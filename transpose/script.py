#!/usr/bin/env python3
"""
profile_transpose_dataset.py
- Sweeps parameter combinations for matrix transpose kernels
- Compiles CUDA file per combo (nvcc --maxrregcount)
- Logs GPU power.draw for POWER_LOG_DURATION seconds and computes average
- Parses bandwidth, time, and correctness for EACH kernel separately
- Writes one row per configuration PER KERNEL to CSV
- Supports multiple kernels: copy, naive, optimized
"""

import subprocess, time, csv, re, math
from itertools import product
from pathlib import Path

# -------------------
# Config
# -------------------
CUDA_SOURCE = "transpose_modified.cu"  # Relative to script directory
CUDA_BINARY = "transpose_bin"
LOG_DIR = Path("power_logs_transpose")
CSV_FILE = Path("transpose_dataset.csv")

# Parameter ranges
TILE_DIMS = [16, 32, 64]
WORK_PER_THREAD = [1, 2, 4]
THREAD_BLOCK_DIMS = [8, 16, 32]  # Will give 64, 256, 1024 threads per block
SHARED = [0, 1]
UNROLL = [0, 1]
GRID_SCALES = [1.0, 1.5, 2.0]
REGISTER_LIMITS = [32, 64, 128]

POWER_LOG_DURATION = 10     # seconds (reduced for Colab - was 60)
NVIDIA_SMI_INTERVAL = 1     # seconds (sampling interval)

# Ensure dirs
SCRIPT_DIR = Path.cwd()
CUDA_SRC_PATH = SCRIPT_DIR / CUDA_SOURCE
CUDA_BIN_PATH = SCRIPT_DIR / CUDA_BINARY
LOG_DIR.mkdir(exist_ok=True)

# Regexes to parse output for different test sections
# The transpose code has three test sections with their own results

# Test 1: Simple Copy
COPY_TIME_RE = re.compile(r"=== Test 1.*?Time per copy:\s*([0-9]*\.?[0-9]+)\s*ms", re.DOTALL)
COPY_BW_RE = re.compile(r"=== Test 1.*?Bandwidth:\s*([0-9]*\.?[0-9]+)\s*GB/s", re.DOTALL)

# Test 2: Naive Transpose
NAIVE_TIME_RE = re.compile(r"=== Test 2.*?Time per transpose:\s*([0-9]*\.?[0-9]+)\s*ms", re.DOTALL)
NAIVE_BW_RE = re.compile(r"=== Test 2.*?Bandwidth:\s*([0-9]*\.?[0-9]+)\s*GB/s", re.DOTALL)
NAIVE_CORRECT_RE = re.compile(r"=== Test 2.*?Correctness:\s*(PASS|FAIL)", re.DOTALL)

# Test 3: Optimized Transpose
OPT_TIME_RE = re.compile(r"=== Test 3.*?Time per transpose:\s*([0-9]*\.?[0-9]+)\s*ms", re.DOTALL)
OPT_BW_RE = re.compile(r"=== Test 3.*?Bandwidth:\s*([0-9]*\.?[0-9]+)\s*GB/s", re.DOTALL)
OPT_CORRECT_RE = re.compile(r"=== Test 3.*?Correctness:\s*(PASS|FAIL)", re.DOTALL)

# Performance summary
SPEEDUP_RE = re.compile(r"Speedup \(optimized vs naive\):\s*([0-9]*\.?[0-9]+)x")

# General patterns
THREADS_RE = re.compile(r"THREAD_BLOCK_DIM:\s*(\d+)x(\d+)\s*=\s*(\d+)\s*threads/block")
CUDA_ERR_RE = re.compile(r"(CUDA|ERROR).*?error[:\s]+(.*?)(?:\n|$)", re.IGNORECASE)

def compile_cuda(tile_dim, work, thread_dim, shared, unroll, grid, reg):
    print(f"[Compiling] TILE={tile_dim}, WORK={work}, THREAD_DIM={thread_dim}, SHARED={shared}, UNROLL={unroll}, GRID={grid}, REG={reg}")
    cmd = [
        "nvcc", "-O3",
        "-arch=sm_89",  # NVIDIA RTX 5000 Ada Generation (compute capability 8.9)
        f"--maxrregcount={reg}",
        f"-DTILE_DIM={tile_dim}",
        f"-DWORK_PER_THREAD={work}",
        f"-DTHREAD_BLOCK_DIM={thread_dim}",
        f"-DUSE_SHARED={shared}",
        f"-DUSE_UNROLL={unroll}",
        f"-DGRID_SCALE={grid}f",
        f"-DREGISTER_LIMIT={reg}",
        "-o", str(CUDA_BIN_PATH),
        str(CUDA_SRC_PATH)
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print("Compilation failed:\n", proc.stderr)
        return False, proc.stderr
    return True, proc.stdout

def start_power_log(log_path):
    cmd = [
        "nvidia-smi",
        "--query-gpu=power.draw",
        "--format=csv,noheader,nounits",
        "-l", str(NVIDIA_SMI_INTERVAL)
    ]
    f = open(log_path, "w")
    p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.DEVNULL)
    return p, f

def stop_power_log(proc, handle):
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    handle.close()
    time.sleep(0.2)

def parse_power_log(log_path):
    vals = []
    try:
        with open(log_path, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    v = float(line)
                    vals.append(v)
                except:
                    continue
    except Exception:
        return None, 0
    if not vals:
        return None, 0
    return sum(vals)/len(vals), len(vals)

def validate_config(tile, work, thread_dim, shared, unroll, grid, reg):
    """
    Validate configuration against hardware and algorithm constraints.
    Returns (is_valid, skip_reason)
    """
    threads_pb = thread_dim * thread_dim
    
    # 1. Threads per block limit (hardware)
    if threads_pb > 1024:
        return False, "SKIP_THREADS_LIMIT"
    
    # 2. Must cover entire tile (CRITICAL - will produce wrong results)
    if thread_dim * work < tile:
        return False, f"SKIP_INCOMPLETE_TILE_COVERAGE_{thread_dim}x{work}<{tile}"
    
    # 3. Shared memory limit (48KB typical)
    if shared == 1:
        shared_mem_bytes = tile * (tile + 1) * 4  # +1 for bank conflict padding
        if shared_mem_bytes > 49152:  # 48KB
            return False, f"SKIP_SHARED_MEM_LIMIT_{shared_mem_bytes}B>48KB"
    
    # 4. Excessive work per thread (diminishing returns, register pressure)
    if work > 8:
        return False, f"SKIP_EXCESSIVE_WORK_PER_THREAD_{work}>8"
    
    # 5. Non-power-of-2 tile dimensions (poor memory alignment)
    if tile not in [8, 16, 32, 64, 128]:
        return False, f"SKIP_NON_POWER_OF_2_TILE_{tile}"
    
    return True, None

def parse_output(stdout):
    """Parse the output and extract metrics for all three kernels"""
    results = {
        "copy": {},
        "naive": {},
        "optimized": {}
    }

    # Parse Copy kernel (Test 1)
    m = COPY_TIME_RE.search(stdout)
    if m: results["copy"]["time_ms"] = float(m.group(1))
    m = COPY_BW_RE.search(stdout)
    if m: results["copy"]["bandwidth_gbps"] = float(m.group(1))
    results["copy"]["correctness"] = "N/A"  # Copy doesn't have correctness check

    # Parse Naive transpose (Test 2)
    m = NAIVE_TIME_RE.search(stdout)
    if m: results["naive"]["time_ms"] = float(m.group(1))
    m = NAIVE_BW_RE.search(stdout)
    if m: results["naive"]["bandwidth_gbps"] = float(m.group(1))
    m = NAIVE_CORRECT_RE.search(stdout)
    if m: results["naive"]["correctness"] = m.group(1)
    else: results["naive"]["correctness"] = "UNKNOWN"

    # Parse Optimized transpose (Test 3)
    m = OPT_TIME_RE.search(stdout)
    if m: results["optimized"]["time_ms"] = float(m.group(1))
    m = OPT_BW_RE.search(stdout)
    if m: results["optimized"]["bandwidth_gbps"] = float(m.group(1))
    m = OPT_CORRECT_RE.search(stdout)
    if m: results["optimized"]["correctness"] = m.group(1)
    else: results["optimized"]["correctness"] = "UNKNOWN"

    # Parse speedup
    m = SPEEDUP_RE.search(stdout)
    speedup = float(m.group(1)) if m else None

    # Parse threads per block
    m = THREADS_RE.search(stdout)
    threads_per_block = int(m.group(3)) if m else None

    return results, speedup, threads_per_block

def run_and_profile(tile_dim, work, thread_dim, shared, unroll, grid, reg):
    # start power logging
    log_name = f"power_t{tile_dim}_w{work}_td{thread_dim}_s{shared}_u{unroll}_g{grid}_r{reg}.log"
    log_path = LOG_DIR / log_name
    proc, handle = start_power_log(log_path)
    time.sleep(0.5)  # stabilize logging

    # run binary and capture stdout+stderr
    start_time = time.time()
    try:
        run = subprocess.run([str(CUDA_BIN_PATH)], capture_output=True, text=True, timeout=600)
        stdout = run.stdout + "\n" + run.stderr
        exit_code = run.returncode
    except subprocess.TimeoutExpired:
        stdout = ""
        exit_code = -1
    end_time = time.time()
    exec_time = end_time - start_time

    # ensure we collected roughly POWER_LOG_DURATION seconds of samples
    remaining = POWER_LOG_DURATION - (exec_time if exec_time < POWER_LOG_DURATION else 0)
    if remaining > 0:
        time.sleep(remaining)
    stop_power_log(proc, handle)

    avg_power, sample_count = parse_power_log(log_path)

    # parse outputs for all kernels
    kernel_results, speedup, threads_pb = parse_output(stdout)

    # Check for CUDA errors
    cuda_err = None
    m = CUDA_ERR_RE.search(stdout)
    if m: cuda_err = m.group(2).strip()

    return {
        "kernel_results": kernel_results,
        "speedup": speedup,
        "threads_per_block": threads_pb if threads_pb else thread_dim * thread_dim,
        "avg_power_watts": avg_power,
        "exec_time_sec": exec_time,
        "power_log_duration": sample_count * NVIDIA_SMI_INTERVAL,
        "exit_code": exit_code,
        "cuda_error_str": cuda_err,
        "power_log_path": str(log_path)
    }

def main():
    combos = list(product(TILE_DIMS, WORK_PER_THREAD, THREAD_BLOCK_DIMS,
                         SHARED, UNROLL, GRID_SCALES, REGISTER_LIMITS))
    total = len(combos)
    print(f"Total configurations: {total}")
    print(f"Total rows (3 kernels per config): {total * 3}")

    header = [
        "kernel_name",           # copy, naive, optimized
        "tile_dim",
        "work_per_thread",
        "thread_block_dim",
        "threads_per_block",
        "use_shared",
        "use_unroll",
        "grid_scale",
        "register_limit",
        "time_ms",
        "bandwidth_gbps",
        "correctness",
        "speedup_vs_naive",     # only for optimized
        "avg_power_watts",
        "exec_time_sec",
        "power_log_duration",
        "exit_code",
        "cuda_error_str",
        "power_log_path"
    ]

    first = not CSV_FILE.exists()
    with open(CSV_FILE, "a", newline="") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=header)
        if first:
            writer.writeheader()

        for idx, (tile, work, thread_dim, shared, unroll, grid, reg) in enumerate(combos, start=1):
            print(f"\n=== Run {idx}/{total} ===")
            threads_pb = thread_dim * thread_dim

            # Validate configuration
            is_valid, skip_reason = validate_config(tile, work, thread_dim, shared, unroll, grid, reg)
            if not is_valid:
                print(f"Skipping config: {skip_reason}")
                # Write skip rows for all three kernels
                for kernel_name in ["copy", "naive", "optimized"]:
                    row = {k: "" for k in header}
                    row.update({
                        "kernel_name": kernel_name,
                        "tile_dim": tile,
                        "work_per_thread": work,
                        "thread_block_dim": thread_dim,
                        "threads_per_block": threads_pb,
                        "use_shared": shared,
                        "use_unroll": unroll,
                        "grid_scale": grid,
                        "register_limit": reg,
                        "correctness": skip_reason
                    })
                    writer.writerow(row)
                csvf.flush()
                continue

            # Old threads limit check is now redundant (covered by validate_config)
            # but keeping for backwards compatibility
            if threads_pb > 1024:
                print(f"Skipping config: threads_per_block={threads_pb} > 1024")
                # Write skip rows for all three kernels
                for kernel_name in ["copy", "naive", "optimized"]:
                    row = {k: "" for k in header}
                    row.update({
                        "kernel_name": kernel_name,
                        "tile_dim": tile,
                        "work_per_thread": work,
                        "thread_block_dim": thread_dim,
                        "threads_per_block": threads_pb,
                        "use_shared": shared,
                        "use_unroll": unroll,
                        "grid_scale": grid,
                        "register_limit": reg,
                        "correctness": "SKIP_THREADS_LIMIT"
                    })
                    writer.writerow(row)
                csvf.flush()
                continue

            ok, comp_out = compile_cuda(tile, work, thread_dim, shared, unroll, grid, reg)
            if not ok:
                print("Compilation failed; writing COMPILE_FAIL rows.")
                # Write fail rows for all three kernels
                for kernel_name in ["copy", "naive", "optimized"]:
                    row = {k: "" for k in header}
                    row.update({
                        "kernel_name": kernel_name,
                        "tile_dim": tile,
                        "work_per_thread": work,
                        "thread_block_dim": thread_dim,
                        "threads_per_block": threads_pb,
                        "use_shared": shared,
                        "use_unroll": unroll,
                        "grid_scale": grid,
                        "register_limit": reg,
                        "correctness": "COMPILE_FAIL",
                        "cuda_error_str": str(comp_out)[:200]  # Truncate long errors
                    })
                    writer.writerow(row)
                csvf.flush()
                continue

            res = run_and_profile(tile, work, thread_dim, shared, unroll, grid, reg)

            if res["exit_code"] != 0:
                print(f"Binary failed exit_code={res['exit_code']} cuda_err={res['cuda_error_str']}")
            else:
                print(f"Power(W): {res['avg_power_watts']}")
                for kname in ["copy", "naive", "optimized"]:
                    kr = res["kernel_results"].get(kname, {})
                    bw = kr.get("bandwidth_gbps", "N/A")
                    tm = kr.get("time_ms", "N/A")
                    corr = kr.get("correctness", "N/A")
                    print(f"  {kname:12s}: BW={bw} GB/s, Time={tm} ms, Correct={corr}")

            # Write one row per kernel
            for kernel_name in ["copy", "naive", "optimized"]:
                kr = res["kernel_results"].get(kernel_name, {})

                row = {
                    "kernel_name": kernel_name,
                    "tile_dim": tile,
                    "work_per_thread": work,
                    "thread_block_dim": thread_dim,
                    "threads_per_block": res["threads_per_block"],
                    "use_shared": shared,
                    "use_unroll": unroll,
                    "grid_scale": grid,
                    "register_limit": reg,
                    "time_ms": ("%.6f" % kr["time_ms"]) if "time_ms" in kr else "",
                    "bandwidth_gbps": ("%.3f" % kr["bandwidth_gbps"]) if "bandwidth_gbps" in kr else "",
                    "correctness": kr.get("correctness", "UNKNOWN"),
                    "speedup_vs_naive": ("%.3f" % res["speedup"]) if kernel_name == "optimized" and res["speedup"] else "",
                    "avg_power_watts": ("%.3f" % res["avg_power_watts"]) if res["avg_power_watts"] else "",
                    "exec_time_sec": ("%.3f" % res["exec_time_sec"]),
                    "power_log_duration": res["power_log_duration"],
                    "exit_code": res["exit_code"],
                    "cuda_error_str": res["cuda_error_str"] if res["cuda_error_str"] else "",
                    "power_log_path": res["power_log_path"]
                }
                writer.writerow(row)

            csvf.flush()
            time.sleep(0.5)

    print("\nAll runs finished. CSV saved to:", CSV_FILE.resolve())

if __name__ == "__main__":
    main()
