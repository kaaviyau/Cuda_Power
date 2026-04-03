// transpose_param_test.cu
// Matrix Transpose with tunable parameters similar to matmul
// Testing parameters:
// TILE_DIM = 32 (tile size for transpose)
// WORK_PER_THREAD = 2 (elements processed per thread)
// THREADS_PER_BLOCK = 256 (16x16 thread block)
// USE_SHARED = 1 (shared memory enabled)
// USE_UNROLL = 1 (loop unrolling enabled)
// GRID_SCALE = 2.0 (grid oversubscription)
// REGISTER_LIMIT = 32

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define TILE_DIM 32
#define WORK_PER_THREAD 2
#define THREAD_BLOCK_DIM 16 // 16x16 = 256 threads per block
#define USE_SHARED 1
#define USE_UNROLL 1
#define GRID_SCALE 1.0f
#define REGISTER_LIMIT 32

// Transpose kernel with tunable parameters
// Each thread transposes WORK_PER_THREAD elements
__global__
__launch_bounds__(THREAD_BLOCK_DIM *THREAD_BLOCK_DIM, REGISTER_LIMIT) void transposeOptimized(float *odata, const float *idata, int width, int height, int grid_x0, int grid_y0)
{
 int bx = blockIdx.x;
 int by = blockIdx.y;
 int tx = threadIdx.x;
 int ty = threadIdx.y;
 if (bx >= grid_x0 || by >= grid_y0)
  return;

#if USE_SHARED
 // Shared memory with padding to avoid bank conflicts
 __shared__ float tile[TILE_DIM][TILE_DIM + 1];
#endif

 // Calculate input base
 int yBase = by * TILE_DIM;

#if USE_SHARED
 // Load data into shared memory with coalesced reads
 // Each thread loads up to WORK_PER_THREAD elements
#if USE_UNROLL
#pragma unroll
#endif
 for (int j = 0; j < TILE_DIM; j += THREAD_BLOCK_DIM)
 {
  int col = tx + j;
  if (col < TILE_DIM)
  {
#if USE_UNROLL
#pragma unroll
#endif
   for (int w = 0; w < WORK_PER_THREAD; w++)
   {
    int row = ty + w * THREAD_BLOCK_DIM;
    if (row < TILE_DIM)
    {
     int x = bx * TILE_DIM + col;
     int y = yBase + row;
     if (x < width && y < height)
     {
      tile[row][col] = idata[y * width + x];
     }
    }
   }
  }
 }

 __syncthreads();

 // Write transposed data from shared memory (coalesced write)
 // Swap block coordinates for transpose
 int xBase = by * TILE_DIM;
 yBase = bx * TILE_DIM;

#if USE_UNROLL
#pragma unroll
#endif
 for (int j = 0; j < TILE_DIM; j += THREAD_BLOCK_DIM)
 {
  int col = tx + j;
  if (col < TILE_DIM)
  {
#if USE_UNROLL
#pragma unroll
#endif
   for (int w = 0; w < WORK_PER_THREAD; w++)
   {
    int row = ty + w * THREAD_BLOCK_DIM;
    if (row < TILE_DIM)
    {
     int x = xBase + col;
     int y = yBase + row;
     if (x < height && y < width)
     {
      // Transpose: read tile[col][row] instead of tile[row][col]
      odata[y * height + x] = tile[col][row];
     }
    }
   }
  }
 }
#else
 // Direct transpose without shared memory
#if USE_UNROLL
#pragma unroll
#endif
 for (int w = 0; w < WORK_PER_THREAD; w++)
 {
  int row = ty + w * THREAD_BLOCK_DIM;
  if (row < TILE_DIM)
  {
   int y = yBase + row;
   int x = xIndex;
   int outx = by * TILE_DIM + tx;
   int outy = bx * TILE_DIM + row;

   if (x < width && y < height && outx < height && outy < width)
   {
    odata[outy * height + outx] = idata[y * width + x];
   }
  }
 }
#endif
}

// Simple copy kernel for baseline comparison
__global__ void simpleCopy(float *odata, const float *idata, int width, int height)
{
 int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
 int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

 for (int i = 0; i < TILE_DIM; i += THREAD_BLOCK_DIM)
 {
  int y = yIndex + i;
  if (xIndex < width && y < height)
  {
   int index = y * width + xIndex;
   odata[index] = idata[index];
  }
 }
}

// Naive transpose (for comparison)
__global__ void transposeNaive(float *odata, const float *idata, int width, int height)
{
 int xBase = blockIdx.x * TILE_DIM;
 int yBase = blockIdx.y * TILE_DIM;

 for (int j = 0; j < TILE_DIM; j += THREAD_BLOCK_DIM)
 {
  int x = xBase + threadIdx.x + j;
  for (int i = 0; i < TILE_DIM; i += THREAD_BLOCK_DIM)
  {
   int y = yBase + threadIdx.y + i;
   if (x < width && y < height)
   {
    int index_in = y * width + x;
    int index_out = x * height + y;
    odata[index_out] = idata[index_in];
   }
  }
 }
}

void initMatrix(float *data, int size)
{
 for (int i = 0; i < size; i++)
 {
  data[i] = (float)i;
 }
}

void computeTransposeGold(float *gold, const float *idata, int width, int height)
{
 for (int y = 0; y < height; y++)
 {
  for (int x = 0; x < width; x++)
  {
   gold[x * height + y] = idata[y * width + x];
  }
 }
}

bool verifyTranspose(const float *result, const float *reference, int size, float tolerance)
{
 for (int i = 0; i < size; i++)
 {
  float diff = fabs(result[i] - reference[i]);
  if (diff > tolerance)
  {
   return false;
  }
 }
 return true;
}

int main()
{
 printf("\n=== CUDA Matrix Transpose - Parameter Test ===\n");
 printf("TILE_DIM: %d\n", TILE_DIM);
 printf("WORK_PER_THREAD: %d\n", WORK_PER_THREAD);
 printf("THREAD_BLOCK_DIM: %dx%d = %d threads/block\n",
        THREAD_BLOCK_DIM, THREAD_BLOCK_DIM, THREAD_BLOCK_DIM * THREAD_BLOCK_DIM);
 printf("USE_SHARED: %d\n", USE_SHARED);
 printf("USE_UNROLL: %d\n", USE_UNROLL);
 printf("GRID_SCALE: %.1f\n", GRID_SCALE);
 printf("REGISTER_LIMIT: %d\n", REGISTER_LIMIT);
 printf("================================================\n\n");

 // Matrix dimensions
 int width = 1024;  // columns
 int height = 1024; // rows

 printf("Matrix dimensions: %d x %d\n", width, height);
 printf("Transposed dimensions: %d x %d\n\n", height, width);

 size_t size = (size_t)width * (size_t)height;
 size_t mem_size = sizeof(float) * size;

 printf("Memory requirements: %.2f MB\n\n", mem_size / (1024.0 * 1024.0));

 // Allocate host memory
 float *h_idata = (float *)malloc(mem_size);
 float *h_odata = (float *)malloc(mem_size);
 float *h_gold = (float *)malloc(mem_size);

 if (!h_idata || !h_odata || !h_gold)
 {
  printf("ERROR: Host memory allocation failed!\n");
  return 1;
 }

 // Initialize input matrix
 printf("Initializing matrix...\n");
 initMatrix(h_idata, size);

 // Compute reference solution
 printf("Computing reference transpose...\n");
 computeTransposeGold(h_gold, h_idata, width, height);

 // Allocate device memory
 float *d_idata = NULL, *d_odata = NULL;
 cudaError_t err;

 err = cudaMalloc((void **)&d_idata, mem_size);
 if (err != cudaSuccess)
 {
  printf("ERROR: cudaMalloc d_idata failed: %s\n", cudaGetErrorString(err));
  free(h_idata);
  free(h_odata);
  free(h_gold);
  return 1;
 }

 err = cudaMalloc((void **)&d_odata, mem_size);
 if (err != cudaSuccess)
 {
  printf("ERROR: cudaMalloc d_odata failed: %s\n", cudaGetErrorString(err));
  cudaFree(d_idata);
  free(h_idata);
  free(h_odata);
  free(h_gold);
  return 1;
 }

 // Copy data to device
 printf("Copying data to device...\n");
 err = cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice);
 if (err != cudaSuccess)
 {
  printf("ERROR: cudaMemcpy failed: %s\n", cudaGetErrorString(err));
  cudaFree(d_idata);
  cudaFree(d_odata);
  free(h_idata);
  free(h_odata);
  free(h_gold);
  return 1;
 }

 // Calculate grid dimensions
 int grid_x0 = (width + TILE_DIM - 1) / TILE_DIM;
 int grid_y0 = (height + TILE_DIM - 1) / TILE_DIM;

 // Apply grid scaling
 int grid_x = (int)ceilf(grid_x0 * GRID_SCALE);
 int grid_y = (int)ceilf(grid_y0 * GRID_SCALE);

 if (grid_x < 1)
  grid_x = 1;
 if (grid_y < 1)
  grid_y = 1;

 dim3 threads(THREAD_BLOCK_DIM, THREAD_BLOCK_DIM);
 dim3 grid(grid_x, grid_y);

 printf("\nLaunch configuration:\n");
 printf("Grid: (%d, %d) = %d blocks\n", grid.x, grid.y, grid.x * grid.y);
 printf("Threads per block: (%d, %d) = %d threads\n",
        threads.x, threads.y, threads.x * threads.y);
 printf("Total threads: %d\n\n", grid.x * grid.y * threads.x * threads.y);

 // Create timing events
 cudaEvent_t start, stop;
 cudaEventCreate(&start);
 cudaEventCreate(&stop);

 // Test 1: Simple Copy (baseline)
 printf("=== Test 1: Simple Copy (Baseline) ===\n");
 simpleCopy<<<grid, threads>>>(d_odata, d_idata, width, height);
 cudaDeviceSynchronize();

 int nIter = 100;
 cudaEventRecord(start, 0);
 for (int i = 0; i < nIter; i++)
 {
  simpleCopy<<<grid, threads>>>(d_odata, d_idata, width, height);
 }
 cudaEventRecord(stop, 0);
 cudaEventSynchronize(stop);

 float copyTime;
 cudaEventElapsedTime(&copyTime, start, stop);
 float copyBandwidth = 2.0f * mem_size * nIter / (copyTime * 1e6);

 printf("Time per copy: %.6f ms\n", copyTime / nIter);
 printf("Bandwidth: %.2f GB/s\n\n", copyBandwidth);

 // Test 2: Naive Transpose
 printf("=== Test 2: Naive Transpose ===\n");
 cudaMemset(d_odata, 0, mem_size);
 transposeNaive<<<grid, threads>>>(d_odata, d_idata, width, height);
 cudaDeviceSynchronize();

 cudaEventRecord(start, 0);
 for (int i = 0; i < nIter; i++)
 {
  transposeNaive<<<grid, threads>>>(d_odata, d_idata, width, height);
 }
 cudaEventRecord(stop, 0);
 cudaEventSynchronize(stop);

 float naiveTime;
 cudaEventElapsedTime(&naiveTime, start, stop);
 float naiveBandwidth = 2.0f * mem_size * nIter / (naiveTime * 1e6);

 cudaMemcpy(h_odata, d_odata, mem_size, cudaMemcpyDeviceToHost);
 bool naiveCorrect = verifyTranspose(h_odata, h_gold, size, 0.01f);

 printf("Time per transpose: %.6f ms\n", naiveTime / nIter);
 printf("Bandwidth: %.2f GB/s\n", naiveBandwidth);
 printf("Correctness: %s\n\n", naiveCorrect ? "PASS \u2713" : "FAIL \u2717");

 // Test 3: Optimized Transpose with tuning parameters
 printf("=== Test 3: Optimized Transpose (Tuned Parameters) ===\n");
 cudaMemset(d_odata, 0, mem_size);
 transposeOptimized<<<grid, threads>>>(d_odata, d_idata, width, height, grid_x0, grid_y0);

 err = cudaGetLastError();
 if (err != cudaSuccess)
 {
  printf("ERROR: Kernel launch failed: %s\n", cudaGetErrorString(err));
  cudaFree(d_idata);
  cudaFree(d_odata);
  free(h_idata);
  free(h_odata);
  free(h_gold);
  return 1;
 }

 cudaDeviceSynchronize();

 cudaEventRecord(start, 0);
 for (int i = 0; i < nIter; i++)
 {
  transposeOptimized<<<grid, threads>>>(d_odata, d_idata, width, height, grid_x0, grid_y0);
 }
 cudaEventRecord(stop, 0);
 cudaEventSynchronize(stop);

 float optimizedTime;
 cudaEventElapsedTime(&optimizedTime, start, stop);
 float optimizedBandwidth = 2.0f * mem_size * nIter / (optimizedTime * 1e6);

 cudaMemcpy(h_odata, d_odata, mem_size, cudaMemcpyDeviceToHost);
 bool optimizedCorrect = verifyTranspose(h_odata, h_gold, size, 0.01f);

 printf("Time per transpose: %.6f ms\n", optimizedTime / nIter);
 printf("Bandwidth: %.2f GB/s\n", optimizedBandwidth);
 printf("Correctness: %s\n\n", optimizedCorrect ? "PASS \u2713" : "FAIL \u2717");

 // Performance summary
 printf("=== Performance Summary ===\n");
 printf("Simple Copy:         %.2f GB/s (baseline)\n", copyBandwidth);
 printf("Naive Transpose:     %.2f GB/s (%.1f%% of copy)\n",
        naiveBandwidth, 100.0f * naiveBandwidth / copyBandwidth);
 printf("Optimized Transpose: %.2f GB/s (%.1f%% of copy)\n",
        optimizedBandwidth, 100.0f * optimizedBandwidth / copyBandwidth);
 printf("Speedup (optimized vs naive): %.2fx\n", naiveTime / optimizedTime);
 printf("===========================\n\n");

 // Cleanup
 cudaFree(d_idata);
 cudaFree(d_odata);
 free(h_idata);
 free(h_odata);
 free(h_gold);
 cudaEventDestroy(start);
 cudaEventDestroy(stop);

 bool allPassed = naiveCorrect && optimizedCorrect;
 printf("=== Test %s ===\n", allPassed ? "PASSED" : "FAILED");

 return allPassed ? 0 : 1;
}
