#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <ctime>

#define MATRIX_SIZE 4096
#define BLOCK_SIZE 256 
#define BM 64
#define BN 64
#define BK 8
#define TM 4

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// Error checking macros for CUDA and cuBLAS
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUBLAS(call) { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error in %s:%d: %d\n", __FILE__, __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
}


// so grid.x covers columns and grid.y covers rows.
__global__ void sgemm1DBlockTiling(int M, int N, int K, float alpha,
                                   const float *A, const float *B, float beta,
                                   float *C) {
    // Determine which tile of C this block is responsible for.
    const unsigned int tileRow = blockIdx.y; // each tile covers BM rows
    const unsigned int tileCol = blockIdx.x; // each tile covers BN columns

    // We assume BN=64 threads per row, so we have 256/64 = 4 thread rows.
    const int threadCol = threadIdx.x % BN;      // 0..63
    const int threadRow = threadIdx.x / BN;        // 0..3

    // Allocate shared memory tiles.
    __shared__ float As[BM * BK]; // tile from A: 64x8 elements
    __shared__ float Bs[BK * BN]; // tile from B: 8x64 elements

    // Adjust pointers to the beginning of the tile in global memory.
    // Each tile of A starts at row (tileRow*BM) and B starts at column (tileCol*BN).
    A += tileRow * BM * K;
    B += tileCol * BN;
    C += tileRow * BM * N + tileCol * BN;

    // Each thread maintains an accumulator for TM output rows.
    float threadResults[TM];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        threadResults[i] = 0.0f;

    // Loop over the K dimension in blocks of BK.
    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // Load A tile (size BM x BK) into shared memory.
        // Use a strided loop so that all 256 threads cooperate.
        for (int idx = threadIdx.x; idx < BM * BK; idx += blockDim.x) {
            int row = idx / BK;
            int col = idx % BK;
            As[idx] = A[row * K + col];
        }
        // Load B tile (size BK x BN) into shared memory.
        for (int idx = threadIdx.x; idx < BK * BN; idx += blockDim.x) {
            int row = idx / BN;
            int col = idx % BN;
            Bs[idx] = B[row * N + col];
        }
        __syncthreads();

        // Compute dot products for each element of the output sub-tile.
        // For each position in the K dimension tile.
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            float tmpB = Bs[dotIdx * BN + threadCol]; // value from B tile
            // Each thread computes TM consecutive rows.
            for (int resIdx = 0; resIdx < TM; ++resIdx) {
                // (threadRow*TM + resIdx) gives the row in the tile
                threadResults[resIdx] += 
                    As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
            }
        }
        __syncthreads();

        // Advance A and B pointers to the next tile in K.
        A += BK;
        B += BK * N;
    }

    // Write the computed results from registers back to global memory.
    for (int resIdx = 0; resIdx < TM; ++resIdx) {
        // Row index in the output tile: (threadRow * TM + resIdx)
        int row = threadRow * TM + resIdx;
        C[row * N + threadCol] = alpha * threadResults[resIdx] + 
                                 beta * C[row * N + threadCol];
    }
}

// Function to fill a matrix with random values
void fillMatrixRandom(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = static_cast<float>(std::rand()) / RAND_MAX; // Random float in [0, 1)
    }
}

// Function to get the current time in seconds
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    // Seed the random number generator
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    // Matrix dimensions
    const int M = MATRIX_SIZE;
    const int N = MATRIX_SIZE;
    const int K = MATRIX_SIZE;

    // Host memory allocation
    float* h_A = (float*)malloc(M * K * sizeof(float));
    float* h_B = (float*)malloc(K * N * sizeof(float));
    float* h_C_1D = (float*)malloc(M * N * sizeof(float));
    float* h_C_cublas = (float*)malloc(M * N * sizeof(float));

    // Fill matrices A and B with random values
    fillMatrixRandom(h_A, M, K);
    fillMatrixRandom(h_B, K, N);

    // Initialize result matrices to zero
    for (int i = 0; i < M * N; ++i) {
        h_C_1D[i] = 0.0f;
        h_C_cublas[i] = 0.0f;
    }

    // CUDA setup
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void **)&d_A, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **)&d_B, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **)&d_C, M * N * sizeof(float)));

    // Copy matrices A and B to the device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));

    // --- 1D Block Tiling SGEMM ---
    dim3 blockDim(BLOCK_SIZE); // Threads per block
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM)); // Blocks in grid

    double start_time = get_time();
    sgemm1DBlockTiling<<<gridDim, blockDim>>>(M, N, K, 1.0f, d_A, d_B, 0.0f, d_C);
    CHECK_CUDA(cudaDeviceSynchronize());
    double end_time = get_time();

    CHECK_CUDA(cudaMemcpy(h_C_1D, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // --- cuBLAS SGEMM ---
    double start_cublas = get_time();
    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
    CHECK_CUDA(cudaDeviceSynchronize());
    double end_cublas = get_time();

    CHECK_CUDA(cudaMemcpy(h_C_cublas, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Calculate performance metrics
    double gflops = 2.0 * M * N * K * 1e-9; // Total operations in GFLOPs

    double one_D_tile_time = end_time - start_time;
    double cublas_time = end_cublas - start_cublas;
    double one_D_tile = gflops / one_D_tile_time;
    double cublas_gflops = gflops / cublas_time;
    double relative_performance = (cublas_time / one_D_tile_time) * 100;

    // Print results
    printf("1D Block Tiling SGEMM Time: %f seconds\n", one_D_tile_time);
    printf("cuBLAS SGEMM Time: %f seconds\n", cublas_time);
    printf("1D Block Tiling SGEMM GFLOPs: %f\n", one_D_tile);
    printf("cuBLAS SGEMM GFLOPs: %f\n", cublas_gflops);
    printf("Performance relative to cuBLAS: %f%%\n", relative_performance);

    // Clean up
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUBLAS(cublasDestroy(handle));

    free(h_A);
    free(h_B);
    free(h_C_1D);
    free(h_C_cublas);

    return 0;
}
