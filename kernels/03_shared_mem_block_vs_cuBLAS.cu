#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <ctime>

#define MATRIX_SIZE 4096
#define BLOCK_SIZE 32

// Macro to compute the ceiling of the division of two numbers
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

// CUDA kernel for single-precision matrix multiplication using shared memory
__global__ void sgemm_shared_memory_block(int num_rows_C, int num_cols_C, int inner_dim, float alpha,
                                          const float *matrix_A, const float *matrix_B,
                                          float beta, float *matrix_C) {
    // The row and column of the output block this thread block is responsible for
    const uint output_block_row = blockIdx.x;
    const uint output_block_col = blockIdx.y;

    // Shared memory buffers to store tiles of matrix_A and matrix_B
    __shared__ float tile_A[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float tile_B[BLOCK_SIZE * BLOCK_SIZE];

    // The row and column of the thread within the block
    const uint thread_col = threadIdx.x % BLOCK_SIZE;
    const uint thread_row = threadIdx.x / BLOCK_SIZE;

    // Move pointers to the starting positions of the current block
    matrix_A += output_block_row * BLOCK_SIZE * inner_dim; // Start of the row in matrix_A
    matrix_B += output_block_col * BLOCK_SIZE;             // Start of the column in matrix_B
    matrix_C += output_block_row * BLOCK_SIZE * num_cols_C + output_block_col * BLOCK_SIZE; // Start of the block in matrix_C

    // Accumulator for the dot product
    float dot_product_result = 0.0f;

    // Loop over the inner dimension in steps of BLOCK_SIZE
    for (int tile_idx = 0; tile_idx < inner_dim; tile_idx += BLOCK_SIZE) {
        // Each thread loads one element from matrix_A and matrix_B into shared memory
        tile_A[thread_row * BLOCK_SIZE + thread_col] = matrix_A[thread_row * inner_dim + thread_col];
        tile_B[thread_row * BLOCK_SIZE + thread_col] = matrix_B[thread_row * num_cols_C + thread_col];

        // Synchronize to ensure all threads have loaded their data into shared memory
        __syncthreads();

        // Move pointers to the next tile in matrix_A and matrix_B
        matrix_A += BLOCK_SIZE;
        matrix_B += BLOCK_SIZE * num_cols_C;

        // Compute the dot product for the current tile
        for (int dot_idx = 0; dot_idx < BLOCK_SIZE; ++dot_idx) {
            dot_product_result += tile_A[thread_row * BLOCK_SIZE + dot_idx] *
                                 tile_B[dot_idx * BLOCK_SIZE + thread_col];
        }

        // Synchronize to ensure all threads have finished computing before loading the next tile
        __syncthreads();
    }

    // Write the result back to matrix_C, scaled by alpha and beta
    matrix_C[thread_row * num_cols_C + thread_col] =
        alpha * dot_product_result + beta * matrix_C[thread_row * num_cols_C + thread_col];
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
    float* h_C_SMEM = (float*)malloc(M * N * sizeof(float));
    float* h_C_cublas = (float*)malloc(M * N * sizeof(float));

    // Fill matrices A and B with random values
    fillMatrixRandom(h_A, M, K);
    fillMatrixRandom(h_B, K, N);

    // Initialize result matrices to zero
    for (int i = 0; i < M * N; ++i) {
        h_C_SMEM[i] = 0.0f;
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

    // --- Shared Memory SGEMM ---
    dim3 blockDim(BLOCK_SIZE * BLOCK_SIZE); // Threads per block
    dim3 gridDim(CEIL_DIV(M, BLOCK_SIZE), CEIL_DIV(N, BLOCK_SIZE)); // Blocks in grid

    double start_time = get_time();
    sgemm_shared_memory_block<<<gridDim, blockDim>>>(M, N, K, 1.0f, d_A, d_B, 0.0f, d_C);
    CHECK_CUDA(cudaDeviceSynchronize());
    double end_time = get_time();

    CHECK_CUDA(cudaMemcpy(h_C_SMEM, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // --- cuBLAS SGEMM ---
    double start_cublas = get_time();
    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
    CHECK_CUDA(cudaDeviceSynchronize());
    double end_cublas = get_time();

    CHECK_CUDA(cudaMemcpy(h_C_cublas, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Calculate performance metrics
    double gflops = 2.0 * M * N * K * 1e-9; // Total operations in GFLOPs

    double SMEM_time = end_time - start_time;
    double cublas_time = end_cublas - start_cublas;
    double SMEM_gflops = gflops / SMEM_time;
    double cublas_gflops = gflops / cublas_time;
    double relative_performance = (cublas_time / SMEM_time) * 100;

    // Print results
    printf("Shared Memory SGEMM Time: %f seconds\n", SMEM_time);
    printf("cuBLAS SGEMM Time: %f seconds\n", cublas_time);
    printf("Shared Memory SGEMM GFLOPs: %f\n", SMEM_gflops);
    printf("cuBLAS SGEMM GFLOPs: %f\n", cublas_gflops);
    printf("Performance relative to cuBLAS: %f%%\n", relative_performance);

    // Clean up
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUBLAS(cublasDestroy(handle));

    free(h_A);
    free(h_B);
    free(h_C_SMEM);
    free(h_C_cublas);

    return 0;
}
