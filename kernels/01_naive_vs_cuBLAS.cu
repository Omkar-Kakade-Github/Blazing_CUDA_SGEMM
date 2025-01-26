#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <ctime>
#include <cmath>

#define MATRIX_SIZE 4096

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

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < M && y < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            sum += A[x * K + i] * B[i * N + y];
        }

        C[x * N + y] = alpha * sum + beta * C[x * N + y];
    }
}

// Function to fill a matrix with random values
void fillMatrixRandom(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = (static_cast<float>(std::rand())) / RAND_MAX; // Random float in [0, 1)
    }
}

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
    float* h_C_naive = (float*)malloc(M * N * sizeof(float));
    float* h_C_cublas = (float*)malloc(M * N * sizeof(float));

    // Fill matrices A and B with random values
    fillMatrixRandom(h_A, M, K);
    fillMatrixRandom(h_B, K, N);

    // Initialize result matrices to zero
    for (int i = 0; i < M * N; ++i) {
        h_C_naive[i] = 0.0f;
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

    // --- Naive SGEMM ---
    dim3 blockDim(32, 32, 1);
    dim3 gridDim((M + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    double start_naive = get_time();
    sgemm_naive<<<gridDim, blockDim>>>(M, N, K, 1.0f, d_A, d_B, 0.0f, d_C);
    CHECK_CUDA(cudaDeviceSynchronize());
    double end_naive = get_time();

    CHECK_CUDA(cudaMemcpy(h_C_naive, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // --- cuBLAS SGEMM ---
    double start_cublas = get_time();
    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
    CHECK_CUDA(cudaDeviceSynchronize());
    double end_cublas = get_time();

    CHECK_CUDA(cudaMemcpy(h_C_cublas, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    double gflops = 2.0 * M * N * K * 1e-9; // Total operations in GFLOPs

    // Calculate performance relative to cuBLAS
    double naive_time = end_naive - start_naive;
    double cublas_time = end_cublas - start_cublas;
    double naive_gflops = gflops / naive_time;
    double cublas_gflops = gflops / cublas_time;
    double relative_performance = (cublas_time / naive_time) * 100;

    // Print results
    printf("Naive SGEMM Time: %f seconds\n", naive_time);
    printf("cuBLAS SGEMM Time: %f seconds\n", cublas_time);
    printf("Naive SGEMM GFLOPs: %f\n", naive_gflops);
    printf("cuBLAS SGEMM GFLOPs: %f\n", cublas_gflops);
    printf("Performance relative to cuBLAS: %f%%\n", relative_performance);

    // Clean up
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUBLAS(cublasDestroy(handle));

    free(h_A);
    free(h_B);
    free(h_C_naive);
    free(h_C_cublas);

    return 0;
}
