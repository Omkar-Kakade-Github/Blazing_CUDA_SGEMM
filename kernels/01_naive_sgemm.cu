#include "sgemm_kernels.h"
#include <cstdio>

__global__ void sgemm_naive_kernel(int M, int N, int K, 
                                   float alpha, float *A, float *B, 
                                   float beta, float *C) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < M && y < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            sum += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = alpha * sum + beta * C[x * N + y];
    }
}

// Wrapper function to launch the naïve SGEMM kernel.
void run_naive_sgemm(int M, int N, int K, float alpha, 
                     float* d_A, float* d_B, 
                     float beta, float* d_C) {
    dim3 blockDim(32, 32);
    dim3 gridDim((M + blockDim.x - 1) / blockDim.x,
                 (N + blockDim.y - 1) / blockDim.y);
    sgemm_naive_kernel<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
}
