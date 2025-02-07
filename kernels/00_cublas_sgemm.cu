#include "sgemm_kernels.h"
#include <cstdio>

void run_cublas_sgemm(cublasHandle_t handle, int M, int N, int K, 
                      float alpha, float* d_A, float* d_B, 
                      float beta, float* d_C) {

    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, CUDA_R_32F,
               N, d_A, CUDA_R_32F, K, &beta, d_C, CUDA_R_32F, N,
               CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}
