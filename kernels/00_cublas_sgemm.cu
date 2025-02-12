#include "sgemm_kernels.h"
#include <cstdio>

void run_cublas_sgemm(cublasHandle_t handle, int M, int N, int K, 
                      float alpha, float* d_A, float* d_B, 
                      float beta, float* d_C) {

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, 
                &alpha, d_B, N, d_A, K, &beta, d_C, N);
}
