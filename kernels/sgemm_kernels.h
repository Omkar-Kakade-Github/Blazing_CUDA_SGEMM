#ifndef SGEMM_KERNELS_H
#define SGEMM_KERNELS_H

#include <cuda_runtime.h>
#include <cublas_v2.h>

#ifdef __cplusplus
extern "C" {
#endif

// Launches the naïve SGEMM kernel
void run_naive_sgemm(int M, int N, int K, float alpha,
                     float *d_A, float *d_B,
                     float beta, float *d_C);
                     
void run_global_mem_coalesce(int M, int N, int K, float alpha,
                     float *d_A, float *d_B,
                     float beta, float *d_C);

void run_shared_memory_block(int M, int N, int K, float alpha,
                     float *d_A, float *d_B,
                     float beta, float *d_C);

void run_1d_block_tiling(int M, int N, int K, float alpha,
                     float *d_A, float *d_B,
                     float beta, float *d_C);

void run_2d_block_tiling(int M, int N, int K, float alpha,
                     float *d_A, float *d_B,
                     float beta, float *d_C);   
                     
void run_vectorized(int M, int N, int K, float alpha,
                     float *d_A, float *d_B,
                     float beta, float *d_C);                     

// Launches the cuBLAS SGEMM (wrapped as a kernel call)
void run_cublas_sgemm(cublasHandle_t handle, int M, int N, int K, 
                      float alpha, float* d_A, float* d_B, 
                      float beta, float* d_C);

#ifdef __cplusplus
}
#endif

#endif // SGEMM_KERNELS_H
