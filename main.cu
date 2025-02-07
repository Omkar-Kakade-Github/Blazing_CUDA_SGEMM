// main.cu
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <sys/time.h>
#include <cmath>
#include "sgemm_kernels.h"

#define MATRIX_SIZE 4096

// Error-checking macros
#define CHECK_CUDA(call) {                                             \
    cudaError_t err = call;                                            \
    if (err != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__,\
                cudaGetErrorString(err));                              \
        exit(EXIT_FAILURE);                                            \
    }                                                                  \
}

#define CHECK_CUBLAS(call) {                                           \
    cublasStatus_t status = call;                                      \
    if (status != CUBLAS_STATUS_SUCCESS) {                             \
        fprintf(stderr, "cuBLAS error in %s:%d: %d\n", __FILE__, __LINE__, status); \
        exit(EXIT_FAILURE);                                            \
    }                                                                  \
}

// Global cuBLAS handle (used by the wrapper below)
cublasHandle_t g_handle;

// Wrapper for the cuBLAS SGEMM so that it matches the common kernel signature.
void run_cublas_kernel(int M, int N, int K, float alpha, 
                       const float* d_A, const float* d_B, 
                       float beta, float* d_C) {
    // This wrapper calls the function defined in cublas_sgemm.cu using the global handle.
    run_cublas_sgemm(g_handle, M, N, K, alpha, d_A, d_B, beta, d_C);
}

// Returns current time in seconds.
double get_time() {
    struct timeval time;
    gettimeofday(&time, NULL);
    return (time.tv_sec + time.tv_usec * 1e-6);
}

// Fill a matrix with random values in [0, 1)
void fillMatrixRandom(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// Compare two matrices; returns true if they match (within a tolerance).
bool compareResults(const float* ref, const float* test, int size) {
    const float tol = 1e-4f;
    for (int i = 0; i < size; i++) {
        if (fabs(ref[i] - test[i]) > tol) {
            fprintf(stderr, "Mismatch at index %d: ref %f vs test %f\n", 
                    i, ref[i], test[i]);
            return false;
        }
    }
    return true;
}

// Structure to hold a kernel function pointer and its name.
struct KernelFunc {
    const char* name;
    // Function pointer signature for a kernel wrapper.
    void (*kernelFunc)(int, int, int, float, const float*, const float*, float, float*);
};

int main() {
    // Matrix dimensions.
    const int M = MATRIX_SIZE;
    const int N = MATRIX_SIZE;
    const int K = MATRIX_SIZE;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // Allocate host memory.
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C_ref = (float*)malloc(size_C);    // Reference result (computed by cuBLAS)
    float *h_C_kernel = (float*)malloc(size_C);   // Result from each kernel

    // Seed the random number generator and fill A and B.
    srand(time(NULL));
    fillMatrixRandom(h_A, M, K);
    fillMatrixRandom(h_B, K, N);

    // Initialize the reference result to zero.
    for (int i = 0; i < M * N; ++i) {
        h_C_ref[i] = 0.0f;
    }

    // Allocate device memory.
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_A, size_A));
    CHECK_CUDA(cudaMalloc((void**)&d_B, size_B));
    CHECK_CUDA(cudaMalloc((void**)&d_C, size_C));

    // Copy matrices A and B to device memory.
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    // Create cuBLAS handle and assign it to our global handle.
    CHECK_CUBLAS(cublasCreate(&g_handle));

    // --- Compute the reference result with cuBLAS ---
    // (We use this as a correctness check for the kernels.)
    CHECK_CUDA(cudaMemset(d_C, 0, size_C));
    run_cublas_kernel(M, N, K, 1.0f, d_A, d_B, 0.0f, d_C);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_C_ref, d_C, size_C, cudaMemcpyDeviceToHost));

    // Define an array of kernel implementations to test.
    // Make sure to list cuBLAS first so that we can use its time as the reference.
    KernelFunc kernels[] = {
        {"cuBLAS SGEMM", run_cublas_kernel},
        {"Naive SGEMM", run_naive_sgemm},
        {"Global Mem Coalescing", run_global_mem_coalesce},
        {"Shared Mem Block", run_shared_memory_block},
        {"1D Block Tiling", run_1d_block_tiling},
        {"2D Block Tiling", run_2d_block_tiling},
        // Add additional kernels here, e.g. {"Tiled SGEMM", run_tiled_sgemm}
    };
    int numKernels = sizeof(kernels) / sizeof(KernelFunc);

    // Arrays to store benchmark results.
    double* avg_times = new double[numKernels];
    double* gflops_arr = new double[numKernels];

    // For each kernel, perform 5 warmup runs and then 20 benchmark runs.
    for (int k = 0; k < numKernels; k++) {
        printf("-------------------------------------------------\n");
        printf("Running kernel: %s ...\n", kernels[k].name);

        // Warmup runs (not timed).
        for (int i = 0; i < 5; i++) {
            CHECK_CUDA(cudaMemset(d_C, 0, size_C));
            kernels[k].kernelFunc(M, N, K, 1.0f, d_A, d_B, 0.0f, d_C);
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        // Benchmark runs.
        double total_time = 0.0;
        for (int i = 0; i < 20; i++) {
            CHECK_CUDA(cudaMemset(d_C, 0, size_C));
            double start = get_time();
            kernels[k].kernelFunc(M, N, K, 1.0f, d_A, d_B, 0.0f, d_C);
            CHECK_CUDA(cudaDeviceSynchronize());
            double end = get_time();
            total_time += (end - start);
        }
        double avg_time = total_time / 20.0;
        avg_times[k] = avg_time;
        double gflops = ((2.0 * M * N * K) + (M * N)) / (avg_time * 1e9);
        gflops_arr[k] = gflops;

        // Copy the result from the last run back to host.
        CHECK_CUDA(cudaMemcpy(h_C_kernel, d_C, size_C, cudaMemcpyDeviceToHost));

        // Verify that the result matches the reference.
        bool correct = compareResults(h_C_ref, h_C_kernel, M * N);
        if (correct) {
            printf("Completed kernel: %s\n", kernels[k].name);
            printf("   Average Time: %f sec, GFLOPS: %f\n", avg_time, gflops);
        } else {
            printf("Kernel %s produced an incorrect result!\n", kernels[k].name);
        }
    }

    // Determine the average time for cuBLAS (assumed to be the first kernel).
    double cublas_avg_time = avg_times[0];

    // Print a summary of performance relative to cuBLAS.
    printf("\n================== Performance Summary ==================\n");
    printf("%-15s %-15s %-15s %-20s\n", "Kernel", "Avg Time (sec)", "GFLOPS", "Relative Performance (%)");
    for (int k = 0; k < numKernels; k++) {
        double relative_perf = (cublas_avg_time / avg_times[k]) * 100.0;
        printf("%-15s %-15f %-15f %-20f\n", kernels[k].name, avg_times[k], gflops_arr[k], relative_perf);
    }
    printf("Note: Relative performance is computed as (cuBLAS_time / kernel_time)*100, so cuBLAS should be 100%%.\n");

    // Clean up.
    delete[] avg_times;
    delete[] gflops_arr;
    free(h_A);
    free(h_B);
    free(h_C_ref);
    free(h_C_kernel);
    CHECK_CUBLAS(cublasDestroy(g_handle));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}
