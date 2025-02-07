#include "sgemm_kernels.h"
#include <cstdio>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

// CUDA kernel for single-precision matrix multiplication using shared memory
__global__ void sgemm_shared_memory_block(int num_rows_C, int num_cols_C, int inner_dim, float alpha,
                                            const float *matrix_A, const float *matrix_B,
                                            float beta, float *matrix_C) {
    // The row and column of the output block this thread block is responsible for
    const unsigned int output_block_row = blockIdx.x;
    const unsigned int output_block_col = blockIdx.y;

    // Shared memory buffers to store tiles of matrix_A and matrix_B
    __shared__ float tile_A[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float tile_B[BLOCK_SIZE * BLOCK_SIZE];

    // The row and column of the thread within the block
    const unsigned int thread_col = threadIdx.x % BLOCK_SIZE;
    const unsigned int thread_row = threadIdx.x / BLOCK_SIZE;

    // Move pointers to the starting positions of the current block.
    // For matrix_A, advance by (output_block_row * BLOCK_SIZE) rows.
    // For matrix_B, advance by (output_block_col * BLOCK_SIZE) columns.
    matrix_A += output_block_row * BLOCK_SIZE * inner_dim;
    matrix_B += output_block_col * BLOCK_SIZE;
    matrix_C += output_block_row * BLOCK_SIZE * num_cols_C + output_block_col * BLOCK_SIZE;

    // Accumulator for the dot product.
    float dot_product_result = 0.0f;

    // Loop over the inner dimension in tiles of BLOCK_SIZE.
    // Note: This code assumes that inner_dim is a multiple of BLOCK_SIZE.
    for (int tile_idx = 0; tile_idx < inner_dim; tile_idx += BLOCK_SIZE) {
        // Each thread loads one element from matrix_A and matrix_B into shared memory.
        // The pointer arithmetic uses the current matrix_A and matrix_B pointers.
        tile_A[thread_row * BLOCK_SIZE + thread_col] = matrix_A[thread_row * inner_dim + thread_col];
        tile_B[thread_row * BLOCK_SIZE + thread_col] = matrix_B[thread_row * num_cols_C + thread_col];

        // Synchronize to ensure that the entire tile is loaded.
        __syncthreads();

        // Compute the dot product for this tile.
        for (int dot_idx = 0; dot_idx < BLOCK_SIZE; ++dot_idx) {
            dot_product_result += tile_A[thread_row * BLOCK_SIZE + dot_idx] *
                                  tile_B[dot_idx * BLOCK_SIZE + thread_col];
        }

        // Synchronize before loading the next tile.
        __syncthreads();

        // Advance the pointers to the next tile.
        matrix_A += BLOCK_SIZE;
        matrix_B += BLOCK_SIZE * num_cols_C;
    }

    // Write the computed value back to matrix_C (with scaling by alpha and beta).
    matrix_C[thread_row * num_cols_C + thread_col] =
        alpha * dot_product_result + beta * matrix_C[thread_row * num_cols_C + thread_col];
}

// Wrapper function to launch the shared memory SGEMM kernel.
// It follows the same signature as your other kernel launchers.
void run_shared_memory_block(int num_rows_C, int num_cols_C, int inner_dim, 
                             float alpha, const float* d_A, const float* d_B, 
                             float beta, float* d_C) {
    // Each thread block will have BLOCK_SIZE*BLOCK_SIZE threads.
    dim3 blockDim(BLOCK_SIZE * BLOCK_SIZE);
    // Grid dimensions are set so that each block computes a BLOCK_SIZE x BLOCK_SIZE tile.
    dim3 gridDim((num_rows_C + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (num_cols_C + BLOCK_SIZE - 1) / BLOCK_SIZE);
    sgemm_shared_memory_block<<<gridDim, blockDim>>>(num_rows_C, num_cols_C, inner_dim,
                                                     alpha, d_A, d_B, beta, d_C);
}
