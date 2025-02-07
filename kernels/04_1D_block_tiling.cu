#include "sgemm_kernels.h"
#include <cstdio>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BN, const int BK, const int TM>
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

void run_1d_block_tiling(int M, int N, int K, float alpha, const float *d_A,
                         const float *d_B, float beta, float *d_C) {

    const uint BM = 64;
    const uint BN = 64;
    const uint BK = 8;
    const uint TM = 8;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / TM);                        
    sgemm1DBlockTiling<BM, BN, BK, TM>
        <<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
}
