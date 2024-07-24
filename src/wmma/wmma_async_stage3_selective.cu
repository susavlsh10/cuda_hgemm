#include "common.h"
#include <mma.h>
#include <cuda_fp16.h>

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define BLOCK_ROWS 256
#define BLOCK_COLS 128

#define WARP_ROWS 64
#define WARP_COLS 64

#define BLOCK_ROW_WARPS 2  // BLOCK_COLS / WARP_COLS
#define BLOCK_COL_WARPS 4  // BLOCK_ROWS / WARP_ROWS

#define BLOCK_ROW_TILES 8   // BLOCK_COLS / WMMA_N
#define BLOCK_COL_TILES 16  // BLOCK_ROWS / WMMA_M

#define WARP_ROW_TILES 4  // WARP_COLS / WMMA_N
#define WARP_COL_TILES 4  // WARP_ROWS / WMMA_M

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8      // BLOCK_ROW_WARPS * BLOCK_COL_WARPS
#define THREADS_PER_BLOCK 256  // WARP_SIZE * WARPS_PER_BLOCK

#define CHUNK_K 2  // 32 / WMMA_K

#define THREAD_COPY_BYTES 16

#define CHUNK_LINE_BYTES 64          // CHUNK_K * WMMA_K * sizeof(half)
#define CHUNK_COPY_LINES_PER_WARP 8  // WARP_SIZE * THREAD_COPY_BYTES / CHUNK_LINE_BYTES
#define CHUNK_COPY_LINE_LANES 4      // WARP_SIZE / CHUNK_COPY_LINES_PER_WARP

#define SMEM_PADDING 8

#define AB_SMEM_STRIDE 40  // CHUNK_K * WMMA_K + SMEM_PADDING

#define C_SMEM_STRIDE 136  // BLOCK_COLS + SMEM_PADDING
#define C_SMEM_OFFSET 64   // WARP_COLS

#define BLOCK_STRIDE 16

#define K_STAGE 3

using namespace nvcuda;

__global__ void wmmaAsyncStage3KernelSelective(const half *__restrict__ A, const half *__restrict__ B, half *__restrict__ C,
                                               const int *__restrict__ index_A, const int *__restrict__ index_B,
                                               size_t M, size_t N, size_t K, size_t num_active_A, size_t num_active_B) {
    const size_t M_tiles = div_ceil(M, WMMA_M);
    const size_t N_tiles = div_ceil(N, WMMA_N);
    const size_t K_tiles = div_ceil(K, WMMA_K);

    const size_t block_tile_i =
        (blockIdx.z % 2) ? ((gridDim.y - blockIdx.y - 1) * BLOCK_COL_TILES) : (blockIdx.y * BLOCK_COL_TILES);
    const size_t block_tile_j = (blockIdx.z * gridDim.x + blockIdx.x) * BLOCK_ROW_TILES;

    if (block_tile_i >= num_active_A || block_tile_j >= num_active_B) {
        return;
    }

    extern __shared__ half smem[][AB_SMEM_STRIDE];

    const size_t warp_id = threadIdx.x / WARP_SIZE;
    const size_t lane_id = threadIdx.x % WARP_SIZE;

    constexpr size_t B_smem_idx_off = BLOCK_ROWS;
    constexpr size_t smem_stage_off = BLOCK_ROWS + BLOCK_COLS;

    half *smem_warp_tile_ptr = &smem[0][0] + (warp_id / BLOCK_ROW_WARPS) * C_SMEM_STRIDE * WARP_ROWS +
                               (warp_id % BLOCK_ROW_WARPS) * C_SMEM_OFFSET;

    half *smem_warp_stream_ptr = &smem[0][0] + warp_id * WMMA_M * 2 * C_SMEM_STRIDE;

    const size_t gmem_idx = (block_tile_i + warp_id * 2) * WMMA_M * N + block_tile_j * WMMA_N;
    half *src_gmem_warp_stream_ptr = &C[gmem_idx];

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> C_frag[WARP_COL_TILES][WARP_ROW_TILES];

#pragma unroll
    for (size_t i = 0; i < WARP_COL_TILES; ++i) {
#pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
            wmma::fill_fragment(C_frag[i][j], 0.0);
        }
    }

    const half *A_warp_ptr = &A[index_A[block_tile_i] * K];
    const half *B_warp_ptr = &B[index_B[block_tile_j] * K];

    constexpr size_t A_smem_iters = BLOCK_ROWS / (CHUNK_COPY_LINES_PER_WARP * WARPS_PER_BLOCK);
    constexpr size_t B_smem_iters = BLOCK_COLS / (CHUNK_COPY_LINES_PER_WARP * WARPS_PER_BLOCK);

    size_t smem_store_idx = 0;
    size_t smem_load_idx = 0;

    size_t smem_store_off = 0;
    size_t smem_load_off = 0;

    size_t A_smem_idx = 0;
    int4 *A_lane_ptr = nullptr;

    size_t B_smem_idx = 0;
    int4 *B_lane_ptr = nullptr;

    A_smem_idx = smem_store_off + BLOCK_ROWS / WARPS_PER_BLOCK * warp_id;
    A_lane_ptr = (int4 *)(A_warp_ptr + (lane_id / CHUNK_COPY_LINE_LANES) * K) + (lane_id % CHUNK_COPY_LINE_LANES);
    A_smem_idx += lane_id / CHUNK_COPY_LINE_LANES;

#pragma unroll
    for (size_t i = 0; i < A_smem_iters; ++i) {
        uint32_t A_smem_lane_addr =
            __cvta_generic_to_shared(&smem[A_smem_idx][0]) + (lane_id % CHUNK_COPY_LINE_LANES) * THREAD_COPY_BYTES;

        CP_ASYNC_CG(A_smem_lane_addr, A_lane_ptr, THREAD_COPY_BYTES);

        A_lane_ptr = (int4 *)((half *)A_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
        A_smem_idx += CHUNK_COPY_LINES_PER_WARP;
    }

    B_smem_idx = smem_store_off + B_smem_idx_off + BLOCK_COLS / WARPS_PER_BLOCK * warp_id;
    B_lane_ptr = (int4 *)(B_warp_ptr + (lane_id / CHUNK_COPY_LINE_LANES) * K) + (lane_id % CHUNK_COPY_LINE_LANES);
    B_smem_idx += lane_id / CHUNK_COPY_LINE_LANES;

#pragma unroll
    for (size_t i = 0; i < B_smem_iters; ++i) {
        uint32_t B_smem_lane_addr =
            __cvta_generic_to_shared(&smem[B_smem_idx][0]) + (lane_id % CHUNK_COPY_LINE_LANES) * THREAD_COPY_BYTES;

        CP_ASYNC_CG(B_smem_lane_addr, B_lane_ptr, THREAD_COPY_BYTES);

        B_lane_ptr = (int4 *)((half *)B_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
        B_smem_idx += CHUNK_COPY_LINES_PER_WARP;
    }

    CP_ASYNC_COMMIT_GROUP();

    smem_store_idx = (smem_store_idx + 1) % K_STAGE;
    smem_store_off = smem_store_idx * smem_stage_off;

    A_smem_idx = smem_store_off + BLOCK_ROWS / WARPS_PER_BLOCK * warp_id;
    A_lane_ptr = (int4 *)(A_warp_ptr + CHUNK_K * WMMA_K + (lane_id / CHUNK_COPY_LINE_LANES) * K) +
                 (lane_id % CHUNK_COPY_LINE_LANES);
    A_smem_idx += lane_id / CHUNK_COPY_LINE_LANES;

#pragma unroll
    for (size_t i = 0; i < A_smem_iters; ++i) {
        uint32_t A_smem_lane_addr =
            __cvta_generic_to_shared(&smem[A_smem_idx][0]) + (lane_id % CHUNK_COPY_LINE_LANES) * THREAD_COPY_BYTES;

        CP_ASYNC_CG(A_smem_lane_addr, A_lane_ptr, THREAD_COPY_BYTES);

        A_lane_ptr = (int4 *)((half *)A_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
        A_smem_idx += CHUNK_COPY_LINES_PER_WARP;
    }

    B_smem_idx = smem_store_off + B_smem_idx_off + BLOCK_COLS / WARPS_PER_BLOCK * warp_id;
    B_lane_ptr = (int4 *)(B_warp_ptr + CHUNK_K * WMMA_K + (lane_id / CHUNK_COPY_LINE_LANES) * K) +
                 (lane_id % CHUNK_COPY_LINE_LANES);
    B_smem_idx += lane_id / CHUNK_COPY_LINE_LANES;

#pragma unroll
    for (size_t i = 0; i < B_smem_iters; ++i) {
        uint32_t B_smem_lane_addr =
            __cvta_generic_to_shared(&smem[B_smem_idx][0]) + (lane_id % CHUNK_COPY_LINE_LANES) * THREAD_COPY_BYTES;

        CP_ASYNC_CG(B_smem_lane_addr, B_lane_ptr, THREAD_COPY_BYTES);

        B_lane_ptr = (int4 *)((half *)B_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
        B_smem_idx += CHUNK_COPY_LINES_PER_WARP;
    }

    CP_ASYNC_COMMIT_GROUP();
    CP_ASYNC_WAIT_GROUP(1);

    __syncthreads();

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> A_frag[2][WARP_COL_TILES];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> B_frag[2][WARP_ROW_TILES];

    size_t reg_store

_idx = 0;
    size_t reg_load_idx = 1;

    smem_load_idx = 0;
    smem_load_off = 0;

#pragma unroll
    for (size_t k_step = 0; k_step < K_tiles + CHUNK_K; k_step += CHUNK_K) {
        if (k_step < K_tiles) {
            smem_load_idx = (smem_load_idx + 1) % K_STAGE;
            smem_load_off = smem_load_idx * smem_stage_off;

#pragma unroll
            for (size_t ik = 0; ik < CHUNK_K; ++ik) {
                uint32_t smem_copy_base_addr = smem_load_off + ik * WMMA_K;
#pragma unroll
                for (size_t i = 0; i < WARP_COL_TILES; ++i) {
                    uint32_t A_frag_lane_addr = __cvta_generic_to_shared(
                        &smem[smem_copy_base_addr + i * WMMA_M][warp_id % WARP_COL_TILES * WMMA_N]);

                    wmma::load_matrix_sync(A_frag[reg_load_idx][i], A_frag_lane_addr, AB_SMEM_STRIDE);
                }
#pragma unroll
                for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
                    uint32_t B_frag_lane_addr = __cvta_generic_to_shared(
                        &smem[B_smem_idx_off + smem_copy_base_addr + j * WMMA_N][warp_id % WARP_ROW_TILES * WMMA_M]);

                    wmma::load_matrix_sync(B_frag[reg_load_idx][j], B_frag_lane_addr, AB_SMEM_STRIDE);
                }
                reg_store_idx = (reg_store_idx + 1) % 2;
                reg_load_idx = (reg_load_idx + 1) % 2;
            }
        }

#pragma unroll
        for (size_t ik = 0; ik < CHUNK_K; ++ik) {
#pragma unroll
            for (size_t i = 0; i < WARP_COL_TILES; ++i) {
#pragma unroll
                for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
                    wmma::mma_sync(C_frag[i][j], A_frag[reg_store_idx][i], B_frag[reg_store_idx][j], C_frag[i][j]);
                }
            }
        }
    }

    __syncthreads();

#pragma unroll
    for (size_t i = 0; i < WARP_COL_TILES; ++i) {
#pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
            half *C_smem_warp_tile_ptr = smem_warp_tile_ptr + i * C_SMEM_STRIDE * WMMA_M + j * WMMA_N;

            wmma::store_matrix_sync(C_smem_warp_tile_ptr, C_frag[i][j], C_SMEM_STRIDE, wmma::mem_row_major);
        }
    }

    __syncthreads();

#pragma unroll
    for (size_t i = 0; i < 2; ++i) {
        __pipeline_memcpy_async(smem_warp_stream_ptr + i * C_SMEM_STRIDE * WMMA_M + lane_id * 2,
                                src_gmem_warp_stream_ptr + i * C_SMEM_STRIDE * WMMA_M + lane_id * 2, 16, 0);
    }

    __pipeline_commit();
    __pipeline_wait_prior(0);
}