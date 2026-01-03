// src/cc_cuda_warp_frontier_async.cu
//
// Warp-per-vertex, FRONTIER-based, ASYNCHRONOUS (in-place) relax + push (CORRECTED).
//
// Key change vs earlier buggy frontier version:
//  - If u is active, we ALWAYS push labels[u] to neighbors, and activate neighbors that
//    actually changed (atomicMin returned > labels[u]).
//  - We still optionally do a "pull relax" step to maybe lower labels[u] from neighbors first,
//    but pushing does not depend on "u decreased in this kernel".
//
// This avoids early termination where a vertex u gets lowered by someone else’s push
// but never propagates to its other neighbors.
//
// Mapping is overflow-safe:
//   u = blockIdx.x * warps_per_block + warp_in_block
//
// Frontier is a bitset (uint32 words). next frontier contains vertices whose label changed.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cinttypes>

#include <cuda_runtime.h>

#include "cc_cuda.h"
#include "graph.h"

#ifndef CC_CUDA_DEFAULT_BLOCK
#define CC_CUDA_DEFAULT_BLOCK 256
#endif

static inline int cc_cuda_check(cudaError_t err, const char *what)
{
    if (err != cudaSuccess)
    {
        std::fprintf(stderr, "CUDA error during %s: %s\n", what, cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

__global__ static void init_labels_kernel(uint32_t n, uint32_t *labels)
{
    uint32_t idx = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < n) labels[idx] = idx;
}

static __device__ __forceinline__ uint32_t warp_reduce_min_u32(uint32_t x)
{
    const unsigned mask = 0xFFFFFFFFu;
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        uint32_t y = __shfl_down_sync(mask, x, offset);
        x = (y < x) ? y : x;
    }
    return x;
}

static __device__ __forceinline__ uint32_t bit_word(uint32_t v) { return v >> 5; }
static __device__ __forceinline__ uint32_t bit_mask(uint32_t v) { return 1u << (v & 31u); }

/**
 * Correct frontier async kernel:
 *   if u active:
 *     - (optional) relax labels[u] from neighbors (pull-min)
 *     - push labels[u] to neighbors (atomicMin)
 *     - if any neighbor label changed, mark neighbor active next round and set any_next
 */
__global__ static void cc_frontier_async_warp_kernel(
    const uint32_t n,
    const uint64_t *row_ptr,
    const uint32_t *col_idx,
    uint32_t *labels,             // in-place
    const uint32_t *active_bits,  // bitset words
    uint32_t *next_active_bits,   // bitset words
    unsigned int *any_next)       // flag (1 if next frontier non-empty)
{
    const uint32_t lane = static_cast<uint32_t>(threadIdx.x) & 31u;
    const uint32_t warp_in_block = static_cast<uint32_t>(threadIdx.x) >> 5;
    const uint32_t warps_per_block = static_cast<uint32_t>(blockDim.x) >> 5;

    const uint32_t u = static_cast<uint32_t>(blockIdx.x) * warps_per_block + warp_in_block;
    if (u >= n) return;

    // Frontier check: lane0 reads, broadcast
    uint32_t is_active = 0u;
    if (lane == 0u)
    {
        const uint32_t w = bit_word(u);
        const uint32_t m = bit_mask(u);
        is_active = ((active_bits[w] & m) != 0u) ? 1u : 0u;
    }
    is_active = __shfl_sync(0xFFFFFFFFu, is_active, 0);
    if (is_active == 0u) return;

    const uint64_t start = row_ptr[u];
    const uint64_t end   = row_ptr[u + 1];

    // --- Optional RELAX (pull) step: try to lower u from neighbors first ---
    uint32_t old_u = labels[u];
    uint32_t best = old_u;

    for (uint64_t e = start + static_cast<uint64_t>(lane); e < end; e += 32ull)
    {
        const uint32_t v = col_idx[e];
        const uint32_t lv = labels[v];  // async read
        if (lv < best) best = lv;
    }

    best = warp_reduce_min_u32(best);
    best = __shfl_sync(0xFFFFFFFFu, best, 0);

    if (lane == 0u && best < old_u)
    {
        atomicMin(reinterpret_cast<unsigned int *>(&labels[u]),
                  static_cast<unsigned int>(best));
    }

    // Load u label after potential relaxation and broadcast to warp
    uint32_t lu = 0u;
    if (lane == 0u) lu = labels[u];
    lu = __shfl_sync(0xFFFFFFFFu, lu, 0);

    // --- PUSH step: always push lu to neighbors; activate neighbors that changed ---
    unsigned int any_changed_in_warp = 0u;

    for (uint64_t e = start + static_cast<uint64_t>(lane); e < end; e += 32ull)
    {
        const uint32_t v = col_idx[e];

        const unsigned int prev =
            atomicMin(reinterpret_cast<unsigned int *>(&labels[v]),
                      static_cast<unsigned int>(lu));

        if (static_cast<uint32_t>(prev) > lu)
        {
            const uint32_t w2 = bit_word(v);
            const uint32_t m2 = bit_mask(v);
            atomicOr(reinterpret_cast<unsigned int *>(&next_active_bits[w2]),
                     static_cast<unsigned int>(m2));
            any_changed_in_warp = 1u;
        }
    }

    // Warp-wide OR to see if any lane changed something
    const unsigned int mask = 0xFFFFFFFFu;
    const unsigned int changed_mask = __ballot_sync(mask, any_changed_in_warp != 0u);

    if (lane == 0u && changed_mask != 0u)
    {
        atomicExch(any_next, 1u);
    }
}

int compute_connected_components_cuda_warp_frontier_async(
    const CSRGraph *G,
    uint32_t *labels,
    const CudaCCOptions *opt,
    uint32_t *out_iter,
    double *out_kernel_seconds)
{
    if (!G || !labels || !G->row_ptr || !G->col_idx)
    {
        std::fprintf(stderr, "Invalid input to compute_connected_components_cuda_warp_frontier_async\n");
        return 2;
    }

    const uint32_t n = G->n;
    if (n == 0)
    {
        if (out_iter) *out_iter = 0;
        if (out_kernel_seconds) *out_kernel_seconds = 0.0;
        return 0;
    }

    int device_id = 0;
    int block_size = CC_CUDA_DEFAULT_BLOCK;
    int verbose = 0;

    if (opt)
    {
        device_id = (opt->device_id >= 0) ? opt->device_id : device_id;
        block_size = (opt->block_size > 0) ? opt->block_size : block_size;
        verbose = opt->verbose;
    }

    // Must be multiple of 32
    if ((block_size & 31) != 0)
    {
        int rounded = block_size & ~31;
        if (rounded < 32) rounded = 256;
        if (verbose)
            std::fprintf(stderr, "[cc-cuda][frontier] block_size %d not multiple of 32, using %d\n",
                         block_size, rounded);
        block_size = rounded;
    }

    if (cc_cuda_check(cudaSetDevice(device_id), "setting CUDA device") != 0)
        return 3;

    uint64_t *d_row_ptr = nullptr;
    uint32_t *d_col_idx = nullptr;
    uint32_t *d_labels = nullptr;

    uint32_t *d_active = nullptr;
    uint32_t *d_next_active = nullptr;

    unsigned int *d_any_next = nullptr;
    unsigned int h_any_next = 1u;

    cudaEvent_t event_start = nullptr, event_stop = nullptr;
    float kernel_ms_total = 0.0f;

    int rc = 0;

    const size_t row_ptr_bytes = static_cast<size_t>(n + 1u) * sizeof(uint64_t);
    const size_t col_idx_bytes = static_cast<size_t>(G->m) * sizeof(uint32_t);
    const size_t labels_bytes  = static_cast<size_t>(n) * sizeof(uint32_t);

    const uint32_t words = static_cast<uint32_t>((static_cast<uint64_t>(n) + 31ull) / 32ull);
    const size_t bits_bytes = static_cast<size_t>(words) * sizeof(uint32_t);

    if (verbose)
    {
        std::fprintf(stderr, "[cc-cuda][frontier-async-warp] n=%" PRIu32 ", m=%" PRIu64 "\n", G->n, G->m);
        std::fprintf(stderr, "[cc-cuda][frontier-async-warp] labels=%.3f MB, frontier_bits=%.3f MB (x2)\n",
                     labels_bytes / (1024.0 * 1024.0),
                     bits_bytes / (1024.0 * 1024.0));
        std::fprintf(stderr, "[cc-cuda][frontier-async-warp] block_size=%d\n", block_size);
    }

    const uint32_t warps_per_block = static_cast<uint32_t>(block_size) >> 5;
    const uint32_t grid_size =
        static_cast<uint32_t>((static_cast<uint64_t>(n) + static_cast<uint64_t>(warps_per_block) - 1ull) /
                              static_cast<uint64_t>(warps_per_block));

    uint32_t iters = 0;
    if (cc_cuda_check(cudaMalloc(reinterpret_cast<void **>(&d_row_ptr), row_ptr_bytes), "cudaMalloc(d_row_ptr)") != 0 ||
        cc_cuda_check(cudaMalloc(reinterpret_cast<void **>(&d_col_idx), col_idx_bytes), "cudaMalloc(d_col_idx)") != 0 ||
        cc_cuda_check(cudaMalloc(reinterpret_cast<void **>(&d_labels), labels_bytes), "cudaMalloc(d_labels)") != 0 ||
        cc_cuda_check(cudaMalloc(reinterpret_cast<void **>(&d_active), bits_bytes), "cudaMalloc(d_active)") != 0 ||
        cc_cuda_check(cudaMalloc(reinterpret_cast<void **>(&d_next_active), bits_bytes), "cudaMalloc(d_next_active)") != 0 ||
        cc_cuda_check(cudaMalloc(reinterpret_cast<void **>(&d_any_next), sizeof(unsigned int)), "cudaMalloc(d_any_next)") != 0)
    {
        rc = 4;
        goto cleanup;
    }

    if (cc_cuda_check(cudaMemcpy(d_row_ptr, G->row_ptr, row_ptr_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(row_ptr)") != 0 ||
        cc_cuda_check(cudaMemcpy(d_col_idx, G->col_idx, col_idx_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(col_idx)") != 0)
    {
        rc = 5;
        goto cleanup;
    }

    // init labels[i]=i
    {
        const uint32_t init_grid =
            static_cast<uint32_t>((static_cast<uint64_t>(n) + static_cast<uint64_t>(block_size) - 1ull) /
                                  static_cast<uint64_t>(block_size));
        init_labels_kernel<<<init_grid, static_cast<uint32_t>(block_size)>>>(n, d_labels);
        if (cc_cuda_check(cudaGetLastError(), "launch init_labels_kernel") != 0 ||
            cc_cuda_check(cudaDeviceSynchronize(), "sync init_labels_kernel") != 0)
        {
            rc = 6;
            goto cleanup;
        }
    }

    // initial frontier: all vertices active
    if (cc_cuda_check(cudaMemset(d_active, 0xFF, bits_bytes), "memset d_active") != 0 ||
        cc_cuda_check(cudaMemset(d_next_active, 0x00, bits_bytes), "memset d_next_active") != 0)
    {
        rc = 7;
        goto cleanup;
    }

    if (cc_cuda_check(cudaEventCreate(&event_start), "cudaEventCreate(start)") != 0 ||
        cc_cuda_check(cudaEventCreate(&event_stop), "cudaEventCreate(stop)") != 0)
    {
        rc = 8;
        goto cleanup;
    }

    for (;;)
    {
        if (cc_cuda_check(cudaMemset(d_next_active, 0x00, bits_bytes), "clear d_next_active") != 0 ||
            cc_cuda_check(cudaMemset(d_any_next, 0x00, sizeof(unsigned int)), "clear d_any_next") != 0)
        {
            rc = 9;
            goto cleanup;
        }

        cudaEventRecord(event_start, 0);

        cc_frontier_async_warp_kernel<<<grid_size, static_cast<uint32_t>(block_size)>>>(
            n, d_row_ptr, d_col_idx, d_labels, d_active, d_next_active, d_any_next);

        if (cc_cuda_check(cudaGetLastError(), "launch cc_frontier_async_warp_kernel") != 0 ||
            cc_cuda_check(cudaDeviceSynchronize(), "sync cc_frontier_async_warp_kernel") != 0)
        {
            rc = 10;
            goto cleanup;
        }

        cudaEventRecord(event_stop, 0);
        cudaEventSynchronize(event_stop);

        float kernel_ms = 0.0f;
        cudaEventElapsedTime(&kernel_ms, event_start, event_stop);
        kernel_ms_total += kernel_ms;

        if (cc_cuda_check(cudaMemcpy(&h_any_next, d_any_next, sizeof(unsigned int), cudaMemcpyDeviceToHost),
                          "copy d_any_next") != 0)
        {
            rc = 11;
            goto cleanup;
        }

        ++iters;

        if (verbose && (iters % 5u) == 0u)
        {
            std::printf("  iter=%" PRIu32 " last=%.3fms any_next=%u\n", iters, kernel_ms, h_any_next);
            std::fflush(stdout);
        }

        if (h_any_next == 0u) break;

        uint32_t *tmp = d_active;
        d_active = d_next_active;
        d_next_active = tmp;
    }

    if (cc_cuda_check(cudaMemcpy(labels, d_labels, labels_bytes, cudaMemcpyDeviceToHost), "copy labels back") != 0)
    {
        rc = 12;
        goto cleanup;
    }

    if (out_iter) *out_iter = iters;
    if (out_kernel_seconds) *out_kernel_seconds = static_cast<double>(kernel_ms_total) / 1000.0;

cleanup:
    if (event_start) cudaEventDestroy(event_start);
    if (event_stop)  cudaEventDestroy(event_stop);

    cudaFree(d_any_next);
    cudaFree(d_next_active);
    cudaFree(d_active);
    cudaFree(d_labels);
    cudaFree(d_col_idx);
    cudaFree(d_row_ptr);

    return rc;
}
