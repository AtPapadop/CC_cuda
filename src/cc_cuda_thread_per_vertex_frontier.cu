// src/cc_cuda_thread_frontier_async.cu
//
// Thread-per-vertex, FRONTIER-based, ASYNCHRONOUS (in-place) relax + push.
//
// This mirrors the “fast” pthreads behavior more closely than synchronous pull:
//  - Maintain an active frontier (bitset).
//  - For each active vertex u:
//      (optional) relax u from neighbors (pull-min)
//      push labels[u] to neighbors via atomicMin
//      activate neighbors that actually changed
//  - Terminate when next frontier is empty.
//
// Correctness note (important):
//  - We DO NOT only push when u decreases in this kernel. If u is active, we push its
//    current label, because u may have been decreased earlier by someone else’s push.
//    Gating pushes on "u decreased here" can terminate early with wrong CC counts.
//
// Uses uint32 labels; CSR row_ptr is uint64; col_idx is uint32.
// Frontier bitset uses uint32 words.
//
// API:
//   int compute_connected_components_cuda_thread_frontier_async(...)
// Add prototype in include/cc_cuda.h.

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

/**
 * Initialize labels on device: labels[i] = i
 */
__global__ static void init_labels_kernel(uint32_t n, uint32_t *labels)
{
    uint32_t idx = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < n)
        labels[idx] = idx;
}

static __device__ __forceinline__ uint32_t bit_word(uint32_t v) { return v >> 5; }
static __device__ __forceinline__ uint32_t bit_mask(uint32_t v) { return 1u << (v & 31u); }

/**
 * Thread-per-vertex frontier async kernel.
 *
 * If u is active:
 *   - relax u from neighbors (pull-min) and apply atomicMin to labels[u]
 *   - read lu = labels[u]
 *   - push lu to neighbors (atomicMin)
 *   - if neighbor changed, set its bit in next frontier
 *
 * any_next is set to 1 if any vertex is activated for next round.
 */
__global__ static void cc_frontier_async_thread_kernel(
    const uint32_t n,
    const uint64_t *row_ptr,
    const uint32_t *col_idx,
    uint32_t *labels,            // in-place
    const uint32_t *active_bits, // bitset words
    uint32_t *next_active_bits,  // bitset words
    unsigned int *any_next)      // flag
{
    const uint32_t u = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (u >= n)
        return;

    const uint32_t w = bit_word(u);
    const uint32_t m = bit_mask(u);
    if ((active_bits[w] & m) == 0u)
        return;

    const uint64_t start = row_ptr[u];
    const uint64_t end = row_ptr[u + 1];

    // Optional relax (pull) step: try to lower labels[u] from neighbor labels.
    uint32_t old_u = labels[u];
    uint32_t best = old_u;

    for (uint64_t e = start; e < end; ++e)
    {
        const uint32_t v = col_idx[e];
        const uint32_t lv = labels[v]; // async read
        if (lv < best)
            best = lv;
    }

    if (best < old_u)
    {
        atomicMin(reinterpret_cast<unsigned int *>(&labels[u]),
                  static_cast<unsigned int>(best));
    }

    // Push current u label to neighbors; activate neighbors that changed.
    const uint32_t lu = labels[u];

    unsigned int local_any = 0u;

    for (uint64_t e = start; e < end; ++e)
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
            local_any = 1u;
        }
    }

    if (local_any != 0u)
        atomicExch(any_next, 1u);
}

int compute_connected_components_cuda_thread_frontier_async(
    const CSRGraph *G,
    uint32_t *labels,
    const CudaCCOptions *opt,
    uint32_t *out_iter,
    double *out_kernel_seconds)
{
    if (!G || !labels || !G->row_ptr || !G->col_idx)
    {
        std::fprintf(stderr, "Invalid input to compute_connected_components_cuda_thread_frontier_async\n");
        return 2;
    }

    const uint32_t n = G->n;
    if (n == 0)
    {
        if (out_iter)
            *out_iter = 0;
        if (out_kernel_seconds)
            *out_kernel_seconds = 0.0;
        return 0;
    }

    int device_id = 0;
    int block_size = CC_CUDA_DEFAULT_BLOCK;
    int verbose = 0;

    if (opt)
    {
        device_id = opt->device_id >= 0 ? opt->device_id : device_id;
        block_size = opt->block_size > 0 ? opt->block_size : block_size;
        verbose = opt->verbose;
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
    const size_t labels_bytes = static_cast<size_t>(n) * sizeof(uint32_t);

    const uint32_t words =
        static_cast<uint32_t>((static_cast<uint64_t>(n) + 31ull) / 32ull);
    const size_t bits_bytes = static_cast<size_t>(words) * sizeof(uint32_t);

    const uint32_t grid_size =
        static_cast<uint32_t>((static_cast<uint64_t>(n) + static_cast<uint64_t>(block_size) - 1ull) /
                              static_cast<uint64_t>(block_size));

    if (verbose)
    {
        std::fprintf(stderr, "[cc-cuda][frontier-async-thread] n=%" PRIu32 ", m=%" PRIu64 "\n", G->n, G->m);
        std::fprintf(stderr, "[cc-cuda][frontier-async-thread] labels=%.3f MB, frontier_bits=%.3f MB (x2)\n",
                     labels_bytes / (1024.0 * 1024.0),
                     bits_bytes / (1024.0 * 1024.0));
        std::fprintf(stderr, "[cc-cuda][frontier-async-thread] block_size=%d\n", block_size);
    }
    uint32_t iters = 0;

    // Allocate
    if (cc_cuda_check(cudaMalloc(reinterpret_cast<void **>(&d_row_ptr), row_ptr_bytes), "allocating d_row_ptr") != 0 ||
        cc_cuda_check(cudaMalloc(reinterpret_cast<void **>(&d_col_idx), col_idx_bytes), "allocating d_col_idx") != 0 ||
        cc_cuda_check(cudaMalloc(reinterpret_cast<void **>(&d_labels), labels_bytes), "allocating d_labels") != 0 ||
        cc_cuda_check(cudaMalloc(reinterpret_cast<void **>(&d_active), bits_bytes), "allocating d_active") != 0 ||
        cc_cuda_check(cudaMalloc(reinterpret_cast<void **>(&d_next_active), bits_bytes), "allocating d_next_active") != 0 ||
        cc_cuda_check(cudaMalloc(reinterpret_cast<void **>(&d_any_next), sizeof(unsigned int)), "allocating d_any_next") != 0)
    {
        rc = 4;
        goto cleanup;
    }

    // Copy graph
    if (cc_cuda_check(cudaMemcpy(d_row_ptr, G->row_ptr, row_ptr_bytes, cudaMemcpyHostToDevice), "copying row_ptr to device") != 0 ||
        cc_cuda_check(cudaMemcpy(d_col_idx, G->col_idx, col_idx_bytes, cudaMemcpyHostToDevice), "copying col_idx to device") != 0)
    {
        rc = 5;
        goto cleanup;
    }

    // Init labels
    init_labels_kernel<<<grid_size, static_cast<uint32_t>(block_size)>>>(n, d_labels);
    if (cc_cuda_check(cudaGetLastError(), "launching init_labels_kernel") != 0 ||
        cc_cuda_check(cudaDeviceSynchronize(), "synchronizing after init_labels_kernel") != 0)
    {
        rc = 6;
        goto cleanup;
    }

    // Init frontier: all vertices active
    if (cc_cuda_check(cudaMemset(d_active, 0xFF, bits_bytes), "memset d_active") != 0 ||
        cc_cuda_check(cudaMemset(d_next_active, 0x00, bits_bytes), "memset d_next_active") != 0 ||
        cc_cuda_check(cudaMemset(d_any_next, 0x00, sizeof(unsigned int)), "memset d_any_next") != 0)
    {
        rc = 7;
        goto cleanup;
    }

    if (cc_cuda_check(cudaEventCreate(&event_start), "creating event_start") != 0 ||
        cc_cuda_check(cudaEventCreate(&event_stop), "creating event_stop") != 0)
    {
        rc = 8;
        goto cleanup;
    }

    std::printf("Starting frontier-async thread-per-vertex iterations...\n");

    for (;;)
    {
        // Clear next frontier + flag
        if (cc_cuda_check(cudaMemset(d_next_active, 0x00, bits_bytes), "clearing d_next_active") != 0 ||
            cc_cuda_check(cudaMemset(d_any_next, 0x00, sizeof(unsigned int)), "clearing d_any_next") != 0)
        {
            rc = 9;
            goto cleanup;
        }

        cudaEventRecord(event_start, 0);

        cc_frontier_async_thread_kernel<<<grid_size, static_cast<uint32_t>(block_size)>>>(
            n, d_row_ptr, d_col_idx, d_labels, d_active, d_next_active, d_any_next);

        if (cc_cuda_check(cudaGetLastError(), "launching cc_frontier_async_thread_kernel") != 0 ||
            cc_cuda_check(cudaDeviceSynchronize(), "synchronizing after cc_frontier_async_thread_kernel") != 0)
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
                          "copying d_any_next") != 0)
        {
            rc = 11;
            goto cleanup;
        }

        ++iters;

        if (verbose && (iters % 5u) == 0u)
        {
            std::printf("  iter=%" PRIu32 " (last kernel %.3f ms) any_next=%u\n",
                        iters, kernel_ms, h_any_next);
            std::fflush(stdout);
        }

        if (h_any_next == 0u)
            break;

        // Swap frontiers
        uint32_t *tmp = d_active;
        d_active = d_next_active;
        d_next_active = tmp;
    }

    if (cc_cuda_check(cudaMemcpy(labels, d_labels, labels_bytes, cudaMemcpyDeviceToHost), "copying labels to host") != 0)
    {
        rc = 12;
        goto cleanup;
    }

    if (out_iter)
        *out_iter = iters;
    if (out_kernel_seconds)
        *out_kernel_seconds = static_cast<double>(kernel_ms_total) / 1000.0;

cleanup:
    if (event_start)
        cudaEventDestroy(event_start);
    if (event_stop)
        cudaEventDestroy(event_stop);

    cudaFree(d_any_next);
    cudaFree(d_next_active);
    cudaFree(d_active);
    cudaFree(d_labels);
    cudaFree(d_col_idx);
    cudaFree(d_row_ptr);

    return rc;
}
