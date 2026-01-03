// src/cc_cuda_block_per_vertex.cu
//
// Block-per-vertex (cooperative), PULL-style (double-buffer) min-label propagation.
// No atomics on labels. Each iteration computes:
//   next[v] = min(curr[v], min_{u in N(v)} curr[u])
// and sets a global changed flag if any v changed.
//
// Mapping:
//   - Each CUDA block processes one vertex at a time using all threads cooperatively.
//   - To avoid launching n blocks (too many for huge n), we use a grid-stride loop:
//       for (v = blockIdx.x; v < n; v += gridDim.x) { ... }
//
// This is a good baseline "block-cooperative pull" implementation.

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

static __device__ __forceinline__ uint32_t u32_min(uint32_t a, uint32_t b)
{
    return (b < a) ? b : a;
}

/**
 * Block-cooperative pull kernel (grid-stride over vertices).
 *
 * Shared memory: blockDim.x * sizeof(uint32_t) bytes.
 */
__global__ static void lp_pull_block_per_vertex_kernel(
    const uint32_t n,
    const uint64_t *row_ptr,
    const uint32_t *col_ind,
    const uint32_t *curr,
    uint32_t *next,
    unsigned int *changed)
{
    extern __shared__ uint32_t s_min[];

    const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
    const uint32_t stride_v = static_cast<uint32_t>(gridDim.x);

    for (uint32_t v = static_cast<uint32_t>(blockIdx.x); v < n; v += stride_v)
    {
        const uint64_t row_start = row_ptr[v];
        const uint64_t row_end = row_ptr[v + 1];

        uint32_t local = curr[v];

        // Cooperative scan over adjacency list
        for (uint64_t e = row_start + static_cast<uint64_t>(tid); e < row_end; e += static_cast<uint64_t>(blockDim.x))
        {
            const uint32_t u = col_ind[e];
            const uint32_t lu = curr[u];
            local = u32_min(local, lu);
        }

        // Reduce within block
        s_min[tid] = local;
        __syncthreads();

        // Parallel reduction (min)
        for (uint32_t offset = static_cast<uint32_t>(blockDim.x) >> 1; offset > 0; offset >>= 1)
        {
            if (tid < offset)
            {
                s_min[tid] = u32_min(s_min[tid], s_min[tid + offset]);
            }
            __syncthreads();
        }

        if (tid == 0u)
        {
            const uint32_t best = s_min[0];
            next[v] = best;
            if (best != curr[v])
                atomicExch(changed, 1u);
        }

        __syncthreads();
    }
}

int compute_connected_components_cuda_block_per_vertex(
    const CSRGraph *G,
    uint32_t *labels,
    const CudaCCOptions *opt,
    uint32_t *out_iter,
    double *out_kernel_seconds)
{
    if (!G || !labels || !G->row_ptr || !G->col_idx)
    {
        std::fprintf(stderr, "Invalid input to compute_connected_components_cuda_block_per_vertex\n");
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
        device_id = (opt->device_id >= 0) ? opt->device_id : device_id;
        block_size = (opt->block_size > 0) ? opt->block_size : block_size;
        verbose = opt->verbose;
    }

    if (cc_cuda_check(cudaSetDevice(device_id), "setting CUDA device") != 0)
        return 3;

    // Heuristic: cap grid size to a small multiple of SMs to avoid absurdly large launches.
    cudaDeviceProp prop;
    if (cc_cuda_check(cudaGetDeviceProperties(&prop, device_id), "getting device properties") != 0)
        return 3;

    uint32_t grid_size = 0;
    {
        const uint32_t sm = static_cast<uint32_t>(prop.multiProcessorCount > 0 ? prop.multiProcessorCount : 1);
        const uint32_t cap = sm * 8u; // tuneable: 4-16 typically fine
        grid_size = (n < cap) ? n : cap;
        if (grid_size == 0u)
            grid_size = 1u;
    }

    uint64_t *d_row_ptr = nullptr;
    uint32_t *d_col_idx = nullptr;

    uint32_t *d_curr = nullptr;
    uint32_t *d_next = nullptr;

    unsigned int *d_changed = nullptr;

    cudaEvent_t event_start = nullptr, event_stop = nullptr;
    float kernel_ms_total = 0.0f;

    int rc = 0;

    const size_t row_ptr_bytes = static_cast<size_t>(n + 1u) * sizeof(uint64_t);
    const size_t col_idx_bytes = static_cast<size_t>(G->m) * sizeof(uint32_t);
    const size_t labels_bytes = static_cast<size_t>(n) * sizeof(uint32_t);
    const size_t changed_bytes = sizeof(unsigned int);

    if (verbose)
    {
        std::fprintf(stderr, "[cc-cuda][block-pull] n=%" PRIu32 ", m=%" PRIu64 "\n", G->n, G->m);
        std::fprintf(stderr, "[cc-cuda][block-pull] row_ptr=%.3f MB, col_idx=%.3f MB, labels=%.3f MB (x2)\n",
                     row_ptr_bytes / (1024.0 * 1024.0),
                     col_idx_bytes / (1024.0 * 1024.0),
                     labels_bytes / (1024.0 * 1024.0));
        std::fprintf(stderr, "[cc-cuda][block-pull] block_size=%d, grid_size=%" PRIu32 " (cap based on SMs)\n",
                     block_size, grid_size);
    }

    uint32_t iters = 0;
    unsigned int h_changed = 1u;
    const size_t shared_bytes = static_cast<size_t>(block_size) * sizeof(uint32_t);

    if (cc_cuda_check(cudaMalloc(reinterpret_cast<void **>(&d_row_ptr), row_ptr_bytes), "allocating d_row_ptr") != 0 ||
        cc_cuda_check(cudaMalloc(reinterpret_cast<void **>(&d_col_idx), col_idx_bytes), "allocating d_col_idx") != 0 ||
        cc_cuda_check(cudaMalloc(reinterpret_cast<void **>(&d_curr), labels_bytes), "allocating d_curr") != 0 ||
        cc_cuda_check(cudaMalloc(reinterpret_cast<void **>(&d_next), labels_bytes), "allocating d_next") != 0 ||
        cc_cuda_check(cudaMalloc(reinterpret_cast<void **>(&d_changed), changed_bytes), "allocating d_changed") != 0)
    {
        rc = 4;
        goto cleanup;
    }

    if (cc_cuda_check(cudaMemcpy(d_row_ptr, G->row_ptr, row_ptr_bytes, cudaMemcpyHostToDevice), "copying row_ptr to device") != 0 ||
        cc_cuda_check(cudaMemcpy(d_col_idx, G->col_idx, col_idx_bytes, cudaMemcpyHostToDevice), "copying col_idx to device") != 0)
    {
        rc = 5;
        goto cleanup;
    }

    // init curr labels
    {
        const uint32_t init_grid =
            static_cast<uint32_t>((static_cast<uint64_t>(n) + static_cast<uint64_t>(block_size) - 1ull) /
                                  static_cast<uint64_t>(block_size));
        init_labels_kernel<<<init_grid, static_cast<uint32_t>(block_size)>>>(n, d_curr);
        if (cc_cuda_check(cudaGetLastError(), "launching init_labels_kernel") != 0 ||
            cc_cuda_check(cudaDeviceSynchronize(), "synchronizing after init_labels_kernel") != 0)
        {
            rc = 6;
            goto cleanup;
        }
    }

    if (cc_cuda_check(cudaEventCreate(&event_start), "creating event_start") != 0 ||
        cc_cuda_check(cudaEventCreate(&event_stop), "creating event_stop") != 0)
    {
        rc = 7;
        goto cleanup;
    }

    std::printf("Starting block-per-vertex kernel iterations...\n");
    for (;;)
    {
        if (cc_cuda_check(cudaMemset(d_changed, 0, changed_bytes), "clearing changed flag") != 0)
        {
            rc = 8;
            goto cleanup;
        }

        cudaEventRecord(event_start, 0);

        lp_pull_block_per_vertex_kernel<<<grid_size, static_cast<uint32_t>(block_size), shared_bytes>>>(
            n, d_row_ptr, d_col_idx, d_curr, d_next, d_changed);

        if (cc_cuda_check(cudaGetLastError(), "launching lp_pull_block_per_vertex_kernel") != 0 ||
            cc_cuda_check(cudaDeviceSynchronize(), "synchronizing after lp_pull_block_per_vertex_kernel") != 0)
        {
            rc = 9;
            goto cleanup;
        }

        cudaEventRecord(event_stop, 0);
        cudaEventSynchronize(event_stop);

        float kernel_ms = 0.0f;
        cudaEventElapsedTime(&kernel_ms, event_start, event_stop);
        kernel_ms_total += kernel_ms;

        if (cc_cuda_check(cudaMemcpy(&h_changed, d_changed, changed_bytes, cudaMemcpyDeviceToHost),
                          "copying changed flag to host") != 0)
        {
            rc = 10;
            goto cleanup;
        }

        ++iters;

        if (verbose && (iters % 10u) == 0u)
        {
            std::printf("  iter=%" PRIu32 " (last kernel %.3f ms)\n", iters, kernel_ms);
            std::fflush(stdout);
        }

        // swap buffers
        uint32_t *tmp = d_curr;
        d_curr = d_next;
        d_next = tmp;

        if (h_changed == 0u)
            break;
    }

    if (cc_cuda_check(cudaMemcpy(labels, d_curr, labels_bytes, cudaMemcpyDeviceToHost), "copying labels to host") != 0)
    {
        rc = 11;
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

    cudaFree(d_changed);
    cudaFree(d_next);
    cudaFree(d_curr);
    cudaFree(d_col_idx);
    cudaFree(d_row_ptr);

    return rc;
}
