// src/cc_cuda_thread_per_vertex.cu
//
// Thread-per-vertex, PULL-style (double-buffer) min-label propagation.
// No atomics on labels. Each iteration computes:
//   next[v] = min(curr[v], min_{u in N(v)} curr[u])
// and sets a global changed flag if any v changed.
//
// This is typically much faster than atomic push on large graphs.

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
    if (idx < n) labels[idx] = idx;
}

/**
 * Pull-based label propagation kernel (Jacobi / double-buffer):
 * next[v] = min(curr[v], min_{u in N(v)} curr[u])
 *
 * @param n Number of vertices
 * @param row_ptr CSR row pointer array (uint64 offsets)
 * @param col_ind CSR column indices array (uint32 vertex ids)
 * @param curr Current labels (read-only this iteration)
 * @param next Next labels (written)
 * @param changed Global flag set to 1 if any label changed
 */
__global__ static void lp_pull_per_v_kernel(
    const uint32_t n,
    const uint64_t *row_ptr,
    const uint32_t *col_ind,
    const uint32_t *curr,
    uint32_t *next,
    unsigned int *changed)
{
    uint32_t v = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (v >= n) return;

    uint32_t best = curr[v];

    uint64_t row_start = row_ptr[v];
    uint64_t row_end   = row_ptr[v + 1];

    for (uint64_t offset = row_start; offset < row_end; ++offset)
    {
        uint32_t u = col_ind[offset];
        uint32_t lu = curr[u];
        if (lu < best) best = lu;
    }

    next[v] = best;

    if (best != curr[v])
        atomicExch(changed, 1u);
}

int compute_connected_components_cuda_thread_per_vertex(
    const CSRGraph *G,
    uint32_t *labels,
    const CudaCCOptions *opt,
    uint32_t *out_iter,
    double *out_kernel_seconds)
{
    if (!G || !labels || !G->row_ptr || !G->col_idx)
    {
        std::fprintf(stderr, "Invalid input to compute_connected_components_cuda_thread_per_vertex\n");
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
        device_id = opt->device_id >= 0 ? opt->device_id : device_id;
        block_size = opt->block_size > 0 ? opt->block_size : block_size;
        verbose = opt->verbose;
    }

    if (cc_cuda_check(cudaSetDevice(device_id), "setting CUDA device") != 0)
        return 3;

    uint64_t *d_row_ptr = nullptr;
    uint32_t *d_col_idx = nullptr;

    uint32_t *d_curr = nullptr;
    uint32_t *d_next = nullptr;

    unsigned int *d_changed = nullptr;

    cudaEvent_t event_start = nullptr, event_stop = nullptr;
    float kernel_ms_total = 0.0f;

    int rc = 0;

    const size_t row_ptr_bytes = static_cast<size_t>(n + 1) * sizeof(uint64_t);
    const size_t col_idx_bytes = static_cast<size_t>(G->m) * sizeof(uint32_t);
    const size_t labels_bytes  = static_cast<size_t>(n) * sizeof(uint32_t);
    const size_t changed_bytes = sizeof(unsigned int);

    if (verbose)
    {
        std::fprintf(stderr, "[cc-cuda][pull] n=%" PRIu32 ", m=%" PRIu64 "\n", G->n, G->m);
        std::fprintf(stderr, "[cc-cuda][pull] row_ptr=%.3f MB, col_idx=%.3f MB, labels=%.3f MB (x2)\n",
                     row_ptr_bytes / (1024.0 * 1024.0),
                     col_idx_bytes / (1024.0 * 1024.0),
                     labels_bytes / (1024.0 * 1024.0));
    }

    uint32_t iters = 0;
    unsigned int h_changed = 1;

    const uint32_t grid_size = static_cast<uint32_t>((n + (uint32_t)block_size - 1u) / (uint32_t)block_size);

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
    init_labels_kernel<<<grid_size, (uint32_t)block_size>>>(n, d_curr);
    if (cc_cuda_check(cudaGetLastError(), "launching init_labels_kernel") != 0 ||
        cc_cuda_check(cudaDeviceSynchronize(), "synchronizing after init_labels_kernel") != 0)
    {
        rc = 6;
        goto cleanup;
    }

    if (cc_cuda_check(cudaEventCreate(&event_start), "creating event_start") != 0 ||
        cc_cuda_check(cudaEventCreate(&event_stop), "creating event_stop") != 0)
    {
        rc = 7;
        goto cleanup;
    }

    std::printf("Starting kernel iterations...\n");
    for (;;)
    {
        // clear changed flag on device
        if (cc_cuda_check(cudaMemset(d_changed, 0, changed_bytes), "clearing changed flag") != 0)
        {
            rc = 8;
            goto cleanup;
        }

        if (event_start) cudaEventRecord(event_start, 0);

        lp_pull_per_v_kernel<<<grid_size, (uint32_t)block_size>>>(n, d_row_ptr, d_col_idx, d_curr, d_next, d_changed);

        if (cc_cuda_check(cudaGetLastError(), "launching lp_pull_per_v_kernel") != 0 ||
            cc_cuda_check(cudaDeviceSynchronize(), "synchronizing after lp_pull_per_v_kernel") != 0)
        {
            rc = 9;
            goto cleanup;
        }

        if (event_stop)
        {
            cudaEventRecord(event_stop, 0);
            cudaEventSynchronize(event_stop);

            if ((iters % 10) == 0 && verbose)
            {
                printf("  iter=%u\n", iters);
                fflush(stdout);
            }

            float kernel_ms = 0.0f;
            cudaEventElapsedTime(&kernel_ms, event_start, event_stop);
            kernel_ms_total += kernel_ms;
        }

        if (cc_cuda_check(cudaMemcpy(&h_changed, d_changed, changed_bytes, cudaMemcpyDeviceToHost), "copying changed flag to host") != 0)
        {
            rc = 10;
            goto cleanup;
        }

        ++iters;

        // swap buffers
        uint32_t *tmp = d_curr;
        d_curr = d_next;
        d_next = tmp;

        if (h_changed == 0) break;
    }

    if (cc_cuda_check(cudaMemcpy(labels, d_curr, labels_bytes, cudaMemcpyDeviceToHost), "copying labels to host") != 0)
    {
        rc = 11;
        goto cleanup;
    }

    if (out_iter) *out_iter = iters;
    if (out_kernel_seconds) *out_kernel_seconds = static_cast<double>(kernel_ms_total) / 1000.0;

cleanup:
    if (event_start) cudaEventDestroy(event_start);
    if (event_stop)  cudaEventDestroy(event_stop);

    cudaFree(d_changed);
    cudaFree(d_next);
    cudaFree(d_curr);
    cudaFree(d_col_idx);
    cudaFree(d_row_ptr);

    return rc;
}
