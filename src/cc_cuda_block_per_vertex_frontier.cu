// src/cc_cuda_block_frontier_async.cu
//
// Block-per-vertex (cooperative), FRONTIER-based, ASYNCHRONOUS (in-place) relax + push.
//
// Frontier is a bitset (uint32 words). Each iteration processes only active vertices:
//
// For each active vertex u:
//   1) (optional) relax u from neighbors (pull-min) and apply atomicMin(labels[u], best)
//   2) read lu = labels[u]
//   3) push lu to neighbors: atomicMin(labels[v], lu)
//      if a neighbor v changed, mark v active in next frontier
//
// Correctness note:
//   We ALWAYS push from an active u, even if u didn't decrease "in this kernel",
//   because u may have been decreased earlier by someone else's push.
//   Gating pushes on "u changed here" can terminate early with wrong CC counts.
//
// Mapping:
//   - We do NOT launch n blocks. We cap gridDim.x to a multiple of SMs and use a grid-stride
//     loop over vertices: for (u = blockIdx.x; u < n; u += gridDim.x)
//   - Each block cooperatively processes u’s adjacency list.
//
// Uses uint32 labels; CSR row_ptr is uint64, col_idx is uint32.

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
    if (idx < n)
        labels[idx] = idx;
}

static __device__ __forceinline__ uint32_t u32_min(uint32_t a, uint32_t b)
{
    return (b < a) ? b : a;
}

static __device__ __forceinline__ uint32_t bit_word(uint32_t v) { return v >> 5; }
static __device__ __forceinline__ uint32_t bit_mask(uint32_t v) { return 1u << (v & 31u); }

/**
 * Block-per-vertex frontier async kernel (grid-stride over vertices).
 *
 * Shared memory used for block min reduction and (optionally) for a block-wide "any changed" flag.
 *
 * any_next is set to 1 if any vertex is activated for next iteration.
 */
__global__ static void cc_frontier_async_block_kernel(
    const uint32_t n,
    const uint64_t *row_ptr,
    const uint32_t *col_idx,
    uint32_t *labels,            // in-place
    const uint32_t *active_bits, // bitset words
    uint32_t *next_active_bits,  // bitset words
    unsigned int *any_next)      // flag
{
    extern __shared__ uint32_t sdata[];

    const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
    const uint32_t stride_v = static_cast<uint32_t>(gridDim.x);

    for (uint32_t u = static_cast<uint32_t>(blockIdx.x); u < n; u += stride_v)
    {
        // Frontier check (cheap): skip if not active
        const uint32_t w = bit_word(u);
        const uint32_t m = bit_mask(u);
        if ((active_bits[w] & m) == 0u)
            continue;

        const uint64_t start = row_ptr[u];
        const uint64_t end = row_ptr[u + 1];

        // -------- Optional relax (pull) step: compute min over neighbor labels --------
        uint32_t local_min = labels[u];

        for (uint64_t e = start + static_cast<uint64_t>(tid); e < end; e += static_cast<uint64_t>(blockDim.x))
        {
            const uint32_t v = col_idx[e];
            const uint32_t lv = labels[v]; // async read
            local_min = u32_min(local_min, lv);
        }

        // Reduce local_min across block into sdata[0]
        sdata[tid] = local_min;
        __syncthreads();

        for (uint32_t off = static_cast<uint32_t>(blockDim.x) >> 1; off > 0; off >>= 1)
        {
            if (tid < off)
                sdata[tid] = u32_min(sdata[tid], sdata[tid + off]);
            __syncthreads();
        }

        if (tid == 0u)
        {
            const uint32_t best = sdata[0];
            atomicMin(reinterpret_cast<unsigned int *>(&labels[u]),
                      static_cast<unsigned int>(best));
        }
        __syncthreads();

        // Load current label of u after relax
        const uint32_t lu = labels[u];

        // -------- Push step: push lu to neighbors; activate changed neighbors --------
        unsigned int block_any = 0u;

        for (uint64_t e = start + static_cast<uint64_t>(tid); e < end; e += static_cast<uint64_t>(blockDim.x))
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
                block_any = 1u;
            }
        }

        // Block-wide OR: if any thread saw a change, set any_next.
        // Use ballot to avoid shared atomics.
        const unsigned int mask32 = 0xFFFFFFFFu;
        const unsigned int any_mask = __ballot_sync(mask32, block_any != 0u);

        if (tid == 0u && any_mask != 0u)
            atomicExch(any_next, 1u);

        __syncthreads();
    }
}

int compute_connected_components_cuda_block_frontier_async(
    const CSRGraph *G,
    uint32_t *labels,
    const CudaCCOptions *opt,
    uint32_t *out_iter,
    double *out_kernel_seconds)
{
    if (!G || !labels || !G->row_ptr || !G->col_idx)
    {
        std::fprintf(stderr, "Invalid input to compute_connected_components_cuda_block_frontier_async\n");
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

    cudaDeviceProp prop;
    if (cc_cuda_check(cudaGetDeviceProperties(&prop, device_id), "getting device properties") != 0)
        return 3;

    // Cap grid to a multiple of SMs to keep launch overhead reasonable.
    uint32_t grid_size = 0;
    {
        const uint32_t sm = static_cast<uint32_t>(prop.multiProcessorCount > 0 ? prop.multiProcessorCount : 1);
        const uint32_t cap = sm * 8u; // tuneable
        grid_size = (n < cap) ? n : cap;
        if (grid_size == 0u)
            grid_size = 1u;
    }

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

    if (verbose)
    {
        std::fprintf(stderr, "[cc-cuda][block-frontier-async] n=%" PRIu32 ", m=%" PRIu64 "\n", G->n, G->m);
        std::fprintf(stderr, "[cc-cuda][block-frontier-async] labels=%.3f MB, frontier_bits=%.3f MB (x2)\n",
                     labels_bytes / (1024.0 * 1024.0),
                     bits_bytes / (1024.0 * 1024.0));
        std::fprintf(stderr, "[cc-cuda][block-frontier-async] block_size=%d grid_size=%" PRIu32 "\n",
                     block_size, grid_size);
    }

    const size_t shared_bytes = static_cast<size_t>(block_size) * sizeof(uint32_t);

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
    {
        const uint32_t init_grid =
            static_cast<uint32_t>((static_cast<uint64_t>(n) + static_cast<uint64_t>(block_size) - 1ull) /
                                  static_cast<uint64_t>(block_size));
        init_labels_kernel<<<init_grid, static_cast<uint32_t>(block_size)>>>(n, d_labels);
        if (cc_cuda_check(cudaGetLastError(), "launching init_labels_kernel") != 0 ||
            cc_cuda_check(cudaDeviceSynchronize(), "synchronizing after init_labels_kernel") != 0)
        {
            rc = 6;
            goto cleanup;
        }
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

    std::printf("Starting block-frontier-async iterations...\n");

    for (;;)
    {
        if (cc_cuda_check(cudaMemset(d_next_active, 0x00, bits_bytes), "clearing next frontier") != 0 ||
            cc_cuda_check(cudaMemset(d_any_next, 0x00, sizeof(unsigned int)), "clearing any_next") != 0)
        {
            rc = 9;
            goto cleanup;
        }

        cudaEventRecord(event_start, 0);

        cc_frontier_async_block_kernel<<<grid_size, static_cast<uint32_t>(block_size), shared_bytes>>>(
            n, d_row_ptr, d_col_idx, d_labels, d_active, d_next_active, d_any_next);

        if (cc_cuda_check(cudaGetLastError(), "launching cc_frontier_async_block_kernel") != 0 ||
            cc_cuda_check(cudaDeviceSynchronize(), "synchronizing after cc_frontier_async_block_kernel") != 0)
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
                          "copying any_next") != 0)
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

        // swap frontiers
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
