// src/cc_cuda_thread_afforest.cu
//
// Vertex-per-thread (Afforest-style sampling + SV hooking + compression).
//
// Phases:
//  1) parent[v] = v
//  2) Sampling (k-out): for each vertex sample k neighbors and hook roots
//  3) Compression passes (pointer jumping)
//  4) Pick “giant” root by sampling roots -> host mode
//  5) Finish: scan full adjacency for vertices not in giant, hook + compress until stable
//  6) Output labels[v] = find_root(v)

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cinttypes>

#include <algorithm>
#include <vector>

#include <cuda_runtime.h>

#include "cc_cuda.h"
#include "graph.h"

#ifndef CC_CUDA_DEFAULT_BLOCK
#define CC_CUDA_DEFAULT_BLOCK 256
#endif

#ifndef CC_AFFOREST_K
#define CC_AFFOREST_K 2
#endif

#ifndef CC_AFFOREST_SAMPLE_ROUNDS
#define CC_AFFOREST_SAMPLE_ROUNDS 2
#endif

#ifndef CC_AFFOREST_COMPRESS_PASSES
#define CC_AFFOREST_COMPRESS_PASSES 4
#endif

#ifndef CC_AFFOREST_SAMPLE_SIZE
#define CC_AFFOREST_SAMPLE_SIZE 1000000u
#endif

#ifndef CC_AFFOREST_MAX_ITERS
#define CC_AFFOREST_MAX_ITERS 1000
#endif

static_assert(sizeof(uint32_t) == sizeof(unsigned int), "uint32_t must match unsigned int for atomics.");

static inline int cc_cuda_check(cudaError_t err, const char *what)
{
    if (err != cudaSuccess)
    {
        std::fprintf(stderr, "CUDA error during %s: %s\n", what, cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

__device__ __forceinline__ uint32_t d_min_u32(uint32_t a, uint32_t b) { return (b < a) ? b : a; }
__device__ __forceinline__ uint32_t d_max_u32(uint32_t a, uint32_t b) { return (a < b) ? b : a; }

__device__ __forceinline__ uint32_t d_hash_u32(uint32_t x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

__device__ __forceinline__ uint32_t d_find_root(uint32_t *parent, uint32_t x)
{
    while (true)
    {
        uint32_t px = parent[x];
        if (px == x) return x;
        uint32_t ppx = parent[px];
        parent[x] = ppx; // path halving
        x = px;
    }
}

__global__ static void init_parent_kernel(uint32_t n, uint32_t *parent)
{
    uint32_t v = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (v < n) parent[v] = v;
}

__global__ static void compress_step_kernel(uint32_t n, uint32_t *parent)
{
    uint32_t v = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (v >= n) return;

    uint32_t p = parent[v];
    uint32_t pp = parent[p];
    parent[v] = pp;
}

__global__ static void sample_k_out_kernel(
    uint32_t n,
    const uint64_t *row_ptr,
    const uint32_t *col_idx,
    uint32_t *parent,
    uint32_t k,
    uint32_t round_seed,
    unsigned int *changed)
{
    uint32_t u = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (u >= n) return;

    uint64_t start = row_ptr[u];
    uint64_t end   = row_ptr[u + 1];
    uint64_t deg64 = (end > start) ? (end - start) : 0ull;
    if (deg64 == 0ull) return;

    uint32_t deg = (deg64 > 0xFFFFFFFFull) ? 0xFFFFFFFFu : static_cast<uint32_t>(deg64);

    for (uint32_t i = 0; i < k; ++i)
    {
        // pick pseudo-random neighbor to avoid bias in real world graphs
        uint32_t h = d_hash_u32(u ^ (round_seed + 0x9e3779b9u * i));
        uint32_t off = (deg > 0u) ? (h % deg) : 0u;

        uint32_t v = col_idx[start + static_cast<uint64_t>(off)];

        uint32_t ru = d_find_root(parent, u);
        uint32_t rv = d_find_root(parent, v);
        if (ru == rv) continue;

        uint32_t hi = d_max_u32(ru, rv);
        uint32_t lo = d_min_u32(ru, rv);

        unsigned int prev = atomicMin(reinterpret_cast<unsigned int *>(&parent[hi]),
                                      static_cast<unsigned int>(lo));
        if (static_cast<uint32_t>(prev) > lo)
            atomicExch(changed, 1u);
    }
}

__global__ static void sample_roots_kernel(
    uint32_t n,
    uint32_t *parent,
    uint32_t stride,
    uint32_t sample_n,
    uint32_t *out_roots)
{
    uint32_t i = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= sample_n) return;

    uint64_t u64 = static_cast<uint64_t>(i) * static_cast<uint64_t>(stride);
    if (u64 >= static_cast<uint64_t>(n)) u64 = static_cast<uint64_t>(n - 1u);

    uint32_t u = static_cast<uint32_t>(u64);
    out_roots[i] = d_find_root(parent, u);
}

__global__ static void finish_hook_kernel_skip_giant(
    uint32_t n,
    const uint64_t *row_ptr,
    const uint32_t *col_idx,
    uint32_t *parent,
    uint32_t giant_root,
    unsigned int *changed)
{
    uint32_t u = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (u >= n) return;

    uint32_t ru = d_find_root(parent, u);
    if (giant_root != 0xFFFFFFFFu && ru == giant_root) return;

    uint64_t start = row_ptr[u];
    uint64_t end   = row_ptr[u + 1];

    for (uint64_t e = start; e < end; ++e)
    {
        uint32_t v = col_idx[e];

        uint32_t rv = d_find_root(parent, v);
        if (ru == rv) continue;

        uint32_t hi = d_max_u32(ru, rv);
        uint32_t lo = d_min_u32(ru, rv);

        unsigned int prev = atomicMin(reinterpret_cast<unsigned int *>(&parent[hi]),
                                      static_cast<unsigned int>(lo));
        if (static_cast<uint32_t>(prev) > lo)
            atomicExch(changed, 1u);

        // refresh ru sometimes
        ru = d_find_root(parent, u);
        if (giant_root != 0xFFFFFFFFu && ru == giant_root) break;
    }
}

__global__ static void write_labels_kernel(uint32_t n, uint32_t *parent, uint32_t *labels_out)
{
    uint32_t v = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (v >= n) return;

    labels_out[v] = d_find_root(parent, v);
}

static uint32_t host_mode_root(std::vector<uint32_t> &roots)
{
    if (roots.empty()) return 0xFFFFFFFFu;
    std::sort(roots.begin(), roots.end());

    uint32_t best_val = roots[0];
    size_t best_count = 1;

    uint32_t cur_val = roots[0];
    size_t cur_count = 1;

    for (size_t i = 1; i < roots.size(); ++i)
    {
        if (roots[i] == cur_val)
        {
            ++cur_count;
        }
        else
        {
            if (cur_count > best_count)
            {
                best_count = cur_count;
                best_val = cur_val;
            }
            cur_val = roots[i];
            cur_count = 1;
        }
    }

    if (cur_count > best_count)
    {
        best_count = cur_count;
        best_val = cur_val;
    }

    return best_val;
}

int compute_connected_components_cuda_thread_afforest(
    const CSRGraph *G,
    uint32_t *labels,
    const CudaCCOptions *opt,
    uint32_t *out_iter,
    double *out_kernel_seconds)
{
    int rc;
    int device_id;
    int block_size;
    int verbose;

    uint32_t n;
    uint32_t sample_n;
    uint32_t stride;
    uint32_t grid_v;
    uint32_t grid_sample;
    uint32_t giant_root;
    uint32_t iters;

    size_t row_ptr_bytes;
    size_t col_idx_bytes;
    size_t parent_bytes;
    size_t sample_bytes;

    uint64_t *d_row_ptr;
    uint32_t *d_col_idx;
    uint32_t *d_parent;
    uint32_t *d_labels_out;
    unsigned int *d_changed;
    uint32_t *d_sample_roots;

    cudaEvent_t ev_start;
    cudaEvent_t ev_stop;
    float kernel_ms_total;

    std::vector<uint32_t> h_roots;

    rc = 0;
    device_id = 0;
    block_size = CC_CUDA_DEFAULT_BLOCK;
    verbose = 0;

    d_row_ptr = nullptr;
    d_col_idx = nullptr;
    d_parent = nullptr;
    d_labels_out = nullptr;
    d_changed = nullptr;
    d_sample_roots = nullptr;

    ev_start = nullptr;
    ev_stop = nullptr;
    kernel_ms_total = 0.0f;

    giant_root = 0xFFFFFFFFu;
    iters = 0u;

    if (!G || !labels || !G->row_ptr || !G->col_idx)
    {
        std::fprintf(stderr, "Invalid input to compute_connected_components_cuda_thread_afforest\n");
        return 2;
    }

    n = G->n;
    if (n == 0u)
    {
        if (out_iter) *out_iter = 0u;
        if (out_kernel_seconds) *out_kernel_seconds = 0.0;
        return 0;
    }

    if (opt)
    {
        device_id = (opt->device_id >= 0) ? opt->device_id : device_id;
        block_size = (opt->block_size > 0) ? opt->block_size : block_size;
        verbose = opt->verbose;
    }

    if (cc_cuda_check(cudaSetDevice(device_id), "cudaSetDevice") != 0)
        return 3;

    row_ptr_bytes = static_cast<size_t>(n + 1u) * sizeof(uint64_t);
    col_idx_bytes = static_cast<size_t>(G->m) * sizeof(uint32_t);
    parent_bytes  = static_cast<size_t>(n) * sizeof(uint32_t);

    sample_n = CC_AFFOREST_SAMPLE_SIZE;
    if (sample_n > n) sample_n = n;
    if (sample_n == 0u) sample_n = 1u;

    stride = static_cast<uint32_t>(
        (static_cast<uint64_t>(n) + static_cast<uint64_t>(sample_n) - 1ull) /
        static_cast<uint64_t>(sample_n));
    if (stride == 0u) stride = 1u;

    sample_bytes = static_cast<size_t>(sample_n) * sizeof(uint32_t);

    grid_v = static_cast<uint32_t>(
        (static_cast<uint64_t>(n) + static_cast<uint64_t>(block_size) - 1ull) /
        static_cast<uint64_t>(block_size));

    grid_sample = static_cast<uint32_t>(
        (static_cast<uint64_t>(sample_n) + static_cast<uint64_t>(block_size) - 1ull) /
        static_cast<uint64_t>(block_size));

    if (verbose)
    {
        std::fprintf(stderr, "[cc-cuda][afforest-thread] n=%" PRIu32 ", m=%" PRIu64 "\n", G->n, G->m);
        std::fprintf(stderr, "[cc-cuda][afforest-thread] parent=%.3f MB, sample=%.3f MB (sample_n=%" PRIu32 ", stride=%" PRIu32 ")\n",
                     parent_bytes / (1024.0 * 1024.0),
                     sample_bytes / (1024.0 * 1024.0),
                     sample_n, stride);
        std::fprintf(stderr, "[cc-cuda][afforest-thread] K=%d, sample_rounds=%d, compress_passes=%d, finish_cap=%d\n",
                     CC_AFFOREST_K, CC_AFFOREST_SAMPLE_ROUNDS, CC_AFFOREST_COMPRESS_PASSES, CC_AFFOREST_MAX_ITERS);
    }

    h_roots.resize(sample_n);

    // Allocate device memory
    if (cc_cuda_check(cudaMalloc(reinterpret_cast<void **>(&d_row_ptr), row_ptr_bytes), "cudaMalloc(d_row_ptr)") != 0 ||
        cc_cuda_check(cudaMalloc(reinterpret_cast<void **>(&d_col_idx), col_idx_bytes), "cudaMalloc(d_col_idx)") != 0 ||
        cc_cuda_check(cudaMalloc(reinterpret_cast<void **>(&d_parent), parent_bytes), "cudaMalloc(d_parent)") != 0 ||
        cc_cuda_check(cudaMalloc(reinterpret_cast<void **>(&d_labels_out), parent_bytes), "cudaMalloc(d_labels_out)") != 0 ||
        cc_cuda_check(cudaMalloc(reinterpret_cast<void **>(&d_changed), sizeof(unsigned int)), "cudaMalloc(d_changed)") != 0 ||
        cc_cuda_check(cudaMalloc(reinterpret_cast<void **>(&d_sample_roots), sample_bytes), "cudaMalloc(d_sample_roots)") != 0)
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

    if (cc_cuda_check(cudaEventCreate(&ev_start), "cudaEventCreate(start)") != 0 ||
        cc_cuda_check(cudaEventCreate(&ev_stop), "cudaEventCreate(stop)") != 0)
    {
        rc = 6;
        goto cleanup;
    }

    // Init parent
    cudaEventRecord(ev_start, 0);
    init_parent_kernel<<<grid_v, static_cast<uint32_t>(block_size)>>>(n, d_parent);
    if (cc_cuda_check(cudaGetLastError(), "launch init_parent_kernel") != 0 ||
        cc_cuda_check(cudaDeviceSynchronize(), "sync init_parent_kernel") != 0)
    {
        rc = 7;
        goto cleanup;
    }
    cudaEventRecord(ev_stop, 0);
    cudaEventSynchronize(ev_stop);
    {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, ev_start, ev_stop);
        kernel_ms_total += ms;
    }

    // Phase A: sampling rounds
    for (int r = 0; r < CC_AFFOREST_SAMPLE_ROUNDS; ++r)
    {
        if (cc_cuda_check(cudaMemset(d_changed, 0, sizeof(unsigned int)), "memset changed") != 0)
        {
            rc = 8;
            goto cleanup;
        }

        cudaEventRecord(ev_start, 0);

        sample_k_out_kernel<<<grid_v, static_cast<uint32_t>(block_size)>>>(
            n, d_row_ptr, d_col_idx, d_parent,
            static_cast<uint32_t>(CC_AFFOREST_K),
            static_cast<uint32_t>(0xA5A5A5A5u + static_cast<uint32_t>(r) * 1013904223u),
            d_changed);

        for (int i = 0; i < CC_AFFOREST_COMPRESS_PASSES; ++i)
            compress_step_kernel<<<grid_v, static_cast<uint32_t>(block_size)>>>(n, d_parent);

        if (cc_cuda_check(cudaGetLastError(), "launch sample_k_out / compress") != 0 ||
            cc_cuda_check(cudaDeviceSynchronize(), "sync sample phase") != 0)
        {
            rc = 9;
            goto cleanup;
        }

        cudaEventRecord(ev_stop, 0);
        cudaEventSynchronize(ev_stop);
        {
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, ev_start, ev_stop);
            kernel_ms_total += ms;
        }
    }

    // Extra compress after sampling
    cudaEventRecord(ev_start, 0);
    for (int i = 0; i < CC_AFFOREST_COMPRESS_PASSES * 2; ++i)
        compress_step_kernel<<<grid_v, static_cast<uint32_t>(block_size)>>>(n, d_parent);

    if (cc_cuda_check(cudaGetLastError(), "launch compress after sampling") != 0 ||
        cc_cuda_check(cudaDeviceSynchronize(), "sync compress after sampling") != 0)
    {
        rc = 10;
        goto cleanup;
    }
    cudaEventRecord(ev_stop, 0);
    cudaEventSynchronize(ev_stop);
    {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, ev_start, ev_stop);
        kernel_ms_total += ms;
    }

    // Pick giant root (sample roots -> host mode)
    cudaEventRecord(ev_start, 0);
    sample_roots_kernel<<<grid_sample, static_cast<uint32_t>(block_size)>>>(
        n, d_parent, stride, sample_n, d_sample_roots);

    if (cc_cuda_check(cudaGetLastError(), "launch sample_roots_kernel") != 0 ||
        cc_cuda_check(cudaDeviceSynchronize(), "sync sample_roots_kernel") != 0)
    {
        rc = 11;
        goto cleanup;
    }
    cudaEventRecord(ev_stop, 0);
    cudaEventSynchronize(ev_stop);
    {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, ev_start, ev_stop);
        kernel_ms_total += ms;
    }

    if (cc_cuda_check(cudaMemcpy(h_roots.data(), d_sample_roots, sample_bytes, cudaMemcpyDeviceToHost),
                      "cudaMemcpy(sample_roots D2H)") != 0)
    {
        rc = 12;
        goto cleanup;
    }

    giant_root = host_mode_root(h_roots);
    if (verbose)
        std::fprintf(stderr, "[cc-cuda][afforest-thread] giant_root=%" PRIu32 "\n", giant_root);

    // Phase B: finish
    iters = 0u;
    for (;;)
    {
        if (cc_cuda_check(cudaMemset(d_changed, 0, sizeof(unsigned int)), "memset changed (finish)") != 0)
        {
            rc = 13;
            goto cleanup;
        }

        cudaEventRecord(ev_start, 0);

        finish_hook_kernel_skip_giant<<<grid_v, static_cast<uint32_t>(block_size)>>>(
            n, d_row_ptr, d_col_idx, d_parent, giant_root, d_changed);

        for (int i = 0; i < CC_AFFOREST_COMPRESS_PASSES; ++i)
            compress_step_kernel<<<grid_v, static_cast<uint32_t>(block_size)>>>(n, d_parent);

        if (cc_cuda_check(cudaGetLastError(), "launch finish_hook / compress") != 0 ||
            cc_cuda_check(cudaDeviceSynchronize(), "sync finish phase") != 0)
        {
            rc = 14;
            goto cleanup;
        }

        cudaEventRecord(ev_stop, 0);
        cudaEventSynchronize(ev_stop);
        {
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, ev_start, ev_stop);
            kernel_ms_total += ms;
        }

        unsigned int h_changed = 0u;
        if (cc_cuda_check(cudaMemcpy(&h_changed, d_changed, sizeof(unsigned int), cudaMemcpyDeviceToHost),
                          "cudaMemcpy(changed D2H)") != 0)
        {
            rc = 15;
            goto cleanup;
        }

        ++iters;

        if (verbose && (iters % 5u) == 0u)
            std::fprintf(stderr, "  finish_iter=%" PRIu32 " changed=%u\n", iters, h_changed);

        if (h_changed == 0u) break;
        if (iters >= static_cast<uint32_t>(CC_AFFOREST_MAX_ITERS)) break;
    }

    // Final compress + write labels
    cudaEventRecord(ev_start, 0);
    for (int i = 0; i < CC_AFFOREST_COMPRESS_PASSES * 3; ++i)
        compress_step_kernel<<<grid_v, static_cast<uint32_t>(block_size)>>>(n, d_parent);

    write_labels_kernel<<<grid_v, static_cast<uint32_t>(block_size)>>>(n, d_parent, d_labels_out);

    if (cc_cuda_check(cudaGetLastError(), "launch final compress/write_labels") != 0 ||
        cc_cuda_check(cudaDeviceSynchronize(), "sync final compress/write") != 0)
    {
        rc = 16;
        goto cleanup;
    }

    cudaEventRecord(ev_stop, 0);
    cudaEventSynchronize(ev_stop);
    {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, ev_start, ev_stop);
        kernel_ms_total += ms;
    }

    if (cc_cuda_check(cudaMemcpy(labels, d_labels_out, parent_bytes, cudaMemcpyDeviceToHost),
                      "cudaMemcpy(labels D2H)") != 0)
    {
        rc = 17;
        goto cleanup;
    }

    if (out_iter) *out_iter = iters;
    if (out_kernel_seconds) *out_kernel_seconds = static_cast<double>(kernel_ms_total) / 1000.0;

cleanup:
    if (ev_start) cudaEventDestroy(ev_start);
    if (ev_stop) cudaEventDestroy(ev_stop);

    cudaFree(d_sample_roots);
    cudaFree(d_changed);
    cudaFree(d_labels_out);
    cudaFree(d_parent);
    cudaFree(d_col_idx);
    cudaFree(d_row_ptr);

    return rc;
}
