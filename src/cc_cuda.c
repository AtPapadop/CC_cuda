#define _GNU_SOURCE

#include "cc_cuda.h"

int compute_connected_components_cuda(const CSRGraph *G,
                                      uint32_t *labels,
                                      const CudaCCOptions *opt,
                                      uint32_t *out_iters,
                                      double *out_kernel_seconds,
                                      CudaCCMethod method)
{
    switch (method)
    {
    case CUDA_CC_METHOD_THREAD_PER_VERTEX:
        return compute_connected_components_cuda_thread_per_vertex(G, labels, opt, out_iters, out_kernel_seconds);
    case CUDA_CC_METHOD_THREAD_FRONTIER_ASYNC:
        return compute_connected_components_cuda_thread_frontier_async(G, labels, opt, out_iters, out_kernel_seconds);
    case CUDA_CC_METHOD_THREAD_AFOREST:
        return compute_connected_components_cuda_thread_afforest(G, labels, opt, out_iters, out_kernel_seconds);
    case CUDA_CC_METHOD_WARP_PER_VERTEX:
        return compute_connected_components_cuda_warp_per_vertex(G, labels, opt, out_iters, out_kernel_seconds);
    case CUDA_CC_METHOD_WARP_FRONTIER_ASYNC:
        return compute_connected_components_cuda_warp_frontier_async(G, labels, opt, out_iters, out_kernel_seconds);
    case CUDA_CC_METHOD_WARP_AFOREST:
        return compute_connected_components_cuda_warp_afforest(G, labels, opt, out_iters, out_kernel_seconds);
    case CUDA_CC_METHOD_BLOCK_PER_VERTEX:
        return compute_connected_components_cuda_block_per_vertex(G, labels, opt, out_iters, out_kernel_seconds);
    case CUDA_CC_METHOD_BLOCK_FRONTIER_ASYNC:
        return compute_connected_components_cuda_block_frontier_async(G, labels, opt, out_iters, out_kernel_seconds);
    case CUDA_CC_METHOD_BLOCK_AFOREST:
        return compute_connected_components_cuda_block_afforest(G, labels, opt, out_iters, out_kernel_seconds);
    default:
        return -1; // Unsupported method
    }
}