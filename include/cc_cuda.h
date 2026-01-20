#pragma once

#include <stdint.h>
#include "graph.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int device_id;
    int block_size;
    int verbose;
} CudaCCOptions;

typedef enum {
    CUDA_CC_METHOD_THREAD_PER_VERTEX, // 0
    CUDA_CC_METHOD_THREAD_FRONTIER_ASYNC, // 1
    CUDA_CC_METHOD_THREAD_AFOREST, // 2
    CUDA_CC_METHOD_WARP_PER_VERTEX, // 3
    CUDA_CC_METHOD_WARP_FRONTIER_ASYNC, // 4
    CUDA_CC_METHOD_WARP_AFOREST, // 5
    CUDA_CC_METHOD_BLOCK_PER_VERTEX, // 6
    CUDA_CC_METHOD_BLOCK_FRONTIER_ASYNC, // 7
    CUDA_CC_METHOD_BLOCK_AFOREST // 8
} CudaCCMethod;


/**
 * Compute connected components using CUDA with per-vertex atomic push-based label propagation
 *
 * @param G Input graph in CSR format
 * @param labels Output array of size G->n to store component labels
 * @param opt CudaCCOptions structure with options (device_id, block_size, verbose) (can be nullptr for defaults)
 * @param out_iter Pointer to store the number of iterations performed (can be nullptr)
 * @param out_kernel_seconds Pointer to store the total kernel execution time in seconds (can be nullptr)
 * @return 0 on success, non-zero on failure
 */
int compute_connected_components_cuda_thread_per_vertex(const CSRGraph *G,
                                                        uint32_t *labels,
                                                        const CudaCCOptions *opt,
                                                        uint32_t *out_iters,
                                                        double *out_kernel_seconds);

int compute_connected_components_cuda_thread_frontier_async(const CSRGraph *G,
                                                            uint32_t *labels,
                                                            const CudaCCOptions *opt,
                                                            uint32_t *out_iter,
                                                            double *out_kernel_seconds);

int compute_connected_components_cuda_thread_afforest(const CSRGraph *G,
                                                      uint32_t *labels,
                                                      const CudaCCOptions *opt,
                                                      uint32_t *out_iter,
                                                      double *out_kernel_seconds);


int compute_connected_components_cuda_warp_per_vertex(const CSRGraph *G,
                                                      uint32_t *labels,
                                                      const CudaCCOptions *opt,
                                                      uint32_t *out_iter,
                                                      double *out_kernel_seconds);         
                                               
int compute_connected_components_cuda_warp_frontier_async(const CSRGraph *G,
                                                          uint32_t *labels,
                                                          const CudaCCOptions *opt,
                                                          uint32_t *out_iters,
                                                          double *out_kernel_seconds);

int compute_connected_components_cuda_warp_afforest(const CSRGraph *G,
                                                    uint32_t *labels,
                                                    const CudaCCOptions *opt,
                                                    uint32_t *out_iters,
                                                    double *out_kernel_seconds);

int compute_connected_components_cuda_block_per_vertex(const CSRGraph *G,
                                                        uint32_t *labels,
                                                        const CudaCCOptions *opt,
                                                        uint32_t *out_iters,
                                                        double *out_kernel_seconds);
                                                        
int compute_connected_components_cuda_block_frontier_async(const CSRGraph *G,
                                                          uint32_t *labels,
                                                          const CudaCCOptions *opt,
                                                          uint32_t *out_iters,
                                                          double *out_kernel_seconds);

int compute_connected_components_cuda_block_afforest(const CSRGraph *G,
                                                      uint32_t *labels,
                                                      const CudaCCOptions *opt,
                                                      uint32_t *out_iters,
                                                      double *out_kernel_seconds);
                                                                                                               
int compute_connected_components_cuda(const CSRGraph *G,
                                      uint32_t *labels,
                                      const CudaCCOptions *opt,
                                      uint32_t *out_iters,
                                      double *out_kernel_seconds,
                                      CudaCCMethod method);                                                       
#ifdef __cplusplus
}
#endif