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
#ifdef __cplusplus
}
#endif