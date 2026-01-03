#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Compressed Sparse Row (CSR) representation of a graph.
typedef struct
{
  uint32_t n;
  uint64_t m;
  uint64_t *row_ptr;
  uint32_t *col_idx;
} CSRGraph;

// Load an undirected graph from Matrix Market file into CSR form.
// symmetrize = 1 → ensure undirected by adding reverse edges.
// drop_self_loops = 1 → skip edges (i,i).
// Returns 0 on success.
int load_csr_from_mtx(const char *path, int symmetrize, int drop_self_loops, CSRGraph *out);

// Load an undirected graph from MATLAB .mat file into CSR form.
// Returns 0 on success.
int load_csr_from_mat(const char *path, CSRGraph *out);

// Load an undirected graph from file (either .mtx/.txt or .mat) into CSR form.
// symmetrize = 1 → ensure undirected by adding reverse edges.
// drop_self_loops = 1 → skip edges (i,i).
// Returns 0 on success.
int load_csr_from_file(const char *path, int symmetrize, int drop_self_loops, CSRGraph *out);

// Free memory allocated for CSR graph.
void free_csr(CSRGraph *g);

#ifdef __cplusplus
}
#endif
