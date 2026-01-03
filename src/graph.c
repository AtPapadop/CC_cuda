#define _POSIX_C_SOURCE 200112L

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <strings.h>
#include <inttypes.h>
#include <errno.h>
#include <matio.h>

#include "graph.h"

int load_csr_from_file(const char *path, int symmetrize, int drop_self_loops, CSRGraph *out)
{
  const char *ext = strrchr(path, '.');
  (void)symmetrize;
  (void)drop_self_loops;
  if (!ext)
    ext = "";

  if (strcasecmp(ext, ".mtx") == 0 || strcasecmp(ext, ".txt") == 0)
  {
    return -1;
  }
  else if (strcasecmp(ext, ".mat") == 0)
  {
    return load_csr_from_mat(path, out);
  }
  else
  {
    fprintf(stderr, "Unsupported file extension: %s\n", ext);
    return 99;
  }
}


int load_csr_from_mat(const char *path, CSRGraph *out)
{
  memset(out, 0, sizeof(*out));

  mat_t *matfp = Mat_Open(path, MAT_ACC_RDONLY);
  if (!matfp)
  {
    fprintf(stderr, "Error opening .mat file: %s\n", path);
    return 1;
  }

  matvar_t *var = Mat_VarReadNext(matfp);
  while (var)
  {
    if (strcmp(var->name, "Problem") == 0 && var->data_type == MAT_T_STRUCT)
    {
      matvar_t *fieldA = Mat_VarGetStructFieldByName(var, "A", 0);
      if (fieldA)
        var = fieldA;
      break;
    }
    if (var->class_type == MAT_C_SPARSE)
      break;
    Mat_VarFree(var);
    var = Mat_VarReadNext(matfp);
  }

  if (!var)
  {
    fprintf(stderr, "No sparse matrix found in .mat file %s\n", path);
    Mat_Close(matfp);
    return 2;
  }

  mat_sparse_t *sparse = var->data;
  uint32_t n = (uint32_t)var->dims[0];
  uint32_t mcols = (uint32_t)var->dims[1];

  uint64_t nz = (uint64_t)sparse->jc[mcols];

  // Build CSR from column-compressed format in MATLAB (CSC)
  uint64_t *row_ptr = (uint64_t *)calloc((size_t)n + 1, sizeof(uint64_t));
  uint32_t *col_idx = (uint32_t *)malloc(sizeof(uint32_t) * (size_t)nz);
  if (!row_ptr || !col_idx)
  {
    fprintf(stderr, "Memory allocation failed\n");
    Mat_VarFree(var);
    Mat_Close(matfp);
    free(row_ptr);
    free(col_idx);
    return 3;
  }

  // Count entries per row
  for (uint32_t c = 0; c < mcols; ++c)
  {
    for (uint64_t j = (uint64_t)sparse->jc[c]; j < (uint64_t)sparse->jc[c + 1]; ++j)
    {
      int r = sparse->ir[j];
      if (r >= 0 && (uint32_t)r < n)
        row_ptr[r + 1]++;
    }
  }
  for (uint32_t i = 0; i < n; ++i)
    row_ptr[i + 1] += row_ptr[i];

  // Fill col_idx
  uint64_t *head = (uint64_t *)malloc(sizeof(uint64_t) * n);
  memcpy(head, row_ptr, sizeof(uint64_t) * n);

  for (uint32_t c = 0; c < mcols; ++c)
  {
    for (uint64_t j = (uint64_t)sparse->jc[c]; j < (uint64_t)sparse->jc[c + 1]; ++j)
    {
      int r = sparse->ir[j];
      if (r < 0 || (uint32_t)r >= n)
        continue;
      col_idx[head[r]++] = c;
    }
  }

  free(head);
  Mat_VarFree(var);
  Mat_Close(matfp);

  out->n = n;
  out->m = row_ptr[n];
  out->row_ptr = row_ptr;
  out->col_idx = col_idx;
  return 0;
}

void free_csr(CSRGraph *g)
{
  if (g)
  {
    free(g->row_ptr);
    free(g->col_idx);
    g->n = 0;
    g->m = 0;
    g->row_ptr = NULL;
    g->col_idx = NULL;
  }
}
