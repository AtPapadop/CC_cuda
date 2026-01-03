#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif
// Status codes for results writer functions
typedef enum
{
    RESULTS_WRITER_OK = 0,
    RESULTS_WRITER_IO_ERROR = -1,
    RESULTS_WRITER_MEMORY_ERROR = -2,
    RESULTS_WRITER_INVALID_ARGS = -3
} results_writer_status;

// Append a column of timing results to a CSV file.
// Returns RESULTS_WRITER_OK on success or an error code on failure.
results_writer_status append_times_column(const char *filename, const char *column_name, const double *values, size_t count);

// Ensure that the directory at 'path' exists, creating it if necessary.
// Returns 0 on success, -1 on failure.
int results_writer_ensure_directory(const char *path);

// Join 'dir' and 'file' into 'dest', ensuring not to exceed 'dest_size'.
// Returns 0 on success, -1 on failure.
int results_writer_join_path(char *dest, size_t dest_size, const char *dir, const char *file);

// Extract the matrix stem (filename without directory and extension) from 'matrix_path'.
// Returns 0 on success, -1 on failure.
int results_writer_matrix_stem(const char *matrix_path, char *dest, size_t dest_size);

// Build a results file path in 'dest' using 'output_dir', 'prefix', and the matrix stem from 'matrix_path'.
// Returns 0 on success, -1 on failure.
int results_writer_build_results_path(char *dest, size_t dest_size, const char *output_dir,
                                      const char *prefix, const char *matrix_path);

#ifdef __cplusplus
}
#endif
