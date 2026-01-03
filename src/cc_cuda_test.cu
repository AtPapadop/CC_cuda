// src/cc_cuda_test.cu
//
// Quick test driver for CUDA CC (atomic push, per-vertex thread mapping).
// Mirrors cc_pthreads test structure: loads graph, sweeps block size, repeats runs,
// writes CSV timing columns, optionally writes labels.
//
// Build (example):
//   nvcc -O3 -Iinclude -o cc_cuda_test src/cc_cuda_test.cu src/cc_cuda_thread_per_vertex.cu \
//        src/graph.c src/opt_parser.c src/results_writer.c -lmatio
//
// Run:
//   ./cc_cuda_test [OPTIONS] <matrix-file-path>
//
// Notes:
// - Expects load_csr_from_file/free_csr from graph.h
// - Uses opt_parser + results_writer helpers like your pthreads test
// - Calls compute_connected_components_cuda_thread_per_vertex (declared in cc_cuda.h)

#define _GNU_SOURCE
#include <errno.h>
#include <getopt.h>
#include <inttypes.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <cuda_runtime.h>

#include "cc_cuda.h"
#include "graph.h"
#include "opt_parser.h"
#include "results_writer.h"

static void print_usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s [OPTIONS] <matrix-file-path>\n\n"
            "Options:\n"
            "  -b, --block-sizes SPEC  CUDA block sizes to sweep (default 256; comma/range syntax)\n"
            "  -d, --device ID         CUDA device id (default 0)\n"
            "  -r, --runs N            Number of runs to average (default 1)\n"
            "  -o, --output DIR        Output directory (default 'results')\n"
            "  -s, --save-labels       Write cuda_labels.txt (default disabled)\n"
            "  -v, --verbose           Print extra info\n"
            "  -h, --help              Show this message\n",
            prog);
}

static double now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int cmp_u32(const void *a, const void *b)
{
    uint32_t x = *(const uint32_t *)a;
    uint32_t y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

static uint32_t count_unique_labels_u32(const uint32_t *labels, uint32_t n)
{
    if (!labels || n == 0) return 0;

    uint32_t *tmp = (uint32_t *)malloc((size_t)n * sizeof(uint32_t));
    if (!tmp) return 0;

    memcpy(tmp, labels, (size_t)n * sizeof(uint32_t));
    qsort(tmp, (size_t)n, sizeof(uint32_t), cmp_u32);

    uint32_t uniq = 1;
    for (uint32_t i = 1; i < n; ++i)
        if (tmp[i] != tmp[i - 1]) ++uniq;

    free(tmp);
    return uniq;
}

int main(int argc, char **argv)
{
    int runs = 1;
    const char *path = NULL;
    const char *output_dir = "results";
    const char *block_spec = "256";
    int save_labels = 0;
    int verbose = 0;
    int device_id = 0;

    const struct option long_opts[] = {
        {"block-sizes", required_argument, NULL, 'b'},
        {"device", required_argument, NULL, 'd'},
        {"runs", required_argument, NULL, 'r'},
        {"output", required_argument, NULL, 'o'},
        {"save-labels", no_argument, NULL, 's'},
        {"verbose", no_argument, NULL, 'v'},
        {"help", no_argument, NULL, 'h'},
        {NULL, 0, NULL, 0}
    };

    int opt, opt_index = 0;
    while ((opt = getopt_long(argc, argv, "b:d:r:o:svh", long_opts, &opt_index)) != -1)
    {
        switch (opt)
        {
        case 'b':
            if (!optarg || *optarg == '\0')
            {
                fprintf(stderr, "Block size specification must not be empty.\n");
                return EXIT_FAILURE;
            }
            block_spec = optarg;
            break;
        case 'd':
        {
            int parsed = 0;
            if (opt_parse_positive_int(optarg, &parsed) != 0)
            {
                fprintf(stderr, "Invalid device id: %s\n", optarg);
                return EXIT_FAILURE;
            }
            device_id = parsed;
            break;
        }
        case 'r':
        {
            int parsed = 0;
            if (opt_parse_positive_int(optarg, &parsed) != 0)
            {
                fprintf(stderr, "Invalid run count: %s\n", optarg);
                return EXIT_FAILURE;
            }
            runs = parsed;
            break;
        }
        case 'o':
            if (optarg[0] == '\0')
            {
                fprintf(stderr, "Output directory must not be empty.\n");
                return EXIT_FAILURE;
            }
            output_dir = optarg;
            break;
        case 's':
            save_labels = 1;
            break;
        case 'v':
            verbose = 1;
            break;
        case 'h':
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        default:
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    if (optind >= argc)
    {
        fprintf(stderr, "Missing matrix file path.\n");
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }
    path = argv[optind];

    OptIntList block_sizes;
    opt_int_list_init(&block_sizes);
    if (opt_parse_range_list(block_spec, &block_sizes, "block sizes") != 0)
    {
        opt_int_list_free(&block_sizes);
        return EXIT_FAILURE;
    }
    if (block_sizes.size == 0)
    {
        fprintf(stderr, "Block size specification must yield at least one value.\n");
        opt_int_list_free(&block_sizes);
        return EXIT_FAILURE;
    }

    if (results_writer_ensure_directory(output_dir) != 0)
    {
        fprintf(stderr, "Failed to create output directory '%s': %s\n", output_dir, strerror(errno));
        opt_int_list_free(&block_sizes);
        return EXIT_FAILURE;
    }

    char labels_path[PATH_MAX];
    int labels_path_ready = 0;
    if (save_labels)
    {
        if (results_writer_join_path(labels_path, sizeof(labels_path), output_dir, "cuda_labels.txt") != 0)
        {
            fprintf(stderr, "Output path too long for labels file: %s\n", strerror(errno));
            opt_int_list_free(&block_sizes);
            return EXIT_FAILURE;
        }
        labels_path_ready = 1;
    }

    int results_path_ready = 0;
    char results_path[PATH_MAX];
    results_path[0] = '\0';
    if (results_writer_build_results_path(results_path, sizeof(results_path), output_dir, "results_cuda", path) != 0)
    {
        fprintf(stderr, "Warning: Failed to build results path: %s\n", strerror(errno));
    }
    else
    {
        results_path_ready = 1;
    }

    // CUDA device info
    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);
    if (dev_count <= 0)
    {
        fprintf(stderr, "No CUDA devices found.\n");
        opt_int_list_free(&block_sizes);
        return EXIT_FAILURE;
    }
    if (device_id < 0 || device_id >= dev_count)
    {
        fprintf(stderr, "Invalid device id %d (found %d device(s)).\n", device_id, dev_count);
        opt_int_list_free(&block_sizes);
        return EXIT_FAILURE;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    printf("Using CUDA device %d: %s\n", device_id, prop.name);

    printf("Loading graph: %s\n", path);

    CSRGraph G;
    if (load_csr_from_file(path, 1, 1, &G) != 0)
    {
        fprintf(stderr, "Failed to load graph from %s\n", path);
        opt_int_list_free(&block_sizes);
        return EXIT_FAILURE;
    }

    uint32_t *labels = (uint32_t *)malloc((size_t)G.n * sizeof(uint32_t));
    if (!labels)
    {
        fprintf(stderr, "Memory allocation failed\n");
        free_csr(&G);
        opt_int_list_free(&block_sizes);
        return EXIT_FAILURE;
    }

    double *run_times = (double *)malloc((size_t)runs * sizeof(double));
    if (!run_times)
    {
        fprintf(stderr, "Memory allocation failed\n");
        free(labels);
        free_csr(&G);
        opt_int_list_free(&block_sizes);
        return EXIT_FAILURE;
    }

    printf("Sweeping %zu block size option%s (%d run%s each).\n",
           block_sizes.size,
           block_sizes.size == 1 ? "" : "s",
           runs,
           runs == 1 ? "" : "s");

    for (size_t idx = 0; idx < block_sizes.size; idx++)
    {
        int bs = block_sizes.values[idx];
        printf("Computing connected components with block size %d (%d run%s)...\n",
               bs, runs, runs == 1 ? "" : "s");

        CudaCCOptions opt_cc;
        memset(&opt_cc, 0, sizeof(opt_cc));
        opt_cc.device_id = device_id;
        opt_cc.block_size = bs;
        opt_cc.verbose = verbose;

        double total_time = 0.0;
        uint32_t last_iters = 0;
        double last_kernel_sec = 0.0;

        for (int run = 0; run < runs; run++)
        {
            double start = now_seconds();

            uint32_t iters = 0;
            double kernel_sec = 0.0;

            int rc = compute_connected_components_cuda_block_frontier_async(&G, labels, &opt_cc, &iters, &kernel_sec);
            if (rc != 0)
            {
                fprintf(stderr, "CUDA CC failed (rc=%d) on run %d.\n", rc, run + 1);
                free(run_times);
                free(labels);
                free_csr(&G);
                opt_int_list_free(&block_sizes);
                return EXIT_FAILURE;
            }

            double elapsed = now_seconds() - start;
            total_time += elapsed;
            run_times[run] = elapsed;

            last_iters = iters;
            last_kernel_sec = kernel_sec;

            printf("  Run %d: total %.6f s (kernel %.6f s), iters=%" PRIu32 "\n",
                   run + 1, elapsed, kernel_sec, iters);
        }

        double average = total_time / runs;
        printf("Average for block size %d: total %.6f s (last kernel %.6f s), iters=%" PRIu32 "\n",
               bs, average, last_kernel_sec, last_iters);

        if (results_path_ready)
        {
            char column_name[64];
            snprintf(column_name, sizeof(column_name), "bs=%d", bs);
            results_writer_status csv_status = append_times_column(results_path, column_name, run_times, (size_t)runs);
            if (csv_status != RESULTS_WRITER_OK)
                fprintf(stderr, "Warning: Failed to update %s (error %d)\n", results_path, (int)csv_status);
        }
    }

    uint32_t num_components = count_unique_labels_u32(labels, G.n);
    printf("Number of connected components (last run): %" PRIu32 "\n", num_components);

    if (labels_path_ready)
    {
        FILE *fout = fopen(labels_path, "w");
        if (!fout)
        {
            fprintf(stderr, "Failed to open output file %s.\n", labels_path);
            free(run_times);
            free(labels);
            free_csr(&G);
            opt_int_list_free(&block_sizes);
            return EXIT_FAILURE;
        }

        for (uint32_t i = 0; i < G.n; i++)
            fprintf(fout, "%" PRIu32 "\n", labels[i]);

        fclose(fout);
        printf("Labels written to %s\n", labels_path);
    }
    else
    {
        printf("Labels not saved (pass --save-labels to enable).\n");
    }

    if (results_path_ready)
        printf("Timing results written to %s\n", results_path);

    free(run_times);
    free(labels);
    free_csr(&G);
    opt_int_list_free(&block_sizes);

    return EXIT_SUCCESS;
}
