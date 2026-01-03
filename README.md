# CC CUDA Connected Components

CUDA implementations of push-based connected components using several thread mappings (thread/warp/block, optional frontier specialization). The `cc_cuda_test` harness loads CSR graphs, sweeps CUDA block sizes, and records timing statistics for benchmarking.

## Repository Layout
- `include/` – public headers (`cc_cuda.h`, graph loader, option parser, CSV writer).
- `src/` – CUDA kernels (`cc_cuda_*`), graph/CLI utilities, and the `cc_cuda_test.cu` driver.
- `build/` – object files produced by `make` (created on demand).
- `bin/` – final executables such as `cc_cuda_test` (created on demand).
- `Makefile` – builds the driver plus kernels with NVCC/GCC.

## Requirements
- CUDA toolkit (NVCC + runtime). Default path is `/usr/local/cuda`; override via `CUDA_HOME`.
- GCC/G++ 12 (configurable by editing `Makefile`).
- [`libmatio`](https://github.com/tbeu/matio) for `.mat` graphs. Adjust `MATIO_INC`, `MATIO_LIB`, `MATIO_LDLIBS` in the Makefile when installed in non-standard locations.

## Building
```bash
make            # builds bin/cc_cuda_test using src/cc_cuda_block_frontier_async.cu + helpers
make clean      # removes build/ and bin/
make print      # echoes the key compiler/linker flags
```
Override toolchain bits on the command line if needed:
```bash
make CUDA_HOME=/opt/cuda CC=gcc-11 CXX=g++-11
```

## Running the Test Harness
`cc_cuda_test` expects a graph stored in MATLAB/CSR format that `load_csr_from_file()` understands (see `include/graph.h`). Example:
```bash
./bin/cc_cuda_test -b 128,256 -r 5 -d 0 -o results ../CC/data/foo.mat
```
Key options (see `src/cc_cuda_test.cu`):
| Flag | Meaning |
| ---- | ------- |
| `-b`, `--block-sizes` | Sweep one or more CUDA block sizes (comma lists or `start:end:step`). |
| `-d`, `--device` | CUDA device ID (default 0). |
| `-r`, `--runs` | Number of repetitions per block size (default 1). |
| `-o`, `--output` | Directory for CSV files and optional labels (`results` by default). |
| `-s`, `--save-labels` | Write component labels for the final run to `cuda_labels.txt`. |
| `-v`, `--verbose` | Propagate verbose logging into the CUDA kernels/utilities. |
| `-h`, `--help` | Print usage text. |

During a run the program prints per-run timing breakdowns plus the number of connected components detected in the final sweep. When `--save-labels` is set, component IDs are written one-per-line so downstream scripts can verify correctness.

If the `results_writer_*` helpers can construct a CSV path (e.g., `results/results_cuda_<graph>.csv`), each block size appends a column containing the individual run times. This makes it easy to chart scaling behavior offline.

## Selecting a Kernel
`include/cc_cuda.h` exposes six variants:
- `compute_connected_components_cuda_thread_per_vertex`
- `compute_connected_components_cuda_thread_frontier_async`
- `compute_connected_components_cuda_warp_per_vertex`
- `compute_connected_components_cuda_warp_frontier_async`
- `compute_connected_components_cuda_block_per_vertex`
- `compute_connected_components_cuda_block_frontier_async`

`cc_cuda_test` currently calls the block-frontier async flavor. Swap in a different implementation by editing `src/cc_cuda_test.cu` (look for `compute_connected_components_cuda_block_frontier_async(...)`). All kernels accept the same `CudaCCOptions` struct (device, block size, verbosity) and report iteration counts plus cumulative kernel time.

## Troubleshooting
- Use `-v` to surface detailed stderr breadcrumbs (allocation stages, kernel launches, etc.).
- `cuda-memcheck ./bin/cc_cuda_test ...` can help catch invalid accesses.
- If builds fail with missing `matio`, either install the library or remove the dependency by pruning MATIO flags in the Makefile.
- For performance experiments, pin the GPU clocks (`nvidia-smi -lgc`) and run on an otherwise idle system.

## License
No explicit license file is provided. Treat the code as private coursework/research unless instructed otherwise.
