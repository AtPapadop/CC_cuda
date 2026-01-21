#!/bin/bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  test.sh [-r RUNS] [-o OUTPUT_DIR] INPUT

Options:
  -r RUNS        Number of runs (default: 10)
  -o OUTPUT_DIR  Output directory (default: ../results)
  -h             Show this help

Example:
  ./test.sh -r 20 -o ../results/myrun data/input.bin
EOF
}

RUNS=10
OUTPUT="../results"

# Parse flags
while getopts ":r:o:h" opt; do
  case "$opt" in
    r) RUNS="$OPTARG" ;;
    o) OUTPUT="$OPTARG" ;;
    h) usage; exit 0 ;;
    \?) echo "Error: Unknown option -$OPTARG" >&2; usage; exit 2 ;;
    :)  echo "Error: Option -$OPTARG requires an argument" >&2; usage; exit 2 ;;
  esac
done
shift $((OPTIND - 1))

# Required positional arg: INPUT
if [[ $# -lt 1 ]]; then
  echo "Error: INPUT is required." >&2
  usage
  exit 2
fi
INPUT="$1"

# Ensure output directory exists
mkdir -p "$OUTPUT"

B_LIST="64,128,256,512,1024"

for m in {0..8}; do
  echo "Running: m=$m runs=$RUNS output=$OUTPUT input=$INPUT"
  ../bin/cc_cuda_test -b "$B_LIST" -r "$RUNS" -m "$m" -o "$OUTPUT" "$INPUT"
done
