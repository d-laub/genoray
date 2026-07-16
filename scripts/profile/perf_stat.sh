#!/usr/bin/env bash
# scripts/profile/perf_stat.sh <src> <out-dir> <ref> <threads>
# task-clock vs wall tells us whether N cores are actually busy.
set -euo pipefail
src=$1 out=$2 ref=$3 threads=$4
perf stat -e task-clock,context-switches,cpu-migrations,cache-misses,instructions,cycles \
  -- pixi run python /carter/users/dlaub/svar_bench/run_svar2.py "$src" "$out" "$ref" "$threads"
