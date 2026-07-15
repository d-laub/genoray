#!/usr/bin/env bash
# scripts/profile/perf_record.sh <src> <out-dir> <ref> <threads>
# Confirms inflate/parse dominance and tracks the executor's share.
set -euo pipefail
src=$1 out=$2 ref=$3 threads=$4
perf record -g --call-graph dwarf -o perf.data \
  -- pixi run python /carter/users/dlaub/svar_bench/run_svar2.py "$src" "$out" "$ref" "$threads"
perf report -i perf.data --stdio | head -60
