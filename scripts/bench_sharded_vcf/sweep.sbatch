#!/bin/bash
#SBATCH --job-name=genoray-pr140
#SBATCH --partition=carter-compute
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/carter/users/dlaub/.claude/jobs/ddd927d5/tmp/sweep_%j.log

set -euo pipefail

WT=/carter/users/dlaub/projects/genoray/.claude/worktrees/bench-pr140-reader-workers
JD=/carter/users/dlaub/.claude/jobs/ddd927d5/tmp
DATA=$JD/data
PX="pixi run --manifest-path $WT/pixi.toml"

echo "=== node: $(hostname)  nproc=$(nproc) ==="
taskset -cp $$ || true

# ---- 1. Generate the remaining shape-sweep datasets (constant ~100M cells) ----
gen() {
  local out=$1; shift
  if [ -f "$out" ]; then echo "have $out"; return; fi
  $PX python $JD/gen_vcf.py --out "$out" "$@" --threads 3
}
gen $DATA/s250_v400k.vcf.gz   --samples 250  --variants 400000 &
gen $DATA/s4000_v25k.vcf.gz   --samples 4000 --variants 25000  &
gen $DATA/s1000_v100k_8c.vcf.gz --samples 1000 --variants 100000 \
    --contigs chr15,chr16,chr17,chr18,chr19,chr20,chr21,chr22 &
wait
ls -la $DATA

# ---- 2. Core-count sweep on the baseline shape ----
# Pinned CPU sets so `available_parallelism()` and the real core count agree.
first_cpus() {  # N -> comma list of the first N CPUs THIS job actually owns
  python3 - "$1" <<'PY'
import os, sys
n = int(sys.argv[1])
cpus = sorted(os.sched_getaffinity(0))
if n > len(cpus):
    sys.exit(f"requested {n} cpus, allocation has {len(cpus)}")
print(",".join(str(c) for c in cpus[:n]))
PY
}

run_sweep() {  # cores outdir workers htslib src label json
  local cores=$1 out=$2 workers=$3 hts=$4 src=$5 label=$6 js=$7
  local cpuset
  cpuset=$(first_cpus "$cores")
  echo "### sweep cores=$cores cpuset=$cpuset label=$label workers=$workers hts=$hts"
  # Slurm hands out NON-CONTIGUOUS cpu ids, so pin to the real allocated ids
  # rather than 0-(N-1); `available_parallelism()` then reports `cores`.
  taskset -c "$cpuset" $PX python $JD/bench_sharded.py \
    --src "$src" --outdir "$out" --reps 3 --threads "$cores" \
    --workers "$workers" --htslib "$hts" \
    --json-out "$js" --label "$label"
}

run_sweep 16 $JD/o16 1,3,5,7,9,11,14,20 0 $DATA/s1000_v100k.vcf.gz \
  s1000_v100k_c16_hts0 $JD/sw_c16_hts0.json
run_sweep 16 $JD/o16 3,11 1 $DATA/s1000_v100k.vcf.gz \
  s1000_v100k_c16_hts1 $JD/sw_c16_hts1.json

run_sweep 32 $JD/o32 3,7,11,19,27,40 0 $DATA/s1000_v100k.vcf.gz \
  s1000_v100k_c32_hts0 $JD/sw_c32_hts0.json
run_sweep 32 $JD/o32 19,27 1 $DATA/s1000_v100k.vcf.gz \
  s1000_v100k_c32_hts1 $JD/sw_c32_hts1.json

run_sweep 12 $JD/o12 1,3,5,7,11 0 $DATA/s1000_v100k.vcf.gz \
  s1000_v100k_c12_hts0 $JD/sw_c12_hts0.json

# ---- 3. Shape sweep at 16 cores: does the knee move with samples/variants? ----
run_sweep 16 $JD/oB 1,3,5,7,11,14 0 $DATA/s250_v400k.vcf.gz \
  s250_v400k_c16 $JD/sw_B.json
run_sweep 16 $JD/oC 1,3,5,7,11,14 0 $DATA/s4000_v25k.vcf.gz \
  s4000_v25k_c16 $JD/sw_C.json

# ---- 4. Multi-contig at 16 cores: 8 contigs, concurrent_chroms > 1 ----
run_sweep 16 $JD/oD 1,2,3,5,7,11 0 $DATA/s1000_v100k_8c.vcf.gz \
  s1000_v100k_8contig_c16 $JD/sw_D.json

echo "=== ALL DONE ==="
