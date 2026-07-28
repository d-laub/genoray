#!/bin/bash
#SBATCH --job-name=genoray-pr140-b
#SBATCH --partition=carter-compute
#SBATCH --cpus-per-task=48
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/carter/users/dlaub/.claude/jobs/ddd927d5/tmp/sweep2_%j.log

set -euo pipefail

WT=/carter/users/dlaub/projects/genoray/.claude/worktrees/bench-pr140-reader-workers
JD=/carter/users/dlaub/.claude/jobs/ddd927d5/tmp
DATA=$JD/data
PX="pixi run --manifest-path $WT/pixi.toml"

echo "=== node: $(hostname)  nproc=$(nproc) ==="

first_cpus() {
  python3 - "$1" <<'PY'
import os, sys
n = int(sys.argv[1]); cpus = sorted(os.sched_getaffinity(0))
if n > len(cpus): sys.exit(f"requested {n}, have {len(cpus)}")
print(",".join(str(c) for c in cpus[:n]))
PY
}

run_sweep() {  # cores outdir workers htslib overshard src label json
  local cores=$1 out=$2 workers=$3 hts=$4 osh=$5 src=$6 label=$7 js=$8
  local cpuset; cpuset=$(first_cpus "$cores")
  echo "### sweep cores=$cores label=$label workers=$workers hts=$hts overshard=$osh"
  taskset -c "$cpuset" $PX python $JD/bench_sharded.py \
    --src "$src" --outdir "$out" --reps 5 --threads "$cores" \
    --workers "$workers" --htslib "$hts" --overshard "$osh" \
    --json-out "$js" --label "$label"
}

# ---- A. Multi-contig REGRESSION regime -------------------------------------
# 48 cores + 8 contigs: planner gives concurrent_chroms=7, so main's
# processing_threads=5 per contig (35 shard threads, oversubscribed) vs the
# PR's reader_workers=2 (14). Does the PR's tighter bound cost throughput?
run_sweep 48 $JD/p2A 1,2,3,5,8 0 4 $DATA/s1000_v100k_8c.vcf.gz \
  s1000_v100k_8c_c48 $JD/sw2_A.json

# ---- B. Single-contig at 48 cores ------------------------------------------
# main W=35 vs PR W=43 -- both far past the knee; confirms the plateau.
run_sweep 48 $JD/p2B 3,7,11,19,27,35,43 0 4 $DATA/s1000_v100k.vcf.gz \
  s1000_v100k_c48 $JD/sw2_B.json

# ---- C. OVERSHARD_FACTOR axis (never swept before) -------------------------
# Each work unit builds a FRESH VcfRecordSource (reopen + reload the index), so
# more shards = more open/seek overhead but better skew absorption. w fixed at 7.
run_sweep 16 $JD/p2C 7 0 1,2,4,8,16 $DATA/s1000_v100k.vcf.gz \
  overshard_c16_w7 $JD/sw2_C.json

# ---- D. Multi-contig re-run at 16 cores with more reps ---------------------
# The 3-rep pass on a shared node gave a non-monotonic curve (26/26/26/6.7/11.7/21.5s);
# re-measure with 5 reps to separate a real threshold from node noise.
run_sweep 16 $JD/p2D 1,2,3,5,7,11 0 4 $DATA/s1000_v100k_8c.vcf.gz \
  s1000_v100k_8c_c16_rerun $JD/sw2_D.json

echo "=== ALL DONE ==="
