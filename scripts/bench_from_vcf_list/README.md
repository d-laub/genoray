# from_vcf_list benchmark & profiling harness

Reusable harness for measuring and profiling `SparseVar2.from_vcf_list` peak RAM
and wall-time as a function of cohort size (issue #120). All commands run from
this directory inside the pixi env (`pixi run bash -lc '...'`) unless noted.

## 1. Generate a synthetic cohort

```bash
python generate_cohort.py /tmp/cohort --n-files 2000 --n-variants 30000 \
    --contig chr1 --contig-len 1000000 --shared-frac 0.1 --indel-frac 0.1 --seed 0
```

Each file is a single-sample bgzipped+indexed VCF; shared sites carry identical
REF/ALT across files so the k-way merge actually joins. Emits `manifest.txt`
(one path per line).

## 2. RAM / wall-time sweep (Python entry, /usr/bin/time)

```bash
for k in 100 500 1000 2000; do
  python run_bench.py --manifest /tmp/cohort/manifest.txt --out /tmp/out_$k \
      --chrom chr1 --subset $k --profiler time --results results.csv
done
```

`results.csv` gets one `n_files,wall_s,maxrss_kb,profiler` row per run.

## 3. Python allocations (memray)

```bash
python run_bench.py --manifest /tmp/cohort/manifest.txt --out /tmp/out_m \
    --chrom chr1 --subset 500 --profiler memray
memray flamegraph bench.memray
```

## 4. Rust allocations (dhat) — native binary

```bash
cargo run --profile profiling --no-default-features --features conversion,dhat-heap \
    --bin bench_from_vcf_list -- /tmp/cohort/manifest.txt /tmp/out_dhat chr1
# open dhat-heap.json at https://nnethercote.github.io/dh_view/dh_view.html
```

## 5. CPU hotspots (perf) — native binary

```bash
RUSTFLAGS="-C force-frame-pointers=yes" cargo build --profile profiling \
    --no-default-features --features conversion --bin bench_from_vcf_list
perf record -g -- target/profiling/bench_from_vcf_list /tmp/cohort/manifest.txt /tmp/out_p chr1
perf report
```

## 6. Call counts (callgrind) — small N

```bash
valgrind --tool=callgrind target/profiling/bench_from_vcf_list /tmp/cohort_small/manifest.txt /tmp/out_c chr1
callgrind_annotate callgrind.out.*
```

## 7. Codegen of a hot fn (cargo-show-asm)

```bash
cargo asm --no-default-features --features conversion genoray_core::vcf_list_reader::...
```

## Notes

- The native binary (`bench_from_vcf_list`) calls `orchestrator::run_vcf_list`
  directly (no Python), so it is the clean target for dhat/perf/callgrind/asm.
  Pass an optional 4th arg `reference.fa` to run with a reference; omit it for
  `no_reference` mode.
- `run_bench.py` runs the conversion in a subprocess so `/usr/bin/time -v`
  captures the whole RSS including Rust threads. Sweep N via repeated `--subset`
  invocations against the same manifest.
