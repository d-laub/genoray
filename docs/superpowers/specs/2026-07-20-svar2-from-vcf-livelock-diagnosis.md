# SVAR2 `from_vcf` Concurrent-Chromosome Livelock — Root-Cause Diagnosis (#135)

**Status:** Phase 1 gate. This is the written root cause + evidence that gates re-planning the Phase 2 fix. No fix is applied here.

**Date:** 2026-07-20
**Box:** 8 physical cores, run under a SLURM memory cgroup (`/slurm/uid_1111/job_13273029`).
**Build:** `maturin develop --release` at commit `24bb91e` (Task 4 instrumentation: `GENORAY_TRACE` heartbeats + a de-lied `shard=` reader-CPU telemetry column).

---

## 1. TL;DR root cause

Under **≥2 concurrent chromosomes**, `SparseVar2.from_vcf` **accumulates reader/assembler memory without bound and never makes durable forward progress**, growing to **~21.8 GB RSS in ~30 s** and then being **OOM-killed** (on a larger-RAM box this presents as an indefinite hang — the reported "livelock"). It is **not a logical (cyclic-wait) deadlock**: the reader threads are pinned at ~100 % CPU (`shard=~100%`) actively working, and under reduced data pressure the pipeline crawls forward a few chunks before still OOMing.

The structural driver is **the absence of any cross-lane (total) memory budget** on the reader/assembler side. Each chromosome runs **one** shard-worker thread (`shard_exec::run` spawns `workers = processing_threads` threads, `src/shard_exec.rs:193,211`; the failing regime is `workers=1` — see `[plan]` below). The `OVERSHARD_FACTOR=4` (`src/orchestrator.rs:58,390-396`) split produces **4 shard work-*units*, not 4 threads**: they are seeded on a FIFO queue and pulled **sequentially** by that single worker (`src/shard_exec.rs:201-205,237`). So `threads=32` → **5 concurrent chromosomes** → **~5 worker threads** (plus BGZF decode threads) on 8 cores, each holding an in-progress `ChunkAssembler` staging **16 007-sample-wide** records/FORMAT data. All the pipeline channels are *bounded* (`tx_res=workers*2`, `tx_dense=6`, `tx_sparse=8`, `tx_long=2`) and the `ReorderBuffer` stays near-empty at `workers=1` (chunks arrive in ordinal order, streamed via the `ordinal==head` fast path, `src/shard_exec.rs:83-84`) — so the runaway is **not** a backlog of buffered chunks and **not** reorder-buffer HOL. It is the **aggregate reader/assembler footprint of ~5 concurrent lanes**, each buffering a shard's worth of 16 007-sample records, with nothing bounding the *sum* across lanes → ~21.8 GB → OOM.

> The *exact* allocation that grows to 21.8 GB is not yet pinned: a single in-progress chunk is only ~480 MB (5000 × 16 007 × ~6 B), so 5 lanes × one chunk ≈ 2.4 GB cannot explain 21.8 GB. The likely culprit is per-shard record/FORMAT staging inside the assembler (a shard's fetch window is ~1/4 of a chromosome — tens of thousands of records × 16 007 samples), i.e. the known "within-contig FORMAT-staging" cost, scaled by ~5 concurrent lanes. **Phase 2 must heap-profile this before choosing a lever** (see §6).

At **1 concurrent chromosome** (`threads=6`) the same code makes progress and never OOMs, because a single lane's assembler footprint stays within budget and the one executor drains it.

---

## 2. Reproduction

### Cohort (the real #135 data, user-provided)
`/carter/shared/data/gdc/somatic/wgs_DR45/results/gdc_wgs_DR45.bcf`
- **88 GB** BCF, **16 007 samples**, **2 779 contigs**.
- Native `##FORMAT=<ID=VAF,Number=A,Type=Float,...>` (somatic tumor allele fractions).

### Command (the repro harness from Task 3)
```bash
python scripts/from_vcf_livelock/repro.py \
    --bcf /carter/shared/data/gdc/somatic/wgs_DR45/results/gdc_wgs_DR45.bcf \
    --out <out> --threads 32 --timeout 300
# repro.py runs SparseVar2.from_vcf(out, bcf, no_reference=True,
#   format_fields=[FormatField("VAF")], threads=32, chunk_size=5000, overwrite=True)
```

### Smallest reproducing ingredient: **scale**, not thread config
The synthetic 2 000-sample / 517 MB cohort (Task 2) does **not** reproduce the livelock — it *completes* in the identical `threads=32` / 5-concurrent regime (~156 s, valid 950 MB store). It is simply too small to out-run the drain before finishing. **Sample width (16 007) and total data volume are the missing ingredient**: they make the reader buffer grow to the OOM ceiling before a chromosome can finish. Any synthetic repro must therefore be at real-cohort scale (thousands of samples, tens of GB) to trip it — this is why the synthetic harness alone was insufficient and the real cohort was required.

> Fixture note (separate, real bug): the synthetic `vcfixture gt-vaf` cohort contained symbolic `<DEL>` ALTs, which `no_reference=True` short-read conversion rejects on the first shard (`ValueError: symbolic/breakend ALT '<DEL>' ... is out of scope (short-read only)`). That crash is unrelated to the livelock and was filtered out (`bcftools view -e 'ALT[*]~"<"'`) before any livelock testing. See §7.

---

## 3. Pipeline configs (both regimes)

**Failing regime — `threads=32`:**
```
Pipeline Config: 5 concurrent chromosomes | 2 HTSlib decompression threads each
    (30 pipeline/BGZF threads, 1 reader-side processing/shard threads).
[plan chr1]  workers=1 shards=4     [plan chr4]  workers=1 shards=4
[plan chr7]  workers=1 shards=4     [plan chr10] workers=1 shards=4
[plan chr13] workers=1 shards=4
```
This is the **exact issue regime** named in the plan: 5 concurrent, htslib=2, processing_threads=1. Note **`shards=4` despite `workers=1`** — the over-shard factor.

**Control regime — `threads=6`:**
```
Pipeline Config: 1 concurrent chromosomes | 1 HTSlib decompression threads each
    (5 pipeline/BGZF threads, 1 reader-side processing/shard threads).
```

---

## 4. Per-lane heartbeat evidence (`GENORAY_TRACE=1`)

### 4a. Failing regime (`threads=32`, full cohort) — stalls at ordinal 0
Every one of the 5 concurrent lanes processes **chunk/ordinal 0 fully, end-to-end**, then **no lane ever advances to ordinal 1**:
```
[trace chrN] reader: shard 0 assembled chunk (unit ordinal 0) local=0 rows=5000
[trace chrN] reader: forwarded ordinal 0 to tx_dense
[trace chrN] exec: dense2sparse enter chunk 0
[trace chrN] exec: dense2sparse exit chunk 0
[trace chrN] exec: sent SparseChunk 0
        (for N in chr1, chr4, chr7, chr10, chr13 — then silence; no "ordinal 1")
```
So the stall is **not** at pipeline start-up and **not** inside `dense2sparse` — chunk 0 flows reader → executor → writer cleanly. The pipeline wedges **between forwarding chunk 0 and producing the next chunk**, i.e. the single per-lane worker is stuck **assembling the next chunk** (`shard=~100%` CPU) while the executor idles — the read/assembler side, not the encode or write side.

### 4b. The de-lied telemetry (the smoking gun)
The pre-existing monitor reported `read=0%` for the whole run — which is a **lie**: the `read-{chrom}` thread it sampled is the *parked periodic-monitor helper* (`src/orchestrator.rs:340`), not the threads doing I/O. The Task-4 `shard=` column (aggregating the real `shard-worker-*` TIDs via a per-chromosome registry) reveals the truth:
```
[chrN t=5s] tx_dense=0/6 tx_sparse=0/8 tx_long=0/2 | cpu read=0% shard=102% exec=0% cw=0% lw=0%
```
- `shard=~100%` (102/101/88 %): the shard-reader threads are **pegged at a full core each**, actively working.
- `exec=0%`: the executor is **idle**, starved of the next in-order chunk.
- `tx_dense/tx_sparse/tx_long = 0`: the channels are drained (chunk 0 already consumed) and nothing new is being emitted.

**100 % CPU + zero forward progress = a livelock (active spin/work without progress), not a deadlock.**

### 4c. OOM confirmation
```
Memory cgroup out of memory: Killed process (python)
    total-vm:29032264kB, anon-rss:21866932kB   (≈ 21.8 GB RSS)
```
~21.8 GB accumulated in ~30 s while committing **zero** durable chunks past ordinal 0.

### 4d. Control contrast (`threads=6`, 1 concurrent) — bug does not fire
```
[chr1 t=30s] tx_dense=0/6 ... | cpu read=0% exec=39% cw=0% lw=0%
```
- `exec` is **active (~39 %)** — the executor runs and drains.
- Ran the full 200 s timeout with **no OOM** (memory bounded by the single lane's read-ahead).
- `tx_dense` still 0 only because a 5 000-var × 16 007-sample chunk is genuinely slow to assemble at this scale — it is progressing, not wedged. (On the small synthetic cohort the identical `threads=6` path streams `chunk 0…58` and completes.)

### 4e. Scale/pressure gradient (rules out hard HOL deadlock)
Restricting to `regions=[chr1..chr6 :1-40 000 000]` (still `shards=4`, ~6 concurrent) **advances** ordinal 0→1→2→3 and *then* OOMs (`RC=137`, SIGKILL). Less data pressure ⇒ it limps a few chunks forward before the buffer still runs away. This proves the wedge is a **progressive unbounded-buffering livelock**, not a fixed cyclic-wait deadlock: with enough slack it moves; it just never bounds its memory.

---

## 5. Hypothesis adjudication

Plan hypotheses were (a) reader contention / oversubscription, (b) ReorderBuffer head-of-line (HOL), (c) thread oversubscription.

- **(a) reader contention — CONFIRMED as the primary driver, but as *memory* contention, not thread over-count.** The failing regime is `workers=1` per chromosome (§3 `[plan]`), so it is **~5 concurrent worker threads**, not 20 — the 4 shards are *sequential units*, not concurrent readers (`src/shard_exec.rs:193,211,237`). What contends is **memory**: ~5 concurrent assemblers each staging 16 007-sample-wide shard data, with no bound on the sum. `shard=~100% / exec=0%` is that single per-lane worker pegged on assembling the next chunk while the executor idles for input.
- **(c) thread oversubscription — SECONDARY.** ~5 worker threads + up to 30 BGZF decode threads on 8 cores means each worker gets a fraction of a core, so assembling the next chunk is *slow* — but slowness alone doesn't OOM. Oversubscription aggravates; the OOM is the memory-budget defect.
- **(b) ReorderBuffer HOL — REFUTED for this regime.** With `workers=1`, chunks are produced in ordinal order and the `ReorderBuffer` streams them via the `ordinal==head` fast path (`src/shard_exec.rs:83-84`); its `pending` map only accumulates when `workers>1` (out-of-order completion). The reorder buffer is **not** the memory sink here. (HOL could matter in `workers≥2` regimes — a `threads=32` + few-contigs config gives `processing_threads=7` — but that is a *different* regime than #135's `processing_threads=1`.)
- **The `read=0%` telemetry actively misled diagnosis** — it hid that the reader/assembler, not the executor, is the CPU-bound and memory-growing party. Fixing it (the `shard=` column) was necessary to localize the stall.

---

## 6. Recommended Phase 2 fix direction

**Phase-2 pre-requisite (do this FIRST):** heap-profile a failing `threads=32` run (e.g. `dhat`, or the existing `--dhat` bench path) to identify *which* allocation grows to ~21.8 GB — the per-shard `ChunkAssembler` record/FORMAT staging, the indexed-fetch `RecordSource` buffer, or something else. The channels and reorder buffer are bounded/empty at `workers=1` (§5), so the sink is in the reader/assembler; pinning the exact allocation is what tells you which lever below actually moves peak RSS. **Do not commit to a lever before this** — the arithmetic (one chunk ≈ 480 MB, 5 lanes ≈ 2.4 GB ≠ 21.8 GB) shows a single in-flight chunk per lane is not the story.

Then, ordered by expected leverage:

1. **Bound *total* (cross-lane) reader/assembler memory with real backpressure.** Every pipeline channel is already bounded, but nothing bounds the **sum** of the ~5 concurrent lanes' in-progress assembler footprints. A conversion-wide memory budget (a shared semaphore/permit sized in bytes, acquired before a lane assembles a chunk and released as it drains) would make a lane **block** rather than let aggregate RSS scale with `concurrent_chroms`. This is the direct fix for the OOM and the highest-leverage change.
2. **Cap `concurrent_chroms` (and/or chunk width) as a function of sample count / available RAM.** At 16 007 samples each lane's per-chunk footprint is large; 5 concurrent lanes multiply it. Deriving `concurrent_chroms` from `available_mem / per-lane-footprint` (not just `(threads-1)/6`) bounds peak directly. Reducing `chunk_size` at high sample counts shrinks the per-lane footprint similarly.
3. **(Lower leverage / verify first) Reconsider `OVERSHARD_FACTOR` only for its *assembler* cost, not thread count.** With `workers=1` the 4 shards are sequential, so they do **not** multiply concurrent threads — dropping the factor will **not** by itself fix the OOM. It may still matter if per-shard fetch-window staging is the allocation the profile implicates (a bigger shard = a bigger fetch window buffered at once); decide after the heap profile. The PGEN path already reasons about shard/memory tradeoffs (`src/orchestrator.rs:508-532`); the VCF path (`:390-396`) does not.
4. **Make `max_mem` actually cap peak.** Peak here is reader/assembler-dominated and currently uncounted (matches the known "`max_mem` can't cap peak (reader-dominated)" limitation). Once (1) exists, wire the total-memory budget to `max_mem` so the user knob is meaningful.

## 6a. Acceptance test the fix MUST satisfy

- **Correctness (byte-identical):** on a fixed input, the `threads=32` (≥2 concurrent) output store must be **byte-identical** to the `threads=6` (1 concurrent) output store — differential comparison of the two `<out>` directories (`meta.json` + every `{chrom}/{dense,fields,indel,var_key}` array). The concurrency fix must not change results. (The clean synthetic cohort already completes in both regimes and is the cheap CI oracle for this invariant; the real cohort is the scale oracle.)
- **Liveness (no OOM / no hang):** `repro.py --bcf <real 88GB BCF> --threads 32 --timeout <T>` must **complete** (exit 0), with peak RSS bounded well under the cohort's total size and independent of `concurrent_chroms`. The existing `tests/test_from_vcf_livelock.py::test_multi_concurrent_completes` (currently `xfail(strict=True)`) flips to **pass** when fixed.
- **No regression at 1 concurrent:** `threads=6` throughput must not degrade.

---

## 7. Reproduction assets (this branch)

- `scripts/from_vcf_livelock/generate_repro.py` — synthetic multi-contig VAF cohort generator (control/CI oracle; does **not** reproduce the livelock by itself — see §2).
- `scripts/from_vcf_livelock/repro.py` — watchdog harness (`--bcf/--out/--threads/--timeout`; exit 0 = completed, 124 = timed out, other = crash/OOM).
- `tests/test_from_vcf_livelock.py` — regression: `test_single_concurrent_completes` (control) + `test_multi_concurrent_completes` (`xfail(strict=True)`, flips to pass once fixed).
- `GENORAY_TRACE=1` — env-gated reader/exec heartbeats + the `[plan]` line; the monitor's `shard=` column de-lies reader CPU (commit `24bb91e`).

**Known caveat carried forward:** the synthetic-fixture reheader trick (`Number=1`→`Number=A`) and the `<DEL>` symbolic-ALT contamination are documented in `scripts/from_vcf_livelock/README.md`; the symbolic-ALT crash is a *separate* short-read-mode robustness issue, not the livelock.

---

## 8. Phase 1 gate

Root cause established with direct heartbeat + OOM + de-lied-CPU evidence. **Do not begin Phase 2 here** — this finding gates a re-planned fix (§6) validated against the §6a acceptance tests.
