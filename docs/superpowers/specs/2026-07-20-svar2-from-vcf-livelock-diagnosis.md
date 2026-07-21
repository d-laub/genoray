# SVAR2 `from_vcf` Concurrent-Chromosome Livelock — Root-Cause Diagnosis (#135)

**Status:** Phase 1 gate. This is the written root cause + evidence that gates re-planning the Phase 2 fix. No fix is applied here.

**Date:** 2026-07-20
**Box:** 8 physical cores, run under a SLURM memory cgroup (`/slurm/uid_1111/job_13273029`).
**Build:** `maturin develop --release` at commit `24bb91e` (Task 4 instrumentation: `GENORAY_TRACE` heartbeats + a de-lied `shard=` reader-CPU telemetry column).

---

## 1. TL;DR root cause

Under **≥2 concurrent chromosomes**, `SparseVar2.from_vcf` **buffers variant chunks in memory without bound and never makes durable forward progress**, growing to **~21.8 GB RSS in ~30 s** and then being **OOM-killed** (on a larger-RAM box this presents as an indefinite hang — the reported "livelock"). It is **not a logical (cyclic-wait) deadlock**: the reader threads are pinned at ~100 % CPU (`shard=~100%`) actively working, and under reduced data pressure the pipeline crawls forward a few chunks before still OOMing.

The structural driver is **read-side over-decomposition with no total-memory backpressure**: the VCF reader splits every chromosome into `processing_threads × OVERSHARD_FACTOR` shards — **4 shards even when `processing_threads == 1`** (`src/orchestrator.rs:58,390-396`). With `threads=32` the budget picks **5 concurrent chromosomes** (`Pipeline Config` below), so **~5 × 4 = 20 shard-reader threads** (plus BGZF decode threads) run on 8 cores, each reading **16 007-sample-wide** records into a per-chromosome global-ordinal reorder buffer far faster than the single per-chromosome executor can drain them. Buffered chunks accumulate unbounded → OOM.

At **1 concurrent chromosome** (`threads=6`) the same code makes progress and never OOMs, because a single lane's read-ahead stays within what the one executor can consume.

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
So the stall is **not** at pipeline start-up and **not** inside `dense2sparse` — chunk 0 flows reader → executor → writer cleanly. The pipeline wedges **between forwarding ordinal 0 and producing ordinal 1**, i.e. in the **multi-shard reader / reorder stage**.

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

- **(a) + (c) — CONFIRMED and are the primary driver.** `OVERSHARD_FACTOR=4` (`src/orchestrator.rs:58`) makes the VCF path spawn `processing_threads × 4` shard readers per chromosome regardless of downstream parallelism (`processing_threads=1` still ⇒ 4 shards, `src/orchestrator.rs:390-396`). Across 5 concurrent chromosomes that is ~20 shard-reader threads + BGZF threads on 8 cores — the `shard=~100%` pegging with `exec=0%` is exactly this over-decomposed read side starving/outrunning the single drain.
- **(b) HOL — CONTRIBUTING, not the whole story.** The per-chromosome reorder buffer must emit chunks in global-ordinal order; the full-cohort run forwards only ordinal 0 then buffers un-emittable later ordinals (consistent with a lagging shard blocking the head). But it is **not a pure HOL deadlock** — §4e shows it can advance when pressure is lower. HOL aggravates the memory runaway; the root defect is the **absence of a bound on total in-flight (read-ahead + reorder) memory**.
- **The `read=0%` telemetry actively misled diagnosis** — it hid that the readers, not the executor, are the CPU-bound party. Fixing it (the `shard=` column) was necessary to localize the stall.

---

## 6. Recommended Phase 2 fix direction

Ordered by expected leverage:

1. **Bound total in-flight reader memory with real backpressure.** The `tx_dense` channel is bounded (cap 6), but the **per-shard read-ahead and/or the reorder buffer are not** — shards keep reading and buffering while the reorder head is blocked. The reader side must **block** (not buffer) when the reorder buffer / downstream is behind, so total buffered memory is bounded regardless of concurrency or cohort size. This is the direct fix for the OOM.
2. **Do not over-shard when there is no downstream parallelism to feed.** With `processing_threads == 1`, `OVERSHARD_FACTOR × 1 = 4` shard readers per chromosome add memory + contention with no drain benefit. Gate the over-shard factor on effective downstream concurrency **and** on `concurrent_chroms`, so total live shard-reader threads stay bounded by cores (e.g. cap `concurrent_chroms × shards_per_chrom` at physical cores). The existing PGEN path already reasons about this (`src/orchestrator.rs:508-532`); the VCF path (`:390-396`) does not.
3. **Make `max_mem` actually cap peak.** Peak here is reader-dominated and currently uncounted (matches the known "`max_mem` can't cap peak (reader-dominated)" limitation). Once (1) bounds read-ahead, wire that bound to `max_mem` so the user knob is meaningful.

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
