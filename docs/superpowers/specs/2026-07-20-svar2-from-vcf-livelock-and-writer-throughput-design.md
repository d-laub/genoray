# SVAR2 single-source writers: fix the `from_vcf` concurrent-chromosome livelock, then a throughput pass

**Date:** 2026-07-20
**Issue:** [#135](https://github.com/d-laub/genoray/issues/135) — `from_vcf` livelocks with ≥2 concurrent chromosomes; large-cohort throughput too low.
**Status:** design — Phases 0–1 concrete; Phase 2 is a set of diagnosis-keyed branches to be re-planned after the spike; Phase 3 concrete.

## Problem

`SparseVar2.from_vcf` **livelocks with zero forward progress** when converting a large multi-sample BCF with **more than one concurrent chromosome**. The encode (`exec`) stage shows intermittent CPU, but the commit/writer stages (`cw`, `lw`) never fire — over a 16.8 h run every contig stayed at `tx_dense=0/6 tx_sparse=0/8 tx_long=0/2`, `cw=0% lw=0%`, staging dir empty. The same conversion is **correct and completes at 1 concurrent chromosome** (a 2 Mb chr22 slice finished in 602 s). Memory is bounded (~33 GB RSS) — not a memory-growth problem. A secondary concern: even the working single-stream path is ~290 variants/s single-threaded, impractical for 16k-sample / 100M+-variant whole-genome cohorts.

Reported at genoray 3.2.1 on the GDC WGS somatic merged BCF: 16,007 samples, 348,259,675 variants, 83 GB, `no_reference=True`, FORMAT `GT` + `VAF` (`Number=A, Float`), the call using `threads=32, chunk_size=5000` (→ "5 concurrent chromosomes").

## Scope (agreed)

1. **Livelock first, then throughput.** The livelock blocks the very concurrency throughput depends on, so it is fixed before any concurrent-path optimization.
2. **Optimize all three single-source writers independently** — `from_vcf`, `from_pgen`, `from_svar1` — end to end, including their distinct reader front-ends, not just the shared writer stage.
3. **`from_vcf_list` is out of scope.** It is already optimized (min-heap merge, cross-contig `malloc_trim` RAM fix) and forces `concurrent_chroms = 1` by construction, so it does not exhibit this bug.
4. **Fixture generation via `vcfixture-rs 0.3.0` `bulk` CLI** — synthetic, private-data-free repro and benchmark inputs.

## Architecture (as traced, read-only)

`from_vcf` runs a pure-Rust htslib reader with no Python/GIL in the loop (the whole pipeline runs under `py.detach`, `src/lib.rs:181`).

- **Entry:** `SparseVar2.from_vcf` (`python/genoray/_svar2.py:749`) → Rust `run_conversion_pipeline` (`src/lib.rs:143`) → fans contigs to `orchestrator::process_chromosome` (`src/orchestrator.rs:254`), **one call per contig**.
- **Concurrency:** `plan_thread_budget` (`src/budget.rs:31`): `concurrent_chroms = min(usable_cores / 6, n_chroms)` (6 = 4 pipeline + 2 htslib threads per chrom). Per-contig lanes run as `par_iter` over a rayon pool of `concurrent_chroms` workers (`src/lib.rs:217,225`); each `process_chromosome` blocks its rayon worker in `.join()` (`orchestrator.rs:753`) while its own `std::thread`s do the work.
- **Per-contig stages & bounded channels** (`orchestrator.rs:315-317`): `read → exec` via `bounded(6)`; `exec → cw` via `bounded(8)`; `exec(bank) → lw` via `bounded(2)`. Threads: `read-{chrom}`, `exec-{chrom}` (`executor::run_compute_engine`, `src/executor.rs:22`), `cw-{chrom}` (`writer::run_io_writer`), `lw-{chrom}` (`writer::run_long_allele_writer`).
- **Sharded reader path (what actually runs here):** whole-contig conversion fills explicit `[0, contig_len)` ranges to enable sub-contig sharding (`_svar2.py:722-740`). `plan_vcf_shards` tiles `[0,len)` into `processing_threads × OVERSHARD_FACTOR(4)` bp-slices, so even `processing_threads=1` yields 4 shards → the **sharded** branch (`orchestrator.rs:383-443`) via `shard_exec::run` (`src/shard_exec.rs:141`). Shard results pass through a `ReorderBuffer` (`shard_exec.rs:37`) before `tx_dense`.
- **Shared across `from_*`:** all three converge on `process_chromosome` and share the identical exec/writer/channels; they diverge **only** in the reader `SourceSpec` (Vcf `lib.rs:230`, Pgen `lib.rs:371`, Svar1 `lib.rs:1014`). PGEN also uses `shard_exec::run`; svar1 uses the single-reader loop.

### Two facts that reframe the symptom

- **No proven hard cross-lane deadlock exists.** Per-contig channels/threads are self-contained; nothing (no shared writer, output mutex, or cross-contig channel) is shared across lanes except the outer rayon pool, the one physical BCF + its htslib decode threads, and stdout. So the ≥2-chromosome trigger implies **concurrency-gated contention or starvation**, not a lock cycle — the fix must be found empirically.
- **The telemetry lies about the reader.** In the sharded path the sampled `read-{chrom}` thread just parks in `rx_res.recv()`; the real read CPU is on untracked `shard-worker-{i}` threads. **`read=0%` does not prove the reader is idle** (`shard_exec.rs:271`, `monitor.rs:102-105`). `tx_dense=0` is consistent with *either* the `ReorderBuffer` holding chunks back *or* the reader never assembling one full `chunk_size`-variant `DenseChunk`.

## Phase 0 — Repro harness (`vcfixture-rs 0.3.0 bulk`)

**Goal:** a private-data-free BCF that trips the livelock at the **smallest scale that still spawns ≥2 concurrent chromosomes**. The bug is concurrency-gated, so the 83 GB / 348 M-variant scale is unnecessary for diagnosis.

- Install: `cargo install vcfixture --features cli` (crates.io `vcfixture = 0.3.0`; repo `d-laub/vcfixture-rs`).
- Generate a BCF with: **≥2 contigs** (so `concurrent_chroms ≥ 2`), a somatic-style **`VAF` FORMAT field** (`Number=A, Type=Float`) to exercise the F×N staging path, ploidy 2, `no_reference=True`-compatible (already-normalized).
- **Sizing:** start ~2–4k samples and a few M variants (~GB-scale); scale up only until the livelock reproduces. `threads` chosen so the `Pipeline Config` log line reports ≥2 concurrent chromosomes on the dev box.

**Open item to resolve first (harness detail, not a plan blocker):** confirm `vcfixture bulk` can emit a somatic `VAF Number=A Float` FORMAT field (via a somatic profile or a dialed payload preset — it ships fitted profiles such as `germline-1kgp`). If it cannot, fall back to post-processing its output to inject the `VAF` field. Resolve by reading the bulk-generation guide / `vcfixture bulk --help` before building the fixture.

**Deliverable:** a documented generator command + a small wrapper under `scripts/` (sibling to `scripts/bench_from_vcf_list/`) that emits the BCF + its `.csi`, reproducible by seed.

## Phase 1 — Instrument & diagnose (systematic-debugging)

Reproduce the livelock deterministically, then add heartbeat instrumentation to disambiguate the three failure modes the current telemetry cannot separate — *reader never assembles chunk 0* vs *exec never finishes a chunk* vs *cw never receives*:

- Heartbeat at `shard_exec.rs:217` (per `read_next_chunk` return) — is the reader producing `DenseChunk`s at all?
- Heartbeat at `executor.rs:34` / `executor.rs:47` (per `dense2sparse_vk` enter/exit and per `tx_sparse.send`) — is exec completing chunks end-to-end?
- Log `processing_threads`, `workers`, and `shards.len()` for the actual repro config.
- Fix the misleading telemetry: sample the `shard-worker-*` threads (real read CPU), not the parked `read-*` thread.

**Hypotheses to bisect (none yet proven; the spike decides):**
- **(a) Reader contention** — up to `concurrent_chroms × shards × htslib_threads` concurrent `IndexedReader`s seeking into one physical BCF → per-reader throughput collapse, so no lane assembles chunk 0 in observable time.
- **(b) `ReorderBuffer` head-of-line blocking** (`shard_exec.rs:71-95`) — non-head shards buffer until shard 0 sends `Done`; bites only at `processing_threads ≥ 2`. Confirm which `workers` regime the repro is in.
- **(c) htslib decode-thread oversubscription** — `concurrent_chroms × shards × 2` decode threads ≫ cores.

**Deliverable:** a written root-cause finding (which stage stalls and why), with the instrumented evidence. **This ends the spike; Phase 2 is then re-planned concretely against the finding.**

## Phase 2 — Fix the livelock (re-planned after Phase 1)

Specified as branches keyed on the Phase-1 root cause; the concrete fix plan is written after the spike, not now:

- If **(a) reader contention:** cap concurrent `IndexedReader`s per physical file; bound total htslib threads to the core budget; possibly serialize shard readers within a lane.
- If **(b) `ReorderBuffer` HOL:** stream non-head shard chunks without waiting on shard 0's `Done`, or re-order the forwarding contract.
- If **(c) thread oversubscription:** clamp `concurrent_chroms × shards × htslib_threads` to `usable_cores` in `plan_thread_budget`.

**Acceptance:** the synthetic repro converts to completion with ≥2 concurrent chromosomes, byte-identical output to the 1-concurrent-chromosome path (differential test), forward progress visible in `tx_*`/`cw`/`lw` telemetry.

## Phase 3 — Throughput pass (performant-py-rust), three writers independently

Only after Phase 2. Profile each writer end to end, including its distinct reader.

- **Correctness oracle:** `vcfixture` `GroundTruth` arrays (positions, genotypes, per-allele metadata) as the reference; matched **PGEN** (`plink2`) and **svar1** (`genoray` convert) derived from the *same* VCF so all three decode identically; reuse the existing `from_svar1`-vs-`from_vcf` parity test. Every candidate must match the oracle (exact, or documented tolerance for dosages).
- **Swept benchmark + baseline:** known starting point ~290 variants/s single-stream. Parameterize by samples × variants (and concurrency once the livelock is fixed); sweep the dominating dimension; report **cpu_s** (load-robust on this shared box, per the `bench_from_vcf_list` README convention). Record the baseline number before optimizing.
- **Profile-driven, one change at a time:** `from_vcf` is on record as ~78% htslib-reader-bound (inflate+parse), so its ceiling is likely the reader; `from_pgen` (pgenlib) and `from_svar1` (mmap) may bottleneck elsewhere — the profile decides. Re-run the oracle **and** the benchmark after each change; keep only wins that stay correct. Stop at the target or the Amdahl ceiling, and state which stopped each writer.

## Cross-cutting build & measurement discipline

- `cargo test --no-default-features` and `cargo check --no-default-features` (else the pyo3 test binary fails to link: `undefined symbol: _Py_Dealloc`; and the query-core gating has separate CI coverage).
- Set `CARGO_TARGET_DIR` off NFS (NFS `target/` bus-errors `cargo test` in debug and the lint hooks).
- Park fixtures + bench data in `$CLAUDE_JOB_DIR/tmp` — `/tmp` is reaped mid-session.
- **`maturin develop --release` before any Python-level perf/e2e run** — `pixi run test` does *not* rebuild the Rust `.so`, so Python benches can silently run stale code. Release `.so` ≈ 4 MB, debug ≈ 79 MB.
- Always check process exit status + real output before believing a profile (a panicked/aborted run can print a tiny, misleading number).

## Public API impact

No public-API change is expected (a concurrency bug fix + internal throughput work). If any name reachable from `import genoray` without underscores changes, `skills/genoray-api/SKILL.md` **must** be updated in the same PR (per repo policy). The spec notes the gate; it is likely a no-op here.

## Out of scope

- `from_vcf_list` optimization (already done).
- Peak-RAM work (resolved for the multi-file path; single-source RAM is bounded per #135).
- Any GPU work.
