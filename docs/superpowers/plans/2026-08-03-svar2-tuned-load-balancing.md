# SVAR2 Tuned Load Balancing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make SVAR2 VCF conversion plan contig concurrency under explicit core and memory constraints, order contigs longest-first, and size reader workers from a measured rate instead of a hardcoded knee.

**Architecture:** Four independent units behind clean seams — a pure-arithmetic planner (`budget.rs`), an index-derived contig cost estimator (`contig_cost.rs`), an optional two-chunk rate probe (`tune.rs`), and a Python-side memory-budget detector (`_utils.py`) — wired together in `lib.rs` between contig discovery and rayon dispatch. Output bytes must not move; the digest-invariance test is the gate.

**Tech Stack:** Rust (pyo3 0.29, rust-htslib 1.0, rayon, crossbeam-channel), Python 3.10+ (numpy, polars), pixi for envs, pytest + `cargo test` for tests.

**Spec:** `docs/superpowers/specs/2026-08-03-svar2-tuned-load-balancing-design.md`

## Global Constraints

- **Rust tests MUST run `--no-default-features --features conversion`.** Without it the pyo3 test binary fails to link with `undefined symbol: _Py_Dealloc`.
- **`CARGO_TARGET_DIR` MUST point off NFS** before any cargo command: `export CARGO_TARGET_DIR="$CLAUDE_JOB_DIR/tmp/cargo-target"`. NFS `target/` bus-errors the linker.
- **`pixi run test` does NOT rebuild the Rust extension.** Run `maturin develop --release` before any Python-level test of a Rust change, or you are testing stale code.
- **Never write scratch to `/tmp`** — it is reaped on this cluster. Use `$CLAUDE_JOB_DIR/tmp`.
- **Never detach a process** — no `nohup`, `setsid`, `disown`, or trailing `&`. Long/compute-heavy runs go to `sbatch`. Run all cargo/maturin commands in the foreground.
- **Conventional Commits** (`feat:`, `fix:`, `docs:`, `test:`, `perf:`). **Never edit `CHANGELOG.md`** — commitizen owns it.
- **Public API changes require updating `skills/genoray-api/SKILL.md` in the same PR.**
- **Run all commands from the worktree root**, not the main checkout.
- Fitted-law constants are copied verbatim from the spec: `RAM_BASE_MB = 932.0`, `RAM_PER_SAMPLE_MB = 0.01115`, `RAM_KAPPA = 1.371`, `MEM_BUDGET_FRACTION = 0.8`, `W_MAX = 16`.
- `chunk_bytes` formula, verbatim: `chunk_size * ((n_samples * ploidy) / 8 + n_format_fields * n_samples * 4)`.

## File Structure

| file | responsibility |
|---|---|
| `src/budget.rs` (modify) | Pure arithmetic. Adds `plan_sharded` — concurrency under core + memory constraints. Existing `plan_thread_budget` stays for the monolithic/PGEN paths. |
| `src/contig_cost.rs` (create) | Per-contig work estimates from index metadata, and the longest-first ordering. No variant data is read. |
| `src/tune.rs` (create) | The optional probe: measures `t_read` / `t_exec` on two chunks, and the pure `workers_from_rates`. |
| `src/lib.rs` (modify) | Wiring: estimate → probe → plan → order → dispatch. New `max_mem_bytes` / `tune` pyfunction args. |
| `python/genoray/_utils.py` (modify) | `detect_memory_budget()` — cgroup limit first, `/proc/meminfo` fallback. |
| `python/genoray/_svar2.py` (modify) | `from_vcf(..., max_mem=None, tune=False)`. |
| `tests/_oracle.py` (modify) | `store_digest(path)` helper for the invariance test. |
| `tests/test_svar2_schedule_invariance.py` (create) | The gate: output digest identical across `(cc, w, order)`. |
| `tests/test_memory_budget.py` (create) | `detect_memory_budget` tiers. |
| `skills/genoray-api/SKILL.md` (modify) | Documents `max_mem` / `tune`. |

**Parallelism.** Tasks 1–4 are mutually independent and should be dispatched in parallel via `superpowers:dispatching-parallel-agents` with `superpowers:subagent-driven-development`. Task 5 depends on all of 1–4. Tasks 6 and 7 depend on 5 and are parallel with each other. **Use Sonnet or weaker for implementers**; reserve Opus for review and for fixing a critical implementer failure.

---

### Task 1: Memory budget detection (Python)

**Files:**
- Modify: `python/genoray/_utils.py`
- Test: `tests/test_memory_budget.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: `detect_memory_budget(fraction: float = 0.8) -> int` returning bytes; module constant `MEM_BUDGET_FRACTION = 0.8`.

**Why cgroup first:** under Slurm, `/proc/meminfo` reports the whole node while the job is capped by its cgroup. Reading the node's memory hands the planner a budget it does not have — on exactly the allocations where the planner matters most.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_memory_budget.py
"""detect_memory_budget must prefer the cgroup limit over /proc/meminfo.

Under Slurm those differ: /proc/meminfo reports the node, the cgroup reports
the job. Planning against the node OOMs the job.
"""

import pytest

from genoray._utils import MEM_BUDGET_FRACTION, detect_memory_budget


def _write(path, text):
    path.write_text(text)
    return path


def test_prefers_cgroup_v2_over_meminfo(tmp_path, monkeypatch):
    v2 = _write(tmp_path / "memory.max", "8589934592\n")  # 8 GiB
    meminfo = _write(tmp_path / "meminfo", "MemTotal:       263192360 kB\n")
    monkeypatch.setattr("genoray._utils._CGROUP_V2", v2)
    monkeypatch.setattr("genoray._utils._CGROUP_V1", tmp_path / "absent")
    monkeypatch.setattr("genoray._utils._MEMINFO", meminfo)
    assert detect_memory_budget() == int(8589934592 * MEM_BUDGET_FRACTION)


def test_cgroup_v2_max_falls_through_to_v1(tmp_path, monkeypatch):
    # cgroup v2 writes the literal "max" when the group is uncapped.
    v2 = _write(tmp_path / "memory.max", "max\n")
    v1 = _write(tmp_path / "limit_in_bytes", "4294967296\n")  # 4 GiB
    meminfo = _write(tmp_path / "meminfo", "MemTotal:       263192360 kB\n")
    monkeypatch.setattr("genoray._utils._CGROUP_V2", v2)
    monkeypatch.setattr("genoray._utils._CGROUP_V1", v1)
    monkeypatch.setattr("genoray._utils._MEMINFO", meminfo)
    assert detect_memory_budget() == int(4294967296 * MEM_BUDGET_FRACTION)


def test_cgroup_v1_sentinel_falls_through_to_meminfo(tmp_path, monkeypatch):
    # cgroup v1 writes a huge sentinel (PAGE_COUNTER_MAX) when uncapped; it is
    # not a real limit and must not be believed.
    v1 = _write(tmp_path / "limit_in_bytes", "9223372036854771712\n")
    meminfo = _write(tmp_path / "meminfo", "MemTotal:       1048576 kB\n")  # 1 GiB
    monkeypatch.setattr("genoray._utils._CGROUP_V2", tmp_path / "absent")
    monkeypatch.setattr("genoray._utils._CGROUP_V1", v1)
    monkeypatch.setattr("genoray._utils._MEMINFO", meminfo)
    assert detect_memory_budget() == int(1024**3 * MEM_BUDGET_FRACTION)


def test_all_sources_absent_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("genoray._utils._CGROUP_V2", tmp_path / "absent")
    monkeypatch.setattr("genoray._utils._CGROUP_V1", tmp_path / "absent")
    monkeypatch.setattr("genoray._utils._MEMINFO", tmp_path / "absent")
    with pytest.raises(RuntimeError, match="could not detect a memory budget"):
        detect_memory_budget()


def test_fraction_is_applied_and_never_the_whole_limit(tmp_path, monkeypatch):
    v2 = _write(tmp_path / "memory.max", "1000000000\n")
    monkeypatch.setattr("genoray._utils._CGROUP_V2", v2)
    monkeypatch.setattr("genoray._utils._CGROUP_V1", tmp_path / "absent")
    monkeypatch.setattr("genoray._utils._MEMINFO", tmp_path / "absent")
    got = detect_memory_budget()
    assert got == 800_000_000
    assert got < 1_000_000_000
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run pytest tests/test_memory_budget.py -v`
Expected: FAIL with `ImportError: cannot import name 'detect_memory_budget'`

- [ ] **Step 3: Implement**

Append to `python/genoray/_utils.py`:

```python
from pathlib import Path

# Planning to 100% of a limit means the first prediction error is an OOM kill.
# The RAM law it feeds predicts peak RSS with R^2=0.9040 and a 3% hold-out
# error; this headroom covers that residual plus everything the law does not
# model -- the interpreter, glibc arena fragmentation, the merge tail.
MEM_BUDGET_FRACTION = 0.8

# Module-level so tests can point them at fixtures.
_CGROUP_V2 = Path("/sys/fs/cgroup/memory.max")
_CGROUP_V1 = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
_MEMINFO = Path("/proc/meminfo")

# cgroup v1 writes a near-INT64_MAX sentinel rather than a real limit when the
# group is uncapped. Anything at or above this is "no limit", not a budget.
_CGROUP_V1_UNLIMITED = 1 << 62


def _read_int(path: Path) -> int | None:
    """Parse a single integer from `path`, or None if absent/unparseable."""
    try:
        text = path.read_text().strip()
    except OSError:
        return None
    try:
        return int(text)
    except ValueError:
        return None  # cgroup v2 writes the literal "max" when uncapped.


def _meminfo_total(path: Path) -> int | None:
    try:
        for line in path.read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except (OSError, IndexError, ValueError):
        return None
    return None


def detect_memory_budget(fraction: float = MEM_BUDGET_FRACTION) -> int:
    """Bytes of memory the conversion planner may plan against.

    Prefers the cgroup limit over `/proc/meminfo`. Under Slurm the two differ:
    `/proc/meminfo` reports the node, the cgroup reports the job. Planning
    against the node hands the planner a budget it does not have, on exactly
    the allocations where the planner matters most.
    """
    limit = _read_int(_CGROUP_V2)
    if limit is None:
        v1 = _read_int(_CGROUP_V1)
        limit = v1 if v1 is not None and v1 < _CGROUP_V1_UNLIMITED else None
    if limit is None:
        limit = _meminfo_total(_MEMINFO)
    if limit is None:
        raise RuntimeError(
            "could not detect a memory budget (no cgroup limit and no "
            "/proc/meminfo); pass max_mem explicitly"
        )
    return int(limit * fraction)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run pytest tests/test_memory_budget.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add python/genoray/_utils.py tests/test_memory_budget.py
git commit -m "feat(svar2): detect the conversion memory budget from the cgroup

Under Slurm /proc/meminfo reports the node while the job is capped by its
cgroup, so reading the node hands a planner a budget it does not have -- on
exactly the allocations where planning matters. Prefer the cgroup limit,
treat cgroup v2's \"max\" and v1's PAGE_COUNTER_MAX sentinel as absent, and
apply a 0.8 fraction so the first prediction error is not an OOM kill."
```

---

### Task 2: Constrained concurrency planner (`budget.rs`)

**Files:**
- Modify: `src/budget.rs`
- Test: `src/budget.rs` inline `mod tests` (follow the existing style — assert whole structs)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `pub struct PlanInputs { pub usable_cores: usize, pub n_contigs: usize, pub n_samples: usize, pub chunk_bytes: u64, pub max_mem_bytes: Option<u64>, pub reader_workers: usize }`
  - `pub struct ShardedPlan { pub concurrent_chroms: usize, pub reader_workers: usize }`
  - `pub enum PlanError { InsufficientMemory { needed_mb: f64, budget_mb: f64 } }`
  - `pub fn plan_sharded(inp: PlanInputs) -> Result<ShardedPlan, PlanError>`
  - `pub const RAM_BASE_MB: f64`, `RAM_PER_SAMPLE_MB: f64`, `RAM_KAPPA: f64`

Do **not** delete `plan_thread_budget` — the monolithic VCF reader and the PGEN path still call it (`lib.rs:433`, `lib.rs:1144`) and are out of scope.

- [ ] **Step 1: Write the failing tests**

Add to `src/budget.rs`'s `mod tests`:

```rust
    // 48 cores -> 47 usable. w=2 -> demand 3/contig -> 15 concurrent, under
    // the 22 available. The OLD planner returned 7 here, because it charged
    // 6 cores per contig for 4 mostly-blocked pipeline threads plus an
    // HTSlib pool the sharded path never allocates.
    #[test]
    fn core_bound_concurrency() {
        let plan = plan_sharded(PlanInputs {
            usable_cores: 47,
            n_contigs: 22,
            n_samples: 4_000,
            chunk_bytes: 10_937_000,
            max_mem_bytes: None,
            reader_workers: 2,
        })
        .unwrap();
        assert_eq!(
            plan,
            ShardedPlan {
                concurrent_chroms: 15,
                reader_workers: 2
            }
        );
    }

    // Fewer contigs than cores allow: never spawn a pipeline with no contig.
    #[test]
    fn contig_count_bounds_concurrency() {
        let plan = plan_sharded(PlanInputs {
            usable_cores: 47,
            n_contigs: 4,
            n_samples: 4_000,
            chunk_bytes: 10_937_000,
            max_mem_bytes: None,
            reader_workers: 2,
        })
        .unwrap();
        assert_eq!(plan.concurrent_chroms, 4);
    }

    // The memory constraint must actually bind, or it is decoration.
    // S=500,000, ploidy 2, no FORMAT fields, chunk_size 25,000:
    //   chunk_bytes = 25_000 * (500_000*2/8) = 3.125e9 B = 3125 MB
    //   base        = 932 + 0.01115*500_000 = 6507 MB
    //   per-contig  = 1.371 * (2 + 1) * 3125 = 12852.2 MB
    //   budget      = 52428 MB  ->  (52428 - 6507)/12852.2 = 3.57 -> 3
    // The core bound alone would have allowed 15.
    #[test]
    fn memory_bound_beats_core_bound_at_biobank_scale() {
        let plan = plan_sharded(PlanInputs {
            usable_cores: 47,
            n_contigs: 22,
            n_samples: 500_000,
            chunk_bytes: 3_125_000_000,
            max_mem_bytes: Some(52_428 * 1_000_000),
            reader_workers: 2,
        })
        .unwrap();
        assert_eq!(plan.concurrent_chroms, 3);
    }

    // A budget below the cohort baseline cannot fit even one contig. Failing
    // loudly beats planning cc=0 (which dispatches nothing and "succeeds"
    // with an empty store) or cc=1 (which OOMs).
    #[test]
    fn budget_below_baseline_is_an_error() {
        let err = plan_sharded(PlanInputs {
            usable_cores: 47,
            n_contigs: 22,
            n_samples: 500_000,
            chunk_bytes: 3_125_000_000,
            max_mem_bytes: Some(5_000 * 1_000_000),
            reader_workers: 2,
        })
        .unwrap_err();
        match err {
            PlanError::InsufficientMemory {
                needed_mb,
                budget_mb,
            } => {
                assert!(needed_mb > budget_mb);
                assert!((budget_mb - 5_000.0).abs() < 1.0);
            }
        }
    }

    // Degenerate hardware must still produce a runnable plan.
    #[test]
    fn single_core_single_contig_still_runs() {
        let plan = plan_sharded(PlanInputs {
            usable_cores: 1,
            n_contigs: 1,
            n_samples: 250,
            chunk_bytes: 64_000,
            max_mem_bytes: None,
            reader_workers: 4,
        })
        .unwrap();
        assert_eq!(
            plan,
            ShardedPlan {
                concurrent_chroms: 1,
                reader_workers: 4
            }
        );
    }

    // Zero contigs is a caller bug, not a plan: clamp rather than divide by it.
    #[test]
    fn zero_contigs_clamps_to_one() {
        let plan = plan_sharded(PlanInputs {
            usable_cores: 47,
            n_contigs: 0,
            n_samples: 250,
            chunk_bytes: 64_000,
            max_mem_bytes: None,
            reader_workers: 2,
        })
        .unwrap();
        assert_eq!(plan.concurrent_chroms, 1);
    }
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
export CARGO_TARGET_DIR="$CLAUDE_JOB_DIR/tmp/cargo-target"
pixi run bash -lc 'cargo test --no-default-features --features conversion budget:: -- --nocapture'
```
Expected: FAIL to compile — `cannot find function plan_sharded`

- [ ] **Step 3: Implement**

Add to `src/budget.rs`:

```rust
// Peak-RSS coefficients from the scale-bench RAM law, fitted 2026-08-03:
//   peak_rss_mb ~ 932 + 0.01115*samples + 1.371*(w+pending)*chunk_bytes
//   R^2 = 0.9040, n = 44
// See docs/superpowers/specs/2026-08-03-svar2-tuned-load-balancing-design.md.
// These are load-bearing in production, not just in the bench: a bad refit
// becomes an OOM. Change them only alongside a refit that says so.
pub const RAM_BASE_MB: f64 = 932.0;
pub const RAM_PER_SAMPLE_MB: f64 = 0.01115;
pub const RAM_KAPPA: f64 = 1.371;

/// Inputs to the sharded-VCF concurrency plan. Every field is data the caller
/// already has before opening a single record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlanInputs {
    pub usable_cores: usize,
    pub n_contigs: usize,
    pub n_samples: usize,
    /// Bytes of one FULL dense chunk:
    /// `chunk_size * (n_samples*ploidy/8 + n_format_fields*n_samples*4)`.
    pub chunk_bytes: u64,
    /// `None` means the caller declined a budget; only the core bound applies.
    pub max_mem_bytes: Option<u64>,
    pub reader_workers: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShardedPlan {
    pub concurrent_chroms: usize,
    pub reader_workers: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PlanError {
    /// The budget cannot fit the cohort baseline plus one contig's chunks.
    InsufficientMemory { needed_mb: f64, budget_mb: f64 },
}

impl std::fmt::Display for PlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlanError::InsufficientMemory {
                needed_mb,
                budget_mb,
            } => write!(
                f,
                "max_mem is {budget_mb:.0f} MB but converting this cohort needs \
                 at least {needed_mb:.0f} MB for one concurrent contig; raise \
                 max_mem or lower chunk_size"
            ),
        }
    }
}

/// Plan contig concurrency for the sharded VCF reader.
///
/// Per-contig CPU demand is `1 + reader_workers`: one executor
/// (`run_compute_engine`, a serial recv loop, pegged at ~100% of one core) plus
/// the shard readers. It is NOT `PIPELINE_THREADS_PER_CHROM + htslib_threads`
/// -- the dispatcher and both writers are nearly always blocked (a measured
/// 22-contig run put 16 threads on 2.02 cores), and the HTSlib decode pool is
/// `SHARDED_VCF_HTSLIB_THREADS_PER_READER` = 0 on this path because shard
/// readers decompress inline.
///
/// Memory bounds concurrency independently: each concurrent contig holds
/// `reader_workers + pending` chunks in flight, where `pending` is the reorder
/// buffer's structural floor `reader_workers - 1` (the units ahead of the head
/// keep everything they produce buffered even with perfectly balanced readers).
pub fn plan_sharded(inp: PlanInputs) -> Result<ShardedPlan, PlanError> {
    let w = inp.reader_workers.max(1);
    let n_contigs = inp.n_contigs.max(1);
    let usable = inp.usable_cores.max(1);

    let core_bound = (usable / (1 + w)).max(1);

    let cc = match inp.max_mem_bytes {
        None => std::cmp::min(core_bound, n_contigs),
        Some(budget) => {
            let budget_mb = budget as f64 / 1e6;
            let baseline_mb = RAM_BASE_MB + RAM_PER_SAMPLE_MB * inp.n_samples as f64;
            let pending = w.saturating_sub(1);
            let per_contig_mb =
                RAM_KAPPA * (w + pending) as f64 * (inp.chunk_bytes as f64 / 1e6);
            let headroom_mb = budget_mb - baseline_mb;
            if headroom_mb < per_contig_mb {
                return Err(PlanError::InsufficientMemory {
                    needed_mb: baseline_mb + per_contig_mb,
                    budget_mb,
                });
            }
            let mem_bound = (headroom_mb / per_contig_mb).floor() as usize;
            std::cmp::min(std::cmp::min(core_bound, n_contigs), mem_bound.max(1))
        }
    };

    Ok(ShardedPlan {
        concurrent_chroms: cc,
        reader_workers: w,
    })
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
export CARGO_TARGET_DIR="$CLAUDE_JOB_DIR/tmp/cargo-target"
pixi run bash -lc 'cargo test --no-default-features --features conversion budget:: -- --nocapture'
```
Expected: all `budget::tests` pass, including the pre-existing `plan_thread_budget` tests (unchanged).

- [ ] **Step 5: Document which path the old constants bind**

Edit the doc comments on `PIPELINE_THREADS_PER_CHROM`, `MIN_HTSLIB_THREADS`, and `MAX_HTSLIB_THREADS` to state they govern the **monolithic** reader path only. The current comments read as if they bound both paths, which is how the sharded path came to be charged for a decode pool it never allocates.

- [ ] **Step 6: Commit**

```bash
git add src/budget.rs
git commit -m "feat(svar2): plan contig concurrency under core and memory constraints

plan_thread_budget charges every contig MIN_THREADS_PER_CHROM = 6 cores: four
pipeline threads plus two HTSlib decode threads. On the sharded path the
HTSlib pool does not exist (readers decompress inline) and only the executor
is CPU-bound -- a measured 22-contig run put 16 threads on 2.02 cores. So it
reserved ~6 cores for something consuming 0.4-1.0, and returned cc=7 on a
48-core box with 22 contigs.

plan_sharded charges 1 + reader_workers and bounds concurrency by the fitted
RAM law as well, which is what the scale bench's H3 verdict asked for: the
invariant is in-flight bytes, not worker count. A budget too small for one
contig is an error rather than a cc=0 plan that silently writes nothing."
```

---

### Task 3: Per-contig cost estimates (`contig_cost.rs`)

**Files:**
- Create: `src/contig_cost.rs`
- Modify: `src/lib.rs` (add `mod contig_cost;` next to the other `#[cfg(feature = "conversion")] mod` declarations)
- Test: `src/contig_cost.rs` inline `mod tests`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `pub fn estimate_contig_costs(vcf_path: &str, chroms: &[String]) -> std::collections::HashMap<String, u64>`
  - `pub fn order_longest_first(chroms: &[String], costs: &std::collections::HashMap<String, u64>) -> Vec<String>`

Only *ratios* matter — these values order contigs and do nothing else. Never read variant data.

**On the fallback chain:** the spec lists three tiers; this task implements two. Tier 1 is `hts_idx_get_stat` (exact record counts). Tier 2 is the header contig length. The spec's middle tier — the linear index's compressed byte extent — is dropped: reaching it means walking CSI internals through far more `unsafe` than the two surviving tiers, for an estimator whose output is a sort key. Step 6 amends the spec to match.

**Tier 1 may not work.** `hts_idx_get_stat` is documented for BAM; whether CSI/TBI indexes over VCF populate the mapped-record count is not guaranteed. Step 1's test is written to *decide this*: if tier 1 returns zeros, the test fails and the implementer removes tier 1 and ships tier 2 alone. Do not "fix" the test by weakening the assertion.

- [ ] **Step 1: Write the failing tests**

```rust
// src/contig_cost.rs -- append after the implementation.
#[cfg(test)]
mod tests {
    use super::*;
    use rust_htslib::bcf::{Format, Header, Writer};
    use std::collections::HashMap;

    /// Build a 3-contig VCF with deliberately unequal record counts
    /// (chrA=5, chrB=40, chrC=15) and index it. Mirrors the fixture pattern
    /// already used in `src/vcf_reader.rs`'s tests.
    fn three_contig_vcf(dir: &std::path::Path) -> String {
        let path = dir.join("cost.vcf.gz");
        let mut header = Header::new();
        header.push_record(b"##contig=<ID=chrA,length=100000>");
        header.push_record(b"##contig=<ID=chrB,length=100000>");
        header.push_record(b"##contig=<ID=chrC,length=100000>");
        header.push_record(
            b"##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">",
        );
        header.push_sample(b"s1");
        {
            let mut w = Writer::from_path(&path, &header, false, Format::Vcf).unwrap();
            for (chrom, n) in [("chrA", 5u32), ("chrB", 40), ("chrC", 15)] {
                let rid = w.header().name2rid(chrom.as_bytes()).unwrap();
                for i in 0..n {
                    let mut rec = w.empty_record();
                    rec.set_rid(Some(rid));
                    rec.set_pos((i as i64 + 1) * 100);
                    rec.set_alleles(&[b"A", b"C"]).unwrap();
                    w.write(&rec).unwrap();
                }
            }
        }
        let p = path.to_str().unwrap().to_string();
        rust_htslib::bcf::index::build(
            &p,
            None,
            1,
            rust_htslib::bcf::index::Type::Csi(14),
        )
        .unwrap();
        p
    }

    #[test]
    fn estimates_rank_contigs_by_true_record_count() {
        let dir = tempfile::tempdir().unwrap();
        let path = three_contig_vcf(dir.path());
        let chroms: Vec<String> = ["chrA", "chrB", "chrC"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let costs = estimate_contig_costs(&path, &chroms);
        // Only the ORDER is contractual -- the absolute unit differs per tier.
        // If this fails with all-equal costs, tier 1 returned nothing useful
        // for CSI-over-VCF; delete tier 1 and ship the header-length tier.
        assert!(costs["chrB"] > costs["chrC"], "costs = {costs:?}");
        assert!(costs["chrC"] > costs["chrA"], "costs = {costs:?}");
    }

    #[test]
    fn orders_longest_first() {
        let costs: HashMap<String, u64> = [
            ("chrA".to_string(), 5u64),
            ("chrB".to_string(), 40),
            ("chrC".to_string(), 15),
        ]
        .into_iter()
        .collect();
        let chroms: Vec<String> = ["chrA", "chrB", "chrC"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(order_longest_first(&chroms, &costs), vec!["chrB", "chrC", "chrA"]);
    }

    /// An unestimated contig sorts FIRST. Guessing high costs a slightly worse
    /// order; guessing low risks starting the longest job last, which is the
    /// exact failure the ordering exists to prevent.
    #[test]
    fn unknown_contig_sorts_first() {
        let costs: HashMap<String, u64> =
            [("chrA".to_string(), 5u64), ("chrB".to_string(), 40)]
                .into_iter()
                .collect();
        let chroms: Vec<String> = ["chrA", "chrB", "chrZ"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(order_longest_first(&chroms, &costs), vec!["chrZ", "chrB", "chrA"]);
    }

    /// Ordering must be total and deterministic -- rayon dispatch order feeds
    /// the output layout, and a tie broken by HashMap iteration order would
    /// make the run non-reproducible.
    #[test]
    fn ties_break_deterministically_by_name() {
        let costs: HashMap<String, u64> = [
            ("chrA".to_string(), 7u64),
            ("chrB".to_string(), 7),
            ("chrC".to_string(), 7),
        ]
        .into_iter()
        .collect();
        let chroms: Vec<String> = ["chrC", "chrA", "chrB"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        for _ in 0..20 {
            assert_eq!(
                order_longest_first(&chroms, &costs),
                vec!["chrA", "chrB", "chrC"]
            );
        }
    }

    #[test]
    fn missing_index_falls_back_without_panicking() {
        let dir = tempfile::tempdir().unwrap();
        let chroms = vec!["chrA".to_string()];
        let costs = estimate_contig_costs(
            dir.path().join("absent.vcf.gz").to_str().unwrap(),
            &chroms,
        );
        assert!(costs.is_empty() || costs.contains_key("chrA"));
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
export CARGO_TARGET_DIR="$CLAUDE_JOB_DIR/tmp/cargo-target"
pixi run bash -lc 'cargo test --no-default-features --features conversion contig_cost:: -- --nocapture'
```
Expected: FAIL to compile — `file not found for module contig_cost`

- [ ] **Step 3: Implement**

```rust
//! Per-contig work estimates used to order contigs longest-first before
//! dispatch. Reads index/header metadata only -- never variant data.
//!
//! Only RATIOS matter: these values are a sort key and nothing else. That is
//! why a coarse tier is an acceptable fallback and why the absolute unit is
//! allowed to differ between tiers.

use rust_htslib::bcf::Read;
use std::collections::HashMap;
use std::ffi::CString;

/// Exact per-contig record counts from the `.csi`/`.tbi`, or `None` if the
/// index is absent or carries no per-reference statistics.
///
/// `hts_idx_get_stat` is documented for BAM; whether CSI/TBI over VCF
/// populates the mapped count is not guaranteed, which is why every failure
/// mode here returns `None` and lets the caller fall back rather than
/// reporting a confident zero.
fn counts_from_index(vcf_path: &str, chroms: &[String]) -> Option<HashMap<String, u64>> {
    let reader = rust_htslib::bcf::Reader::from_path(vcf_path).ok()?;
    let header = reader.header();

    let c_path = CString::new(vcf_path).ok()?;
    // SAFETY: `c_path` is a valid NUL-terminated string that outlives the
    // call; `hts_idx_load` returns null on any failure, which is checked.
    let idx = unsafe {
        rust_htslib::htslib::hts_idx_load(
            c_path.as_ptr(),
            rust_htslib::htslib::HTS_FMT_CSI as i32,
        )
    };
    if idx.is_null() {
        return None;
    }

    let mut out = HashMap::new();
    let mut any_nonzero = false;
    for chrom in chroms {
        let Some(rid) = header.name2rid(chrom.as_bytes()).ok() else {
            continue;
        };
        let mut mapped: u64 = 0;
        let mut unmapped: u64 = 0;
        // SAFETY: `idx` is non-null and owned here; `rid` came from this
        // file's own header; both out-params are valid for the call.
        let ret = unsafe {
            rust_htslib::htslib::hts_idx_get_stat(idx, rid as i32, &mut mapped, &mut unmapped)
        };
        if ret == 0 && mapped > 0 {
            any_nonzero = true;
            out.insert(chrom.clone(), mapped);
        }
    }
    // SAFETY: `idx` was produced by `hts_idx_load` and is destroyed exactly
    // once, here, on every path that got past the null check.
    unsafe { rust_htslib::htslib::hts_idx_destroy(idx) };

    any_nonzero.then_some(out)
}

/// Header contig lengths. Always available, and a reasonable proxy for a
/// whole-genome VCF; a poor one for exome or targeted data, which is why it
/// is the fallback rather than the primary source.
fn lengths_from_header(vcf_path: &str, chroms: &[String]) -> HashMap<String, u64> {
    use rust_htslib::bcf::header::HeaderRecord;

    let Ok(reader) = rust_htslib::bcf::Reader::from_path(vcf_path) else {
        return HashMap::new();
    };
    // rust-htslib 1.0's HeaderView has no rid2length; contig lengths are only
    // reachable through the structured header records.
    let wanted: std::collections::HashSet<&str> =
        chroms.iter().map(|s| s.as_str()).collect();
    let mut out = HashMap::new();
    for rec in reader.header().header_records() {
        let HeaderRecord::Contig { values, .. } = rec else {
            continue;
        };
        let (Some(id), Some(len)) = (values.get("ID"), values.get("length")) else {
            continue;
        };
        if wanted.contains(id.as_str()) {
            if let Ok(n) = len.parse::<u64>() {
                out.insert(id.clone(), n);
            }
        }
    }
    out
}

/// Per-contig work estimates, best source available.
pub fn estimate_contig_costs(vcf_path: &str, chroms: &[String]) -> HashMap<String, u64> {
    match counts_from_index(vcf_path, chroms) {
        Some(counts) => {
            tracing::debug!(source = "index", n = counts.len(), "contig cost estimates");
            counts
        }
        None => {
            let lens = lengths_from_header(vcf_path, chroms);
            tracing::debug!(
                source = "header_length",
                n = lens.len(),
                "contig cost estimates (index carried no per-contig counts)"
            );
            lens
        }
    }
}

/// Order contigs most-expensive-first (LPT). Rayon's work stealing does the
/// dynamic balancing; descending order is the whole scheduling contribution.
///
/// A contig with no estimate sorts FIRST: guessing high costs a slightly worse
/// order, guessing low risks starting the longest job last. Ties break by name
/// so dispatch order is deterministic across runs.
pub fn order_longest_first(chroms: &[String], costs: &HashMap<String, u64>) -> Vec<String> {
    let mut out = chroms.to_vec();
    out.sort_by(|a, b| {
        let ca = costs.get(a).copied().unwrap_or(u64::MAX);
        let cb = costs.get(b).copied().unwrap_or(u64::MAX);
        cb.cmp(&ca).then_with(|| a.cmp(b))
    });
    out
}
```

Add to `src/lib.rs` beside the other conversion-gated modules:

```rust
#[cfg(feature = "conversion")]
mod contig_cost;
```

`tempfile` is already a `[dev-dependencies]` entry, so the tests above need no
manifest change. **Do not edit `Cargo.toml` in this task** — Task 4 moves
`tempfile` to `[dependencies]`, and two parallel tasks editing the same
manifest section conflict.

- [ ] **Step 4: Run tests to verify they pass**

```bash
export CARGO_TARGET_DIR="$CLAUDE_JOB_DIR/tmp/cargo-target"
pixi run bash -lc 'cargo test --no-default-features --features conversion contig_cost:: -- --nocapture'
```
Expected: 5 passed.

**If `estimates_rank_contigs_by_true_record_count` fails with equal costs:** tier 1 does not work for CSI-over-VCF. Delete `counts_from_index` entirely, have `estimate_contig_costs` call `lengths_from_header` directly, change that test to build its fixture with unequal *contig lengths* (chrA=10000, chrB=800000, chrC=300000) instead of unequal record counts, and record the finding in the Step 6 spec amendment. Do not weaken the assertion to make tier 1 look like it works.

- [ ] **Step 5: Verify the no-conversion build still compiles**

```bash
export CARGO_TARGET_DIR="$CLAUDE_JOB_DIR/tmp/cargo-target"
pixi run bash -lc 'cargo check --no-default-features'
```
Expected: clean. The query-core build has no other CI coverage; `contig_cost` must stay behind the `conversion` feature gate.

- [ ] **Step 6: Amend the spec's fallback chain**

Edit the "`contig_cost.rs` — per-contig work estimates" section of
`docs/superpowers/specs/2026-08-03-svar2-tuned-load-balancing-design.md` to list the tiers actually implemented, with one sentence on why the byte-extent tier was dropped (walking CSI internals through more `unsafe` than the rest of the module, for a sort key) and what the tier-1 test found.

- [ ] **Step 7: Commit**

```bash
git add src/contig_cost.rs src/lib.rs docs/superpowers/specs/2026-08-03-svar2-tuned-load-balancing-design.md
git commit -m "feat(svar2): estimate per-contig conversion cost from index metadata

Contigs are wildly unequal and were dispatched in whatever order the caller
supplied, so the longest could start last. These estimates order them
longest-first; rayon's work stealing does the rest.

Only ratios matter -- the values are a sort key and nothing else -- so a
coarse fallback tier is acceptable and the absolute unit is allowed to differ
between tiers. An unestimated contig sorts first: guessing high costs a
slightly worse order, guessing low risks the failure the ordering exists to
prevent. Ties break by name so dispatch order stays deterministic."
```

---

### Task 4: The reader-rate probe (`tune.rs`)

**Files:**
- Create: `src/tune.rs`
- Modify: `src/lib.rs` (add `#[cfg(feature = "conversion")] mod tune;`)
- Test: `src/tune.rs` inline `mod tests`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `pub struct Rates { pub t_read_s: f64, pub t_exec_s: f64 }`
  - `pub fn workers_from_rates(rates: &Rates) -> usize`
  - `pub fn probe_rates(vcf_path: &str, chrom: &str, samples: &[&str], chunk_size: usize, ploidy: usize, fields: &[crate::field::FieldSpec]) -> Result<Rates, crate::error::ConversionError>`
  - `pub const W_MAX: usize = 16;`
  - `pub const PROBE_CHUNKS: usize = 2;`

Implement and land the pure half first — it is the half the planner depends on, and it is testable without touching a file.

- [ ] **Step 1: Write the failing tests for the pure half**

```rust
// src/tune.rs
#[cfg(test)]
mod tests {
    use super::*;

    /// To keep the executor fed, w readers must supply at least as fast as one
    /// executor drains: w/t_read >= 1/t_exec, so w = ceil(t_read/t_exec).
    #[test]
    fn workers_are_the_read_to_exec_ratio_rounded_up() {
        assert_eq!(
            workers_from_rates(&Rates {
                t_read_s: 0.30,
                t_exec_s: 0.10
            }),
            3
        );
        // 0.35/0.10 = 3.5 -> 4: rounding DOWN starves the executor, which is
        // the bottleneck this whole probe exists to keep busy.
        assert_eq!(
            workers_from_rates(&Rates {
                t_read_s: 0.35,
                t_exec_s: 0.10
            }),
            4
        );
    }

    /// A reader faster than the executor still needs one worker, not zero.
    #[test]
    fn floor_is_one_worker() {
        assert_eq!(
            workers_from_rates(&Rates {
                t_read_s: 0.01,
                t_exec_s: 0.50
            }),
            1
        );
    }

    /// W_MAX bounds the damage from a probe that hit a pathological prefix --
    /// an all-reference stretch reads far faster than it converts. The harness
    /// never observed a knee above 7.
    #[test]
    fn clamped_at_w_max() {
        assert_eq!(
            workers_from_rates(&Rates {
                t_read_s: 100.0,
                t_exec_s: 0.001
            }),
            W_MAX
        );
    }

    /// A zero or negative t_exec is a broken measurement (clock granularity on
    /// a tiny chunk), not an infinitely fast executor. Fall back rather than
    /// dividing by it.
    #[test]
    fn degenerate_exec_time_falls_back_to_one() {
        assert_eq!(
            workers_from_rates(&Rates {
                t_read_s: 1.0,
                t_exec_s: 0.0
            }),
            1
        );
        assert_eq!(
            workers_from_rates(&Rates {
                t_read_s: 1.0,
                t_exec_s: -1.0
            }),
            1
        );
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
export CARGO_TARGET_DIR="$CLAUDE_JOB_DIR/tmp/cargo-target"
pixi run bash -lc 'cargo test --no-default-features --features conversion tune:: -- --nocapture'
```
Expected: FAIL to compile — `file not found for module tune`

- [ ] **Step 3: Implement the pure half**

```rust
//! Optional runtime calibration of the per-contig reader count.
//!
//! The scale bench fitted a knee at w ~ 3-7 that moves with cohort size. That
//! fit was taken on synthetic corpora on one machine, and node speed on this
//! cluster varies by 2.08x; `t_read` and `t_exec` also move with compression
//! ratio, field count, and ploidy, none of which a fitted knee sees. This
//! measures the ratio on the actual input, on the actual machine.

/// Per-chunk timings from the probe.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rates {
    /// Seconds for ONE shard worker to produce one dense chunk.
    pub t_read_s: f64,
    /// Seconds for `dense2sparse_vk` to consume one dense chunk.
    pub t_exec_s: f64,
}

/// Safety clamp, not a tuning parameter: it bounds the damage from a probe
/// that measured a pathological prefix. The harness never observed a knee
/// above 7, so this sits well clear of any real workload -- reaching it means
/// the probe, not the workload, is what to look at.
pub const W_MAX: usize = 16;

/// Chunks timed per probe. Two is enough to get past first-chunk warmup while
/// staying negligible against a real conversion.
pub const PROBE_CHUNKS: usize = 2;

/// Readers needed to keep one executor fed: `w/t_read >= 1/t_exec`.
///
/// Rounds UP — rounding down starves the executor, which is the serial stage
/// the whole probe exists to keep busy.
pub fn workers_from_rates(rates: &Rates) -> usize {
    if !(rates.t_exec_s > 0.0) || !rates.t_read_s.is_finite() || rates.t_read_s <= 0.0 {
        // A non-positive or non-finite measurement is a broken clock reading
        // on a small chunk, not an infinitely fast stage. Do not divide by it.
        return 1;
    }
    let w = (rates.t_read_s / rates.t_exec_s).ceil();
    if !w.is_finite() {
        return 1;
    }
    (w as usize).clamp(1, W_MAX)
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
export CARGO_TARGET_DIR="$CLAUDE_JOB_DIR/tmp/cargo-target"
pixi run bash -lc 'cargo test --no-default-features --features conversion tune:: -- --nocapture'
```
Expected: 4 passed.

- [ ] **Step 5: Commit the pure half**

```bash
git add src/tune.rs src/lib.rs
git commit -m "feat(svar2): derive reader count from a read/exec rate ratio

w readers keep one executor fed when w/t_read >= 1/t_exec, so w is
ceil(t_read/t_exec) -- rounding up, since rounding down starves the serial
stage this exists to keep busy. Clamped at W_MAX=16, well clear of the
knee of 3-7 the scale bench observed, so reaching the clamp flags a bad
probe rather than silently over-provisioning."
```

- [ ] **Step 6: Write the failing test for the probe**

```rust
    /// The probe returns positive timings and a plausible worker count on a
    /// real file. Bounds, not a pinned number: this measures a machine, and
    /// node speed here varies 2.08x.
    #[test]
    fn probe_returns_usable_rates_on_a_fixture() {
        let dir = tempfile::tempdir().unwrap();
        let path = probe_fixture_vcf(dir.path(), 200, 64); // 200 variants, 64 samples
        let samples: Vec<String> = (0..64).map(|i| format!("s{i}")).collect();
        let refs: Vec<&str> = samples.iter().map(|s| s.as_str()).collect();
        let rates = probe_rates(&path, "chr1", &refs, 64, 2, &[]).unwrap();
        assert!(rates.t_read_s > 0.0, "rates = {rates:?}");
        assert!(rates.t_exec_s > 0.0, "rates = {rates:?}");
        let w = workers_from_rates(&rates);
        assert!((1..=W_MAX).contains(&w), "w = {w}");
    }

    /// A contig absent from the file is a caller error, and must surface as
    /// one rather than as a silent w=1 that quietly under-provisions.
    #[test]
    fn probe_errors_on_an_unknown_contig() {
        let dir = tempfile::tempdir().unwrap();
        let path = probe_fixture_vcf(dir.path(), 50, 8);
        let samples: Vec<String> = (0..8).map(|i| format!("s{i}")).collect();
        let refs: Vec<&str> = samples.iter().map(|s| s.as_str()).collect();
        assert!(probe_rates(&path, "chrNope", &refs, 32, 2, &[]).is_err());
    }
```

Write `probe_fixture_vcf(dir, n_variants, n_samples) -> String` in the same
`mod tests`, following the fixture pattern in `src/vcf_reader.rs`'s tests
(`bcf::Writer` + `bcf::index::build` with `Type::Csi(14)`). Every record needs
a GT for all samples, since the probe densifies them.

- [ ] **Step 7: Run to verify it fails**

```bash
export CARGO_TARGET_DIR="$CLAUDE_JOB_DIR/tmp/cargo-target"
pixi run bash -lc 'cargo test --no-default-features --features conversion tune:: -- --nocapture'
```
Expected: FAIL — `cannot find function probe_rates`

- [ ] **Step 8: Implement `probe_rates`**

Time `PROBE_CHUNKS` chunks through both stages using **one** `ChunkAssembler`
over an unsharded `VcfRecordSource`. No writers, no rayon pool, no output
directory. Passing `pool = None` to `read_next_chunk` keeps packing sequential,
which is what makes `t_read_s` one worker's rate rather than a pool's.

```rust
/// Measure one reader's chunk-production rate against one executor's
/// chunk-consumption rate, on a bounded prefix of `chrom`.
///
/// Deliberately unsharded and unpooled: `t_read_s` must be ONE worker's rate,
/// or the ratio it feeds means nothing.
pub fn probe_rates(
    vcf_path: &str,
    chrom: &str,
    samples: &[&str],
    chunk_size: usize,
    ploidy: usize,
    fields: &[crate::field::FieldSpec],
) -> Result<Rates, crate::error::ConversionError> {
    use std::time::Instant;

    let src = crate::vcf_reader::VcfRecordSource::new(
        vcf_path,
        chrom,
        samples,
        0, // htslib_threads: inline decode, matching a shard worker
        ploidy,
        fields,
        Vec::new(), // whole contig
        crate::svar2_view::OverlapMode::Pos,
    )?;
    let mut asm = crate::chunk_assembler::ChunkAssembler::new(
        Box::new(src),
        samples.len(),
        ploidy,
        None,  // no FASTA: left-alignment cost is not what is being compared
        chrom,
        false, // skip_out_of_scope
        crate::normalize::CheckRef::Error,
        fields,
    )?;

    // The bank is a required sink for dense2sparse_vk, not an output. It lands
    // in a tempdir that is dropped on return, so the probe writes no
    // user-visible bytes.
    let tmp = tempfile::tempdir().map_err(|e| {
        crate::error::ConversionError::Input(format!("probe tempdir: {e}"))
    })?;
    let mut bank = crate::nrvk::LongAlleleTableWriter::new(
        &tmp.path().join("probe_bank"),
        8 * 1024 * 1024,
    )?;

    let mut read_s = 0.0f64;
    let mut exec_s = 0.0f64;
    let mut chunks = 0usize;

    while chunks < PROBE_CHUNKS {
        let t0 = Instant::now();
        let Some(chunk) = asm.read_next_chunk(chunk_size, chunks, None)? else {
            break; // Short contig: fewer than PROBE_CHUNKS chunks exist.
        };
        read_s += t0.elapsed().as_secs_f64();

        let t1 = Instant::now();
        let _ = crate::rvk::dense2sparse_vk(&chunk, &mut bank, false, fields);
        exec_s += t1.elapsed().as_secs_f64();

        chunks += 1;
    }

    if chunks == 0 {
        return Err(crate::error::ConversionError::Input(format!(
            "probe found no records on contig {chrom}"
        )));
    }
    Ok(Rates {
        t_read_s: read_s / chunks as f64,
        t_exec_s: exec_s / chunks as f64,
    })
}
```

Two signatures above are load-bearing and worth confirming against the source
before writing: `ChunkAssembler::new` (`src/chunk_assembler.rs:397`) and
`LongAlleleTableWriter::new` (`src/nrvk.rs`). Adjust the call, never the two
contractual properties — exactly one reader with `pool = None`, and separate
timers bracketing `read_next_chunk` and `dense2sparse_vk`.

`tempfile` is currently a `[dev-dependencies]` entry (`Cargo.toml:47`) and this
is non-test code, so move it to `[dependencies]` as
`tempfile = { version = "3.10", optional = true }` and add `"dep:tempfile"` to
the `conversion` feature list. **This task owns that manifest edit** — no other
task in this plan touches `Cargo.toml`.

Then re-run `cargo check --no-default-features` to confirm the query-core build
still links: `tune` is conversion-gated and must not leak into it.

- [ ] **Step 9: Run tests to verify they pass**

```bash
export CARGO_TARGET_DIR="$CLAUDE_JOB_DIR/tmp/cargo-target"
pixi run bash -lc 'cargo test --no-default-features --features conversion tune:: -- --nocapture'
```
Expected: 6 passed.

- [ ] **Step 10: Commit**

```bash
git add src/tune.rs
git commit -m "feat(svar2): probe read and exec rates on two chunks

Times one shard worker's chunk production against dense2sparse_vk's chunk
consumption on a bounded prefix, so the reader count comes from a measurement
on this input and this machine rather than a knee fitted elsewhere. One
reader and no shards, or t_read is not one worker's rate and the ratio it
feeds means nothing. Writes no user-visible bytes."
```

---

### Task 5: Wire the planner into conversion

**Depends on Tasks 1, 2, 3, 4.**

**Files:**
- Modify: `src/lib.rs:149` (pyfunction signature) and `src/lib.rs:204-291` (the `py.detach` block)
- Modify: `python/genoray/_svar2.py:634-840` (`from_vcf`)
- Test: covered by Task 6; this task's gate is that the existing suite stays green

**Interfaces:**
- Consumes: `budget::plan_sharded`, `budget::PlanInputs`, `budget::PlanError`, `contig_cost::{estimate_contig_costs, order_longest_first}`, `tune::{probe_rates, workers_from_rates}`, `genoray._utils.detect_memory_budget`, `genoray._utils.parse_memory`.
- Produces: `from_vcf(..., max_mem: int | str | None = None, tune: bool = False)`.

- [ ] **Step 1: Add the Rust arguments**

In `src/lib.rs:149`, add `max_mem_bytes=None, tune=false` to the `#[pyo3(signature = ...)]` immediately before `log_level`, and the matching parameters `max_mem_bytes: Option<u64>, tune: bool` to `run_conversion_pipeline`.

- [ ] **Step 2: Replace the planning block**

Inside the `py.detach` closure, replace the `plan_thread_budget` call at `src/lib.rs:222-226` with:

```rust
            // Per-variant dense-chunk cost, matching what the RAM law was
            // fitted against: packed presence grid plus staged FORMAT values.
            let n_format = fields
                .iter()
                .filter(|f| f.category == crate::field::FieldCategory::Format)
                .count();
            let per_variant_bytes =
                (samples.len() * ploidy / 8 + n_format * samples.len() * 4) as u64;
            let chunk_bytes = per_variant_bytes * chunk_size as u64;

            let reader_workers = if tune {
                // Probe the LARGEST contig: it dominates the makespan, so its
                // rates are the ones worth matching.
                let costs = crate::contig_cost::estimate_contig_costs(&vcf_path, &chroms);
                let target = crate::contig_cost::order_longest_first(&chroms, &costs);
                match target.first() {
                    Some(c) => match crate::tune::probe_rates(
                        &vcf_path,
                        c,
                        &sample_refs,
                        chunk_size,
                        ploidy,
                        &fields,
                    ) {
                        Ok(rates) => {
                            let w = crate::tune::workers_from_rates(&rates);
                            tracing::info!(
                                t_read_s = rates.t_read_s,
                                t_exec_s = rates.t_exec_s,
                                reader_workers = w,
                                "tuned reader workers from probe"
                            );
                            w
                        }
                        Err(e) => {
                            // A failed probe must not fail the conversion; it
                            // is an optimization, and the planner's default is
                            // a measured knee, not a guess.
                            tracing::warn!(error = %e, "probe failed; using default reader workers");
                            DEFAULT_READER_WORKERS
                        }
                    },
                    None => DEFAULT_READER_WORKERS,
                }
            } else {
                DEFAULT_READER_WORKERS
            };

            let sharded = crate::budget::plan_sharded(crate::budget::PlanInputs {
                usable_cores: available_cores.saturating_sub(1).max(1),
                n_contigs: chroms.len(),
                n_samples: samples.len(),
                chunk_bytes,
                max_mem_bytes,
                reader_workers,
            });
            let sharded = match sharded {
                Ok(p) => p,
                Err(e) => return vec![Err(crate::error::ConversionError::from(e))],
            };

            let plan = crate::budget::plan_thread_budget(available_cores, chroms.len());
            let concurrent_chroms =
                orchestrator::bench_concurrent_chroms(sharded.concurrent_chroms);
            let htslib_threads = plan.htslib_threads; // monolithic path only
            let reader_workers = sharded.reader_workers;
            let processing_threads = plan.processing_threads;
```

Note that FORMAT fields enter `chunk_bytes` through the `n_format` term, so a
run with configured fields automatically gets a smaller `cc`. That is the
spec's "be conservative when fields are configured" risk item, and it needs no
separate rule — but it does mean `fields` must be parsed *before* this block,
which it already is (`src/lib.rs:176-179`).

Define `const DEFAULT_READER_WORKERS: usize = 3;` near the top of `lib.rs` with a
comment recording that 3 is the low end of the scale bench's measured knee of
3-7, chosen low because the executor is the bottleneck and surplus readers
steal cores from *other* contigs' executors.

Add a `From<budget::PlanError> for ConversionError` impl in `src/error.rs` so
the insufficient-memory case reaches Python as a clear error.

- [ ] **Step 3: Order the contigs before dispatch**

Replace `chroms.par_iter()` at `src/lib.rs:258-259` with an ordered copy:

```rust
            let costs = crate::contig_cost::estimate_contig_costs(&vcf_path, &chroms);
            let ordered = crate::contig_cost::order_longest_first(&chroms, &costs);
            let results = pool.install(|| {
                ordered
                    .par_iter()
                    .with_min_len(1)
                    .map(|chrom| {
```

`with_min_len(1)` keeps a single contig stealable, so an idle worker can take
one rather than waiting for a split.

**`chroms` must stay in its original order everywhere else.** `finalize_fields`
and `write_meta` at `src/lib.rs:300-314` both take `&chroms`, and the store's
contig order is part of its layout — reordering it there changes output bytes,
which Task 6 will catch. Only the dispatch iterates `ordered`.

- [ ] **Step 4: Add the Python arguments**

In `python/genoray/_svar2.py`, add to `from_vcf`'s keyword-only block:

```python
        max_mem: int | str | None = None,
        tune: bool = False,
```

and before the `_core.run_conversion_pipeline` call at line 821:

```python
        # `None` means a DETECTED budget, not unbounded. Unbounded preserves
        # exactly the biobank-scale OOM exposure the byte-budgeted planner
        # exists to remove.
        if max_mem is None:
            max_mem_bytes = detect_memory_budget()
        else:
            max_mem_bytes = parse_memory(max_mem)
```

passing `max_mem_bytes` and `tune` positionally in the same order as the Rust
signature. Import `detect_memory_budget` alongside the existing `parse_memory`
import.

Document both in the `from_vcf` docstring, including that `max_mem=None` means
a detected budget rather than no limit.

- [ ] **Step 5: Build and run the full suite**

```bash
export CARGO_TARGET_DIR="$CLAUDE_JOB_DIR/tmp/cargo-target"
pixi run bash -lc 'cargo test --no-default-features --features conversion'
pixi run bash -lc 'maturin develop --release'
pixi run pytest tests/ -q -m "not network"
```
Expected: all green. Run this via `sbatch`, not on the login node — the login node runs at 30-75 loadavg and a full build plus suite there is both slow and unreliable.

- [ ] **Step 6: Commit**

```bash
git add src/lib.rs src/error.rs python/genoray/_svar2.py
git commit -m "feat(svar2): plan and order contigs before dispatch

Concurrency now comes from plan_sharded under core and memory constraints
rather than plan_thread_budget's core arithmetic, contigs dispatch
longest-first so the longest cannot start last, and reader_workers can be
probed from the largest contig with --tune.

max_mem=None means a DETECTED budget, not unbounded -- a default behavior
change. Unbounded preserves exactly the biobank-scale OOM exposure the
byte-budgeted planner exists to remove.

chroms keeps its original order everywhere except dispatch: finalize_fields
and write_meta consume it as store layout, so reordering it there would move
output bytes."
```

---

### Task 6: Digest invariance gate

**Depends on Task 5.**

**Files:**
- Modify: `tests/_oracle.py`
- Create: `tests/test_svar2_schedule_invariance.py`

**Interfaces:**
- Consumes: `from_vcf(..., max_mem=, tune=)`, env hooks `GENORAY_CONCURRENT_CHROMS` / `GENORAY_READER_WORKERS` (`src/orchestrator.rs:80`, `:448`).
- Produces: `tests/_oracle.py::store_digest(path) -> str`.

This is the gate the whole change rides on. `cc`, `w`, and contig order all
move, and each is a chance to perturb chunk ordinals, per-chunk ledgers, or
long-allele bank offsets.

- [ ] **Step 1: Add the digest helper**

Append to `tests/_oracle.py`:

```python
def store_digest(store: Path) -> str:
    """Order-independent hash of every file in a .svar store.

    Deliberately a second copy of `scripts/bench_svar2/probe.py:digest` --
    coupling a standalone bench script to the test package is worse than eight
    duplicated lines. Both must stay in agreement; if one changes, change both.
    """
    h = hashlib.sha256()
    for p in sorted(store.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(store).as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:16]
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_svar2_schedule_invariance.py
"""Scheduling must not change output bytes.

concurrent_chroms, reader_workers, and contig dispatch order all move under
the tuned planner. Each is an opportunity to perturb chunk ordinals, per-chunk
ledgers, or long-allele bank offsets. If this test fails, nothing else in the
tuned-load-balancing change matters.
"""

import os

import pytest

from genoray import SparseVar2

from ._oracle import store_digest

# (concurrent_chroms, reader_workers) -- spans the corners the planner can now
# reach: one contig at a time with many readers, and many contigs with few.
SCHEDULES = [(1, 1), (1, 12), (4, 3), (8, 2)]


@pytest.fixture(scope="module")
def multi_contig_vcf(tmp_path_factory):
    """Eight contigs with DIFFERENT record counts.

    Unequal counts are the point: with equal contigs, longest-first ordering
    is a no-op and the invariance test proves nothing about reordering.
    """
    import subprocess

    d = tmp_path_factory.mktemp("sched")
    contigs = {f"chr{i}": 4 * i for i in range(1, 9)}  # 4, 8, ... 32 records
    length = 4 * max(contigs.values()) + 10

    header = [
        "##fileformat=VCFv4.2",
        '##FILTER=<ID=PASS,Description="">',
        *[f"##contig=<ID={c},length={length}>" for c in contigs],
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="">',
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1",
    ]
    rows = [
        f"{c}\t{4 * j + 1}\t.\tA\tG\t.\t.\t.\tGT\t0|1\t1|1"
        for c, n in contigs.items()
        for j in range(n)
    ]
    vcf = d / "sched.vcf"
    vcf.write_text("\n".join(header + rows) + "\n")

    vcf_gz = d / "sched.vcf.gz"
    subprocess.run(f"bgzip -c {vcf} > {vcf_gz}", shell=True, check=True)
    subprocess.run(["bcftools", "index", str(vcf_gz)], check=True)
    return vcf_gz


def _convert(vcf, out, cc, w):
    # The bench hooks are read in-process by the Rust orchestrator
    # (src/orchestrator.rs:80 and :448), so they must be set on os.environ --
    # a subprocess env would never reach this process's pipeline.
    os.environ.update(
        GENORAY_CONCURRENT_CHROMS=str(cc), GENORAY_READER_WORKERS=str(w)
    )
    try:
        SparseVar2.from_vcf(out, vcf, no_reference=True, chunk_size=64)
    finally:
        os.environ.pop("GENORAY_CONCURRENT_CHROMS", None)
        os.environ.pop("GENORAY_READER_WORKERS", None)
    return store_digest(out)


def test_digest_is_invariant_across_schedules(multi_contig_vcf, tmp_path):
    digests = {}
    for cc, w in SCHEDULES:
        out = tmp_path / f"cc{cc}_w{w}.svar"
        digests[(cc, w)] = _convert(multi_contig_vcf, out, cc, w)
    assert len(set(digests.values())) == 1, f"schedule changed output: {digests}"


def test_max_mem_too_small_raises_rather_than_writing_an_empty_store(
    multi_contig_vcf, tmp_path
):
    out = tmp_path / "tiny.svar"
    with pytest.raises(Exception, match="max_mem"):
        SparseVar2.from_vcf(
            out, multi_contig_vcf, no_reference=True, chunk_size=64, max_mem="1M"
        )


def test_tune_does_not_change_output(multi_contig_vcf, tmp_path):
    a = tmp_path / "untuned.svar"
    b = tmp_path / "tuned.svar"
    SparseVar2.from_vcf(a, multi_contig_vcf, no_reference=True, chunk_size=64)
    SparseVar2.from_vcf(
        b, multi_contig_vcf, no_reference=True, chunk_size=64, tune=True
    )
    assert store_digest(a) == store_digest(b)
```

Fill in `multi_contig_vcf` using whichever VCF-building helper `tests/conftest.py`
already provides; if none exists, write one with `cyvcf2`/`pysam` following the
pattern in `tests/test_from_vcf_livelock_fixture.py`. The contigs **must** have
different record counts.

- [ ] **Step 3: Run to verify it fails**

```bash
pixi run pytest tests/test_svar2_schedule_invariance.py -v
```
Expected: FAIL — `store_digest` missing, or `max_mem` unknown kwarg, before Task 5 lands.

- [ ] **Step 4: Run against the implementation**

```bash
export CARGO_TARGET_DIR="$CLAUDE_JOB_DIR/tmp/cargo-target"
pixi run bash -lc 'maturin develop --release'
pixi run pytest tests/test_svar2_schedule_invariance.py -v
```
Expected: 3 passed.

**If `test_digest_is_invariant_across_schedules` fails, stop and report.** A
digest that moves with the schedule is a correctness bug in the pipeline, not a
test to adjust. Note which files differ (`diff -r` the two stores) before
escalating.

- [ ] **Step 5: Commit**

```bash
git add tests/_oracle.py tests/test_svar2_schedule_invariance.py
git commit -m "test(svar2): gate scheduling changes on output-byte invariance

concurrent_chroms, reader_workers, and contig dispatch order all move under
the tuned planner, and each can perturb chunk ordinals, per-chunk ledgers, or
long-allele bank offsets. Pins the store digest across four schedules
spanning the corners the planner can now reach, and across tune on/off."
```

---

### Task 7: Public API docs and a bench point

**Depends on Task 5. Parallel with Task 6.**

**Files:**
- Modify: `skills/genoray-api/SKILL.md`
- Modify: `scripts/bench_svar2/plans/build_plans.py`
- Test: `tests/bench/test_build_plans.py`

- [ ] **Step 1: Document the new kwargs**

Add `max_mem` and `tune` to `from_vcf`'s entry in `skills/genoray-api/SKILL.md`.
State plainly that `max_mem=None` means a **detected** budget (cgroup limit ×
0.8), not unbounded, since that is a behavior change a downstream reader will
otherwise assume the other way.

- [ ] **Step 2: Write the failing test for the new bench points**

Add to `tests/bench/test_build_plans.py`:

```python
def test_concurrency_plan_spans_the_measured_corners(tmp_path):
    """The contig axis compared cc=1,w=12 against cc=4,w=3 and found 2.99x.
    The new planner can reach cc=15; the sweep must cover that, or it cannot
    tell whether the planner's choice is the good one."""
    plans = build(tmp_path, threads=48)
    cc_values = {
        p.concurrent_chroms for p in plans["concurrency"] if p.concurrent_chroms
    }
    assert max(cc_values) >= 15
    assert 1 in cc_values
```

- [ ] **Step 3: Run to verify it fails**

```bash
pixi run pytest tests/bench/test_build_plans.py::test_concurrency_plan_spans_the_measured_corners -v
```
Expected: FAIL — `KeyError: 'concurrency'`

- [ ] **Step 4: Add the sweep axis**

Add to `build_plans.py`, beside the existing axis constants:

```python
# The contig axis only ever compared cc=1 against cc=4, which cannot say
# whether the planner's cc=15 is the good choice or merely better than cc=1.
# w is pinned at the low end of the measured 3-7 knee: the executor is the
# bottleneck, so surplus readers steal cores from OTHER contigs' executors.
CONCURRENCY_CHROMS = (1, 4, 8, 15, 22)
CONCURRENCY_READER_WORKERS = 3
```

and inside `build()`, alongside the other axes:

```python
    for cc in CONCURRENCY_CHROMS:
        concurrency.append(
            SweepPoint(
                corpus=str(corpus_dir / "s4000_c22.manifest.json"),
                reader_workers=CONCURRENCY_READER_WORKERS,
                concurrent_chroms=cc,
                shard_htslib=0,
                overshard=4,
                chunk_size=10_937,
                threads=threads,
                reps=3,
                rss_ceiling_mb=None,
            )
        )
```

Unpack `concurrency` alongside the other lists at the top of `build()`, and add
`"concurrency": concurrency` to the returned dict. Update
`test_all_five_plans_are_produced` to six plans and rename it accordingly.

- [ ] **Step 5: Run tests**

```bash
pixi run pytest tests/bench/ -q
```
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add skills/genoray-api/SKILL.md scripts/bench_svar2/plans/build_plans.py tests/bench/test_build_plans.py
git commit -m "docs(svar2): document max_mem/tune and sweep the concurrency axis

The contig axis only ever compared cc=1 against cc=4. The new planner reaches
cc=15 on a 48-core box, so the sweep needs points there to say whether its
choice is the good one. SKILL.md states that max_mem=None is a detected
budget rather than unbounded, since a reader will otherwise assume the
opposite."
```

---

## Verification before the PR

Run on a dedicated allocation with `--nodelist` pinned — node speed here varies
2.08×, well over any effect this change should be judged by.

```bash
export CARGO_TARGET_DIR="$CLAUDE_JOB_DIR/tmp/cargo-target"
pixi run bash -lc 'cargo test --no-default-features --features conversion'
pixi run bash -lc 'cargo check --no-default-features'
pixi run bash -lc 'maturin develop --release'
pixi run pytest tests/ -q -m "not network"
```

Then measure the thing the change exists for: the 22-contig corpus at
`S=4,000` previously ran 11.74 s using 5.24 of 48 cores at `cc=4, w=3`, against
35.11 s at `cc=1, w=12`. The planner should now choose `cc=15` unprompted and
beat 11.74 s. Report the wall time, the cores used, and the node.
