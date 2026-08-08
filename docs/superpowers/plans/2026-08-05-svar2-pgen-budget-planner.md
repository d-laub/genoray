# PGEN Budget Planner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Tasks marked **[parallel: group X]** in the dependency map below may be dispatched concurrently via superpowers:dispatching-parallel-agents.

**Goal:** Give `SparseVar2.from_pgen` the same core-and-memory-bounded concurrency planner `from_vcf` received in PR #141, with PGEN's own measured RAM-law coefficients.

**Architecture:** `budget::plan_sharded` is generalized from VCF-only to backend-parameterized by extracting its three fitted peak-RSS coefficients into a `RamLaw` struct. PGEN supplies `reader_workers = 1` (its single `pgenlib` reader thread), exact per-contig costs derived from `.pvar` index ranges at zero I/O cost, and its own `RamLaw::PGEN` fitted from a new bench arm. A shared `processing_threads_for` helper makes both backends size the merge tail against the concurrency actually dispatched.

**Tech Stack:** Rust (pyo3, rayon), Python 3.10+, pixi, `vcfixture` bulk CLI (Rust, external), `plink2`, Slurm.

**Spec:** `docs/superpowers/specs/2026-08-05-svar2-pgen-budget-planner-design.md`

## Global Constraints

Every task's requirements implicitly include this section.

- **Rust tests MUST be run as `cargo test --no-default-features --features conversion`.** Dropping `extension-module` is required or the pyo3 test binary will not link; keeping `conversion` is required or you silently skip the entire conversion path (bare `--no-default-features` runs 189 tests and compiles neither the executor nor the orchestrator; the correct invocation runs 439+).
- **Set `CARGO_TARGET_DIR` off NFS before any cargo command:** `export CARGO_TARGET_DIR=$CLAUDE_JOB_DIR/tmp/cargo-target` (or another node-local path). An NFS `target/` directory causes bus errors in lint hooks and `cargo test` failures to mmap object files. Do **not** use bare `/tmp` for anything you need to survive — it is reaped mid-session on this cluster.
- **`pixi run test` does NOT rebuild the Rust extension.** Any Python-level verification of a Rust change requires `pixi run maturin develop --release` first, or the test silently runs stale code.
- **Python tests:** `pixi run pytest tests/ -q -m "not network"`.
- **Core gate:** `cargo check --no-default-features` must stay clean — this is the query-core build GenVarLoader links against, and it is a separate CI job from the conversion build.
- **Never background a long build or test.** Run cargo/maturin/pytest in the foreground and wait. A backgrounded build that returns early produces a task reported complete against stale artifacts.
- **Compute-heavy work goes to `sbatch`, not the interactive session.** The login node runs at 30–75 loadavg; the same measurement has been observed at 5 s on a dedicated allocation and 300 s on the login node.
- **Any cross-point timing comparison must pin `--nodelist`** and stamp the node in the results. Node speed on this cluster varies by 2.08× (151.9 s vs 73.2 s for an identical job).
- **Conventional Commits** (`feat:`, `fix:`, `perf:`, `docs:`, `test:`, `refactor:`). Never edit `CHANGELOG.md` — commitizen owns it. Never bump the version by hand.
- **Public API rule:** anything reachable from `import genoray` without a leading underscore that is added, renamed, or changed in meaning REQUIRES updating `skills/genoray-api/SKILL.md` in the same commit. `from_pgen(max_mem=)` is public.
- **Coordinate convention:** ranges are 0-based, half-open `[start, end)`.
- **prek hooks must be installed** before committing (`pixi run prek-install`).

## Dependency Map

Two tracks run concurrently. Track B produces the constants Track A's final wiring consumes.

```
Track A (Rust planner)          Track B (bench + fit)
──────────────────────          ─────────────────────
[parallel: group 1]             [parallel: group 1]
  Task 1  RamLaw                  Task 6  derived profile
  Task 3  ContigCosts::exact      Task 7  pgen_corpus.py   (needs 6)
     │                            Task 8  probe/plans      (needs 7)
  Task 2  processing_threads_for     │
     │    (needs 1)                Task 9  sweep + fit     (needs 8)
     │                               │
     └──────────────┬────────────────┘
                    │
              Task 10  RamLaw::PGEN constant  (needs 1, 9)
                    │
              Task 4  wire run_pgen_conversion_pipeline (needs 2, 3, 10)
                    │
              Task 5  from_pgen(max_mem=) + CLI + SKILL.md (needs 4)
                    │
              Task 11 PGEN schedule-invariance tests (needs 5)
```

**Group 1 (dispatch concurrently):** Task 1, Task 3, Task 6.
Task 1 and Task 2 both edit `src/budget.rs` and must not run concurrently. Task 3 edits only `src/contig_cost.rs`. Track B touches only `scripts/bench_svar2/`.

---

### Task 1: Extract `RamLaw` from the loose RAM constants

Behavior-preserving refactor. `plan_sharded` reads its three coefficients from a struct on `PlanInputs` instead of three module constants, so a second backend can supply its own.

**Files:**
- Modify: `src/budget.rs:99-200` (constants, `PlanInputs`, `plan_sharded`)
- Modify: `src/lib.rs:283-291` (the one production `plan_sharded` call site)
- Test: `src/budget.rs` (inline `#[cfg(test)] mod tests`)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `pub struct RamLaw { pub base_mb: f64, pub per_sample_mb: f64, pub kappa: f64 }` (derives `Debug, Clone, Copy, PartialEq`)
  - `RamLaw::VCF` associated constant
  - `PlanInputs` gains `pub ram: RamLaw`
  - `PlanInputs` **loses its `Eq` derive** (it now holds `f64`s)

- [ ] **Step 1: Write the failing test**

Add to `src/budget.rs`'s `mod tests`:

```rust
#[test]
fn ram_law_vcf_reproduces_the_fitted_coefficients() {
    assert_eq!(RamLaw::VCF.base_mb, 932.0);
    assert_eq!(RamLaw::VCF.per_sample_mb, 0.01115);
    assert_eq!(RamLaw::VCF.kappa, 1.371);
}

#[test]
fn plan_sharded_uses_the_supplied_ram_law_not_a_global() {
    // Two identical inputs differing ONLY in the law: a law with twice the
    // kappa must halve the memory-bound concurrency. If plan_sharded still
    // read module constants, both would return the same cc.
    let base = PlanInputs {
        usable_cores: 64,
        n_contigs: 32,
        n_samples: 1_000,
        chunk_bytes: 100_000_000,
        max_mem_bytes: Some(20_000_000_000),
        reader_workers: 1,
        ram: RamLaw {
            base_mb: 1000.0,
            per_sample_mb: 0.0,
            kappa: 1.0,
        },
    };
    let doubled = PlanInputs {
        ram: RamLaw {
            kappa: 2.0,
            ..base.ram
        },
        ..base
    };
    let a = plan_sharded(base).unwrap().concurrent_chroms;
    let b = plan_sharded(doubled).unwrap().concurrent_chroms;
    assert_eq!(a, 2 * b, "cc must scale inversely with kappa: {a} vs {b}");
}
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
export CARGO_TARGET_DIR=$CLAUDE_JOB_DIR/tmp/cargo-target
cargo test --no-default-features --features conversion budget:: 2>&1 | tail -20
```

Expected: FAIL to compile — `cannot find type RamLaw in this scope`.

- [ ] **Step 3: Replace the constants with the struct**

In `src/budget.rs`, replace lines 99–106 (the comment block and the three `pub const`s) with:

```rust
/// Fitted peak-RSS coefficients for one conversion backend:
///   peak_rss_mb ~ base_mb + per_sample_mb*samples + kappa*(w+pending)*chunk_bytes
///
/// These are load-bearing in production, not just in the bench: a bad refit
/// becomes an OOM. Change a law only alongside a refit that says so, and
/// record that refit's R^2 and n in the constant's doc comment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RamLaw {
    pub base_mb: f64,
    pub per_sample_mb: f64,
    pub kappa: f64,
}

impl RamLaw {
    /// Sharded VCF path. Fitted 2026-08-03, R^2 = 0.9040, n = 44.
    /// See docs/superpowers/specs/2026-08-03-svar2-tuned-load-balancing-design.md.
    pub const VCF: RamLaw = RamLaw {
        base_mb: 932.0,
        per_sample_mb: 0.01115,
        kappa: 1.371,
    };
}
```

- [ ] **Step 4: Add the field to `PlanInputs` and drop `Eq`**

`PlanInputs` currently derives `Eq`, which `f64` does not implement. Change the derive line and add the field:

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
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
    /// Which backend's fitted peak-RSS law to plan against.
    pub ram: RamLaw,
}
```

- [ ] **Step 5: Point `plan_sharded` at the supplied law**

In `plan_sharded`'s `Some(budget)` arm, replace the two constant-reading lines:

```rust
            let baseline_mb = inp.ram.base_mb + inp.ram.per_sample_mb * inp.n_samples as f64;
            let pending = w.saturating_sub(1);
            let per_contig_mb =
                inp.ram.kappa * (w + pending) as f64 * (inp.chunk_bytes as f64 / 1e6);
```

- [ ] **Step 6: Update the production call site**

In `src/lib.rs`, inside `run_conversion_pipeline`'s `plan_sharded(crate::budget::PlanInputs { ... })` literal (around line 283), add the field:

```rust
                reader_workers: DEFAULT_READER_WORKERS,
                ram: crate::budget::RamLaw::VCF,
            });
```

- [ ] **Step 7: Fix every `PlanInputs` literal in the existing tests**

The `mod tests` block builds `PlanInputs` in several places. Each literal needs `ram: RamLaw::VCF,` added. Compile errors point at each one:

```bash
export CARGO_TARGET_DIR=$CLAUDE_JOB_DIR/tmp/cargo-target
cargo test --no-default-features --features conversion budget:: 2>&1 | rg "^error" -A 4 | head -40
```

Add the field until it compiles. Do **not** change any existing test's expected values — this refactor must not move a single number.

- [ ] **Step 8: Run the full budget + planner test set**

```bash
export CARGO_TARGET_DIR=$CLAUDE_JOB_DIR/tmp/cargo-target
cargo test --no-default-features --features conversion 2>&1 | tail -15
```

Expected: PASS, with the same test count as before this task plus 2. Every pre-existing `plan_sharded` / `plan_thread_budget` assertion must still hold unchanged — this is the proof the refactor is behavior-preserving.

- [ ] **Step 9: Check the core gate**

```bash
export CARGO_TARGET_DIR=$CLAUDE_JOB_DIR/tmp/cargo-target
cargo check --no-default-features 2>&1 | tail -5
```

Expected: clean.

- [ ] **Step 10: Commit**

```bash
git add src/budget.rs src/lib.rs
git commit -m "refactor(svar2): extract the fitted RAM coefficients into a RamLaw struct

plan_sharded read three module constants, which hardcoded it to the VCF
backend. Moving them onto PlanInputs lets a second backend supply its own
fitted law without duplicating the planner. PlanInputs loses its Eq derive
because RamLaw holds f64s; nothing depended on it.

Behavior-preserving: every existing planner assertion holds unchanged."
```

---

### Task 2: Add `processing_threads_for` and switch the VCF pipeline to it

`processing_threads` currently comes from `plan_thread_budget`, computed against *that function's* concurrency — not the `cc` the sharded path actually dispatches. It sizes the merge tail (`merge.rs`'s gather pool, `dense_merge`'s transpose), so the mismatch wastes real parallelism.

**Files:**
- Modify: `src/budget.rs` (new public function + tests)
- Modify: `src/lib.rs:301` (`let processing_threads = plan.processing_threads;`)
- Test: `src/budget.rs` inline tests

**Interfaces:**
- Consumes: Task 1's `RamLaw` (same file; do not run concurrently with Task 1).
- Produces: `pub fn processing_threads_for(usable_cores: usize, cc: usize, w: usize) -> usize`

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn processing_threads_for_returns_the_cores_left_after_executors_and_readers() {
    // 47 usable, 11 contigs at (1 executor + 3 readers) = 44 spent, 3 left.
    assert_eq!(processing_threads_for(47, 11, 3), 3);
    // PGEN shape: w = 1, so each contig costs 2.
    assert_eq!(processing_threads_for(47, 22, 1), 3);
}

#[test]
fn processing_threads_for_floors_at_one_when_oversubscribed() {
    // Never 0: the merge tail must always get a usable thread count, and a
    // rayon pool of 0 threads panics at build time.
    assert_eq!(processing_threads_for(4, 8, 3), 1);
    assert_eq!(processing_threads_for(1, 1, 1), 1);
}
```

- [ ] **Step 2: Run to verify it fails**

```bash
export CARGO_TARGET_DIR=$CLAUDE_JOB_DIR/tmp/cargo-target
cargo test --no-default-features --features conversion processing_threads_for 2>&1 | tail -10
```

Expected: FAIL — `cannot find function processing_threads_for`.

- [ ] **Step 3: Implement it**

Add to `src/budget.rs`, next to `plan_sharded`:

```rust
/// Cores left after the planned concurrency's executors and readers.
///
/// Sizes the merge tail — `merge.rs`'s var_key gather pool and
/// `dense_merge`'s bit-transpose — which runs per contig once its pipeline
/// drains. Both backends use this so the tail is sized against the
/// concurrency actually dispatched, not against a different planner's
/// hypothetical one.
///
/// Floors at 1: `rayon::ThreadPoolBuilder::num_threads(0)` means "use the
/// global default", not "no threads", so returning 0 here would silently
/// oversubscribe rather than serialize.
pub fn processing_threads_for(usable_cores: usize, cc: usize, w: usize) -> usize {
    usable_cores.saturating_sub(cc * (1 + w)).max(1)
}
```

- [ ] **Step 4: Run to verify it passes**

```bash
export CARGO_TARGET_DIR=$CLAUDE_JOB_DIR/tmp/cargo-target
cargo test --no-default-features --features conversion processing_threads_for 2>&1 | tail -10
```

Expected: PASS (2 tests).

- [ ] **Step 5: Switch the VCF pipeline to it**

In `src/lib.rs`, `run_conversion_pipeline` currently has (around line 299–301):

```rust
            let plan = crate::budget::plan_thread_budget(available_cores, chroms.len());
            let concurrent_chroms =
                orchestrator::bench_concurrent_chroms(sharded.concurrent_chroms);
            let htslib_threads = plan.htslib_threads; // monolithic path only
            let reader_workers = sharded.reader_workers;
            let processing_threads = plan.processing_threads;
```

Change only the last line:

```rust
            // Sized against the concurrency this path actually dispatches
            // (`concurrent_chroms`, from `plan_sharded`) — NOT against
            // `plan_thread_budget`'s own `concurrent_chroms`, which models the
            // monolithic reader's 6-cores-per-contig shape and is only still
            // consulted here for `htslib_threads`.
            let processing_threads = crate::budget::processing_threads_for(
                available_cores.saturating_sub(1).max(1),
                concurrent_chroms,
                reader_workers,
            );
```

- [ ] **Step 6: Verify the whole Rust suite still passes**

```bash
export CARGO_TARGET_DIR=$CLAUDE_JOB_DIR/tmp/cargo-target
cargo test --no-default-features --features conversion 2>&1 | tail -15
```

Expected: PASS. This changes `from_vcf`'s merge-tail width, so any e2e test asserting on output BYTES must still pass — `processing_threads` must not affect output content, only how fast it is produced. A failure here is a real bug, not an expectation to update.

- [ ] **Step 7: Verify Python e2e is unaffected**

```bash
pixi run maturin develop --release
pixi run pytest tests/ -q -m "not network" 2>&1 | tail -5
```

Expected: same pass count as before the task.

- [ ] **Step 8: Commit**

```bash
git add src/budget.rs src/lib.rs
git commit -m "perf(svar2): size the merge tail against the dispatched concurrency

processing_threads came from plan_thread_budget, computed against that
function's concurrent_chroms rather than the cc plan_sharded actually
dispatches. Since #141 those differ, so the merge tail -- merge.rs's
gather pool and dense_merge's transpose, both parallel as of 8625f70 and
e31d815 -- was sized off a stale number.

processing_threads_for is shared so both backends size it the same way."
```

---

### Task 3: Add `ContigCosts::exact`

**[parallel: group 1]** — touches only `src/contig_cost.rs`.

PGEN's `.pvar` index ranges *are* exact per-contig record counts. This constructor lets the PGEN path build costs with zero I/O and no htslib involvement.

**Files:**
- Modify: `src/contig_cost.rs:255-265` (add an `impl` block after the struct)
- Test: `src/contig_cost.rs` inline tests

**Interfaces:**
- Consumes: nothing.
- Produces: `ContigCosts::exact(values: HashMap<String, u64>) -> ContigCosts`

- [ ] **Step 1: Write the failing test**

Add to `src/contig_cost.rs`'s `mod tests`:

```rust
#[test]
fn exact_marks_counts_as_exact_and_orders_longest_first() {
    let mut m = HashMap::new();
    m.insert("chr1".to_string(), 100u64);
    m.insert("chr2".to_string(), 500u64);
    m.insert("chr3".to_string(), 250u64);
    let costs = ContigCosts::exact(m);

    assert!(
        costs.exact_counts,
        "counts from .pvar ranges are exact record counts, not a fallback tier"
    );
    let chroms = vec![
        "chr1".to_string(),
        "chr2".to_string(),
        "chr3".to_string(),
    ];
    assert_eq!(
        order_longest_first(&chroms, &costs),
        vec!["chr2".to_string(), "chr3".to_string(), "chr1".to_string()]
    );
}
```

Note `order_longest_first(&chroms, &costs)` relies on `ContigCosts`' existing `Deref` to `HashMap` — no `.values` access needed.

- [ ] **Step 2: Run to verify it fails**

```bash
export CARGO_TARGET_DIR=$CLAUDE_JOB_DIR/tmp/cargo-target
cargo test --no-default-features --features conversion contig_cost:: 2>&1 | tail -10
```

Expected: FAIL — `no function or associated item named exact found`.

- [ ] **Step 3: Implement it**

Add immediately after the `ContigCosts` struct definition in `src/contig_cost.rs`:

```rust
impl ContigCosts {
    /// Costs known exactly without probing any index.
    ///
    /// Used by the PGEN path, where the `.pvar` variant-index range
    /// `[lo, hi)` for a contig IS its record count. Marked `exact_counts`
    /// because these are genuine record counts, which is what licenses a
    /// `min(chunk_size, records)` resident-chunk estimate downstream.
    ///
    /// Nothing here touches htslib, so none of this module's index-tier
    /// hazards (tabix-vs-header id spaces, `hts_idx_get_stat` bounds) are
    /// reachable through this constructor.
    pub fn exact(values: HashMap<String, u64>) -> Self {
        ContigCosts {
            values,
            exact_counts: true,
        }
    }
}
```

- [ ] **Step 4: Run to verify it passes**

```bash
export CARGO_TARGET_DIR=$CLAUDE_JOB_DIR/tmp/cargo-target
cargo test --no-default-features --features conversion contig_cost:: 2>&1 | tail -10
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/contig_cost.rs
git commit -m "feat(svar2): add ContigCosts::exact for index-free per-contig costs

The PGEN path already knows each contig's exact record count from its
.pvar index range, so it needs neither the tabix tier nor the header-length
fallback -- and reaches none of their FFI hazards."
```

---

### Task 6: Generate and commit the variant-skewed bench profile

**[parallel: group 1]** — touches only `scripts/bench_svar2/`.

`vcfixture bulk --records` splits contigs proportional to fitted `density_per_kb`, which is near-uniform (measured max/min 1.38× across 22 autosomes) where real 1kGP is 6.07×. A flat corpus makes longest-first dispatch unmeasurable. `density_per_kb` has exactly one consumer in vcfixture (`distribute_by_density`), so rescaling it to the fitted `n_variants` fixes the split and changes nothing else.

**Files:**
- Create: `scripts/bench_svar2/profiles/derive_varskew.py`
- Create: `scripts/bench_svar2/profiles/germline-1kgp-varskew.json` (generated, committed)
- Test: `tests/bench/test_derive_varskew.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `derive_varskew(profile: dict) -> dict` — pure transform, importable by the test.

- [ ] **Step 1: Write the failing test**

Create `tests/bench/test_derive_varskew.py`:

```python
"""The derived profile must reproduce the source profile's per-contig
variant PROPORTIONS, because that is the only thing it exists to fix."""

from __future__ import annotations

from scripts.bench_svar2.profiles.derive_varskew import derive_varskew


def _profile(counts: dict[str, int]) -> dict:
    return {
        "name": "src",
        "provenance": {"n_samples_source": 3202, "fitted_on": "2026-07-16"},
        "fitted": {
            "contigs": [
                # Densities are deliberately near-uniform, like the real
                # profile: that is what makes the density split useless.
                {"id": cid, "n_variants": n, "density_per_kb": 23.0}
                for cid, n in counts.items()
            ]
        },
    }


def test_density_is_rescaled_to_variant_counts():
    src = _profile({"1": 6_000_000, "2": 1_000_000})
    out = derive_varskew(src)
    d = {c["id"]: c["density_per_kb"] for c in out["fitted"]["contigs"]}
    # distribute_by_density weights by density, so the RATIO is what matters.
    assert d["1"] / d["2"] == 6.0


def test_source_profile_is_not_mutated():
    src = _profile({"1": 6_000_000, "2": 1_000_000})
    derive_varskew(src)
    assert src["fitted"]["contigs"][0]["density_per_kb"] == 23.0


def test_provenance_records_the_source_and_the_transform():
    out = derive_varskew(_profile({"1": 10, "2": 20}))
    assert out["name"] == "germline-1kgp-varskew"
    assert out["provenance"]["derived_from"] == "src"
    assert "density_per_kb" in out["provenance"]["derivation"]


def test_variant_counts_are_left_alone():
    """Only the split weight changes. n_variants stays as fitted, so the
    file still documents the real cohort it came from."""
    out = derive_varskew(_profile({"1": 6_000_000, "2": 1_000_000}))
    n = {c["id"]: c["n_variants"] for c in out["fitted"]["contigs"]}
    assert n == {"1": 6_000_000, "2": 1_000_000}
```

- [ ] **Step 2: Run to verify it fails**

```bash
pixi run pytest tests/bench/test_derive_varskew.py -q 2>&1 | tail -10
```

Expected: FAIL — `ModuleNotFoundError: scripts.bench_svar2.profiles.derive_varskew`.

- [ ] **Step 3: Implement the transform**

Create `scripts/bench_svar2/profiles/__init__.py` (empty) and `scripts/bench_svar2/profiles/derive_varskew.py`:

```python
"""Derive a variant-skewed vcfixture profile from a fitted one.

`vcfixture bulk`'s `Size::Records` splits records across contigs
proportional to each contig's `density_per_kb` (`distribute_by_density` in
vcfixture's src/bulk/mod.rs). For the human profiles those densities are
near-uniform -- ~21-26 variants/kb across the autosomes -- so a 22-contig
corpus comes out at max/min 1.38x where the real cohort is 6.07x. An even
corpus makes contig-level scheduling unmeasurable: every contig finishes at
once, so longest-first dispatch and makespan tails look like no-ops.

`density_per_kb` is read in EXACTLY ONE place in vcfixture -- that split --
and nothing else consumes it (positions, gap distribution, SFS, class mix
all ignore it). So overwriting it with `n_variants` changes the
apportionment and nothing else about the generated data.

This is a workaround for vcfixture-rs#15 (no way to express per-contig
counts through `Size`). If that lands, prefer it: this transform depends on
`density_per_kb` having a single consumer, and would break silently if that
ever stops holding.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

DERIVED_NAME = "germline-1kgp-varskew"


def derive_varskew(profile: dict) -> dict:
    """Return a copy of `profile` whose per-contig `density_per_kb` is the
    contig's fitted `n_variants`.

    Only the ratios between contigs matter to `distribute_by_density`, so
    the absolute units of the rewritten field are irrelevant -- it is a
    split weight after this transform, not a density.
    """
    out = copy.deepcopy(profile)
    for contig in out["fitted"]["contigs"]:
        contig["density_per_kb"] = float(contig["n_variants"])
    out["name"] = DERIVED_NAME
    out.setdefault("provenance", {})
    out["provenance"]["derived_from"] = profile.get("name", "unknown")
    out["provenance"]["derivation"] = (
        "density_per_kb overwritten with n_variants so vcfixture's "
        "distribute_by_density apportions records by true per-contig variant "
        "counts instead of near-uniform densities (vcfixture-rs#15)"
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "source",
        type=Path,
        help="path to a fitted vcfixture profile JSON (e.g. "
        "vcfixture-rs/profiles/germline-1kgp.json)",
    )
    ap.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path(__file__).parent / f"{DERIVED_NAME}.json",
    )
    args = ap.parse_args()
    derived = derive_varskew(json.loads(args.source.read_text()))
    args.out.write_text(json.dumps(derived, indent=1) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify it passes**

```bash
pixi run pytest tests/bench/test_derive_varskew.py -q 2>&1 | tail -5
```

Expected: PASS (4 tests).

- [ ] **Step 5: Generate the committed profile**

The source profile lives in the sibling `vcfixture-rs` checkout. If it is absent, clone it or fetch `profiles/germline-1kgp.json` from `d-laub/vcfixture-rs`.

```bash
pixi run python -m scripts.bench_svar2.profiles.derive_varskew \
  /carter/users/dlaub/projects/vcfixture-rs/profiles/germline-1kgp.json
```

- [ ] **Step 6: Verify the derived profile has the expected skew**

```bash
pixi run python -c "
import json
p = json.load(open('scripts/bench_svar2/profiles/germline-1kgp-varskew.json'))
auto = [c for c in p['fitted']['contigs'] if c['id'] in {str(i) for i in range(1, 23)}]
w = [c['density_per_kb'] for c in auto]
print('n autosomes:', len(auto))
print('max/min split weight:', round(max(w) / min(w), 3))
"
```

Expected: `n autosomes: 22` and `max/min split weight: 6.072`. If it prints ~1.38 the transform did not apply; if it prints 22 contigs but a different ratio, the source profile has been refitted upstream — record the new ratio in the commit message rather than forcing the old one.

- [ ] **Step 7: Commit**

```bash
git add scripts/bench_svar2/profiles/ tests/bench/test_derive_varskew.py
git commit -m "feat(bench): add a variant-skewed vcfixture profile for the PGEN corpus

vcfixture bulk splits records by fitted density_per_kb, which is near
uniform across the human autosomes (max/min 1.38x) where the real cohort is
6.07x. An even corpus makes contig-level scheduling unmeasurable.

density_per_kb has exactly one consumer in vcfixture, so rewriting it with
n_variants fixes the split and changes nothing else. Committed rather than
derived at bench time: a fitted RAM law wants a frozen corpus distribution.

Workaround for vcfixture-rs#15."
```

---

### Task 7: PGEN corpus generation module

**Files:**
- Create: `scripts/bench_svar2/pgen_corpus.py`
- Test: `tests/bench/test_pgen_corpus.py`

**Interfaces:**
- Consumes: Task 6's `scripts/bench_svar2/profiles/germline-1kgp-varskew.json`; `CorpusManifest` from `scripts/bench_svar2/records.py`.
- Produces:
  - `resolve_vcfixture() -> Path` (raises `FileNotFoundError` with an install line)
  - `PgenCorpusSpec` dataclass: `samples: int`, `variants: int`, `contigs: tuple[str, ...]`, `seed: int`
  - `generate(spec: PgenCorpusSpec, outdir: Path) -> CorpusManifest` — returns a manifest whose `path` is the `.pgen`

- [ ] **Step 1: Write the failing test**

Create `tests/bench/test_pgen_corpus.py`:

```python
"""Corpus generation shells out to two external binaries. The tests that
need them are skipped when they are absent -- the vcfixture bulk CLI is NOT
the PyPI `vcfixture` package and is not in this project's pixi env, so an
unguarded test passes locally and fails only in CI."""

from __future__ import annotations

import shutil

import pytest

from scripts.bench_svar2.pgen_corpus import (
    PgenCorpusSpec,
    apportion,
    resolve_vcfixture,
)


def _have_tools() -> bool:
    try:
        resolve_vcfixture()
    except FileNotFoundError:
        return False
    return shutil.which("plink2") is not None


needs_tools = pytest.mark.skipif(
    not _have_tools(), reason="vcfixture bulk CLI and/or plink2 unavailable"
)


def test_resolve_vcfixture_error_names_the_install_command(monkeypatch):
    monkeypatch.delenv("VCFIXTURE_BIN", raising=False)
    monkeypatch.setattr(shutil, "which", lambda _: None)
    with pytest.raises(FileNotFoundError, match="cargo install vcfixture"):
        resolve_vcfixture()


def test_apportion_splits_by_weight_and_sums_exactly():
    counts = apportion(total=100, weights={"chr1": 6.0, "chr2": 4.0})
    assert counts == {"chr1": 60, "chr2": 40}
    assert sum(counts.values()) == 100


def test_apportion_never_yields_a_zero_count():
    """A contig with no records would be declared in the header but absent
    from the data -- the exact shape that crashed from_vcf_list in #122."""
    counts = apportion(total=3, weights={"a": 1000.0, "b": 1.0, "c": 1.0})
    assert min(counts.values()) >= 1
    assert sum(counts.values()) == 3


def test_stem_is_unique_per_shape():
    """Manifests land FLAT in one corpora dir and `model._load_manifests`
    keys them by FILENAME, globbing `*.manifest.json` non-recursively. Two
    corpora sharing a stem would silently overwrite each other's manifest."""
    from scripts.bench_svar2.pgen_corpus import corpus_stem

    a = corpus_stem(PgenCorpusSpec(4000, 250_000, ("chr1",), 42))
    b = corpus_stem(PgenCorpusSpec(4000, 500_000, ("chr1",), 42))
    c = corpus_stem(PgenCorpusSpec(32_000, 250_000, ("chr1",), 42))
    assert len({a, b, c}) == 3
    # build_plans and tests/bench/test_build_plans.py both parse the shape
    # back out of this stem.
    assert a == "pgen_s4000_v250000"


@needs_tools
def test_generate_produces_a_readable_pgen(tmp_path):
    from genoray import SparseVar2

    from scripts.bench_svar2.pgen_corpus import generate

    spec = PgenCorpusSpec(
        samples=20, variants=6000, contigs=("chr1", "chr2", "chr3"), seed=7
    )
    manifest = generate(spec, tmp_path)

    assert manifest.path.endswith(".pgen")
    assert manifest.samples == 20
    assert (tmp_path / "pgen_s20_v6000.manifest.json").exists()

    store = tmp_path / "roundtrip.svar"
    SparseVar2.from_pgen(
        store,
        manifest.path,
        no_reference=True,
        skip_out_of_scope=True,
        log_level="off",
    )
    sv = SparseVar2(store)
    # plink2 strips the chr prefix unless --output-chr is passed; generate()
    # passes it, so the store must carry the prefixed names.
    assert sv.contigs == ["chr1", "chr2", "chr3"]


@needs_tools
def test_generate_is_cached(tmp_path):
    from scripts.bench_svar2.pgen_corpus import generate

    spec = PgenCorpusSpec(samples=10, variants=3000, contigs=("chr1",), seed=1)
    first = generate(spec, tmp_path)
    pgen = tmp_path / "pgen_s10_v3000.pgen"
    mtime = pgen.stat().st_mtime_ns
    second = generate(spec, tmp_path)
    assert first == second
    assert pgen.stat().st_mtime_ns == mtime, (
        "a cached corpus must not be regenerated"
    )
```

- [ ] **Step 2: Run to verify it fails**

```bash
pixi run pytest tests/bench/test_pgen_corpus.py -q 2>&1 | tail -10
```

Expected: FAIL — `ModuleNotFoundError: scripts.bench_svar2.pgen_corpus`.

- [ ] **Step 3: Implement the module**

Create `scripts/bench_svar2/pgen_corpus.py`:

```python
"""Generate a human-genome-shaped PGEN corpus for the SVAR2 conversion bench.

    vcfixture bulk --profile germline-1kgp-varskew.json ... -o corpus.bcf
    plink2 --bcf corpus.bcf --make-pgen --output-chr chrM --out corpus

Distinct from `scale_corpus.py`, which hand-writes VCF text for the VCF arm.
This one streams from a profile fitted on real 1kGP data (3,202 samples,
73.6M variants), which is what "human-genome-like" requires: a real site
frequency spectrum, gap distribution, variant-class mix, ti/tv, and indel
length distribution.

The two corpora come from different generators, so the VCF and PGEN RAM laws
are NOT comparable coefficient-by-coefficient. Each is fitted on, and used
for, its own backend.

Gotchas, all measured rather than assumed:

- plink2 strips the `chr` prefix from .pvar DATA rows while copying
  `##contig=<ID=chr1>` header lines verbatim, producing an internally
  inconsistent .pvar. `from_pgen` reads the data rows, so the store's
  contigs would silently become `1..22`. `--output-chr chrM` prevents it.
- The germline profile emits symbolic ALTs (`<DEL>`) at a low rate and
  plink2 passes them straight through into the .pvar. Convert with
  `skip_out_of_scope=True`.
- germline-1kgp has multiallelic_rate 0.0, so this corpus does NOT exercise
  the multiallelic allele_idx_offsets path. The fitted law should not be
  claimed to cover multiallelic-heavy cohorts.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
from pathlib import Path

from scripts.bench_svar2.records import CorpusManifest

PROFILE = Path(__file__).parent / "profiles" / "germline-1kgp-varskew.json"

# Bumped whenever a change to this module would alter the generated bytes,
# so a stale cached corpus is regenerated instead of silently reused.
GENERATOR_VERSION = 1


@dataclasses.dataclass(frozen=True)
class PgenCorpusSpec:
    samples: int
    variants: int
    contigs: tuple[str, ...]
    seed: int


def resolve_vcfixture() -> Path:
    """Locate the `vcfixture` bulk CLI.

    This is NOT the PyPI `vcfixture` package pinned in pixi.toml -- that
    ships no console script (no entry_points.txt, no bin/vcfixture). The
    bulk generator is a separate Rust binary. Shelling out to a bare
    `vcfixture` therefore passes on a dev box that happens to have it built
    and fails in CI with FileNotFoundError, so resolution is explicit and
    the error says how to fix it.
    """
    env = os.environ.get("VCFIXTURE_BIN")
    if env:
        p = Path(env)
        if p.is_file():
            return p
        raise FileNotFoundError(f"VCFIXTURE_BIN={env} is not a file")
    found = shutil.which("vcfixture")
    if found:
        return Path(found)
    raise FileNotFoundError(
        "vcfixture bulk CLI not found. It is a Rust binary, separate from the "
        "PyPI `vcfixture` package (which has no CLI). Install it with "
        "`cargo install vcfixture --features cli`, or point VCFIXTURE_BIN at "
        "an existing build."
    )


def apportion(total: int, weights: dict[str, float]) -> dict[str, int]:
    """Split `total` across keys proportional to `weights`, summing exactly.

    Largest-remainder, with a floor of 1 per key: a contig declared in the
    header but carrying zero records is the exact shape that crashed
    from_vcf_list in issue #122, and it is not a case this bench wants to
    spend a corpus on.
    """
    if total < len(weights):
        raise ValueError(
            f"cannot give {len(weights)} contigs at least one record each "
            f"out of {total}"
        )
    total_w = sum(weights.values())
    exact = {k: total * w / total_w for k, w in weights.items()}
    counts = {k: max(1, int(v)) for k, v in exact.items()}
    # Distribute the remainder by descending fractional part, breaking ties
    # on key so the result never depends on dict iteration order.
    short = total - sum(counts.values())
    order = sorted(weights, key=lambda k: (-(exact[k] - int(exact[k])), k))
    i = 0
    while short > 0:
        counts[order[i % len(order)]] += 1
        short -= 1
        i += 1
    while short < 0:
        k = order[i % len(order)]
        if counts[k] > 1:
            counts[k] -= 1
            short += 1
        i += 1
    return counts


def _weights_for(contigs: tuple[str, ...]) -> dict[str, float]:
    """Per-contig split weights from the derived profile.

    The derived profile already carries n_variants in `density_per_kb`
    (see profiles/derive_varskew.py), so this reads the field vcfixture
    itself would use.
    """
    profile = json.loads(PROFILE.read_text())
    by_id = {}
    for c in profile["fitted"]["contigs"]:
        by_id[c["id"]] = float(c["density_per_kb"])
        by_id[f"chr{c['id']}"] = float(c["density_per_kb"])
    missing = [c for c in contigs if c not in by_id]
    if missing:
        raise ValueError(f"profile has no fitted entry for: {', '.join(missing)}")
    return {c: by_id[c] for c in contigs}


def corpus_stem(spec: PgenCorpusSpec) -> str:
    """Filename stem for a corpus of this shape.

    Corpora land FLAT in one directory because `model._load_manifests` globs
    `*.manifest.json` non-recursively and keys by FILENAME -- a per-corpus
    subdirectory would make every manifest `corpus.manifest.json` and they
    would collide. The shape is encoded here so `plans/build_plans.py` can
    name the corpus it wants without consulting a manifest first.
    """
    return f"pgen_s{spec.samples}_v{spec.variants}"


def generate(spec: PgenCorpusSpec, outdir: Path) -> CorpusManifest:
    """Generate (or reuse) a PGEN corpus in `outdir`.

    Cached on the full spec plus GENERATOR_VERSION: `vcfixture --seed`
    produces byte-identical output regardless of thread count, so a corpus
    is reproducible and there is no reason to pay for it twice.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    stem = corpus_stem(spec)
    manifest_path = outdir / f"{stem}.manifest.json"
    key = {**dataclasses.asdict(spec), "generator_version": GENERATOR_VERSION}

    if manifest_path.exists():
        cached = json.loads(manifest_path.read_text())
        if cached.get("_key") == key and Path(cached["path"]).exists():
            payload = {k: v for k, v in cached.items() if k != "_key"}
            payload["contigs"] = tuple(payload["contigs"])
            payload["format_fields"] = tuple(payload["format_fields"])
            return CorpusManifest(**payload)

    vcfixture = resolve_vcfixture()
    bcf = outdir / f"{stem}.bcf"
    subprocess.run(
        [
            str(vcfixture),
            "bulk",
            "--profile",
            str(PROFILE),
            "--samples",
            str(spec.samples),
            "--contigs",
            ",".join(spec.contigs),
            "--records",
            str(spec.variants),
            "--payload",
            "gt-only",
            "--format",
            "bcf",
            "--seed",
            str(spec.seed),
            "-o",
            str(bcf),
        ],
        check=True,
    )

    # --output-chr chrM: keep the `chr` prefix on .pvar DATA rows. Without it
    # plink2 writes `1` in the body while copying `##contig=<ID=chr1>` into
    # the header, and from_pgen (which reads the body) yields a store whose
    # contigs silently disagree with the source BCF's.
    subprocess.run(
        [
            "plink2",
            "--bcf",
            str(bcf),
            "--make-pgen",
            "--output-chr",
            "chrM",
            "--out",
            str(outdir / stem),
        ],
        check=True,
    )

    pgen = outdir / f"{stem}.pgen"
    manifest = CorpusManifest(
        path=str(pgen),
        samples=spec.samples,
        variants=spec.variants,
        contigs=spec.contigs,
        format_fields=(),
        ploidy=2,
        cells=spec.samples * spec.variants,
        compressed_bytes=pgen.stat().st_size,
        seed=spec.seed,
        generator_version=GENERATOR_VERSION,
    )
    payload = dataclasses.asdict(manifest)
    payload["_key"] = key
    manifest_path.write_text(json.dumps(payload, indent=1) + "\n")
    return manifest
```

- [ ] **Step 4: Run to verify it passes**

```bash
export VCFIXTURE_BIN=/carter/users/dlaub/projects/vcfixture-rs/target/release/vcfixture
pixi run pytest tests/bench/test_pgen_corpus.py -q 2>&1 | tail -10
```

Expected: PASS. If `VCFIXTURE_BIN` is unset the two `@needs_tools` tests skip and the other three still pass — verify that too:

```bash
env -u VCFIXTURE_BIN pixi run pytest tests/bench/test_pgen_corpus.py -q 2>&1 | tail -5
```

Expected: 3 passed, 2 skipped.

- [ ] **Step 5: Commit**

```bash
git add scripts/bench_svar2/pgen_corpus.py tests/bench/test_pgen_corpus.py
git commit -m "feat(bench): generate a human-genome-shaped PGEN corpus

vcfixture bulk against the variant-skewed profile, then plink2 --make-pgen.
Records three measured gotchas: plink2 strips the chr prefix from .pvar data
rows without --output-chr, the germline profile emits symbolic ALTs that
survive into the .pvar, and the bulk CLI is a Rust binary unrelated to the
PyPI vcfixture package -- so its absence must skip, not fail."
```

---

### Task 8: Teach the probe and plan builder about the PGEN backend

**Files:**
- Modify: `scripts/bench_svar2/records.py` (add `backend` to `SweepPoint`)
- Modify: `scripts/bench_svar2/probe.py:172-190` (`_build_cmd`)
- Modify: `scripts/bench_svar2/plans/build_plans.py` (a `pgen` family)
- Test: `tests/bench/test_probe.py`, `tests/bench/test_build_plans.py`

**Interfaces:**
- Consumes: Task 7's `PgenCorpusSpec` / `generate`.
- Produces: `SweepPoint.backend: str` (`"vcf"` default, `"pgen"`); a `"pgen"` key in `build()`'s returned dict.

- [ ] **Step 1: Write the failing test**

Add to `tests/bench/test_probe.py`:

```python
def test_build_cmd_dispatches_on_backend():
    from scripts.bench_svar2.probe import _build_cmd
    from scripts.bench_svar2.records import CorpusManifest, SweepPoint

    manifest = CorpusManifest(
        path="/tmp/corpus.pgen",
        samples=10,
        variants=100,
        contigs=("chr1",),
        format_fields=(),
        ploidy=2,
        cells=1000,
        compressed_bytes=1,
        seed=0,
        generator_version=1,
    )
    point = SweepPoint(
        corpus=manifest.path,
        reader_workers=1,
        concurrent_chroms=4,
        shard_htslib=0,
        overshard=4,
        chunk_size=1000,
        threads=8,
        reps=1,
        backend="pgen",
    )
    cmd = _build_cmd(point, manifest, Path("/tmp/store.svar"))
    assert "pgen" in cmd
    assert "vcf" not in cmd
    # Symbolic ALTs survive plink2 into the .pvar, so the PGEN arm must skip
    # them or every conversion aborts on the first <DEL>.
    assert "--skip-symbolics-and-breakends" in cmd


def test_build_cmd_defaults_to_vcf():
    from scripts.bench_svar2.probe import _build_cmd
    from scripts.bench_svar2.records import CorpusManifest, SweepPoint

    manifest = CorpusManifest(
        path="/tmp/corpus.vcf.gz",
        samples=10,
        variants=100,
        contigs=("chr1",),
        format_fields=(),
        ploidy=2,
        cells=1000,
        compressed_bytes=1,
        seed=0,
        generator_version=1,
    )
    point = SweepPoint(
        corpus=manifest.path,
        reader_workers=1,
        concurrent_chroms=None,
        shard_htslib=0,
        overshard=4,
        chunk_size=1000,
        threads=8,
        reps=1,
    )
    assert "vcf" in _build_cmd(point, manifest, Path("/tmp/s.svar"))
```

- [ ] **Step 2: Run to verify it fails**

```bash
pixi run pytest tests/bench/test_probe.py -q -k backend 2>&1 | tail -10
```

Expected: FAIL — `SweepPoint.__init__() got an unexpected keyword argument 'backend'`.

- [ ] **Step 3: Add the field**

In `scripts/bench_svar2/records.py`, add to `SweepPoint` after `rss_ceiling_mb`:

```python
    # Which conversion path this point measures. Defaults to "vcf" so every
    # existing point's `point_id` -- a hash over all fields -- is unchanged
    # and recorded sweeps stay resumable.
    backend: str = "vcf"
```

Placing it last with a default is required: `point_id` hashes `dataclasses.asdict(self)`, so inserting a field earlier (or without a default) would change every existing point's id and invalidate recorded baselines.

- [ ] **Step 4: Dispatch in `_build_cmd`**

Replace `scripts/bench_svar2/probe.py`'s `_build_cmd` with:

```python
def _build_cmd(point: SweepPoint, manifest: CorpusManifest, store: Path) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "genoray._cli",
        "write",
        point.backend,
        manifest.path,
        str(store),
        "--no-reference",
        "--log-level",
        "info",
        "--overwrite",
        "-@",
        str(point.threads),
        "--chunk-size",
        str(point.chunk_size),
    ]
    if point.backend == "pgen":
        # The germline-1kgp profile emits symbolic ALTs at a low rate and
        # plink2 passes them into the .pvar, where check_ref="e" would abort
        # the whole conversion on the first one.
        cmd.append("--skip-symbolics-and-breakends")
    return cmd
```

- [ ] **Step 5: Run to verify it passes**

```bash
pixi run pytest tests/bench/test_probe.py -q 2>&1 | tail -5
```

Expected: PASS, including the pre-existing probe tests.

- [ ] **Step 6: Write the plan-family test**

Add to `tests/bench/test_build_plans.py`:

```python
def test_pgen_family_has_two_v_ladders_at_different_sample_counts():
    """A ladder that holds S*V constant forces the cohort exponent to ~1
    arithmetically -- it cannot identify beta no matter how many points it
    has. Two V-ladders at DIFFERENT S is the minimum that can."""
    from scripts.bench_svar2.plans.build_plans import build

    plans = build(Path("/tmp/corpora"), threads=48)
    pgen = plans["pgen"]
    assert pgen, "pgen family must not be empty"
    assert all(p.backend == "pgen" for p in pgen)

    by_samples: dict[int, set[int]] = {}
    for p in pgen:
        s, v = _shape_of(p)
        by_samples.setdefault(s, set()).add(v)
    ladders = [s for s, vs in by_samples.items() if len(vs) >= 2]
    assert len(ladders) >= 2, (
        f"need >=2 V-ladders at different S to identify the cohort exponent; "
        f"got {by_samples}"
    )


def test_pgen_concurrency_axis_holds_workers_at_one():
    """PGEN pins P=1 (sub-contig sharding disabled), so a reader_workers
    axis would measure nothing."""
    from scripts.bench_svar2.plans.build_plans import build

    for p in build(Path("/tmp/corpora"), threads=48)["pgen"]:
        assert p.reader_workers == 1
```

`_shape_of` reads the corpus shape a point refers to. Implement it in the test module next to the tests:

```python
def _shape_of(point) -> tuple[int, int]:
    """(samples, variants) encoded in the corpus path, e.g. `.../s4000_v250000/`."""
    import re

    m = re.search(r"s(\d+)_v(\d+)", point.corpus)
    assert m, f"corpus path does not encode its shape: {point.corpus}"
    return int(m.group(1)), int(m.group(2))
```

- [ ] **Step 7: Run to verify it fails**

```bash
pixi run pytest tests/bench/test_build_plans.py -q -k pgen 2>&1 | tail -10
```

Expected: FAIL — `KeyError: 'pgen'`.

- [ ] **Step 8: Add the `pgen` family**

In `scripts/bench_svar2/plans/build_plans.py`, add near the other ladder constants:

```python
# Two V-ladders at DIFFERENT cohort widths. A single ladder, or two ladders
# holding S*V constant, pins the cohort exponent to ~1 by construction --
# that failure has already produced one published-then-retracted interval
# in this project, so the shape of this ladder is load-bearing.
PGEN_LADDERS = ((4_000, (250_000, 500_000, 1_000_000)),
                (32_000, (250_000, 500_000, 1_000_000)))
# Concurrency axis: bracket the core bound (usable/2 = 23 on a 48-core box)
# so the sweep can show where the GIL stops paying, if it does.
PGEN_CONCURRENCY = (1, 4, 8, 11, 16, 22)
PGEN_CONCURRENCY_AT = (4_000, 1_000_000)
```

Then inside `build()`, add the family (returning it in the dict alongside the existing ones):

```python
    pgen = []
    for s, vs in PGEN_LADDERS:
        for v in vs:
            corpus = corpus_dir / f"pgen_s{s}_v{v}.pgen"
            cs = _chunk_size_for(v)
            # w is always 1: from_pgen pins P=1, so there is no reader-worker
            # axis to sweep. The RAM-law points leave concurrency unset so the
            # planner's own choice is what gets measured.
            pgen.append(
                SweepPoint(
                    corpus=str(corpus),
                    reader_workers=1,
                    concurrent_chroms=None,
                    shard_htslib=0,
                    overshard=4,
                    chunk_size=cs,
                    threads=threads,
                    reps=3,
                    backend="pgen",
                )
            )
    s_cc, v_cc = PGEN_CONCURRENCY_AT
    corpus_cc = corpus_dir / f"pgen_s{s_cc}_v{v_cc}.pgen"
    for cc in PGEN_CONCURRENCY:
        pgen.append(
            SweepPoint(
                corpus=str(corpus_cc),
                reader_workers=1,
                concurrent_chroms=cc,
                shard_htslib=0,
                overshard=4,
                chunk_size=_chunk_size_for(v_cc),
                threads=threads,
                reps=3,
                backend="pgen",
            )
        )
```

Add `"pgen": pgen` to the dict `build()` returns.

- [ ] **Step 9: Run to verify it passes**

```bash
pixi run pytest tests/bench/ -q 2>&1 | tail -5
```

Expected: PASS — the whole bench test suite, not just the new tests.

- [ ] **Step 10: Commit**

```bash
git add scripts/bench_svar2/ tests/bench/
git commit -m "feat(bench): add a PGEN backend axis to the probe and plan builder

SweepPoint.backend defaults to \"vcf\" and is added LAST so point_id -- a
hash over all fields -- is unchanged for every existing point and recorded
sweeps stay resumable.

The pgen family runs two V-ladders at different S, because a single ladder
(or two at constant S*V) pins the cohort exponent arithmetically."
```

---

### Task 9: Run the sweep and fit `RamLaw::PGEN`

This task's deliverable is **measurements**, not code: the fitted coefficients plus a results document. It must run on a dedicated Slurm allocation.

**Files:**
- Create: `scripts/bench_svar2/sweep_pgen.sbatch`
- Create: `docs/superpowers/plans/results/2026-08-05-pgen-ram-law-fit.md`

**Interfaces:**
- Consumes: Task 8's `pgen` plan family; Task 7's `generate`/`PgenCorpusSpec`; and four existing helpers in `scripts/bench_svar2/model.py`, none of which should be reimplemented:
  - `_load_manifests(manifests_dir: Path) -> dict[str, CorpusManifest]` — globs `*.manifest.json` FLAT, keyed by filename
  - `load_sweep(name: str, results_dir: Path, plans_dir: Path, manifests) -> _LoadedSweep` — joins ndjson ↔ plan ↔ manifests; never raises, records failures in `.excluded`
  - `_ram_rows(*sweeps: _LoadedSweep) -> list[RamRow]` — applies `_resident_chunk_size`, which is what keeps `kappa` from being fitted ~10× low
  - `fit_ram_law(rows: Sequence[RamRow]) -> RamLaw` — returns `base_mb`, `per_sample_mb`, `kappa`, `r2`, `n`
- Produces: three fitted `f64`s + R² + n, consumed by Task 10; and the measured `cc` curve, which decides whether a `PGEN_MAX_CONCURRENT` cap is needed.

- [ ] **Step 1: Write the sbatch script**

Create `scripts/bench_svar2/sweep_pgen.sbatch`, modelled on the existing `sweep_scale.sbatch`:

Modelled on `sweep_scale.sbatch`, which is the reference for every convention here.

```bash
#!/bin/bash
#SBATCH --job-name=svar2-pgen
#SBATCH --partition=carter-compute
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --nodelist=carter-cn-04
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.log

set -euo pipefail

# --nodelist is pinned, not optional: node speed on this cluster varies by
# 2.08x, so an unpinned sweep cannot support any cross-point comparison.
# Change WT to this worktree's path before submitting.
WT=/carter/users/dlaub/projects/genoray/.claude/worktrees/svar2-pgen-budget-planner
# Slurm's default cwd is the SUBMISSION directory, so every
# `python -m scripts.bench_svar2.<mod>` below needs this to resolve as a package.
cd "$WT"
PX="pixi run --manifest-path $WT/pixi.toml"

# NOT $CLAUDE_JOB_DIR/tmp: /tmp inside a Slurm job is a PRIVATE slurmtmpfs bind
# mount, so a job writing there dies with exit 53 and no log. /local/$USER is
# visible from the compute node.
SCRATCH="/local/$USER/pgen-sweep"
mkdir -p "$SCRATCH"/{corpora,plans,out}

export VCFIXTURE_BIN="${VCFIXTURE_BIN:?set VCFIXTURE_BIN to the vcfixture bulk binary}"

# Threads must track --cpus-per-task: hardcoding 48 while lowering the
# allocation would oversubscribe it and measure against a pool the job does
# not own. Note that changing --cpus-per-task changes the REGIME, not just the
# cost -- results at different widths must not be pooled.
THREADS=$($PX python -c 'import os;print(len(os.sched_getaffinity(0)))')
echo "=== node $(hostname) nproc=$(nproc) allocated=$THREADS ==="

# Rebuild the extension FIRST. `pixi run` does NOT rebuild it, so without this
# the sweep silently measures whatever .so is installed -- possibly one
# predating the planner change entirely. `set -e` makes a build failure abort
# in the first minute rather than after hours of useless data.
export CARGO_TARGET_DIR="$SCRATCH/cargo-target"
export PATH="$HOME/.cargo/bin:$PATH"
$PX maturin develop --release

# --- corpora (generation is compute-heavy; it belongs on the allocation) ------
$PX python -c "
from pathlib import Path
from scripts.bench_svar2.pgen_corpus import PgenCorpusSpec, generate
from scripts.bench_svar2.plans.build_plans import PGEN_LADDERS
contigs = tuple(f'chr{i}' for i in range(1, 23))
out = Path('$SCRATCH/corpora')
for s, vs in PGEN_LADDERS:
    for v in vs:
        m = generate(PgenCorpusSpec(s, v, contigs, seed=42), out)
        print(m.path, m.compressed_bytes, flush=True)
"

# --- plans and sweep ---------------------------------------------------------
$PX python -m scripts.bench_svar2.plans.build_plans \
  --corpus-dir "$SCRATCH/corpora" --out-dir "$SCRATCH/plans" --threads "$THREADS"

$PX python -m scripts.bench_svar2.sweep \
  --plan "$SCRATCH/plans/pgen.json" \
  --results "$SCRATCH/out/pgen.ndjson" \
  --outdir "$SCRATCH/out/pgen"

echo "=== ALL DONE (node $(hostname)) ==="
```

Corpus shapes come from `PGEN_LADDERS` rather than being repeated here, so the corpora cannot drift out of sync with the plan points built against them — the same single-source-of-truth pattern `sweep_scale.sbatch` uses for its hold-out and V-ladders.

- [ ] **Step 2: Submit and wait**

```bash
sbatch scripts/bench_svar2/sweep_pgen.sbatch
```

Poll with `squeue -u $USER`. Do not proceed until it finishes; confirm with:

```bash
sacct -j <jobid> --format=JobID,JobName,State,ExitCode,MaxRSS,Elapsed
```

A non-zero `ExitCode` means the data is incomplete — `run_sweep` resumes (it skips points already fsynced to the ndjson), so resubmit rather than re-deriving anything.

- [ ] **Step 3: Fit the law**

`model.py` already has the join and the fit: `load_sweep` reads the ndjson and joins it against the plan and manifests, `_ram_rows` builds `RamRow`s (applying `_resident_chunk_size`, which is what keeps `kappa` from being dragged ~10× low by nominal chunk sizes), and `fit_ram_law` fits. Do not hand-roll any of it.

```bash
pixi run python -c "
from pathlib import Path
from scripts.bench_svar2.model import _load_manifests, _ram_rows, fit_ram_law, load_sweep

scratch = Path('/local/$USER/pgen-sweep')
manifests = _load_manifests(scratch / 'corpora')
sweep = load_sweep('pgen', scratch / 'out', scratch / 'plans', manifests)
print('excluded:', sweep.excluded)
rows = _ram_rows(sweep)
print('n rows:', len(rows))
law = fit_ram_law(rows)
print(law)
"
```

Expected: an empty (or explained) `excluded` list, `n rows` equal to the number of successful points, and a `RamLaw` with `base_mb`, `per_sample_mb`, `kappa`, `r2`, `n`. A non-empty `excluded` list means points were dropped — read the reasons before trusting the fit.

- [ ] **Step 4: Sanity-check the fit before trusting it**

Three checks, all of which must pass:

1. **R² ≥ 0.8.** The VCF fit reached 0.9040. A markedly worse fit means the law's functional form does not describe the PGEN path, and the right response is to investigate, not to ship the coefficients.
2. **`kappa > 0` and `per_sample_mb >= 0`.** A negative coefficient means the design matrix is collinear — most likely the ladders collapsed to constant S×V.
3. **Residual check for an `n_contigs` term.** Plot or tabulate residuals against contig count. The spec's prediction is that the eager `PgenReader` pool costs tens of MB against a multi-GB baseline, i.e. noise. **Add an `n_contigs` term only if the residuals say so** — do not add it speculatively.

- [ ] **Step 5: Read the concurrency curve**

Tabulate wall time against `concurrent_chroms` for the six `PGEN_CONCURRENCY` points. Two outcomes:

- **Monotonic improvement to the core bound (cc≈22):** PGEN is executor-bound, the GIL is not binding, no cap is needed. Record the numbers.
- **Flattens or degrades before cc=22:** PGEN is GIL-bound past that point. Record the knee; Task 10 adds a `PGEN_MAX_CONCURRENT` constant set to it, **documented with this measurement**.

- [ ] **Step 6: Write the results document**

Create `docs/superpowers/plans/results/2026-08-05-pgen-ram-law-fit.md` recording: the node, the allocation, the commit SHA, the corpus specs, the fitted coefficients with R² and n, the concurrency table, the residual check for an `n_contigs` term, and the cap decision with its justification. Include the raw ndjson path.

- [ ] **Step 7: Commit**

```bash
git add scripts/bench_svar2/sweep_pgen.sbatch docs/superpowers/plans/results/
git commit -m "test(bench): fit the PGEN RAM law and measure the concurrency curve

Run on <node>, <cpus>/<mem>, commit <sha>. Fitted coefficients, R^2, and n
recorded in the results doc alongside the cc curve and the residual check
for an n_contigs term."
```

---

### Task 10: Define `RamLaw::PGEN` from the fit

Separate from Task 9 because the fit is a measurement and this is the decision to ship it. A reviewer can accept the measurement and still reject the constants.

**Files:**
- Modify: `src/budget.rs` (add the `PGEN` associated constant)
- Test: `src/budget.rs` inline tests

**Interfaces:**
- Consumes: Task 1's `RamLaw`; Task 9's fitted coefficients.
- Produces: `RamLaw::PGEN`; optionally `pub const PGEN_MAX_CONCURRENT: usize`.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn ram_law_pgen_is_a_usable_law() {
    // Guards against a placeholder shipping: a zero kappa would make the
    // memory bound vacuous and silently restore the unbounded planning this
    // whole change exists to remove.
    assert!(RamLaw::PGEN.kappa > 0.0, "kappa must be positive");
    assert!(RamLaw::PGEN.base_mb > 0.0, "baseline must be positive");
    assert!(RamLaw::PGEN.per_sample_mb >= 0.0);
}

#[test]
fn pgen_memory_bound_actually_binds() {
    // A budget that fits the baseline plus exactly two contigs must plan 2,
    // not the core bound. Uses RamLaw::PGEN's real coefficients, so it fails
    // if a future refit makes the law nonsensical.
    let chunk_bytes = 100_000_000u64;
    let baseline_mb = RamLaw::PGEN.base_mb + RamLaw::PGEN.per_sample_mb * 1000.0;
    let per_contig_mb = RamLaw::PGEN.kappa * 1.0 * (chunk_bytes as f64 / 1e6);
    let budget = ((baseline_mb + 2.0 * per_contig_mb) * 1e6) as u64;

    let plan = plan_sharded(PlanInputs {
        usable_cores: 64,
        n_contigs: 22,
        n_samples: 1_000,
        chunk_bytes,
        max_mem_bytes: Some(budget),
        reader_workers: 1,
        ram: RamLaw::PGEN,
    })
    .unwrap();
    assert_eq!(plan.concurrent_chroms, 2);
}

#[test]
fn pgen_budget_too_small_for_one_contig_is_an_error_not_a_silent_cc_of_one() {
    // Below the baseline + one contig, planning must FAIL. Clamping to cc=1
    // and proceeding would OOM at the exact scale the budget exists to
    // protect, and would do it after writing a partial store.
    let err = plan_sharded(PlanInputs {
        usable_cores: 64,
        n_contigs: 22,
        n_samples: 1_000_000,
        chunk_bytes: 10_000_000_000,
        max_mem_bytes: Some(1_000_000),
        reader_workers: 1,
        ram: RamLaw::PGEN,
    })
    .unwrap_err();
    match err {
        PlanError::InsufficientMemory {
            needed_mb,
            budget_mb,
        } => assert!(
            needed_mb > budget_mb,
            "needed {needed_mb} must exceed budget {budget_mb}"
        ),
    }
    // The message must name the two knobs a caller can actually turn.
    let msg = err.to_string();
    assert!(msg.contains("max_mem"), "{msg}");
    assert!(msg.contains("chunk_size"), "{msg}");
}
```

- [ ] **Step 2: Run to verify it fails**

```bash
export CARGO_TARGET_DIR=$CLAUDE_JOB_DIR/tmp/cargo-target
cargo test --no-default-features --features conversion ram_law_pgen 2>&1 | tail -10
```

Expected: FAIL — `no associated item named PGEN found`.

- [ ] **Step 3: Add the constant with the fitted values**

In `src/budget.rs`'s `impl RamLaw`, add — substituting the real numbers from Task 9's results document:

```rust
    /// PGEN path. Fitted <DATE from Task 9>, R^2 = <R2>, n = <N>.
    /// See docs/superpowers/plans/results/2026-08-05-pgen-ram-law-fit.md.
    ///
    /// NOT comparable coefficient-by-coefficient with `RamLaw::VCF`: the two
    /// corpora come from different generators (vcfixture bulk vs
    /// scale_corpus.py), so each law is valid only for its own backend.
    ///
    /// Fitted on a corpus with multiallelic_rate 0.0, so this law is not
    /// claimed to cover multiallelic-heavy cohorts.
    pub const PGEN: RamLaw = RamLaw {
        base_mb: /* fitted */,
        per_sample_mb: /* fitted */,
        kappa: /* fitted */,
    };
```

If Task 9 found a concurrency knee below the core bound, also add:

```rust
/// Measured ceiling on useful PGEN contig concurrency: `pgenlib` holds the
/// GIL through decode, so past this point extra concurrent contigs buy no
/// wall time while still costing memory. Measured <DATE> on <NODE> --
/// see the results doc. NOT a guess; if a future pgenlib release drops the
/// GIL through decode, re-measure before raising it.
pub const PGEN_MAX_CONCURRENT: usize = /* measured knee */;
```

- [ ] **Step 4: Run to verify it passes**

```bash
export CARGO_TARGET_DIR=$CLAUDE_JOB_DIR/tmp/cargo-target
cargo test --no-default-features --features conversion 2>&1 | tail -10
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/budget.rs
git commit -m "feat(svar2): add the fitted RamLaw::PGEN coefficients

Fitted on <node> at <sha>, R^2 = <r2>, n = <n>; see the results doc. Records
that the law is not cross-comparable with RamLaw::VCF (different corpus
generators) and does not cover multiallelic-heavy cohorts."
```

---

### Task 4: Wire `run_pgen_conversion_pipeline` to the planner

**Files:**
- Modify: `src/lib.rs:417-520` (signature, `#[pyo3(signature = ...)]`, and the planning block inside `py.detach`)
- Test: `tests/test_e2e.rs` (Rust e2e) plus the Python gate added in Task 11

**Interfaces:**
- Consumes: `RamLaw::PGEN` (Task 10), `processing_threads_for` (Task 2), `ContigCosts::exact` (Task 3).
- Produces: `run_pgen_conversion_pipeline` gains a `max_mem_bytes: Option<u64>` parameter positioned after `sample_perm` and before `log_level`.

- [ ] **Step 1: Add the parameter to the signature**

In `src/lib.rs`, update the `#[pyo3(signature = ...)]` attribute for `run_pgen_conversion_pipeline` to insert `max_mem_bytes=None` after `sample_perm`:

```rust
#[pyo3(signature = (pgen_path, pvar_path, reference_path, chroms, contig_ranges, output_dir, samples, chunk_size, max_threads, long_allele_capacity, skip_out_of_scope, signatures, dosage_fields, readers, dosage_readers, check_ref, region_ranges, regions_overlap, sample_perm, max_mem_bytes=None, log_level = "info".to_string(), receiver = None))]
```

and add the matching Rust parameter after `sample_perm: Vec<usize>,`:

```rust
    max_mem_bytes: Option<u64>,
```

- [ ] **Step 2: Replace the planning block**

Inside the `py.detach(...)` closure, replace these lines:

```rust
            let plan = crate::budget::plan_thread_budget(available_cores, jobs.len());
            let concurrent_chroms = orchestrator::bench_concurrent_chroms(plan.concurrent_chroms);
            let processing_threads = plan.processing_threads;
            tracing::info!(
                concurrent_chroms,
                processing_threads,
                "pipeline config (PGEN)"
            );
```

with:

```rust
            // Per-contig costs are EXACT here and cost no I/O: a contig's
            // .pvar index range [lo, hi) IS its record count. The VCF path
            // needs contig_cost's index tiers for this; PGEN does not, and so
            // reaches none of their FFI hazards.
            let costs = crate::contig_cost::ContigCosts::exact(
                jobs.iter()
                    .map(|(chrom, (lo, hi), _, _)| (chrom.clone(), (hi - lo) as u64))
                    .collect(),
            );

            // Every dosage field is FORMAT-category by construction on this
            // path, so unlike the VCF path there is no INFO to filter out.
            let per_variant_bytes =
                (samples.len() * ploidy / 8 + fields.len() * samples.len() * 4) as u64;

            // The RAM law was fitted against RESIDENT chunk bytes, not the
            // nominal chunk_size: BitGrid3::zeros is alloc_zeroed, so an
            // oversized chunk_size costs address space rather than RSS.
            // The VCF path guards this narrowing with `if costs.exact_counts`
            // because its header-length fallback tier yields BASE PAIRS, a
            // different unit. Here the counts are exact by construction, so
            // the guard would always be true -- do not "restore" it.
            let resident_chunk_size = costs
                .values
                .values()
                .copied()
                .max()
                .map_or(chunk_size, |max_records| {
                    chunk_size.min(max_records as usize)
                });
            let chunk_bytes = per_variant_bytes * resident_chunk_size as u64;

            // reader_workers = 1: from_pgen pins P = 1, so a contig's demand
            // is exactly one executor plus one pgenlib reader -- which is
            // what plan_sharded's `1 + reader_workers` already models.
            let sharded = crate::budget::plan_sharded(crate::budget::PlanInputs {
                usable_cores: available_cores.saturating_sub(1).max(1),
                n_contigs: jobs.len(),
                n_samples: samples.len(),
                chunk_bytes,
                max_mem_bytes,
                reader_workers: 1,
                ram: crate::budget::RamLaw::PGEN,
            });
            let sharded = match sharded {
                Ok(p) => p,
                Err(e) => return vec![Err(crate::error::ConversionError::from(e))],
            };
            let concurrent_chroms =
                orchestrator::bench_concurrent_chroms(sharded.concurrent_chroms);
            let processing_threads = crate::budget::processing_threads_for(
                available_cores.saturating_sub(1).max(1),
                concurrent_chroms,
                sharded.reader_workers,
            );
            tracing::info!(
                concurrent_chroms,
                reader_workers = sharded.reader_workers,
                processing_threads,
                "pipeline config (PGEN)"
            );

            // Longest-first DISPATCH order. Sort the jobs themselves, not a
            // separate name list: each tuple carries that contig's own
            // pgenlib reader pool and dosage-reader pool, so reordering names
            // alone would pair contigs with the wrong readers.
            //
            // `chroms` keeps its original order for finalize_fields/write_meta
            // below -- the store's on-disk contig order is part of its layout.
            // `results` therefore comes back in DISPATCH order; that is safe
            // only because the sole consumer sums every entry unconditionally.
            // A future change that zips `results` positionally against
            // `chroms` would silently misattribute per-contig results.
            let order = crate::contig_cost::order_longest_first(&chroms, &costs);
            let rank: std::collections::HashMap<&str, usize> = order
                .iter()
                .enumerate()
                .map(|(i, c)| (c.as_str(), i))
                .collect();
            let mut jobs = jobs;
            jobs.sort_by_key(|(chrom, _, _, _)| rank[chrom.as_str()]);
```

If Task 10 added `PGEN_MAX_CONCURRENT`, clamp immediately after `bench_concurrent_chroms`:

```rust
            let concurrent_chroms =
                concurrent_chroms.min(crate::budget::PGEN_MAX_CONCURRENT);
```

- [ ] **Step 3: Build and check both gates**

```bash
export CARGO_TARGET_DIR=$CLAUDE_JOB_DIR/tmp/cargo-target
cargo check --no-default-features 2>&1 | tail -5
cargo test --no-default-features --features conversion 2>&1 | tail -15
```

Expected: both clean/PASS. The Python caller does not yet pass `max_mem_bytes`, but it defaults to `None` in the pyo3 signature, so existing calls keep working.

- [ ] **Step 4: Verify PGEN output is byte-identical to before the change**

Longest-first reordering and a different `cc` must not move a single output byte. Build a reference store at the previous commit, then one at HEAD, and compare.

```bash
pixi run maturin develop --release
pixi run pytest tests/ -q -m "not network" -k pgen 2>&1 | tail -5
```

Expected: PASS. Task 11 adds the dedicated invariance gate; this step is the quick check that nothing already-tested regressed.

- [ ] **Step 5: Commit**

```bash
git add src/lib.rs
git commit -m "feat(svar2)!: plan PGEN conversion under core and memory budgets

run_pgen_conversion_pipeline planned with plan_thread_budget, which charges
6 cores per contig -- four pipeline threads plus an HTSlib pool PGEN never
allocates -- and bounded memory not at all. It now plans with plan_sharded
against RamLaw::PGEN at reader_workers=1, sizes the merge tail against the
concurrency it actually dispatches, and dispatches longest-first using exact
.pvar-derived per-contig counts."
```

---

### Task 5: Expose `max_mem` on `from_pgen`, the CLI, and the skill doc

**Files:**
- Modify: `python/genoray/_svar2.py` (`from_pgen` signature ~line 895, docstring, `_core` call ~line 1215)
- Modify: `python/genoray/_cli/__main__.py` (`write_pgen` ~line 257 and its `from_pgen` call ~line 368)
- Modify: `skills/genoray-api/SKILL.md`
- Test: `tests/test_svar2_pgen_schedule_invariance.py` (Task 11)

**Interfaces:**
- Consumes: Task 4's `max_mem_bytes` parameter.
- Produces: `SparseVar2.from_pgen(..., max_mem: int | str | None = None)`; CLI `--max-mem`.

- [ ] **Step 1: Add the parameter**

In `from_pgen`'s signature, add after `chunk_size: int | None = None,`:

```python
        max_mem: int | str | None = None,
```

- [ ] **Step 2: Document it**

Add to the docstring, after the `chunk_size` paragraph:

```
        max_mem: byte budget the concurrency planner may use, as an int or a
        string like `"64GiB"` (see `parse_memory`). **This is a WHOLE-PROCESS
        planning budget**: the number of contigs converted concurrently is
        chosen so the cohort baseline plus each concurrent contig's in-flight
        chunk buffers fit inside it, in addition to the existing core-count
        bound.

        **Same meaning as** :meth:`from_vcf`'s `max_mem`, NOT
        :meth:`from_vcf_list`'s. Both are whole-process budgets, but
        `from_vcf_list` has no concurrency planner to spend one on (its
        contigs run strictly sequentially), so it instead derives its own
        per-chunk `chunk_size` from the budget. Here the budget buys
        concurrency and `chunk_size` keeps its independent default.

        **`None` (the default) means a DETECTED budget** -- 80% of the cgroup
        memory limit (or `/proc/meminfo` total outside a cgroup) -- **not
        unbounded**. Unbounded planning preserves exactly the biobank-scale
        OOM exposure the byte-budgeted planner exists to remove. If detection
        itself fails (no cgroup limit and no readable `/proc/meminfo` -- every
        macOS run, and any host without `/proc`), a warning is issued and
        planning degrades to core-bound rather than raising.
```

- [ ] **Step 3: Resolve the budget before the `_core` call**

Immediately before `from ._logging import write_reporting` in `from_pgen`, add the same block `from_vcf` uses (`python/genoray/_svar2.py:854-867`):

```python
        # `None` means a DETECTED budget, not unbounded -- unbounded preserves
        # exactly the biobank-scale OOM exposure the byte-budgeted planner
        # exists to remove. Detection is an optimization, not a requirement:
        # it raises RuntimeError when there is no cgroup limit AND no readable
        # /proc/meminfo, which is every macOS run (this project ships an
        # osx-arm64 wheel). That must not fail the whole conversion.
        if max_mem is None:
            try:
                max_mem_bytes = detect_memory_budget()
            except RuntimeError as e:
                warnings.warn(
                    f"could not detect a memory budget ({e}); planning "
                    "concurrency by core count only. Pass max_mem explicitly "
                    "to plan against a byte budget.",
                    stacklevel=2,
                )
                max_mem_bytes = None
        else:
            max_mem_bytes = parse_memory(max_mem)
```

`warnings`, `detect_memory_budget`, and `parse_memory` are already imported at module scope by `from_vcf`'s implementation — verify with `rg -n "^import warnings|detect_memory_budget, parse_memory" python/genoray/_svar2.py` rather than adding duplicates.

- [ ] **Step 4: Pass it through**

In the `_core.run_pgen_conversion_pipeline(...)` call, insert after `sample_perm`:

```python
                max_mem_bytes,
```

- [ ] **Step 5: Add the CLI flag**

In `write_pgen`, add after `chunk_size: int | None = None,`:

```python
    max_mem: Annotated[str | None, Parameter(name="--max-mem")] = None,
```

Document it in the command's docstring, and pass it in the `SparseVar2.from_pgen(...)` call:

```python
        chunk_size=chunk_size,
        max_mem=max_mem,
```

- [ ] **Step 6: Update the skill doc**

In `skills/genoray-api/SKILL.md`, find the `from_pgen` entry and add `max_mem` to its parameter list, with the whole-process-budget semantics and the detected-default behavior. Match the wording already used for `from_vcf`'s `max_mem` so the two read consistently.

- [ ] **Step 7: Verify**

```bash
pixi run maturin develop --release
pixi run pytest tests/ -q -m "not network" 2>&1 | tail -5
pixi run python -c "
from genoray import SparseVar2
import inspect
sig = inspect.signature(SparseVar2.from_pgen)
assert 'max_mem' in sig.parameters, sig
print('max_mem present:', sig.parameters['max_mem'])
"
pixi run python -m genoray._cli write pgen --help 2>&1 | rg -- "--max-mem"
```

Expected: tests PASS, `max_mem` present with default `None`, `--max-mem` in the CLI help.

- [ ] **Step 8: Commit**

```bash
git add python/genoray/_svar2.py python/genoray/_cli/__main__.py skills/genoray-api/SKILL.md
git commit -m "feat(svar2): add max_mem to from_pgen

Whole-process planning budget with from_vcf's semantics -- the budget buys
concurrency, chunk_size keeps its own default -- explicitly NOT
from_vcf_list's meaning, where the same kwarg derives chunk_size because
that path has no planner. None means a DETECTED budget, not unbounded."
```

---

### Task 11: PGEN schedule-invariance gate

The correctness gate for everything above. If output bytes move with the schedule, longest-first reordering is unsafe and nothing else matters.

**Files:**
- Create: `tests/test_svar2_pgen_schedule_invariance.py`

**Interfaces:**
- Consumes: Task 5's `from_pgen(max_mem=)`; `tests/_oracle.store_digest`; Task 7's `pgen_corpus` helpers are NOT used — this test builds its own tiny fixture so it runs in CI without external binaries.

- [ ] **Step 1: Write the test**

Create `tests/test_svar2_pgen_schedule_invariance.py`:

```python
"""Scheduling must not change PGEN output bytes.

concurrent_chroms and contig dispatch order both move under the planner.
Each is an opportunity to perturb chunk ordinals, per-chunk ledgers, or
long-allele bank offsets. Mirrors test_svar2_schedule_invariance.py for the
VCF path.

The fixture is built with plink2 from a hand-written VCF rather than from
scripts/bench_svar2/pgen_corpus.py: this gate must run in CI, and the
vcfixture bulk CLI is not available there.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from genoray import SparseVar2

from tests import _oracle

# PGEN pins P=1, so reader_workers has no axis to sweep -- only cc moves.
SCHEDULES = [1, 2, 4, 8]

CHUNK_SIZE = 8

pytestmark = pytest.mark.skipif(
    shutil.which("plink2") is None, reason="plink2 not available"
)


@pytest.fixture(scope="module")
def multi_contig_pgen(tmp_path_factory):
    """Eight contigs with DIFFERENT record counts.

    Unequal counts are the point: with equal contigs, longest-first ordering
    is a no-op and this test proves nothing about reordering.
    """
    d = tmp_path_factory.mktemp("pgen_sched")
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

    # --output-chr chrM keeps the `chr` prefix on .pvar DATA rows; without it
    # plink2 writes `1` in the body while copying ##contig=<ID=chr1> into the
    # header, and from_pgen reads the body.
    subprocess.run(
        [
            "plink2",
            "--vcf",
            str(vcf),
            "--make-pgen",
            "--output-chr",
            "chrM",
            "--out",
            str(d / "sched"),
        ],
        check=True,
    )
    return d / "sched.pgen"


def _convert(pgen, out, cc, monkeypatch):
    # Read in-process by the Rust orchestrator, so it must be set on
    # os.environ -- a subprocess env would never reach this pipeline.
    monkeypatch.setenv("GENORAY_CONCURRENT_CHROMS", str(cc))
    SparseVar2.from_pgen(
        out, pgen, no_reference=True, chunk_size=CHUNK_SIZE, log_level="off"
    )
    return _oracle.store_digest(out)


def test_digest_is_invariant_across_schedules(
    multi_contig_pgen, tmp_path, monkeypatch
):
    digests = {cc: _convert(multi_contig_pgen, tmp_path / f"cc{cc}.svar", cc, monkeypatch)
               for cc in SCHEDULES}
    assert len(set(digests.values())) == 1, f"schedule changed output: {digests}"


def test_dispatch_order_is_longest_first_and_still_writes_meta_in_file_order(
    multi_contig_pgen, tmp_path, monkeypatch
):
    """Dispatch order must not leak into the store's layout: meta.json's
    contig order is part of the on-disk format and comes from `chroms`, not
    from the (reordered) dispatch list."""
    out = tmp_path / "order.svar"
    _convert(multi_contig_pgen, out, 4, monkeypatch)
    sv = SparseVar2(out)
    assert sv.contigs == [f"chr{i}" for i in range(1, 9)]


def test_max_mem_too_small_raises_rather_than_writing_an_empty_store(
    multi_contig_pgen, tmp_path
):
    out = tmp_path / "tiny.svar"
    with pytest.raises(Exception, match="max_mem"):
        SparseVar2.from_pgen(
            out,
            multi_contig_pgen,
            no_reference=True,
            chunk_size=CHUNK_SIZE,
            max_mem="1M",
            log_level="off",
        )
    assert not out.exists(), "a rejected max_mem budget must not create the store dir"


def test_max_mem_none_plans_against_a_detected_budget(
    multi_contig_pgen, tmp_path, monkeypatch
):
    """`None` must mean DETECTED, not unbounded -- unbounded is the OOM
    exposure the byte budget exists to remove."""
    import genoray._svar2 as svar2

    called = []
    real = svar2.detect_memory_budget

    def spy():
        called.append(True)
        return real()

    monkeypatch.setattr(svar2, "detect_memory_budget", spy)
    SparseVar2.from_pgen(
        tmp_path / "detected.svar",
        multi_contig_pgen,
        no_reference=True,
        chunk_size=CHUNK_SIZE,
        log_level="off",
    )
    assert called, "from_pgen(max_mem=None) must consult detect_memory_budget"


def test_detection_failure_warns_and_still_converts(
    multi_contig_pgen, tmp_path, monkeypatch
):
    """Detection raises on any host without a cgroup limit AND without a
    readable /proc/meminfo -- every macOS run. That must degrade to
    core-bound planning, not fail the conversion."""
    import genoray._svar2 as svar2

    def boom():
        raise RuntimeError("no cgroup limit and no /proc/meminfo")

    monkeypatch.setattr(svar2, "detect_memory_budget", boom)
    out = tmp_path / "degraded.svar"
    with pytest.warns(UserWarning, match="could not detect a memory budget"):
        SparseVar2.from_pgen(
            out,
            multi_contig_pgen,
            no_reference=True,
            chunk_size=CHUNK_SIZE,
            log_level="off",
        )
    assert (out / "meta.json").exists(), "degraded planning must still produce a store"
```

- [ ] **Step 2: Run it**

```bash
pixi run maturin develop --release
pixi run pytest tests/test_svar2_pgen_schedule_invariance.py -q 2>&1 | tail -10
```

Expected: PASS (5 tests). A digest mismatch here is a **real bug in the reordering**, not a test to relax — investigate before proceeding.

- [ ] **Step 3: Run the full suite**

```bash
pixi run pytest tests/ -q -m "not network" 2>&1 | tail -5
export CARGO_TARGET_DIR=$CLAUDE_JOB_DIR/tmp/cargo-target
cargo test --no-default-features --features conversion 2>&1 | tail -5
cargo check --no-default-features 2>&1 | tail -3
```

Expected: all PASS/clean.

- [ ] **Step 4: Commit**

```bash
git add tests/test_svar2_pgen_schedule_invariance.py
git commit -m "test(svar2): gate PGEN output bytes against the schedule

concurrent_chroms and longest-first dispatch order must not move a single
output byte. Builds its fixture with plink2 from a hand-written VCF rather
than the bench corpus module, so the gate runs in CI where the vcfixture
bulk CLI is unavailable."
```

---

## Verification Checklist

Run before opening the PR:

- [ ] `cargo test --no-default-features --features conversion` — all pass
- [ ] `cargo check --no-default-features` — clean (query-core gate)
- [ ] `pixi run maturin develop --release && pixi run pytest tests/ -q -m "not network"` — all pass
- [ ] `pixi run pytest tests/bench/ -q` — all pass
- [ ] `env -u VCFIXTURE_BIN pixi run pytest tests/bench/test_pgen_corpus.py -q` — external-tool tests SKIP, not fail
- [ ] `skills/genoray-api/SKILL.md` documents `from_pgen(max_mem=)`
- [ ] `RamLaw::PGEN` carries its R², n, and results-doc reference; no placeholder values
- [ ] The results document records the node, allocation, and commit SHA of the fit
