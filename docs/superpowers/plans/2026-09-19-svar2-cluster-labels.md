# SVAR2 per-mutation cluster labels (`annotate_clusters`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** add a post-hoc `SparseVar2.annotate_clusters(...)` that ports SigProfilerClusters' `findClustersOfClusters` (`correction=False`) and `findClustersOfClusters_noVAF` into a Rust kernel and stores the per-(sample, mutation) subclass as the `cluster_class` FORMAT field (`u8`).

**Spec:** `docs/superpowers/specs/2026-09-19-svar2-cluster-labels-design.md` — read it before starting. It carries the full upstream algorithm walkthrough, the label codebook, every deliberate deviation, and the storage contract.

**Architecture:** a pure Rust classifier (`src/cluster/classify.rs`) over one sample's owned mutation list; a Rust writer (`src/cluster/write.rs`) that gathers mutations from the reader's var_key + dense SNP views, classifies, and writes all four `values.bin` streams (temp + fsync + rename); a pyo3 binding on `PyContigReader` (`src/py_cluster.rs`); a Python mixin (`python/genoray/_svar2_clusters.py`) that validates inputs, resolves `contigs=`, 255-fills out-of-scope contigs, and stamps `meta.json` last.

**Tech stack:** Rust (pyo3 0.29, memmap2, rayon, thiserror, `svar2-codec`), Python 3.10+, pixi tasks, pytest, `vcfixture.VcfBuilder`. The upstream parity oracle (`SigProfilerClusters`) is installed only in the `sigprofiler` pixi env.

---

## Global constraints

- The new Rust modules must compile with `--no-default-features` (query-core consumers like GenVarLoader link this crate): do NOT gate `cluster`/`py_cluster` on `conversion`, and do not import `rust-htslib`, `tracing`, or any writer/convert module from them.
- Keep both guards green: `pixi run -e lint check-core` (`cargo check --no-default-features`) and `pixi run -e lint test-rust` (`cargo test --no-default-features --features conversion`).
- Label codes are a wire contract: `0` nonclustered, `1` doublet, `2` MBS, `3` omikli, `4` kataegis, `5` other, `255` not annotated. Pinned in Rust unit tests AND Python tests.
- `cluster_class` is written to all four sub-streams (`var_key_snp`, `var_key_indel`, `dense_snp`, `dense_indel`); every written file's byte length is asserted against its sub-stream's element count before rename. The reader opens all four, so a missing or short file misindexes the field.
- `meta.json` is stamped only after every in-scope contig's files are renamed into place; the stamp itself is same-directory temp + fsync + `os.replace`.
- No new Python runtime dependency. `import genoray` stays light: the new mixin module imports only stdlib + numpy at module scope (mirror `_svar2_mutcat.py`'s deferral discipline).
- Rust rounding of `|ΔVAF|` must match Python `round(x, 4)`: `(x * 10_000.0).round_ties_even() / 10_000.0`. The ClassIII greedy re-split compares the RAW `|ΔVAF| < vaf_cut` (upstream does; keep the asymmetry).
- Conventional Commits. Pre-commit/pre-push hooks are installed; run `pixi run -e lint lint` before each commit if you want the same checks locally.
- Public API addition: `skills/genoray-api/SKILL.md` MUST be updated in the same PR (repo rule). Never touch `CHANGELOG.md`.
- `pixi run pytest` auto-rebuilds the editable extension via maturin's import hook. If a Rust change does not appear to take effect, run `pixi run maturin develop` once and re-run.

## File map

| path | action |
|---|---|
| `src/cluster/mod.rs` | create: constants, version, submodules |
| `src/cluster/classify.rs` | create: pure classifier + unit tests |
| `src/cluster/write.rs` | create: gather/classify/write four `values.bin` + unit tests |
| `src/py_cluster.rs` | create: `PyContigReader::annotate_clusters` + `fill_cluster_labels` |
| `src/lib.rs` | modify: register `pub mod cluster;` and `pub mod py_cluster;` |
| `python/genoray/_svar2_clusters.py` | create: `_ClustersMixin` + codebook constants |
| `python/genoray/_svar2.py` | modify: inherit `_ClustersMixin` |
| `tests/test_svar2_clusters.py` | create: e2e, ordering pin, validation, atomicity, rerun |
| `tests/test_clusters_calibration.py` | create: upstream parity (skips when not installed) |
| `pixi.toml` | modify: `sigprofilerclusters` in the sigprofiler feature |
| `skills/genoray-api/SKILL.md` | modify: document `annotate_clusters` + codebook |

---

### Task 1: Rust classifier core (IMD, grouping, decision tree)

**Files:**
- Create: `src/cluster/mod.rs`
- Create: `src/cluster/classify.rs`
- Modify: `src/lib.rs` (add `pub mod cluster;` next to the other `pub mod` declarations, NOT inside a `#[cfg(feature = "conversion")]` block)

- [ ] **Step 1: Write the failing tests first.** Create `src/cluster/classify.rs` containing ONLY the test module below plus the public type/function stubs it needs (empty bodies returning `vec![]`/`NONCLUSTERED`), so the module compiles and the tests fail. The signatures are frozen here; the implementation must match them.

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn m(pos: u32, alt: u8, vaf: Option<f64>) -> Mutation {
        Mutation { pos, alt, vaf }
    }

    fn snps(positions: &[u32]) -> Vec<Mutation> {
        positions.iter().map(|&p| m(p, 0, None)).collect()
    }

    #[test]
    fn imds_ends_and_single_mutation() {
        assert!(imds(&[]).is_empty());
        assert_eq!(imds(&snps(&[5])), vec![SINGLE_MUTATION_IMD]);
        // ends look only inward: first uses the down gap, last the up gap
        assert_eq!(imds(&snps(&[10, 14, 30])), vec![4.0, 4.0, 16.0]);
    }

    #[test]
    fn grouping_chains_below_10kb_and_breaks_at_10kb() {
        // cutoff huge so every mutation is clustered; only GROUP_GAP splits
        let muts = snps(&[1, 2, 10_002, 10_003]);
        assert_eq!(
            clustered_events(&muts, &imds(&muts), 1e6),
            vec![vec![0, 1, 2, 3]]
        );
        // a gap of exactly GROUP_GAP starts a new event
        let muts = snps(&[1, 2, 10_003, 10_004]);
        assert_eq!(
            clustered_events(&muts, &imds(&muts), 1e6),
            vec![vec![0, 1], vec![2, 3]]
        );
    }

    #[test]
    fn grouping_skips_nonclustered_mutations() {
        // 20 is beyond the cutoff -> not part of any event
        let muts = snps(&[10, 11, 20]);
        let imd = imds(&muts);
        assert_eq!(clustered_events(&muts, &imd, 5.0), vec![vec![0, 1]]);
    }

    #[test]
    fn tree_leaves() {
        // gap above cutoff -> NONCLUSTERED
        assert_eq!(cluster_sample(&snps(&[10, 20]), 5.0, 0.1), vec![NONCLUSTERED; 2]);
        // adjacent pair -> DOUBLET
        assert_eq!(cluster_sample(&snps(&[10, 11]), 5.0, 0.1), vec![DOUBLET; 2]);
        // 3+ adjacent -> MBS
        assert_eq!(cluster_sample(&snps(&[10, 11, 12]), 5.0, 0.1), vec![MBS; 3]);
        // one line gap, no failures -> OMIKLI
        assert_eq!(cluster_sample(&snps(&[10, 12]), 5.0, 0.1), vec![OMIKLI; 2]);
        // two line gaps -> OMIKLI (distancesLine > 1 but len < 4)
        assert_eq!(cluster_sample(&snps(&[10, 12, 14]), 5.0, 0.1), vec![OMIKLI; 3]);
        // two line gaps and 4+ mutations -> KATAEGIS
        assert_eq!(cluster_sample(&snps(&[10, 12, 14, 16]), 5.0, 0.1), vec![KATAEGIS; 4]);
        // d == cutoff counts toward the line (<=), d == 1 does not
        assert_eq!(cluster_sample(&snps(&[10, 12]), 2.0, 0.1), vec![OMIKLI; 2]);
        // multiallelic distance-0 pair -> DOUBLET
        let pair = vec![m(10, 0, None), m(10, 1, None)];
        assert_eq!(cluster_sample(&pair, 5.0, 0.1), vec![DOUBLET; 2]);
    }

    #[test]
    fn failed_gap_no_vaf_is_other() {
        // 11 -> 100 exceeds the cutoff inside one 10 kb chain; upstream drops
        // the whole event in no-VAF mode, we label every member OTHER
        assert_eq!(cluster_sample(&snps(&[10, 11, 100, 101]), 20.0, 0.1), vec![OTHER; 4]);
    }

    #[test]
    fn singleton_event_is_other() {
        // clustered by IMD but beyond the 10 kb chain limit from the event
        // before it -> upstream skips len-1 events, we label OTHER
        assert_eq!(
            cluster_sample(&snps(&[1, 2, 20_000]), 25_000.0, 0.1),
            vec![DOUBLET, DOUBLET, OTHER]
        );
    }

    #[test]
    fn vaf_failure_resplits_greedily() {
        // 0.5 -> 0.9 breaks VAF consistency; {10, 12} stay a line, {14} is
        // left over and becomes OTHER
        let muts = [m(10, 0, Some(0.5)), m(12, 0, Some(0.5)), m(14, 0, Some(0.9))];
        assert_eq!(cluster_sample(&muts, 5.0, 0.1), vec![OMIKLI, OMIKLI, OTHER]);
        // a failed gap re-splits too (VAF mode) instead of going OTHER
        let muts = [
            m(10, 0, Some(0.5)),
            m(11, 0, Some(0.5)),
            m(100, 0, Some(0.5)),
            m(101, 0, Some(0.5)),
        ];
        assert_eq!(cluster_sample(&muts, 20.0, 0.1), vec![DOUBLET; 4]);
        // a failure-free event with one line gap classifies with the outer
        // tree: distancesLine == 1, zeroDistances == 1 -> OMIKLI
        let muts = [
            m(10, 0, Some(0.5)),
            m(11, 0, Some(0.5)),
            m(100, 0, Some(0.5)),
            m(101, 0, Some(0.5)),
        ];
        assert_eq!(cluster_sample(&muts, 100.0, 0.1), vec![OMIKLI; 4]);
        // a failed gap splits the event into two consistent runs
        let muts = [
            m(10, 0, Some(0.5)),
            m(11, 0, Some(0.5)),
            m(12, 0, Some(0.5)),
            m(200, 0, Some(0.5)),
            m(201, 0, Some(0.5)),
        ];
        assert_eq!(
            cluster_sample(&muts, 20.0, 0.1),
            vec![MBS, MBS, MBS, DOUBLET, DOUBLET]
        );
    }

    #[test]
    fn vaf_rounding_matches_python_round() {
        // |0.2 - 0.1| = 0.10000000000000003 in f64; round(..., 4) == 0.1,
        // which is NOT > vaf_cut, so this pair is consistent -> DOUBLET
        let muts = [m(10, 0, Some(0.1)), m(11, 0, Some(0.2))];
        assert_eq!(cluster_sample(&muts, 5.0, 0.1), vec![DOUBLET; 2]);
    }
}
```

Run: `pixi run -e lint cargo test --no-default-features --features conversion cluster::classify`
Expected: compiles (with stubs) and the tests FAIL.

- [ ] **Step 2: Implement `src/cluster/mod.rs`.**

```rust
//! SigProfilerClusters' per-mutation subclassification, ported.
//!
//! Faithful port of `SigProfilerClusters.classifyFunctions.
//! findClustersOfClusters` with `correction=False` and of
//! `findClustersOfClusters_noVAF`. The decision tree, the sentinels, and the
//! deliberate deviations are documented in
//! `docs/superpowers/specs/2026-09-19-svar2-cluster-labels-design.md`.

pub mod classify;
pub mod write;

/// Storage schema version stamped into `meta.json` as `cluster_version`.
pub const CLUSTER_VERSION: u32 = 1;

/// Wire codebook for the `cluster_class` FORMAT field (pinned again by tests
/// in Rust and Python).
pub use classify::{
    DOUBLET, KATAEGIS, MBS, NONCLUSTERED, NOT_ANNOTATED, OMIKLI, OTHER,
};
```

- [ ] **Step 3: Implement `src/cluster/classify.rs` (replace the stubs; keep the tests).**

```rust
//! Pure classifier: one sample's mutations in, one label per mutation out.

/// Codebook (mirrored by `python/genoray/_svar2_clusters.py`).
pub const NONCLUSTERED: u8 = 0;
pub const DOUBLET: u8 = 1;
pub const MBS: u8 = 2;
pub const OMIKLI: u8 = 3;
pub const KATAEGIS: u8 = 4;
pub const OTHER: u8 = 5;
pub const NOT_ANNOTATED: u8 = 255;

/// A gap of at least this many bases starts a new event (upstream's
/// hard-coded `cutoff = 10001`, `CF:761`).
pub const GROUP_GAP: u32 = 10_001;

/// Upstream's IMD sentinel for a chromosome carrying one mutation
/// (`SigProfilerClusters.py:470-485`).
pub const SINGLE_MUTATION_IMD: f64 = 1_000_000.0;

/// One mutation of one sample. `alt` is a 2-bit code (`decode_snp_2bit`);
/// `vaf` is `Some(-1.5)` for a missing value in VAF mode (upstream's fill
/// sentinel) and `None` in no-VAF mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mutation {
    pub pos: u32,
    pub alt: u8,
    pub vaf: Option<f64>,
}

/// Step 2: for each mutation, the minimum distance to its neighbour(s). The
/// ends look only inward; a lone mutation gets [`SINGLE_MUTATION_IMD`].
pub fn imds(muts: &[Mutation]) -> Vec<f64> {
    match muts.len() {
        0 => Vec::new(),
        1 => vec![SINGLE_MUTATION_IMD],
        n => (0..n)
            .map(|i| {
                let up = if i == 0 {
                    f64::INFINITY
                } else {
                    (muts[i].pos - muts[i - 1].pos) as f64
                };
                let down = if i + 1 == n {
                    f64::INFINITY
                } else {
                    (muts[i + 1].pos - muts[i].pos) as f64
                };
                up.min(down)
            })
            .collect(),
    }
}

/// Step 3 + Step 4: drop mutations whose IMD exceeds `cutoff` (upstream's
/// pre-filter, already applied to its clustered input file), then group the
/// survivors into maximal runs where each consecutive gap is `< GROUP_GAP`.
/// The returned indices point into `muts`.
pub fn clustered_events(muts: &[Mutation], imd: &[f64], cutoff: f64) -> Vec<Vec<usize>> {
    debug_assert_eq!(muts.len(), imd.len());
    let mut events: Vec<Vec<usize>> = Vec::new();
    for (i, &d) in imd.iter().enumerate() {
        if d > cutoff {
            continue;
        }
        let extends = events
            .last()
            .is_some_and(|ev| muts[i].pos - muts[*ev.last().unwrap()].pos < GROUP_GAP);
        if extends {
            events.last_mut().unwrap().push(i);
        } else {
            events.push(vec![i]);
        }
    }
    events
}

/// Upstream's adjacent-pair counts (`CF:1035-1047`, `correction=False`).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct PairCounts {
    /// `1 < d <= cutoff` (upstream `distancesLine`)
    line: usize,
    /// `d > cutoff` (upstream `distancesFailed`)
    failed: usize,
    /// `d > 1` (upstream `zeroDistances`)
    non_adjacent: usize,
    /// `round(|dvaf|, 4) > vaf_cut`; only pairs where both values exist
    /// (upstream `vafs`)
    vaf_bad: usize,
}

fn pair_counts(muts: &[Mutation], ev: &[usize], cutoff: f64, vaf_cut: f64) -> PairCounts {
    let mut c = PairCounts::default();
    for w in ev.windows(2) {
        let (a, b) = (muts[w[0]], muts[w[1]]);
        let d = (b.pos - a.pos) as f64;
        if d > 1.0 {
            c.non_adjacent += 1;
            if d <= cutoff {
                c.line += 1;
            }
        }
        if d > cutoff {
            c.failed += 1;
        }
        if let (Some(x), Some(y)) = (a.vaf, b.vaf) {
            if round4((y - x).abs()) > vaf_cut {
                c.vaf_bad += 1;
            }
        }
    }
    c
}

/// Python's `round(x, 4)` (banker's rounding; `round_ties_even` matches).
fn round4(x: f64) -> f64 {
    (x * 10_000.0).round_ties_even() / 10_000.0
}

/// The Step 6 decision tree for an event with no failures and no VAF
/// inconsistency (`CF:1184-1210`).
fn classify_consistent(muts: &[Mutation], ev: &[usize], cutoff: f64, vaf_cut: f64) -> u8 {
    let c = pair_counts(muts, ev, cutoff, vaf_cut);
    debug_assert_eq!(c.failed, 0, "classify_consistent on a failed event");
    debug_assert_eq!(c.vaf_bad, 0, "classify_consistent on a VAF-broken event");
    if c.line > 1 {
        if ev.len() >= 4 { KATAEGIS } else { OMIKLI }
    } else if c.non_adjacent == 0 {
        if ev.len() == 2 { DOUBLET } else { MBS }
    } else {
        OMIKLI
    }
}

/// Label one event in place (Steps 5-8). Upstream skips len-1 events; we
/// label them `OTHER` so every mutation has a value. On a VAF or gap failure
/// the event is re-split greedily (ClassIII, VAF mode only, `CF:1270-1510`):
/// take the first remaining mutation, append every later one whose RAW
/// `|dvaf| < vaf_cut` and whose gap to the last KEPT mutation is `<= cutoff`,
/// classify that run with the same tree, repeat on the leftovers. Runs of one
/// and leftovers of one become `OTHER`.
fn label_event(
    muts: &[Mutation],
    ev: &[usize],
    cutoff: f64,
    vaf_cut: f64,
    labels: &mut [u8],
) {
    if ev.len() == 1 {
        labels[ev[0]] = OTHER;
        return;
    }
    let c = pair_counts(muts, ev, cutoff, vaf_cut);
    if c.failed == 0 && c.vaf_bad == 0 {
        let lab = classify_consistent(muts, ev, cutoff, vaf_cut);
        for &i in ev {
            labels[i] = lab;
        }
        return;
    }
    if muts[ev[0]].vaf.is_none() {
        // no-VAF mode has no re-split; upstream drops the event entirely
        for &i in ev {
            labels[i] = OTHER;
        }
        return;
    }
    let mut remaining: Vec<usize> = ev.to_vec();
    while remaining.len() > 1 {
        let mut run = vec![remaining[0]];
        let mut last = remaining[0];
        for &m in &remaining[1..] {
            let d = (muts[m].pos - muts[last].pos) as f64;
            let dv = (muts[m].vaf.unwrap() - muts[last].vaf.unwrap()).abs();
            if dv < vaf_cut && d <= cutoff {
                run.push(m);
                last = m;
            }
        }
        if run.len() > 1 {
            let lab = classify_consistent(muts, &run, cutoff, vaf_cut);
            for &i in &run {
                labels[i] = lab;
            }
        } else {
            labels[run[0]] = OTHER;
        }
        remaining.retain(|i| !run.contains(i));
    }
    if let [i] = remaining[..] {
        labels[i] = OTHER;
    }
}

/// Steps 2-8 for one sample. `muts` must be sorted by `(pos, alt)` with
/// `(pos, alt)` unique. Returns one label per mutation, `NONCLUSTERED` where
/// no event covers it.
pub fn cluster_sample(muts: &[Mutation], cutoff: f64, vaf_cut: f64) -> Vec<u8> {
    let mut labels = vec![NONCLUSTERED; muts.len()];
    if muts.is_empty() {
        return labels;
    }
    let imd = imds(muts);
    for ev in clustered_events(muts, &imd, cutoff) {
        label_event(muts, &ev, cutoff, vaf_cut, &mut labels);
    }
    labels
}

#[cfg(test)]
mod tests {
    // ... the Step 1 test module, unchanged ...
}
```

- [ ] **Step 4: Register the module in `src/lib.rs`.** Add `pub mod cluster;` (and, in Task 3, `pub mod py_cluster;`) beside the other `pub mod` lines, outside any feature gate.

- [ ] **Step 5: Run the tests.**

Run: `pixi run -e lint cargo test --no-default-features --features conversion cluster::classify`
Expected: all tests pass.

Run: `pixi run -e lint check-core`
Expected: clean (no `conversion`-gated imports).

- [ ] **Step 6: Commit.**

```bash
pixi run -e lint cargo fmt
pixi run -e lint lint
git add src/cluster/mod.rs src/cluster/classify.rs src/lib.rs
git commit -m "feat(clusters): port SigProfilerClusters IMD grouping and subclassifier"
```

---

### Task 2: Rust writer (gather, classify, write four streams)

**Files:**
- Create: `src/cluster/write.rs`

- [ ] **Step 1: Write the file with its unit tests.** No separate stub step here: the pure helper tests below are the first things to run, and they fail only if the helpers are wrong.

```rust
//! Gather a contig's mutations, classify per sample, and write the four
//! `cluster_class` value streams.

use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};

use memmap2::MmapMut;
use rayon::prelude::*;

use crate::field::StorageDtype;
use crate::layout::{ContigPaths, FieldSub};
use crate::query::field::{FieldValue, FieldView};
use crate::query::reader::ContigReader;
use crate::query::sidecar::as_bytes;
use crate::svar2_codec;

use super::classify::{Mutation, NONCLUSTERED, NOT_ANNOTATED, cluster_sample};

/// Name of the FORMAT field this module owns.
pub const CLUSTER_CLASS: &str = "cluster_class";

/// Everything that can go wrong while annotating one contig.
#[derive(Debug, thiserror::Error)]
pub enum ClusterError {
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error("{0}")]
    Invalid(String),
}

/// Where a mutation came from, so its label can be written back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Origin {
    /// Absolute index into `vk_snp`'s call stream.
    VkCall(usize),
    /// Dense SNP column; the label lands at `col * n_samples + sample`.
    DenseCol(usize),
}

struct SampleMutations {
    muts: Vec<Mutation>,
    origins: Vec<Origin>,
}

fn field_f64(v: FieldValue) -> f64 {
    match v {
        FieldValue::Bool(b) => u8::from(b) as f64,
        FieldValue::I8(x) => x as f64,
        FieldValue::U8(x) => x as f64,
        FieldValue::I16(x) => x as f64,
        FieldValue::U16(x) => x as f64,
        FieldValue::I32(x) => x as f64,
        FieldValue::U32(x) => x as f64,
        FieldValue::F16(x) => x.to_f64(),
        FieldValue::F32(x) => x as f64,
    }
}

/// A missing float (NaN) becomes upstream's `-1.5` VAF fill sentinel.
fn map_missing(v: f64) -> f64 {
    if v.is_nan() { -1.5 } else { v }
}

/// All of `sample`'s SNPs on this contig, from both the var_key and dense
/// streams, sorted by `(pos, alt)` and deduped (a dense variant carried by
/// both haplotypes is ONE mutation; a var_key/dense overlap keeps var_key,
/// matching mutcat's precedence).
fn gather_sample(
    reader: &ContigReader,
    snp_positions: &[u32],
    dense_positions: &[u32],
    sample: usize,
    vaf_vk: Option<&FieldView>,
    vaf_dense: Option<&FieldView>,
) -> SampleMutations {
    let mut muts: Vec<Mutation> = Vec::new();
    let mut origins: Vec<Origin> = Vec::new();
    for p in 0..reader.ploidy {
        let col = sample * reader.ploidy + p;
        for call in reader.vk_snp.column(col) {
            let pos = snp_positions[call];
            let key = svar2_codec::unpack_snp_key_at(as_bytes(&reader.vk_snp.keys), call);
            let alt = svar2_codec::decode_snp_2bit(key);
            let vaf = vaf_vk.map(|v| map_missing(field_f64(v.value_at(call))));
            muts.push(Mutation { pos, alt, vaf });
            origins.push(Origin::VkCall(call));
        }
        if let Some(dense) = &reader.dense_snp {
            let keys = as_bytes(&dense.keys);
            dense.for_each_carried(col, |dcol| {
                let pos = dense_positions[dcol];
                let alt = svar2_codec::decode_snp_2bit(svar2_codec::unpack_snp_key_at(keys, dcol));
                let vaf = vaf_dense.map(|v| map_missing(field_f64(v.format_at(dcol, sample))));
                muts.push(Mutation { pos, alt, vaf });
                origins.push(Origin::DenseCol(dcol));
            });
        }
    }
    let mut order: Vec<usize> = (0..muts.len()).collect();
    order.sort_by_key(|&i| {
        let rank = matches!(origins[i], Origin::DenseCol(_)) as u8;
        (muts[i].pos, muts[i].alt, rank)
    });
    let mut out = SampleMutations {
        muts: Vec::with_capacity(muts.len()),
        origins: Vec::with_capacity(muts.len()),
    };
    for i in order {
        let m = muts[i];
        let duplicate = out
            .muts
            .last()
            .is_some_and(|last| last.pos == m.pos && last.alt == m.alt);
        if !duplicate {
            out.muts.push(m);
            out.origins.push(origins[i]);
        }
    }
    out
}

/// Classify every sample of `reader` and write all four `cluster_class`
/// `values.bin` files for `paths`. `cutoffs` has one entry per cohort sample.
/// `vaf` is `(field name, stored dtype)`; `None` runs the no-VAF port.
pub fn annotate_contig(
    reader: &ContigReader,
    paths: &ContigPaths,
    cutoffs: &[f64],
    vaf: Option<(&str, StorageDtype)>,
    vaf_cut: f64,
) -> Result<(), ClusterError> {
    let n_samples = reader.n_samples;
    if cutoffs.len() != n_samples {
        return Err(ClusterError::Invalid(format!(
            "imd_cutoff has {} values for {n_samples} samples",
            cutoffs.len()
        )));
    }
    let snp_positions = reader.vk_snp.positions();
    let dense_positions: &[u32] = reader
        .dense_snp
        .as_ref()
        .map(|d| d.positions())
        .unwrap_or(&[]);
    let vk_calls = reader.vk_snp.offsets.last().copied().unwrap_or(0) as usize;

    let (vaf_vk, vaf_dense) = match vaf {
        Some((name, dtype)) => {
            let vk = FieldView::open(paths, "format", name, FieldSub::VkSnp, dtype, n_samples)?;
            let dense =
                FieldView::open(paths, "format", name, FieldSub::DenseSnp, dtype, n_samples)?;
            let dense_elems = reader
                .dense_snp
                .as_ref()
                .map(|d| d.n_dense_variants * n_samples)
                .unwrap_or(0);
            if vk.len() != vk_calls || dense.len() != dense_elems {
                return Err(ClusterError::Invalid(format!(
                    "VAF field {name:?} has {} var_key and {} dense values, expected \
                     {vk_calls} and {dense_elems}",
                    vk.len(),
                    dense.len()
                )));
            }
            (Some(vk), Some(dense))
        }
        None => (None, None),
    };

    // Gather + classify in parallel; the mmap writes stay single-threaded.
    // If `ContigReader` turns out not to be `Sync` (the long-allele LUT reader
    // is the only suspect), drop `.into_par_iter()` to a plain `(0..n_samples)`
    // map; correctness is identical.
    let classified: Vec<(SampleMutations, Vec<u8>)> = (0..n_samples)
        .into_par_iter()
        .map(|s| {
            let g = gather_sample(
                reader,
                snp_positions,
                dense_positions,
                s,
                vaf_vk.as_ref(),
                vaf_dense.as_ref(),
            );
            let labels = cluster_sample(&g.muts, cutoffs[s], vaf_cut);
            (g, labels)
        })
        .collect();

    let seen_vk: usize = classified
        .iter()
        .map(|(g, _)| {
            g.origins
                .iter()
                .filter(|o| matches!(o, Origin::VkCall(_)))
                .count()
        })
        .sum();
    if seen_vk != vk_calls {
        return Err(ClusterError::Invalid(format!(
            "gathered {seen_vk} var_key SNP calls, the stream has {vk_calls}"
        )));
    }

    let vk_indel_calls = reader.vk_indel.offsets.last().copied().unwrap_or(0) as usize;
    let dense_snp_elems = reader
        .dense_snp
        .as_ref()
        .map(|d| d.n_dense_variants * n_samples)
        .unwrap_or(0);
    let dense_indel_elems = reader
        .dense_indel
        .as_ref()
        .map(|d| d.n_dense_variants * n_samples)
        .unwrap_or(0);

    write_values(
        &paths.field_values("format", CLUSTER_CLASS, FieldSub::VkSnp),
        vk_calls,
        |buf| {
            buf.fill(NONCLUSTERED);
            for (g, labels) in &classified {
                for (i, origin) in g.origins.iter().enumerate() {
                    if let Origin::VkCall(call) = *origin {
                        buf[call] = labels[i];
                    }
                }
            }
        },
    )?;
    write_values(
        &paths.field_values("format", CLUSTER_CLASS, FieldSub::VkIndel),
        vk_indel_calls,
        |buf| buf.fill(NOT_ANNOTATED),
    )?;
    write_values(
        &paths.field_values("format", CLUSTER_CLASS, FieldSub::DenseSnp),
        dense_snp_elems,
        |buf| {
            buf.fill(NOT_ANNOTATED);
            for (sample, (g, labels)) in classified.iter().enumerate() {
                for (i, origin) in g.origins.iter().enumerate() {
                    if let Origin::DenseCol(col) = *origin {
                        buf[col * n_samples + sample] = labels[i];
                    }
                }
            }
        },
    )?;
    write_values(
        &paths.field_values("format", CLUSTER_CLASS, FieldSub::DenseIndel),
        dense_indel_elems,
        |buf| buf.fill(NOT_ANNOTATED),
    )?;
    Ok(())
}

/// Write 255 into all four streams (for contigs outside `contigs=` scope), so
/// selecting `cluster_class` and decoding an unannotated contig is coherent.
pub fn fill_contig(reader: &ContigReader, paths: &ContigPaths) -> Result<(), ClusterError> {
    let n_samples = reader.n_samples;
    let streams = [
        (
            FieldSub::VkSnp,
            reader.vk_snp.offsets.last().copied().unwrap_or(0) as usize,
        ),
        (
            FieldSub::VkIndel,
            reader.vk_indel.offsets.last().copied().unwrap_or(0) as usize,
        ),
        (
            FieldSub::DenseSnp,
            reader
                .dense_snp
                .as_ref()
                .map(|d| d.n_dense_variants * n_samples)
                .unwrap_or(0),
        ),
        (
            FieldSub::DenseIndel,
            reader
                .dense_indel
                .as_ref()
                .map(|d| d.n_dense_variants * n_samples)
                .unwrap_or(0),
        ),
    ];
    for (sub, len) in streams {
        write_values(
            &paths.field_values("format", CLUSTER_CLASS, sub),
            len,
            |buf| buf.fill(NOT_ANNOTATED),
        )?;
    }
    Ok(())
}

/// Create `path.tmp` at `len` bytes, hand the mmap to `fill`, then fsync +
/// rename (the `mutcat::sidecar` pattern). `len == 0` writes an empty file.
fn write_values(path: &Path, len: usize, fill: impl FnOnce(&mut [u8])) -> io::Result<()> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    let tmp = temp_path(path);
    remove_if_exists(&tmp)?;
    if len == 0 {
        File::create(&tmp)?;
        return fs::rename(&tmp, path);
    }
    let file = File::create(&tmp)?;
    file.set_len(len as u64)?;
    // SAFETY: `tmp` is private to this call; nothing else maps or writes it
    // between `map_mut` and the rename below.
    let mut mm = unsafe { MmapMut::map_mut(&file)? };
    fill(&mut mm[..]);
    mm.flush()?;
    drop(mm);
    file.sync_all()?;
    fs::rename(&tmp, path)
}

/// `path` with `.tmp` appended to its file name, so staging shares the
/// destination's directory and filesystem.
fn temp_path(path: &Path) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(".tmp");
    path.with_file_name(name)
}

fn remove_if_exists(path: &Path) -> io::Result<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_values_writes_len_and_replaces() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested").join("values.bin");
        write_values(&path, 4, |buf| buf.copy_from_slice(&[1, 2, 3, 4])).unwrap();
        assert_eq!(fs::read(&path).unwrap(), vec![1, 2, 3, 4]);
        // a second call replaces the contents and leaves no temp behind
        write_values(&path, 2, |buf| buf.fill(9)).unwrap();
        assert_eq!(fs::read(&path).unwrap(), vec![9, 9]);
        assert!(!temp_path(&path).exists());
    }

    #[test]
    fn write_values_handles_zero_len() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("values.bin");
        write_values(&path, 0, |_| unreachable!()).unwrap();
        assert!(path.is_file());
        assert_eq!(fs::metadata(&path).unwrap().len(), 0);
    }

    #[test]
    fn failed_write_leaves_destination_untouched() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("values.bin");
        write_values(&path, 1, |buf| buf[0] = 5).unwrap();
        // Block the temp path with a directory: the staged write must fail
        // before the committed destination is replaced.
        fs::create_dir(temp_path(&path)).unwrap();
        assert!(write_values(&path, 1, |buf| buf[0] = 9).is_err());
        assert_eq!(fs::read(&path).unwrap(), vec![5]);
    }

    #[test]
    fn missing_vaf_becomes_the_upstream_sentinel() {
        assert_eq!(map_missing(f64::NAN), -1.5);
        assert_eq!(map_missing(0.4), 0.4);
    }
}
```

- [ ] **Step 2: Run the writer unit tests.**

Run: `pixi run -e lint cargo test --no-default-features --features conversion cluster::write`
Expected: all pass.

- [ ] **Step 3: Commit.**

```bash
pixi run -e lint cargo fmt
git add src/cluster/write.rs
git commit -m "feat(clusters): write cluster_class sidecar streams from vk+dense mutations"
```

---

### Task 3: pyo3 binding on `PyContigReader`

**Files:**
- Create: `src/py_cluster.rs`
- Modify: `src/lib.rs` (add `pub mod py_cluster;`)

- [ ] **Step 1: Create `src/py_cluster.rs`.**

```rust
//! Python-facing cluster labels: annotate + fill on `PyContigReader`.

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use crate::cluster::write::{ClusterError, annotate_contig, fill_contig};
use crate::field::StorageDtype;
use crate::layout::ContigPaths;
use crate::py_query::PyContigReader;

fn cluster_err(context: &str, e: ClusterError) -> PyErr {
    match e {
        ClusterError::Invalid(msg) => {
            pyo3::exceptions::PyValueError::new_err(format!("{context}: {msg}"))
        }
        ClusterError::Io(err) => {
            pyo3::exceptions::PyIOError::new_err(format!("{context}: {err}"))
        }
    }
}

#[pymethods]
impl PyContigReader {
    /// Classify this contig's samples against `cutoffs` (one per cohort
    /// sample) and write all four `cluster_class` value streams.
    #[pyo3(signature = (base_out_dir, chrom, cutoffs, vaf=None, vaf_cut=0.1))]
    fn annotate_clusters(
        &self,
        base_out_dir: &str,
        chrom: &str,
        cutoffs: PyReadonlyArray1<f64>,
        vaf: Option<(String, String)>,
        vaf_cut: f64,
    ) -> PyResult<()> {
        let paths = ContigPaths::new(base_out_dir, chrom);
        let cutoffs = cutoffs.as_slice()?;
        let vaf = match &vaf {
            Some((name, dtype)) => {
                let dtype = StorageDtype::from_meta_str(dtype).ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "unknown VAF storage dtype {dtype:?}"
                    ))
                })?;
                Some((name.as_str(), dtype))
            }
            None => None,
        };
        annotate_contig(&self.inner, &paths, cutoffs, vaf, vaf_cut)
            .map_err(|e| cluster_err(&format!("annotate clusters {chrom}"), e))
    }

    /// Write 255-filled `cluster_class` streams for this contig (out of scope).
    fn fill_cluster_labels(&self, base_out_dir: &str, chrom: &str) -> PyResult<()> {
        let paths = ContigPaths::new(base_out_dir, chrom);
        fill_contig(&self.inner, &paths)
            .map_err(|e| cluster_err(&format!("fill cluster labels {chrom}"), e))
    }
}
```

- [ ] **Step 2: Register in `src/lib.rs`.** Add `pub mod py_cluster;` beside `pub mod py_mutcat;` etc. (same feature gating as the other `py_*` modules — `extension-module` is implied by `#[pyclass]` usage, and pyo3's `multiple-pymethods` is already on).

- [ ] **Step 3: Build and check.**

Run: `pixi run -e lint cargo check --no-default-features --features conversion`
Expected: clean.

Run: `pixi run -e lint check-core`
Expected: clean.

- [ ] **Step 4: Commit.**

```bash
pixi run -e lint cargo fmt
git add src/py_cluster.rs src/lib.rs
git commit -m "feat(clusters): expose PyContigReader.annotate_clusters and fill_cluster_labels"
```

---

### Task 4: Python mixin + `SparseVar2` wiring

**Files:**
- Create: `python/genoray/_svar2_clusters.py`
- Modify: `python/genoray/_svar2.py` (import + base class)

- [ ] **Step 1: Write `python/genoray/_svar2_clusters.py`.**

```python
"""SVAR2 cluster-label surface: ``annotate_clusters`` on ``SparseVar2``.

Ports SigProfilerClusters' ``findClustersOfClusters`` (VAF mode) and
``findClustersOfClusters_noVAF`` (no-VAF mode) with ``correction=False`` into
a post-hoc annotation that writes the ``cluster_class`` FORMAT field. The
algorithm and every deliberate deviation are documented in
``docs/superpowers/specs/2026-09-19-svar2-cluster-labels-design.md``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

# Wire codebook; the Rust classifier pins the same values in
# ``src/cluster/classify.rs`` and the tests pin both.
NONCLUSTERED = 0
DOUBLET = 1
MBS = 2
OMIKLI = 3
KATAEGIS = 4
OTHER = 5
NOT_ANNOTATED = 255

#: Storage schema version stamped as ``cluster_version`` in ``meta.json``.
CLUSTER_VERSION = 1

if TYPE_CHECKING:
    pass


class _ClustersMixin:
    """SigProfilerClusters subclassification over a finished SVAR2 store.

    Provided by the concrete ``SparseVar2`` host class (see
    ``SparseVar2.__init__``); declared here so the mixin's use of them
    type-checks in isolation.
    """

    path: Path
    contigs: list[str]
    available_samples: list[str]
    available_fields: dict[str, Any]
    _readers: dict[str, Any]

    def annotate_clusters(
        self,
        *,
        imd_cutoff: "float | Mapping[str, float]",
        vaf_field: str | None = None,
        vaf_cut: float = 0.1,
        contigs: "Sequence[str] | None" = None,
    ) -> None:
        """Subclassify clustered mutations per sample into ``cluster_class``.

        Args:
            imd_cutoff: Inter-mutational-distance cutoff(s), in bases. Either a
                single float for every sample or a mapping from sample name to
                cutoff (all samples required). Mutations whose minimum
                neighbour distance is ``<=`` their cutoff are clustered.
                Upstream derives these per sample by simulation; feed the same
                ``imds.pickle`` values for label-for-label parity.
            vaf_field: Optional FORMAT field holding per-sample VAF/CCF. When
                given, VAF-consistency participates in the decision tree and
                failed events are greedily re-split. Must be a 2- or 4-byte
                float field.
            vaf_cut: Maximum ``|delta VAF|`` for adjacent mutations to stay in
                one event (upstream default ``0.1``).
            contigs: If given, only these contigs (resolved against
                ``self.contigs``, alternate naming accepted) are annotated;
                every other contig gets 255-filled ``cluster_class`` streams
                so decoding stays coherent. ``None`` (default) annotates every
                contig in the store.

        Notes:
            Stamps ``meta.json`` with ``cluster_version``, ``cluster_contigs``,
            ``cluster_cutoff``, ``cluster_vaf_field``, ``cluster_vaf_cut``, and
            the ``cluster_class`` FORMAT entry, after every in-scope contig's
            files are in place. Re-running is unconditional.

        Raises:
            ValueError: Unknown/missing samples in a ``imd_cutoff`` mapping, a
                non-positive cutoff or ``vaf_cut``, a ``vaf_field`` that is
                missing / not FORMAT / not a 2- or 4-byte float, or a
                ``contigs=`` name absent from the store.
        """
        from genoray._svar2_fields import _META_DTYPE

        if vaf_cut <= 0:
            raise ValueError(f"vaf_cut must be > 0, got {vaf_cut!r}")

        if isinstance(imd_cutoff, Mapping):
            unknown = [s for s in imd_cutoff if s not in self.available_samples]
            if unknown:
                raise ValueError(
                    f"imd_cutoff names unknown samples: {unknown}; "
                    f"available samples: {self.available_samples}"
                )
            missing = [s for s in self.available_samples if s not in imd_cutoff]
            if missing:
                raise ValueError(f"imd_cutoff is missing samples: {missing}")
            cutoffs = np.array(
                [float(imd_cutoff[s]) for s in self.available_samples], dtype=np.float64
            )
        else:
            cutoffs = np.full(len(self.available_samples), float(imd_cutoff), dtype=np.float64)
        if not np.all(cutoffs > 0):
            raise ValueError(f"imd_cutoff values must be > 0, got {cutoffs.tolist()}")

        vaf: tuple[str, str] | None = None
        if vaf_field is not None:
            field = self.available_fields.get(vaf_field)
            if field is None:
                matches = [f for f in self.available_fields.values() if f.name == vaf_field]
                field = matches[0] if len(matches) == 1 else None
            if field is None:
                raise ValueError(
                    f"vaf_field {vaf_field!r} is not in the store; available fields: "
                    f"{sorted(self.available_fields)}"
                )
            if field.category != "format":
                raise ValueError(
                    f"vaf_field {vaf_field!r} is an {field.category} field; expected FORMAT"
                )
            if field.dtype.kind != "f" or field.dtype.itemsize not in (2, 4):
                raise ValueError(
                    f"vaf_field {vaf_field!r} has dtype {field.dtype}; "
                    "expected a 2- or 4-byte float"
                )
            vaf = (field.name, _META_DTYPE[field.dtype])

        if contigs is None:
            scope = list(self.contigs)
        else:
            # _resolve_contigs raises ValueError naming the first miss.
            resolved = self._resolve_contigs(contigs)  # type: ignore[missing-attribute]
            scope = list(dict.fromkeys(resolved))
            if not scope:
                raise ValueError("contigs= resolved to no store contigs")

        for contig in scope:
            self._readers[contig].annotate_clusters(
                str(self.path), contig, cutoffs, vaf, vaf_cut
            )

        if contigs is not None:
            for contig in self.contigs:
                if contig not in scope:
                    self._readers[contig].fill_cluster_labels(str(self.path), contig)

        meta_path = self.path / "meta.json"
        meta = json.loads(meta_path.read_text())
        meta["cluster_version"] = CLUSTER_VERSION
        meta["cluster_contigs"] = scope
        meta["cluster_cutoff"] = (
            dict(imd_cutoff) if isinstance(imd_cutoff, Mapping) else float(imd_cutoff)
        )
        meta["cluster_vaf_field"] = vaf_field
        meta["cluster_vaf_cut"] = vaf_cut
        fields = [
            f
            for f in meta.get("fields") or []
            if not (f["name"] == "cluster_class" and f["category"] == "format")
        ]
        fields.append(
            {"name": "cluster_class", "category": "format", "dtype": "u8", "default": None}
        )
        meta["fields"] = fields
        tmp_path = meta_path.with_name(meta_path.name + ".tmp")
        with open(tmp_path, "w") as f:
            f.write(json.dumps(meta))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, meta_path)
```

- [ ] **Step 2: Wire the mixin into `SparseVar2`.** In `python/genoray/_svar2.py`:

```python
from genoray._svar2_clusters import _ClustersMixin
```

and change the class line to:

```python
class SparseVar2(_BatchQueryMixin, _DecodeMixin, _MutcatMixin, _ClustersMixin):
```

- [ ] **Step 3: Import smoke test.**

Run: `pixi run python -c "import genoray; from genoray import SparseVar2; print(hasattr(SparseVar2, 'annotate_clusters'))"`
Expected: `True`.

- [ ] **Step 4: Commit.**

```bash
git add python/genoray/_svar2_clusters.py python/genoray/_svar2.py
git commit -m "feat(clusters): add SparseVar2.annotate_clusters post-hoc API"
```

---

### Task 5: End-to-end Python tests

**Files:**
- Create: `tests/test_svar2_clusters.py`

Fixture design (tiny cohort, so the cost model routes the recurrent doublet DENSE and the rare SNPs/indel var_key — one store exercises all four sub-streams): 3 samples on `chr1`:

| pos | ref/alt | s0 | s1 | s2 | VAF | stream |
|---|---|---|---|---|---|---|
| 100 | C>A | 1\|0 | 1\|0 | 0\|0 | 0.5 | dense (2/3 carriers) |
| 101 | C>G | 1\|0 | 0\|1 | 0\|0 | 0.5 | dense |
| 200 | A>T | 0\|0 | 0\|0 | 1\|0 | 0.5 | var_key |
| 300 | A>AT | 0\|0 | 0\|0 | 0\|1 | 0.5 | var_key_indel |
| 5000 | G>T | 0\|0 | 0\|0 | 1\|0 | 0.5 | var_key |

With `imd_cutoff=1000`: s0/s1's 100/101 pair is a DOUBLET; s2's 200 and 5000 are lone (NONCLUSTERED); the indel is NOT_ANNOTATED.

- [ ] **Step 1: Write the test file.**

```python
"""End-to-end tests for ``SparseVar2.annotate_clusters``.

Fixture (3 samples, chr1): a doublet at 100/101 carried by s0+s1 (the cost
model routes it DENSE), rare SNPs at 200/5000 and a rare indel at 300
carried by s2 (var_key / var_key_indel). With imd_cutoff=1000 the doublet is
class 1, s2's lone SNPs are class 0, and the indel is 255.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from vcfixture import Number, Seq, Type, VcfBuilder

from genoray import SparseVar2
from genoray._svar2_clusters import (
    CLUSTER_VERSION,
    DOUBLET,
    KATAEGIS,
    MBS,
    NONCLUSTERED,
    NOT_ANNOTATED,
    OMIKLI,
    OTHER,
)
from genoray._svar2_fields import FormatField


def _fixture(tmp_path: Path) -> Path:
    doc = (
        VcfBuilder(samples=["s0", "s1", "s2"], contigs=[("chr1", None)])
        .fmt("GT")
        .fmt("VAF", Number.ONE, Type.FLOAT)
        .record(
            "chr1", 100, ref="C", alt=[Seq("A")],
            gt=["1|0", "1|0", "0|0"], VAF=[[0.5], [0.5], [0.5]],
        )
        .record(
            "chr1", 101, ref="C", alt=[Seq("G")],
            gt=["1|0", "0|1", "0|0"], VAF=[[0.5], [0.5], [0.5]],
        )
        .record(
            "chr1", 200, ref="A", alt=[Seq("T")],
            gt=["0|0", "0|0", "1|0"], VAF=[[0.5], [0.5], [0.5]],
        )
        .record(
            "chr1", 300, ref="A", alt=[Seq("AT")],
            gt=["0|0", "0|0", "0|1"], VAF=[[0.5], [0.5], [0.5]],
        )
        .record(
            "chr1", 5000, ref="G", alt=[Seq("T")],
            gt=["0|0", "0|0", "1|0"], VAF=[[0.5], [0.5], [0.5]],
        )
    )
    return doc.write(tmp_path / "clusters.vcf.gz", bgzip=True, index=True)


def _store(tmp_path: Path) -> SparseVar2:
    out = tmp_path / "clusters.svar2"
    SparseVar2.from_vcf(
        out,
        _fixture(tmp_path),
        no_reference=True,
        format_fields=[FormatField("VAF", dtype="f32")],
    )
    return SparseVar2(out)


def test_codebook_is_pinned():
    assert (NONCLUSTERED, DOUBLET, MBS, OMIKLI, KATAEGIS, OTHER, NOT_ANNOTATED) == (
        0, 1, 2, 3, 4, 5, 255
    )
    assert CLUSTER_VERSION == 1


def test_annotate_clusters_end_to_end(tmp_path: Path):
    store = _store(tmp_path)
    store.annotate_clusters(imd_cutoff=1000.0, vaf_field="VAF")

    meta = json.loads((store.path / "meta.json").read_text())
    assert meta["cluster_version"] == CLUSTER_VERSION
    assert meta["cluster_contigs"] == ["chr1"]
    assert meta["cluster_cutoff"] == 1000.0
    assert meta["cluster_vaf_field"] == "VAF"
    assert meta["cluster_vaf_cut"] == 0.1
    assert {
        "name": "cluster_class",
        "category": "format",
        "dtype": "u8",
        "default": None,
    } in meta["fields"]

    sv = SparseVar2(store.path).with_fields(["cluster_class"])
    rag = sv.decode("chr1", [(0, 1_000_000)])
    # carrier-only flat (R, S, P) order: s0h0, s0h1, s1h0, s1h1, s2h0, s2h1
    assert rag["pos"].lengths.reshape(-1).tolist() == [2, 0, 1, 1, 2, 1]
    labels = np.asarray(rag["cluster_class"].data)
    np.testing.assert_array_equal(
        labels,
        np.array(
            [DOUBLET, DOUBLET, DOUBLET, DOUBLET, NONCLUSTERED, NONCLUSTERED, NOT_ANNOTATED],
            dtype=np.uint8,
        ),
    )


def test_dense_stream_is_dense_row_major(tmp_path: Path):
    store = _store(tmp_path)
    store.annotate_clusters(imd_cutoff=1000.0, vaf_field="VAF")

    dense_dir = store.path / "chr1" / "dense" / "snp"
    positions = np.frombuffer((dense_dir / "positions.bin").read_bytes(), dtype=np.uint32)
    assert len(positions) >= 2, "fixture must route the recurrent doublet dense"
    values = np.frombuffer(
        (
            store.path
            / "chr1"
            / "fields"
            / "format"
            / "cluster_class"
            / "dense_snp"
            / "values.bin"
        ).read_bytes(),
        dtype=np.uint8,
    )
    expected = np.full(len(positions) * 3, NOT_ANNOTATED, dtype=np.uint8)
    for row, pos in enumerate(positions):
        if pos in (100, 101):
            expected[row * 3 + 0] = DOUBLET
            expected[row * 3 + 1] = DOUBLET
    np.testing.assert_array_equal(values, expected)


def test_out_of_scope_contigs_are_255_filled(tmp_path: Path):
    # second contig, one lone SNP, untouched by contigs=["chr1"]
    doc = (
        VcfBuilder(samples=["s0", "s1", "s2"], contigs=[("chr1", None), ("chr2", None)])
        .fmt("GT")
        .fmt("VAF", Number.ONE, Type.FLOAT)
        .record(
            "chr1", 100, ref="C", alt=[Seq("A")],
            gt=["1|0", "1|0", "0|0"], VAF=[[0.5], [0.5], [0.5]],
        )
        .record(
            "chr1", 101, ref="C", alt=[Seq("G")],
            gt=["1|0", "0|1", "0|0"], VAF=[[0.5], [0.5], [0.5]],
        )
        .record(
            "chr2", 50, ref="A", alt=[Seq("T")],
            gt=["1|0", "0|0", "0|0"], VAF=[[0.5], [0.5], [0.5]],
        )
    )
    out = tmp_path / "two.svar2"
    SparseVar2.from_vcf(
        out,
        doc.write(tmp_path / "two.vcf.gz", bgzip=True, index=True),
        no_reference=True,
        format_fields=[FormatField("VAF", dtype="f32")],
    )
    store = SparseVar2(out)
    store.annotate_clusters(imd_cutoff=1000.0, vaf_field="VAF", contigs=["chr1"])

    meta = json.loads((out / "meta.json").read_text())
    assert meta["cluster_contigs"] == ["chr1"]
    chr2_values = out / "chr2" / "fields" / "format" / "cluster_class"
    assert (chr2_values / "var_key_snp" / "values.bin").read_bytes() == bytes([NOT_ANNOTATED])
    assert (chr2_values / "var_key_indel" / "values.bin").stat().st_size == 0
    sv = SparseVar2(out).with_fields(["cluster_class"])
    rag = sv.decode("chr2", [(0, 1_000_000)])
    assert np.asarray(rag["cluster_class"].data).tolist() == [NOT_ANNOTATED]


def test_validation_errors(tmp_path: Path):
    store = _store(tmp_path)
    with pytest.raises(ValueError, match="unknown samples"):
        store.annotate_clusters(imd_cutoff={"s0": 10.0, "nope": 10.0})
    with pytest.raises(ValueError, match="missing samples"):
        store.annotate_clusters(imd_cutoff={"s0": 10.0})
    with pytest.raises(ValueError, match="must be > 0"):
        store.annotate_clusters(imd_cutoff=0.0)
    with pytest.raises(ValueError, match="vaf_cut"):
        store.annotate_clusters(imd_cutoff=10.0, vaf_field="VAF", vaf_cut=0.0)
    with pytest.raises(ValueError, match="not in the store"):
        store.annotate_clusters(imd_cutoff=10.0, vaf_field="NOPE")
    with pytest.raises(ValueError, match="not found in store"):
        store.annotate_clusters(imd_cutoff=10.0, contigs=["chr9"])


def test_failed_write_leaves_meta_unstamped(tmp_path: Path):
    store = _store(tmp_path)
    before = (store.path / "meta.json").read_bytes()
    # Block the var_key_snp staging path with a directory so the Rust write
    # fails before anything is renamed into place.
    blocked = (
        store.path / "chr1" / "fields" / "format" / "cluster_class" / "var_key_snp"
    )
    blocked.mkdir(parents=True)
    (blocked / "values.bin.tmp").mkdir()
    with pytest.raises(OSError):
        store.annotate_clusters(imd_cutoff=1000.0, vaf_field="VAF")
    assert (store.path / "meta.json").read_bytes() == before
    assert not (blocked / "values.bin").exists()


def test_rerun_is_idempotent(tmp_path: Path):
    store = _store(tmp_path)
    store.annotate_clusters(imd_cutoff=1000.0, vaf_field="VAF")
    first = (
        store.path
        / "chr1"
        / "fields"
        / "format"
        / "cluster_class"
        / "dense_snp"
        / "values.bin"
    ).read_bytes()
    store.annotate_clusters(imd_cutoff=1000.0, vaf_field="VAF")
    second = (
        store.path
        / "chr1"
        / "fields"
        / "format"
        / "cluster_class"
        / "dense_snp"
        / "values.bin"
    ).read_bytes()
    assert first == second
```

- [ ] **Step 2: Run.**

Run: `pixi run pytest tests/test_svar2_clusters.py -v`
Expected: all pass. If the Rust extension is stale, run `pixi run maturin develop` and re-run.

- [ ] **Step 3: Commit.**

```bash
git add tests/test_svar2_clusters.py
git commit -m "test(clusters): end-to-end annotate_clusters coverage"
```

---

### Task 6: Upstream parity oracle

**Files:**
- Create: `tests/test_clusters_calibration.py`
- Modify: `pixi.toml` (`[feature.sigprofiler.pypi-dependencies]`)

- [ ] **Step 1: Add the dependency.** In `pixi.toml`, under `[feature.sigprofiler.pypi-dependencies]`, add:

```toml
sigprofilerclusters = ">=1.2, <2"
```

Then run `pixi run -e sigprofiler python -c "import SigProfilerClusters.classifyFunctions"` and expect no error (first run resolves and installs the env). If resolution fails on `SigProfilerSimulator`, install it into the same feature (`sigprofilersimulator = ">=1.2"`); `classifyFunctions` does not import it, but its `setup.py` requires it.

- [ ] **Step 2: Write the parity test.**

Fixture rows (one sample `S0`, one contig `chr1`, all gaps between events >= 10001 so upstream's grouping equals ours; every mutation's min-neighbour gap <= cutoff so upstream's pre-filter keeps them all). With `cutoff=1000`, `vaf_cut=0.1`:

| event | positions | VAFs | upstream | ours |
|---|---|---|---|---|
| E1 | 1000, 1001 | .5, .5 | ClassIA | DOUBLET |
| E2 | 21000, 21001, 21002 | .5 | ClassIB | MBS |
| E3 | 41000, 41005 | .5 | ClassIC | OMIKLI |
| E4 | 61000, 61002, 61004, 61006 | .5 | ClassII | KATAEGIS |
| E5 | 81000, 81002, 81004 | .5, .5, .9 | ClassIC, ClassIC, ClassIII | OMIKLI, OMIKLI, OTHER |
| E6 | 101000, 101001, 101100, 101101 | .5 | ClassIC (greedy chains the 99-gap) | OMIKLI |

```python
"""Parity oracle: our classifier vs the REAL SigProfilerClusters.

Runs only where SigProfilerClusters is installed (the `sigprofiler` pixi
env). Uses upstream's own functions with ``correction=False`` and the same
IMD cutoffs, then compares per-mutation labels.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
from vcfixture import Number, Seq, Type, VcfBuilder

from genoray import SparseVar2
from genoray._svar2_clusters import DOUBLET, KATAEGIS, MBS, NONCLUSTERED, OMIKLI, OTHER
from genoray._svar2_fields import FormatField

cf = pytest.importorskip("SigProfilerClusters.classifyFunctions")

CUTOFF = 1000.0

# (positions, vafs) per event; see the plan table.
EVENTS = [
    ([1000, 1001], [0.5, 0.5]),
    ([21000, 21001, 21002], [0.5, 0.5, 0.5]),
    ([41000, 41005], [0.5, 0.5]),
    ([61000, 61002, 61004, 61006], [0.5, 0.5, 0.5, 0.5]),
    ([81000, 81002, 81004], [0.5, 0.5, 0.9]),
    ([101000, 101001, 101100, 101101], [0.5, 0.5, 0.5, 0.5]),
]
LABEL_TO_CODE = {
    "ClassIA": DOUBLET,
    "ClassIB": MBS,
    "ClassIC": OMIKLI,
    "ClassII": KATAEGIS,
    "ClassIII": OTHER,
}


def _mutations() -> list[tuple[int, float]]:
    out: list[tuple[int, float]] = []
    for positions, vafs in EVENTS:
        out.extend(zip(positions, vafs))
    return out


def _upstream_rows() -> list[str]:
    header = [
        "project", "samples", "ID", "genome", "mutType", "chr", "start", "end",
        "ref", "alt", "mutClass", "IMDplot", "IMD", "VAF/CCF",
    ]
    rows = ["\t".join(header)]
    for pos, vaf in _mutations():
        rows.append(
            "\t".join(
                ["T", "S0", ".", "GRCh37", "SNP", "chr1", str(pos), str(pos),
                 "C", "A", "SOMATIC", "1", "1", str(vaf)]
            )
        )
    return rows


def _run_upstream(tmp_path: Path, *, vaf: bool) -> dict[tuple[str, int], str]:
    pp = tmp_path / "proj"
    project_path = pp / "output" / "vcf_files" / "T_clustered"
    if vaf:
        target = project_path / "T_clustered_vaf.txt"
    else:
        target = project_path / "SNV" / "T_clustered.txt"
    target.parent.mkdir(parents=True)
    target.write_text("\n".join(_upstream_rows()) + "\n")
    sims = pp / "output" / "simulations" / "data"
    sims.mkdir(parents=True)
    with open(sims / "imds.pickle", "wb") as f:
        pickle.dump({"S0": CUTOFF}, f)

    # generateMatrices spawns an mp.Pool and renders plots; label comparison
    # does not need it.
    cf.generateMatrices = lambda *a, **k: None
    if vaf:
        cf.findClustersOfClusters(
            "T", False, str(pp) + "/", 1_000_000, {}, {}, str(pp / "log.txt"),
            "GRCh37", 1, {}, correction=False,
        )
    else:
        cf.findClustersOfClusters_noVAF(
            "T", False, str(pp) + "/", 1_000_000, {}, {}, str(pp / "log.txt"),
            "GRCh37", 1, {}, correction=False,
        )

    labels: dict[tuple[str, int], str] = {}
    for name in ("class1a", "class1b", "class1c", "class2", "class3"):
        path = project_path / "subclasses" / name / f"T_clustered_{name}.txt"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            row = line.split("\t")
            if len(row) < 8:
                continue
            subclass = next((t for t in reversed(row) if t.startswith("Class")), None)
            if subclass is not None:
                labels[(row[1], int(row[7]))] = subclass
    return labels


def _our_labels(tmp_path: Path, *, vaf: bool) -> dict[tuple[str, int], int]:
    doc = VcfBuilder(samples=["S0"], contigs=[("chr1", None)]).fmt("GT")
    if vaf:
        doc = doc.fmt("VAF", Number.ONE, Type.FLOAT)
    for pos, value in _mutations():
        kwargs = {"VAF": [[value]]} if vaf else {}
        doc = doc.record(
            "chr1", pos, ref="C", alt=[Seq("A")], gt=["1|0"], **kwargs
        )
    out = tmp_path / ("vaf.svar2" if vaf else "novaf.svar2")
    SparseVar2.from_vcf(
        out,
        doc.write(tmp_path / ("vaf.vcf.gz" if vaf else "novaf.vcf.gz"), bgzip=True, index=True),
        no_reference=True,
        format_fields=[FormatField("VAF", dtype="f32")] if vaf else None,
    )
    store = SparseVar2(out)
    store.annotate_clusters(
        imd_cutoff=CUTOFF, vaf_field="VAF" if vaf else None
    )
    sv = SparseVar2(out).with_fields(["cluster_class"])
    rag = sv.decode("chr1", [(0, 1_000_000)])
    positions = np.asarray(rag["pos"].data)
    codes = np.asarray(rag["cluster_class"].data)
    return {("S0", int(p)): int(c) for p, c in zip(positions, codes)}


def test_vaf_parity(tmp_path: Path):
    upstream = _run_upstream(tmp_path, vaf=True)
    ours = _our_labels(tmp_path, vaf=True)
    assert ours, "fixture produced no labels"
    assert NONCLUSTERED not in ours.values(), "fixture must be closed under the pre-filter"
    for key, subclass in upstream.items():
        assert ours[key] == LABEL_TO_CODE[subclass], f"{key}: {subclass}"
    # upstream skips len-1 events; those are our OTHER
    for key, code in ours.items():
        if key not in upstream:
            assert code == OTHER, f"{key} unlabeled upstream but ours={code}"


def test_no_vaf_parity(tmp_path: Path):
    upstream = _run_upstream(tmp_path, vaf=False)
    ours = _our_labels(tmp_path, vaf=False)
    assert ours, "fixture produced no labels"
    assert NONCLUSTERED not in ours.values(), "fixture must be closed under the pre-filter"
    for key, subclass in upstream.items():
        assert ours[key] == LABEL_TO_CODE[subclass], f"{key}: {subclass}"
    # no-VAF upstream drops failing events; those are our OTHER
    for key, code in ours.items():
        if key not in upstream:
            assert code == OTHER, f"{key} dropped upstream but ours={code}"
```

- [ ] **Step 3: Run the parity tests.**

Run: `pixi run -e sigprofiler pytest tests/test_clusters_calibration.py -v`
Expected: both tests pass. If a label differs, do NOT adjust the expected value — treat it as a port bug and reconcile against `CF:1184-1210` / `CF:1270-1510`.

- [ ] **Step 4: Commit.**

```bash
git add pixi.toml pixi.lock tests/test_clusters_calibration.py
git commit -m "test(clusters): upstream SigProfilerClusters parity oracle"
```

---

### Task 7: Document the public API

**Files:**
- Modify: `skills/genoray-api/SKILL.md`

- [ ] **Step 1: Add `annotate_clusters` to the SVAR2 section** (next to `annotate_mutations`/`mutation_matrix`), with: the signature, the `imd_cutoff` float-or-mapping contract, `vaf_field` requirements (FORMAT, 2/4-byte float, missing -> upstream `-1.5` sentinel), the label codebook table, the read-back recipe (`with_fields(["cluster_class"])` + `decode`), the `contigs=` 255-fill behavior, and the note that cutoffs come from the caller (upstream simulation) so the port is label-for-label reproducible with upstream's `imds.pickle`.

- [ ] **Step 2: Verify the docs render as intended.**

Run: `pixi run -e lint prek run --files skills/genoray-api/SKILL.md`
Expected: clean.

- [ ] **Step 3: Commit.**

```bash
git add skills/genoray-api/SKILL.md
git commit -m "docs(clusters): document annotate_clusters and the cluster_class codebook"
```

---

### Task 8: Final verification

- [ ] **Step 1: Full Rust suite + core guard.**

```bash
pixi run -e lint test-rust
pixi run -e lint check-core
```
Expected: both clean.

- [ ] **Step 2: Full Python suite (default env).**

```bash
pixi run pytest tests -x -q
```
Expected: all pass (or only pre-existing failures; compare against `main` if unsure).

- [ ] **Step 3: Parity suite (sigprofiler env).**

```bash
pixi run -e sigprofiler pytest tests/test_clusters_calibration.py -q
```
Expected: pass.

- [ ] **Step 4: Lint hooks.**

```bash
pixi run -e lint lint
```
Expected: clean.

- [ ] **Step 5: Report** the exact commands run and their outcomes (verification-before-completion), then hand off for review/merge.
