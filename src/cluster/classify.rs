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
        if let (Some(x), Some(y)) = (a.vaf, b.vaf)
            && round4((y - x).abs()) > vaf_cut
        {
            c.vaf_bad += 1;
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
fn label_event(muts: &[Mutation], ev: &[usize], cutoff: f64, vaf_cut: f64, labels: &mut [u8]) {
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
        assert_eq!(
            cluster_sample(&snps(&[10, 20]), 5.0, 0.1),
            vec![NONCLUSTERED; 2]
        );
        // adjacent pair -> DOUBLET
        assert_eq!(cluster_sample(&snps(&[10, 11]), 5.0, 0.1), vec![DOUBLET; 2]);
        // 3+ adjacent -> MBS
        assert_eq!(cluster_sample(&snps(&[10, 11, 12]), 5.0, 0.1), vec![MBS; 3]);
        // one line gap, no failures -> OMIKLI
        assert_eq!(cluster_sample(&snps(&[10, 12]), 5.0, 0.1), vec![OMIKLI; 2]);
        // two line gaps -> OMIKLI (distancesLine > 1 but len < 4)
        assert_eq!(
            cluster_sample(&snps(&[10, 12, 14]), 5.0, 0.1),
            vec![OMIKLI; 3]
        );
        // two line gaps and 4+ mutations -> KATAEGIS
        assert_eq!(
            cluster_sample(&snps(&[10, 12, 14, 16]), 5.0, 0.1),
            vec![KATAEGIS; 4]
        );
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
        assert_eq!(
            cluster_sample(&snps(&[10, 11, 100, 101]), 20.0, 0.1),
            vec![OTHER; 4]
        );
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
        let muts = [
            m(10, 0, Some(0.5)),
            m(12, 0, Some(0.5)),
            m(14, 0, Some(0.9)),
        ];
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

    #[test]
    fn wire_codes_are_pinned() {
        assert_eq!(
            (
                NONCLUSTERED,
                DOUBLET,
                MBS,
                OMIKLI,
                KATAEGIS,
                OTHER,
                NOT_ANNOTATED
            ),
            (0, 1, 2, 3, 4, 5, 255)
        );
    }
}
