//! Reader-free spine algorithms for the SVAR2 query decode core (M6.1).
//!
//! `gather_keys` and `merge_keys` are the pure half of what M5's `query.rs`
//! did inline: overlap-resolve a position run into undecoded `(position, key)`
//! pairs, and merge already-sorted runs. They carry **uniform 32-bit keys** and
//! never touch alleles or the LUT — decoding is the consumer's job. Splitting
//! them out lets both `overlap_sample` and the batched `overlap_batch` share one
//! gather/merge and one uniform-key convention.

use crate::search::{SearchTree, overlap_range};

/// An undecoded variant reference: a genomic start plus its **uniform 32-bit
/// key** (SNPs re-expanded via [`crate::rvk::snp_code_to_key`]; indel keys
/// stored uniform already). A key with LSB set references the long-allele LUT,
/// resolved downstream — never here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyRef {
    pub position: u32,
    pub key: u32,
}

#[allow(clippy::too_many_arguments)]
pub fn gather_keys(
    positions: &[u32],
    max_region_length: u32,
    q_start: u32,
    q_end: u32,
    del_len: impl Fn(usize) -> u32,
    carried: impl Fn(usize) -> bool,
    to_key: impl Fn(usize) -> u32,
    out: &mut Vec<KeyRef>,
) {
    if positions.is_empty() {
        return;
    }
    let v_ends: Vec<u32> = positions
        .iter()
        .enumerate()
        .map(|(i, &p)| p + 1 + del_len(i))
        .collect();
    let tree = SearchTree::new(positions);
    let (s_idx, e_idx) = overlap_range(&tree, &v_ends, max_region_length, q_start, q_end);
    for (i, &position) in positions.iter().enumerate().take(e_idx).skip(s_idx) {
        if carried(i) {
            out.push(KeyRef {
                position,
                key: to_key(i),
            });
        }
    }
}

/// K-way merge of already position-sorted runs into one position-sorted list.
/// Stable across ties (earlier run wins). `O(total × n_runs)` with `n_runs`
/// small (2 within a channel; more only if M11 adds a `pointer` sub-stream).
pub fn merge_keys(runs: Vec<Vec<KeyRef>>) -> Vec<KeyRef> {
    let total: usize = runs.iter().map(|r| r.len()).sum();
    let mut heads = vec![0usize; runs.len()];
    let mut out = Vec::with_capacity(total);
    for _ in 0..total {
        let mut best: Option<usize> = None;
        for r in 0..runs.len() {
            if heads[r] >= runs[r].len() {
                continue;
            }
            match best {
                // Keep `best` on ties so the earlier run emits first (stable).
                Some(b) if runs[b][heads[b]].position <= runs[r][heads[r]].position => {}
                _ => best = Some(r),
            }
        }
        let b = best.expect("total accounts for every remaining element");
        out.push(runs[b][heads[b]]);
        heads[b] += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kr(position: u32, key: u32) -> KeyRef {
        KeyRef { position, key }
    }

    #[test]
    fn test_gather_keys_snp_half_open() {
        // positions [10, 20, 30], v_end = pos + 1, max_del 0; query [15, 25):
        // only index 1 (pos 20) overlaps. to_key(i) = i as a marker key.
        let positions = [10u32, 20, 30];
        let mut out = Vec::new();
        gather_keys(
            &positions,
            0,
            15,
            25,
            |_| 0,
            |_| true,
            |i| i as u32,
            &mut out,
        );
        assert_eq!(out, vec![kr(20, 1)]);
    }

    #[test]
    fn test_gather_keys_deletion_spans_query_start() {
        // v0 start 2 deletes 6 bases -> v_end 9; v1 SNP at 10. query [5, 7):
        // only v0 (2..9) spans it. max_region_length 6 covers the deletion.
        let positions = [2u32, 10];
        let dels = [6u32, 0];
        let mut out = Vec::new();
        gather_keys(
            &positions,
            6,
            5,
            7,
            |i| dels[i],
            |_| true,
            |i| 100 + i as u32,
            &mut out,
        );
        assert_eq!(out, vec![kr(2, 100)]);
    }

    #[test]
    fn test_gather_keys_carried_filter() {
        // Only even indices carried; query covers all.
        let positions = [10u32, 20, 30, 40];
        let mut out = Vec::new();
        gather_keys(
            &positions,
            0,
            0,
            100,
            |_| 0,
            |i| i % 2 == 0,
            |i| i as u32,
            &mut out,
        );
        assert_eq!(out, vec![kr(10, 0), kr(30, 2)]);
    }

    #[test]
    fn test_gather_keys_empty_positions() {
        let mut out = Vec::new();
        gather_keys(&[], 0, 0, 100, |_| 0, |_| true, |_| 0, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn test_merge_keys_orders_by_position() {
        let runs = vec![
            vec![kr(10, 0), kr(30, 2)],
            vec![kr(20, 1)],
            vec![],
            vec![kr(25, 9)],
        ];
        let merged = merge_keys(runs);
        let positions: Vec<u32> = merged.iter().map(|k| k.position).collect();
        assert_eq!(positions, vec![10, 20, 25, 30]);
    }

    #[test]
    fn test_merge_keys_ties_keep_earlier_run_first() {
        let runs = vec![vec![kr(50, 111)], vec![kr(50, 222)]];
        assert_eq!(merge_keys(runs), vec![kr(50, 111), kr(50, 222)]);
    }
}
