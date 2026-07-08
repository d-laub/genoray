//! Read-bound per-class gather: find_ranges emits per-class dense ranges and
//! gather_ranges_readbound replays them into BatchResultSplit without building
//! the contig-wide DenseUnion.
mod common;

use common::{SynthRecord, build_contig};
use genoray_core::bits_get_bit;
use genoray_core::query::{
    BatchResultSplit, ContigReader, HapCalls, KeyRef, decode_keyref_alt_pub, find_ranges,
    gather_haps_readbound, gather_ranges_readbound, overlap_batch,
};
use genoray_core::search;
use tempfile::tempdir;

fn synth_reader(out: &std::path::Path) -> ContigReader {
    let samples = ["S0", "S1"];
    let records = vec![
        SynthRecord {
            pos: 100,
            ref_allele: b"A",
            alts: vec![&b"C"[..]],
            gt: vec![1, 0, 0, 0],
        },
        // Dense SNP: AC=4 (all 4 haps carry) routes to the Dense class, not
        // VarKey. Cost model (n_samples=2, ploidy=2 → np=4): SNP dense_bits =
        // 32+2+4 = 38, var_key_bits = 34·x; dense wins for x >= 2, so x=4 is
        // Dense. This exercises the dense_snp / dense_snp_present channel of
        // BatchResultSplit (the pos-100 SNP above stays VarKey at AC=1).
        SynthRecord {
            pos: 150,
            ref_allele: b"A",
            alts: vec![&b"C"[..]],
            gt: vec![1, 1, 1, 1],
        },
        // Partially-carried dense SNP: AC=2 (haps 0,1 carry; haps 2,3 don't).
        // Same cost model as pos-150 (np=4, snp per_call=34): dense_bits=38 <
        // var_key_bits=34·x for x>=2, so this also routes to Dense — but unlike
        // pos-150 (AC=4, all-carried) it exercises BOTH a TRUE and a FALSE
        // dense-SNP present bit. Different ALT (`G` vs pos-150's `C`) so an
        // allele-level parity check actually discriminates a wrong-allele bug.
        SynthRecord {
            pos: 175,
            ref_allele: b"A",
            alts: vec![&b"G"[..]],
            gt: vec![1, 1, 0, 0],
        },
        SynthRecord {
            pos: 200,
            ref_allele: b"A",
            alts: vec![&b"AT"[..]],
            gt: vec![0, 1, 1, 1],
        },
        SynthRecord {
            pos: 300,
            ref_allele: b"AT",
            alts: vec![&b"A"[..]],
            gt: vec![1, 1, 0, 1],
        },
    ];
    build_contig(out, "chr1", &samples, 2, &records);
    ContigReader::open(out.to_str().unwrap(), "chr1", 2, 2).unwrap()
}

#[test]
fn test_find_ranges_emits_per_class_dense_ranges() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();
    let reader = synth_reader(&out);
    let regions = vec![(0u32, 1_000_000u32), (250u32, 400u32)];

    let rb = find_ranges(&reader, &regions, None);
    // Both per-class range vectors are per-region (dense is cohort-shared).
    assert_eq!(rb.dense_snp_range.len(), regions.len());
    assert_eq!(rb.dense_indel_range.len(), regions.len());
    // Each per-class window is a subset of that class's table; ranges are valid.
    for &(s, e) in rb.dense_snp_range.iter().chain(rb.dense_indel_range.iter()) {
        assert!(s <= e);
    }
    // Region 0 spans the whole contig: it must see the dense SNP (pos 150,
    // AC=4) and the dense indels. The union window must equal snp∪indel counts.
    let (us0, ue0) = rb.dense_range[0];
    let snp0 = rb.dense_snp_range[0].1 - rb.dense_snp_range[0].0;
    let indel0 = rb.dense_indel_range[0].1 - rb.dense_indel_range[0].0;
    // Empirical coverage guard: the fixture MUST route at least one SNP to the
    // Dense class, else the dense_snp / dense_snp_present channel of
    // BatchResultSplit is never exercised with real content and a SNP-class
    // bit-indexing / v_end bug would pass silently. If this fires, the cost
    // model routed the AC=4 SNP to VarKey — escalate rather than weaken.
    assert!(
        snp0 >= 1,
        "fixture must route >=1 SNP to Dense so the dense-SNP channel is covered \
         (dense_snp_range[0] = {:?})",
        rb.dense_snp_range[0]
    );
    assert_eq!(
        ue0 - us0,
        snp0 + indel0,
        "union window size must equal sum of per-class window sizes"
    );
}

/// Merge vk ⋈ dense_snp ⋈ dense_indel for one hap and decode — the gvl-side
/// reconstruction, expressed as a test oracle.
fn readbound_decode_hap(
    br: &BatchResultSplit,
    _reader: &ContigReader,
    r: usize,
    s: usize,
    p: usize,
) -> Vec<(u32, i32)> {
    use genoray_core::query::KeyRef;
    let h = (r * br.n_samples + s) * br.ploidy + p;
    let mut merged: Vec<KeyRef> = br.vk[br.vk_off[h]..br.vk_off[h + 1]].to_vec();

    let (ss, se) = br.dense_snp_range[r];
    let bit0 = br.dense_snp_present_off[h];
    for (k, j) in (ss..se).enumerate() {
        if genoray_core::bits_get_bit(&br.dense_snp_present, bit0 + k) {
            merged.push(br.dense_snp[j]);
        }
    }
    let (is_, ie_) = br.dense_indel_range[r];
    let bit0 = br.dense_indel_present_off[h];
    for (k, j) in (is_..ie_).enumerate() {
        if genoray_core::bits_get_bit(&br.dense_indel_present, bit0 + k) {
            merged.push(br.dense_indel[j]);
        }
    }
    // Stable position sort (var_key already ahead of dense within its own run).
    merged.sort_by_key(|kr| kr.position);
    merged
        .into_iter()
        .map(|kr| (kr.position, kr.key as i32))
        .collect()
}

#[test]
fn test_readbound_reconstructs_union_per_hap() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();
    let reader = synth_reader(&out);
    let regions = vec![(0u32, 1_000_000u32), (250u32, 400u32), (150u32, 250u32)];

    let oracle = overlap_batch(&reader, &regions);
    let rb = find_ranges(&reader, &regions, None);
    let got = gather_ranges_readbound(&reader, &rb);

    assert_eq!(got.n_regions, oracle.n_regions);
    assert_eq!(got.n_samples, oracle.n_samples);
    assert_eq!(got.ploidy, oracle.ploidy);

    for r in 0..oracle.n_regions {
        for s in 0..oracle.n_samples {
            for p in 0..oracle.ploidy {
                // Oracle: decode via the shipped union decode_hap, keep the full
                // (position, ilen, alt) triple. Comparing only (position, ilen)
                // would miss a wrong-allele-at-correct-position bug for SNPs,
                // since every SNP has ilen == 0 regardless of which base it
                // decodes to.
                let hc: HapCalls = oracle.decode_hap(&reader, r, s, p);
                let want: Vec<(u32, i32, Vec<u8>)> = hc
                    .positions
                    .iter()
                    .zip(hc.ilens.iter())
                    .zip(hc.alts.iter())
                    .map(|((&a, &b), c)| (a, b, c.clone()))
                    .collect();
                let got_keys = readbound_decode_hap(&got, &reader, r, s, p);
                // Decode the read-bound raw keys the same way, keeping the ALT.
                let got_dec: Vec<(u32, i32, Vec<u8>)> = got_keys
                    .iter()
                    .map(|&(pos, key)| {
                        let (ilen, alt) = decode_keyref_alt_pub(pos, key as u32, &reader);
                        (pos, ilen, alt)
                    })
                    .collect();
                assert_eq!(got_dec, want, "hap (r={r}, s={s}, p={p})");
            }
        }
    }
}

#[test]
fn test_readbound_gather_builds_no_search_tree() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();
    let reader = synth_reader(&out);
    let regions = vec![(0u32, 1_000_000u32), (250u32, 400u32)];

    let rb = find_ranges(&reader, &regions, None);
    let before = search::search_tree_build_count();
    let _ = gather_ranges_readbound(&reader, &rb);
    assert_eq!(
        search::search_tree_build_count(),
        before,
        "gather_ranges_readbound must build zero SearchTrees (no dense_union)"
    );
}

#[test]
fn test_readbound_subset_matches_full_selected_haps() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();
    let reader = synth_reader(&out); // 2 samples, ploidy 2
    let regions = vec![(0u32, 1_000_000u32)];

    let full = gather_ranges_readbound(&reader, &find_ranges(&reader, &regions, None));
    // Select only sample 1.
    let sub = gather_ranges_readbound(&reader, &find_ranges(&reader, &regions, Some(&[1])));
    assert_eq!(sub.n_samples, 1);
    for p in 0..reader_ploidy(&reader) {
        let a = readbound_decode_hap(&sub, &reader, 0, 0, p); // selected slot 0 == orig sample 1
        let b = readbound_decode_hap(&full, &reader, 0, 1, p); // orig sample 1
        assert_eq!(a, b, "subset ploid {p}");
    }
}

fn reader_ploidy(_r: &ContigReader) -> usize {
    2
}

#[test]
fn test_flat_gather_matches_cartesian_full_cohort() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();
    let reader = synth_reader(&out); // 2 samples, ploidy 2
    let regions = vec![(0u32, 1_000_000u32), (250u32, 400u32)];
    let ploidy = 2usize;

    let rb = find_ranges(&reader, &regions, None);
    let cart = gather_ranges_readbound(&reader, &rb);

    // Enumerate flat queries in the SAME order cart lays out haps:
    // region-major, samples 0..S, so query q = r*S + s, orig sample = s.
    let s_n = rb.n_samples;
    let mut region_starts = Vec::new();
    let mut orig_samples = Vec::new();
    let mut vk_snp_range = Vec::new();
    let mut vk_indel_range = Vec::new();
    let mut dsr = Vec::new();
    let mut dir_ = Vec::new();
    for r in 0..regions.len() {
        for s in 0..s_n {
            region_starts.push(rb.region_starts[r]);
            orig_samples.push(rb.sample_cols[s]);
            dsr.push(rb.dense_snp_range[r]);
            dir_.push(rb.dense_indel_range[r]);
            for p in 0..ploidy {
                let row = r * (s_n * ploidy) + s * ploidy + p;
                vk_snp_range.push(rb.vk_snp_range[row]);
                vk_indel_range.push(rb.vk_indel_range[row]);
            }
        }
    }
    let flat = gather_haps_readbound(
        &reader,
        &region_starts,
        &orig_samples,
        &vk_snp_range,
        &vk_indel_range,
        &dsr,
        &dir_,
        ploidy,
    );

    // Compare decoded per-hap. cart hap (r,s,p) == flat query q=r*S+s, ploid p.
    for r in 0..regions.len() {
        for s in 0..s_n {
            for p in 0..ploidy {
                let a = readbound_decode_hap(&cart, &reader, r, s, p);
                let b = readbound_decode_hap(&flat, &reader, r * s_n + s, 0, p);
                assert_eq!(a, b, "flat vs cartesian (r={r}, s={s}, p={p})");
            }
        }
    }
}

/// Byte-identical (per-hap RAW field) regression test for the
/// `gather_haps_readbound` asm-fix pass: asserts the *raw* `vk` `KeyRef`
/// sequence (order + position + key, no sort/decode) and the raw
/// `dense_snp_present`/`dense_indel_present` bits for every hap match the
/// independent `gather_ranges_readbound` (cartesian) oracle exactly.
///
/// This is deliberately stronger than `test_flat_gather_matches_cartesian_
/// full_cohort` above (which decodes+sorts before comparing, so it cannot
/// see a merge-order regression): the perf pass replaced the per-hap
/// `spine::merge_keys(vec![snp_run, indel_run])` call in
/// `gather_haps_readbound` with a hand-inlined two-pointer merge, and only
/// `gather_ranges_readbound` (untouched) still calls `spine::merge_keys`.
/// Comparing raw, unsorted `vk` slices between the two functions is exactly
/// the check that would fail if the inlined merge's tie-break or ordering
/// diverged from `merge_keys`'s stable (earlier-run-wins-ties) semantics.
#[test]
fn test_gather_haps_readbound_byte_identical() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();
    let reader = synth_reader(&out); // 2 samples, ploidy 2
    let regions = vec![(0u32, 1_000_000u32), (250u32, 400u32), (150u32, 250u32)];
    let ploidy = 2usize;

    let rb = find_ranges(&reader, &regions, None);
    let cart = gather_ranges_readbound(&reader, &rb);

    // Flatten in the same region-major, sample-major order `cart` lays out
    // haps in, so flat query q = r*S+s, ploid p <-> cart hap (r, s, p).
    let s_n = rb.n_samples;
    let mut region_starts = Vec::new();
    let mut orig_samples = Vec::new();
    let mut vk_snp_range = Vec::new();
    let mut vk_indel_range = Vec::new();
    let mut dsr = Vec::new();
    let mut dir_ = Vec::new();
    for r in 0..regions.len() {
        for s in 0..s_n {
            region_starts.push(rb.region_starts[r]);
            orig_samples.push(rb.sample_cols[s]);
            dsr.push(rb.dense_snp_range[r]);
            dir_.push(rb.dense_indel_range[r]);
            for p in 0..ploidy {
                let row = r * (s_n * ploidy) + s * ploidy + p;
                vk_snp_range.push(rb.vk_snp_range[row]);
                vk_indel_range.push(rb.vk_indel_range[row]);
            }
        }
    }
    let flat: BatchResultSplit = gather_haps_readbound(
        &reader,
        &region_starts,
        &orig_samples,
        &vk_snp_range,
        &vk_indel_range,
        &dsr,
        &dir_,
        ploidy,
    );

    let mut any_nonempty_vk = false;
    let mut any_dense_snp_present = false;
    let mut any_dense_indel_present = false;

    for r in 0..regions.len() {
        for s in 0..s_n {
            let cart_h_base = (r * cart.n_samples + s) * cart.ploidy;
            let flat_q = r * s_n + s;
            let flat_h_base = flat_q * ploidy; // flat.n_samples == 1
            for p in 0..ploidy {
                let ch = cart_h_base + p;
                let fh = flat_h_base + p;

                // Raw vk KeyRef sequence: exact order, exact (position, key).
                let cart_vk: &[KeyRef] = &cart.vk[cart.vk_off[ch]..cart.vk_off[ch + 1]];
                let flat_vk: &[KeyRef] = &flat.vk[flat.vk_off[fh]..flat.vk_off[fh + 1]];
                assert_eq!(
                    flat_vk, cart_vk,
                    "raw vk mismatch (r={r}, s={s}, p={p}): flat={flat_vk:?} cart={cart_vk:?}"
                );
                if !cart_vk.is_empty() {
                    any_nonempty_vk = true;
                }

                // Raw dense/snp presence bits over this region's dense_snp window.
                let (ss, se) = rb.dense_snp_range[r];
                let cart_bit0 = cart.dense_snp_present_off[ch];
                let flat_bit0 = flat.dense_snp_present_off[fh];
                for k in 0..(se - ss) {
                    let c = bits_get_bit(&cart.dense_snp_present, cart_bit0 + k);
                    let f = bits_get_bit(&flat.dense_snp_present, flat_bit0 + k);
                    assert_eq!(
                        f, c,
                        "dense_snp_present bit mismatch (r={r}, s={s}, p={p}, k={k})"
                    );
                    any_dense_snp_present |= c;
                }

                // Raw dense/indel presence bits over this region's dense_indel window.
                let (is_, ie_) = rb.dense_indel_range[r];
                let cart_bit0 = cart.dense_indel_present_off[ch];
                let flat_bit0 = flat.dense_indel_present_off[fh];
                for k in 0..(ie_ - is_) {
                    let c = bits_get_bit(&cart.dense_indel_present, cart_bit0 + k);
                    let f = bits_get_bit(&flat.dense_indel_present, flat_bit0 + k);
                    assert_eq!(
                        f, c,
                        "dense_indel_present bit mismatch (r={r}, s={s}, p={p}, k={k})"
                    );
                    any_dense_indel_present |= c;
                }
            }
        }
    }

    // Coverage guards: fail loudly if the fixture stopped exercising the
    // channels this test exists to protect (rather than passing vacuously).
    assert!(
        any_nonempty_vk,
        "fixture must exercise the var_key (vk) merge path"
    );
    assert!(
        any_dense_snp_present,
        "fixture must exercise dense_snp_present"
    );
    assert!(
        any_dense_indel_present,
        "fixture must exercise dense_indel_present"
    );
}

/// Dedicated fixture for the same-position SNP+indel tie-break: one SNP and
/// one insertion at the identical position (400), both carried by the SAME
/// hap (sample S0, ploid 0) and both with a single carrier (AC=1) so the cost
/// model keeps them `VarKey` (never routed to `Dense` — see
/// `cost_model::choose_representation`: AC=1 always yields `dense_bits >
/// var_key_bits`). Two records sharing a POS is unusual for real VCFs but
/// unconstrained by the reader; both share REF `A` so the shared 'N'-filler
/// FASTA stamp is consistent between them.
fn tie_break_reader(out: &std::path::Path) -> ContigReader {
    let samples = ["S0", "S1"];
    let records = vec![
        SynthRecord {
            pos: 400,
            ref_allele: b"A",
            alts: vec![&b"C"[..]],
            gt: vec![1, 0, 0, 0], // S0 ploid0 only
        },
        SynthRecord {
            pos: 400,
            ref_allele: b"A",
            alts: vec![&b"AT"[..]],
            gt: vec![1, 0, 0, 0], // S0 ploid0 only — same hap as the SNP above
        },
    ];
    build_contig(out, "chr1", &samples, 2, &records);
    ContigReader::open(out.to_str().unwrap(), "chr1", 2, 2).unwrap()
}

/// Targeted regression for the SNP+indel same-position tie-break that
/// `test_gather_haps_readbound_byte_identical`'s distinct-position fixture
/// (100/150/175/200/300) never exercises: `merge_keys`'s k-way merge picks
/// the earlier run (`snp_run`, index 0) on ties, so a SNP and an indel at the
/// same position must decode with the SNP first. `gather_haps_readbound`'s
/// hand-inlined two-pointer merge encodes the same rule as `si <= ii` (favor
/// `snp_run` on equality) — see the comment above that loop in `query.rs`.
///
/// If that `<=` were weakened to `<`, the equal-position branch would fall
/// through to the `else` arm and push the indel first, flipping the decoded
/// order to (indel, snp). This test would then fail both assertions below:
/// the explicit decode order check, and the byte-identical `vk` comparison
/// against the `merge_keys`-based `gather_ranges_readbound` oracle (which
/// does not share the bug, so `cart_vk` would stay (snp, indel) while
/// `flat_vk` flipped to (indel, snp) — an outright mismatch).
#[test]
fn test_gather_haps_readbound_tie_break_snp_before_indel() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();
    let reader = tie_break_reader(&out);
    let regions = vec![(0u32, 1_000_000u32)];
    let ploidy = 2usize;

    let rb = find_ranges(&reader, &regions, None);
    let cart = gather_ranges_readbound(&reader, &rb); // oracle: spine::merge_keys

    // Flatten in the same region-major, sample-major order `cart` uses.
    let s_n = rb.n_samples;
    let mut region_starts = Vec::new();
    let mut orig_samples = Vec::new();
    let mut vk_snp_range = Vec::new();
    let mut vk_indel_range = Vec::new();
    let mut dsr = Vec::new();
    let mut dir_ = Vec::new();
    for r in 0..regions.len() {
        for s in 0..s_n {
            region_starts.push(rb.region_starts[r]);
            orig_samples.push(rb.sample_cols[s]);
            dsr.push(rb.dense_snp_range[r]);
            dir_.push(rb.dense_indel_range[r]);
            for p in 0..ploidy {
                let row = r * (s_n * ploidy) + s * ploidy + p;
                vk_snp_range.push(rb.vk_snp_range[row]);
                vk_indel_range.push(rb.vk_indel_range[row]);
            }
        }
    }
    let flat: BatchResultSplit = gather_haps_readbound(
        &reader,
        &region_starts,
        &orig_samples,
        &vk_snp_range,
        &vk_indel_range,
        &dsr,
        &dir_,
        ploidy,
    );

    // Locate S0 ploid0 in both layouts (region 0, sample slot for S0, p=0).
    let s0_slot = rb
        .sample_cols
        .iter()
        .position(|&orig| orig == 0)
        .expect("S0 must be present");
    let cart_h = s0_slot * cart.ploidy; // region 0
    let flat_q = s0_slot; // region 0
    let flat_h = flat_q * ploidy; // flat.n_samples == 1

    let cart_vk: &[KeyRef] = &cart.vk[cart.vk_off[cart_h]..cart.vk_off[cart_h + 1]];
    let flat_vk: &[KeyRef] = &flat.vk[flat.vk_off[flat_h]..flat.vk_off[flat_h + 1]];

    // Byte-identical check (same style as test_gather_haps_readbound_byte_identical):
    // this is the assertion that would catch a wrong tie-break, since `cart_vk`
    // is built via `spine::merge_keys` (untouched) and `flat_vk` via the
    // hand-inlined merge under test.
    assert_eq!(
        flat_vk, cart_vk,
        "raw vk mismatch at the same-position SNP+indel tie: flat={flat_vk:?} cart={cart_vk:?}"
    );

    // Explicit, self-documenting decode of the tie-break: exactly 2 calls at
    // position 400 for this hap, SNP (ilen=0, ALT "C") strictly before the
    // insertion (ilen=1, ALT "AT").
    assert_eq!(cart_vk.len(), 2, "expected exactly SNP+indel at pos 400");
    let decoded: Vec<(u32, i32, Vec<u8>)> = cart_vk
        .iter()
        .map(|kr| {
            let (ilen, alt) = decode_keyref_alt_pub(kr.position, kr.key, &reader);
            (kr.position, ilen, alt)
        })
        .collect();
    assert_eq!(
        decoded,
        vec![(400, 0, b"C".to_vec()), (400, 1, b"AT".to_vec())],
        "merge_keys tie-break must emit the SNP before the indel at an equal position"
    );
}
