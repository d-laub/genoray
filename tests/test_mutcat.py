from __future__ import annotations

import numpy as np
import polars as pl

from genoray._mutcat import (
    DBS78,
    DBS78_INDEX,
    DBS78_OFFSET,
    ID83,
    ID83_INDEX,
    SBS96,
    SENTINELS,
    SBS96_INDEX,
    build_entry_codes,
    classify_dbs78,
    classify_id83,
    classify_sbs96,
    classify_variants,
    code_ranges,
)
from genoray._reference import Reference


def test_codebook_sizes():
    assert len(SBS96) == 96
    assert len(DBS78) == 78
    assert len(ID83) == 83


def test_sbs96_labels_well_formed():
    assert SBS96[0] == "A[C>A]A"
    assert SBS96[-1] == "T[T>G]T"
    assert all(lbl[1:2] == "[" or "[" in lbl for lbl in SBS96)


def test_dbs78_known_members():
    assert "AC>CA" in DBS78
    assert "TT>GG" in DBS78


def test_id83_known_members():
    assert ID83[0] == "1:Del:C:0"
    assert "5:Del:M:5" in ID83
    assert ID83[-1] == "5:Del:M:5"


def test_code_ranges_are_contiguous_and_disjoint():
    r = code_ranges()
    assert r["SBS96"] == (0, 96)
    assert r["DBS78"] == (96, 174)
    assert r["ID83"] == (174, 257)


def test_sentinels_outside_category_ranges():
    # sentinels must not collide with 0..256
    assert all(v < 0 for v in SENTINELS.values())
    assert SENTINELS["DBS_PARTNER"] != SENTINELS["UNCLASSIFIED"]


def test_sbs96_pyrimidine_direct():
    # context A[C>A]G : ref=C (pyrimidine), keep as-is
    code = classify_sbs96(five=b"A", ref=b"C", alt=b"A", three=b"G")
    assert code == SBS96_INDEX["A[C>A]G"]


def test_sbs96_purine_folds_to_revcomp():
    # ref=G>T in context T_A : revcomp -> T[C>A]A
    # original T[G>T]A ; revcomp substitution G>T -> C>A ; flanks: comp(A)=T (new 5'),
    # comp(T)=A (new 3') -> T[C>A]A
    code = classify_sbs96(five=b"T", ref=b"G", alt=b"T", three=b"A")
    assert code == SBS96_INDEX["T[C>A]A"]


def test_sbs96_unclassified_cases():
    unc = SENTINELS["UNCLASSIFIED"]
    # identity mutation (ref == alt)
    assert classify_sbs96(five=b"A", ref=b"C", alt=b"C", three=b"G") == unc
    # non-ACGT ref
    assert classify_sbs96(five=b"A", ref=b"N", alt=b"C", three=b"G") == unc
    # non-ACGT flank
    assert classify_sbs96(five=b"N", ref=b"C", alt=b"A", three=b"G") == unc
    # empty bytes
    assert classify_sbs96(five=b"A", ref=b"", alt=b"C", three=b"G") == unc


def test_dbs78_direct_member():
    # AC>CA is directly in the catalogue
    assert classify_dbs78(b"AC", b"CA") == DBS78_INDEX["AC>CA"]


def test_dbs78_folds_to_revcomp_member():
    # GT>TG : revcomp of ref GT is AC, revcomp of alt TG is CA -> AC>CA
    assert classify_dbs78(b"GT", b"TG") == DBS78_INDEX["AC>CA"]


def test_dbs78_non_doublet_unclassified():
    assert classify_dbs78(b"AC", b"AC") == SENTINELS["UNCLASSIFIED"]  # no change
    assert classify_dbs78(b"ACG", b"TTT") == SENTINELS["UNCLASSIFIED"]  # >2bp
    assert classify_dbs78(b"AN", b"CA") == SENTINELS["UNCLASSIFIED"]  # non-ACGT base


def _ref_fn(seq: bytes):
    """Build a reference-fetch callable over a single contig given as bytes."""
    arr = np.frombuffer(seq, dtype=np.uint8)

    def fetch(start: int, end: int) -> bytes:
        out = np.full(end - start, ord("N"), dtype=np.uint8)
        s, e = max(start, 0), min(end, len(arr))
        if e > s:
            out[s - start : e - start] = arr[s:e]
        return out.tobytes()

    return fetch


def test_id83_1bp_deletion_in_homopolymer():
    # ref: ...A CCCCC G...  delete one C from a run of 5 C's
    # contig: index: 0=A,1..5=CCCCC,6=G ; anchor at pos 0 (A), REF="AC", ALT="A"
    # deleted base C, the run downstream of the deletion has 4 remaining C's -> repeat class
    fetch = _ref_fn(b"ACCCCCG")
    code = classify_id83(pos=0, ref=b"AC", alt=b"A", fetch=fetch)
    # SigProfiler confirmed: repeat count = homopolymer_length - 1 (4 remaining C's -> bucket 4)
    assert code == ID83_INDEX["1:Del:C:4"]


def test_id83_1bp_insertion_T():
    # insert a T with no downstream T repeat
    fetch = _ref_fn(b"AGGGG")
    code = classify_id83(pos=0, ref=b"A", alt=b"AT", fetch=fetch)
    assert code == ID83_INDEX["1:Ins:T:0"]


def test_id83_non_indel_unclassified():
    fetch = _ref_fn(b"ACGT")
    assert (
        classify_id83(pos=0, ref=b"A", alt=b"C", fetch=fetch)
        == SENTINELS["UNCLASSIFIED"]
    )


def test_classify_variants_mixed(tmp_path):
    import pysam

    fa = tmp_path / "ref.fa"
    # chr1: A C G T A C G T  (0..7)
    fa.write_text(">chr1\nACGTACGT\n")
    pysam.faidx(str(fa))
    ref = Reference.from_path(fa)

    index = pl.DataFrame(
        {
            "CHROM": ["chr1", "chr1"],
            "POS": [1, 2],  # 0-based
            "REF": ["C", "G"],
            "ALT": [["A"], ["GT"]],  # SNV ; insertion
        }
    )
    codes = classify_variants(index, ref)
    assert codes.dtype == np.int16
    assert len(codes) == 2
    # first is an SNV -> within SBS96 range
    assert 0 <= codes[0] < 96
    # second is a 1bp insertion -> within ID-83 range
    lo, hi = code_ranges()["ID83"]
    assert lo <= codes[1] < hi


def test_build_entry_codes_marks_adjacent_dbs():
    # 3 variants. var 0 and 1 are SNVs at adjacent positions p, p+1.
    # var 2 is an isolated SNV.
    var_code = np.array([10, 11, 12], dtype=np.int16)  # arbitrary SBS codes
    var_pos = np.array([100, 101, 200], dtype=np.int64)
    var_contig = np.array([0, 0, 0], dtype=np.int32)
    var_is_snv = np.array([True, True, True])
    var_ref_b = np.frombuffer(b"ACG", np.uint8).copy()  # ref base per variant
    var_alt_b = np.frombuffer(b"GTA", np.uint8).copy()  # alt base per variant

    # one sample, ploidy 1, track has all three variant indices
    data = np.array([0, 1, 2], dtype=np.int32)
    offsets = np.array([0, 3], dtype=np.int64)

    codes = build_entry_codes(
        data,
        offsets,
        var_code,
        var_pos,
        var_contig,
        var_is_snv,
        var_ref_b,
        var_alt_b,
    )
    # entry for var0 -> a DBS code (>=DBS78_OFFSET), var1 -> DBS_PARTNER, var2 unchanged
    assert codes[0] >= DBS78_OFFSET and codes[0] < DBS78_OFFSET + 78
    assert codes[0] == classify_dbs78(b"AC", b"GT")
    assert codes[1] == SENTINELS["DBS_PARTNER"]
    assert codes[2] == 12


def test_build_entry_codes_run_of_three_stays_sbs():
    # 3 adjacent SNVs at positions 100, 101, 102.
    # A run of >=3 adjacent SNVs must all stay as individual SBS — none should
    # be collapsed into a DBS or marked DBS_PARTNER.
    var_code = np.array([10, 11, 12], dtype=np.int16)
    var_pos = np.array([100, 101, 102], dtype=np.int64)
    var_contig = np.array([0, 0, 0], dtype=np.int32)
    var_is_snv = np.array([True, True, True])
    var_ref_b = np.frombuffer(b"ACG", np.uint8).copy()
    var_alt_b = np.frombuffer(b"GTA", np.uint8).copy()

    data = np.array([0, 1, 2], dtype=np.int32)
    offsets = np.array([0, 3], dtype=np.int64)

    codes = build_entry_codes(
        data,
        offsets,
        var_code,
        var_pos,
        var_contig,
        var_is_snv,
        var_ref_b,
        var_alt_b,
    )
    assert codes.tolist() == [10, 11, 12]


def test_build_entry_codes_no_false_pair_when_not_adjacent():
    var_code = np.array([10, 12], dtype=np.int16)
    var_pos = np.array([100, 105], dtype=np.int64)
    var_contig = np.array([0, 0], dtype=np.int32)
    var_is_snv = np.array([True, True])
    var_ref_b = np.frombuffer(b"AC", np.uint8).copy()
    var_alt_b = np.frombuffer(b"GT", np.uint8).copy()
    data = np.array([0, 1], dtype=np.int32)
    offsets = np.array([0, 2], dtype=np.int64)
    codes = build_entry_codes(
        data,
        offsets,
        var_code,
        var_pos,
        var_contig,
        var_is_snv,
        var_ref_b,
        var_alt_b,
    )
    assert codes.tolist() == [10, 12]


def test_build_entry_codes_no_pair_across_tracks():
    # Two SNVs at adjacent genomic positions (100, 101) on the same contig,
    # but placed in separate tracks: track 0 contains variant 0, track 1
    # contains variant 1.  The per-track loop must not pair entries across the
    # track boundary, so both codes must remain their original var_code values.
    var_code = np.array([10, 11], dtype=np.int16)
    var_pos = np.array([100, 101], dtype=np.int64)
    var_contig = np.array([0, 0], dtype=np.int32)
    var_is_snv = np.array([True, True])
    var_ref_b = np.frombuffer(b"AC", np.uint8).copy()
    var_alt_b = np.frombuffer(b"GT", np.uint8).copy()

    # data[0] -> variant 0 (track 0), data[1] -> variant 1 (track 1)
    data = np.array([0, 1], dtype=np.int32)
    offsets = np.array([0, 1, 2], dtype=np.int64)  # two single-entry tracks

    codes = build_entry_codes(
        data,
        offsets,
        var_code,
        var_pos,
        var_contig,
        var_is_snv,
        var_ref_b,
        var_alt_b,
    )
    # Neither entry should be promoted to a DBS or marked DBS_PARTNER
    assert codes.tolist() == [10, 11]
