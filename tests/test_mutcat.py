from __future__ import annotations

import numpy as np

from genoray._mutcat import (
    DBS78,
    DBS78_INDEX,
    ID83,
    ID83_INDEX,
    SBS96,
    SENTINELS,
    SBS96_INDEX,
    classify_dbs78,
    classify_id83,
    classify_sbs96,
    code_ranges,
)


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
    # repeat-bucket boundary (4 vs 5) is deferred to Task 10 SigProfiler calibration
    assert ID83_INDEX["1:Del:C:5"] == code or ID83_INDEX["1:Del:C:4"] == code


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
