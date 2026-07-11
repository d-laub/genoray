from __future__ import annotations

import numba as nb
import numpy as np
import polars as pl

from genoray._mutcat import (
    DBS78,
    DBS78_INDEX,
    DBS78_OFFSET,
    ID83,
    ID83_INDEX,
    SBS96,
    Sentinel,
    SBS96_INDEX,
    _ID1_CODE,
    _IDM_CODE,
    _IDR_CODE,
    _REF_MISMATCH,
    _dbs78_codes,
    _id83_codes_for_contig,
    _sbs96_codes,
    build_entry_codes,
    classify_dbs78,
    classify_variants,
    code_ranges,
    count_matrix,
)
from genoray._reference import Reference

from _mutcat_oracle import (
    _classify_variants_scalar,
    classify_id83,
    classify_sbs96,
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
    assert all(v < 0 for v in Sentinel)
    assert Sentinel.DBS_PARTNER != Sentinel.UNCLASSIFIED


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
    unc = Sentinel.UNCLASSIFIED
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
    assert classify_dbs78(b"AC", b"AC") == Sentinel.UNCLASSIFIED  # no change
    assert classify_dbs78(b"ACG", b"TTT") == Sentinel.UNCLASSIFIED  # >2bp
    assert classify_dbs78(b"AN", b"CA") == Sentinel.UNCLASSIFIED  # non-ACGT base


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
        classify_id83(pos=0, ref=b"A", alt=b"C", fetch=fetch) == Sentinel.UNCLASSIFIED
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
            "POS": [2, 3],  # 1-based (VCF convention); classify_variants converts
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
    assert codes[1] == Sentinel.DBS_PARTNER
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


def test_id83_2bp_deletion_ref_absent_returns_mismatch():
    # REF="CGG" -> ALT="C": deleted unit "GG". Downstream reference is "AAAA..."
    # so the deleted unit is NOT present at scan_start -> REF/reference mismatch.
    downstream = b"A" * 20

    def fetch(s: int, e: int) -> bytes:
        return downstream[: e - s]

    assert classify_id83(pos=0, ref=b"CGG", alt=b"C", fetch=fetch) == _REF_MISMATCH


def test_id83_1bp_deletion_ref_absent_returns_mismatch():
    # REF="CG" -> ALT="C": deleted unit "G". Downstream reference is "AAAA..."
    downstream = b"A" * 20

    def fetch(s: int, e: int) -> bytes:
        return downstream[: e - s]

    assert classify_id83(pos=0, ref=b"CG", alt=b"C", fetch=fetch) == _REF_MISMATCH


def test_id83_2bp_deletion_ref_present_classifies():
    # n_rep=2 means the MH guard (n_rep <= 1) doesn't fire, so this falls through
    # to the repeat branch -> 2:Del:R:1, confirming the guard doesn't reject a
    # legitimately repeated deleted unit.
    downstream = b"GG" * 2 + b"A" * 16

    def fetch(s: int, e: int) -> bytes:
        return downstream[: e - s]

    code = classify_id83(pos=0, ref=b"CGG", alt=b"C", fetch=fetch)
    assert code != _REF_MISMATCH
    assert code == ID83_INDEX["2:Del:R:1"]


def test_id83_insertion_no_repeat_not_mismatch():
    # An insertion with no downstream repeat must NOT trigger the deletion-only
    # _REF_MISMATCH guard (guards are gated on is_del).
    downstream = b"A" * 20

    def fetch(s: int, e: int) -> bytes:
        return downstream[: e - s]

    code = classify_id83(pos=0, ref=b"C", alt=b"CG", fetch=fetch)
    assert code != _REF_MISMATCH


def test_sbs96_arithmetic_matches_codebook():
    # The vectorized substitution LUT + arithmetic must reproduce SBS96_INDEX
    # for every one of the 96 labels (pyrimidine-folded form, no boundary issues).
    from genoray._mutcat import _BASE2IDX, _SUB_LUT, _BASES, _SBS_SUBS

    for sub_idx, sub in enumerate(_SBS_SUBS):
        r_char, a_char = sub[0], sub[2]
        r, a = _BASE2IDX[ord(r_char)], _BASE2IDX[ord(a_char)]
        assert _SUB_LUT[r, a] == sub_idx
        for five_idx, five in enumerate(_BASES):
            for three_idx, three in enumerate(_BASES):
                label = f"{five}[{sub}]{three}"
                code = sub_idx * 16 + five_idx * 4 + three_idx
                assert code == SBS96_INDEX[label]


def test_sbs96_codes_match_scalar_on_random_snvs():
    rng = np.random.default_rng(0)
    bases = np.frombuffer(b"ACGT", np.uint8)
    n = 500
    seq = np.frombuffer(bytes(bases[rng.integers(0, 4, 200)]), np.uint8)
    # interior positions only so flanks exist
    p0 = rng.integers(1, len(seq) - 1, n).astype(np.int64)
    ref_b = seq[p0].copy()  # use the actual reference base as REF
    alt_b = bases[rng.integers(0, 4, n)].copy()
    got = _sbs96_codes(seq, p0, ref_b, alt_b)
    for i in range(n):
        five = bytes(seq[p0[i] - 1 : p0[i]])
        three = bytes(seq[p0[i] + 1 : p0[i] + 2])
        exp = classify_sbs96(
            five, bytes(ref_b[i : i + 1]), bytes(alt_b[i : i + 1]), three
        )
        assert got[i] == exp, (i, five, ref_b[i], alt_b[i], three)


def test_dbs78_codes_match_scalar():
    bases = b"ACGT"
    ref0, ref1, alt0, alt1, exp = [], [], [], [], []
    for r0 in bases:
        for r1 in bases:
            for a0 in bases:
                for a1 in bases:
                    ref0.append(r0)
                    ref1.append(r1)
                    alt0.append(a0)
                    alt1.append(a1)
                    exp.append(classify_dbs78(bytes([r0, r1]), bytes([a0, a1])))
    got = _dbs78_codes(
        np.array(ref0, np.uint8),
        np.array(ref1, np.uint8),
        np.array(alt0, np.uint8),
        np.array(alt1, np.uint8),
    )
    assert list(got) == exp


def test_id83_luts_match_index():
    for ki, k in enumerate(("Del", "Ins")):
        for bi, b in enumerate(("C", "T")):
            for r in range(6):
                assert _ID1_CODE[ki, bi, r] == ID83_INDEX[f"1:{k}:{b}:{r}"]
        for si, s in enumerate(("2", "3", "4", "5")):
            for r in range(6):
                assert _IDR_CODE[ki, si, r] == ID83_INDEX[f"{s}:{k}:R:{r}"]
    for si, s in enumerate(("2", "3", "4", "5")):
        cap = {"2": 1, "3": 2, "4": 3, "5": 5}[s]
        for m in range(1, cap + 1):
            assert _IDM_CODE[si, m] == ID83_INDEX[f"{s}:Del:M:{m}"]


def test_id83_kernel_matches_scalar_random():
    # Build a random contig and a set of indels; compare to classify_id83.
    rng = np.random.default_rng(7)
    bases = b"ACGT"
    seq = np.frombuffer(
        bytes(np.frombuffer(bases, np.uint8)[rng.integers(0, 4, 400)]), np.uint8
    )

    def fetch(s, e, _seq=seq):
        out = np.full(e - s, ord("N"), np.uint8)
        a, b = max(s, 0), min(e, len(_seq))
        if b > a:
            out[a - s : b - s] = _seq[a:b]
        return bytes(out)

    refs, alts, p0s = [], [], []
    for _ in range(300):
        p = int(rng.integers(2, len(seq) - 10))
        anchor = bytes(seq[p : p + 1])
        size = int(rng.integers(1, 5))
        unit = bytes(np.frombuffer(bases, np.uint8)[rng.integers(0, 4, size)])
        if rng.random() < 0.5:  # deletion
            refs.append(anchor + unit)
            alts.append(anchor)
        else:  # insertion
            refs.append(anchor)
            alts.append(anchor + unit)
        p0s.append(p)

    # flat buffers for the kernel
    ref_data = np.frombuffer(b"".join(refs), np.uint8)
    alt_data = np.frombuffer(b"".join(alts), np.uint8)
    ref_off = np.concatenate(([0], np.cumsum([len(r) for r in refs]))).astype(np.int64)
    alt_off = np.concatenate(([0], np.cumsum([len(a) for a in alts]))).astype(np.int64)
    p0 = np.array(p0s, np.int64)

    got = _id83_codes_for_contig(
        seq,
        p0,
        ref_data,
        ref_off[:-1],
        (ref_off[1:] - ref_off[:-1]),
        alt_data,
        alt_off[:-1],
        (alt_off[1:] - alt_off[:-1]),
    )
    for i in range(len(refs)):
        exp = classify_id83(int(p0[i]), refs[i], alts[i], fetch)
        assert got[i] == exp, (i, refs[i], alts[i], int(p0[i]), got[i], exp)


def _random_index_and_ref(tmp_path, rng):
    import pysam

    seq = "".join(rng.choice(list("ACGT"), size=300))
    fa = tmp_path / "ref.fa"
    fa.write_text(f">chr1\n{seq}\n")
    pysam.faidx(str(fa))
    ref = Reference.from_path(fa)

    chrom, pos, refs, alts = [], [], [], []
    for _ in range(400):
        p = int(rng.integers(2, len(seq) - 8))  # 0-based interior
        kind = rng.integers(0, 4)
        anchor = seq[p]
        if kind == 0:  # SNV
            alt = rng.choice([b for b in "ACGT" if b != anchor])
            refs.append(anchor)
            alts.append(alt)
        elif kind == 1:  # native doublet
            refs.append(seq[p : p + 2])
            alts.append("".join(rng.choice(list("ACGT"), 2)))
        elif kind == 2:  # deletion
            size = int(rng.integers(1, 4))
            refs.append(seq[p : p + 1 + size])
            alts.append(anchor)
        else:  # insertion
            size = int(rng.integers(1, 4))
            refs.append(anchor)
            alts.append(anchor + "".join(rng.choice(list("ACGT"), size)))
        chrom.append("chr1")
        pos.append(p + 1)  # store 1-based

    index = pl.DataFrame(
        {"CHROM": chrom, "POS": pos, "REF": refs, "ALT": [[a] for a in alts]}
    )
    return index, ref


def test_classify_variants_matches_scalar(tmp_path):
    rng = np.random.default_rng(123)
    index, ref = _random_index_and_ref(tmp_path, rng)
    got = classify_variants(index, ref)
    exp = _classify_variants_scalar(index, ref)
    assert list(got) == list(exp)


def test_classify_variants_contig_boundary(tmp_path):
    import pysam

    fa = tmp_path / "ref.fa"
    fa.write_text(">chr1\nACGT\n")
    pysam.faidx(str(fa))
    ref = Reference.from_path(fa)
    # SNV at first and last base: no 5'/3' flank -> UNCLASSIFIED, matching scalar
    index = pl.DataFrame(
        {
            "CHROM": ["chr1", "chr1"],
            "POS": [1, 4],
            "REF": ["A", "T"],
            "ALT": [["C"], ["G"]],
        }
    )
    got = classify_variants(index, ref)
    exp = _classify_variants_scalar(index, ref)
    assert list(got) == list(exp) == [Sentinel.UNCLASSIFIED] * 2


def test_entry_codes_thread_invariant():
    rng = np.random.default_rng(5)
    n_var = 200
    var_code = rng.integers(0, 96, n_var).astype(np.int16)
    var_pos = np.sort(rng.integers(0, 10_000, n_var)).astype(np.int64)
    var_contig = np.zeros(n_var, np.int32)
    var_is_snv = np.ones(n_var, np.bool_)
    var_ref_b = np.frombuffer(b"ACGT", np.uint8)[rng.integers(0, 4, n_var)].copy()
    var_alt_b = np.frombuffer(b"ACGT", np.uint8)[rng.integers(0, 4, n_var)].copy()
    # 8 tracks over the variant indices
    data = np.tile(np.arange(n_var, dtype=np.int32), 8)
    offsets = (np.arange(9) * n_var).astype(np.int64)

    args = (
        data,
        offsets,
        var_code,
        var_pos,
        var_contig,
        var_is_snv,
        var_ref_b,
        var_alt_b,
    )
    prev = nb.get_num_threads()
    try:
        nb.set_num_threads(1)
        a = build_entry_codes(*args)
        nb.set_num_threads(max(2, prev))
        b = build_entry_codes(*args)
    finally:
        nb.set_num_threads(prev)
    assert np.array_equal(a, b)


def test_count_matrix_thread_invariant():
    rng = np.random.default_rng(9)
    n_samples, ploidy = 6, 2
    per_track = 50
    n_slots = n_samples * ploidy
    entry_codes = rng.integers(0, 96, n_slots * per_track).astype(np.int16)
    offsets = (np.arange(n_slots + 1) * per_track).astype(np.int64)
    names = [f"s{i}" for i in range(n_samples)]

    prev = nb.get_num_threads()
    try:
        nb.set_num_threads(1)
        a = count_matrix(entry_codes, offsets, ploidy, n_samples, names, "SBS96", False)
        nb.set_num_threads(max(2, prev))
        b = count_matrix(entry_codes, offsets, ploidy, n_samples, names, "SBS96", False)
    finally:
        nb.set_num_threads(prev)
    assert a.equals(b)


def test_not_annotated_sentinel_and_version():
    from genoray._mutcat import MUTCAT_VERSION

    # distinct from the existing sentinels and from MISSING (-3)
    assert Sentinel.NOT_ANNOTATED == -4
    # Guard against a future accidental duplicate value: use __members__ (which
    # includes aliases) rather than iterating the enum (which drops aliases and
    # would make this a tautology).
    assert len(set(Sentinel.__members__.values())) == len(Sentinel.__members__)
    # on-disk semantics changed -> version bumped
    assert MUTCAT_VERSION == 3


def test_classify_variants_contig_scope(tmp_path):
    import pysam
    import polars as pl
    from genoray._reference import Reference
    from genoray._mutcat import classify_variants, Sentinel

    # two contigs; chr2 is a valid SNV that WOULD classify if in scope
    fa = tmp_path / "ref.fa"
    fa.write_text(">chr1\nACGTACGTAC\n>chr2\nACGTACGTAC\n")
    pysam.faidx(str(fa))
    ref = Reference.from_path(fa)

    index = pl.DataFrame(
        {
            "CHROM": ["chr1", "chr2"],
            "POS": np.array([2, 2], dtype=np.int32),  # 1-based; 0-based idx 1 -> REF=C
            "REF": ["C", "C"],
            "ALT": [["A"], ["A"]],
        }
    )

    # scope to chr1 only: chr2 must be NOT_ANNOTATED, chr1 must be classified
    out = classify_variants(index, ref, contigs=["chr1"])
    assert out[0] >= 0  # chr1 classified to a real SBS code
    assert out[1] == Sentinel.NOT_ANNOTATED

    # contigs=None -> both classified (unchanged behavior)
    out_all = classify_variants(index, ref, contigs=None)
    assert out_all[0] >= 0 and out_all[1] >= 0
