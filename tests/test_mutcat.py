from __future__ import annotations

from genoray._mutcat import (
    DBS78,
    ID83,
    SBS96,
    SENTINELS,
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
