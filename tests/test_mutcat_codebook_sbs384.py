from __future__ import annotations

from genoray._mutcat import codebook as cb


def test_sbs384_layout():
    assert cb.SBS384_OFFSET == 257
    assert cb.N_CODES == 641
    assert len(cb.SBS384) == 384
    # [T, U, N, B] blocks of 96, each label "<strand>:<sbs96 label>"
    assert cb.SBS384[0] == f"T:{cb.SBS96[0]}"
    assert cb.SBS384[96] == f"U:{cb.SBS96[0]}"
    assert cb.SBS384[192] == f"N:{cb.SBS96[0]}"
    assert cb.SBS384[288] == f"B:{cb.SBS96[0]}"


def test_sbs192_is_tu_view():
    assert cb.labels("SBS384") == cb.SBS384
    assert cb.labels("SBS192") == cb.SBS384[:192]
    assert cb.code_ranges()["SBS384"] == (257, 641)
    assert cb.code_ranges()["SBS192"] == (257, 449)
    assert cb.code_ranges()["SBS96"] == (0, 96)


def test_version_bumped():
    assert cb.MUTCAT_VERSION == 4
