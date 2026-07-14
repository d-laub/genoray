"""COSMIC mutation catalogue codebooks and code space (SBS-96, DBS-78, ID-83)."""

from __future__ import annotations

from enum import IntEnum
from typing import Literal

# ---- SBS-96 (COSMIC order: substitution outer, 5' base, 3' base inner) ----
_SBS_SUBS = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]
_BASES = ["A", "C", "G", "T"]
SBS96: list[str] = [
    f"{five}[{sub}]{three}" for sub in _SBS_SUBS for five in _BASES for three in _BASES
]

# ---- DBS-78 (canonical COSMIC list) ----
DBS78: list[str] = [
    "AC>CA",
    "AC>CG",
    "AC>CT",
    "AC>GA",
    "AC>GG",
    "AC>GT",
    "AC>TA",
    "AC>TG",
    "AC>TT",
    "AT>CA",
    "AT>CC",
    "AT>CG",
    "AT>GA",
    "AT>GC",
    "AT>TA",
    "CC>AA",
    "CC>AG",
    "CC>AT",
    "CC>GA",
    "CC>GG",
    "CC>GT",
    "CC>TA",
    "CC>TG",
    "CC>TT",
    "CG>AT",
    "CG>GC",
    "CG>GT",
    "CG>TA",
    "CG>TC",
    "CG>TT",
    "CT>AA",
    "CT>AC",
    "CT>AG",
    "CT>GA",
    "CT>GC",
    "CT>GG",
    "CT>TA",
    "CT>TC",
    "CT>TG",
    "GC>AA",
    "GC>AG",
    "GC>AT",
    "GC>CA",
    "GC>CG",
    "GC>TA",
    "TA>AT",
    "TA>CG",
    "TA>CT",
    "TA>GC",
    "TA>GG",
    "TA>GT",
    "TC>AA",
    "TC>AG",
    "TC>AT",
    "TC>CA",
    "TC>CG",
    "TC>CT",
    "TC>GA",
    "TC>GG",
    "TC>GT",
    "TG>AA",
    "TG>AC",
    "TG>AT",
    "TG>CA",
    "TG>CC",
    "TG>CT",
    "TG>GA",
    "TG>GC",
    "TG>GT",
    "TT>AA",
    "TT>AC",
    "TT>AG",
    "TT>CA",
    "TT>CC",
    "TT>CG",
    "TT>GA",
    "TT>GC",
    "TT>GG",
]


def _build_id83() -> list[str]:
    out: list[str] = []
    # 1bp del/ins, by base C/T, repeat count 0..5(+)
    for kind in ("Del", "Ins"):
        for base in ("C", "T"):
            for r in range(6):
                out.append(f"1:{kind}:{base}:{r}")
    # >1bp del/ins at repeats, size 2..5(+), repeat 0..5(+)
    for kind in ("Del", "Ins"):
        for size in ("2", "3", "4", "5"):
            for r in range(6):
                out.append(f"{size}:{kind}:R:{r}")
    # microhomology deletions
    for size, mh_max in (("2", 1), ("3", 2), ("4", 3), ("5", 5)):
        for m in range(1, mh_max + 1):
            out.append(f"{size}:Del:M:{m}")
    return out


ID83: list[str] = _build_id83()

assert len(SBS96) == 96 and len(DBS78) == 78 and len(ID83) == 83

# ---- SBS-384 (transcriptional strand bias): SigProfiler [T, U, N, B] blocks ----
# T=Transcribed, U=Untranscribed, N=Nontranscribed (intergenic), B=Bidirectional.
# Label form "<strand>:<sbs96 label>", e.g. "T:A[C>A]A". SBS-192 is the {T,U} view
# (SBS384[:192]); no separate stored range.
_STRANDS = ["T", "U", "N", "B"]
SBS384: list[str] = [f"{s}:{lbl}" for s in _STRANDS for lbl in SBS96]

assert len(SBS384) == 384

# ---- unified int16 code space ----
SBS96_OFFSET = 0
DBS78_OFFSET = SBS96_OFFSET + len(SBS96)  # 96
ID83_OFFSET = DBS78_OFFSET + len(DBS78)  # 174
SBS384_OFFSET = ID83_OFFSET + len(ID83)  # 257
N_CODES = SBS384_OFFSET + len(SBS384)  # 641


class Sentinel(IntEnum):
    """Negative sentinel codes in the int16 mutation-code space (closed set)."""

    DBS_PARTNER = -1  # 3' half of an adjacency doublet; never counted
    UNCLASSIFIED = -2  # symbolic/complex/MNV>2bp/non-ACGT
    MISSING = -3
    NOT_ANNOTATED = -4  # entry on a contig excluded from the annotation scope


# index -> label, for building DataFrames
SBS96_INDEX = {lbl: SBS96_OFFSET + i for i, lbl in enumerate(SBS96)}
DBS78_INDEX = {lbl: DBS78_OFFSET + i for i, lbl in enumerate(DBS78)}
ID83_INDEX = {lbl: ID83_OFFSET + i for i, lbl in enumerate(ID83)}

MUTCAT_VERSION = 4

Kind = Literal["SBS96", "DBS78", "ID83", "SBS192", "SBS384"]

_LABELS: dict[str, list[str]] = {
    "SBS96": SBS96,
    "DBS78": DBS78,
    "ID83": ID83,
    "SBS384": SBS384,
    "SBS192": SBS384[:192],
}


def code_ranges() -> dict[Kind, tuple[int, int]]:
    """Half-open ``[start, end)`` code range per matrix kind.

    ``SBS192`` is the {T, U} sub-view of the ``SBS384`` block, so its range is
    the first 192 codes of that block.
    """
    return {
        "SBS96": (SBS96_OFFSET, DBS78_OFFSET),
        "DBS78": (DBS78_OFFSET, ID83_OFFSET),
        "ID83": (ID83_OFFSET, SBS384_OFFSET),
        "SBS384": (SBS384_OFFSET, N_CODES),
        "SBS192": (SBS384_OFFSET, SBS384_OFFSET + 192),
    }


def labels(kind: Kind) -> list[str]:
    """Return the ordered label list for a given mutation category kind."""
    if kind not in _LABELS:
        raise ValueError(
            f"Unknown mutation kind {kind!r}; choose from {list(_LABELS)}."
        )
    return _LABELS[kind]
