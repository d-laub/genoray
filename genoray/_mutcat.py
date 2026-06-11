"""COSMIC mutation catalogue codebooks and code space (SBS-96, DBS-78, ID-83)."""

from __future__ import annotations

from collections.abc import Callable

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

# ---- unified int16 code space ----
SBS96_OFFSET = 0
DBS78_OFFSET = SBS96_OFFSET + len(SBS96)  # 96
ID83_OFFSET = DBS78_OFFSET + len(DBS78)  # 174
N_CODES = ID83_OFFSET + len(ID83)  # 257

SENTINELS: dict[str, int] = {
    "DBS_PARTNER": -1,  # 3' half of an adjacency doublet; never counted
    "UNCLASSIFIED": -2,  # symbolic/complex/MNV>2bp/non-ACGT
    "MISSING": -3,
}

# index -> label, for building DataFrames
SBS96_INDEX = {lbl: SBS96_OFFSET + i for i, lbl in enumerate(SBS96)}
DBS78_INDEX = {lbl: DBS78_OFFSET + i for i, lbl in enumerate(DBS78)}
ID83_INDEX = {lbl: ID83_OFFSET + i for i, lbl in enumerate(ID83)}

MUTCAT_VERSION = 1

_LABELS: dict[str, list[str]] = {"SBS96": SBS96, "DBS78": DBS78, "ID83": ID83}


def code_ranges() -> dict[str, tuple[int, int]]:
    """Half-open ``[start, end)`` code range per matrix kind."""
    return {
        "SBS96": (SBS96_OFFSET, DBS78_OFFSET),
        "DBS78": (DBS78_OFFSET, ID83_OFFSET),
        "ID83": (ID83_OFFSET, N_CODES),
    }


def labels(kind: str) -> list[str]:
    """Return the ordered label list for a given mutation category kind."""
    if kind not in _LABELS:
        raise ValueError(
            f"Unknown mutation kind {kind!r}; choose from {list(_LABELS)}."
        )
    return _LABELS[kind]


# ---- SBS-96 single-variant classifier ----
_COMP = {ord("A"): ord("T"), ord("T"): ord("A"), ord("C"): ord("G"), ord("G"): ord("C")}
_PYR = {ord("C"), ord("T")}


def _comp(b: int) -> int:
    return _COMP.get(b, b)


def classify_sbs96(five: bytes, ref: bytes, alt: bytes, three: bytes) -> int:
    """Return the SBS-96 code for one SNV, or SENTINELS['UNCLASSIFIED'].

    Each argument is expected to be a single-base bytes object (e.g. b"A").
    Empty or otherwise invalid input yields SENTINELS['UNCLASSIFIED'].
    """
    if not (five and ref and alt and three):
        return SENTINELS["UNCLASSIFIED"]
    f, r, a, t = five[0], ref[0], alt[0], three[0]
    if r not in _COMP or a not in _COMP or r == a:
        return SENTINELS["UNCLASSIFIED"]
    if f not in _COMP or t not in _COMP:
        return SENTINELS["UNCLASSIFIED"]
    if r not in _PYR:  # purine ref -> fold to reverse complement
        r, a = _comp(r), _comp(a)
        f, t = _comp(t), _comp(f)  # flanks swap and complement
    label = f"{chr(f)}[{chr(r)}>{chr(a)}]{chr(t)}"
    return SBS96_INDEX[label]


# ---- DBS-78 doublet classifier ----


def _revcomp(seq: bytes) -> bytes:
    return bytes(_comp(b) for b in reversed(seq))


def classify_dbs78(ref: bytes, alt: bytes) -> int:
    """Return the DBS-78 code for a 2bp doublet, or SENTINELS['UNCLASSIFIED'].

    Tries the literal ``REF>ALT`` first, then its reverse-complement, since
    DBS-78 collapses strand-equivalent doublets.
    """
    if len(ref) != 2 or len(alt) != 2 or ref == alt:
        return SENTINELS["UNCLASSIFIED"]
    if any(b not in _COMP for b in ref) or any(b not in _COMP for b in alt):
        return SENTINELS["UNCLASSIFIED"]
    key = f"{ref.decode()}>{alt.decode()}"
    if key in DBS78_INDEX:
        return DBS78_INDEX[key]
    rc_key = f"{_revcomp(ref).decode()}>{_revcomp(alt).decode()}"
    if rc_key in DBS78_INDEX:
        return DBS78_INDEX[rc_key]
    return SENTINELS["UNCLASSIFIED"]


# ---- ID-83 indel classifier ----


def _size_bucket(n: int) -> str:
    return "5" if n >= 5 else str(n)


def _repeat_bucket(n: int) -> int:
    return 5 if n >= 5 else n


def classify_id83(
    pos: int, ref: bytes, alt: bytes, fetch: Callable[[int, int], bytes]
) -> int:
    """Classify a single indel into one of the 83 ID channels.

    ``pos`` is the 0-based anchor (REF[0]) position. REF/ALT must share the
    anchor base (standard left-aligned VCF representation).
    """
    if len(ref) == len(alt):  # SNV or MNV, not an indel
        return SENTINELS["UNCLASSIFIED"]
    if ref[0] != alt[0]:
        return SENTINELS["UNCLASSIFIED"]  # not anchored; complex
    is_del = len(ref) > len(alt)
    indel = ref[1:] if is_del else alt[1:]
    ilen = len(indel)
    if any(b not in _COMP for b in indel):
        return SENTINELS["UNCLASSIFIED"]

    # downstream sequence begins just after the anchor (deletions) / after pos (ins)
    # The first changed base sits at pos+1.
    scan_start = pos + 1
    window = fetch(
        scan_start, scan_start + ilen * 6 + ilen
    )  # 7*ilen: room for >=5 repeats + MH scan

    # count tandem repeats of `indel` immediately downstream
    n_rep = 0
    i = 0
    while i + ilen <= len(window) and window[i : i + ilen] == indel:
        n_rep += 1
        i += ilen

    kind = "Del" if is_del else "Ins"

    if ilen == 1:
        base = indel.decode()
        # fold purine to pyrimidine for the 1bp channel
        if base in ("A", "G"):
            base = chr(_comp(ord(base)))
        rep = _repeat_bucket(n_rep - 1 if is_del else n_rep)
        return ID83_INDEX[f"1:{kind}:{base}:{rep}"]

    # >=2bp: repeat channel unless a microhomology deletion applies
    size = _size_bucket(ilen)
    if is_del:
        mh = _microhomology_len(indel, window, ilen)
        if mh > 0 and n_rep <= 1:
            mh_cap = {2: 1, 3: 2, 4: 3}.get(ilen, 5)
            return ID83_INDEX[f"{size}:Del:M:{min(mh, mh_cap)}"]
    rep = _repeat_bucket(n_rep - 1 if is_del else n_rep)
    return ID83_INDEX[f"{size}:{kind}:R:{rep}"]


def _microhomology_len(indel: bytes, downstream: bytes, ilen: int) -> int:
    """Length of partial-match microhomology between the deleted unit and the
    sequence flanking the deletion (downstream side), capped at ilen-1."""
    mh = 0
    for k in range(1, ilen):
        if downstream[:k] == indel[:k]:
            mh = max(mh, k)
    return mh
