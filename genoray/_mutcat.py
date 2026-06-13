"""COSMIC mutation catalogue codebooks and code space (SBS-96, DBS-78, ID-83)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numba as nb
import numpy as np
import polars as pl
from loguru import logger
from numpy.typing import NDArray

from ._reference import Reference

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

# Internal boundary signal (NOT a public sentinel): a deletion whose deleted unit
# is absent in the reference at scan_start, i.e. REF disagrees with the reference
# genome. classify_variants maps this to UNCLASSIFIED and aggregates a warning.
_REF_MISMATCH = -99

# index -> label, for building DataFrames
SBS96_INDEX = {lbl: SBS96_OFFSET + i for i, lbl in enumerate(SBS96)}
DBS78_INDEX = {lbl: DBS78_OFFSET + i for i, lbl in enumerate(DBS78)}
ID83_INDEX = {lbl: ID83_OFFSET + i for i, lbl in enumerate(ID83)}

MUTCAT_VERSION = 2

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
        if is_del and n_rep == 0:
            return _REF_MISMATCH
        rep = _repeat_bucket(n_rep - 1 if is_del else n_rep)
        return ID83_INDEX[f"1:{kind}:{base}:{rep}"]

    # >=2bp: repeat channel unless a microhomology deletion applies
    size = _size_bucket(ilen)
    if is_del:
        mh = _microhomology_len(indel, window, ilen)
        if mh > 0 and n_rep <= 1:
            mh_cap = {2: 1, 3: 2, 4: 3}.get(ilen, 5)
            return ID83_INDEX[f"{size}:Del:M:{min(mh, mh_cap)}"]
    if is_del and n_rep == 0:
        return _REF_MISMATCH
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


_BASE2IDX = np.full(256, -1, dtype=np.int64)
_BASE2IDX[[ord("A"), ord("C"), ord("G"), ord("T")]] = [0, 1, 2, 3]

# (ref_idx, alt_idx) -> SBS substitution index 0..5, for pyrimidine-folded refs.
# _SBS_SUBS order: C>A, C>G, C>T, T>A, T>C, T>G  (encoding A=0,C=1,G=2,T=3)
_SUB_LUT = np.full((4, 4), -1, dtype=np.int64)
for _si, _sub in enumerate(_SBS_SUBS):
    _SUB_LUT[_BASE2IDX[ord(_sub[0])], _BASE2IDX[ord(_sub[2])]] = _si

_UNCL = np.int16(SENTINELS["UNCLASSIFIED"])


def _sbs96_codes(
    seq: NDArray[np.uint8],
    p0: NDArray[np.int64],
    ref_b: NDArray[np.uint8],
    alt_b: NDArray[np.uint8],
) -> NDArray[np.int16]:
    """SBS-96 codes for SNVs on one contig. ``p0`` is the 0-based REF position."""
    n = len(seq)
    r = _BASE2IDX[ref_b]
    a = _BASE2IDX[alt_b]
    fpos = p0 - 1
    tpos = p0 + 1
    in_f = (fpos >= 0) & (fpos < n)
    in_t = (tpos >= 0) & (tpos < n)
    f = _BASE2IDX[seq[np.clip(fpos, 0, n - 1)]]
    t = _BASE2IDX[seq[np.clip(tpos, 0, n - 1)]]
    valid = (r >= 0) & (a >= 0) & (f >= 0) & (t >= 0) & (r != a) & in_f & in_t
    purine = (r == 0) | (r == 2)  # A or G
    rr = np.where(purine, 3 - r, r)
    aa = np.where(purine, 3 - a, a)
    # flanks swap and complement when folding: new 5' = comp(old 3'), new 3' = comp(old 5')
    ff = np.where(purine, 3 - t, f)
    tt = np.where(purine, 3 - f, t)
    sub = _SUB_LUT[np.clip(rr, 0, 3), np.clip(aa, 0, 3)]
    code = (sub * 16 + ff * 4 + tt).astype(np.int16)
    return np.where(valid, code, _UNCL)


def _build_dbs_table() -> np.ndarray:
    """tbl[r0, r1, a0, a1] -> DBS-78 code or UNCLASSIFIED for doublets not in
    the (folded) catalogue. Bases encoded A=0,C=1,G=2,T=3."""
    tbl = np.full((4, 4, 4, 4), SENTINELS["UNCLASSIFIED"], dtype=np.int16)
    bases = b"ACGT"
    for r0 in range(4):
        for r1 in range(4):
            for a0 in range(4):
                for a1 in range(4):
                    ref = bytes([bases[r0], bases[r1]])
                    alt = bytes([bases[a0], bases[a1]])
                    tbl[r0, r1, a0, a1] = classify_dbs78(ref, alt)
    return tbl


_DBS_TABLE = _build_dbs_table()
_DBS_PARTNER = SENTINELS["DBS_PARTNER"]


@nb.njit(nogil=True, cache=True)
def _entry_codes_kernel(
    data: NDArray[np.int32],
    offsets: NDArray[np.int64],
    var_code: NDArray[np.int16],
    var_pos: NDArray[np.int64],
    var_contig: NDArray[np.int32],
    var_is_snv: NDArray[np.bool_],
    ref_idx: NDArray[np.int64],
    alt_idx: NDArray[np.int64],
    dbs_table: NDArray[np.int16],
    out: NDArray[np.int16],
    dbs_partner: np.int16,
):
    for slot in range(len(offsets) - 1):
        o_s, o_e = offsets[slot], offsets[slot + 1]
        j = o_s
        while j < o_e:
            v = data[j]
            out[j] = var_code[v]
            # try to pair with the next entry in this track
            if j + 1 < o_e and var_is_snv[v]:
                w = data[j + 1]
                if (
                    var_is_snv[w]
                    and var_contig[v] == var_contig[w]
                    and var_pos[w] - var_pos[v] == 1
                ):
                    # isolated pair only: neither a forward nor a backward
                    # adjacent SNV must exist; otherwise this pair is part of
                    # a run of >=3 adjacent SNVs and must stay individual SBS.
                    isolated = True
                    if j + 2 < o_e:
                        x = data[j + 2]
                        if (
                            var_is_snv[x]
                            and var_contig[w] == var_contig[x]
                            and var_pos[x] - var_pos[w] == 1
                        ):
                            isolated = False
                    if isolated and j > o_s:
                        p = data[j - 1]
                        if (
                            var_is_snv[p]
                            and var_contig[p] == var_contig[v]
                            and var_pos[v] - var_pos[p] == 1
                        ):
                            isolated = False
                    if isolated:
                        ri0 = ref_idx[v]
                        ri1 = ref_idx[w]
                        ai0 = alt_idx[v]
                        ai1 = alt_idx[w]
                        if ri0 >= 0 and ri1 >= 0 and ai0 >= 0 and ai1 >= 0:
                            code = dbs_table[ri0, ri1, ai0, ai1]
                            out[j] = code
                            out[j + 1] = dbs_partner
                            j += 2
                            continue
            j += 1


def build_entry_codes(
    data: NDArray[np.int32],
    offsets: NDArray[np.int64],
    var_code: NDArray[np.int16],
    var_pos: NDArray[np.int64],
    var_contig: NDArray[np.int32],
    var_is_snv: NDArray[np.bool_],
    var_ref_b: NDArray[np.uint8],
    var_alt_b: NDArray[np.uint8],
) -> NDArray[np.int16]:
    """Return int16 per-entry codes aligned to ``data`` (genos.data)."""
    ref_idx = _BASE2IDX[var_ref_b]
    alt_idx = _BASE2IDX[var_alt_b]
    out = np.empty(len(data), dtype=np.int16)
    _entry_codes_kernel(
        data.astype(np.int32),
        offsets.astype(np.int64),
        var_code,
        var_pos.astype(np.int64),
        var_contig.astype(np.int32),
        var_is_snv.astype(np.bool_),
        ref_idx,
        alt_idx,
        _DBS_TABLE,
        out,
        np.int16(_DBS_PARTNER),
    )
    return out


@nb.njit(nogil=True, cache=True)
def _count_kernel(
    data_codes: NDArray[np.int16],
    offsets: NDArray[np.int64],
    ploidy: np.int64,
    n_samples: np.int64,
    n_codes: np.int64,
    per_sample: np.bool_,
    out: NDArray[np.int64],
) -> None:
    """out[sample, code] accumulator over genotype entries.

    ``data_codes`` is the per-entry int16 code array (aligned to genos.data).
    When ``per_sample`` is True, a code is counted at most once per sample.
    """
    for slot in range(len(offsets) - 1):
        sample = slot // ploidy
        o_s, o_e = offsets[slot], offsets[slot + 1]
        for j in range(o_s, o_e):
            code = data_codes[j]
            if code < 0 or code >= n_codes:
                continue
            if per_sample:
                if out[sample, code] == 0:
                    out[sample, code] = 1
            else:
                out[sample, code] += 1


def count_matrix(
    entry_codes: np.ndarray,
    offsets: np.ndarray,
    ploidy: int,
    n_samples: int,
    sample_names: list[str],
    kind: Literal["SBS96", "DBS78", "ID83"],
    per_sample: bool,
) -> "pl.DataFrame":
    counts = np.zeros((n_samples, N_CODES), dtype=np.int64)
    _count_kernel(
        entry_codes.astype(np.int16),
        offsets.astype(np.int64),
        np.int64(ploidy),
        np.int64(n_samples),
        np.int64(N_CODES),
        np.bool_(per_sample),
        counts,
    )
    lo, hi = code_ranges()[kind]
    block = counts[:, lo:hi]  # (n_samples, n_categories)
    out: dict[str, Any] = {"MutationType": labels(kind)}
    for s_i, name in enumerate(sample_names):
        out[name] = block[s_i]
    return pl.DataFrame(out)


def classify_variants(index: pl.DataFrame, reference: Reference) -> np.ndarray:
    """Return an int16 array of intrinsic mutation codes, one per row of ``index``.

    ``index`` must have columns CHROM, POS (1-based int, VCF convention), REF
    (str), ALT (List[str]; first ALT used). POS is converted to a 0-based
    reference coordinate internally. Reference context is fetched per contig.
    """
    chrom = index["CHROM"].to_numpy()
    pos = index["POS"].to_numpy().astype(np.int64)
    ref = index["REF"].to_list()
    alt0 = index["ALT"].list.first().to_list()

    out = np.full(index.height, SENTINELS["UNCLASSIFIED"], dtype=np.int16)

    n_mismatch = 0
    mismatch_examples: list[str] = []

    for i in range(index.height):
        r = ref[i]
        a = alt0[i]
        if a is None or r is None:
            continue
        rb, ab = r.encode(), a.encode()
        c = str(chrom[i])
        p = int(pos[i]) - 1  # index POS is 1-based (VCF); reference.fetch is 0-based

        if len(rb) == 1 and len(ab) == 1:  # SNV
            five = reference.fetch(c, p - 1, p).tobytes()
            three = reference.fetch(c, p + 1, p + 2).tobytes()
            out[i] = classify_sbs96(five, rb, ab, three)
        elif len(rb) == 2 and len(ab) == 2:  # native MNV doublet
            out[i] = classify_dbs78(rb, ab)
        elif len(rb) != len(ab):  # indel
            _c = c  # capture per-iteration contig for closure

            def _fetch(s: int, e: int, _c: str = _c) -> bytes:
                return reference.fetch(_c, s, e).tobytes()

            code = classify_id83(p, rb, ab, _fetch)
            if code == _REF_MISMATCH:
                n_mismatch += 1
                if len(mismatch_examples) < 5:
                    mismatch_examples.append(f"{c}:{int(pos[i])}")
                out[i] = SENTINELS["UNCLASSIFIED"]
            else:
                out[i] = code
        # else: MNV>2bp or symbolic -> stays UNCLASSIFIED

    if n_mismatch:
        examples = ", ".join(mismatch_examples)
        logger.warning(
            f"{n_mismatch}/{index.height} deletions have REF disagreeing with the "
            f"reference genome at their position (e.g. {examples}) — wrong reference "
            "build? These were marked UNCLASSIFIED."
        )

    return out
