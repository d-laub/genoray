"""Scalar, per-record mutation-catalogue classifiers — the slow ground-truth
oracle used only by tests to validate the shipped vectorized path
(``genoray._mutcat.classify_variants``). Relocated out of shipped code in SP-7.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import polars as pl
from loguru import logger

from genoray._mutcat import (
    ID83_INDEX,
    SBS96_INDEX,
    Sentinel,
    _COMP,
    _REF_MISMATCH,
    _comp,
    classify_dbs78,
)
from genoray._reference import Reference

_PYR = {ord("C"), ord("T")}


def classify_sbs96(five: bytes, ref: bytes, alt: bytes, three: bytes) -> int:
    """Return the SBS-96 code for one SNV, or Sentinel.UNCLASSIFIED.

    Each argument is expected to be a single-base bytes object (e.g. b"A").
    Empty or otherwise invalid input yields Sentinel.UNCLASSIFIED.
    """
    if not (five and ref and alt and three):
        return Sentinel.UNCLASSIFIED
    f, r, a, t = five[0], ref[0], alt[0], three[0]
    if r not in _COMP or a not in _COMP or r == a:
        return Sentinel.UNCLASSIFIED
    if f not in _COMP or t not in _COMP:
        return Sentinel.UNCLASSIFIED
    if r not in _PYR:  # purine ref -> fold to reverse complement
        r, a = _comp(r), _comp(a)
        f, t = _comp(t), _comp(f)  # flanks swap and complement
    label = f"{chr(f)}[{chr(r)}>{chr(a)}]{chr(t)}"
    return SBS96_INDEX[label]


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
        return Sentinel.UNCLASSIFIED
    if ref[0] != alt[0]:
        return Sentinel.UNCLASSIFIED  # not anchored; complex
    is_del = len(ref) > len(alt)
    indel = ref[1:] if is_del else alt[1:]
    ilen = len(indel)
    if any(b not in _COMP for b in indel):
        return Sentinel.UNCLASSIFIED

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


def _classify_variants_scalar(index: pl.DataFrame, reference: Reference) -> np.ndarray:
    """Return an int16 array of intrinsic mutation codes, one per row of ``index``.

    ``index`` must have columns CHROM, POS (1-based int, VCF convention), REF
    (str), ALT (List[str]; first ALT used). POS is converted to a 0-based
    reference coordinate internally. Reference context is fetched per contig.
    """
    chrom = index["CHROM"].to_numpy()
    pos = index["POS"].to_numpy().astype(np.int64)
    ref = index["REF"].to_list()
    alt0 = index["ALT"].list.first().to_list()

    out = np.full(index.height, Sentinel.UNCLASSIFIED, dtype=np.int16)

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
                out[i] = Sentinel.UNCLASSIFIED
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
