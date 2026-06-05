"""Adapt vcfixture GroundTruth to genoray's array conventions.

GroundTruth.genotypes is (records, samples, ploidy); genoray reads are
(samples, ploidy[+phasing], variants). These helpers transpose/slice truth so
tests assert genoray output against a decoded oracle instead of literals.

The oracle supplies variant VALUES. It does NOT model genoray's range-selection
logic (spanning deletions): callers pass the variant indices a query returns.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from vcfixture import GroundTruth

Index = slice | Sequence[int] | NDArray[np.intp]


def genos(truth: GroundTruth, idx: Index) -> NDArray[np.int32]:
    """(samples, ploidy, variants), -1 = missing."""
    sub = truth.genotypes[idx]  # (v, s, p)
    return np.ascontiguousarray(sub.transpose(1, 2, 0))


def phasing(truth: GroundTruth, idx: Index) -> NDArray[np.bool_]:
    """(samples, variants) — True where fully phased."""
    sub = truth.phasing[idx]  # (v, s)
    return np.ascontiguousarray(sub.transpose(1, 0))


def dosages(truth: GroundTruth, idx: Index, field: str = "DS") -> NDArray[np.float32]:
    """(samples, variants); missing/"."/NaN -> np.nan.

    Mirrors genoray's dosage_field convention: one scalar per (sample, variant).
    For Number=A fields the first ALT's dosage is used (the biallelic fixture is
    1:1; the multiallelic fixture has no PGEN dosage path).
    """
    fmt = truth.format
    n_rec = len(fmt)
    rec_ids = list(range(n_rec))[idx] if isinstance(idx, slice) else list(idx)
    n_smp = len(truth.samples)
    out = np.full((n_smp, len(rec_ids)), np.nan, np.float32)
    for vi, ri in enumerate(rec_ids):
        per_sample = fmt[ri]
        for si in range(n_smp):
            val = per_sample[si].get(field)
            if isinstance(val, (list, tuple)):
                val = val[0] if len(val) else None
            if val is None:
                continue
            fval = float(val)
            out[si, vi] = np.nan if np.isnan(fval) else fval
    return out


def split_phased(gp: NDArray) -> tuple[NDArray, NDArray[np.bool_]]:
    """Split genoray phased output (s, ploidy+1, v) into (genos, phasing)."""
    g, p = np.array_split(gp, 2, axis=1)
    return g, p.squeeze(1).astype(bool)


def expected_ilen(truth: GroundTruth, idx: Index) -> list[list[int | None]]:
    """Per-record list of per-ALT expected ILEN from the vcfixture oracle.

    DEL -> -|svlen|, INS/DUP -> +|svlen|, else (unsupported type or no svlen)
    -> None. Mirrors genoray.exprs.symbolic_ilen for the symbolic case. For
    literal ALTs, ILEN is len(ALT) - len(REF).

    Note: AlleleTruth has no .alt attribute; literal sequences are read from
    truth.alts[ri][alt_idx] (list[list[str]] on GroundTruth).
    """
    n_rec = len(truth.pos)
    rec_ids = list(range(n_rec))[idx] if isinstance(idx, slice) else list(idx)
    out: list[list[int | None]] = []
    for ri in rec_ids:
        ref = truth.ref[ri]
        row: list[int | None] = []
        for ai, at in enumerate(truth.alts_truth[ri]):
            if at.is_sequence:
                alt_str = truth.alts[ri][ai]
                row.append(len(alt_str) - len(ref))
            elif at.sv_type == "DEL" and at.svlen is not None:
                row.append(-abs(int(at.svlen)))
            elif at.sv_type in ("INS", "DUP") and at.svlen is not None:
                row.append(abs(int(at.svlen)))
            else:
                row.append(None)
        out.append(row)
    return out
