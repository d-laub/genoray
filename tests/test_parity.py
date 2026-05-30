"""Cross-backend parity for *_with_length on the indels fixture.

VCF and PGEN are dense/variant-major: each extends to a shared variant boundary
sufficient for the worst-case haplotype to reach the query length. Their windows
are NOT byte-identical to each other — the two backends use different extension
strategies (VCF checks length every N variants per-variant; PGEN grabs doubling
batches), so one may over-include relative to the other. Both windows are
supersets of every haplotype's minimal set, however.

SparseVar is sparse/sample-major: it extends each (sample, haplotype)
independently to its own minimal length. The bridge ``_dense2sparse_with_length``
trims a dense window down to that per-haplotype-minimal form. So the meaningful,
invariant-respecting parity is:

    bridge(VCF window)  == SparseVar.read_ranges_with_length
    bridge(PGEN window) == SparseVar.read_ranges_with_length

i.e. both dense backends carry the variants needed for every haplotype, and the
bridge reconciles either one to the canonical sparse result.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from genoray import PGEN, VCF, SparseVar
from genoray._svar import _dense2sparse_with_length
from genoray._types import V_IDX_TYPE

ddir = Path(__file__).parent / "data"

# (label, contig, start, end, clamped)
REGIONS = [
    ("A_big_deletion", "chr1", 999, 1010, False),
    ("B_per_haplotype", "chr1", 1999, 2006, False),
    ("C_snp_dense", "chr1", 2999, 3030, False),
    ("D_contig_end_clamp", "chr1", 4999, 5040, True),
]


def _collect_pgen(pgen, contig, start, end):
    """Concatenate a single range's chunks into one dense window + var_idxs.

    PGEN exposes the window's global variant indices directly (3rd tuple field).
    """
    mode = PGEN.GenosPhasingDosages
    gen = pgen._chunk_ranges_with_length(contig, start, end, "1g", mode)
    genos_parts, dose_parts, idx_parts = [], [], []
    for range_ in gen:
        for chunk, _e, v_idxs in range_:
            g, _p, d = chunk
            genos_parts.append(np.asarray(g))
            dose_parts.append(np.asarray(d))
            idx_parts.append(np.asarray(v_idxs))
        break  # single range queried
    genos = np.concatenate(genos_parts, axis=-1)
    dosages = np.concatenate(dose_parts, axis=-1)
    var_idxs = np.concatenate(idx_parts)
    return genos, dosages, var_idxs


def _collect_vcf(vcf, contig, start, end):
    """Concatenate a single range's chunks into one dense window.

    phasing=False so genos is (samples, ploidy, variants) with no phasing row —
    that's exactly the shape the bridge needs (it only inspects ALT carriers).
    VCF does NOT expose variant indices, so the caller reconstructs them from the
    contiguous genomic-order window (see the test).
    """
    vcf.phasing = False
    mode = VCF.Genos16Dosages
    gen = vcf._chunk_ranges_with_length(contig, start, end, "1g", mode)
    genos_parts, dose_parts = [], []
    for range_ in gen:
        for chunk, _e, _n_ext in range_:
            gp, d = chunk
            genos_parts.append(np.asarray(gp))
            dose_parts.append(np.asarray(d))
        break
    genos = np.concatenate(genos_parts, axis=-1)
    dosages = np.concatenate(dose_parts, axis=-1)
    return genos, dosages


def _ragged_to_lists(rag):
    """Return the logical per-haplotype variant-index lists from a Ragged.

    Works for both offset conventions used in this codebase:
    - (2, n_haps) start/end pair matrix  (SparseVar.read_ranges_with_length[0])
    - (n_haps+1,) monotonic cumsum       (_dense2sparse_with_length)
    """
    offsets = np.asarray(rag.offsets)
    data = np.asarray(rag.data)
    if offsets.ndim == 2:
        return [
            data[offsets[0, i] : offsets[1, i]].tolist()
            for i in range(offsets.shape[1])
        ]
    return [data[offsets[i] : offsets[i + 1]].tolist() for i in range(len(offsets) - 1)]


def _global_attrs(svar):
    """Global 0-based starts and ILENs, indexed by global variant index."""
    v_starts = (svar.index["POS"] - 1).to_numpy()
    ilens = svar.index["ILEN"].list.first().to_numpy()
    return v_starts, ilens


@pytest.mark.parametrize("label,contig,start,end,clamped", REGIONS)
def test_pgen_window_bridges_to_svar(label, contig, start, end, clamped):
    pgen = PGEN(ddir / "indels.pgen", dosage_path=ddir / "indels.pgen")
    svar = SparseVar(ddir / "indels.pgen.svar")

    genos, dosages, var_idxs = _collect_pgen(pgen, contig, start, end)
    v_starts_g, ilens_g = _global_attrs(svar)
    idx = var_idxs.astype(np.intp)
    brag, _bdrag = _dense2sparse_with_length(
        genos.astype(np.int8),
        var_idxs,
        start,
        end,
        v_starts_g[idx].astype(np.int32),  # window-local
        ilens_g[idx].astype(np.int32),  # window-local
        dosages,
    )

    actual = svar.read_ranges_with_length(contig, start, end)[0]
    assert _ragged_to_lists(actual) == _ragged_to_lists(brag)


@pytest.mark.parametrize("label,contig,start,end,clamped", REGIONS)
def test_vcf_window_bridges_to_svar(label, contig, start, end, clamped):
    vcf = VCF(ddir / "indels.vcf.gz", dosage_field="DS")
    svar = SparseVar(ddir / "indels.vcf.svar")

    genos, dosages = _collect_vcf(vcf, contig, start, end)
    n = genos.shape[-1]

    # VCF does not expose variant indices. The window is a contiguous run of
    # variants in genomic (== global index) order starting at the first variant
    # at/after the query start, so reconstruct the global indices by searching
    # the (sorted) index POS.
    v_starts_g, ilens_g = _global_attrs(svar)
    w_start = int(np.searchsorted(v_starts_g, start))
    var_idxs = np.arange(w_start, w_start + n, dtype=V_IDX_TYPE)

    brag, _bdrag = _dense2sparse_with_length(
        genos.astype(np.int8),
        var_idxs,
        start,
        end,
        v_starts_g[w_start : w_start + n].astype(np.int32),  # window-local
        ilens_g[w_start : w_start + n].astype(np.int32),  # window-local
        dosages,
    )

    actual = svar.read_ranges_with_length(contig, start, end)[0]
    assert _ragged_to_lists(actual) == _ragged_to_lists(brag)
