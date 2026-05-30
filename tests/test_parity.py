"""Cross-backend parity for *_with_length on the indels fixture.

PGEN and VCF are dense/variant-major: they extend to one shared boundary, so
their dense outputs must be identical. SparseVar extends each haplotype
independently; its output must equal _dense2sparse_with_length applied to the
dense window (the parity bridge).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from genoray import PGEN, VCF, SparseVar
from genoray._svar import _dense2sparse_with_length

ddir = Path(__file__).parent / "data"

# (label, contig, start, end, clamped)
REGIONS = [
    ("A_big_deletion", "chr1", 999, 1010, False),
    ("B_per_haplotype", "chr1", 1999, 2006, False),
    ("C_snp_dense", "chr1", 2999, 3030, False),
    ("D_contig_end_clamp", "chr1", 4999, 5040, True),
]


def _collect_pgen(pgen, contig, start, end):
    """Concatenate a single range's chunks into one dense window + var_idxs."""
    mode = PGEN.GenosPhasingDosages
    gen = pgen._chunk_ranges_with_length(contig, start, end, "1g", mode)
    genos_parts, dose_parts, idx_parts = [], [], []
    end_pos = None
    for range_ in gen:
        for chunk, e, v_idxs in range_:
            g, p, d = chunk
            genos_parts.append(np.asarray(g))
            dose_parts.append(np.asarray(d))
            idx_parts.append(np.asarray(v_idxs))
            end_pos = e
        break  # single range queried
    genos = np.concatenate(genos_parts, axis=-1)
    dosages = np.concatenate(dose_parts, axis=-1)
    var_idxs = np.concatenate(idx_parts)
    return genos, dosages, var_idxs, end_pos


def _collect_vcf(vcf, contig, start, end):
    # phasing=False so genos shape is (samples, ploidy, variants) without the
    # phasing indicator row, avoiding the (2,2) vs (2,3) broadcast error in
    # _ext_genos_dosages_with_length when phasing=True.
    vcf.phasing = False
    mode = VCF.Genos16Dosages
    gen = vcf._chunk_ranges_with_length(contig, start, end, "1g", mode)
    genos_parts, dose_parts = [], []
    end_pos = None
    for range_ in gen:
        for chunk, e, _n_ext in range_:
            gp, d = chunk
            # gp shape: (samples, ploidy, variants) — no phasing row when phasing=False
            genos_parts.append(np.asarray(gp))
            dose_parts.append(np.asarray(d))
            end_pos = e
        break
    genos = np.concatenate(genos_parts, axis=-1)
    dosages = np.concatenate(dose_parts, axis=-1)
    return genos, dosages, end_pos


def _ragged_to_lists(rag):
    """Return the logical per-haplotype variant-index lists from a Ragged.

    Works for both offset conventions used in this codebase:
    - (2, n_haps) start/end pair matrix  (SparseVar.read_ranges_with_length[0])
    - (n_haps+1,) monotonic cumsum       (_dense2sparse_with_length)
    """
    offsets = np.asarray(rag.offsets)
    data = np.asarray(rag.data)
    if offsets.ndim == 2:
        # (2, n_haps) convention: row 0 = starts, row 1 = ends
        return [
            data[offsets[0, i] : offsets[1, i]].tolist()
            for i in range(offsets.shape[1])
        ]
    else:
        # (n_haps+1,) monotonic convention
        return [
            data[offsets[i] : offsets[i + 1]].tolist() for i in range(len(offsets) - 1)
        ]


@pytest.mark.parametrize("label,contig,start,end,clamped", REGIONS)
def test_pgen_equals_vcf_dense(label, contig, start, end, clamped):
    pgen = PGEN(ddir / "indels.pgen", dosage_path=ddir / "indels.pgen")
    vcf = VCF(ddir / "indels.vcf.gz", dosage_field="DS")

    pg, pd, _vidx, _pe = _collect_pgen(pgen, contig, start, end)
    vg, vd, _ve = _collect_vcf(vcf, contig, start, end)

    np.testing.assert_array_equal(pg.astype(np.int16), vg.astype(np.int16))
    np.testing.assert_allclose(pd, vd, rtol=1e-4, equal_nan=True)


@pytest.mark.parametrize("label,contig,start,end,clamped", REGIONS)
def test_svar_equals_bridge(label, contig, start, end, clamped):
    pgen = PGEN(ddir / "indels.pgen", dosage_path=ddir / "indels.pgen")
    svar = SparseVar(ddir / "indels.pgen.svar")

    genos, dosages, var_idxs, _end = _collect_pgen(pgen, contig, start, end)
    # GLOBAL per-variant attributes, then slice to WINDOW-LOCAL via var_idxs
    v_starts_global = (svar.index["POS"] - 1).to_numpy()
    ilens_global = svar.index["ILEN"].list.first().to_numpy()
    var_idxs_int = var_idxs.astype(np.intp)
    v_starts_window = v_starts_global[var_idxs_int].astype(np.int32)
    ilens_window = ilens_global[var_idxs_int].astype(np.int32)

    bridged = _dense2sparse_with_length(
        genos.astype(np.int8),
        var_idxs,
        start,
        end,
        v_starts_window,
        ilens_window,
        dosages,
    )
    brag, bdrag = bridged

    actual = svar.read_ranges_with_length(contig, start, end)
    # actual shape: (1 range, n_samples, ploidy, None); bridge shape: (n_samples, ploidy, None)
    # Index out the single range before comparing.
    actual_r = actual[0]

    # The two Ragged objects may use different internal offset conventions
    # ((2, n_haps) start/end pairs vs (n_haps+1,) cumsum), so compare the
    # logical per-haplotype variant-index lists directly.
    assert _ragged_to_lists(actual_r) == _ragged_to_lists(brag)
