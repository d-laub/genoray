from __future__ import annotations

from pathlib import Path

import numpy as np
from seqpro.rag import Ragged

from genoray import VCF, SparseVar

ddir = Path(__file__).parent / "data"
VCF_PATH = (
    ddir / "biallelic.vcf.gz"
)  # 2 samples (sample1, sample2), contigs chr1/chr2/chr3

_SENTINEL = np.iinfo(np.int64).max


def _rag_to_set(r: Ragged) -> set[int]:
    """Safely convert a 1-D Ragged entry to a set of ints, returning empty set on sentinels."""
    offsets = r._layout.offsets
    # offsets is a list of arrays; the last level is starts/stops as [[start],[stop]]
    arr = offsets[-1] if isinstance(offsets, (list, tuple)) else offsets
    if arr.flat[0] == _SENTINEL:
        return set()
    return set(r.to_numpy().flatten().tolist())


def _haploid_call_sets(sv: SparseVar) -> dict[tuple[str, int], set[int]]:
    """Map (contig, sample_idx) -> set of global variant indices present (ploidy=1)."""
    out: dict[tuple[str, int], set[int]] = {}
    for c in sv.contigs:
        rag = sv.read_ranges(c)  # (1, n_samples, 1, ~v)
        for i in range(sv.n_samples):
            out[(c, i)] = _rag_to_set(rag[0, i, 0])
    return out


def _diploid_union_call_sets(sv: SparseVar) -> dict[tuple[str, int], set[int]]:
    """Map (contig, sample_idx) -> union of both haplotypes' variant indices (ploidy=2)."""
    out: dict[tuple[str, int], set[int]] = {}
    for c in sv.contigs:
        rag = sv.read_ranges(c)  # (1, n_samples, 2, ~v)
        for i in range(sv.n_samples):
            hap0 = _rag_to_set(rag[0, i, 0])
            hap1 = _rag_to_set(rag[0, i, 1])
            out[(c, i)] = hap0 | hap1
    return out


def test_from_vcf_haploid_metadata_and_or(tmp_path: Path):
    dip = tmp_path / "dip.svar"
    hap = tmp_path / "hap.svar"
    SparseVar.from_vcf(dip, VCF(VCF_PATH), max_mem="1g", overwrite=True)
    SparseVar.from_vcf(hap, VCF(VCF_PATH), max_mem="1g", overwrite=True, haploid=True)

    sv_dip = SparseVar(dip)
    sv_hap = SparseVar(hap)

    # metadata + shape
    assert sv_hap.ploidy == 1
    assert sv_hap.genos.shape[1] == 1
    # same variant set as the diploid build (no variants gained/lost by collapse)
    assert sv_hap.n_variants == sv_dip.n_variants

    # OR invariant: haploid call set == union of the two diploid haplotype sets
    assert _haploid_call_sets(sv_hap) == _diploid_union_call_sets(sv_dip)


def _dose_pairs(sv: SparseVar) -> dict[int, set[tuple[int, float]]]:
    """Map sample_idx -> set of (variant_idx, rounded dosage) over all contigs."""
    out: dict[int, set[tuple[int, float]]] = {i: set() for i in range(sv.n_samples)}
    for c in sv.contigs:
        rag = sv.read_ranges(c)  # record array: .genos, .dosages
        for i in range(sv.n_samples):
            for p in range(sv.ploidy):
                g_entry = rag.genos[0, i, p]
                offsets = g_entry._layout.offsets
                arr = offsets[-1] if isinstance(offsets, (list, tuple)) else offsets
                if arr.flat[0] == _SENTINEL:
                    continue
                vi = g_entry.to_numpy().flatten()
                ds = rag.dosages[0, i, p].to_numpy().flatten()
                for v, d in zip(vi.tolist(), ds.tolist()):
                    out[i].add((int(v), round(float(d), 4)))
    return out


def test_from_vcf_haploid_with_dosages(tmp_path: Path):
    dip = tmp_path / "dip.svar"
    hap = tmp_path / "hap.svar"
    SparseVar.from_vcf(
        dip,
        VCF(VCF_PATH, dosage_field="DS"),
        max_mem="1g",
        overwrite=True,
        with_dosages=True,
    )
    SparseVar.from_vcf(
        hap,
        VCF(VCF_PATH, dosage_field="DS"),
        max_mem="1g",
        overwrite=True,
        with_dosages=True,
        haploid=True,
    )

    sv_dip = SparseVar(dip, fields=["dosages"])
    sv_hap = SparseVar(hap, fields=["dosages"])

    # genos and dosages share offsets, so equal entry counts per build
    rag = sv_hap.read_ranges(sv_hap.contigs[0])
    assert rag.genos.data.shape == rag.dosages.data.shape
    assert rag.dosages.data.dtype == np.float32

    # every (variant, dosage) pair present in the haploid build also exists in the
    # diploid build for that sample (dosage is per-(sample,variant), so collapse
    # preserves the value; a hom-ALT call simply stops being double-stored).
    dip_pairs = _dose_pairs(sv_dip)
    hap_pairs = _dose_pairs(sv_hap)
    for i in range(sv_hap.n_samples):
        assert hap_pairs[i].issubset(dip_pairs[i])


def test_from_vcf_haploid_sample_subset_af(tmp_path: Path):
    hap = tmp_path / "hap_sub.svar"
    SparseVar.from_vcf(
        hap,
        VCF(VCF_PATH),
        max_mem="1g",
        overwrite=True,
        haploid=True,
        samples=["sample1"],
    )
    sv = SparseVar(hap, attrs="AF")
    assert sv.ploidy == 1
    assert list(sv.available_samples) == ["sample1"]
    # haploid AF denominator is n_out * 1 = 1 survivor sample, so every surviving
    # variant (MAC>0) has AF in (0, 1]; with one haploid sample AF is exactly 1.0.
    afs = sv.index["AF"].to_numpy()
    assert afs.size == sv.n_variants
    assert np.all(afs > 0.0)
    assert np.all(afs <= 1.0)
