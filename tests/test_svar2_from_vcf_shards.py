from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import numpy as np
import pytest

from genoray import SparseVar2


_REF = "CAAAATCAGAGT"


def _write_ref(d: Path) -> Path:
    ref = d / "ref.fa"
    ref.write_text(f">chr1\n{_REF}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)
    return ref


def _write_vcf(
    d: Path,
    name: str,
    rows: str,
    samples: tuple[str, ...],
    *,
    suffix: str = ".vcf.gz",
) -> Path:
    header = (
        "##fileformat=VCFv4.2\n"
        f"##contig=<ID=chr1,length={len(_REF)}>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
        + "\t".join(samples)
        + "\n"
    )
    plain = d / f"{name}.vcf"
    plain.write_text(header + rows)
    gz = d / f"{name}{suffix}"
    with open(gz, "wb") as fh:
        subprocess.run(["bgzip", "-c", str(plain)], check=True, stdout=fh)
    subprocess.run(["bcftools", "index", str(gz)], check=True)
    return gz


def test_from_vcf_shards_accepts_vcf_bgz_suffix(tmp_path: Path) -> None:
    ref = _write_ref(tmp_path)
    source = _write_vcf(
        tmp_path,
        "native",
        "chr1\t8\t.\tA\tT\t.\t.\t.\tGT\t1|0\t0|1\n",
        ("S0", "S1"),
        suffix=".vcf.bgz",
    )

    out = tmp_path / "native.svar2"
    SparseVar2.from_vcf_shards(
        out,
        [(source, ("chr1", 0, len(_REF)))],
        ref,
        threads=2,
    )

    assert SparseVar2(out).available_samples == ["S0", "S1"]


def test_from_vcf_shards_closes_each_header_before_opening_next(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cyvcf2 import VCF as RealVCF

    class TrackingVCF:
        active = 0
        max_active = 0

        def __init__(self, path: str) -> None:
            self._inner = RealVCF(path)
            self._closed = False
            type(self).active += 1
            type(self).max_active = max(type(self).max_active, type(self).active)

        def __getattr__(self, name: str):
            return getattr(self._inner, name)

        def close(self) -> None:
            if not self._closed:
                self._inner.close()
                self._closed = True
                type(self).active -= 1

    monkeypatch.setattr("cyvcf2.VCF", TrackingVCF)
    ref = _write_ref(tmp_path)
    samples = ("S0", "S1")
    shard_a = _write_vcf(
        tmp_path,
        "a",
        "chr1\t8\t.\tA\tT\t.\t.\t.\tGT\t1|0\t0|1\n",
        samples,
    )
    shard_b = _write_vcf(
        tmp_path,
        "b",
        "chr1\t9\t.\tGAG\tG\t.\t.\t.\tGT\t1|0\t1|1\n",
        samples,
    )

    SparseVar2.from_vcf_shards(
        tmp_path / "headers.svar2",
        [
            (shard_a, ("chr1", 0, 8)),
            (shard_b, ("chr1", 8, len(_REF))),
        ],
        ref,
        threads=2,
    )

    assert TrackingVCF.active == 0
    assert TrackingVCF.max_active == 1


def test_from_vcf_shards_skips_field_header_scan_when_no_fields_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("field headers should not be reopened without fields")

    monkeypatch.setattr("genoray._svar2._resolve_fields", fail_if_called)
    ref = _write_ref(tmp_path)
    source = _write_vcf(
        tmp_path,
        "native",
        "chr1\t8\t.\tA\tT\t.\t.\t.\tGT\t1|0\t0|1\n",
        ("S0", "S1"),
    )

    SparseVar2.from_vcf_shards(
        tmp_path / "no-fields.svar2",
        [(source, ("chr1", 0, len(_REF)))],
        ref,
        threads=2,
    )


def _assert_store_equal(a: Path, b: Path) -> None:
    left = SparseVar2(a)
    right = SparseVar2(b)
    assert left.available_samples == right.available_samples
    assert left.contigs == right.contigs
    np.testing.assert_array_equal(
        left.region_counts("chr1", [(0, len(_REF))]),
        right.region_counts("chr1", [(0, len(_REF))]),
    )
    left_rag = left.decode("chr1", [(0, len(_REF))])
    right_rag = right.decode("chr1", [(0, len(_REF))])
    for field in ("pos", "ilen", "allele"):
        np.testing.assert_array_equal(
            np.asarray(left_rag[field].data), np.asarray(right_rag[field].data)
        )
        np.testing.assert_array_equal(
            np.asarray(left_rag[field].lengths),
            np.asarray(right_rag[field].lengths),
        )


def _hash_store(store: Path) -> bytes:
    digest = hashlib.sha256()
    for path in sorted(p for p in store.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(store)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.digest()


def test_from_vcf_shards_matches_single_vcf_across_left_align_boundary(
    tmp_path: Path,
) -> None:
    """Padded native cohort-shard workers must preserve normalized ordering.

    The deletion starts in shard B at 0-based POS 8, then left-aligns to POS 6,
    before shard A's SNP at POS 7.  Normalizing each physical file independently
    without boundary padding and normalized-POS ownership would be out of order.
    """
    ref = _write_ref(tmp_path)
    samples = ("S0", "S1")
    row_a = "chr1\t8\t.\tA\tT\t.\t.\t.\tGT\t1|0\t0|1\n"
    row_b = "chr1\t9\t.\tGAG\tG\t.\t.\t.\tGT\t1|0\t1|1\n"
    shard_a = _write_vcf(tmp_path, "a", row_a, samples)
    shard_b = _write_vcf(tmp_path, "b", row_b, samples)
    oracle = _write_vcf(tmp_path, "oracle", row_a + row_b, samples)

    oracle_out = tmp_path / "oracle.svar2"
    shard_out = tmp_path / "shards.svar2"
    SparseVar2.from_vcf(oracle_out, oracle, ref, threads=1, chunk_size=1)
    SparseVar2.from_vcf_shards(
        shard_out,
        [
            (shard_a, ("chr1", 0, 8)),
            (shard_b, ("chr1", 8, len(_REF))),
        ],
        ref,
        threads=4,
        chunk_size=1,
    )

    _assert_store_equal(shard_out, oracle_out)
    assert _hash_store(shard_out) == _hash_store(oracle_out)
    rag = SparseVar2(shard_out).decode("chr1", [(0, len(_REF))])
    assert sorted(set(np.asarray(rag["pos"].data).tolist())) == [6, 7]


def test_from_vcf_shards_rejects_sample_order_mismatch(tmp_path: Path) -> None:
    ref = _write_ref(tmp_path)
    shard_a = _write_vcf(
        tmp_path,
        "a",
        "chr1\t8\t.\tA\tT\t.\t.\t.\tGT\t1|0\t0|1\n",
        ("S0", "S1"),
    )
    shard_b = _write_vcf(
        tmp_path,
        "b",
        "chr1\t9\t.\tGAG\tG\t.\t.\t.\tGT\t1|0\t1|1\n",
        ("S1", "S0"),
    )

    with pytest.raises(ValueError, match="sample.*order|identical sample"):
        SparseVar2.from_vcf_shards(
            tmp_path / "bad.svar2",
            [
                (shard_a, ("chr1", 0, 8)),
                (shard_b, ("chr1", 8, len(_REF))),
            ],
            ref,
            threads=2,
        )


def test_from_vcf_shards_rejects_overlapping_ownership(tmp_path: Path) -> None:
    ref = _write_ref(tmp_path)
    samples = ("S0", "S1")
    shard_a = _write_vcf(
        tmp_path,
        "a",
        "chr1\t8\t.\tA\tT\t.\t.\t.\tGT\t1|0\t0|1\n",
        samples,
    )
    shard_b = _write_vcf(
        tmp_path,
        "b",
        "chr1\t9\t.\tGAG\tG\t.\t.\t.\tGT\t1|0\t1|1\n",
        samples,
    )

    with pytest.raises(ValueError, match="ownership.*overlap|overlapping.*shard"):
        SparseVar2.from_vcf_shards(
            tmp_path / "overlap.svar2",
            [
                (shard_a, ("chr1", 0, 9)),
                (shard_b, ("chr1", 8, len(_REF))),
            ],
            ref,
            threads=2,
        )


def test_from_vcf_shards_regions_and_samples_match_single_vcf(tmp_path: Path) -> None:
    ref = _write_ref(tmp_path)
    samples = ("S0", "S1")
    row_a = "chr1\t8\t.\tA\tT\t.\t.\t.\tGT\t1|0\t0|1\n"
    row_b = "chr1\t9\t.\tGAG\tG\t.\t.\t.\tGT\t1|0\t1|1\n"
    shard_a = _write_vcf(tmp_path, "a", row_a, samples)
    shard_b = _write_vcf(tmp_path, "b", row_b, samples)
    oracle = _write_vcf(tmp_path, "oracle", row_a + row_b, samples)

    oracle_out = tmp_path / "oracle_subset.svar2"
    shard_out = tmp_path / "shards_subset.svar2"
    kwargs = {"regions": ("chr1", 7, len(_REF)), "samples": ["S1"]}
    SparseVar2.from_vcf(oracle_out, oracle, ref, threads=1, **kwargs)
    SparseVar2.from_vcf_shards(
        shard_out,
        [
            (shard_a, ("chr1", 0, 8)),
            (shard_b, ("chr1", 8, len(_REF))),
        ],
        ref,
        threads=4,
        **kwargs,
    )

    _assert_store_equal(shard_out, oracle_out)
    assert _hash_store(shard_out) == _hash_store(oracle_out)


def test_from_vcf_shards_rejects_non_pos_overlap_mode(tmp_path: Path) -> None:
    ref = _write_ref(tmp_path)
    shard = _write_vcf(
        tmp_path,
        "a",
        "chr1\t8\t.\tA\tT\t.\t.\t.\tGT\t1|0\t0|1\n",
        ("S0", "S1"),
    )
    with pytest.raises(ValueError, match="only regions_overlap='pos'"):
        SparseVar2.from_vcf_shards(
            tmp_path / "unsupported.svar2",
            [(shard, ("chr1", 0, len(_REF)))],
            ref,
            regions_overlap="variant",
        )
