"""E2E tests for `SparseVar2.from_pgen(dosages=...)` -- PGEN dosages stored and
read back as SVAR2 FORMAT fields.

Fixtures are self-contained (built with `plink2 --make-pgen ... 'dosage=DS'`,
mirroring `tests/data/gen_from_vcf.sh`) rather than reusing a named repo
fixture, since no `.pgen` carrying a dosage track is committed by name.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from genoray import DosageField, SparseVar2
from tests.test_svar2_from_svar1 import _write_dosage_vcf
from tests.test_svar2_from_vcf import _write_ref

# Carrier-only decode order for `_write_dosage_vcf`'s two variants
# (chr1:3 A>G, chr1:7 C>CAT; samples S0, S1), across the single query region
# (0, 40): cells are visited (S0,h0), (S0,h1), (S1,h0), (S1,h1).
#   S0: hap0 carries var0 (DS=1.0); hap1 carries var1 (DS=1.0)
#   S1: hap0 does NOT carry var0 (GT 0|0); hap1 carries var1... wait S1 var1
#   GT is 1|1, so BOTH haps carry var1 (DS=2.0); S1 does not carry var0 at all.
_EXPECTED_DOSAGES = [1.0, 1.0, 2.0, 2.0]


def _make_pgen(d: Path, vcf: Path, name: str, *, dosage: bool) -> Path:
    """Build a `.pgen`/`.pvar`/`.psam` trio from `vcf`, matching the
    `--output-chr chrM` convention `tests/test_svar2_from_pgen.py` uses to
    keep `chr1`-style contig names (rather than plink2's default renaming).
    """
    vcf_args = ["--vcf", str(vcf)]
    if dosage:
        vcf_args.append("dosage=DS")
    subprocess.run(
        [
            "plink2",
            "--make-pgen",
            "--output-chr",
            "chrM",
            *vcf_args,
            "--out",
            str(d / name),
        ],
        check=True,
    )
    return d / f"{name}.pgen"


def _write_mismatched_dosage_vcf(d: Path) -> Path:
    """Same variants as `_write_dosage_vcf` but different sample names."""
    body = (
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=40>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        '##FORMAT=<ID=DS,Number=1,Type=Float,Description="Dosage">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tT0\tT1\n"
        "chr1\t3\t.\tA\tG\t.\t.\t.\tGT:DS\t1|0:1.0\t0|0:0.0\n"
        "chr1\t7\t.\tC\tCAT\t.\t.\t.\tGT:DS\t0|1:1.0\t1|1:2.0\n"
    )
    plain = d / "mismatch.vcf"
    plain.write_text(body)
    gz = d / "mismatch.vcf.gz"
    with open(gz, "wb") as fh:
        subprocess.run(["bgzip", "-c", str(plain)], check=True, stdout=fh)
    subprocess.run(["bcftools", "index", str(gz)], check=True)
    return gz


def _write_single_variant_dosage_vcf(d: Path) -> Path:
    """Same samples/first variant as `_write_dosage_vcf` but only ONE
    variant, so a `.pvar` built from this does NOT align 1:1 with the
    two-variant fixture.
    """
    body = (
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=40>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        '##FORMAT=<ID=DS,Number=1,Type=Float,Description="Dosage">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1\n"
        "chr1\t3\t.\tA\tG\t.\t.\t.\tGT:DS\t1|0:1.0\t0|0:0.0\n"
    )
    plain = d / "single.vcf"
    plain.write_text(body)
    gz = d / "single.vcf.gz"
    with open(gz, "wb") as fh:
        subprocess.run(["bgzip", "-c", str(plain)], check=True, stdout=fh)
    subprocess.run(["bcftools", "index", str(gz)], check=True)
    return gz


def _write_dosage_vcf_v2(d: Path) -> Path:
    """Same GT carrier pattern as `_write_dosage_vcf` but distinct DS
    values, so a second dosage field sourced from this file is
    distinguishable from the first (`_EXPECTED_DOSAGES`) -- catches a
    field-ordering/mislabel bug between two `DosageField`s.

    Carrier-only decode order (see `_EXPECTED_DOSAGES` above) yields
    `[0.3, 0.7, 1.9, 1.9]`. DS values stay in plink2's expected
    `[0, ploidy]` dosage range (unlike raw allele counts) so plink2 accepts
    them as valid `DS`, while still being distinct from `_EXPECTED_DOSAGES`.
    """
    body = (
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=40>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        '##FORMAT=<ID=DS,Number=1,Type=Float,Description="Dosage">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1\n"
        "chr1\t3\t.\tA\tG\t.\t.\t.\tGT:DS\t1|0:0.3\t0|0:0.0\n"
        "chr1\t7\t.\tC\tCAT\t.\t.\t.\tGT:DS\t0|1:0.7\t1|1:1.9\n"
    )
    plain = d / "ds_v2.vcf"
    plain.write_text(body)
    gz = d / "ds_v2.vcf.gz"
    with open(gz, "wb") as fh:
        subprocess.run(["bgzip", "-c", str(plain)], check=True, stdout=fh)
    subprocess.run(["bcftools", "index", str(gz)], check=True)
    return gz


_EXPECTED_DOSAGES_V2 = [0.3, 0.7, 1.9, 1.9]


def test_from_pgen_self_dosages_round_trip(tmp_path: Path):
    ref = _write_ref(tmp_path)
    vcf = _write_dosage_vcf(tmp_path)
    pgen = _make_pgen(tmp_path, vcf, "self_ds", dosage=True)

    out = tmp_path / "out.svar2"
    SparseVar2.from_pgen(
        out,
        pgen,
        ref,
        dosages=[DosageField(name="DS", source="self")],
        threads=1,
    )
    sv = SparseVar2(out, fields=["DS"])
    assert "DS" in sv.available_fields
    assert sv.available_fields["DS"].category == "format"

    rag = sv.decode("chr1", [(0, 40)])
    ds = np.asarray(rag["DS"].data)
    np.testing.assert_allclose(ds, _EXPECTED_DOSAGES, rtol=1e-5)


def test_from_pgen_separate_dosage_file(tmp_path: Path):
    ref = _write_ref(tmp_path)
    vcf = _write_dosage_vcf(tmp_path)
    hardcall_pgen = _make_pgen(tmp_path, vcf, "hard", dosage=False)
    dosage_pgen = _make_pgen(tmp_path, vcf, "dose", dosage=True)

    out = tmp_path / "out.svar2"
    SparseVar2.from_pgen(
        out,
        hardcall_pgen,
        ref,
        dosages=[DosageField(name="VAF", source=dosage_pgen)],
        threads=1,
    )
    sv = SparseVar2(out, fields=["VAF"])
    assert "VAF" in sv.available_fields

    rag = sv.decode("chr1", [(0, 40)])
    vaf = np.asarray(rag["VAF"].data)
    np.testing.assert_allclose(vaf, _EXPECTED_DOSAGES, rtol=1e-5)


def test_from_pgen_dosage_psam_mismatch_raises(tmp_path: Path):
    ref = _write_ref(tmp_path)
    vcf = _write_dosage_vcf(tmp_path)
    hardcall_pgen = _make_pgen(tmp_path, vcf, "hard2", dosage=False)

    mismatch_vcf = _write_mismatched_dosage_vcf(tmp_path)
    mismatched_dosage_pgen = _make_pgen(
        tmp_path, mismatch_vcf, "mismatch_ds", dosage=True
    )

    with pytest.raises(ValueError, match="samples"):
        SparseVar2.from_pgen(
            tmp_path / "out.svar2",
            hardcall_pgen,
            ref,
            dosages=[DosageField(name="VAF", source=mismatched_dosage_pgen)],
            threads=1,
        )


def test_from_pgen_no_dosages_unchanged(tmp_path: Path):
    ref = _write_ref(tmp_path)
    vcf = _write_dosage_vcf(tmp_path)
    pgen = _make_pgen(tmp_path, vcf, "nodose", dosage=False)

    out = tmp_path / "out.svar2"
    SparseVar2.from_pgen(out, pgen, ref, threads=1)
    sv = SparseVar2(out)
    assert sv.available_fields == {}


def test_from_pgen_dosage_non_pgen_source_raises(tmp_path: Path):
    """A dosage `source` must be a `.pgen` path (or the literal `"self"`)."""
    vcf = _write_dosage_vcf(tmp_path)
    pgen = _make_pgen(tmp_path, vcf, "badext", dosage=False)

    with pytest.raises(ValueError, match=r"\.pgen"):
        SparseVar2.from_pgen(
            tmp_path / "out.svar2",
            pgen,
            no_reference=True,
            dosages=[DosageField(name="x", source="something.txt")],
            threads=1,
        )


def test_from_pgen_dosage_duplicate_name_raises(tmp_path: Path):
    vcf = _write_dosage_vcf(tmp_path)
    pgen = _make_pgen(tmp_path, vcf, "dup", dosage=False)

    with pytest.raises(ValueError, match="duplicate"):
        SparseVar2.from_pgen(
            tmp_path / "out.svar2",
            pgen,
            no_reference=True,
            dosages=[
                DosageField(name="DS", source="self"),
                DosageField(name="DS", source="self"),
            ],
            threads=1,
        )


def test_from_pgen_dosage_reserved_mutcat_name_raises(tmp_path: Path):
    vcf = _write_dosage_vcf(tmp_path)
    pgen = _make_pgen(tmp_path, vcf, "mutcat_reserved", dosage=False)

    with pytest.raises(ValueError, match="mutcat"):
        SparseVar2.from_pgen(
            tmp_path / "out.svar2",
            pgen,
            no_reference=True,
            dosages=[DosageField(name="mutcat", source="self")],
            threads=1,
        )


def test_from_pgen_dosage_variant_count_mismatch_raises(tmp_path: Path):
    """A separate dosage `.pgen` whose `.pvar` has a different variant count
    than `source`'s cannot align 1:1.
    """
    vcf = _write_dosage_vcf(tmp_path)
    hardcall_pgen = _make_pgen(tmp_path, vcf, "hard_count", dosage=False)

    single_vcf = _write_single_variant_dosage_vcf(tmp_path)
    mismatched_count_pgen = _make_pgen(
        tmp_path, single_vcf, "mismatch_count_ds", dosage=True
    )

    with pytest.raises(ValueError, match="1:1"):
        SparseVar2.from_pgen(
            tmp_path / "out.svar2",
            hardcall_pgen,
            no_reference=True,
            dosages=[DosageField(name="VAF", source=mismatched_count_pgen)],
            threads=1,
        )


def test_from_pgen_multi_dosage_fields_ordering(tmp_path: Path):
    """Two `DosageField`s -- one `source="self"`, one a separate `.pgen` --
    must each decode to THEIR OWN values, not get rotated/swapped.
    """
    ref = _write_ref(tmp_path)
    vcf = _write_dosage_vcf(tmp_path)
    vcf_v2 = _write_dosage_vcf_v2(tmp_path)

    # `source="self"` needs the hardcall file to itself carry a dosage track.
    hardcall_pgen = _make_pgen(tmp_path, vcf, "multi_self", dosage=True)
    dose2_pgen = _make_pgen(tmp_path, vcf_v2, "multi_dose2", dosage=True)

    out = tmp_path / "out.svar2"
    SparseVar2.from_pgen(
        out,
        hardcall_pgen,
        ref,
        dosages=[
            DosageField(name="DS", source="self"),
            DosageField(name="DS2", source=dose2_pgen),
        ],
        threads=1,
    )
    sv = SparseVar2(out, fields=["DS", "DS2"])
    assert set(sv.available_fields) == {"DS", "DS2"}

    rag = sv.decode("chr1", [(0, 40)])
    ds = np.asarray(rag["DS"].data)
    ds2 = np.asarray(rag["DS2"].data)
    np.testing.assert_allclose(ds, _EXPECTED_DOSAGES, rtol=1e-5)
    # PGEN's fixed-point dosage encoding (~1/32768 precision) needs a looser
    # tolerance for these small fractional values than the integral dosages
    # above.
    np.testing.assert_allclose(ds2, _EXPECTED_DOSAGES_V2, atol=1e-3)
    assert not np.allclose(ds, ds2)
