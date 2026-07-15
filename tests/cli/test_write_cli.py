from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from genoray import SparseVar, SparseVar2

_REF = "ACAGTACATGGGTACTAGCTAGGCTAACCGGTTAACCGGT"


def _run(argv: list[str], *, columns: int | None = None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    if columns is not None:
        env["COLUMNS"] = str(columns)
    return subprocess.run(
        [sys.executable, "-m", "genoray._cli", *argv],
        capture_output=True,
        text=True,
        env=env,
    )


def _ref(d: Path) -> Path:
    ref = d / "ref.fa"
    ref.write_text(f">chr1\n{_REF}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)
    return ref


def _pgen(d: Path, vcf: Path) -> Path:
    subprocess.run(
        [
            "plink2",
            "--make-pgen",
            "--output-chr",
            "chrM",
            "--vcf",
            str(vcf),
            "--out",
            str(d / "in"),
        ],
        check=True,
    )
    return d / "in.pgen"


def _vcf(d: Path, *, symbolic: bool) -> Path:
    body = (
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=40>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1\n"
        "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\t0|0\n"
        "chr1\t7\t.\tC\tCAT\t.\t.\t.\tGT\t0|1\t1|1\n"
    )
    if symbolic:
        # POS 20 (1-based) in _REF is 'T'; the anchor REF base must match the
        # FASTA or the pipeline raises RefMismatch before it can even reach
        # the skip-out-of-scope logic being tested here.
        body += "chr1\t20\t.\tT\t<DEL>\t.\t.\t.\tGT\t0|1\t0|0\n"
    plain = d / "in.vcf"
    plain.write_text(body)
    gz = d / "in.vcf.gz"
    with open(gz, "wb") as fh:
        subprocess.run(["bgzip", "-c", str(plain)], check=True, stdout=fh)
    subprocess.run(["bcftools", "index", str(gz)], check=True)
    return gz


def test_write_defaults_to_svar2(tmp_path: Path):
    ref = _ref(tmp_path)
    vcf = _vcf(tmp_path, symbolic=False)
    out = tmp_path / "store"
    r = _run(["write", str(vcf), str(out), "--reference", str(ref), "--threads", "1"])
    assert r.returncode == 0, r.stderr
    sv = SparseVar2(out)
    assert sv.contigs == ["chr1"]


def test_write_no_reference(tmp_path: Path):
    vcf = _vcf(tmp_path, symbolic=False)
    out = tmp_path / "store2"
    r = _run(["write", str(vcf), str(out), "--no-reference", "--threads", "1"])
    assert r.returncode == 0, r.stderr
    assert (out / "meta.json").exists()


def test_write_svar2_regions_and_samples(tmp_path: Path):
    ref = _ref(tmp_path)
    vcf = _vcf(tmp_path, symbolic=False)
    out = tmp_path / "regioned"
    r = _run(
        [
            "write",
            str(vcf),
            str(out),
            "--reference",
            str(ref),
            "--regions",
            "chr1:1-4",
            "--samples",
            "S0",
            "--threads",
            "1",
        ]
    )
    assert r.returncode == 0, r.stderr
    sv = SparseVar2(out)
    assert sv.available_samples == ["S0"]
    assert int(sv.region_counts("chr1", [(0, 40)]).sum()) == 1


def test_write_svar2_regions_file_and_samples_file(tmp_path: Path):
    ref = _ref(tmp_path)
    vcf = _vcf(tmp_path, symbolic=False)
    regions = tmp_path / "regions.bed"
    regions.write_text("chr1\t3\t8\n")
    samples = tmp_path / "samples.txt"
    samples.write_text("S1\n")
    out = tmp_path / "file_args"
    r = _run(
        [
            "write",
            str(vcf),
            str(out),
            "--reference",
            str(ref),
            "--regions-file",
            str(regions),
            "--samples-file",
            str(samples),
            "--threads",
            "1",
        ]
    )
    assert r.returncode == 0, r.stderr
    sv = SparseVar2(out)
    assert sv.available_samples == ["S1"]
    assert int(sv.region_counts("chr1", [(0, 40)]).sum()) == 2


def test_write_requires_reference_xor(tmp_path: Path):
    vcf = _vcf(tmp_path, symbolic=False)
    out = tmp_path / "store3"
    r = _run(["write", str(vcf), str(out), "--threads", "1"])
    assert r.returncode != 0
    assert "reference" in (r.stderr + r.stdout)


def test_write_skip_symbolic(tmp_path: Path):
    ref = _ref(tmp_path)
    vcf = _vcf(tmp_path, symbolic=True)
    out = tmp_path / "store4"
    r = _run(
        [
            "write",
            str(vcf),
            str(out),
            "--reference",
            str(ref),
            "--skip-symbolics-and-breakends",
            "--threads",
            "1",
        ]
    )
    assert r.returncode == 0, r.stderr
    assert (out / "meta.json").exists()


def test_write_svar2_has_single_skip_flag():
    # --help lists the new collapsed flag. Wide COLUMNS avoids the rich help
    # table wrapping the long flag name across lines.
    r = _run(["write", "--help"], columns=200)
    assert r.returncode == 0, r.stderr
    assert "--skip-symbolics-and-breakends" in r.stdout
    # The docstring's cross-reference note mentions svar1's --no-symbolic /
    # --no-breakend by name for context, so we can't grep --help for their
    # absence; see test_write_no_{symbolic,breakend}_removed_from_svar2 below
    # for the behavioral check that they're no longer accepted options here.


def test_write_no_symbolic_removed_from_svar2(tmp_path: Path):
    ref = _ref(tmp_path)
    vcf = _vcf(tmp_path, symbolic=False)
    out = tmp_path / "store5"
    r = _run(
        [
            "write",
            str(vcf),
            str(out),
            "--reference",
            str(ref),
            "--no-symbolic",
            "--threads",
            "1",
        ]
    )
    assert r.returncode != 0
    assert "no-symbolic" in (r.stdout + r.stderr).lower()


def test_write_no_breakend_removed_from_svar2(tmp_path: Path):
    ref = _ref(tmp_path)
    vcf = _vcf(tmp_path, symbolic=False)
    out = tmp_path / "store6"
    r = _run(
        [
            "write",
            str(vcf),
            str(out),
            "--reference",
            str(ref),
            "--no-breakend",
            "--threads",
            "1",
        ]
    )
    assert r.returncode != 0
    assert "no-breakend" in (r.stdout + r.stderr).lower()


def test_write_dispatches_pgen(tmp_path: Path):
    """`genoray write` already advertises VCF/PGEN in its help; make it true."""
    ref = _ref(tmp_path)
    vcf = _vcf(tmp_path, symbolic=False)
    pgen = _pgen(tmp_path, vcf)
    out = tmp_path / "pgen.svar2"
    r = _run(["write", str(pgen), str(out), "--reference", str(ref), "--threads", "1"])
    assert r.returncode == 0, r.stderr
    sv = SparseVar2(out)
    assert sv.contigs == ["chr1"]


def test_write_pgen_rejects_nondefault_ploidy(tmp_path: Path):
    ref = _ref(tmp_path)
    vcf = _vcf(tmp_path, symbolic=False)
    pgen = _pgen(tmp_path, vcf)
    out = tmp_path / "pgen_ploidy.svar2"
    r = _run(
        [
            "write",
            str(pgen),
            str(out),
            "--reference",
            str(ref),
            "--ploidy",
            "1",
            "--threads",
            "1",
        ]
    )
    assert r.returncode != 0
    assert "diploid" in (r.stdout + r.stderr).lower()


def test_write_pgen_rejects_regions_without_plink(tmp_path: Path):
    pgen = tmp_path / "fake.pgen"
    pgen.write_bytes(b"")
    out = tmp_path / "fake.svar2"
    r = _run(
        [
            "write",
            str(pgen),
            str(out),
            "--no-reference",
            "--regions",
            "chr1:1-4",
        ]
    )
    assert r.returncode != 0
    assert "vcf/bcf only" in (r.stdout + r.stderr).lower()


def test_write_svar1_still_works(tmp_path: Path):
    vcf = _vcf(tmp_path, symbolic=False)
    out = tmp_path / "v1.svar"
    r = _run(["write", "svar1", str(vcf), str(out), "--max-mem", "64m"])
    assert r.returncode == 0, r.stderr
    sv = SparseVar(out)
    assert sv.n_variants >= 1
