from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from genoray import SparseVar, SparseVar2
from genoray._cli.__main__ import app

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
    r = _run(
        ["write", "vcf", str(vcf), str(out), "--reference", str(ref), "--threads", "1"]
    )
    assert r.returncode == 0, r.stderr
    sv = SparseVar2(out)
    assert sv.contigs == ["chr1"]


def test_write_no_reference(tmp_path: Path):
    vcf = _vcf(tmp_path, symbolic=False)
    out = tmp_path / "store2"
    r = _run(["write", "vcf", str(vcf), str(out), "--no-reference", "--threads", "1"])
    assert r.returncode == 0, r.stderr
    assert (out / "meta.json").exists()


def test_write_svar2_regions_and_samples(tmp_path: Path):
    ref = _ref(tmp_path)
    vcf = _vcf(tmp_path, symbolic=False)
    out = tmp_path / "regioned"
    r = _run(
        [
            "write",
            "vcf",
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
            "vcf",
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
    r = _run(["write", "vcf", str(vcf), str(out), "--threads", "1"])
    assert r.returncode != 0
    assert "reference" in (r.stderr + r.stdout)


def test_write_skip_symbolic(tmp_path: Path):
    ref = _ref(tmp_path)
    vcf = _vcf(tmp_path, symbolic=True)
    out = tmp_path / "store4"
    r = _run(
        [
            "write",
            "vcf",
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
    r = _run(["write", "vcf", "--help"], columns=200)
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
            "vcf",
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
            "vcf",
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
    r = _run(
        [
            "write",
            "pgen",
            str(pgen),
            str(out),
            "--reference",
            str(ref),
            "--threads",
            "1",
        ]
    )
    assert r.returncode == 0, r.stderr
    sv = SparseVar2(out)
    assert sv.contigs == ["chr1"]


def test_write_pgen_rejects_ploidy_flag(tmp_path: Path):
    """`write pgen` drops --ploidy entirely (PGEN is always diploid), so
    passing it is now an unrecognized-option error rather than the old
    'diploid' ValueError raised by the merged `write` command."""
    ref = _ref(tmp_path)
    vcf = _vcf(tmp_path, symbolic=False)
    pgen = _pgen(tmp_path, vcf)
    out = tmp_path / "pgen_ploidy.svar2"
    r = _run(
        [
            "write",
            "pgen",
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


def test_write_pgen_no_ploidy_flag():
    """`write pgen` drops --ploidy entirely (PGEN is always diploid)."""
    r = _run(["write", "pgen", "--help"], columns=200)
    assert r.returncode == 0, r.stderr
    assert "--ploidy" not in r.stdout


def test_write_pgen_regions_dispatch_without_plink(tmp_path: Path):
    """--regions/--samples now route to `from_pgen` (no longer VCF/BCF-only);
    this exercises that dispatch without needing a real plink2-built PGEN --
    a bare `.pgen` with no `.pvar` sibling fails deterministically inside
    `from_pgen` itself (missing sibling), never with the old blanket
    rejection message."""
    pgen = tmp_path / "fake.pgen"
    pgen.write_bytes(b"")
    out = tmp_path / "fake.svar2"
    r = _run(
        [
            "write",
            "pgen",
            str(pgen),
            str(out),
            "--no-reference",
            "--regions",
            "chr1:1-4",
        ]
    )
    assert r.returncode != 0
    assert "vcf/bcf only" not in (r.stdout + r.stderr).lower()
    assert "pvar" in (r.stdout + r.stderr).lower()


def test_write_svar1_still_works(tmp_path: Path):
    vcf = _vcf(tmp_path, symbolic=False)
    out = tmp_path / "v1.svar"
    r = _run(["write-svar1", str(vcf), str(out), "--max-mem", "64m"])
    assert r.returncode == 0, r.stderr
    sv = SparseVar(out)
    assert sv.n_variants >= 1


def test_write_pgen_regions_and_samples(tmp_path: Path):
    """Task 6: --regions/--samples now route through to `from_pgen`."""
    vcf = _vcf(tmp_path, symbolic=False)
    pgen = _pgen(tmp_path, vcf)
    out = tmp_path / "pgen_regioned.svar2"
    r = _run(
        [
            "write",
            "pgen",
            str(pgen),
            str(out),
            "--no-reference",
            "--regions",
            "chr1:1-4",
            "--samples",
            "S1",
            "--threads",
            "1",
        ]
    )
    assert r.returncode == 0, r.stderr
    sv = SparseVar2(out)
    assert sv.available_samples == ["S1"]


def _single_sample_vcf(d: Path, name: str, sample: str, rows: str) -> Path:
    """A single-sample bgzipped+indexed VCF, for the vcf-list (directory)
    input form."""
    header = (
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=40>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{sample}\n"
    )
    plain = d / f"{name}.vcf"
    plain.write_text(header + rows)
    gz = d / f"{name}.vcf.gz"
    with open(gz, "wb") as fh:
        subprocess.run(["bgzip", "-c", str(plain)], check=True, stdout=fh)
    subprocess.run(["bcftools", "index", str(gz)], check=True)
    return gz


def test_write_vcf_list_rejects_samples(tmp_path: Path):
    vcf_dir = tmp_path / "vcfs"
    vcf_dir.mkdir()
    _single_sample_vcf(vcf_dir, "a", "S0", "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\n")
    _single_sample_vcf(vcf_dir, "b", "S1", "chr1\t7\t.\tC\tCAT\t.\t.\t.\tGT\t0|1\n")
    out = tmp_path / "list_out"
    r = _run(
        ["write", "vcf", str(vcf_dir), str(out), "--no-reference", "--samples", "S0"]
    )
    assert r.returncode != 0
    assert "not supported for multi-file" in (r.stdout + r.stderr).lower()


def test_write_vcf_list_dispatches(tmp_path: Path):
    """Sanity check that the vcf-list (directory) form is actually reachable
    (not just erroring) now that source-kind resolution routes it to
    `from_vcf_list` instead of falling through to the single-VCF `from_vcf`
    path (which would fail trying to bgzip-check/index a directory)."""
    vcf_dir = tmp_path / "vcfs2"
    vcf_dir.mkdir()
    _single_sample_vcf(vcf_dir, "a", "S0", "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\n")
    _single_sample_vcf(vcf_dir, "b", "S1", "chr1\t7\t.\tC\tCAT\t.\t.\t.\tGT\t0|1\n")
    out = tmp_path / "list_out2"
    r = _run(
        ["write", "vcf", str(vcf_dir), str(out), "--no-reference", "--threads", "1"]
    )
    assert r.returncode == 0, r.stderr
    sv = SparseVar2(out)
    assert sv.available_samples == ["S0", "S1"]


def test_write_svar1_regions_and_samples(tmp_path: Path):
    """Task 6: --regions/--samples now route through to `from_svar1` for a
    `.svar` (SVAR1) source. The legacy VCF->SVAR1 writer used to build the
    `.svar` fixture now lives at the top-level `write-svar1`."""
    vcf = _vcf(tmp_path, symbolic=False)
    v1 = tmp_path / "v1.svar"
    r = _run(["write-svar1", str(vcf), str(v1), "--max-mem", "64m"])
    assert r.returncode == 0, r.stderr

    out = tmp_path / "v1_to_v2.svar2"
    r = _run(
        [
            "write",
            "svar1",
            str(v1),
            str(out),
            "--no-reference",
            "--samples",
            "S1",
            "--threads",
            "1",
        ]
    )
    assert r.returncode == 0, r.stderr
    sv = SparseVar2(out)
    assert sv.available_samples == ["S1"]


def test_write_bare_is_removed():
    # bare `write SOURCE OUT` no longer resolves to a converter now that
    # `write` has only subcommands (vcf/pgen/svar1): cyclopts treats
    # "x.vcf.gz" as an unknown subcommand and exits non-zero.
    with pytest.raises(SystemExit) as exc:
        app(["write", "x.vcf.gz", "out.svar2", "--no-reference"])
    assert exc.value.code != 0


def test_write_vcf_fields_parsed(tmp_path: Path, monkeypatch):
    ref = _ref(tmp_path)
    vcf = _vcf(tmp_path, symbolic=False)
    captured: dict = {}

    def fake_from_vcf(out, source, reference, **kw):
        captured.update(kw)
        return 0

    monkeypatch.setattr(SparseVar2, "from_vcf", staticmethod(fake_from_vcf))
    with pytest.raises(SystemExit) as exc:
        app(
            [
                "write",
                "vcf",
                str(vcf),
                str(tmp_path / "o.svar2"),
                "--reference",
                str(ref),
                "--fields",
                "INFO/AF",
                "--fields",
                "FMT/AD",
            ]
        )
    assert exc.value.code == 0
    assert captured["info_fields"] == ["AF"]
    assert captured["format_fields"] == ["AD"]


def test_write_pgen_dosages_parsed(tmp_path: Path, monkeypatch):
    ref = _ref(tmp_path)
    vcf = _vcf(tmp_path, symbolic=False)
    pgen = _pgen(tmp_path, vcf)
    captured: dict = {}

    def fake_from_pgen(out, source, reference, **kw):
        captured.update(kw)
        return 0

    monkeypatch.setattr(SparseVar2, "from_pgen", staticmethod(fake_from_pgen))
    with pytest.raises(SystemExit) as exc:
        app(
            [
                "write",
                "pgen",
                str(pgen),
                str(tmp_path / "o2.svar2"),
                "--reference",
                str(ref),
                "--dosages",
                "DS=self",
                "--dosages",
                "VAF=/x/vaf.pgen",
            ]
        )
    assert exc.value.code == 0
    ds = {d.name: str(d.source) for d in captured["dosages"]}
    assert ds == {"DS": "self", "VAF": "/x/vaf.pgen"}


def test_write_dosages_malformed_entry_raises(tmp_path: Path):
    ref = _ref(tmp_path)
    vcf = _vcf(tmp_path, symbolic=False)
    pgen = _pgen(tmp_path, vcf)
    with pytest.raises(ValueError, match="NAME=self"):
        app(
            [
                "write",
                "pgen",
                str(pgen),
                str(tmp_path / "bad.svar2"),
                "--reference",
                str(ref),
                "--dosages",
                "not-a-valid-entry",
            ],
            exit_on_error=False,
        )


def test_write_svar1_legacy_still_exists(tmp_path: Path, monkeypatch):
    """`write-svar1` (top-level) is the legacy VCF/PGEN->SVAR1 (v1) writer."""
    vcf = _vcf(tmp_path, symbolic=False)
    called: dict = {}
    monkeypatch.setattr(
        SparseVar,
        "from_vcf",
        staticmethod(lambda *a, **k: called.setdefault("hit", True)),
    )
    with pytest.raises(SystemExit) as exc:
        app(["write-svar1", str(vcf), str(tmp_path / "o.svar")])
    assert exc.value.code == 0
    assert called.get("hit")


def test_write_from_svar1_fields_and_empty(tmp_path: Path, monkeypatch):
    """`write svar1` (SVAR1->SVAR2) gains --fields and --empty-fields."""
    vcf = _vcf(tmp_path, symbolic=False)
    v1 = tmp_path / "v1.svar"
    r = _run(["write-svar1", str(vcf), str(v1), "--max-mem", "64m"])
    assert r.returncode == 0, r.stderr

    captured: dict = {}

    def fake_from_svar1(out, source, reference, **kw):
        captured.update(kw)
        return 0

    monkeypatch.setattr(SparseVar2, "from_svar1", staticmethod(fake_from_svar1))
    with pytest.raises(SystemExit) as exc:
        app(
            [
                "write",
                "svar1",
                str(v1),
                str(tmp_path / "v1_to_v2.svar2"),
                "--no-reference",
                "--empty-fields",
            ]
        )
    assert exc.value.code == 0
    assert captured["fields"] == []
