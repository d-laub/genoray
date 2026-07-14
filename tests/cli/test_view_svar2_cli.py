import subprocess
import sys
from pathlib import Path

from genoray import SparseVar2, _core


def _run(argv):
    return subprocess.run(
        [sys.executable, "-m", "genoray._cli", *argv], capture_output=True, text=True
    )


def _dense_flip_svar2(tmp_path: Path) -> Path:
    """A 3-sample store (S0,S1 hom alt, S2 het -> dense at the full
    population) with a FORMAT field `DP`, used to prove the CLI's default
    `reroute` (unset -> `"auto"`) preserves representation for a 1-sample
    subset instead of re-routing it (see `test_svar2_write_view.py`'s
    `_dense_flip_store` for the same fixture/reasoning at the Python-API
    level, confirmed empirically there via `svar2_variant_stats`)."""
    ref_seq = "ACAGTACATGGGTACTAGCTAGGCTAACCGGTTAACCGGT" * 3
    ref = tmp_path / "ref.fa"
    ref.write_text(f">chr1\n{ref_seq}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)

    plain = tmp_path / "dense_flip.vcf"
    plain.write_text(
        "##fileformat=VCFv4.2\n"
        f"##contig=<ID=chr1,length={len(ref_seq)}>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Depth">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1\tS2\n"
        "chr1\t3\t.\tA\tG\t.\t.\t.\tGT:DP\t1|1:10\t1|1:20\t1|0:30\n"
    )
    gz = tmp_path / "dense_flip.vcf.gz"
    with open(gz, "wb") as out_f:
        subprocess.run(["bgzip", "-c", str(plain)], check=True, stdout=out_f)
    subprocess.run(["bcftools", "index", str(gz)], check=True)

    out = tmp_path / "src.svar2"
    SparseVar2.from_vcf(out, gz, ref, format_fields=["DP"], threads=1, overwrite=True)
    return out


def test_cli_split_then_concat(tiny_svar2, tmp_path):
    r = _run(["split", str(tiny_svar2), str(tmp_path / "parts")])
    assert r.returncode == 0, r.stderr
    parts = sorted((tmp_path / "parts").glob("*.svar2"))
    assert len(parts) == 2
    r = _run(["concat", str(tmp_path / "m.svar2"), *map(str, parts)])
    assert r.returncode == 0, r.stderr
    assert set(SparseVar2(tmp_path / "m.svar2").contigs) == set(
        SparseVar2(tiny_svar2).contigs
    )


def test_cli_split_with_contigs(tiny_svar2, tmp_path):
    out = tmp_path / "chr1_only.svar2"
    r = _run(["split", str(tiny_svar2), str(out), "--contigs", "chr1"])
    assert r.returncode == 0, r.stderr
    contigs = SparseVar2(out).contigs
    assert contigs == ["chr1"]
    assert "chr2" not in contigs


def test_cli_view_svar2_region_subset(tiny_svar2, tmp_path):
    # tiny_svar2 is a two-contig (chr1, chr2) store; chr1's variants sit at
    # POS 1 and 3 (1-based). Restricting to chr1:1-40 should keep only chr1.
    out = tmp_path / "v.svar2"
    r = _run(["view", str(tiny_svar2), str(out), "-r", "chr1:1-40"])
    assert r.returncode == 0, r.stderr
    assert SparseVar2(out).contigs == ["chr1"]


def test_cli_view_svar2_defaults_to_auto(tmp_path):
    """`genoray view` (no `--reroute`/`--no-reroute` flag) must default to
    `"auto"`, not `True`: with a FORMAT field carried, `"auto"` must preserve
    the source-dense variant's representation rather than re-route it -- the
    same protection the Python API gets by default."""
    src = _dense_flip_svar2(tmp_path)
    out = tmp_path / "v.svar2"
    r = _run(
        [
            "view",
            str(src),
            str(out),
            "-s",
            "S2",
            "-f",
            "DP",
        ]
    )
    assert r.returncode == 0, r.stderr
    assert SparseVar2(out).available_fields["DP"] is not None
    _ii, sd_out, *_ = _core.svar2_variant_stats(str(out), "chr1", [0])
    assert int(sd_out[0]) == 1, "CLI default ('auto') must preserve representation"


def test_cli_view_svar1_still_works(tiny_svar, tmp_path):
    out = tmp_path / "v.svar"
    r = _run(["view", "svar1", str(tiny_svar), str(out), "-r", "chr1:1-100"])
    assert r.returncode == 0, r.stderr


def test_cli_view_svar2_no_reroute_succeeds(tiny_svar2, tmp_path):
    """`--no-reroute` (reroute=False, the representation-preserving direct
    slice) is now implemented and produces a readable store."""
    out = tmp_path / "v.svar2"
    r = _run(
        [
            "view",
            str(tiny_svar2),
            str(out),
            "-r",
            "chr1:1-40",
            "--no-reroute",
        ]
    )
    assert r.returncode == 0, r.stderr
    assert SparseVar2(out).contigs == ["chr1"]
