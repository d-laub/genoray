import subprocess
import os
from pathlib import Path


def test_generate_repro_emits_multicontig_vaf_bcf(tmp_path: Path):
    out = tmp_path / "cohort"
    subprocess.run(
        [
            "python",
            "scripts/from_vcf_livelock/generate_repro.py",
            "--out",
            str(out),
            "--samples",
            "40",
            "--contigs",
            "chr1,chr2,chr3,chr4,chr5,chr6",
            "--target-size",
            "8MB",
            "--seed",
            "0",
        ],
        check=True,
        env={**os.environ},
    )
    bcf = out / "cohort.bcf"
    assert bcf.exists() and (out / "cohort.bcf.csi").exists()
    hdr = subprocess.run(
        ["bcftools", "view", "-h", str(bcf)], capture_output=True, text=True, check=True
    ).stdout
    assert hdr.count("##contig=") >= 5
    assert "ID=VAF" in hdr and "Number=A" in hdr and "Type=Float" in hdr
    contigs = subprocess.run(
        ["bcftools", "index", "-s", str(bcf)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert len([line for line in contigs.splitlines() if line.strip()]) >= 5
