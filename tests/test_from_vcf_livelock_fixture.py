import os
import shutil
import subprocess
from pathlib import Path

import pytest

# generate_repro.py shells out to the ``vcfixture`` CLI binary (the Rust
# ``--features cli`` build, not the ``vcfixture`` Python package, which has no
# bulk generator). It is not installed in CI, so skip this smoke test when the
# binary can't be resolved — same "skip when the external resource is absent"
# convention as test_from_vcf_livelock.py. The resolution mirrors
# scripts/from_vcf_livelock/generate_repro.py::resolve_vcfixture.
_VCFIXTURE_FALLBACK = Path("/tmp/vcfixture-cli/bin/vcfixture")
_HAVE_VCFIXTURE = shutil.which("vcfixture") is not None or _VCFIXTURE_FALLBACK.exists()


@pytest.mark.skipif(
    not _HAVE_VCFIXTURE,
    reason=(
        "vcfixture CLI binary not on PATH or at /tmp/vcfixture-cli/bin/vcfixture; "
        "generate_repro.py needs it (install: cargo install vcfixture --features cli)"
    ),
)
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
