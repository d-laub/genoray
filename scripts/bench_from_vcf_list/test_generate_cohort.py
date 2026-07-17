"""The generator must model a private-somatic cohort: the union of variant positions
grows ~linearly with the number of files. A fixed position pool makes the union
asymptote, which hides every O(V x N) cost in the merge (issue #120)."""

import subprocess
from pathlib import Path

from generate_cohort import generate_cohort, union_positions


def test_union_grows_linearly_with_n(tmp_path: Path) -> None:
    small = union_positions(
        n_files=50, n_variants=200, shared_frac=0.1, contig_len=10_000_000
    )
    big = union_positions(
        n_files=500, n_variants=200, shared_frac=0.1, contig_len=10_000_000
    )
    # 10x the files => ~10x the private positions (shared pool is fixed, so allow slack).
    ratio = big / small
    assert 8.0 < ratio < 10.5, (
        f"union ratio {ratio} — pool saturating, cohort is unrealistic"
    )


def test_format_fields_are_emitted(tmp_path: Path) -> None:
    manifest = generate_cohort(
        tmp_path,
        n_files=3,
        n_variants=10,
        contigs=["1", "2"],
        contig_len=100_000,
        shared_frac=0.1,
        indel_frac=0.1,
        seed=0,
        format_fields=["VAF", "DP"],
    )
    first = Path(manifest).read_text().splitlines()[0]
    out = subprocess.run(
        ["bcftools", "view", "-H", first], capture_output=True, text=True, check=True
    ).stdout
    assert "GT:VAF:DP" in out, f"FORMAT column missing requested fields:\n{out[:200]}"


def test_multiple_contigs_present(tmp_path: Path) -> None:
    manifest = generate_cohort(
        tmp_path,
        n_files=2,
        n_variants=10,
        contigs=["1", "2", "3"],
        contig_len=100_000,
        shared_frac=0.1,
        indel_frac=0.1,
        seed=0,
        format_fields=["VAF"],
    )
    first = Path(manifest).read_text().splitlines()[0]
    out = subprocess.run(
        ["bcftools", "index", "-s", first], capture_output=True, text=True, check=True
    ).stdout
    assert {line.split("\t")[0] for line in out.strip().splitlines()} == {"1", "2", "3"}
