"""The generator must model a private-somatic cohort: the union of variant positions
grows ~linearly with the number of files. A fixed position pool makes the union
asymptote, which hides every O(V x N) cost in the merge (issue #120)."""

import subprocess
from pathlib import Path

from generate_cohort import generate_cohort, required_contig_len, union_positions


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


def _distinct_positions(manifest: Path) -> set[tuple[str, int]]:
    """True (contig, pos) union actually emitted across a cohort's files."""
    positions: set[tuple[str, int]] = set()
    for vcf_path in Path(manifest).read_text().splitlines():
        out = subprocess.run(
            ["bcftools", "query", "-f", "%CHROM\t%POS\n", vcf_path],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for line in out.splitlines():
            chrom, pos = line.split("\t")
            positions.add((chrom, int(pos)))
    return positions


def test_generated_union_matches_estimate_and_grows_linearly(tmp_path: Path) -> None:
    """`test_union_grows_linearly_with_n` above only checks union_positions's
    arithmetic against itself -- it never calls generate_cohort, so it can't
    catch a generator bug (e.g. shared/private fractions swapped, contig salting
    collisions). This test closes that gap empirically: generate two cohorts at
    a small but discriminating scale, count the ACTUAL distinct (contig, pos)
    union bcftools sees on disk, and check it against both union_positions's
    prediction and n_files."""
    n_variants = 30
    shared_frac = 0.1

    # contig_len=None exercises required_contig_len's auto-sizing (untested
    # otherwise). At this n_variants, required_contig_len is floor-dominated
    # (max(1_000_000, ...)) for both file counts below, so both cohorts land on
    # the same contig_len and the comparison isolates the effect of n_files.
    contig_len_20 = required_contig_len(20, n_variants, shared_frac)
    contig_len_200 = required_contig_len(200, n_variants, shared_frac)
    assert contig_len_20 == contig_len_200 == 1_000_000, (
        "test assumes both file counts are floor-dominated so growth reflects "
        "n_files alone -- adjust n_variants if this assumption breaks"
    )

    small_manifest = generate_cohort(
        tmp_path / "small",
        n_files=20,
        n_variants=n_variants,
        contig_len=None,
        shared_frac=shared_frac,
        seed=0,
    )
    big_manifest = generate_cohort(
        tmp_path / "big",
        n_files=200,
        n_variants=n_variants,
        contig_len=None,
        shared_frac=shared_frac,
        seed=0,
    )

    small_union = len(_distinct_positions(small_manifest))
    big_union = len(_distinct_positions(big_manifest))

    predicted_small = union_positions(20, n_variants, shared_frac, contig_len_20)
    predicted_big = union_positions(200, n_variants, shared_frac, contig_len_200)

    # Load-bearing assertion: ties the birthday-problem estimator to what the
    # generator actually emits. p = n_priv/contig_len ~ 27/1e6 ~ 2.7e-5, so
    # cross-file collisions are rare and the estimate should be tight.
    assert abs(small_union - predicted_small) / predicted_small < 0.15, (
        f"empirical union {small_union} vs predicted {predicted_small:.1f} "
        "(n_files=20) -- estimator and generator disagree"
    )
    assert abs(big_union - predicted_big) / predicted_big < 0.15, (
        f"empirical union {big_union} vs predicted {predicted_big:.1f} "
        "(n_files=200) -- estimator and generator disagree"
    )

    # 10x the files => ~10x the union at this low collision probability; allow
    # slack for the fixed shared pool + sampling noise (same bound as the
    # formula-only test above).
    ratio = big_union / small_union
    assert 8.0 < ratio < 10.5, (
        f"empirical union ratio {ratio} — generator's actual output isn't "
        "growing linearly with n_files"
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
