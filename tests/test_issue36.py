"""Regression test for issue #36.

var_counts drops queries with 0 variants from the group_by result,
returning a short array. Counts then misalign: region K gets region K+1's
count. A non-zero count assigned to an empty region causes _fill_genos to
pre-allocate a buffer it can't fill -> ValueError.

Minimal reprex: 2 regions where the first has 0 variants.
"""

from __future__ import annotations

from pathlib import Path

from genoray import VCF

ddir = Path(__file__).parent / "data"


def test_chunk_with_length_empty_region_misalignment():
    """Two regions: first has 0 variants, second has >0 variants.

    Before fix: var_counts returns [count_region1] (length 1 not 2).
    zip misaligns — region 0 gets count_region1, tries to read that many
    variants from a 0-variant range -> ValueError.

    After fix: var_counts returns [0, count_region1]; region 0 is
    correctly skipped; region 1 reads normally.

    biallelic.vcf.gz has variants on chr1 at 81262 and 81265.
    Region [1, 100) on chr1 is empty; region [81260, 81270) has variants.
    """
    vcf = VCF(ddir / "biallelic.vcf.gz")
    contig = "chr1"
    starts = [1, 81260]
    ends = [100, 81270]

    results = [
        list(region_chunks)
        for region_chunks in vcf._chunk_ranges_with_length(contig, starts, ends)
    ]
    # Empty region: one chunk with 0 variants (not a ValueError)
    assert len(results[0]) == 1
    assert results[0][0][0].shape[-1] == 0, "empty region chunk should have 0 variants"
    # Populated region: chunk with actual variants
    assert len(results[1]) >= 1
    assert results[1][0][0].shape[-1] > 0, "populated region chunk should have variants"
