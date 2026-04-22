"""Regression test for issue #36.

var_counts drops queries with 0 variants from the group_by result,
returning a short array. Counts then misalign: region K gets region K+1's
count. A non-zero count assigned to an empty region causes _fill_genos to
pre-allocate a buffer it can't fill -> ValueError.

Minimal reprex: 2 regions where the first has 0 variants.
"""

from __future__ import annotations

from pathlib import Path

import pooch
import pytest

from genoray import VCF

VCF_URL = "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV/1kGP_high_coverage_Illumina.chr22.filtered.SNV_INDEL_SV_phased_panel.vcf.gz"
TBI_URL = VCF_URL + ".tbi"
VCF_HASH = "md5:a2653cc7a1c8d03a96ca4f14d0fabdd2"

CACHE = Path(__file__).parent / "data" / "issue36_cache"


pytestmark = pytest.mark.network


@pytest.fixture(scope="session")
def vcf_1kgp() -> VCF:
    CACHE.mkdir(parents=True, exist_ok=True)
    vcf_path = Path(
        pooch.retrieve(
            url=VCF_URL, known_hash=VCF_HASH, path=CACHE, fname="chr22.vcf.gz"
        )
    )
    pooch.retrieve(url=TBI_URL, known_hash=None, path=CACHE, fname="chr22.vcf.gz.tbi")
    vcf = VCF(vcf_path)
    vcf._write_gvi_index()
    vcf._load_index()
    return vcf


def test_chunk_with_length_empty_region_misalignment(vcf_1kgp: VCF):
    """Two regions: first has 0 variants, second has >0 variants.

    Before fix: var_counts returns [count_region1] (length 1 not 2).
    zip misaligns — region 0 gets count_region1, tries to read that many
    variants from a 0-variant range -> ValueError.

    After fix: var_counts returns [0, count_region1]; region 0 is
    correctly skipped; region 1 reads normally.

    Region [17165208, 17165244) is 36 bp on chr22 — very likely 0 variants.
    Region [17181485, 17181576) is 91 bp — likely has variants.
    """
    contig = "chr22"
    starts = [17165208, 17181485]
    ends = [17165244, 17181576]

    for region_chunks in vcf_1kgp._chunk_ranges_with_length(contig, starts, ends):
        for _chunk, _end, _n_ext in region_chunks:
            pass
