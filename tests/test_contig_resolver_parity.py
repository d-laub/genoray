"""Anti-drift guard: the Rust FASTA contig resolver (src/vcf_reader.rs
`resolve_contig_name`) MUST agree with genoray's Python ContigNormalizer on this
shared table. The identical table is asserted in Rust in vcf_reader.rs's tests
module; keep the two in lockstep.
"""

import pytest

from genoray._contigs import ContigNormalizer

# (fasta_contigs, query, expected_or_None)
PARITY_CASES = [
    (["chr1", "chr2", "chrM"], "1", "chr1"),
    (["chr1", "chr2", "chrM"], "chr1", "chr1"),
    (["1", "2", "MT"], "chr1", "1"),
    (["1", "2", "MT"], "1", "1"),
    (["chr1", "chrM"], "MT", "chrM"),  # mito alias
    (["chr1", "chrM"], "chrMT", "chrM"),
    (["1", "MT"], "chrM", "MT"),
    (["chr1", "chr2"], "chrZ", None),  # genuine miss
    (["chr1", "chr2"], "MT", None),  # no mito in reference
]


@pytest.mark.parametrize("contigs,query,expected", PARITY_CASES)
def test_contignormalizer_matches_parity_table(contigs, query, expected):
    assert ContigNormalizer(contigs).norm(query) == expected
