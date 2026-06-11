from __future__ import annotations

import pysam
import pytest

from genoray._reference import Reference


@pytest.fixture
def tiny_fasta(tmp_path):
    # contig "chr1": positions 0..9 = ACGTACGTAC
    fa = tmp_path / "ref.fa"
    fa.write_text(">chr1\nACGTACGTAC\n>chr2\nTTTTGGGG\n")
    pysam.faidx(str(fa))  # writes ref.fa.fai
    return fa


def test_fetch_single_window(tiny_fasta):
    ref = Reference.from_path(tiny_fasta)
    # 0-based, half-open [2, 5) -> "GTA"
    seq = ref.fetch("chr1", 2, 5)
    assert bytes(seq) == b"GTA"


def test_fetch_handles_chr_prefix_mismatch(tiny_fasta):
    ref = Reference.from_path(tiny_fasta)
    # query without "chr" prefix must still resolve via ContigNormalizer
    seq = ref.fetch("1", 0, 4)
    assert bytes(seq) == b"ACGT"


def test_fetch_out_of_bounds_pads_with_N(tiny_fasta):
    ref = Reference.from_path(tiny_fasta)
    # left of contig start -> padded with N
    seq = ref.fetch("chr1", -2, 3)
    assert bytes(seq) == b"NNACG"


def test_fetch_right_oob_pads_with_N(tiny_fasta):
    ref = Reference.from_path(tiny_fasta)
    # chr1 has 10 bases; positions 8,9 = "AC", then 2 N pads
    seq = ref.fetch("chr1", 8, 12)
    assert bytes(seq) == b"ACNN"
