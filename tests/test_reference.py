from __future__ import annotations

import numpy as np
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


def _write_fasta(path, seq, contig="chr1"):
    path.write_text(f">{contig}\n{seq}\n")
    pysam.faidx(str(path))


def test_contig_array_matches_fetch(tmp_path):
    fa = tmp_path / "ref.fa"
    _write_fasta(fa, "ACGTACGTAC")
    ref = Reference.from_path(fa)
    arr = ref.contig_array("chr1")
    assert isinstance(arr, np.ndarray) and arr.dtype == np.uint8
    assert bytes(arr) == b"ACGTACGTAC"
    # equals fetch over the full span
    assert bytes(ref.fetch("chr1", 0, 10)) == bytes(arr)
    # chr-prefix normalization still works
    assert bytes(ref.contig_array("1")) == b"ACGTACGTAC"
