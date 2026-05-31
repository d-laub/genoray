"""Regression test for sample/genotype alignment in :meth:`VCF.set_samples`.

Reproduces the bug surfaced upstream as
`mcvickerlab/GenVarLoader#159 <https://github.com/mcvickerlab/GenVarLoader/issues/159>`_.

The bug appears whenever the input VCF's sample column order isn't already
alphabetical: ``set_samples`` was computing ``np.argsort(s_idx)`` where it
should have been computing the permutation that sends caller-requested
sample order to VCF column order. With pre-sorted samples (the case
exercised by every other fixture in this test directory) the buggy and
correct permutations happen to coincide, so the bug was invisible until
someone fed in a cohort BCF with samples in non-alphabetical order.

The fixture used here has samples laid out in file order
``[sample_C, sample_A, sample_B]`` — deliberately not alphabetical — with
one phased SNV at ``chr1:100 T>A`` and unambiguous per-sample GTs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from genoray._vcf import VCF
from tests import _oracle
from tests.data.fixtures import FIXTURES

DDIR = Path(__file__).parent / "data"

# GroundTruth oracle for the three_samples_unsorted fixture.
# File-order samples: [sample_C(0), sample_A(1), sample_B(2)]
_THREE = FIXTURES["three_samples_unsorted"]().truth()


@pytest.mark.parametrize(
    "requested",
    [
        # Alphabetical user order against non-alphabetical VCF column order —
        # the case `gvl.write` triggers in practice.
        ["sample_A", "sample_B", "sample_C"],
        # Non-alphabetical user order — both directions of the permutation
        # have to compose correctly.
        ["sample_C", "sample_B", "sample_A"],
        # Subset, non-alphabetical.
        ["sample_C", "sample_A"],
    ],
)
def test_set_samples_preserves_genotype_alignment(requested: list[str]) -> None:
    """``gt[i, p, 0]`` must equal the oracle genotype for ``requested[i]`` for every (i, p)."""
    vcf = VCF(DDIR / "three_samples_unsorted.vcf.gz", phasing=False)
    vcf.set_samples(requested)
    # read returns shape (n_samples, ploidy, n_variants); we have 1 variant.
    gt = vcf.read("chr1", 99, 100, VCF.Genos8)
    # Build expected from oracle: genos(_THREE, [0]) -> (3, 2, 1) in file order
    # [sample_C(0), sample_A(1), sample_B(2)]; index axis-0 to match requested order.
    sample_positions = [_THREE.samples.index(n) for n in requested]
    expected = _oracle.genos(_THREE, [0])[sample_positions, :, 0].astype(np.int8)
    assert gt is not None, "fixture variant chr1:100 missing from VCF"
    np.testing.assert_array_equal(
        gt[..., 0],
        expected,
        err_msg=(
            f"requested order: {requested!r}\n"
            f"observed: {gt[..., 0].tolist()}\n"
            f"expected: {expected.tolist()}"
        ),
    )
    assert vcf.current_samples == requested
