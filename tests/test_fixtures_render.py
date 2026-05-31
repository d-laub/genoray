from __future__ import annotations

import cyvcf2
import pytest

from tests.data.fixtures import FIXTURES


@pytest.mark.parametrize(
    "name", ["biallelic", "multiallelic", "indels", "three_samples_unsorted"]
)
def test_fixture_renders_and_parses(tmp_path, name):
    builder = FIXTURES[name]()
    path = builder.write(tmp_path / f"{name}.vcf")
    vcf = cyvcf2.VCF(str(path))
    records = list(vcf)
    assert len(records) > 0


def test_biallelic_record_count():
    builder = FIXTURES["biallelic"]()
    assert len(builder.build().records) == 6


def test_indels_positions():
    builder = FIXTURES["indels"]()
    pos = [r.pos for r in builder.build().records]
    assert pos[0] == 1000 and 5000 in pos
    assert 1011 in pos and 1021 in pos
    assert 3031 in pos and 3070 in pos
