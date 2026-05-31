from __future__ import annotations

import pytest

from tests.data.fixtures import FIXTURES


@pytest.fixture(scope="session")
def truths():
    """name -> vcfixture GroundTruth for every fixture."""
    return {name: build().truth() for name, build in FIXTURES.items()}


@pytest.fixture(scope="session")
def biallelic_truth(truths):
    return truths["biallelic"]


@pytest.fixture(scope="session")
def multiallelic_truth(truths):
    return truths["multiallelic"]


@pytest.fixture(scope="session")
def indels_truth(truths):
    return truths["indels"]


@pytest.fixture(scope="session")
def three_samples_truth(truths):
    return truths["three_samples_unsorted"]
