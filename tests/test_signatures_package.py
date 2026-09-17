"""Guard the private import surface of genoray._signatures.

genoray's underscore modules are imported across repos (GenVarLoader), and
two modules in this package import from _signatures directly. The package
split must not move any of these names.
"""

from __future__ import annotations

import importlib

import pytest

# Every name that resolved as `genoray._signatures.<name>` before the package
# split. Adding to this list is fine; removing from it is a breaking change.
PRESERVED_NAMES = [
    "fit_signatures",
    "cosmic_signatures",
    "Criterion",
    "_fit_one",
    "_cosine",
    "_nnls",
    "_poisson_ll",
    "_load_signature_file",
    "_COSMIC_REGISTRY",
    "_KIND_TOKEN",
]


@pytest.mark.parametrize("name", PRESERVED_NAMES)
def test_name_still_importable_from_signatures(name: str):
    mod = importlib.import_module("genoray._signatures")
    assert hasattr(mod, name), f"genoray._signatures.{name} disappeared"


def test_signatures_is_a_package():
    mod = importlib.import_module("genoray._signatures")
    assert hasattr(mod, "__path__"), "_signatures should be a package"


def test_internal_importers_still_work():
    """The two sibling modules that import from _signatures directly."""
    from genoray._svar._annotate import SparseVarAnnotateMixin  # noqa: F401
    from genoray._svar2_mutcat import _MutcatMixin  # noqa: F401
