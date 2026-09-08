from __future__ import annotations

import dataclasses

import pytest

from genoray import Tuning


def test_defaults_are_all_none():
    t = Tuning()
    assert t._as_ffi() == {
        "concurrent_chroms": None,
        "reader_workers": None,
        "overshard": None,
        "dense_cap": None,
        "merge_threads": None,
        "sample_interval": None,
    }


def test_frozen_and_kw_only():
    t = Tuning(reader_workers=20)
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.reader_workers = 3  # type: ignore[misc]
    with pytest.raises(TypeError):
        Tuning(2)  # type: ignore[misc]  # positional args rejected


def test_replace_works():
    t = Tuning(reader_workers=20, overshard=40)
    assert dataclasses.replace(t, overshard=4).overshard == 4


@pytest.mark.parametrize(
    "field",
    ["concurrent_chroms", "reader_workers", "overshard", "dense_cap", "merge_threads"],
)
@pytest.mark.parametrize("bad", [0, -1])
def test_below_one_rejected(field, bad):
    with pytest.raises(ValueError, match=field):
        Tuning(**{field: bad})


def test_sample_interval_zero_allowed_negative_rejected():
    assert Tuning(sample_interval=0).sample_interval == 0
    with pytest.raises(ValueError, match="sample_interval"):
        Tuning(sample_interval=-1)


# The applicability matrix, straight from the spec's section A table.
@pytest.mark.parametrize(
    ("backend", "field"),
    [
        ("pgen", "reader_workers"),
        ("pgen", "overshard"),
        ("vcf_list", "concurrent_chroms"),
        ("vcf_list", "reader_workers"),
        ("vcf_list", "overshard"),
        ("svar1", "reader_workers"),
        ("svar1", "overshard"),
    ],
)
def test_inapplicable_field_rejected_naming_field_and_backend(backend, field):
    t = Tuning(**{field: 2})
    with pytest.raises(ValueError) as excinfo:
        t._check_backend(backend)
    msg = str(excinfo.value)
    assert field in msg
    assert backend in msg


@pytest.mark.parametrize(
    ("backend", "field"),
    [
        ("vcf", "reader_workers"),
        ("vcf", "overshard"),
        ("vcf", "concurrent_chroms"),
        ("pgen", "concurrent_chroms"),
        ("svar1", "concurrent_chroms"),
        ("vcf_list", "dense_cap"),
        ("vcf_list", "merge_threads"),
        ("vcf_list", "sample_interval"),
    ],
)
def test_applicable_field_accepted(backend, field):
    Tuning(**{field: 2})._check_backend(backend)  # must not raise


def test_none_fields_never_rejected():
    for backend in ("vcf", "pgen", "vcf_list", "svar1"):
        Tuning()._check_backend(backend)


def test_unknown_backend_is_a_programming_error():
    with pytest.raises(KeyError):
        Tuning()._check_backend("nope")
