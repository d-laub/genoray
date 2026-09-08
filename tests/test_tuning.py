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


# `None` is absent on purpose: it is the valid "let the planner choose" value.
@pytest.mark.parametrize("bad", ["8", 2.5, 3.0, b"4", (), [2]])
def test_non_integer_values_are_rejected(bad: object):
    with pytest.raises(ValueError, match="must be None or an int"):
        Tuning(reader_workers=bad)  # pyrefly: ignore[bad-argument-type]


@pytest.mark.parametrize("bad", [True, False])
def test_bool_is_rejected_despite_being_an_int(bad: bool):
    # `bool` is an `int` AND defines `__index__`, so it slips through both the
    # isinstance and the operator.index checks without an explicit guard.
    with pytest.raises(ValueError, match="must be None or an int"):
        Tuning(reader_workers=bad)


def test_numpy_integers_are_accepted_and_coerced():
    # This is a numpy-centric library: a caller who derives a knob from an array
    # gets an np.int64, which is not an `int`. It must be accepted, and it must
    # reach the FFI seam as a plain `int`.
    np = pytest.importorskip("numpy")
    t = Tuning(reader_workers=np.int64(4), sample_interval=np.uint8(0))
    assert t.reader_workers == 4
    assert type(t.reader_workers) is int
    assert t.sample_interval == 0
    assert type(t.sample_interval) is int
    assert all(v is None or type(v) is int for v in t._as_ffi().values())


def test_numpy_integers_below_the_minimum_are_still_rejected():
    np = pytest.importorskip("numpy")
    with pytest.raises(ValueError, match="reader_workers"):
        Tuning(reader_workers=np.int64(0))


def test_every_inapplicable_field_is_reported_at_once():
    # Fixing one knob per round-trip is the failure mode; both must be named.
    t = Tuning(reader_workers=2, overshard=3)
    with pytest.raises(ValueError) as excinfo:
        t._check_backend("pgen")
    msg = str(excinfo.value)
    assert "reader_workers" in msg
    assert "overshard" in msg
    assert "pgen" in msg
