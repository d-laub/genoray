"""Tests for issue #153: the conversion pipeline's grouped arguments.

`run_conversion_pipeline` had grown to 20 user-facing parameters, most of them
positionally interchangeable by type. The two field manifests were the sharp
edge: `info_fields` and `format_fields` have the identical type, so transposing
them at a call site compiled and ran, routing every INFO field into the FORMAT
section and vice versa.

These tests pin the two properties that make that unrepresentable rather than
merely unlikely -- the entry point takes no positional arguments at all, and the
structs cannot be built positionally either.
"""

from __future__ import annotations

import dataclasses

import pytest

from genoray import _core
from genoray._pipeline_args import FieldSpec, PlanSettings, RegionSpec

_SPECS = [FieldSpec, RegionSpec, PlanSettings]


@pytest.mark.parametrize("spec", _SPECS, ids=lambda s: s.__name__)
def test_specs_cannot_be_built_positionally(spec):
    """`kw_only=True` is the guarantee. Without it, `FieldSpec(format_, info)`
    would be the same silent transposition one level down."""
    with pytest.raises(TypeError):
        spec([])


@pytest.mark.parametrize("spec", _SPECS, ids=lambda s: s.__name__)
def test_specs_are_frozen(spec):
    """They cross the FFI boundary by attribute read; nothing should be able to
    mutate one between construction and the call."""
    inst = spec()
    name = dataclasses.fields(inst)[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(inst, name, None)


def test_field_spec_keeps_info_and_format_apart():
    """The transposition the issue names: distinguishable by name, not order."""
    info = [("AF", "info", "f32", None, None)]
    fmt = [("DS", "format", "f32", None, None)]
    spec = FieldSpec(info=info, format=fmt)
    assert spec.info == info
    assert spec.format == fmt


def test_run_conversion_pipeline_takes_no_positional_arguments():
    """Keyword-only at the boundary: the remaining bare strings (vcf_path,
    output_dir, check_ref, log_level) have no position to get wrong."""
    with pytest.raises(TypeError):
        _core.run_conversion_pipeline("some.vcf")


def test_run_conversion_pipeline_requires_the_grouped_arguments():
    """`regions`/`fields`/`plan` have no defaults, so a caller cannot omit one
    and silently convert an empty contig list."""
    with pytest.raises(TypeError):
        _core.run_conversion_pipeline(
            vcf_path="some.vcf",
            reference_path=None,
            output_dir="out",
        )


def test_run_conversion_pipeline_reads_the_specs_by_attribute():
    """The Rust side extracts fields by name, so any object with the right
    attributes works -- which is what makes the dataclasses a transport shape
    rather than a type the FFI is coupled to. A bad `regions_overlap` proves it
    got that far: the mode is parsed up front, before any output is written."""
    with pytest.raises(ValueError, match="overlap"):
        _core.run_conversion_pipeline(
            vcf_path="does-not-exist.vcf",
            reference_path=None,
            output_dir="out",
            regions=RegionSpec(chroms=["chr1"], regions_overlap="nonsense"),
            fields=FieldSpec(),
            plan=PlanSettings(),
        )
