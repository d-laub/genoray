"""Tests for issues #153 and #200: the conversion pipelines' grouped arguments.

`run_conversion_pipeline` had grown to 20 user-facing parameters, most of them
positionally interchangeable by type. The two field manifests were the sharp
edge: `info_fields` and `format_fields` have the identical type, so transposing
them at a call site compiled and ran, routing every INFO field into the FORMAT
section and vice versa.

These tests pin the two properties that make that unrepresentable rather than
merely unlikely -- the entry point takes no positional arguments at all, and the
structs cannot be built positionally either.

#200 extended the same treatment to the four sibling entry points. Two of them
carried their own version of the hazard: `run_vcf_list_conversion_pipeline` has
the identical `info_fields`/`format_fields` pair #153 names, and
`run_svar1_conversion_pipeline` has four identically typed
`{ref,alt}_{bytes,offsets}_per_contig` vectors. The tests below cover all five
entry points together, since the point of #200 is that they now agree.
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


# ---- #200: the four sibling entry points ----

_ENTRY_POINTS = [
    "run_conversion_pipeline",
    "run_pgen_conversion_pipeline",
    "run_vcf_list_conversion_pipeline",
    "run_svar1_conversion_pipeline",
    "run_slice_view",
]


@pytest.mark.parametrize("name", _ENTRY_POINTS)
def test_entry_points_take_no_positional_arguments(name):
    """Item 1 of #200. This alone removes the positional hazard for the bare
    strings, and for svar1's four interchangeable per-contig byte vectors."""
    with pytest.raises(TypeError):
        getattr(_core, name)("some-path")


def test_vcf_list_rejects_max_mem_bytes():
    """#200's caveat: this path plans by core count and never honours a byte
    budget. `PlanSettings` carries the field for the pipelines that do use it,
    so this one rejects it rather than dropping it silently -- a silent drop
    would trade a positional hazard for a semantic one."""
    with pytest.raises(ValueError, match="max_mem_bytes"):
        _core.run_vcf_list_conversion_pipeline(
            vcf_paths=["does-not-exist.vcf"],
            reference_path=None,
            output_dir="out",
            regions=RegionSpec(chroms=["chr1"]),
            fields=FieldSpec(),
            plan=PlanSettings(max_mem_bytes=1 << 30),
            contig_membership=[[True]],
        )


def test_svar1_rejects_max_mem_bytes():
    """Same contract as the VCF-list path."""
    with pytest.raises(ValueError, match="max_mem_bytes"):
        _core.run_svar1_conversion_pipeline(
            svar1_dir="does-not-exist",
            reference_path=None,
            output_dir="out",
            regions=RegionSpec(chroms=["chr1"], samples=["S0"]),
            plan=PlanSettings(max_mem_bytes=1 << 30),
            contig_starts=[0],
            contig_lens=[0],
            pos_per_contig=[[]],
            ref_bytes_per_contig=[[]],
            ref_offsets_per_contig=[[0]],
            alt_bytes_per_contig=[[]],
            alt_offsets_per_contig=[[0]],
            format_fields=[],
            format_src_dtypes=[],
            sample_idx=[0],
        )


def test_pgen_still_honours_max_mem_bytes():
    """The counterpart: `max_mem_bytes` is real on the PGEN path, so setting it
    must NOT raise. A bad `regions_overlap` proves the call got past the budget
    check and into the fail-fast band, before any output byte is written."""
    with pytest.raises(ValueError, match="overlap"):
        _core.run_pgen_conversion_pipeline(
            pgen_path="does-not-exist.pgen",
            pvar_path="does-not-exist.pvar",
            reference_path=None,
            output_dir="out",
            regions=RegionSpec(chroms=["chr1"], regions_overlap="nonsense"),
            plan=PlanSettings(max_mem_bytes=1 << 30),
            contig_ranges=[(0, 0)],
            dosage_fields=[],
            readers=[[]],
            dosage_readers=[[]],
            sample_perm=[],
        )
