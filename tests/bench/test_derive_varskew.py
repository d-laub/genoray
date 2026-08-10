"""The derived profile must reproduce the source profile's per-contig
variant PROPORTIONS, because that is the only thing it exists to fix."""

from __future__ import annotations

from scripts.bench_svar2.profiles.derive_varskew import derive_varskew


def _profile(counts: dict[str, int]) -> dict:
    return {
        "name": "src",
        "provenance": {"n_samples_source": 3202, "fitted_on": "2026-07-16"},
        "fitted": {
            "contigs": [
                # Densities are deliberately near-uniform, like the real
                # profile: that is what makes the density split useless.
                {"id": cid, "n_variants": n, "density_per_kb": 23.0}
                for cid, n in counts.items()
            ]
        },
    }


def test_density_is_rescaled_to_variant_counts():
    src = _profile({"1": 6_000_000, "2": 1_000_000})
    out = derive_varskew(src)
    d = {c["id"]: c["density_per_kb"] for c in out["fitted"]["contigs"]}
    # distribute_by_density weights by density, so the RATIO is what matters.
    assert d["1"] / d["2"] == 6.0


def test_source_profile_is_not_mutated():
    src = _profile({"1": 6_000_000, "2": 1_000_000})
    derive_varskew(src)
    assert src["fitted"]["contigs"][0]["density_per_kb"] == 23.0


def test_provenance_records_the_source_and_the_transform():
    out = derive_varskew(_profile({"1": 10, "2": 20}))
    assert out["name"] == "germline-1kgp-varskew"
    assert out["provenance"]["derived_from"] == "src"
    assert "density_per_kb" in out["provenance"]["derivation"]


def test_variant_counts_are_left_alone():
    """Only the split weight changes. n_variants stays as fitted, so the
    file still documents the real cohort it came from."""
    out = derive_varskew(_profile({"1": 6_000_000, "2": 1_000_000}))
    n = {c["id"]: c["n_variants"] for c in out["fitted"]["contigs"]}
    assert n == {"1": 6_000_000, "2": 1_000_000}
