"""Corpus generation shells out to two external binaries. The tests that
need them are skipped when they are absent -- the vcfixture bulk CLI is NOT
the PyPI `vcfixture` package and is not in this project's pixi env, so an
unguarded test passes locally and fails only in CI."""

from __future__ import annotations

import shutil

import pytest

from scripts.bench_svar2.pgen_corpus import (
    PgenCorpusSpec,
    resolve_vcfixture,
)


def _have_tools() -> bool:
    try:
        resolve_vcfixture()
    except FileNotFoundError:
        return False
    return shutil.which("plink2") is not None


needs_tools = pytest.mark.skipif(
    not _have_tools(), reason="vcfixture bulk CLI and/or plink2 unavailable"
)


def test_resolve_vcfixture_error_names_the_install_command(monkeypatch):
    monkeypatch.delenv("VCFIXTURE_BIN", raising=False)
    monkeypatch.setattr(shutil, "which", lambda _: None)
    with pytest.raises(FileNotFoundError, match="cargo install vcfixture"):
        resolve_vcfixture()


def test_stem_is_unique_per_shape():
    """Manifests land FLAT in one corpora dir and `model._load_manifests`
    keys them by FILENAME, globbing `*.manifest.json` non-recursively. Two
    corpora sharing a stem would silently overwrite each other's manifest."""
    from scripts.bench_svar2.pgen_corpus import corpus_stem

    a = corpus_stem(PgenCorpusSpec(4000, 250_000, ("chr1",), 42))
    b = corpus_stem(PgenCorpusSpec(4000, 500_000, ("chr1",), 42))
    c = corpus_stem(PgenCorpusSpec(32_000, 250_000, ("chr1",), 42))
    assert len({a, b, c}) == 3
    # build_plans and tests/bench/test_build_plans.py both parse the shape
    # back out of this stem.
    assert a == "pgen_s4000_v250000"


@needs_tools
def test_generate_produces_a_readable_pgen(tmp_path):
    from genoray import SparseVar2

    from scripts.bench_svar2.pgen_corpus import generate

    spec = PgenCorpusSpec(
        samples=20, variants=6000, contigs=("chr1", "chr2", "chr3"), seed=7
    )
    manifest = generate(spec, tmp_path)

    assert manifest.path.endswith(".pgen")
    assert manifest.samples == 20
    assert (tmp_path / "pgen_s20_v6000.manifest.json").exists()

    store = tmp_path / "roundtrip.svar"
    SparseVar2.from_pgen(
        store,
        manifest.path,
        no_reference=True,
        skip_out_of_scope=True,
        log_level="off",
    )
    sv = SparseVar2(store)
    # plink2 strips the chr prefix unless --output-chr is passed; generate()
    # passes it, so the store must carry the prefixed names.
    assert sv.contigs == ["chr1", "chr2", "chr3"]


@needs_tools
def test_generate_is_cached(tmp_path):
    from scripts.bench_svar2.pgen_corpus import generate

    spec = PgenCorpusSpec(samples=10, variants=3000, contigs=("chr1",), seed=1)
    first = generate(spec, tmp_path)
    pgen = tmp_path / "pgen_s10_v3000.pgen"
    mtime = pgen.stat().st_mtime_ns
    second = generate(spec, tmp_path)
    assert first == second
    assert pgen.stat().st_mtime_ns == mtime, "a cached corpus must not be regenerated"
