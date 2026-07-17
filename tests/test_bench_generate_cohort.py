# tests/test_bench_generate_cohort.py
from __future__ import annotations
import importlib.util
from pathlib import Path

from cyvcf2 import VCF

_SPEC = (
    Path(__file__).parent.parent
    / "scripts"
    / "bench_from_vcf_list"
    / "generate_cohort.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("generate_cohort", _SPEC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_generate_cohort_shape(tmp_path: Path):
    gen = _load()
    manifest = gen.generate_cohort(
        tmp_path,
        n_files=4,
        n_variants=6,
        contigs=["chr1"],
        contig_len=10_000,
        shared_frac=0.5,
        indel_frac=0.25,
        seed=7,
    )
    paths = [Path(p) for p in manifest.read_text().split()]
    assert len(paths) == 4
    for p in paths:
        assert p.exists() and p.suffix == ".gz"
        v = VCF(str(p))
        assert len(v.samples) == 1  # single-sample
        rows = list(v("chr1"))
        assert rows == sorted(rows, key=lambda r: r.POS)  # position-sorted
    # shared sites: at least one POS appears in >= 2 files, and every shared POS
    # carries an IDENTICAL REF/ALT across the files so genoray's k-way merge
    # (keyed on (pos, ilen, alt)) actually joins them instead of emitting
    # separate rows or hard-erroring on a cross-file REF disagreement.
    from collections import Counter, defaultdict

    pos_counts: Counter[int] = Counter()
    variants_by_pos: dict[int, set[tuple[str, str]]] = defaultdict(set)
    for p in paths:
        for r in VCF(str(p))("chr1"):
            pos_counts[r.POS] += 1
            variants_by_pos[r.POS].add((r.REF, r.ALT[0]))
    shared_positions = [pos for pos, c in pos_counts.items() if c >= 2]
    assert shared_positions  # at least one POS appears in >= 2 files
    for pos in shared_positions:
        assert len(variants_by_pos[pos]) == 1  # identical REF/ALT across files


def test_generate_cohort_no_cross_file_ref_disagreement(tmp_path: Path):
    """At scale, private positions collide ACROSS files (birthday paradox in a
    finite contig). Every position that ends up in >= 2 files -- shared-pool OR a
    coincidental private/private collision -- must carry an identical REF/ALT, or
    from_vcf_list's no_reference k-way merge hard-errors on a cross-file REF
    disagreement. Force heavy private collisions with a tiny contig + many files.
    """
    from collections import Counter, defaultdict

    gen = _load()
    manifest = gen.generate_cohort(
        tmp_path,
        n_files=10,
        n_variants=40,
        contigs=["chr1"],
        contig_len=100,  # tiny -> private positions collide across files
        shared_frac=0.1,
        indel_frac=0.2,
        seed=3,
    )
    paths = [Path(p) for p in manifest.read_text().split()]
    pos_counts: Counter[int] = Counter()
    variants_by_pos: dict[int, set[tuple[str, str]]] = defaultdict(set)
    for p in paths:
        for r in VCF(str(p))("chr1"):
            pos_counts[r.POS] += 1
            variants_by_pos[r.POS].add((r.REF, r.ALT[0]))
    collisions = [pos for pos, c in pos_counts.items() if c >= 2]
    assert collisions  # the tiny contig must actually force cross-file collisions
    for pos in collisions:
        assert len(variants_by_pos[pos]) == 1, (
            f"pos {pos} has disagreeing REF/ALT across files: {variants_by_pos[pos]}"
        )
