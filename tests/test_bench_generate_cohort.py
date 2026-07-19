# tests/test_bench_generate_cohort.py
from __future__ import annotations
import importlib
import sys
from pathlib import Path

from cyvcf2 import VCF

_SCRIPTS_DIR = Path(__file__).parent.parent / "scripts" / "bench_from_vcf_list"


def _load():
    # Import as a real top-level module (dir on sys.path) rather than via
    # spec_from_file_location, so the parallel default (jobs=None ->
    # ProcessPoolExecutor) survives every multiprocessing start method. Under
    # spawn/forkserver -- the Linux default since Python 3.14 -- each worker is
    # a fresh interpreter that re-imports the worker function's module BY NAME;
    # a spec-loaded module isn't importable that way, so the worker dies with
    # ModuleNotFoundError and the pool breaks (fork only worked because the
    # child inherited the parent's sys.modules). spawn/forkserver propagate
    # sys.path to workers, so `import generate_cohort` resolves there too.
    if str(_SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS_DIR))
    return importlib.import_module("generate_cohort")


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
