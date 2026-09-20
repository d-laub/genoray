"""Parity oracle: our classifier vs the REAL SigProfilerClusters.

Runs only where SigProfilerClusters is installed (the ``sigprofiler`` pixi
env). Uses upstream's own functions with ``correction=False`` and the same
IMD cutoffs, then compares per-mutation labels.

Two harness details are load-bearing:

* Off-by-one: :meth:`VcfBuilder.record` takes a 1-based VCF POS and SVAR2
  stores 0-based positions, while upstream keys its output rows by the VCF
  POS. Our decoded position ``p`` therefore maps to upstream key ``p + 1``.
* Upstream ends ``findClustersOfClusters*`` by fanning ``generateMatrices``
  out over a ``multiprocessing.Pool``. The real function needs a
  SigProfilerMatrixGenerator reference genome and renders plots, and a
  lambda cannot be pickled to a worker at all, so the tests replace the
  module's pool with an in-process shim and ``generateMatrices`` with a
  no-op. No processes are spawned and nothing is pickled.
* Upstream's VAF path only classifies a group when it reads the blank-line
  separator that follows it, and the first pass never writes one after the
  final group (the no-VAF path has a post-loop block that catches the last
  group; the VAF path does not). The VAF input therefore carries a sentinel
  mutation far enough away to start a new final group, so E6 is actually
  classified instead of silently dropped.
"""

from __future__ import annotations

import pickle
import types
from pathlib import Path

import numpy as np
import pytest
from genoray import SparseVar2
from genoray._svar2_clusters import DOUBLET, KATAEGIS, MBS, NONCLUSTERED, OMIKLI, OTHER
from genoray._svar2_fields import FormatField
from vcfixture import Number, Seq, Type, VcfBuilder

cf = pytest.importorskip("SigProfilerClusters.classifyFunctions")

CUTOFF = 1000.0

# (positions, vafs) per event; see the plan table.
EVENTS = [
    ([1000, 1001], [0.5, 0.5]),
    ([21000, 21001, 21002], [0.5, 0.5, 0.5]),
    ([41000, 41005], [0.5, 0.5]),
    ([61000, 61002, 61004, 61006], [0.5, 0.5, 0.5, 0.5]),
    ([81000, 81002, 81004], [0.5, 0.5, 0.9]),
    ([101000, 101001, 101100, 101101], [0.5, 0.5, 0.5, 0.5]),
]
LABEL_TO_CODE = {
    "ClassIA": DOUBLET,
    "ClassIB": MBS,
    "ClassIC": OMIKLI,
    "ClassII": KATAEGIS,
    "ClassIII": OTHER,
}


def _mutations() -> list[tuple[int, float]]:
    out: list[tuple[int, float]] = []
    for positions, vafs in EVENTS:
        out.extend(zip(positions, vafs))
    return out


def _upstream_rows(*, sentinel: bool) -> list[str]:
    header = [
        "project",
        "samples",
        "ID",
        "genome",
        "mutType",
        "chr",
        "start",
        "end",
        "ref",
        "alt",
        "mutClass",
        "IMDplot",
        "IMD",
        "VAF/CCF",
    ]
    rows = ["\t".join(header)]

    def row(pos: int, vaf: float, chrom: str = "chr1") -> str:
        return "\t".join(
            [
                "T",
                "S0",
                ".",
                "GRCh37",
                "SNP",
                chrom,
                str(pos),
                str(pos),
                "C",
                "A",
                "SOMATIC",
                "1",
                "1",
                str(vaf),
            ]
        )

    for pos, vaf in _mutations():
        rows.append(row(pos, vaf))
    if sentinel:
        # >= cutoff past E6, so it opens a fresh final group. Upstream never
        # classifies the last group on the VAF path, and it is never compared.
        rows.append(row(201_000, 0.5))
    return rows


def _noop_generate_matrices(*args, **kwargs):
    """In-process stand-in for upstream's ``generateMatrices``.

    The real function builds SigProfilerMatrixGenerator matrices and renders
    plots; label parity only needs the subclass text files written before the
    pool fan-out. Module-level so it stays picklable, though the synchronous
    pool below never pickles it.
    """
    return


class _SyncResult:
    """Minimal ``AsyncResult`` for :class:`_SyncPool`."""

    def __init__(self) -> None:
        self._value = None
        self._exc: BaseException | None = None

    def wait(self) -> None:
        return None

    def successful(self) -> bool:
        return self._exc is None

    def get(self):
        if self._exc is not None:
            raise self._exc
        return self._value


class _SyncPool:
    """Run ``apply_async`` jobs in-process instead of spawning workers.

    Upstream only ever submits module-level functions, but the test must not
    pickle them (a lambda is unpicklable) nor spawn processes, so the pool is
    shimmed for the duration of each test.
    """

    def __init__(self, *args, **kwargs) -> None:
        pass

    def apply_async(self, fn, args=(), kwds=None) -> _SyncResult:
        result = _SyncResult()
        try:
            result._value = fn(*args, **(kwds or {}))
        except BaseException as exc:  # noqa: BLE001 - mirror Pool semantics
            result._exc = exc
        return result

    def close(self) -> None:
        return None

    def join(self) -> None:
        return None


def _run_upstream(
    tmp_path: Path, *, vaf: bool, monkeypatch: pytest.MonkeyPatch
) -> dict[tuple[str, int], str]:
    pp = tmp_path / "proj"
    project_path = pp / "output" / "vcf_files" / "T_clustered"
    if vaf:
        target = project_path / "T_clustered_vaf.txt"
    else:
        target = project_path / "SNV" / "T_clustered.txt"
    target.parent.mkdir(parents=True)
    target.write_text("\n".join(_upstream_rows(sentinel=vaf)) + "\n")
    sims = pp / "output" / "simulations" / "data"
    sims.mkdir(parents=True)
    with open(sims / "imds.pickle", "wb") as f:
        pickle.dump({"S0": CUTOFF}, f)

    # generateMatrices spawns an mp.Pool and renders plots; label comparison
    # does not need it.
    monkeypatch.setattr(cf, "generateMatrices", _noop_generate_matrices)
    monkeypatch.setattr(cf, "mp", types.SimpleNamespace(Pool=_SyncPool))
    if vaf:
        cf.findClustersOfClusters(
            "T",
            False,
            str(pp) + "/",
            1_000_000,
            {},
            {},
            str(pp / "log.txt"),
            "GRCh37",
            1,
            {},
            correction=False,
        )
    else:
        cf.findClustersOfClusters_noVAF(
            "T",
            False,
            str(pp) + "/",
            1_000_000,
            {},
            {},
            str(pp / "log.txt"),
            "GRCh37",
            1,
            {},
            correction=False,
        )

    labels: dict[tuple[str, int], str] = {}
    for name in ("class1a", "class1b", "class1c", "class2", "class3"):
        path = project_path / "subclasses" / name / f"T_clustered_{name}.txt"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            row = line.split("\t")
            if len(row) < 8:
                continue
            subclass = next((t for t in reversed(row) if t.startswith("Class")), None)
            if subclass is not None:
                labels[(row[1], int(row[7]))] = subclass
    return labels


def _our_labels(tmp_path: Path, *, vaf: bool) -> dict[tuple[str, int], int]:
    doc = VcfBuilder(samples=["S0"], contigs=[("chr1", None)]).fmt("GT")
    if vaf:
        doc = doc.fmt("VAF", Number.ONE, Type.FLOAT)
    for pos, value in _mutations():
        kwargs = {"VAF": [[value]]} if vaf else {}
        doc = doc.record("chr1", pos, ref="C", alt=[Seq("A")], gt=["1|0"], **kwargs)
    out = tmp_path / ("vaf.svar2" if vaf else "novaf.svar2")
    SparseVar2.from_vcf(
        out,
        doc.write(
            tmp_path / ("vaf.vcf.gz" if vaf else "novaf.vcf.gz"),
            bgzip=True,
            index=True,
        ),
        no_reference=True,
        format_fields=[FormatField("VAF", dtype="f32")] if vaf else None,
    )
    store = SparseVar2(out)
    store.annotate_clusters(imd_cutoff=CUTOFF, vaf_field="VAF" if vaf else None)
    sv = SparseVar2(out).with_fields(["cluster_class"])
    rag = sv.decode("chr1", [(0, 1_000_000)])
    positions = np.asarray(rag["pos"].data)
    codes = np.asarray(rag["cluster_class"].data)
    # SVAR2 positions are 0-based; upstream rows carry the VCF POS (1-based).
    return {("S0", int(p) + 1): int(c) for p, c in zip(positions, codes)}


def test_vaf_parity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    upstream = _run_upstream(tmp_path, vaf=True, monkeypatch=monkeypatch)
    ours = _our_labels(tmp_path, vaf=True)
    assert ours, "fixture produced no labels"
    assert NONCLUSTERED not in ours.values(), (
        "fixture must be closed under the pre-filter"
    )
    for key, subclass in upstream.items():
        assert ours[key] == LABEL_TO_CODE[subclass], f"{key}: {subclass}"
    # upstream skips len-1 events; those are our OTHER
    for key, code in ours.items():
        if key not in upstream:
            assert code == OTHER, f"{key} unlabeled upstream but ours={code}"


def test_no_vaf_parity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    upstream = _run_upstream(tmp_path, vaf=False, monkeypatch=monkeypatch)
    ours = _our_labels(tmp_path, vaf=False)
    assert ours, "fixture produced no labels"
    assert NONCLUSTERED not in ours.values(), (
        "fixture must be closed under the pre-filter"
    )
    for key, subclass in upstream.items():
        assert ours[key] == LABEL_TO_CODE[subclass], f"{key}: {subclass}"
    # no-VAF upstream drops failing events; those are our OTHER
    for key, code in ours.items():
        if key not in upstream:
            assert code == OTHER, f"{key} dropped upstream but ours={code}"
