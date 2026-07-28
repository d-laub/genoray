"""Shared record schema for the SVAR2 scale-bench harness.

Every other module in this package reads or writes these types. They are frozen
so a record cannot be mutated after a run is recorded, and the JSON codec
restores tuples explicitly because JSON has no tuple type -- without that,
round-tripped records compare unequal and resumption silently re-runs points.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class CorpusManifest:
    """Shape of one generated corpus. Written next to the .vcf.gz.

    Consumers read shape from here rather than parsing filenames, so a corpus
    can be renamed or relocated without breaking a sweep.
    """

    path: str
    samples: int
    variants: int
    contigs: tuple[str, ...]
    format_fields: tuple[str, ...]
    ploidy: int
    cells: int
    compressed_bytes: int
    seed: int
    generator_version: int

    @property
    def chunk_bytes(self) -> int:
        """Analytic bytes of one dense chunk at `chunk_size` variants.

        Mirrors `_auto_chunk_size`'s cost model: packed presence grid plus
        staged FORMAT values. Callers multiply by chunk_size.
        """
        grid = (self.samples * self.ploidy) // 8
        fmt = len(self.format_fields) * self.samples * 4
        return grid + fmt


@dataclass(frozen=True)
class SweepPoint:
    """One configuration to measure. `point_id` is content-derived so a
    resumed sweep can skip points it already recorded."""

    corpus: str
    reader_workers: int
    concurrent_chroms: int | None
    shard_htslib: int
    overshard: int
    chunk_size: int
    threads: int
    reps: int
    rss_ceiling_mb: int | None = None

    @property
    def point_id(self) -> str:
        payload = json.dumps(dataclasses.asdict(self), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class ProbeRecord:
    """Result of one instrumented conversion run.

    `ok=False` with `oom_at_rss_mb` set is a legitimate datum, not an error --
    demonstrating that the current chunk_size OOMs at scale is a deliverable.
    """

    point_id: str
    ok: bool
    wall_s: float
    phase1_s: float
    cpu_s: float
    maxrss_mb: float
    digest: str
    dense_cap: int
    dense_occupancy: tuple[int, ...]
    cpu_shard_pct: tuple[float, ...]
    cpu_exec_pct: tuple[float, ...]
    pending_highwater: int
    pending_bytes_highwater: int
    shard_unit_secs: tuple[float, ...]
    oom_at_rss_mb: float | None = None
    error: str | None = None


@dataclass(frozen=True)
class VLaw:
    """phase1_s ~ intercept + slope * variants."""

    slope_s_per_variant: float
    intercept_s: float
    r2: float
    n_points: int
    max_extrapolation_factor: float


@dataclass(frozen=True)
class CostLaw:
    """cost(S) = alpha * S**beta, fitted on logs."""

    name: str
    alpha: float
    beta: float
    beta_ci95: tuple[float, float]
    n_points: int


@dataclass(frozen=True)
class RamLaw:
    """peak_rss_mb ~ base_mb + kappa * (workers + pending_hw) * chunk_bytes."""

    base_mb: float
    kappa: float
    r2: float
    n_points: int


@dataclass(frozen=True)
class Verdict:
    hypothesis: str  # "H1" | "H2" | "H3" | "none"
    rationale: str
    evidence: dict[str, Any] = field(default_factory=dict)


# --- codecs -----------------------------------------------------------------


def _tuple_fields(cls: type) -> dict[str, type]:
    """Field names whose annotation is a tuple type. JSON round-trips them as
    lists, so they must be re-coerced or frozen-dataclass equality fails."""
    hints = typing.get_type_hints(cls)
    out = {}
    for f in dataclasses.fields(cls):
        origin = typing.get_origin(hints[f.name])
        if origin is tuple:
            args = typing.get_args(hints[f.name])
            out[f.name] = args[0] if args else str
    return out


def to_json(obj: Any) -> str:
    return json.dumps(dataclasses.asdict(obj), sort_keys=True)


def from_json(cls: type[T], s: str) -> T:
    raw = json.loads(s)
    coerce = _tuple_fields(cls)
    for name, elem in coerce.items():
        if name in raw and raw[name] is not None:
            raw[name] = tuple(elem(v) for v in raw[name])
    known = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in raw.items() if k in known})


def append_ndjson(path: Path, obj: Any) -> None:
    """Append one record and fsync. A preempted Slurm job must not lose the
    point it just finished paying for."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(to_json(obj) + "\n")
        fh.flush()
        import os

        os.fsync(fh.fileno())


def read_ndjson(path: Path, cls: type[T]) -> list[T]:
    if not Path(path).exists():
        return []
    return [
        from_json(cls, line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]
