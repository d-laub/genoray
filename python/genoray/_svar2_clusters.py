"""SVAR2 cluster-label surface: ``annotate_clusters`` on ``SparseVar2``.

Ports SigProfilerClusters' ``findClustersOfClusters`` (VAF mode) and
``findClustersOfClusters_noVAF`` (no-VAF mode) with ``correction=False`` into
a post-hoc annotation that writes the ``cluster_class`` FORMAT field. The
algorithm and every deliberate deviation are documented in
``docs/superpowers/specs/2026-09-19-svar2-cluster-labels-design.md``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

# Wire codebook; the Rust classifier pins the same values in
# ``src/cluster/classify.rs`` and the tests pin both.
NONCLUSTERED = 0
DOUBLET = 1
MBS = 2
OMIKLI = 3
KATAEGIS = 4
OTHER = 5
NOT_ANNOTATED = 255

#: Storage schema version stamped as ``cluster_version`` in ``meta.json``.
CLUSTER_VERSION = 1


class _ClustersMixin:
    """SigProfilerClusters subclassification over a finished SVAR2 store.

    Provided by the concrete ``SparseVar2`` host class (see
    ``SparseVar2.__init__``); declared here so the mixin's use of them
    type-checks in isolation.
    """

    path: Path
    contigs: list[str]
    available_samples: list[str]
    available_fields: dict[str, Any]
    _readers: dict[str, Any]

    def annotate_clusters(
        self,
        *,
        imd_cutoff: float | Mapping[str, float],
        vaf_field: str | None = None,
        vaf_cut: float = 0.1,
        contigs: Sequence[str] | None = None,
    ) -> None:
        """Subclassify clustered mutations per sample into ``cluster_class``.

        Args:
            imd_cutoff: Inter-mutational-distance cutoff(s), in bases. Either a
                single float for every sample or a mapping from sample name to
                cutoff (all samples required). Mutations whose minimum
                neighbour distance is ``<=`` their cutoff are clustered.
                Upstream derives these per sample by simulation; feed the same
                ``imds.pickle`` values for label-for-label parity.
            vaf_field: Optional FORMAT field holding per-sample VAF/CCF. When
                given, VAF-consistency participates in the decision tree and
                failed events are greedily re-split. Must be a 2- or 4-byte
                float field.
            vaf_cut: Maximum ``|delta VAF|`` for adjacent mutations to stay in
                one event (upstream default ``0.1``).
            contigs: If given, only these contigs (resolved against
                ``self.contigs``, alternate naming accepted) are annotated;
                every other contig gets 255-filled ``cluster_class`` streams
                so decoding stays coherent. ``None`` (default) annotates every
                contig in the store.

        Notes:
            Stamps ``meta.json`` with ``cluster_version``, ``cluster_contigs``,
            ``cluster_cutoff``, ``cluster_vaf_field``, ``cluster_vaf_cut``, and
            the ``cluster_class`` FORMAT entry, after every in-scope contig's
            files are in place. Re-running is unconditional: the entry is
            atomically unadvertised before any contig is rewritten and re-added
            by the final stamp, so a kill mid-run cannot leave the manifest
            pointing at a mix of old and new label files.

        Raises:
            ValueError: Unknown/missing samples in a ``imd_cutoff`` mapping, a
                non-positive cutoff or ``vaf_cut``, a ``vaf_field`` that is
                missing / not FORMAT / not a 2- or 4-byte float, or a
                ``contigs=`` name absent from the store.
        """
        from genoray._svar2_fields import _META_DTYPE

        if vaf_cut <= 0:
            raise ValueError(f"vaf_cut must be > 0, got {vaf_cut!r}")

        if isinstance(imd_cutoff, Mapping):
            unknown = [s for s in imd_cutoff if s not in self.available_samples]
            if unknown:
                raise ValueError(
                    f"imd_cutoff names unknown samples: {unknown}; "
                    f"available samples: {self.available_samples}"
                )
            missing = [s for s in self.available_samples if s not in imd_cutoff]
            if missing:
                raise ValueError(f"imd_cutoff is missing samples: {missing}")
            cutoffs = np.array(
                [float(imd_cutoff[s]) for s in self.available_samples], dtype=np.float64
            )
        else:
            cutoffs = np.full(
                len(self.available_samples), float(imd_cutoff), dtype=np.float64
            )
        if not np.all(cutoffs > 0):
            raise ValueError(f"imd_cutoff values must be > 0, got {cutoffs.tolist()}")

        vaf: tuple[str, str] | None = None
        if vaf_field is not None:
            field = self.available_fields.get(vaf_field)
            if field is None:
                matches = [
                    f for f in self.available_fields.values() if f.name == vaf_field
                ]
                field = matches[0] if len(matches) == 1 else None
            if field is None:
                raise ValueError(
                    f"vaf_field {vaf_field!r} is not in the store; available fields: "
                    f"{sorted(self.available_fields)}"
                )
            if field.category != "format":
                raise ValueError(
                    f"vaf_field {vaf_field!r} is an {field.category} field; expected FORMAT"
                )
            if field.dtype.kind != "f" or field.dtype.itemsize not in (2, 4):
                raise ValueError(
                    f"vaf_field {vaf_field!r} has dtype {field.dtype}; "
                    "expected a 2- or 4-byte float"
                )
            vaf = (field.name, _META_DTYPE[field.dtype])

        if contigs is None:
            scope = list(self.contigs)
        else:
            # _resolve_contigs raises ValueError naming the first miss.
            # pyrefly: ignore [missing-attribute]
            resolved = self._resolve_contigs(contigs)
            scope = list(dict.fromkeys(resolved))
            if not scope:
                raise ValueError("contigs= resolved to no store contigs")
        meta_path = self.path / "meta.json"
        if meta_path.exists():
            # Unadvertise before rewriting any contig: each contig's streams
            # commit independently, so a kill mid-re-run must not leave
            # meta.json pointing at a mix of old and new label files. The
            # final stamp below re-adds the entry (its dedupe keeps it single).
            meta = json.loads(meta_path.read_text())
            meta["fields"] = [
                f
                for f in meta.get("fields") or []
                if not (f["name"] == "cluster_class" and f["category"] == "format")
            ]
            tmp_path = meta_path.with_name(meta_path.name + ".tmp")
            with open(tmp_path, "w") as f:
                f.write(json.dumps(meta))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, meta_path)

        for contig in scope:
            self._readers[contig].annotate_clusters(
                str(self.path), contig, cutoffs, vaf, vaf_cut
            )

        if contigs is not None:
            for contig in self.contigs:
                if contig not in scope:
                    self._readers[contig].fill_cluster_labels(str(self.path), contig)

        meta_path = self.path / "meta.json"
        meta = json.loads(meta_path.read_text())
        meta["cluster_version"] = CLUSTER_VERSION
        meta["cluster_contigs"] = scope
        meta["cluster_cutoff"] = (
            dict(imd_cutoff) if isinstance(imd_cutoff, Mapping) else float(imd_cutoff)
        )
        meta["cluster_vaf_field"] = vaf_field
        meta["cluster_vaf_cut"] = vaf_cut
        fields = [
            f
            for f in meta.get("fields") or []
            if not (f["name"] == "cluster_class" and f["category"] == "format")
        ]
        fields.append(
            {
                "name": "cluster_class",
                "category": "format",
                "dtype": "u8",
                "default": None,
            }
        )
        meta["fields"] = fields
        tmp_path = meta_path.with_name(meta_path.name + ".tmp")
        with open(tmp_path, "w") as f:
            f.write(json.dumps(meta))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, meta_path)
