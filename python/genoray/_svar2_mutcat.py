"""SVAR2 mutational-signature surface: annotate/count/assign on ``SparseVar2``.

Mirrors v1's ``SparseVarAnnotateMixin.annotate_mutations``/``mutation_matrix``/
``assign_signatures`` (``genoray._svar._annotate``), but delegates the actual
classification and counting to the Rust ``PyContigReader`` (per-contig
``annotate_mutations``/``count_matrix`` bindings) instead of doing it in Python
over an in-memory index.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl

from genoray._contigs import ContigNormalizer
from genoray._mutcat import MUTCAT_VERSION, N_CODES, code_ranges, labels
from genoray._mutcat.strand import contig_strand_intervals, load_gene_intervals
from genoray._reference import Reference
from genoray._signatures import _load_signature_file, cosmic_signatures, fit_signatures


class _MutcatMixin:
    """Mutation-catalogue annotation, counting, and signature-assignment methods.

    Provided by the concrete ``SparseVar2`` host class (see ``SparseVar2.__init__``);
    declared here so the mixin's use of them type-checks in isolation. (``n_samples``
    is an ``@property`` on the host, so it can't be redeclared as a plain attribute
    here without a bad-override; it's accessed via an ignore below.)
    """

    path: Path
    contigs: list[str]
    available_samples: list[str]
    _readers: dict[str, Any]

    def annotate_mutations(
        self,
        reference: "Reference | str | Path",
        *,
        gtf: "str | Path | pl.DataFrame | None" = None,
        contigs: "list[str] | None" = None,
    ) -> None:
        """Classify every variant on the in-scope contigs into SBS-96/DBS-78/ID-83 codes and write the per-contig ``mutcat`` sidecar (post-hoc annotation).

        Args:
            reference: Reference genome. A :class:`~genoray._reference.Reference` instance,
                or a path to a FASTA file (with a ``.fai`` index alongside it).
            gtf: Optional gene model (path to a GTF/GTF.gz, or a pre-loaded GTF
                DataFrame). When given, each SNV additionally gets a transcriptional
                strand class stored in a 2-bit ``strand.bin`` sidecar, enabling the
                strand-resolved ``"SBS192"``/``"SBS384"`` matrices. Gene footprints
                are taken from ``feature == "gene"`` rows (full gene body); filter
                the GTF beforehand to restrict biotypes.
            contigs: If given, only these contigs (intersected with ``self.contigs``) are
                annotated. ``None`` (default) annotates every contig in the store.

        Notes:
            Stamps ``meta.json`` with ``mutcat_version``, ``mutcat_contigs``, and
            ``mutcat_strand`` (whether a GTF was supplied). The sidecar files
            themselves are the ground truth checked by :meth:`mutation_matrix`'s
            guards.
        """
        if not isinstance(reference, Reference):
            reference = Reference.from_path(reference)
        scope = (
            self.contigs
            if contigs is None
            else [c for c in self.contigs if c in set(contigs)]
        )

        genes = None
        c_norm = None
        if gtf is not None:
            genes = load_gene_intervals(str(gtf) if isinstance(gtf, Path) else gtf)
            c_norm = ContigNormalizer(self.contigs)

        for contig in scope:
            seq = reference.contig_array(contig).astype(np.uint8, copy=False)
            if genes is not None and c_norm is not None:
                starts, stops, values = contig_strand_intervals(genes, contig, c_norm)
                self._readers[contig].annotate_mutations(
                    str(self.path), contig, seq, starts, stops, values
                )
            else:
                self._readers[contig].annotate_mutations(str(self.path), contig, seq)

        meta_path = self.path / "meta.json"
        meta = json.loads(meta_path.read_text())
        meta["mutcat_version"] = MUTCAT_VERSION
        meta["mutcat_contigs"] = scope
        meta["mutcat_strand"] = gtf is not None
        meta_path.write_text(json.dumps(meta))

    def _is_annotated(self) -> bool:
        """Whether every contig has an on-disk mutcat sidecar.

        Checked directly on disk (not via ``meta.json``) because
        ``from_vcf(..., signatures=True)`` writes sidecars at conversion time
        without stamping ``meta.json``; the ``var_key_snp/code.bin`` file is
        written by ``annotate_contig`` for every contig (even if empty), so its
        absence is ground truth for "never annotated".
        """
        return all(
            (self.path / contig / "mutcat" / "var_key_snp" / "code.bin").exists()
            for contig in self.contigs
        )

    def _is_strand_annotated(self) -> bool:
        """Whether every contig has an on-disk ``strand.bin`` (GTF-annotated).

        Ground truth for whether SBS192/SBS384 can be produced. Like
        ``_is_annotated``, checked on disk rather than via ``meta.json`` because
        the file is what the Rust count reads.
        """
        return all(
            (self.path / contig / "mutcat" / "var_key_snp" / "strand.bin").exists()
            for contig in self.contigs
        )

    def mutation_matrix(
        self,
        kind: Literal["SBS96", "DBS78", "ID83", "SBS192", "SBS384"],
        *,
        count: Literal["allele", "sample"] = "allele",
    ) -> pl.DataFrame:
        """Build a per-sample mutation count matrix.

        Requires :meth:`annotate_mutations` to have been run (or the store to
        have been built with ``from_vcf(..., signatures=True)``). Returns a
        DataFrame with a ``MutationType`` column plus one column per sample
        (rows in fixed codebook order).

        Args:
            kind: One of ``"SBS96"``, ``"DBS78"``, ``"ID83"``, ``"SBS192"``,
                ``"SBS384"``. The last two require :meth:`annotate_mutations` to
                have been run with ``gtf=`` (transcriptional strand annotation).
            count: ``"allele"`` counts every non-ref allele copy; ``"sample"`` counts
                each category at most once per sample (presence/absence), OR-combined
                across contigs.

        Raises:
            ValueError: If the store has not been annotated (no on-disk mutcat sidecar for
                every contig) — calling the underlying Rust ``count_matrix`` on an
                unannotated contig with SNP records panics across the FFI boundary,
                so this is checked up front to raise a clean Python exception instead.
                Also raised for ``"SBS192"``/``"SBS384"`` if the store has not been
                strand-annotated (no on-disk ``strand.bin`` for every contig).
        """
        if kind not in ("SBS96", "DBS78", "ID83", "SBS192", "SBS384"):
            raise ValueError(f"Unknown matrix kind {kind!r}.")
        if count not in ("allele", "sample"):
            raise ValueError(
                f"Unknown count mode {count!r}; choose 'allele' or 'sample'."
            )
        if not self._is_annotated():
            raise ValueError(
                "SparseVar2 is not annotated for mutational signatures; call "
                "annotate_mutations(reference) first, or rebuild with "
                "from_vcf(..., signatures=True)."
            )
        if kind in ("SBS192", "SBS384") and not self._is_strand_annotated():
            raise ValueError(
                f"{kind} requires transcriptional strand annotation; re-run "
                "annotate_mutations(reference, gtf=...) with a gene model."
            )

        per_sample = count == "sample"
        total = np.zeros((self.n_samples, N_CODES), dtype=np.int64)  # type: ignore[missing-attribute]
        for contig in self.contigs:
            total += self._readers[contig].count_matrix(
                str(self.path), contig, per_sample
            )
        if per_sample:
            np.clip(total, 0, 1, out=total)  # presence OR across contigs

        lo, hi = code_ranges()[kind]
        block = total[:, lo:hi]
        out: dict[str, object] = {"MutationType": labels(kind)}
        for si, name in enumerate(self.available_samples):
            out[name] = block[si]
        return pl.DataFrame(out)

    def assign_signatures(
        self,
        kind: Literal["SBS96", "DBS78", "ID83"],
        *,
        reference: "pl.DataFrame | str | Path | None" = None,
        count: Literal["allele", "sample"] = "allele",
        max_delta: float = 0.01,
        min_activity: float = 0.005,
        n_jobs: int = 1,
        backend: str = "loky",
    ) -> pl.DataFrame:
        """Refit this store's mutation catalogue against reference signatures.

        Builds the ``kind`` catalogue via :meth:`mutation_matrix` and decomposes
        it into per-sample activities via :func:`genoray.fit_signatures`.

        Args:
            kind: One of ``"SBS96"``, ``"DBS78"``, ``"ID83"``.
            reference: Reference signatures as a Polars ``DataFrame`` (``MutationType`` +
                signature columns), a path to a COSMIC-style TSV, or ``None`` to
                fetch the default COSMIC set via :func:`genoray.cosmic_signatures`.
            count: Counting unit passed to :meth:`mutation_matrix`.
            max_delta: Forwarded to :func:`genoray.fit_signatures`.
            min_activity: Forwarded to :func:`genoray.fit_signatures`.
            n_jobs: Forwarded to :func:`genoray.fit_signatures` to control per-sample
                parallelism (``1`` (default) runs serially; ``-1`` uses all cores;
                process-based ``"loky"`` backend).
            backend: Forwarded to :func:`genoray.fit_signatures` to control per-sample
                parallelism (``1`` (default) runs serially; ``-1`` uses all cores;
                process-based ``"loky"`` backend).

        Returns:
            pl.DataFrame: One row per sample: ``Sample``, one column per signature, and
            ``cosine_similarity``.
        """
        if kind in ("SBS192", "SBS384"):
            raise NotImplementedError(
                f"{kind} is a transcriptional-strand-bias catalog with no COSMIC "
                "reference signature set for refitting. Use mutation_matrix(kind) "
                "for strand-bias analysis instead."
            )
        catalogue = self.mutation_matrix(kind, count=count)
        if reference is None:
            ref = cosmic_signatures(kind)
        elif isinstance(reference, pl.DataFrame):
            ref = reference
        else:
            ref = _load_signature_file(reference)
        return fit_signatures(
            catalogue,
            ref,
            max_delta=max_delta,
            min_activity=min_activity,
            n_jobs=n_jobs,
            backend=backend,
        )
