from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import numpy as np
import polars as pl
import polars_bio as pb
import polars_config_meta  # noqa: F401
import seqpro as sp
from loguru import logger
from numpy.typing import NDArray
from seqpro.rag import Ragged
from tqdm.auto import tqdm

from .._contigs import ContigNormalizer
from .._mutcat import MUTCAT_VERSION, build_entry_codes, classify_variants, count_matrix
from .._reference import Reference
from .._signatures import _load_signature_file, cosmic_signatures, fit_signatures
from ._io import _open_fmt
from ._kernels import _nb_af_helper


def _empty_annot() -> pl.DataFrame:
    """Return an empty annotation DataFrame with the correct schema."""
    return pl.DataFrame(
        {"varID": [], "gene_id": [], "strand": [], "codon_pos": []},
        schema={
            "varID": pl.UInt32,
            "gene_id": pl.Utf8,
            "strand": pl.Utf8,
            "codon_pos": pl.Int8,
        },
    )


def _get_strand_and_codon_pos(
    cds_df: pl.DataFrame, var_table: pl.DataFrame, contig_normalizer: ContigNormalizer
) -> pl.DataFrame:
    """Calculate strand and codon position for variants overlapping CDS regions.

    Args:
        cds_df (pl.DataFrame): CDS features from GTF with columns: chrom, start, end, strand, frame,
            gene_id, transcript_id, gene_biotype, transcript_support_level, tag
            coordinates should be 1-based
        var_table (pl.DataFrame): Variant table with columns: index, CHROM, POS, ILEN, ...
            POS should be 1-based
        contig_normalizer (ContigNormalizer): Normalizer to match chromosome names between CDS and granges

    Returns:
        pl.DataFrame: Annotation with varID, gene_id, strand, codon_pos
    """
    # Normalize CDS chromosome names to match granges
    # Cast to string first to avoid categorical comparison issues
    cds_df = cds_df.with_columns(
        pl.col("chrom").cast(pl.Utf8).replace(contig_normalizer.contig_map)
    )

    # Filter out CDS features with chromosomes not in granges
    cds_df = cds_df.filter(pl.col("chrom").is_in(contig_normalizer.contigs))
    cds_df.config_meta.set(coordinate_system_zero_based=False)  # type: ignore

    # Prepare var_table for pb.overlap by creating interval columns
    var_intervals = var_table.select(
        pl.col("ILEN").list.first(),
        var_id="index",
        chrom="CHROM",
        start=pl.col("POS"),
        end=pl.col("POS")
        - pl.col("ILEN").list.first().clip(upper_bound=0).fill_null(0),
    )
    var_intervals.config_meta.set(coordinate_system_zero_based=False)  # type: ignore

    # Check if CDS or var_table is empty
    if cds_df.is_empty() or var_table.is_empty():
        return _empty_annot()

    joined_cds = (
        cast(
            pl.LazyFrame,
            pb.overlap(var_intervals, cds_df, projection_pushdown=True),
        )
        .rename(
            {
                "start_1": "pos",
                "start_2": "cds_start",
                "end_2": "cds_end",
            }
        )
        .drop("end_1", "chrom_1", "chrom_2")
        .rename(lambda c: c.replace("_2", "").replace("_1", ""))
        .collect()
    )

    if joined_cds.height == 0:
        return _empty_annot()

    annot = (
        joined_cds
        # Positive strand: (rel_pos - frame) % 3
        # Negative strand: (2 * (rel_pos - frame)) % 3 (reverse complement pattern)
        .with_columns(
            pl.when(
                pl.col("frame").is_not_null()
                & (pl.col("frame") <= 2)
                & (pl.col("ILEN") == 0)
            )
            .then(
                pl.when(pl.col("strand") == "+")
                .then((pl.col("pos") - pl.col("cds_start") - pl.col("frame")) % 3)
                .otherwise(
                    (2 * (pl.col("pos") - pl.col("cds_start") - pl.col("frame"))) % 3
                )
            )
            .cast(pl.Int8)
            .alias("codon_pos")
        )
        # Get the gene_id, strand, and codon_pos.
        # If there are any duplicates, choose the one with the best rank, breaking ties by choosing the first seen.
        .with_columns(
            # Rank 0 is best, higher ranks are worse
            pl.when(pl.col("gene_biotype") == "protein_coding")
            .then(0)
            .otherwise(1)
            .alias("rank_pc"),
            pl.when(
                pl.col("tag").is_not_null()
                & pl.col("tag").str.contains(r"^(canonical|appris_principal)")
            )
            .then(0)
            .otherwise(1)
            .alias("rank_canonical"),
            pl.when(pl.col("transcript_support_level").is_not_null())
            .then(
                pl.col("transcript_support_level")
                .str.extract(r"(\d+)", 1)
                .cast(pl.Int16, strict=False)
            )
            .otherwise(9999)
            .alias("rank_tsl"),
            # Negative span so larger spans get rank 0 (best)
            -(pl.col("cds_end") - pl.col("cds_start")).alias("rank_span"),
        )
        .sort(
            [
                "var_id",
                "rank_pc",
                "rank_canonical",
                "rank_tsl",
                "rank_span",
                "transcript_id",
            ],
            descending=[
                False,
                False,
                False,
                False,
                False,
                False,
            ],  # kept this for code clarity (default also the same)
        )
        .group_by("var_id", maintain_order=True)
        .agg(pl.col("gene_id", "strand", "codon_pos").first())
        # Match the column name used by _empty_annot() and the write-back join.
        .rename({"var_id": "varID"})
    )

    return annot


def _load_gtf(gtf: str | pl.DataFrame) -> pl.DataFrame:
    """Load GTF file as a 1-based polars DataFrame."""
    if isinstance(gtf, pl.DataFrame):
        return gtf.rename({"seqname": "chrom"}, strict=False)

    return (
        sp.gtf.scan(str(gtf))
        .with_columns(
            sp.gtf.attr("gene_id"),
            sp.gtf.attr("transcript_id"),
            sp.gtf.attr("gene_name"),
            sp.gtf.attr("gene_biotype"),
            sp.gtf.attr("transcript_support_level"),
            sp.gtf.attr("level"),
            sp.gtf.attr("tag"),
        )
        .collect()
        .rename({"seqname": "chrom"}, strict=False)
    )


class SparseVarAnnotateMixin:
    def annotate_with_gtf(
        self,
        gtf: str | pl.DataFrame,
        level_filter: int | None = 1,
        write_back: bool = True,
        *,
        strand_encoding: dict[str | None, int] | None = None,
        codon_null_token: int | None = None,
    ) -> pl.DataFrame:
        """Annotate variants with gene_id, strand, and codon_pos from GTF CDS features.

        Computes codon position for SNVs only; indels receive strand but null codon_pos.

        Args:
            gtf (str or pl.DataFrame): Path to GTF file (.gtf or .gtf.gz) or pre-loaded Polars DataFrame.
            level_filter (int or None, default 1): If set, keep rows with GTF 'level' <= level_filter (1 = highest quality).
            write_back (bool, default True): If True, update self.index in-place and write to index.arrow file.
            strand_encoding (dict or None, optional): Encode strand as integers. Example: {'+': 0, '-': 1, None: 2}
            codon_null_token (int or None, optional): Replace null codon_pos with this integer for ML models.

        Returns:
            pl.DataFrame: Columns: varID (UInt32), gene_id (Utf8), strand (Utf8/Int16), codon_pos (Int8/Int16)

        Examples:
            >>> svar = SparseVar("data.svar")
            >>> annot = svar.annotate_with_gtf("gencode.v45.gtf.gz")
            >>> annot.head()
        """
        # Validate inputs
        if level_filter is not None and not isinstance(level_filter, int):
            raise TypeError(
                f"level_filter must be int or None, got {type(level_filter)}"
            )
        if strand_encoding is not None and not isinstance(strand_encoding, dict):
            raise TypeError("strand_encoding must be dict or None")
        if codon_null_token is not None and not isinstance(codon_null_token, int):
            raise TypeError("codon_null_token must be int or None")

        logger.info("Loading GTF for CDS annotation")

        with tqdm(total=3, desc="GTF annotation", unit="step") as pbar:
            # Load GTF
            pbar.set_description("Loading GTF")
            gtf_df = _load_gtf(gtf)
            if level_filter is not None and "level" in gtf_df.columns:
                gtf_df = gtf_df.filter(pl.col("level").cast(pl.Int32) <= level_filter)
            pbar.update(1)

            # CDS Annotation
            pbar.set_description("CDS annotation")

            # Extract CDS features with gene_biotype
            cds_df = gtf_df.filter(pl.col("feature") == "CDS").select(
                "chrom",
                "start",
                "end",
                "strand",
                "frame",
                "gene_id",
                "transcript_id",
                "gene_biotype",
                "transcript_support_level",
                "tag",
            )

            if len(cds_df) == 0:
                annot = _empty_annot()
            else:
                # pyrefly: ignore [missing-attribute]
                annot = _get_strand_and_codon_pos(cds_df, self.index, self._c_norm)
            pbar.update()

            # Apply encoding
            pbar.set_description("Finalizing")
            if strand_encoding is not None:
                str_map = {k: v for k, v in strand_encoding.items() if k is not None}
                null_val = strand_encoding.get(None)
                strand_expr = pl.col("strand").replace_strict(str_map, default=null_val)
                annot = annot.with_columns(strand_expr.cast(pl.Int16).alias("strand"))

            if codon_null_token is not None:
                annot = annot.with_columns(
                    pl.col("codon_pos").fill_null(codon_null_token).cast(pl.Int16)
                )

            # Write back if requested
            if write_back:
                self._load_all_attrs()
                self.index = (
                    self.index.lazy()
                    .with_row_index("varID")
                    # maintain_order="left" is required: index rows are
                    # positionally aligned with the sparse genotype storage, so
                    # the join must not reorder them. The default ('none') lets
                    # polars' hash join reorder output on larger data.
                    .join(annot.lazy(), on="varID", how="left", maintain_order="left")
                    .drop("varID")
                    .collect()
                )
                # pyrefly: ignore [missing-attribute]
                df = self._to_df()
                # pyrefly: ignore [missing-attribute]
                df.write_ipc(self._index_path(self.path))
                logger.info("Wrote gene_id, strand, codon_pos to index.arrow")

            pbar.update(1)

        return annot

    def annotate_mutations(
        self,
        reference: "Reference | str | Path",
        *,
        contigs: "list[str] | None" = None,
        write_back: bool = True,
    ) -> None:
        """Classify every variant into SBS-96 / DBS-78 / ID-83 channels and store a per-genotype-entry ``mutcat`` field (int16, enum-encoded).

        Adjacent SNVs carried on the same haplotype are combined into DBS; the
        5' entry receives the DBS code and the 3' entry a ``DBS_PARTNER`` sentinel.

        Args:
            reference: Reference genome.  A :class:`~genoray._reference.Reference` instance,
                or a path to a FASTA file (with a ``.fai`` index alongside it).
            contigs: If given, only variants on these contigs are classified; entries on
                all other contigs are marked ``NOT_ANNOTATED`` and their contigs are
                never fetched from the reference.  Names are matched via the
                :class:`~genoray._contigs.ContigNormalizer` (so ``chr1``/``1`` both
                work).  Requested contigs absent from the ``.svar`` index are skipped
                with a warning.  A listed contig present in the index but absent from
                the reference still raises (use the allowlist to exclude it instead).
                ``None`` (default) classifies all contigs.
            write_back: If ``True`` (default), persist ``mutcat.npy`` and update
                ``metadata.json`` on disk so that subsequent ``SparseVar(...)``
                opens will see the field.  If ``False``, the ``mutcat`` field lives
                only in memory (``self.fields["mutcat"]``) and is NOT written to
                disk — reopening the file will not find it.  Note: the
                ``metadata.json`` update is not safe against concurrent writers;
                single-writer access is expected (consistent with
                ``annotate_with_gtf``).
        """
        if not isinstance(reference, Reference):
            reference = Reference.from_path(reference)

        # 0. resolve contig scope
        index_chroms = self.index["CHROM"].to_list()
        if contigs is None:
            scoped_contigs: "list[str] | None" = None
            in_scope = np.ones(self.index.height, dtype=np.bool_)
        else:
            # pyrefly: ignore [missing-attribute]
            normalized = self._c_norm.norm(list(contigs))
            unmatched = [c for c, nm in zip(contigs, normalized) if nm is None]
            if unmatched:
                logger.warning(
                    f"annotate_mutations: {len(unmatched)} requested contig(s) not "
                    f"found in the .svar index; they will be skipped: {unmatched}"
                )
            scope_set = {nm for nm in normalized if nm is not None}
            scoped_contigs = sorted(scope_set)
            in_scope = np.array([c in scope_set for c in index_chroms], dtype=np.bool_)

        # 1. intrinsic per-variant codes (scoped)
        var_code = classify_variants(self.index, reference, contigs=scoped_contigs)

        # 2. per-variant arrays needed by the adjacency kernel
        pos = self.index["POS"].to_numpy().astype(np.int64)
        ref0 = self.index["REF"].to_list()
        alt0 = self.index["ALT"].list.first().to_list()
        is_snv = np.array(
            [
                r is not None and a is not None and len(r) == 1 and len(a) == 1
                for r, a in zip(ref0, alt0)
            ],
            dtype=np.bool_,
        )
        # gate adjacency: out-of-scope variants must not be collapsed into DBS
        is_snv &= in_scope
        # contig id per variant — equality semantics only (same contig ↔ same id)
        # pyrefly: ignore [missing-attribute]
        contig_map = {c: i for i, c in enumerate(self.contigs)}
        contig_codes = np.array(
            [contig_map.get(c, -1) for c in self.index["CHROM"].to_list()],
            dtype=np.int32,
        )
        ref_b = np.array([ord(r[0]) if r else 0 for r in ref0], dtype=np.uint8)
        alt_b = np.array([ord(a[0]) if a else 0 for a in alt0], dtype=np.uint8)

        # 3. broadcast to entries + DBS adjacency override
        entry_codes = build_entry_codes(
            # pyrefly: ignore [missing-attribute]
            self.genos.data,
            # pyrefly: ignore [missing-attribute]
            self.genos.offsets,
            var_code,
            pos,
            contig_codes,
            is_snv,
            ref_b,
            alt_b,
        )

        # 4. register in-memory (mirrors how fields are opened in __init__)
        # pyrefly: ignore [missing-attribute]
        shape = (self.n_samples, self.ploidy, None)
        # pyrefly: ignore [missing-attribute]
        self.available_fields["mutcat"] = np.dtype("int16")
        # pyrefly: ignore [missing-attribute]
        self.fields["mutcat"] = Ragged.from_offsets(
            entry_codes,
            shape,
            # pyrefly: ignore [missing-attribute]
            self.genos.offsets,
        )

        # 5. optionally persist
        if write_back:
            mm = np.memmap(
                # pyrefly: ignore [missing-attribute]
                self.path / "mutcat.npy",
                dtype=np.int16,
                mode="w+",
                shape=entry_codes.shape,
            )
            mm[:] = entry_codes
            mm.flush()
            del mm

            # Local import: SparseVarMetadata lives in _core, which imports this
            # mixin at module scope; importing it lazily here avoids a cycle.
            from ._core import SparseVarMetadata

            # pyrefly: ignore [missing-attribute]
            with open(self.path / "metadata.json", "rb") as f:
                meta = SparseVarMetadata.model_validate_json(f.read())
            meta.fields["mutcat"] = "int16"
            meta.mutcat_version = MUTCAT_VERSION
            meta.mutcat_contigs = scoped_contigs
            # pyrefly: ignore [missing-attribute]
            with open(self.path / "metadata.json", "w") as f:
                f.write(meta.model_dump_json())

    def mutation_matrix(
        self,
        kind: Literal["SBS96", "DBS78", "ID83"],
        *,
        count: Literal["allele", "sample"] = "allele",
    ) -> pl.DataFrame:
        """Build a per-sample mutation count matrix.

        Requires :meth:`annotate_mutations` to have been run (or the ``mutcat``
        field to be loaded). Returns a DataFrame with a ``MutationType`` column
        plus one column per sample (rows in fixed codebook order).

        The ``mutcat`` field is resolved in the following priority order:

        1. Already loaded in ``self.fields["mutcat"]`` (e.g. opened with
           ``fields=["mutcat"]``).
        2. Present on disk as ``mutcat.npy`` (written by a prior
           :meth:`annotate_mutations` call with ``write_back=True``) — opened
           lazily and cached into ``self.fields["mutcat"]`` for subsequent
           calls.
        3. Not found at all — raises :class:`ValueError`.

        Args:
            kind: One of ``"SBS96"``, ``"DBS78"``, ``"ID83"``.
            count: ``"allele"`` counts every non-ref allele copy; ``"sample"`` counts
                each category at most once per sample (presence/absence).
        """
        if kind not in ("SBS96", "DBS78", "ID83"):
            raise ValueError(f"Unknown matrix kind {kind!r}.")
        if count not in ("allele", "sample"):
            raise ValueError(
                f"Unknown count mode {count!r}; choose 'allele' or 'sample'."
            )
        # pyrefly: ignore [missing-attribute]
        mut = self.fields.get("mutcat")
        if mut is None:
            # pyrefly: ignore [missing-attribute]
            if "mutcat" in self.available_fields:
                # pyrefly: ignore [missing-attribute]
                shape = (self.n_samples, self.ploidy, None)
                mut = _open_fmt(
                    "mutcat",
                    # pyrefly: ignore [bad-index]
                    self.available_fields["mutcat"],
                    # pyrefly: ignore [missing-attribute]
                    self.path,
                    shape,
                    "r",
                )
                # pyrefly: ignore [missing-attribute]
                self.fields["mutcat"] = mut
            else:
                raise ValueError(
                    "No 'mutcat' field found. Run annotate_mutations() first "
                    "(or open with fields=['mutcat'])."
                )
        return count_matrix(
            np.asarray(mut.data),
            # pyrefly: ignore [missing-attribute]
            np.asarray(self.genos.offsets),
            # pyrefly: ignore [missing-attribute]
            self.ploidy,
            # pyrefly: ignore [missing-attribute]
            self.n_samples,
            # pyrefly: ignore [missing-attribute]
            self.available_samples,
            kind,
            per_sample=(count == "sample"),
        )

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
    ) -> "pl.DataFrame":
        """Refit this object's mutation catalogue against COSMIC signatures.

        Builds the ``kind`` catalogue via :meth:`mutation_matrix` and decomposes
        it into per-sample activities via :func:`genoray.fit_signatures`.

        Args:
            kind: One of ``"SBS96"``, ``"DBS78"``, ``"ID83"``.
            reference: Reference signatures as a Polars ``DataFrame`` (``MutationType`` +
                signature columns), a path to a COSMIC-style TSV, or ``None`` to fetch
                the default COSMIC set via :func:`genoray.cosmic_signatures`.
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

    def cache_afs(self):
        """Cache the allele frequencies on disk. Will also load all possible attributes and add the AF column in-memory."""
        self._load_all_attrs()
        afs = self._compute_afs()
        self.index = self.index.with_columns(AF=pl.Series(afs))
        self._write_afs()

    def _load_all_attrs(self):
        # pyrefly: ignore [missing-attribute]
        idx_df = pl.scan_ipc(self._index_path(self.path))
        schema = idx_df.collect_schema()
        missing = set(schema) - set(self.index.columns)
        missing_attrs = idx_df.select(*missing).collect()
        self.index = self.index.hstack(missing_attrs)

    def _compute_afs(self) -> NDArray[np.float32]:
        # pyrefly: ignore [missing-attribute]
        n_samples, ploidy, _ = cast(tuple[int, int, None], self.genos.shape)
        max_count = n_samples * ploidy
        # pyrefly: ignore [missing-attribute]
        afs = np.zeros(self.n_variants, np.float32)
        # pyrefly: ignore [missing-attribute]
        _nb_af_helper(afs, self.genos.data, self.genos.offsets, max_count)
        return afs

    def _write_afs(self):
        # pyrefly: ignore [missing-attribute]
        df = self._to_df()
        # pyrefly: ignore [missing-attribute]
        df.write_ipc(self._index_path(self.path))
