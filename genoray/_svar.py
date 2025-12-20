from __future__ import annotations

import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Generator, Iterable, Literal, Union, cast, overload

import awkward as ak
import numba as nb
import numpy as np
import polars as pl
import polars_bio as pb
import seqpro as sp
from awkward.contents import Content, NumpyArray
from hirola import HashTable
from loguru import logger
from natsort import natsorted
from numpy.typing import ArrayLike, NDArray
from polars._typing import IntoExpr
from pydantic import BaseModel
from seqpro.rag import OFFSET_TYPE, Ragged, lengths_to_offsets
from tqdm.auto import tqdm

from ._pgen import PGEN
from ._types import DOSAGE_TYPE, DTYPE, INT64_MAX, POS_TYPE, V_IDX_TYPE
from ._utils import ContigNormalizer
from ._var_ranges import var_ranges
from ._vcf import VCF

SparseGenotypes = Ragged[V_IDX_TYPE]
SparseDosages = Ragged[DOSAGE_TYPE]


@overload
def dense2sparse(
    genos: NDArray[np.int8],
    var_idxs: NDArray[V_IDX_TYPE],
    dosages: None = None,
) -> SparseGenotypes: ...
@overload
def dense2sparse(
    genos: NDArray[np.int8],
    var_idxs: NDArray[V_IDX_TYPE],
    dosages: NDArray[DOSAGE_TYPE],
) -> tuple[SparseGenotypes, SparseDosages]: ...
def dense2sparse(
    genos: NDArray[np.int8],
    var_idxs: NDArray[V_IDX_TYPE],
    dosages: NDArray[DOSAGE_TYPE] | None = None,
) -> SparseGenotypes | tuple[SparseGenotypes, SparseDosages]:
    """Convert dense genotypes (and dosages) to sparse genotypes."""
    # (s p v)
    if genos.ndim < 3:
        raise ValueError(
            "Sparse genotypes must have at least 3 dimensions, with the final three dimensions corresponding"
            " to (samples, ploidy, variants)"
        )
    if dosages is not None:
        if dosages.ndim < 2:
            raise ValueError(
                "Sparse dosages must have at least 2 dimensions, with the final two dimensions corresponding"
                " to (samples, variants)"
            )
        if dosages.shape[-1] != genos.shape[-1]:
            raise ValueError(
                "Sparse dosages must have the same number of variants as the genotypes"
            )
        if dosages.shape[-2] != genos.shape[-3]:
            raise ValueError(
                "Sparse dosages must have the same number of samples as the genotypes"
            )

    keep = genos == 1
    data = var_idxs[keep.nonzero()[-1]]
    lengths = keep.sum(-1)
    shape = (*lengths.shape, None)
    offsets = lengths_to_offsets(lengths)
    rag = SparseGenotypes.from_offsets(data, shape, offsets)

    if dosages is not None:
        # (s v) -> (s p v)
        dosage_data = np.broadcast_to(dosages[:, None], genos.shape)[keep]
        _dosages = SparseDosages.from_offsets(dosage_data, shape, offsets)
        return rag, _dosages
    return rag


CURRENT_VERSION = 1


class SparseVarMetadata(BaseModel):
    version: Union[int, None] = None
    samples: list[str]
    ploidy: int
    contigs: list[str]


class SparseVar:
    """Open a Sparse Variant (SVAR) directory.

    Parameters
    ----------
    path
        Path to the SVAR directory.
    attrs
        Expression of attributes to load in addition to the ALT and ILEN columns.
    """

    path: Path
    version: int | None
    available_samples: list[str]
    ploidy: int
    contigs: list[str]
    """Contigs in the order they appear in the dataset. Variants are only sorted within each contig."""
    genos: SparseGenotypes
    dosages: SparseDosages | None
    var_table: pl.DataFrame
    """Table of variants with columns: `CHROM`, `POS`, `REF`, `ALT`, `ILEN`, and any additional
    attributes specified in `attrs` on construction."""
    _c_norm: ContigNormalizer
    _s2i: HashTable
    _c_max_idxs: dict[str, int]
    _is_biallelic: bool

    @property
    def n_samples(self) -> int:
        """Number of samples in the dataset."""
        return len(self.available_samples)

    @property
    def n_variants(self) -> int:
        """Number of variants in the dataset."""
        return self.var_table.height

    @property
    def has_dosages(self) -> bool:
        return (self.path / "dosages.npy").exists()

    def __init__(self, path: str | Path, attrs: IntoExpr | None = None):
        path = Path(path)
        self.path = path
        if not self.path.exists():
            raise FileNotFoundError(f"SVAR directory {self.path} does not exist.")

        with open(path / "metadata.json", "rb") as f:
            metadata = SparseVarMetadata.model_validate_json(f.read())
        contigs = metadata.contigs
        self.version = metadata.version
        self.contigs = contigs
        self.available_samples = metadata.samples
        self.ploidy = metadata.ploidy
        samples = np.array(self.available_samples)
        self._s2i = HashTable(
            len(samples) * 2,  # type: ignore
            dtype=samples.dtype,
        )
        self._s2i.add(samples)

        self._c_norm = ContigNormalizer(contigs)
        self.genos = _open_genos(path, (self.n_samples, self.ploidy, None), "r")
        if (path / "dosages.npy").exists():
            dosage_data = np.memmap(path / "dosages.npy", dtype=DOSAGE_TYPE, mode="r")
            self.dosages = SparseDosages.from_offsets(
                dosage_data, self.genos.shape, self.genos.offsets
            )
        else:
            self.dosages = None
        logger.info("Loading genoray index")
        self.var_table = self._load_var_table(attrs)
        self._is_biallelic = (self.var_table["ALT"].list.len() == 1).all()
        vars_per_contig = self.var_table.group_by("CHROM", maintain_order=True).agg(
            n_variants=pl.len()
        )
        self._c_max_idxs = {
            c: v - 1
            for c, v in zip(
                vars_per_contig["CHROM"], vars_per_contig["n_variants"].cum_sum()
            )
        }
        self._c_max_idxs |= {c: 0 for c in self.contigs if c not in self._c_max_idxs}

    def var_ranges(
        self,
        contig: str,
        starts: ArrayLike = 0,
        ends: ArrayLike = INT64_MAX,
    ) -> NDArray[V_IDX_TYPE]:
        """Get variant index ranges for each query range. i.e.
        For each query range, return the minimum and maximum variant that overlaps.
        Note that this means some variants within those ranges may not actually overlap with
        the query range if there is a deletion that spans the start of the query.

        Parameters
        ----------
        contig
            Contig name.
        starts
            0-based start positions of the ranges.
        ends
            0-based, exclusive end positions of the ranges.

        Returns
        -------
            Shape: :code:`(ranges, 2)`. The first column is the start index of the variant
            and the second column is the end index of the variant.
        """
        return var_ranges(self._c_norm, self.var_table, contig, starts, ends)

    def _find_starts_ends(
        self,
        contig: str,
        starts: ArrayLike = 0,
        ends: ArrayLike = INT64_MAX,
        samples: ArrayLike | None = None,
    ) -> NDArray[OFFSET_TYPE]:
        """Find the start and end offsets of the sparse genotypes for each range.

        Parameters
        ----------
        contig
            Contig name.
        starts
            0-based start positions of the ranges.
        ends
            0-based, exclusive end positions of the ranges.
        samples
            List of sample names to read. If None, read all samples.

        Returns
        -------
            Shape: (ranges, samples, ploidy, 2). The first column is the start index of the variant
            and the second column is the end index of the variant.
        """
        if samples is None:
            samples = np.atleast_1d(np.array(self.available_samples))
        else:
            samples = np.atleast_1d(np.array(samples))
            if missing := set(samples) - set(self.available_samples):  # type: ignore
                raise ValueError(f"Samples {missing} not found in the dataset.")

        s_idxs = cast(NDArray[np.int64], self._s2i[samples])

        starts = np.atleast_1d(np.asarray(starts, POS_TYPE))
        n_ranges = len(starts)

        c = self._c_norm.norm(contig)
        if c is None:
            return np.full((n_ranges, len(samples), self.ploidy, 2), -1, OFFSET_TYPE)

        ends = np.atleast_1d(np.asarray(ends, POS_TYPE))
        # (r 2)
        var_ranges = self.var_ranges(contig, starts, ends)
        # (2 r s p)
        starts_ends = _find_starts_ends(
            self.genos.data, self.genos.offsets, var_ranges, s_idxs, self.ploidy
        )
        return starts_ends

    def _find_starts_ends_with_length(
        self,
        contig: str,
        starts: ArrayLike = 0,
        ends: ArrayLike = INT64_MAX,
        samples: ArrayLike | None = None,
        out: NDArray[OFFSET_TYPE] | None = None,
    ) -> NDArray[OFFSET_TYPE]:
        """Find the start and end offsets of the sparse genotypes for each range.

        Parameters
        ----------
        contig
            Contig name.
        starts
            0-based start positions of the ranges.
        ends
            0-based, exclusive end positions of the ranges.
        samples
            List of sample names to read. If None, read all samples.
        out
            Output array to write to. If None, a new array will be created.

        Returns
        -------
            Shape: (2, ranges, samples, ploidy). The first column is the start index of the variant
            and the second column is the end index of the variant.
        """
        if not self._is_biallelic:
            raise ValueError(
                "Cannot use with_length operations with multiallelic variants."
            )

        if samples is None:
            samples = np.atleast_1d(np.array(self.available_samples))
        else:
            samples = np.atleast_1d(np.array(samples))
            if missing := set(samples) - set(self.available_samples):  # type: ignore
                raise ValueError(f"Samples {missing} not found in the dataset.")

        s_idxs = cast(NDArray[np.int64], self._s2i[samples])

        starts = np.atleast_1d(np.asarray(starts, POS_TYPE))
        n_ranges = len(starts)

        c = self._c_norm.norm(contig)
        if c is None:
            return np.full((n_ranges, len(samples), self.ploidy, 2), -1, OFFSET_TYPE)

        ends = np.atleast_1d(np.asarray(ends, POS_TYPE))
        # (r 2)
        var_ranges = self.var_ranges(contig, starts, ends)

        v_starts = (self.var_table["POS"] - 1).to_numpy()

        # (2 r s p)
        out = _find_starts_ends_with_length(
            self.genos.data,
            self.genos.offsets,
            starts,
            ends,
            var_ranges,
            v_starts,
            self.var_table["ILEN"].list.first().to_numpy(),
            s_idxs,
            self.ploidy,
            self._c_max_idxs[c],
            out,
        )
        return out

    def read_ranges(
        self,
        contig: str,
        starts: ArrayLike = 0,
        ends: ArrayLike = INT64_MAX,
        samples: ArrayLike | None = None,
    ) -> SparseGenotypes:
        """Read the genotypes for the given ranges.

        Parameters
        ----------
        contig
            Contig name.
        starts
            0-based start positions of the ranges.
        ends
            0-based, exclusive end positions of the ranges.
        samples
            List of sample names to read. If None, read all samples.

        Returns
        -------
            SparseGenotypes with shape :code:`(ranges, samples, ploidy, ~variants)`. Note that the genotypes will be backed by
            a memory mapped read-only array of the full file so the only data in memory will be the offsets.
        """
        if samples is None:
            samples = np.atleast_1d(np.array(self.available_samples))
        else:
            samples = np.atleast_1d(np.array(samples))
            if missing := set(samples) - set(self.available_samples):  # type: ignore
                raise ValueError(f"Samples {missing} not found in the dataset.")

        n_samples = len(samples)
        starts = np.atleast_1d(np.asarray(starts, POS_TYPE))
        n_ranges = len(starts)

        # (2 r s p)
        starts_ends = self._find_starts_ends(contig, starts, ends, samples)
        return SparseGenotypes.from_offsets(
            self.genos.data,
            (n_ranges, n_samples, self.ploidy, None),
            starts_ends.reshape(2, -1),
        )

    def read_ranges_with_length(
        self,
        contig: str,
        starts: ArrayLike = 0,
        ends: ArrayLike = INT64_MAX,
        samples: ArrayLike | None = None,
    ) -> SparseGenotypes:
        """Read the genotypes for the given ranges such that each entry of variants is guaranteed to have
        the minimum amount of variants to reach the query length. This can mean either fewer or more variants
        than would be returned than by :code:`read_ranges`, depending on the presence of indels.

        Parameters
        ----------
        contig
            Contig name.
        starts
            0-based start positions of the ranges.
        ends
            0-based, exclusive end positions of the ranges.
        samples
            List of sample names to read. If None, read all samples.

        Returns
        -------
            SparseGenotypes with shape :code:`(ranges, samples, ploidy, ~variants)`. Note that the genotypes will be backed by
            a memory mapped read-only array of the full file so the only data in memory will be the offsets.
        """
        if samples is None:
            samples = np.atleast_1d(np.array(self.available_samples))
        else:
            if missing := set(samples) - set(self.available_samples):  # type: ignore
                raise ValueError(f"Samples {missing} not found in the dataset.")
            samples = np.atleast_1d(np.array(samples))

        n_samples = len(samples)
        starts = np.atleast_1d(np.asarray(starts, POS_TYPE))
        n_ranges = len(starts)

        # (2 r s p)
        starts_ends = self._find_starts_ends_with_length(contig, starts, ends, samples)
        return SparseGenotypes.from_offsets(
            self.genos.data,
            (n_ranges, n_samples, self.ploidy, None),
            starts_ends.reshape(2, -1),
        )

    @classmethod
    def from_vcf(
        cls,
        out: str | Path,
        vcf: VCF,
        max_mem: int | str,
        overwrite: bool = False,
        with_dosages: bool = False,
    ):
        """Create a Sparse Variant (.svar) from a VCF/BCF.

        Parameters
        ----------
        out
            Path to the output directory.
        vcf
            VCF file to write from.
        max_mem
            Maximum memory to use while writing.
        overwrite
            Whether to overwrite the output directory if it exists.
        with_dosages
            Whether to write dosages.
        """
        out = Path(out)

        if out.exists() and not overwrite:
            raise FileExistsError(
                f"Output path {out} already exists. Use overwrite=True to overwrite."
            )
        out.mkdir(parents=True, exist_ok=True)

        if not vcf._index_path().exists():
            logger.info("Genoray VCF index not found, creating index.")
            vcf._write_gvi_index()

        contigs = vcf.contigs
        with open(out / "metadata.json", "w") as f:
            json = SparseVarMetadata(
                version=CURRENT_VERSION,
                contigs=contigs,
                samples=vcf.available_samples,
                ploidy=vcf.ploidy,
            ).model_dump_json()
            f.write(json)

        shutil.copy(vcf._index_path(), cls._index_path(out))

        with TemporaryDirectory() as tdir:
            tdir = Path(tdir)

            shape = (vcf.n_samples, vcf.ploidy)
            c_pbar = tqdm(total=len(contigs), unit=" contig")
            offset = 0
            chunk_idx = 0
            for c in contigs:
                c_pbar.set_description(f"Processing contig {c}")
                v_pbar = tqdm(unit=" variant", position=1)
                v_pbar.set_description("Reading variants")
                with vcf.using_pbar(v_pbar) as vcf:
                    # genos: (s p v)
                    if with_dosages:
                        for genos, dosages in vcf.chunk(
                            c, max_mem=max_mem, mode=VCF.Genos8Dosages
                        ):
                            n_vars = genos.shape[-1]
                            if n_vars == 0:
                                continue
                            var_idxs = np.arange(
                                offset, offset + n_vars, dtype=np.int32
                            )
                            sp_genos, sp_dosages = dense2sparse(
                                genos, var_idxs, dosages
                            )
                            _write_genos(tdir / str(chunk_idx), sp_genos)
                            _write_dosages(tdir / str(chunk_idx), sp_dosages.data)
                            offset += n_vars
                            chunk_idx += 1
                    else:
                        for genos in vcf.chunk(c, max_mem=max_mem, mode=VCF.Genos8):
                            n_vars = genos.shape[-1]
                            if n_vars == 0:
                                continue
                            var_idxs = np.arange(
                                offset, offset + n_vars, dtype=np.int32
                            )
                            sp_genos = dense2sparse(genos, var_idxs)
                            _write_genos(tdir / str(chunk_idx), sp_genos)
                            offset += n_vars
                            chunk_idx += 1
                    v_pbar.close()
                c_pbar.update()
            c_pbar.close()

            logger.info("Concatenating intermediate chunks")
            _concat_data(out, tdir, shape, with_dosages=with_dosages)

    @classmethod
    def from_pgen(
        cls,
        out: str | Path,
        pgen: PGEN,
        max_mem: int | str,
        overwrite: bool = False,
        with_dosages: bool = False,
    ):
        """Create a Sparse Variant (.svar) from a PGEN.

        Parameters
        ----------
        out
            Path to the output directory.
        pgen
            PGEN file to write from.
        max_mem
            Maximum memory to use while writing.
        overwrite
            Whether to overwrite the output directory if it exists.
        with_dosages
            Whether to write dosages.
        """
        out = Path(out)

        if out.exists() and not overwrite:
            raise FileExistsError(
                f"Output path {out} already exists. Use overwrite=True to overwrite."
            )
        out.mkdir(parents=True, exist_ok=True)

        contigs = pgen.contigs
        with open(out / "metadata.json", "w") as f:
            json = SparseVarMetadata(
                version=CURRENT_VERSION,
                contigs=contigs,
                samples=pgen.available_samples,
                ploidy=pgen.ploidy,
            ).model_dump_json()
            f.write(json)

        shutil.copy(pgen._index_path(), cls._index_path(out))

        with TemporaryDirectory() as tdir:
            tdir = Path(tdir)

            shape = (pgen.n_samples, pgen.ploidy)
            n_variants = len(pgen._index)
            pbar = tqdm(total=n_variants, unit=" variant")
            offset = 0
            chunk_idx = 0
            for c in contigs:
                pbar.set_description(f"Contig {c}, readings variants")
                if with_dosages:
                    if pgen._sei is None:
                        raise ValueError("PGEN must be bi-allelic with filters applied")
                    offset, chunk_idx = _process_contig_dosages(
                        pgen.chunk_ranges(c, max_mem=max_mem, mode=PGEN.GenosDosages),
                        tdir,
                        offset,
                        chunk_idx,
                        pbar,
                    )
                else:
                    offset, chunk_idx = _process_contig(
                        pgen.chunk_ranges(c, max_mem=max_mem, mode=PGEN.Genos),
                        tdir,
                        offset,
                        chunk_idx,
                        pbar,
                    )
            pbar.close()

            logger.info("Concatenating intermediate chunks")
            _concat_data(out, tdir, shape, with_dosages=with_dosages)

    @classmethod
    def _index_path(cls, root: Path):
        """Path to the index file."""
        return root / "index.arrow"

    def _load_var_table(self, attrs: IntoExpr | None = None) -> pl.DataFrame:
        """Load the variant table from the index file."""

        min_attrs: set[Any] = {"ALT", "ILEN"}
        if attrs is not None:
            if not isinstance(attrs, str) and isinstance(attrs, Iterable):
                min_attrs.update(attrs)
            else:
                min_attrs.add(attrs)
        attrs = list(min_attrs)

        index = (
            pl.scan_ipc(
                self._index_path(self.path), row_index_name="index", memory_map=False
            )
            .select("index", "CHROM", "POS", "REF", *attrs)
            .collect()
        )

        return index

    def annotate_with_gtf(
        self,
        gtf: str | pl.DataFrame,
        level_filter: int | None = 1,
        write_back: bool = True,
        *,
        strand_encoding: dict[str | None, int] | None = None,
        codon_null_token: int | None = None,
    ) -> pl.DataFrame:
        """
        Annotate variants with gene_id, strand, and codon_pos from GTF CDS features.

        Computes codon position for SNVs only; indels receive strand but null codon_pos.

        Parameters
        ----------
        gtf : str or pl.DataFrame
            Path to GTF file (.gtf or .gtf.gz) or pre-loaded Polars DataFrame.
        level_filter : int or None, default 1
            If set, keep rows with GTF 'level' <= level_filter (1 = highest quality).
        write_back : bool, default True
            If True, update self.var_table in-place and write to index.arrow file.
        strand_encoding : dict or None, optional
            Encode strand as integers. Example: {'+': 0, '-': 1, None: 2}
        codon_null_token : int or None, optional
            Replace null codon_pos with this integer for ML models.

        Returns
        -------
        pl.DataFrame
            Columns: varID (UInt32), gene_id (Utf8), strand (Utf8/Int16), codon_pos (Int8/Int16)

        Examples
        --------
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
                annot = _get_strand_and_codon_pos(cds_df, self.var_table, self._c_norm)
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
                self.var_table = (
                    self.var_table.lazy()
                    .with_row_index("varID")
                    .join(annot.lazy(), on="varID", how="left")
                    .drop("varID")
                    .collect()
                )
                df = self._to_df()
                df.write_ipc(self._index_path(self.path))
                logger.info("Wrote gene_id, strand, codon_pos to index.arrow")

            pbar.update(1)

        return annot

    def cache_afs(self):
        """Cache the allele frequencies on disk. Will also load all possible attributes and add the AF column in-memory."""
        self._load_all_attrs()
        afs = self._compute_afs()
        self.var_table = self.var_table.with_columns(AF=pl.Series(afs))
        self._write_afs()

    def _load_all_attrs(self):
        idx_df = pl.scan_ipc(self._index_path(self.path))
        schema = idx_df.collect_schema()
        missing = set(schema) - set(self.var_table.columns)
        missing_attrs = idx_df.select(*missing).collect()
        self.var_table = self.var_table.hstack(missing_attrs)

    def _compute_afs(self) -> NDArray[np.float32]:
        n_samples, ploidy, _ = cast(tuple[int, int, None], self.genos.shape)
        max_count = n_samples * ploidy
        afs = np.zeros(self.n_variants, np.float32)
        _nb_af_helper(afs, self.genos.data, self.genos.offsets, max_count)
        return afs

    def _write_afs(self):
        df = self._to_df()
        df.write_ipc(self._index_path(self.path))

    def _to_df(self) -> pl.DataFrame:
        return self.var_table.drop("index")

    def _load_genos(self):
        def memmap2array(layout: Content, **kwargs):
            if isinstance(layout, NumpyArray):
                data = layout.data
                if isinstance(data, np.memmap):
                    data = data[:]
                return NumpyArray(data)

        self.genos = ak.transform(memmap2array, self.genos)  # type: ignore


@nb.njit(nogil=True, cache=True)
def _nb_af_helper(afs, v_idxs, offsets, max_count):
    for i in range(len(offsets) - 1):
        o_s, o_e = offsets[i], offsets[i + 1]
        v_slice = v_idxs[o_s:o_e]
        afs[v_slice] += 1
    afs /= max_count


def _process_contig(
    chunker: Generator[Generator[NDArray[np.int8 | np.int32]] | None],
    tdir: Path,
    offset: int,
    chunk_idx: int,
    pbar: tqdm | None = None,
) -> tuple[int, int]:
    for range_ in chunker:
        if range_ is None:
            continue
        # genos: (s p v)
        for genos in range_:
            n_vars = genos.shape[-1]
            if n_vars == 0:
                continue
            var_idxs = np.arange(offset, offset + n_vars, dtype=np.int32)
            sp_genos = dense2sparse(genos.astype(np.int8), var_idxs)
            _write_genos(tdir / str(chunk_idx), sp_genos)
            offset += n_vars
            chunk_idx += 1
            if pbar is not None:
                pbar.update(n_vars)
    return offset, chunk_idx


def _process_contig_dosages(
    chunker: Generator[
        Generator[tuple[NDArray[np.int8 | np.int32], NDArray[DOSAGE_TYPE]]] | None
    ],
    tdir: Path,
    offset: int,
    chunk_idx: int,
    pbar: tqdm | None = None,
) -> tuple[int, int]:
    for range_ in chunker:
        if range_ is None:
            continue
        # genos, dosages: (s p v)
        for genos, dosages in range_:
            n_vars = genos.shape[-1]
            if n_vars == 0:
                continue
            var_idxs = np.arange(offset, offset + n_vars, dtype=np.int32)

            sp_genos, sp_dosages = dense2sparse(
                genos.astype(np.int8), var_idxs, dosages
            )
            _write_genos(tdir / str(chunk_idx), sp_genos)
            _write_dosages(tdir / str(chunk_idx), sp_dosages.data)
            offset += n_vars
            chunk_idx += 1
            if pbar is not None:
                pbar.update(n_vars)
    return offset, chunk_idx


def _open_genos(path: Path, shape: tuple[int | None, ...], mode: Literal["r", "r+"]):
    # Load the memory-mapped files
    var_idxs = np.memmap(path / "variant_idxs.npy", dtype=V_IDX_TYPE, mode=mode)
    offsets = np.memmap(path / "offsets.npy", dtype=np.int64, mode=mode)

    sp_genos = SparseGenotypes.from_offsets(var_idxs, shape, offsets)
    return sp_genos


def _open_dosages(path: Path, shape: tuple[int | None, ...], mode: Literal["r", "r+"]):
    # Load the memory-mapped files
    dosages = np.memmap(path / "dosages.npy", dtype=DOSAGE_TYPE, mode=mode)
    offsets = np.memmap(path / "offsets.npy", dtype=np.int64, mode=mode)

    sp_genos = SparseDosages.from_offsets(dosages, shape, offsets)
    return sp_genos


def _write_genos(path: Path, sp_genos: SparseGenotypes):
    path.mkdir(parents=True, exist_ok=True)

    var_idxs = np.memmap(
        path / "variant_idxs.npy",
        shape=sp_genos.data.shape,
        dtype=sp_genos.data.dtype,
        mode="w+",
    )
    var_idxs[:] = sp_genos.data
    var_idxs.flush()

    offsets = np.memmap(
        path / "offsets.npy",
        shape=sp_genos.offsets.shape,
        dtype=sp_genos.offsets.dtype,
        mode="w+",
    )
    offsets[:] = sp_genos.offsets
    offsets.flush()


def _write_dosages(path: Path, dosages: NDArray[DOSAGE_TYPE]):
    path.mkdir(parents=True, exist_ok=True)

    dosages_memmap = np.memmap(
        path / "dosages.npy",
        shape=dosages.shape,
        dtype=dosages.dtype,
        mode="w+",
    )
    dosages_memmap[:] = dosages
    dosages_memmap.flush()


def _concat_data(
    out_path: Path,
    chunks_path: Path,
    shape: tuple[int, int],
    with_dosages: bool = False,
):
    out_path.mkdir(parents=True, exist_ok=True)

    # [1, 2, 3, ...]
    chunk_dirs = natsorted(chunks_path.iterdir())

    vars_per_sp = np.zeros(shape, dtype=np.int32)
    ls_sp_genos: list[SparseGenotypes] = []
    for chunk_dir in chunk_dirs:
        sp_genos = _open_genos(chunk_dir, (*shape, None), mode="r")
        vars_per_sp += sp_genos.lengths
        ls_sp_genos.append(sp_genos)

    # offsets should be relatively small even for ultra-large datasets
    # scales O(n_samples * ploidy)
    offsets = lengths_to_offsets(vars_per_sp)
    offsets_memmap = np.memmap(
        out_path / "offsets.npy", dtype=offsets.dtype, mode="w+", shape=offsets.shape
    )
    offsets_memmap[:] = offsets
    offsets_memmap.flush()

    var_idxs_memmap = np.memmap(
        out_path / "variant_idxs.npy", dtype=V_IDX_TYPE, mode="w+", shape=offsets[-1]
    )
    _concat_helper(
        var_idxs_memmap,
        offsets,
        [a.data for a in ls_sp_genos],
        [a.offsets for a in ls_sp_genos],
        shape,
    )
    var_idxs_memmap.flush()

    if with_dosages:
        ls_: list[SparseDosages] = []
        for chunk_dir in chunk_dirs:
            sp_dosages = _open_dosages(chunk_dir, (*shape, None), mode="r")
            vars_per_sp += sp_dosages.lengths
            ls_.append(sp_dosages)
        dosages_memmap = np.memmap(
            out_path / "dosages.npy", dtype=DOSAGE_TYPE, mode="w+", shape=offsets[-1]
        )
        _concat_helper(
            dosages_memmap,
            offsets,
            [a.data for a in ls_],
            [a.offsets for a in ls_],
            shape,
        )
        dosages_memmap.flush()


@nb.njit(parallel=True, nogil=True, cache=True)
def _concat_helper(
    out_data: NDArray[DTYPE],
    out_offsets: NDArray[OFFSET_TYPE],
    in_data: list[NDArray[DTYPE]],
    in_offsets: list[NDArray[OFFSET_TYPE]],
    shape: tuple[int, int],
):
    n_samples, ploidy = shape
    n_chunks = len(in_data)
    assert len(in_offsets) == n_chunks
    for s in nb.prange(n_samples):
        for p in nb.prange(ploidy):
            sp = s * ploidy + p
            o_s, o_e = out_offsets[sp], out_offsets[sp + 1]
            sp_out_idxs = out_data[o_s:o_e]
            offset = 0
            for chunk in range(n_chunks):
                i_s, i_e = in_offsets[chunk][sp], in_offsets[chunk][sp + 1]
                chunk_len = i_e - i_s
                sp_out_idxs[offset : offset + chunk_len] = in_data[chunk][i_s:i_e]
                offset += chunk_len


@nb.njit(parallel=True, nogil=True, cache=True)
def _find_starts_ends(
    genos: NDArray[V_IDX_TYPE],
    geno_offsets: NDArray[OFFSET_TYPE],
    var_ranges: NDArray[V_IDX_TYPE],
    sample_idxs: NDArray[np.int64],
    ploidy: int,
):
    """Find the start and end offsets of the sparse genotypes for each range.

    Parameters
    ----------
    genos
        Sparse genotypes
    geno_offsets
        Genotype offsets
    var_ranges
        Shape = (ranges 2) Variant index ranges.

    Returns
    -------
        Shape: (ranges samples ploidy 2). The first column is the start index of the variant
        and the second column is the end index of the variant.
    """
    n_ranges = len(var_ranges)
    n_samples = len(sample_idxs)
    out_offsets = np.empty((2, n_ranges, n_samples, ploidy), dtype=OFFSET_TYPE)
    sorter = np.argsort(var_ranges[:, 0])
    var_ranges = var_ranges[sorter]

    for s in nb.prange(n_samples):
        for p in nb.prange(ploidy):
            s_idx = sample_idxs[s]
            sp = s_idx * ploidy + p
            o_s, o_e = geno_offsets[sp], geno_offsets[sp + 1]
            sp_genos = genos[o_s:o_e]
            # add o_s to make indices relative to whole array
            out_offsets[..., s, p] = np.searchsorted(sp_genos, var_ranges).T + o_s

    no_vars = var_ranges[:, 0] == var_ranges[:, 1]
    out_offsets[:, no_vars] = np.iinfo(OFFSET_TYPE).max

    unsorter = sorter[sorter]
    out_offsets = out_offsets[:, unsorter]

    return out_offsets


@nb.njit(parallel=False, nogil=True, cache=True)
def _find_starts_ends_with_length(
    genos: NDArray[V_IDX_TYPE],
    geno_offsets: NDArray[OFFSET_TYPE],
    q_starts: NDArray[POS_TYPE],
    q_ends: NDArray[POS_TYPE],
    var_ranges: NDArray[V_IDX_TYPE],
    v_starts: NDArray[np.int32],
    ilens: NDArray[np.int32],
    sample_idxs: NDArray[np.int64],
    ploidy: int,
    contig_max_idx: int,
    out: NDArray[OFFSET_TYPE] | None = None,
):
    """Find the start and end offsets of the sparse genotypes for each range.

    Parameters
    ----------
    genos
        Sparse genotypes
    geno_offsets
        Genotype offsets
    var_ranges
        Shape = (ranges 2) Variant index ranges.

    Returns
    -------
        Shape: (2 ranges samples ploidy). The first column is the start index of the variant
        and the second column is the end index of the variant.
    """
    n_ranges = len(q_starts)
    n_samples = len(sample_idxs)
    if out is None:
        out = np.empty((2, n_ranges, n_samples, ploidy), dtype=OFFSET_TYPE)

    sorter = np.argsort(var_ranges[:, 0])
    var_ranges = var_ranges[sorter]

    for s in nb.prange(n_samples):
        for p in nb.prange(ploidy):
            s_idx = sample_idxs[s]
            sp = s_idx * ploidy + p
            o_s, o_e = geno_offsets[sp], geno_offsets[sp + 1]
            sp_genos = genos[o_s:o_e]

            max_idx = np.searchsorted(sp_genos, contig_max_idx + 1)
            start_idxs = np.searchsorted(sp_genos, var_ranges[:, 0])

            for r in range(n_ranges):
                start_idx: np.intp = start_idxs[r]

                if var_ranges[r, 0] == var_ranges[r, 1]:
                    out[:, r, s, p] = np.iinfo(OFFSET_TYPE).max
                    continue

                # add o_s to make indices relative to whole array
                out[0, r, s, p] = start_idx + o_s
                if start_idx == max_idx:
                    # no variants in this range
                    out[1, r, s, p] = start_idx + o_s
                    continue

                q_start: POS_TYPE = q_starts[r]
                q_len: POS_TYPE = q_ends[r] - q_start
                last_v_end: POS_TYPE = q_start
                written_len = 0
                # ensure geno_idx is assigned when start_idx == n_vars
                geno_idx = start_idx
                for geno_idx in range(start_idx, max_idx):
                    v_idx: V_IDX_TYPE = sp_genos[geno_idx]
                    v_start: np.int32 = v_starts[v_idx]
                    ilen: np.int32 = ilens[v_idx]

                    # only add atomized length if v_start >= ref_start
                    maybe_add_one = POS_TYPE(v_start >= q_start)

                    # only variants within query can add to write length
                    if v_start >= q_start:
                        written_len += v_start - last_v_end
                        if written_len >= q_len:
                            geno_idx -= 1
                            break

                        v_write_len = (
                            max(0, ilen)  # insertion length  # type: ignore
                            + maybe_add_one  # maybe add atomized length
                        )

                        # right-clip insertions
                        # Not necessary since it's inconsequential to overshoot the target length
                        # and insertions don't affect the ref length for getting tracks.
                        # Nevertheless, here's the code to clip a final insertion if we ever wanted to:
                        # missing_len = target_len - cum_write_len
                        # clip_right = max(0, v_len - missing_len)
                        # v_len -= clip_right

                        written_len += v_write_len
                        if written_len >= q_len:
                            break

                    v_end = (
                        v_start
                        - min(0, ilen)  # deletion length  # type: ignore
                        + maybe_add_one  # maybe add atomized length
                    )
                    last_v_end = max(last_v_end, v_end)  # type: ignore

                # add o_s to make indices relative to whole array
                out[1, r, s, p] = geno_idx + o_s + 1

    unsorter = sorter[sorter]
    out = out[:, unsorter]

    return out


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
    """
    Calculate strand and codon position for variants overlapping CDS regions.

    Parameters
    ----------
    cds_df : pl.DataFrame
        CDS features from GTF with columns: chrom, start, end, strand, frame,
        gene_id, transcript_id, gene_biotype, transcript_support_level, tag
        coordinates should be 1-based
    var_table : pl.DataFrame
        Variant table with columns: index, CHROM, POS, ILEN, ...
        POS should be 1-based
    contig_normalizer : ContigNormalizer
        Normalizer to match chromosome names between CDS and granges


    Returns
    -------
    pl.DataFrame
        Annotation with varID, gene_id, strand, codon_pos
    """

    # Normalize CDS chromosome names to match granges
    # Cast to string first to avoid categorical comparison issues
    cds_df = cds_df.with_columns(
        pl.col("chrom").cast(pl.Utf8).replace(contig_normalizer.contig_map)
    )

    # Filter out CDS features with chromosomes not in granges
    cds_df = cds_df.filter(pl.col("chrom").is_in(contig_normalizer.contigs))

    # Prepare var_table for pb.overlap by creating interval columns
    var_intervals = var_table.select(
        pl.col("ILEN").list.first(),
        var_id="index",
        chrom="CHROM",
        start=pl.col("POS"),
        end=pl.col("POS") - pl.col("ILEN").list.first().clip(upper_bound=0),
    )

    # Check if CDS or var_table is empty
    if cds_df.is_empty() or var_table.is_empty():
        return _empty_annot()

    joined_cds = (
        cast(
            pl.LazyFrame,
            pb.overlap(
                var_intervals, cds_df, use_zero_based=False, projection_pushdown=True
            ),
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
