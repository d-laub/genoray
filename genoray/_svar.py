from __future__ import annotations

import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Generator, Literal, TypeVar, cast

import numba as nb
import numpy as np
import polars as pl
import pyranges as pr
from hirola import HashTable
from loguru import logger
from natsort import natsorted
from numpy.typing import ArrayLike, NDArray
from polars._typing import IntoExpr
from seqpro._ragged import OFFSET_TYPE, Ragged, lengths_to_offsets
from tqdm.auto import tqdm

from ._pgen import PGEN
from ._utils import ContigNormalizer
from ._vcf import VCF

POS_TYPE = np.int64
V_IDX_TYPE = np.int32
DOSAGE_TYPE = np.float32
INT64_MAX = np.iinfo(POS_TYPE).max
DTYPE = TypeVar("DTYPE", bound=np.generic)


class SparseGenotypes(Ragged[V_IDX_TYPE]):
    """A Ragged array of variant indices with additional guarantees and methods:

    - dtype is :code:`int32`
    - Ragged shape of **at least** 2 dimensions where the final two correspond to (samples, ploidy)
    - :code:`from_dense` to convert dense genotypes to sparse genotypes
    """

    def __attrs_post_init__(self):
        assert self.ndim >= 2, "SparseGenotypes must have at least 2 dimensions"
        assert self.dtype.type == V_IDX_TYPE, "SparseGenotypes must be of type int32"

    @property
    def n_samples(self) -> int:
        """Number of samples"""
        return self.shape[-2]

    @property
    def ploidy(self) -> int:
        """Ploidy"""
        return self.shape[-1]

    @classmethod
    def from_dense(cls, genos: NDArray[np.int8], var_idxs: NDArray[V_IDX_TYPE]):
        """Convert dense genotypes to sparse genotypes.

        Parameters
        ----------
        genos
            Shape = (samples ploidy variants) Genotypes.
        var_idxs
            Shape = (variants) Variant indices.
        """
        # (s p v)
        keep = genos == 1
        data = var_idxs[keep.nonzero()[-1]]
        lengths = keep.sum(-1)
        return cls.from_lengths(data, lengths)


class SparseDosages(Ragged[DOSAGE_TYPE]):
    """A Ragged array of dosages with additional guarantees and methods:

    - dtype is :code:`float32`
    - Ragged shape of **at least** 2 dimensions that should correspond to (samples, ploidy)
    - :code:`from_dense` to convert dense dosages to sparse dosages
    """

    def __attrs_post_init__(self):
        assert self.ndim >= 2, "SparseDosages must have at least 2 dimensions"
        assert self.dtype.type == DOSAGE_TYPE, "SparseDosages must be of type float32"

    @property
    def n_samples(self) -> int:
        """Number of samples"""
        return self.shape[-2]

    @property
    def ploidy(self) -> int:
        """Ploidy"""
        return self.shape[-1]

    @classmethod
    def from_dense(
        cls,
        genos: NDArray[np.int8],
        var_idxs: NDArray[V_IDX_TYPE],
        dosages: NDArray[DOSAGE_TYPE],
    ) -> tuple[SparseGenotypes, SparseDosages]:
        """Convert dense dosages to sparse dosages.

        Parameters
        ----------
        genos
            Shape = (samples ploidy variants) Genotypes.
        var_idxs
            Shape = (variants) Variant indices.
        dosages
            Shape = (samples variants) Cancer cell fractions (or proxy thereof).
        """
        # (s p v)
        keep: NDArray[np.bool_] = genos == 1
        geno_data = var_idxs[keep.nonzero()[-1]]
        lengths = keep.sum(-1)
        offsets = lengths_to_offsets(lengths)
        sp_genos = SparseGenotypes(geno_data, lengths.shape, offsets, lengths)

        # (s v) -> (s p v)
        dosage_data = np.broadcast_to(dosages[:, None], genos.shape)[keep]
        sp_dosages = SparseDosages(dosage_data, lengths.shape, offsets, lengths)

        return sp_genos, sp_dosages


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
    available_samples: list[str]
    ploidy: int
    contigs: list[str]
    genos: SparseGenotypes
    dosages: SparseDosages | None
    granges: pr.PyRanges
    attrs: pl.DataFrame
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
        return len(self.granges)

    @property
    def has_dosages(self) -> bool:
        return (self.path / "dosages.npy").exists()

    def __init__(self, path: str | Path, attrs: IntoExpr | None = None):
        path = Path(path)
        self.path = path

        with open(path / "metadata.json") as f:
            metadata = json.load(f)
        contigs = metadata["contigs"]
        self.contigs = contigs
        self.available_samples = metadata["samples"]
        self.ploidy = metadata["ploidy"]
        samples = np.array(self.available_samples)
        self._s2i = HashTable(
            len(samples) * 2,  # type: ignore
            dtype=samples.dtype,
        )
        self._s2i.add(samples)

        self._c_norm = ContigNormalizer(contigs)
        self.genos = _open_genos(path, (self.n_samples, self.ploidy), "r")
        if (path / "dosages.npy").exists():
            dosage_data = np.memmap(path / "dosages.npy", dtype=DOSAGE_TYPE, mode="r")
            self.dosages = SparseDosages.from_offsets(
                dosage_data, self.genos.shape, self.genos.offsets
            )
        else:
            self.dosages = None
        logger.info("Loading genoray index")
        self.granges, self.attrs = self._load_index(attrs)
        self._is_biallelic = (self.attrs["ALT"].list.len() == 1).all()
        vars_per_contig = np.array(
            [len(self.granges[c]) for c in self.contigs]
        ).cumsum()
        self._c_max_idxs = {c: v - 1 for c, v in zip(self.contigs, vars_per_contig)}

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
        starts = np.atleast_1d(np.asarray(starts, POS_TYPE))
        n_ranges = len(starts)

        c = self._c_norm.norm(contig)
        if c is None:
            return np.full((n_ranges, 2), np.iinfo(V_IDX_TYPE).max, V_IDX_TYPE)

        ends = np.atleast_1d(np.asarray(ends, POS_TYPE))
        queries = pr.PyRanges(
            pl.DataFrame(
                {
                    "Chromosome": np.full(n_ranges, c),
                    "Start": starts,
                    "End": ends,
                }
            )
            .with_row_index("query")
            .to_pandas(use_pyarrow_extension_array=True)
        )
        join = queries.join(self.granges)

        if len(join) == 0:
            return np.full((n_ranges, 2), np.iinfo(V_IDX_TYPE).max, V_IDX_TYPE)

        join = pl.from_pandas(join.df).select("query", "index")

        missing_queries = np.setdiff1d(
            np.arange(n_ranges, dtype=np.uint32),
            join["query"].unique(),
            assume_unique=True,
        ).astype(np.uint32)
        if (missing_queries).size > 0:
            missing_join = pl.DataFrame(
                {"query": missing_queries},
            ).with_columns(index=pl.lit(None).cast(pl.UInt32))
            join = join.vstack(missing_join)

        var_ranges = (
            join.group_by("query")
            .agg(start=pl.col("index").min(), end=pl.col("index").max() + 1)
            .with_columns(pl.col("start", "end").fill_null(np.iinfo(V_IDX_TYPE).max))
            .sort("query")
            .drop("query")
            .to_numpy()
            .astype(V_IDX_TYPE)
        )

        return var_ranges

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
            Shape: (ranges, samples, ploidy, 2). The first column is the start index of the variant
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

        # (2 r s p)
        out = _find_starts_ends_with_length(
            self.genos.data,
            self.genos.offsets,
            starts,
            ends,
            var_ranges,
            self.granges.Start.to_numpy(),
            self.attrs["ILEN"].list.first().to_numpy(),
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
            (n_ranges, n_samples, self.ploidy),
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
            (n_ranges, n_samples, self.ploidy),
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
            json.dump(
                {
                    "contigs": contigs,
                    "samples": vcf.available_samples,
                    "ploidy": vcf.ploidy,
                },
                f,
            )

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
                            var_idxs = np.arange(
                                offset, offset + n_vars, dtype=np.int32
                            )
                            sp_genos, sp_dosages = SparseDosages.from_dense(
                                genos, var_idxs, dosages
                            )
                            _write_genos(tdir / str(chunk_idx), sp_genos)
                            _write_dosages(tdir / str(chunk_idx), sp_dosages.data)
                            offset += n_vars
                            chunk_idx += 1
                    else:
                        for genos in vcf.chunk(c, max_mem=max_mem, mode=VCF.Genos8):
                            n_vars = genos.shape[-1]
                            var_idxs = np.arange(
                                offset, offset + n_vars, dtype=np.int32
                            )
                            sp_genos = SparseGenotypes.from_dense(genos, var_idxs)
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
            json.dump(
                {
                    "contigs": contigs,
                    "samples": pgen.available_samples,
                    "ploidy": pgen.ploidy,
                },
                f,
            )

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

    def _load_index(
        self, attrs: IntoExpr | None = None
    ) -> tuple[pr.PyRanges, pl.DataFrame]:
        """Load the index file and return the granges and attributes."""

        min_attrs: list[Any] = ["ALT", "ILEN"]
        if attrs is not None:
            if isinstance(attrs, list):
                min_attrs.extend(attrs)
            else:
                min_attrs.append(attrs)
        attrs = min_attrs

        index = (
            pl.scan_ipc(
                self._index_path(self.path), row_index_name="index", memory_map=False
            )
            .select("CHROM", "POS", "REF", "index", *attrs)
            .collect()
        )

        granges = pr.PyRanges(
            index.select(
                "index",
                Chromosome="CHROM",
                Start=pl.col("POS") - 1,
                End=pl.col("POS") + pl.col("REF").str.len_bytes() - 1,
            ).to_pandas()
        )
        attr_df = index.select(*attrs)
        return granges, attr_df


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
            var_idxs = np.arange(offset, offset + n_vars, dtype=np.int32)
            sp_genos = SparseGenotypes.from_dense(genos.astype(np.int8), var_idxs)
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
            var_idxs = np.arange(offset, offset + n_vars, dtype=np.int32)

            sp_genos, sp_dosages = SparseDosages.from_dense(
                genos.astype(np.int8), var_idxs, dosages
            )
            _write_genos(tdir / str(chunk_idx), sp_genos)
            _write_dosages(tdir / str(chunk_idx), sp_dosages.data)
            offset += n_vars
            chunk_idx += 1
            if pbar is not None:
                pbar.update(n_vars)
    return offset, chunk_idx


def _open_genos(path: Path, shape: tuple[int, ...], mode: Literal["r", "r+"]):
    # Load the memory-mapped files
    var_idxs = np.memmap(path / "variant_idxs.npy", dtype=V_IDX_TYPE, mode=mode)
    offsets = np.memmap(path / "offsets.npy", dtype=np.int64, mode=mode)

    sp_genos = SparseGenotypes.from_offsets(var_idxs, shape, offsets)
    return sp_genos


def _open_dosages(path: Path, shape: tuple[int, ...], mode: Literal["r", "r+"]):
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
        sp_genos = _open_genos(chunk_dir, shape, mode="r")
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
            sp_dosages = _open_dosages(chunk_dir, shape, mode="r")
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

    for r in nb.prange(n_ranges):
        for s in nb.prange(n_samples):
            for p in nb.prange(ploidy):
                if var_ranges[r, 0] == var_ranges[r, 1]:
                    out_offsets[:, r, s, p] = np.iinfo(OFFSET_TYPE).max
                    continue

                s_idx = sample_idxs[s]
                sp = s_idx * ploidy + p
                o_s, o_e = geno_offsets[sp], geno_offsets[sp + 1]
                sp_genos = genos[o_s:o_e]
                # add o_s to make indices relative to whole array
                out_offsets[:, r, s, p] = np.searchsorted(sp_genos, var_ranges[r]) + o_s

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

    for r in nb.prange(n_ranges):
        for s in nb.prange(n_samples):
            for p in nb.prange(ploidy):
                if var_ranges[r, 0] == var_ranges[r, 1]:
                    out[:, r, s, p] = np.iinfo(OFFSET_TYPE).max
                    continue

                s_idx = sample_idxs[s]
                sp = s_idx * ploidy + p
                o_s, o_e = geno_offsets[sp], geno_offsets[sp + 1]
                sp_genos = genos[o_s:o_e]
                start_idx, max_idx = np.searchsorted(
                    sp_genos, [var_ranges[r, 0], contig_max_idx + 1]
                )

                # add o_s to make indices relative to whole array
                out[0, r, s, p] = start_idx + o_s
                if start_idx == max_idx:
                    # no variants in this range
                    out[1, r, s, p] = start_idx + o_s
                    continue

                q_start: POS_TYPE = q_starts[r]
                q_len: POS_TYPE = q_ends[r] - q_start
                last_v_end = q_start
                written_len = 0
                # ensure geno_idx is assigned when start_idx == n_vars
                geno_idx = start_idx
                for geno_idx in range(start_idx, max_idx):
                    v_idx: V_IDX_TYPE = sp_genos[geno_idx]
                    v_start = v_starts[v_idx]
                    ilen: np.int32 = ilens[v_idx]

                    # only add atomized length if v_start >= ref_start
                    maybe_add_one = POS_TYPE(v_start >= q_start)

                    # only variants within query can add to write length
                    if v_start >= q_start:
                        v_write_len = (
                            (v_start - last_v_end)  # dist between last var and this var
                            + max(0, ilen)  # insertion length
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
                        - min(0, ilen)  # deletion length
                        + maybe_add_one  # maybe add atomized length
                    )
                    last_v_end = max(last_v_end, v_end)

                # add o_s to make indices relative to whole array
                out[1, r, s, p] = geno_idx + o_s + 1

    return out
