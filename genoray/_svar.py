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
CCF_TYPE = np.float32
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


class SparseCCFs(Ragged[CCF_TYPE]):
    """A Ragged array of CCFs with additional guarantees and methods:

    - dtype is :code:`float32`
    - Ragged shape of **at least** 2 dimensions that should correspond to (samples, ploidy)
    - :code:`from_dense` to convert dense CCFs to sparse CCFs
    """

    def __attrs_post_init__(self):
        assert self.ndim >= 2, "SparseCCFs must have at least 2 dimensions"
        assert self.dtype.type == CCF_TYPE, "SparseCCFs must be of type float32"

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
        ccfs: NDArray[CCF_TYPE],
        v_starts: NDArray[POS_TYPE],
        ilens: NDArray[np.int32],
    ) -> tuple[SparseGenotypes, SparseCCFs]:
        """Convert dense CCFs to sparse CCFs. Infers missing CCFs as germline variants
        and sets their CCFs to 1 - sum(overlapping somatic CCFs).

        Parameters
        ----------
        genos
            Shape = (samples ploidy variants) Genotypes.
        var_idxs
            Shape = (variants) Variant indices.
        ccfs
            Shape = (samples variants) Cancer cell fractions (or proxy thereof).
        v_starts
            Shape = (total_variants) 0-based start positions.
        ilens
            Shape = (total_variants) Indel lengths.
        """
        # (s p v)
        keep: NDArray[np.bool_] = genos == 1
        geno_data = var_idxs[keep.nonzero()[-1]]
        lengths = keep.sum(-1)
        offsets = lengths_to_offsets(lengths)
        sp_genos = SparseGenotypes(geno_data, lengths.shape, offsets, lengths)

        # (s v) -> (s p v)
        ccf_data = np.broadcast_to(ccfs[:, None], genos.shape)[keep]
        _infer_germline_ccfs(
            ccfs=ccf_data,
            v_idxs=sp_genos.data,
            v_starts=v_starts,
            ilens=ilens,
            v_offsets=sp_genos.offsets,
        )
        sp_ccfs = SparseCCFs(ccf_data, lengths.shape, offsets, lengths)

        return sp_genos, sp_ccfs


@nb.njit(parallel=True, nogil=True, cache=True)
def _infer_germline_ccfs(
    ccfs: NDArray[CCF_TYPE],
    v_idxs: NDArray[V_IDX_TYPE],
    v_starts: NDArray[POS_TYPE],
    ilens: NDArray[np.int32],
    v_offsets: NDArray[OFFSET_TYPE],
    max_ccf: float = 1.0,
):
    """Infer germline CCFs from the variant indices and variant starts.

    Germline variants are identified by having missing CCFs.
    i.e. they have a variant index but missing CCFs. Germline CCFs are inferred
    to be 1 - sum(overlapping somatic CCFs).

    Parameters
    ----------
    ccfs
        CCFs for the variants.
    v_idxs
        Variant indices for the variants.
    v_starts
        Variant start positions for the variants.
    ilens
        Variant lengths for the variants.
    v_offsets
        Variant offsets for the variants.
    max_ccf
        Maximum CCF value. Default is 1.0.
    """
    n_sp = len(v_offsets) - 1
    for o_idx in nb.prange(n_sp):
        o_s, o_e = v_offsets[o_idx], v_offsets[o_idx + 1]
        n_variants = o_e - o_s
        if n_variants == 0:
            continue

        ccf = ccfs[o_s:o_e]
        if not np.isnan(ccf).any():
            continue

        v_idx = v_idxs[o_s:o_e]
        v_start = v_starts[v_idx]
        ilen = ilens[v_idx]
        v_end = v_start - np.minimum(0, ilen) + 1  # atomized variants
        v_end_sorter = np.argsort(v_end)
        v_end = v_end[v_end_sorter]

        # sorted merge by starts then ends
        starts_ends = np.empty(n_variants * 2, POS_TYPE)
        se_local_idx = np.empty(n_variants * 2, V_IDX_TYPE)
        start_idx = 0
        end_idx = 0
        for i in range(n_variants * 2):
            end = v_end[end_idx]
            if start_idx < n_variants and v_start[start_idx] < end:
                starts_ends[i] = v_start[start_idx]
                se_local_idx[i] = start_idx
                start_idx += 1
            else:
                starts_ends[i] = -end
                se_local_idx[i] = v_end_sorter[end_idx]
                end_idx += 1

        #! this assumes that no germline variants overlap
        #! true with phased genotypes
        running_ccf = CCF_TYPE(0)
        g_idx = V_IDX_TYPE(-1)
        g_end = np.iinfo(POS_TYPE).max
        for i in range(n_variants * 2):
            pos: POS_TYPE = starts_ends[i]
            local_idx: V_IDX_TYPE = se_local_idx[i]
            pos_ccf: CCF_TYPE = ccf[local_idx]
            is_germ = np.isnan(pos_ccf)

            #! without this we will decrement the running CCF before setting the germline CCF
            # this is because tied ends are sorted by start, but the ends are 0-based exclusive
            # so we need to set the germline CCF before we start any decrementing
            if -pos >= g_end:
                ccf[g_idx] = max_ccf - running_ccf
                g_end = np.iinfo(POS_TYPE).max

            if is_germ and pos > 0:
                g_idx = local_idx
                # have to recompute the end because we sorted them above so the local idx points
                # to the wrong place
                g_end = pos - min(0, ilen[local_idx]) + 1
            else:
                # sign of position, with 0 being positive
                running_ccf += (2 * (pos >= 0) - 1) * pos_ccf

        np.nan_to_num(ccf, False, max_ccf)


class SparseVar:
    path: Path
    available_samples: list[str]
    ploidy: int
    contigs: list[str]
    genos: SparseGenotypes
    ccfs: SparseCCFs | None
    granges: pr.PyRanges
    attrs: pl.DataFrame
    _c_norm: ContigNormalizer
    _s2i: HashTable
    _c_max_idxs: dict[str, int]

    @property
    def n_samples(self) -> int:
        """Number of samples in the dataset."""
        return len(self.available_samples)

    @property
    def n_variants(self) -> int:
        """Number of variants in the dataset."""
        return len(self.granges)

    def __init__(self, path: str | Path, attrs: IntoExpr | None = None):
        """Open a Sparse Variant (SVAR) directory.

        Parameters
        ----------
        path
            Path to the SVAR directory.
        attrs
            Expression of attributes to load in addition to the ALT and ILEN columns.
        """
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
        if (path / "ccfs.npy").exists():
            ccf_data = np.memmap(path / "ccfs.npy", dtype=CCF_TYPE, mode="r")
            self.ccfs = SparseCCFs.from_offsets(
                ccf_data, self.genos.shape, self.genos.offsets
            )
        else:
            self.ccfs = None
        self.granges, self.attrs = self._load_index(attrs)
        vars_per_contig = np.array([len(df) for df in self.granges.values()]).cumsum()
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
                {
                    "query": missing_queries,
                    "index": np.full(len(missing_queries), np.iinfo(V_IDX_TYPE).max),
                },
                schema={"query": pl.UInt32, "index": pl.UInt32},
            )
            join = join.vstack(missing_join)

        var_ranges = (
            join.group_by("query")
            .agg(start=pl.col("index").min(), end=pl.col("index").max() + 1)
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
            return np.zeros((n_ranges, len(samples), self.ploidy, 2), OFFSET_TYPE)

        ends = np.atleast_1d(np.asarray(ends, POS_TYPE))
        # (r 2)
        var_ranges = self.var_ranges(contig, starts, ends)
        # (r s p 2)
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
        has_multiallelics = (self.attrs["ALT"].list.len() > 1).any()
        if has_multiallelics:
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
            return np.zeros((n_ranges, len(samples), self.ploidy, 2), OFFSET_TYPE)

        ends = np.atleast_1d(np.asarray(ends, POS_TYPE))
        # (r 2)
        var_ranges = self.var_ranges(contig, starts, ends)

        # (r s p 2)
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

        c = self._c_norm.norm(contig)
        if c is None:
            return SparseGenotypes.from_offsets(
                np.empty((0), V_IDX_TYPE),
                (n_samples, self.ploidy),
                np.zeros((0, 2), OFFSET_TYPE),
            )

        # (r s p 2)
        starts_ends = self._find_starts_ends(contig, starts, ends, samples)
        return SparseGenotypes.from_offsets(
            self.genos.data,
            (n_ranges, n_samples, self.ploidy),
            starts_ends.reshape(-1, 2),
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

        c = self._c_norm.norm(contig)
        if c is None:
            return SparseGenotypes.from_offsets(
                np.empty((0), V_IDX_TYPE),
                (n_samples, self.ploidy),
                np.zeros((0, 2), OFFSET_TYPE),
            )

        # (r s p 2)
        starts_ends = self._find_starts_ends_with_length(contig, starts, ends, samples)
        return SparseGenotypes.from_offsets(
            self.genos.data,
            (n_ranges, n_samples, self.ploidy),
            starts_ends.reshape(-1, 2),
        )

    @classmethod
    def from_vcf(
        cls,
        out: str | Path,
        vcf: VCF,
        max_mem: int | str,
        overwrite: bool = False,
        with_ccfs: bool = False,
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
        with_ccfs
            Whether to write CCFs.
        """
        out = Path(out)

        if out.exists() and not overwrite:
            raise FileExistsError(
                f"Output path {out} already exists. Use overwrite=True to overwrite."
            )
        out.mkdir(parents=True, exist_ok=True)

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

        if not vcf._index_path().exists():
            logger.info("Genoray VCF index not found, creating index.")
            vcf._write_gvi_index()
        shutil.copy(vcf._index_path(), cls._index_path(out))

        with TemporaryDirectory() as tdir:
            tdir = Path(tdir)

            if with_ccfs:
                if vcf._index is None:
                    raise ValueError("VCF must be bi-allelic with index loaded")
                v_starts = vcf._index.gr.Start.to_numpy()
                ilens = vcf._index.df["ILEN"].list.first().to_numpy()

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
                    if with_ccfs:
                        for genos, ccfs in vcf.chunk(
                            c, max_mem=max_mem, mode=VCF.Genos8Dosages
                        ):
                            n_vars = genos.shape[-1]
                            var_idxs = np.arange(
                                offset, offset + n_vars, dtype=np.int32
                            )
                            sp_genos, sp_ccfs = SparseCCFs.from_dense(
                                genos,
                                var_idxs,
                                ccfs,
                                v_starts,  # type: ignore | guaranteed bound by with_ccfs
                                ilens,  # type: ignore | guaranteed bound by with_ccfs
                            )
                            _write_genos(tdir / str(chunk_idx), sp_genos)
                            _write_ccfs(tdir / str(chunk_idx), sp_ccfs.data)
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
            _concat_data(out, tdir, shape, with_ccfs=with_ccfs)

    @classmethod
    def from_pgen(
        cls,
        out: str | Path,
        pgen: PGEN,
        max_mem: int | str,
        overwrite: bool = False,
        with_ccfs: bool = False,
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
        with_ccfs
            Whether to write CCFs.
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
                if with_ccfs:
                    if pgen._sei is None:
                        raise ValueError("PGEN must be bi-allelic with filters applied")
                    offset, chunk_idx = _process_contig_ccfs(
                        pgen.chunk_ranges(c, max_mem=max_mem, mode=PGEN.GenosDosages),
                        tdir,
                        offset,
                        chunk_idx,
                        pgen._sei.v_starts,
                        pgen._sei.ilens,
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
            _concat_data(out, tdir, shape, with_ccfs=with_ccfs)

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


def _process_contig_ccfs(
    chunker: Generator[
        Generator[tuple[NDArray[np.int8 | np.int32], NDArray[CCF_TYPE]]] | None
    ],
    tdir: Path,
    offset: int,
    chunk_idx: int,
    v_starts: NDArray[POS_TYPE],
    ilens: NDArray[np.int32],
    pbar: tqdm | None = None,
) -> tuple[int, int]:
    for range_ in chunker:
        if range_ is None:
            continue
        # genos, ccfs: (s p v)
        for genos, ccfs in range_:
            n_vars = genos.shape[-1]
            var_idxs = np.arange(offset, offset + n_vars, dtype=np.int32)

            sp_genos, sp_ccfs = SparseCCFs.from_dense(
                genos.astype(np.int8), var_idxs, ccfs, v_starts, ilens
            )
            _write_genos(tdir / str(chunk_idx), sp_genos)
            _write_ccfs(tdir / str(chunk_idx), sp_ccfs.data)
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


def _open_ccfs(path: Path, shape: tuple[int, ...], mode: Literal["r", "r+"]):
    # Load the memory-mapped files
    ccfs = np.memmap(path / "ccfs.npy", dtype=CCF_TYPE, mode=mode)
    offsets = np.memmap(path / "offsets.npy", dtype=np.int64, mode=mode)

    sp_genos = SparseCCFs.from_offsets(ccfs, shape, offsets)
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


def _write_ccfs(path: Path, ccfs: NDArray[CCF_TYPE]):
    path.mkdir(parents=True, exist_ok=True)

    ccfs_memmap = np.memmap(
        path / "ccfs.npy",
        shape=ccfs.shape,
        dtype=ccfs.dtype,
        mode="w+",
    )
    ccfs_memmap[:] = ccfs
    ccfs_memmap.flush()


def _concat_data(
    out_path: Path, chunks_path: Path, shape: tuple[int, int], with_ccfs: bool = False
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

    if with_ccfs:
        ls_: list[SparseCCFs] = []
        for chunk_dir in chunk_dirs:
            sp_ccfs = _open_ccfs(chunk_dir, shape, mode="r")
            vars_per_sp += sp_ccfs.lengths
            ls_.append(sp_ccfs)
        ccfs_memmap = np.memmap(
            out_path / "ccfs.npy", dtype=CCF_TYPE, mode="w+", shape=offsets[-1]
        )
        _concat_helper(
            ccfs_memmap,
            offsets,
            [a.data for a in ls_],
            [a.offsets for a in ls_],
            shape,
        )
        ccfs_memmap.flush()


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
    out_offsets = np.empty((n_ranges, n_samples, ploidy, 2), dtype=OFFSET_TYPE)

    for r in nb.prange(n_ranges):
        for s in nb.prange(n_samples):
            for p in nb.prange(ploidy):
                if var_ranges[r, 0] == var_ranges[r, 1]:
                    out_offsets[r, s, p] = np.iinfo(OFFSET_TYPE).max
                    continue

                s_idx = sample_idxs[s]
                sp = s_idx * ploidy + p
                o_s, o_e = geno_offsets[sp], geno_offsets[sp + 1]
                sp_genos = genos[o_s:o_e]
                # add o_s to make indices relative to whole array
                out_offsets[r, s, p] = np.searchsorted(sp_genos, var_ranges[r]) + o_s

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
        Shape: (ranges samples ploidy 2). The first column is the start index of the variant
        and the second column is the end index of the variant.
    """
    n_ranges = len(q_starts)
    n_samples = len(sample_idxs)
    if out is None:
        out = np.empty((n_ranges, n_samples, ploidy, 2), dtype=OFFSET_TYPE)

    for r in nb.prange(n_ranges):
        for s in nb.prange(n_samples):
            for p in nb.prange(ploidy):
                if var_ranges[r, 0] == var_ranges[r, 1]:
                    out[r, s, p] = np.iinfo(OFFSET_TYPE).max
                    continue

                s_idx = sample_idxs[s]
                sp = s_idx * ploidy + p
                o_s, o_e = geno_offsets[sp], geno_offsets[sp + 1]
                sp_genos = genos[o_s:o_e]
                start_idx, max_idx = np.searchsorted(
                    sp_genos, [var_ranges[r, 0], contig_max_idx + 1]
                )

                # add o_s to make indices relative to whole array
                out[r, s, p, 0] = start_idx + o_s
                if start_idx == max_idx:
                    # no variants in this range
                    out[r, s, p, 1] = start_idx + o_s
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
                out[r, s, p, 1] = geno_idx + o_s + 1

    return out
