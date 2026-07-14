"""SVAR2 ``reroute`` measurement spike (concat/split/write_view plan, Task 1).

Decides whether the *separate* ``reroute=False`` (source-representation-preserving)
slice path for ``SparseVar2.write_view`` is ever worth building, or whether
``reroute=True`` (re-run the dense/var_key cost model on the subset) is a
permanent sole mode.

The question is empirical: when you take a **sample subset** of a finished
store, does re-running the cost model pick a *different* representation than the
source store used (a "flip"), and does preserving the source representation
instead cost materially more bytes? This script answers it with an **analytic
recount** — no store is rewritten. For each subset it reads, per variant:

* ``src_dense``  — the source store's routing (which physical stream the variant
  lives in), and
* ``x_sub``      — how many subset haplotypes carry it,

via the ``_core.svar2_variant_stats`` helper (a no-gather CSR walk + dense
popcount over the finished sidecars; see ``ContigReader::variant_stats``). It
then recomputes each variant's subset-optimal representation from the **exact**
integer cost model (reproduced below and asserted against the documented
crossovers) and tabulates flip-% and the on-disk size delta.

Run (after the two stores exist — see the companion report for the build
commands)::

    pixi run -e py310 python scripts/svar2_reroute_spike.py \
        --germline data/chr21.germline.svar2 \
        --somatic  data/gdc.chr21.somatic.svar2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import genoray._core as _core

# --- exact cost model (mirrors src/cost_model.rs; integer, no floats) ---------
# var_key = x * (POS_BITS + key_bits + sidecar + info + format)
# dense   = POS_BITS + key_bits + n_haps + sidecar + info + format * n_samples
# Route to Dense iff strictly cheaper; ties break to var_key.
# The spike builds no-field, no-signature stores, so sidecar = info = format = 0.
POS_BITS = 32
SNP_KEY_BITS = 2
INDEL_KEY_BITS = 32


def key_bits(is_indel: np.ndarray) -> np.ndarray:
    return np.where(is_indel, INDEL_KEY_BITS, SNP_KEY_BITS).astype(np.int64)


def dense_bits(is_indel: np.ndarray, n_samples: int, ploidy: int) -> np.ndarray:
    """On-disk bits if a variant is stored DENSE for an ``n_samples`` cohort."""
    return POS_BITS + key_bits(is_indel) + np.int64(n_samples * ploidy)


def var_key_bits(is_indel: np.ndarray, x: np.ndarray) -> np.ndarray:
    """On-disk bits if a variant is stored VAR_KEY with ``x`` carrier calls."""
    return x.astype(np.int64) * (POS_BITS + key_bits(is_indel))


def routes_dense(
    is_indel: np.ndarray, n_samples: int, ploidy: int, x: np.ndarray
) -> np.ndarray:
    """Vectorized ``choose_representation`` == Dense (strict-cheaper, tie→var_key)."""
    return dense_bits(is_indel, n_samples, ploidy) < var_key_bits(is_indel, x)


def _assert_crossovers() -> None:
    """Pin the Python recount to the Rust crossovers documented in cost_model.rs
    (np = 2000: SNP dense at x >= 60, indel dense at x >= 33)."""
    snp = np.array([False])
    ind = np.array([True])
    assert not routes_dense(snp, 1000, 2, np.array([59]))[0]
    assert routes_dense(snp, 1000, 2, np.array([60]))[0]
    assert not routes_dense(ind, 1000, 2, np.array([32]))[0]
    assert routes_dense(ind, 1000, 2, np.array([33]))[0]


def _meta(store: Path) -> dict:
    return json.loads((store / "meta.json").read_text())


def analyze(store: Path, label: str, sizes: list[int], seed: int = 0) -> list[dict]:
    meta = _meta(store)
    samples = meta["samples"]
    ploidy = int(meta["ploidy"])
    contigs = meta["contigs"]
    n_full = len(samples)
    rng = np.random.default_rng(seed)

    rows: list[dict] = []
    # `n_full` (all samples) is the control: reroute on the full cohort must
    # reproduce the source routing exactly (0 flips, 0 size delta).
    for k in [*sizes, n_full]:
        if k > n_full:
            continue
        subset = (
            np.sort(rng.choice(n_full, size=k, replace=False))
            if k < n_full
            else np.arange(n_full)
        )

        # accumulate across every contig in the store (chr21-only here)
        is_indel_all, src_dense_all, x_full_all, x_sub_all = [], [], [], []
        for chrom in contigs:
            ii, sd, xf, xs = _core.svar2_variant_stats(
                str(store), chrom, subset.tolist()
            )
            is_indel_all.append(ii.astype(bool))
            src_dense_all.append(sd.astype(bool))
            x_full_all.append(xf.astype(np.int64))
            x_sub_all.append(xs.astype(np.int64))
        is_indel = np.concatenate(is_indel_all)
        src_dense = np.concatenate(src_dense_all)
        x_full = np.concatenate(x_full_all)
        x_sub = np.concatenate(x_sub_all)

        # Sanity: the source routing must match choose_representation at the FULL
        # cohort with the full carrier counts (holds iff no fields/signatures).
        src_recomputed = routes_dense(is_indel, n_full, ploidy, x_full)
        src_mismatch = int((src_recomputed != src_dense).sum())

        # A variant is in the VIEW iff >=1 subset hap carries it (MAC>0). MAC=0
        # variants are dropped by both reroute and preserve-source, so exclude.
        present = x_sub > 0
        ii_p = is_indel[present]
        sd_p = src_dense[present]
        xs_p = x_sub[present]
        n_present = int(present.sum())

        reroute_dense = routes_dense(ii_p, k, ploidy, xs_p)
        flips = int((reroute_dense != sd_p).sum())

        d_bits = dense_bits(ii_p, k, ploidy)
        vk_bits = var_key_bits(ii_p, xs_p)
        size_reroute = int(np.where(reroute_dense, d_bits, vk_bits).sum())
        size_preserve = int(np.where(sd_p, d_bits, vk_bits).sum())

        rows.append(
            {
                "store": label,
                "k": k,
                "is_control": k == n_full,
                "n_present": n_present,
                "flips": flips,
                "flip_pct": 100.0 * flips / n_present if n_present else 0.0,
                "size_reroute_bits": size_reroute,
                "size_preserve_bits": size_preserve,
                "size_delta_pct": (100.0 * (size_preserve / size_reroute - 1.0))
                if size_reroute
                else 0.0,
                "src_mismatch": src_mismatch,
                "n_variants_total": int(is_indel.size),
            }
        )
        print(
            f"[{label}] k={k:>6} present={n_present:>9} flips={flips:>9} "
            f"({rows[-1]['flip_pct']:6.2f}%)  size_delta={rows[-1]['size_delta_pct']:+7.3f}%  "
            f"(src_mismatch={src_mismatch})"
        )
    return rows


def _table(rows: list[dict]) -> str:
    hdr = "| store | subset k | variants in view | flips | flip % | size delta % (preserve vs reroute) |"
    sep = "|---|---|---|---|---|---|"
    lines = [hdr, sep]
    for r in rows:
        lines.append(
            f"| {r['store']} | {r['k']} | {r['n_present']:,} | {r['flips']:,} | "
            f"{r['flip_pct']:.2f}% | {r['size_delta_pct']:+.3f}% |"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--germline", type=Path, default=None)
    ap.add_argument("--somatic", type=Path, default=None)
    ap.add_argument("--sizes", type=int, nargs="+", default=[10, 50, 100, 500])
    ap.add_argument("--json-out", type=Path, default=None, help="write rows as JSON")
    args = ap.parse_args()
    if args.germline is None and args.somatic is None:
        ap.error("pass at least one of --germline / --somatic")

    _assert_crossovers()
    print("cost-model crossovers verified against cost_model.rs\n")

    rows: list[dict] = []
    if args.germline is not None:
        rows += analyze(args.germline, "germline", args.sizes)
    if args.somatic is not None:
        rows += analyze(args.somatic, "somatic", args.sizes)

    print("\n" + _table(rows))

    subset_rows = [r for r in rows if not r["is_control"]]
    max_flip = max((r["flip_pct"] for r in subset_rows), default=0.0)
    max_delta = max((abs(r["size_delta_pct"]) for r in subset_rows), default=0.0)
    ctrl_ok = all(
        r["flips"] == 0 and r["src_mismatch"] == 0 for r in rows if r["is_control"]
    )
    print(f"\nmax subset flip% = {max_flip:.2f}%   max |size delta| = {max_delta:.3f}%")
    print(f"full-cohort control clean (0 flips, 0 mismatch): {ctrl_ok}")

    if args.json_out is not None:
        args.json_out.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
