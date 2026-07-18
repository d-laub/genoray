# FORMAT route-before-densify — measured result

Follow-up to `2026-07-16-svar2-from-vcf-list-memory-baseline.md`, whose Section 6
recorded the cpu_s-vs-N exponent gate as **MISSED**: "Design A" delivered a real
~2.5x constant-factor CPU win and a 6.72x churn reduction but left the O(N^2)
asymptote in place (exponent 1.756 -> 1.747, unmoved). Its Section 2 root-cause
named the residual: `chunk_assembler` still densified FORMAT to an
`F x chunk_size x N` grid (`format_staged`) that `rvk` then read back only at
carrier positions — the same round-trip the project had already eliminated for
genotypes, left in place for FORMAT.

This change carries FORMAT values carrier-sparsely into `rvk` (mirroring how the
prior branch carried genotype `Carriers`) instead of densifying them:
`DenseChunk.format_by_carrier`, `rvk` resolves FORMAT per carrier via
`resolve_format`, and `chunk_assembler` skips building `format_staged` entirely
for carrier-bearing chunks. Commits `c70d64e..240c797` (+ the query-core build
fix `e0d3875`).

## Result — the O(N^2) is eliminated

N-sweep, `cpu_s` (user+sys CPU — the load-robust, thread-count-robust metric the
prior baseline used), same `cohort_sweep` (2000 synthetic single-sample VCFs,
one contig), F=2 (VAF float + DP int), `threads=1`, at HEAD `240c797`:

| N | cpu_s | MaxRSS (GB) |
|---|---|---|
| 250 | 21.5 | 1.10 |
| 500 | 36.8 | 1.67 |
| 1000 | 67.3 | 3.07 |
| 2000 | 137.7 | 5.73 |

(N=250 reproduced at 19.8s against the confirmed-fresh release build.)

**cpu_s exponent (log-log fit):**

| build | exponent | verdict |
|---|---|---|
| pre-fix baseline | 1.756 | — |
| after "Design A" | 1.747 | did not move |
| **after FORMAT route-before-densify** | **0.890** | **linear** |

The cleanest, build-profile-independent view is the per-doubling cpu_s ratio
(2x N -> 2x time is linear; -> 4x is quadratic):

| build | 250->500 | 500->1000 | 1000->2000 |
|---|---|---|---|
| after "Design A" | 3.19x | 3.53x | 3.48x |
| after FORMAT route-before-densify | 1.71x | 1.83x | 2.05x |

Design A's own ratios (~3.5x per doubling) are the fingerprint of `~N^1.8`;
this change's ratios (~2x per doubling) are the fingerprint of `~N^1.0`. The
slope is the algorithmic signature and is invariant to the constant-factor
speedup a build profile applies, so this is a valid apples-to-apples comparison
even though the absolute times below use a release build.

## Rigor caveats (do not over-claim)

- **Absolute times are NOT comparable to the prior baselines.** The prior
  baseline/after CSVs were measured with a debug extension (a 79 MB `.so`); the
  numbers above are a release build. Release changes the *constant*, not the
  *slope* — so the exponent / per-doubling ratio is the valid cross-profile
  comparison (each is self-consistent within its own build), and no absolute
  "Nx faster" number is claimed here.
- **Peak RAM still grows with N.** MaxRSS exponent is ~0.80 (Design A: 0.783) —
  essentially unchanged. The plan's aspirational "peak RAM independent of cohort
  size" is **NOT achieved**: the union variant count V grows with N in the
  somatic model, so total work and peak both retain an N-dependence that
  route-before-densify does not remove. At F=2 the FORMAT grid was small, so the
  peak-RAM contribution this change removes is modest; the ~10 GB/chunk figure
  the design targeted was an F=7 concern. The CPU-exponent collapse is the win.

## Correctness

- Store is **byte-identical**: Rust differential test strengthened to a real
  cross-encoding parity (carrier-encoded chunk vs dense-grid sibling), plus a
  Python cross-path oracle (`from_vcf` dense grid vs `from_vcf_list` carrier,
  over a `bcftools merge`, 2 contigs, DP+VAF) — DP and VAF match exactly.
- Rust: 283 lib unittests green; `check-core` (`--no-default-features`
  query-core build) clean; clippy `-D warnings` / `cargo fmt --check` enforced.
- Python: 735 passed / 2 skipped / 16 xfailed (re-run of the from_vcf_list +
  parity subset confirmed green against a freshly-rebuilt release extension).

## Not run (optional, non-blocking per the plan's Task 6)

dhat churn regression (Step 1), the perf top-symbol check that `dense2sparse_vk`
leaves the profile (Step 4), and the full 7089-file Hartwig cohort at F=7
(Step 6) were not run this session. The exponent gate (Step 2) — the one the
prior branch missed and this change targets — is **MET**.

## Summary vs the prior branch's scorecard

| gate | Design A | this change |
|---|---|---|
| cpu_s exponent (~N^1.8 -> ~N^1.0) | **MISSED** (1.747) | **MET** (0.890) |
| byte-identical store output | MET | MET |
| peak RSS independent of N | NOT ACHIEVED | NOT ACHIEVED (~N^0.80) |

Issue #120's scaling wall — the O(N^2) merge cost that opened the project — is
closed on the CPU axis: the from_vcf_list merge is now linear in the number of
input files.
