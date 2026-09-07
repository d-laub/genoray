# Reader-frontier wide-cohort repro — issue #169 (UNMEASURED)

**No run has been performed.** The numbers in the table below are absent, not
pending — every measurement cell reads literally `NOT MEASURED`. This document
exists only to fix the plan and the table shape so a run can be dropped in
without redesigning either.

## Why nothing was run

The coordinator for this task ruled out submitting the `sbatch` job in this
session:

- `sinfo` showed all of `carter-cn-[02-04]` in `mixed` state, and
  `squeue -u dlaub` showed nine of the user's own jobs already running
  (eight `nf-ASCA_*` plus `atac_pool`), three of them started within the
  previous 15 minutes — an actively-submitting production Nextflow pipeline.
  `frontier.sbatch` requests `--exclusive` for up to 8 hours on a pinned node,
  which would hold a whole node against that live pipeline and would itself
  sit `PENDING` until a node drained (the longest of those jobs were 19-20h
  in), producing no result in any case.
- Dropping `--exclusive` is not an acceptable substitute. Shared-node timing
  rows on this cluster have previously run 1.35–83x slow and produced three
  published-then-retracted findings (see
  `svar2-scale-ladder-rows-contaminated`); the same job has measured 151.9s on
  one node vs. 73.2s on another. An unpinned or shared-node timing claim is
  inadmissible for a task whose entire deliverable is a timing comparison.
- The corpus itself (500,000 samples × 200,000 variants, ~1e11 genotype
  cells, hundreds of GB uncompressed) was also not generated, for the same
  reason: it is not something to build on a login node, and building it
  without an intent to immediately run the sweep on a pinned exclusive
  allocation would just leave a large orphaned artifact on `/local`.

`frontier.sbatch` and `frontier_points.py` are written and verified to parse
(see the Task 8 report for the exact verification commands and output). They
have not been executed.

## Reproducing the run

Pick `<node>` from `sinfo` — a node that is `idle` or that will drain soon
enough to actually start the job — and record it in the row it produces.
Run both arms against the **same** node:

```bash
# From a worktree checked out at main:
sbatch --nodelist=<node> --export=ALL,ARM=main scripts/bench_svar2/frontier.sbatch

# From this branch (issue #169 / worktree-issue-169-reader-frontier):
sbatch --nodelist=<node> --export=ALL,ARM=branch scripts/bench_svar2/frontier.sbatch
```

`maturin develop --release` must be run in each arm's own environment before
its sweep starts (the sbatch script assumes each arm's tree already has an
up-to-date build, or builds one as part of its own setup) — otherwise both
arms load one `.so` and the `code_id` column below will be identical, which
invalidates the comparison per the brief's own trap list.

## Results

2 arms × 3 worker counts (`reader_workers` = 3, 8, 20), one contig
(`concurrent_chroms=1`), `chunk_size=5000`, `threads=32`.

| arm | reader_workers | node | git rev | code_id | wall_s | peak_rss_mb | pending_highwater | per-contig span (done: Xs) | store digest |
|---|---|---|---|---|---|---|---|---|---|
| main | 3 | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| main | 8 | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| main | 20 | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| branch | 3 | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| branch | 8 | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| branch | 20 | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |

Acceptance for a real run: the two arms' `code_id` values must differ (proof
that `main` and this branch produced different `.so` artifacts, not the same
one served twice via a shared `CARGO_TARGET_DIR`); the store digest must be
identical across all six rows (proof the sharding change is byte-identical
at this scale, not just at the corpora `tests/test_svar2_schedule_invariance.py`
already covers); `git rev` must differ between the `main` and `branch` rows.

## Which constants the data constrains

No data was collected, so the honest answer is: **none of them**. All four
remain at their originally-chosen values with no measurement behind them —

- `W_TARGET` (`src/budget.rs`, currently `8`)
- `MERGE_RESERVE_DIV` (`src/budget.rs`, currently `4`)
- `UNITS_TARGET_CHUNKS` (`src/shard.rs`, currently `4`)
- `PENDING_BUDGET_CHUNKS` (`src/budget.rs`, currently `8`)

None of their doc comments have been edited to claim a measurement this run
did not make. Whoever executes the reproduction steps above should fill in
the table, then revisit this section with whatever the six rows actually
show — in particular whether `reader_workers=8` (the `W_TARGET` default)
sits near the wall-time knee between 3 and 20 on the branch arm, and whether
`pending_highwater` on the branch's `w=20` row stays under
`PENDING_BUDGET_CHUNKS` as designed rather than saturating it.
