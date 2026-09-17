# SPA-faithful refit for `fit_signatures`

Design for issue #214, items 1-7. Follows #212 (`criterion="bic"`), which is
merged.

## Problem

`genoray.fit_signatures` is greedy forward selection from the empty set.
SigProfilerAssignment's `cosmic_fit` is backward elimination from the saturated
set, with a refinement pass, a different removal criterion, a different score,
two priors genoray lacks, and burden-conserving integer activities.

Forward selection from empty and backward elimination from saturated are
different algorithms with different fixed points. Forward selection is myopic:
it can never recover a pair of signatures that only helps jointly. On synthetic
mixtures of four real COSMIC v3.4 SBS96 signatures (20 reps), backward
elimination recovers more true signatures at every burden tested:

| burden | scenario | forward + cosine | backward + L2 |
|---|---|---|---|
| 100,000 | distinct | 3.50 | **3.85** |
| 100,000 | with flat (SBS5/SBS40a) | 2.75 | **3.25** |
| 1,000 | distinct | 3.30 | **3.60** |

Backward elimination also has better precision at low burden: 0.20 vs 0.55 false
positives at 100 mutations.

This design adds the SPA algorithm as a second strategy. It does not change the
default. Neither estimator is consistent — both scores are scale-invariant, so
both plateau below 4.00 as burden grows. Only #212's `criterion="bic"` reaches
4.00. Adopting SPA's search direction improves recovery; it does not fix
consistency, and the docs must not imply otherwise.

## Scope

In scope: issue #214 items 1-7.

Out of scope, deliberately:

- **Item 8** (exome-renormalized COSMIC signatures). A real correctness gap for
  WES and panel callers, but independent of the fitting algorithm. Gets its own
  issue and PR.
- **Item 9** (SPA behaviours not worth copying). See "Divergences from SPA".

## API

Two frozen dataclasses select and parameterize the algorithm. Each carries only
the knobs its own algorithm uses, so passing a forward threshold to backward
elimination is a type error rather than a silently ignored keyword argument.
This follows the lesson recorded in #200 and #207: a parameter one caller
silently drops is a semantic trap, and the fix is to make the drop
unrepresentable, or failing that, loud.

```python
Criterion = Literal["cosine", "bic"]        # existing, from #212
Metric = Literal["l2", "cosine"]            # new
ActivityScale = Literal["burden", "raw"]    # new


@dataclass(frozen=True)
class Forward:
    """Greedy forward selection from the empty set. genoray's own algorithm."""
    max_delta: float = 0.01
    min_activity: float = 0.005
    criterion: Criterion = "cosine"


@dataclass(frozen=True)
class Spa:
    """Backward elimination from the saturated set. SigProfilerAssignment's."""
    metric: Metric = "l2"
    initial_remove_penalty: float = 0.05
    add_penalty: float = 0.05
    remove_penalty: float = 0.01
    background_sigs: Sequence[str] | None = ("SBS1", "SBS5")
    connected_sigs: bool | Sequence[Sequence[str]] = True
    activity_scale: ActivityScale = "burden"


Strategy = Forward | Spa
```

Every `Spa` default is SigProfilerAssignment's own default, so `Spa()` is
`cosmic_fit` as shipped.

`fit_signatures` gains one keyword argument:

```python
def fit_signatures(
    catalogue: pl.DataFrame,
    reference: pl.DataFrame,
    *,
    strategy: Strategy | None = None,
    max_delta: float = ...,
    min_activity: float = ...,
    criterion: Criterion = ...,
    n_jobs: int = 1,
    backend: str = "loky",
) -> pl.DataFrame: ...
```

### Resolving `strategy` against the legacy keyword arguments

`max_delta`, `min_activity` and `criterion` are kept permanently as shorthand
for the forward path. They are not deprecated. Resolution:

1. `strategy is None` and no legacy argument passed -> `Forward()`.
2. `strategy is None` and one or more legacy arguments passed ->
   `Forward(max_delta=..., min_activity=..., criterion=...)` built from them.
3. `strategy` given and no legacy argument passed -> use `strategy`.
4. `strategy` given **and** any legacy argument explicitly passed -> raise
   `ValueError` naming the conflicting arguments.

Case 4 needs to distinguish "explicitly passed the default value" from "not
passed". Use a module-private sentinel as the declared default:

```python
_UNSET: Any = object()

def fit_signatures(..., max_delta: float = _UNSET, ...):
```

The sentinel is replaced by the real default inside the function. The public
signature in the docstring documents the real defaults (0.01, 0.005,
`"cosine"`), not the sentinel.

`Forward` and `Spa` are exported from the top-level `genoray` namespace via the
existing `_LAZY` table in `python/genoray/__init__.py`, alongside `Criterion`.
`Metric`, `ActivityScale` and `Strategy` are exported too — a user annotating a
variable that holds a strategy needs `Strategy`.

### Naming

`Forward` and `Spa` are terse. They are kept because the alternative
(`ForwardSelection`, `SpaFit`) buys little at the call site, which reads
`strategy=Spa()` either way, and the class docstrings carry the meaning. `Spa`
is the upstream tool's own abbreviation and appears already in this repo as the
`sigprofiler` pixi feature and the `pytest.mark.sigprofiler` marker.

### Downstream plumbing

`SparseVar.assign_signatures` and `SparseVar2.assign_signatures` both forward
`max_delta`/`min_activity`/`n_jobs`/`backend` to `fit_signatures` today. Both
gain `strategy: Strategy | None = None`, forwarded unchanged, with the same
conflict error raised by `fit_signatures` itself rather than re-implemented.

Note neither `assign_signatures` forwards `criterion` today — #212 added it to
`fit_signatures` only. Adding `criterion` to both is in scope as a Boy Scout
fix: it is the same one-line passthrough, and leaving one of the two stop rules
unreachable from the two highest-level entry points is an inconsistency a user
will hit. With `strategy=` added alongside it, a caller who wants BIC can
already write `strategy=Forward(criterion="bic")`, but the flat shorthand should
be consistent across all three functions rather than present on one.

## Algorithm

Read off SigProfilerAssignment `main`: `single_sample.py`
(`fit_signatures`, `add_signatures`, `remove_all_single_signatures`,
`add_remove_signatures`, `add_connected_sigs`, `roundConserveSum`) and
`decompose_subroutines.py` (`process_sample`). The `solver="nnls"`,
`pcawg_rule=False` path is the one reproduced; SPA's `slsqp` branch is not.

Notation: `W` is the (n_types, n_sigs) column-normalized reference, `m` the
sample's count vector, `A` the active index set.

Relative L2 distance, the score throughout:

```
d(A) = ||m - W[:, A] @ nnls(W[:, A], m)||_2 / ||m||_2
```

Lower is better. With `metric="cosine"` the score is `1 - cos_sim(m, recon)`
instead; also lower-is-better, which keeps every comparison below identical.

### Stage 1: saturate

One NNLS over every reference signature, plus `background_sigs` (already
included, since every signature is active). This is SPA's
`exposureAvg_dummy = np.random.rand(n_sigs, n_samples)` passed as
`denovo_exposureAvg` with `cosmic_sigs=True`: every signature starts nonzero.

genoray does not replicate the random draw. Its only effect in SPA is to make
every entry nonzero, which a vector of ones does deterministically. See
"Divergences from SPA".

### Stage 2: initial prune

`remove_all_single_signatures(cutoff=initial_remove_penalty)` with **no**
protected signatures — SPA passes `background_sigs=[]` at this stage, so SBS1
and SBS5 are removable here and protected only later.

```
loop:
    base = d(A)
    P' = remap(P)                       # see "Protection decays", below
    for each position p, signature i in A (skipping p in P'):
        cand = A - {i}
        record the i minimizing d(cand) - base
    if that minimum > cutoff: stop
    if the winning candidate has exactly one nonzero: accept it and stop
    A = nonzeros of the winning exposure vector; base = d(A)
    P' = remap(P')                      # against the winning vector
```

The cutoff is tested against the degradation relative to the **current** fit,
not the saturated one, because SPA reassigns `originalSimilarity = record[2]`
after each accepted layer. Getting this wrong makes the prune far too
permissive.

This is item 3: SPA removes the signature whose removal degrades the fit least,
while that degradation stays under a threshold. It has no `min_activity`
analogue and will drop a high-activity signature that is collinear with others.
`Spa` therefore has no `min_activity` field.

#### Protection decays, and that is reproduced

`remove_all_single_signatures` loops over the **compacted** index space of the
exposure vector's nonzeros, so it remaps its protected set into that space
before each pass, via `get_changed_background_sig_idx`. Dropping a protected
signature whose exposure is zero is deliberate — NNLS has already removed it.
Feeding the compacted result back in as if it were a full-length index on the
next call is not. Concretely, for COSMIC SBS96 with `background_sigs =
("SBS1","SBS5")` (full indices 0 and 4):

```
pass 1 top:    [0, 4] -> [0, 2]   correct: SBS1 and SBS5 are active
pass 1 bottom: [0, 2] -> [0]      2 is read as a full index (SBS3), value 0, dropped
pass 2 onward: [0]    -> [0]      only SBS1 survives, and only because it is column 0
```

So protection holds for the first removal decision of a sweep and then
survives only for signatures whose compacted position still equals their full
index. A background signature the sample does not need is removed.

This was measured against real `cosmic_fit` 1.1.4, not inferred. Upstream's own
`ss.add_remove_signatures`, called on the same `W` and sample, feeds
`{SBS1, SBS2, SBS5, SBS13, SBS17b}` into the sweep with
`background_sigs=[SBS1, SBS5]` and gets `{SBS1, SBS2, SBS17b}` back. Treating
`background_sigs` as an absolute veto instead retains SBS5 on samples where
`cosmic_fit` reports it as zero, and drags every other activity down with it
(SBS7a off by ~900 of 59,000 on the `distinct_high` calibration cell). All
three cells in `tests/test_signatures_calibration.py` agree with `cosmic_fit`
to the mutation once this is copied.

`Spa`'s docstring says so, because `background_sigs` does not mean what its
name suggests.

### Stage 3: refinement layers

`add_remove_signatures`. `A` enters as stage 2's survivors union
`background_sigs`; `P` is the protected set (`background_sigs` resolved to
indices, or empty).

```
best = +inf
loop over layers:
    present = A, expanded by connected_sigs   # NOT A union {c} -- c joins after expansion
    layer_best = +inf
    for each candidate c not in present:
        A_add = add_signatures(present=present, candidate=c, cutoff=add_penalty)
        A_rem = remove_all_single_signatures(A_add, cutoff=remove_penalty,
                                             protected=P)
        pick = A_add if A_rem did not change the support else A_rem
        if d(pick) < layer_best: layer_best = d(pick); layer_pick = pick
    if layer_best >= best: stop, keep the previous layer's result
    best = layer_best; A = layer_pick
```

SPA's own version of the `pick` line is
`np.nonzero(add)[0].all() == np.nonzero(remove)[0].all() and shapes equal`,
which looks like a bug but is not load-bearing: the removal sweep can only
shrink the support, so equal shapes already imply equal supports, and when the
supports are equal the two distances are equal too. "Did not change the
support" is exactly equivalent, and picking either branch under equality gives
the same answer.

`add_signatures` is SPA's own greedy add loop, but here it is called with
`toBeAdded=[c]`, which restricts its candidate pool to `c` alone. It therefore
adds `c` if and only if doing so improves the distance by strictly more than
`add_penalty`, and adds nothing else. The loop, the rounding and the
`finalRecord` bookkeeping inside it all collapse to:

```
d_base = d(present)                 # +inf when present is empty
d_new  = d(present union {c})
result = present union {c} if d_base - d_new > add_penalty else present
```

genoray implements that directly rather than porting the general loop, which
would be dead generality. The equivalence holds only because `toBeAdded` is a
singleton at every call site in `cosmic_fit`.

This is item 2. Each layer performs a full add-and-remove sweep per candidate,
so a decision made early can be undone later, which is what forward selection
cannot do.

### Rounding drives the support

SPA carries an exposure *vector* between stages, not an index set, and derives
the active set from `np.nonzero(exposures)`. Both `add_signatures` and
`remove_all_single_signatures` round what they record — `np.round` and
`roundConserveSum` respectively, after rescaling to the burden. A signature
whose rescaled activity is under 0.5 therefore rounds to zero and drops out of
the support on its own, with no threshold ever testing it.

This is load-bearing at low burden and invisible at high burden, and it is not
something an index-set implementation reproduces. genoray's internal
representation is therefore the same as SPA's: a full-length `float64` exposure
vector, with `_support(h) = np.nonzero(h)[0]`.

One asymmetry to preserve: distances are computed from the **unrounded** NNLS
reconstruction (`newSample = np.dot(W1, weights)` uses raw weights), while the
recorded exposures are rounded. Scoring off the rounded vector would be a
divergence.

### Stage 4: report

With `activity_scale="burden"` (SPA's behaviour, item 7):

```
h = nnls(W[:, A], m)
if sum(h) == 0: return zeros
h = h / sum(h) * sum(m)
h = round_conserve_sum(h)
```

`round_conserve_sum` is SPA's `roundConserveSum`: ceil every entry, then
decrement the entries with the largest rounding residual until the total equals
`round(sum(m))`. The result is integer-valued `float64` summing exactly to the
rounded burden.

With `activity_scale="raw"`, `h` is returned as raw NNLS weights, matching the
forward path. The forward path is unchanged and still returns raw weights that
need not sum to the burden; its docstring now says so explicitly, which is the
minimum item 7 asked for.

Edge cases: a sample with zero total counts returns zeros and cosine 0.0, as
today. A fit where NNLS returns all zeros returns zeros without rescaling
(avoids a divide by zero SPA does not guard).

### Priors

**`background_sigs`** (item 4). Resolved from signature *names* to column
indices against the reference frame's column names. Names not present in the
reference are ignored silently — SPA's `get_indeces` does the same, and it is
what makes the default `("SBS1", "SBS5")` inert for DBS78 and ID83 without a
special case. `None` disables the prior.

Protected signatures are skipped by the removal loops in stage 3 (`if i in
background_sig: continue`) but **not** in stage 2.

**`connected_sigs`** (item 5). `True` uses SPA's groups:

```
[["SBS2", "SBS13"],
 ["SBS7a", "SBS7b", "SBS7c", "SBS7d"],
 ["SBS10a", "SBS10b"],
 ["SBS17a", "SBS17b"]]
```

Whenever any member of a group is in the active set, every member of that group
is added. `False` disables. A `Sequence[Sequence[str]]` supplies custom groups,
which is what makes the feature usable for DBS and ID catalogues, where SPA's
SBS-only groups are inert.

## Divergences from SPA

These are deliberate. Each is documented in the `Spa` docstring.

### The negative-difference guard is not copied

`remove_all_single_signatures` contains:

```python
difference = newSimilarity - originalSimilarity
if difference < 0:
    difference = cutoff + 1e-100
```

If removing a signature *improves* the fit, SPA refuses to remove it.

On the default `metric="l2"` path this branch is unreachable. NNLS over a column
subset can never achieve a smaller residual than NNLS over the superset, and
relative L2 divides by `||m||_2`, which is constant across candidates. So
`newSimilarity >= originalSimilarity` always, up to floating-point noise, and
omitting the guard cannot change a `Spa()` result.

Under `metric="cosine"` the branch *is* reachable: cosine similarity is
scale-invariant, so it is not monotone in the residual norm, and removing a
column can raise it. `Spa(metric="cosine")` therefore diverges from SPA. The
docstring says so, and a test pins the difference rather than leaving it
undiscovered.

### RNG consumption is not copied

SPA computes `x0 = np.random.rand(...)` in three places on the `nnls` path where
it is never used, and draws `exposureAvg_dummy` from the global RNG. genoray's
implementation touches no RNG. A test asserts two `Spa()` fits of the same input
are bit-identical, which SPA cannot guarantee for its own callers.

### `check_rule_negatives` is not implemented

SPA's `check_rule_negatives=[1, 16]` with a 1.5x penalty applies only to mm9,
mm10 and mm39. genoray ships no mouse reference signatures, so the parameter has
nothing to act on. Not exposed.

### The protected-set remap matches by index, not by exposure value

`get_changed_background_sig_idx` finds a protected signature's new position by
looking its exposure *value* up in the list of nonzero exposures
(`list.index`). genoray's `_protected_positions` uses the index directly. The
two differ only when two active signatures carry exactly the same exposure, in
which case SPA takes whichever appears first — arbitrary, and it would make the
result depend on rounding collisions. The decay itself is reproduced faithfully
(see "Protection decays"); only the tie-breaking is not.

### Reference columns are renormalized

`fit_signatures` scales each reference column to sum 1; SPA does not do this for
a custom `signature_database`. This is a pre-existing robustness improvement in
genoray, retained on both paths, and it means activities are in mutation-count
units regardless of how the reference was scaled.

## Output schema

Unchanged: `Sample`, one `Float64` column per reference signature, and a
trailing `cosine_similarity`.

SPA also reports L2 error %, and adding a trailing `l2_relative_error` column
was considered. It is **not** added: the schema is documented in
`skills/genoray-api/SKILL.md` and consumed downstream, and a second trailing
non-signature column breaks any caller slicing signature names positionally.
The cost of the break outweighs the convenience. Reversible later if wanted —
adding a column is a smaller decision than removing one.

Under `activity_scale="burden"` the signature columns hold integer-valued
`float64`. The dtype does not change.

## Module structure

`python/genoray/_signatures.py` is 360 lines after #212 and holds two unrelated
concerns already (the estimator, and a pooch-backed COSMIC download registry).
The SPA path adds roughly 250 lines and a third. Split into a package:

```
python/genoray/_signatures/__init__.py   re-exports only; no logic
python/genoray/_signatures/_fit.py       fit_signatures driver, name->index
                                         resolution, joblib fan-out
python/genoray/_signatures/_strategy.py  Forward, Spa, the Literal aliases,
                                         strategy resolution
python/genoray/_signatures/_common.py    _nnls, _cosine, _rel_l2, _poisson_ll,
                                         _round_conserve_sum
python/genoray/_signatures/_forward.py   forward selection + BIC (moved verbatim)
python/genoray/_signatures/_spa.py       stages 1-4, priors
python/genoray/_signatures/_cosmic.py    _COSMIC_REGISTRY, _load_signature_file,
                                         cosmic_signatures
```

`__init__.py` re-exports every name that exists today at
`genoray._signatures.<name>`. Known private importers in this repo:

- `python/genoray/_svar/_annotate.py` — `_load_signature_file`,
  `cosmic_signatures`, `fit_signatures`
- `python/genoray/_svar2_mutcat.py` — the same three, imported lazily
- `tests/test_signatures.py`, `tests/test_signatures_criterion.py` —
  `_fit_one`, `_poisson_ll`, and the module-level helpers

Genoray's private API is also imported across repos — GenVarLoader reaches into
underscore modules — so `from genoray._signatures import X` must keep working
for every X that resolves today. A test enumerates the pre-split module's names
and asserts each still imports.

`_forward.py` receives the existing `_fit_one` unchanged apart from taking a
`Forward` instead of loose keyword arguments, so the diff stays reviewable and
the regression test below is meaningful.

## Parallelism

`fit_signatures` already fans out per sample over `joblib.Parallel` with the
`loky` process backend. The SPA path slots in unchanged: one `_fit_one_spa(W, m,
spec)` call per sample.

The strategy dataclasses are frozen and hold only scalars, strings and tuples,
so they pickle cleanly to workers. `background_sigs` and `connected_sigs` are
resolved from names to indices **once**, in `fit_signatures`, before the fan-out
— resolution needs the reference column names, and doing it per sample would
repeat it n_samples times.

Cost: stage 3 performs, per layer, one add plus a full removal sweep per
candidate signature. With ~86 SBS96 signatures and an active set of ~5-10 that
is on the order of 10^3-10^4 NNLS solves per sample against a 96-by-k matrix.
Expected to land in the 0.1-1 s/sample range, one to two orders of magnitude
slower than the forward path. Measured and recorded in the PR description rather
than guessed; if it lands far outside that range, that is a finding worth
reporting, not silently absorbing.

## Testing

### Calibration against real SigProfilerAssignment

The payoff of faithfulness, and the test that justifies the word. Extend
`tests/test_signatures_calibration.py`:

- Build 3 or more synthetic SBS96 catalogues from known COSMIC v3.4 mixtures,
  spanning burdens (about 1,000 and about 100,000) and including one mixture
  with a flat signature.
- Run `spa.cosmic_fit` and `fit_signatures(..., strategy=Spa())` on the same
  matrix.
- Assert **identical support** (same set of nonzero signatures) and activities
  agreeing to within 1 count per signature, which is the resolution
  `roundConserveSum` leaves.

Marked `sigprofiler` and `network` like the existing test, so it skips by
default. Requires materializing the `sigprofiler` pixi environment, which is
declared in `pixi.toml` but not currently installed.

Exact agreement is the target. If a residual mismatch survives, it must be
explained and written down in this design before the PR merges — an unexplained
divergence means the reimplementation is not understood.

### Recovery

Reproduce the issue's measurement as a regression guard, asserting the
*ordering* rather than the exact means, which are 20-rep Monte Carlo estimates:

- 4 true COSMIC signatures, fixed seeds, at burden 1,000 and 100,000, in both
  the distinct and the with-flat scenario.
- Assert `Spa()` recovers at least as many true signatures as `Forward()` in
  every cell, and strictly more in at least the two 100,000-burden cells.
- Assert `Spa()` has no more false positives than `Forward()` at burden 100.

### Properties

- Removing a column never lowers relative L2 (validates the reachability claim
  behind skipping the negative-difference guard). Randomized over reference
  subsets and sample vectors.
- `round_conserve_sum(h).sum() == round(h.sum())` and all entries are
  non-negative integers, over randomized `h`.
- `activity_scale="burden"` output sums to `round(sum(m))` exactly, for every
  sample in a randomized catalogue.
- Two `Spa()` fits of the same input are bit-identical.

### Regression

The forward path must be untouched. Compare `fit_signatures` before and after
across randomized references, burdens (30 to 300,000) and
`max_delta`/`min_activity`/`criterion` combinations, asserting zero mismatches.
This mirrors the check #212 ran.

### API surface

- `strategy=Spa()` together with an explicit `max_delta=` raises `ValueError`
  naming both.
- `strategy=Spa()` alone, and `max_delta=` alone, both work.
- `Forward` and `Spa` are reachable from `import genoray`.
- Every name importable from `genoray._signatures` today is still importable
  after the package split.
- `background_sigs=("SBS1", "SBS5")` against a DBS78 reference is inert, not an
  error.

## Documentation

Required by `CLAUDE.md`, not optional:

- `skills/genoray-api/SKILL.md`: the public-name list gains `Forward`, `Spa`,
  `Strategy`, `Metric`, `ActivityScale`; the `fit_signatures` and both
  `assign_signatures` entries gain `strategy=`; the note that forward-path
  activities are raw NNLS weights not summing to the burden, and SPA-path
  activities are integer-valued and do sum to it.
- `docs/source/api.md` and `docs/source/index.md`: whichever mention
  `fit_signatures` arguments.
- Module docstring: the "None of that is reproduced here" paragraph added by
  #212 is now wrong. Rewrite it to say which algorithm each strategy runs, and
  keep the standing warning that neither score is burden-aware and only
  `criterion="bic"` is consistent.
- `CHANGELOG.md`: not touched. Commitizen owns it.

## Risks

- **Faithfulness is asserted, not assumed.** The calibration test is the gate.
  If it cannot be made to pass, the honest outcome is to document the residual
  divergence in this file and rename nothing — not to weaken the assertion until
  it goes green.
- **Package split touches a cross-repo private API.** Mitigated by the
  re-export test, but GenVarLoader should be checked against the branch before
  merge.
- **Stage 3 cost.** If it turns out far worse than estimated, the fallback is to
  document the cost and recommend `n_jobs=-1`, not to silently prune the
  candidate loop, which would stop it being SPA.
