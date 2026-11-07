# The directional demand split

**Status:** implemented, off by default. Set `direction_split_model` on a
`propose` or `hindcast` config to switch it on.
**Code:** [`src/orta/direction_split.py`](src/orta/direction_split.py),
wired into `demand.predict_demand`.
**Validation case:** `datasources/cases/07_improved_operator_loading` — case 06
re-run on the same 270 trains with this one setting changed.

---

## 1. The problem in one paragraph

The national ticketing matrix the model runs on carries **no directional
information**. The period matrices inherit that symmetry, so a pair's two
directions are handed identical demand in every slot. That alone would be a
missing effect. The damage is that the next step divides each direction's
demand by *that direction's* train count — which is not symmetric — so the
model ends up predicting the **higher load per train in the direction with
fewer trains**. Reality is the other way round. The result is that the
frequency divisor is worse than useless: it injects a directional signal with
the sign reversed.

---

## 2. The evidence

### 2.1 The matrix is exactly symmetric

`ODM_for_RDM_2024-25.csv`, 1,597,996 directed pairs:

| check | result |
|---|---|
| directed pairs exactly equal to their reverse | **1,597,996 / 1,597,996 (100.0000%)** |
| cells holding odd integers | 54.5% |
| directed total | 1.532bn journeys |
| implied national journeys from table 1410, (entries+exits)/2 | ~1.23bn |

The odd-integer count matters: an odd integer cannot be half of an integer, so
the matrix is **not** a two-way total that has been halved. It is one figure
duplicated into both directions. And the directed total sits at or above the
national journey count, so it is not understated either.

> This closes a false lead worth recording. A median ODM/count ratio of 0.47
> looks like a factor of two — a return journey recorded once against a
> boarding counted twice. It is not. The 0.47 came from comparing an
> end-to-end pair flow against *all* boardings at the origin, which is
> structurally the larger quantity. There is no factor of two. Do not re-chase
> it.

### 2.2 The symmetry survives into the period matrices

`period_matrices.generate_period_matrices` computes

```
trips(o, d) = annual_journeys(o, d) × factor(journey_type, day, period)
```

Under the default `journey_type_mode: "distance"` the journey type is a
distance band, which is identical for `A→B` and `B→A`. Both inputs are
symmetric, so the output is symmetric **by construction**.

### 2.3 The frequency divisor then points the wrong way

`demand.predict_demand` computes `captured = slot demand / trains in
period`. With symmetric demand, the model's implied per-train directional
asymmetry is exactly `−log(trains_fwd / trains_rev)`.

Measured over **419 bidirectional OD × period cells** across four operator feeds
(Northern, Greater Anglia, Southeastern, GWR). GTR and SWR publish XLSX rather
than CSV and were not in this measurement; the shipped fit
(`fit_direction_split.py`) reads all nine case-06 operators through the
standard readers, so its coefficients rest on more data than the numbers below.

| quantity | value |
|---|---|
| sd(log load-per-train asymmetry) — *what a hindcast is scored on* | **0.969** |
| sd(log total-flow asymmetry) | 1.177 |
| sd(log service-count asymmetry) | 0.748 |
| corr(total-flow asymmetry, service-count asymmetry) | **+0.572** |
| corr(per-train asymmetry, service-count asymmetry) | **−0.077** |

Read those last two together. Operators run **more trains** where more people
travel (+0.572), which leaves **load per train roughly equal in both
directions** (−0.077). The model assumes the opposite.

Consequently:

| | sd(log directional error) |
|---|---|
| with the frequency divisor (current) | **1.177** |
| predicting no directional asymmetry at all | **0.969** |

**The divisor makes direction 48% worse in variance than doing nothing.**

### 2.4 The directional error is repeatable, not noise

Splitting the feeds in half by RSP period and re-measuring each corridor's
directional share gives **r = +0.982**. The 50/50 assumption is unbiased *on
average* (median observed share 0.463) but wrong cell by cell: **68% of cells
fall outside 0.40–0.60**, and the p5–p95 range is 0.11–0.86.

---

## 3. What the fix does

For an unordered pair `{o, d}` in a `(day, period)` slot, the two directions
share the pair's **combined** slot demand according to

```
logit(s) = ln(n_fwd / n_rev)  +  a_p  +  b_p · ln(F_o / F_d)
```

| term | meaning | source |
|---|---|---|
| `n_fwd`, `n_rev` | trains serving the pair in the slot, each way | timetable (the *proposed* one, for a forecast) |
| `F_o`, `F_d` | annual entries and exits at each end | table 1410 |
| `a_p`, `b_p` | fitted intercept and slope, per time period | `direction_split.calibrate` |

The forward direction takes share `s` of `trips(o,d) + trips(d,o)`. **Pair
totals are conserved** — this moves trips between directions, it does not
create or destroy any.

### Why the frequency term is not fitted

`ln(n_fwd / n_rev)` enters with a **fixed coefficient of 1**, and it is there
to *cancel the frequency divisor*. With `a_p = b_p = 0` the split is exactly
proportional to train counts, which makes predicted load per train identical in
both directions — the pair's combined slot demand over the combined train
count. The fitted terms then tilt away from that baseline. So the fit's target
is `y = ln(load_per_train / load_per_train_reverse)`: what is left *after* the
cancellation.

### Where it deliberately does nothing

If either direction has **no train in the slot**, the share is exactly 0.5,
which leaves a symmetric matrix untouched. The split was fitted only on pairs
observed running both ways, and handing a one-way slot its whole combined
demand would double that direction's figure in a regime the fit never saw.
Confining the change to the validated regime is intentional.

---

## 4. How well it works

Fitted on half the corridors, scored on **61 corridors never seen in
training**. Target: log per-train directional asymmetry.

| rule | sd(log error) | variance removed |
|---|---|---|
| current: symmetric demand ÷ trains in period | 1.187 | — |
| split in proportion to trains (`a = b = 0`) | 0.968 | **34%** |
| + footfall ratio, pooled | 0.742 | **61%** |
| + footfall ratio, per-period slopes | 0.631 | **72%** |

Those figures come from the exploratory analysis. The **definitive** run is
through the shipped code path (`fit_direction_split.py --holdout`, which scores
`DirectionSplitModel.share` itself), on the full fitting set of **2,073
bidirectional cells over 592 corridors from eight operators**:

| | sd(log per-train direction error) |
|---|---|
| current: symmetric demand ÷ trains in period | 1.378 |
| fitted split | **0.696** |
| **variance removed on 295 unseen corridors** | **74.5%** |

This is the number to quote. It holds on 295 held-out corridors — roughly five
times the corridor count of the exploratory run — and the fitted coefficients
replicate: the pooled footfall slope moves only from 0.250 (four operators) to
0.275 (eight), and the per-period pattern is unchanged, with the slope near
zero in the am_peak and highest in the pm_peak and evening.

> **SWR is not in the fit.** Its feed loses 199 TIPLOCs to missing CRS mappings
> and yields no usable rows through the standard reader. EMR yields rows but no
> bidirectional cells. The fit therefore rests on AWC, Greater Anglia, GTR,
> GWR, Northern, Southeastern and TPE.

This is **out-of-corridor** validation, which is the standard that matters
here. A per-corridor correction factor would score far better in-sample and
transfer to nothing — it would memorise route identity, and the cases this
model exists for are new stations and new links with no observed flow to look
up. This rule has two coefficients per period and both of its inputs are
available for a service that does not yet exist:

- the **proposed timetable** supplies trains by direction and period — you are
  designing it;
- **footfall** is known for existing stations and forecast for new ones, and
  the gravity model's origin elasticity is 0.508, so a ±50% footfall estimate
  error moves demand only +23% / −30%.

### Measured effect on a loading hindcast — and where the claim above does not carry

Case 07 ran the split over eight operators (GTR pending), 7,474 scored
segments, 202 trains. **The 74.5% does not transfer to segment loads, and the
gap is a measurement error on my part, not a bug.**

What did happen:

| | case 06 | case 07 | |
|---|---|---|---|
| correlation r | 0.441 | **0.552** | better |
| RMSE (passengers) | 116.7 | **107.1** | better |
| GEH < 5 % | 54.8 | **56.7** | better |
| MAPE % | 77.0 | **74.7** | better |
| rmse(log bias) | 0.856 | 0.854 | ~unchanged |
| overall ratio | 0.856 | 0.838 | worse |

The split is firing as designed — it changed **95.5%** of positive-demand legs,
median +4%, sd 0.393 in logs, moving 70% of legs by more than 10%. And it
carries real directional information: on bidirectional link × period cells,
`corr(predicted, observed direction asymmetry)` rises from **0.392 to 0.499**,
and the change pushes predictions the right way
(`corr(change, observed asymmetry) = +0.358`).

But the directional *error* barely moves:

| | sd(predicted asymmetry) | corr | sd(directional error) |
|---|---|---|---|
| observed | 1.256 | — | — |
| case 06 | 0.542 | 0.392 | 1.157 |
| case 07 as shipped | 0.871 | 0.499 | 1.116 (**+7.0%** variance) |
| case 07 optimally damped | 0.627 | 0.499 | 1.088 (+11.4% variance) |

Two things went wrong with the earlier claim.

1. **I validated on the wrong quantity.** The 74.5% was measured on per-train
   directional asymmetry of *boardings at a train's origin* — one number per OD
   pair per period. Case 06 scores *on-board load per segment*, which sums many
   overlapping OD legs. Directional errors on individual legs partly cancel in
   that sum, so a large leg-level intervention yields a small segment-level
   one. The two are not the same target and I should not have documented one as
   if it predicted the other.
2. **The split is now over-dispersed.** Case 06 under-predicted directional
   asymmetry (0.542 against an observed 1.256); case 07 over-corrects to 0.871.
   With a correlation of 0.499 the error-minimising spread is `ρ·σ_observed` =
   0.627, so the shipped version overshoots by about 40%. Damping the logit
   deviation toward the frequency-neutral point by **λ ≈ 0.72** would take the
   gain from +7.0% to +11.4%.

   That λ is derived *from case 07's own results*, so it must not be adopted as
   fitted — it would be tuned on the test set. Estimating a damping factor
   properly, on held-out corridors at the segment level, is the obvious next
   step.

**The honest summary.** At segment level the correlation ceiling is ρ² ≈ 25% of
observed directional variance, not 75%. The change earns its place on the
headline measures — r +0.11 and RMSE −9.5 passengers are not small — but it
should be described as *improving the assignment of demand across directions*,
not as removing three quarters of directional error. Per operator the log-bias
result is a coin flip (AWC, EMR, GWR, SWR better; Greater Anglia, Northern,
Southeastern, TPE worse), which is a further reason not to oversell it.

---

## 5. Using it

### Fit the coefficients

```bash
cd datasources/cases/06_cross_operator_loading
python3 fit_direction_split.py              # writes _baselines/direction_split_model.json
python3 fit_direction_split.py --holdout    # prints the held-out-corridor skill
```

### Switch it on

```yaml
direction_split_model: "/path/to/_baselines/direction_split_model.json"
```

Leaving it unset preserves the previous symmetric behaviour exactly, so every
existing case result reproduces unchanged.

### Measure it

```bash
cd datasources/cases/07_improved_operator_loading
python3 run_improved.py          # case 06's own 270 trains, split switched on
python3 compare_to_case06.py     # the before/after table
```

`compare_to_case06.py` prints a headline comparison, a per-operator table, and
a **directional decomposition** that splits the log bias on bidirectional cells
into a symmetric part (the corridor level error, which this change does not
target) and an antisymmetric part (the directional error, which it does). A run
where the antisymmetric part falls and the symmetric part holds is the change
doing what it claims and nothing more. That is the result to read.

---

## 6. Caveats — read before quoting any of this

1. **Supply follows demand.** Splitting demand in proportion to trains bakes in
   the fact that operators run more trains where more people travel. For
   forecasting a *proposed* service that is exactly what you want — it encodes
   the planner's directional intent — but it is **not a causal claim** and must
   not be written as one. A reviewer will ask, and the honest answer is that
   the timetable is an input, not a result.
2. **Modest sample.** ~410–419 cells on ~123 corridors.
3. **Counted flow is a proxy.** Boardings at the train's origin, which
   conflates service pattern with demand.
4. **Case 07 is in-sample for direction.** Its outturn comes from the same
   feeds the split is fitted on, exactly as case 06 is already in-sample for
   the temporal split. The generalisation number to quote is the
   held-out-corridor figure in §4, not case 07's improvement.
5. **This does not fix the big one.** The per-corridor *level* error — roughly
   symmetric, highly repeatable, worth ~55% of bias variance on bidirectional
   cells — is untouched. It is the ODM being wrong pair by pair, and no
   arithmetic downstream of the matrix can repair it.

---

## 7. What was considered and rejected

**ODME (OD matrix estimation from link counts).** The standard method for
conditioning a prior matrix on observed counts, and the obvious thing to reach
for. Rejected, with a reason worth stating in the paper: the prior's deficiency
here is now *named and specific*, and two covariates fix most of it while
transferring to unbuilt services. General ODME would fit counts on observed
corridors and transfer well across **time** (split-half r = 0.982) but can do
nothing for the launch cases, which have no counts at all. **ODME improves the
baseline, not the forecast.**

**An ML correction layer.** Tested and rejected earlier: gradient boosting on
the residual scores R² = 0.85 cross-validated by train but **0.30** grouped by
route, and even that is inflated by lat/long letting it interpolate from
neighbouring counted corridors. The gap is memorisation of ~172 corridor
correction factors, which would lift case 06's headline and improve the launch
cases by nothing.

**Origin-station temporal profiles** (`journey_type_mode: origin_station`).
This is the only *pre-existing* lever that breaks direction symmetry, since
`A→B` gets A's profile and `B→A` gets B's. Fitted on half the RSP periods and
scored on the other half it correlates +0.426 with the observed directional
share but removes only **12.1%** of the directional error variance, and it
makes the **am_peak worse (−22%)** while helping pm_peak (+31%) and evening
(+20%). A station's overall peakiness does not tell you the split for a
specific pair. Useful for within-period scatter; not the direction fix.

---

## 8. Updates since this was written

**22 September — refitted on segment loads.** The first-stop fit above is
superseded: `direction_split.calibrate_to_segments` fits the same model on the
segment loads a hindcast is scored on (`_baselines/direction_split_model_segments.json`;
slopes am −0.46, early −0.63, interpeak −0.09, pm +0.25, evening +0.47;
rmse(log) 0.858 → 0.726–0.729 with routes or operators held out). See
`datasources/cases/09_tidal_direction_split/RESULTS.md`.

**27 September — two bugs in how the split met the rest of the model**
(`MODEL_REVIEW_2026-09-27.md` §1.1 and §1.3):

1. Destination substitution was handed the *split* cell and divided it by the
   group's *raw* cells, so a dominant London terminal could hold more than its
   whole group and the pivot logit blew up (one leg ×52). Case 09's run through
   the real code is contaminated by it (3% of segments over 1.5×); the offline
   harness figures are the right ones.
2. With the split on observed legs, a calls-only strip counted each candidate
   twice in the forward train count, so both directions tilted towards
   themselves and the split created demand; a joint whole-train strip never
   counted the reverse direction's new trains. `_trains_after` now counts both
   directions the same way.

**27 September — optional station-function term.** A per-period coefficient on
`g_o − g_d`, `g = ln(catchment ÷ jobs within a mile)`, fitted with
`fit_direction_split_segments.py --function`
(`_baselines/direction_split_model_segments_function.json`). Held out on case
06 it improves the footfall-only split slightly (0.718–0.723 against
0.726–0.729) and turns Cambridge South, a destination, most of the way round.
See `MODEL_REVIEW_2026-09-27.md` §3.4.
