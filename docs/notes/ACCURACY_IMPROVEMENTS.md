# Improving predictive accuracy: what the evidence supports

Diagnostic run 2026-09-19 on `cases/06_cross_operator_loading/output/all_segments.csv`
(8,655 complete segments, 217 trains, 9 operators; 8,059 of them with ORR table 1410
coverage at both ends, which is the working sample below). **No code changed.**

Companion to `PAPER_METHODS.md`, `COMPARATOR_MODELS.md` and
`DIRECTIONAL_DEMAND_SPLIT.md`. Written after `paper/REVIEW.md`, and three of the
referee's points are settled here.

Baseline on the working sample: **rmse(log bias) 0.846, overall ratio 0.843,
r = 0.451**. An oracle per-route scalar takes rmse(log) to 0.452 — that is the
memorisation ceiling, not a target.

---

## 1. The headline: one candidate fix looks excellent and must not be shipped

The largest single signal found is that **the model under-predicts in proportion
to the frequency it divides by**. Binned on the demand-weighted median
`existing_frequency` of the legs composing each segment:

| trains in period | 1 | 2 | 3 | 4 | 5 | 6 | 9 | 13 |
|---|---|---|---|---|---|---|---|---|
| predicted ÷ observed | 1.31 | 1.04 | 1.08 | 0.93 | 0.67 | 0.66 | 0.73 | 0.61 |

Regressing log bias on log frequency, clustered by corridor:

> slope **−0.277** (se 0.024, *t* = −11.3), i.e. the divisor should be
> *f*^κ with **κ = 0.723** (95% CI 0.62–0.83)

κ is stable where it matters: 0.69–0.78 across five corridor-grouped CV folds,
0.68–0.76 leave-one-operator-out, and negative in six of nine operator feeds
(GWR and SWR flat). Out-of-route cross-validation gives **+9.2% of log-bias
variance and a ratio of 0.998** — it removes the level bias outright, and it
transfers out of corridor, which the rejected ML layer did not.

It also has a mechanism, and the mechanism makes a prediction that holds. If the
nominal train count over-states the *effective* choice set — because a passenger
picks among the trains near their desired departure, not all of them — then κ
should approach 1 as the period narrows. It does:

| period | width | κ |
|---|---|---|
| am_peak | 2.5 h | **0.949** |
| pm_peak | 3 h | 0.812 |
| evening | 5 h | 0.757 |
| interpeak | 6.5 h | 0.789 |
| early | 7 h | 0.603 |

corr(κ, period width) = **−0.82**. This is PDFH's service-interval penalty and
MOIRA 2.2's non-uniform allocation, arrived at from the residuals.

**Now the check that kills it.** Applying κ = 0.723 to the two hindcasts that
currently work:

| case | demand-weighted mean *f* | uplift | ratio now | ratio after |
|---|---|---|---|---|
| 02 Whitby → Newcastle | 2.53 | ×1.27 | 0.955 | **1.21** |
| 03 Lowestoft → Cambridge | 11.91 | ×1.67 | 0.938 | **1.57** |

The correction that removes case 06's bias breaks both clean cases. Case 03 sits
at ratio 0.94 on legs averaging twelve trains a period, where case 06 says the
ratio at that frequency should be 0.61. **The two measurements are
inconsistent, and the frequency exponent is not a universal property of the
divisor.**

Corroborating that reading: within corridor the slope collapses to −0.101
(*t* = −1.8, not significant). The effect is almost entirely *between* corridors,
so log frequency is standing in for corridor type — urban, short-distance,
season-ticket-heavy — rather than measuring an allocation law. Splitting by trip
length does not reconcile it either (κ = 0.61–0.81 in every quartile).

**Do not adopt κ as a global divisor change.** It is a diagnostic, not a fix. The
defensible next step is to re-run cases 02 and 03 under it rather than assume,
and to test it per demand branch — cases 02/03 run through `gravity_new_link`,
case 06 through pure `gravity_pivot`, and the divisor may not even be the
operative path in both.

This is the same trap the ML correction layer fell into, one level up: it scores
beautifully on the validation case and contradicts the only outturn that is not
the validation case.

## 2. What the evidence does support: rebalance the matrix to table 1410

The ODM's station totals and ORR table 1410 are two independent measures of the
same quantity, and they disagree station by station. Over 2,510 stations:

| | ODM (out+in) ÷ table 1410 entries and exits |
|---|---|
| national | 1.246 |
| median | 1.249 |
| p5 – p95 | **1.00 – 1.65** |

The spread is the point, not the level. Regressing log bias on that ratio:

> coefficient **+1.009** (se 0.273, *t* = +3.7), *R*² 0.012

**A coefficient of 1.0 is the signature of a pure level error.** It says that
where the ODM over-states a station's demand by a factor *R* relative to the
counted total, the model's prediction is too high by exactly *R* — which is what
a station-total rebalance removes by construction. Out-of-route CV gives **+4.9%
of variance and ratio 0.999**.

Two independent literature findings name the mechanism:

- DfT's *Journeys per Season Ticket Study* (2019) measured **378 annual journeys
  per season ticket against the 480 LENNON assumes** — LENNON over-states season
  journeys by ×1.27, against the ×1.25 measured here.
- ORR/Steer's methodology note records that **London season tickets are sold from
  the furthest station in the fare group regardless of actual origin**, because
  identically-priced stations give passengers flexibility at no extra fare. That
  is a *within-corridor spatial misallocation with the corridor total conserved* —
  precisely the diagnostic signature in `case06-route-effect-diagnosis`: operator
  totals roughly conserved (awc 0.99, northern 1.01) while corridor ratios span
  0.10–4.29, and 59% of each train's error already present at the first stop.

Season-ticket intensity, published per station in table 1410, tracks the bias
directly:

| station season-ticket share | ≤0.087 | 0.087–0.12 | 0.12–0.15 | 0.15–0.18 | 0.18–0.23 | >0.23 |
|---|---|---|---|---|---|---|
| predicted ÷ observed | 0.93 | 0.98 | 0.98 | 0.81 | 0.79 | **0.63** |

### Measured: case 08

Implemented as `rebalance_to_table_1410` on the loading config
([`od_matrix.py`](src/ortla/od_matrix.py)) and run over case
06's own 270 trains. Full result in
`datasources/cases/08_table_1410_rebalance/RESULTS.md`.

| measure | case 06 | case 08 | |
|---|---:|---:|---|
| rmse(log bias) | 0.850 | **0.839** | better |
| overall ratio | 0.882 | **0.910** | better |
| GEH < 5 % | 55.6 | **56.4** | better |
| correlation *r* | 0.451 | 0.457 | ~unchanged |
| RMSE (passengers) | 114.4 | 114.8 | marginally worse |

**2.47% of log-bias variance, 95% CI [1.25%, 3.64%]** (paired bootstrap, 2,000
resamples clustered by corridor). Seven of nine operators improve; the two that
worsen (EMR, Greater Anglia) were already over-predicting and are pushed
further up. The largest gain is GWR, the worst under-predictor at x0.548, which
moves to x0.626 with rmse(log) 0.825 -> 0.752. Corridor sd(log ratio) falls
0.722 -> 0.711.

The mechanism holds: binned by how far the rebalance actually moved each
prediction, the segments it barely touched gain 0.2% and the top quintile
(median move 13.6%) gains **7.9%**, from the worst baseline error of any
quintile.

It delivers less than the +4.9% the cross-validation above suggested, for the
same reason the directional split under-delivered in case 07 -- the correction
acts on station marginals and a segment load sums many overlapping legs whose
corrections partly cancel. The CV figure also fitted a free coefficient where
the rebalance imposes the estimated 1.0, so it was an upper bound.

Two things it is not. It does not touch the route effect: an oracle per-corridor
scalar reaches 0.452 against this run's 0.839. And the gravity refit's
improvement (*R*2 0.707 -> 0.718) is **partly mechanical**, because the
constraint is built from the same table 1410 footfall the gravity model
regresses on -- do not quote it as validation.

**Why this is not ODME.** The objection recorded in `frequency-divisor-direction-bug`
and repeated in the paper is that matrix estimation conditions on link counts,
which a new service does not have. Table 1410 is not a link count — it is a
station-level published statistic that exists for every station in Britain, and
SDFT predicts it, in the same unit, for stations that do not exist yet
(`COMPARATOR_MODELS.md` §2). A station-constrained balance therefore **transfers
to an unbuilt station**, which is exactly the exception already identified in
`COMPARATOR_MODELS.md` §4. This is the paper's own §6(ii) recommendation, and it
is the one correction here that survives every test applied to it.

## 3. The smearing factor does contaminate the discovery headlines

`paper/REVIEW.md` point 4 asks whether the ×2.94 per-pair over-prediction washes
out when summed over a candidate's legs. The paper asserts it does — "correct when
summing many pairs and wrong when a single pair's figure is read on its own".

**It does not, at the scale the paper uses it.** Simulating the fitted residual
distribution (sd 1.526, Duan 2.916), median ratio of summed prediction to summed
outturn by pair count:

| pairs summed | 1 | 6 | 21 | 45 | 100 | 300 | 1000 |
|---|---|---|---|---|---|---|---|
| median over-prediction | 2.92 | 1.75 | 1.40 | 1.27 | 1.17 | 1.07 | **1.00** |

Convergence needs roughly a *thousand* pairs. The three worked cases sum 6, 21
and 45 ordered stop pairs. And because the residuals are correlated within a
corridor — split-half *r* = 0.946 on route effects — the independent figures above
are a **floor**; at ρ = 0.5 the same counts give 2.19, 1.96 and 1.87.

This changes a stated conclusion. The paper reports Woking–Reading as "level
with" Gatwick–Birmingham on generated demand:

| case | pairs | quoted | median-corrected |
|---|---|---|---|
| Woking – Reading | 21 | 24,506 | **17,441** |
| Gatwick – Birmingham | 45 | 24,522 | **19,347** |

They are not level. Gatwick–Birmingham is ahead by 11%, and the gap runs the way
it does *because* the two cases sum different numbers of pairs — the referee's
exact objection. Tonbridge–Gatwick, at six pairs, carries the largest inflation of
the three (×1.75).

Report both columns, and state the pair count beside every summed figure.

## 4. Ruled out — do not re-chase

Each of these was a plausible lead and each is dead:

| tested | result |
|---|---|
| **Interchange traffic** the model structurally omits, from table 1410's interchange column | corr(bias, interchange rate) = **+0.003** segment-level, −0.031 at route level. Flat across every quantile. |
| **Stopping pattern** — fast trains carrying more than the uniform share | corr(bias, log stop density) = −0.027; jointly with frequency *t* = +0.3. Nothing. |
| **Per-period κ** instead of one global value | Out-of-route CV **+3.5%** against the global κ's +9.2%. The per-period CIs all overlap 1.0. It overfits. |
| **Capacity / crowding** | Utilisation is 14–24% across every frequency band. Not binding. |
| **A factor of two in the ODM** | Confirmed dead again from the station side: ODM(out+in) ÷ 1410 = 1.25, not 2.0. As recorded in `DIRECTIONAL_DEMAND_SPLIT.md` §2.1 — do not re-chase. |

Also corrected: `case06-route-effect-diagnosis` records the frequency divisor as
ruled out at corr = −0.044. That measurement used raw `existing_frequency`; taken
as the demand-weighted median over the legs actually composing each segment it is
**−0.26 segment-level and −0.32 at train-route level**. The earlier figure was an
aggregation artefact. The conclusion in §1 above still stands, for a different
reason.

## 5. The honest ceiling

Everything transferable found here recovers about **10% of log-bias variance**
(rmse(log) 0.846 → 0.80) and removes the level bias. The route effect is 63%, and
the oracle per-route scalar reaches 0.452. The gap between 0.80 and 0.45 is the
ODM being wrong pair by pair, and §2 is the only lever on it that transfers.

No arithmetic downstream of the matrix repairs the matrix. That should be the
paper's claim, and it is a stronger one than ×0.955.

## 6. What to do, in order

1. ~~**Rebalance the ODM to table 1410 station totals.**~~ **Done** --
   `rebalance_to_table_1410`, measured as case 08 above: 2.47% of variance
   [1.25%, 3.64%], ratio 0.882 -> 0.910, seven of nine operators better. The
   outstanding piece is the out-of-sample test: re-run cases 02 and 03 under it,
   since the decision to apply it came from case 06's residuals.
2. **Correct the discovery figures for smearing at their own pair counts**, and
   report pair count beside every summed demand figure. This is a write-up change
   with no new model run, and it revises a stated comparison.
3. **Adopt SDFT's trip-end model as the footfall input** for cases 01 and 05
   (`COMPARATOR_MODELS.md` §2). Still the highest-value outstanding work: it turns
   two blocked cases into the only genuine tests of the synthetic branch.
4. **Test κ per demand branch on cases 02 and 03 before going near it.** Do not
   ship it on the strength of case 06.
5. **Origin-station temporal profiles as an additive layer**, not a replacement.
   The distance band removes 1% of between-corridor profile variance; the origin
   station removes 30% and the destination 25%. It is unusable only for the
   *new-station* branch, which is two of seven cases — it applies to 02, 03, 04,
   06/07 and the entire national scan. Implement it as a layer alongside bands
   still fitted to all rows, or `generate_period_matrices` silently writes zero
   demand (`case06-temporal-resolution-findings`).
6. **Re-run cases 02 and 03 at the within-pair frequency elasticity** (§7).
   The pivot exponent b4 = 0.64 is cross-sectional; within a pair the
   association is 0.01-0.08 and survives the attenuation check. This is the
   model's mechanism for "what does adding trains do", so it outranks
   everything below it -- but it is not to be changed until those two cases
   have been run under it.
7. **Report uncertainty** — `paper/REVIEW.md` point 1. Cluster by train and by
   corridor, and give *n* trains beside *n* segments everywhere. Every figure in
   this document is clustered by corridor; it costs nothing and the results
   survive it.

## 7. The frequency elasticity is cross-sectional, and within a pair it nearly vanishes

Added 2026-09-22, from the ODM panel the archive CIFs make possible
(`scripts/frequency_panel.py`, `ERA_MATCHED_TIMETABLES.md`).

Section 1 above could not settle the frequency question because every estimate
available was cross-sectional: the apparent exponent collapsed from -0.28
between corridors to -0.10 (*t* = -1.8) within them, so log frequency was
largely standing in for corridor type. The same objection applies to the gravity
model's own **b4 = 0.64**, fitted across pairs in one year.

Five ODM releases and eight published timetables allow the other measurement:
watch **one pair's demand move when that same pair's frequency moves**. First
differencing removes the pair -- its population, its distance, its corridor
type, everything fixed about it.

| span | pairs | clusters | elasticity | se | *t* |
|---|---:|---:|---:|---:|---:|
| 2018-19 -> 2019-20 | 30,950 | 2,325 | **0.016** | 0.005 | 3.0 |
| 2019-20 -> 2022-23 | 27,448 | 2,242 | **0.074** | 0.007 | 10.8 |
| 2022-23 -> 2023-24 | 27,784 | 2,262 | **0.093** | 0.006 | 15.0 |
| 2023-24 -> 2024-25 | 30,303 | 2,343 | **0.032** | 0.006 | 5.3 |
| pooled, span-demeaned | 116,485 | 2,382 | **0.060** | 0.004 | 15.5 |
| pooled, excluding the covid span | 89,037 | 2,381 | **0.051** | 0.004 | 13.4 |

Standard errors clustered by origin station. Against a cross-sectional 0.64,
the within-pair association is **0.02 to 0.09** -- an order of magnitude smaller.

These are the figures of the current panel (`output_paper/analysis/
frequency_panel_elasticity.csv`, 22 September), which added the 2022-23 ODM and
the May 2022 timetable. An earlier version of this section, without them,
reported 0.043 pooled and 0.009 excluding the covid span.

**Robust to the timetables' short-term schedules** (checked 27 September). The
archive timetables book part of some operators' service as short-term ("N")
schedules, which every reader in the model drops (it keeps permanent "P"
schedules only), so 84-107 stations per era lose more than a tenth of their
trains. Rebuilding every era's route frequencies with the schedules resolved,
and as the median of five Wednesdays, gives 0.058 pooled and 0.050 excluding
covid; the rebuild with permanent schedules only reproduces the shipped files
exactly.

### The check that would kill it, and does not

A first difference of a noisily measured regressor is biased toward zero, and
this regressor is noisy: frequency is read off one reference weekday per
extract. If attenuation were producing the small estimate, the elasticity would
have to rise as the frequency change grows relative to the noise. It does not:

| size of the frequency change | pairs | elasticity | se |
|---|---:|---:|---:|
| \|d ln f\| in [0.02, 0.10) | 28,754 | 0.076 | 0.052 |
| \|d ln f\| in [0.10, 0.25) | 31,230 | 0.025 | 0.016 |
| \|d ln f\| in [0.25, 0.60) | 26,851 | 0.080 | 0.009 |
| \|d ln f\| in [0.60, inf) | 25,222 | **0.059** | 0.004 |

**Among pairs whose frequency changed by more than 60% in logs -- where
measurement error is negligible beside the signal -- the elasticity is 0.059
(*t* = 14.8), not 0.64.** There is no monotone rise with the size of the change,
which is what attenuation would look like.

Two further biases both run the *wrong way for the small estimate*, so they
cannot rescue 0.64 either: operators add trains where demand is already growing
(simultaneity, biases up), and the ODM is an allocation of ticket sales rather
than a count, so any service-pattern step in that allocation ties demand to
frequency mechanically (also up). The one bias running the other way besides
measurement error is ramp-up -- a May-2023 frequency increase need not show its
full effect by March 2024 -- and the multi-year spans above do not remove it.

### Why this matters more than section 1

b4 is not a diagnostic. It is the mechanism: `predict_demand` pivots an existing
pair's observed demand by `(f_after / f_before) ** b4`, so b4 *is* the model's
answer to "what does adding trains do". At 0.64 a doubled frequency adds 56% of
demand; at 0.06 it adds 4%.

This is also the cleanest available explanation of a result section 1 left
unreconciled. Case 03 (Lowestoft -> Cambridge) runs on legs averaging twelve
trains a period and comes in at ratio 0.938; case 02 (Whitby, 2.5 trains) at
0.955. A frequency term that is too strong over-predicts most where frequency
is highest and where it moved most -- and those two cases move in that order.

### The test is built, and the obvious form of it measures nothing

`datasources/cases/run_b4_test.py` re-runs cases 02 and 03 across a range of b4
(`ProposeConfig.frequency_coefficient_override` substitutes the coefficient at
load time, leaving the rest of the fit alone). Setting it up turned up a
mis-specification in the test as proposed above.

**A hindcast does not consult b4 on the branch the cases mostly use.**
`HindcastConfig.apply_frequency_dilution` defaults to False, and with it off
`_calibrated_leg` grows an already-served pair's demand *linearly* with the
period's trains rather than by `(f_after / f_before) ** b4` -- deliberately, to
hold per-train load for a train that really ran. Both cases are calls-only
strips, so their `gravity_pivot` legs came back with a frequency ratio of
exactly 1.000 at every b4. Measured that way:

| case | dilution | b4 | ratio | MAPE % | RMSE |
|---|---|---:|---:|---:|---:|
| Lowestoft | off | 0.6386 | 0.870 | 37.57 | 19.65 |
| Lowestoft | off | 0.000 | 0.870 | 37.57 | 19.65 |
| Whitby | off | 0.6386 | 0.960 | 28.93 | 18.45 |
| Whitby | off | 0.000 | 0.957 | 28.95 | 18.44 |

Case 03 is *identical to six decimal places*; case 02 moves 0.3%, and only
through the `gravity_new_link` and `gravity_synthetic` legs, where b4 enters the
prediction's level rather than a ratio. So "re-run the cases at a lower b4"
under the hindcast's own convention answers nothing: b4 is inert there by
construction.

The setting the section-7 claim is actually about is `propose`'s forecasting
treatment, `apply_frequency_dilution: true` -- would a lower b4 have *forecast*
these services better? On case 02 the first two runs say no:

| case | dilution | b4 | ratio | MAPE % | RMSE |
|---|---|---:|---:|---:|---:|
| Whitby | on | 0.6386 | 0.736 | 32.92 | 21.95 |
| Whitby | on | 0.000 | 0.707 | 34.44 | 22.76 |

Lowering b4 makes it worse on all three measures, and the direction is
mechanical: under dilution per-train load goes as `f ** (b4 - 1)`, so a smaller
b4 sheds more load per train and deepens an under-prediction that is already at
0.736. Note also that dilution-on is the *wrong* treatment for a hindcast --
it costs case 02 0.96 -> 0.74 on the ratio at the fitted b4, which is why the
default is off. It is a stress test, not a headline.

### What has *not* been done

**The sweep has not been run out.** Four of the sixteen runs above exist; the
rest (b4 = 0.30, 0.15, 0.06, and case 03 under dilution) are set up and
outstanding, about half an hour of compute:

    cd datasources/cases && python3 run_b4_test.py

**Nothing has been changed in the model, and on this evidence nothing should
be.** Section 1 is a record of a correction that scored beautifully on case 06
and broke both clean hindcasts; the same discipline applies here, in the other
direction. What the panel measures is real, but these two cases are the wrong
instrument for it -- a calls-only extension onto a corridor that already has
trains barely moves frequency, which is exactly the regime in which b4 cannot
show itself. Testing b4 properly needs a case where a pair's direct frequency
genuinely changes.

Also unresolved: this measures the pivot on *already-served* pairs, which is the
branch cases 02, 03 and 06 use. It says nothing about `gravity_new_link`'s
treatment of a pair gaining its first direct service, where the 8.3x
direct-service dummy was already diagnosed as selection.

## Sources

- DfT / Steer (2019). *Journeys per Season Ticket Study.*
  https://assets.publishing.service.gov.uk/media/60085f1fd3bf7f2ab1a1af90/journeys-per-season-ticket-study.pdf
- ORR / Steer (2025). *Station Usage & Origin Destination Matrix 2024/25: historical
  methodological changes.*
  https://dataportal.orr.gov.uk/media/1903/station-usage-and-origin-destination-matrix-steer-historical-methodological-changes.pdf
- Rail Delivery Group. *About the Passenger Demand Forecasting Handbook* (PDFH v6.0,
  service-interval penalty; MOIRA 2.2 allocation).
  https://www.raildeliverygroup.com/pdfc/about-the-pdfh.html
- DfT (2018). *Rail Demand Forecasting Estimation Study.*
  https://assets.publishing.service.gov.uk/media/5a82d78ded915d74e6237e7c/phase2-rail-demand-forecasting-estimation-study.pdf
- Duan, N. (1983). Smearing estimate: a nonparametric retransformation method.
