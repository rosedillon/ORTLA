# Comparator models

Assessment of three open-source rail models against this one, carried out 2026-09-19.
Question asked: does any of them hindcast more accurately than we do, and is there a
method worth adopting?

Repositories examined:

- `station-demand-forecasting-tool/sdft` — Station Demand Forecasting Tool (Young &
  Blainey, University of Southampton). R package, DOI 10.5281/zenodo.4066924.
- `OpenRailAssociation/osrd` — Open Source Railway Designer (SNCF Réseau).
- `itrcrisks/GB-railway-network-model` — ITRC / University of Oxford (Pant, Hall &
  Blainey 2016).

## Summary

Only **SDFT** is a demand model. On the one branch where it overlaps this model —
total demand at a new station — **it is more accurate than our gravity branch**, under
a validation design comparable to ours. Its larger value, though, is that it produces
non-circularly the exact input that currently blocks cases 01 and 05.

**OSRD** has no demand component of any kind: no file in its documentation mentions
demand, passenger load, ridership or footfall. It is an infrastructure, timetabling and
capacity simulator.

**GB-railway-network-model** does not forecast demand either. It distributes *observed*
ORR station usage over timetabled train paths to synthesise an OD matrix. It contains no
calibration and no validation. One structural idea in it is worth taking (§4).

## 1. SDFT: the one real benchmark

### 1.1 What it is

A two-stage model for proposed new local stations in mainland GB.

**Stage 1, station choice.** A multinomial logit over the ten nearest stations to each
of ~1.5M GB unit postcodes. Coefficients in `R/sdr_calculate_probabilities.R`:

| variable | coefficient |
|---|---|
| nearest station (dummy) | 0.69065 |
| sqrt(road access distance) | −2.26183 |
| Category F station (dummy) | −0.67672 |
| ln(daily service frequency) | 1.19857 |
| CCTV (dummy) | 1.07082 |
| car parking spaces (count) | 0.00132 |
| ticket machine (dummy) | 0.98392 |
| bus interchange (dummy) | 0.75848 |

Sample 14,422; McFadden adjusted R² 0.71; 10-fold CV predictive performance 28.6%
against 32.3% in-sample.

**Stage 2, trip end.** OLS on ln(annual entries + exits), calibrated on Network Rail
Category E and F stations using 2011/12 ORR usage. Shipped coefficients, from
`R/sdft_submit.R:1262-1278`:

```
ln V = 3.5153
     + 0.3663 · ln(prw_pop + 1)          probability-weighted catchment population
     + 1.1245 · ln(freq)                 weekday daily train frequency
     + 0.0566 · ln1p(work_pop_2min)      jobs within a 2-minute drive
     + 0.1262 · ln1p(carspaces)
     + 0.2366 · electric
     + 0.3000 · travelcard_boundary
     + 0.7921 · terminus
```

n = 1,792; R² 0.8502; adjusted R² 0.8496; **residual standard error 0.6919** on 1,784 df.

Catchment population is decayed by `((d − 750)/1000 + 1)^−1.5212` beyond 750m of road
access distance, flat within 750m, cut off at a 60-minute drive-time service area
(`R/sdr_calculate_prweighted_population.R`).

> **Cite the code, not the paper.** The conference paper's Table 2 gives 3.67 / 0.37 /
> 1.14 / 0.05 / 0.13 / 0.24 / 0.30 / 0.78 and describes the jobs variable as a
> 1-minute drive. The shipped model is a later revision ("altered application of decay
> function and 2 minute work population"). The paper also labels the trip-end fit
> "McFadden's adjusted R²", which is wrong for an OLS — it is an ordinary adjusted R².

### 1.2 Head-to-head accuracy

SDFT's own validation, plus its Table 3 hindcast on ten stations that opened *after*
its 2011/12 calibration year, converted onto the metrics used in this study:

| model | unit | validation design | n | sd(log) | within ×2 | within ±25% | median ratio |
|---|---|---|---|---|---|---|---|
| SDFT trip end, calibration | station total | in-sample | 1,792 | **0.693** | 68.3%* | 25.3%* | — |
| SDFT trip end, 10-fold CV | station total | station holdout | 1,792 | **0.695** | 68.1%* | 25.2%* | — |
| SDFT, 10 newly-opened stations | station total | true ex-post | 10 | 0.468 | 80% | 30% | 1.15 |
| **this model, gravity branch** | station total | leave-one-station-out | 1,473 | **~0.87** | 57.5% | 19.6% | **1.02** |
| scheme appraisal forecasts | station total | ex-post | 8 | 1.304 | 37.5% | **0%** | 0.72 |
| this model, gravity branch | OD pair | leave-one-station-out | 60,725 | 1.537 | 28.5% | — | 2.94 |

\* implied from the reported MSE (0.48, and 0.483 under CV) assuming lognormal residuals.

Our station-total sd(log) is not reported directly in `gravity-branch-holdout`; it is
inferred, and three independently reported quantities agree closely:

| inferred from | implied sd(log) |
|---|---|
| "within a factor of 2, 57.5%" | 0.869 |
| "within ±25%, 19.6%" | 0.899 |
| "p10–p90 spans 0.36–3.27" | 0.861 |

**The comparison is fair, not flattering to them.** SDFT's 10-fold cross-validation has
one row per station, so it *is* a station holdout — the same design as our
leave-one-station-out, and not the random-pair interpolation trap. On the like-for-like
slice (small stations, which is all SDFT is calibrated on) our own figure is sd(log)
0.90 against their 0.69.

**SDFT carries roughly 37% less error variance than our gravity branch on new-station
totals.** That should be stated plainly rather than argued around.

### 1.3 Where we are nevertheless stronger

- **Bias.** Our median station-total ratio is 1.02 against their 1.15. The Duan smearing
  correction is vindicated; their forecasts run slightly hot.
- **Evidential base.** Their ten-station hindcast is seven Borders Railway stations plus
  three others — roughly four independent observations, all but three on a single line.
  That is why its sd(log) (0.468) comes out *better* than their own calibration residual
  (0.693), which is not credible. Our holdout is 1,473 stations.
- **Frequency elasticity.** Theirs is **1.124** (trip end) and 1.199 (station choice).
  We tested that magnitude — b4 = 1.30 fitted over all pairs — diagnosed it as absorbing
  the served/unserved difference rather than measuring a frequency effect, and cut to
  0.64 among served pairs. Their specification leaves that endogeneity unaddressed, and
  it bites hardest exactly where it matters: a new station given generous service. Their
  Tweedbank forecast over-predicts by 84% on the simple catchment.
- **Temporal drift.** Their dependent variable is 2011/12 and their validation actuals
  2017/18, with six years of national growth absorbed by a regional uplift lookup
  (`data.regional_uplifts`, applied as a flat percentage). Part of the residual in
  Table 3 is growth, not model error.

### 1.4 Scope

SDFT forecasts **new-station totals only**. It produces no OD pairs, no train loads, no
time-of-day, no direction, no new links between existing stations, and no extensions.
It has nothing to say about four of the seven cases in this study (02, 03, 04, 06/07).
It should be presented as a benchmark on one branch, never as a rival to the model.

## 2. The method worth adopting

SDFT's trip-end model outputs **annual entries and exits for a station that does not yet
exist**, from catchment population, service frequency, nearby jobs, parking and three
dummies. None of those inputs is the station's own outturn.

That is precisely the quantity `HindcastConfig.station_footfall` was removed for: filling
it with what a station later recorded makes the hindcast circular. **SDFT is the
non-circular way to fill it**, and it is the same unit as ORR table 1410 entries and
exits — a direct substitution into the gravity model's footfall term.

This unblocks **case 01 (Northumberland Line)** and **case 05 (Cambridge South)**.

Cost. The full pipeline needs PostgreSQL + PostGIS + pgRouting, OS Open Roads, unit
postcode population, and drive-time service areas. But it is needed here for about seven
stations, not nationally: choice sets for postcodes within a 60-minute drive of seven
points. Every input is open data.

Robustness. The origin elasticity in our gravity model is 0.508, so a ±50% error in the
SDFT-supplied footfall moves predicted demand by only +23% / −30%. The substitution does
not need to be precise to be useful.

Honesty requirement. Any case using this must state that station footfall is *modelled*,
not observed, and report the sensitivity — `01_northumberland_line/footfall_sensitivity.py`
already sweeps the assumption and should carry the SDFT value as its central case rather
than as one point on a sweep.

## 3. What none of them fixes

**Case 06's route-level error.** Nothing in any of the three repositories addresses the
per-corridor level error that accounts for 63.5% of log-bias variance. SDFT operates at
station totals; OSRD has no demand; ITRC has no calibration.

**Direction.** Nothing to adopt. SDFT has no direction at all — entries and exits are a
single combined number. ITRC's OD matrix *is* directional, but its only directional
content is timetable path asymmetry, because it assumes entries = exits = half the ORR
annual total at every station (`rail_od_flows.py`, `station_usage['entries'] = 0.5 * ...`).
That is the supply-follows-demand mechanism our directional split already uses, minus the
footfall-ratio term. We have their approach plus a term they lack.

It is worth one citation, though: it is direct evidence that the GB open-model ecosystem
has no directional demand data and papers over the gap by assuming symmetry — which
supports the framing in `DIRECTIONAL_DEMAND_SPLIT.md`.

## 4. The one structural idea from ITRC

The ITRC OD construction is cruder than ours — no ticketing matrix, no distance decay,
station usage only — but its allocation is **origin-constrained**, and ours is not.

Their procedure (`rail_od_flows.py`, steps 6–8):

1. For each timetabled path and each position in it, attractiveness =
   (trains on that path) × (sum of annual exits at all *downstream* stations on that path).
2. A station's observed entries are split across the paths serving it in proportion to
   that path's share of total attractiveness at that node. **Entries are conserved
   exactly at every station by construction.**
3. Within a path, entries boarding at position *i* are distributed to downstream stations
   in proportion to their exits.

Ours computes `ODM pair demand × period factor ÷ trains in period`, which is forced to
sum to nothing. That matches the observed symptom in `MODEL_DEFECTS.md` and the case 06
diagnosis: operator totals roughly conserved while corridor ratios span 0.10–4.29, and
59% of each train's error already present at the first stop.

**Caveat, and it matters.** A Furness / IPF balance to observed station totals is the
same category of fix as ODME: it constrains the baseline on corridors that have counts,
and does nothing for cases 01, 03 and 05, which have none. That was already ruled out,
for good reason, in `frequency-divisor-direction-bug`.

**The exception is the interesting part.** SDFT supplies a new station's row and column
total. A station-constrained balancing step therefore *does* transfer to an unbuilt
station, provided the constraint for that station comes from a trip-end model rather than
from counts. SDFT + station-constrained balancing is the one architecture that attacks
both the new-station blocker and the corridor misallocation at once. Recorded here as a
design direction, not a committed piece of work.

## 5. OSRD

Open Source Railway Designer: infrastructure design, capacity analysis, timetabling,
simulation and short-term path request. It contains a physics-based running-time
simulator (`core/envelope-sim`), a signalling simulation (`core/kt-osrd-signaling`,
`core/kt-osrd-sncf-signaling`), blocking-time conflict detection and STDCM.

In principle it is strictly better than what we do for correction 5 (BPLAN sectional
running times) and correction 6 (the fixed `planning_headway_minutes: 3` rule): it would
give a real blocking-time headway rather than a flat assumption.

**It is not adoptable here.** It consumes RailJSON infrastructure — track geometry,
gradients, speed limits, electrification, signal positions and block boundaries — and the
only signalling module shipped is SNCF's. Network Rail publishes no open signal or block
data for GB, and BPLAN gives sectional running times without signal spacing. Building a
GB RailJSON model, even for one corridor, means sourcing data that is not openly
available, and it would not improve demand accuracy, which is where essentially all of
our error sits.

Worth one sentence in related work: a full microsimulation alternative to the BPLAN
timing and headway checks exists, but the infrastructure data it requires is not open for
Great Britain.

## 6. Also worth a look, not yet pursued

SDFT includes an **abstraction analysis**: a before/after choice set for each postcode,
the percentage change in an existing station's probability-weighted population, and an
assumed elasticity of one applied to its current entries and exits.

That is the right *shape* for the case 04 destination-substitution defect — the 96.9×
invisible London flows, where a new link to Blackfriars cannot capture existing Victoria,
London Bridge or Charing Cross passengers. The structure needed is a choice model over
competing stations.

Its coefficients will not transfer. SDFT's station choice model is entirely access-end
(nearest, road distance, parking, CCTV, ticket machine, bus interchange), and London
terminal choice is a destination-end problem with different attributes. The architecture
transfers; the calibration does not.

## Sources

- Young, M. & Blainey, S. (2019). *An automated online tool to forecast demand for new
  railway stations.* Transport Practitioners' Meeting.
  https://www.marcusyoung.co.uk/files/tpm2019.pdf — model specification, Table 2
  (calibration) and Table 3 (ten-station evaluation).
- Young, M. & Blainey, S. (2018). *Development of railway station choice models to improve
  the representation of station catchments in rail demand models.*
- Young, M. & Blainey, S. (2020). *sdft: The Station Demand Forecasting Tool.* R package
  v0.3.1. https://doi.org/10.5281/zenodo.4066924 — shipped coefficients.
- Pant, R., Hall, J. W. & Blainey, S. P. (2016). *Vulnerability assessment framework for
  interdependent critical infrastructures: case-study for Great Britain's rail network.*
  EJTIR 16(1). https://doi.org/10.18757/ejtir.2016.16.1.3120
- Transport Scotland (2017). *Borders Railway year 1 evaluation.*
- Steer Davies Gleave (2010). *Station usage and demand forecasts for newly opened
  stations.*
