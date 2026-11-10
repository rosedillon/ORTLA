# Model review, 27 September 2026

A review of the 26 September work (SDFT × 0.8 balance, new-station cases
01/03/05/11–15, `LITERATURE_BENCHMARK.md`) and of the code paths it runs
through. It found three bugs in the demand code, two in the inputs, one
coding gap in the timetable, and several claims from 26 September that do not
survive checking. Everything below is uncommitted, like the work it reviews.

Section 1 lists what was wrong and is now fixed, section 2 the claims that
change, section 3 new diagnostics of the whole-train error, section 4 the
re-run results, section 5 what to do next and section 6 the data that would
help.

## 1. Bugs found and fixed

### 1.1 The direction split and destination substitution did not compose

`service_proposal.predict_demand` handed destination substitution the pair's
cell *after* the direction split (up to twice the raw cell) and
`_SubstitutionContext.terms` divided it by the *raw* total of the destination
group. A London terminal holding most of its group then came out holding more
than all of it (share 1.66 for Colchester → Liverpool Street), and the
pivot-logit denominator `s·pull + (1 − s)` went to nearly zero: that leg came
out **52 times** too big.

*Fix:* substitution gets the raw cell; `_on_split_direction` then puts its
result on the leg's direction by the split's own factor; the share is clamped
to 1 as a guard. *Exposure:* only London legs (the only substitution group),
and only runs with the split on observed legs — case 09's run through the real
code, the direction-split tests, and the `_balanced_split_all` variants.
*Test:* `test_substitution_share_of_its_own_group_cannot_exceed_one`
(mutation-checked: it fails without the clamp).

**Case 09's real-code figures are contaminated.** Against the offline harness,
which composes the two correctly, 3% of case 06 segments were inflated more
than 1.5×, up to 10.7× (SWR Portsmouth Harbour → Waterloo). London routes:
ratio 0.757 and rmse(log) 0.733 correctly, against 0.807 and 0.742 as run.
Quote the offline figures in `09_tidal_direction_split/RESULTS.md` §2, not §3.

### 1.2 A slower train made its destination less attractive

The substitution pivot priced the candidate's own journey time as the new time
to its terminal, so any train slower than the best existing journey *lowered*
its terminal's share. The faster train still runs, so adding a slower one
cannot. In a calls-only strip the candidate is still in the timetable it is
compared with, so the pull could only ever fall. Demand-weighted over London
legs, the demand base fell to **0.86** of the raw cell in case 11
(Thameslink), 0.93 in case 12, 0.94 in case 05 (GTR) and 0.98 in case 15;
tenth percentile 0.66.

*Fix:* the pull uses `min(candidate time, best existing time)`. New direct
links that are faster — what substitution exists for (case 04) — are
unchanged. *Test:* `test_a_slower_train_does_not_make_its_destination_less_attractive`.
*Exposure:* whole-train figures of every run with London legs (cases 04, 05,
06/09, 11, 12, 15). New-station on/off is unaffected: new-station legs are
always `gravity_synthetic` and never take a substituted base.

### 1.3 The direction split counted candidates twice, or not at all

The split's forward train count was `existing + k` ("the candidate is itself a
train"). In a calls-only strip the candidate is still in the stripped
timetable, so it was counted twice: every leg tilted towards its own
direction, the two directions' shares summed to more than one, and the split
*created* demand — about 14% on a London leg where a third of the trains are
candidates. In a joint whole-train strip (the Northumberland Line) the
opposite happened: the reverse direction's new trains were never added.

*Fix:* `_trains_after` counts trains after the change the same way both ways —
nothing added when the candidate is already in the timetable, the candidates
each way when it is not. *Tests:* `test_split_counts_a_candidate_still_in_the_timetable_once`,
`test_split_counts_added_trains_both_ways_when_they_are_new`. *Exposure:* runs
with the split on observed legs.

### 1.4 The SDFT port read a two-minute drive as a two-minute walk

SDFT's jobs term is the workplace population inside a **two-minute drive-time**
service area on OS Open Roads (`R/sdft_submit.R`: "2 minute work population
(1 mile)"; `sdr_create_service_areas(..., cost = "time")`). The port used a
200 m radius, which gave Marsh Barton's trading estate and Cambridge South's
campus no jobs at all — and its README blamed coarse output-area centroids.
Now 1,609 m. It barely changes the calibration fit (residual sd 0.879 → 0.851
for 2024-25) and does not change accuracy on real new stations (§2.1), but it
is SDFT's definition.

### 1.5 Every timetable reader drops short-term schedules

`_filter_services`, `_filter_service_calls`, `read_public_calls` and the
departure-time reader keep `stp_indicator == "P"` only. The May 2026 timetable
the hindcasts use is clean (24,433 permanent schedules against 17 "N" and 1
overlay). The archive eras are not: of the trains running on the reference
day, Avanti 59% and Transport for Wales 72% permanent (May 2022), CrossCountry
57% (May 2023), Merseyrail 69%, ScotRail 70% and Northern 85% (June 2024);
84–107 stations per era lose more than a tenth of their trains, and Upholland
showed 2 a day.

Fixed inside the SDFT port only (`running_calls`: overlay over permanent, "N"
in its own right, median of seven Wednesdays). Measured elsewhere it does not
matter: the within-pair elasticity panel gives 0.058 with the schedules
resolved against 0.060 without. **Not yet measured:** era-matched loading
(`timetable_eras`), whose stop-time lookup has the same filter. The core
filter was left alone because changing it would move every May 2026 result
for 18 schedules.

### 1.6 Smaller things

- `loading_analysis.load_table_1410` assigned strings into an int64 column with
  `df.loc[:, col] = ...`; pandas warned on every run that this will raise in a
  future version, which would stop every model run at the first table 1410
  read. *Fixed* with whole-column assignment (checked with the warning raised
  as an error, on three years of table 1410). The other pandas warnings in the
  logs (`concat` with empty frames in `hindcast.py`, `fillna` downcasting) will
  change dtypes, not raise, and are left.
- `ACCURACY_IMPROVEMENTS.md` §7 quoted an earlier run of the frequency panel
  (0.043 pooled, 0.009 excluding covid). The current panel gives **0.060** and
  **0.051**. The conclusion — an order of magnitude below the cross-sectional
  0.64 — holds, and §7 is updated.
- The Thanet Parkway note that Southeastern's counts are "44% of ORR" divides
  weekday-only counts by ORR's all-days total. Like for like (× 0.8) it is
  about 55%.

## 2. Claims from 26 September that change

### 2.1 "Better SDFT inputs are now the biggest lever" — no

The claim rested on four stations whose 2024-25 outturn was their build-up
year, compared with forecasts made at 2022-23 demand levels. A proper test:
every England and Wales station first in the ORR series between 2012-13 and
2023-24 (bar the Elizabeth line's central stations and Heathrow's), 32 of them,
26 mature; calibrate on existing stations only, predict each new station's
2024-25 usage from its catchment and service, same year.

| SDFT port | median | sd(log) | within ×2 |
|---|---|---|---|
| as the cases used it (200 m jobs, permanent schedules only) | 0.90 | 0.67 | 72% |
| as now (one-mile jobs, schedules resolved) | 0.90 | 0.70 | 72% |
| either, every slope refitted to modern data | 0.74–0.88 | 0.67–0.69 | 72–75% |
| *UK scheme appraisals, DfT/SDG 2010 (n = 23)* | *0.94* | *0.61* | *74%* |

The port is about as accurate as scheme appraisals, and **nothing tried here
improves it** — the jobs radius, the train counts, or refitting every slope
(which does want a frequency elasticity near 0.6 against the shipped 1.12, and
a terminus effect near 0.2 against 0.79). At n = 32 the sd of the sd is about
0.09. Mature stations are **under**-forecast (median 0.78–0.88); the 2023
openings look over-forecast because 2024-25 was their build-up year. The big
misses are parkways on fast London lines (Oxford Parkway 0.3–0.4,
Worcestershire Parkway 0.15–0.19) and new-build catchments (Cranbrook
0.13–0.40, Barking Riverside, Meridian Water). Better inputs for those two
kinds of station are the lever, not the jobs radius.

### 2.2 "About as accurate as UK scheme appraisals" — not shown

The 12-station headline (median 1.10, sd(log) 0.62, 42% within ±25%, 75%
within ×2) has three problems, now measured by `new_station_eval.py`
(`new_station_eval_summary.csv`):

1. **Years.** Predictions are at each case's demand year (2022-23 for cases
   11–14, 2023-24 for the Northumberland Line, 2024-25 for Cambridge South and
   Beaulieu Park); the counts are from summer 2026. The counted operators grew
   by 1.08–1.26 over the gap (ORR table 1223, now in
   `datasources/orr-table-1223/`). Year-consistent, the median is 1.34 and only
   25% are within ±25%.
2. **Clustering.** Six of the twelve stations are one line, one footfall model
   and one operator. By scheme (seven), the balanced model has median 1.25,
   sd(log) 0.79 and 57% within ×2 — below the SDG appraisals' 0.61 and 74%.
3. **Uncertainty.** Resampling schemes, the 90% interval on the station-level
   sd(log) is 0.35–0.99 and on the median 0.82–2.45: "level with" cannot be
   told apart from much better or much worse.

Year-consistent *and* with PDFH build-up the station-level figures come back
to about 0.92 / 0.68 / 75%, and by scheme 1.04 / 0.83 / 57%. The design (the
balance, the 0.8) was also chosen after seeing these same stations, so this is
not a clean out-of-sample test and should be described as such. The 32-station
test in §2.1 is the stronger evidence on new-station totals.
(Figures here are before the re-run; §4 has the re-run's.)

## 3. New diagnostics of the whole-train error

### 3.1 The shortfall is trip *length*, not trip count

On case 06's 854 fully counted train-days the model predicts **1.04×** the
counted boardings but **0.86×** the on-board load: its passengers ride 4.36
segments on average against 5.27 counted. The error is how far people travel,
not how many travel.

| operator | trips, predicted ÷ counted | trip length, predicted ÷ counted |
|---|---|---|
| GWR | 0.91 | **0.59** |
| GTR | 1.18 | **0.67** |
| Southeastern | 0.87 | 0.78 |
| SWR | 0.82 | 0.81 |
| Greater Anglia | 1.49 | 0.81 |
| EMR | 1.28 | 0.92 |
| TPE | 0.91 | 0.90 |
| Avanti | 1.08 | 0.94 |
| Northern | 0.88 | 1.14 |

Ruled out, each by direct test:

- **The tidal split.** Applied offline, trip length goes 0.83 → 0.81.
- **Which train people pick.** Every train in a period gets an equal share
  today. A logit over the period's trains on in-vehicle time (one θ, stable at
  0.015–0.05 per minute across held-out folds) improves case 06 a little
  (with the split: rmse(log) 0.718 → 0.711–0.714, r 0.68 → 0.71, RMSE 88.7 →
  85.4–85.9) but makes trips *shorter* (0.78).
- **The counts.** Loads conserve on and off to within 0.6% for eight of the
  nine operators. (TransPennine's do not: its alightings total 45% of its
  boardings, so its on/off counts should not be used for anything.)
- **Interchange stations.** Predicted on+off at the 533 stations is 1.07×
  counted, flat across table 1410's interchange-share bands.

### 3.2 What is left: journeys that change trains, and the Elizabeth line's codes

The model assigns only pairs a single train serves; a journey needing a change
never reaches any train. For operators whose trains run into London that loses
long legs — for GWR, 0.97m a year from the Windsor, Henley, Guildford and
Wokingham branches into Paddington alone, plus cross-London journeys. It fits
the operators with *fewer and shorter* trips (GWR, Southeastern, SWR).

Separately, the timetable books the Elizabeth line's central calls as WHX,
LSX, PDX, FDX and ABX, where the ODM has Whitechapel (ZLW), Liverpool Street,
Paddington, Farringdon (ZFD) and Abbey Wood (ABW). Elizabeth line trains
therefore serve none of those pairs in the model: where another operator does
(Romford or Stratford → Liverpool Street, Slough → Paddington) it gets all the
demand, and where no one else does (anything to Farringdon) the journeys are
dropped. Recoding them changes the train count on only 0.2% of case 06's legs,
and improves Greater Anglia a little (rmse(log) 0.862 → 0.855) — a fix worth
making, not the explanation.

**So the next structural step is a transfer assignment**: route each pair no
single train serves through its best interchange and load both legs.

### 3.3 The tidal split, tested on trains it was not fitted on

Case 06's split was fitted on case 06's trains. The new-station cases count
some 42,000 segments on other trains (the ones that were in case 06's sample
are dropped), so running them with the split on *every* leg is the first large
out-of-sample test of it on existing passengers. Section 4 has the results
after the fixes in §1.1–1.3; before them, the split improved the shape of the
error on six of nine cases (rmse(log) −0.12 to −0.17) but raised the level
everywhere, which is what §1.1 and §1.3 turned out to be doing.

### 3.4 A station-function term in the split

Footfall says how big a station is, not what it is for, so the split points
every small station's morning demand at the big one — right at Beaulieu Park,
wrong at Cambridge South's biomedical campus. `DirectionSplitModel` now takes
an optional per-period term on `g_o − g_d`, with `g = ln(catchment ÷ jobs within
a mile)` (SDFT's catchment on the May 2026 network; `_sdft/station_function.py
--catchment`). It is antisymmetric, so pair totals are still conserved, and old
model files load and behave exactly as before.

| split keyed on (case 06, held out) | routes, seed 0 / 1 | operators |
|---|---|---|
| footfall ratio (shipped) | 0.726 / 0.729 | 0.726 |
| footfall + residents ÷ jobs within 1 km | 0.734 / 0.731 | 0.732 |
| footfall + catchment ÷ jobs within a mile | **0.718 / 0.723** | **0.721** |
| either function measure alone | 0.751–0.756 | 0.753–0.754 |

Replayed on the new-station cases it takes Cambridge South's morning boarding
share from 0.61 to 0.42 (counted 0.16) and its per-train correlation on
boardings from 0.36 to 0.54, lifts the Northumberland Line's per-train
correlation from 0.23 to 0.41, and costs a little at Beaulieu Park (0.80 →
0.73 against 0.97). Model file:
`cases/_baselines/direction_split_model_segments_function.json` (fitted by
`fit_direction_split_segments.py --function`).

## 4. Results after the fixes

Every hindcast that reads the SDFT footfall or has London legs was re-run with
all the fixes above: cases 01, 03, 04, 05 and 11–15, 58 runs in 69 steps, none
failed. The outputs they replace are in `cases/_superseded/20260927_*/`, each
with a README.

### 4.1 New-station totals

New-station legs never take a substituted base (§1.2), so the station totals
moved only with the SDFT forecasts (§1.4–1.5): Brent Cross West +36% (995k →
1.36m), Cambridge South +32%, Marsh Barton +37%, Beaulieu Park +23%, Thanet
Parkway +9%; Ashington −17%, Seaton Delaval −11%.

| station, balanced model | before | after | after, with build-up |
|---|---|---|---|
| Northumberland Line, six pooled | 1.25 | 1.13 | 0.81 |
| Cambridge South (Greater Anglia counts) | 1.57 | 2.07 | 1.05 |
| Brent Cross West | 1.08 | 1.48 | 1.34 |
| Thanet Parkway | 2.45 | 2.67 | 2.52 |
| Headbolt Lane | 2.69 | 2.72 | 2.50 |
| Beaulieu Park | 0.82 | 1.00 | 0.63 |
| Marsh Barton (8 train-days) | 0.26 | 0.35 | 0.34 |

| balanced model, after (before) | median | sd(log) | within ±25% | within ×2 |
|---|---|---|---|---|
| 12 stations, as reported | 1.12 (1.10) | 0.57 (0.62) | 50% (42%) | 67% (75%) |
| 12 stations, year-consistent | 1.31 (1.34) | 0.57 | 42% (25%) | 67% |
| 12 stations, year-consistent + build-up | 0.90 (0.92) | 0.64 (0.68) | 25% | 75% (75%) |
| 7 schemes, as reported | 1.48 (1.25) | 0.71 (0.79) | 29% | 43% (57%) |
| 7 schemes, year-consistent + build-up | 1.13 (1.04) | 0.75 (0.83) | 29% | 57% (57%) |
| *UK scheme appraisals, SDG 2010 (n = 23)* | *0.94* | *0.61* | *52%* | *74%* |

Resampling schemes, the 90% interval on the station median is now 1.00–2.67
(was 0.82–2.45) and on the station sd(log) 0.33–0.91 (0.35–0.99).

By station the spread narrowed a little and the median did not move. By
scheme the model now over-predicts by about half, and three of seven schemes
are within a factor of two. **The comparison with scheme appraisals is still
not shown.** SDFT's own jobs definition made the two stations in large job
centres worse (Brent Cross West, Cambridge South) and the two outer ones
better (Beaulieu Park, Marsh Barton). That fits §2.1: the port's errors belong
to particular kinds of station, and no single input setting improves them all.

Three stations need a reading:
- **Cambridge South.** Its 2.07 is mostly its first five weeks of operation:
  with PDFH build-up it is 1.05.
- **Thanet Parkway.** Its 2.67 is two factors multiplied: SDFT forecasts 1.49
  times ORR's 2024-25 figure, and Southeastern's counts come to about 55% of
  ORR's weekday-equivalent usage.
- **Headbolt Lane.** Its 2.72 is Northern's modelled share of the station and
  its passengers changing to Merseyrail (`LITERATURE_BENCHMARK.md` §3.3).

### 4.2 Whole-train loads

Segment loads, calibrated joint mode; each superseded run against its re-run.

| case (counted operator), balanced | ratio of totals | rmse(log) |
|---|---|---|
| 11 Brent Cross West (Thameslink) | 0.66 → 0.71 | 0.798 → 0.778 |
| 12 Thanet Parkway (Southeastern) | 0.53 → 0.55 | 0.850 → 0.833 |
| 05 Cambridge South (Greater Anglia) | 0.62 → 0.64 | 0.786 → 0.769 |
| 05 Cambridge South (GTR) | 0.52 → 0.53 | 0.854 → 0.861 |
| 15 Beaulieu Park (Greater Anglia) | 1.00 → 1.01 | 0.883 → 0.885 |
| 04a Maidstone East → Blackfriars (Southeastern) | 0.17 → 0.18 | 1.678 → 1.641 |
| 01 Northumberland Line (Northern) | 0.86 → 0.75 | 0.750 → 0.750 |
| 03, 13, 14 | within 0.5% | unchanged |

- **London-bound trains.** The substitution fix (§1.2) lifts them by 2–7%,
  about what its exposure measurement said. It also cuts their error a little,
  except on GTR's Cambridge South trains.
- **Northumberland Line.** Its trains lose 14% with the lower SDFT forecasts.
- **Case 04a.** It is still at a fifth of its counted load. That shortfall,
  diagnosed in `MODEL_FIXES.md`, is not something these fixes touch.

**The whole-train level is still low almost everywhere:** 0.53–0.75 on the
London operators' trains and on the Northumberland Line. This is §3.1's
short-trip shortfall, showing up on trains outside case 06.

### 4.3 The tidal split on 38,653 segments it was not fitted on

`cases/_results/split_variant_eval.py`: segment loads, calibrated joint mode,
with the trains in case 06's sample dropped. Each cell gives rmse(log) and
Pearson r.

| case (operator) | segments | balanced | split, new station's legs | split, every leg | every leg + function |
|---|---|---|---|---|---|
| 01 Northumberland Line (Northern) | 2,095 | 0.750 / −0.05 | 0.664 / 0.29 | 0.664 / 0.29 | 0.653 / 0.31 |
| 03 Lowestoft–Cambridge (GA) | 235 | 0.419 / 0.89 | 0.419 / 0.89 | 0.533 / 0.91 | 0.508 / 0.89 |
| 05 Cambridge South (GA) | 6,863 | 0.750 / 0.57 | 0.741 / 0.58 | 0.661 / 0.81 | 0.681 / 0.80 |
| 05 Cambridge South (GTR) | 2,984 | 0.861 / 0.56 | 0.856 / 0.56 | 0.769 / 0.65 | 0.800 / 0.64 |
| 11 Brent Cross West (Thameslink) | 15,203 | 0.779 / 0.44 | 0.778 / 0.45 | 0.721 / 0.57 | 0.721 / 0.60 |
| 12 Thanet Parkway (Southeastern) | 4,669 | 0.833 / 0.41 | 0.820 / 0.43 | 0.803 / 0.65 | 0.828 / 0.68 |
| 13 Marsh Barton (GWR) | 73 | 0.841 / 0.39 | 0.841 / 0.38 | 0.864 / 0.34 | 0.874 / 0.32 |
| 14 Headbolt Lane (Northern) | 2,860 | 0.735 / 0.54 | 0.750 / 0.50 | 0.693 / 0.70 | 0.669 / 0.73 |
| 15 Beaulieu Park (GA) | 3,671 | 0.888 / 0.38 | 0.874 / 0.41 | 0.567 / 0.77 | 0.542 / 0.76 |
| **pooled rmse(log) / median r** | 38,653 | 0.792 / 0.44 | 0.783 / 0.45 | **0.706 / 0.65** | 0.711 / 0.68 |

- **The gain is in splitting existing passengers' legs.** Pooled rmse(log)
  goes 0.792 → 0.706 and the median r 0.44 → 0.65. rmse(log) is better in
  seven of nine runs and r in eight, across five operators. The split has five
  slopes, fitted on 270 other trains, so this cannot be memorisation. It is the
  first evidence that case 06's main finding, that the route effect is mostly
  tidal, holds on other trains.
- **The level barely moves:** −1.5% to +5% on counted segments, apart from case
  03 (§4.4).
- **The station-function term does not carry over to whole trains.** Pooled it
  scores 0.711 against 0.706. It helps 01, 14 and 15 and hurts both case 05
  runs, 12 and 13. Keep it optional.
- **Per-train totals move less than segments.** The split mostly fixes *where
  along the train* the load sits, which a Thameslink train running through
  London needs most. Case 11's train-level r goes 0.48 → 0.46 while its segment
  r goes 0.44 → 0.57.

### 4.4 Case 03: four trains, all running in the peak direction

With the split on every leg, case 03's predicted load rises 36% across all 235
segments, counted or not. This is not a bug: the four stripped trains all run
in the peak direction.
- **Inward.** The 05:38 and 06:34 from Lowestoft go to Cambridge. Ely →
  Cambridge in the morning rises 41%.
- **Outward.** The 20:46 runs from Stansted Airport back to Lowestoft.
  Cambridge → Ely in the evening rises 37%.
- **Opposite direction.** Those flows ride other trains, which the split lowers.

The split conserves each pair's total in each period. A sample made only of
peak-direction trains therefore gets the whole of the tide.

Per train, the picture is the one recorded on 22 September, before any of
these fixes:
- the 06:34 goes from 0.79 to 1.03 and the 20:46 from 0.77 to 1.03, so both
  are corrected;
- the 05:38 goes from 1.17–1.22 to 1.69–1.76, so it is over-boosted.

The suspect is the early-period slope (−0.63), learnt mostly from London
commuter trains. Four trains cannot settle it; the national check in §5 can.

## 5. What to do next

1. **A national base-year check: the larger hindcast.** Scaling up the
   strip-and-re-add hindcasts will not find the whole-train error. That error
   is in how the matrix is loaded onto trains, and case 06 already shows it in
   sample. The stripping machinery is also where all three bugs in §1.1–1.3
   lived. The check:
   - assigns the full ODM to the full May 2026 timetable, with every train
     present;
   - compares every counted segment and station call of the ten operators with
     count feeds, about 60% of weekday trains;
   - needs a vectorised assignment (about a million legs per weekday), which
     then runs in minutes;
   - must first reproduce case 06's 270 trains, apart from the known n/(n+1)
     factor.

   It gives enough data to split the error by corridor, operator, period,
   direction, distance and trip length, and a TAG M3.2-style base-year figure
   for the paper. The fixes to test on it, in this order:
   - a transfer assignment (§3.2);
   - the Elizabeth line codes;
   - the split on every leg, with its slopes refitted, with and without the
     function term (§4.3–4.4);
   - journey-time train choice (§3.1);
   - origin-station period profiles.
2. **Guardrails for (1).** Errors are shared within a corridor (split-half r =
   0.95 on case 06), so the unit of evidence is the corridor, not the
   train-day:
   - hold out whole corridors when testing a fix;
   - fit no per-corridor factors, which would only memorise the answer;
   - screen the counts systematically (GTR's restarts, TransPennine's on/off);
   - fix the candidate fixes and the metrics before looking at the results.
3. **An event study of real timetable changes**, for forecasting. Every
   service change between counted timetables (December 2023 to May 2026) is a
   hindcast against a real before-timetable, not a synthetic strip. That tests
   frequency response, new direct links and extensions on hundreds of changes
   instead of six trains. It needs the December 2024 and May 2025 archive CIFs
   (§6).
4. **Use the split on every leg** wherever a hindcast uses the split
   (`direction_split_legs: all`), with the footfall-only model (§4.3). Refit
   its slopes on (1) rather than on case 06's 270 trains.
5. **Re-run cases 06 and 09 with the fixes, and refit the split.** §1.2 moves
   case 06's London legs (GTR +4.7%, Southeastern +3.7%), and the shipped
   split was fitted on the old figures. Replace case 09's §3 figures.
6. **Short-term schedules (§1.5).** Measure how much they affect era-matched
   loading (`timetable_eras`). Change the core `stp_indicator == "P"` filter
   only alongside (1), since it moves every May 2026 result.
7. **Write-ups.** `LITERATURE_BENCHMARK.md` is revised with these results.
   `paper/PAPER.md` still:
   - quotes "16%" (line 36), and §4.2 describes the split fitted on
     first-stop boardings (morning slope +0.08, case 07); the segment fit
     (morning −0.46, case 09 offline, §4.3 here) replaced it on 22 September;
   - says Cambridge South opened in January 2026 (lines 248 and 299); it opened
     on 28 June 2026;
   - does not cover cases 11–15;
   - lists "no substitution between destinations" as the largest gap (line
     369);
   - carries recommendations in §6 that are now mostly done or superseded.
8. **Tidy-up.**
   - Delete `cases/_superseded/` (about 4.6 GB) once these results are
     accepted.
   - Re-run or remove case 04's sensitivity folders (`_diluted`, `_theta010`,
     `_theta015`), which predate the fixes.

## 6. Data that would help

| data | what it unblocks |
|---|---|
| Full CIF archives for December 2024 and May 2025 | the event study (§5.3); the versions built from those two eras' working timetables have known faults |
| Merseyrail counts, at least Kirkby–Headbolt Lane | Headbolt Lane is counted only on Northern's trains, which carry a minority of its passengers; Merseyrail coverage for §5.1 |
| How the ODM allocates "London Terminals" tickets to stations | destination substitution, and every London leg |
| Any directional O-D data: NRTS, LENNON by ticket type, MOIRA | the ODM has no direction, so the split is fitted from counts alone |
| Route allocations for journeys that change trains (MOIRA or ORCATS) | checking a transfer assignment (§3.2) |
| TransPennine's count definitions | its alightings total 45% of its boardings |
| GTR's counter behaviour at Cambridge South | the restarts that make those counts unusable |

## 7. Addendum, 28 September: the national check (§5.1) is built

Case 16 (`datasources/cases/16_national_base_year/`, `PLAN.md` then
`RESULTS.md`) assigns the ODM to every train in the May 2026 timetable with
`ortla.network_assignment`, and scores it on 490,537 counted
segments of nine operators. It reproduces case 06 exactly, leg for leg. Three
things above change.

1. **§3.1 is corrected: trips are too few *and* too short.** With every
   train present the level is 0.70, not case 06's 0.88. Boardings are 0.81 of
   the counts and trip length 0.86. Case 06's boardings looked right only
   because stripping a train prices it over n trains instead of n + 1.
2. **§3.2's leads, tested at scale:**
   - **Transfer assignment** (one change, the model's 20-minute penalty) is
     the biggest single gain found: held-out rmse(log) 0.898 → 0.754, level
     0.69 → 0.84, boardings 0.79 → 0.98, every operator better. It does not
     lengthen trips (0.88 → 0.87), so the short trips have another cause.
   - **The Elizabeth line codes** are one case of a wider fault. Twenty
     timetable codes are second codes of ODM stations, so their legs find no
     matrix cell and their pairs' trains are split between the two codes.
     Reading platform 4 (`RDZ`) carries every SWR and North Downs train.
     Recoding them helps a little (0.906 → 0.898), five folds of five.
3. **§4.3's split holds nationally, with gentler slopes at the ends of the
   day.** On held-out corridors it takes rmse(log) 0.754 → 0.676. Refitted
   nationally, the early slope is −0.42 against case 06's −0.63, and the
   evening +0.31 against +0.47. That explains case 03 (§4.4). The national fit
   is `cases/_baselines/direction_split_model_national.json`.

Rejected on held-out corridors:
- moving a slot's orphaned demand to the nearest period with trains;
- journey-time train choice, which improves the shape but pulls the level
  down by 0.035.

The second verdict stands as registered, but the level test does not fit that
fix. Journey-time choice only moves each slot's trips between the slot's
trains, so it cannot lose demand. The level falls because trips move onto
faster trains, which cross fewer counted segments, and onto operators without
counts. So equal sharing flatters the 0.84 below; under journey-time choice it
is 0.80.

With every adopted fix the national figures are:

| | before | after |
|---|---|---|
| level | 0.70 | 0.84 |
| rmse(log) | 0.91 | 0.67 |
| r | 0.50 | 0.65 |
| GEH < 5 | 59% | 70% |

**What is left in London is Southeastern's, not the whole London Terminals
group** (corrected 28 September; this section first read the group as erring
in opposite directions everywhere):
- **Southeastern: the ODM books too many trips to London Bridge.** On the same
  trains, the model sets down at London Bridge 66% of the load arriving there
  on Cannon Street trains (41% counted) and 56% on Charing Cross trains (35%).
  Before any fix it was already 60% and 48%, so most of the excess is the
  ODM's; the transfer assignment adds 7–8 points by routing changes through
  London Bridge. Waterloo East → Charing Cross is right (24% against 22%).
- **SWR: not an allocation effect.** SWR's feed never counts Vauxhall →
  Waterloo (none of 1,830 segments). "Into Waterloo" (0.48) is therefore only
  fast trains arriving non-stop from Woking (0.46), Surbiton (0.62) and
  Basingstoke (0.38), and "into Vauxhall" (1.19) is mostly the Windsor lines
  (Queenstown Road → Vauxhall 2.49; Clapham Junction → Vauxhall 1.03). The ODM
  books 92–94% of Woking, Basingstoke and Winchester's morning-peak London
  trips to Waterloo itself. Fast trains under and stopping trains over is what
  sharing a slot equally across its trains produces.
- Liverpool Street (1.03) is not over, and Manchester's members show no split.

So the ODM's London Terminals allocation (§6) is a smaller data ask than it
first looked: Southeastern's error can be measured from its own on/off counts,
and SWR's is not an allocation effect. The model changes these results call
for are:
- a transfer assignment in the per-train path;
- the second codes as station aliases;
- the national split as the shipped model.

All three were made on 28 September (§7.1).

### 7.1 The fixes in the model, checked leg for leg

What changed (`ortla`; the first two are on by default):
- **Second codes.** `station_aliases.DEFAULT_STATION_RECODES` holds the twenty
  second codes. They are rewritten in the timetable, and in any count feed
  matched to it, as soon as each is read (`station_recodes`; `{}` turns it
  off). Unlike an alias, this makes trains at both codes count as one
  station's trains.
- **Transfers.** `network_assignment.TransferDemand` routes every pair with no
  direct train through its best single change, with case 16's detour limit
  (`transfer_assignment`). Routing uses the full timetable, the one after the
  change, not the stripped one. The routed trips join each leg's cell before
  the split and the frequency response. `hindcast_demand_by_leg.csv` records
  them as `existing_transfer_trips`, and `hindcast_transfer_routes.csv`
  records what was routed each day.
- **Composition.** The split's factor comes from the pair's own cells and
  scales the routed trips as well, as case 16 applied it. Destination
  substitution still compares raw ODM cells, and the routed trips join its
  result afterwards, so §1.1's bug cannot return by this route.
- **The national split.** `make_configs.py` points every split variant at
  `direction_split_model_national.json`, except the `_fn` ones, which keep
  case 09's fit. Each launch-case hindcast without SDFT footfall gains a
  `_split_all` variant.

**Checked leg for leg.** Case 06, rerun through the per-train path, prices
all 185,438 of its legs as case 16's own code does. The routed trips, the
demand, the train count and the captured trips agree on every leg
(`16_national_base_year/check_port.py`). There are five new tests, 77 in all.

Case 06's in-sample level rises from 0.88 to 1.11. That is expected. Each
stripped train is still priced over the trains left in its slot, n rather
than n + 1 (§7, item 1), and the routed trips come on top. Case 16 measures
the level; case 06 now checks the code.

**A second double count, not fixed here.** The older aliases in
`station_aliases` (`SPL`/`SPX` → `STP`, `WJL` → `WIJ`, ...) are lookups only,
so trains at the two codes are counted separately. Where both serve a pair in the same slot,
each gets the whole cell. That happens in 69 of Wednesday's 316,776 slots,
and the one that matters is Bedford → St Pancras: 13 Thameslink trains at
`SPL` and 5 EMR trains at `STP` in the morning peak each take the full cell.
Moving them to the recodes is the fix, to be judged on held-out corridors as
the second codes were.

### 7.2 Every case rerun with the fixes

All 68 hindcasts were rerun (`run_cases.py --force --phase hindcast`), none
failed. `_results/compare_before_after.py` scores every run against its
predecessor in `_results/before_20260928/`. Segment loads, calibrated joint
mode, with case 06's trains dropped.

**Out of sample, pooled (the new-station cases' 38,653 counted segments):**

| variant | level | rmse(log) | r |
|---|---|---|---|
| balanced, no split | 0.70 → 0.85 | 0.792 → 0.752 | 0.49 → 0.49 |
| split on every leg | 0.72 → 0.86 | 0.706 → 0.650 | 0.66 → 0.66 |

The fixes lift the level and cut the error, as nationally, but do not improve
the correlation. By case, they help where the model was low and push past 1
the cases already at or above it:
- **Better:** Cambridge South, Greater Anglia (rmse(log) 0.661 → 0.580) and
  GTR (0.769 → 0.679); Brent Cross West (0.721 → 0.653).
- **Worse:** Headbolt Lane (level 1.19 → 1.30); Beaulieu Park (0.567 →
  0.617); and both clean hindcasts. Case 02's two trains go from 0.955 to
  1.044, and to 1.211 with the national split on every leg. Case 03's four go
  from 0.86 to 1.10: transfers alone put its 06:34 and 20:46 almost exactly
  right (1.01 and 0.95), but the 05:38 goes from 1.2 to 1.5, and adding the
  national split puts all four over (2.1 for the 05:38). Both cases' trains
  all run with the tide.
- **Level right, shape wrong:** Thanet Parkway goes from 0.54 to 1.01, but its
  r falls from 0.65 to 0.52: London-bound morning trains are now at 2.12 of
  their counts. The reason is below.
- **Unchanged:** new-station totals (the fixes never touch a new station's
  own legs), the Northumberland Line and Marsh Barton. Case 04a is still at
  0.20.

**The reference date is missing trains, and transfers now reroute their
demand.** The timetable is one extract, taken on 11 September, and schedules
that ended before then are not in it. On the reference date (Wednesday 17
June) it holds 22,187 trains, against 23,039 on 16 September:
- SWR is at 0.89 of its September count, London Overground 0.90,
  Southeastern 0.91 and Merseyrail 0.91.
- High Speed 1 has one train, at 23:40. Its weekday schedules in the extract
  start on 7 September.

Before this change, a pair whose only direct service was missing lost its
demand. Now that demand is routed through a change onto other lines. Of all
routed trips, 8.1% (1.9m a year) are for 638 pairs that have a direct train in
September but not on 17 June: Ashford, Canterbury West and Folkestone West to
St Pancras via London Bridge, and the Overground's Lea Valley lines via
Cheshunt. That is an upper bound, since some September links may be new. It
barely moves the national level, but it dominates Thanet Parkway's London-bound
trains, and it feeds some of the London Bridge set-down excess the transfer
step added in §7. The fix is a timetable for the count window, a CIF extract
from June 2026, not a model change.
