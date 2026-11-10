# Review of the literature research pack, 29 September 2026

`MODEL_LITERATURE_RESEARCH_PACK(1).md` (dated 29 September)
proposes a schedule-based train-choice layer as the model's next change. This
note checks the pack three ways:
- against the model as it now stands;
- against the papers it cites, all of which are in `Transport for the North/lit`;
- against case 16's 490,537 counted segments.

The new measurements are in section 4. Their scripts are in
`datasources/cases/16_national_base_year/lit_checks/` (section 7).

## 1. Verdict

The pack's advice on method is sound:
- separate demand from assignment;
- hold out whole corridors;
- never tune on the trains being scored;
- do not treat the ODM as truth;
- report level and shape together.

Its central hypothesis does not hold. It says equal sharing between trains is
the largest remaining error. On case 16 it is the smaller part:
- **The ceiling is low.** Suppose every train got its counted share of its
  station, direction and period group. rmse(log) would fall from 0.663 to
  0.563, and no train-choice rule can do better than that.
- **Timetable features explain little of it.** The features a schedule-based
  model uses remove about 3% of that within-period error on held-out corridors.
  Departure time within the period removes 7–9%.
- **Most of the error is out of reach.** 70% of the remaining squared error is in
  how much flow passes a station, in one direction, in one period. No
  train-choice rule can move that. If those levels were perfect, rmse(log) would
  be 0.363.

The pack also does not know what the model already does (section 2). Several
of its highest-ranked sources support less than it claims (section 3).

## 2. What the pack gets wrong about the model

| the pack says | the model now | where |
|---|---|---|
| each OD flow is shared equally across its eligible trains | within a (pair, period) slot, yes. Before that, demand is split by direction (the national tidal split), and pairs no train serves are routed through one change | case 16, fixes 1–3, adopted 28 Sept |
| Experiment 1: replace equal sharing with a journey-time logit | done as case 16's fix 4 (θ = 0.03 per minute). rmse(log) 0.676 → 0.660, level −0.035; rejected as registered | `fix_split_choice.py` |
| constrain ODM station totals to external station totals | a no-op. Table 1410 *is* the ODM's marginals: 96.7% of stations agree within 0.1% once the years match | 22 Sept |
| check whether the ODM is directional | it is exactly symmetric, 1,597,996 of 1,597,996 pairs | case 06 |
| Experiment 7: within-pair frequency response | done: 0.05–0.06, against the cross-sectional 0.64 | `scripts/frequency_panel.py` |
| probabilistic station catchments for new stations | SDFT's choice model is ported. On 32 held-out new stations it is at appraisal level | `_sdft/sdft_footfall.py` |
| no machine learning, no per-corridor constants | agreed, and tested: R² 0.85 falls to 0.30 once the route is held out | case 06 |
| set success thresholds before calibrating | TAG M3.2 already gives them (section 4.5) | |

## 3. What the cited literature actually says

**Cascetta & Coppola (2016)** is the pack's main evidence.
- **Against on-board counts, the gain depends on frequency.** Where trains run
  two to five an hour, the schedule-based model beats the frequency-based one
  only a little: RMSE% 0.280 against 0.308. At 0.5–1 an hour it wins by a lot:
  0.246 against 0.436.
- **The ±40–60% the pack quotes is model against model.** It compares the two
  models on timetables the authors altered artificially, not on counts.
- **It needs data we do not have.** The schedule-based model needs each
  traveller's desired departure time. They had 3,237 surveyed ones.
- **Their conclusion points away from the pack.** They find frequency-based
  accuracy depends mainly on how evenly the trains are spaced. The demand
  profile within the period matters only "marginally".
- **Their frequency-based model is richer than equal sharing.** It prices each
  service type separately.

**TAG M3.2 (May 2024)** supports timetable-based assignment for low-frequency
and long-distance services (§2.7.4, B.1.5). Its B.1.8 describes our equal
sharing exactly, and says it fails where headways are irregular. It also says
three things the pack leaves out:
- **Matrix validation uses grouped counts.** Screenline totals should be within
  15% for 95% of screenlines (table 4). An individual service is held to within
  25% only on links carrying more than 150 passengers an hour (table 5).
- **Base years may use matrix estimation.** It is allowed, within limits (§8.2),
  to meet those criteria. A forecasting model that refuses matrix estimation
  should not expect to meet them, and should say so (section 4.5).
- **The GB industry has its own standard approach.** TAG names it (MOIRA, and
  PDFH's rooftop model for infrequent services) but does not describe it. The
  pack never says that GB practice is *incremental*: MOIRA pivots off observed
  ticket sales.

**ORR's quality and methodology report (December 2025)** confirms four things
that bear on the model:
- **London Terminals allocation.** Tickets to "London Terminals" have been
  allocated to the member stations from ORCATS modelled routes (the Central
  Allocations File) since 2022-23. Before that, the 2001 LATS survey was used.
  This is a documented mechanism for Southeastern's London Bridge excess (case
  16 RESULTS §6).
- **Split tickets.** From 2022-23, a journey made on split tickets is counted
  once, end to end.
- **Season tickets.** An annual season is converted to 480 journeys a year.
  The 2019 DfT/Steer study found 378.
- **Interchanges.** ORR publishes interchanges for every station, estimated from
  the same Central Allocations File. That gives the transfer assignment a check
  that does not use the counts (section 4.4).

**Wheat & Wardman (2017) and Wardman (2022)** measure how much demand changes
with journey time, headway and interchange. That is the pivot, not which train
people pick. Two of their numbers matter here:
- **Interchange penalty.** Freely estimated, it is about 100 minutes on journeys
  of about 100 miles. They recommend PDFH's values plus 75%; PDFH's are 19, 40
  and 65 minutes at 30, 100 and 200 miles for non-season tickets. The model uses
  a flat 20 minutes. That does not matter yet: a change route is offered only
  where no direct train exists, and the penalty is the same at every station, so
  it never decides a choice.
- **Frequency penalty.** Their PDFH values (0.70–0.90 of a 30-minute headway,
  0.45–0.65 of an hourly one), with a GJT elasticity of about −1.1, imply a
  frequency elasticity of roughly 0.15–0.25. That is for going from hourly to
  half-hourly on a 30–60 minute journey; the figure is rough arithmetic.
  Our within-pair 0.05–0.06 is below that range, and the cross-sectional 0.64
  far above it. That is worth a sentence in the paper.

**Cordera et al. (2018)** studied one 23-station regional line and 26,371
passengers. The doubly constrained gravity model fits best, but that is
mechanical when both margins are observed. The idea that transfers is Poisson
(PPML) estimation, for two reasons:
- it predicts the mean directly, so no retransformation factor is needed (our
  Duan smearing factor is 2.9);
- it keeps pairs with zero flow.

**The other sources:**
- **RTRI (2016)** estimates the choice between fast and slow trains from 775
  surveyed passengers, and reports hit rates, not train counts.
- **Burger et al. (2023)** is an online experiment on crowding and switching
  trains, not a source of allocation parameters.
- **Chen, Cheng & Sun (2025)** estimate one bus route's OD for each trip from
  boarding and alighting counts. For us it could only be a diagnostic: the
  counted load already fixes passenger-kilometres, and the split between pairs
  comes from the prior.
- **Spiess & Florian, Nguyen & Pallottino, and Florian & Constantin** describe
  frequency-based "strategy" assignment for urban transit. It fits London's
  turn-up-and-go links, not timetabled rail.
- **Flyvbjerg et al.** The pack says their *Transport Reviews* paper was
  retracted. Nothing in the folder confirms it; check before citing it either
  way.

**Missing from the pack:**
- MOIRA and ORCATS themselves. The ORR/LeighFisher open-access report the pack
  lists is not in the folder.
- PDFH's rooftop model.
- Incremental forecasting as the GB standard.
- Off-peak ticket validity. TAG §3.3 would count it as a fare that affects train
  choice.

## 4. New measurements on case 16

All figures are on the adopted-fixes variant (`codes_transfers_split`): 490,537
counted segment-days. Averaging each train-segment over its weekdays gives
117,449 train-segments.

### 4.1 Where the remaining error sits

Day-to-day variation is 11.7% of the squared log error of segment-days. With the
days averaged out (train-segments, rmse(log) 0.663), the rest splits as follows:

| level | share of squared log error | rmse(log) if this level were perfect |
|---|---|---|
| overall level | 5.5% | |
| station × direction (from station, train's destination) | 43.1% | 0.475 |
| period within that | 21.4% | 0.363 (together with the row above) |
| trains within one station, direction and period | 30.0% | 0.563 |

For comparison, a perfect scale factor per corridor (operator × the train's
ends) would give 0.496, and per corridor × period 0.447.

### 4.2 Train choice within a period: the pack's H1

The test covers 19,064 station × direction × period groups with two or more
trains, which hold 94% of train-segments. Each feature a schedule-based model
would use was fitted on four fifths of the corridors and scored on the rest:

| feature | within-group error removed, held out |
|---|---|
| journey time to the train's destination, against the group's fastest | 1.8% |
| departure-time catchment (the window of desired departures a train owns) | 0.1% |
| rooftop GJT share (desired departure spread evenly; cost = displacement × a + journey time; logit), best of 12 settings | 3.3% |
| half-hour of departure within the period | 7.4% |
| half-hour × London-bound | 8.7% |

**Timetable spacing does not explain the error.** The catchment does not
correlate with the counted shares at all (+0.004), so spacing is not what makes
one train busier than its neighbours.

**Timing within the period does, a little:**
- trains at the end of the peaks are over-predicted: am 09:00–09:30 by +0.21,
  pm 18:30–19:00 by +0.21;
- trains just before the peaks are under-predicted: 06:30 by −0.24, 15:30 by
  −0.20.

That pattern comes from demand bunching in the middle of the peak and from
off-peak validity boundaries, not from timetable spacing. It is case 10's
departure-time weighting, not the pack's schedule-based layer, and case 10
showed how thin slots break it.

On this evidence a schedule-based layer could gain about 0.01–0.02 of
rmse(log). It is not the next thing to build.

### 4.3 Time of day: 21% of the error, and hard to predict ex ante

Held out, a mean per period removes 7% of the period error. Adding London-bound
takes it to 10%. Season-ticket share at either end adds nothing, and station
size adds 1%.

This repeats case 06's finding: nothing known for a new station predicts its
time-of-day profile. The origin station's own profile explained 30% there, but
it exists only for stations that already exist.

### 4.4 The transfer assignment piles interchange onto a few hubs

Predicted ÷ counted, on the five operators whose on and off counts conserve:

| station | alighting, before transfers | adopted | boarding, before | adopted |
|---|---|---|---|---|
| Clapham Junction | 0.57 | 1.60 | 0.55 | 1.37 |
| London Bridge | 0.87 | 1.38 | 0.90 | 1.27 |
| Farringdon | 0.75 | 1.45 | 0.70 | 1.22 |
| East Croydon | 0.92 | 1.26 | 0.92 | 1.26 |
| Highbury & Islington | 1.50 | 1.98 | 1.42 | 1.84 |
| Gatwick Airport | 1.06 | 1.30 | 1.09 | 1.33 |

The summed on/off error across all stations is unchanged: 783,000 a day before
transfers, 786,000 after. Transfers restored the missing trips (boardings 0.79 →
0.98) but put their changes in the wrong places.

**Against ORR's published interchanges** (table 1410, 2022-23), Wednesday's
routes:
- carry 154m changes a year against ORR's 213m;
- correlate with ORR at 0.58 in logs, over the 200 stations with more than
  100,000 interchanges;
- put 34% of changes at the five busiest vias, where ORR has 25%.

**Three causes are visible:**
- **The rule is all-or-nothing on the day's fastest times.** Each unserved pair
  gets one via, the station with the fastest trains of the day, whenever those
  trains run.
- **There is no Underground.** A cross-London journey by Tube has no route in
  the model. So changes at the London termini come out near zero: Waterloo,
  Euston, King's Cross, Paddington and Victoria get 0–0.5m against ORR's 2–5m
  each. The model sends those journeys via Farringdon (8.2m against ORR's 0.64m),
  Whitechapel (9.5m) or Clapham Junction instead.
- **The reference date is missing trains.** Cheshunt carries 4.1m changes
  against ORR's 0.23m, because the Overground trains that should carry those
  journeys are missing from the timetable extract.

Clapham Junction's national total is about right (18.4m against 19.1m) even
though its counted trains come out 1.6 times over. So the error there is in
which trains and which pairs carry the changes, not only in how many there are.

**Correction, later on 29 September** (case 17, `NEXT_INVESTIGATION_REVIEW_2026-09-29.md`).
Two of the causes above are wrong:
- **The ORR comparison used the wrong year.** It used 2022-23, whose Central
  Allocations File lacks Elizabeth line interchanges (Whitechapel 579). ORR's
  2024-25 table, the ODM's own year, has Farringdon at 9.3m and Whitechapel at
  13.8m. So the model's 8.2m and 9.5m are close, not over-routing.
- **The Underground cause is much smaller than stated.** A change at a London
  terminus in ORR's figures is a National Rail change there, not a Tube
  transfer (Vauxhall and Old Street show none). And the 2024-25 ODM already
  splits contactless rail–Tube–rail journeys in two, and cuts journeys that
  start or end on the Underground to their National Rail part (Steer, 2025,
  §1.57–1.60). A Tube link can therefore matter only for through tickets
  across London.

Case 17 redoes this comparison against 2024-25.

### 4.5 TAG M3.2's criteria

| criterion (TAG table 5, a typical weekday) | case 16 | TAG asks |
|---|---|---|
| link-hours carrying at least 150 an hour (counted), within 25% | 38% of 20,506 | 95% |
| station × direction × period groups carrying at least 150 a day, within 15% | 26% of 12,963 | 95% |
| train-segments within 25% | 35% | – |

An uncalibrated model is a long way from a TAG base year. The paper should say
so, and give the reason: TAG base years reach these figures with matrix
estimation on the same counts.

## 5. What to do next

In order of what the evidence says each is worth:

1. **Close the reference-date gap first.** The rule set on 28 September still
   holds: adopt nothing more until a CIF from inside the count window is in.
   Cheshunt, Thanet Parkway and part of London Bridge are contaminated. *Done
   the same day, from a file already held: section 8.*
2. **Fix how the transfer assignment chooses routes.** Judge it against ORR's
   interchanges and hub on/off. Three changes:
   - a logit over the vias within some minutes of the best, instead of one via;
   - connections that fit the timetable (the earliest arrival through the via
     within the period), instead of the day's fastest times;
   - links between the London termini by Underground, with TfL journey times.

   ORR publishes interchanges by station every year, so they are a target that
   does not use the counts. This is the pack's H2, reached a different way.
3. **Within-period timing, not a schedule-based layer.** Use case 10's weights,
   fixed for thin slots (shrink a slot's weights towards equal as its train count
   falls), and add off-peak validity boundaries as a feature. Worth about 8% of
   the within-period error.
4. **The 43% at station × direction level is the ODM.** Southeastern's London
   Bridge share of London Terminals trips can be measured from its own counts
   (case 16 RESULTS §6). Elsewhere the fix needs directional or route data (the
   Central Allocations File, NRTS). Whether counts may be used to correct it
   depends on what the paper claims (question 1).
5. **Gravity branch: PPML instead of log-OLS with smearing.** It only matters for
   new stations and new links. It removes the 2.9× retransformation factor and
   keeps zero flows.
6. **Report the TAG criteria and an incremental comparison alongside
   rmse(log).** In a pivot, the base-year error should cancel out of the
   predicted change. A paper about forecasting should show that it does.

## 6. Questions, and Dillon's answers (29 September)

1. **What is the paper's primary claim?** All three: base-year loads,
   the response to a change in service, and new-station totals. The weight is on
   forecasting new services and new stations accurately.
2. **May time-of-day profiles come from the counts at existing stations?** Left
   to judgement. Decision: yes, for existing stations only, with three
   safeguards:
   - in a hindcast, only counts from before the change, and never the case's own
     trains;
   - on case 16, scored on held-out corridors;
   - a new station falls back to the profile it gets now.

   A new service between existing stations is the commonest forecast, and its
   stations' counted profiles are known before it runs, so using them is
   ex-ante information, not leakage. The work sits after the transfer routing
   in the queue (section 5).
3. **What data can be had now?** None of the June 2026 CIF, the Central
   Allocations File or MOIRA. But a June extract turned out to be held already
   (section 8).
4. **Is routing cross-London journeys by Underground in scope?** Yes.

## 7. Scripts

In `datasources/cases/16_national_base_year/lit_checks/`, writing to
`output/lit_checks/`:
- `oracle_decomposition.py`: section 4.1; it also attaches departure times to
  the segments;
- `within_period_features.py`: the timetable features in section 4.2;
- `extra_checks.py`: the half-hours in section 4.2, and sections 4.3, 4.4
  (hubs) and 4.5;
- `interchange_check.py`: section 4.4, against ORR's interchanges.

## 8. Addendum: the reference-date gap, closed from a file already held

Every case uses Wednesday 17 June 2026, but the model's timetable was the
daily feed's full extract of 11 September. That extract no longer holds
schedules that ended before it was taken. `timetable-data/various/May26
publsihed 240626.cif` is the May 2026 *published* timetable, extracted on 24
June, which is inside the count window. It had never been read.

**It holds the missing trains.** In the model's own filters on 17 June it has
23,024 trains against 22,187 (+837). The gains are SWR +175, London
Overground +161 and Southeastern +157; 147 trains call at Ebbsfleet
International against 1. All 105,144 of its schedules carry a blank STP
indicator, as a published extract does, and the reader takes blank as
permanent.
- Config: `configs/paper/eras/timetable_may26.yml`
  → `output_paper/timetable_may26/`.
- `run_national.py` gained `--timetable`, whose default is unchanged.
- Run: `output/may26_codes_transfers_split/`, the adopted fixes on the June
  timetable.

| on the 489,882 segments both runs score | rmse(log) | level | r |
|---|---|---|---|
| 11 September extract | 0.675 | 0.838 | 0.654 |
| 24 June extract | 0.677 | 0.800 | **0.729** |

**By operator:**
- **SWR improves most:** rmse(log) 0.879 → 0.841, r 0.415 → 0.571.
- **Southeastern:** r rises (0.573 → 0.628), but rmse(log) worsens (0.762 →
  0.810) as its level falls (0.869 → 0.777).
- **The restored trains** add 15,235 segments, 7,673 of them Southeastern's
  and 4,729 SWR's. They come out 1.31 times their counts.

**At the hubs:**
- **Cheshunt's boardings** go from 2.38 to 1.09 times counted.
- **London Bridge's alightings** go from 1.38 to 1.20.
- **Summed station on/off error** falls from 786,000 to 761,000 a day.
- **Unchanged:** Farringdon, Finsbury Park, East Croydon, Highbury &
  Islington, Gatwick and Blackfriars. That is the routing problem in section
  4.4, not the timetable.

**Judged by case 16's rule, it fails: 0 of 5 folds on both seeds.** The rule
wants rmse(log) better *and* the level towards 1. But the rule was written for
model changes. This is the timetable that ran on the reference date. The level
it lowers was being held up by demand that the missing trains' passengers had
been routed onto the counted trains through changes. The same point was made
about fix 4 on 28 September: equal sharing flatters the level.

**Recommendation:** make the June extract the reference timetable. Rerun every
case on it, committing in the paper repo first. Then take up section 5 from
item 2.
