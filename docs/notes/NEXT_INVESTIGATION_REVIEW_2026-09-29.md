# The second investigation brief, reviewed and tested (29 September 2026)

This reviews the next-model-investigation brief and records what was
run. The experiments are case 17 (`datasources/cases/17_station_direction_error/`,
plan in `PLAN.md`, results in `RESULTS.md`), on the June 2026 timetable
throughout.

## 1. Verdict on the brief

The brief's direction is right. Several of its specifics are wrong, and one
would have produced a leaky experiment.

**Right, and done:**
- **The June 2026 timetable as reference.** Whether to use it is a question of
  which data are valid, not a model change for the fold rule to judge. Every
  case 17 result uses it.
- **The error is mainly station × direction × period level.** Treating that as
  the working hypothesis holds up: 42.8% + 20.7% of train-segment error on
  the June timetable.
- **Transfer routing before train choice**, with Underground links in scope.
- **Within-period timing** by revisiting case 10's weights, with a thin-slot
  guard.
- **Classifying every gain by what is known ex ante**, and holding out whole
  corridors.

**Wrong or misleading:**

| brief says | what is actually the case |
|---|---|
| §1: the file is at `data/timetable-data/...` | `datasources/timetable-data/various/May26 publsihed 240626.cif` (built by `configs/paper/eras/timetable_may26.yml`). |
| §2: the decomposition table, with day-to-day 11.7% as a fifth row | Day-to-day is a share of segment-day error; the other four are shares of train-segment error and sum to 100%. The rows don't add. |
| §4: build station profiles from observed counts, then score on held-out corridors without leaking | Contradictory on case 16. A held-out corridor's stations' counts are what is held out. The only valid source for an existing station is counts from **before** the change. Five operators' feeds hold them (January–April 2026), which case 16's filter had set aside. That test is E2, and it is incremental forecasting (a pivot), not a model fix. |
| §4: D_sdt = D_s × p_sdt | Counts measure loads and train on/off, not a station's demand. And the station total is the one thing not to force: forcing each station's counted boardings and alightings makes loads worse (rmse(log) 0.685 → 0.721, E1). |
| §6: step 5, "exact train share" | Exact loads, which is trivial. Rebuilt from each train's own counted on/off, loads reach only 0.295: the feeds' on/off do not add up to their loads. |
| §7: ORR interchanges are "an external target that is not derived from the counts" | True, but they are not observations either. They are ORCATS output (ODM × CAF routes), and National Rail changes only. They benchmark our routing against the industry's model, not against reality. |
| §7: Farringdon, Whitechapel and Clapham Junction are over-routed because there is no Underground | Came from my own comparison against ORR's **2022-23** file, whose CAF lacks the Elizabeth line (Whitechapel 579). In 2024-25, the ODM's year: Farringdon 9.3m, Whitechapel 13.8m, close to the model. Clapham Junction's total matches too; its excess on counted trains is which trains carry the changes. |
| §7: Waterloo, Euston, King's Cross, Paddington, Victoria flows need a Tube network | Mostly already handled in the ODM. Journeys that start or end on the Underground are cut to their rail part, and since 2024-25 contactless rail–Tube–rail journeys are split in two (Steer, 2025, §1.57–1.60). Only through tickets across London still need a Tube link in the middle. "Train then Tube" routes would double-count. |
| §8: add off-peak validity boundaries as features | They fall on the period bins' edges, so the bins already carry them. Without ticket type by time they cannot be told apart from other edge effects. |
| §5: predictors such as footfall, season share, region | Held out properly, most of the apparent gain is statistical shrinkage of noisy predictions. The station's own footfall adds nothing (E3). |
| §4 metrics: GEH | TAG's GEH is for hourly link flows. Here it is computed per train-segment, so it is comparable only between variants. |

## 2. Answers to §14

### A. The largest actionable source of error

**Fixable with data already held:**
- **At existing stations, the station × direction × period level itself.** It
  is stable over months. Calibrating the model to counts taken 2–5 months
  earlier takes rmse(log) from 0.685 to 0.473 on the five operators with such
  counts (E2). That is two-thirds of what the exact oracle (0.37) would give.
  It is ex ante wherever counts exist before the change.
- **Demand lost between periods.** Routing each period's unserved pairs
  through a change in that period (T1) recovers it: 0.680 → 0.654, adopted.
- **Timing within the period**: 0.680 → 0.672 held out, adopted (E5).
- **Noisy predictions.** Shrinking them by fitted slope gives 0.669 → 0.63–0.65
  held out. That is a forecasting calibration, not a model change.

**Fixable only with new data:**
- **The routes of through-London journeys.** TfL's journey planner is open and
  was used, but the Underground links failed the fold rule (T3). The routes
  ORR's figures come from would need the CAF.
- **The industry's own routes.** The CAF (ORCATS) would replace T1/T2's
  guesswork with the routes ORR's interchanges come from. Not obtainable.
- **Station on/off at the over-predicted stations** (Balham, Richmond,
  Highbury & Islington). Needs a second source, such as TfL's RODS/NUMBAT for
  the London Overground and Underground interchange stations.

**Not observable with what exists:**
- **A new station's direction and time profile in an area with no counts.**
  Held out by region, station features beyond shrinkage recover about 0.02 of
  rmse(log) (E3).
- **Day-to-day count noise**: 10.7% of segment-day error.
- **The on/off counts' own inconsistency**, which floors any on/off method at
  0.295.

### B. Which experiments moved the result (measured)

| experiment | change in rmse(log) | verdict |
|---|---|---|
| E2: calibrate to pre-change counts (existing stations) | 0.685 → 0.473 | the largest by far; a forecasting mode, not a model change |
| E3: shrinkage of noisy predictions | 0.669 → 0.614–0.634 | holds held out; a calibration |
| T1: route changes within each period | 0.680 → 0.654 | adopted |
| E5: departure-time weights | 0.680 → 0.672 | adopted; adds to T1 (0.645 together) |
| T2: logit over vias | 0.654 → 0.651 | not adopted (fails one fold) |
| T3: Underground links | 0.654 → 0.655 | fails |
| E3: station attributes, new area | about −0.02 beyond shrinkage | not usable |

A schedule-based train-choice layer (the first pack's proposal) stays capped at
0.669 → 0.567 even with perfect shares, and timetable features recover about 3%
of that.

### C. What data can realistically be obtained

- **TfL Unified API** (open, no key): used here. StopPoint search found 75 of
  London's 340 National Rail stations within 300 m of a Tube or DLR station.
  The journey planner gave Underground, DLR and walking times between them.
  Anonymous use is limited to about 50 calls a minute. A free key
  (api-portal.tfl.gov.uk) raises that about tenfold and would make a both-ways,
  several-times-of-day matrix practical.
- **TfL crowding and RODS/NUMBAT open data**: Underground link loads and
  entries and exits by access mode. It could check the model's cross-London
  transfers at the termini, which ORR's interchanges (National Rail only)
  cannot. Not yet downloaded.
- **ORR table 1410, 2024-25, with interchanges**: downloaded
  (`datasources/table-1410/table_1410_2024_25_interchanges.csv`).
- **Steer's methodology reports for ORR**: downloaded to `lit/fetched/`.
  They describe the ODM's handling of Tube legs and the ORCATS allocation.
- **Earlier count periods**: already held, from April 2024, for five operators.
  They are what makes E2 possible, and they would support a national before/after
  event study, if the matching timetables were complete. The December 2025
  extract is not: it lacks the trains run under short-term schedules.
- **Not obtainable** (as before): CAF/ORCATS, MOIRA, LENNON route codes.

### D–F. Experiments and results

All are in case 17, reproducible from its scripts. The default model and
configuration are unchanged. Results are in `RESULTS.md` there; section 3
below is the summary table.

## 3. Results against the June-timetable baseline

| variant | rmse(log) | level | r | station error/day | trip length | held-out result |
|---|---|---|---|---|---|---|
| B0: case 16 fixes, June timetable | 0.680 | 0.819 | 0.699 | 760,600 | 0.862 | — |
| T1: routing by period | 0.654 | 0.845 | 0.699 | 766,700 | 0.863 | adopted (4/5, 5/5) |
| T2: + logit over vias | 0.651 | 0.851 | 0.699 | 761,800 | 0.868 | not adopted over T1 (3/5, 5/5) |
| T3: + Underground links | 0.655 | 0.862 | 0.700 | 775,000 | 0.860 | fails (3/5, 3/5) |
| E5: timing weights | 0.672 | 0.820 | 0.711 | 760,900 | 0.862 | adopted (5/5, 5/5) |
| **T1 + E5** | **0.645** | 0.846 | 0.712 | 766,700 | 0.864 | adopted over T1 (5/5, 5/5) |
| E2: calibrated to pre-change counts* | 0.473 | 0.93 | 0.85 | n/a | n/a | 5/5, 5/5 (nothing fitted) |
| E2: pre-change counts alone* | 0.481 | 0.92 | 0.83 | n/a | n/a | 4/5, 4/5 |

\* The five operators with pre-change counts (B0 0.685 on the same segments);
existing stations only.

The E3 rows (new stations) are in case 17's `RESULTS.md`. In short:
- a station never seen, in an area with no counts, is predictable only by
  shrinkage (0.669 → 0.63);
- a new station on a counted line is predictable from its neighbours' errors
  (0.669 → 0.430).

**The answer to the brief's closing question:**
- **What causes the error.** It comes from how each station's demand divides by
  direction and time, and from which pairs load which trains. It does not come
  from the ODM's station totals.
- **Existing stations.** It is stable over months, so counts from before a
  change remove two-thirds of it.
- **New stations.** It can be reduced only where the station's line is already
  counted.
- **The model itself.** The legitimate changes found here (period routing and
  timing weights) take 0.035 off rmse(log). That is real, and small beside
  what counts do.

## Literature, source by source

The brief asks five things of each source: what it models, what data it needs,
whether this project holds that data, which of our problems it bears on, and
what can be borrowed without building MOIRA. The problems are:
- **SDP**: station × direction × period demand;
- **path**: transfer and path choice;
- **train**: allocation to trains within a period;
- **new**: new-station demand;
- **change**: response to a frequency or service change.

Everything in the folder was read. TAG M2.2 and two Steer methodology reports
for ORR are public and were downloaded to `lit/fetched/`. MOIRA, PDFH and
ORCATS are industry-licensed. Only Steer's one-page ORCATS overview is public,
so those rows say what the public material says and no more.

### GB practice and data

| source | what it models | data it needs | held here? | bears on | what to borrow |
|---|---|---|---|---|---|
| **TAG M3.2** (May 2024) | PT assignment practice: when to use timetable- or frequency-based assignment, and how to validate | Counts by link and screenline, a base matrix | Counts yes, as train-segments; hourly screenlines can be built | train, path | Report tables 4 and 5 (grouped counts within 15%; links over 150 pax/h within 25%). Say plainly that without matrix estimation they will not be met: 38% of link-hours pass. |
| **TAG M2.2** (May 2020) | Building the base-year matrix: a prior from observed data, then refinement. Counts may adjust it through matrix estimation, "part of the matrix development process", with the change monitored (§6.3.11). Checks on trip-length distribution and sector totals; PT screenlines within ±15% (§6.2.22). | Ticket data, surveys, counts | ODM yes, counts yes | SDP | E2's calibration to pre-change counts is the kind of count-based refinement M2.2 allows in a base year. The M2.2 way to report it: its change to the prior, its trip-length distribution, sector totals. |
| **ORR / Steer methodology** (Dec 2025 quality report; Nov 2025 "historical methodological changes") | How the ODM and table 1410 are built from LENNON: London Terminals allocated by CAF; journeys starting or ending on the Underground cut to their National Rail part (§1.57–1.60); since 2024/25 contactless rail–Tube–rail journeys split in two; Elizabeth line interchanges at Paddington and Liverpool Street added in 2024/25 | – | Yes (reports fetched; ODM and 1410 held) | path, SDP | An ODM pair's ends are where passengers enter and leave National Rail, so the Tube links are needed only in the middle of a journey (T3 changed before running). Compare interchanges with 2024-25, not 2022-23 (Whitechapel 13.8m against 579). |
| **ORCATS** (Steer 2017/18 report, appendix C: the only public description) | Route shares for each flow from the timetable: faster and more frequent routes take more; the output is the CAF | Timetable, proprietary parameters | No | path | Spread a flow over routes (T2) instead of all-or-nothing. Treat ORR's interchanges as ORCATS output, a benchmark rather than an observation. |
| **MOIRA** (not public) | Incremental forecasts: pivots LENNON demand on PDFH elasticities to changes in generalised journey time; ORCATS-style allocation | LENNON, timetable, PDFH | ODM only | change, path | Pivoting. E2 shows it is the dominant lever at existing stations. |
| **PDFH** (not held; via Wheat & Wardman) | Elasticities to GJT, fares, income; rooftop wait for infrequent services; interchange penalties by distance | Licensed | No | change, train | Values quoted in Wheat & Wardman: interchange penalties 19/40/65 min at 30/100/200 miles, frequency elasticity about 0.15–0.25. |

### Demand response and distribution

| source | what it models | data it needs | held here? | bears on | what to borrow |
|---|---|---|---|---|---|
| **Wheat & Wardman (2017)** | Demand response to timetable changes (journey time, headway, interchange) on GB flows, from ticket-sales panels | Flow-level ticket panel over time | No panel; one ODM year | change, path | A penalty per change that rises with distance (about 100 min at about 100 miles), where the model uses a flat 20 min. A frequency elasticity target of 0.15–0.25 against the model's 0.05–0.06 within a pair. |
| **Wardman (2022)** | Meta-analysis of GB time elasticities | – | – | change | Priors for the pivot's elasticities. |
| **Cordera et al. (2018)** | Trip distribution on one 23-station line, with spatial effects between stations; doubly constrained gravity | 26,371 surveyed OD trips | No survey; ODM yes | new | Poisson (PPML) estimation for the gravity branch (not run here; PLAN.md). |
| **Young & Blainey (2018); Young (thesis)** | Station choice logit, giving probabilistic catchments for new-station trip-end models (the basis of SDFT) | Revealed-preference surveys (Wales, Scotland), an open-data route planner | SDFT already ported | new | Already borrowed. E3 adds that the new station's load error can be predicted from its counted neighbours on the same line, not from its attributes. |

### Assignment: schedule-based, frequency-based and path choice

| source | what it models | data it needs | held here? | bears on | what to borrow |
|---|---|---|---|---|---|
| **Cascetta & Coppola (2016; 2012)** | Schedule- against frequency-based assignment for high-speed rail; elastic demand | 3,237 surveyed desired departure times; HSR counts | Counts yes; departure times no | train, path | Their own finding: frequency-based is close at 2–5 trains an hour (RMSE% 0.280 against 0.308). The ±40–60% is model against model. |
| **Nuzzolo, Crisalli & Rosati (2012)** | Schedule-based dynamic assignment with vehicle capacity; joint choice of departure time, stop and run; frequent users | Desired departure times, capacities, run-level network | Capacities partly (counts' mean capacity); no departure times | train | Capacity limits on crowded peak trains, later. |
| **Eltved, Nielsen & Rasmussen (2019)** | Mixed schedule- and frequency-based path choice: choice sets by a cost threshold, then MNL | Timetable, route choice parameters | Timetable yes; parameters borrowed | path | The shape of T2: a window around the best route, then logit. |
| **Spiess & Florian (1989); Nguyen & Pallottino (1988); Florian & Constantin (2012)** | Frequency-based "optimal strategy" (hyperpath) assignment; logit over strategies | Line frequencies | Yes | path | Half the headway as the wait for the connecting train (T1's cost). Strategies suit London's turn-up-and-go links, not timetabled rail. |
| **Hamdouch et al. (2014)** (`Content.pdf`) | Schedule-based equilibrium with strategies and unreliable supply | Space-time network, reliability data | No | path, train | Nothing at this scale. |
| **Fukasawa & Shibata (RTRI)** | Choice between express and local trains in urban railways | Web survey (775 people) | No | train | Fast against stopping choice (case 16 fix 4's territory). |
| **Burger et al. (2023)** | Stated willingness to shift within the peak | Online experiment | No | train | Evidence that peak-edge shifting exists, as E5's half-hour pattern shows. |
| **Robenek et al. (2018)** | Timetable design with logit demand | Synthetic | – | – | Nothing. |

### OD estimation from boarding and alighting counts

| source | what it models | data it needs | held here? | bears on | what to borrow |
|---|---|---|---|---|---|
| **Ji et al. (2015)**; **Chen, Cheng & Sun (2025)**; the frequency-based path-flow estimator (Transp. Res. E 2022); **Burgalat et al. (2026 review)** | One route's OD per trip or period, from APC on/off counts with a prior (EM, IPF, Bayesian); network OD from APC and AFC | Stop-level on/off by trip, and a prior | On/off for five operators; ODM prior | SDP | As a base-year diagnostic, and only with counts from before a change. E1 shows the on/off counts do not add up to the counted loads: exact on/off per train reaches rmse(log) 0.295, not 0. That caps any on/off-based estimation. |

### General references

Cascetta's *Transportation Systems Analysis*, Boyles et al., McNally's
*The Four Step Model* and Train's *Discrete Choice Methods* are textbooks, cited
for definitions only. The claim that Flyvbjerg et al.'s *Transport Reviews*
paper was retracted is still unverified.
