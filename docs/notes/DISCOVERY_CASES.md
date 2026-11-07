# Discovery cases: unserved direct-service markets on the GB network

A national scan for station pairs that have **no direct train today but high
estimated latent demand**, and three candidates worked up through Scenario
Studio. Companion to `PAPER_METHODS.md`, which documents the demand model this
relies on.

Run on 17 September 2026 against the 11 September 2026 CIF extract.
Configs: `configs/paper/discovery_national.yml`, `configs/paper/scenario_discovery_cases.yml`.
Outputs: `datasources/output_paper/discovery/`, `datasources/output_paper/discovery_cases/`.

> **Superseded figures (2026-10-01).** The tables below predate cases 24 and 25 and
> are stale in five ways:
>
> - stale matrices;
> - the Reading second-code bug in scenario/propose;
> - no new-link step;
> - no transfers in scenario;
> - frequency priced as `f ** 0.4`.
>
> The scan also ranked on the gravity model's slot prediction, not on each corridor's
> observed market, and read stations' second codes as separate stations. See
> `datasources/cases_local/25_new_service_forecasts/RESULTS.md` for the current
> figures.

---

## 1. The three cases

| # | Case | Track miles | Run time | Best slot | Feasible slots | Total pax/yr | **Generated pax/yr** |
|---|------|------------|----------|-----------|----------------|--------------|----------------------|
| 1 | **Gatwick Airport ↔ Birmingham New Street** | 137.8 | 2h33 | 08:35, 4/day | 2 of 25 | 117,156 | **24,522** |
| 2 | **Tonbridge ↔ Gatwick Airport** | 25.9 | 33 min | 07:00, 4/day | 11 of 25 | 8,863 | **859** |
| 3 | **Woking ↔ Reading** | 28.4 | 47 min | 09:00, 4/day | 12 of 25 | 208,235 | **24,506** |

All three are **direction-symmetrically unserved**: no through train in either
direction on a typical weekday, anywhere in the May-2026 timetable.

Read the two demand columns carefully — they are different things:

- **Total pax/yr** is everyone riding the train across *all* its legs,
  including intermediate pairs that already have service. It is a train-loading
  figure, not a measure of the market gap.
- **Generated pax/yr** is the model's estimate of genuinely new demand, after
  abstraction from existing services. This is the number that answers "what does
  the new direct link actually unlock", and it is the one to quote.

---

## 2. What was run

| Step | Command / config | Output |
|---|---|---|
| National timetable, `crs_reference` set (§3.1) | `timetable` / `configs/paper/timetable.yml` | `output_paper/timetable` |
| **City-market service table** (§3.2) | `scripts/discovery_inputs.py` | `output_paper/timetable_city_market` |
| BPLAN timing graph (Jul-26 PIF) | `bplan` / `configs/paper/bplan.yml` | `output_paper/bplan` |
| Demand baseline, all 9 operators | `loading` / `configs/paper/loading_all.yml` | `output_paper/loading_all` |
| National discovery scan | `discover` / `configs/paper/discovery_national.yml` | `output_paper/discovery` |
| Annual-basis re-rank (§4) | post-processing of the 400-pair shortlist | — |
| Scenario Studio on the 3 cases | `scenario` / `configs/paper/scenario_discovery_cases.yml` | `output_paper/discovery_cases` |

Operator coverage is the full `loading_all` set — Northern, TPE, Avanti, EMR,
GWR, GTR, SWR, Southeastern and Greater Anglia. CrossCountry is excluded because
its feed is a 0–100 busyness index, not a count (`PAPER_METHODS.md` §Data). The
*timetable* side is the whole CIF, so XC's trains are present for the
existing-service and headway tests even though its loading data is not.

The demand model is the calibrated gravity model already documented in
`PAPER_METHODS.md`:

```
ln J = -6.256 + 0.508 ln F_o + 0.512 ln F_d - 0.614 ln d + 0.644 ln f
n = 73,982 pairs   R² = 0.707   Duan smearing 2.916
```

It was refitted after the §3.1 correction, because its `ln f` term reads the
same route-frequency table the defect corrupted: pairs into Waterloo, Victoria,
London Bridge and Clapham Junction had no direct train recorded, so they failed
the fit's "at least one direct weekday train" test and were dropped from it.
The effect on the model itself turns out to be small and in the right direction
— 2,576 more pairs, R² 0.696 → 0.707, and `ln_trains` 0.639 → 0.644 — which is
reassuring for the other results in `PAPER_METHODS.md` that share this model.
The pre-correction fit is kept beside it as `gravity_model_prefix.json`.

---

## 3. Four corrections made before any result was trusted

The scan does not work out of the box at national scale. Each of these
materially changed the answer; the first two are defects in the input data, the
second two are specification errors in how "already served" was being tested.

### 3.1 Britain's busiest stations were missing from the timetable

`route_frequency_summary.csv` from the national `timetable` run contained **no
rows at all** for Waterloo, Victoria, London Bridge, Clapham Junction or
Vauxhall — while showing `WOK→SUR` at 99 trains/day. That is impossible.

Cause: the CIF publishes several TIPLOCs per station for different platform
groups, and for some of them the `TiplocV1` reference record carries a null CRS:

```json
{"tiploc_code":"WATRLMN","nalco":"559801","stanox":"87212",
 "crs_code":null,"description":null,"tps_description":"LONDON WATERLOO"}
```

`load_tiploc_crs_map` ([timetable.py:130](src/orta/timetable.py#L130))
maps these to `""`, and `read_public_calls`
([timetable.py:349](src/orta/timetable.py#L349)) then drops every call
at a location with no CRS. **192,589 public call rows** across 77 TIPLOCs were
affected; 12 were real stations, and 10 stations lost *every* call row.

The bias is not random — it is almost entirely SWR and South London
(`CLPHMJM/C/W/1` alone account for 45,559 rows). Left uncorrected, every SWR
flow reads as having no London service, and Waterloo's 57.8m entries/exits make
those phantom pairs dominate the demand ranking. This is precisely the territory
the scan was asked to cover.

**Fix.** `tps_description` survives even when `crs_code` is null, so the CRS is
recoverable by name. `load_tiploc_crs_map` now takes a `station_reference` and,
for any TIPLOC the CIF publishes without a CRS, matches the description against
station names — exactly first, then re-trying after stripping one platform-group
qualifier at a time (`(E)`, `DC`, `ELL`, `NLL`, `HIGH LEVEL`, `PLATS 0-2`,
`STATION`). Only an exact match on the whole remaining name is accepted.

Nothing else is stripped, and that is the safety property that matters:
rail-replacement bus stops, ferry piers and depots never match a station name,
so they stay unmapped. "OKEHAMPTON WEST STREET", "GOUROCK PIER" and "LONDON ROAD
DEPOT" are all left alone — mapping any of them to a rail CRS would fabricate
train service that does not exist. Descriptions too abbreviated to match
("STOURBRIDGE JN (TOWN PLAT)") are also left unmapped rather than guessed, and
are supplied explicitly through `tiploc_crs_overrides`.

On the national CIF this recovers **39 TIPLOCs**, one override aside — more than
the 24 found by hand, since it also picks up Moorgate, Farringdon, Staines,
Norwich Low Level, both Yeovils, Orpington's platform TIPLOCs and Gourock
*station* (while still refusing Gourock *Pier*).

The fix validates itself: the national top routes become Waterloo–Clapham
Junction (652/day), Waterloo–Vauxhall and London Bridge–Charing Cross. Before
it, the busiest route in Britain was Liverpool James Street–Birkenhead. Call
rows 6,866,436 → 6,902,077; route pairs 81,625 → 84,309.

Set `crs_reference` on the `timetable` config to enable it (see
`configs/paper/timetable.yml`). It is opt-in rather than automatic because it
needs a station list the CIF does not supply, and a run without it is unchanged.
Covered by `tests/test_network_inputs.py`.

### 3.2 A London terminus is a gateway, not a destination

With the raw per-station table, the model's top-ranked "unserved" pairs were
`GTW→LST`, `RDG→LST`, `WOK→LST`, `SLO→LST`, `BHM→LST`. Every one is an artefact.

Table 1410 gives Liverpool Street 80.4m entries and exits, the overwhelming
majority of whom are travelling to or from somewhere else in London. The gravity
model reads that as destination *mass*, so anything paired with a London
terminus scores enormously. And "no direct train Reading → Liverpool Street" is
true but meaningless: the real market, Reading → London, runs 204 times a day
from Paddington.

The same problem recurs one level down with non-terminus London stations —
Gatwick→Stratford scored as a gap while Gatwick→Victoria runs 142 times a day.

**Correction.** The existing-service test runs at the level of the **city
market**, not the individual station. `discover` reads `daily_trains` from a
`timetable` run's summary, so a zone-aware table was written and the tool
pointed at it, unmodified:

```
served(O,D) = max over o' ∈ N(O), d' ∈ N(D) of direct trains(o', d')
  N(X) = X's conurbation if it is in one, else every station within 2 miles
```

Conurbations are radius-defined around 29 centres (London 20 mi from Charing
Cross → 446 stations; Birmingham/Manchester/Glasgow/Leeds/Liverpool 12 mi;
smaller cities 6–10 mi). A pair with **both ends inside one conurbation** is a
metro journey, not an inter-urban market, and is suppressed outright.

Effect: 1,562,799 pairs reclassified as already served and 78,450 suppressed as
intra-conurbation. `RDG→LST` now reads 204, `GTW→SRA` 261, `WOK→LST` 217.

### 3.3 A corridor is served in both directions or neither

Chichester→Liverpool Street ranked 5th after the corrections above, and
Chichester→Victoria showed **1 train/day**. Chichester is a city of 30,000 — a
genuine finding, if true.

It is not. The counter is correct, but the CIF contains **32 through
Victoria→Chichester workings and 1 Chichester→Victoria** on the reference
Wednesday. The corridor is well served; only the *direction* is asymmetric in
the schedule data (Southern's Coastway services divide and join, so the
northbound Chichester portion is not a single through schedule).

`_drop_well_served_pairs` tests ordered pairs, so `CCH→VIC` at 1/day passed the
≤1 filter while `VIC→CCH` at 32/day did not. A direct link is a property of a
corridor, not a direction.

**Correction.** A pair is treated as served if **either** direction exceeds the
threshold. This reclassifies a further 14,935 pairs as served (3,995,483 →
3,980,548 candidates), removes every Chichester entry from the promoted list —
`CCH→LST` had ranked *first* before it — and lets Reading–Brighton, another pair
on the withdrawn CrossCountry corridor, take the vacated slot.

Now `DiscoveryConfig.existing_service_both_directions`, default true, applied
inside `_drop_well_served_pairs`. Set it false to test each ordered pair on its
own. Covered by `tests/test_network_inputs.py`.

### 3.4 The pre-rank is mis-specified against the model it gates

Stage 3a shortlists on a footfall proxy `F_o · F_d / d²`
([discovery.py](src/orta/discovery.py), `_proxy_rank_demand`).
The fitted model it gates has `ln_distance = -0.626`, not -2 — a distance
exponent wrong by a factor of three, which systematically starves the shortlist
of long-distance candidates. At the default `demand_proxy_shortlist_size: 500`,
0.00% of the shortlist was over 100 miles.

Two changes, neither touching the library:

- `demand_proxy_shortlist_size: 60000`, so the proxy is not the binding
  constraint and the calibrated model does the ranking.
- `max_candidates_per_origin: 2600`. The per-origin cap keeps the *nearest*
  pairs in band, so with a 300-mile ceiling the default cap of 40 would have
  silently collapsed the scan to roughly 50 miles.

---

## 4. Ranking basis: annual, not peak-slot

`discover` scores demand in one (day, period) slot. The slot factor is a
per-band constant, and GWR's Monday AM peak weights the bands unequally:

| Band | Monday AM-peak factor | Relative to commute |
|---|---|---|
| commute | 0.03255 | 1.00 |
| regional | 0.02107 | 0.65 |
| long_distance | 0.02013 | 0.62 |

So a peak-slot ranking penalises long-distance candidates by 38% for no reason
other than the choice of slot. The effect is not academic: on the AM-peak slot
`WOK→RDG` (261) outranks `GTW→BHM` (169), but dividing out the band factors
gives 8,023 against 8,408 — **the ranking flips**.

Candidates were therefore re-ranked on the flat **annual** estimate, at a
common 1 train/day so the frequency term is neutral across pairs, then filtered
for BPLAN running-time feasibility. Shortlisting for exact track distances used
the fitted model's own functional form on straight-line distance, rather than
the d⁻² proxy.

---

## 5. The funnel

| Stage | Pairs |
|---|---|
| Ordered station pairs considered (2,612 stations) | 6,819,932 |
| Within 15–300 miles straight-line | 5,636,732 |
| No/poor existing direct service, **city-market and direction aware** | 3,980,548 |
| Demand-shortlisted (calibrated gravity model) | 400 |
| Running-time feasible on the physical BPLAN network | 400 |
| Promoted to the full Scenario Studio sweep | 15 |

All 400 shortlisted pairs were physically routable — unsurprising, since they
are pairs of existing stations on a connected network. Feasibility bites at the
*path* level, not the route level (§6.1).

### Top 12, annual basis, direction-symmetric

| # | Pair | Track mi | Band | Min run | Annual trips @1/day |
|---|------|---------|------|---------|---------------------|
| 1 | Woking – Gatwick Airport | 32.2 | regional | 30 min | 8,583 |
| 2 | **Gatwick Airport – Birmingham New Street** | 137.8 | long_distance | 105 min | 8,408 |
| 3 | **Woking – Reading** | 28.4 | commute | 26 min | 8,023 |
| 4 | **Tonbridge – Gatwick Airport** | 25.9 | commute | 22 min | 7,957 |
| 5 | Reading – Brighton | 76.2 | regional | 56 min | 7,072 |
| 6 | Brighton – Birmingham New Street | 161.9 | long_distance | 102 min | 7,004 |
| 7 | Stansted Airport – Gatwick Airport | 66.3 | regional | 70 min | 6,670 |
| 8 | Slough – Gatwick Airport | 46.3 | regional | 44 min | 6,292 |
| 9 | Tunbridge Wells – Gatwick Airport | 30.5 | regional | 26 min | 6,126 |
| 10 | Chelmsford – Gatwick Airport | 62.0 | regional | 58 min | 5,935 |
| 11 | Guildford – Brighton | 50.7 | regional | 44 min | 5,922 |
| 12 | Gatwick Airport – Manchester Piccadilly | 208.5 | long_distance | 174 min | 5,666 |

**The single clearest finding of the scan is that one corridor dominates it.**
Entries 2, 5, 6, 12 — and 21 (Guildford–Birmingham), 24 (Woking–Birmingham) and
30 (Reading–Leeds) below the cut — all lie on the route
Brighton–Gatwick–Reading–Birmingham–Manchester. That is not a coincidence:
CrossCountry ran exactly this service until **13 December 2008**, when the
re-let franchise dropped everything south-east of Guildford
([Wikipedia: CrossCountry](https://en.wikipedia.org/wiki/CrossCountry)). The
model, which knows nothing of that history, independently re-identifies the
withdrawn route as the largest unserved direct-service market in Britain.

The second theme is Gatwick feeder gaps (1, 4, 7, 8, 9, 10). Gatwick has 16.5m
entries and exits and direct trains to London, Brighton and the Sussex coast,
but to almost nowhere else.

---

## 6. The cases

### Case 1 — Gatwick Airport ↔ Birmingham New Street

**Restoration of the withdrawn CrossCountry link.** 137.8 track miles via
Redhill, Guildford, Reading, Oxford, Banbury, Leamington Spa, Coventry and
Birmingham International. Class 220, 130.0 min pure running time,
`traction_exact_share` 0.744.

| Trains/day | Total pax/yr | Generated pax/yr | Generated share |
|---|---|---|---|
| 1 | 74,927 | 11,631 | 15.5% |
| 2 | 98,867 | 15,347 | 15.5% |
| 4 | 130,455 | 20,250 | 15.5% |

Best-ranked feasible option 08:35, 4/day: 117,156 total, 24,522 generated.
Optimised schedule:

```
GTW 08:35 → RDH 08:42/44 → GLD 09:06/08 → RDG 09:37/39 → OXF 09:58/59
→ BAN 10:16/17 → LMS 10:33/34 → COV 10:46/48 → BHI 10:57/59 → BHM 11:08
```

**The binding constraint is paths, not demand.** Only **2 of 25** departure
slots cleared the headway test — by far the worst of the three — with a tightest
gap of 3.0 minutes. A flexible-dwell search did find a 07:56 departure clearing
every leg independently. Restoring this service is a capacity problem on the
Coventry–Birmingham and Reading approaches, not a market problem.

Why this case: the largest unserved long-distance market in the scan; two of the
largest stations outside London (16.5m and 30.7m entries/exits); and a withdrawn
service, which makes it the only one of the three with a natural experiment
behind it.

### Case 2 — Tonbridge ↔ Gatwick Airport

**Kent's missing airport link.** 25.9 track miles via Edenbridge and Redhill,
entirely over existing line. Class 153, 27.0 min pure running time,
`traction_exact_share` 0.667.

| Trains/day | Total pax/yr | Generated pax/yr | Generated share |
|---|---|---|---|
| 1 | 5,091 | 493 | 9.7% |
| 2 | 6,717 | 651 | 9.7% |
| 4 | 8,863 | 859 | 9.7% |

Best option 07:00, 4/day. Schedule `TON 07:00 → EBR 07:11/12 → RDH 07:22/24 →
GTW 07:30`. **11 of 25** slots feasible, tightest gap 11.0 minutes.

This is the most *deliverable* case and the weakest on demand — and it is
included for exactly that reason. It needs no new infrastructure, no
electrification and a single two-car unit; it ranked 4th on the scan. But once
the full model runs over a realistic stopping pattern, it generates only ~859
new trips a year, because the intermediate stations are small and the end-to-end
market is thin. **The scan's ranking and the worked-up appraisal disagree, and
the appraisal is the one to believe.** A scan score is an invitation to look,
not a result.

### Case 3 — Woking ↔ Reading

**An orbital link between two major interchanges.** 28.4 track miles via
Guildford and the North Downs line. Class 158, 36.0 min pure running time,
`traction_exact_share` 0.497 — the weakest timing evidence of the three.

| Trains/day | Total pax/yr | Generated pax/yr | Generated share |
|---|---|---|---|
| 1 | 119,599 | 14,075 | 11.8% |
| 2 | 157,812 | 18,572 | 11.8% |
| 4 | 208,235 | 24,506 | 11.8% |

Best option 09:00, 4/day: 208,235 total, 24,506 generated. Schedule
`WOK 09:00 → GLD 09:06/08 → NCM 09:18/19 → FNN 09:23/24 → BAW 09:28/29 →
WKM 09:37/38 → RDG 09:45`. **12 of 25** slots feasible, tightest gap 5.0 minutes.

The 208,235 total is the largest of the three by a wide margin, but most of it
is riders on already-served intermediate legs — the North Downs corridor is
busy. On generated demand it is level with Gatwick–Birmingham (24,506 against
24,522) for a fifth of the distance and six times the path availability.

Woking (5.3m) and Reading (12.4m) are both major interchanges, and a Woking–
Reading passenger currently changes at Guildford or Ascot. The case is an
orbital, London-avoiding link — a different market type from the other two.

### Runner-up: the North

All three cases are in the South East. That is a result, not a selection bias:
after the city-market and direction-symmetry corrections, the high-demand
unserved pairs concentrate on the withdrawn CrossCountry corridor and Gatwick's
feeder gaps. The highest-ranked northern candidate is **Derby ↔ Manchester
Piccadilly** (rank 29; 69.5 track miles; 73 min; 5,044 annual trips at 1/day),
which has no through train on a typical weekday — a legacy of the Peak District
route closures. It was not taken through Scenario Studio.

---

## 7. What the literature says about the mechanism

The model's own frequency coefficient cannot carry the argument for a *new*
direct link. `b4 = 0.639` is estimated among **already-served** pairs, and
`PAPER_METHODS.md` §2 is explicit that the ×8.3 direct-service dummy reflects
selection rather than the effect of a new link, so no new-link uplift is
estimated. The case that removing an interchange unlocks demand therefore rests
on external evidence:

- **Interchange penalty.** Wardman (1983) estimated the penalty at ~**35 minutes**
  of equivalent journey time on average, falling to ~13 minutes when the
  connection is guaranteed to be on time, with business travellers valuing it
  highest. This is the standard framing in PDFH's generalised journey time,
  where GJT combines in-vehicle time, a headway penalty and an interchange
  penalty ([Worsley, RAC Foundation, 2012](https://www.racfoundation.org/wp-content/uploads/2017/11/pdfh-worsley-dec2012.pdf);
  [RDG, About PDFH](https://www.raildeliverygroup.com/pdfc/about-the-pdfh.html)).
- **Demand effect.** Cross-sectional analysis of ticket sales by White & Holt
  (1979) and Wardman (1983) found the presence of an interchange reduced
  inter-urban rail demand by around **45%**. That is the order of magnitude the
  three cases are reaching for — and it carries the same cross-sectional caveat
  as our own `b4`.
- **Generated vs abstracted.** ORR and industry monitoring of open-access
  operators — the closest real-world analogue to "a new direct link into an
  unserved market" — reports roughly **2 to 2.5 journeys generated for every 1
  abstracted**, and 51–67p of industry revenue generated per journey
  ([ORR, Monitoring open access, 2024](https://www.orr.gov.uk/sites/default/files/2024-07/monitoring-open-access-july-2024-update.pdf);
  [House of Commons Library, 2025](https://commonslibrary.parliament.uk/research-briefings/cdp-2025-0029/)).

That last figure is the sharpest check on these results, and it cuts against
them: it implies a generated share around 67–71%, where this model produces
**9.7–15.5%**. The model is between four and seven times more conservative than
the open-access evidence. Two readings, and they are not exclusive:

1. The model is genuinely conservative by construction — it refuses a new-link
   uplift, so it only ever captures the frequency effect on an existing market.
   The generated figures here should be read as a **floor**.
2. Open-access route selection is heavily self-selected — operators propose
   exactly the routes where generation will be highest — so 2–2.5:1 is an upper
   bound that a randomly chosen unserved pair would not reach.

The honest position is that the true generation for these three sits somewhere
between the two, and the model does not pin it down. For precedent on appraising
new links of this kind, ATOC's *Connecting Communities* (2009) identified 14
schemes it judged commercially viable at BCR > 1.0
([RDG archive](https://www.raildeliverygroup.com/publications/archive/299-2009-06-connecting-communities/file.html)).

---

## 8. Limitations

**These are not appraisals.** Capital and operating costs are set to 0.0
throughout, so the BCR (`inf`) and NPV (£0.01–0.34m) in the outputs are
arithmetic artefacts, not results. No cost has been invented. Turning any of
these into a business case needs rolling-stock, capex and opex assumptions that
are not in this repo.

**Absolute demand is modest and frequency-dependent.** The scan's headline
figures (~5,000–8,600 annual trips) are the model's estimate for a pair with
*one* train a day, where the `ln f` term is zero. They are a comparable floor
across candidates, not a forecast of what the market would be at a real service
level. Do not quote them as "demand for the route".

**BPLAN timing coverage for modern DMUs is sparse on secondary lines.** No
passenger unit has exact timings on the Redhill–Tonbridge or North Downs links;
`traction_exact_share` is 0.667 (case 2) and 0.497 (case 3), against 0.744 for
case 1. Across plausible traction codes the end-to-end running time varies
24.5–28.5 min for case 2 and 35.0–49.0 min for case 3. The quoted times are
towards the fast end of that range. Demand results were stable under the traction
change (117,156 / 8,863 / 208,235 before and after), so this affects the
timetable, not the ranking.

**The London-terminus footfall problem is contained, not solved.** §3.2 fixes
the *existing-service* test. The *demand* model still uses each terminus's own
table 1410 footfall, which is interchange-inflated. None of the three cases has
a London end, so they are unaffected — but any future candidate that does should
be treated with suspicion.

**The headway test is a planning rule, not signalling capacity.**
`headway_mode: fixed` at 3 minutes was used, because the empirical rule
(tightest existing gap minus 1 minute) rejects any frequency increase by
construction — `PAPER_METHODS.md` correction 6. The "2 of 25 feasible" result for
case 1 is a real signal of a congested corridor, but it is not a capacity study.

**One timetable snapshot.** A single CIF extract (11 September 2026, snapshotted
on Wednesday 18 November 2026 — the May-2026 timetable), permanent public
passenger calls only. No seasonal, short-term or freight variation.

**Conurbation zones are radius-defined and hand-specified.** 29 centres with
6–20 mile radii. A city not on the list gets only a 2-mile neighbourhood rule,
so a multi-station town outside the list could still generate a false gap.

**Inherited limitations.** Everything in `PAPER_METHODS.md` §Known limitations
applies — in particular that predicted train loads cover only journeys between
the train's own calling points, and that interchanging passengers are not
assigned to it.

---

## 9. Reproducing

From the repo root, after the `timetable` and `bplan` paper configs:

```sh
PYTHONPATH=src python3 scripts/discovery_inputs.py
PYTHONPATH=src python3 -m orta discover configs/paper/discovery_national.yml
PYTHONPATH=src python3 -m orta scenario configs/paper/scenario_discovery_cases.yml
```

§3.1 and §3.3 are now in the library — `crs_reference` on the `timetable` config
and `existing_service_both_directions` on the discovery config — so only the
city-market table of §3.2 needs a script beside it. The annual re-rank of §4 is
post-processing of the 400-pair shortlist and is not part of the scripted
pipeline.

`output_paper/timetable_fixed/` is the superseded, downstream-patched copy of the
national timetable. Nothing points at it any more and it can be deleted.
