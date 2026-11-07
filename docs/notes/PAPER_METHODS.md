# Paper methods log

What the pipeline does for the hindcast study, and every data or modelling decision
behind it. Configs are in `configs/paper/`; outputs go to
`datasources/output_paper/`.

## Pipeline

| Step | Command / config | Output |
|---|---|---|
| National timetable (CIF extracted 11 Sep 2026) | `timetable` / `timetable.yml` | `output_paper/timetable` |
| BPLAN timing graph (Jul-26 PIF) | `bplan` / `bplan.yml` | `output_paper/bplan` |
| Demand baselines | `loading` / `loading_all.yml`, `loading_northern_pre_dec25.yml`, `loading_tpe_2223.yml` | `output_paper/loading_*` |
| Case 1 hindcasts | `hindcast` / `case1_*.yml` | `output_paper/case1_*` |
| Methods diagnostics | `scripts/paper_diagnostics.py` | `output_paper/analysis` |

Run any step with `PYTHONPATH=src python3 -m orta <command> <config>`.

## Data

- **Loading data**: Northern, TPE, Avanti, EMR, GWR, GTR, SWR, Southeastern and Greater
  Anglia are read into one schema (`loading_data.py`). CrossCountry is excluded:
  its feed is a 0-100 busyness figure, not a count.
- **Boardings**: Northern and TPE publish passengers on/off. For the others, boardings
  and alightings are derived from the change in load between consecutive counted calls
  (net figures, so a lower bound on true boardings). A call is only treated as the
  terminus (everyone alights) when it is the service's destination.
- **Republished data**: Southeastern/Greater Anglia publish some windows twice (railway
  period and calendar month; capped and uncapped). Uncapped files are preferred, then
  railway periods; overlapping windows are dropped. Rows identical in every column are
  dropped; the same headcode recurring with different values is kept (different runs).
- **Impossible loads**: values above 3,000 passengers are voided.
- **SWR** railway periods without dates in the file are dated from the period code
  (P01 from 1 April, P02 from the first Sunday on/after 29 April, then 4-weekly).
- **ODMs**: releases without TLC codes (2018-19, 2020-21, 2021-22, 2022-23) are mapped
  NLC -> TLC from releases that carry both.

- **Archive timetables**: the published timetable for a past period, in 80-column
  fixed-width CIF (`cif_archive.py`). May-16, Dec-16, May-18, May-19, Dec-19,
  May-23, Dec-23 and Jun-24 are held as archive extracts, and Dec-25, May-26 and
  Dec-26 as published-timetable extracts. Cleaned per era by
  `configs/paper/eras/`. See `ERA_MATCHED_TIMETABLES.md`.

## Model corrections made before producing results

1. **Footfall-share demand model** (the original fallback for legs without direct
   service) summed average daily boardings over every row of the loading file, so its
   calibration grew with the number of periods loaded, and it assigned a station's
   whole boardings to every leg from that station. It is kept as `demand_model: legacy`
   (with the period scaling fixed) for comparison.
2. **Calibrated gravity model** (`gravity_model.py`, default `demand_model: calibrated`):
   OLS on every directly served ODM pair (journeys, footfall at both ends, track
   distance, >= 1 direct weekday train), `ln J = b0 + b1 ln F_o + b2 ln F_d + b3 ln d +
   b4 ln f`, with lognormal smearing for the mean. Fitted over *all* pairs with
   `ln(1 + f)`, b4 was 1.30 (a 40-fold rise for a new 16-trains/day link) because it
   absorbs the served/unserved difference; among served pairs b4 = 0.64 (ODM 2024-25),
   at the upper end of published frequency elasticities and still cross-sectional.
   Adding a direct-service dummy gave x8.3, which reflects selection, not the effect of
   a new link, so no new-link uplift is estimated. Use:
   - pair already served: observed slot demand x `(f_after / f_before)^b4`;
   - first direct trains, observed indirect demand: max(observed, served-pair prediction);
   - no observed demand (new station): the served-pair prediction.
   New slot demand is shared across all trains in the period; growth over the old total
   is generated demand attributed to the added trains.
3. **Captured vs generated trips**: a new train mostly takes riders from existing trains.
   `predict_demand` now reports both. Scenario Studio appraises abstracted riders as the
   do-minimum (full journey-time benefit, no new revenue) and generated riders as new
   (rule of half, new revenue, mode shift). The commercial case uses generated riders.
4. **Joint vs marginal addition**: `propose` treats a candidate as the only new train
   (`existing + 1`). Hindcasts report this and the joint case where all the service's
   trains are added at once (`existing + k`).
5. **Running-time check**: BPLAN publishes pass-to-pass and start/stop timings; the
   parser kept only the fastest (pass-to-pass). Legs now use the start-from-stand
   timing on the first link and the stop timing on the last. Each leg reports
   `traction_exact_share`, the share timed for the requested traction rather than the
   fastest-traction fallback. A leg is infeasible only if more than 0.5 min short:
   candidate times are whole public minutes while BPLAN times in half minutes (six real
   GWR legs were exactly 0.5 min short on public times).
6. **Headway check**: the empirical rule (tightest existing gap minus 1 minute) reflects
   current frequency, not signalling capacity, so it rejects any frequency increase. A
   `headway_mode: fixed` rule (`planning_headway_minutes`, default 3) is added; hindcasts
   report conflicts under both.
7. **Loading time periods** are classified by the boarding station's scheduled
   departure, not the train's departure from its origin. A passenger boards a median
   half hour after the train left its origin, so origin timing put 10-21% of an
   operator's boardings in the wrong period, and not at random: it inflated the
   "early" share by 124-180% and cut the AM peak by up to 19%.
   Not every feed publishes the time. Northern publishes none; Avanti and GTR publish
   the call's scheduled *arrival*; Southeastern and Greater Anglia omit it for about
   half their rows, where they also synthesise a pseudo-headcode from the origin time
   and route in place of a signalling ID. `loading_data.py` fills these from the
   CIF, which carries the same GBTT departure the other operators publish -- so the
   imputed value is that quantity, not a proxy for it. A published departure always
   wins; the rest are matched to their schedule through a ladder of keys (exact;
   destination dropped; headcode dropped; nearest origin departure within 5 minutes),
   and any key mapping to two different departures is dropped rather than guessed.
   Every row records the rung it matched on in `departure_time_source`.
   Validated by hiding a published departure and recovering it (hold-out, in
   `paper_diagnostics.py`): 83-98% exact to the minute, 98-99.6% within 5 minutes, and
   the wrong-period share falls from 16.2% to 2.2% (GWR), 21.5% to 6.6% (TPE) and
   11.0% to 3.5% (Southeastern). Per rung, the wrong-period share is 0.2% (exact),
   0.1% (no destination), 0.2% (no headcode) and 1.3% (nearest origin), against 18%
   for no imputation. Over all operators, boardings timed from the train's origin fall
   from 29.3% to 9.9%, and those timed by a true scheduled departure at the boarding
   stop rise from 23.1% to 59.6%.
   Switched off with `impute_departure_times: false`.
8. **TAG appraisal**: values of time are grown from the data book value year (2023), not
   the opening year, and flows are discounted to the data book present-value year, not the
   start of construction.
9. **Calls-only strip** (`StripRule.remove_calls_at`): the original strip removed whole
   schedules, which is right for a new train but wrong for a new station inserted into
   existing services, or an existing train extended over new ground -- there it also
   deletes service that was there before. Calls-only keeps the matched schedules in the
   before-network without their stops at the named stations. Used by the Whitby,
   Lowestoft and Cambridge South cases in `datasources/cases/`. Caveat: legs that do not
   touch a removed station still see the candidate's own train, so their existing
   frequency is one too high; `hindcast_demand_by_leg.csv` flags the legs that are the
   new ground as `leg_is_new_ground`. Where an archive extract covers the period
   before a scheme opened -- Jun-24 for the Northumberland Line -- the before
   network can now be the timetable as published rather than a synthesised one,
   which removes this caveat instead of flagging it.
10. **New-station footfall** (`HindcastConfig.station_footfall`): the gravity model needs
   annual entries and exits at both ends of a leg, and a station that opened after
   table 1410's release year appears in it at all. The calibrated model then returns
   nothing for every leg touching that station and the legacy model substitutes the
   other end's footfall, over-predicting by one to two orders of magnitude -- the first
   Northumberland Line run predicted a mean load of 0.02 against an actual 48. Footfall
   for such stations is now supplied through `station_footfall`, and a warning names any
   called station with no figure. The value is an exogenous forecast input, not anything
   measured after opening, which would make a hindcast circular.

## Known limitations to state in the paper

- **A new station cannot be forecast without an exogenous footfall input.** The demand
  model is calibrated on station usage statistics, so its accuracy at a station that has
  none is bounded by the accuracy of a forecast the model does not produce.
  `datasources/cases/01_northumberland_line/footfall_sensitivity.py` sweeps the
  assumption and reports predicted against observed load at each level, which states the
  case as "what the scheme's forecast would have had to assume" rather than as a miss.

- The timetable is a single CIF extract (11 Sep 2026), snapshotted on Wednesday
  18 November 2026, so the May-26 timetable; "before" networks are built by removing
  the service's schedules, not from historical timetables.
- Headway and existing-service counts use permanent public passenger calls only
  (no freight, non-stopping trains or short-term variations).
- Predicted train loads cover only journeys between the train's own calling points;
  passengers interchanging onto or off the train are not assigned to it.
- Temporal factors come from sampled trains; sampling coverage varies by operator/day.
- Imputed departure times are matched on service identity against a single timetable
  extract, not the timetable in force when the count was taken, so a retimed or
  withdrawn train does not match. 16% of Northern boardings and 25-30% of
  Southeastern's and Greater Anglia's still fall back to the origin departure on that
  account, and 64% of GTR's keep the published arrival rather than a departure.
  **Addressed 2026-09-22.** The parser now reads fixed-width CIF
  (`cif_archive.py`), and `LoadingConfig.timetable_eras` resolves each count
  against the timetable in force when it was taken. Four of the six eras
  spanning the counts are held (Dec-23, Jun-24, Dec-25, plus the May-26 feed);
  the Dec-24 and May-25 extracts are not, so 43% of counts still fall back to a
  single extract. Where an era is held the wrong-period share falls 17%
  relative, and rows whose era *is* the baseline are unchanged, which is the
  control. See `ERA_MATCHED_TIMETABLES.md`.
- Loading rows at a service's biggest alighting station have no trip-length band and are
  excluded from temporal factors.
- Early outturn (e.g. GWR's first three periods) precedes demand ramp-up.
