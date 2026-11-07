# How accurate is the model, against other models and the literature?

Written 2026-09-26 after the fixes in §4 were built and the new-station cases
(01, 03, 05, 11–15) re-run. §1 sets out the comparable published evidence,
§2 puts this model on the same scale, §3 lists why the model is still wrong
and what each cause is worth, and §4 lists what changed.

**Revised 2026-09-27.** A review (`MODEL_REVIEW_2026-09-27.md`) found three bugs
in the demand code and two in the SDFT port, and every affected case was
re-run. §2, §3 and §5 carry the re-run figures. The 26 September headline,
"about as accurate on new-station totals as UK scheme appraisals", is
withdrawn: it does not survive year-consistency, scheme clustering or a
confidence interval (§2.1).

All ratios are **forecast ÷ outturn**. The spread is the sample standard
deviation of the log ratio (ddof = 1), so 0.5 means a typical station is off by
a factor of about e^0.5 ≈ 1.65 either way.

## 1. What the literature says

### 1.1 New-station totals: the closest comparison

| Source | What was forecast | n | Median ratio | sd(log) | Within ±25% | Within ×2 |
|---|---|---|---|---|---|---|
| Steer Davies Gleave for DfT (2010), Figure 3.2 | UK scheme appraisal forecasts, annual journeys, 2008/9 outturn | 23 | 0.94 | 0.61 | 52% | 74% |
| Young & Blainey, SDFT (2019), Table 3 | national trip-end model, 2017/18 entries+exits | 10 | 1.15 | 0.49 | 30% | 80% |
| Same table, the appraisal forecasts | Borders Railway and Conon Bridge business cases | 8 | 0.72 | 1.39 | 12% | 38% |

Computed here from the published tables. SDFT's ten are seven Borders
Railway stations and three others, so they are about four independent
observations, not ten.

What those studies found drives the error:

- **Getting the station's function wrong.** Examples are a leisure centre
  missed at Glasshoughton (outturn 2.65× forecast) and Newcraighall and
  Edinburgh Park treated as residential. DfT's guidance note for promoters
  lists this as one of the three most common causes of inaccurate forecasts.
  The other two are wrong train-service assumptions and methods that miss
  local characteristics, such as new housing or regeneration.
- **Wrong input assumptions.** When the backcast was re-run with outturn
  housing, Aylesbury Vale Parkway went from 100% out to within 10%. When it
  was re-run with the outturn bus link, Liverpool South Parkway went from
  −27% to −11%. The method was fine; the inputs were not.
- **Productions and attractions.** On average 27% of a new station's demand is
  *attracted* (the home end is elsewhere), and 48% at destination stations.
  Produced demand is concentrated: at most stations 80% goes to just three
  destinations.
- **Build-up.** Appraisals assume 50–70% of mature demand in year 1, 75–90%
  in year 2 and 90–100% in year 3. PDFH (B12, major new services) gives 70%,
  85% and 95% at the end of years 1–3, and 100% by year 4. SDFT notes that a
  station may take 5–6 years to mature (Preston & Dargay 2005; Blainey 2009).
- **Reopenings are under-forecast as often as over.** Examples:
  - Borders Railway stations ran up to 8× forecast in year 1 (Transport
    Scotland 2017).
  - Ebbw Vale Parkway ran at 5.5× forecast.
  - The Northumberland Line's outline business case expected 704,000 journeys
    in 2025, and about 920,000 were made in its first year.

### 1.2 Rail and transit forecasts in general

- **Flyvbjerg, Holm & Buhl (2005)**, 210 projects in 14 nations: rail
  passenger forecasts were too high in 9 of 10 projects, by 106% on average,
  and accuracy did not improve over 30 years.
- **Nicolaisen & Driscoll (2014)**, a review of 12 ex-post studies: rail tends
  to over-forecast and roads to under-forecast. Every study blames exogenous
  inputs (growth, car ownership, fuel prices) for part of the error.
- **Hoque, Zhang, Schmitt & Erhardt (2024)**, US Capital Investment Grants
  projects: ridership averages 24.6% below forecast and about 70% of projects
  over-predict. Accuracy has improved since 2000.

The UK new-station record above does not share this optimism bias. Its
median is close to 1, and its error is spread rather than biased.

### 1.3 Assignment validation standards

TAG unit M3.2 (May 2024), Tables 4 and 5, sets **base-year** standards. They
are not forecast standards:

- modelled flows within 15% of counts across screenlines, grouped services and
  grouped stations, for about 95% of cases;
- within 25% on individual services, links and stations, again for about 95%.

Neither applies where the observed flow is below 150 passengers an hour. There
the guidance asks for the differences to be reported and explained with
histograms or scatter plots (§4.3.11).

Every new station in these cases is far below 150 an hour: the busiest,
Brent Cross West, counts about 12 on and off per train. So the TAG criteria
do not formally apply at the new-station level. They are quoted here as the
scale practitioners work to. The whole-train comparisons are closer to TAG's
"individual service" level, but these are forecasts, not base-year fits.

### 1.4 The open-source comparators

The three repositories assessed on 2026-09-19 are covered in
`COMPARATOR_MODELS.md`. Only SDFT forecasts demand. Its trip-end model is now
the model's footfall input for new stations, so on new-station totals the
comparison in §2 is with SDFT's own error carried through our allocation.

## 2. This model on the same scale

The re-run results are in `datasources/cases/_results/new_station_eval.csv`,
written by `new_station_eval.py` beside it.

**Measure.** The measure is predicted against counted boardings plus
alightings at the new station, per counted weekday train-day, on the modelled
operator's trains (calibrated model, joint mode). Two things make it harsher
than the literature's measure:
- the literature compares annual station totals, while this also tests the
  split between operators and over weekdays;
- counts are one operator's trains over weeks, not a year of ticket sales.

**Scope.** There are 12 stations: six on the Northumberland Line, plus
Cambridge South, Brent Cross West, Thanet Parkway, Marsh Barton, Headbolt Lane
and Beaulieu Park. Two sets of counts are excluded:
- GTR's counts at Cambridge South, which are contaminated (§3.2);
- case 03's Cambridge South counts, which cover only 5 train-days.

### 2.1 New-station totals

| | n | Median ratio | sd(log) | Within ±25% | Within ×2 | Median abs % error |
|---|---|---|---|---|---|---|
| **This model, unbalanced gravity** | 12 | 0.50 | 1.01 | 8% | 33% | 65% |
| **This model, SDFT × 0.8 balance** | 12 | **1.12** | **0.57** | **50%** | **67%** | **33%** |
| … year-consistent | 12 | 1.31 | 0.57 | 42% | 67% | 47% |
| … year-consistent, with PDFH build-up | 12 | 0.90 | 0.64 | 25% | 75% | 35% |
| … by scheme (7), as reported | 7 | 1.48 | 0.71 | 29% | 43% | 65% |
| … by scheme, year-consistent, with build-up | 7 | 1.13 | 0.75 | 29% | 57% | 59% |
| UK scheme appraisals (SDG 2010) | 23 | 0.94 | 0.61 | 52% | 74% | 24% |
| SDFT's own hindcast | 10 | 1.15 | 0.49 | 30% | 80% | 30% |
| Borders/Highland appraisals | 8 | 0.72 | 1.39 | 12% | 38% | 85% |

**Not shown: that the model is as accurate as UK scheme appraisals.** Station
by station the balanced model's spread is a little narrower than the
appraisals' (0.57 against 0.61), but three things stop that being a like-for-like
result:
- **Years.** Predictions are at each case's demand year (2022-23 to 2024-25),
  and the counts are from 2026. Grown to 2025-26 by each operator's growth in
  ORR table 1223, the median is 1.31.
- **Clustering.** Six of the twelve stations are one line. By scheme the
  median is 1.48, the spread 0.71, and three of seven schemes are within ×2,
  worse than the appraisals.
- **Uncertainty.** Resampling schemes, the 90% interval on the station median
  is 1.00–2.67 and on the sd(log) 0.33–0.91.

The balance and its 0.8 were also chosen after seeing these same stations.
The stronger evidence on new-station totals is SDFT's own held-out test on 32
real new stations (§3.1), which is appraisal-level.

| Station | Unbalanced | Balanced | + build-up | Year-consistent, + build-up |
|---|---|---|---|---|
| Brent Cross West | 4.29 | **1.48** | 1.34 | 1.70 |
| Cambridge South (GA counts) | 3.93 | **2.07** | 1.05 | 1.14 |
| Northumberland Line, pooled | 0.31 | **1.13** | 0.81 | 0.94 |
| Ashington / Seaton Delaval / Newsham | 0.21 / 0.35 / 0.28 | 1.50 / 0.99 / 0.81 | 1.17 / 0.77 / 0.60 | 1.37 / 0.91 / 0.71 |
| Blyth Bebside / Northumberland Park / Bedlington | 0.42 / 0.36 / 0.46 | 1.17 / 0.82 / 1.07 | 0.76 / 0.47 / 0.59 | 0.89 / 0.55 / 0.69 |
| Beaulieu Park | 0.87 | 1.00 | 0.63 | 0.68 |
| Thanet Parkway | 1.72 | 2.67 | 2.52 | 3.07 |
| Headbolt Lane | 1.41 | 2.72 | 2.50 | 3.05 |
| Marsh Barton (8 train-days) | 0.54 | 0.35 | 0.34 | 0.41 |

"Unbalanced" is the gravity model on SDFT's footfall, with no balance. The
SDFT forecasts behind every column changed on 27 September: Brent Cross West
+36%, Cambridge South +32%, Marsh Barton +37%, Beaulieu Park +23%, Ashington
−17%. That is why the figures differ from the 26 September version.

### 2.2 Whole trains

Whole-train figures mostly measure existing passengers (§3.5). The exception
is the Northumberland Line, where the new stations are most of the load; there
the balance and the direction split together are a large improvement:

| Case 01 train-days (n = 315) | Ratio | RMSE | Mean GEH | MAPE | r |
|---|---|---|---|---|---|
| SDFT footfall, unbalanced | 0.21 | 46.1 | 6.56 | 71% | 0.23 |
| + balance | 0.75 | 28.9 | 3.51 | 58% | 0.21 |
| + direction split on new-station legs | 0.75 | **25.3** | **2.78** | **41%** | **0.55** |
| + the split's station-function term | 0.75 | 24.8 | 2.69 | 39% | 0.58 |

The level fell from 0.87 on 26 September to 0.75 with the corrected SDFT
forecasts (Ashington −17%, Seaton Delaval −11%).

Elsewhere the new station is 5–20% of the load, so balancing it barely moves
whole trains. Splitting the *existing* passengers' legs by direction does. The
table covers 38,653 counted segments on trains outside case 06's sample, where
the split was fitted. Figures are rmse(log) / r of segment loads.

| Case | Ratio, unbalanced → balanced | Balanced | Split on every leg |
|---|---|---|---|
| 05 Cambridge South (GA) | 0.68 → 0.64 | 0.750 / 0.57 | 0.661 / 0.81 |
| 05 Cambridge South (GTR) | 0.56 → 0.53 | 0.861 / 0.56 | 0.769 / 0.65 |
| 11 Brent Cross West | 0.77 → 0.71 | 0.779 / 0.44 | 0.721 / 0.57 |
| 12 Thanet Parkway | 0.53 → 0.55 | 0.833 / 0.41 | 0.803 / 0.65 |
| 14 Headbolt Lane | 0.94 → 1.19 | 0.735 / 0.54 | 0.693 / 0.70 |
| 15 Beaulieu Park | 0.99 → 0.99 | 0.888 / 0.38 | 0.567 / 0.77 |
| all nine runs, pooled (median r) | | 0.792 / 0.44 | **0.706 / 0.65** |

- **Brent Cross West** gets slightly worse with the balance, because its
  unbalanced over-prediction had been hiding an under-prediction of the
  existing passengers.
- **The split's gain** is the first evidence that the tidal split fitted on
  case 06 holds on other trains (`MODEL_REVIEW_2026-09-27.md` §4.3).
- **Case 03 is the exception.** Its four trains all run in the peak direction,
  and the split over-boosts the earliest one (§4.4 there).
- **The whole-train level stays at 0.53–0.75** on the London operators' trains
  (§3.5).

### 2.3 Direction at the new station

The split fixes the morning direction at residential stations. Share of
morning-peak on+off that is boarding:

| Station | Counted | Unsplit | Split | Split + function term |
|---|---|---|---|---|
| Beaulieu Park | 0.97 | 0.49 | 0.78 | 0.72 |
| Thanet Parkway | 0.89 | 0.46 | 0.72 | 0.77 |
| Brent Cross West | 0.81 | 0.52 | 0.64 | 0.62 |
| Northumberland Line | 0.74 | 0.50 | 0.58 | 0.57 |
| Headbolt Lane | 0.42 | 0.52 | 0.36 | 0.50 |
| Cambridge South (GA) | 0.16 | 0.48 | 0.59 | 0.40 |

It moves Beaulieu Park's per-train correlation from −0.02 to **0.64**. Keyed
on footfall alone, it pushes Cambridge South the wrong way, because Cambridge
South is a destination. The station-function term (§3.4) turns it back towards
the count, at a small cost at Beaulieu Park.

### 2.4 Against TAG

No new station reaches 150 passengers an hour, so under M3.2 §4.3.11 the
station comparisons are reportable, not pass/fail. As a scale reference only:
6 of the 12 are within TAG's 25% for individual stations after the balance, and
1 before.

## 3. Why the model is still wrong, largest first

### 3.1 SDFT's error passes straight through the balance

This is the intended trade: the balance swaps the gravity model's unbounded
error for SDFT's, and where SDFT is off, so is the station. How large SDFT's
error is on real new stations is now measured (`MODEL_REVIEW_2026-09-27.md`
§2.1). The port was calibrated on existing stations only, then asked to
predict the 2024-25 usage of the 32 England and Wales stations that opened
between 2012-13 and 2023-24. Result: median 0.90, sd(log) 0.70, 72% within ×2
— about what scheme appraisals achieve.
- **Mature stations** are under-forecast (median 0.78–0.88).
- **Parkways on fast London lines** are the big misses: Oxford Parkway 0.3–0.4,
  Worcestershire Parkway 0.15–0.19.
- **New-build catchments** are the other: Cranbrook 0.13–0.40, Barking
  Riverside, Meridian Water.

Nothing tried improves it: not the jobs radius (now SDFT's own two-minute
drive, about a mile, instead of 200 m), not resolving short-term schedules in
the train counts, and not refitting every slope to modern data. The 26
September conclusion, that SDFT's trip-end model is the biggest remaining
lever, rested on four stations. Their 2024-25 usage was their build-up year,
and it was compared with forecasts made at 2022-23 demand. **It does not hold.**
Better inputs for parkways and new-build catchments are the lever, and only
for those kinds of station.

### 3.2 The outturn itself disagrees

- **Thanet Parkway:** Southeastern's weekday counts, at full coverage, come
  to about 55,000 a year. Like for like, that is about 55% of ORR's 124,000 cut
  to weekdays (× 0.8), not the 44% first quoted. With SDFT at 1.49 × ORR's
  2024-25 figure, that factor of about 1.8 accounts for the whole balanced
  ratio of 2.67.
- **GTR:** the counter restarts mid-journey on 15% of Thameslink train-days
  (127 of 869, at City Thameslink, Harlington, Gatwick, Luton and elsewhere).
  At Cambridge South it restarted on 60% before the screen. Even after the
  screen, a few partial restarts (5% of train-days) inflate GTR's mean there
  to 22 on+off per train, against medians of 1 off and 2.5 on. Treat GTR's Cambridge South
  counts as unusable.
- **Cambridge South** opened on 28 June 2026, so the Greater Anglia counts are
  its first five weeks. The old window started four weeks before opening.
- **Marsh Barton** has only 8 counted train-days in the one usable GWR
  outturn week.

### 3.3 Operator share at stations other operators serve

Headbolt Lane's total is shared by gravity over every train that calls there.
Northern's modelled share is 16.6%, against a rough estimate from the counts
of 8–10%. Headbolt Lane is also an interchange: 58% of Northern's
morning-peak on+off there is alighting (people changing onto Merseyrail),
which a direct-pairs model cannot represent.

### 3.4 Station function

The direction split keys on footfall ratio, so it treats every small station
as residential. That is right for Beaulieu Park, Thanet Parkway, Brent Cross
West and the Northumberland Line. It is wrong for destinations (Cambridge
South's biomedical campus, Marsh Barton's trading estate) and for Headbolt
Lane's interchange.

A production/attraction indicator fixes the destinations. It is
`_sdft/station_function.py`: residents ÷ jobs within 1 km, fitted on 461
existing stations. It explains 45% of the variance in morning-peak boarding
share and gets Cambridge South (0.15 against 0.13) and Thanet Parkway (0.92
against 0.94) right out of sample. It fails at Beaulieu Park (0.17 against
0.98), where Census 2021 predates the estate and the parkway catchment is far
wider than 1 km. The literature's warning about "the function of the station"
is exactly this.

Since 27 September this is an optional term in the direction split
(`DirectionSplitModel.station_function`). It is keyed on SDFT's catchment ÷
jobs within a mile, and fitted on case 06's segments
(`_baselines/direction_split_model_segments_function.json`).
- **Case 06, held out:** it improves the split a little (rmse(log) 0.726 →
  0.721).
- **At the new stations:** it turns Cambridge South's morning share towards
  the count (§2.3).
- **On whole trains outside case 06:** it adds nothing (pooled rmse(log) 0.706
  → 0.711).

So it stays optional.

### 3.5 Existing passengers

On cases 11–15 the whole-train error is mostly the operator's base, not the
new station. It has three parts.
- **Base year.** The 2022-23 ODM is used against 2026 counts. On 26
  September, growth on each corridor's own pairs between the 2022-23 and
  2024-25 ODMs (Thameslink 1.14, Southeastern 1.22, GWR 1.10, Northern 1.26)
  moved the train ratios nearer 1 on average, but no narrower.
- **Direction.** The tidal split on existing passengers' legs cuts pooled
  rmse(log) from 0.792 to 0.706 on these cases (§2.2).
- **Trip length.** On case 06's fully counted trains the model's passengers
  ride 4.36 segments against 5.27 counted. Trips are about 19% too short, not
  too few. This is what holds the level at 0.53–0.75 on the London operators'
  trains. The leading explanation is that journeys needing a change are never
  assigned to any train (`MODEL_REVIEW_2026-09-27.md` §3.1–3.2).

### 3.6 Build-up and weekday share

**Build-up.** PDFH build-up improves the scheme median (1.48 → 1.05) but not
the spread (0.71 → 0.73). Our own counts show a fast early plateau: Beaulieu Park and
Northumberland Park rise about 20% over their first 5–8 months. So PDFH's 50%
at opening is probably too low for stations like these.

**Weekday share.** Each operator's weekday share of annual demand (0.74 GTR to
0.81 Southeastern) would lower most targets by 3–7% against the assumed 0.8.

### 3.7 Network data

- **Track distances:** 10% of the pairs the cases use have impossible routed
  distances. Headbolt Lane to Kirkby routes at 39.5 miles against 1 mile
  straight-line.
- **Station codes:** the timetable books three stations under codes the
  demand data does not hold, including Moorfields as XMR.
- **Northern's feed** has no departure times, so its stations cannot enter
  any by-period fit.

These fed wrong distances and missing footfall into the gravity model before
this re-run.

## 4. What changed

### 4.1 On 2026-09-26

**Model**
- **Station balance:** `forecast_footfall_weekday_share` holds each
  SDFT-footfall station's weekday throughput to SDFT × 0.8. It covers every
  operator's trains at the station, and is fitted jointly where a pair links
  two new stations.
- **Direction split for new-station legs:** `direction_split_legs:
  synthetic` applies the existing fitted split to those legs.
- **Distance fallback:** routed distances below 0.9× or above 3× the
  straight line are replaced by 1.2× straight line, in both the gravity fit
  and prediction.
- **Aliases:** XMR → MRF, XLV → LVC and WJL → WIJ.

**Data quality**
- **Count screen:** train-days whose counts restart mid-journey are dropped.
- **Empty inputs:** a date window with no counts, or a hindcast with no
  period matrices, now stops with an error. It used to predict zero and report
  success, as Marsh Barton's first run did.
- **Marsh Barton:** baseline from the first GWR count week, outturn from the
  second.
- **Cambridge South:** outturn from its opening day.

**Speed**
- **Timetable lookup:** a per-day index, 84× faster per pair.
- **Parsed counts:** hindcasts reuse the parsed-count cache; re-parsing GTR's
  workbooks was about 43 minutes per run.
- **Matrices:** each file is parsed once rather than once per column.
- **BPLAN graph:** built vectorised, 28 s → 7 s.
- **Parallel runner:** `run_cases.py --jobs N`, with GTR steps weighted by
  memory.
- **Result:** every output identical (checked file by file), and 39 steps now
  take 35 minutes against 2 h 41 min for the previous 13. Brent Cross West went
  from 63 to 7 minutes.

### 4.2 On 2026-09-27

From the review in `MODEL_REVIEW_2026-09-27.md` (§1 there has the detail and
the tests):
- **Split and substitution** did not compose. On runs with the split on
  existing passengers' legs, a London terminal could hold more than its whole
  group, and Colchester → Liverpool Street came out 52 times too big.
- **Slower trains** made their London terminal less attractive, so every
  stopping train's London legs lost demand (−14% on Thameslink).
- **Train counts in the split**: a candidate still in the timetable was
  counted twice, and a new train's reverse-running fellows not at all.
- **SDFT port**: jobs are now counted within SDFT's two-minute drive (about a
  mile) rather than 200 m, and short-term schedules are resolved in the
  calibration-era train counts.
- **Table 1410 reader**: it no longer writes strings into an integer column,
  which a later pandas would have refused.
- **Station-function term**: added to the direction split, optional.

Every case that reads the SDFT footfall or has London legs was re-run: 58
runs, none failed.

## 5. What would help most next

1. **A national base-year check.** Assign the full ODM to the full timetable
   and compare every counted train of the ten operators with feeds, testing
   fixes with whole corridors held out (`MODEL_REVIEW_2026-09-27.md` §5). The
   whole-train level (0.53–0.75 on the London operators) and the short trips
   (§3.5) are problems of assigning the base matrix, which the new-station
   cases cannot isolate.
2. **A transfer assignment**, the leading explanation for the short trips.
3. **The split on every leg** wherever the split is used, with its slopes
   refitted on (1). The early-period slope is suspect.
4. **An event study of real timetable changes**, for frequency and new-link
   response. It needs the December 2024 and May 2025 CIF archives.
5. **SDFT inputs for parkways and new-build catchments** (§3.1), the only
   kinds of station where better inputs could help.
6. **Resolve Thanet Parkway's two outturn sources**, and get GTR to fix or
   explain the restarts.
7. **Re-run cases 02 and 06–10** on the fixed code. Case 04 is re-run, but its
   sensitivity folders are not.

## Sources

- Steer Davies Gleave for DfT (2010). *Station Usage and Demand Forecasts for
  Newly Opened Railway Lines and Stations*, final report.
  <https://assets.publishing.service.gov.uk/media/5a7997aced915d07d35b6af2/demand-forecasting-report.pdf>
- DfT. *Guidance note on passenger demand forecasting for third party funded
  local rail schemes*.
  <https://assets.publishing.service.gov.uk/media/5a79dbed40f0b66d161aea80/guidance-note.pdf>
- Young, M. & Blainey, S. (2019). *An automated online tool to forecast demand
  for new railway stations*. <https://www.marcusyoung.co.uk/files/tpm2019.pdf>
- Blainey, S. (2010). Trip end models of local rail demand in England and
  Wales. *Journal of Transport Geography* 18(1), 153–165.
- DfT (2024). *TAG unit M3.2 Public Transport Assignment Modelling*.
  <https://assets.publishing.service.gov.uk/media/666af32effd07973a043d110/tag-unit-m3.2-public-transport-assignment-modelling.pdf>
- Flyvbjerg, B., Holm, M. K. S. & Buhl, S. L. (2005). How (in)accurate are
  demand forecasts in public works projects? *JAPA* 71(2), 131–146.
- Nicolaisen, M. S. & Driscoll, P. A. (2014). Ex-post evaluations of demand
  forecast accuracy: a literature review. *Transport Reviews* 34(4), 540–557.
- Hoque, J. M., Zhang, I., Schmitt, D. & Erhardt, G. D. (2024). Are public
  transit investments based on accurate forecasts? *Transportation Research
  Part A* 186.
- Northumberland County Council (2025), first-year patronage release; New
  Civil Engineer, 19 Dec 2025.
