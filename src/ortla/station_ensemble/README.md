# Station ensemble

`ortla station-ensemble` forecasts a station's annual entries and exits. It fits thirteen
trip-end models to established stations, averages them, and adds nearby stations' residuals.
It is the more accurate of the two new-station models here; `ortla new-station` is SDFT's model
with the same borrowing.

| | Cross-validated, 1,799 stations | 69 openings, 1998-2023 |
| --- | ---: | ---: |
| rmse(log) | 0.526 | 0.440 |
| Median forecast / actual | | 0.94 |
| Within a factor of 2 | | 0.84 |
| 80% interval, typical width; coverage | | x/1.78; 0.77 |

These are the scores `station_ensemble_scores.csv` reports (row `ensemble`).

The cross-validation holds out 25 km cells, five folds. The openings are kept out of every fit,
the gravity model's pairs included. Scotland is never fitted to: on its 331 established stations
the same forecast scores 0.585.

## How it works

1. Stations (`frame.py`). Every station open in the June 2024 timetable with Knowledgebase
   attributes. Catchments follow SDFT: each postcode's residents are shared among its ten
   nearest stations by a logit on distance, trains and facilities. Station choice counts every
   service SRAM lists. The trip end counts ticketed trains only, leaving out the Underground, the
   Metro, Supertram, Eurostar and heritage lines, whose passengers are not in ORR's figures.
2. Reach of the network (`gravity.py`). A Poisson gravity model of the ORR origin-destination
   matrix gives each station a mass. A station's accessibility is the other stations' masses it
   reaches by train. Each destination counts in proportion to how much better this station serves
   it than its neighbours do.
3. Terms (`terms.py`). SDFT's terms (residents, trains, jobs, car spaces, electric, terminus,
   Travelcard), plus visitor beds, car catchments, jobs at workplace zones, request stops, venues,
   Sunday service, staffing and operator. The ridge member reads all 67.
4. Members (`members.py`). The same trip end fitted by least squares, boosted trees, a random
   forest, Poisson regression, median regression, ridge, splines and analogues. Three members
   differ in their inputs instead: SDFT's form, residents shared by service, and a three-year
   footfall target. Kriging alone is the thirteenth.
5. Ensemble (`ensemble.py`). The plain mean of the thirteen, plus the in-sample residuals of
   established stations, weighted exp(-km / 4) and shrunk by 0.5. The interval comes from the
   cross-validated errors in the forecast's size fifth.

Proposed stations (`proposals.py`) are added to the network as the config describes them, and
every term is rebuilt with them open. Their neighbours' forecasts with and without them are
written too.

## Data

Set `data_dir` to a folder laid out like this:

| Path under `data_dir` | Source |
| --- | --- |
| `table-1410/by_year/table_1410_YYYY_YY.csv` | ORR table 1410, estimates of station usage, every year from 1997-98 |
| `ODM_for_RDM_2024-25.csv` | ORR origin-destination matrix, 2024-25 (Rail Data Marketplace) |
| `stations.csv` | Station list with `crsCode`, `lat`, `long` |
| `station-covariates/station_covariates.csv` | NRE Knowledgebase stations: position, facilities, car spaces, staffing, operator |
| `output_paper/timetable_jun24/timetable_calls.csv` | `ortla timetable` on the June 2024 CIF (reference Wednesday 18 September 2024) |
| `output_paper/timetable_may26/timetable_calls.csv` | `ortla timetable` on the 24 June 2026 CIF |
| `timetable-data/new-schedule/CIF_ALL_FULL_DAILY_toc-full.CIF` | Network Rail full CIF extract (request stops) |
| `sram/SRAM_Summer24_O04_StopsAndPasses.parquet` | Network Rail SRAM, Summer 2024 stops and passes |
| `sdft-inputs/census2021_postcode_population.csv` | Census 2021 residents by postcode (NOMIS) |
| `sdft-inputs/codepoint_open/Data/CSV/` | OS Code-Point Open |
| `sdft-inputs/census2021_wp001/WP001_oa.csv` | Census 2021 workplace population by output area |
| `sdft-inputs/oa2021_pwc_osgb.csv`, `oa2011_pwc_osgb.csv` | Output area population-weighted centroids, British grid |
| `sdft-inputs/census2021_ts/census2021-ts045-oa.csv` | Census 2021 TS045, cars or vans |
| `sdft-inputs/census2011_scotland/` | Scotland's Census 2011 KS101SC, KS404SC and output area centroids; BRES 2024 jobs by data zone and data zone centroids |
| `sdft-inputs/census2011_extra/` | Census 2011 WP101 (workplace population) by output area; jobs by workplace zone and zone centroids |
| `sdft-inputs/naptan/naptan.csv` | NaPTAN stops (rail access points, bus stops) |
| `sdft-inputs/osm/pois.csv` | OpenStreetMap places to stay (`scripts/fetch_station_ensemble_osm.py beds`) |
| `osm_geofabrik_2026-11-11/*.csv` | OpenStreetMap points by feature (`scripts/fetch_station_ensemble_osm.py osm`) |
| `osm_overpass_2026-11-11/wikidata_enwiki.csv` | Wikidata items with an English article (`scripts/fetch_station_ensemble_osm.py wikidata`) |
| `lookups/oa11_msoa11.csv` | 2011 output area to MSOA lookup |
| `census2011_wu03ew/wu03ew_msoa.csv` | Census 2011 WU03EW, commuting between MSOAs |

The first run takes about ten minutes, most of it six fits of the gravity model. Set `cache_dir`
to keep them; with the cache, a run takes a minute or two. Empty it when the inputs change.

## Outputs

- `station_ensemble_scores.csv`: each member's scores and the ensemble's (row `ensemble`).
- `station_ensemble_openings.csv`: the 69 openings' forecasts, intervals and outturns.
- `station_ensemble_forecasts.csv`: each proposed station's forecast and 80% interval, with its
  inputs.
- `station_ensemble_neighbours.csv`: existing stations near a proposal, forecast without and with it.

## Caveats

- Staffing and Sunday service come from today's timetable and Knowledgebase. At an opening they
  may follow demand seen since it opened. With both predicted from the other inputs instead, the
  openings score 0.489 rather than 0.472 on the least-squares member. For a station not yet open,
  enter what the promoter plans.
- Parkway openings are forecast about a quarter too high on average, and established parkways'
  errors run from x0.26 to x3.6. Use the interval.
- Forecasts are medians. For a mean, multiply by the lift for the forecast's size fifth:
  1.27, 1.16, 1.12, 1.09 and 0.93 from smallest to largest.
