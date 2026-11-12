# The reference train-loading model

The model behind the accuracy table in the main README, frozen on 10 November 2026. `build.sh`
rebuilds it from the raw files: every intermediate is written fresh, and the scores come out the
same.

```sh
reference/build.sh /path/to/datasources /path/to/output          # everything; several hours
reference/build.sh /path/to/datasources /path/to/output assign   # one step again
```

The steps, in order:

1. `timetables`: `ortla timetable` on two CIFs. The 11 September 2026 extract gives the timetable
   the operators' counts are matched to. The 24 June 2026 CIF is the one assigned, because the
   September extract lacks about 850 trains on 17 June 2026, nearly all of them HS1.
2. `loading`: `ortla loading` for each of the nine counted operators. Each writes its period
   matrices: the ODM split by day and period using that operator's counts.
3. `means`: `means.py` averages the period matrices for trains of operators without counts.
   A flow with either end within 20 km of Charing Cross takes the mean of Greater Anglia, GTR,
   Southeastern and South Western; every other flow takes the mean of all nine.
4. `assign`: `ortla assign` loads every train on the five weekdays of the week of 17 June 2026
   and scores the loads against the counts. It writes `assign_scores.csv` and the other tables
   to `OUT/output`.

`configs/` holds the configs with `{DATA}` and `{OUT}` in place of the folders; `build.sh` fills
them in and writes the result to `OUT/cfg`.

## Inputs

Under the data folder:

| Path | Source |
| --- | --- |
| `ODM_for_RDM_2024-25.csv` | ORR origin-destination matrix, 2024-25 (Rail Data Marketplace) |
| `old-odm/ODM_for_rdm_2023-24.csv`, `old-odm/ODM_for_rdm_2019-20.csv` | Earlier ODM years, for the station size correction |
| `table-1410/by_year/table_1410_2024_25.csv` | ORR estimates of station usage |
| `stations.csv` | Station list with CRS code, latitude and longitude |
| `NWR_TrackModel20260904/NWR_TrackCentreLines.shp` | Network Rail track model |
| `sram/SRAM_Summer24_O03_ODHourlyRoutes.parquet` | Network Rail SRAM, Summer 2024 hourly OD profile |
| `timetable-data/new-schedule/CIF_ALL_FULL_DAILY_toc-full.json` | Network Rail full timetable extract, 11 September 2026 |
| `timetable-data/various/May26 publsihed 240626.cif` | The published May 2026 timetable, CIF of 24 June 2026 |
| `<operator>-loading-data/` | Each operator's loading counts (see the configs for the file names) |

The counts are the operators' own and are not published. Avanti West Coast, East Midlands
Railway and TransPennine Express each have one file; the other six have a folder of files.
