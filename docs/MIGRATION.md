# Upgrading from older versions

## Module names before October 2026

The modules were renamed and some were merged on 27 October 2026. Scripts written
against the old names need their imports changing as follows.

| Old module | Now in |
| --- | --- |
| `inputs` | `config` |
| `station_aliases` | `stations` |
| `distance_calculator` | `track_distance` |
| `matrix_manipulation`, `matrix_balance` | `od_matrix` |
| `operator_loading`, `departure_times` | `loading_data` |
| `loading_analysis` | `period_matrices` (station lists and table 1410 moved to `stations`) |
| `service_proposal` | `demand`, `feasibility`, `propose` and `service_lookup` |
| `hsp_feed` | `feasibility` |
| `fares_feed`, `fares_lookup` | `fares` |
| `economic_appraisal`, `tag_databook` | `appraisal` |
| `scenario_discovery` | `discovery` |

The clock and period helpers that several modules shared now live in `clock`, and a
few private helpers lost their leading underscore when they became shared (for example
`_filter_services` is `timetable.filter_services`). Hourly and half-hourly period
matrices are now written without a colon in the file name (`odm_Monday_0700.csv`), so
they can be copied to Windows; folders written with the old names still read.

## The old name

Until November 2026 the package was called `open-rail-model` and imported as
`open_rail_model`. It is `ortla` now, and the commands are `ortla` and `ortla-gui`. A
small alias package keeps `import open_rail_model...` working, so older scripts and
pickles still load; it hands back the same modules as `ortla`.

