<h1 align="center">ORTA</h1>

<p align="center">
  <b>Open Rail Train-level Assignment</b><br>
  Passenger demand for every train on the rail network in Great Britain
</p>

<p align="center">
  <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white">
  <img alt="Licence: GPL v3" src="https://img.shields.io/badge/licence-GPL%20v3-blue">
  <img alt="Tests: 252 passing" src="https://img.shields.io/badge/tests-252%20passing-2ea44f">
  <img alt="Code style: ruff" src="https://img.shields.io/badge/code%20style-ruff-D7FF64?logo=ruff&logoColor=black">
  <img alt="Desktop app: Qt" src="https://img.shields.io/badge/desktop%20app-Qt-41CD52?logo=qt&logoColor=white">
</p>

<p align="center">
  <img src="docs/images/links.png" width="49%" alt="The 1,500 busiest links between calls on a Wednesday">
  <img src="docs/images/boardings.png" width="49%" alt="Passengers boarding at each station on a Wednesday">
</p>

ORTA turns the ORR's annual origin to destination matrix (ODM) and operators' loading
counts into demand by day of the week and time of day, then puts that demand onto the
timetable train by train. One run loads all of the roughly 23,000 trains that run on a
weekday and scores every load it can match against the counts.

The same model answers the planning questions: how many people would use a new or
retimed service, whether it fits the timetable, what it is worth on DfT TAG values,
what it would earn an operator, and how many people a new station would draw. Services
that have already launched can be hindcast, so a forecast can be checked against what
actually happened.

Everything runs from the command line or from a desktop app with interactive charts
and maps.

## Contents

- [Accuracy](#accuracy)
- [What it does](#what-it-does)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Commands](#commands)
- [The adopted model](#the-adopted-model)
- [Desktop app](#desktop-app)
- [Data](#data)
- [Repository layout](#repository-layout)
- [Development](#development)
- [Licence](#licence)

## Accuracy

With the adopted settings, `orta assign` was run over all five weekdays of the
timetable and scored against nine operators' counts.

| Operator | Counted segments | Predicted / counted | rmse(log) |
| --- | ---: | ---: | ---: |
| Avanti West Coast | 6,144 | 1.03 | 0.41 |
| Govia Thameslink Railway | 174,566 | 0.98 | 0.46 |
| Greater Anglia | 50,362 | 0.99 | 0.46 |
| Northern | 141,134 | 1.03 | 0.46 |
| Southeastern | 74,402 | 0.95 | 0.49 |
| South Western Railway | 50,614 | 0.95 | 0.54 |
| TransPennine Express | 14,179 | 1.07 | 0.60 |
| Great Western Railway | 31,727 | 0.97 | 0.61 |
| East Midlands Railway | 4,685 | 1.11 | 0.74 |
| **All nine** | **547,813** | **0.98** | **0.49** |

A segment is one train between two calls. The rmse(log) is the root mean square of
log(predicted / counted), so 0.49 means a typical segment is within a factor of about
1.6 of its count. These figures are in sample. Measured on corridors held out of
fitting, the same settings score 0.461 on a screened set of counts. South Western
Railway is scored on its counts from before April 2026, when the basis of its counts
changed.

<p align="center">
  <img src="docs/images/scatter.png" width="85%" alt="Predicted against counted load on 538,527 segments">
</p>

## What it does

```mermaid
flowchart LR
    ODM[ORR ODM] --> matrix
    counts[Operator counts] --> loading
    matrix --> loading[loading<br>period matrices]
    CIF[CIF timetable] --> timetable
    loading --> assign
    timetable --> assign
    loading --> propose & scenario & hindcast & discover
    timetable --> propose & scenario & hindcast & discover
    propose --> commercial
    SDFT[Station data] --> newstation[new-station]
```

- **Demand by period.** The annual ODM is split by day of the week and time period
  using the operators' counts, with a station size correction and an hourly OD profile.
- **Train loading.** Each slot's demand is shared between the trains that serve it.
  A small neural network reads each train's place in the day's timetable (the gap to
  the trains either side, the stops between and so on) to decide its share. One-change
  journeys are routed through interchanges, and a fitted direction split tells the
  morning peak into London from the evening peak out.
- **Forecasting.** A candidate service gets the share it would take if it were added to
  the timetable, plus an uplift for station pairs that gain a direct link. Feasibility
  checks use BPLAN running times, headways and HSP. Appraisal uses the TAG Data Book,
  and the commercial case reports NPV, IRR and payback.
- **New stations.** A port of the Station Demand Forecasting Tool's model, with 80% and
  90% intervals and the demand taken from neighbouring stations.
- **Hindcasts.** A launched service is stripped from the timetable, predicted, and
  compared with its counts.

## Installation

ORTA needs Python 3.11 or later. It is not on PyPI yet, so install it from a clone:

```sh
git clone https://github.com/rosedillon/ORTLA.git
cd ORTLA
pip install -e ".[gui]"        # add ,dev for the test and lint tools
```

## Quick start

Every command reads one YAML file. Copy the template for it from
[`examples/`](examples) into `configs/` (which git ignores, since it holds your own
paths and any NRDP password), fill in the paths, and run it:

```sh
mkdir -p configs
cp examples/matrix.yml examples/loading.yml examples/assign.yml configs/
orta matrix  configs/matrix.yml     # annual station to station matrix
orta loading configs/loading.yml    # matrices by day and period
orta assign  configs/assign.yml     # load every train and score it
```

`assign` writes five tables to its output folder:

| File | One row per |
| --- | --- |
| `assign_segments.csv` | Train between two calls, with load, boardings, alightings and the count where there is one. |
| `assign_trains.csv` | Train, with its peak load and where it occurs. |
| `assign_stations.csv` | Station and day: boardings and alightings. |
| `assign_links.csv` | Pair of consecutive calls: passengers over the day. |
| `assign_scores.csv` | Operator: segments scored, predicted / counted and rmse(log). |

`orta -h` lists the commands. Configs are read as strict YAML, which rejects inline
lists such as `[a, b]`, so write lists one item per line:

```yaml
days:
  - Tuesday
  - Wednesday
```

## Commands

Each command is run as `orta <command> <config.yml>`.

| Command | What it does |
| --- | --- |
| `distance` | Track distance along a sequence of stations. |
| `matrix` | Build the annual station to station matrix from an ODM release. |
| `loading` | Read the loading counts and build the period matrices. |
| `lookup` | Expected trips between station pairs, by day and period. |
| `assign` | Load every train in the timetable and score the loads against counts. |
| `timetable` | Clean a CIF timetable and count route frequencies. |
| `services` | List the timetabled trains between station pairs. |
| `bplan` | Clean a Network Rail BPLAN Geography extract into a timing graph. |
| `propose` | Check candidate services against the timetable and predict their demand. |
| `fares` | Look up RSP standard fares for station pairs. |
| `commercial` | Operator's commercial case (NPV, IRR, payback) for proposed services. |
| `hindcast` | Strip a launched service, predict it, and compare with the counts. |
| `scenario` | Sweep departure times and frequencies for a candidate and appraise each. |
| `discover` | Scan the network for new direct services worth a closer look. |
| `new-station` | Forecast a new station's annual entries and exits, with 80% and 90% intervals. |

The loading readers cover Northern, TransPennine Express, Avanti West Coast, East
Midlands Railway, Great Western Railway, Govia Thameslink Railway, South Western
Railway, Southeastern and Greater Anglia.

## The adopted model

Every model setting is off by default, so a config without them runs the plain model.
The `assign`, `propose`, `scenario` and `hindcast` templates already carry the settings
that scored best on held-out corridors:

```yaml
direction_split_model: national
slot_share_model: neural
slot_share_profile: "/path/to/SRAM_Summer24_O03_ODHourlyRoutes.parquet"
transfer_scale: 1.20
transfer_scale_london: 2.30
transfer_choice_theta: 0.2
```

The `loading` template carries the station size correction
(`station_size_exponent: -0.097`) and SRAM's hourly OD profile (`od_profile_path`).
Build the matrices that `assign` and the forecasting commands read with both.

Three fitted models ship inside the package, so a config can name one instead of
giving a path:

| Name | Setting | What it is |
| --- | --- | --- |
| `national` | `direction_split_model` | The direction split fitted to every counted operator. |
| `m3` | `slot_share_model` | Each train's share of its slot from its timing and journey time. |
| `neural` | `slot_share_model` | `m3` plus a small neural network on 24 timetable features. |

## Desktop app

```sh
orta-gui            # or: python -m orta.gui
```

The app runs every command. Its Analysis page builds a form from each command's
config, with help for each setting, and you can switch between the form and the YAML
at any point. Runs go to a background queue. When one finishes, its outputs open as
interactive charts, maps and tables made for that command: a hindcast gets a scorecard
by model variant, the error by hour and along the line and a map of where it runs over
or under, while `assign` gets the charts shown on this page.

| Page | What it does |
| --- | --- |
| Dashboard | Start or open a project, with a card for each command. |
| Pipeline | Run several commands in order, each reading what the one before wrote. |
| Jobs | Follow, cancel and review background runs. |
| Maps | Draw the whole track network, coloured by daily trains. |
| Help | A Quick Start page that sets up a data folder, and a reference page per command. |

A project (a `.orta` file) remembers every command's settings. Charts work offline;
map tiles need an internet connection.

## Data

ORTA reads published industry data. None of it is included here.

| Data | Source | Used by |
| --- | --- | --- |
| Origin to destination matrix (ODM) | ORR | `matrix` |
| Estimates of station usage (table 1410) | ORR | the ODM rebalance, `new-station` |
| Timetable (CIF) | Network Rail open data | `timetable`, `assign`, forecasting |
| BPLAN Geography | Network Rail | `bplan`, `propose` |
| Fares feed (RSP) | Rail Delivery Group, through NRDP | `fares`, `commercial` |
| Loading counts | Train operators | `loading`, `assign` scores |
| Hourly OD profile (SRAM) | Network Rail | `loading`, slot share |

## Repository layout

| Folder | Contents |
| --- | --- |
| `src/orta` | The package, one module per job, and `gui/` for the desktop app. |
| `tests` | Unit tests, including hand-worked checks of the model's arithmetic. |
| `examples` | A template config for every command. |
| `scripts` | Data preparation, such as the station covariates `new-station` reads. |
| `docs` | Notes on why the model is built the way it is, and what was tried. |

<details>
<summary>Package layout</summary>

| Module | Contents |
| --- | --- |
| `config` | The config model for every command, read from YAML. |
| `clock` | Clock times, weekdays and the periods a day is split into. |
| `stations` | Station codes, aliases and second codes, station lists, table 1410 footfall. |
| `track_distance` | Distances along the track model. |
| `od_matrix` | Read, build and rebalance the annual ODM. |
| `loading_data` | Read each operator's counts into one schema and time each row. |
| `period_matrices` | The `loading` command: temporal factors and period matrices. |
| `od_profile` | Reweight the period matrices towards SRAM's hourly OD profile. |
| `direction_split` | Split a direction-symmetric matrix by direction. |
| `trip_lookup` | The `lookup` command. |
| `timetable` | CIF cleaning, the calls table and route frequencies. |
| `cif_archive` | Read a fixed-width archive CIF as if it were the JSON feed. |
| `wtt_workbook` | Read the Working Timetable books. |
| `service_lookup` | The trains running between two stations on a given day. |
| `bplan_geography` | BPLAN Geography: minimum running times and running lines. |
| `feasibility` | Running-time, headway and HSP checks; departure-window search. |
| `gravity_model` | The calibrated gravity model of annual journeys. |
| `network_assignment` | Load a whole day's timetable; routed one-change journeys. |
| `slot_net` | The neural slot share's network and the timetable features it reads. |
| `assign` | The `assign` command: every train's loads, scored against counts. |
| `models` | The fitted models that ship with the package (`national`, `m3`, `neural`). |
| `demand` | Predict a candidate's demand leg by leg, and its on-board loads. |
| `fares` | Download, clean and look up the RSP fares feed. |
| `propose` | The `propose` command. |
| `appraisal` | TAG Data Book values and the economic appraisal. |
| `commercial_case` | The operator's cashflow, NPV, IRR and payback. |
| `scenario` | Scenario Studio: sweep, check and appraise a candidate. |
| `discovery` | The network scan for new direct services. |
| `hindcast` | Strip, predict and compare a launched service. |
| `new_station` | The `new-station` command: SDFT's station choice, catchment and trip end. |
| `gui` | The desktop app (optional; needs the `gui` extra). |

</details>

Upgrading from `open-rail-model`, or from module names before October 2026? See
[docs/MIGRATION.md](docs/MIGRATION.md). A small alias package keeps
`import open_rail_model` working.

## Development

```sh
pip install -e ".[dev,gui]"
pytest            # unit tests, doctests and the desktop app's tests
ruff check .      # lint
ruff format .     # format
```

The desktop app's tests are skipped without the `gui` extra. On a machine with no
display, set `QT_QPA_PLATFORM=offscreen`. The code follows PEP 8 with 99-character
lines, and docstrings use the numpy layout. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Licence

ORTA is free software under the [GNU General Public License v3](LICENSE).
