"""Configuration models for every command, read from YAML files.

Each command is driven by a :class:`caf.toolkit.BaseConfig`, a validated
pydantic model loaded from a YAML file. The settings models
(:class:`StationsConfig`, :class:`TrackCalculatorConfig` and so on)
describe data sources that several commands share, and each runnable
model implements :meth:`BaseRunConfig.run` so the command line can
execute it.

Paths, thresholds and other parameters live here, so the analysis
modules hold no file locations of their own.
"""

from __future__ import annotations

import abc
import itertools
import pathlib  # noqa: TC003  (needed at runtime for pydantic field resolution)
from typing import Literal

import caf.toolkit as ctk
import pandas as pd
import pydantic

from ortla import (
    assign,
    bplan_geography,
    commercial_case,
    discovery,
    fares,
    hindcast,
    new_station,
    od_matrix,
    period_matrices,
    propose,
    scenario,
    service_lookup,
    timetable,
    track_distance,
    trip_lookup,
)
from ortla.clock import (
    format_hhmm,
    hhmm_minutes,
    hourly_period_order,
    hourly_time_cutoffs,
    interval_period_order,
    interval_time_cutoffs,
)


# # # SETTINGS MODELS # # #
class StationsConfig(ctk.BaseConfig):
    """Location and column names of the station coordinates dataset."""

    path: pathlib.Path
    crs: str = "crsCode"
    lat: str = "lat"
    lon: str = "long"


class CoordinateSystems(ctk.BaseConfig):
    """Coordinate reference systems used when projecting geometries."""

    input_crs: str = "EPSG:4326"
    working_crs: str = "EPSG:27700"


class MapConfig(ctk.BaseConfig):
    """Styling and output options for the interactive route map."""

    default_output: pathlib.Path | None = None
    route_colour: str = "red"
    route_weight: int = 4
    station_colour: str = "blue"
    station_marker_type: str = "marker"


class TrackCalculatorConfig(ctk.BaseConfig):
    """What the track distance calculator needs to build its graph."""

    track_shapefile: pathlib.Path
    stations: StationsConfig
    coordinate_systems: CoordinateSystems = pydantic.Field(default_factory=CoordinateSystems)
    map: MapConfig = pydantic.Field(default_factory=MapConfig)


class RouteTypeConfig(ctk.BaseConfig):
    """Distance bands (miles) used to label a journey by route type.

    ``bands`` maps a label to its inclusive upper bound in miles, in
    increasing order; any distance above the last bound takes
    ``fallback``.
    """

    bands: dict[str, float] = pydantic.Field(
        default_factory=lambda: {"commute": 30.0, "regional": 100.0}
    )
    fallback: str = "long_distance"


class AnalysisConfig(ctk.BaseConfig):
    """Preferred orderings for day-of-week and time-period output."""

    day_order: list[str] = pydantic.Field(default_factory=list)
    time_order: list[str] = pydantic.Field(default_factory=list)


_MIN_STATIONS_TO_BATCH = 2


class StationPairMixin(ctk.BaseConfig):
    """One origin/destination pair, or a station list to batch over.

    With ``station_codes`` set (2+ codes), the command runs once for
    every ordered pair among them (both directions, since a corridor's
    trains generally differ by direction) instead of the single
    ``origin``/ ``destination`` pair. That builds network-wide coverage
    from a per-pair command without retyping origin and destination for
    every corridor.
    """

    origin: str | None = None
    destination: str | None = None
    station_codes: list[str] = pydantic.Field(default_factory=list)

    @pydantic.model_validator(mode="after")
    def _resolve_station_pairs(self) -> StationPairMixin:
        has_pair = bool(self.origin) and bool(self.destination)
        has_batch = len(self.station_codes) >= _MIN_STATIONS_TO_BATCH
        if has_pair and has_batch:
            raise ValueError("Set either origin/destination or station_codes, not both.")
        if not has_pair and not has_batch:
            raise ValueError(
                "Set either both origin and destination, or station_codes with "
                "at least 2 stations to batch over."
            )
        return self

    def pairs(self) -> list[tuple[str, str]]:
        """Return the ``(origin, destination)`` pairs to run."""
        if len(self.station_codes) >= _MIN_STATIONS_TO_BATCH:
            # Both directions, sorted for a stable run order.
            codes = sorted({c.strip().upper() for c in self.station_codes if c.strip()})
            pairs = list(itertools.permutations(codes, 2))
            if not pairs:
                raise ValueError(
                    f"station_codes {self.station_codes!r} has fewer than 2 "
                    "distinct station codes; nothing to run."
                )
            return pairs
        if self.origin is None or self.destination is None:  # guarded by the validator
            raise ValueError("Both origin and destination must be set for a single pair.")
        return [(self.origin.strip().upper(), self.destination.strip().upper())]


# # # RUNNABLE MODELS # # #
class BaseRunConfig(ctk.BaseConfig, abc.ABC):
    """Base class for a runnable command configuration."""

    output_path: pathlib.Path

    @property
    def logging_path(self) -> pathlib.Path:
        """The run's log file; creates ``output_path`` if needed."""
        self.output_path.mkdir(exist_ok=True, parents=True)
        return self.output_path / "rail_model.log"

    @abc.abstractmethod
    def run(self) -> None:
        """Execute the command."""


class DistanceConfig(BaseRunConfig):
    """Calculate the track distance along a sequence of stations."""

    calculator: TrackCalculatorConfig
    station_codes: list[str] = pydantic.Field(default_factory=list)
    map_output: pathlib.Path | None = None
    units: str = "miles"

    def run(self) -> None:
        """Route through ``station_codes`` and report the distance."""
        track_distance.run_distance(self)


class MatrixConfig(BaseRunConfig):
    """Build the annual origin-destination matrix from the ODM."""

    source_odm: pathlib.Path
    # ODM releases carrying both NLC and TLC columns, used to map an
    # NLC-only source_odm (older releases) to TLCs.
    nlc_reference_paths: list[pathlib.Path] = pydantic.Field(default_factory=list)
    # Stations outside this list are blanked with ``fill_value``; unset
    # keeps every station (national).
    valid_stations: pathlib.Path | None = None
    chunk_size: int = 500_000
    keep_columns: list[str] = pydantic.Field(
        default_factory=lambda: ["origin_tlc", "destination_tlc", "journeys"]
    )
    fill_value: str = "!"

    def run(self) -> None:
        """Build the OD matrix and write it to ``output_path``."""
        od_matrix.run_matrix(self)


class TimetableEra(ctk.BaseConfig):
    """One published timetable and the dates it was in force.

    The loading counts span six published timetables. A count can only
    be matched to the schedule that was actually running when it was
    taken, so each era carries its own cleaned calls table and validity
    window; see :mod:`ortla.loading_data`.
    """

    # Short label for the timetable, used in logs and in the provenance
    # report.
    name: str
    # Cleaned calls table for this era, from a `timetable` run over its
    # extract.
    calls_path: pathlib.Path
    # First and last day this timetable was in force, as published.
    valid_from: str
    valid_to: str


class LoadingConfig(BaseRunConfig):
    """Analyse loading counts and build the period matrices."""

    table_1410: pathlib.Path
    # stations.csv (crsCode + tiploc columns); supplies the TIPLOC ->
    # CRS map for operators that publish TIPLOCs (EMR, SWR).
    station_list: pathlib.Path
    source_odm: pathlib.Path
    # ODM releases with both NLC and TLC columns, for mapping an
    # NLC-only source_odm.
    nlc_reference_paths: list[pathlib.Path] = pydantic.Field(default_factory=list)
    distance: TrackCalculatorConfig
    # `timetable` route_frequency_summary.csv: direct trains/day per
    # pair, the service term of the calibrated gravity model (not fitted
    # when unset).
    route_frequency_path: pathlib.Path | None = None
    # Scale ODM cells so each station's marginal matches its share of
    # table 1410 entries and exits (ortla.od_matrix). The two
    # sources disagree station by station (p5-p95 of the ratio
    # 1.00-1.65) and that disagreement predicts corridor bias with a
    # coefficient of 1.0. Only the shares move: the grand total is
    # preserved, so the 1.246x definitional gap between LENNON sales and
    # ORR station counts is not propagated. Off by default: a run
    # without it is unchanged.
    rebalance_to_table_1410: bool = False
    # Rebalance station marginals to table 1410 entries and exits raised
    # to (1 + this), so with -0.097 small stations gain demand against
    # large ones (grand total kept). Counts exceed the ODM at small
    # stations by ~25% per tenfold drop in size, which fits the
    # ticketless and unrecorded travel ORR lists as a limitation (case
    # 29, 762 counted stations, operator fixed effects). Held out on
    # station halves it took the adopted model's rmse(log) from 0.620 to
    # 0.601-0.603 and the level from 0.809 to 0.817. Implies
    # rebalance_to_table_1410. None leaves the matrix unchanged.
    station_size_exponent: float | None = None
    # Network Rail SRAM's hourly OD demand (output O03, parquet or CSV).
    # When set, each weekday's period matrices move every pair's split
    # over periods and directions towards it by od_profile_strength, the
    # profile smoothed by od_profile_smoothing trips towards ours; pair
    # day totals are kept (ortla.od_profile). Case 37:
    # strength 0.5, smoothing 10 took rmse(log) from 0.562 to 0.532
    # (held out by corridor 0.558 to 0.519), with the direction split
    # left on. None leaves the matrices unchanged.
    od_profile_path: pathlib.Path | None = None
    od_profile_strength: float = 0.5
    od_profile_smoothing: float = 10.0
    # Optional `timetable` calls table, used to extend the TIPLOC -> CRS
    # map beyond stations.csv's one TIPLOC per station.
    tiploc_crs_calls_path: pathlib.Path | None = None
    # Fill a call's scheduled departure from that same calls table where
    # the operator publishes none (Northern) or publishes the arrival
    # instead (Avanti, GTR), so time periods come from the boarding stop
    # rather than the train's origin. No effect without
    # tiploc_crs_calls_path.
    impute_departure_times: bool = True
    # One cleaned calls table per published timetable, so a count is
    # matched against the schedule that was running when it was taken
    # rather than against a single extract. Counts outside every window
    # fall back to tiploc_crs_calls_path. Empty leaves the original
    # single-extract behaviour untouched.
    timetable_eras: list[TimetableEra] = pydantic.Field(default_factory=list)
    # Minutes of slack on the origin departure for the loosest match
    # rung.
    departure_match_tolerance_minutes: int = 5
    # Operator key -> loading data file or folder. Keys are the readers
    # in ortla.loading_data.READERS: northern, tpe, awc, emr,
    # gwr, gtr, swr, xc, southeastern, greater_anglia.
    loading_paths: dict[str, pathlib.Path] = pydantic.Field(default_factory=dict)
    # Only use count periods starting within this window ("YYYY-MM-DD"),
    # e.g. to build a pre-launch baseline. Undated rows are dropped when
    # either is set.
    date_from: str | None = None
    date_to: str | None = None
    # Write period matrices only for these operator keys (all when
    # unset). A national set of matrices is several GB per operator.
    matrix_operators: list[str] | None = None
    # Pickle each operator's standardised data so re-runs skip the slow
    # XLSX parsing. cache_dir (default: output_path/_cache) can be
    # shared between runs with different date filters or ODMs; the
    # classified ODM is cached there too.
    cache_standardised: bool = True
    cache_dir: pathlib.Path | None = None
    airport_stations: list[str] = pydantic.Field(default_factory=list)
    time_periods_cutoffs: dict[str, str] = pydantic.Field(
        default_factory=lambda: {
            "am_peak": "7:00",
            "interpeak": "9:30",
            "pm_peak": "16:00",
            "evening": "19:00",
        }
    )
    route_type: RouteTypeConfig = pydantic.Field(default_factory=RouteTypeConfig)
    analysis: AnalysisConfig = pydantic.Field(default_factory=AnalysisConfig)
    demand_column: str = "Mean Passengers On"
    journey_type_column: str = "route_type_pax_off"
    # Bucket by clock hour (24 one-hour periods) instead of the named
    # periods. When true, overrides ``time_periods_cutoffs`` and the
    # analysis time order.
    hourly: bool = False
    # Equal buckets of this many minutes instead of the named periods.
    # Takes precedence over ``hourly`` (which is
    # ``period_minutes: 60``).
    period_minutes: int | None = None
    # What the temporal factors are grouped by. "distance" uses
    # ``journey_type_column``'s distance band; "origin_station" gives
    # each origin with at least ``journey_type_min_rows`` rows its own
    # profile and falls back to the distance band elsewhere. Case 06
    # measured the three distance bands removing 1% of the variation
    # between corridors' real time-of-day profiles (98.7% of it is
    # within band), against 30% for the origin station, so distance
    # explains almost none of it.
    journey_type_mode: str = "distance"
    journey_type_min_rows: int = 200

    @property
    def resolved_cache_dir(self) -> pathlib.Path:
        """Where standardised loadings and the typed ODM are cached."""
        return self.cache_dir or self.output_path / "_cache"

    @pydantic.model_validator(mode="after")
    def _resolve_hourly(self) -> LoadingConfig:
        minutes = self.period_minutes or (60 if self.hourly else None)
        if minutes:
            self.time_periods_cutoffs = interval_time_cutoffs(minutes)
            self.analysis.time_order = interval_period_order(minutes)
        return self

    def run(self) -> None:
        """Run the loading analysis and build the period matrices."""
        period_matrices.run_loading(self)


class LookupConfig(BaseRunConfig, StationPairMixin):
    """Look up expected trips in the period matrices."""

    period_matrices_dir: pathlib.Path
    day_order: list[str] = pydantic.Field(
        default_factory=lambda: [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
    )
    period_order: list[str] = pydantic.Field(
        default_factory=lambda: [
            "early",
            "am_peak",
            "interpeak",
            "pm_peak",
            "evening",
        ]
    )
    fill_value: str = "!"
    per_day: bool = False
    journey_type: bool = False
    route_type: RouteTypeConfig = pydantic.Field(default_factory=RouteTypeConfig)
    distance: TrackCalculatorConfig | None = None
    # Set true when the period matrices were built with the loading
    # command's ``hourly`` mode, so the day x hour grid is ordered by
    # clock hour.
    hourly: bool = False

    @pydantic.model_validator(mode="after")
    def _resolve_hourly(self) -> LookupConfig:
        if self.hourly:
            self.period_order = hourly_period_order()
        return self

    def run(self) -> None:
        """Look up the station pair and report the day x period grid."""
        trip_lookup.run_lookup(self)


class ServiceLookupConfig(BaseRunConfig, StationPairMixin):
    """List the timetabled trains between station pairs, by time."""

    # Cleaned calls table produced by the ``timetable`` command.
    timetable_calls_path: pathlib.Path
    # Optional time filter on the departure from the origin. Precedence:
    # ``time_range`` (an ``["HH:MM", "HH:MM"]`` pair, start
    # inclusive/end exclusive) beats ``hour`` (a single clock hour,
    # 0-23) beats ``period`` (a named period from
    # ``time_periods_cutoffs``). With none set the whole day is listed.
    period: str | None = None
    hour: int | None = None
    # An ``["HH:MM", "HH:MM"]`` pair (start inclusive, end exclusive).
    time_range: list[str] | None = None
    # Ordered period name -> start time, matching the loading/timetable
    # configs, used to label each train's departure period.
    time_periods_cutoffs: dict[str, str] = pydantic.Field(
        default_factory=lambda: {
            "am_peak": "7:00",
            "interpeak": "9:30",
            "pm_peak": "16:00",
            "evening": "19:00",
        }
    )
    period_order: list[str] = pydantic.Field(
        default_factory=lambda: [
            "early",
            "am_peak",
            "interpeak",
            "pm_peak",
            "evening",
        ]
    )
    # Typical-weekday snapshot controls, matching the ``timetable``
    # command so the listed trains reconcile with the route-frequency
    # counts. reference_weekday is 0=Mon .. 6=Sun (used only when
    # reference_weekdays is unset).
    reference_weekday: int = 2
    # Optional list of weekdays (0=Mon .. 6=Sun) to average over, e.g.
    # Mon-Fri as [0, 1, 2, 3, 4]. When set, each train is weighted by
    # the share of these days it runs, giving a mean typical-weekday
    # frequency. Overrides reference_weekday.
    reference_weekdays: list[int] | None = None
    reference_date: str | None = None
    # Bucket by clock hour (24 one-hour periods) instead of the named
    # periods. When true, overrides ``time_periods_cutoffs`` and
    # ``period_order``.
    hourly: bool = False

    @pydantic.model_validator(mode="after")
    def _resolve_hourly(self) -> ServiceLookupConfig:
        if self.hourly:
            self.time_periods_cutoffs = hourly_time_cutoffs()
            self.period_order = hourly_period_order()
        return self

    def run(self) -> None:
        """List the origin -> destination trains in the time filter."""
        service_lookup.run_service_lookup(self)


class FaresLookupConfig(BaseRunConfig, StationPairMixin):
    """Look up RSP standard fares for one or many station pairs.

    Only the permanent standard fares in the official feed are covered
    (Anytime, Off-Peak, Season and so on). The
    :mod:`ortla.fares` module docstring says what this data
    does and doesn't include; in particular, it has no date-varying
    Advance pricing.
    """

    # An already-extracted feed directory (containing the .FFL/.LOC/.TTY
    # files); set this to skip downloading. Leave unset and set
    # ``download: true`` to fetch a fresh copy from the National Rail
    # Data Portal instead (credentials come from the
    # NRDP_USERNAME/NRDP_PASSWORD environment variables, never from this
    # config).
    raw_feed_path: pathlib.Path | None = None
    download: bool = False
    # NRDP account credentials for `download`. Optional: NRDP_USERNAME/
    # NRDP_PASSWORD environment variables are used as a fallback when
    # either is left unset here. Prefer the environment variables: a
    # password in a config file is plain text.
    username: str | None = None
    password: str | None = None
    # Reuse the cleaned
    # fares_flow.csv/fares_locations.csv/fares_tickets.csv already in
    # output_path (from a previous run) instead of re-parsing the feed.
    # The raw feed is ~100 MB and slow to parse again for a run that
    # only changes the station pairs.
    reuse_existing_clean: bool = False

    def run(self) -> None:
        """Look up fares for the configured station pair(s)."""
        fares.run_fares_lookup(self)


class TimetableConfig(BaseRunConfig):
    """Clean and analyse CIF timetable data for our stations."""

    # Network Rail CIF schedule export as newline-delimited JSON, a
    # fixed-width archive extract, or a folder of Working Timetable
    # books. See ortla.cif_archive and
    # ortla.wtt_workbook.
    cif_path: pathlib.Path
    # TIPLOC reference for a ``cif_path`` that carries no location
    # records of its own. The Working Timetable books name their
    # locations rather than coding them, so they need a CIF extract (any
    # era, ideally a near one) to resolve those names to TIPLOCs and CRS
    # codes.
    tiploc_cif_path: pathlib.Path | None = None
    # Valid-station list (CRS codes); services touching one of these are
    # kept. Unset keeps every service (national).
    station_list: pathlib.Path | None = None
    # Station list with stationName + crsCode columns, used to recover
    # TIPLOCs the CIF publishes with a null crs_code (platform-group
    # variants such as WATRLMN, VICTRIE, CLPHMJM) by matching their
    # tps_description. Without it every call at those locations is
    # dropped, which removes Waterloo, Victoria, London Bridge, Clapham
    # Junction and Vauxhall from a national run. Defaults to
    # station_list; set this explicitly in national mode, where
    # station_list is unset.
    crs_reference: pathlib.Path | None = None
    # Explicit TIPLOC -> CRS entries, applied after the CIF and name
    # recovery and winning over both, for descriptions too abbreviated
    # to match a station name (e.g. "STOURBRIDGE JN (TOWN PLAT)").
    tiploc_crs_overrides: dict[str, str] = pydantic.Field(default_factory=dict)
    # Skip the slow CIF clean and reuse an existing timetable_calls.csv
    # when it is already present, to re-run only the route analysis.
    reuse_existing_calls: bool = False
    # Reference weekday for the "typical weekday" frequency (0=Mon ..
    # 6=Sun). Used to pick a representative date when ``reference_date``
    # is not set.
    reference_weekday: int = 2
    # Representative calendar date (``YYYY-MM-DD``) the frequency
    # snapshot is taken on; only permanent schedules whose date range
    # covers it are counted, so seasonal variants of the same train do
    # not stack. Auto-picked (a ``reference_weekday`` near the middle of
    # the data) when left unset.
    reference_date: str | None = None
    # Clock-hour window (inclusive start, exclusive end) treated as
    # daytime when averaging trains per hour.
    daytime_hours: tuple[int, int] = (7, 19)
    # Drop Merseyrail (ATOC code "ME") services before the route
    # frequency analysis, since its metro-style timetable can dominate
    # route counts.
    exclude_merseyrail: bool = False

    def run(self) -> None:
        """Clean the CIF and count typical-weekday route frequencies."""
        timetable.run_timetable(self)


class BplanGeographyConfig(BaseRunConfig):
    """Clean a BPLAN Geography extract and build the timing graph."""

    # Tab-delimited BPLAN Geography extract (LOC/NWK/TLK/TLD/PLT/REF
    # records).
    geography_path: pathlib.Path

    def run(self) -> None:
        """Clean the extract, build the timing graph, log a summary."""
        bplan_geography.run_bplan(self)


class ServiceStop(ctk.BaseConfig):
    """One calling point of a candidate service, with explicit times."""

    station: str  # CRS code
    arrival: str | None = None  # "HH:MM"; None for the origin
    departure: str | None = None  # "HH:MM"; None for the terminus
    # Only used by the demand model's gravity/catchment case, for a
    # station with no or thin existing loading data (a new or
    # barely-served stop).
    estimated_annual_footfall: int | None = None


class DepartureWindow(ctk.BaseConfig):
    """A window of departure shifts to search for a clear path.

    Drives two independent searches:

    * Every stop time of the candidate moves by each trial amount. A
      uniform shift changes no leg's running time, so running-time
      feasibility is checked once against the candidate as typed, and
      only headway feasibility varies across the window. See
      :func:`ortla.feasibility.search_departure_window`.
    * For a multi-stop candidate a single shift can't always clear every
      leg: LIV -> WGW -> BON might need +5 min on the first leg but +15
      min on the second to avoid existing trains.
      :func:`ortla.feasibility.search_flexible_slot` finds
      each leg's clearest departure on its own and bridges them by
      flexing the intermediate dwells within
      ``min_dwell_minutes``/``max_dwell_minutes``. It runs alongside the
      uniform sweep, and both appear as ranked options in Scenario
      Studio.
    """

    earliest_shift_minutes: float = -30.0  # relative to the candidate's own stop times
    latest_shift_minutes: float = 30.0
    step_minutes: float = 5.0

    # Bounds on how far an intermediate stop's dwell may be stretched or
    # squeezed to bridge two legs' independently found clear slots.
    min_dwell_minutes: float = 1.0
    max_dwell_minutes: float = 15.0


class CandidateService(ctk.BaseConfig):
    """A new or altered service to check for feasibility and demand."""

    name: str
    operator: str | None = None
    # BPLAN traction/timing-load code (see bplan_traction_loads.csv),
    # used to look up minimum sectional running times, e.g. "150",
    # "158", "185".
    traction_code: str
    days_of_week: list[str]
    stops: list[ServiceStop]
    # Assumed seats, used by the loading/VoC estimate.
    capacity: float | None = None
    # If set, `propose` searches whole-candidate time shifts in this
    # window (writing propose_{name}_search_results.csv) instead of only
    # checking the candidate's stop times as typed.
    search_window: DepartureWindow | None = None
    # Set for a service on track that doesn't exist yet in the current
    # BPLAN Geography/timetable data (a new build or reopening, e.g.
    # Skipton-Colne). The running-time and headway checks both assume
    # today's physical network, so for a new alignment they either find
    # no route or route the long way round the existing network and
    # reject a realistic direct running time as "infeasible" against
    # that false minimum. When true, both checks (and the
    # departure-window and flexible-slot searches that rely on them) are
    # skipped for every leg of this candidate; demand and appraisal
    # still run.
    new_infrastructure: bool = False
    # When set, the unit works back: a second candidate,
    # `{name}_return`, leaves the last stop this many minutes after
    # arriving and calls at the same stops in reverse, each leg taking
    # the time it took outbound.
    return_after_minutes: float | None = None

    def return_working(self) -> CandidateService:
        """Build the return working (see ``return_after_minutes``)."""
        # A stop with only one time set is taken to have no dwell, as in
        # the running-time check, so every return leg gets both its
        # times.
        end = hhmm_minutes(self.stops[-1].arrival or self.stops[-1].departure)
        start = end + (self.return_after_minutes or 0.0)
        last = len(self.stops) - 1
        stops = [
            stop.model_copy(
                update={
                    "arrival": None
                    if idx == 0
                    else format_hhmm(start + end - hhmm_minutes(stop.departure or stop.arrival)),
                    "departure": None
                    if idx == last
                    else format_hhmm(start + end - hhmm_minutes(stop.arrival or stop.departure)),
                }
            )
            for idx, stop in enumerate(reversed(self.stops))
        ]
        return self.model_copy(
            update={
                "name": f"{self.name}_return",
                "stops": stops,
                "return_after_minutes": None,
            }
        )


class HistoricPerformanceConfig(ctk.BaseConfig):
    """Optional HSP historic-performance check for `propose` candidates.

    When set on :class:`ProposeConfig`, each candidate leg's departure
    is compared with existing services on the same corridor from the
    Historic Service Performance API (see
    :mod:`ortla.feasibility`), reporting how reliably a
    comparable existing service ran to time. It is context on how
    reliable the corridor is in practice, not a feasibility check like
    the running-time and headway checks: HSP has no data for a new
    service, only for what already runs near the candidate's departure
    time on the corridor.
    """

    from_date: str
    to_date: str
    # Candidate departure +/- this many minutes is queried for existing
    # comparable services (HSP buckets by time window, not exact
    # departure).
    window_minutes: float = 30.0
    tolerance_minutes: str = "5"
    # Below this percentage-within-tolerance, the nearest existing
    # service is flagged as poor_reliability.
    poor_reliability_percent: float = 75.0
    # Below this many matched historic services, the nearest existing
    # service is flagged insufficient_sample instead of trusted.
    min_sample_size: int = 5
    # NRDP account credentials, the same account as `fares`. Optional:
    # NRDP_USERNAME/NRDP_PASSWORD environment variables are used as a
    # fallback when either is left unset here.
    username: str | None = None
    password: str | None = None


class ProposeConfig(BaseRunConfig):
    """Check candidate services fit the timetable; predict demand."""

    # A service with `return_after_minutes` set is followed by its
    # return working, which is checked and appraised as a candidate of
    # its own.
    services: list[CandidateService]
    # Cleaned calls table produced by the `timetable` command.
    timetable_calls_path: pathlib.Path
    # Output directory of a previous `bplan` command run.
    bplan_output_path: pathlib.Path
    # Period matrices directory from the `loading` command (one
    # operator's {operator}_period_matrices folder, or a manually
    # blended equivalent), laid out like
    # LookupConfig.period_matrices_dir. Optional: when unset, demand
    # prediction is skipped and only the feasibility checks run.
    period_matrices_dir: pathlib.Path | None = None
    # Pivot-model demand elasticity: new_total = existing_trips *
    # (new_frequency / existing_frequency) ** frequency_elasticity. 0.4
    # is a typical planning value, not calibrated locally, so check it.
    frequency_elasticity: float = 0.4
    # "calibrated": every leg uses the ODM-fitted gravity model
    # (ortla.gravity_model): observed slot demand scaled by
    # its frequency response, or its synthetic prediction for an
    # unobserved pair. "legacy": the original pivot
    # (frequency_elasticity) with the footfall-share fallback, kept for
    # comparison.
    demand_model: str = "calibrated"
    # Override the gravity model's fitted frequency coefficient `b4` for
    # this run, leaving every other coefficient as calibrated. The
    # fitted value is cross-sectional (busy pairs attract trains as well
    # as the reverse), and the within-pair panel puts the same
    # association an order of magnitude lower
    # (docs/notes/ACCURACY_IMPROVEMENTS.md section 7). This exists so
    # that claim can be tested on a hindcast before anything is changed.
    # None = use the fitted value.
    frequency_coefficient_override: float | None = None
    # Exponent of the pivot on a pair that already has direct trains,
    # (f_after / f_before) ** e, in place of the cross-sectional b4
    # (about 0.62). On held-out ODM spans b4 was the worst choice in
    # every span and 0 to 0.1 the best (case 53, posthoc_grid). 0.1
    # allows for the noise in archive timetables, which biases the
    # within-pair estimate down. The synthetic prediction for an
    # unserved pair keeps b4. None = b4 (runs before 2026-10-26).
    served_pair_frequency_elasticity: float | None = 0.1
    # Timetable CRS -> the CRS the ODM, table 1410 and the fares feed
    # hold the same physical station under (see
    # ortla.stations). Unset takes the built-in HS1/domestic
    # pairs; `{}` disables aliasing.
    station_aliases: dict[str, str] | None = None
    # Timetable CRS -> the CRS of the station it is a second code of
    # (the Elizabeth line's central calls, Reading platform 4, ...).
    # Unlike an alias it is rewritten in the timetable, and in any count
    # feed matched to it, as soon as they are read, so trains at either
    # code count as one station's trains. Unset takes
    # ortla.stations.DEFAULT_STATION_RECODES; `{}` turns it
    # off.
    station_recodes: dict[str, str] | None = None
    # Route each pair no single train serves through its best single
    # change on the day's full timetable, and add its trips to both
    # legs' matrix cells (network_assignment.TransferDemand). Without
    # it a journey needing a change reaches no train. A route slower in
    # the train than `transfer_detour_base_minutes` plus
    # `transfer_detour_minutes_per_km` times its straight-line km is
    # dropped (the km need `distance.stations`), and the change costs
    # `interchange_penalty_minutes`. Used by `hindcast`.
    transfer_assignment: bool = True
    transfer_detour_base_minutes: float = 30.0
    transfer_detour_minutes_per_km: float = 2.0
    # "period": route each period on its own trains, so a pair with
    # direct trains in some periods still changes trains in the others
    # (case 17's T1, adopted again in case 23). "day": the original
    # rule, a change only for pairs no train links all day; set it to
    # reproduce runs before 2026-10-01.
    transfer_routing: Literal["period", "day"] = "period"
    # Multiply the routed one-change trips; `transfer_scale_london`
    # replaces `transfer_scale` on trains that start or end at a London
    # terminal. Case 33 fitted 2.30 and 1.20 together with a
    # slot_share_model of theta 0.06 (in
    # the packaged slot_share_model "m3"), so use the three
    # together. 1.0 and None reproduce runs before 2026-10-25.
    transfer_scale: float = 1.0
    transfer_scale_london: float | None = None
    # Case 88: pairs with a direct train also change trains by choice, a
    # share 1 / (1 + exp(theta x (change minutes - direct minutes)))
    # moving from the direct legs to the best change's two legs (minutes
    # include half the headway and the change penalty). 0.2 took
    # held-out rmse(log) from 0.492 to 0.481 with the transfer scales
    # kept. None reproduces runs before 2026-10-27.
    transfer_choice_theta: float | None = None
    # Where each day's one-change routes are kept and reused by later
    # runs on the same timetable, matrices and settings (a leave-one-out
    # starts one per batch). None = route every run.
    transfer_cache_dir: pathlib.Path | None = None
    # Hold direct legs to the same detour limit: a leg slower than it
    # (the long way round a loop) carries nothing where its slot has a
    # leg within it, and is not counted among the slot's trains. Needs
    # `distance.stations`. Case 19 (addendum L2). Used by `hindcast` and
    # `propose`.
    direct_detour_limit: bool = True
    # Destination sets that compete for the same trips, so a new direct
    # service to one member abstracts passengers from the others instead
    # of only growing its own near-empty O-D cell. Unset takes the
    # built-in London terminals; `{}` disables destination substitution.
    destination_groups: dict[str, list[str]] | None = None
    # Logit scale on generalised journey time (per minute), used for the
    # destination-substitution and direct-vs-interchange route splits.
    # 0.05 is a typical planning value, not calibrated locally, so check
    # it.
    gjt_logit_theta: float = 0.05
    # Minutes of generalised time charged for a change of train when
    # pricing the interchange journey a new direct link replaces.
    interchange_penalty_minutes: float = 20.0
    # Whether demand grows sub-linearly with frequency (the fitted `b4`)
    # while being shared over linearly more trains, so per-train load
    # falls as trains are added. Right for forecasting a new service,
    # wrong for a hindcast re-inserting a train that already ran (see
    # HindcastConfig).
    apply_frequency_dilution: bool = True
    # Floor on a new direct link's demand (`gravity_new_link`): the
    # larger of the share of the pair's observed demand the link wins
    # and this. One of "synthetic" (the served-pair gravity prediction,
    # a smeared mean), "median" (that divided by the smearing factor) or
    # "none". The larger of an observed figure and a noisy estimate
    # selects the estimate's upward errors: on case 20's served pairs
    # the smeared floor won 77% of the time at a median 2.95x the
    # observed cell, where "median" gave 1.01x. "median" since
    # 2026-10-01 (case 20 addendum); it also edged "synthetic" on the
    # clean hindcasts 02 and 03. Set "synthetic" to reproduce earlier
    # runs.
    new_link_floor: Literal["synthetic", "median", "none"] = "median"
    # Log growth of a pair's market when it gains a direct link of at
    # least `new_link_min_trains` a day, scaled by the share the direct
    # route wins and counted as generated. Case 24 (ODM 2018-19 to
    # 2024-25, pairs gaining or losing a direct link, station x year
    # effects): 0.28 from losses (the conservative arm), 0.45 from
    # gains; 1-3 trains a day moved demand 0-8%. 0 reproduces runs
    # before 2026-10-01, where a new link generated nothing.
    new_link_uplift: float = 0.28
    new_link_min_trains: int = 4
    # Gravity/catchment model, used for a leg with no/thin existing
    # demand data (a new or barely-served stop). Both must be set for it
    # to run; legs otherwise falling back to it stay "insufficient_data"
    # without them.
    table_1410: pathlib.Path | None = None
    distance: TrackCalculatorConfig | None = None
    route_type: RouteTypeConfig = pydantic.Field(default_factory=RouteTypeConfig)
    # Route-type bands calibrated from fewer loading rows than this are
    # not trusted for the gravity model (too little data).
    gravity_min_loading_observations: int = 30
    # Fitted directional split (a `direction_split_model.json` written
    # by `ortla.direction_split`). The ODM is exactly
    # symmetric, so without this a pair's two directions get the same
    # slot demand and the only directional signal in the model is the
    # frequency divisor, which points the wrong way: it predicts the
    # higher load per train in the direction with fewer trains. With it
    # set, a pair's combined slot demand is redistributed between its
    # directions in proportion to service provision and the footfall
    # ratio of the two ends. Slots where either direction has no train
    # are left alone. "national" names the packaged fit on every
    # counted operator. None keeps demand symmetric.
    direction_split_model: pathlib.Path | None = None
    # Which legs the split acts on: "observed" (legs with matrix demand,
    # the case it was fitted and verified on), "synthetic" (only legs
    # with no matrix demand, i.e. a new station's gravity legs, which
    # otherwise come out the same size both ways in every period), or
    # "all". Ignored without `direction_split_model`.
    direction_split_legs: Literal["observed", "synthetic", "all"] = "observed"
    # How a slot's demand divides between its trains (a
    # `slot_share_model.json`,
    # ortla.network_assignment.SlotShareModel): by departure
    # half-hour and in-train minutes, so a fast train takes more of a
    # slot than a stopper. Case 23. The packaged models can be named
    # instead of a path: "m3" (cases 23 and 33, theta 0.06) or "neural"
    # (m3 with case 117's network on each train's place in its slot and
    # run; fitted with slot_share_profile on, so set that too). None
    # shares every slot equally over its trains.
    slot_share_model: pathlib.Path | None = None
    # Network Rail SRAM's hourly OD demand (O03, as
    # LoadingConfig.od_profile_path). When set with slot_share_model, a
    # train also takes more of its slot the more of its pair's profile
    # demand starts in the hour it leaves (SlotShareModel.with_profile).
    # Case 39, on top of the od_profile matrices: rmse(log) 0.532 to
    # 0.526, held out by corridor 0.519 to 0.511.
    slot_share_profile: pathlib.Path | None = None
    slot_share_profile_strength: float = 0.75
    slot_share_profile_smoothing: float = 2.0
    # Assumed seats for a candidate without its own `capacity` set, used
    # by the loading/VoC estimate.
    default_capacity: float | None = None
    # Cleaned fares feed directory (a `fares` command output_path,
    # holding fares_flow.csv/fares_locations.csv/fares_tickets.csv).
    # When set, each demand leg is priced at its O-D pair's median adult
    # standard-class fare (ortla.fares.median_fare) times
    # fare_yield_factor to give a predicted annual revenue. Optional:
    # revenue is left blank when unset (or when a pair has no such fare
    # in the feed).
    fares_dir: pathlib.Path | None = None
    # What a journey pays against the median listed fare. The median
    # counts each product once; passengers lean on seasons, railcards
    # and off-peak tickets. Weighted by 2024-25 ODM journeys the median
    # averages GBP 13.34 a journey; ORR's fares income is GBP 11.5bn
    # over the ODM's 1.53bn journeys, GBP 7.50 (case 26). Simplification: one
    # national factor; a long-distance/commuter split if the cases need
    # it.
    fare_yield_factor: float = 0.56
    # Reference weekday and date, as in the `timetable` and `services`
    # commands; they pick the timetable snapshot the headway check runs
    # against. 0=Mon .. 6=Sun.
    reference_weekday: int = 2
    reference_date: str | None = None
    # Ordered period name -> start time; used only to label existing
    # trains' time period in the headway-check output.
    time_periods_cutoffs: dict[str, str] = pydantic.Field(
        default_factory=lambda: {
            "am_peak": "7:00",
            "interpeak": "9:30",
            "pm_peak": "16:00",
            "evening": "19:00",
        }
    )
    # Equal buckets of this many minutes instead of the named periods.
    # Must match the ``period_minutes`` the period matrices were built
    # with, or every leg reads the wrong matrix file.
    period_minutes: int | None = None
    # Safety margin subtracted from the tightest gap already scheduled
    # on a segment, and the hard floor used when too few existing trains
    # exist to derive an empirical minimum. This is an empirical proxy,
    # not signalling-headway data (see propose.py's module docstring).
    headway_safety_margin_minutes: float = 1.0
    min_headway_floor_minutes: float = 2.0
    # "empirical" (tightest existing gap, see above) or "fixed" (the
    # same planning headway, planning_headway_minutes, on every
    # segment).
    headway_mode: str = "empirical"
    planning_headway_minutes: float = 3.0
    # Optional HSP historic-performance check (see
    # HistoricPerformanceConfig), skipped when unset.
    historic_performance: HistoricPerformanceConfig | None = None

    @pydantic.model_validator(mode="after")
    def _resolve_period_minutes(self) -> ProposeConfig:
        if self.period_minutes:
            self.time_periods_cutoffs = interval_time_cutoffs(self.period_minutes)
        return self

    @pydantic.model_validator(mode="after")
    def _add_return_workings(self) -> ProposeConfig:
        names = {service.name for service in self.services}
        self.services = [
            *self.services,
            *(
                service.return_working()
                for service in self.services
                if service.return_after_minutes is not None
                and f"{service.name}_return" not in names
            ),
        ]
        return self

    def run(self) -> None:
        """Check every candidate's running times and headways."""
        propose.run_propose(self)


class CommercialServiceDef(ctk.BaseConfig):
    """One service to appraise, with its demand and fare."""

    name: str
    annual_passengers: int
    avg_fare: float  # £ per passenger


class CommercialCaseConfig(BaseRunConfig):
    """Commercial case: whether a proposed service pays its way.

    Fare revenue is treated as direct income to the operator. It is not
    netted off as a transfer, as WebTAG's socioeconomic appraisal does;
    Scenario Studio uses ``appraisal.py`` for that. Produces NPV, IRR
    and payback period at the operator's own discount rate.
    """

    proposed_services: list[CommercialServiceDef] = pydantic.Field(default_factory=list)
    do_minimum_services: list[CommercialServiceDef] = pydantic.Field(default_factory=list)

    # Appraisal timing
    appraisal_year: int = 2028
    appraisal_period: int = 30
    construction_period: int = 3

    # Demand Ramp-Up
    demand_ramp_up_years: int = 3
    demand_ramp_up_profile: list[float] = pydantic.Field(
        default_factory=lambda: [0.5, 0.75, 0.9, 1.0]
    )
    demand_growth_rate: float = 0.0  # real-terms pa, over and above ramp-up

    # Costs (£ million)
    capital_cost: float = 0.0
    annual_operating_cost: float = 0.0
    cost_inflation_rate: float = 0.0

    # Revenue growth (real-terms pa, over and above demand growth)
    fare_growth_rate: float = 0.0

    # The operator's own cost of capital / required return.
    discount_rate: float = 0.08

    # Optional: import demand from a propose run
    propose_output_path: pathlib.Path | None = None

    def run(self) -> None:
        """Run the commercial case."""
        commercial_case.run_commercial_case_command(self)


class DiscoveryConfig(BaseRunConfig):
    """Scan the network for high-potential new direct-service scenarios.

    Finds station pairs with no or poor existing direct service but high
    estimated latent demand, synthesises a plausible direct-service
    candidate for the most promising ones, and appraises each through
    the same optimiser Scenario Studio uses
    (:func:`ortla.scenario.generate_scenarios`), so it finds
    scenarios worth a closer look without anyone first picking the
    origin-destination pair.

    With ``default_capital_cost_m``/``default_annual_operating_cost_m``
    left at 0.0, the ranking is a demand-driven guide to which pairs are
    worth investigating, not a bankable appraisal. Real costs belong in
    a full Scenario Studio run once a candidate is promoted.
    """

    # Station universe + straight-line pre-filter.
    station_list: pathlib.Path
    distance: TrackCalculatorConfig
    min_distance_miles: float = 15.0
    max_distance_miles: float = 150.0
    # Per-origin cap on the straight-line pre-filter, so the candidate
    # count stays bounded (~n_stations * this) however dense that part
    # of the network is; a distance band alone doesn't guarantee that.
    max_candidates_per_origin: int = 40

    # Existing-service filter, from a `timetable` run's whole-network
    # summary.
    timetable_output_path: pathlib.Path
    max_existing_daily_trains: float = 1.0
    # Judge a pair on the busier of its two directions. A direct link is
    # a property of a corridor, not a direction, and real timetables are
    # not symmetric: Chichester -> London Victoria has one through
    # working on a typical Wednesday against 32 the other way. Set false
    # to test each ordered pair on its own, which will surface such
    # corridors as gaps.
    existing_service_both_directions: bool = True

    # Demand scoring: each corridor's observed market, from the period
    # matrices (see `discovery._score_observed_market`). `table_1410`
    # and the gravity settings serve the full sweep's legs with no
    # observed flow.
    table_1410: pathlib.Path
    period_matrices_dir: pathlib.Path
    route_type: RouteTypeConfig = pydantic.Field(default_factory=RouteTypeConfig)
    gravity_min_loading_observations: int = 30
    demand_ranked_shortlist_size: int = 50

    # Running-time feasibility, on the physical BPLAN network.
    timetable_calls_path: pathlib.Path
    bplan_output_path: pathlib.Path
    # How many running-time-feasible pairs get a synthesized candidate
    # and a full Scenario Studio sweep, the largest cost of a run.
    promote_to_sweep_n: int = 20
    # True: promote that many from each route-type band (commute,
    # regional, long distance) instead, so short branch corridors do not
    # fill the sweep.
    promote_per_route_type: bool = False

    # Candidate synthesis. There is no corridor-based traction lookup,
    # so one default is used. `minimum_running_time` falls back to the
    # fastest traction recorded on a link when the requested code has no
    # entry, so an imprecise default costs some accuracy instead of
    # giving wrong numbers.
    default_traction_code: str = "150"
    default_days_of_week: list[str] = pydantic.Field(
        default_factory=lambda: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    )
    anchor_departure_time: str = "08:00"
    running_time_margin_minutes: float = 5.0
    # Call at up to this many stations on the route, the busiest by
    # table 1410 entries and exits at or above the floor; 0 keeps
    # candidates non-stop.
    max_intermediate_stops: int = 0
    intermediate_stop_min_footfall: float = 1_000_000.0
    dwell_minutes: float = 1.0
    # As ProposeConfig. "empirical" rejects any train that raises a
    # segment's frequency, which every synthesized candidate with
    # intermediate stops does; "fixed" applies planning_headway_minutes
    # everywhere.
    headway_mode: str = "empirical"
    planning_headway_minutes: float = 3.0
    # Always applied to a synthesized candidate: a single anchor time is
    # a guess, and sweeping a window gives the optimiser a better chance
    # of a good slot.
    search_window: DepartureWindow = pydantic.Field(default_factory=DepartureWindow)

    # Full sweep (reuses `generate_scenarios` unchanged).
    fares_dir: pathlib.Path | None = None
    tag_databook_path: pathlib.Path
    frequency_options: list[int] = pydantic.Field(default_factory=lambda: [1, 2])
    default_capital_cost_m: float = 0.0
    default_annual_operating_cost_m: float = 0.0
    # As ScenarioAssumptionsConfig: per train-km over each candidate's
    # track distance, and the service day the wait-time benefit is
    # spread over.
    operating_cost_per_train_km: float = 0.0
    operating_hours_per_day: float = 0.0
    # Appraise each candidate's return working too (see
    # CandidateService.return_after_minutes) and rank on the round trip.
    return_after_minutes: float | None = None

    def run(self) -> None:
        """Run the discovery scan and write the ranked candidates."""
        discovery.run_discovery_command(self)


class ScenarioAssumptionsConfig(ctk.BaseConfig):
    """Appraisal levers for the ``scenario`` command.

    Mirrors :class:`ortla.scenario.ScenarioAssumptions` (same
    names and defaults); see there for what each lever does.
    """

    appraisal_year: int = 2028
    appraisal_period: int = 60
    construction_period: int = 3
    demand_ramp_up_years: int = 3
    demand_ramp_up_profile: list[float] = pydantic.Field(
        default_factory=lambda: [0.5, 0.75, 0.9, 1.0]
    )
    purpose_work: float = 0.12
    purpose_commuting: float = 0.47
    purpose_other: float = 0.41
    capital_cost_m: float = 0.0
    annual_operating_cost_m: float = 0.0
    operating_cost_per_train_km: float = 0.0
    cost_inflation_rate: float = 0.0
    optimism_bias_capex: float = 0.66
    optimism_bias_opex: float = 0.41
    indirect_tax_correction_factor: float = 1.19
    mode_shift_car_pct: float = 0.20
    avg_car_trip_km: float = 30.0
    car_co2_per_km: float = 0.120
    indirect_tax_car_pence_per_km: float = 3.0
    operating_hours_per_day: float = 0.0
    wait_time_vot_multiplier: float = 2.0
    max_average_wait_minutes: float = 15.0
    default_seats_per_train: float = 0.0
    frequency_options: list[int] = pydantic.Field(default_factory=lambda: [1])
    default_time_saving_minutes: float = 0.0
    default_avg_fare: float = 0.0
    commercial_discount_rate: float = 0.08
    commercial_fare_growth_rate: float = 0.0


class ScenarioConfig(ProposeConfig):
    """Scenario Studio from the command line.

    Each entry in ``services`` is optimised independently: swept across
    its ``search_window`` and ``assumptions.frequency_options`` for
    feasibility and demand, every feasible variant appraised (WebTAG +
    commercial case), and the ranked options written out.
    """

    tag_databook_path: pathlib.Path
    assumptions: ScenarioAssumptionsConfig = pydantic.Field(
        default_factory=ScenarioAssumptionsConfig
    )

    def run(self) -> None:
        """Optimise and appraise every candidate service."""
        scenario.run_scenario_command(self)


class StripRule(ctk.BaseConfig):
    """Which timetable schedules make up the already-launched service.

    ``train_uids``, when given, is used on its own. Otherwise a schedule
    is stripped when it matches every set criterion: operator, a
    schedule start on/after ``start_date_from`` (new schedules), public
    calls at all of ``calls_at`` (in that order, or reversed with
    ``both_directions``), and ``origin``/``destination`` as its ends.

    ``remove_calls_at`` switches the strip from whole schedules to calls
    only: the matched schedules keep running in the "before" network,
    minus their stops at those stations. Use it when the service was not
    a new train but a change to an existing one (a new station inserted
    into today's services, or an existing train extended over new
    ground), where removing the whole schedule would also delete service
    that was there before.
    """

    train_uids: list[str] = pydantic.Field(default_factory=list)
    atoc_code: str | None = None
    calls_at: list[str] = pydantic.Field(default_factory=list)
    origin: str | None = None
    destination: str | None = None
    both_directions: bool = True
    in_order: bool = True
    start_date_from: str | None = None
    remove_calls_at: list[str] = pydantic.Field(default_factory=list)


class OutturnConfig(ctk.BaseConfig):
    """Observed loading data for the stripped trains."""

    # Operator key from ortla.loading_data.READERS.
    operator: str
    path: pathlib.Path
    date_from: str | None = None
    date_to: str | None = None
    # Drop train-days whose counts restart mid-journey: at least
    # ``count_reset_min_load`` passengers on board arriving at an
    # intermediate call, under a fifth of them still aboard leaving it,
    # and nobody counted boarding. Nobody boarding at Gatwick or
    # Cambridge over weeks of counts is a counter restarting, not
    # passengers, and it zeroes the load onward.
    screen_count_resets: bool = True
    count_reset_min_load: float = 10.0
    # Where the parsed feed is pickled and reused (the `loading`
    # command's `cache_dir` works, and shares its cache). GTR's feed is
    # 30 workbooks that take ~45 minutes to parse, and without this
    # every hindcast parses them again. None = parse every time.
    cache_dir: pathlib.Path | None = None


class OdmOutturnConfig(ctk.BaseConfig):
    """Before/after annual ODMs for an O-D level hindcast comparison."""

    before_path: pathlib.Path
    after_path: pathlib.Path
    # ODMs with both NLC and TLC columns, to map NLC-only releases.
    nlc_reference_paths: list[pathlib.Path] = pydantic.Field(default_factory=list)
    # Compare only pairs with an end at one of these stations (default:
    # every pair the service links).
    pairs_touching: list[str] = pydantic.Field(default_factory=list)


class HindcastConfig(ProposeConfig):
    """Strip a launched service, predict it, compare with outturn."""

    case_name: str
    strip: StripRule
    # A hindcast re-inserts a train that really ran and compares against
    # that train's own observed load, so the frequency dilution
    # `propose` applies when forecasting a new service would be charged
    # against a service that was never diluted. Off by default here; set
    # True to reproduce `propose`'s forecasting treatment.
    apply_frequency_dilution: bool = False
    # Whether the period matrices were observed with the stripped trains
    # running (an in-sample matrix, as in cases 06, 16 and 20) rather
    # than before them (cases 01-05). When true, an observed cell is
    # already the demand after the change, so each train takes an equal
    # share of it, as in the national assignment. Left false on an
    # in-sample matrix, the strip removes the train but not its
    # passengers: the re-inserted train takes cell / (n - 1) instead of
    # cell / n, and a train that was its slot's only train is priced as
    # a new link (case 20: +0.20 log pooled).
    matrix_includes_stripped: bool = False
    # Train-level outturn from loading data, and/or O-D outturn from
    # ODMs.
    outturn: OutturnConfig | None = None
    outturn_odm: OdmOutturnConfig | None = None
    # Ex-ante footfall for called stations that post-date
    # ``table_1410``: a CSV with ``crs`` and ``forecast_entries_exits``
    # columns, as written by ``cases/_sdft/sdft_footfall.py``. If the
    # file has a ``footfall_year`` column, only the rows for
    # ``forecast_footfall_year`` are read. A forecast is only ever given
    # to a station absent from table 1410, never over an observed
    # figure. It must come from a model of the station's catchment and
    # service (SDFT's trip-end model), never from the station's own
    # later outturn: that would make the hindcast circular, so there is
    # no option to enter a station's footfall by hand.
    forecast_footfall_path: pathlib.Path | None = None
    forecast_footfall_year: str | None = None
    # Hold each station that took a forecast footfall to it. The
    # calibrated gravity model predicts every directly served pair on
    # its own, so a new station's total follows how many big stations
    # its trains reach rather than its catchment: 2.7x its whole
    # footfall at Brent Cross West, 0.1-0.5x on the Northumberland Line.
    # When set, the synthetic demand on every pair the station has a
    # direct train to, on any operator's trains and not only the
    # stripped ones, is scaled so its weekday total is the forecast x
    # this share (the share of annual entries and exits made on
    # weekdays), pro rata to the modelled weekdays in ``days``. Balanced
    # jointly when a pair links two such stations. Calibrated model
    # only. None = unbalanced.
    forecast_footfall_weekday_share: float | None = None
    # BPLAN traction code for the running-time check (CIF timing loads
    # such as "802" don't name a BPLAN code on their own).
    traction_code: str
    days: list[str] = pydantic.Field(
        default_factory=lambda: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    )
    # Demand models to compare (see ProposeConfig.demand_model).
    demand_models: list[str] = pydantic.Field(default_factory=lambda: ["calibrated", "legacy"])
    # Candidates are rebuilt from the timetable, not listed by hand.
    services: list[CandidateService] = pydantic.Field(default_factory=list)

    def run(self) -> None:
        """Run the hindcast."""
        hindcast.run_hindcast(self)


class ProposedStation(ctk.BaseConfig):
    """A station that is not in the forecast timetable yet."""

    # A code of your own for the outputs; it must not clash with a real
    # CRS code.
    code: str
    latitude: float
    longitude: float
    # Weekday trains calling, both directions together.
    trains_per_day: float
    car_spaces: int = 0
    cctv: bool = True
    ticket_machine: bool = True
    bus_interchange: bool = False
    travelcard_boundary: bool = False
    # An unstaffed station counts as SDFT's category F.
    staffed: bool = False
    # Whether most of its trains are electric, and whether most start or
    # end there.
    electric: bool = False
    terminus: bool = False


class NewStationConfig(BaseRunConfig):
    """Forecast new stations' entries and exits with SDFT's model."""

    # Stations already in the forecast timetable and the covariates file
    # to forecast as if new: their catchments are drawn with them open.
    stations: list[str] = pydantic.Field(default_factory=list)
    # Stations not in the timetable yet, located and described by hand.
    proposed: list[ProposedStation] = pydantic.Field(default_factory=list)
    # Census 2021 residents by postcode (NOMIS pcd_p001).
    postcode_population: pathlib.Path
    # OS Code-Point Open's CSV folder (Data/CSV), for postcode
    # centroids.
    codepoint_dir: pathlib.Path
    # Census 2021 workplace population (WP001) by output area.
    workplace_population: pathlib.Path
    # Output area population-weighted centroids on the British grid
    # (columns oa21cd, e, n).
    output_area_centroids: pathlib.Path
    # NRE Knowledgebase station covariates: position, car spaces, CCTV,
    # ticket machine, bus interchange, Travelcard boundary, staffing.
    station_covariates: pathlib.Path
    # `timetable` calls tables and their reference dates (a Wednesday):
    # one for the footfall year the constant is fitted to, one for the
    # timetable the new stations will open into.
    calibration_timetable: pathlib.Path
    calibration_reference_date: str
    forecast_timetable: pathlib.Path
    forecast_reference_date: str
    # Table 1410 for the footfall year the constant is fitted to, and
    # the year before it (stations not open all through that year are
    # left out of the fit).
    calibration_footfall: pathlib.Path
    previous_footfall: pathlib.Path | None = None
    # Table 1410 for 2011-12, SDFT's calibration year: only stations
    # below small_station_threshold in it are used for the fit.
    domain_footfall: pathlib.Path
    small_station_threshold: float = 1_500_000
    # Borrow nearby established stations' residuals, weighted exp(-km / borrow_km) and shrunk
    # by borrow_kappa (cases 107, 160). 0 gives SDFT's forecast alone. The intervals come
    # from the calibration stations' left-out residuals either way.
    borrow_km: float = 2.0
    borrow_kappa: float = 0.25
    # Existing stations within this distance of a new one are listed
    # with their catchments without and with it.
    neighbour_km: float = 15.0
    # Where to keep the postcode and timetable passes between runs.
    cache_dir: pathlib.Path | None = None

    def run(self) -> None:
        """Forecast the new stations."""
        new_station.run_new_station(self)


class AssignOperatorConfig(ctk.BaseConfig):
    """One operator's trains: its matrices and optional counts."""

    # The timetable's ATOC codes for the operator's trains (GTR: TL, SN,
    # GN, GX).
    atoc_codes: list[str]
    # The operator's period matrices from the `loading` command.
    period_matrices_dir: pathlib.Path
    # The operator's loading data, to score the loads against (file or
    # folder, as `loading`'s loading_paths). The operator key must be a
    # reader in ortla.loading_data.READERS.
    counts: pathlib.Path | None = None
    # Only counts from count periods starting in this window
    # ("YYYY-MM-DD").
    counts_date_from: str | None = None
    counts_date_to: str | None = None
    # Where the parsed counts are kept between runs (`loading`'s
    # cache_dir works and shares its cache).
    counts_cache_dir: pathlib.Path | None = None


class AssignConfig(BaseRunConfig):
    """Load every train in the timetable from the period matrices."""

    # Cleaned calls table from the `timetable` command.
    timetable_calls_path: pathlib.Path
    # A date the timetable is in force, for which schedules run; each
    # weekday in `days` takes its week.
    reference_date: str
    days: list[Literal["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]] = pydantic.Field(
        default_factory=lambda: ["Wednesday"]
    )
    # Operator key (a loading data reader: northern, gwr, ...) -> its
    # ATOC codes, matrices and counts.
    operators: dict[str, AssignOperatorConfig] = pydantic.Field(default_factory=dict)
    # Matrices for the trains of every operator not listed above (any
    # operator's matrices will do: they spread the same ODM over the
    # day). Unset leaves those trains out.
    other_period_matrices_dir: pathlib.Path | None = None
    # Station coordinates, for the detour limit, the transfer routes'
    # distances and the neural slot share's.
    stations: StationsConfig | None = None
    # Table 1410 entries and exits, for the direction split's footfall
    # ratio.
    table_1410: pathlib.Path | None = None
    # A fitted direction split (`direction_split_model.json`), or
    # "national" for the packaged one; None keeps each pair's demand the
    # same both ways.
    direction_split_model: pathlib.Path | None = None
    # How a slot's demand divides between its trains: a model file, or
    # a packaged model by name, "m3" or "neural" (see
    # ProposeConfig.slot_share_model). None shares slots equally.
    slot_share_model: pathlib.Path | None = None
    # SRAM's hourly OD profile for the slot share's hourly term (see
    # ProposeConfig.slot_share_profile).
    slot_share_profile: pathlib.Path | None = None
    slot_share_profile_strength: float = 0.75
    slot_share_profile_smoothing: float = 2.0
    # Route pairs with no direct train in a period through their best
    # single change, as ProposeConfig.transfer_assignment with
    # transfer_routing "period".
    transfer_assignment: bool = True
    transfer_scale: float = 1.0
    transfer_scale_london: float | None = None
    transfer_choice_theta: float | None = None
    transfer_detour_base_minutes: float = 30.0
    transfer_detour_minutes_per_km: float = 2.0
    interchange_penalty_minutes: float = 20.0
    # Hold direct legs to the detour limit (needs `stations`).
    direct_detour_limit: bool = True
    # CIF joins and divides (columns main_uid, portion_uid, category VV
    # or JJ, crs): each portion runs through its joint section
    # (`network_assignment.merge_portions`). None treats portions as
    # separate trains.
    associations_path: pathlib.Path | None = None
    # How the counts record a joint section: "unit" (each portion under
    # its own train, so a portion runs as a full train of its own) or
    # "joined" (the main train's count is the whole train: portions'
    # joint loads fold into it).
    portion_counts: Literal["unit", "joined"] = "joined"
    # Stations open for part of the ODM year -> the share it was open
    # (e.g. 0.29 for 15 December). Their matrix cells and footfall are
    # scaled up to a full year.
    odm_open_share: dict[str, float] | None = None
    # As ProposeConfig.
    station_aliases: dict[str, str] | None = None
    station_recodes: dict[str, str] | None = None
    time_periods_cutoffs: dict[str, str] = pydantic.Field(
        default_factory=lambda: {
            "am_peak": "7:00",
            "interpeak": "9:30",
            "pm_peak": "16:00",
            "evening": "19:00",
        }
    )

    @property
    def reference_timestamp(self) -> pd.Timestamp:
        """``reference_date`` as a timestamp."""
        return pd.Timestamp(self.reference_date)

    def operator_groups(self) -> list[tuple[str, list[str], pathlib.Path]]:
        """Return (name, ATOC codes, matrices) for each group."""
        groups = [(k, v.atoc_codes, v.period_matrices_dir) for k, v in self.operators.items()]
        if self.other_period_matrices_dir is not None:
            groups.append((assign.OTHER, [], self.other_period_matrices_dir))
        return groups

    def run(self) -> None:
        """Load every train and score it."""
        assign.run_assign(self)


# # # COMMAND REGISTRY # # #
# Every command the tool offers: its name, the config model that drives
# it and a one-line description. The command line and the desktop app
# are both built from this, so a new command needs a line here and its
# config class above.
COMMANDS: dict[str, tuple[type[BaseRunConfig], str]] = {
    "distance": (DistanceConfig, "calculate track distance along a sequence of stations"),
    "matrix": (MatrixConfig, "build the annual origin-destination matrix"),
    "loading": (LoadingConfig, "analyse loading data and build temporal period matrices"),
    "lookup": (LookupConfig, "look up expected trips for one station pair"),
    "timetable": (
        TimetableConfig,
        "clean and analyse CIF timetable data (national unless station_list is set)",
    ),
    "services": (ServiceLookupConfig, "list the timetabled trains from station A to station B"),
    "bplan": (
        BplanGeographyConfig,
        "clean a BPLAN Geography extract and build the minimum running-time graph",
    ),
    "propose": (ProposeConfig, "check candidate new/altered services for timetable feasibility"),
    "fares": (FaresLookupConfig, "look up official RSP standard fares for one or many pairs"),
    "commercial": (
        CommercialCaseConfig,
        "commercial case: whether proposed services pay their way (NPV, IRR, payback)",
    ),
    "hindcast": (
        HindcastConfig,
        "strip an already-launched service, predict it and compare with outturn",
    ),
    "scenario": (
        ScenarioConfig,
        "Scenario Studio: sweep, appraise and rank variants of candidate services",
    ),
    "discover": (
        DiscoveryConfig,
        "scan the network for high-potential new direct-service scenarios",
    ),
    "assign": (AssignConfig, "load every train in the timetable: national train loads"),
    "new-station": (
        NewStationConfig,
        "forecast a new station's annual entries and exits with SDFT's model",
    ),
}
