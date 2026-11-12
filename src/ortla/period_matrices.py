"""The ``loading`` command: temporal factors and period matrices.

Reads every configured operator's loading data (see
:mod:`ortla.loading_data`), labels each counted row with its
time period and journey type, and turns the counts into day x period
demand factors. Applied to the annual ODM, the factors give one matrix
per (day of week, time period), which is what the rest of the model
prices trains against.
"""

from __future__ import annotations

import logging
import pathlib
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from tqdm import tqdm

from ortla import gravity_model, loading_data, od_profile
from ortla.clock import classify_times
from ortla.od_matrix import read_odm, rebalance_to_station_totals, station_targets
from ortla.stations import load_table_1410, tiploc_to_crs_map
from ortla.track_distance import TrackDistanceCalculator

if TYPE_CHECKING:
    from ortla.config import LoadingConfig, RouteTypeConfig

logger = logging.getLogger(__name__)


def load_operators(config: LoadingConfig) -> pd.DataFrame:
    """Read every operator in ``config.loading_paths``.

    See :mod:`ortla.loading_data` for the per-operator
    readers. Rows are optionally restricted to count periods starting
    within ``config.date_from``/``config.date_to``, which is how a
    "before" baseline is built from pre-launch data only.
    """
    tiploc_to_crs = tiploc_to_crs_map(config.station_list, config.tiploc_crs_calls_path)

    cache_dir = config.resolved_cache_dir / "standardised" if config.cache_standardised else None
    chunks = []
    for operator, path in config.loading_paths.items():
        df = loading_data.load_operator(operator, path, tiploc_to_crs, cache_dir=cache_dir)
        df = _filter_dates(df, operator, config.date_from, config.date_to)
        if not df.empty:
            chunks.append(df)

    if not chunks:
        logger.warning("No operator data files found.")
        return pd.DataFrame(columns=loading_data.CANONICAL_COLUMNS)
    return pd.concat(chunks, ignore_index=True)


def _filter_dates(
    df: pd.DataFrame, operator: str, date_from: str | None, date_to: str | None
) -> pd.DataFrame:
    """Keep rows whose count period starts in the date range."""
    if df.empty or (date_from is None and date_to is None):
        return df
    start = pd.to_datetime(df["Period Start Date"], errors="coerce")
    keep = start.notna()
    if date_from is not None:
        keep &= start >= pd.Timestamp(date_from)
    if date_to is not None:
        keep &= start <= pd.Timestamp(date_to)
    undated = int(start.isna().sum())
    if undated:
        logger.warning(
            "%s: %d row(s) have no count-period date and were excluded by the date filter.",
            operator,
            undated,
        )
    logger.info("%s: %d of %d rows within the date filter.", operator, int(keep.sum()), len(df))
    if not keep.any():
        # An empty window is a config error, not an empty operator:
        # carrying on writes a baseline with no matrices, and every
        # hindcast built on it predicts zero while still reporting
        # success.
        raise ValueError(
            f"{operator}: none of its {len(df)} rows fall within the date filter "
            f"[{date_from}, {date_to}]; the feed's count periods start between "
            f"{start.min():%Y-%m-%d} and {start.max():%Y-%m-%d}."
        )
    return df[keep]


def classify_route_type(distance_miles: float | None, route_cfg: RouteTypeConfig) -> str:
    """Route-type label for a distance in miles (``"Unknown"`` when None)."""
    if distance_miles is None:
        return "Unknown"

    for name, max_miles in route_cfg.bands.items():
        if distance_miles <= max_miles:
            return name

    return route_cfg.fallback


def _departure_for_classification(data: pd.DataFrame) -> pd.Series:
    """Return the time each row's period is read from.

    Prefers the resolved departure (published, else imputed from the
    timetable; see :mod:`ortla.loading_data`), then the
    operator's own published time, and only then the train's origin
    departure, which times a row a median half hour early.
    """
    if loading_data.RESOLVED_COLUMN in data.columns:
        return data[loading_data.RESOLVED_COLUMN].astype("string")
    return (
        data["Departure Time"]
        .astype("string")
        .fillna(data["Origin Departure Time"].astype("string"))
    )


def _classify_distances(miles: pd.Series, route_cfg: RouteTypeConfig) -> pd.Series:
    """Vectorised :func:`classify_route_type`."""
    out = pd.Series(route_cfg.fallback, index=miles.index, dtype=object)
    assigned = pd.Series(False, index=miles.index)
    for name, max_miles in route_cfg.bands.items():
        hit = ~assigned & (miles <= max_miles)
        out[hit] = name
        assigned |= hit
    out[miles.isna()] = "Unknown"
    return out


def _distance_between(
    stations: pd.Series, targets: pd.Series, calc: TrackDistanceCalculator
) -> pd.Series:
    """Track miles from each row's station to its target station.

    One shortest-path tree per distinct target (the track graph is
    undirected). A station's distance to itself, or to a missing
    station, is ``NaN``.
    """
    trees: dict[str, dict[str, float]] = {}
    for target in targets.dropna().unique():
        try:
            trees[target] = calc.distances_from(target, unit="miles")
        except ValueError:
            trees[target] = {}
    values = [
        trees.get(target, {}).get(station, np.nan)
        if isinstance(target, str) and station != target
        else np.nan
        for station, target in zip(stations, targets, strict=True)
    ]
    return pd.Series(values, index=stations.index, dtype=float)


def _compute_analysis_columns(
    data: pd.DataFrame,
    service_key: list[str],
    station_entries: dict[str, int],
    calc: TrackDistanceCalculator,
    config: LoadingConfig,
    airport_stations: set[str],
) -> pd.DataFrame:
    """Add the per-row analysis columns for every service.

    For each service (``service_key``): the stop with the most
    table-1410 footfall and the stop with the most alightings are its
    "biggest stations"; each row is then classified by its track
    distance to those, by time period (from the boarding station's
    departure, else the service's origin departure) and by airport
    status.
    """
    crs = data["Departure from CRS"].astype("string").str.strip().str.upper()
    service = data.groupby(
        [data[c].astype("string").fillna("") for c in service_key], sort=False
    ).ngroup()

    footfall = crs.map(station_entries).astype(float).fillna(0.0)
    missing = set(crs.dropna().unique()) - set(station_entries)
    if missing:
        logger.warning(
            "%d stations have no table-1410 entry (treated as 0 footfall, so "
            "they can never be biggest_station_1410; likely newly opened): %s",
            len(missing),
            ", ".join(sorted(missing)),
        )
    best_1410 = crs.loc[footfall.groupby(service).idxmax()]
    best_1410.index = footfall.groupby(service).idxmax().index
    biggest_1410 = service.map(best_1410)

    off = pd.to_numeric(data["Mean Passengers Off"], errors="coerce")
    per_stop = off.groupby([service, crs]).max().dropna()
    if per_stop.empty:
        biggest_pax = pd.Series(pd.NA, index=data.index, dtype="string")
    else:
        top = per_stop.groupby(level=0).idxmax()
        biggest_pax = service.map(pd.Series([s for _, s in top], index=top.index))

    dist_1410 = _distance_between(crs, biggest_1410, calc)
    dist_pax = _distance_between(crs, biggest_pax, calc)

    origin = data["Origin CRS"].astype("string").str.strip().str.upper()
    dest = data["Destination CRS"].astype("string").str.strip().str.upper()
    return pd.DataFrame(
        {
            "biggest_station_1410": biggest_1410,
            "biggest_station_pax_off": biggest_pax,
            "distance_to_1410_miles": dist_1410,
            "distance_to_pax_off_miles": dist_pax,
            # Classify by when the passenger boarded: the station's own
            # departure time where the feed gives one, else the train's
            # origin departure (Northern publishes no per-station
            # times).
            "time_period": classify_times(
                _departure_for_classification(data),
                config.time_periods_cutoffs,
            ),
            "route_type_1410": _classify_distances(dist_1410, config.route_type),
            "route_type_pax_off": _classify_distances(dist_pax, config.route_type),
            # Origin-station journey types, used when
            # ``journey_type_mode="origin_station"``. Thin origins fall
            # back to their distance band (see compute_temporal_factors
            # and generate_period_matrices), so this column is a
            # candidate label, not the final one.
            "route_type_origin": "O:" + origin,
            "is_airport": origin.isin(airport_stations) | dest.isin(airport_stations),
        },
        index=data.index,
    )


def analyse_loading_data(
    config: LoadingConfig,
    calc: TrackDistanceCalculator | None = None,
) -> pd.DataFrame:
    """Load every configured operator's counts and add the analysis columns.

    For each service (operator, headcode, day, ends, period, departure): the biggest station by
    table 1410 footfall and by passengers off, the track distance, and the period, route type and
    airport flags.
    """
    if calc is None:
        logger.info("Initialising track distance calculator...")
        calc = TrackDistanceCalculator(config.distance)

    station_entries = load_table_1410(config.table_1410)
    data = load_operators(config)
    if data.empty:
        logger.warning("No operator loading data loaded; nothing to analyse.")
        return data
    data = data[data["Departure from CRS"].notna()].reset_index(drop=True)
    if config.impute_departure_times and config.tiploc_crs_calls_path is not None:
        data = loading_data.add_resolved_departures(
            data,
            config.tiploc_crs_calls_path,
            config.departure_match_tolerance_minutes,
            config.timetable_eras,
        )

    service_key = [
        "Train Operator",
        "Headcode",
        "Service Week Day",
        "Origin CRS",
        "Destination CRS",
        "RSP Period",
        # The same headcode can run several times a day; without the
        # departure time those separate trains get lumped into one
        # "service" (and can straddle time periods). Including it makes
        # each group one train run.
        "Origin Departure Time",
    ]
    logger.info("Classifying %d loading rows ...", len(data))
    result_df = _compute_analysis_columns(
        data, service_key, station_entries, calc, config, set(config.airport_stations)
    )
    data = pd.concat([data, result_df], axis=1)

    logger.info("Analysis complete. %d rows processed.", len(data))
    return data


def compute_temporal_factors(
    analysed_data: pd.DataFrame,
    config: LoadingConfig,
    demand_col: str | None = None,
    journey_type_col: str | None = None,
) -> pd.DataFrame:
    """Day x period shares of boardings per journey type, summing to 1 within each type.

    Applying a type's factors to a pair's annual journeys splits them over days and periods.
    ``demand_col`` and ``journey_type_col`` default to the config's.
    """
    demand_col = demand_col or config.demand_column
    journey_type_col = journey_type_col or config.journey_type_column
    df = analysed_data.copy()

    # Under origin_station mode the origin profiles are computed *in
    # addition* to the distance bands, not instead of them: the bands
    # stay estimated from every row so they remain a dense fallback for
    # the ODM pairs no origin profile covers. Overwriting the column
    # instead leaves the bands fitted to the thin-origin residue,
    # covering a third of the day x period cells, and every pair falling
    # back to one lands on an empty cell and is written as zero demand.
    origin_layer = (
        config.journey_type_mode == "origin_station"
        and journey_type_col == config.journey_type_column
        and "route_type_origin" in df.columns
    )

    # Ensure demand column is numeric
    if demand_col in df.columns:
        df.loc[:, demand_col] = pd.to_numeric(
            df[demand_col].astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        )

    # Drop rows that can't be assigned to a segment: unknown/missing
    # journey type or time period would otherwise leak into the factors
    # without mapping to any OD pair. Track each cause separately so the
    # warning shows which side is responsible.
    bad_journey_type = df[journey_type_col].isna() | (df[journey_type_col] == "Unknown")
    bad_time_period = df["time_period"].isna() | (df["time_period"] == "Unknown")
    usable = df[~bad_journey_type & ~bad_time_period]
    dropped = len(df) - len(usable)
    if dropped:
        only_journey_type = int((bad_journey_type & ~bad_time_period).sum())
        only_time_period = int((bad_time_period & ~bad_journey_type).sum())
        both = int((bad_journey_type & bad_time_period).sum())
        logger.warning(
            "Excluded %d of %d rows from factor calculation: %d unknown "
            "journey type only (%r), %d unknown time period only, %d unknown "
            "for both.",
            dropped,
            len(df),
            only_journey_type,
            journey_type_col,
            only_time_period,
            both,
        )
        # When the journey-type column is a route_type_* classification,
        # an "Unknown" usually means the service had no biggest station
        # to measure distance from. Surface how much of the loss is
        # that.
        biggest_col = journey_type_col.replace("route_type", "biggest_station")
        if biggest_col in df.columns:
            no_biggest = int((bad_journey_type & df[biggest_col].isna()).sum())
            if no_biggest:
                logger.warning(
                    "Of the unknown-journey-type rows, %d are services with "
                    "no %s (no station to measure distance from).",
                    no_biggest,
                    biggest_col,
                )

    def _factors_for(frame: pd.DataFrame, label_col: str) -> pd.DataFrame:
        out = (
            frame.groupby([label_col, "Service Week Day", "time_period"])[demand_col]
            .sum()
            .rename("boardings")
            .reset_index()
            .rename(columns={label_col: "journey_type"})
        )
        # Normalise within each journey type so the day x time cells sum
        # to 1.
        out.loc[:, "factor"] = out.groupby("journey_type")["boardings"].transform(
            lambda s: s / s.sum()
        )
        return out

    grouped = _factors_for(usable, journey_type_col)

    if origin_layer:
        counts = usable["route_type_origin"].value_counts()
        keep = counts[counts >= config.journey_type_min_rows].index
        covered = usable[usable["route_type_origin"].isin(keep)]
        logger.info(
            "Origin-station journey types: %d origins of %d have >= %d rows "
            "and get their own profile, alongside the distance bands which "
            "stay fitted to all %d rows as the fallback.",
            len(keep),
            len(counts),
            config.journey_type_min_rows,
            len(usable),
        )
        if not covered.empty:
            grouped = pd.concat(
                [grouped, _factors_for(covered, "route_type_origin")],
                ignore_index=True,
            )

    # Order rows by the config's preferred day/time ordering for
    # readability. Sorted by position rather than cast to a Categorical,
    # which would turn a day or period missing from the order into NaN
    # and drop its matrices; those sort after the listed ones instead.
    orders = {
        "Service Week Day": config.analysis.day_order,
        "time_period": config.analysis.time_order,
    }

    def _position(column: pd.Series) -> pd.Series:
        order = orders.get(column.name)
        if not order:
            return column
        return column.map({label: i for i, label in enumerate(order)}).fillna(len(order))

    return grouped.sort_values(
        ["journey_type", "Service Week Day", "time_period"], key=_position, kind="stable"
    ).reset_index(drop=True)


def classify_od_pairs(
    config: LoadingConfig,
    calc: TrackDistanceCalculator | None = None,
) -> pd.DataFrame:
    """Return the annual OD matrix with each pair's journey type and track distance in miles.

    Distances use one shortest-path tree per origin, not one route per pair.
    """
    logger.info("Loading annual OD matrix from %s ...", config.source_odm)
    odm = read_odm(config.source_odm, config.nlc_reference_paths)

    if calc is None:
        calc = TrackDistanceCalculator(config.distance)

    logger.info("Classifying %d OD pairs by O->D distance...", len(odm))
    journey_type: dict[tuple[str, str], str] = {}
    distance: dict[tuple[str, str], float] = {}
    pairs_by_origin = odm.groupby("origin_tlc")["destination_tlc"].unique()
    for origin_key, dests in tqdm(
        pairs_by_origin.items(),
        total=len(pairs_by_origin),
        desc="Routing origins",
    ):
        origin = str(origin_key)
        try:
            dist_map = calc.distances_from(origin, unit="miles", plausible=True)
        except ValueError:
            dist_map = {}
        for dest_code in dests:
            dest = str(dest_code)
            miles = dist_map.get(dest)
            distance[(origin, dest)] = np.nan if miles is None else miles
            journey_type[(origin, dest)] = classify_route_type(
                miles if origin != dest else 0.0, config.route_type
            )

    pairs = list(zip(odm["origin_tlc"], odm["destination_tlc"], strict=True))
    odm["journey_type"] = [journey_type[pair] for pair in pairs]
    odm["distance_miles"] = [distance[pair] for pair in pairs]
    unknown = odm["journey_type"].eq("Unknown")
    if unknown.any():
        logger.warning(
            "%d OD pairs (%.2f%% of journeys) have no track route and get no period-matrix trips.",
            int(unknown.sum()),
            100 * odm.loc[unknown, "journeys"].sum() / odm["journeys"].sum(),
        )
        # Nationally the share is small, but it falls on a few stations,
        # which then have no demand at all: name them, or they pass for
        # quiet stations.
        lost = odm.loc[unknown].groupby("origin_tlc")["journeys"].sum()
        total = odm.groupby("origin_tlc")["journeys"].sum()
        share = (lost / total.reindex(lost.index)).sort_values(ascending=False)
        stranded = share[share > 0.5]  # noqa: PLR2004
        if not stranded.empty:
            logger.warning(
                "%d stations have no track route for most of their journeys and so "
                "no period-matrix demand: %s.",
                len(stranded),
                ", ".join(f"{s} ({int(total[s]):,})" for s in stranded.index[:30]),
            )
    return odm


def generate_period_matrices(
    factors: pd.DataFrame,
    config: LoadingConfig,
    calc: TrackDistanceCalculator | None = None,
    typed_odm: pd.DataFrame | None = None,
    output_dir: pathlib.Path | str | None = None,
) -> list[pathlib.Path]:
    """Write one wide OD matrix per day and period; return the paths.

    ``trips(o, d) = annual_journeys(o, d) x factor`` of the pair's journey type. Written to
    ``output_dir``, by default ``<output_path>/period_matrices``.
    """
    odm = classify_od_pairs(config, calc) if typed_odm is None else typed_odm

    # Under origin_station mode the factors are keyed on "O:<origin>"
    # for origins the feed covers well enough, and on the distance band
    # otherwise. The ODM has to carry the same labels or its pairs would
    # map to no factor and be silently written as zero demand.
    if config.journey_type_mode == "origin_station":
        known = set(factors["journey_type"].astype(str))
        origin_label = "O:" + odm["origin_tlc"].astype(str)
        matched = origin_label.isin(known)
        odm = odm.copy()
        odm["journey_type"] = origin_label.where(matched, odm["journey_type"])
        logger.info(
            "Origin-station journey types: %d of %d ODM pairs (%.1f%% of "
            "journeys) matched an origin profile; the rest keep their band.",
            int(matched.sum()),
            len(odm),
            100 * odm.loc[matched, "journeys"].sum() / max(odm["journeys"].sum(), 1),
        )

    out_dir = pathlib.Path(output_dir or (config.output_path / "period_matrices"))
    out_dir.mkdir(parents=True, exist_ok=True)

    profile = None
    profile_path = getattr(config, "od_profile_path", None)
    if profile_path is not None:
        profile = od_profile.read_profile(profile_path, config.time_periods_cutoffs)
        logger.info("OD profile %s: %d cells", profile_path, len(profile))

    written: list[pathlib.Path] = []
    days: dict[str, list[tuple[str, pd.Series]]] = {}
    for (day, period), grp in factors.groupby(["Service Week Day", "time_period"], observed=True):
        factor_by_type = dict(zip(grp["journey_type"], grp["factor"], strict=True))
        mapped = odm["journey_type"].map(factor_by_type)
        if mapped.isna().any():
            lost = odm.loc[mapped.isna(), "journeys"].sum()
            if lost > 0:
                logger.warning(
                    "%s %s: %d OD pairs (%.2f%% of journeys) have no factor for "
                    "their journey type and are written as zero demand.",
                    day,
                    period,
                    int(mapped.isna().sum()),
                    100 * lost / max(odm["journeys"].sum(), 1),
                )
        days.setdefault(str(day), []).append((str(period), odm["journeys"] * mapped.fillna(0.0)))

    for day, periods in days.items():
        if profile is not None and day in od_profile.WEEKDAYS:
            cells = pd.concat(
                [
                    pd.DataFrame(
                        {
                            "origin": odm["origin_tlc"].to_numpy(object),
                            "destination": odm["destination_tlc"].to_numpy(object),
                            "period": period,
                            "trips": trips.to_numpy(float),
                        }
                    )
                    for period, trips in periods
                ],
                ignore_index=True,
            )
            f = od_profile.reweight(
                cells,
                profile,
                getattr(config, "od_profile_strength", 0.5),
                getattr(config, "od_profile_smoothing", 10.0),
            )
            parts = np.split(f, len(periods))
            periods = [  # noqa: PLW2901 (replaced by the reweighted cells)
                (period, trips * part)
                for (period, trips), part in zip(periods, parts, strict=True)
            ]
        for period, trips in periods:
            path = out_dir / f"odm_{day}_{period.replace(':', '')}.csv"
            _write_period_matrix(odm, trips, path)
            written.append(path)

    return written


def _write_period_matrix(odm: pd.DataFrame, trips: pd.Series, path: pathlib.Path) -> None:
    wide = pd.DataFrame(
        {
            "origin_tlc": odm["origin_tlc"],
            "destination_tlc": odm["destination_tlc"],
            "trips": trips,
        }
    ).pivot_table(
        index="origin_tlc",
        columns="destination_tlc",
        values="trips",
        aggfunc="sum",
        fill_value=0.0,
    )
    wide.to_csv(path)
    logger.info(
        "Wrote %s (%d origins x %d destinations)",
        path.name,
        wide.shape[0],
        wide.shape[1],
    )


def validate_against_loading(
    factors: pd.DataFrame,
    analysed_data: pd.DataFrame,
    typed_odm: pd.DataFrame,
    n_flows: int = 10,
    min_observations: int = 30,
    demand_col: str = "Mean Passengers On",
) -> pd.DataFrame:
    """Compare each busy flow's modelled day x period split with the one counted on its trains.

    For the ``n_flows`` busiest flows with ``min_observations`` loading rows, the journey type's
    profile is compared with the profile counted on trains running that exact origin to
    destination, by total variation distance (0.5 x sum of |modelled - observed| shares; 0
    identical, 1 no overlap). A large distance means the journey type is too coarse for that flow.
    """
    cells = ["Service Week Day", "time_period"]

    loads = analysed_data.copy()
    loads["o"] = loads["Origin CRS"].astype(str).str.strip().str.upper()
    loads["d"] = loads["Destination CRS"].astype(str).str.strip().str.upper()

    # Rank candidate flows by observed demand, keeping only flows that
    # also exist in the classified OD matrix.
    typed_pairs = set(zip(typed_odm["origin_tlc"], typed_odm["destination_tlc"], strict=True))
    flow_totals = (
        loads.groupby(["o", "d"])
        .agg(obs=(demand_col, "size"), demand=(demand_col, "sum"))
        .reset_index()
    )
    flow_totals = flow_totals[
        (flow_totals["obs"] >= min_observations)
        & flow_totals["o"].ne(flow_totals["d"])
        & flow_totals.apply(lambda r: (r["o"], r["d"]) in typed_pairs, axis=1)
    ].sort_values("demand", ascending=False)

    journey_type_by_pair = dict(
        zip(
            zip(
                typed_odm["origin_tlc"],
                typed_odm["destination_tlc"],
                strict=True,
            ),
            typed_odm["journey_type"],
            strict=True,
        )
    )

    # Under origin_station mode a covered origin's pairs take its
    # "O:<origin>" profile, as generate_period_matrices applies it,
    # rather than their band's.
    known = set(factors["journey_type"].astype(str))

    rows = []
    for _, flow in flow_totals.head(n_flows).iterrows():
        o, d = flow["o"], flow["d"]
        journey_type = f"O:{o}" if f"O:{o}" in known else journey_type_by_pair[(o, d)]

        modelled = factors[factors["journey_type"] == journey_type].set_index(cells)["factor"]
        modelled = modelled / modelled.sum()

        flow_rows = loads[(loads["o"] == o) & (loads["d"] == d)]
        observed = flow_rows.groupby(cells, observed=True)[demand_col].sum()
        if observed.sum() == 0:
            continue
        observed = observed / observed.sum()

        aligned = pd.concat(
            [modelled.rename("modelled"), observed.rename("observed")],
            axis=1,
        ).fillna(0.0)
        tvd = 0.5 * (aligned["modelled"] - aligned["observed"]).abs().sum()

        rows.append(
            {
                "origin": o,
                "destination": d,
                "journey_type": journey_type,
                "loading_rows": int(flow["obs"]),
                "total_variation_distance": round(tvd, 3),
            }
        )

    result = pd.DataFrame(rows)
    if not result.empty:
        logger.info(
            "Validation: mean total-variation distance over %d flows = %.3f "
            "(0 = perfect, lower is better).",
            len(result),
            result["total_variation_distance"].mean(),
        )
    return result


def run_loading(config: LoadingConfig) -> None:
    """Run the loading analysis and temporal factoring, one set of outputs per operator.

    Writes ``{operator}_loading_analysis.csv``, ``{operator}_temporal_factors.csv``,
    ``{operator}_period_matrices/`` and ``{operator}_factor_validation.csv``.
    """
    output_path = config.output_path
    output_path.mkdir(parents=True, exist_ok=True)

    # Build the calculator once and reuse it for every stage.
    calc = TrackDistanceCalculator(config.distance)

    result = analyse_loading_data(config, calc)
    if result.empty:
        logger.warning("No loading data to process.")
        return

    # The OD classification depends only on the annual ODM and the
    # distance graph, not on the operator, so compute it once and cache
    # it beside the outputs (it is ~2,600 national shortest-path
    # searches). v3: v2 caches were written before stations were snapped
    # past disconnected track fragments, and were never refreshed
    # because staleness looked only at the ODM. Sunbury,
    # Poulton-le-Fylde, Sandy and 17 more served stations kept no route,
    # so every pair touching them was written as zero demand in every
    # period matrix. The cache is now stale when any of its
    # inputs is newer than it, or when the settings the classification
    # reads (route-type bands, distance, NLC references) differ from the
    # ones in the ``.key`` file beside it; a changed band would
    # otherwise reuse old labels.
    typed_odm_cache = config.resolved_cache_dir / f"typed_odm_v3_{config.source_odm.stem}.pkl"
    typed_odm_cache.parent.mkdir(parents=True, exist_ok=True)
    settings_path = typed_odm_cache.with_suffix(".key")
    settings = repr((config.route_type, config.distance, config.nlc_reference_paths))
    inputs_mtime = max(
        pathlib.Path(p).stat().st_mtime
        for p in (
            config.source_odm,
            config.distance.track_shapefile,
            config.distance.stations.path,
            *config.nlc_reference_paths,
        )
    )
    if (
        typed_odm_cache.exists()
        and typed_odm_cache.stat().st_mtime >= inputs_mtime
        and settings_path.exists()
        and settings_path.read_text(encoding="utf-8") == settings
    ):
        logger.info("Using cached OD classification from %s", typed_odm_cache)
        typed_odm = pd.read_pickle(typed_odm_cache)  # noqa: S301 (our own cache)
    else:
        typed_odm = classify_od_pairs(config, calc)
        typed_odm.to_pickle(typed_odm_cache)
        settings_path.write_text(settings, encoding="utf-8")

    # Constrain station marginals to their table 1410 shares before
    # anything downstream reads the matrix, so the gravity fit and the
    # period matrices both see the rebalanced figures. Applied after the
    # cache rather than inside it, so one classification cache serves
    # both settings.
    if config.rebalance_to_table_1410 or config.station_size_exponent is not None:
        targets = station_targets(
            load_table_1410(config.table_1410), config.station_size_exponent or 0.0
        )
        typed_odm, balance_report = rebalance_to_station_totals(typed_odm, targets)
        balance_report.stations.to_csv(output_path / "table_1410_rebalance.csv", index=False)

    # Gravity model for legs without enough direct service to pivot
    # from.
    frequency = gravity_model.load_direct_frequency(config.route_frequency_path)
    if not frequency:
        logger.warning(
            "No route_frequency_path set; gravity model fitted without the "
            "direct-frequency term cannot price new trains and is not written."
        )
    else:
        model = gravity_model.calibrate(
            typed_odm, load_table_1410(config.table_1410), frequency, config.source_odm.name
        )
        model.save(output_path / gravity_model.MODEL_FILENAME)

    operators = [str(op).strip() for op in result["Train Operator"].unique() if pd.notna(op)]
    logger.info("Found %d operators: %s", len(operators), ", ".join(operators))

    for operator in operators:
        operator_data = result[result["Train Operator"].astype(str).str.strip() == operator].copy()

        if operator_data.empty:
            logger.warning("No data for operator: %s", operator)
            continue

        logger.info("Processing operator: %s (%d rows)", operator, len(operator_data))

        safe_operator_name = operator.lower().replace(" ", "_").replace("/", "_")
        output_file = output_path / f"{safe_operator_name}_loading_analysis.csv"
        operator_data.to_csv(output_file, index=False)
        logger.info("Results saved to %s", output_file)

        factors = compute_temporal_factors(operator_data, config)
        factors_file = output_path / f"{safe_operator_name}_temporal_factors.csv"
        factors.to_csv(factors_file, index=False)
        logger.info("Factors saved to %s", factors_file)

        # Generate period matrices for this operator (national matrices
        # are large, so only for the operators asked for when a list is
        # given).
        if config.matrix_operators is None or operator in config.matrix_operators:
            matrices_dir = output_path / f"{safe_operator_name}_period_matrices"
            matrices = generate_period_matrices(
                factors, config, calc=calc, typed_odm=typed_odm, output_dir=matrices_dir
            )
            logger.info("Wrote %d period matrices for %s.", len(matrices), operator)

        validation = validate_against_loading(
            factors, operator_data, typed_odm, demand_col=config.demand_column
        )
        validation_file = output_path / f"{safe_operator_name}_factor_validation.csv"
        validation.to_csv(validation_file, index=False)
        logger.info("Validation saved to %s", validation_file)

    logger.info(
        "Loading analysis complete. Processed %d rows across %d operators.",
        len(result),
        len(operators),
    )
