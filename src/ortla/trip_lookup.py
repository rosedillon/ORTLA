"""Look up the expected trips between two stations by time.

The loading step produces one small matrix per day of the week and time
period, in files named ``odm_{Day}_{period}.csv``. Each one holds, for
every station pair, how many of that pair's yearly trips happen on that
day and in that period.

This module takes an origin and a destination, opens each of those
matrices, reads the one cell for that pair, and lays the answers out as
a grid: days down the side, time periods across the top. Because the
matrices together split the whole year up, adding every cell for a pair
back together gives its yearly total.

We can also label the pair by how far apart the two stations are along
the track (commute / regional / long_distance); see
:func:`classify_journey_type`.
"""

from __future__ import annotations

import logging
import pathlib
from typing import TYPE_CHECKING

import pandas as pd

from ortla.clock import WEEKDAYS_PER_YEAR, period_from_file_label, to_per_day
from ortla.period_matrices import classify_route_type
from ortla.track_distance import TrackDistanceCalculator

if TYPE_CHECKING:
    from ortla.config import LookupConfig, RouteTypeConfig

logger = logging.getLogger(__name__)


def _parse_day_period(path: pathlib.Path) -> tuple[str, str]:
    """Read the day and period in an ``odm_{day}_{period}.csv`` name."""
    stem = path.stem.removeprefix("odm_")
    day, _, period = stem.partition("_")
    return day, period_from_file_label(period)


def lookup_trips_batch(
    pairs: list[tuple[str, str]],
    matrices_dir: pathlib.Path | str,
    day_order: list[str],
    period_order: list[str],
    fill_value: str = "!",
) -> dict[tuple[str, str], pd.DataFrame]:
    """Look up many (origin, destination) pairs in one pass over the matrices."""
    normalized_pairs = [(orig.strip().upper(), dest.strip().upper()) for orig, dest in pairs]
    matrices_dir = pathlib.Path(matrices_dir)
    files = sorted(matrices_dir.glob("odm_*.csv"))
    if not files:
        raise FileNotFoundError(f"No matrices found in {matrices_dir}")

    all_origins = {orig for orig, _ in normalized_pairs}
    all_dests = {dest for _, dest in normalized_pairs}

    pair_records: dict[tuple[str, str], list[dict[str, object]]] = {
        pair: [] for pair in normalized_pairs
    }

    for path in files:
        day, period = _parse_day_period(path)
        logger.info("Reading %s (%s %s) ...", path.name, day, period)

        try:
            available_cols = set(pd.read_csv(path, nrows=0).columns)
        except (OSError, ValueError):  # unreadable or not a CSV
            available_cols = set()

        needed_dests = available_cols & all_dests
        if not needed_dests or "origin_tlc" not in available_cols:
            for pair in normalized_pairs:
                pair_records[pair].append({"day": day, "period": period, "trips": None})
            continue

        usecols = ["origin_tlc", *needed_dests]
        df = pd.read_csv(path, usecols=usecols, dtype=str)
        df["origin_tlc"] = df["origin_tlc"].astype(str).str.strip().str.upper()
        df = df[df["origin_tlc"].isin(all_origins)]
        df = df.set_index("origin_tlc")

        for orig, dest in normalized_pairs:
            if dest not in needed_dests or orig not in df.index:
                trips = None
            else:
                val = df.loc[orig, dest]
                if isinstance(val, pd.Series):
                    val = val.iloc[0]
                if val == fill_value or pd.isna(val):
                    trips = None
                else:
                    try:
                        trips = float(val)
                    except (ValueError, TypeError):
                        trips = None
            pair_records[(orig, dest)].append({"day": day, "period": period, "trips": trips})

    grids: dict[tuple[str, str], pd.DataFrame] = {}
    for (orig, dest), records in pair_records.items():
        grid = (
            pd.DataFrame.from_records(records)  # noqa: PD010
            .pivot(index="day", columns="period", values="trips")
            .reindex(index=day_order)
            .reindex(columns=period_order)
        )
        grid.index.name = f"{orig} -> {dest}"
        grids[(orig, dest)] = grid

    return grids


def classify_journey_type(
    origin_tlc: str,
    destination_tlc: str,
    route_cfg: RouteTypeConfig,
    calc: TrackDistanceCalculator,
) -> tuple[str, float | None]:
    """Band a pair by its track distance as the temporal factoring does.

    Returns the label (commute, regional, long_distance) and the distance in miles, or
    ``"Unknown"`` and None without a track route.
    """
    origin_tlc = origin_tlc.strip().upper()
    destination_tlc = destination_tlc.strip().upper()

    try:
        distance_miles = calc.calculate_distance(origin_tlc, destination_tlc, unit="miles")
    except ValueError:
        return "Unknown", None

    return classify_route_type(distance_miles, route_cfg), distance_miles


def run_lookup(config: LookupConfig) -> None:
    """Report the day x period grid for one pair, or every pair among ``station_codes``.

    The track distance calculator, needed only for ``journey_type``, is built once.
    """
    config.output_path.mkdir(parents=True, exist_ok=True)
    pairs = config.pairs()

    calc = None
    if config.journey_type:
        if config.distance is None:
            raise ValueError("journey_type requires a 'distance' calculator config.")
        calc = TrackDistanceCalculator(config.distance)

    logger.info(
        "Looking up trips for %d station pair(s) in a single pass over matrices...", len(pairs)
    )
    grids = lookup_trips_batch(
        pairs,
        config.period_matrices_dir,
        config.day_order,
        config.period_order,
        config.fill_value,
    )

    for origin, destination in pairs:
        norm_orig = origin.strip().upper()
        norm_dest = destination.strip().upper()
        grid = grids[(norm_orig, norm_dest)]
        annual = grid.sum().sum()

        if config.per_day:
            output_grid = to_per_day(grid)
            logger.info(
                "Expected trips per typical day (%s -> %s):\n%s",
                norm_orig,
                norm_dest,
                output_grid.to_string(),
            )
            logger.info(
                "Total per typical day: %.4f (annual total %.1f / %.2f)",
                output_grid.sum().sum(),
                annual,
                WEEKDAYS_PER_YEAR,
            )
        else:
            output_grid = grid
            logger.info(
                "Annual expected trips grid (%s -> %s):\n%s",
                norm_orig,
                norm_dest,
                grid.to_string(),
            )
            logger.info("Annual expected trips: %.1f", annual)

        out_path = config.output_path / f"lookup_{norm_orig}_{norm_dest}.csv"
        output_grid.to_csv(out_path)
        logger.info("Lookup grid written to %s", out_path)

        if calc is not None:
            band, miles = classify_journey_type(norm_orig, norm_dest, config.route_type, calc)
            if miles is None:
                logger.info("Journey type: %s (no track route found)", band)
            else:
                logger.info("Journey type: %s (%.1f miles O->D)", band, miles)
