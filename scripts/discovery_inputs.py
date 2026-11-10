"""Build the city-market table the national discovery scan needs.

``discover``'s existing-service filter reads ``daily_trains`` from a
`timetable` run and tests one station pair at a time. At national scale
that asks the wrong question twice over, and both are documented in
DISCOVERY_CASES.md section 3.2:

* **A London terminus is a gateway, not a destination.** Table 1410
  gives Liverpool Street 80.4m entries and exits, almost all of them
  travelling to or from somewhere else in London, and the gravity model
  reads that as destination mass. So "no direct train Reading ->
  Liverpool Street" scores as a large unserved market when the real
  market, Reading -> London, runs 204 times a day from Paddington.
* **Both ends in one conurbation is a metro journey**, not an
  inter-urban rail market, whatever the distance between the two
  stations.

This rewrites ``daily_trains`` so a pair counts as served when any
station in the origin's city zone has a direct train to any station in
the destination's zone, and suppresses intra-conurbation pairs outright.
``discover`` is then pointed at the result and runs unmodified.

Direction symmetry (section 3.3) is handled inside ``discover`` itself,
by ``DiscoveryConfig.existing_service_both_directions``; it is applied
here too so the written table reads the same way round.

The null-``crs_code`` defect of section 3.1 is now fixed in the library:
set ``crs_reference`` on the `timetable` config (see
``configs/paper/timetable.yml``) and the platform-group TIPLOCs are
recovered during the clean. A calls file produced before that fix needs
re-cleaning, not patching.

Run from the repo root, after the `timetable` and `bplan` paper configs:

    PYTHONPATH=src python3 scripts/discovery_inputs.py
"""

from __future__ import annotations

import collections
import logging
import os
import pathlib
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ortla.config import TrackCalculatorConfig
from ortla.discovery import _pairs_within_distance_band
from ortla.stations import (
    canonical_crs,
    read_station_list,
    resolve_aliases,
    resolve_recodes,
)
from ortla.track_distance import TrackDistanceCalculator

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

DATA = pathlib.Path(os.environ.get("ORM_DATA", "/Volumes/ENGG341/datasources"))
PAPER = pathlib.Path(os.environ.get("ORM_OUT", DATA / "output_paper"))  # ORM_OUT: a regeneration
TIMETABLE_DIR = PAPER / os.environ.get("ORM_TIMETABLE", "timetable")
OUTPUT_DIR = PAPER / "timetable_city_market"

# Conurbation centre CRS -> radius in miles. A pair with both ends
# inside one of these is a metro journey, not an inter-urban market.
CONURBATIONS = {
    "CHX": 20.0,
    "BHM": 12.0,
    "MAN": 12.0,
    "GLC": 12.0,
    "LDS": 12.0,
    "LIV": 12.0,
    "NCL": 12.0,
    "EDB": 12.0,
    "SHF": 12.0,
    "CDF": 12.0,
    "BRI": 12.0,
    "NOT": 10.0,
    "LEI": 10.0,
    "SOU": 10.0,
    "PMS": 10.0,
    "BTN": 6.0,
    "COV": 8.0,
    "DBY": 8.0,
    "SOT": 8.0,
    "NWP": 8.0,
    "HUL": 8.0,
    "PLY": 8.0,
    "ABD": 8.0,
    "SWA": 8.0,
    "YRK": 8.0,
    "PRE": 8.0,
    "MDB": 8.0,
    "SUN": 8.0,
    "BDQ": 8.0,
}
# A station outside every conurbation still shares a market with its
# immediate neighbours. Windsor & Eton Central and Windsor & Eton
# Riverside serve one town, and only the second has a direct London
# service.
NEIGHBOURHOOD_MILES = 2.0
MIN_MILES, MAX_MILES = 15.0, 300.0
# Written for an intra-conurbation pair, well above any real frequency,
# so the pair is dropped whatever `max_existing_daily_trains` is set to.
SUPPRESSED = 9999.0
METRES_PER_MILE = 1609.344


def _distance_config() -> TrackCalculatorConfig:
    return TrackCalculatorConfig.model_validate(
        {
            "track_shapefile": str(DATA / "NWR_TrackModel20260904/NWR_TrackCentreLines.shp"),
            "stations": {
                "path": str(DATA / "stations.csv"),
                "crs": "crsCode",
                "lat": "lat",
                "lon": "long",
            },
            "coordinate_systems": {
                "input_crs": "EPSG:4326",
                "working_crs": "EPSG:27700",
            },
        }
    )


def _assign_zones(codes: list[str], coords: np.ndarray, tree: cKDTree) -> dict[str, str]:
    """Map each station CRS to the conurbation centre it falls in."""
    position = {code: i for i, code in enumerate(codes)}
    zone: dict[str, str] = {}
    for centre, radius in CONURBATIONS.items():
        if centre not in position:
            logger.warning("Conurbation centre %s is not a known station; skipped.", centre)
            continue
        for j in tree.query_ball_point(coords[position[centre]], radius * METRES_PER_MILE):
            zone.setdefault(codes[j], centre)
    return zone


def _suppression_rows(
    pairs: list[tuple[str, str, float]],
    zone: dict[str, str],
    served: Callable[[str, str], float],
) -> tuple[list[tuple[str, str, float]], int]:
    """Build the (origin, destination, daily_trains) suppressing rows.

    Pairs left out are read as zero by ``_drop_well_served_pairs``, so
    only the ones to suppress need writing.
    """
    rows: list[tuple[str, str, float]] = []
    n_intra = 0
    for origin, destination, _miles in pairs:
        if origin in zone and zone[origin] == zone.get(destination):
            rows.append((origin, destination, SUPPRESSED))
            n_intra += 1
            continue
        # A direct link is a property of a corridor, not a direction.
        value = max(served(origin, destination), served(destination, origin))
        if value > 1.0:
            rows.append((origin, destination, value))
    return rows, n_intra


def build_city_market_table(
    timetable_dir: pathlib.Path = TIMETABLE_DIR,
    output_dir: pathlib.Path = OUTPUT_DIR,
) -> pathlib.Path:
    """Rewrite daily_trains so the test sees cities and direction."""
    calculator = TrackDistanceCalculator(_distance_config())
    locations = calculator.station_locations
    codes = sorted(locations)
    position = {code: i for i, code in enumerate(codes)}
    coords = np.array([locations[code] for code in codes])
    tree = cKDTree(coords)

    zone = _assign_zones(codes, coords, tree)
    members = collections.defaultdict(list)
    for code, centre in zone.items():
        members[centre].append(code)

    def neighbourhood(code: str) -> list[str]:
        if code in zone:
            return members[zone[code]]
        return [
            codes[j]
            for j in tree.query_ball_point(
                coords[position[code]], NEIGHBOURHOOD_MILES * METRES_PER_MILE
            )
        ]

    def zone_key(code: str) -> str:
        return zone.get(code) or code

    raw = pd.read_csv(timetable_dir / "route_frequency_summary.csv")
    # The timetable books some calls under a station's second code
    # (Retford low level XRO, Worcestershire Parkway high level WPH),
    # which no zone holds: read as their own codes, Paddington ->
    # Worcestershire Parkway's 20 trains a day were lost and it scored
    # as an unserved market (case 25).
    recodes = resolve_recodes(None)
    aliases = resolve_aliases(None)

    def station(code: str) -> str:
        return canonical_crs(recodes.get(code, code), aliases)

    # Collapse to zone level once, so the per-pair lookup below is O(1)
    # rather than |N(origin)| x |N(destination)| over 5.6m pairs.
    zone_max: dict[tuple[str, str], float] = collections.defaultdict(float)
    for origin, destination, trains in zip(
        raw.origin_crs.str.upper(),
        raw.destination_crs.str.upper(),
        raw.daily_trains,
        strict=True,
    ):
        key = (zone_key(station(origin)), zone_key(station(destination)))
        if trains > zone_max[key]:
            zone_max[key] = float(trains)

    def served(origin: str, destination: str) -> float:
        origins = (
            {zone_key(origin)} if origin in zone else {zone_key(x) for x in neighbourhood(origin)}
        )
        destinations = (
            {zone_key(destination)}
            if destination in zone
            else {zone_key(x) for x in neighbourhood(destination)}
        )
        return max(
            (zone_max.get((a, b), 0.0) for a in origins for b in destinations),
            default=0.0,
        )

    stations, _ = read_station_list(DATA / "stations.csv")
    pairs = _pairs_within_distance_band(calculator, stations, MIN_MILES, MAX_MILES, len(codes))
    rows, n_intra = _suppression_rows(pairs, zone, served)

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["origin_crs", "destination_crs", "daily_trains"]).to_csv(
        output_dir / "route_frequency_summary.csv", index=False
    )
    logger.info(
        "City-market table: %d pairs in band, %d already served, %d intra-conurbation, "
        "%d left as candidates -> %s",
        len(pairs),
        len(rows) - n_intra,
        n_intra,
        len(pairs) - len(rows),
        output_dir,
    )
    return output_dir


def main() -> None:
    """Build the city-market table from the paper's timetable run."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_city_market_table()


if __name__ == "__main__":
    main()
