"""Build the SDFT station covariates from the Knowledgebase feed.

The Station Demand Forecasting Tool predicts a new station's annual
entries and exits without using that station's own outturn, which is
exactly the non-circular footfall that
``HindcastConfig.station_footfall`` needs for cases 01 and 05. Its two
fitted models between them take eight station attributes, and every one
that is not derivable from the timetable comes from the National Rail
Knowledgebase stations feed:

    trip end      ln(carspaces + 1), travelcard_boundary
    station choice  cctv, carspaces, ticket machine, bus, category F

This pulls the feed and writes one row per CRS in the shape those models
read. Frequency, electrification and terminus status are not here; they
come from the timetable cleans, which already know them.

The feed needs a National Rail Data Marketplace subscription. Pass the
consumer key in the environment, never in a config::

    export RDM_CONSUMER_KEY=...
    python3 scripts/build_station_covariates.py

Usage::

    python3 scripts/build_station_covariates.py \
        [--out PATH] [--cache PATH]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sys
import urllib.request

import pandas as pd

LOG = logging.getLogger(__name__)

FEED_URL = (
    "https://api1.raildata.org.uk/"
    "1010-nationalrail-knowledgebase-stations-feed-_json_---production5_0/stations"
)
KEY_VARIABLE = "RDM_CONSUMER_KEY"
DEFAULT_OUT = pathlib.Path(
    "/Volumes/ENGG341/datasources/station-covariates/station_covariates.csv"
)
# SDFT's station-choice model splits category F (the smallest, unstaffed
# halts) from everything else. The feed carries staffing rather than the
# ORR category, and unstaffed is the attribute the coefficient is
# actually standing for.
_UNSTAFFED = "unstaffed"
_REQUEST_TIMEOUT = 300


def fetch_stations(cache: pathlib.Path | None) -> list[dict]:
    """Return the station records, from ``cache`` if it exists."""
    if cache is not None and cache.exists():
        LOG.info("Reading cached feed from %s", cache)
        return json.loads(cache.read_text())["stations"]

    key = os.environ.get(KEY_VARIABLE)
    if not key:
        raise SystemExit(
            f"Set {KEY_VARIABLE} to the consumer key of your National Rail Data "
            "Marketplace subscription to the Knowledgebase stations feed."
        )
    LOG.info("Fetching %s", FEED_URL)
    # S310: FEED_URL is a fixed https constant, not caller-supplied.
    request = urllib.request.Request(FEED_URL, headers={"x-apikey": key})  # noqa: S310
    with urllib.request.urlopen(request, timeout=_REQUEST_TIMEOUT) as response:  # noqa: S310
        payload = json.load(response)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(payload))
        LOG.info("Cached feed to %s", cache)
    return payload["stations"]


def _car_spaces(station: dict) -> int:
    """Return the station's parking spaces, summing its car parks.

    The summary count is null for a third of stations that nonetheless
    list individual car parks, so falling back to the sum recovers them.
    """
    car_parks = station.get("carParks") or {}
    total = car_parks.get("numberOfSpaces")
    if total is not None:
        return int(total)
    listed = car_parks.get("carParks") or []
    spaces = [park.get("numberOfSpaces") for park in listed]
    return sum(int(space) for space in spaces if space is not None)


def _available(value: object) -> bool:
    """Return whether a feed block reports its facility as available."""
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        return bool(value.get("available"))
    return False


def build_covariates(stations: list[dict]) -> pd.DataFrame:
    """Return one row per station in the shape the SDFT models read."""
    rows = []
    for station in stations:
        crs = station.get("crsCode")
        if not crs:
            continue
        staffing = station.get("staffingLevel") or ""
        operator = station.get("stationOperator") or {}
        rows.append(
            {
                "crs": crs,
                "station_name": station.get("name"),
                "latitude": (station.get("location") or {}).get("latitude"),
                "longitude": (station.get("location") or {}).get("longitude"),
                "postcode": (station.get("address") or {}).get("postcode"),
                "car_spaces": _car_spaces(station),
                "cctv": _available((station.get("stationFacilities") or {}).get("cctvAvailable")),
                "ticket_machine": _available(
                    (station.get("ticketBuying") or {}).get("ticketMachinesAvailable")
                ),
                "bus_interchange": _available((station.get("transportLinks") or {}).get("bus")),
                "travelcard_boundary": bool(
                    (station.get("ticketBuying") or {}).get("londonFareZone")
                ),
                "london_fare_zone": (station.get("ticketBuying") or {}).get("londonFareZone"),
                "unstaffed": staffing.strip().lower() == _UNSTAFFED,
                "staffing_level": staffing or None,
                "operator_code": operator.get("operatorCode"),
                "minimum_connection_time": station.get("minimumConnectionTime"),
            }
        )
    return pd.DataFrame(rows).sort_values("crs").reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    """Fetch the feed and write the covariate table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--cache",
        type=pathlib.Path,
        default=None,
        help="read the feed from this file if it exists, else save it there",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")

    stations = fetch_stations(args.cache)
    covariates = build_covariates(stations)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    covariates.to_csv(args.out, index=False)
    LOG.info("Wrote %d stations to %s", len(covariates), args.out)

    counts = {
        "with car spaces": int((covariates["car_spaces"] > 0).sum()),
        "with CCTV": int(covariates["cctv"].sum()),
        "with ticket machine": int(covariates["ticket_machine"].sum()),
        "with bus link": int(covariates["bus_interchange"].sum()),
        "inside a London fare zone": int(covariates["travelcard_boundary"].sum()),
        "unstaffed": int(covariates["unstaffed"].sum()),
    }
    for name, count in counts.items():
        LOG.info("  %-28s %5d", name, count)


if __name__ == "__main__":
    sys.exit(main())
