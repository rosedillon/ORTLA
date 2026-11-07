"""Download, clean and look up the official RSP fares feed.

The Fares and Associated Data Feed is published by Rail Delivery Group
and Rail Settlement Plan (RSP) and distributed through the National Rail
Data Portal (NRDP, https://opendata.nationalrail.co.uk). It is
fixed-format flat text, one record type per line (a leading ``/`` marks
a comment). The field positions below follow the official interface
specification, RSPS5045 ("Fares and Associated Data Feed Interface
Specification").

Only the ``FLOW`` file (``.FFL``, point-to-point adult fares), the
``LOCATIONS`` file (``.LOC``, NLC, CRS and name) and the
``TICKET TYPES`` file (``.TTY``, ticket descriptions) are read, with the
station clusters (``.FSC``). Together these give a queryable table of
the standard fares (Anytime, Off-Peak, Season and so on) between any two
stations.

The feed has two limits:

* It has no date-varying Advance fares. Those are set by each operator's
  own yield-managed booking system and are not in this static industry
  feed, so nothing here gives the price for travel on a particular date.
* Fares can be set for a station cluster (``ORIGIN_CODE`` or
  ``DESTINATION_CODE`` is a cluster NLC covering several stations, such
  as ``Q814``) or a station group (``1072``, LONDON TERMINALS) rather
  than per station. About half the feed's fare rows are. The cluster
  file and the ``.LOC`` file's group members are read into
  :data:`CLUSTERS_CSV`, so a station resolves to every code it is priced
  under.

The feed also carries negotiated and wholesale products (Group, Tour
Operator, School Group, Inclusive Tour, Comp, Staff, child concession)
that have no individually advertised price. They are priced at a handful
of fixed placeholder amounts nationwide rather than by distance. Every
looked-up row is flagged in a ``non_retail_fare`` column (see
:func:`_flag_non_retail`), so they stay visible but do not distort a
"cheapest fare" read.

To download, POST credentials to ``/authenticate`` for a token, then GET
the feed with that token in an ``X-Auth-Token`` header. Credentials come
from :class:`~orta.config.FaresLookupConfig`'s ``username``
and ``password`` fields, falling back to the
``NRDP_USERNAME``/``NRDP_PASSWORD`` environment variables when unset.
"""

from __future__ import annotations

import collections
import dataclasses
import datetime
import io
import json
import logging
import os
import urllib.parse
import urllib.request
import zipfile
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import pathlib

    from orta.config import FaresLookupConfig

logger = logging.getLogger(__name__)


_AUTH_URL = "https://opendata.nationalrail.co.uk/authenticate"


_FARES_URL = "https://opendata.nationalrail.co.uk/api/staticfeeds/2.0/fares"


_REQUEST_TIMEOUT_SECONDS = 120


# Cleaned CSV filenames written by clean_fares_feed(), read back by
# fares.py.
FLOW_CSV = "fares_flow.csv"


LOCATIONS_CSV = "fares_locations.csv"


TICKETS_CSV = "fares_tickets.csv"


CLUSTERS_CSV = "fares_clusters.csv"


# (start, end) character spans are 0-indexed/exclusive-end python
# slices, transcribed from RSPS5045's 1-indexed inclusive "Position"
# column.
_FLOW_FIELDS = {
    "update_marker": (0, 1),
    "record_type": (1, 2),
    "origin_code": (2, 6),
    "destination_code": (6, 10),
    "route_code": (10, 15),
    "status_code": (15, 18),
    "usage_code": (18, 19),
    "direction": (19, 20),
    "end_date": (20, 28),
    "start_date": (28, 36),
    "toc": (36, 39),
    "cross_london_ind": (39, 40),
    "flow_id": (42, 49),
}


_FARE_FIELDS = {
    "update_marker": (0, 1),
    "record_type": (1, 2),
    "flow_id": (2, 9),
    "ticket_code": (9, 12),
    "fare_pence": (12, 20),
    "restriction_code": (20, 22),
}


_LOCATION_FIELDS = {
    "update_marker": (0, 1),
    "record_type": (1, 2),
    "nlc_code": (36, 40),
    "description": (40, 56),
    "crs_code": (56, 59),
}


_TICKET_TYPE_FIELDS = {
    "update_marker": (0, 1),
    "ticket_code": (1, 4),
    "description": (28, 43),
    "tkt_class": (43, 44),
    "tkt_type": (44, 45),
    "tkt_group": (45, 46),
}


def _authenticate(username: str, password: str) -> str:
    """Post credentials to NRDP ``/authenticate``; return the token."""
    body = urllib.parse.urlencode({"username": username, "password": password}).encode()
    request = urllib.request.Request(  # noqa: S310 (fixed https URL, not user input)
        _AUTH_URL,
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=_REQUEST_TIMEOUT_SECONDS) as response:  # noqa: S310
        payload = json.loads(response.read())
    token = payload.get("token")
    if not token:
        raise ValueError("NRDP authentication response did not include a token.")
    return token


def _download_zip(token: str) -> bytes:
    """GET the fares feed zip from NRDP using an auth token."""
    request = urllib.request.Request(  # noqa: S310 (fixed https URL, not user input)
        _FARES_URL, headers={"X-Auth-Token": token}
    )
    with urllib.request.urlopen(request, timeout=_REQUEST_TIMEOUT_SECONDS) as response:  # noqa: S310
        return response.read()


def fetch_fares_feed(
    dest_dir: pathlib.Path, username: str | None = None, password: str | None = None
) -> pathlib.Path:
    """Authenticate with NRDP, download the fares feed and extract it.

    Parameters
    ----------
    dest_dir : pathlib.Path
        Directory the extracted ``.FFL``/``.LOC``/``.TTY`` etc. files
        are written to (created if missing).
    username, password : str | None, optional
        NRDP account credentials. Either falls back to the
        ``NRDP_USERNAME``/``NRDP_PASSWORD`` environment variables when
        left unset here.

    Returns
    -------
    pathlib.Path
        ``dest_dir``, for chaining into :func:`clean_fares_feed`.

    Raises
    ------
    ValueError
        If neither the arguments nor the environment variables supply
        both a username and a password.
    """
    username = username or os.environ.get("NRDP_USERNAME")
    password = password or os.environ.get("NRDP_PASSWORD")
    if not username or not password:
        raise ValueError(
            "No NRDP credentials given: set username/password (e.g. in "
            "configs/fares.yml) or the NRDP_USERNAME/NRDP_PASSWORD "
            "environment variables, for a National Rail Data Portal account "
            "(https://opendata.nationalrail.co.uk)."
        )

    logger.info("Authenticating with the National Rail Data Portal ...")
    token = _authenticate(username, password)

    logger.info("Downloading the fares feed from %s ...", _FARES_URL)
    archive_bytes = _download_zip(token)

    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        archive.extractall(dest_dir)
    logger.info("Fares feed extracted to %s", dest_dir)
    return dest_dir


def _find_one(raw_feed_dir: pathlib.Path, suffix: str) -> pathlib.Path:
    """Return the single file with ``suffix`` in ``raw_feed_dir``."""
    # A name starting "." is macOS metadata (``._RJFA.FFL`` on an exFAT
    # drive), not the feed.
    matches = sorted(p for p in raw_feed_dir.glob(f"*{suffix}") if not p.name.startswith("."))
    if not matches:
        raise FileNotFoundError(f"No *{suffix} file found in {raw_feed_dir}.")
    return matches[0]


def _slice_row(line: str, fields: dict[str, tuple[int, int]]) -> dict[str, str]:
    return {name: line[start:end].strip() for name, (start, end) in fields.items()}


def _iso(ddmmyyyy: str) -> str:
    """Write a ``DDMMYYYY`` feed date as ``YYYYMMDD`` so it sorts."""
    return ddmmyyyy[4:8] + ddmmyyyy[2:4] + ddmmyyyy[0:2]


def parse_flow_file(path: pathlib.Path, on_date: datetime.date | None = None) -> pd.DataFrame:
    """Parse a ``.FFL`` file into one row per fare (flows and fares).

    Only flows valid on ``on_date`` (default today) are kept. Ahead of a
    fares change the feed carries the old and the new flow for the same
    journey, each with its own validity dates, and keeping both would
    mix their prices.

    Parameters
    ----------
    path : pathlib.Path
        Path to the ``RJFAtnnn.FFL`` file.
    on_date : datetime.date | None, optional
        Date the fares must be valid on; today when ``None``.

    Returns
    -------
    pd.DataFrame
        Columns: ``origin_code``, ``destination_code``, ``route_code``,
        ``direction``, ``toc``, ``cross_london_ind``, ``ticket_code``,
        ``fare_pence``, ``restriction_code``.
    """
    day = (on_date or datetime.date.today()).strftime("%Y%m%d")
    flows: dict[str, dict[str, str]] = {}
    fares = []
    with path.open(encoding="latin-1") as handle:
        for line in handle:
            if not line or line.startswith("/"):
                continue
            record_type = line[1:2]
            if record_type == "F":
                row = _slice_row(line, _FLOW_FIELDS)
                if row["update_marker"] != "D" and _iso(row["start_date"]) <= day <= _iso(
                    row["end_date"]
                ):
                    flows[row["flow_id"]] = row
            elif record_type == "T":
                row = _slice_row(line, _FARE_FIELDS)
                if row["update_marker"] != "D":
                    fares.append(row)

    fare_df = pd.DataFrame(fares)
    if fare_df.empty:
        return pd.DataFrame(
            columns=[
                "origin_code",
                "destination_code",
                "route_code",
                "direction",
                "toc",
                "cross_london_ind",
                "ticket_code",
                "fare_pence",
                "restriction_code",
            ]
        )

    flow_df = pd.DataFrame.from_dict(flows, orient="index")
    joined = fare_df.merge(flow_df, on="flow_id", how="inner", suffixes=("", "_flow"))
    joined.loc[:, "fare_pence"] = pd.to_numeric(joined["fare_pence"], errors="coerce")
    return joined[
        [
            "origin_code",
            "destination_code",
            "route_code",
            "direction",
            "toc",
            "cross_london_ind",
            "ticket_code",
            "fare_pence",
            "restriction_code",
        ]
    ].dropna(subset=["fare_pence"])


def parse_locations_file(path: pathlib.Path) -> pd.DataFrame:
    """Parse a ``.LOC`` file into an NLC -> CRS and name lookup.

    Parameters
    ----------
    path : pathlib.Path
        Path to the ``RJFAtnnn.LOC`` file.

    Returns
    -------
    pd.DataFrame
        Columns: ``nlc_code``, ``crs_code``, ``description``. Locations
        with no NLC (non-GB) or no CRS code are dropped.
    """
    rows = []
    with path.open(encoding="latin-1") as handle:
        for line in handle:
            if not line or line.startswith("/") or line[1:2] != "L":
                continue
            row = _slice_row(line, _LOCATION_FIELDS)
            if row["update_marker"] != "D":
                rows.append(row)

    locations = pd.DataFrame(rows)
    if locations.empty:
        return pd.DataFrame(columns=["nlc_code", "crs_code", "description"])
    locations = locations[(locations["nlc_code"] != "") & (locations["crs_code"] != "")]
    return locations[["nlc_code", "crs_code", "description"]].drop_duplicates("nlc_code")


def parse_ticket_types_file(path: pathlib.Path) -> pd.DataFrame:
    """Parse a ``.TTY`` file into a ticket-code lookup.

    Parameters
    ----------
    path : pathlib.Path
        Path to the ``RJFAtnnn.TTY`` file.

    Returns
    -------
    pd.DataFrame
        Columns: ``ticket_code``, ``description``, ``tkt_class``,
        ``tkt_type``, ``tkt_group``.
    """
    rows = []
    with path.open(encoding="latin-1") as handle:
        for line in handle:
            if not line or line.startswith("/"):
                continue
            row = _slice_row(line, _TICKET_TYPE_FIELDS)
            if row["update_marker"] != "D":
                rows.append(row)

    tickets = pd.DataFrame(rows)
    if tickets.empty:
        return pd.DataFrame(
            columns=["ticket_code", "description", "tkt_class", "tkt_type", "tkt_group"]
        )
    return tickets[
        ["ticket_code", "description", "tkt_class", "tkt_type", "tkt_group"]
    ].drop_duplicates("ticket_code")


def parse_clusters(raw_feed_dir: pathlib.Path) -> pd.DataFrame:
    """Every (cluster or group NLC, member NLC) pair in the feed.

    Station clusters come from the ``.FSC`` file (``R`` + cluster NLC +
    member NLC + dates) and station groups (LONDON TERMINALS, BIRMINGHAM
    STNS, ...) from the ``.LOC`` file's ``M`` records, whose 7-digit UIC
    codes carry the NLC in digits 3-6.

    Returns
    -------
    pd.DataFrame
        Columns: ``cluster_code``, ``nlc_code``.
    """
    rows = []
    with _find_one(raw_feed_dir, ".FSC").open(encoding="latin-1") as handle:
        rows += [(line[1:5], line[5:9]) for line in handle if line[:1] == "R"]
    with _find_one(raw_feed_dir, ".LOC").open(encoding="latin-1") as handle:
        rows += [(line[4:8], line[19:23]) for line in handle if line[:2] == "RM"]
    return pd.DataFrame(rows, columns=["cluster_code", "nlc_code"]).drop_duplicates()


def clean_fares_feed(raw_feed_dir: pathlib.Path, output_path: pathlib.Path) -> dict[str, int]:
    """Parse the extracted feed and write the four cleaned lookup CSVs.

    Parameters
    ----------
    raw_feed_dir : pathlib.Path
        Directory containing the extracted ``.FFL``/``.LOC``/``.TTY``
        files (see :func:`fetch_fares_feed`).
    output_path : pathlib.Path
        Directory the cleaned CSVs (:data:`FLOW_CSV`,
        :data:`LOCATIONS_CSV`, :data:`TICKETS_CSV` and
        :data:`CLUSTERS_CSV`) are written to.

    Returns
    -------
    dict[str, int]
        Rows written per cleaned file.
    """
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Parsing flow/fare records ...")
    flow = parse_flow_file(_find_one(raw_feed_dir, ".FFL"))
    flow.to_csv(output_path / FLOW_CSV, index=False)

    logger.info("Parsing location records ...")
    locations = parse_locations_file(_find_one(raw_feed_dir, ".LOC"))
    locations.to_csv(output_path / LOCATIONS_CSV, index=False)

    logger.info("Parsing ticket type records ...")
    tickets = parse_ticket_types_file(_find_one(raw_feed_dir, ".TTY"))
    tickets.to_csv(output_path / TICKETS_CSV, index=False)

    clusters = parse_clusters(raw_feed_dir)
    clusters.to_csv(output_path / CLUSTERS_CSV, index=False)

    counts = {
        "flow": len(flow),
        "locations": len(locations),
        "tickets": len(tickets),
        "clusters": len(clusters),
    }
    logger.info("Fares feed cleaned: %s", ", ".join(f"{k}={v}" for k, v in counts.items()))
    return counts


_OUTPUT_COLUMNS = [
    "origin",
    "origin_name",
    "destination",
    "destination_name",
    "ticket_code",
    "ticket_description",
    "tkt_class",
    "tkt_type",
    "fare_pence",
    "fare_gbp",
    "restriction_code",
    "toc",
    "route_code",
    "non_retail_fare",
]


# Confirmed against the live feed: negotiated products with no
# individually advertised price (Group, Tour Operator, School Group,
# Special Events, Family, LNER/Avanti Comp, ...) are priced at one of
# these three round-number placeholders on every route nationally, never
# a real distance-based amount.
_NON_RETAIL_SENTINEL_PENCE = {99900, 99999, 999800}  # GBP 999.00 / 999.99 / 9998.00


# Wholesale/settlement products (Inclusive Tour, Comp, Staff, child
# concession) are priced near-identically for every route nationally,
# with almost no relationship to distance, so they are a nominal token
# rather than a retail fare. A ticket code only counts if it shows this
# near-flat price across enough routes to rule out coincidence (a real
# distance-based fare would take far more than 3 distinct values across
# that many routes).
_NON_RETAIL_FLAT_MIN_ROUTES = 20


_NON_RETAIL_FLAT_MAX_UNIQUE_PRICES = 3


_NON_RETAIL_FLAT_MAX_PENCE = 20


def _flag_non_retail(flow: pd.DataFrame) -> pd.Series:
    """Flag fares that aren't genuine point-to-point retail prices.

    See :data:`_NON_RETAIL_SENTINEL_PENCE` and the flat-price rule
    below. Both patterns were found by inspecting the full national
    feed, not guessed from a single station pair.
    """
    sentinel = flow["fare_pence"].isin(_NON_RETAIL_SENTINEL_PENCE)

    per_ticket = flow.groupby("ticket_code", observed=True)["fare_pence"].agg(
        nunique="nunique", count="count", price="max"
    )
    flat_low_codes = per_ticket[
        (per_ticket["nunique"] <= _NON_RETAIL_FLAT_MAX_UNIQUE_PRICES)
        & (per_ticket["count"] >= _NON_RETAIL_FLAT_MIN_ROUTES)
        & (per_ticket["price"] <= _NON_RETAIL_FLAT_MAX_PENCE)
    ].index
    flat_low = flow["ticket_code"].isin(flat_low_codes)

    return sentinel | flat_low


def _cleaned_files_exist(output_path: pathlib.Path) -> bool:
    return all((output_path / name).exists() for name in (FLOW_CSV, LOCATIONS_CSV, TICKETS_CSV))


@dataclasses.dataclass
class FaresTable:
    """The cleaned feed, queryable by station CRS.

    ``fares`` holds one row per flow fare, keyed by NLC: a station's
    own, a cluster's or a group's. ``codes`` maps each CRS to every NLC
    it is priced under, so :func:`lookup_pair` finds cluster and group
    fares too.
    """

    fares: pd.DataFrame
    codes: dict[str, frozenset[str]]
    names: dict[str, str]


def build_fares_table(output_path: pathlib.Path) -> FaresTable:
    """Load the cleaned feed CSVs into a :class:`FaresTable`.

    Parameters
    ----------
    output_path : pathlib.Path
        Directory containing the cleaned CSVs written by
        :func:`clean_fares_feed`. One cleaned
        before clusters were read gets them from its ``raw`` folder.
    """
    flow = pd.read_csv(
        output_path / FLOW_CSV,
        dtype={"fare_pence": "int64"}
        | dict.fromkeys(
            (
                "origin_code",
                "destination_code",
                "route_code",
                "direction",
                "toc",
                "cross_london_ind",
                "ticket_code",
                "restriction_code",
            ),
            "category",
        ),
    )
    flow["non_retail_fare"] = _flag_non_retail(flow)
    tickets = pd.read_csv(output_path / TICKETS_CSV, dtype=str)
    flow = flow.merge(
        tickets.rename(columns={"description": "ticket_description"}),
        on="ticket_code",
        how="left",
    )
    flow["fare_gbp"] = flow["fare_pence"] / 100

    locations = pd.read_csv(output_path / LOCATIONS_CSV, dtype=str)
    clusters_path = output_path / CLUSTERS_CSV
    if not clusters_path.exists():
        parse_clusters(output_path / "raw").to_csv(clusters_path, index=False)
    clusters = pd.read_csv(clusters_path, dtype=str)
    member_of = clusters.groupby("nlc_code")["cluster_code"].agg(set).to_dict()

    codes: dict[str, set[str]] = collections.defaultdict(set)
    for crs, nlc in zip(locations["crs_code"], locations["nlc_code"], strict=True):
        # A group (London Terminals) can itself sit in a cluster.
        groups = member_of.get(nlc, set())
        codes[crs] |= {nlc, *groups, *(c for g in groups for c in member_of.get(g, ()))}
    names = dict(zip(locations["crs_code"], locations["description"], strict=True))
    return FaresTable(flow, {k: frozenset(v) for k, v in codes.items()}, names)


def lookup_pair(table: FaresTable, origin: str, destination: str) -> pd.DataFrame:
    """Every fare that prices ``origin`` -> ``destination``.

    Flows set between any code either station is priced under (its own
    NLC, its clusters, its groups) match. A reversible flow
    (``direction == "R"``) also prices the reverse direction.

    Parameters
    ----------
    table : FaresTable
        From :func:`build_fares_table`.
    origin, destination : str
        Upper-cased CRS codes.

    Returns
    -------
    pd.DataFrame
        Matching fares, columns as in :data:`_OUTPUT_COLUMNS`, cheapest
        first.
    """
    fares = table.fares
    o = table.codes.get(origin, frozenset())
    d = table.codes.get(destination, frozenset())
    forward = fares["origin_code"].isin(o) & fares["destination_code"].isin(d)
    reverse = (
        (fares["direction"] == "R")
        & fares["origin_code"].isin(d)
        & fares["destination_code"].isin(o)
    )
    matches = fares[forward | reverse].assign(
        origin=origin,
        destination=destination,
        origin_name=table.names.get(origin),
        destination_name=table.names.get(destination),
    )
    matches = matches.drop_duplicates(
        subset=["ticket_code", "restriction_code", "fare_pence", "route_code"]
    )
    return matches[_OUTPUT_COLUMNS].sort_values("fare_pence").reset_index(drop=True)


# The ordinary adult products. Matched on the description's start; the
# feed also carries hundreds of others on the same flows (family,
# small-group, child, "Kid with Adult" at GBP 2, "SALE ADVANCE" at GBP
# 1, PAYG/CPAY "INFO"/"TEST" entries, concessions), which no list of
# exclusions keeps up with.
_ADULT_PRODUCTS = (
    "ADVANCE",
    "ANYTIME",
    "OFF-PEAK",
    "SUPER OFFPEAK",
    "SUP OFFPK",
    "LUMOFIXED",
    "SEMI FLEX",
)


# Standard Premium is first class in all but name.
_NOT_ADULT_STANDARD = ("STDPREM",)


# Below this a standard-class fare is a token, not what a passenger
# pays.
_MIN_RETAIL_FARE_GBP = 1.0


def median_fare(table: FaresTable, origin: str, destination: str) -> float | None:
    """Median fare per journey for one O-D pair, in GBP.

    The feed has no sales weights, so each ordinary adult standard-class
    product on the pair (:data:`_ADULT_PRODUCTS`: Advance rungs,
    Anytime, Off-Peak, Super Off-Peak) counts once, singles at face
    value and returns at half (a return is two journeys). Seasons are
    left out (their per-journey price depends on how often the holder
    travels), as are first class and Standard Premium, non-retail
    placeholders (GBP 999 / 9998 group and tour prices) and anything
    under :data:`_MIN_RETAIL_FARE_GBP`. The median shrugs off what is
    left of the tails.

    Returns
    -------
    float | None
        The median, or ``None`` if the pair has no such fare in the
        feed.
    """
    fares = lookup_pair(table, origin, destination)
    description = fares["ticket_description"].fillna("").str.upper()
    retail = fares[
        (fares["tkt_class"] == "2")
        & fares["tkt_type"].isin(("S", "R"))
        & ~fares["non_retail_fare"].astype(bool)
        & description.str.startswith(_ADULT_PRODUCTS)
        & ~description.str.contains("|".join(_NOT_ADULT_STANDARD))
    ]
    per_journey = retail["fare_gbp"].where(retail["tkt_type"] == "S", retail["fare_gbp"] / 2)
    per_journey = per_journey[per_journey >= _MIN_RETAIL_FARE_GBP]
    return float(per_journey.median()) if not per_journey.empty else None


def _log_summary(fares: pd.DataFrame, origin: str, destination: str) -> None:
    if fares.empty:
        logger.warning("No fares found for %s -> %s.", origin, destination)
        return

    retail = fares[~fares["non_retail_fare"]]
    flagged = len(fares) - len(retail)
    if flagged:
        logger.warning(
            "%d of %d fare(s) for %s -> %s are non-retail placeholders (Group/Tour "
            "Operator/Inclusive Tour/child concession/etc, not real point-to-point "
            "prices); see the 'non_retail_fare' column.",
            flagged,
            len(fares),
            origin,
            destination,
        )
    if retail.empty:
        logger.warning(
            "No standard retail fares for %s -> %s (only non-retail placeholders).",
            origin,
            destination,
        )
        return
    logger.info(
        "%d retail fare(s) for %s -> %s (%.2f - %.2f GBP):\n%s",
        len(retail),
        origin,
        destination,
        retail["fare_gbp"].min(),
        retail["fare_gbp"].max(),
        retail[
            [
                "ticket_code",
                "ticket_description",
                "fare_gbp",
                "restriction_code",
                "toc",
                "route_code",
            ]
        ].to_string(index=False),
    )


def run_fares_lookup(config: FaresLookupConfig) -> None:
    """Build or reuse the fares table and look up every pair.

    Parameters
    ----------
    config : FaresLookupConfig
        Fares-lookup configuration.

    Raises
    ------
    ValueError
        If neither ``raw_feed_path`` nor ``download`` is set.
    """
    config.output_path.mkdir(parents=True, exist_ok=True)

    if config.reuse_existing_clean and _cleaned_files_exist(config.output_path):
        logger.info("Reusing already-cleaned fares feed in %s.", config.output_path)
    else:
        if config.raw_feed_path is not None:
            raw_feed_dir = config.raw_feed_path
        elif config.download:
            raw_feed_dir = fetch_fares_feed(
                config.output_path / "raw", config.username, config.password
            )
        else:
            raise ValueError(
                "Set raw_feed_path to an already-extracted feed directory, or "
                "download: true to fetch it from the National Rail Data Portal."
            )
        clean_fares_feed(raw_feed_dir, config.output_path)

    table = build_fares_table(config.output_path)

    pairs = config.pairs()
    logger.info("Looking up fares for %d station pair(s).", len(pairs))
    for origin, destination in pairs:
        fares = lookup_pair(table, origin, destination)
        _log_summary(fares, origin, destination)
        out_path = config.output_path / f"fares_{origin}_{destination}.csv"
        fares.to_csv(out_path, index=False)
        logger.info(
            "Wrote %d fare(s) for %s -> %s -> %s", len(fares), origin, destination, out_path
        )
