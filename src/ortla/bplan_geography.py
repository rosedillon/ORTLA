"""Network Rail BPLAN Geography and minimum sectional running times.

BPLAN Geography is a tab-delimited extract, one record type per line,
keyed by the record type in the first column:

* ``LOC``: TIPLOC register (name, OSGB easting/northing, STANOX).
* ``NWK``: network links, the physical topology graph between adjacent
  TIPLOCs (running line, direction, distance in metres, route
  availability).
* ``TLK``: timing links, the minimum sectional running time between two
  adjacent TIPLOCs for a given traction/timing-load code, on a given
  running line (``+MM'SS`` format).
* ``TLD``: traction/timing-load definitions (``TLK``'s traction code ->
  max speed, description, category) that ``TLK`` keys off.
* ``PLT``/``REF``: platform and reference-code tables, not used here.

The field layout below was worked out from a sample extract, without the
official interface spec, and checked against known stations: ``LOC``
easting/northing for Euston (529500, 182700) and Kings Cross (530200,
183200) match their OSGB National Grid references, and ``NWK``/``TLK``
values around Leeds and the slow Aberdare branch give plausible
distances and running speeds. It is probably right, but it has not been
checked against the spec.

The file (~1.2M lines) is too large to load whole, so it is streamed
once and only the four record types above are kept. A prefix check skips
the field split for the record types we don't need, which are most of
the file's non-``TLK`` lines.
"""

from __future__ import annotations

import csv
import itertools
import logging
import re
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Callable, Iterator

    from ortla.config import BplanGeographyConfig

logger = logging.getLogger(__name__)

# Record types we keep; everything else (REF, PLT, PIT, PIF headers) is
# skipped by the prefix check in _iter_records before any
# field-splitting.
_KEPT_RECORD_TYPES = ("LOC", "NWK", "TLK", "TLD")

_LOC_HEADER = ["tiploc", "name", "easting", "northing", "stanox"]
_NWK_HEADER = [
    "from_tiploc",
    "to_tiploc",
    "running_line",
    "start_date",
    "end_date",
    "direction",
    "distance_metres",
    "route_availability",
]
_TLK_HEADER = [
    "from_tiploc",
    "to_tiploc",
    "running_line",
    "traction_code",
    "timing_load_speed_mph",
    "entry_speed",
    "exit_speed",
    "start_date",
    "end_date",
    "sectional_running_time_minutes",
]
_TLD_HEADER = ["traction_code", "max_speed_mph", "description", "category"]

_HEADERS = {"LOC": _LOC_HEADER, "NWK": _NWK_HEADER, "TLK": _TLK_HEADER, "TLD": _TLD_HEADER}
_OUTPUT_NAMES = {
    "LOC": "bplan_locations.csv",
    "NWK": "bplan_network_links.csv",
    "TLK": "bplan_timing_links.csv",
    "TLD": "bplan_traction_loads.csv",
}

# Parses a BPLAN sectional-running-time field like "+04'00" -> sign, mm,
# ss.
_SRT_PATTERN = re.compile(r"^([+-]?)(\d+)'(\d+)$")

# Minimum tab-split field count each record type needs for the columns
# we read (the last index used below, plus one).
_MIN_FIELDS = {"LOC": 11, "NWK": 15, "TLK": 14, "TLD": 8}


def parse_srt_minutes(value: str) -> float | None:
    """Minutes in a BPLAN ``+MM'SS`` running time (``"+04'00"`` is 4.0), None if blank."""
    match = _SRT_PATTERN.match(value.strip())
    if not match:
        return None
    sign, minutes, seconds = match.groups()
    total = int(minutes) + int(seconds) / 60
    return -total if sign == "-" else total


def _iter_records(path: pathlib.Path) -> Iterator[tuple[str, list[str]]]:
    """Stream ``(record_type, fields)`` for the record types kept.

    The BPLAN extract mixes several encodings of place names (including
    international TIPLOCs such as Eurostar destinations), so this reads
    as latin-1 to avoid decode errors rather than assuming UTF-8.
    """
    prefixes = tuple(f"{rt}\t" for rt in _KEPT_RECORD_TYPES)
    with path.open(encoding="latin-1") as handle:
        for line in handle:
            if not line.startswith(prefixes):
                continue
            fields = line.rstrip("\n").split("\t")
            yield fields[0], fields


def _loc_row(fields: list[str]) -> dict | None:
    if len(fields) < _MIN_FIELDS["LOC"]:
        return None
    return {
        "tiploc": fields[2].strip().upper(),
        "name": fields[3].strip(),
        "easting": fields[6] or None,
        "northing": fields[7] or None,
        "stanox": fields[10].strip(),
    }


def _nwk_row(fields: list[str]) -> dict | None:
    if len(fields) < _MIN_FIELDS["NWK"]:
        return None
    return {
        "from_tiploc": fields[2].strip().upper(),
        "to_tiploc": fields[3].strip().upper(),
        "running_line": fields[4].strip(),
        "start_date": fields[6],
        "end_date": fields[7],
        "direction": fields[8].strip(),
        "distance_metres": fields[10] or None,
        "route_availability": fields[14].strip(),
    }


def _tlk_row(fields: list[str]) -> dict | None:
    if len(fields) < _MIN_FIELDS["TLK"]:
        return None
    srt = parse_srt_minutes(fields[13])
    if srt is None:
        return None
    return {
        "from_tiploc": fields[2].strip().upper(),
        "to_tiploc": fields[3].strip().upper(),
        "running_line": fields[4].strip(),
        "traction_code": fields[5].strip(),
        "timing_load_speed_mph": fields[7] or None,
        # 0 = starting from / stopping at this end, -1 = passing at line
        # speed.
        "entry_speed": fields[9].strip(),
        "exit_speed": fields[10].strip(),
        "start_date": fields[11],
        "end_date": fields[12],
        "sectional_running_time_minutes": srt,
    }


def _tld_row(fields: list[str]) -> dict | None:
    if len(fields) < _MIN_FIELDS["TLD"]:
        return None
    return {
        "traction_code": fields[2].strip(),
        "max_speed_mph": fields[4] or None,
        "description": fields[6].strip(),
        "category": fields[7].strip(),
    }


_ROW_BUILDERS = {"LOC": _loc_row, "NWK": _nwk_row, "TLK": _tlk_row, "TLD": _tld_row}


def clean_bplan_geography(config: BplanGeographyConfig) -> dict[str, int]:
    """Stream the BPLAN extract into one CSV per record type; return rows written per type."""
    config.output_path.mkdir(parents=True, exist_ok=True)

    writers: dict[str, csv.DictWriter] = {}
    handles = []
    counts = dict.fromkeys(_KEPT_RECORD_TYPES, 0)
    try:
        for record_type in _KEPT_RECORD_TYPES:
            handle = (config.output_path / _OUTPUT_NAMES[record_type]).open(
                "w", newline="", encoding="utf-8"
            )
            handles.append(handle)
            writer = csv.DictWriter(handle, fieldnames=_HEADERS[record_type])
            writer.writeheader()
            writers[record_type] = writer

        logger.info("Streaming BPLAN Geography from %s ...", config.geography_path)
        n_lines = 0
        for record_type, fields in _iter_records(config.geography_path):
            n_lines += 1
            row = _ROW_BUILDERS[record_type](fields)
            if row is None:
                continue
            writers[record_type].writerow(row)
            counts[record_type] += 1
            if n_lines % 500_000 == 0:
                logger.info("Processed %d lines ...", n_lines)
    finally:
        for handle in handles:
            handle.close()

    logger.info(
        "BPLAN Geography cleaned: %s",
        ", ".join(f"{rt}={n}" for rt, n in counts.items()),
    )
    return counts


# BPLAN traction codes that are not trains on Network Rail track (see
# bplan_traction_loads.csv): ships, Underground and Tyne & Wear Metro
# stock. Every code starting "BUS" is a bus.
_NON_RAIL_TRACTION = frozenset({"SHIP", "HSS", "SEALYX", "LUL", "20", "TWM"})


class BplanTimingGraph:
    """Minimum running time between any two TIPLOCs, by traction.

    Built once from the cleaned ``bplan_timing_links.csv`` (see
    :func:`clean_bplan_geography`) and reused for many lookups, each a
    Dijkstra search, as in
    :class:`~ortla.track_distance.TrackDistanceCalculator`.

    Where several timing-link records exist for the same (from, to,
    traction), from different running lines or superseded historical
    variants, the fastest is kept, since we want the minimum achievable
    time: if any recorded variant did it faster, this data alone cannot
    show that a candidate timed at or above that is infeasible.
    """

    def __init__(self, bplan_output_path: pathlib.Path) -> None:
        """Load ``bplan_timing_links.csv`` from a ``bplan`` run and build the directed graph."""
        path = bplan_output_path / "bplan_timing_links.csv"
        logger.info("Loading BPLAN timing links from %s ...", path)
        links = pd.read_csv(path, dtype=str)
        # Plain assignment, not .loc: .loc keeps the str column's object
        # dtype, which sends the group-by minimums below to pure Python
        # (~14 s).
        links["sectional_running_time_minutes"] = pd.to_numeric(
            links["sectional_running_time_minutes"], errors="coerce"
        )
        links = links.dropna(subset=["sectional_running_time_minutes"])
        # Bus, ship, Underground and Metro timings are not trains on
        # Network Rail track: a bus link Kenton -> Preston Park put
        # Harrow 19.5 minutes from Brighton for any traction without a
        # timing of its own.
        code = links["traction_code"].fillna("").str.strip().str.upper()
        links = links[~(code.str.startswith("BUS") | code.isin(_NON_RAIL_TRACTION))]

        # Each link has timings for passing (-1) or starting/stopping
        # (0) at either end; older outputs without the flags are all
        # treated as passing.
        if "entry_speed" not in links.columns:
            links["entry_speed"] = "-1"
            links["exit_speed"] = "-1"
        links["starts"] = links["entry_speed"].astype(str).str.strip().eq("0")
        links["stops"] = links["exit_speed"].astype(str).str.strip().eq("0")

        # Fastest time per (link, timing kind, traction) in one
        # group-by: a group-by per link took ~25 s, a fixed cost on
        # every propose/hindcast. "srt" is the passing time, or the
        # fastest of any kind on a link that has no passing timing.
        links["kind"] = np.select(
            [
                ~links["starts"] & ~links["stops"],
                links["starts"] & ~links["stops"],
                ~links["starts"] & links["stops"],
            ],
            ["srt", "srt_start", "srt_stop"],
            default="srt_start_stop",
        )
        srt = "sectional_running_time_minutes"
        by_kind = ["from_tiploc", "to_tiploc", "kind", "traction_code"]
        fastest = links.groupby(by_kind)[srt].min()
        any_kind = links.groupby(["from_tiploc", "to_tiploc", "traction_code"])[srt].min()
        edges: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
        for (origin, dest, kind, traction), minutes in fastest.items():
            edges.setdefault(
                (origin, dest),
                {"srt": {}, "srt_start": {}, "srt_stop": {}, "srt_start_stop": {}},
            )[kind][traction] = minutes
        no_passing = {pair for pair, timings in edges.items() if not timings["srt"]}
        for (origin, dest, traction), minutes in any_kind.items():
            if (origin, dest) in no_passing:
                edges[(origin, dest)]["srt"][traction] = minutes
        self.graph = nx.DiGraph()
        for (origin, dest), timings in sorted(edges.items()):
            self.graph.add_edge(origin, dest, **timings)

        logger.info(
            "Built BPLAN timing graph: %d TIPLOCs, %d directed links.",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )

    @staticmethod
    def _weight_for(traction_code: str) -> Callable[[str, str, dict], float | None]:
        """Edge weight: the traction's own timing, else the fastest."""

        def _weight(_u: str, _v: str, data: dict) -> float | None:
            srt = data.get("srt", {})
            if traction_code in srt:
                return srt[traction_code]
            return min(srt.values()) if srt else None

        return _weight

    def minimum_running_time(
        self,
        from_tiploc: str,
        to_tiploc: str,
        traction_code: str,
    ) -> float | None:
        """Minimum running time in minutes between two TIPLOCs, None if no route.

        Sums sectional running times along the fastest path, since consecutive calls are often not
        adjacent in the graph (non-stop timing points lie between). A link with no time for
        ``traction_code`` (BPLAN's traction or timing-load code, ``"150"``, ``"185"``) takes its
        fastest recorded one.
        """
        return self.running_time_detail(from_tiploc, to_tiploc, traction_code)[0]

    def fastest_path(
        self,
        from_tiploc: str,
        to_tiploc: str,
        traction_code: str,
    ) -> list[str] | None:
        """TIPLOCs on the fastest route, ends included, or ``None``."""
        from_tiploc = from_tiploc.strip().upper()
        to_tiploc = to_tiploc.strip().upper()
        if from_tiploc not in self.graph or to_tiploc not in self.graph:
            return None
        try:
            return nx.dijkstra_path(
                self.graph,
                from_tiploc,
                to_tiploc,
                weight=self._weight_for(traction_code),
            )
        except nx.NetworkXNoPath:
            return None

    @staticmethod
    def _timing(data: dict, kind: str, traction_code: str) -> tuple[float | None, bool]:
        """One link's timing of ``kind``: ``(minutes, exact_match)``.

        Falls back to the passing timing when the link has no start/stop
        variant, and to the fastest traction when the code has no entry.
        """
        for key in (kind, "srt"):
            timings = data.get(key) or {}
            if traction_code in timings:
                return timings[traction_code], True
        timings = data.get(kind) or data.get("srt") or {}
        return (min(timings.values()), False) if timings else (None, False)

    def running_time_detail(
        self,
        from_tiploc: str,
        to_tiploc: str,
        traction_code: str,
    ) -> tuple[float | None, float | None]:
        """Stop-to-stop running time and the share timed exactly.

        The route is the fastest passing path; its first link then takes
        the start-from-stand timing and its last link the stop timing (a
        single link takes the start-and-stop timing), since a calling
        train must accelerate away and brake to a stand. Returns
        ``(minutes, exact_share)`` where ``exact_share`` is the fraction
        of the minutes taken from ``traction_code``'s own timings rather
        than the fastest-traction fallback.
        """
        path = self.fastest_path(from_tiploc, to_tiploc, traction_code)
        if path is None:
            return None, None
        edges = list(itertools.pairwise(path))
        if not edges:
            return 0.0, None
        total = exact = 0.0
        for idx, (u, v) in enumerate(edges):
            first, last = idx == 0, idx == len(edges) - 1
            kind = (
                "srt_start_stop"
                if first and last
                else "srt_start"
                if first
                else "srt_stop"
                if last
                else "srt"
            )
            minutes, matched = self._timing(self.graph[u][v], kind, traction_code)
            if minutes is None:
                return None, None
            total += minutes
            exact += minutes if matched else 0.0
        return total, (exact / total if total else None)


class BplanNetworkLinks:
    """Distinct running lines per TIPLOC pair, from the network links.

    ``NWK`` records the physical topology (see the module docstring),
    and more than one distinct ``running_line`` value recorded between
    the same TIPLOC pair is the closest proxy this pipeline has for
    "this segment is multi-track". It is not a platform count (no such
    field is parsed; see ``PLT`` in the module docstring), and NWK's
    running-line labelling has not been checked against the spec either.
    Use it as context when reviewing a flagged headway conflict, not as
    a reason to suppress the conflict automatically.
    """

    def __init__(self, bplan_output_path: pathlib.Path) -> None:
        """Load ``bplan_network_links.csv`` from a ``bplan`` run and count lines per pair."""
        path = bplan_output_path / "bplan_network_links.csv"
        logger.info("Loading BPLAN network links from %s ...", path)
        links = pd.read_csv(path, dtype=str)
        links = links.dropna(subset=["from_tiploc", "to_tiploc", "running_line"])
        self._counts = (
            links.groupby(["from_tiploc", "to_tiploc"])["running_line"].nunique().to_dict()
        )

    def running_line_count(self, from_tiploc: str, to_tiploc: str) -> int | None:
        """Distinct running lines on the directed segment, None if the pair is not in the data."""
        key = (from_tiploc.strip().upper(), to_tiploc.strip().upper())
        return self._counts.get(key)


def run_bplan(config: BplanGeographyConfig) -> None:
    """Clean the BPLAN geography extract and log a summary."""
    counts = clean_bplan_geography(config)
    if not counts.get("TLK"):
        logger.warning("No timing links parsed; BplanTimingGraph would be empty.")
        return

    graph = BplanTimingGraph(config.output_path)
    logger.info(
        "BPLAN Geography ready: %d locations, %d network links, %d timing links, "
        "%d traction definitions -> timing graph with %d TIPLOCs / %d directed links.",
        counts["LOC"],
        counts["NWK"],
        counts["TLK"],
        counts["TLD"],
        graph.graph.number_of_nodes(),
        graph.graph.number_of_edges(),
    )
