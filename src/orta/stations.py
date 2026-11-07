"""Station codes, station lists and station footfall.

A handful of stations carry two CRS codes: one for their domestic
platforms and one for the international or high-speed platforms beside
them. ``stations.csv`` lists both as separate rows, the CIF timetable
books high-speed services under the second code, and the ODM matrices
and ORR table 1410 only ever use the first. Nothing in the feeds maps
between them, so a leg into St Pancras booked as ``SPX`` finds no
``STP`` column in the matrices, falls through to the synthetic estimate
and loses the real flow.

Everything that takes a station code from the timetable into the demand
data (the period matrices, table 1410, the fares feed) canonicalises it
here first. The codes in :data:`DEFAULT_STATION_RECODES` are also
rewritten in the timetable, and in the count feeds matched to it, as
soon as they are read.

This module also reads the station lists (``stations.csv`` and the
headerless list of names and codes) and ORR's table 1410 of entries and
exits.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Iterable

logger = logging.getLogger(__name__)


# Alias -> canonical CRS. The alias is the code the CIF timetable uses;
# the canonical is the code the ODM and table 1410 use for the same
# platform group.
DEFAULT_STATION_ALIASES: dict[str, str] = {
    "SPX": "STP",  # London St Pancras International (HS1)
    "SPL": "STP",  # St Pancras Low Level (Thameslink box)
    "SDI": "SFA",  # Stratford International (international platforms)
    "EBF": "EBD",  # Ebbsfleet International (international platforms)
    # Merseyrail and London Overground calls the CIF books under a
    # second code. stations.csv maps each TIPLOC to the canonical code;
    # without these a Merseyrail train from Headbolt Lane appeared to
    # stop at a station with no footfall.
    "XMR": "MRF",  # Moorfields (TIPLOC MORFNL)
    "XLV": "LVC",  # Liverpool Central (TIPLOC LVRPLCL)
    "WJL": "WIJ",  # Willesden Junction low level (TIPLOC WLSDNJL)
}


# Second codes the CIF timetable gives some platforms of a station the
# ODM holds under one code. Unlike the aliases above, these are
# rewritten in the timetable, and in any count feed matched to it, as
# soon as it is read: trains at both codes serve the same pairs, so they
# have to be counted as one station's trains when a slot's demand is
# shared among them, not only looked up under one code. Found by the
# national base-year check (case 16), each confirmed by its neighbouring
# calls; recoding them took held-out rmse(log) from 0.906 to 0.898
# there.
DEFAULT_STATION_RECODES: dict[str, str] = {
    # The Elizabeth line's central calls.
    "WHX": "ZLW",  # Whitechapel
    "LSX": "LST",  # Liverpool Street
    "PDX": "PAD",  # Paddington
    "FDX": "ZFD",  # Farringdon
    "ABX": "ABW",  # Abbey Wood
    "RDZ": "RDG",  # Reading platforms 4A/B (RDNG4AB): SWR, GWR North Downs
    "LVL": "LIV",  # Liverpool Lime Street low level (LVRPLSL)
    "ALE": "LPY",  # Liverpool South Parkway high level (ALERTN)
    "GQL": "GLQ",  # Glasgow Queen Street low level (GLGQLL)
    "GCL": "GLC",  # Glasgow Central low level (GLGCLL)
    "WJH": "WIJ",  # Willesden Junction high level (WLSDJHL)
    "XHZ": "HHY",  # Highbury & Islington, East London line (HIGHBYE)
    "HII": "HHY",  # Highbury & Islington, North London line (HIGHBYA)
    "XIC": "CNN",  # Canonbury, East London line (CNNBELL)
    "XGQ": "SGB",  # Smethwick Galton Bridge high level (GALTILL)
    "XUP": "UPM",  # Upminster platform 6 (UPMNSP6)
    "TAH": "TAM",  # Tamworth high level (TMWTHHL)
    "LIF": "LTV",  # Lichfield Trent Valley high level (LCHTTVH)
    "XRO": "RET",  # Retford low level (RTFDLL)
    "WPH": "WOP",  # Worcestershire Parkway high level (WORCPHL)
    # The aliases above, recoded too (2026-10-20): as lookups only, a
    # Thameslink train at SPL and an EMR train at STP each took the
    # whole Bedford to St Pancras cell.
    **DEFAULT_STATION_ALIASES,
}


# Destination sets whose members compete for the same trips, so a new
# direct service to one of them abstracts from the others rather than
# only growing its own O-D cell (see `demand.SubstitutionContext`).
# Keyed by a label used only in logs and the demand output.
DEFAULT_DESTINATION_GROUPS: dict[str, list[str]] = {
    "london_terminals": [
        "BFR",  # Blackfriars
        "CHX",  # Charing Cross
        "CST",  # Cannon Street
        "EUS",  # Euston
        "FST",  # Fenchurch Street
        "KGX",  # King's Cross
        "LBG",  # London Bridge
        "LST",  # Liverpool Street
        "MYB",  # Marylebone
        "PAD",  # Paddington
        "STP",  # St Pancras
        "VIC",  # Victoria
        "WAT",  # Waterloo
        "ZFD",  # Farringdon
        "EPH",  # Elephant & Castle
        "CTK",  # City Thameslink
        "MOG",  # Moorgate
        "OLD",  # Old Street
        "VXH",  # Vauxhall
    ],
}


# Terminals whose trains count as London-bound for the timing weights
# (case 17 common.LONDON).
LONDON_TERMINALS = frozenset(
    {
        "BFR",
        "CHX",
        "CST",
        "CTK",
        "EPH",
        "EUS",
        "FST",
        "KGX",
        "LBG",
        "LST",
        "MYB",
        "MOG",
        "OLD",
        "PAD",
        "STP",
        "VIC",
        "VXH",
        "WAT",
        "WAE",
        "ZFD",
        "SPX",
        "SPL",
    }
)


def resolve_aliases(configured: dict[str, str] | None) -> dict[str, str]:
    """Upper-case an alias map, or return the default aliases.

    ``None`` takes the defaults; a supplied map replaces them outright,
    so a run can drop the built-ins with ``station_aliases: {}``. Codes
    are upper-cased and trimmed, an entry mapping a code to itself is
    dropped, and chains (``A -> B -> C``) are collapsed so every alias
    resolves in one step.
    """
    source = DEFAULT_STATION_ALIASES if configured is None else configured
    pairs = {
        str(alias).strip().upper(): str(canonical).strip().upper()
        for alias, canonical in source.items()
        if str(alias).strip() and str(canonical).strip()
    }
    resolved: dict[str, str] = {}
    for alias, first in pairs.items():
        canonical, seen = first, {alias}
        while canonical in pairs and canonical not in seen:
            seen.add(canonical)
            canonical = pairs[canonical]
        if canonical != alias:
            resolved[alias] = canonical
    return resolved


def resolve_recodes(configured: dict[str, str] | None) -> dict[str, str]:
    """Upper-case a recode map, or return the default recodes.

    As :func:`resolve_aliases`: ``None`` takes the defaults and ``{}``
    turns recoding off.
    """
    return resolve_aliases(DEFAULT_STATION_RECODES if configured is None else configured)


def recode_columns(
    frame: pd.DataFrame, recodes: dict[str, str], columns: Iterable[str]
) -> pd.DataFrame:
    """``frame`` with second codes in ``columns`` recoded."""
    present = [column for column in columns if column in frame.columns]
    if not recodes or not present:
        return frame
    return frame.assign(**{column: frame[column].replace(recodes) for column in present})


def canonical_crs(code: str | None, aliases: dict[str, str]) -> str:
    """Return the code that ``code``'s demand is held under.

    The result is trimmed and upper-cased.
    """
    if code is None:
        return ""
    normalized = str(code).strip().upper()
    return aliases.get(normalized, normalized)


def with_alias_keys(mapping: dict[str, int], aliases: dict[str, str]) -> dict[str, int]:
    """Copy a CRS mapping, giving each alias its station's value.

    Used for lookups that index by whatever code the timetable supplied
    (table 1410 footfall, most usefully), so their call sites need no
    alias handling of their own. An alias already present in ``mapping``
    keeps its own value.
    """
    expanded = dict(mapping)
    for alias, canonical in aliases.items():
        if alias not in expanded and canonical in expanded:
            expanded[alias] = expanded[canonical]
    return expanded


def load_table_1410(path: pathlib.Path) -> dict[str, int]:
    """Load table 1410 and return a TLC -> entries/exits mapping.

    Parameters
    ----------
    path : pathlib.Path
        Path to the table 1410 CSV.

    Returns
    -------
    dict[str, int]
        Mapping of station TLC to total entries and exits.
    """
    logger.info("Loading table 1410 from %s ...", path)

    df = pd.read_csv(path)

    tlc_col = "Three Letter Code\n(TLC)"
    entries_col = "Entries and exits - All tickets"

    # Whole-column assignment, not ``df.loc[:, col] = ...``: the
    # per-year files read their entries as int64, and writing strings
    # into it in place is deprecated (a FutureWarning on every read, an
    # error in a later pandas).
    df[tlc_col] = df[tlc_col].str.strip().str.upper()
    df[entries_col] = df[entries_col].astype(str).str.replace(",", "", regex=False)
    df[entries_col] = pd.to_numeric(df[entries_col], errors="coerce")
    df = df.dropna(subset=[tlc_col, entries_col])

    station_entries = dict(zip(df[tlc_col], df[entries_col].astype(int), strict=True))
    logger.info("Loaded entries/exits for %d stations.", len(station_entries))
    return station_entries


def read_station_list(path: pathlib.Path) -> tuple[set[str], dict[str, str]]:
    """Read a station list's CRS codes and TIPLOC -> CRS pairs.

    Two layouts are read. ``stations.csv`` has a header with a
    ``crsCode`` column and, usually, a ``tiploc`` column; the headerless
    list has a station name and a code on each row.

    Parameters
    ----------
    path : pathlib.Path
        The station list.

    Returns
    -------
    tuple[set[str], dict[str, str]]
        The upper-cased CRS codes, and the TIPLOC -> CRS map (empty for
        a list with no ``tiploc`` column). Rows with no code are left
        out of both.
    """
    logger.info("Loading the station list from %s ...", path)
    frame = pd.read_csv(path, dtype=str)
    if "crsCode" not in frame.columns:
        frame = pd.read_csv(path, header=None, names=["stationName", "crsCode"], dtype=str)
    crs = frame["crsCode"].str.strip().str.upper()
    codes = set(crs.dropna()) - {""}
    tiploc_to_crs: dict[str, str] = {}
    if "tiploc" in frame.columns:
        tiploc = frame["tiploc"].str.strip().str.upper()
        usable = crs.notna() & crs.ne("") & tiploc.notna() & tiploc.ne("")
        tiploc_to_crs = dict(zip(tiploc[usable], crs[usable], strict=True))
    logger.info("Loaded %d stations and %d TIPLOC mappings.", len(codes), len(tiploc_to_crs))
    return codes, tiploc_to_crs


def tiploc_to_crs_map(
    station_list: pathlib.Path | None, calls_path: pathlib.Path | None = None
) -> dict[str, str]:
    """TIPLOC -> CRS from the timetable's calls and the station list.

    The station list names one TIPLOC per station; the timetable knows
    every TIPLOC a station is booked under. Where both name a TIPLOC the
    station list wins.

    Parameters
    ----------
    station_list : pathlib.Path | None
        A ``stations.csv``-style list (see :func:`read_station_list`).
    calls_path : pathlib.Path | None, optional
        A ``timetable_calls.csv`` from the ``timetable`` command.

    Returns
    -------
    dict[str, str]
        Upper-cased TIPLOC -> CRS.
    """
    mapping: dict[str, str] = {}
    if calls_path is not None:
        pairs = (
            pd.read_csv(calls_path, usecols=["tiploc", "crs"], dtype=str)
            .dropna()
            .drop_duplicates()
        )
        mapping = dict(zip(pairs["tiploc"], pairs["crs"], strict=True))
    if station_list is not None:
        mapping.update(read_station_list(station_list)[1])
    return mapping
