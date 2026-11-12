"""Read every operator's loading counts into one schema.

Operators publish their loading data (through the Rail Data Marketplace)
in different layouts. There is one reader per layout, and each returns
rows of :data:`CANONICAL_COLUMNS`, one row per (train, day of week,
calling station, count period), so the rest of the model never needs to
know which operator a row came from.

Two things differ across operators:

* Boardings against load. Northern and TPE report passengers on and off
  at each stop; everyone else reports only the load on the train at each
  stop. For those, boardings and alightings are derived from the change
  in load between consecutive calls (``on = max(load_i - load_{i-1},
  0)``). That is a net figure: passengers swapping at the same stop
  cancel out, so derived boardings are a lower bound on true boardings.
* CrossCountry is not read: its published feed is a busyness figure from
  0 to 100, not a passenger count.

Readers are registered in :data:`READERS`, keyed by the operator names
used in ``LoadingConfig.loading_paths``.

Departure times
---------------
Operators publish the GBTT departure of the call a count was taken at,
except Northern, which publishes only the train's departure from its
origin, and Avanti and GTR, which publish the call's scheduled arrival.
Classifying a Northern row by the origin departure times it a median 29
minutes early and puts about 15% of its calls in the wrong time period.

:func:`add_resolved_departures` fills the gap from the CIF timetable,
which carries the same GBTT departure the other operators publish, so an
imputed value is that quantity and not a proxy for it. A loading row is
matched to its schedule through a ladder of progressively looser keys,
and every row records the rung it matched on, so the imputation can be
audited, reported or a rung excluded.

A count can only match the timetable that was running when it was taken.
The counts span April 2024 to August 2026, six published timetables, so
resolving them all against one extract fails for every row whose train
was retimed or withdrawn since, which is most of them.
``timetable_eras`` supplies one calls table per published timetable with
the dates it was in force, and each count is matched against its own
era; rows outside every window fall back to the single extract.
"""

from __future__ import annotations

import datetime
import functools
import hashlib
import logging
import pathlib
import re
from collections.abc import Callable
from typing import TYPE_CHECKING

import pandas as pd
from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from ortla.config import TimetableEra

logger = logging.getLogger(__name__)


CANONICAL_COLUMNS = [
    "Train Operator",
    "Unique ID",
    "Headcode",
    "Service Week Day",
    "Origin CRS",
    "Destination CRS",
    "Origin Departure Time",
    "Departure from CRS",
    "Departure Time",
    "Call Sequence",
    "RSP Period",
    "Period Start Date",
    "Period End Date",
    "Sample Size",
    "Mean Capacity",
    "Mean Load on Departure",
    "Mean Passengers On",
    "Mean Passengers Off",
    "load_units",
]


# Bump an operator's entry when its reader's output changes, so its
# cached standardised frame is rebuilt.
READER_VERSION = {"gtr": 2, "swr": 4, "southeastern": 4, "greater_anglia": 4, "emr": 4, "tpe": 4}


_DEFAULT_READER_VERSION = 3


# Readers that map TIPLOCs to CRS codes. Only their caches depend on the
# map: the loading and hindcast commands build different maps, and
# keying every operator on it made the two re-parse GTR's 3.7 GB of XLSX
# on every run.
_USES_TIPLOC_MAP = {"emr", "swr"}


_MAX_LOAD = 3000


_DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


_DAY_BY_PREFIX = {d[:3].lower(): d for d in _DAY_NAMES}


# Identifies one run of one train within one count period, for deriving
# boardings from load changes along its calling pattern.
_INSTANCE_KEY = [
    "Train Operator",
    "Unique ID",
    "Headcode",
    "Service Week Day",
    "Origin CRS",
    "Destination CRS",
    "Origin Departure Time",
    "Period Start Date",
]


# Small normalisers
def _clean_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Strip non-breaking and repeated spaces from column names."""
    df.columns = [re.sub(r"\s+", " ", str(c).replace("\xa0", " ")).strip() for c in df.columns]
    return df


def _crs(series: pd.Series) -> pd.Series:
    out = series.astype("string").str.strip().str.upper()
    # Not "NAN": that is Nantwich's code. A missing cell is already <NA>
    # here.
    return out.mask(out.isin(["", "NONE", "<NA>"]))


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype("string").str.replace(",", "", regex=False).str.rstrip("%"),
        errors="coerce",
    )


def _day(series: pd.Series) -> pd.Series:
    """Map any day spelling ("Mon", "1-Sun") to its full name."""
    text = series.astype("string").str.strip().str.lower().str.replace(r"^\d+-", "", regex=True)
    return text.str[:3].map(_DAY_BY_PREFIX)


def _time(series: pd.Series) -> pd.Series:
    """Write clock times as ``HH:MM:SS``.

    Accepts "6:02", "06:02:00" and ``datetime.time`` values.
    """

    def _one(value: object) -> str | None:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        if isinstance(value, datetime.time | datetime.datetime | pd.Timestamp):
            return value.strftime("%H:%M:%S")
        text = str(value).strip()
        match = re.match(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?", text)
        if not match:
            return None
        return f"{int(match.group(1)) % 24:02d}:{match.group(2)}:{match.group(3) or '00'}"

    return series.map(_one).astype("string")


def _date(series: pd.Series, dayfirst: bool = False) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", dayfirst=dayfirst).dt.normalize()


def _canonical(df: pd.DataFrame) -> pd.DataFrame:
    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[CANONICAL_COLUMNS]


def derive_on_off(df: pd.DataFrame) -> pd.DataFrame:
    """Fill boardings and alightings from changes in load.

    Only rows whose ``Mean Passengers On`` is missing are filled, so
    operators that report real boardings keep them. Calls are ordered by
    ``Call Sequence``. A call at the service's destination is the
    terminus (everyone alights, departing load 0); operators that count
    departures only often omit that row, in which case alightings at the
    terminus stay unknown.
    """
    need = df["Mean Passengers On"].isna() & df["Mean Load on Departure"].notna()
    ordered = df["Call Sequence"].notna()
    if not (need & ordered).any():
        return df
    if not ordered.all():
        # Calls with no known order can't be differenced; leave them as
        # is.
        return pd.concat([derive_on_off(df[ordered]), df[~ordered]])
    df = df.sort_values([*_INSTANCE_KEY, "Call Sequence"], kind="stable")
    key = [df[c].astype("string").fillna("") for c in _INSTANCE_KEY]
    grouped = df.groupby(key, sort=False)["Mean Load on Departure"]
    is_first = grouped.cumcount().eq(0)
    is_terminus = grouped.cumcount(ascending=False).eq(0) & df["Departure from CRS"].eq(
        df["Destination CRS"]
    ).fillna(False)
    load = df["Mean Load on Departure"].astype(float).mask(is_terminus, 0.0)
    # Compare with the last *counted* call; before any counted call the
    # train's starting load is unknown, so boardings stay unknown too.
    last_known = load.groupby(key, sort=False).ffill()
    previous = last_known.groupby(key, sort=False).shift(1).mask(is_first, 0.0)
    delta = load - previous
    fill = df["Mean Passengers On"].isna()
    df.loc[fill, "Mean Passengers On"] = delta.clip(lower=0)[fill]
    df.loc[fill, "Mean Passengers Off"] = (-delta).clip(lower=0)[fill]
    return df


def _read_csv_any_encoding(file: pathlib.Path) -> pd.DataFrame:
    """Read a CSV that may be UTF-8, with or without BOM, or cp1252."""
    try:
        return pd.read_csv(file, dtype=str, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(file, dtype=str, encoding="cp1252", low_memory=False)


def _files(path: pathlib.Path, patterns: tuple[str, ...]) -> list[pathlib.Path]:
    if path.is_file():
        return [path]
    found = sorted({p for pattern in patterns for p in path.glob(pattern)})
    return [p for p in found if not p.name.startswith(("~$", "."))]


# Readers, one per published layout
def read_northern(path: pathlib.Path, **_: object) -> pd.DataFrame:
    """Northern: period CSVs with passengers on, off and load."""
    chunks = []
    # The folder also holds formation/capacity tables; only period files
    # are loading data.
    for file in tqdm(_files(path, ("Loading_Data_*.csv",)), desc="Northern"):
        raw = pd.read_csv(file, dtype=str, low_memory=False)
        df = pd.DataFrame(
            {
                "Train Operator": "northern",
                # No UID is published; the timetable tag separates the
                # same headcode running under two timetables within one
                # period.
                "Unique ID": raw["Timetable"].astype("string"),
                "Headcode": raw["Headcode"],
                "Service Week Day": _day(raw["Service Week Day"]),
                "Origin CRS": _crs(raw["Origin CRS"]),
                "Destination CRS": _crs(raw["Destination CRS"]),
                "Origin Departure Time": _time(raw["Origin Departure Time"]),
                "Departure from CRS": _crs(raw["Departure from CRS"]),
                "RSP Period": raw["RSP Period"],
                "Period Start Date": _date(raw["Period Start Date"]),
                "Period End Date": _date(raw["Period End Date"]),
                "Sample Size": _num(raw["Sample Size"]),
                "Mean Capacity": _num(raw["Mean Capacity"]),
                "Mean Load on Departure": _num(raw["Mean Load on Departure"]),
                "Mean Passengers On": _num(raw["Mean Passengers On"]),
                "Mean Passengers Off": _num(raw["Mean Passengers Off"]),
                "load_units": "passengers",
            }
        )
        chunks.append(_canonical(df))
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def _each_file(
    path: pathlib.Path, read_one: Callable[..., pd.DataFrame], **kw: object
) -> pd.DataFrame:
    """Read every CSV in ``path``, numbering the files as periods."""
    frames = [read_one(f, str(i), **kw) for i, f in enumerate(_files(path, ("*.csv",)), 1)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def read_tpe(path: pathlib.Path, **_: object) -> pd.DataFrame:
    """TransPennine Express: one CSV per period, on, off and load."""
    return _each_file(path, _read_tpe_file)


def _read_tpe_file(path: pathlib.Path, period: str) -> pd.DataFrame:
    raw = _clean_headers(pd.read_csv(path, dtype=str, low_memory=False))
    df = pd.DataFrame(
        {
            "Train Operator": "tpe",
            "Unique ID": raw.get("RSID"),
            "Headcode": raw["Headcode"],
            "Service Week Day": _day(raw["Service Week Day"]),
            "Origin CRS": _crs(raw["Origin CRS"]),
            "Destination CRS": _crs(raw["Destination CRS"]),
            "Origin Departure Time": _time(raw["Origin Departure Time"]),
            "Departure from CRS": _crs(raw["Departure from CRS"]),
            "Departure Time": _time(raw["Departure from GBTT time"]),
            "RSP Period": period,
            "Sample Size": _num(raw["Sample Size"]),
            "Mean Load on Departure": _num(raw["Mean Occupancy"]),
            "Mean Passengers On": _num(raw["Mean Passenger On"]),
            "Mean Passengers Off": _num(raw["Mean Passenger Off"]),
            "load_units": "passengers",
        }
    )
    df = _drop_forks(df, _crs(raw["Next station CRS"]))
    df["Call Sequence"] = _order_by_time(df)
    return _canonical(df)


def _drop_forks(df: pd.DataFrame, next_crs: pd.Series) -> pd.DataFrame:
    """Drop the paths a TPE train did not mostly take.

    The feed can list two paths for one train: rows leaving the same
    station at the same time for two different next stations, the
    second (a diversion) with a load averaged over the few runs that
    took it. At each such fork the busier row is kept, then rows at
    stations reached only through a dropped row are dropped too.
    """
    key = ["Unique ID", "Service Week Day", "Origin Departure Time"]
    df = df.assign(_next=next_crs.to_numpy())
    at = [*key, "Departure from CRS", "Departure Time"]
    load = df["Mean Load on Departure"].fillna(-1)
    fork = df.groupby(at, dropna=False)["_next"].transform("nunique") > 1
    keep = ~fork | (load == load.groupby([df[c] for c in at], dropna=False).transform("max"))
    train = df[key].astype(str).agg("|".join, axis=1)
    here = train + "|" + df["Departure from CRS"].astype(str)
    while True:
        reached = set(train[keep] + "|" + df.loc[keep, "_next"].astype(str))
        reached |= set(train + "|" + df["Origin CRS"].astype(str))
        new = keep & here.isin(reached)
        if new.equals(keep):
            break
        keep = new
    return df[keep].drop(columns="_next")


def _order_by_time(df: pd.DataFrame) -> pd.Series:
    """Call order from station times, unwrapping past midnight."""
    origin = pd.to_timedelta(df["Origin Departure Time"].fillna("00:00:00"), errors="coerce")
    station = pd.to_timedelta(df["Departure Time"], errors="coerce")
    minutes = (station - origin).dt.total_seconds() / 60
    return minutes.where(minutes >= 0, minutes + 24 * 60)


def read_awc(path: pathlib.Path, **_: object) -> pd.DataFrame:
    """Avanti West Coast: one CSV of mean loads per period."""
    return _each_file(path, _read_awc_file)


def _read_awc_file(path: pathlib.Path, period: str) -> pd.DataFrame:
    raw = _clean_headers(pd.read_csv(path, dtype=str, low_memory=False))
    df = pd.DataFrame(
        {
            "Train Operator": "awc",
            "Unique ID": raw["Unique ID"],
            "Headcode": raw["Train ID (Headcode)"],
            "Service Week Day": _day(raw["Day of Count"]),
            "Origin CRS": _crs(raw["Origin of Service (CRS)"]),
            "Destination CRS": _crs(raw["Destination of Service (CRS)"]),
            "Origin Departure Time": _time(raw["Timetabled Departure from Origin"]),
            "Departure from CRS": _crs(raw["Station CRS"]),
            "Departure Time": _time(raw["Station Scheduled Arrival Time"]),
            "RSP Period": period,
            "Sample Size": _num(raw["Number of Counts"]),
            "Mean Capacity": _num(raw["Standard Seated Capacity"])
            + _num(raw["First Class Seated Capacity"]).fillna(0),
            "Mean Load on Departure": _num(raw["Total Load (Mean)"]),
            "load_units": "passengers",
        }
    )
    df["Call Sequence"] = _order_by_time(df)
    return _canonical(df)


def read_emr(
    path: pathlib.Path, tiploc_to_crs: dict[str, str] | None = None, **_: object
) -> pd.DataFrame:
    """East Midlands Railway: dated CSVs with ends as TIPLOCs or names.

    Files up to May 2026 give TIPLOCs; later ones give names.
    """
    return _each_file(path, _read_emr_file, tiploc_to_crs=tiploc_to_crs or {})


def _read_emr_file(
    path: pathlib.Path, _period: str, tiploc_to_crs: dict[str, str]
) -> pd.DataFrame:
    raw = _clean_headers(pd.read_csv(path, dtype=str, low_memory=False))

    # From August 2026 the ends are station names ("London St.
    # Pancras"); match them to the file's own counted stations, exactly
    # or by unique prefix, after dropping punctuation.
    def key(s: pd.Series) -> pd.Series:
        return s.fillna("").str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)

    names = dict(zip(key(raw["StationName"]), raw["StationCRS"], strict=True))

    def by_name(name: str) -> str | None:
        hits = {crs for k, crs in names.items() if name and k.startswith(name)}
        return names.get(name) or (hits.pop() if len(hits) == 1 else None)

    def end_crs(col: str) -> pd.Series:
        tiploc = raw[col].str.strip().str.upper().map(tiploc_to_crs)
        return _crs(tiploc.fillna(key(raw[col]).map(by_name)))

    df = pd.DataFrame(
        {
            "Train Operator": "emr",
            "Unique ID": raw["UniqueID"],
            "Headcode": raw["TrainID"],
            "Service Week Day": _day(raw["DayOfCount"]),
            "Origin CRS": end_crs("Origin"),
            "Destination CRS": end_crs("Destination"),
            "Origin Departure Time": _time(raw["TimetabledDepartureTimeFromOrigin"]),
            "Departure from CRS": _crs(raw["StationCRS"]),
            "Departure Time": _time(
                raw["StationScheduledDepartureTime"].fillna(raw["StationScheduledArrivalTime"])
            ),
            "Call Sequence": _num(raw["CallingPointNumber"]),
            "RSP Period": raw["StartOfCountPeriod"],
            "Period Start Date": _date(raw["StartOfCountPeriod"]),
            "Period End Date": _date(raw["EndOfCountPeriod"]),
            "Sample Size": _num(raw["NumberOfCounts"]),
            "Mean Capacity": _num(raw["StandardSeatingCapacity"])
            + _num(raw["FirstClassSeatingCapacity"]).fillna(0),
            "Mean Load on Departure": _num(raw["TotalLoad_Mean"]),
            "load_units": "passengers",
        }
    )
    return _canonical(_fill_origin_destination(df))


def _fill_origin_destination(df: pd.DataFrame) -> pd.DataFrame:
    """Fill a missing service origin or destination from its calls."""
    key = ["Headcode", "Service Week Day", "Origin Departure Time", "Period Start Date"]
    key_cols = [df[c].astype("string").fillna("") for c in key]
    ordered = df.sort_values("Call Sequence", kind="stable")
    first = ordered.groupby(key_cols, sort=False)["Departure from CRS"].transform("first")
    last = ordered.groupby(key_cols, sort=False)["Departure from CRS"].transform("last")
    df["Origin CRS"] = df["Origin CRS"].fillna(first.reindex(df.index))
    df["Destination CRS"] = df["Destination CRS"].fillna(last.reindex(df.index))
    return df


def _gwr_frame(raw: pd.DataFrame, period: str) -> pd.DataFrame:
    raw = _clean_headers(raw)
    dayfirst = not pd.api.types.is_datetime64_any_dtype(raw["Start of count period"])
    return pd.DataFrame(
        {
            "Train Operator": "gwr",
            "Unique ID": raw["Unique ID (UID)"],
            "Headcode": raw["Train ID (4 Char)"],
            "Service Week Day": _day(raw["Day of Count Text"]),
            "Origin CRS": _crs(raw["Origin of service CRS"]),
            "Destination CRS": _crs(raw["Destination of service CRS"]),
            "Origin Departure Time": _time(raw["Timetabled departure time from origin"]),
            "Departure from CRS": _crs(raw["Departure from CRS"]),
            "Departure Time": _time(raw["Departure from GBTT time"]),
            "Call Sequence": _num(raw["Calling Sequence"]),
            "RSP Period": period,
            "Period Start Date": _date(raw["Start of count period"], dayfirst=dayfirst),
            "Period End Date": _date(raw["End of count period"], dayfirst=dayfirst),
            "Sample Size": _num(raw["Number of counts"]),
            "Mean Capacity": _num(raw["Standard seating capacity"])
            + _num(raw["First class seating capacity"]).fillna(0),
            "Mean Load on Departure": _num(raw["TotalLoad"]),
            "load_units": "passengers",
        }
    )


def read_gwr(path: pathlib.Path, **_: object) -> pd.DataFrame:
    """Great Western Railway: one CSV or XLSX per rail period."""
    chunks = []
    for file in tqdm(_files(path, ("*.csv", "*.xlsx")), desc="GWR"):
        match = re.search(r"Period\s*(\d{4})", file.name)
        period = match.group(1) if match else file.stem
        if file.suffix.lower() == ".xlsx":
            raw = pd.read_excel(file, dtype={"Calling Sequence": str})
        else:
            raw = _read_csv_any_encoding(file)
        df = _gwr_frame(raw, period)
        # Calls of one train/day can carry slightly different count
        # windows within the period; give them one shared window so the
        # calls group into a single run when boardings are derived.
        run = [df["Unique ID"].astype(str), df["Service Week Day"].astype(str)]
        df["Period Start Date"] = df.groupby(run)["Period Start Date"].transform("min")
        df["Period End Date"] = df.groupby(run)["Period End Date"].transform("max")
        chunks.append(_canonical(df))
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def read_gtr(path: pathlib.Path, **_: object) -> pd.DataFrame:
    """Govia Thameslink Railway: one XLSX per period, a sheet a week.

    Boardings are derived per dated run, then everything is averaged to
    one row per (train, day of week, station) within the rail period.
    """
    chunks = []
    for file in tqdm(_files(path, ("*.xlsx",)), desc="GTR"):
        sheets = pd.read_excel(file, sheet_name=None)
        for sheet in sheets.values():
            if sheet.empty or "Station CRS" not in sheet.columns:
                continue
            raw = _clean_headers(sheet)
            date = _date(raw["Date"])
            is_arrival = (
                raw["Count on arrival or departure?"].astype(str).str.lower().eq("arrival")
            )
            load = _num(raw["Total load"])
            df = pd.DataFrame(
                {
                    "Train Operator": "gtr",
                    "Unique ID": raw["Unique ID (UID)"],
                    "Headcode": raw["Train ID"],
                    "Service Week Day": _day(raw["Day of Count"]),
                    "Origin CRS": _crs(raw["Origin of service CRS"]),
                    "Destination CRS": _crs(raw["Destination of service CRS"]),
                    "Origin Departure Time": _time(raw["Timetabled departure time from origin"]),
                    "Departure from CRS": _crs(raw["Station CRS"]),
                    "Departure Time": _time(raw["Station Scheduled Arrival Time"]),
                    "Call Sequence": _num(raw["Calling point number"]),
                    "RSP Period": raw["Rail period"].astype(str),
                    "Period Start Date": date,
                    "Period End Date": date,
                    "Sample Size": _num(raw["Number of counts"]),
                    "Mean Capacity": _num(raw["Standard seating capacity"])
                    + _num(raw["First class seating capacity"]).fillna(0),
                    # An arrival count is the load arriving, not
                    # departing; only the terminus is counted that way,
                    # where departing load is 0.
                    "Mean Load on Departure": load.mask(is_arrival),
                    "load_units": "passengers",
                }
            )
            chunks.append(derive_on_off(_canonical(df)))
    if not chunks:
        return pd.DataFrame()
    daily = pd.concat(chunks, ignore_index=True)
    return _average_dated_runs(daily)


def _average_dated_runs(daily: pd.DataFrame) -> pd.DataFrame:
    """Collapse dated runs to a mean row per train, day and station."""
    key = [
        "Train Operator",
        "Unique ID",
        "Headcode",
        "Service Week Day",
        "Origin CRS",
        "Destination CRS",
        "Origin Departure Time",
        "Departure from CRS",
        "RSP Period",
        "load_units",
    ]
    for col in key:
        daily[col] = daily[col].astype("string").fillna("")
    out = (
        daily.groupby(key, sort=False)
        .agg(
            **{
                "Departure Time": ("Departure Time", "first"),
                "Call Sequence": ("Call Sequence", "median"),
                "Period Start Date": ("Period Start Date", "min"),
                "Period End Date": ("Period End Date", "max"),
                "Sample Size": ("Mean Load on Departure", "count"),
                "Mean Capacity": ("Mean Capacity", "mean"),
                "Mean Load on Departure": ("Mean Load on Departure", "mean"),
                "Mean Passengers On": ("Mean Passengers On", "mean"),
                "Mean Passengers Off": ("Mean Passengers Off", "mean"),
            }
        )
        .reset_index()
    )
    for col in key:
        out[col] = out[col].replace("", pd.NA)
    return _canonical(out)


_SWR_PERIOD_DATES = re.compile(
    r"(\d{1,2})(?:st|nd|rd|th)?\s+(\w+)\s+(\d{4})\s*-\s*(\d{1,2})(?:st|nd|rd|th)?\s+(\w+)\s+(\d{4})"
)


def _swr_period_dates(notes: pd.DataFrame) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    for value in notes.astype(str).to_numpy().ravel():
        match = _SWR_PERIOD_DATES.search(value)
        if match:
            d1, m1, y1, d2, m2, y2 = match.groups()
            start = pd.to_datetime(f"{d1} {m1} {y1}", errors="coerce")
            end = pd.to_datetime(f"{d2} {m2} {y2}", errors="coerce")
            return start, end
    return None, None


def _rail_period_dates(code: str) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Dates of a railway period such as ``"RY27 P03"``.

    P01 starts on 1 April; P02 starts on the first Sunday on or after 29
    April and every later period is four weeks long, ending the day
    before the next. P13 ends on 31 March.
    """
    match = re.search(r"RY(\d{2})\s*P(\d{2})", code)
    if not match:
        return None, None
    year = 2000 + int(match.group(1)) - 1
    period = int(match.group(2))
    april = pd.Timestamp(year=year, month=4, day=1)
    period_2_start = april + pd.Timedelta(days=28)
    period_2_start += pd.Timedelta(days=(6 - period_2_start.weekday()) % 7)

    def _start(n: int) -> pd.Timestamp:
        return april if n == 1 else period_2_start + pd.Timedelta(weeks=4 * (n - 2))

    end = (
        pd.Timestamp(year=year + 1, month=3, day=31)
        if period == 13
        else _start(period + 1) - pd.Timedelta(days=1)
    )
    return _start(period), end


def read_swr(
    path: pathlib.Path, tiploc_to_crs: dict[str, str] | None = None, **_: object
) -> pd.DataFrame:
    """South Western Railway: XLSX, two-row header, TIPLOC locations."""
    tiploc_to_crs = tiploc_to_crs or {}
    chunks = []
    unmapped: set[str] = set()
    for file in tqdm(_files(path, ("*.xlsx",)), desc="SWR"):
        book = pd.read_excel(file, sheet_name=None, header=None)
        # The file name's railway period gives the true dates. The Notes
        # sheet runs a week early and twice has the wrong year, and the
        # Rail Period column labels both RY26 P01 and P02 "RY25 P13".
        label = re.search(r"RY\d{2}\s*P\d{2}", file.name)
        start, end = _rail_period_dates(label.group(0)) if label else (None, None)
        if start is None:
            start, end = _swr_period_dates(book.get("Notes", pd.DataFrame()))
        grid = book["Loadings Data"]
        top = grid.iloc[0].ffill()
        sub = grid.iloc[1]
        names = [
            str(s).strip() if pd.notna(s) else str(t).strip()
            for t, s in zip(top, sub, strict=True)
        ]
        raw = grid.iloc[2:].copy()
        # Layouts differ by year: "Calendar Day" became "ODT", and the
        # load was once headed "Average Train Load on Departure".
        names = [
            "ODT"
            if n == "Calendar Day"
            else "Average Train Load"
            if n.startswith("Average Train Load")
            else n
            for n in names
        ]
        raw.columns = names
        raw = raw.dropna(subset=["Location"])
        if start is None:
            start, end = _rail_period_dates(str(raw["Rail Period"].dropna().iloc[0]))
        tiplocs = raw["Location"].astype(str).str.strip().str.upper()
        unmapped.update(tiplocs[~tiplocs.isin(tiploc_to_crs)].unique())
        df = pd.DataFrame(
            {
                "Train Operator": "swr",
                "Unique ID": raw["Train UID"],
                "Headcode": raw["Headcode"],
                "Service Week Day": _day(raw["ODT"]),
                "Origin CRS": _crs(
                    raw["Origin"].astype(str).str.strip().str.upper().map(tiploc_to_crs)
                ),
                "Destination CRS": _crs(
                    raw["Destination"].astype(str).str.strip().str.upper().map(tiploc_to_crs)
                ),
                "Origin Departure Time": _time(raw["Origin Time"]),
                "Departure from CRS": _crs(tiplocs.map(tiploc_to_crs)),
                "Departure Time": _time(raw["Dep"].fillna(raw["Arr"])),
                "RSP Period": label.group(0) if label else raw["Rail Period"].astype(str),
                "Period Start Date": start,
                "Period End Date": end,
                "Sample Size": _num(raw["Number of Readings"]),
                "Mean Capacity": _num(raw["Standard Seated"])
                + _num(raw["First Class Seats"]).fillna(0),
                "Mean Load on Departure": _num(raw["Average Train Load"]),
                "load_units": "passengers",
            }
        )
        # Rows are published in calling order within each train.
        df["Call Sequence"] = df.groupby(
            [raw["Train UID"].astype(str), raw["ODT"].astype(str)], sort=False
        ).cumcount()
        chunks.append(_canonical(df.dropna(subset=["Departure from CRS"])))
    if unmapped:
        logger.warning(
            "SWR: %d TIPLOC(s) have no CRS mapping and were dropped (mostly "
            "non-station timing points): %s",
            len(unmapped),
            ", ".join(sorted(unmapped)[:40]),
        )
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def _average_counts_frame(
    raw: pd.DataFrame, operator: str, window: tuple[pd.Timestamp, pd.Timestamp] | None = None
) -> pd.DataFrame:
    """Southeastern and Greater Anglia passenger count CSVs.

    ``window`` is the count window in the file name. It replaces the
    dates in the cells, which come as yyyy/mm/dd, dd/mm/yyyy or ISO, so
    some files read month first.
    """
    raw = _clean_headers(raw)
    if "Train UId" in raw.columns:  # layout from Aug 2025 onwards
        load = _num(raw["Total Load (Mean)"])
        is_arrival = raw["Count Event"].astype(str).str.lower().eq("arrival")
        df = pd.DataFrame(
            {
                "Unique ID": raw["Train UId"],
                "Headcode": raw["Headcode"],
                "Service Week Day": _day(raw["Day Of Count"]),
                "Origin CRS": _crs(raw["Origin CRS"]),
                "Destination CRS": _crs(raw["Destination CRS"]),
                "Origin Departure Time": _time(raw["Origin Time"]),
                "Departure from CRS": _crs(raw["Station CRS"]),
                "Departure Time": _time(raw["Station Departure"].fillna(raw["Station Arrival"])),
                "Call Sequence": _num(raw["Calling Point Number"]),
                "Period Start Date": _date(raw["Start Date"]),
                "Period End Date": _date(raw["End Date"]),
                "Sample Size": _num(raw["Number of Counts"]),
                "Mean Capacity": _num(raw["Standard Seating Capacity"])
                + _num(raw["First Class Seating Capacity"]).fillna(0),
                "Mean Load on Departure": load.mask(is_arrival),
            }
        )
    else:  # earlier layout: no UID/headcode, one "Order" per call
        origin_time = _time(raw["Start Time"])
        df = pd.DataFrame(
            {
                "Headcode": origin_time.str[:5].str.replace(":", "")
                + raw["Origin CRS"].astype(str)
                + raw["Destination CRS"].astype(str),
                "Service Week Day": _day(raw["Day of Week"]),
                "Origin CRS": _crs(raw["Origin CRS"]),
                "Destination CRS": _crs(raw["Destination CRS"]),
                "Origin Departure Time": origin_time,
                "Departure from CRS": _crs(raw["CRS"]),
                "Call Sequence": _num(raw["Order"]),
                "Period Start Date": _date(raw["Start Date"]),
                "Period End Date": _date(raw["End Date"]),
                "Sample Size": _num(raw["Observations"]),
                # Some files also publish an uncapped load; the capped
                # figure is truncated at the train's capacity, so prefer
                # uncapped.
                "Mean Load on Departure": _num(
                    raw["Average Loading (Uncapped)"]
                    if "Average Loading (Uncapped)" in raw.columns
                    else raw["Average Loading"]
                ),
            }
        )
    if window is not None:
        df["Period Start Date"], df["Period End Date"] = window
    df["Train Operator"] = operator
    df["RSP Period"] = df["Period Start Date"].dt.strftime("%Y-%m-%d")
    df["load_units"] = "passengers"
    return _canonical(df)


_WINDOW_IN_NAME = re.compile(r"(\d{8}) to (\d{8})")


_RAIL_PERIOD_DAYS = 28


def _name_window(file: pathlib.Path) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    match = _WINDOW_IN_NAME.search(file.name)
    return (pd.Timestamp(match.group(1)), pd.Timestamp(match.group(2))) if match else None


def select_count_windows(files: list[pathlib.Path]) -> list[pathlib.Path]:
    """Drop files that re-publish an already-covered count window.

    Southeastern and Greater Anglia publish some windows twice: a
    railway period (4 weeks from a Sunday) alongside a calendar month,
    and in 2024 a capped and an uncapped copy of the same period.
    Keeping both would count those weeks twice. Files with an uncapped
    load column are preferred, then railway-period files (they tile the
    year without overlap); any file whose window overlaps a kept file by
    more than half is dropped.
    """
    windows = []
    for file in files:
        match = _WINDOW_IN_NAME.search(file.name)
        if not match:
            windows.append((file, None, None))
            continue
        start = pd.Timestamp(match.group(1))
        end = pd.Timestamp(match.group(2))
        windows.append((file, start, end))

    def priority(item: tuple[pathlib.Path, pd.Timestamp | None, pd.Timestamp | None]) -> tuple:
        file, start, end = item
        if start is None:
            return (2, 0, file.name)
        is_rail_period = (end - start).days + 1 == _RAIL_PERIOD_DAYS
        uncapped = any("Uncapped" in c for c in pd.read_csv(file, nrows=0).columns)
        return (0 if uncapped else 1, 0 if is_rail_period else 1, file.name)

    kept: list[tuple[pathlib.Path, pd.Timestamp | None, pd.Timestamp | None]] = []
    for item in sorted(windows, key=priority):
        _, start, end = item
        if start is not None:
            length = (end - start).days + 1
            overlap = max(
                (
                    (min(end, k_end) - max(start, k_start)).days + 1
                    for _, k_start, k_end in kept
                    if k_start is not None
                ),
                default=0,
            )
            if overlap > length / 2:
                logger.info("Skipping %s: its window is already covered.", item[0].name)
                continue
        kept.append(item)
    return sorted(file for file, _, _ in kept)


def _average_counts_reader(operator: str, label: str) -> Callable[..., pd.DataFrame]:
    def reader(path: pathlib.Path, **_: object) -> pd.DataFrame:
        files = select_count_windows(_files(path, ("*.csv",)))
        chunks = [
            _average_counts_frame(
                pd.read_csv(file, dtype=str, low_memory=False), operator, _name_window(file)
            )
            for file in tqdm(files, desc=label)
        ]
        return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

    reader.__doc__ = f"{label}: 'Average Passenger Counts' CSVs, one per period."
    return reader


Reader = Callable[..., pd.DataFrame]


READERS: dict[str, Reader] = {
    "northern": read_northern,
    "tpe": read_tpe,
    "awc": read_awc,
    "emr": read_emr,
    "gwr": read_gwr,
    "gtr": read_gtr,
    "swr": read_swr,
    "southeastern": _average_counts_reader("southeastern", "Southeastern"),
    "greater_anglia": _average_counts_reader("greater_anglia", "Greater Anglia"),
}


def _dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows republished identically in more than one file.

    The same headcode recurs for good reasons (a count period spanning a
    timetable change lists a train under both timetables, and operators
    without dates publish several runs), so rows are only dropped when
    all values, including the loads, match.
    """
    before = len(df)
    df = df.drop_duplicates()
    if len(df) < before:
        logger.info("Dropped %d identical row(s) republished across files.", before - len(df))
    return df


def load_operator(
    operator: str,
    path: pathlib.Path,
    tiploc_to_crs: dict[str, str] | None = None,
    *,
    cache_dir: pathlib.Path | None = None,
) -> pd.DataFrame:
    """Read one operator's loading data into :data:`CANONICAL_COLUMNS`.

    With ``cache_dir`` set, the standardised frame is pickled there and
    reused while the source files (names, sizes and mtimes), the source
    path and the TIPLOC map are all unchanged, as recorded in a ``.key``
    file beside it (the XLSX operators are slow to parse).
    """
    if operator not in READERS:
        raise ValueError(f"Unknown operator {operator!r}; expected one of {sorted(READERS)}.")
    if not path.exists():
        logger.warning("Loading data for %s not found at %s; skipped.", operator, path)
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    version = READER_VERSION.get(operator, _DEFAULT_READER_VERSION)
    cache = cache_dir / f"{operator}_v{version}.pkl" if cache_dir else None
    key = hashlib.sha256(
        repr(
            (
                str(path.resolve()),
                [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in _files(path, ("*",))],
                sorted((tiploc_to_crs or {}).items()) if operator in _USES_TIPLOC_MAP else None,
            )
        ).encode()
    ).hexdigest()
    key_path = cache.with_suffix(".key") if cache is not None else None
    if (
        cache is not None
        and cache.exists()
        and key_path.exists()
        and key_path.read_text(encoding="utf-8") == key
    ):
        logger.info("Using cached standardised %s data from %s", operator, cache)
        return pd.read_pickle(cache)  # noqa: S301 (our own cache file)

    logger.info("Reading %s loading data from %s ...", operator, path)
    df = READERS[operator](path, tiploc_to_crs=tiploc_to_crs)
    if df.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    df = df.dropna(subset=["Departure from CRS", "Service Week Day"])
    # No British train carries more than ~3,000 people; larger values
    # are keying errors in the source.
    impossible = (df["load_units"] == "passengers") & (df["Mean Load on Departure"] > _MAX_LOAD)
    if impossible.any():
        logger.warning(
            "%s: voided %d impossible load value(s) above %d.",
            operator,
            int(impossible.sum()),
            _MAX_LOAD,
        )
        df.loc[
            impossible, ["Mean Load on Departure", "Mean Passengers On", "Mean Passengers Off"]
        ] = pd.NA
    df = _dedupe(df)
    df = derive_on_off(df).reset_index(drop=True)
    logger.info("Loaded %d %s rows.", len(df), operator)

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(cache)
        key_path.write_text(key, encoding="utf-8")
    return df


# Operators whose "Departure Time" is really the call's scheduled
# arrival: their feeds publish no departure, so a timetabled departure
# is preferred and the published arrival becomes the fallback.
ARRIVAL_TIME_OPERATORS = frozenset({"awc", "gtr"})


# Loading-data operator key -> the ATOC codes it runs under in the
# timetable. Used to keep a headcode from matching another operator's
# train.
OPERATOR_ATOC_CODES: dict[str, tuple[str, ...]] = {
    "northern": ("NT",),
    "tpe": ("TP",),
    "awc": ("VT",),
    "emr": ("EM",),
    "gwr": ("GW",),
    "gtr": ("TL", "GN", "GX", "SN"),
    "swr": ("SW",),
    "southeastern": ("SE",),
    "greater_anglia": ("LE",),
    "xc": ("XC",),
}


# Length of an ``HHMM`` time, the CIF's format.
_HHMM_LENGTH = 4


RESOLVED_COLUMN = "Resolved Departure Time"


SOURCE_COLUMN = "departure_time_source"


# The count period's start date, which decides which timetable was in
# force.
COUNT_DATE_COLUMN = "Period Start Date"


# Label for counts falling outside every configured era.
_NO_ERA = "no matching era"


# Ladder rungs, tightest first. Each is a (label, join keys) pair; the
# last rung matches on the nearest origin departure within a tolerance
# instead.
_EXACT_KEYS = ["operator", "headcode", "origin_crs", "dest_crs", "origin_dep", "stop_crs"]


_NO_DEST_KEYS = ["operator", "headcode", "origin_crs", "origin_dep", "stop_crs"]


# Southeastern and Greater Anglia synthesise a pseudo-headcode from the
# origin time and route for about half their rows, so those can never
# match a real signalling ID. The route, the origin departure minute and
# the calling station still pin the schedule without it.
_NO_HEADCODE_KEYS = ["operator", "origin_crs", "dest_crs", "origin_dep", "stop_crs"]


_NEAREST_BY = ["operator", "headcode", "origin_crs", "dest_crs", "stop_crs"]


def _hhmm(times: pd.Series) -> pd.Series:
    """``HH:MM:SS`` or ``HHMM`` as ``HHMM``, without a half minute."""
    text = times.astype("string").str.replace(":", "", regex=False)
    return text.str[:_HHMM_LENGTH].where(text.str.len() >= _HHMM_LENGTH)


def _minutes(hhmm: pd.Series) -> pd.Series:
    """``HHMM`` -> minutes since midnight."""
    value = pd.to_numeric(hhmm, errors="coerce")
    return (value // 100) * 60 + (value % 100)


def _as_clock(hhmm: pd.Series) -> pd.Series:
    """``HHMM`` -> ``HH:MM:SS``, the format the operators publish."""
    text = hhmm.astype("string")
    return (text.str[:2] + ":" + text.str[2:4] + ":00").where(text.notna())


def _atoc_to_operator(operators: Collection[str]) -> dict[str, str]:
    """ATOC code -> loading operator key, for imputed operators."""
    return {
        code: operator for operator in operators for code in OPERATOR_ATOC_CODES.get(operator, ())
    }


@functools.lru_cache(maxsize=1)
def _read_all_departures(calls_path: pathlib.Path, chunksize: int) -> pd.DataFrame:
    """Every operator's public calls with a GBTT departure, read once.

    The calls table is most of a gigabyte and a hold-out test
    re-resolves each operator in turn, so the parsed result is memoised:
    the file is read once per process and each caller filters the frame
    it needs.
    """
    atoc_map = _atoc_to_operator(OPERATOR_ATOC_CODES)
    columns = [
        "atoc_code",
        "stp_indicator",
        "is_public_call",
        "signalling_id",
        "svc_origin_crs",
        "svc_dest_crs",
        "svc_origin_dep",
        "crs",
        "public_departure",
    ]
    frames = []
    for chunk in pd.read_csv(calls_path, usecols=columns, dtype=str, chunksize=chunksize):
        keep = chunk[
            (chunk["stp_indicator"] == "P")
            & (chunk["is_public_call"].astype(str) == "True")
            & chunk["atoc_code"].isin(atoc_map)
            & chunk["public_departure"].notna()
            & chunk["crs"].notna()
        ]
        if not keep.empty:
            frames.append(keep)
    if not frames:
        return pd.DataFrame()

    calls = pd.concat(frames, ignore_index=True)
    frame = pd.DataFrame(
        {
            "operator": calls["atoc_code"].map(atoc_map),
            "headcode": calls["signalling_id"],
            "origin_crs": calls["svc_origin_crs"],
            "dest_crs": calls["svc_dest_crs"],
            "origin_dep": _hhmm(calls["svc_origin_dep"]),
            "stop_crs": calls["crs"],
            "stop_dep": _hhmm(calls["public_departure"]),
        }
    )
    # Match the "string" dtype the loading side keys on, so the merges
    # line up.
    return frame.astype("string").dropna().drop_duplicates()


def load_timetable_departures(
    calls_path: pathlib.Path,
    operators: Collection[str],
    chunksize: int = 1_000_000,
) -> pd.DataFrame:
    """Permanent public calls with a GBTT departure, for ``operators``.

    One row per (operator, headcode, origin, destination, origin
    departure, calling station). Short-term variations are excluded,
    matching how the rest of the model reads the timetable.
    """
    wanted = {op for op in operators if op in OPERATOR_ATOC_CODES}
    if not wanted:
        return pd.DataFrame()
    calls = _read_all_departures(pathlib.Path(calls_path), chunksize)
    if calls.empty:
        return calls
    return calls[calls["operator"].isin(wanted)]


def _unambiguous(calls: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Collapse to keys that map to exactly one departure time.

    A headcode that calls at the same station at two different times
    under one key is dropped rather than guessed at, and those rows fall
    through to the next rung.
    """
    grouped = calls.groupby(keys, sort=False)["stop_dep"].agg(["first", "nunique"])
    kept = grouped[grouped["nunique"] == 1].drop(columns="nunique")
    return kept.rename(columns={"first": "stop_dep"}).reset_index()


def _match_nearest(
    unresolved: pd.DataFrame, calls: pd.DataFrame, tolerance_minutes: int
) -> pd.Series:
    """Match on the nearest origin departure within tolerance.

    Catches trains whose booked origin departure has shifted by a minute
    or two between the count and the timetable extract.
    """
    left = unresolved.copy()
    left["origin_min"] = _minutes(left["origin_dep"])
    right = _unambiguous(calls, [*_NEAREST_BY, "origin_dep"])
    right["origin_min"] = _minutes(right["origin_dep"])
    left = left[left["origin_min"].notna()]
    right = right[right["origin_min"].notna()]
    if left.empty or right.empty:
        return pd.Series(dtype="string")

    left = left.sort_values("origin_min")
    right = right.drop(columns="origin_dep").sort_values("origin_min")
    matched = pd.merge_asof(
        left,
        right,
        on="origin_min",
        by=_NEAREST_BY,
        direction="nearest",
        tolerance=tolerance_minutes,
    )
    return matched.set_index("row_key")["stop_dep"].dropna()


def resolve_departure_times(
    data: pd.DataFrame,
    calls_path: pathlib.Path,
    tolerance_minutes: int = 5,
) -> pd.DataFrame:
    """Add the resolved departure time and its provenance to ``data``.

    A published GBTT departure always wins. Rows without one (Northern,
    and any operator's gaps) are matched to the timetable through the
    key ladder; Avanti and GTR, which publish an arrival, prefer the
    timetabled departure and keep their arrival as the fallback.
    Anything still unmatched falls back to the train's origin departure.
    """
    operator = data["Train Operator"].astype("string")
    published = data["Departure Time"].astype("string")
    is_arrival = operator.isin(ARRIVAL_TIME_OPERATORS)

    resolved = published.where(published.notna() & ~is_arrival)
    source = pd.Series(pd.NA, index=data.index, dtype="string")
    source[resolved.notna()] = "published"

    need = resolved.isna()
    if need.any():
        keyed = pd.DataFrame(
            {
                "operator": operator[need],
                "headcode": data.loc[need, "Headcode"].astype("string"),
                "origin_crs": data.loc[need, "Origin CRS"].astype("string"),
                "dest_crs": data.loc[need, "Destination CRS"].astype("string"),
                "origin_dep": _hhmm(data.loc[need, "Origin Departure Time"]),
                "stop_crs": data.loc[need, "Departure from CRS"].astype("string"),
            }
        )
        # Resolve one row per distinct train-stop, then map back: a
        # train recurs across every count period in the file.
        trains = (
            keyed.drop_duplicates()
            .dropna(subset=["operator", "origin_crs", "origin_dep", "stop_crs"])
            .reset_index(drop=True)
        )
        trains["row_key"] = trains.index
        logger.info(
            "Imputing departure times for %d rows (%d distinct train calls) ...",
            int(need.sum()),
            len(trains),
        )
        calls = load_timetable_departures(calls_path, set(keyed["operator"].dropna()))
        if calls.empty:
            logger.warning("No timetable calls matched these operators; nothing imputed.")
        else:
            found = _resolve_ladder(trains, calls, tolerance_minutes)
            keys = pd.MultiIndex.from_frame(keyed[_EXACT_KEYS])
            lookup = trains.set_index(pd.MultiIndex.from_frame(trains[_EXACT_KEYS]))["row_key"]
            lookup = lookup[~lookup.index.duplicated()]
            row_key = pd.Series(keys.map(lookup), index=keyed.index)
            for label, times in found.items():
                hit = pd.Series(row_key.map(times), index=keyed.index)
                fill = hit.notna() & resolved.isna()
                resolved[fill] = _as_clock(hit[fill])
                source[fill] = label

    # Avanti and GTR keep their published arrival before falling back
    # further.
    arrival_left = resolved.isna() & is_arrival & published.notna()
    resolved[arrival_left] = published[arrival_left]
    source[arrival_left] = "published_arrival"

    origin = data["Origin Departure Time"].astype("string")
    fallback = resolved.isna() & origin.notna()
    resolved[fallback] = origin[fallback]
    source[fallback] = "origin_fallback"
    source[resolved.isna()] = "unresolved"

    return pd.DataFrame({RESOLVED_COLUMN: resolved, SOURCE_COLUMN: source}, index=data.index)


def _resolve_ladder(
    trains: pd.DataFrame, calls: pd.DataFrame, tolerance_minutes: int
) -> dict[str, pd.Series]:
    """Run the key ladder, returning row_key -> departure per rung."""
    found: dict[str, pd.Series] = {}
    left = trains

    exact = _unambiguous(calls, _EXACT_KEYS)
    hit = left.merge(exact, on=_EXACT_KEYS, how="inner")
    found["timetable_exact"] = hit.set_index("row_key")["stop_dep"]
    left = left[~left["row_key"].isin(found["timetable_exact"].index)]

    if not left.empty:
        no_dest = _unambiguous(calls, _NO_DEST_KEYS)
        hit = left.merge(no_dest, on=_NO_DEST_KEYS, how="inner")
        found["timetable_no_destination"] = hit.set_index("row_key")["stop_dep"]
        left = left[~left["row_key"].isin(found["timetable_no_destination"].index)]

    if not left.empty:
        no_headcode = _unambiguous(calls, _NO_HEADCODE_KEYS)
        hit = left.merge(no_headcode, on=_NO_HEADCODE_KEYS, how="inner")
        found["timetable_no_headcode"] = hit.set_index("row_key")["stop_dep"]
        left = left[~left["row_key"].isin(found["timetable_no_headcode"].index)]

    if not left.empty:
        nearest = _match_nearest(left, calls, tolerance_minutes)
        found["timetable_nearest_origin"] = nearest

    for label, series in found.items():
        logger.info("  %-26s %7d train calls", label, len(series))
    return found


def _era_of(data: pd.DataFrame, eras: Sequence[TimetableEra]) -> pd.Series:
    """Label each count row with the timetable in force that day.

    Counts are dated by the period they were taken over. A period that
    straddles a timetable change is assigned by its start date. It
    cannot be split between eras because the count itself is not split.
    """
    taken = pd.to_datetime(data[COUNT_DATE_COLUMN], errors="coerce")
    era = pd.Series(pd.NA, index=data.index, dtype="string")
    for entry in eras:
        covered = (taken >= pd.Timestamp(entry.valid_from)) & (
            taken <= pd.Timestamp(entry.valid_to)
        )
        era[covered & era.isna()] = entry.name
    return era


def add_resolved_departures(
    data: pd.DataFrame,
    calls_path: pathlib.Path,
    tolerance_minutes: int = 5,
    eras: Sequence[TimetableEra] = (),
) -> pd.DataFrame:
    """Attach the resolved departure columns and log the provenance mix.

    With ``eras``, each count is resolved against the timetable that was
    in force when it was taken, and anything outside every era's window
    against ``calls_path``. Without them every count is resolved against
    ``calls_path``, which is the original behaviour.
    """
    if not eras or COUNT_DATE_COLUMN not in data.columns:
        resolved = resolve_departure_times(data, calls_path, tolerance_minutes)
    else:
        era = _era_of(data, eras)
        by_era = {entry.name: entry.calls_path for entry in eras}
        parts = []
        for name, rows in data.groupby(era.fillna(_NO_ERA), sort=False):
            era_path = by_era.get(name, calls_path)
            logger.info(
                "Resolving %d rows counted under the %s timetable from %s ...",
                len(rows),
                name,
                era_path,
            )
            parts.append(resolve_departure_times(rows, era_path, tolerance_minutes))
            # One era's calls table is most of a gigabyte parsed; drop
            # it before reading the next, or the run holds every era in
            # memory at once.
            _read_all_departures.cache_clear()
        resolved = pd.concat(parts).reindex(data.index)

    data = pd.concat([data.drop(columns=resolved.columns, errors="ignore"), resolved], axis=1)
    mix = resolved[SOURCE_COLUMN].value_counts(normalize=True)
    logger.info(
        "Departure times resolved: %s",
        ", ".join(f"{label} {share:.1%}" for label, share in mix.items()),
    )
    return data
