"""Where the ensemble's inputs live under the data folder, and the readers for them.

Every reader returns plain frames: points on the British National Grid (columns ``e``,
``n``), or values indexed by CRS code. Nothing here fits anything.
"""

from __future__ import annotations

import collections
import dataclasses
import logging
import re
from importlib import resources
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pyproj import Transformer

from ortla import new_station as ns
from ortla.stations import resolve_aliases, resolve_recodes
from ortla.timetable import PASSENGER_CATEGORIES

if TYPE_CHECKING:
    import pathlib

logger = logging.getLogger(__name__)

_TO_GRID = Transformer.from_crs(4326, 27700, always_xy=True)
# Operators whose passengers are not in ORR's footfall: London Underground, Tyne and Wear
# Metro, Supertram, Eurostar and heritage railways.
UNTICKETED_OPERATORS = ("LT", "TW", "SJ", "ES", "NY", "WR", "LS", "MV", "YG")
# Workplace population per BRES job: England and Wales WP001 (2021) over GB BRES (2024).
SCOTTISH_JOBS_SCALE = 27_671_490 / (28_412_000 + 1_349_000)
PLACES_TO_STAY = (
    "hotel", "guest_house", "hostel", "camp_site", "caravan_site", "apartment", "chalet", "motel",
)  # fmt: skip


@dataclasses.dataclass(frozen=True)
class DataPaths:
    """Every input file, laid out under one data folder (see the README)."""

    root: pathlib.Path

    def __truediv__(self, relative: str) -> pathlib.Path:
        return self.root / relative

    @property
    def sdft(self) -> pathlib.Path:
        return self.root / "sdft-inputs"

    @property
    def footfall_dir(self) -> pathlib.Path:
        return self.root / "table-1410" / "by_year"

    def footfall(self, year: int) -> pathlib.Path:
        """Table 1410 for the year starting in April ``year``."""
        return self.footfall_dir / f"table_1410_{year}_{str(year + 1)[2:]}.csv"

    @property
    def calibration_calls(self) -> pathlib.Path:
        """``ortla timetable`` calls for the footfall year (June 2024 timetable)."""
        return self.root / "output_paper" / "timetable_jun24" / "timetable_calls.csv"

    @property
    def forecast_calls(self) -> pathlib.Path:
        """``ortla timetable`` calls for the forecast timetable (24 June 2026 CIF)."""
        return self.root / "output_paper" / "timetable_may26" / "timetable_calls.csv"

    @property
    def full_cif(self) -> pathlib.Path:
        """A full CIF extract, read for each station's request stops."""
        return self.root / "timetable-data" / "new-schedule" / "CIF_ALL_FULL_DAILY_toc-full.CIF"

    @property
    def sram_stops(self) -> pathlib.Path:
        return self.root / "sram" / "SRAM_Summer24_O04_StopsAndPasses.parquet"

    @property
    def osm(self) -> pathlib.Path:
        """Point files written by ``scripts/fetch_station_ensemble_osm.py``."""
        return self.root / "osm_geofabrik_2026-11-11"


CALIBRATION_DATE = "2024-09-18"  # a Wednesday in the calibration timetable
FORECAST_DATE = "2026-06-17"  # a Wednesday in the forecast timetable
FORECAST_WEEK = ("2026-06-22", "2026-06-28")  # the week whose Sundays are counted


def _number(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values.astype(str).str.replace(",", ""), errors="coerce")


def to_grid(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Latitude and longitude to British National Grid metres, as an (n, 2) array."""
    return np.column_stack(_TO_GRID.transform(np.asarray(lon, float), np.asarray(lat, float)))


def to_lat_lon(frame: pd.DataFrame) -> pd.DataFrame:
    """Latitude and longitude of rows with grid columns ``e`` and ``n``."""
    lon, lat = _TO_GRID.transform(
        frame["e"].to_numpy(), frame["n"].to_numpy(), direction="INVERSE"
    )
    return pd.DataFrame({"latitude": lat.round(6), "longitude": lon.round(6)}, index=frame.index)


def footfall_series(paths: DataPaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read entries and exits by station and year, and each station's latest name and region.

    Years are labelled by the April they start in.
    """
    years, meta = {}, []
    for path in sorted(paths.footfall_dir.glob("table_1410_*.csv")):
        if path.name.startswith("._"):
            continue
        t = pd.read_csv(path).iloc[:, :4]
        t.columns = ["name", "region", "tlc", "ee"]
        t["ee"] = _number(t["ee"])
        t = t.dropna(subset=["tlc"]).drop_duplicates("tlc").set_index("tlc")
        years[int(re.search(r"_(\d{4})_\d\d\.csv$", path.name).group(1))] = t["ee"]
        meta.append(t[["name", "region"]])
    usage = pd.DataFrame(years).fillna(0.0)
    meta = pd.concat(meta)
    return usage, meta[~meta.index.duplicated(keep="last")]


def first_years(usage: pd.DataFrame) -> pd.Series:
    """Return the first year each station has any recorded use."""
    used = usage > 0
    return used.idxmax(axis=1)[used.any(axis=1)].astype(int)


def footfall(paths: DataPaths, year: int) -> pd.DataFrame:
    """One year of table 1410: entries and exits, and the region."""
    return ns.observed(paths.footfall(year))


def sram_services(stops_path: pathlib.Path, operators: tuple[str, ...] | None = None) -> pd.Series:
    """Count services a day at each station in SRAM's stops file.

    With ``operators``, only those operators' services are counted.
    """
    s = pd.read_parquet(
        stops_path, columns=["VehicleJourneyNo", "ATOC", "CRS", "AlightAllowed", "BoardAllowed"]
    )
    s = s[(s["ATOC"] != "ZZ") & ((s["AlightAllowed"] > 0) | (s["BoardAllowed"] > 0))]
    s = s.assign(crs=s["CRS"].astype(str).str.strip().str.upper().replace(resolve_recodes(None)))
    s = s[s["crs"].str.fullmatch(r"[A-Z]{3}")]
    s = s[s.groupby("VehicleJourneyNo")["crs"].transform("nunique") >= 2]
    if operators is not None:
        s = s[s["ATOC"].isin(operators)]
    return s.drop_duplicates(["VehicleJourneyNo", "crs"]).groupby("crs").size().astype(float)


def naptan_rail_points(paths: DataPaths) -> pd.DataFrame:
    """Mean NaPTAN position of each station's rail access points."""
    nap = pd.read_csv(
        paths.sdft / "naptan" / "naptan.csv",
        low_memory=False,
        usecols=["ATCOCode", "StopType", "Easting", "Northing"],
    )
    rail = nap[nap["StopType"] == "RLY"].dropna(subset=["Easting"])
    tiploc_crs = _tiploc_crs().drop_duplicates("tiploc").set_index("tiploc")["crs"]
    rail = rail.assign(crs=rail["ATCOCode"].str[4:].map(tiploc_crs)).dropna(subset=["crs"])
    return rail.groupby("crs")[["Easting", "Northing"]].mean().set_axis(["e", "n"], axis=1)


def _tiploc_crs() -> pd.DataFrame:
    """Read the package's TIPLOC to CRS map (built from CORPUS)."""
    with resources.files("ortla.models").joinpath("corpus_tiploc_crs.csv").open() as fh:
        return pd.read_csv(fh, comment="#")


def bus_stops(paths: DataPaths) -> np.ndarray:
    nap = pd.read_csv(
        paths.sdft / "naptan" / "naptan.csv",
        low_memory=False,
        usecols=["StopType", "Status", "Easting", "Northing"],
    )
    nap = nap[(nap["StopType"] == "BCT") & (nap["Status"] == "active")].dropna()
    return nap[["Easting", "Northing"]].to_numpy(float)


def scottish_residents(paths: DataPaths) -> pd.DataFrame:
    """Census 2011 residents by Scottish output area, at its population-weighted centroid."""
    scot = paths.sdft / "census2011_scotland"
    pop = _number(pd.read_csv(scot / "KS101SC.csv", index_col=0)["All people"].replace("-", "0"))
    cen = pd.read_csv(scot / "oa2011_pwc_scotland_osgb.csv").set_index("oa11cd")
    return cen.join(pop.rename("pop"), how="inner")


def scottish_workplaces(paths: DataPaths) -> pd.DataFrame:
    """BRES 2024 jobs by Scottish data zone, scaled to workplace population."""
    scot = paths.sdft / "census2011_scotland"
    bres = pd.read_csv(scot / "bres_dz2011.csv").set_index("GEOGRAPHY_CODE")["OBS_VALUE"]
    dz = pd.read_csv(scot / "dz2011_centroids_osgb.csv").set_index("dz11cd")
    dz = dz.join(bres.rename("jobs"), how="inner")
    return dz.assign(jobs=dz["jobs"] * SCOTTISH_JOBS_SCALE)


def cars_by_area(paths: DataPaths) -> pd.DataFrame:
    """Read households' cars or vans by output area.

    Census 2021 for England and Wales, 2011 for Scotland.
    """
    ew = pd.read_csv(paths.sdft / "census2021_ts" / "census2021-ts045-oa.csv")
    band = [c for c in ew.columns if c.startswith("Number of cars or vans:")]
    cars = ew[band[2]] + 2 * ew[band[3]] + 3 * ew[band[4]]  # one, two, three or more
    ew = pd.DataFrame({"oa": ew["geography code"], "cars": cars}).merge(
        pd.read_csv(paths.sdft / "oa2021_pwc_osgb.csv").rename(columns={"oa21cd": "oa"}), on="oa"
    )
    scot = paths.sdft / "census2011_scotland"
    sc = pd.read_csv(scot / "KS404SC.csv", thousands=",")
    sc = sc.rename(columns={sc.columns[0]: "oa", "All cars or vans in the households": "cars"})
    sc["cars"] = _number(sc["cars"]).fillna(0)
    cen = pd.read_csv(scot / "oa2011_pwc_scotland_osgb.csv")
    sc = sc[["oa", "cars"]].merge(cen.rename(columns={cen.columns[0]: "oa"}), on="oa")
    out = pd.concat([ew, sc[ew.columns]])
    return out[out["cars"] > 0]


def accommodation(paths: DataPaths) -> np.ndarray:
    """Hotels, guest houses, hostels, campsites and other places to stay (OpenStreetMap)."""
    poi = pd.read_csv(paths.sdft / "osm" / "pois.csv").rename(columns=lambda c: c.lstrip("@"))
    poi = poi[poi["tourism"].isin(PLACES_TO_STAY)]
    return to_grid(poi["lat"].to_numpy(), poi["lon"].to_numpy())


def osm_points(paths: DataPaths, name: str) -> np.ndarray:
    """One of the OpenStreetMap point files (``malls``, ``venues``, ...)."""
    p = pd.read_csv(
        paths.osm / f"{name}.csv", on_bad_lines="skip", keep_default_na=False, na_values=[""]
    )
    p = p.dropna(subset=["lat", "lon"])
    if name == "airports":
        p = p[p["iata"].notna()]
    return to_grid(p["lat"].to_numpy(float), p["lon"].to_numpy(float))


def wikipedia_places(paths: DataPaths) -> pd.DataFrame:
    """Read places with an English Wikipedia article, with their sitelink counts.

    Articles about stations are left out.
    """
    wd = pd.read_csv(paths / "osm_overpass_2026-11-11/wikidata_enwiki.csv", on_bad_lines="skip")
    wd = wd.dropna(subset=["lat", "lon"])
    wd = wd[~wd["title"].astype(str).str.lower().str.contains("station")]
    xy = to_grid(wd["lat"].to_numpy(float), wd["lon"].to_numpy(float))
    return pd.DataFrame(
        {"e": xy[:, 0], "n": xy[:, 1], "sitelinks": wd["sitelinks"].to_numpy(float)}
    )


def workplace_zone_jobs(paths: DataPaths) -> pd.DataFrame:
    """Census 2011 jobs by workplace zone (England and Wales), at the zone's centroid."""
    extra = paths.sdft / "census2011_extra"
    jobs = pd.read_csv(extra / "wz_jobs.csv").set_index("GEOGRAPHY_CODE")["OBS_VALUE"]
    cen = pd.read_csv(extra / "wz2011_pwc_osgb.csv").set_index("wz11cd")
    return cen.join(jobs.rename("jobs"), how="inner")


def workplace_population_2011(paths: DataPaths) -> pd.DataFrame:
    """Census 2011 workplace population by output area, at the output area's centroid."""
    cen = pd.read_csv(paths.sdft / "oa2011_pwc_osgb.csv").set_index("oa11cd")
    wp = pd.read_csv(paths.sdft / "census2011_extra" / "wp101_oa.csv").set_index("GEOGRAPHY_CODE")
    return wp.join(cen, how="inner").rename(columns={"OBS_VALUE": "jobs"})


def commuting(paths: DataPaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read Census 2011 commuters between MSOAs (all modes).

    Also returns each MSOA's position, the mean of its output areas' centroids.
    """
    cen = pd.read_csv(paths.sdft / "oa2011_pwc_osgb.csv").set_index("oa11cd")
    lookup = (
        pd.read_csv(paths / "lookups/oa11_msoa11.csv").set_index("OA11CD").join(cen, how="inner")
    )
    centroids = lookup.groupby("MSOA11CD")[["e", "n"]].mean()
    flows = pd.read_csv(paths / "census2011_wu03ew/wu03ew_msoa.csv", usecols=[0, 1, 2])
    flows.columns = ["o", "d", "n"]
    flows = flows[flows["o"].isin(centroids.index) & flows["d"].isin(centroids.index)]
    return flows, centroids


def request_stop_share(cif: pathlib.Path) -> pd.Series:
    """Share of each station's timetabled public calls that are request stops (CIF activity R)."""
    calls: collections.Counter[str] = collections.Counter()
    request: collections.Counter[str] = collections.Counter()
    with cif.open(encoding="latin-1") as fh:
        for line in fh:
            if line[:2] not in ("LI", "LT"):
                continue
            activity = line[42:54] if line[:2] == "LI" else line[25:37]
            codes = {activity[i : i + 2] for i in range(0, 12, 2)}
            if codes & {"T ", "R ", "TF"}:
                tiploc = line[2:9].strip()
                calls[tiploc] += 1
                request[tiploc] += "R " in codes
    by_tiploc = pd.DataFrame({"calls": calls, "request": request})
    by_crs = (
        by_tiploc.join(_tiploc_crs().set_index("tiploc"), how="inner")
        .groupby("crs")[["calls", "request"]]
        .sum()
    )
    return by_crs["request"] / by_crs["calls"]


def day_calls(calls_path: pathlib.Path, weekday: int, date: str) -> pd.DataFrame:
    """Read the public calls of every schedule valid on ``date`` (``weekday`` 0 is Monday)."""
    cols = [
        "train_uid",
        "days_runs",
        "start_date",
        "end_date",
        "crs",
        "public_departure",
        "is_public_call",
    ]
    t = pd.read_csv(calls_path, dtype=str, usecols=cols)
    return t[
        (t["is_public_call"] == "True")
        & (t["days_runs"].str[weekday] == "1")
        & (t["start_date"] <= date)
        & (t["end_date"] >= date)
    ]


def trains_by_weekday(calls_path: pathlib.Path, week: tuple[str, str]) -> pd.DataFrame:
    """Count ticketed passenger trains calling at each station on each day of a week.

    Only permanent schedules are counted.
    """
    cols = [
        "train_uid",
        "stp_indicator",
        "atoc_code",
        "days_runs",
        "start_date",
        "end_date",
        "train_category",
        "crs",
        "is_public_call",
    ]
    parts = []
    for chunk in pd.read_csv(calls_path, dtype=str, usecols=cols, chunksize=1_000_000):
        keep = (
            chunk["stp_indicator"].eq("P")
            & chunk["is_public_call"].eq("True")
            & chunk["crs"].notna()
            & chunk["train_category"].fillna("").str[:1].isin(PASSENGER_CATEGORIES)
            & ~chunk["atoc_code"].isin(UNTICKETED_OPERATORS)
            & (chunk["start_date"] <= week[1])
            & (chunk["end_date"] >= week[0])
        )
        parts.append(chunk[keep])
    calls = pd.concat(parts).drop_duplicates(["train_uid", "start_date", "crs"])
    runs = np.array([list(map(int, s)) for s in calls["days_runs"]])
    days = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
    out = pd.DataFrame(
        {
            day: calls[runs[:, i] == 1].drop_duplicates(["train_uid", "crs"]).groupby("crs").size()
            for i, day in enumerate(days)
        }
    ).fillna(0)
    return out.assign(weekday=out[days[:5]].mean(axis=1))


def station_aliases() -> dict[str, str]:
    """Second codes and recodes merged, as the skim reads the timetable."""
    return {**resolve_aliases(None), **resolve_recodes(None)}
