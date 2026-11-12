"""The station terms the ensemble's members choose from.

Terms come from three places: the place around the station (residents, jobs, cars, visitor
beds, venues and other OpenStreetMap features, Wikipedia articles), its timetable (trains,
request stops, the day's shape, where its trains go, Sunday service) and the Knowledgebase
(car spaces, staffing, operator). :func:`place_terms` and :func:`servable_commuting` work from
positions alone, so a proposed station is described the same way as an open one.

Gaps are filled with the scored stations' mean, except counts and shares, which are 0.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree

from ortla.station_ensemble import sources

if TYPE_CHECKING:
    from ortla.station_ensemble.frame import StationFrame
    from ortla.station_ensemble.sources import DataPaths

SDFT_TERMS = ["ln_pop", "ln_trains", "ln_jobs", "ln_car", "electric", "travelcard", "terminus"]
MIN_GROUP = 15  # stations a region or operator needs among the scored set to get its own term
AM_PEAK = (6.5, 9.0)  # departure hours
EVENING = 19.0
# (term, OpenStreetMap file, radius in metres, presence only)
OSM_COUNTS = [
    ("ln_malls", "malls", 800, False),
    ("ln_retail_parks", "retail_parks", 1000, False),
    ("ferry", "ferry", 500, True),
    ("airport", "airports", 3000, True),
    ("castle", "castles", 1500, True),
    ("beach", "beaches", 2000, True),
    ("ln_peaks", "peaks", 10_000, False),
    ("ln_venues", "venues", 1000, False),
    ("ln_offices", "offices", 1000, False),
    ("ln_schools", "schools", 1000, False),
    ("ln_supermarkets", "supermarkets", 800, False),
    ("bus_station", "bus_stations", 400, True),
    ("metro", "metro", 300, True),
    ("park_and_ride", "parking", 3000, True),
]
OSM_NEAREST = [("ln_km_town", "places"), ("ln_km_motorway", "motorway_j")]
# Column order of the full term table; the members index it by name, but the tree members'
# results depend on the order.
# fmt: off
ORDER = [
    "ln_pop", "ln_jobs", "ln_car", "ln_trains", "electric", "terminus", "travelcard", "ln_beds",
    "car_catchment_5km", "car_catchment_10km", "car_catchment_trains", "parkway_or_airport",
    "ln_jobs_zones", "ln_jobs_2011", "ln_wiki_places", "ln_wiki_links", "request_share",
    "am_peak_share", "evening_share", "ln_direct_destinations", "ln_malls", "ln_retail_parks",
    "ferry", "airport", "castle", "beach", "ln_peaks", "ln_venues", "ln_offices", "ln_schools",
    "ln_supermarkets", "ln_bus_stops", "bus_station", "metro", "ln_km_town", "ln_km_motorway",
    "park_and_ride", "ln_servable_commuters", "servable_share", "underground_managed",
]
# fmt: on


@dataclasses.dataclass
class Places:
    """Point data read once and queried by position."""

    beds: np.ndarray
    bus_stops: np.ndarray
    osm: dict[str, np.ndarray]
    wiki: pd.DataFrame
    jobs_zones: pd.DataFrame
    jobs_2011: pd.DataFrame
    cars: pd.DataFrame
    flows: pd.DataFrame
    msoa: pd.DataFrame

    @classmethod
    def load(cls, paths: DataPaths) -> Places:
        names = {name for _, name, _, _ in OSM_COUNTS} | {name for _, name in OSM_NEAREST}
        flows, msoa = sources.commuting(paths)
        return cls(
            beds=sources.accommodation(paths),
            bus_stops=sources.bus_stops(paths),
            osm={name: sources.osm_points(paths, name) for name in sorted(names)},
            wiki=sources.wikipedia_places(paths),
            jobs_zones=sources.workplace_zone_jobs(paths),
            jobs_2011=sources.workplace_population_2011(paths),
            cars=sources.cars_by_area(paths),
            flows=flows,
            msoa=msoa,
        )


def _count(points: np.ndarray, xy: np.ndarray, radius: float) -> np.ndarray:
    return np.array([len(v) for v in cKDTree(points).query_ball_point(xy, radius)])


def _decayed_jobs(jobs: pd.DataFrame, xy: np.ndarray) -> np.ndarray:
    """Jobs within 2 km, each weighted exp(-km)."""
    pts, counts = jobs[["e", "n"]].to_numpy(float), jobs["jobs"].to_numpy(float)
    out = []
    for i, near in enumerate(cKDTree(pts).query_ball_point(xy, 2000)):
        km = np.hypot(*(pts[near] - xy[i]).T) / 1000
        out.append(float((counts[near] * np.exp(-km)).sum()) if near else 0.0)
    return np.array(out)


def place_terms(xy: np.ndarray, places: Places) -> pd.DataFrame:
    """Terms that depend only on where the station is."""
    out = {"ln_beds": np.log1p(_count(places.beds, xy, 2000))}
    for term, name, radius, flag in OSM_COUNTS:
        c = _count(places.osm[name], xy, radius)
        out[term] = (c > 0).astype(float) if flag else np.log1p(c)
    for term, name in OSM_NEAREST:
        out[term] = np.log(np.maximum(cKDTree(places.osm[name]).query(xy)[0] / 1000, 0.1))
    wiki = places.wiki
    near = cKDTree(wiki[["e", "n"]].to_numpy()).query_ball_point(xy, 1000)
    links = wiki["sitelinks"].to_numpy()
    out["ln_wiki_places"] = np.log1p([len(v) for v in near])
    out["ln_wiki_links"] = np.log1p([links[v].sum() if v else 0 for v in near])
    out["jobs_zones"] = _decayed_jobs(places.jobs_zones, xy)
    out["jobs_2011"] = _decayed_jobs(places.jobs_2011, xy)
    out["ln_bus_stops"] = np.log1p(_count(places.bus_stops, xy, 400))
    return pd.DataFrame(out)


def car_catchment(
    xy: np.ndarray, cars: pd.DataFrame, weight: np.ndarray, scale_km: float
) -> np.ndarray:
    """Each area's cars shared among stations within 25 km by ``weight`` x exp(-km / scale)."""
    has = np.flatnonzero(weight > 0)
    tree = cKDTree(xy[has])
    zones = cars[["e", "n"]].to_numpy(float)
    count = cars["cars"].to_numpy(float)
    out = np.zeros(len(xy))
    for z, near in enumerate(tree.query_ball_point(zones, 25_000)):
        if not near:
            continue
        k = has[np.asarray(near)]
        u = weight[k] * np.exp(-np.hypot(*(xy[k] - zones[z]).T) / 1000 / scale_km)
        out[k] += count[z] * u / u.sum()
    return out


def servable_commuting(
    xy: np.ndarray, reach: list[set[int]], places: Places
) -> tuple[np.ndarray, np.ndarray]:
    """Count commuters living near each station who work near a station its trains reach.

    Homes are MSOAs within 2 km of the station, workplaces MSOAs within 1.5 km of a station
    in ``reach`` (positions in ``xy``) and not among the homes. Returns the number and its
    share of all commuters from those homes.
    """
    centroids = places.msoa
    pos = {m: k for k, m in enumerate(centroids.index)}
    f = places.flows
    commuters = sp.csr_matrix(
        (f["n"].to_numpy(float), (f["o"].map(pos).to_numpy(), f["d"].map(pos).to_numpy())),
        shape=(len(centroids), len(centroids)),
    )
    tree = cKDTree(centroids.to_numpy(float))
    work = tree.query_ball_point(xy, 1500)
    home = tree.query_ball_point(xy, 2000)
    served, total = np.zeros(len(xy)), np.zeros(len(xy))
    for i, homes in enumerate(home):
        if not homes:
            continue
        dest = set().union(*(work[k] for k in reach[i] if k != i)) - set(homes)
        rows = commuters[homes]
        total[i] = rows.sum()
        served[i] = rows[:, sorted(dest)].sum() if dest else 0.0
    return served, np.where(total > 0, served / np.maximum(total, 1), 0)


def timetable_terms(
    index: pd.Index, xy: np.ndarray, paths: DataPaths, places: Places
) -> pd.DataFrame:
    """Read the timetable terms of stations open in the forecast timetable."""
    wed = sources.day_calls(paths.forecast_calls, 2, sources.FORECAST_DATE)
    dep = wed.dropna(subset=["public_departure"])
    hour = (
        dep["public_departure"].str[:2].astype(int)
        + dep["public_departure"].str[2:4].astype(int) / 60
    )
    departures = dep.groupby("crs").size()

    def share_of_departures(when: pd.Series) -> np.ndarray:
        return (dep[when].groupby("crs").size() / departures).reindex(index).fillna(0).to_numpy()

    calling = wed.groupby("train_uid")["crs"].agg(set)
    reach = {
        c: set().union(*(calling[u] for u in uids))
        for c, uids in wed.groupby("crs")["train_uid"].agg(set).items()
    }
    pos = {c: k for k, c in enumerate(index)}
    served, share = servable_commuting(
        xy, [{pos[c] for c in reach.get(code, ()) if c in pos} for code in index], places
    )
    week = sources.trains_by_weekday(paths.forecast_calls, sources.FORECAST_WEEK).reindex(index)
    request = sources.request_stop_share(paths.full_cif).reindex(index).fillna(0)
    destinations = pd.Series({c: len(v) for c, v in reach.items()}).reindex(index).fillna(0)
    return pd.DataFrame(
        {
            "request_share": request.to_numpy(),
            "am_peak_share": share_of_departures((hour >= AM_PEAK[0]) & (hour < AM_PEAK[1])),
            "evening_share": share_of_departures(hour >= EVENING),
            "direct_destinations": destinations.to_numpy(),
            "servable_commuters": served,
            "servable_share": share,
            "sunday_ratio": (np.log(week["sun"] + 1) - np.log(week["weekday"] + 1)).to_numpy(),
        },
        index=index,
    )


def knowledgebase_terms(paths: DataPaths, index: pd.Index) -> pd.DataFrame:
    cov = pd.read_csv(paths / "station-covariates/station_covariates.csv").drop_duplicates("crs")
    cov = cov.set_index("crs").reindex(index)
    staffing = cov["staffing_level"]
    return pd.DataFrame(
        {
            "operator": cov["operator_code"].fillna("?"),
            "staffed_full_time": (staffing == "Full Time").astype(float).where(staffing.notna()),
            "staffed_part_time": (staffing == "Part Time").astype(float).where(staffing.notna()),
        },
        index=index,
    )


def car_catchment_by_trains(
    xy: np.ndarray,
    cars: pd.DataFrame,
    ln_car: np.ndarray,
    freq: np.ndarray,
    reference: np.ndarray | None = None,
) -> np.ndarray:
    """Return ln(1 + cars) shared by car spaces x sqrt(trains / median trains), within 5 km scale.

    The median is over ``reference`` (the open stations) when given, else over ``freq``.
    """
    median = np.median(freq if reference is None else reference)
    return np.log1p(car_catchment(xy, cars, ln_car * np.sqrt(freq / median), 5.0))


def _fill(values: np.ndarray, scored: np.ndarray) -> np.ndarray:
    return np.where(np.isnan(values), np.nanmean(values[scored]), values)


def _station_terms(sf: StationFrame, place: pd.DataFrame, places: Places) -> pd.DataFrame:
    """SDFT's terms, car catchments and the terms that need only the station's position."""
    f = sf.f
    t = pd.DataFrame(index=f.index)
    t["ln_pop"] = np.log1p(f["prw_pop"])
    t["ln_jobs"] = np.log1p(f["jobs"])
    t["ln_car"] = np.log1p(f["car_spaces"])
    t["ln_trains"] = np.log(f["freq"])
    for c in ("electric", "terminus", "travelcard"):
        t[c] = f[c].astype(float)
    ln_car = t["ln_car"].to_numpy()
    t["car_catchment_5km"] = np.log1p(car_catchment(sf.xy, places.cars, ln_car, 5.0))
    t["car_catchment_10km"] = np.log1p(car_catchment(sf.xy, places.cars, ln_car, 10.0))
    t["car_catchment_trains"] = car_catchment_by_trains(
        sf.xy, places.cars, ln_car, f["freq"].to_numpy(float)
    )
    named = f["name"].str.contains("Parkway|Airport|International")
    t["parkway_or_airport"] = (named | (f["car_spaces"] >= 500)).astype(float)
    for term, raw in (("ln_jobs_zones", "jobs_zones"), ("ln_jobs_2011", "jobs_2011")):
        v = np.log1p(place[raw].to_numpy(float))
        # Workplace zones and the 2011 workplace counts cover England and Wales only.
        t[term] = np.where(sf.scotland | np.isnan(v), t["ln_jobs"], v)
    for c in place.columns.intersection(ORDER):
        t[c] = place[c].to_numpy()
    return t


def build(
    sf: StationFrame,
    place: pd.DataFrame,
    timetable: pd.DataFrame,
    kb: pd.DataFrame,
    places: Places,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Return every station's terms, filled, with the names of the region and operator columns."""
    scored, scotland = sf.scored, sf.scotland
    t = _station_terms(sf, place, places)
    for c in ("request_share", "am_peak_share", "evening_share"):
        t[c] = timetable[c].to_numpy()
    t["ln_direct_destinations"] = np.log1p(timetable["direct_destinations"].to_numpy())
    # The census commuting flows cover England and Wales only.
    t["ln_servable_commuters"] = np.where(
        scotland, np.nan, np.log1p(timetable["servable_commuters"])
    )
    t["servable_share"] = np.where(scotland, np.nan, timetable["servable_share"])
    t["underground_managed"] = (kb["operator"] == "LT").astype(float).to_numpy()
    t = t[ORDER]
    for c in ("ln_servable_commuters", "servable_share", "ln_km_town", "ln_km_motorway"):
        t[c] = _fill(t[c].to_numpy(float), scored)

    region = sf.f["region"].fillna("?")
    regions = [
        r
        for r in sorted(region.unique())
        if r not in ("?", "London") and (region[scored] == r).sum() >= MIN_GROUP
    ]
    for r in regions:
        t[f"region_{r}"] = (region == r).astype(float).to_numpy()
    counts = kb["operator"][scored].value_counts()
    # Northern, the largest, is the base the other operators are measured against.
    operators = [o for o, k in counts.items() if k >= MIN_GROUP and o != "NT"]
    for o in operators:
        t[f"operator_{o}"] = (kb["operator"] == o).astype(float).to_numpy()
    t["sunday_ratio"] = _fill(timetable["sunday_ratio"].to_numpy(float), scored)
    for c in ("staffed_full_time", "staffed_part_time"):
        t[c] = _fill(kb[c].to_numpy(float), scored)
    return t, [f"region_{r}" for r in regions], [f"operator_{o}" for o in operators]
