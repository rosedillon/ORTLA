"""The station table the ensemble fits on, and the sets it fits and is scored on.

Stations are those open in the calibration timetable (June 2024) with Knowledgebase attributes.
Two daily train counts are kept. ``freq_all`` is every service SRAM lists at the station, and
it is what passengers choose stations by in SDFT's catchment logit. ``freq`` leaves out the
Underground, Tyne and Wear Metro, Supertram, Eurostar and heritage trains, whose passengers
are not in ORR's footfall. The trip end reads ``freq``.

Positions are the Knowledgebase's, moved to NaPTAN's rail access points where the two are
more than a kilometre apart.
"""

from __future__ import annotations

import dataclasses
import logging
import zlib
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import cKDTree

from ortla import new_station as ns
from ortla.station_ensemble import sources

if TYPE_CHECKING:
    import pandas as pd

    from ortla.station_ensemble.sources import DataPaths

logger = logging.getLogger(__name__)

FOOTFALL_YEAR = 2024  # 2024-25, the year fitted to
DOMAIN_YEAR = 2011  # SDFT's calibration year; its small stations define the scored set
SMALL_STATION = 1_500_000
FOLDS = 5
FOLD_CELL_M = 25_000
NAPTAN_TOLERANCE_M = 1000
# Usage that starts in the series but is not a new station: reopened entrances, renamed or
# merged codes, and stations closed for rebuilding.
# fmt: off
NOT_OPENINGS = frozenset({
    "AMR", "CFO", "CLW", "RIC", "HOH", "SVS", "WEH", "SRU", "TOM", "GFD", "HHY", "ZFD", "OLD",
    "ROE", "SDE", "WPE", "SQE", "ZCW", "ZLW", "HAF", "HXX", "HWV", "BDS", "TCR", "ACY", "CDN",
    "OKE", "BHO",
})
# Kept out of fitting and scoring altogether: group tickets, interchange-only codes and
# stations whose usage series is broken.
EXCLUDE = frozenset({
    "TCR", "BDS", "ZLW", "CWX", "WWC", "ZCW", "HXX", "HWV", "HAF", "HTR", "WCX", "CUS",
})
# fmt: on


@dataclasses.dataclass
class StationFrame:
    """Stations, their inputs, and which sets each belongs to.

    Attributes
    ----------
    f
        One row per station: position (``e``, ``n``), SDFT's attributes, ``freq`` and
        ``freq_all``, catchment ``prw_pop``, ``jobs`` within a mile, ``region`` and ``name``.
    y
        ln entries and exits in 2024-25 (nan where there were none).
    wide
        Fitted on: England and Wales, used in 2023-24 and 2024-25, not opened since 1998.
    scored
        The part of ``wide`` that was under 1.5 million in 2011-12 (SDFT's domain); the cross
        validation is scored on these.
    openings
        England and Wales stations opened 1998-2023, scored as forecasts.
    scotland_established, scotland_openings
        Scottish stations never fitted to, scored as a transfer test.
    fold
        Spatial fold of each station (25 km cells).
    excluded
        Stations kept out of stage 1 and the fitted set: every station first used 1998-2023,
        and :data:`EXCLUDE`.
    first_year
        First year of recorded use, by station.
    """

    f: pd.DataFrame
    y: np.ndarray
    wide: np.ndarray
    scored: np.ndarray
    openings: list[str]
    scotland_established: np.ndarray
    scotland_openings: list[str]
    fold: np.ndarray
    excluded: set[str]
    first_year: pd.Series

    @property
    def index(self) -> pd.Index:
        return self.f.index

    @property
    def xy(self) -> np.ndarray:
        return self.f[["e", "n"]].to_numpy(float)

    @property
    def scotland(self) -> np.ndarray:
        return (self.f["region"] == "Scotland").to_numpy()

    def positions(self, codes: list[str]) -> np.ndarray:
        return self.index.get_indexer(codes)


def cells(xy: np.ndarray) -> np.ndarray:
    """Spatial fold of each point: its 25 km cell, hashed to one of five folds."""
    return np.array(
        [
            zlib.crc32(f"0:{int(e // FOLD_CELL_M)}_{int(n // FOLD_CELL_M)}".encode()) % FOLDS
            for e, n in xy
        ]
    )


def jobs_within_mile(stations: pd.DataFrame, jobs: pd.DataFrame) -> np.ndarray:
    """Sum the jobs within a mile (straight line) of each station, as SDFT does."""
    near = cKDTree(jobs[["e", "n"]].to_numpy()).query_ball_point(
        stations[["e", "n"]].to_numpy(), ns.JOBS_RADIUS
    )
    counts = jobs["jobs"].to_numpy()
    return np.array([float(counts[i].sum()) for i in near])


def build(
    paths: DataPaths, people: pd.DataFrame, calibration_service: pd.DataFrame
) -> StationFrame:
    """Assemble the station table from the raw inputs."""
    jobs = ns.workplaces(
        paths.sdft / "census2021_wp001" / "WP001_oa.csv", paths.sdft / "oa2021_pwc_osgb.csv"
    )
    attrs = ns.station_attributes(paths / "station-covariates/station_covariates.csv")
    f = ns.features(attrs, calibration_service, people, jobs)
    fold = cells(f[["e", "n"]].to_numpy(float))  # on the Knowledgebase positions

    sram = sources.sram_services(paths.sram_stops)
    f["freq_all"] = sram.reindex(f.index).fillna(f["freq"])
    gone = (
        sources.sram_services(paths.sram_stops, sources.UNTICKETED_OPERATORS)
        .reindex(f.index)
        .fillna(0)
    )
    naptan = sources.naptan_rail_points(paths).reindex(f.index)
    off = np.hypot(f["e"] - naptan["e"], f["n"] - naptan["n"]) > NAPTAN_TOLERANCE_M
    f.loc[off, ["e", "n"]] = naptan.loc[off, ["e", "n"]].to_numpy()
    logger.info("Moved to NaPTAN's position: %s", ", ".join(f.index[off]))
    f["prw_pop"] = ns.catchments(f.assign(freq=f["freq_all"]), people)
    f["jobs"] = jobs_within_mile(f, jobs)
    f["freq"] = np.maximum(f["freq_all"] - gone, 1.0)

    usage, meta = sources.footfall_series(paths)
    obs = sources.footfall(paths, FOOTFALL_YEAR)
    f["region"] = obs["region"].reindex(f.index)
    f["name"] = meta["name"].reindex(f.index).fillna("")
    scotland = (f["region"] == "Scotland").to_numpy()
    f.loc[scotland, "prw_pop"] = ns.catchments(
        f.assign(freq=f["freq_all"]), sources.scottish_residents(paths)
    ).to_numpy()[scotland]
    f.loc[scotland, "jobs"] = jobs_within_mile(f, sources.scottish_workplaces(paths))[scotland]

    first = sources.first_years(usage)
    years = usage.columns
    panel = first[(first > years[0]) & (first < years[-1]) & ~first.index.isin(NOT_OPENINGS)]
    sets = _sets(paths, f, obs, first, panel)
    y = np.log(obs["footfall"].reindex(f.index).where(lambda s: s > 0).to_numpy(float))
    scot_open = [
        c
        for c in panel.index
        if c in f.index and scotland[f.index.get_loc(c)] and np.isfinite(y[f.index.get_loc(c)])
    ]
    return StationFrame(
        f=f,
        y=y,
        wide=sets["wide"],
        scored=sets["scored"],
        openings=sets["openings"],
        scotland_established=np.flatnonzero(
            scotland & np.isfinite(y) & ~f.index.isin(sets["excluded"])
        ),
        scotland_openings=scot_open,
        fold=fold,
        excluded=sets["excluded"],
        first_year=first,
    )


def _sets(
    paths: DataPaths, f: pd.DataFrame, obs: pd.DataFrame, first: pd.Series, panel: pd.Series
) -> dict:
    """Split the stations into the fitted, scored and held-out sets.

    Openings are England and Wales stations first used 1998-2023 and used in 2024-25; those
    first used before 2012 must also pass ``NOT_OPENINGS``. Every station first used in those
    years is kept out of fitting, along with ``EXCLUDE``.
    """
    prior = sources.footfall(paths, FOOTFALL_YEAR - 1)
    domain = sources.footfall(paths, DOMAIN_YEAR)["footfall"]
    domain = domain.index[(domain > 0) & (domain < SMALL_STATION)]

    def opened(years: pd.Series, lo: int, hi: int, drop: frozenset[str]) -> list[str]:
        return sorted(
            c
            for c, y in years.items()
            if lo <= y <= hi
            and c not in drop
            and c in f.index
            and obs["region"].get(c, "") != "Scotland"
            and obs["footfall"].get(c, 0) > 0
        )

    early_any = opened(first, 1998, 2011, EXCLUDE)
    early = opened(panel, 1998, 2011, frozenset())
    late = opened(first, 2012, 2023, EXCLUDE)
    held_out = set(early_any + early + late)
    excluded = held_out | {c for c, y in first.items() if 1998 <= y <= 2023} | set(EXCLUDE)

    used = (obs["footfall"].reindex(f.index) > 0) & (f["region"] != "Scotland")
    used &= f.index.isin(prior.index[prior["footfall"] > 0])
    scored = used & f.index.isin(domain) & ~f.index.isin(list(held_out))
    wide = used & ~f.index.isin(list(excluded))
    return {
        "wide": wide.to_numpy(),
        "scored": scored.to_numpy(),
        "openings": early + late,
        "excluded": excluded,
    }
