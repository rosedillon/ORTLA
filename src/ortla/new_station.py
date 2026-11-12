"""Forecast a new station's annual entries and exits: SDFT plus its neighbours' residuals.

A port of the Station Demand Forecasting Tool (University of Southampton,
github.com/station-demand-forecasting-tool/sdft), in its three stages:

* Station choice (``R/sdr_calculate_probabilities.R``): each postcode chooses among its ten
  nearest stations by a logit on being nearest, sqrt(km), category F, ln(daily trains), CCTV,
  car spaces, a ticket machine and a bus interchange.
* Catchment (``R/sdr_calculate_prweighted_population.R``): residents x choice probability x
  ``((d - 750) / 1000 + 1) ** -1.5212`` beyond 750 m, summed per station.
* Trip end (``R/sdft_submit.R``): ln V linear in ln(catchment + 1), ln(daily trains),
  ln(1 + jobs within a mile), ln(1 + car spaces), electric, Travelcard boundary and terminus.

Where it departs from SDFT:

* Residents are Census 2021 by postcode at Code-Point Open centroids (England and Wales).
* Access distance is the straight line x 1.3, not a road route.
* Category F is "unstaffed" in the NRE Knowledgebase. Electric and terminus mean at least half
  the station's weekday calls are electric or start or end there.
* Daily trains resolve the timetable's short-term schedules and are the median over seven
  Wednesdays. Archive timetables book much of some operators' service as overlays.
* Jobs are Census 2021 workplace population by output area within a mile, straight line (SDFT's
  "two-minute drive").
* The constant is refitted to the footfall year as the median log residual over SDFT's own kind
  of station: under 1.5 million entries and exits in 2011-12, England or Wales, open all year.

Then the forecast borrows the residuals of established stations nearby, weighted exp(-km / 2)
and shrunk by kappa 0.25: half of SDFT's unexplained variance is spatial at about 10 km. On 69
openings this takes rmse(log) from 0.765 to 0.687 (cases 107, 160). The 80% and 90% intervals
are the quantiles of the calibration stations' leave-one-out residuals; on the 69 openings they
covered 0.84 and 0.93 (case 160).

The forecast uses the forecast timetable with the new stations in every postcode's choice set,
so their catchments come out of their neighbours'. It is a median, not a mean.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

from ortla.stations import canonical_crs, resolve_aliases
from ortla.timetable import PASSENGER_CATEGORIES

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Callable

    from ortla.config import NewStationConfig

logger = logging.getLogger(__name__)

# SDFT's station choice and trip-end coefficients.
CHOICE = {
    "nearest": 0.69065,
    "sqrt_km": -2.26183,
    "cat_f": -0.67672,
    "ln_freq": 1.19857,
    "cctv": 1.07082,
    "car_spaces": 0.00132,
    "ticket_machine": 0.98392,
    "bus": 0.75848,
}
TRIP_END = {
    "const": 3.5153,
    "ln_pop": 0.3663,
    "ln_freq": 1.1245,
    "ln_jobs": 0.0566,
    "ln_car": 0.1262,
    "electric": 0.2366,
    "travelcard": 0.3000,
    "terminus": 0.7921,
}
CIRCUITY = 1.3
CHOICE_SET = 10
JOBS_RADIUS = 1609.344  # metres: SDFT's two-minute drive, "1 mile"
FREQUENCY_WEEKS = range(-6, 7, 2)  # Wednesdays either side of the reference date
STP_PRIORITY = {"C": 0, "O": 1, "N": 2, "P": 3}
_CALL_COLUMNS = [
    "train_uid",
    "stp_indicator",
    "start_date",
    "end_date",
    "days_runs",
    "train_category",
    "power_type",
    "svc_origin_crs",
    "svc_origin_dep",
    "svc_dest_crs",
    "crs",
    "is_public_call",
]
_TO_GRID = Transformer.from_crs(4326, 27700, always_xy=True)


def postcodes(population: pathlib.Path, codepoint: pathlib.Path) -> pd.DataFrame:
    """Read residents by postcode, at Code-Point Open centroids."""
    pop = pd.read_csv(population)
    pop = pop.groupby(pop["Postcode"].str.replace(" ", "").str.upper())["Count"].sum()
    xy = pd.concat(
        pd.read_csv(f, header=None, usecols=[0, 1, 2, 3], names=["postcode", "quality", "e", "n"])
        for f in sorted(codepoint.glob("*.csv"))
    )
    xy = xy[xy["quality"] < 90]  # 90: no coordinates
    xy.index = xy["postcode"].str.replace(" ", "").str.upper()
    out = pd.DataFrame({"pop": pop}).join(xy[["e", "n"]], how="inner")
    logger.info(
        "Postcodes with residents and a centroid: %s (%.1f%% of residents)",
        f"{len(out):,}",
        100 * out["pop"].sum() / pop.sum(),
    )
    return out


def workplaces(jobs: pathlib.Path, centroids: pathlib.Path) -> pd.DataFrame:
    """Read workplace population by output area, at its centroid."""
    counts = pd.read_csv(jobs)
    counts.columns = ["oa21cd", "jobs"]
    return pd.read_csv(centroids).merge(counts, on="oa21cd", how="inner")


def _flag(values: pd.Series) -> np.ndarray:
    return values.astype(str).str.lower().eq("true").astype(float).to_numpy()


def station_attributes(covariates: pathlib.Path) -> pd.DataFrame:
    """Read each station's position and the attributes SDFT uses."""
    cov = pd.read_csv(covariates)
    east, north = _TO_GRID.transform(cov["longitude"].to_numpy(), cov["latitude"].to_numpy())
    return pd.DataFrame(
        {
            "e": east,
            "n": north,
            "car_spaces": pd.to_numeric(cov["car_spaces"], errors="coerce").fillna(0).to_numpy(),
            "cctv": _flag(cov["cctv"]),
            "ticket_machine": _flag(cov["ticket_machine"]),
            "bus": _flag(cov["bus_interchange"]),
            "travelcard": _flag(cov["travelcard_boundary"]),
            "cat_f": _flag(cov["unstaffed"]),
        },
        index=cov["crs"].str.upper(),
    )


def running_calls(calls: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    """Return the calls running on ``date``.

    Each train runs its valid schedule of highest CIF priority: an overlay replaces the
    permanent schedule for its dates and an "N" schedule runs in its own right.
    """
    key = ["train_uid", "stp_indicator", "start_date", "end_date", "days_runs"]
    schedules = calls[key].drop_duplicates()
    starts = pd.to_datetime(schedules["start_date"], errors="coerce")
    ends = pd.to_datetime(schedules["end_date"], errors="coerce")
    runs = schedules["days_runs"].fillna("").str[date.weekday()].eq("1")
    valid = schedules[(starts <= date) & (ends >= date) & runs]
    valid = (
        valid.assign(priority=valid["stp_indicator"].map(STP_PRIORITY).fillna(9))
        .sort_values("priority")
        .drop_duplicates("train_uid")
    )
    valid = valid[valid["stp_indicator"] != "C"]
    return calls.merge(valid[key], on=key, how="inner")


def service(calls_path: pathlib.Path, reference_date: str) -> pd.DataFrame:
    """Count each station's weekday trains, the median over seven Wednesdays.

    A station is electric when most of its trains are, and a terminus when most start or
    end there.
    """
    logger.info("Reading %s", calls_path)
    calls = pd.read_csv(calls_path, usecols=_CALL_COLUMNS, dtype=str)
    calls = calls[
        (calls["is_public_call"] == "True")
        & calls["train_category"].fillna("").str[:1].isin(PASSENGER_CATEGORIES)
        & calls["crs"].notna()
    ]
    aliases = resolve_aliases(None)
    days = []
    for weeks in FREQUENCY_WEEKS:
        day = running_calls(calls, pd.Timestamp(reference_date) + pd.Timedelta(days=7 * weeks))
        day = day.assign(
            crs=day["crs"].map(lambda c: canonical_crs(c, aliases)),
            electric=day["power_type"].isin(["EMU", "E"]).astype(float),
            terminus=(
                (day["crs"] == day["svc_origin_crs"]) | (day["crs"] == day["svc_dest_crs"])
            ).astype(float),
        ).drop_duplicates(["train_uid", "start_date", "svc_origin_dep", "crs"])
        days.append(
            day.groupby("crs").agg(
                freq=("train_uid", "size"),
                electric=("electric", "mean"),
                terminus=("terminus", "mean"),
            )
        )

    def mostly(column: str) -> pd.Series:
        return (pd.concat([d[column] for d in days], axis=1).mean(axis=1) >= 0.5).astype(float)  # noqa: PLR2004

    out = pd.DataFrame(
        {
            "freq": pd.concat([d["freq"] for d in days], axis=1).fillna(0).median(axis=1),
            "electric": mostly("electric"),
            "terminus": mostly("terminus"),
        }
    )
    return out[out["freq"] > 0]


def independent_services(stops_path: pathlib.Path) -> pd.Series:
    """Count each station's services in Network Rail SRAM's stops file (O04 StopsAndPasses).

    A service counts once per station where passengers may board or alight. SRAM reads its
    timetable from a different extract, so it catches schedules missing from ours (case 205).
    """
    s = pd.read_parquet(
        stops_path, columns=["VehicleJourneyNo", "ATOC", "CRS", "AlightAllowed", "BoardAllowed"]
    )
    s = s[(s["ATOC"] != "ZZ") & ((s["AlightAllowed"] > 0) | (s["BoardAllowed"] > 0))]
    aliases = resolve_aliases(None)
    crs = s["CRS"].astype(str).map(lambda c: canonical_crs(c, aliases))
    s = s.assign(crs=crs)[crs.str.fullmatch(r"[A-Z]{3}")]
    s = s[s.groupby("VehicleJourneyNo")["crs"].transform("nunique") >= 2]
    return s.drop_duplicates(["VehicleJourneyNo", "crs"]).groupby("crs").size().astype(float)


def frequency_disagreements(
    trains: pd.Series, independent: pd.Series, factor: float = 2.0
) -> pd.DataFrame:
    """Return stations whose daily trains differ from an independent count by over ``factor``."""
    both = pd.DataFrame({"freq": trains, "independent": independent}).dropna()
    ratio = both["freq"] / both["independent"]
    out = both.assign(ratio=ratio.round(3))
    return out[(ratio > factor) | (ratio < 1 / factor)].sort_values("ratio")


def catchments(stations: pd.DataFrame, people: pd.DataFrame) -> pd.Series:
    """Return each station's choice- and distance-weighted catchment population."""
    tree = cKDTree(stations[["e", "n"]].to_numpy())
    dist, idx = tree.query(people[["e", "n"]].to_numpy(), k=CHOICE_SET)
    dist = np.maximum(dist * CIRCUITY, 1.0)
    attr = {
        k: stations[k].to_numpy()[idx]
        for k in ("cat_f", "cctv", "car_spaces", "ticket_machine", "bus")
    }
    utility = (
        # Every station as near as the nearest counts as nearest, so a tie
        # isn't broken by the tree's order (case 195, N4).
        CHOICE["nearest"] * (dist == dist[:, :1])
        + CHOICE["sqrt_km"] * np.sqrt(dist / 1000)
        + CHOICE["cat_f"] * attr["cat_f"]
        + CHOICE["ln_freq"] * np.log(stations["freq"].to_numpy()[idx])
        + CHOICE["cctv"] * attr["cctv"]
        + CHOICE["car_spaces"] * attr["car_spaces"]
        + CHOICE["ticket_machine"] * attr["ticket_machine"]
        + CHOICE["bus"] * attr["bus"]
    )
    prob = np.exp(utility - utility.max(axis=1, keepdims=True))
    prob /= prob.sum(axis=1, keepdims=True)
    decay = np.where(dist <= 750, 1.0, ((dist - 750) / 1000 + 1) ** -1.5212)
    weighted = people["pop"].to_numpy()[:, None] * prob * decay
    return pd.Series(
        np.bincount(idx.ravel(), weighted.ravel(), minlength=len(stations)),
        index=stations.index,
    )


def features(
    attrs: pd.DataFrame, trains: pd.DataFrame, people: pd.DataFrame, jobs: pd.DataFrame
) -> pd.DataFrame:
    """Collect each open station's trip-end inputs."""
    open_ = attrs.join(trains, how="inner")
    open_ = open_[open_["freq"] > 0]
    open_["prw_pop"] = catchments(open_, people)
    near = cKDTree(jobs[["e", "n"]].to_numpy()).query_ball_point(
        open_[["e", "n"]].to_numpy(), JOBS_RADIUS
    )
    counts = jobs["jobs"].to_numpy()
    open_["jobs"] = [float(counts[i].sum()) for i in near]
    return open_


def trip_end(f: pd.DataFrame, const: float) -> pd.Series:
    """Return SDFT's log trip end for each row of ``f``."""
    t = TRIP_END
    return (
        const
        + t["ln_pop"] * np.log(f["prw_pop"] + 1)
        + t["ln_freq"] * np.log(f["freq"])
        + t["ln_jobs"] * np.log1p(f["jobs"])
        + t["ln_car"] * np.log1p(f["car_spaces"])
        + t["electric"] * f["electric"]
        + t["travelcard"] * f["travelcard"]
        + t["terminus"] * f["terminus"]
    )


def observed(path: pathlib.Path) -> pd.DataFrame:
    """Read a year of table 1410: entries and exits, and the region."""
    df = pd.read_csv(path)
    df.columns = [c.replace("\n", " ").strip() for c in df.columns]
    df.index = df["Three Letter Code (TLC)"].astype(str).str.strip().str.upper()
    value = pd.to_numeric(
        df["Entries and exits - All tickets"].astype(str).str.replace(",", ""), errors="coerce"
    )
    region = df["Region"] if "Region" in df.columns else pd.Series("", index=df.index)
    return pd.DataFrame({"footfall": value, "region": region})


def calibration_set(
    f: pd.DataFrame, footfall: pd.DataFrame, previous: pd.DataFrame | None, domain: pd.Index
) -> pd.DataFrame:
    """Select the stations the constant is fitted on.

    Those with footfall this year, outside Scotland, small in SDFT's own year and, when the
    previous year is given, open all through it.
    """
    cal = f.join(footfall, how="inner")
    keep = (cal["footfall"] > 0) & (cal["region"] != "Scotland") & cal.index.isin(domain)
    if previous is not None:
        keep &= cal.index.isin(previous.index[previous["footfall"] > 0])
    return cal[keep]


def distance_weights(to: pd.DataFrame, source: pd.DataFrame, scale_km: float) -> np.ndarray:
    """exp(-km / scale) between stations, zero from a station to itself."""
    km = (
        np.hypot(
            to["e"].to_numpy()[:, None] - source["e"].to_numpy()[None, :],
            to["n"].to_numpy()[:, None] - source["n"].to_numpy()[None, :],
        )
        / 1000
    )
    weights = np.exp(-km / scale_km)
    weights[to.index.to_numpy()[:, None] == source.index.to_numpy()[None, :]] = 0.0
    return weights


def borrow(weights: np.ndarray, residual: np.ndarray, kappa: float) -> np.ndarray:
    """Weighted mean of neighbours' log residuals, shrunk towards 0 by ``kappa``."""
    return weights @ residual / (kappa + weights.sum(axis=1))


def _proposed_rows(config: NewStationConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not config.proposed:
        return pd.DataFrame(), pd.DataFrame()
    codes = [p.code.upper() for p in config.proposed]
    east, north = _TO_GRID.transform(
        [p.longitude for p in config.proposed], [p.latitude for p in config.proposed]
    )
    attrs = pd.DataFrame(
        {
            "e": east,
            "n": north,
            "car_spaces": [float(p.car_spaces) for p in config.proposed],
            "cctv": [float(p.cctv) for p in config.proposed],
            "ticket_machine": [float(p.ticket_machine) for p in config.proposed],
            "bus": [float(p.bus_interchange) for p in config.proposed],
            "travelcard": [float(p.travelcard_boundary) for p in config.proposed],
            "cat_f": [float(not p.staffed) for p in config.proposed],
        },
        index=codes,
    )
    trains = pd.DataFrame(
        {
            "freq": [float(p.trains_per_day) for p in config.proposed],
            "electric": [float(p.electric) for p in config.proposed],
            "terminus": [float(p.terminus) for p in config.proposed],
        },
        index=codes,
    )
    return attrs, trains


def _cached(
    config: NewStationConfig, name: str, build: Callable[[], pd.DataFrame]
) -> pd.DataFrame:
    if config.cache_dir is None:
        return build()
    path = config.cache_dir / f"{name}.pkl"
    if path.exists():
        return pd.read_pickle(path)  # noqa: S301  (written by this module)
    out = build()
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    out.to_pickle(path)
    return out


def run_new_station(config: NewStationConfig) -> pd.DataFrame:
    """Forecast each new station and write four files to ``config.output_path``.

    ``new_station_forecasts.csv`` (SDFT's forecast, the borrowed forecast, its intervals and
    inputs), ``new_station_calibration.csv``, ``new_station_calibration_stations.csv`` and
    ``new_station_neighbours.csv`` (nearby stations' catchments without and with the new ones).
    Station rows carry latitude and longitude.
    """
    config.output_path.mkdir(parents=True, exist_ok=True)
    people = _cached(
        config,
        "postcodes",
        lambda: postcodes(config.postcode_population, config.codepoint_dir),
    )
    jobs = workplaces(config.workplace_population, config.output_area_centroids)
    attrs = station_attributes(config.station_covariates)
    extra_attrs, extra_trains = _proposed_rows(config)
    attrs = pd.concat([attrs.drop(extra_attrs.index, errors="ignore"), extra_attrs])

    cal_trains = _cached(
        config,
        f"service_{config.calibration_reference_date}",
        lambda: service(config.calibration_timetable, config.calibration_reference_date),
    )
    trains = _cached(
        config,
        f"service_{config.forecast_reference_date}",
        lambda: service(config.forecast_timetable, config.forecast_reference_date),
    )
    trains = pd.concat([trains.drop(extra_trains.index, errors="ignore"), extra_trains])

    checked = _check_frequencies(config, cal_trains, trains)

    domain = observed(config.domain_footfall)["footfall"]
    domain = domain.index[(domain > 0) & (domain < config.small_station_threshold)]
    footfall = observed(config.calibration_footfall)
    previous = observed(config.previous_footfall) if config.previous_footfall else None
    cal = calibration_set(features(attrs, cal_trains, people, jobs), footfall, previous, domain)
    residual = np.log(cal["footfall"]) - trip_end(cal, TRIP_END["const"])
    const = TRIP_END["const"] + float(residual.median())
    fitted = np.log(cal["footfall"]) - trip_end(cal, const)

    new = [c.upper() for c in config.stations] + list(extra_trains.index)
    after = features(attrs, trains, people, jobs)
    missing = [c for c in new if c not in after.index]
    if missing:
        logger.warning("No covariates or no trains in the forecast timetable: %s", missing)
    present = [c for c in new if c in after.index]

    # A new station's own past footfall must not inform its forecast.
    source = cal.drop(present, errors="ignore")
    r = fitted.reindex(source.index).to_numpy()
    if config.borrow_km:
        km, kappa = config.borrow_km, config.borrow_kappa
        lift = borrow(distance_weights(after.loc[present], source, km), r, kappa)
        left_out = r - borrow(distance_weights(source, source, km), r, kappa)
    else:
        lift, left_out = np.zeros(len(present)), r
    half_80, half_90 = np.exp(np.quantile(np.abs(left_out), [0.8, 0.9]))

    sdft = np.exp(trip_end(after.loc[present], const))
    forecast = sdft * np.exp(lift)
    # The trip end has a floor of exp(const) x frequency terms at zero
    # population and jobs, so an empty place still forecast thousands a
    # year (case 195, N2). With nobody to serve, forecast nothing.
    rows = after.loc[present]
    empty = ((rows["prw_pop"] < 1) & (rows["jobs"] < 1)).to_numpy()
    forecast = np.where(empty, 0.0, forecast)
    warnings = _warnings(rows, cal, empty, forecast, config.small_station_threshold)
    if checked is not None:
        for i, crs in enumerate(present):
            if crs in checked.index:
                r = checked.loc[crs]
                note = f"{r['freq']:g} trains a day against {r['independent']:g} in SRAM"
                warnings[i] = "; ".join(filter(None, [warnings[i], note]))
    for crs, text in zip(present, warnings, strict=True):
        if text:
            logger.warning("%s: %s", crs, text)
    out = pd.DataFrame(
        {
            "sdft_forecast": sdft.round(),
            "neighbour_adjustment": np.round(lift, 3),
            "forecast_entries_exits": forecast.round(),
            "low_80": (forecast / half_80).round(),
            "high_80": (forecast * half_80).round(),
            "low_90": (forecast / half_90).round(),
            "high_90": (forecast * half_90).round(),
            "warning": warnings,
        },
        index=present,
    )
    inputs = ["prw_pop", "freq", "jobs", "car_spaces", "electric", "terminus", "travelcard"]
    out = out.join(after.loc[present, [*inputs, "cat_f"]].round(1))
    out = out.join(_lat_lon(after.loc[present]))
    out.index.name = "crs"
    out.to_csv(config.output_path / "new_station_forecasts.csv")

    pd.DataFrame(
        [
            {
                "calibration_stations": len(cal),
                "shipped_const": TRIP_END["const"],
                "calibrated_const": round(const, 4),
                "residual_sd_log": round(float(fitted.std()), 3),
                "within_x2": round(float((fitted.abs() < np.log(2)).mean()), 3),
                "borrow_km": config.borrow_km,
                "left_out_sd_log": round(float(np.std(left_out)), 3),
                "interval_80": round(float(half_80), 3),
                "interval_90": round(float(half_90), 3),
            }
        ]
    ).to_csv(config.output_path / "new_station_calibration.csv", index=False)
    stations = pd.DataFrame(
        {
            "footfall": cal["footfall"],
            "fitted": np.exp(trip_end(cal, const)).round(),
            "log_residual": fitted.round(3),
        }
    ).join(_lat_lon(cal))
    stations.index.name = "crs"
    stations.to_csv(config.output_path / "new_station_calibration_stations.csv")

    before = features(attrs.drop(present), trains.drop(present, errors="ignore"), people, jobs)
    near = _neighbours(before, after, present, config.neighbour_km)
    if not near.empty:
        near = near.join(_lat_lon(before.loc[near["neighbour"]]).reset_index(drop=True))
    near.to_csv(config.output_path / "new_station_neighbours.csv", index=False)
    logger.info("Calibrated constant %.4f on %d stations", const, len(cal))
    logger.info(
        "Forecasts:\n%s", out[["sdft_forecast", "forecast_entries_exits", "low_80", "high_80"]]
    )
    return out


def _warnings(
    rows: pd.DataFrame,
    cal: pd.DataFrame,
    empty: np.ndarray,
    forecast: np.ndarray,
    domain_max: float,
) -> list[str]:
    """Say where a forecast rests on nothing or reaches past the calibration stations."""
    out = []
    for i, (_, r) in enumerate(rows.iterrows()):
        notes = []
        if empty[i]:
            notes.append("no population or jobs in reach (outside the census inputs?); forecast 0")
        for column, label in (("freq", "trains a day"), ("car_spaces", "car spaces")):
            top = float(cal[column].max())
            if r[column] > top:
                notes.append(f"{label} {r[column]:g} above any calibration station's ({top:g})")
        if forecast[i] > domain_max:
            notes.append(f"forecast above SDFT's size domain ({domain_max:,.0f})")
        out.append("; ".join(notes))
    return out


def _check_frequencies(
    config: NewStationConfig, cal_trains: pd.DataFrame, trains: pd.DataFrame
) -> pd.DataFrame | None:
    """Compare both timetables' daily trains with SRAM's and write the disagreements."""
    if config.frequency_check is None:
        return None
    independent = independent_services(config.frequency_check)
    found = []
    for label, frame in (("calibration", cal_trains), ("forecast", trains)):
        bad = frequency_disagreements(frame["freq"], independent)
        if len(bad):
            logger.warning(
                "%s timetable: %d stations' daily trains differ from SRAM's by over x2 "
                "(missing schedules?); see new_station_frequency_check.csv",
                label,
                len(bad),
            )
        found.append(bad.assign(timetable=label))
    out = pd.concat(found)
    out.index.name = "crs"
    out.to_csv(config.output_path / "new_station_frequency_check.csv")
    return out[out["timetable"] == "forecast"]


def _lat_lon(frame: pd.DataFrame) -> pd.DataFrame:
    lon, lat = _TO_GRID.transform(
        frame["e"].to_numpy(), frame["n"].to_numpy(), direction="INVERSE"
    )
    return pd.DataFrame({"latitude": lat.round(6), "longitude": lon.round(6)}, index=frame.index)


def _neighbours(
    before: pd.DataFrame, after: pd.DataFrame, new: list[str], radius_km: float
) -> pd.DataFrame:
    """List each new station's neighbours with their catchments without and with it."""
    rows = []
    for crs in new:
        km = np.hypot(before["e"] - after.loc[crs, "e"], before["n"] - after.loc[crs, "n"]) / 1000
        for other in km.index[km <= radius_km]:
            pop_before = float(before.loc[other, "prw_pop"])
            pop_after = float(after.loc[other, "prw_pop"]) if other in after.index else 0.0
            rows.append(
                {
                    "new_station": crs,
                    "neighbour": other,
                    "km": round(float(km[other]), 2),
                    "catchment_before": round(pop_before),
                    "catchment_after": round(pop_after),
                    "catchment_change": round(pop_after / pop_before - 1, 4)
                    if pop_before > 0
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)
