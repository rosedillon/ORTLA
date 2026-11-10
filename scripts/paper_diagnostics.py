"""Diagnostics behind the paper's methods section.

1. gravity_specifications.csv: the frequency term fitted three ways.
2. time_period_misclassification.csv: boardings put in the wrong time
   period when classified by the train's origin departure instead of the
   boarding stop.
3. temporal_profile_shift.csv: how much that moves the day x period
   profiles.
4. northern_timetable_match.csv: share of Northern loading trains found
   in the timetable (headcode, origin, destination, origin departure),
   by quarter.
5. departure_time_imputation.csv: coverage of the timetable-imputed
   departure times, and their accuracy where the operator also publishes
   one (hold-out: hide the published time, recover it, compare).

Run from the repo root after the `timetable` and `loading` paper
configs:
    PYTHONPATH=src python3 scripts/paper_diagnostics.py
"""

from __future__ import annotations

import os
import pathlib

import numpy as np
import pandas as pd

from ortla import loading_data
from ortla.clock import classify_times
from ortla.gravity_model import load_direct_frequency
from ortla.stations import load_table_1410

DS = pathlib.Path(os.environ.get("ORM_DATA", "/Volumes/ENGG341/datasources"))
# ORM_OUT and ORM_TIMETABLE point a regeneration at its own outputs
# (paper/regen).
OUT = pathlib.Path(os.environ.get("ORM_OUT", DS / "output_paper"))
ANALYSIS = OUT / "analysis"
CACHE = OUT / "_cache" / "standardised"
CUTOFFS = {"am_peak": "7:00", "interpeak": "9:30", "pm_peak": "16:00", "evening": "19:00"}
TIMETABLE = OUT / os.environ.get("ORM_TIMETABLE", "timetable")
CALLS = TIMETABLE / "timetable_calls.csv"
# Footfall for the ODM year the paper runs (2024-25). Written by
# `build_station_footfall.py`; see `od_matrix.py` for why the year must
# match the matrix.
TABLE_1410 = DS / "table-1410" / "by_year" / "table_1410_2024_25.csv"


def _ols(x: np.ndarray, y: np.ndarray, names: list[str]) -> dict[str, float]:
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    return {
        **dict(zip(names, map(float, beta), strict=True)),
        "r_squared": float(1 - resid.var() / y.var()),
        "n_pairs": len(y),
    }


def gravity_specifications() -> pd.DataFrame:
    footfall = load_table_1410(TABLE_1410)
    frequency = load_direct_frequency(TIMETABLE / "route_frequency_summary.csv")
    rows = []
    for odm in sorted((OUT / "_cache").glob("typed_odm_v[0-9]_*.pkl")):
        df = pd.read_pickle(odm)
        df["fo"] = df["origin_tlc"].map(footfall)
        df["fd"] = df["destination_tlc"].map(footfall)
        df["f"] = [
            frequency.get(pair, 0.0)
            for pair in zip(df["origin_tlc"], df["destination_tlc"], strict=True)
        ]
        df = df[
            (df["journeys"] > 0)
            & (df["fo"] > 0)
            & (df["fd"] > 0)
            & (df["distance_miles"] >= 1)
            & (df["origin_tlc"] != df["destination_tlc"])
        ]
        y = np.log(df["journeys"].to_numpy(float))
        one = np.ones(len(df))
        base = [one, np.log(df["fo"]), np.log(df["fd"]), np.log(df["distance_miles"])]
        names = ["intercept", "ln_footfall_origin", "ln_footfall_destination", "ln_distance"]
        direct = (df["f"].to_numpy() > 0).astype(float)
        ln_f = np.log(np.maximum(df["f"].to_numpy(float), 1.0))
        served = direct > 0
        odm_name = odm.stem.split("_", 3)[3]
        specs = [
            (
                "all pairs, ln(1+trains)",
                np.column_stack([*base, np.log1p(df["f"])]),
                y,
                [*names, "frequency_term"],
            ),
            (
                "served pairs only, ln(trains) [used]",
                np.column_stack([c[served] for c in base] + [ln_f[served]]),
                y[served],
                [*names, "frequency_term"],
            ),
            (
                "all pairs, direct dummy + direct x ln(trains)",
                np.column_stack([*base, direct, direct * ln_f]),
                y,
                [*names, "direct_link_term", "frequency_term"],
            ),
        ]
        for label, x, yy, cols in specs:
            fit = _ols(x, yy, cols)
            rows.append({"odm": odm_name, "specification": label, **fit})
    out = pd.DataFrame(rows)
    out["direct_link_multiplier"] = np.exp(out.get("direct_link_term"))
    out["uplift_0_to_16_trains_all_pairs_spec"] = np.where(
        out["specification"].str.startswith("all pairs, ln(1+"),
        17 ** out["frequency_term"],
        np.nan,
    )
    return out


def time_period_checks() -> tuple[pd.DataFrame, pd.DataFrame]:
    misclass, shift = [], []
    for op in ["gwr", "tpe", "awc", "emr", "swr", "southeastern", "greater_anglia"]:
        cache = next(CACHE.glob(f"{op}_v*.pkl"), None)
        if cache is None:
            continue
        d = pd.read_pickle(cache)
        d = d[d["Departure Time"].notna() & d["Origin Departure Time"].notna()]
        on = pd.to_numeric(d["Mean Passengers On"], errors="coerce").fillna(0.0)
        by_origin = classify_times(d["Origin Departure Time"].astype("string"), CUTOFFS)
        by_station = classify_times(d["Departure Time"].astype("string"), CUTOFFS)
        minutes = (
            pd.to_timedelta(d["Departure Time"]) - pd.to_timedelta(d["Origin Departure Time"])
        ).dt.total_seconds() / 60
        minutes = minutes.where(minutes >= 0, minutes + 1440)
        changed = by_origin != by_station
        misclass.append(
            {
                "operator": op,
                "loading_rows": len(d),
                "boardings_misclassified_share": on[changed].sum() / on.sum(),
                "rows_misclassified_share": changed.mean(),
                "median_minutes_origin_to_stop": minutes.median(),
                "p90_minutes_origin_to_stop": minutes.quantile(0.9),
            }
        )

    for op in ["gwr", "tpe", "awc", "emr", "swr", "southeastern", "greater_anglia"]:
        path = OUT / "loading_all" / f"{op}_loading_analysis.csv"
        if not path.exists():  # a regeneration builds each operator on its own
            path = OUT / "_baselines" / f"{op}_case06" / f"{op}_loading_analysis.csv"
        if not path.exists():
            continue
        cols = [
            "Origin Departure Time",
            "Departure Time",
            "Service Week Day",
            "route_type_pax_off",
            "Mean Passengers On",
        ]
        a = pd.read_csv(path, usecols=cols, dtype=str)
        a = a[a["route_type_pax_off"].isin(["commute", "regional", "long_distance"])]
        a = a[a["Departure Time"].notna()]
        a["on"] = pd.to_numeric(a["Mean Passengers On"], errors="coerce")
        a["origin"] = classify_times(a["Origin Departure Time"].astype("string"), CUTOFFS)
        a["station"] = classify_times(a["Departure Time"].astype("string"), CUTOFFS)
        for band, g in a.groupby("route_type_pax_off"):
            fo = g.groupby(["Service Week Day", "origin"])["on"].sum()
            fs = g.groupby(["Service Week Day", "station"])["on"].sum()
            x = pd.concat([(fo / fo.sum()).rename("o"), (fs / fs.sum()).rename("s")], axis=1)
            x = x.fillna(0.0)
            x.index.names = ["day", "period"]
            weekly = x.groupby(level="period").sum()
            row = {
                "operator": op,
                "journey_type": band,
                "total_variation_distance": 0.5 * (x["o"] - x["s"]).abs().sum(),
            }
            for period, r in weekly.iterrows():
                row[f"{period}_share_by_station_time"] = r["s"]
                row[f"{period}_share_by_origin_time"] = r["o"]
                row[f"{period}_pct_change_if_origin_time"] = (
                    100 * (r["o"] - r["s"]) / r["s"] if r["s"] else np.nan
                )
            shift.append(row)
    return pd.DataFrame(misclass), pd.DataFrame(shift)


def departure_time_imputation() -> pd.DataFrame:
    """Coverage and accuracy of the timetable-imputed departure times.

    For operators that publish a departure, the published value is
    hidden and recovered from the timetable, so the error is measured
    against ground truth. Northern publishes none, so only its coverage
    can be reported -- the accuracy of the rungs it matches on is read
    across from the others.
    """
    rows = []
    for op in [
        "northern",
        "gwr",
        "tpe",
        "awc",
        "gtr",
        "emr",
        "swr",
        "southeastern",
        "greater_anglia",
    ]:
        cache = next(CACHE.glob(f"{op}_v*.pkl"), None)
        if cache is None:
            continue
        d = pd.read_pickle(cache).reset_index(drop=True)
        on = pd.to_numeric(d["Mean Passengers On"], errors="coerce").fillna(0.0)
        published = d["Departure Time"].astype("string")

        live = loading_data.resolve_departure_times(d, CALLS)
        src = live[loading_data.SOURCE_COLUMN]
        imputed = src.str.startswith("timetable")
        row = {
            "operator": op,
            "loading_rows": len(d),
            "publishes_departure": bool(published.notna().any())
            and op not in loading_data.ARRIVAL_TIME_OPERATORS,
            "imputed_row_share": imputed.mean(),
            "imputed_boardings_share": on[imputed].sum() / on.sum() if on.sum() else np.nan,
            "origin_fallback_boardings_share": (
                on[src == "origin_fallback"].sum() / on.sum() if on.sum() else np.nan
            ),
        }

        # Hold-out: only meaningful where a true published departure
        # exists.
        if published.notna().any() and op not in loading_data.ARRIVAL_TIME_OPERATORS:
            keep = published.notna() & d["Origin Departure Time"].notna()
            truth = published[keep]
            hidden = d[keep].copy()
            hidden["Departure Time"] = pd.NA
            got = loading_data.resolve_departure_times(hidden, CALLS)
            rung = got[loading_data.SOURCE_COLUMN]
            hit = rung.str.startswith("timetable")
            err = (
                pd.to_timedelta(got[loading_data.RESOLVED_COLUMN]) - pd.to_timedelta(truth)
            ).dt.total_seconds() / 60
            err = err.where(err.abs() < 720, err - 1440 * (err > 0) + 1440 * (err < 0))
            on_keep = on[keep]
            p_true = classify_times(truth, CUTOFFS)
            p_got = classify_times(got[loading_data.RESOLVED_COLUMN], CUTOFFS)
            p_origin = classify_times(hidden["Origin Departure Time"].astype("string"), CUTOFFS)
            row.update(
                {
                    "holdout_exact_to_minute": (err[hit].abs() < 0.5).mean(),  # noqa: PLR2004
                    "holdout_within_5min": (err[hit].abs() <= 5).mean(),
                    "holdout_p95_abs_error_min": err[hit].abs().quantile(0.95),
                    "boardings_wrong_period_imputed": (
                        on_keep[p_got != p_true].sum() / on_keep.sum()
                    ),
                    "boardings_wrong_period_origin_only": (
                        on_keep[p_origin != p_true].sum() / on_keep.sum()
                    ),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def northern_timetable_match() -> pd.DataFrame:
    calls = pd.read_csv(
        CALLS,
        usecols=["signalling_id", "svc_origin_crs", "svc_dest_crs", "svc_origin_dep", "atoc_code"],
        dtype=str,
    )
    calls = calls[calls["atoc_code"] == "NT"].drop_duplicates()
    keys = set(
        zip(
            calls["signalling_id"],
            calls["svc_origin_crs"],
            calls["svc_dest_crs"],
            calls["svc_origin_dep"].str[:4],
            strict=True,
        )
    )
    n = pd.read_pickle(next(CACHE.glob("northern_v*.pkl")))
    trains = n.drop_duplicates(
        ["Headcode", "Origin CRS", "Destination CRS", "Origin Departure Time", "RSP Period"]
    ).copy()
    dep = trains["Origin Departure Time"].astype(str).str[:5].str.replace(":", "")
    trains["matched"] = [
        k in keys
        for k in zip(
            trains["Headcode"], trains["Origin CRS"], trains["Destination CRS"], dep, strict=True
        )
    ]
    trains["quarter"] = pd.to_datetime(trains["Period Start Date"]).dt.to_period("Q").astype(str)
    out = (
        trains.groupby("quarter")
        .agg(
            trains=("matched", "size"), matched=("matched", "sum"), match_rate=("matched", "mean")
        )
        .reset_index()
    )
    total = pd.DataFrame(
        [
            {
                "quarter": "all",
                "trains": len(trains),
                "matched": int(trains["matched"].sum()),
                "match_rate": trains["matched"].mean(),
            }
        ]
    )
    return pd.concat([out, total], ignore_index=True)


def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    gravity_specifications().to_csv(ANALYSIS / "gravity_specifications.csv", index=False)
    departure_time_imputation().to_csv(ANALYSIS / "departure_time_imputation.csv", index=False)
    misclass, shift = time_period_checks()
    misclass.to_csv(ANALYSIS / "time_period_misclassification.csv", index=False)
    shift.to_csv(ANALYSIS / "temporal_profile_shift.csv", index=False)
    northern_timetable_match().to_csv(ANALYSIS / "northern_timetable_match.csv", index=False)
    print(f"Written to {ANALYSIS}")


if __name__ == "__main__":
    main()
