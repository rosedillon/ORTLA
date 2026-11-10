"""Measure what an era-matched timetable buys the departure imputation.

Every loading count is currently matched against one timetable extract
(May-26). The counts span April 2024 to August 2026, six published
timetables, so a train retimed or withdrawn since cannot match, and the
row falls back to the train's departure from its origin, which times it
a median 29 minutes early.

This runs the imputation both ways over the same rows and reports:

* the provenance mix, single-extract against era-matched;
* a hold-out accuracy test on rows that *do* carry a published departure
  -- hide it, re-impute, and compare against the truth it was hiding.

The hold-out is the honest measure: coverage alone would reward a looser
match.

Usage::

    python3 scripts/era_matched_departures.py <loading output dir> \
        [--operators a,b]
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from ortla import loading_data
from ortla.config import TimetableEra

DATASOURCES = pathlib.Path("/Volumes/ENGG341/datasources")
OUTPUT = DATASOURCES / "output_paper"

# The published timetables held as CIF, with the dates each was in
# force. Dec-24 and May-25 are missing, which is why 43% of the counts
# still fall back.
ERAS = [
    TimetableEra(
        name="dec23",
        calls_path=OUTPUT / "timetable_dec23" / "timetable_calls.csv",
        valid_from="2023-12-10",
        valid_to="2024-06-01",
    ),
    TimetableEra(
        name="jun24",
        calls_path=OUTPUT / "timetable_jun24" / "timetable_calls.csv",
        valid_from="2024-06-02",
        valid_to="2024-12-14",
    ),
    TimetableEra(
        name="dec25",
        calls_path=OUTPUT / "timetable_dec25" / "timetable_calls.csv",
        valid_from="2025-12-14",
        valid_to="2026-05-16",
    ),
    TimetableEra(
        name="may26",
        calls_path=OUTPUT / "timetable" / "timetable_calls.csv",
        valid_from="2026-05-17",
        valid_to="2026-12-12",
    ),
]
# The Dec-24 and Jun-25 eras are deliberately absent. They exist only as
# Working Timetable books, and the reconstruction from those books is
# sound for call counts but not for a service's endpoints: the books do
# not record a terminating arrival at several London termini, so about a
# third of schedules come out with the wrong origin or destination.
# Departure imputation matches on exactly those endpoints, and measured
# against the May-26 baseline the two eras made it worse: Northern's
# exact-match rung fell from 60.8% to 53.5%. They are used in
# scripts/frequency_panel.py, where only the call counts matter and they
# agree with a CIF era to 0.3% nationally.
BASELINE_CALLS = OUTPUT / "timetable" / "timetable_calls.csv"

KEY_COLUMNS = [
    "Train Operator",
    "Headcode",
    "Origin CRS",
    "Destination CRS",
    "Origin Departure Time",
    "Departure from CRS",
    "Departure Time",
    "Period Start Date",
    "departure_time_source",
]
# Time periods are classified off the resolved departure, so an
# imputation is only worth anything if it lands in the right one.
PERIOD_EDGES = [(0, 7), (7, 9.5), (9.5, 16), (16, 19), (19, 24)]
PERIOD_NAMES = ["early", "am_peak", "interpeak", "pm_peak", "evening"]

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def to_period(times: pd.Series) -> pd.Series:
    """Classify HH:MM:SS departures into the model's five periods."""
    hours = pd.to_timedelta(times.astype(str), errors="coerce").dt.total_seconds() / 3600
    period = pd.Series(pd.NA, index=times.index, dtype="string")
    for (lo, hi), name in zip(PERIOD_EDGES, PERIOD_NAMES, strict=True):
        period[(hours >= lo) & (hours < hi)] = name
    return period


def load_operator(path: pathlib.Path) -> pd.DataFrame:
    available = set(pd.read_csv(path, nrows=0).columns)
    columns = [c for c in KEY_COLUMNS if c in available]
    data = pd.read_csv(path, usecols=columns, dtype=str)
    data["Period Start Date"] = pd.to_datetime(data["Period Start Date"], errors="coerce")
    return data


def provenance(data: pd.DataFrame, eras) -> pd.DataFrame:
    """Resolve departures with and without era matching."""
    single = loading_data.add_resolved_departures(data, BASELINE_CALLS)
    loading_data._read_all_departures.cache_clear()
    multi = loading_data.add_resolved_departures(data, BASELINE_CALLS, eras=eras)
    return pd.DataFrame(
        {
            "single": single[loading_data.SOURCE_COLUMN],
            "single_time": single[loading_data.RESOLVED_COLUMN],
            "era": multi[loading_data.SOURCE_COLUMN],
            "era_time": multi[loading_data.RESOLVED_COLUMN],
        }
    )


def report_mix(result: pd.DataFrame) -> None:
    mix = pd.DataFrame(
        {
            "single extract": result["single"].value_counts(normalize=True),
            "era matched": result["era"].value_counts(normalize=True),
        }
    ).fillna(0)
    mix["change"] = mix["era matched"] - mix["single extract"]
    print("\nProvenance of the resolved departure (share of rows):")
    print((mix * 100).round(2).to_string())


def report_holdout(data: pd.DataFrame, eras) -> None:
    """Hide a published departure, re-impute it, score the result."""
    truth = data[data["departure_time_source"] == "published"].copy()
    if truth.empty:
        print("\nNo rows with a published departure to hold out.")
        return
    actual = truth["Departure Time"].astype("string")
    hidden = truth.drop(columns=["Departure Time"]).assign(**{"Departure Time": pd.NA})

    rows = {}
    rows_time: dict[str, pd.Series] = {}
    for label, kwargs in (
        ("single extract", {}),
        ("era matched", {"eras": eras}),
    ):
        loading_data._read_all_departures.cache_clear()
        got = loading_data.add_resolved_departures(hidden.copy(), BASELINE_CALLS, **kwargs)
        source = got[loading_data.SOURCE_COLUMN]
        imputed = source.str.startswith("timetable", na=False)
        recovered = got[loading_data.RESOLVED_COLUMN]
        rows_time["single" if not kwargs else "era"] = recovered
        exact = (recovered == actual) & imputed
        wrong_period = to_period(recovered) != to_period(actual)
        rows[label] = {
            "matched to the timetable": imputed.mean(),
            "exact to the minute (of matched)": exact.sum() / max(imputed.sum(), 1),
            "in the wrong time period (all rows)": wrong_period.mean(),
        }
    out = pd.DataFrame.from_dict(rows, orient="index")
    print(f"\nHold-out on {len(truth):,} rows that publish a departure:")
    print((out * 100).round(2).to_string())

    # Most of the counts fall in the Dec-24 and May-25 gap, where era
    # matching has nothing to offer and the row is resolved against the
    # same extract as before. Splitting on that says what the two
    # missing extracts are worth.
    era = loading_data._era_of(truth, eras).fillna("no era held")
    covered = era != "no era held"
    scored = pd.DataFrame(
        {
            "era": era,
            "single_wrong": to_period(pd.Series(rows_time["single"])) != to_period(actual),
            "era_wrong": to_period(pd.Series(rows_time["era"])) != to_period(actual),
        }
    )
    by_era = scored.groupby("era")[["single_wrong", "era_wrong"]].agg(["mean", "size"])
    print("\nWrong time period, by the timetable era the count falls in:")
    print(by_era.to_string())
    print(
        f"\nCounts inside an era held as CIF: {covered.mean():.1%}; "
        f"the rest fall in the Dec-24 / May-25 gap."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("loading_dir", type=pathlib.Path)
    parser.add_argument("--operators", default="")
    parser.add_argument("--holdout", action="store_true")
    args = parser.parse_args()

    eras = [e for e in ERAS if e.calls_path.exists()]
    missing = [e.name for e in ERAS if not e.calls_path.exists()]
    if missing:
        print(f"WARNING: no calls table for {missing}; those eras are skipped.")

    wanted = {o for o in args.operators.split(",") if o}
    files = sorted(
        p for p in args.loading_dir.glob("*_loading_analysis.csv") if not p.name.startswith(".")
    )
    for path in files:
        operator = path.name.replace("_loading_analysis.csv", "")
        if wanted and operator not in wanted:
            continue
        data = load_operator(path)
        print(f"\n{'=' * 70}\n{operator}: {len(data):,} rows\n{'=' * 70}")
        if args.holdout:
            report_holdout(data, eras)
        else:
            report_mix(provenance(data, eras))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
