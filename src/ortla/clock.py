"""Clock times, weekdays and the periods a day is split into.

The timetable writes times as ``HHMM`` (``"0831H"`` for a half minute);
the loading data and candidate services use ``HH:MM`` or ``HH:MM:SS``.
"""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pathlib

    import pandas as pd

WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
WEEKDAY_INDEX = {day: index for index, day in enumerate(WEEKDAYS)}

# A period matrix holds a year of trips for one weekday; divide by this
# for one typical day.
WEEKDAYS_PER_YEAR = 365.25 / 7

MINUTES_PER_DAY = 24 * 60
UNKNOWN_PERIOD = "Unknown"

# Length of each named period in minutes, used for a slot's headway.
PERIOD_MINUTES = {
    "early": 180.0,
    "am_peak": 150.0,
    "interpeak": 390.0,
    "pm_peak": 180.0,
    "evening": 300.0,
}


def hourly_time_cutoffs() -> dict[str, str]:
    """Cutoffs for 24 clock hours labelled ``HH:00``, with no early period."""
    return {f"{hour:02d}:00": f"{hour}:00" for hour in range(24)}


def hourly_period_order() -> list[str]:
    return [f"{hour:02d}:00" for hour in range(24)]


def interval_time_cutoffs(minutes: int) -> dict[str, str]:
    """Cutoffs for equal periods of ``minutes`` from midnight, labelled ``HH:MM``.

    Case 06: the five named periods explain 60.6% of the variation in
    load by departure time within a corridor, half-hours 85.7% and the
    exact minute 87.4%, so 30 is the interval worth using.
    """
    if minutes <= 0 or MINUTES_PER_DAY % minutes:
        raise ValueError(f"period_minutes must divide 1440, got {minutes}")
    return {
        f"{m // 60:02d}:{m % 60:02d}": f"{m // 60}:{m % 60:02d}"
        for m in range(0, MINUTES_PER_DAY, minutes)
    }


def interval_period_order(minutes: int) -> list[str]:
    return list(interval_time_cutoffs(minutes))


def period_matrix_path(directory: pathlib.Path, day: str, period: str) -> pathlib.Path:
    """Return the matrix file for one day and period.

    Windows forbids ``:`` in file names, so ``"07:00"`` is written
    ``odm_Monday_0700.csv``. Folders written before 2026-10-27 with the
    colon still read where the file system allows it.
    """
    path = directory / f"odm_{day}_{period.replace(':', '')}.csv"
    if ":" in period and not path.exists():
        legacy = directory / f"odm_{day}_{period}.csv"
        if legacy.exists():
            return legacy
    return path


def period_from_file_label(label: str) -> str:
    """``"0700"`` in a matrix file name back to ``"07:00"``."""
    if len(label) == 4 and label.isdigit():
        return f"{label[:2]}:{label[2:]}"
    return label


def _cutoff_minutes(cutoffs: dict[str, str]) -> list[int]:
    return [int(h) * 60 + int(m) for h, m in (v.split(":") for v in cutoffs.values())]


def classify_time_period(departure_time: str, cutoffs: dict[str, str]) -> str:
    """Return the period of an ``HH:MM[:SS]`` time.

    Times before the first cutoff are ``"early"``; unreadable times are
    ``"Unknown"``.
    """
    text = departure_time.strip()
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            clock = datetime.datetime.strptime(text, fmt).time()
            break
        except ValueError:
            continue
    else:
        return UNKNOWN_PERIOD
    minutes = clock.hour * 60 + clock.minute + clock.second / 60
    return str(classify_minutes(np.array([minutes]), cutoffs)[0])


def classify_times(times: pd.Series, cutoffs: dict[str, str]) -> pd.Series:
    """:func:`classify_time_period` over a column, parsing each time once."""
    mapping = {t: classify_time_period(str(t), cutoffs) for t in times.dropna().unique()}
    return times.map(mapping).fillna(UNKNOWN_PERIOD)


def classify_minutes(minutes: np.ndarray, cutoffs: dict[str, str]) -> np.ndarray:
    """:func:`classify_time_period` on minutes after midnight (NaN unknown)."""
    labels = np.array(["early", *cutoffs], dtype=object)
    missing = np.isnan(minutes)
    bounds = np.asarray(_cutoff_minutes(cutoffs), dtype=float)
    out = labels[np.searchsorted(bounds, np.where(missing, 0, minutes), side="right")]
    out[missing] = UNKNOWN_PERIOD
    return out


def period_minutes(cutoffs: dict[str, str]) -> dict[str, float]:
    """Each period's length in minutes.

    A period runs to the next cutoff and the last to midnight. The early
    period starts at 04:00, when the first trains run.
    """
    starts = _cutoff_minutes(cutoffs)
    lengths = dict(
        zip(cutoffs, np.diff([*starts, MINUTES_PER_DAY]).astype(float).tolist(), strict=True)
    )
    first = starts[0] if starts else MINUTES_PER_DAY
    early = first - 240 if first > 240 else first
    return {"early": float(early), **lengths} if early > 0 else lengths


def hhmm_minutes(hhmm: str | None) -> int | None:
    """Minutes after midnight for ``"HH:MM[:SS]"``, or None."""
    if not hhmm:
        return None
    hours, sep, mins = str(hhmm).strip().partition(":")
    if not sep or not (hours.isdigit() and mins[:2].isdigit()):
        return None
    return int(hours) * 60 + int(mins[:2])


def format_hhmm(minutes: float) -> str:
    """``"HH:MM"`` for minutes after midnight, wrapped into one day."""
    total = round(minutes) % MINUTES_PER_DAY
    return f"{total // 60:02d}:{total % 60:02d}"


def cif_hhmm(value: object) -> str:
    """Write a timetable time such as ``"0831H"`` as ``"08:31"`` (``""`` if unreadable)."""
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    if len(digits) < 4:
        return ""
    return f"{digits[:2]}:{digits[2:4]}"


def cif_time_minutes(value: object) -> int | None:
    """Minutes after midnight for a timetable time such as ``"0905H"``."""
    text = "" if value is None else str(value).strip()
    if len(text) < 4 or not text[:4].isdigit():
        return None
    return int(text[:2]) * 60 + int(text[2:4])


def cif_minutes(times: pd.Series) -> np.ndarray:
    """Vectorised :func:`cif_time_minutes`, NaN where unreadable."""
    digits = times.astype("string").fillna("").str.strip().str[:4]
    ok = digits.str.fullmatch(r"\d{4}").fillna(value=False).to_numpy(dtype=bool)
    out = np.full(len(digits), np.nan)
    if ok.any():
        picked = digits[ok]
        out[ok] = (
            picked.str[:2].astype(int).to_numpy() * 60 + picked.str[2:4].astype(int).to_numpy()
        )
    return out


def elapsed(depart: float, arrive: float) -> float:
    """Minutes from departure to arrival, wrapping at midnight."""
    return arrive - depart if arrive >= depart else arrive - depart + MINUTES_PER_DAY


def to_per_day(annual: pd.DataFrame) -> pd.DataFrame:
    """Convert an annual day x period grid to trips on one typical day."""
    return annual / WEEKDAYS_PER_YEAR
