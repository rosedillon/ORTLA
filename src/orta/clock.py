"""Clock times, weekdays and the periods a day is split into.

The timetable writes times as ``HHMM`` (``"0831"``, or ``"0831H"`` for a
half minute), the loading data and the candidate services as ``HH:MM``
or ``HH:MM:SS``. These helpers read both, label a time with its period
and convert between annual and typical-day figures.
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

# How often each weekday comes round in a year (365.25 / 7, about 52.2).
# A period matrix holds a whole year's trips for one weekday, so
# dividing by this gives the trips on one typical day of that kind.
WEEKDAYS_PER_YEAR = 365.25 / 7

MINUTES_PER_DAY = 24 * 60

# The label for a time that could not be read.
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
    """Cutoffs for 24 one-hour periods labelled ``HH:00``.

    The first cutoff is 00:00, so nothing falls before it and there is
    no ``early`` period: every departure lands in its own clock hour.

    Returns
    -------
    dict[str, str]
        Period label -> start time.
    """
    return {f"{hour:02d}:00": f"{hour}:00" for hour in range(24)}


def hourly_period_order() -> list[str]:
    """Labels of :func:`hourly_time_cutoffs` in clock order.

    Returns
    -------
    list[str]
        ``"00:00"`` to ``"23:00"``.
    """
    return [f"{hour:02d}:00" for hour in range(24)]


def interval_time_cutoffs(minutes: int) -> dict[str, str]:
    """Cutoffs for ``1440 / minutes`` equal periods labelled ``HH:MM``.

    The first cutoff is 00:00, so every departure lands in a period and
    there is no ``early`` one. Case 06 found the five named periods
    capture 60.6% of the variation in load by departure time within a
    corridor, against 87.4% at the exact minute and 85.7% for
    half-hourly periods, which is why 30 is the interval worth using.

    Parameters
    ----------
    minutes : int
        Length of each period; must divide 1440.

    Returns
    -------
    dict[str, str]
        Period label -> start time.

    Raises
    ------
    ValueError
        If ``minutes`` does not divide the day.
    """
    if minutes <= 0 or MINUTES_PER_DAY % minutes:
        raise ValueError(f"period_minutes must divide 1440, got {minutes}")
    return {
        f"{m // 60:02d}:{m % 60:02d}": f"{m // 60}:{m % 60:02d}"
        for m in range(0, MINUTES_PER_DAY, minutes)
    }


def interval_period_order(minutes: int) -> list[str]:
    """Labels of :func:`interval_time_cutoffs` in clock order.

    Parameters
    ----------
    minutes : int
        Length of each period.

    Returns
    -------
    list[str]
        The period labels, earliest first.
    """
    return list(interval_time_cutoffs(minutes))


def period_matrix_path(directory: pathlib.Path, day: str, period: str) -> pathlib.Path:
    """Return the file holding the matrix for one day and period.

    Windows does not allow ``:`` in a file name, so an hourly or
    half-hourly label such as ``"07:00"`` is written without it
    (``odm_Monday_0700.csv``). A folder written with the colon, as
    matrices were before 2026-10-27, still reads where the file system
    allows it.

    Parameters
    ----------
    directory : pathlib.Path
        Folder of period matrices.
    day, period : str
        Day of the week and period label.

    Returns
    -------
    pathlib.Path
        Where the matrix is, or should be written.
    """
    path = directory / f"odm_{day}_{period.replace(':', '')}.csv"
    if ":" in period and not path.exists():
        legacy = directory / f"odm_{day}_{period}.csv"
        if legacy.exists():
            return legacy
    return path


def period_from_file_label(label: str) -> str:
    """Read the period in a matrix file name: ``"0700"`` as ``"07:00"``.

    Parameters
    ----------
    label : str
        Period part of a matrix file name.

    Returns
    -------
    str
        The period label.
    """
    if len(label) == 4 and label.isdigit():
        return f"{label[:2]}:{label[2:]}"
    return label


def classify_time_period(departure_time: str, cutoffs: dict[str, str]) -> str:
    """Label a departure time with its time period.

    Parameters
    ----------
    departure_time : str
        Time as ``HH:MM:SS`` or ``HH:MM``.
    cutoffs : dict[str, str]
        Period name -> start time (``"HH:MM"``), in order. Times before
        the first cutoff are ``"early"``.

    Returns
    -------
    str
        The period, or ``"Unknown"`` when the time cannot be read.
    """
    try:
        clock = datetime.datetime.strptime(departure_time.strip(), "%H:%M:%S").time()
    except ValueError:
        try:
            clock = datetime.datetime.strptime(departure_time.strip(), "%H:%M").time()
        except ValueError:
            return UNKNOWN_PERIOD

    names = list(cutoffs.keys())
    starts = [datetime.datetime.strptime(v, "%H:%M").time() for v in cutoffs.values()]
    if clock < starts[0]:
        return "early"
    for i in range(len(starts) - 1):
        if clock < starts[i + 1]:
            return names[i]
    return names[-1]


def classify_times(times: pd.Series, cutoffs: dict[str, str]) -> pd.Series:
    """:func:`classify_time_period` over a column, each time once.

    Parameters
    ----------
    times : pd.Series
        Times as text.
    cutoffs : dict[str, str]
        Period name -> start time.

    Returns
    -------
    pd.Series
        Period labels, ``"Unknown"`` where a time is missing.
    """
    mapping = {t: classify_time_period(str(t), cutoffs) for t in times.dropna().unique()}
    return times.map(mapping).fillna(UNKNOWN_PERIOD)


def classify_minutes(minutes: np.ndarray, cutoffs: dict[str, str]) -> np.ndarray:
    """:func:`classify_time_period` on minutes after midnight.

    Parameters
    ----------
    minutes : np.ndarray
        Minutes after midnight, NaN where unknown.
    cutoffs : dict[str, str]
        Period name -> start time (``"HH:MM"``), in order.

    Returns
    -------
    np.ndarray
        Period labels: ``"early"`` before the first cutoff and
        ``"Unknown"`` where the time is missing.
    """
    names = list(cutoffs)
    bounds = []
    for value in cutoffs.values():
        hours, mins = value.split(":")
        bounds.append(int(hours) * 60 + int(mins))
    labels = np.array(["early", *names], dtype=object)
    missing = np.isnan(minutes)
    index = np.searchsorted(
        np.asarray(bounds, dtype=float), np.where(missing, 0, minutes), side="right"
    )
    out = labels[index]
    out[missing] = UNKNOWN_PERIOD
    return out


def period_minutes(cutoffs: dict[str, str]) -> dict[str, float]:
    """Each period's length in minutes.

    A period runs to the next cutoff and the last one to midnight. The
    early period runs from 04:00, when the first trains do, so the named
    periods give :data:`PERIOD_MINUTES`. Half-hourly cutoffs start at
    00:00 and have no early period.

    Parameters
    ----------
    cutoffs : dict[str, str]
        Period name -> start time, in order.

    Returns
    -------
    dict[str, float]
        Period name -> minutes.
    """
    starts = []
    for value in cutoffs.values():
        hours, mins = value.split(":")
        starts.append(int(hours) * 60 + int(mins))
    lengths = dict(
        zip(cutoffs, np.diff([*starts, MINUTES_PER_DAY]).astype(float).tolist(), strict=True)
    )
    first = starts[0] if starts else MINUTES_PER_DAY
    early = first - 240 if first > 240 else first
    return {"early": float(early), **lengths} if early > 0 else lengths


def hhmm_minutes(hhmm: str | None) -> int | None:
    """Minutes after midnight for an ``"HH:MM"`` string.

    Seconds after the minutes (``"08:31:00"``) are ignored.

    Parameters
    ----------
    hhmm : str | None
        Clock time.

    Returns
    -------
    int | None
        Minutes, or ``None`` when the text is not a clock time.
    """
    if not hhmm:
        return None
    hours, sep, mins = str(hhmm).strip().partition(":")
    if not sep or not (hours.isdigit() and mins[:2].isdigit()):
        return None
    return int(hours) * 60 + int(mins[:2])


def format_hhmm(minutes: float) -> str:
    """``"HH:MM"`` for minutes after midnight, wrapped into one day.

    Parameters
    ----------
    minutes : float
        Minutes after midnight; rounded to the minute.

    Returns
    -------
    str
        The clock time, on the following day where it passes midnight.
    """
    total = round(minutes) % MINUTES_PER_DAY
    return f"{total // 60:02d}:{total % 60:02d}"


def cif_hhmm(value: object) -> str:
    """Write a timetable time such as ``"0831H"`` as ``"HH:MM"``.

    Parameters
    ----------
    value : object
        The time as the timetable writes it.

    Returns
    -------
    str
        ``"HH:MM"``, or ``""`` when there are fewer than four digits.
    """
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    if len(digits) < 4:
        return ""
    return f"{digits[:2]}:{digits[2:4]}"


def cif_time_minutes(value: object) -> int | None:
    """Minutes after midnight for a timetable time such as ``"0905"``.

    Parameters
    ----------
    value : object
        The time as the timetable writes it; a half-minute ``H`` is
        dropped.

    Returns
    -------
    int | None
        Minutes, or ``None`` without four leading digits.
    """
    text = "" if value is None else str(value).strip()
    if len(text) < 4 or not text[:4].isdigit():
        return None
    return int(text[:2]) * 60 + int(text[2:4])


def cif_minutes(times: pd.Series) -> np.ndarray:
    """Vectorised :func:`cif_time_minutes`.

    Parameters
    ----------
    times : pd.Series
        Timetable times such as ``"0831"`` or ``"0831H"``.

    Returns
    -------
    np.ndarray
        Minutes after midnight, NaN without four leading digits.
    """
    text = times.astype("string").fillna("").str.strip()
    digits = text.str[:4]
    ok = digits.str.fullmatch(r"\d{4}").fillna(value=False).to_numpy(dtype=bool)
    out = np.full(len(text), np.nan)
    if ok.any():
        picked = digits[ok]
        out[ok] = (
            picked.str[:2].astype(int).to_numpy() * 60 + picked.str[2:4].astype(int).to_numpy()
        )
    return out


def elapsed(depart: float, arrive: float) -> float:
    """Minutes from a departure to an arrival, wrapping at midnight.

    Parameters
    ----------
    depart, arrive : float
        Minutes after midnight.

    Returns
    -------
    float
        The journey time.
    """
    return arrive - depart if arrive >= depart else arrive - depart + MINUTES_PER_DAY


def to_per_day(annual: pd.DataFrame) -> pd.DataFrame:
    """Turn an annual day x period grid into trips on one typical day.

    Each cell of an annual grid is the year's trips on that weekday and
    period, so dividing by :data:`WEEKDAYS_PER_YEAR` gives the trips on
    one typical day of that weekday.

    Parameters
    ----------
    annual : pd.DataFrame
        Annual trips.

    Returns
    -------
    pd.DataFrame
        Trips per typical day, with the same shape and labels.
    """
    return annual / WEEKDAYS_PER_YEAR
