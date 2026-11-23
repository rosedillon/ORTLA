"""Put candidate services to the working-timetable path check.

:mod:`ortla.path_check` works in TIPLOCs, half-minutes and schedules. This module gives it a
candidate's stations and public times, and gives back rows a planner can read.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING

import pandas as pd

from ortla import path_check
from ortla.clock import MINUTES_PER_DAY, format_hhmm, hhmm_minutes
from ortla.service_lookup import StationTiplocs, crs_to_tiploc_map

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ortla.bplan_geography import BplanTimingGraph
    from ortla.config import CandidateService, PathCheckConfig, ServiceStop

logger = logging.getLogger(__name__)

PATH_COLUMNS = [
    "candidate_name",
    "day",
    "shift_minutes",
    "departure",
    "path_clear",
    "rule",
    "place",
    "time",
    "train",
    "train_kind",
    "detail",
    "unchecked_platforms",
]


def call_times(stops: Sequence[ServiceStop]) -> list[tuple[int, int]] | None:
    """Return each stop's (arrive, depart) in half-minutes, running past midnight."""
    times: list[tuple[int, int]] = []
    latest = 0
    for stop in stops:
        arrive = hhmm_minutes(stop.arrival or stop.departure)
        depart = hhmm_minutes(stop.departure or stop.arrival)
        if arrive is None or depart is None:
            return None
        pair = []
        for minutes in (arrive, depart):
            at = minutes
            while times and at * 2 < latest - MINUTES_PER_DAY:
                at += MINUTES_PER_DAY
            latest = max(latest, round(at * 2))
            pair.append(round(at * 2))
        times.append((pair[0], pair[1]))
    return times


@dataclasses.dataclass
class CandidatePaths:
    """A path checker and the TIPLOCs of each station in the candidates' timetable."""

    checker: path_check.PathChecker
    places: dict[str, frozenset[str]]
    # Each station's busiest TIPLOC, its main platforms.
    main: dict[str, frozenset[str]] = dataclasses.field(default_factory=dict)
    # Which of them each operator's trains use.
    platforms: StationTiplocs | None = None

    @classmethod
    def build(
        cls,
        config: PathCheckConfig,
        calls: pd.DataFrame,
        reference_date: pd.Timestamp,
        graph: BplanTimingGraph,
        bplan_output_path: object,
    ) -> CandidatePaths:
        """Read the working timetable and its rules. ``calls`` have recoded station codes."""
        logger.info("Loading working timetable from %s ...", config.cif_path)
        timetable = path_check.WorkingTimetable.read(config.cif_path, reference_date)
        rules = load_rules(config, bplan_output_path)
        known = calls[["crs", "tiploc"]].dropna().astype(str)
        known = known.assign(
            crs=known["crs"].str.strip().str.upper(),
            tiploc=known["tiploc"].str.strip().str.upper(),
        )
        places, main = {}, {}
        for crs, own in known.groupby("crs"):
            places[crs] = frozenset(own["tiploc"])
            main[crs] = frozenset([own["tiploc"].value_counts().idxmax()])
        checker = path_check.PathChecker(timetable, rules, graph)
        return cls(checker, places, main, crs_to_tiploc_map(calls))

    def find_calls(
        self,
        calls: Sequence[tuple[str, int, int]],
        day: int,
        traction: str = "",
        ignore: frozenset[int] = frozenset(),
        hints: Sequence[list[str]] = (),
        own: dict[str, str] | None = None,
    ) -> path_check.PathResult:
        """Find a path through ``(station, arrive, depart)`` calls.

        The check leaves a station from the platform group it arrived at, and the commonest way
        in can be a bay with no way on (Reading 4A and 4B towards Oxford). So where that finds
        no path, the platforms the service's operator uses (``own``) are tried, then each
        station's main ones.
        """
        stations = [station.strip().upper() for station, _, _ in calls]

        def through(groups: dict[str, frozenset[str]]) -> path_check.PathResult:
            timed = [
                (groups.get(station, frozenset()), arrive, depart)
                for station, (_, arrive, depart) in zip(stations, calls, strict=True)
            ]
            return self.checker.find(timed, day, traction, ignore, hints)

        found = through(self.places)
        if found.clear or all(len(self.places.get(x, ())) < 2 for x in stations):
            return found
        by_own = through({c: frozenset([t]) for c, t in own.items()}) if own else None
        if by_own is not None and by_own.clear:
            return by_own
        by_main = through(self.main)
        if by_main.clear:
            return by_main
        tries = [x for x in (by_own, by_main, found) if x is not None]
        return max(tries, key=_says_most)

    def find(
        self,
        candidate: CandidateService,
        day: str,
        shift_minutes: float = 0.0,
        ignore: frozenset[int] = frozenset(),
        hints: Sequence[list[str]] = (),
    ) -> path_check.PathResult | None:
        """Find a path on ``day``; None where there is nothing to check."""
        times = call_times(candidate.stops)
        if candidate.new_infrastructure or times is None:
            return None
        move = round(2 * shift_minutes)
        calls = [
            (stop.station, arrive + move, depart + move)
            for stop, (arrive, depart) in zip(candidate.stops, times, strict=True)
        ]
        number = path_check.day_numbers([day])[0]
        own = None
        if self.platforms is not None:
            stations = [stop.station for stop in candidate.stops]
            own = self.platforms.for_service(candidate.operator, stations)
        return self.find_calls(calls, number, candidate.traction_code, ignore, hints, own)

    def check(
        self,
        candidate: CandidateService,
        shifts: Sequence[float] = (0.0,),
        *,
        ignore: frozenset[int] = frozenset(),
        hints: Sequence[list[str]] = (),
    ) -> pd.DataFrame:
        """Check each shift on each running day: one row a shift and day, with what stops it."""
        times = call_times(candidate.stops)
        rows = []
        for shift in shifts:
            for day in candidate.days_of_week:
                found = self.find(candidate, day, shift, ignore, hints)
                if found is None or times is None:
                    continue
                rows.append(self.row(found, candidate.name, day, shift, times[0][1] / 2 + shift))
        return pd.DataFrame(rows, columns=PATH_COLUMNS)

    @staticmethod
    def row(
        found: path_check.PathResult, name: str, day: str, shift: float, departs: float
    ) -> dict[str, object]:
        """Write one result as a row."""
        stopped = found.conflict
        return {
            "candidate_name": name,
            "day": day,
            "shift_minutes": round(shift, 1),
            "departure": format_hhmm(departs),
            "path_clear": found.clear,
            "rule": stopped.rule if stopped else "",
            "place": stopped.place if stopped else "",
            "time": stopped.time if stopped else "",
            "train": stopped.train if stopped else "",
            "train_kind": stopped.kind if stopped else "",
            "detail": stopped.detail if stopped else "",
            "unchecked_platforms": found.unchecked_platforms,
        }

    def check_together(self, candidates: Sequence[CandidateService]) -> pd.DataFrame:
        """Check trains that run together: each also keeps clear of those before it in the list.

        The trains given paths are added to this checker's timetable, so use it for one package.
        """
        frames = []
        for candidate in candidates:
            frames.append(self.check(candidate))
            for day in candidate.days_of_week:
                found = self.find(candidate, day)
                if found is not None and found.clear:
                    self.occupy(candidate.name, day, found.points)
        return pd.concat(frames, ignore_index=True) if frames else self._none()

    def occupy(self, name: str, day: str, points: Sequence[path_check.Point]) -> None:
        """Put a timed train in the timetable for later candidates to keep clear of."""
        timetable = self.checker.timetable
        train = path_check.Train(
            name, "", name, "proposed", 1 << path_check.day_numbers([day])[0], list(points)
        )
        timetable.trains.append(train)
        timetable._index(len(timetable.trains) - 1, train, None)
        self.checker._cache.clear()

    @staticmethod
    def _none() -> pd.DataFrame:
        return pd.DataFrame(columns=PATH_COLUMNS)


def _says_most(failed: path_check.PathResult) -> tuple[bool, int]:
    """Order failed tries by how much each says about what is really in the way.

    A train in the way says more than platforms with no route on or too long a run from them,
    and then so does having got further along. On a tie the first tried is kept.
    """
    rule = failed.conflict.rule if failed.conflict else ""
    return rule not in ("no route", "running time"), len(failed.points)


def load_rules(config: PathCheckConfig, bplan_output_path: object) -> path_check.Rules:
    """Read the saved rules, or learn them from the earlier timetable and save them."""
    if config.rules_path.exists():
        return path_check.Rules.load(config.rules_path)
    if config.learn_from is None or config.learn_date is None:
        raise ValueError(
            f"path_check: no rules at {config.rules_path}. Set learn_from (an earlier CIF) and "
            "learn_date to learn them."
        )
    if config.learn_from.resolve() == config.cif_path.resolve():
        raise ValueError("path_check: learn_from must be an earlier timetable than cif_path.")
    logger.info("Learning path rules from %s ...", config.learn_from)
    layout = (
        path_check.TrackLayout(config.track_shapefile, bplan_output_path)
        if config.track_shapefile is not None
        else None
    )
    earlier = path_check.WorkingTimetable.read(config.learn_from, pd.Timestamp(config.learn_date))
    rules = path_check.learn_rules(earlier, layout=layout)
    rules.save(config.rules_path)
    return rules
