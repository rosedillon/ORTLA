"""Find a train a path through the working timetable.

The headway check in :mod:`ortla.feasibility` compares public departures of trains that call at
both ends of a leg. This reads the working timetable instead: every passenger, freight and empty
train, at every timing point it passes, with its running line and platform. A train has a path
when, at the times asked for at its calls,

* on each link it is clear of trains on the same line at both ends, and passes none inside it;
* it meets no opposing train on a single line;
* it crosses no conflicting movement at a junction or station throat;
* a platform it can reach is free at each call.

Open data have no Timetable Planning Rules or register of conflicting movements, so those are
learnt from an earlier timetable (:func:`learn_rules`): the tightest separations it repeats, and
the pairs of movements it never schedules together though chance often would. A single line is
one that Network Rail's track model shows as single and on which the earlier timetable never
overlaps opposing trains. A rule with no evidence is a default or is not applied, so a path found
here is one that breaks none of these rules. It isn't an offered path: signalling, gauge, power
supply, platform length and the train's own diagram are not checked.

Times are whole half-minutes from midnight of the day a train starts.
"""

from __future__ import annotations

import bisect
import collections
import dataclasses
import heapq
import itertools
import json
import logging
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString
from shapely.geometry import Point as Position

from ortla import cif_archive
from ortla.clock import WEEKDAY_INDEX

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Iterable, Sequence

    from ortla.bplan_geography import BplanTimingGraph

logger = logging.getLogger(__name__)

DAY = 2880  # half-minutes
_STP_PRIORITY = {"C": 0, "N": 1, "O": 2, "P": 3}
_NOT_TRAINS = frozenset({"B", "S", "4", "5"})  # train status: bus and ship
_NOT_TRAIN_CATEGORIES = frozenset({"BR", "BS", "SS"})
_LONGEST_TURNROUND = 480
_LONGEST_PATHING = 120
_MOST_ROUTES = 4
_SAME_PATH_POINTS = 3
_STOP = "|"  # the side of a movement that is a stand at the location


@dataclasses.dataclass(frozen=True)
class Margins:
    """The defaults, and how much evidence a learnt rule needs."""

    headway: int = 6
    platform: int = 4
    junction: int = 4
    repeats: int = 2  # pairs of schedules that must show a tighter value
    shared: int = 3  # pairs that must overlap before a platform or link takes two
    junction_expected: float = 8.0


@dataclasses.dataclass(frozen=True)
class Point:
    """One timing point of a train."""

    tiploc: str
    arrive: int
    depart: int
    stops: bool
    platform: str = ""
    line: str = ""  # the running line it leaves on
    public: bool = False


@dataclasses.dataclass
class Train:
    """One schedule, and the days of the reference week it runs."""

    uid: str
    operator: str
    headcode: str
    kind: str
    days: int
    points: list[Point]
    as_required: bool = False

    def runs(self, day: int) -> bool:
        """Whether it starts on ``day`` (Monday 0)."""
        return bool(self.days >> (day % 7) & 1)

    def describe(self) -> str:
        """Operator, headcode and kind, for a planner to look up."""
        name = " ".join(x for x in (self.operator, self.headcode) if x) or self.uid
        return f"{name} ({self.kind}{', runs as required' if self.as_required else ''})"


def half_minutes(value: str | None) -> int | None:
    """Half-minutes in a working time (``"1026H"``), or None."""
    if not value:
        return None
    text = value.strip()
    if not text[:4].isdigit():
        return None
    return int(text[:2]) * 120 + int(text[2:4]) * 2 + text.endswith("H")


def clock(at: int) -> str:
    """``at`` as a working time of day."""
    at %= DAY
    return f"{at // 120:02d}:{at % 120 // 2:02d}{'½' if at % 2 else ''}"


def _kind(status: str, category: str) -> str:
    if category.startswith("E"):
        return "empty"
    if status in {"F", "2", "3"} or category[:1] in {"A", "B", "D", "H", "J", "M", "P", "Z"}:
        return "freight or other"
    return "passenger"


def _points(locations: list[dict]) -> list[Point]:
    points: list[Point] = []
    latest = None
    for place in locations:
        arrive, depart, passing = (
            half_minutes(place.get(key)) for key in ("arrival", "departure", "pass")
        )
        start = next((t for t in (passing, arrive, depart) if t is not None), None)
        if start is None:
            continue
        end = next((t for t in (passing, depart, arrive) if t is not None), start)
        if latest is not None:
            while start < latest - DAY // 2:
                start += DAY
        while end < start:
            end += DAY
        latest = end
        points.append(
            Point(
                str(place["tiploc_code"]).strip().upper(),
                start,
                end,
                stops=passing is None,
                platform=(place.get("platform") or "").strip(),
                line=(place.get("line") or "").strip(),
                public=bool(place.get("public_arrival") or place.get("public_departure")),
            )
        )
    return points


class WorkingTimetable:
    """Every train of the reference date's week, indexed by link, movement and platform."""

    def __init__(
        self,
        trains: list[Train],
        related: dict[int, set[int]] | None = None,
        turnrounds: dict[int, int] | None = None,
    ) -> None:
        """Index ``trains``. ``turnrounds`` extends a train's last platform stay to its next."""
        self.trains = trains
        self.related = related or {}
        self.links: dict[tuple[str, str], list[tuple[int, int, str, int]]] = (
            collections.defaultdict(list)
        )
        self.moves: dict[tuple[str, str, str], list[tuple[int, int]]] = collections.defaultdict(
            list
        )
        self.stays: dict[tuple[str, str], list[tuple[int, int, int]]] = collections.defaultdict(
            list
        )
        self.platforms_from: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
        self.platforms_to: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
        self.lines: dict[tuple[str, str], collections.Counter[str]] = collections.defaultdict(
            collections.Counter
        )
        self.next_places: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
        self.places_after: dict[str, set[str]] = collections.defaultdict(set)
        self.quickest: dict[tuple[str, str], int] = {}
        self.calls_at: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
        for number, train in enumerate(trains):
            self._index(number, train, (turnrounds or {}).get(number))

    def _index(  # noqa: C901 (one pass over the points, as it is run a million times)
        self, number: int, train: Train, stays_until: int | None
    ) -> None:
        points, last = train.points, len(train.points) - 1
        passenger = train.kind == "passenger"
        for i, here in enumerate(points):
            before = points[i - 1].tiploc if i else _STOP
            after = points[i + 1].tiploc if i < last else _STOP
            if here.stops:
                if i:
                    self.moves[before, here.tiploc, _STOP].append((here.arrive, number))
                if i < last:
                    self.moves[_STOP, here.tiploc, after].append((here.depart, number))
                if here.public and passenger:
                    self.calls_at[here.tiploc].append((number, i))
            else:
                self.moves[before, here.tiploc, after].append((here.depart, number))
            if here.platform:
                until = here.depart
                if i == last and stays_until is not None:
                    until = max(until, stays_until)
                self.stays[here.tiploc, here.platform].append((here.arrive, until, number))
                if here.public and passenger:
                    self.platforms_from[here.tiploc, before].add(here.platform)
                    self.platforms_to[here.tiploc, after].add(here.platform)
            if i == last:
                continue
            link = (here.tiploc, after)
            enter, leave = here.depart, points[i + 1].arrive
            self.links[link].append((enter, leave, here.line, number))
            self.lines[link][here.line] += 1 + 1000 * passenger  # prefer lines passengers use
            self.quickest[link] = min(self.quickest.get(link, DAY), max(leave - enter, 1))
            self.places_after[here.tiploc].add(after)
            if i + 1 < last:
                self.next_places[link].add(points[i + 2].tiploc)

    @classmethod
    def read(cls, cif_path: pathlib.Path, reference_date: pd.Timestamp) -> WorkingTimetable:
        """Read the trains running in ``reference_date``'s week from a CIF."""
        monday = reference_date - pd.Timedelta(days=reference_date.weekday())
        dates = [(monday + pd.Timedelta(days=d)).strftime("%Y-%m-%d") for d in range(7)]
        chosen: dict[tuple[str, int], tuple[int, str, int]] = {}
        kept: list[Train | None] = []
        for record in cif_archive.iter_schedule_records(cif_path):
            runs = record.get("schedule_days_runs") or ""
            start = (record.get("schedule_start_date") or "")[:10]
            end = (record.get("schedule_end_date") or "")[:10]
            days = [
                d
                for d, date in enumerate(dates)
                if len(runs) > d and runs[d] == "1" and start <= date <= end
            ]
            if not days or record.get("transaction_type") == "Delete":
                continue
            stp = record.get("CIF_stp_indicator") or "P"
            uid = record.get("CIF_train_uid") or ""
            segment = record.get("schedule_segment") or {}
            status = record.get("train_status") or ""
            category = segment.get("CIF_train_category") or ""
            train = None
            rail = status not in _NOT_TRAINS and category not in _NOT_TRAIN_CATEGORIES
            if rail and stp != "C":
                points = _points(segment.get("schedule_location") or [])
                if len(points) > 1:
                    train = Train(
                        uid,
                        record.get("atoc_code") or "",
                        segment.get("signalling_id") or "",
                        _kind(status, category),
                        0,
                        points,
                        "Q" in (segment.get("CIF_operating_characteristics") or ""),
                    )
            # The highest-priority schedule of a train on a day is the one that runs, a
            # cancellation or a bus in its place included.
            rank = _STP_PRIORITY.get(stp, 3)
            number = len(kept)
            kept.append(train)
            for day in days:
                held = chosen.get((uid, day))
                if held is None or rank < held[0] or (rank == held[0] and start > held[1]):
                    chosen[uid, day] = (rank, start, number)
        for (_uid, day), (_rank, _start, number) in chosen.items():
            if (train := kept[number]) is not None:
                train.days |= 1 << day
        trains = [train for train in kept if train is not None and train.days]
        related, turnrounds = cls._associations(cif_path, trains, dates)
        cls._alternatives(trains, related)
        logger.info("Working timetable: %d trains in the week of %s.", len(trains), dates[0])
        return cls(trains, related, turnrounds)

    @staticmethod
    def _associations(  # noqa: C901
        cif_path: pathlib.Path, trains: list[Train], dates: list[str]
    ) -> tuple[dict[int, set[int]], dict[int, int]]:
        by_uid: dict[str, list[int]] = collections.defaultdict(list)
        for number, train in enumerate(trains):
            by_uid[train.uid].append(number)
        related: dict[int, set[int]] = collections.defaultdict(set)
        turnrounds: dict[int, int] = {}
        for link in cif_archive.iter_association_records(cif_path):
            start, end = link.get("assoc_start_date") or "", link.get("assoc_end_date") or "9999"
            if start > dates[-1] or end < dates[0]:
                continue
            place = (link.get("location") or "").strip().upper()
            next_day = link.get("date_indicator")
            offset = DAY if next_day == "N" else -DAY if next_day == "P" else 0
            for main in by_uid.get(link.get("main_train_uid") or "", ()):
                for other in by_uid.get(link.get("assoc_train_uid") or "", ()):
                    if not offset and not trains[main].days & trains[other].days:
                        continue
                    related[main].add(other)
                    related[other].add(main)
                    if link.get("category") != "NP":
                        continue
                    last, first = trains[main].points[-1], trains[other].points[0]
                    if last.tiploc != place or first.tiploc != place:
                        continue
                    if last.platform != first.platform:
                        continue
                    leaves = first.depart + offset
                    if 0 <= leaves - last.arrive <= _LONGEST_TURNROUND:
                        turnrounds[main] = leaves
        return related, turnrounds

    @staticmethod
    def _alternatives(trains: list[Train], related: dict[int, set[int]]) -> None:
        """Relate an as-required path to a train timed identically, as its alternative."""
        optional = [n for n, train in enumerate(trains) if train.as_required]
        at: dict[tuple[str, int], list[int]] = {
            (p.tiploc, p.depart): [] for n in optional for p in trains[n].points
        }
        for number, train in enumerate(trains):
            for point in train.points:
                if (point.tiploc, point.depart) in at:
                    at[point.tiploc, point.depart].append(number)
        for number in optional:
            train = trains[number]
            same = collections.Counter(
                other
                for point in train.points
                for other in at[point.tiploc, point.depart]
                if other != number and trains[other].days & train.days
            )
            for other, points in same.items():
                if points >= _SAME_PATH_POINTS:
                    related[number].add(other)
                    related[other].add(number)


def _link_key(a: str, b: str) -> str:
    return f"{a}>{b}"


def _move_key(move: tuple[str, str, str]) -> str:
    return ">".join(move)


@dataclasses.dataclass
class Rules:
    """What an earlier timetable says the railway will take."""

    margins: Margins = dataclasses.field(default_factory=Margins)
    enter: dict[str, int] = dataclasses.field(default_factory=dict)
    leave: dict[str, int] = dataclasses.field(default_factory=dict)
    overtakes: set[str] = dataclasses.field(default_factory=set)
    single: set[str] = dataclasses.field(default_factory=set)
    crossing: dict[str, dict[str, int]] = dataclasses.field(default_factory=dict)
    reoccupy: dict[str, int] = dataclasses.field(default_factory=dict)
    shared: set[str] = dataclasses.field(default_factory=set)

    def save(self, path: pathlib.Path) -> None:
        """Write the rules as JSON."""
        body = dataclasses.asdict(self)
        for key in ("overtakes", "single", "shared"):
            body[key] = sorted(body[key])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(body, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: pathlib.Path) -> Rules:
        """Read what :meth:`save` wrote."""
        body = json.loads(path.read_text(encoding="utf-8"))
        body["margins"] = Margins(**body["margins"])
        for key in ("overtakes", "single", "shared"):
            body[key] = set(body[key])
        return cls(**body)


def _repeated_minimum(gaps: list[int], default: int, repeats: int) -> int:
    """Return the gap the timetable repeats, where that is tighter than the default."""
    if len(gaps) < repeats:
        return default
    return max(0, min(default, sorted(gaps)[repeats - 1]))


class TrackLayout:
    """Which links Network Rail's track model shows as single."""

    _SAMPLES = (0.2, 0.35, 0.5, 0.65, 0.8)
    _WITHIN_METRES = 250
    _MOST_EASTING = 700_000  # BPLAN writes 999999 where it has no position
    _MOST_NORTHING = 999_999

    def __init__(self, track_shapefile: pathlib.Path, bplan_output_path: pathlib.Path) -> None:
        """Read the track centre lines and BPLAN's position for each timing point."""
        track = gpd.read_file(track_shapefile)
        self._track, self._index = track, track.sindex
        places = pd.read_csv(bplan_output_path / "bplan_locations.csv", dtype={"tiploc": str})
        east = pd.to_numeric(places["easting"], errors="coerce")
        north = pd.to_numeric(places["northing"], errors="coerce")
        known = east.between(1, self._MOST_EASTING) & north.between(1, self._MOST_NORTHING - 1)
        self._at = {
            code: Position(x, y)
            for code, x, y in zip(places["tiploc"][known], east[known], north[known], strict=True)
        }

    def single(self, link: tuple[str, str]) -> bool:
        """Whether the nearest route anywhere between the two has only a bidirectional track.

        Track IDs begin 1 and 2 for the up and down running lines and 3 for bidirectional ones.
        """
        if link[0] not in self._at or link[1] not in self._at:
            return False
        along = LineString([self._at[link[0]], self._at[link[1]]])
        for fraction in self._SAMPLES:
            middle = along.interpolate(fraction, normalized=True)
            near = self._index.query(middle, predicate="dwithin", distance=self._WITHIN_METRES)
            if not len(near):
                continue
            tracks = self._track.iloc[near]
            route = tracks["ELR"].iloc[int(np.argmin(tracks.distance(middle).to_numpy()))]
            kinds = {str(x)[:1] for x in tracks["TRACK_ID"][tracks["ELR"] == route]}
            if "3" in kinds and not {"1", "2"} <= kinds:
                return True
        return False


def learn_rules(
    timetable: WorkingTimetable,
    day: int = 2,
    margins: Margins | None = None,
    layout: TrackLayout | None = None,
) -> Rules:
    """Learn separations, single lines, conflicts and shared platforms.

    Evidence is taken from every day of the week, counting a pair of schedules once. Without a
    ``layout`` no link is taken to be single. ``day`` is
    the day whose traffic says how often an overlap would happen by chance. Paths that only run
    as required are left out: alternatives are planned on top of each other. Give it a timetable
    earlier than the one being checked, or the check is circular.
    """
    margins = margins or Margins()
    rules = Rules(margins)
    _learn_links(timetable, rules, layout)
    _learn_crossings(timetable, rules, day)
    _learn_platforms(timetable, rules)
    logger.info(
        "Path rules: %d links, %d single lines, %d crossing movements, %d shared platforms.",
        len(rules.enter),
        len(rules.single) // 2,
        len(rules.crossing),
        len(rules.shared),
    )
    return rules


def _planned(timetable: WorkingTimetable, number: int) -> bool:
    return not timetable.trains[number].as_required


def _unrelated(timetable: WorkingTimetable, a: int, b: int) -> bool:
    return (
        b not in timetable.related.get(a, ())
        and timetable.trains[a].uid != timetable.trains[b].uid
    )


def _learn_links(timetable: WorkingTimetable, rules: Rules, layout: TrackLayout | None) -> None:
    margins, trains = rules.margins, timetable.trains
    for link, events in timetable.links.items():
        rows = sorted(row for row in events if _planned(timetable, row[3]))
        if not rows:
            continue
        key = _link_key(*link)
        pairs: tuple[dict, dict] = ({}, {})
        for line in {row[2] for row in rows}:
            own = [row for row in rows if row[2] == line]
            for side in (0, 1):
                ordered = sorted(own, key=lambda row, side=side: row[side])
                for weekday in range(7):
                    running = [row for row in ordered if trains[row[3]].runs(weekday)]
                    for a, b in itertools.pairwise(running):
                        pairs[side][a[3], b[3]] = (b[side] - a[side], b[1] < a[1])
        kept = [
            [found for ids, found in own.items() if _unrelated(timetable, *ids)] for own in pairs
        ]
        for learnt, found, default in zip(
            (rules.enter, rules.leave), kept, (margins.headway, margins.headway), strict=True
        ):
            learnt[key] = _repeated_minimum([gap for gap, _ in found], default, margins.repeats)
        if sum(gap > 0 and passed for gap, passed in kept[0]) >= margins.shared:
            rules.overtakes.add(key)
        if layout is not None and link < (link[1], link[0]):
            _learn_single(timetable, rules, link, rows, layout)


def _learn_single(
    timetable: WorkingTimetable,
    rules: Rules,
    link: tuple[str, str],
    rows: list[tuple[int, int, str, int]],
    layout: TrackLayout,
) -> None:
    opposite, trains = (link[1], link[0]), timetable.trains
    back = sorted(r for r in timetable.links.get(opposite, ()) if _planned(timetable, r[3]))
    if not back or len({r[2] for r in rows}) > 1 or len({r[2] for r in back}) > 1:
        return  # a choice of running line is more than one track
    if not layout.single(link):
        return
    one, two = (np.array([row[:2] for row in own]) for own in (rows, back))
    meets = (one[:, None, 0] < two[None, :, 1]) & (two[None, :, 0] < one[:, None, 1])
    for i, j in np.argwhere(meets):
        a, b = rows[i][3], back[j][3]
        if trains[a].days & trains[b].days and _unrelated(timetable, a, b):
            return
    rules.single |= {_link_key(*link), _link_key(*opposite)}


def _learn_crossings(timetable: WorkingTimetable, rules: Rules, day: int) -> None:
    margin, trains = rules.margins.junction, timetable.trains
    at: dict[str, list[tuple]] = collections.defaultdict(list)
    for move, events in timetable.moves.items():
        planned = sorted(event for event in events if _planned(timetable, event[1]))
        today = [time for time, number in planned if trains[number].runs(day)]
        if today:
            at[move[1]].append((move, planned, today))
    for moves in at.values():
        if len(moves) < 2:
            continue
        times = [time for *_, today in moves for time in today]
        span = max(times) - min(times) + 1
        for (first, a, a_today), (second, b, b_today) in itertools.combinations(moves, 2):
            if len(a_today) * len(b_today) * 2 * margin / span < rules.margins.junction_expected:
                continue
            # Trains following or converging are separated on the link they share, and the two
            # directions of one route have a track each unless the line is single.
            if first[0] == second[0] != _STOP or first[2] == second[2] != _STOP:
                continue
            if first[0] == second[2] and first[2] == second[0]:
                continue
            if _ever_together(timetable, a, b, margin):
                continue
            rules.crossing.setdefault(_move_key(first), {})[_move_key(second)] = margin
            rules.crossing.setdefault(_move_key(second), {})[_move_key(first)] = margin


def _ever_together(
    timetable: WorkingTimetable, a: list[tuple[int, int]], b: list[tuple[int, int]], margin: int
) -> bool:
    trains, low = timetable.trains, 0
    for time, number in a:
        while low < len(b) and b[low][0] <= time - margin:
            low += 1
        at = low
        while at < len(b) and b[at][0] < time + margin:
            other = b[at][1]
            if trains[number].days & trains[other].days and _unrelated(timetable, number, other):
                return True
            at += 1
    return False


def _learn_platforms(timetable: WorkingTimetable, rules: Rules) -> None:
    trains = timetable.trains
    for (place, platform), events in timetable.stays.items():
        rows = sorted(row for row in events if _planned(timetable, row[2]))
        pairs: dict[tuple[int, int], int] = {}
        for weekday in range(7):
            holder = None  # whoever has been there latest, a long turnround included
            for row in rows:
                if not trains[row[2]].runs(weekday):
                    continue
                if holder is not None:
                    pairs[holder[2], row[2]] = row[0] - holder[1]
                if holder is None or row[1] >= holder[1]:
                    holder = row
        gaps = [gap for ids, gap in pairs.items() if _unrelated(timetable, *ids)]
        key = f"{place}:{platform}"
        if sum(gap < 0 for gap in gaps) >= rules.margins.shared:
            rules.shared.add(key)
        rules.reoccupy[key] = _repeated_minimum(
            [gap for gap in gaps if gap >= 0], rules.margins.platform, rules.margins.repeats
        )


@dataclasses.dataclass(frozen=True)
class Conflict:
    """What stops a path, and where."""

    rule: str
    place: str
    time: str
    train: str = ""
    kind: str = ""
    detail: str = ""


@dataclasses.dataclass
class PathResult:
    """A path, or the first thing that stops one."""

    points: list[Point]
    conflict: Conflict | None = None
    unchecked_platforms: int = 0

    @property
    def clear(self) -> bool:
        """Whether a path was found."""
        return self.conflict is None


class PathChecker:
    """Check timed trains, and find paths for untimed ones, against one working timetable."""

    def __init__(
        self,
        timetable: WorkingTimetable,
        rules: Rules,
        graph: BplanTimingGraph | None = None,
    ) -> None:
        """Check against ``timetable`` by ``rules``; ``graph`` supplies running times."""
        self.timetable, self.rules, self.graph = timetable, rules, graph
        self._cache: dict[tuple, tuple] = {}

    def excluded(self, numbers: Iterable[int]) -> frozenset[int]:
        """``numbers`` and the trains that form, join or divide from them."""
        out = set(numbers)
        for number in tuple(out):
            out |= self.timetable.related.get(number, set())
        return frozenset(out)

    def _on_day(self, table: str, key: tuple, day: int) -> tuple:
        """Return an index's events around ``day`` as columns by time, across midnight."""
        cached = self._cache.get((table, key, day))
        if cached is None:
            found = []
            for *values, number in getattr(self.timetable, table).get(key, ()):
                train = self.timetable.trains[number]
                times = values[:1] if table == "moves" else values[:2]
                for offset in (-1, 0, 1):
                    if train.runs(day + offset):
                        shifted = [t + offset * DAY for t in times]
                        found.append((*shifted, *values[len(times) :], number))
            found.sort()
            cached = tuple(np.array(column) for column in zip(*found, strict=True))
            self._cache[table, key, day] = cached
        return cached

    @staticmethod
    def _keep(numbers: np.ndarray, ignore: frozenset[int]) -> np.ndarray:
        return ~np.isin(numbers, list(ignore)) if ignore else np.ones(len(numbers), dtype=bool)

    def _link(
        self,
        link: tuple[str, str],
        line: str,
        enter: np.ndarray,
        leave: np.ndarray,
        day: int,
        ignore: frozenset[int],
    ) -> tuple[np.ndarray, str, int | None]:
        """Which timings of a link are clear, and what blocks the first that isn't."""
        key, rules = _link_key(*link), self.rules
        clear = np.ones(len(enter), dtype=bool)
        blame: tuple[str, int | None] = ("", None)
        same = self._on_day("links", link, day)
        if same:
            others_in, others_out, others_line, others = same
            first_gap = rules.enter.get(key, rules.margins.headway)
            last_gap = rules.leave.get(key, rules.margins.headway)
            longest = int((others_out - others_in).max())
            lo = np.searchsorted(others_in, enter.min() - first_gap - last_gap - longest)
            hi = np.searchsorted(others_in, leave.max() + first_gap + last_gap, side="right")
            near = (others_line[lo:hi] == line) & self._keep(others[lo:hi], ignore)
            if near.any():
                a, b = others_in[lo:hi][near], others_out[lo:hi][near]
                d_in, d_out = enter[:, None] - a, leave[:, None] - b
                tight = (np.abs(d_in) < first_gap) | (np.abs(d_out) < last_gap)
                passing = np.zeros_like(tight)
                if key not in rules.overtakes:
                    passing = d_in * d_out < 0
                hit = tight | passing
                if hit.any():
                    row, col = np.argwhere(hit)[0]
                    blame = (
                        "headway" if tight[row, col] else "overtaking",
                        int(others[lo:hi][near][col]),
                    )
                    clear = ~hit.any(axis=1)
        if key in rules.single:
            opposing = self._on_day("links", (link[1], link[0]), day)
            if opposing:
                others_in, others_out, _, others = opposing
                near = (others_in < leave.max()) & (others_out > enter.min())
                near &= self._keep(others, ignore)
                hit = (others_in[near] < leave[:, None]) & (others_out[near] > enter[:, None])
                if hit.any():
                    if blame[1] is None:
                        blame = ("single line", int(others[near][np.argwhere(hit)[0][1]]))
                    clear &= ~hit.any(axis=1)
        return clear, *blame

    def _move(
        self, move: tuple[str, str, str], at: np.ndarray, day: int, ignore: frozenset[int]
    ) -> tuple[np.ndarray, str, int | None]:
        clear, blocker = np.ones(len(at), dtype=bool), None
        for name, margin in self.rules.crossing.get(_move_key(move), {}).items():
            events = self._on_day("moves", tuple(name.split(">")), day)
            if not events:
                continue
            times, others = events
            near = (times > at.min() - margin) & (times < at.max() + margin)
            near &= self._keep(others, ignore)
            hit = np.abs(at[:, None] - times[near]) < margin
            if hit.any():
                if blocker is None:
                    blocker = int(others[near][np.argwhere(hit)[0][1]])
                clear &= ~hit.any(axis=1)
        return clear, "junction", blocker

    def _platform(
        self, place: str, platform: str, stay: tuple[int, int], day: int, ignore: frozenset[int]
    ) -> int | None:
        """Return the train in the way at a platform, or None if it is free."""
        events = self._on_day("stays", (place, platform), day)
        if not events:
            return None
        arrives, leaves, others = events
        key = f"{place}:{platform}"
        shared = key in self.rules.shared
        gap = 0 if shared else self.rules.reoccupy.get(key, self.rules.margins.platform)
        hit = (arrives < stay[1] + gap) & (leaves > stay[0] - gap) & self._keep(others, ignore)
        if hit.sum() <= int(shared):
            return None
        return int(others[hit][0])

    def _conflict(
        self, rule: str, place: str, at: int, blocker: int | None, detail: str = ""
    ) -> Conflict:
        train = None if blocker is None else self.timetable.trains[blocker]
        return Conflict(
            rule,
            place,
            clock(at),
            train.describe() if train else "",
            train.kind if train else "",
            detail,
        )

    def check(  # noqa: C901
        self,
        points: Sequence[Point],
        day: int,
        ignore: frozenset[int] = frozenset(),
        *,
        every: bool = False,
    ) -> list[Conflict]:
        """Check a fully timed train; the first conflict, or ``every`` one."""
        found: list[Conflict] = []
        last = len(points) - 1
        for i, here in enumerate(points):
            before = points[i - 1].tiploc if i else _STOP
            after = points[i + 1].tiploc if i < last else _STOP
            moves = [((before, here.tiploc, after), here.depart)]
            if here.stops:
                moves = [((before, here.tiploc, _STOP), here.arrive)] if i else []
                if i < last:
                    moves.append(((_STOP, here.tiploc, after), here.depart))
            for move, at in moves:
                clear, rule, blocker = self._move(move, np.array([at]), day, ignore)
                if not clear[0]:
                    found.append(self._conflict(rule, here.tiploc, at, blocker))
            if here.platform:
                stay = (here.arrive, here.depart)
                blocker = self._platform(here.tiploc, here.platform, stay, day, ignore)
                if blocker is not None:
                    found.append(self._conflict("platform", here.tiploc, here.arrive, blocker))
            if i < last:
                times = np.array([here.depart]), np.array([points[i + 1].arrive])
                clear, rule, blocker = self._link(
                    (here.tiploc, after), here.line, *times, day, ignore
                )
                if not clear[0]:
                    found.append(self._conflict(rule, here.tiploc, here.depart, blocker))
            if found and not every:
                return found[:1]
        return found

    # Routes and running times

    def routes(self, origins: frozenset[str], destinations: frozenset[str]) -> list[list[str]]:
        """Timing-point sequences between two calls: the commonest used, then the quickest."""
        key = ("routes", origins, destinations)
        if key in self._cache:
            return self._cache[key]
        used: collections.Counter[tuple[str, ...]] = collections.Counter()
        ends: dict[int, list[int]] = collections.defaultdict(list)
        for place in destinations:
            for number, i in self.timetable.calls_at.get(place, ()):
                ends[number].append(i)
        for later in ends.values():
            later.sort()
        for place in origins:
            for number, i in self.timetable.calls_at.get(place, ()):
                later = ends.get(number, [])
                at = bisect.bisect_right(later, i)
                if at < len(later):
                    train = self.timetable.trains[number]
                    route = tuple(p.tiploc for p in train.points[i : later[at] + 1])
                    used[route] += train.days.bit_count()
        found = [list(route) for route, _ in used.most_common(_MOST_ROUTES)]
        quickest = self._quickest_route(origins, destinations)
        if quickest and quickest not in found:
            found.append(quickest)
        self._cache[key] = found
        return found

    def _quickest_route(  # noqa: C901
        self, origins: frozenset[str], destinations: frozenset[str]
    ) -> list[str] | None:
        """Quickest route turning only where some train makes that turn."""
        timetable = self.timetable
        best: dict[tuple[str, str], int] = {}
        back: dict[tuple[str, str], tuple[str, str]] = {}
        queue: list[tuple[int, str, str]] = []
        for start in sorted(origins):
            for after in timetable.places_after.get(start, ()):
                minutes = timetable.quickest[start, after]
                if minutes < best.get((start, after), DAY):
                    best[start, after] = minutes
                    heapq.heappush(queue, (minutes, start, after))
        while queue:
            spent, before, here = heapq.heappop(queue)
            if spent > best[before, here]:
                continue
            if here in destinations:
                route, state = [here], (before, here)
                while state in back:
                    state = back[state]
                    route.append(state[1])
                return [state[0], *reversed(route)]
            for after in timetable.next_places.get((before, here), ()):
                if after == before:
                    continue
                total = spent + timetable.quickest.get((here, after), DAY)
                if total < best.get((here, after), 4 * DAY):
                    best[here, after] = total
                    back[here, after] = (before, here)
                    heapq.heappush(queue, (total, here, after))
        return None

    def _running(self, route: list[str], traction: str) -> np.ndarray:
        """Cumulative minimum half-minutes along ``route``, starting and stopping at its ends."""
        key = ("running", tuple(route), traction)
        if key not in self._cache:
            edges = list(itertools.pairwise(route))
            needed = [0.0]
            for i, (a, b) in enumerate(edges):
                minutes = None
                if self.graph is not None and self.graph.graph.has_edge(a, b):
                    first, last = i == 0, i == len(edges) - 1
                    kind = "srt" + ("_start" if first else "") + ("_stop" if last else "")
                    minutes, _ = self.graph._timing(self.graph.graph[a][b], kind, traction)
                if minutes is None:
                    minutes = self.timetable.quickest.get((a, b), 0) / 2
                needed.append(needed[-1] + 2 * minutes)
            self._cache[key] = np.array(needed)
        return self._cache[key]

    # The search

    def _leg(
        self,
        route: list[str],
        leave: int,
        arrive: int,
        traction: str,
        day: int,
        ignore: frozenset[int],
    ) -> tuple[list[Point], Conflict | None]:
        """Time one leg between its calls, spending its spare time where that finds a path.

        A public departure is the working one cut to the minute and an arrival the working one
        rounded up, so each end may move half a minute into the leg.
        """
        needed = self._running(route, traction)
        given = arrive - leave
        # Public times are whole minutes and BPLAN's are halves, so allow the same rounding
        # as the running-time check.
        if given < needed[-1] - 1.1:
            detail = f"needs {needed[-1] / 2:.1f} minutes and has {given / 2:.0f}"
            return [], self._conflict("running time", route[0], leave, None, detail)
        if not needed[-1]:
            base = np.zeros(len(needed), dtype=np.int64)
        elif given - needed[-1] <= _LONGEST_PATHING:
            base = np.rint(needed * min(1.0, given / needed[-1])).astype(np.int64)
        else:
            base = np.rint(needed * (given - _LONGEST_PATHING) / needed[-1]).astype(np.int64)
        base = np.maximum.accumulate(base) + leave
        spare = max(0, int(arrive - base[-1]))
        delays = np.arange(spare + 1)

        starts = delays <= 1
        clear, rule, blocker = self._move((_STOP, route[0], route[1]), leave + delays, day, ignore)
        if not clear[starts].any():
            return [], self._conflict(rule, route[0], leave, blocker)
        reach = starts & clear
        choices: list[list[tuple[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]] = []
        last = len(route) - 2
        for i, link in enumerate(itertools.pairwise(route)):
            before, after = np.nonzero(reach[:, None] & (delays[:, None] <= delays))
            if i == last:
                final = after >= spare - 1
                before, after = before[final], after[final]
            enter, leaves = base[i] + before, base[i + 1] + after
            onto = (link[0], link[1], _STOP if i == last else route[i + 2])
            through, rule, blocker = self._move(onto, base[i + 1] + delays, day, ignore)
            allowed = through[after]
            if not allowed.any():
                at = int(base[i + 1] + (after.min() if len(after) else spare))
                return [], self._conflict(rule, link[1], at, blocker)
            options, ok = [], np.zeros(len(enter), dtype=bool)
            blame: tuple[str, int | None] = ("", None)
            lines = self.timetable.lines.get(link, {"": 0})
            for line in sorted(lines, key=lambda name: (-lines[name], name)):
                free, *why = self._link(link, line, enter, leaves, day, ignore)
                free &= allowed
                options.append((line, free))
                ok |= free
                blame = blame if blame[1] is not None else (why[0], why[1])
            if not ok.any():
                return [], self._conflict(blame[0], link[0], int(enter.min()), blame[1])
            table = np.zeros((spare + 1, spare + 1), dtype=bool)
            table[before[ok], after[ok]] = True
            choices.append([(line, (before, after, free)) for line, free in options])
            reach = table.any(axis=0)

        return self._trace(route, base, spare, reach, choices), None

    @staticmethod
    def _trace(
        route: list[str],
        base: np.ndarray,
        spare: int,
        reach: np.ndarray,
        choices: list[list[tuple[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]],
    ) -> list[Point]:
        """Walk back, keeping as near an even spread of the spare time as the timetable allows."""
        last = len(route) - 2
        taken = int(np.nonzero(reach)[0][-1])
        used, picked = [taken], []
        for i in range(last, -1, -1):
            even = spare * (base[i] - base[0]) / max(1, base[-1] - base[0])
            # Every delay reach holds has a step back to it.
            _, taken, line = min(
                (
                    (abs(float(value) - even), int(value), line)
                    for line, (before, after, free) in choices[i]
                    for value in before[free & (after == taken)]
                ),
                key=lambda option: option[0],
            )
            used.append(taken)
            picked.append(line)
        used.reverse()
        picked.reverse()
        return [
            Point(
                place,
                int(base[i] + used[i]),
                int(base[i] + used[i]),
                stops=i in (0, last + 1),
                line=picked[i] if i <= last else "",
            )
            for i, place in enumerate(route)
        ]

    def find(
        self,
        calls: Sequence[tuple[Iterable[str], int, int]],
        day: int,
        traction: str = "",
        ignore: frozenset[int] = frozenset(),
        hints: Sequence[list[str]] = (),
    ) -> PathResult:
        """Find a path through ``calls``, each ``(its TIPLOCs, arrive, depart)``.

        The times at calls are kept. ``hints`` are routes to try first, one for each leg, as
        when an existing train is retimed.
        """
        points: list[Point] = []
        start = frozenset(calls[0][0])
        for i, ((_, _, depart), (places, arrive, _)) in enumerate(itertools.pairwise(calls)):
            ends = frozenset(places)
            routes = ([hints[i]] if i < len(hints) else []) + self.routes(start, ends)
            if not routes and points:
                # The next leg may leave from another of the station's platform groups.
                start = frozenset(calls[i][0])
                routes = self.routes(start, ends)
            if not routes:
                return PathResult(
                    points,
                    self._conflict("no route", sorted(start)[0] if start else "", depart, None),
                )
            failures: list[Conflict] = []
            for route in routes:
                timed, problem = self._leg(route, depart, arrive, traction, day, ignore)
                if problem:
                    failures.append(problem)
                    continue
                if points:
                    timed[0] = dataclasses.replace(timed[0], arrive=points.pop().arrive)
                points += timed
                start = frozenset([route[-1]])
                break
            else:
                # A train in the way says more than a route too slow for the time given.
                failures.sort(key=lambda found: found.rule == "running time")
                return PathResult(points, failures[0])
        return self._platforms(points, day, ignore)

    def _platforms(self, points: list[Point], day: int, ignore: frozenset[int]) -> PathResult:
        timetable, unknown = self.timetable, 0
        for i, here in enumerate(points):
            if not here.stops:
                continue
            before = points[i - 1].tiploc if i else _STOP
            after = points[i + 1].tiploc if i + 1 < len(points) else _STOP
            come = timetable.platforms_from.get((here.tiploc, before), set())
            go = timetable.platforms_to.get((here.tiploc, after), set())
            # An end of the train can use a platform that trains only call through.
            if i == 0:
                options = go
            elif i + 1 == len(points):
                options = come
            else:
                options = (come & go) or come or go
            if not options:
                unknown += 1
                continue
            blocker = None
            for platform in sorted(options):
                found = self._platform(
                    here.tiploc, platform, (here.arrive, here.depart), day, ignore
                )
                if found is None:
                    points[i] = dataclasses.replace(here, platform=platform, public=True)
                    break
                blocker = blocker if blocker is not None else found
            else:
                detail = f"none of {len(options)} platforms free"
                problem = self._conflict("platform", here.tiploc, here.arrive, blocker, detail)
                return PathResult(points, problem, unknown)
        return PathResult(points, None, unknown)


def day_numbers(days: Iterable[str]) -> list[int]:
    """Weekday names as Monday 0."""
    return [WEEKDAY_INDEX[day] for day in days]
