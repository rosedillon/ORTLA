"""The path check on small made-up timetables."""

from __future__ import annotations

from ortla import path_check as pc

WEDNESDAY = 2


def _train(uid: str, *points: pc.Point, days: int = 1 << WEDNESDAY) -> pc.Train:
    return pc.Train(uid, "XX", uid, "passenger", days, list(points))


def _through(uid: str, at: int, platform: str = "", line: str = "", days: int = 1 << WEDNESDAY):
    return _train(
        uid,
        pc.Point("A", at, at, stops=True, public=True, platform=platform, line=line),
        pc.Point("J", at + 10, at + 10, stops=False, line=line),
        pc.Point("B", at + 20, at + 20, stops=True, public=True),
        days=days,
    )


def _find(checker: pc.PathChecker, leave: int, arrive: int, day: int = WEDNESDAY):
    return checker.find([({"A"}, leave, leave), ({"B"}, arrive, arrive)], day)


def test_a_train_on_top_of_another_has_no_path() -> None:
    checker = pc.PathChecker(pc.WorkingTimetable([_through("T", 1000)]), pc.Rules())
    result = _find(checker, 1000, 1020)
    assert not result.clear
    assert result.conflict.rule == "headway"
    assert _find(checker, 1010, 1030).clear


def test_it_takes_the_other_line_where_there_is_one() -> None:
    trains = [_through("F", 1000, line="FL"), _through("S", 500, line="SL")]
    result = _find(pc.PathChecker(pc.WorkingTimetable(trains), pc.Rules()), 1000, 1020)
    assert result.clear
    assert {point.line for point in result.points[:-1]} == {"SL"}


def test_spare_time_is_spent_to_keep_clear_and_calls_keep_their_times() -> None:
    ahead = _train(
        "T",
        pc.Point("A", 988, 988, stops=True, public=True),
        pc.Point("J", 1010, 1010, stops=False),
        pc.Point("B", 1020, 1020, stops=True, public=True),
    )
    quick = _through("Q", 100)
    checker = pc.PathChecker(pc.WorkingTimetable([ahead, quick]), pc.Rules())
    result = _find(checker, 1000, 1040)
    assert result.clear
    start, junction, end = result.points
    assert start.depart in (1000, 1001)
    assert end.arrive in (1039, 1040)
    assert junction.depart >= 1016


def test_too_little_time_is_said() -> None:
    checker = pc.PathChecker(pc.WorkingTimetable([_through("T", 100)]), pc.Rules())
    assert _find(checker, 1000, 1010).conflict.rule == "running time"


def test_single_line_platform_and_junction() -> None:
    other = _train(
        "O",
        pc.Point("B", 1000, 1000, stops=True, public=True),
        pc.Point("J", 1010, 1010, stops=False),
        pc.Point("A", 1020, 1020, stops=True, public=True),
    )
    timetable = pc.WorkingTimetable([_through("T", 100, platform="1"), other])
    single = pc.Rules(single={"A>J", "J>A", "B>J", "J>B"})
    assert _find(pc.PathChecker(timetable, pc.Rules()), 1004, 1024).clear
    assert _find(pc.PathChecker(timetable, single), 1004, 1024).conflict.rule == "single line"
    # Meeting exactly at J is a crossing at a loop.
    assert _find(pc.PathChecker(timetable, single), 1000, 1020).clear

    crossing = pc.Rules(crossing={"A>J>B": {"B>J>A": 4}})
    assert _find(pc.PathChecker(timetable, crossing), 1000, 1020).conflict.rule == "junction"

    stood = _train(
        "P",
        pc.Point("A", 990, 1010, stops=True, platform="1"),
        pc.Point("C", 1030, 1030, stops=True),
    )
    timetable = pc.WorkingTimetable([_through("T", 100, platform="1"), stood])
    assert _find(pc.PathChecker(timetable, pc.Rules()), 1000, 1020).conflict.rule == "platform"


def test_a_train_from_the_day_before_is_still_there_after_midnight() -> None:
    tuesday = 1 << (WEDNESDAY - 1)
    late = _through("L", pc.DAY + 20, days=tuesday)
    checker = pc.PathChecker(pc.WorkingTimetable([late]), pc.Rules())
    assert not _find(checker, 20, 40).clear
    assert _find(checker, 20, 40, day=WEDNESDAY + 1).clear


def test_rules_are_learnt_from_what_repeats() -> None:
    trains = [_through(f"T{i}", 1000 + 3 * i) for i in range(5)]
    rules = pc.learn_rules(pc.WorkingTimetable(trains))
    assert rules.enter["A>J"] == 3
    assert not rules.single  # nothing is single without the track model


def test_rules_survive_being_saved(tmp_path) -> None:
    rules = pc.Rules(single={"A>J"}, crossing={"A>J>B": {"B>J>A": 4}})
    rules.save(tmp_path / "rules.json")
    assert pc.Rules.load(tmp_path / "rules.json") == rules
