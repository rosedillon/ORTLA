"""Pricing services together, and who gains from a shorter wait."""

from __future__ import annotations

import types

import pandas as pd

from ortla import package, scenario
from ortla.appraisal import AppraisalParameters, _annual_wait_time_benefit
from ortla.config import CandidateService, ServiceStop


def _train(name: str, departure: str) -> CandidateService:
    return CandidateService(
        name=name,
        traction_code="158",
        days_of_week=["Wednesday"],
        stops=[
            ServiceStop(station="AAA", departure=departure),
            ServiceStop(station="BBB", arrival="23:00"),
        ],
    )


def test_every_train_is_told_about_the_rest_of_the_package(monkeypatch):
    seen = {}

    def predict(config, train, *args, added_trains, **kwargs):
        seen[train.name] = (
            added_trains("Wednesday", "am_peak", "AAA", "BBB"),
            added_trains("Wednesday", "day", "AAA", "BBB"),
        )
        return pd.DataFrame({"candidate_name": [train.name]})

    monkeypatch.setattr(package, "predict_demand", predict)
    monkeypatch.setattr(package, "transfer_demand", lambda *a: None)
    config = types.SimpleNamespace(
        time_periods_cutoffs={"am_peak": "7:00", "interpeak": "9:30", "pm_peak": "16:00"}
    )
    context = types.SimpleNamespace(
        calls=pd.DataFrame(),
        prepared=pd.DataFrame(),
        reference_date=pd.Timestamp("2026-06-17"),
        gravity=None,
        fares=None,
    )
    trains = [_train("a", "07:10"), _train("b", "08:10"), _train("c", "12:10")]
    package.package_demand(config, trains, context)
    assert seen == {"a": (2, 3), "b": (2, 3), "c": (2, 3)}


def test_every_rider_on_the_pair_gains_the_shorter_wait(monkeypatch):
    monkeypatch.setattr(scenario, "day_services", lambda *a: None)
    monkeypatch.setattr(scenario, "od_services", lambda *a: range(48))
    monkeypatch.setattr(scenario, "matrix_cell", lambda *a: (100.0, "observed"))
    config = types.SimpleNamespace(
        station_aliases={}, time_periods_cutoffs={"am_peak": "7:00"}, period_matrices_dir=None
    )
    monkeypatch.setattr(scenario, "period_matrix_path", lambda *a: None)
    legs = pd.DataFrame(
        {
            "candidate_name": ["a"],
            "day": ["Wednesday"],
            "from_station": ["AAA"],
            "to_station": ["BBB"],
        }
    )
    context = types.SimpleNamespace(prepared=None, reference_date=None)
    values = types.SimpleNamespace(operating_hours_per_day=16.0, max_average_wait_minutes=15.0)
    saved = scenario.pair_wait_minutes(config, [_train("a", "07:10")], context, legs, values)
    # 48 trains in 16 hours wait 10 minutes; 49 wait 480/49. Two periods of 100 riders.
    assert abs(saved - 200 * (10 - 480 / 49)) < 1e-9

    monkeypatch.setattr(scenario, "od_services", lambda *a: range(4))  # both waits at the cap
    assert scenario.pair_wait_minutes(config, [_train("a", "07:10")], context, legs, values) == 0


def test_a_given_wait_saving_replaces_the_end_to_end_rule():
    tag = types.SimpleNamespace(
        price_year=2023,
        vot_work_rail=30.0,
        vot_commuting=30.0,
        vot_other=30.0,
        vot_for_year=lambda value, *a: value,
    )
    params = AppraisalParameters(tag=tag, operating_hours_per_day=16.0, wait_minutes_saved=600.0)
    assert abs(_annual_wait_time_benefit(params, 2028, 1.0) - 10 * 30 * 2 * 1.19) < 1e-9
