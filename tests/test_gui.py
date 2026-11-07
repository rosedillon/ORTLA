"""The desktop app: start-up, forms, projects, pipelines and charts."""

from __future__ import annotations

import os
import pathlib

import numpy as np
import pandas as pd
import pytest
import yaml

pytest.importorskip("PySide6.QtWebEngineWidgets")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from orta.config import COMMANDS
from orta.gui import charts
from orta.gui.charts import maps, plotly_charts
from orta.gui.constants import ThemeMode
from orta.gui.docs import command_page
from orta.gui.services.job_queue import JobQueue
from orta.gui.services.pipeline import PipelineStage, apply_dependencies
from orta.gui.views.analysis import AnalysisView
from orta.gui.views.pipeline import _PRESETS, PipelineView
from orta.gui.widgets.config_form import ConfigForm, field_comments

EXAMPLES = pathlib.Path(__file__).parents[1] / "examples"


@pytest.fixture(autouse=True)
def app(tmp_path, monkeypatch):
    # Keep the app's settings file out of the real home folder.
    monkeypatch.setenv("HOME", str(tmp_path))
    return QApplication.instance() or QApplication([])


def test_window_starts():
    from orta.gui.app import RailModelApp

    window = RailModelApp()
    assert window._stack.currentIndex() == 0
    assert window._analysis._form is not None


@pytest.mark.parametrize("command", COMMANDS)
def test_every_command_has_a_form_and_a_page(command):
    model = COMMANDS[command][0]
    form = ConfigForm(model)
    assert form.get_data() == {} or set(form.get_data()) <= set(model.model_fields)
    assert f"orta {command}" in command_page(command)


@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("[!.]*.yml")), ids=lambda p: p.stem)
def test_examples_survive_the_form(path):
    model = COMMANDS[path.stem][0]
    data = yaml.safe_load(path.read_text())
    form = ConfigForm(model)
    form.set_data(data)
    assert model.model_validate(form.get_data()) == model.model_validate(data)


def test_field_help_comes_from_comments():
    help_text = field_comments(COMMANDS["loading"][0])
    assert "SRAM" in help_text["od_profile_path"]
    # Inherited fields keep their parent's comment.
    assert field_comments(COMMANDS["hindcast"][0])["table_1410"]


def test_edits_survive_switching_commands():
    view = AnalysisView(JobQueue())
    view.select_command("distance")
    view._form.set_data({"output_path": "out/distance"})
    view.select_command("matrix")
    view.select_command("distance")
    assert view._current_data()["output_path"] == "out/distance"


def test_projects_round_trip():
    view = AnalysisView(JobQueue())
    view.load_all({"distance": {"output_path": "/a"}, "matrix": {"output_path": "/b"}}, "matrix")
    assert view._current_data()["output_path"] == "/b"
    view.select_command("distance")
    assert view._current_data()["output_path"] == "/a"
    captured = view.capture_all()
    assert captured["distance"]["output_path"] == "/a"
    assert captured["matrix"]["output_path"] == "/b"


def test_new_project_forgets_edits():
    view = AnalysisView(JobQueue())
    view.select_command("distance")
    view._form.set_data({"output_path": "out/leaky"})
    view.load_all({}, "distance")
    assert view._current_data().get("output_path") != "out/leaky"


def test_presets_fill_the_pipeline():
    view = PipelineView(JobQueue())
    for sequence in _PRESETS.values():
        assert set(sequence) <= set(COMMANDS)
        view._load_preset(sequence)
        assert [s.command for s in view._stages] == sequence


def test_pipeline_wires_outputs_into_propose():
    done = {
        "timetable": PipelineStage("timetable", {}, pathlib.Path("/tt")),
        "bplan": PipelineStage("bplan", {}, pathlib.Path("/bp")),
    }
    data = apply_dependencies("propose", {"services": []}, done)
    assert data["timetable_calls_path"] == str(pathlib.Path("/tt/timetable_calls.csv"))
    assert data["bplan_output_path"] == str(pathlib.Path("/bp"))
    assert "period_matrices_dir" not in data
    assert apply_dependencies("matrix", {"source_odm": "x.csv"}, {}) == {"source_odm": "x.csv"}


def test_hindcast_charts(tmp_path):
    pd.DataFrame(
        {
            "train_uid": ["A1", "A2"],
            "day": ["Monday", "Monday"],
            "period": ["am_peak", "interpeak"],
            "actual_load": [80.0, 30.0],
            "predicted_load_calibrated_marginal": [60.0, 35.0],
        }
    ).to_csv(tmp_path / "hindcast_trains.csv", index=False)
    titles = [t for t, _ in charts.build_figures("hindcast", tmp_path)]
    assert titles[0] == "Predicted against counted, by train"
    assert "Load by period" in titles
    assert "Error distribution, by train" in titles


@pytest.mark.parametrize("command", COMMANDS)
def test_every_command_copes_with_an_empty_folder(command, tmp_path):
    assert charts.build_figures(command, tmp_path) == []


def _stations_file(tmp_path):
    # Columns in a different order from the one the reader asks for.
    path = tmp_path / "stations.csv"
    pd.DataFrame(
        {
            "lat": [53.48, 53.41, 53.80],
            "long": [-2.23, -2.98, -1.55],
            "crsCode": ["MAN", "LIV", "LDS"],
        }
    ).to_csv(path, index=False)
    return path


def test_stations_found_in_any_config(tmp_path):
    path = _stations_file(tmp_path)
    found = maps.find_stations(
        {"output_path": "x"}, {"distance": {"stations": {"path": str(path)}}}
    )
    assert found.loc["LIV", "lon"] == pytest.approx(-2.98)
    assert found.loc["LDS", "lat"] == pytest.approx(53.80)
    assert maps.find_stations({"output_path": "x"}) is None


def test_flow_and_station_maps(tmp_path):
    stations = maps.find_stations({"s": str(_stations_file(tmp_path))})
    pairs = pd.DataFrame(
        {
            "origin": ["MAN", "MAN", "LIV"],
            "destination": ["LIV", "LDS", "XXX"],
            "trips": [10.0, 30.0, 5.0],
        }
    )
    fig = maps.flow_map(pairs, stations, "trips", "flows", ThemeMode.LIGHT)
    # The pair with an unknown station is dropped; the rest are drawn.
    assert (
        sum(int(np.isfinite(np.asarray(tr.lat, dtype=float)).sum()) for tr in fig.data[:-1]) == 4
    )
    placed = maps.with_positions(pd.DataFrame({"crs": ["MAN", "XXX"], "v": [1.0, 2.0]}), stations)
    assert list(placed["crs"]) == ["MAN"]
    assert maps.station_map(placed, "v", "t", ThemeMode.DARK) is not None


def test_commercial_charts(tmp_path):
    pd.DataFrame(
        {
            "year": [2026, 2027],
            "revenue": [5e6, 6e6],
            "operating_cost": [4e6, 4e6],
            "capital_cost": [2e6, 0.0],
            "net_cashflow": [-1e6, 2e6],
            "cumulative_net_cashflow": [-1e6, 1e6],
            "cumulative_npv": [-1e6, 0.9e6],
            "proposed_passengers": [1e5, 1.1e5],
            "baseline_passengers": [0, 0],
        }
    ).to_csv(tmp_path / "commercial_case_cashflow.csv", index=False)
    pd.Series({"npv_m": 0.9}).to_csv(tmp_path / "commercial_case_summary.csv", header=False)
    pd.DataFrame({"scenario": ["Demand +20%", "Demand -20%"], "npv_m": [2.0, -0.2]}).to_csv(
        tmp_path / "commercial_case_sensitivity.csv", index=False
    )
    titles = [t for t, _ in charts.build_figures("commercial", tmp_path)]
    assert titles == ["Cumulative cashflow", "Annual cashflow", "Passengers", "Sensitivity"]


def test_new_station_map_needs_no_station_file(tmp_path):
    pd.DataFrame(
        {
            "crs": ["NEW"],
            "forecast_entries_exits": [5e5],
            "low_80": [2e5],
            "high_80": [1.2e6],
            "low_90": [1.4e5],
            "high_90": [1.8e6],
            "latitude": [52.0],
            "longitude": [0.1],
        }
    ).to_csv(tmp_path / "new_station_forecasts.csv", index=False)
    titles = [t for t, _ in charts.build_figures("new-station", tmp_path)]
    assert "Map: new stations and neighbours" in titles


def test_chart_view_works_offline():
    from orta.gui.widgets.chart_view import ChartView, _folder

    ChartView().set_figure(plotly_charts.go.Figure())
    assert (_folder() / "plotly.min.js").stat().st_size > 1_000_000
