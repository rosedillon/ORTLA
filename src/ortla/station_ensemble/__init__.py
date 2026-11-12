"""Station footfall from an ensemble of thirteen trip-end models (``ortla station-ensemble``).

Forecasts a station's annual entries and exits. The thirteen members share inputs: a
structural gravity model's reach of the network (:mod:`.gravity`), SDFT's catchment and trip
end terms, and terms for the place, the timetable and the Knowledgebase (:mod:`.terms`). They
differ in how they are fitted (:mod:`.members`). Their mean, with nearby stations' residuals
borrowed, is the forecast (:mod:`.ensemble`); proposed stations go through
:mod:`.proposals`.

Fitted on 2,111 established stations in England and Wales (2024-25 footfall). Cross-validated
on 25 km cells, 1,799 of them score rmse(log) 0.526 after borrowing. On the 69 stations opened
1998-2023 (kept out of every fit) the forecast scores 0.440, with a median forecast-to-actual
ratio of 0.94 and 80% intervals about x/1.8 wide.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ortla.station_ensemble import ensemble, proposals
from ortla.station_ensemble.sources import DataPaths

if TYPE_CHECKING:
    from ortla.config import StationEnsembleConfig

logger = logging.getLogger(__name__)


def run(config: StationEnsembleConfig) -> None:
    """Fit the ensemble, write its scores and the openings' forecasts, and forecast proposals."""
    out = config.output_path
    out.mkdir(parents=True, exist_ok=True)
    paths = DataPaths(config.data_dir)
    model = ensemble.fit(paths, ensemble.Cache(config.cache_dir))
    scores = ensemble.scores(model)
    scores.round(4).to_csv(out / "station_ensemble_scores.csv")
    ensemble.openings_table(model).to_csv(out / "station_ensemble_openings.csv")
    logger.info("Scores:\n%s", scores.round(3).to_string())
    if config.proposed:
        forecasts, neighbours = proposals.forecast(
            model, paths, config.proposed, config.neighbour_km
        )
        forecasts.to_csv(out / "station_ensemble_forecasts.csv")
        neighbours.to_csv(out / "station_ensemble_neighbours.csv")
        logger.info(
            "Forecasts:\n%s", forecasts[["name", "forecast_entries_exits", "low_80", "high_80"]]
        )
