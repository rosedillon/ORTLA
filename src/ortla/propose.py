"""The ``propose`` command: check candidates and predict demand.

For each candidate in the config it runs the running-time and headway
checks (:mod:`ortla.feasibility`), the optional historic
performance check, the departure-window search when the candidate has a
window, and the demand and loading prediction
(:mod:`ortla.demand`), writing one set of CSVs per candidate
and a summary of every problem found.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from ortla.bplan_geography import BplanNetworkLinks, BplanTimingGraph
from ortla.demand import (
    build_fares_context,
    build_gravity_context,
    candidate_calls,
    estimate_loading,
    predict_demand,
    transfer_demand,
)
from ortla.feasibility import (
    check_headway,
    check_historic_performance,
    check_running_times,
    recode_stops,
    search_departure_window,
)
from ortla.service_lookup import crs_to_tiploc_map
from ortla.stations import recode_columns, resolve_recodes
from ortla.timetable import read_public_calls, resolve_reference_date

if TYPE_CHECKING:
    from ortla.config import CandidateService, ProposeConfig
    from ortla.demand import FaresContext, GravityContext
    from ortla.network_assignment import TransferDemand

logger = logging.getLogger(__name__)


_BANNER = "=" * 70


def _format_issues_banner(issues: list[str]) -> str:
    """Build a hard-to-miss block of every issue and its fix.

    Per-candidate log lines get buried in a run's INFO output, so every
    infeasible leg and headway conflict across all candidates is
    collected into one banner, logged once at the end of the run and
    also written to ``propose_summary.txt``.
    """
    header = f"PROPOSE: {len(issues)} issue(s) found; suggested fixes below"
    lines = [_BANNER, header, _BANNER]
    for issue in issues:
        lines.append(issue)
        lines.append("")
    lines.append(_BANNER)
    return "\n".join(lines)


def _run_historic_performance_check(
    config: ProposeConfig, candidate: CandidateService, issues: list[str]
) -> None:
    """Check one candidate against HSP and write its outputs.

    Split out of :func:`run_propose` to keep its branch count down. It
    logs and reports issues the same way as the running-time and headway
    checks.
    """
    historic = check_historic_performance(config, candidate)
    if historic.empty:
        return

    historic_path = config.output_path / f"propose_{candidate.name}_historic_performance.csv"
    historic.to_csv(historic_path, index=False)
    n_poor = int((historic["status"] == "poor_reliability").sum())
    n_failed = int((historic["status"] == "lookup_failed").sum())
    logger.info(
        "  Historic performance check: %d leg/day-type evaluation(s), "
        "%d poor-reliability, %d lookup failure(s) -> %s",
        len(historic),
        n_poor,
        n_failed,
        historic_path,
    )
    hp = config.historic_performance
    window = f"{hp.from_date} to {hp.to_date}" if hp is not None else "historic window"
    for _, leg in historic[historic["status"] == "poor_reliability"].iterrows():
        percent = leg["percent_within_tolerance"]
        # HSP can return a service with no punctuality figure at all,
        # which is also reported as poor reliability.
        percent_text = "an unknown share" if pd.isna(percent) else f"{percent:.0f}%"
        issues.append(
            f"[{candidate.name}] POOR HISTORIC RELIABILITY: {leg['day_type']} "
            f"{leg['from_station']} -> {leg['to_station']} dep "
            f"{leg['candidate_departure']}\n"
            f"    nearest existing service ({leg['toc_code']} dep "
            f"{leg['nearest_existing_departure']}) only "
            f"{percent_text} within "
            f"{leg['tolerance_minutes']} min over "
            f"{leg['matched_historic_services']} historic services "
            f"({window})\n"
            f"    Consider whether the candidate is realistic on this "
            f"corridor without infrastructure/timetable changes."
        )


def _report_running_times(
    config: ProposeConfig,
    candidate: CandidateService,
    crs_tiploc: dict[str, str],
    graph: BplanTimingGraph,
    issues: list[str],
) -> int:
    """Check one candidate's running times and write its outputs.

    Returns the number of infeasible legs and appends a suggested-fix
    line to ``issues`` for each. Split out of :func:`run_propose` to
    keep its branch count down.
    """
    running = check_running_times(candidate, crs_tiploc, graph)
    running_path = config.output_path / f"propose_{candidate.name}_running_times.csv"
    running.to_csv(running_path, index=False)
    n_infeasible = int((running["status"] == "infeasible").sum()) if not running.empty else 0
    n_unverifiable = int((running["status"] == "unverifiable").sum()) if not running.empty else 0
    logger.info(
        "  Running-time check: %d legs, %d infeasible, %d unverifiable (no BPLAN route) -> %s",
        len(running),
        n_infeasible,
        n_unverifiable,
        running_path,
    )
    if n_infeasible:
        for _, leg in running[running["status"] == "infeasible"].iterrows():
            issues.append(
                f"[{candidate.name}] RUNNING-TIME INFEASIBLE: "
                f"{leg['from_station']} -> {leg['to_station']}\n"
                f"    {-leg['shortfall_minutes']:.1f} min short of the "
                f"{leg['minimum_minutes']:.1f} min BPLAN minimum\n"
                f"    FIX: depart by {leg['suggested_departure']} (same arrival) or "
                f"arrive by {leg['suggested_arrival']} (same departure)"
            )
    return n_infeasible


def _report_demand_and_loading(
    config: ProposeConfig,
    candidate: CandidateService,
    calls: pd.DataFrame,
    reference_date: pd.Timestamp,
    gravity: GravityContext | None,
    fares: FaresContext | None,
    transfers: TransferDemand | None = None,
) -> None:
    """Predict one candidate's demand and loading; write outputs.

    Split out of :func:`run_propose` to keep its branch count down. The
    demand table feeds the loading and VoC profile.
    """
    demand = predict_demand(
        config, candidate, calls, reference_date, gravity, fares, transfers=transfers
    )
    if demand.empty:
        return

    demand_path = config.output_path / f"propose_{candidate.name}_demand_by_leg.csv"
    demand.to_csv(demand_path, index=False)
    # Every label the demand model gave a leg, legacy and calibrated
    # alike.
    models = ", ".join(
        f"{count} {label}" for label, count in demand["demand_model_used"].value_counts().items()
    )
    total_captured = demand["predicted_annual_trips_captured"].sum(skipna=True)
    total_revenue = demand["predicted_annual_revenue_gbp"].sum(skipna=True)
    n_priced = int(demand["predicted_annual_revenue_gbp"].notna().sum())
    revenue_note = (
        f", GBP {total_revenue:,.0f} predicted annual revenue "
        f"({n_priced}/{len(demand)} legs priced)"
        if fares is not None
        else ""
    )
    logger.info(
        "  Demand prediction: %d legs (%s), %.0f total predicted annual trips captured%s -> %s",
        len(demand),
        models,
        total_captured,
        revenue_note,
        demand_path,
    )

    loading = estimate_loading(candidate, demand, config)
    if loading.empty:
        return

    loading_path = config.output_path / f"propose_{candidate.name}_loading_profile.csv"
    loading.to_csv(loading_path, index=False)
    peak = loading.loc[loading["voc"].idxmax()] if loading["voc"].notna().any() else None
    if peak is not None:
        logger.info(
            "  Loading profile: peak VoC %.1f%% on %s -> %s (%s) -> %s",
            100 * peak["voc"],
            peak["from_station"],
            peak["to_station"],
            peak["day"],
            loading_path,
        )
    else:
        logger.info(
            "  Loading profile written (no capacity configured, VoC unavailable) -> %s",
            loading_path,
        )


def run_propose(config: ProposeConfig) -> None:
    """Check every candidate's feasibility and predict its demand.

    Writes, per candidate, to ``config.output_path``:

    * ``propose_{name}_running_times.csv`` and ``propose_{name}_headway_conflicts.csv``.
    * ``propose_{name}_historic_performance.csv`` when ``historic_performance`` is set.
    * ``propose_{name}_demand_by_leg.csv`` when ``period_matrices_dir`` is set, priced into
      ``predicted_annual_revenue_gbp`` when ``fares_dir`` is set.
    * ``propose_{name}_loading_profile.csv``: on-board load and VoC per segment.

    Any infeasible leg or headway conflict also goes into ``propose_summary.txt`` with its
    suggested fix, and into one banner at the end of the log.
    """
    config.output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading cleaned calls from %s ...", config.timetable_calls_path)
    calls = read_public_calls(config.timetable_calls_path)
    crs_tiploc = crs_to_tiploc_map(calls)
    # After the TIPLOC map, as in the hindcast: running times are looked
    # up by each stop's own TIPLOC, while service counts need one code
    # per station.
    recodes = resolve_recodes(config.station_recodes)
    calls = recode_columns(calls, recodes, ["crs"])

    reference_date = resolve_reference_date(calls, config)
    logger.info(
        "Headway check reference snapshot: %s (%s).",
        reference_date.date(),
        reference_date.day_name(),
    )

    logger.info("Loading BPLAN timing graph from %s ...", config.bplan_output_path)
    graph = BplanTimingGraph(config.bplan_output_path)
    try:
        network_links = BplanNetworkLinks(config.bplan_output_path)
    except FileNotFoundError:
        # segment_running_lines is only informational, so an older bplan
        # output without this file must not stop the run.
        logger.warning(
            "No bplan_network_links.csv under %s, so headway conflicts won't "
            "carry a segment_running_lines hint.",
            config.bplan_output_path,
        )
        network_links = None

    gravity = build_gravity_context(config)
    fares = build_fares_context(config)

    issues: list[str] = []
    for candidate in config.services:
        logger.info("Checking candidate service '%s' ...", candidate.name)

        n_infeasible = _report_running_times(config, candidate, crs_tiploc, graph, issues)
        recoded = recode_stops(candidate, recodes)

        headway = check_headway(
            config,
            recoded,
            calls,
            reference_date,
            crs_tiploc=crs_tiploc,
            network_links=network_links,
        )
        headway_path = config.output_path / f"propose_{candidate.name}_headway_conflicts.csv"
        headway.to_csv(headway_path, index=False)
        n_conflicts = int(headway["conflict"].sum()) if not headway.empty else 0
        logger.info(
            "  Headway check: %d segment-day evaluations, %d conflicts -> %s",
            len(headway),
            n_conflicts,
            headway_path,
        )
        if n_conflicts:
            for _, leg in headway[headway["conflict"]].iterrows():
                issues.append(
                    f"[{candidate.name}] HEADWAY CONFLICT: {leg['day']} "
                    f"{leg['from_station']} -> {leg['to_station']} "
                    f"dep {leg['candidate_departure']}\n"
                    f"    conflicts with {leg['conflicting_train']}\n"
                    f"    FIX: try {leg['suggested_departure']} instead "
                    f"({leg['shift_minutes']:+.0f} min)"
                )

        if config.historic_performance is not None:
            _run_historic_performance_check(config, candidate, issues)

        # Journeys changing onto the candidate ride it too (see
        # predict_demand).
        transfers = (
            transfer_demand(
                config,
                pd.concat([calls, candidate_calls(recoded, reference_date)]),
                reference_date,
            )
            if config.period_matrices_dir is not None
            else None
        )
        if candidate.search_window is not None:
            search_results = search_departure_window(
                config,
                candidate,
                crs_tiploc,
                graph,
                calls,
                reference_date,
                gravity,
                fares,
                transfers=transfers,
            )
            if not search_results.empty:
                search_path = config.output_path / f"propose_{candidate.name}_search_results.csv"
                search_results.to_csv(search_path, index=False)
                n_feasible = int(search_results["feasible"].sum())
                logger.info(
                    "  Departure-window search: %d trial shift(s), %d feasible -> %s",
                    len(search_results),
                    n_feasible,
                    search_path,
                )

        if n_infeasible or n_conflicts:
            logger.warning(
                "Candidate '%s': %d infeasible leg(s), %d headway conflict(s); "
                "see the summary at the end of this run.",
                candidate.name,
                n_infeasible,
                n_conflicts,
            )
        else:
            logger.info("Candidate '%s': no running-time or headway issues found.", candidate.name)

        _report_demand_and_loading(
            config, recoded, calls, reference_date, gravity, fares, transfers
        )

    if issues:
        banner = _format_issues_banner(issues)
        summary_path = config.output_path / "propose_summary.txt"
        summary_path.write_text(banner, encoding="utf-8")
        logger.warning("\n%s", banner)
        logger.warning("Suggested fixes also written to %s", summary_path)
    else:
        logger.info("No running-time or headway issues found for any candidate.")
