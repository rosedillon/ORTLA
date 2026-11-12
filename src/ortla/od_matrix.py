"""The annual origin-destination matrix: read, build and rebalance.

An OD matrix answers "how many rail journeys were made from station A to
station B in a year?" for every pair of stations. The source ODM file is
one long list of (origin, destination, journeys) rows. :func:`read_odm`
reads any release of it, and :func:`build_odm_matrix` lays the journeys
out as a square grid, one row per origin and one column per destination.
Stations are identified by their three-letter code (TLC); a pair
involving a station outside the valid list is blanked with a fill marker
(``"!"`` by default) rather than a number.

Rebalancing to table 1410
-------------------------
:func:`rebalance_to_station_totals` constrains the matrix to ORR table
1410's station totals. The two are not independent measures. ORR's
quality and methodology annex (*Estimates of station usage 2023-24*,
Annex 1) records that LENNON feeds the MOIRA 2.2 base matrix and that
ORR then makes its own adjustments: tickets sold to "London Terminal"
are allocated, demand is allocated within group stations outside
London, TfL data replaces the Elizabeth line's pay-as-you-go journeys,
uncollected ticket-on-departure sales are excluded and journeys with the
same origin and destination are reallocated. It adds that "the resulting
dataset is used to produce the Origin Destination Matrix ... the ODM is
then used to derive estimates for the number of entries and exits at
each station". Table 1410 is therefore derived from an adjusted ODM, not
a second measurement of it.

So the constraint does not reconcile two independent sources. It imports
ORR's station-level adjustments into a matrix taken before them. That is
still worth doing, because the model has no other way to see those
adjustments, but it is not external validation, and the gain in the
gravity fit that follows is mechanical. Neither should be quoted as
corroboration.

Measured with year-matched inputs they barely disagree. Against the
2024-25 table 1410, 96.7% of the 2,586 stations in the 2024-25 ODM have
a marginal within 0.1% of their published entries and exits, and the
national totals agree to four significant figures (3.065e9 either way).
The stations that do differ are ORR's named reallocations: Manchester
Victoria at 1.22, Manchester Airport at 0.65, London Road Brighton at
0.60, and a tail of group and terminal stations. Those 85 or so stations
are all the constraint changes.

Match the years. Read against a 2024-25 ODM, the 2022-23 table 1410
gives a median ODM/1410 ratio of 1.25 (p5-p95 1.00-1.65), and 1.246 is
just the growth in national station usage over those two years. On
year-matched inputs the ratio has median 1.000 and p5-p95 1.00-1.00,
and the constraint moves the national gravity fit from R^2 0.7083 to
0.7085.

The season-ticket mechanism is still real and documented. DfT's
*Journeys per Season Ticket Study* (2019) measured 378 annual journeys
per season ticket against the 480 LENNON assumes, and ORR and Steer
record that London season tickets are sold from the furthest station in
a fare group wherever the holder boards. Table 1410 inherits that error
from the ODM, so comparing the two cannot find it.

Nor is this matrix estimation from counts. ODME conditions a prior on
link counts, which a service that does not exist cannot have. Table 1410
is a published statistic for every station in Britain, and SDFT predicts
the same quantity, in the same unit, for stations that do not exist yet,
so the constraint transfers to an unbuilt station. That separates it
from the fixes ruled out in ``docs/notes/DIRECTIONAL_DEMAND_SPLIT.md``
section 7.

Only the shares are rebalanced. The grand total is kept exactly, which
on year-matched inputs is very nearly a no-op in itself. Keeping the
total also means a mismatched footfall year cannot leak a level shift
into the matrix; it leaks a shape error instead, which is how the
artefact above went unnoticed.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ortla.stations import read_station_list

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Mapping

    from ortla.config import MatrixConfig

logger = logging.getLogger(__name__)


def build_nlc_to_tlc(reference_paths: list[pathlib.Path]) -> dict[str, str]:
    """NLC -> TLC map from ODM files that carry both codes.

    Older ODM releases (e.g. 2018-19, 2020-21 to 2022-23) only publish
    NLCs; later releases publish both, so they supply the mapping.
    """
    pairs = []
    for path in reference_paths:
        for side in ("origin", "destination"):
            cols = [f"{side}_nlc", f"{side}_tlc"]
            df = pd.read_csv(path, usecols=cols, dtype=str).dropna().drop_duplicates()
            df.columns = ["nlc", "tlc"]
            pairs.append(df)
    table = pd.concat(pairs, ignore_index=True)
    table["nlc"] = table["nlc"].str.strip().str.zfill(4)
    table["tlc"] = table["tlc"].str.strip().str.upper()
    # An NLC keeps one TLC; take the most common where releases
    # disagree.
    return table.groupby("nlc")["tlc"].agg(lambda s: s.value_counts().index[0]).to_dict()


def read_odm(
    path: pathlib.Path,
    nlc_reference_paths: list[pathlib.Path] | None = None,
    chunk_size: int = 1_000_000,
) -> pd.DataFrame:
    """Read any ODM release as origin, destination and journeys.

    Releases without TLC columns are mapped from NLCs via
    :func:`build_nlc_to_tlc` over ``nlc_reference_paths``; rows whose
    NLC has no TLC are dropped (and counted in the log).
    """
    header = pd.read_csv(path, nrows=0).columns
    has_tlc = {"origin_tlc", "destination_tlc"} <= set(header)
    usecols = (
        ["origin_tlc", "destination_tlc", "journeys"]
        if has_tlc
        else ["origin_nlc", "destination_nlc", "journeys"]
    )
    mapping = None
    if not has_tlc:
        if not nlc_reference_paths:
            raise ValueError(f"{path} has no TLC columns; set nlc_reference_paths to map NLCs.")
        mapping = build_nlc_to_tlc(nlc_reference_paths)

    chunks = []
    dropped = 0.0
    for chunk in pd.read_csv(path, usecols=usecols, dtype=str, chunksize=chunk_size):
        journeys = pd.to_numeric(
            chunk["journeys"].str.replace(",", "", regex=False), errors="coerce"
        )
        if has_tlc:
            origin = chunk["origin_tlc"].str.strip().str.upper()
            dest = chunk["destination_tlc"].str.strip().str.upper()
        else:
            origin = chunk["origin_nlc"].str.strip().str.zfill(4).map(mapping)
            dest = chunk["destination_nlc"].str.strip().str.zfill(4).map(mapping)
        frame = pd.DataFrame({"origin_tlc": origin, "destination_tlc": dest, "journeys": journeys})
        missing = frame[["origin_tlc", "destination_tlc"]].isna().any(axis=1)
        dropped += frame.loc[missing, "journeys"].sum()
        chunks.append(frame[~missing & frame["journeys"].notna()])
    odm = pd.concat(chunks, ignore_index=True)
    odm = odm.groupby(["origin_tlc", "destination_tlc"], as_index=False)["journeys"].sum()
    total = odm["journeys"].sum()
    logger.info(
        "Read %s: %d OD pairs, %.1fm journeys (%.2f%% dropped with no TLC).",
        path.name,
        len(odm),
        total / 1e6,
        100 * dropped / (total + dropped) if total + dropped else 0.0,
    )
    return odm


def build_odm_matrix(config: MatrixConfig) -> pd.DataFrame:
    """Turn the flat ODM rows into a square matrix of journeys.

    Every station in the data gets a row and a column. Pairs with an end outside the valid-station
    list keep their place but hold the fill marker instead of a number.
    """
    fill_value = config.fill_value

    valid_stations = read_station_list(config.valid_stations)[0] if config.valid_stations else None
    odm = read_odm(config.source_odm, config.nlc_reference_paths, config.chunk_size)

    # read_odm has already summed each pair. The same stations go on
    # both axes so the matrix is square.
    all_stations = sorted(set(odm["origin_tlc"]) | set(odm["destination_tlc"]))

    matrix = odm.pivot_table(
        index="origin_tlc",
        columns="destination_tlc",
        values="journeys",
        aggfunc="sum",
        fill_value=0,
    )
    # Pairs with no journeys are missing from the pivot; fill those gaps
    # with 0.
    matrix = matrix.reindex(index=all_stations, columns=all_stations, fill_value=0)

    # Switch to object cells so a single grid can hold both numbers and
    # the "!" marker, then blank out every row/column for a station
    # outside our list.
    if valid_stations is not None:
        matrix = matrix.astype(object)
        invalid_rows = ~matrix.index.isin(valid_stations)
        invalid_cols = ~matrix.columns.isin(valid_stations)
        matrix.loc[invalid_rows, :] = fill_value
        matrix.loc[:, invalid_cols] = fill_value

    matrix.index.name = "origin_tlc"
    matrix.columns.name = "destination_tlc"

    logger.info("Built %dx%d ODM matrix.", matrix.shape[0], matrix.shape[1])
    return matrix


def run_matrix(config: MatrixConfig) -> None:
    """Build the OD matrix and save it to ``config.output_path``."""
    matrix = build_odm_matrix(config)

    config.output_path.mkdir(parents=True, exist_ok=True)
    output_file = config.output_path / "odm_matrix.csv"
    matrix.to_csv(output_file)
    logger.info("ODM matrix saved to %s", output_file)


@dataclasses.dataclass(frozen=True)
class BalanceReport:
    """What a rebalance did, for the record and the diagnostics."""

    stations: pd.DataFrame
    """Per-station ``target``, ``before``, ``after`` and ``factor``."""

    iterations: int
    """Scaling passes run."""

    converged: bool
    """Whether the worst constrained station reached ``tolerance``."""

    max_residual: float
    """Largest \\|after/target - 1\\| over constrained stations."""

    constrained: int
    """Stations carrying a table 1410 total."""

    unconstrained: int
    """Matrix stations with no table 1410 total, left to float."""

    total_before: float
    """Grand total before, which the rebalance preserves."""

    def summary(self) -> str:
        """One-line description for the log."""
        return (
            f"{self.constrained} stations constrained, {self.unconstrained} "
            f"floating; {self.iterations} iterations, max residual "
            f"{self.max_residual:.2%}"
            f"{'' if self.converged else ' (NOT converged)'}"
        )


def station_targets(
    station_entries: Mapping[str, float], size_exponent: float = 0.0
) -> dict[str, float]:
    """Station totals to rebalance to, from entries and exits.

    Only the shares matter, as :func:`rebalance_to_station_totals` keeps
    the grand total. A negative exponent moves demand towards small
    stations (see ``LoadingConfig.station_size_exponent``); 0 gives the
    table 1410 shares.
    """
    return {s: float(e) ** (1.0 + size_exponent) for s, e in station_entries.items() if e > 0}


def rebalance_to_station_totals(
    odm: pd.DataFrame,
    station_entries: Mapping[str, float],
    *,
    max_iterations: int = 100,
    tolerance: float = 1e-4,
    journeys_column: str = "journeys",
) -> tuple[pd.DataFrame, BalanceReport]:
    """Scale ODM cells so station totals match table 1410; return the copy and a report.

    Scaling is symmetric, ``M'(a, b) = r(a) M(a, b) r(b)``, so a symmetric matrix stays symmetric
    and the direction split still sees what it was fitted on. A station's total is its row plus its
    column, which is what table 1410 publishes as entries and exits. Stations absent from table
    1410, including every one opened since, are left unscaled. Iterates until every constrained
    station is within ``tolerance`` of its target. Raises ValueError when no station has a table
    1410 total.
    """
    balanced = odm.copy()
    origin = balanced["origin_tlc"].to_numpy(dtype=object)
    destination = balanced["destination_tlc"].to_numpy(dtype=object)
    journeys = balanced[journeys_column].to_numpy(dtype=float)

    codes, index = np.unique(np.concatenate([origin, destination]), return_inverse=True)
    origin_ix = index[: len(origin)]
    dest_ix = index[len(origin) :]
    n_stations = len(codes)

    target = np.full(n_stations, np.nan)
    for position, code in enumerate(codes):
        entries = station_entries.get(str(code))
        if entries is not None and entries > 0:
            target[position] = float(entries)
    constrained = np.isfinite(target)
    if not constrained.any():
        raise ValueError(
            "No station in the OD matrix carries a table 1410 total, so there "
            "is nothing to constrain. Check that table_1410 and source_odm use "
            "the same station codes."
        )

    def marginals(weights: np.ndarray) -> np.ndarray:
        return np.bincount(origin_ix, weights=weights, minlength=n_stations) + np.bincount(
            dest_ix, weights=weights, minlength=n_stations
        )

    before = marginals(journeys)
    total_before = float(journeys.sum())

    # Preserve the grand total: only the shares across stations are
    # corrected. On year-matched inputs the two totals already agree, so
    # this scale is ~1.0; it is a guard against a mismatched footfall
    # year silently re-levelling the whole matrix.
    scale = before[constrained].sum() / target[constrained].sum()
    target = target * scale

    ratios = np.ones(n_stations)
    weights = journeys
    max_residual = np.inf
    iterations = 0
    for iterations in range(1, max_iterations + 1):  # noqa: B007 (read after the loop)
        current = marginals(weights)
        # A station with a target but no demand in the matrix cannot be
        # scaled into existence; leave it rather than divide by zero.
        movable = constrained & (current > 0)
        residual = np.abs(current[movable] / target[movable] - 1.0)
        max_residual = float(residual.max()) if residual.size else 0.0
        if max_residual <= tolerance:
            break
        # Damped by a half because each cell carries the factor at both
        # ends.
        step = np.ones(n_stations)
        step[movable] = np.sqrt(target[movable] / current[movable])
        ratios = ratios * step
        weights = ratios[origin_ix] * journeys * ratios[dest_ix]

    converged = max_residual <= tolerance
    if not converged:
        logger.warning(
            "Table 1410 rebalance did not converge in %d iterations "
            "(max residual %.3f%%); using the last iterate.",
            max_iterations,
            100 * max_residual,
        )

    # Floating stations shift the total slightly; restore it exactly.
    total_after = float(weights.sum())
    if total_after > 0:
        weights = weights * (total_before / total_after)
    balanced[journeys_column] = weights

    after = marginals(weights)
    report = BalanceReport(
        stations=pd.DataFrame(
            {
                "station": codes,
                "target": np.where(constrained, target, np.nan),
                "before": before,
                "after": after,
                "factor": np.divide(after, before, out=np.ones_like(after), where=before > 0),
            }
        ),
        iterations=iterations,
        converged=converged,
        max_residual=max_residual,
        constrained=int(constrained.sum()),
        unconstrained=int((~constrained).sum()),
        total_before=total_before,
    )
    logger.info("Table 1410 rebalance: %s", report.summary())
    return balanced, report
