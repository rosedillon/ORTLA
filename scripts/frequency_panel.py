"""Estimate the frequency elasticity *within* a station pair, over time.

The gravity model's frequency term (b4 = 0.64 among served pairs) is
cross-sectional: it is fitted across pairs in one year, so it compares a
half-hourly urban corridor with a two-hourly rural one and attributes
the demand difference to frequency.
``docs/notes/ACCURACY_IMPROVEMENTS.md`` §1 shows the frequency term
breaks down there: the apparent exponent falls from -0.28 between
corridors to -0.10 (t = -1.8) within them, so log frequency is largely
standing in for corridor type.

Getting past that means watching one pair's demand move when that same
pair's frequency moves. This builds that panel: the ORR ODM for several
financial years against the timetable published in each of those years,
read from the archive CIFs by :mod:`ortla.cif_archive`.
Differencing removes everything fixed about the pair (its population,
its distance, its corridor type), leaving the association between a
change in service and a change in demand on the same flow.

This is an association, not a causal elasticity. Operators add trains
where demand is growing, so frequency is endogenous to demand; without
an instrument the estimate is an upper bound on the causal effect. What
it does remove is the between-corridor confound that makes the
cross-sectional figure uninterpretable, and that is the specific defect
it is here to address.

Usage::

    python3 scripts/frequency_panel.py [--min-journeys N] [--out DIR]
"""

from __future__ import annotations

import argparse
import itertools
import logging
import os
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from ortla.gravity_model import load_direct_frequency
from ortla.od_matrix import read_odm

DATASOURCES = pathlib.Path(os.environ.get("ORTLA_DATA", "datasources"))
# The era timetables; ORM_OUT points a regeneration at its own
# (paper/regen).
OUTPUT = pathlib.Path(os.environ.get("ORM_OUT", DATASOURCES / "output_paper"))

# Each financial year of the ODM, with the published timetable that was
# in force for most of it. A year with no archive extract cannot enter
# the panel.
PANEL = [
    ("2018-19", DATASOURCES / "old-odm" / "ODM_for_rdm_2018-19.csv", ["may18", "dec18"]),
    ("2019-20", DATASOURCES / "old-odm" / "ODM_for_rdm_2019-20.csv", ["may19", "dec19"]),
    ("2022-23", DATASOURCES / "old-odm" / "ODM_for_rdm_2022-23.csv", ["may22"]),
    ("2023-24", DATASOURCES / "old-odm" / "ODM_for_rdm_2023-24.csv", ["may23", "dec23"]),
    ("2024-25", DATASOURCES / "ODM_for_RDM_2024-25.csv", ["jun24"]),
]
# 2020-21 and 2021-22 stay out: no archive extract covers either,
# because the covid emergency timetables were never published as an
# archive CIF. That gap is why the covid span enters the panel only as
# the 2019-20 -> 2022-23 jump. Releases that carry both codes, used to
# map the NLC-only releases.
NLC_REFERENCES = [
    DATASOURCES / "ODM_for_RDM_2024-25.csv",
    DATASOURCES / "old-odm" / "ODM_for_rdm_2023-24.csv",
]


def gapped(span: str) -> bool:
    """Whether a ``"2019-20 -> 2022-23"`` span skips a financial year.

    The covid years have no archive extract, so the panel steps straight
    over them. Those steps are not comparable with a one-year difference
    and are reported separately. Read off the span itself, since a year
    whose timetable is missing drops out of the panel and widens its
    neighbours' span.
    """
    before, after = span.split(" -> ")
    return int(after[:4]) - int(before[:4]) > 1


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def unordered(frame: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Collapse to unordered pairs.

    The ODM carries no directional information (its two directions are
    the same estimate), so an ordered pair would double-count and invent
    a direction the data does not have.
    """
    lo = np.minimum(frame[a], frame[b])
    hi = np.maximum(frame[a], frame[b])
    return frame.assign(pair_a=lo, pair_b=hi)


def load_frequency(eras: list[str]) -> pd.DataFrame:
    """Weekday trains a day per pair, averaged over an era."""
    frames = []
    for era in eras:
        path = OUTPUT / f"timetable_{era}" / "route_frequency_summary.csv"
        if not path.exists():
            logger.warning("  no route frequency for %s (%s)", era, path)
            continue
        # Second codes folded in (PDX into PAD), as the ODM holds them.
        routes = pd.DataFrame(
            [(o, d, n) for (o, d), n in load_direct_frequency(path).items()],
            columns=["origin_crs", "destination_crs", "daily_trains"],
        )
        routes = unordered(routes, "origin_crs", "destination_crs")
        # Average the two directions: a pair's service level is the
        # service in each direction, not their sum.
        frames.append(routes.groupby(["pair_a", "pair_b"], as_index=False)["daily_trains"].mean())
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames)
    return combined.groupby(["pair_a", "pair_b"], as_index=False)["daily_trains"].mean()


def build_panel(min_journeys: float) -> pd.DataFrame:
    rows = []
    for year, odm_path, eras in PANEL:
        logger.info("%s: reading %s", year, odm_path.name)
        odm = read_odm(odm_path, nlc_reference_paths=NLC_REFERENCES)
        odm = unordered(odm, "origin_tlc", "destination_tlc")
        odm = odm.groupby(["pair_a", "pair_b"], as_index=False)["journeys"].sum()
        odm = odm[odm["pair_a"] != odm["pair_b"]]

        freq = load_frequency(eras)
        if freq.empty:
            logger.warning("  skipping %s: no timetable", year)
            continue
        merged = odm.merge(freq, on=["pair_a", "pair_b"], how="inner")
        merged = merged[(merged["journeys"] >= min_journeys) & (merged["daily_trains"] > 0)]
        merged["year"] = year
        logger.info(
            "  %s: %d directly served pairs above %g journeys",
            year,
            len(merged),
            min_journeys,
        )
        rows.append(merged)
    return pd.concat(rows, ignore_index=True)


def first_differences(panel: pd.DataFrame) -> pd.DataFrame:
    """Change in log demand against log frequency, pair by pair."""
    years = [y for y, _, _ in PANEL if y in set(panel["year"])]
    out = []
    for before, after in itertools.pairwise(years):
        a = panel[panel["year"] == before].set_index(["pair_a", "pair_b"])
        b = panel[panel["year"] == after].set_index(["pair_a", "pair_b"])
        both = a.join(b, how="inner", lsuffix="_0", rsuffix="_1")
        diff = pd.DataFrame(
            {
                "d_ln_journeys": np.log(both["journeys_1"] / both["journeys_0"]),
                "d_ln_frequency": np.log(both["daily_trains_1"] / both["daily_trains_0"]),
                "span": f"{before} -> {after}",
            }
        ).reset_index()
        out.append(diff.replace([np.inf, -np.inf], np.nan).dropna())
    return pd.concat(out, ignore_index=True)


def regress(diff: pd.DataFrame, label: str) -> dict:
    """OLS of d_ln_journeys on d_ln_frequency, dyadic clustering.

    Residuals on flows sharing a station are correlated (the route
    effect in ``docs/notes/ACCURACY_IMPROVEMENTS.md`` is 63% of the
    variance), so an unclustered standard error here would be far too
    small. A flow belongs to both its stations' clusters: clustering on
    pair_a alone split a station's flows by whether its partner sorted
    before or after it. Every two flows sharing a station enter the
    variance (Aronow, Samii and Assenova 2015), so the station sums
    count a pair's own observations twice and the pair sums take one
    copy back off.
    """
    y = diff["d_ln_journeys"].to_numpy()
    x = diff["d_ln_frequency"].to_numpy()
    design = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta

    bread = np.linalg.inv(design.T @ design)
    scores = pd.DataFrame(design * resid[:, None])
    a, b = diff["pair_a"].to_numpy(), diff["pair_b"].to_numpy()
    by_station = scores.groupby(a).sum().add(scores.groupby(b).sum(), fill_value=0)
    by_pair = scores.groupby([a, b]).sum()
    meat = (by_station.T @ by_station - by_pair.T @ by_pair).to_numpy()
    cov = bread @ meat @ bread
    se = float(np.sqrt(cov[1, 1])) if cov[1, 1] > 0 else np.nan

    total = float(((y - y.mean()) ** 2).sum())
    return {
        "span": label,
        "n_pairs": len(diff),
        "n_clusters": len(by_station),
        "elasticity": float(beta[1]),
        "se": se,
        "t": float(beta[1] / se) if se else np.nan,
        "r2": 1 - float((resid**2).sum()) / total if total else np.nan,
        "median_d_ln_f": float(np.median(x)),
        "share_f_changed": float((np.abs(x) > 0.05).mean()),  # noqa: PLR2004 (5%)
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-journeys", type=float, default=100)
    parser.add_argument("--out", type=pathlib.Path, default=OUTPUT / "analysis")
    args = parser.parse_args()

    panel = build_panel(args.min_journeys)
    args.out.mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.out / "frequency_panel.csv", index=False)

    diff = first_differences(panel)
    diff.to_csv(args.out / "frequency_panel_differences.csv", index=False)

    results = [regress(d, span) for span, d in diff.groupby("span")]
    # Pooled, with a span dummy absorbing each period's national growth,
    # so the elasticity is identified off differences between pairs
    # within a span, not off the national demand trend.
    pooled = diff.copy()
    pooled["d_ln_journeys"] -= pooled.groupby("span")["d_ln_journeys"].transform("mean")
    pooled["d_ln_frequency"] -= pooled.groupby("span")["d_ln_frequency"].transform("mean")
    results.append(regress(pooled, "pooled, span-demeaned"))

    # Attenuation check. A first difference of a noisily measured
    # regressor is biased toward zero, and Delta ln f is noisy: it is
    # read off one reference weekday per extract. If that is what is
    # driving the small estimate, the elasticity must rise as the
    # frequency change gets large relative to the noise. If it does not,
    # the small estimate is the signal.
    bands = [(0.02, 0.10), (0.10, 0.25), (0.25, 0.60), (0.60, np.inf)]
    for lo, hi in bands:
        size = pooled["d_ln_frequency"].abs()
        band = pooled[(size >= lo) & (size < hi)]
        if len(band) > 200:
            label = f"|d ln f| in [{lo:.2f}, {hi:.2f})"
            results.append(regress(band, label))

    # Any span that skips a financial year jumps the pandemic, when
    # frequency and demand both collapsed for reasons that have nothing
    # to do with a service decision. Derive it from each span's years
    # rather than naming it, so that adding or losing an era cannot
    # silently leave this row equal to the pooled one.
    clean = diff[~diff["span"].map(gapped)].copy()
    for column in ("d_ln_journeys", "d_ln_frequency"):
        clean[column] -= clean.groupby("span")[column].transform("mean")
    results.append(regress(clean, "pooled, excluding the covid span"))

    table = pd.DataFrame(results)
    table.to_csv(args.out / "frequency_panel_elasticity.csv", index=False)
    print("\nWithin-pair frequency elasticity (first differences):")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
