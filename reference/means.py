"""Mean period matrices for operators without their own counts.

An operator without counts borrows the counted operators' temporal factors. A flow with
either end within 20 km of Charing Cross takes the mean of the four London operators'
period matrices; every other flow takes the mean over all nine counted operators.

    python3 means.py DATA OUT

reads ``OUT/baselines/<op>/<op>_period_matrices`` (written by ``ortla loading``) and
``DATA/stations.csv``, and writes three folders of matrices into OUT.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

LONDON_OPS = ["greater_anglia", "gtr", "southeastern", "swr"]
ALL_OPS = [*LONDON_OPS, "northern", "tpe", "emr", "gwr", "awc"]
CHARING_CROSS = (51.5080, -0.1247)
EARTH_RADIUS_KM = 6371.0


def matrix_dir(out: Path, op: str) -> Path:
    return out / "baselines" / op / f"{op}_period_matrices"


def mean_of(out: Path, ops: list[str], dest: Path) -> None:
    """Write the cell-wise mean of the operators' matrices, for the file names they share."""
    dest.mkdir(exist_ok=True)
    names = set.intersection(
        *(
            {p.name for p in matrix_dir(out, op).glob("*.csv") if not p.name.startswith("._")}
            for op in ops
        )
    )
    for name in sorted(names):
        mats = [pd.read_csv(matrix_dir(out, op) / name, index_col=0) for op in ops]
        ref = mats[0]
        if any(m.shape != ref.shape for m in mats):
            raise ValueError(f"{name}: operators' matrices differ in shape")
        stack = np.stack(
            [m.reindex(index=ref.index, columns=ref.columns).to_numpy(float) for m in mats]
        )
        pd.DataFrame(np.nanmean(stack, axis=0), index=ref.index, columns=ref.columns).to_csv(
            dest / name
        )
    print(f"{dest.name}: {len(names)} matrices", flush=True)


def london_stations(data: Path, radius_km: float) -> set[str]:
    st = pd.read_csv(data / "stations.csv", dtype=str).dropna(subset=["lat", "long"])
    lat, lon = np.radians(st["lat"].astype(float)), np.radians(st["long"].astype(float))
    lat0, lon0 = np.radians(CHARING_CROSS)
    hav = (
        np.sin((lat - lat0) / 2) ** 2 + np.cos(lat0) * np.cos(lat) * np.sin((lon - lon0) / 2) ** 2
    )
    km = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(hav))
    return set(st.loc[km <= radius_km, "crsCode"].str.upper())


def flow_matrices(data: Path, out: Path, radius_km: float = 20.0) -> None:
    """London mean where either end is inside the radius, national mean elsewhere."""
    inside = london_stations(data, radius_km)
    london, national = out / "london_mean_period_matrices", out / "national_mean_period_matrices"
    dest = out / f"flow_period_matrices_{radius_km:g}km"
    dest.mkdir(exist_ok=True)
    for path in sorted(london.glob("odm_*.csv")):
        if not (national / path.name).exists():
            continue
        lm = pd.read_csv(path, index_col=0)
        nm = pd.read_csv(national / path.name, index_col=0).reindex(
            index=lm.index, columns=lm.columns
        )
        near = lm.index.isin(inside)[:, None] | lm.columns.isin(inside)[None, :]
        pd.DataFrame(np.where(near, lm, nm), index=lm.index, columns=lm.columns).to_csv(
            dest / path.name
        )
    print(f"{len(inside)} stations within {radius_km:g} km", flush=True)


def main(data: Path, out: Path) -> None:
    mean_of(out, LONDON_OPS, out / "london_mean_period_matrices")
    mean_of(out, ALL_OPS, out / "national_mean_period_matrices")
    flow_matrices(data, out)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python3 means.py DATA OUT")
    main(Path(sys.argv[1]), Path(sys.argv[2]))
