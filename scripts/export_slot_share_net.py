"""Write the packaged slot share models from cases 33 and 117.

``slot_share_m3.json`` is case 33's timing table and theta 0.06;
``slot_share_neural.json`` adds case 117's adopted network (H1, the fit
on all rows, three seeds) with the station table it was trained on: km
to London Bridge and log(1 + 2024-25 entries and exits). Needs PyTorch
to read the fitted weights.

    python3 scripts/export_slot_share_net.py /Volumes/ENGG341/cases
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import torch

from ortla.models import BUILTIN
from ortla.network_assignment import SlotShareModel
from ortla.slot_net import SlotShareNet


def main(cases: pathlib.Path) -> None:
    """Write both models into the package."""
    m3 = SlotShareModel.load(cases / "_baselines_size" / "slot_share_model_m3.json")
    m3.save(BUILTIN["m3"])
    saved = torch.load(
        cases / "117_neural_slot_share" / "output" / "runs" / "all_H1_rows.pt",
        weights_only=False,
    )
    st = pd.read_pickle(cases / "31_sdft_correlates" / "output" / "stations.pkl")
    st = st[st.index.notna() & ~st.index.duplicated()]
    mean, scale = saved["stats"]
    net = SlotShareNet(
        mean=np.asarray(mean, dtype=float),
        scale=np.asarray(scale, dtype=float),
        fits=tuple(
            (
                f["hidden.weight"].numpy(),
                f["hidden.bias"].numpy(),
                f["out.weight"].numpy().ravel(),
                np.float32(f["out.bias"].item()),
            )
            for f in saved["nets"]
        ),
        km_london=st["km_london"].astype(float),
        ln_footfall=np.log1p(st["ee_2425"].astype(float)),
    )
    SlotShareModel(
        timing=m3.timing,
        theta=m3.theta,
        source="case 33 m3 timing and theta, case 117 H1 network (all rows, 3 seeds)",
        net=net,
    ).save(BUILTIN["neural"])


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]))
