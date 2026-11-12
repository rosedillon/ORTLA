"""Save today's c2c loadings (Rail Data Marketplace API 1075) if they differ from the last save.

The feed has no dates: each call returns sums over the last 28 days, so a new day's file shares
most of its runs with yesterday's. Files 28 days apart don't overlap.

The endpoint and key live outside the repo, in ``~/.config/ortla/rdm.env``:

    C2C_URL=https://...
    RDM_API_KEY=...

    python3 scripts/fetch_c2c.py [output folder]
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import pathlib
import sys
import urllib.request

ENV = pathlib.Path.home() / ".config" / "ortla" / "rdm.env"
OUT = pathlib.Path(os.environ.get("ORTLA_DATA", "datasources")) / "c2c-loading-data"


def _env() -> dict[str, str]:
    lines = ENV.read_text().splitlines()
    return dict(line.split("=", 1) for line in lines if "=" in line and not line.startswith("#"))


def _digest(records: list[dict]) -> str:
    rows = sorted(json.dumps(r, sort_keys=True) for r in records)
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


def main(out: pathlib.Path = OUT) -> None:
    env = _env()
    if not env["C2C_URL"].startswith("https://"):
        sys.exit("c2c: C2C_URL must be an https address")
    headers = {
        "x-apikey": env["RDM_API_KEY"],
        "User-Agent": "ortla",
    }  # urllib's own agent gets 403
    request = urllib.request.Request(env["C2C_URL"], headers=headers)  # noqa: S310
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 (https checked above)
        data = json.load(response)
    if not data.get("done", True):
        sys.exit("c2c: response says it is not the whole set (done is false); nothing saved")
    saved = sorted(out.glob("c2c_loadings_fetched_*.json"))
    last = json.loads(saved[-1].read_text())["records"] if saved else []
    if len(data["records"]) < len(last) / 2:
        n = len(data["records"])
        sys.exit(f"c2c: {n} records against {len(last)} last time; looks partial, nothing saved")
    if saved and _digest(last) == _digest(data["records"]):
        print(f"c2c: same as {saved[-1].name}; nothing saved")
        return
    path = out / f"c2c_loadings_fetched_{datetime.date.today():%Y-%m-%d}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    print(f"c2c: {len(data['records'])} records to {path.name}")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else OUT)
