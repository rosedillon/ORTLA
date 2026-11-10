"""Build ``ortla/models/corpus_tiploc_crs.csv`` from Network Rail's CORPUS extract.

    python3 scripts/build_corpus.py datasources/corpus/NLC20261106.xml

CORPUS gives a CRS on one TIPLOC per station. Platform TIPLOCs (CLPHMJC, VICTRIE) carry none and
come from the timetable instead (``stations.tiploc_to_crs_map``).
"""

from __future__ import annotations

import csv
import pathlib
import sys
import xml.etree.ElementTree as ET

NS = "{http://tempuri.org/XMLSchema.xsd}"
OUT = pathlib.Path(__file__).resolve().parents[1] / "src" / "ortla" / "models"


def read(path: pathlib.Path) -> list[dict[str, str]]:
    rows = []
    for _, item in ET.iterparse(path):  # noqa: S314
        if item.tag == f"{NS}TSDBDataItem":
            row = dict.fromkeys(("NLCTiplocCode", "NLCCrsCode"), "")
            row.update({child.tag.removeprefix(NS): (child.text or "").strip() for child in item})
            rows.append(row)
            item.clear()
    return rows


def tiploc_to_crs(rows: list[dict[str, str]]) -> dict[str, str]:
    return {
        r["NLCTiplocCode"]: r["NLCCrsCode"] for r in rows if r["NLCTiplocCode"] and r["NLCCrsCode"]
    }


def main() -> None:
    source = pathlib.Path(sys.argv[1])
    mapping = tiploc_to_crs(read(source))
    with (OUT / "corpus_tiploc_crs.csv").open("w", newline="", encoding="utf-8") as out:
        out.write(f"# TIPLOC -> CRS from Network Rail CORPUS ({source.name})\n")
        writer = csv.writer(out)
        writer.writerow(["tiploc", "crs"])
        writer.writerows(sorted(mapping.items()))
    print(f"{len(mapping):,} TIPLOCs written")


if __name__ == "__main__":
    main()
