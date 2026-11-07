"""Split DfT table 1415 into one table-1410 CSV per financial year.

Why this exists
---------------
``stations.load_table_1410`` takes a single year of station
entries-and-exits and the loading pipeline uses it two ways: as the
marginal target for the Furness rebalance of the ODM, and as the
footfall term in the gravity fit. Both are only meaningful if the
footfall year matches the ODM year. The published single-year table 1410
downloads each cover one year, so running several hindcast years off one
file silently mixes them.

Table 1415 is the same series as table 1410, all years in one sheet, so
one parse gives every year the pipeline might want on a consistent
station base. The national total rose 24.6% between 2022-23 and 2024-25,
and the shape moved too: 17% of stations shifted their share of the
national total by more than 20%. The rebalance cannot absorb the wrong
year.

Usage
-----
    python scripts/build_station_footfall.py TABLE_1415.ods OUT_DIR

Writes ``OUT_DIR/table_1410_<start>_<end>.csv`` (e.g.
``table_1410_2024_25.csv``) for every year column in the sheet, with the
two columns ``load_table_1410`` reads plus the station name and region
for traceability.

The ODS is parsed directly out of the zip because ``odfpy`` is not a
dependency of this project and this script runs once per data refresh.
"""

from __future__ import annotations

import csv
import pathlib
import re
import sys
import xml.etree.ElementTree as ET
import zipfile

NS = {
    "table": "urn:oasis:names:tc:opendocument:xmlns:table:1.0",
    "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
    "office": "urn:oasis:names:tc:opendocument:xmlns:office:1.0",
}
SHEET = "1415a_Entries_and_Exits"

# "Apr 2024 to Mar 2025", sometimes with a "[b]" break-in-series marker.
_YEAR_HEADER = re.compile(r"Apr (\d{4}) to Mar (\d{4})")

# Script name, source ODS, output directory.
_ARGC = 3

# Table 1415 uses these in place of numbers: not available, not
# applicable.
_MISSING = {"[x]", "[z]", "", None}

# load_table_1410 looks these up by name; the TLC one carries a real
# newline.
TLC_COLUMN = "Three Letter Code\n(TLC)"
ENTRIES_COLUMN = "Entries and exits - All tickets"


def _read_sheet(path: pathlib.Path, name: str) -> list[list[str | None]]:
    """Return one ODS sheet as rows, expanding repeat counts."""
    # The input is a DfT spreadsheet the user downloaded, not untrusted
    # network data, and defusedxml is not a dependency here.
    root = ET.fromstring(zipfile.ZipFile(path).read("content.xml"))  # noqa: S314
    for tbl in root.iter(f"{{{NS['table']}}}table"):
        if tbl.get(f"{{{NS['table']}}}name") != name:
            continue
        rows: list[list[str | None]] = []
        for row in tbl.findall("table:table-row", NS):
            cells: list[str | None] = []
            for cell in row.findall("table:table-cell", NS):
                repeat = int(cell.get(f"{{{NS['table']}}}number-columns-repeated", 1))
                value = cell.get(f"{{{NS['office']}}}value")
                if value is None:
                    text = "".join("".join(p.itertext()) for p in cell.findall("text:p", NS))
                    value = text or None
                # A trailing run of empty cells pads the row to the
                # sheet width and can repeat thousands of times; cap it.
                cells.extend([value] * min(repeat, 200))
            while cells and cells[-1] is None:
                cells.pop()
            rows.append(cells)
        return rows
    raise SystemExit(f"sheet {name!r} not found in {path}")


def _number(value: str | None) -> int | None:
    if value in _MISSING:
        return None
    try:
        return round(float(str(value).replace(",", "")))
    except ValueError:
        return None


def main(argv: list[str]) -> int:
    """Write one CSV per year of table 1415 into the given directory."""
    if len(argv) != _ARGC:
        print(__doc__)
        return 2
    source = pathlib.Path(argv[1])
    out_dir = pathlib.Path(argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_sheet(source, SHEET)
    header_index = next(i for i, r in enumerate(rows) if "Sort" in r[:1])
    header = rows[header_index]
    body = rows[header_index + 1 :]

    name_col = header.index("Station name")
    tlc_col = header.index("Three Letter Code(TLC)")
    region_col = header.index("Region")

    years = [
        (i, m.group(1), m.group(2)[2:])
        for i, h in enumerate(header)
        if h and (m := _YEAR_HEADER.match(h))
    ]

    for col, start, end in years:
        out = out_dir / f"table_1410_{start}_{end}.csv"
        written = 0
        with out.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Station name", "Region", TLC_COLUMN, ENTRIES_COLUMN])
            for row in body:
                if len(row) <= col:
                    continue
                tlc = (row[tlc_col] or "").strip().upper()
                entries = _number(row[col])
                # Stations that had not opened, or had closed, in a
                # given year carry [z]; dropping them keeps each file a
                # true snapshot.
                if not tlc or tlc in _MISSING or entries is None:
                    continue
                writer.writerow([row[name_col], row[region_col] or "", tlc, entries])
                written += 1
        print(f"{out.name}: {written} stations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
