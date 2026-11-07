"""The data folder behind the Quick Start page.

:data:`DATA_SOURCES` says where each input belongs inside a chosen data
folder. :func:`create_structure` makes the folders and
:func:`write_configs` copies the templates in ``examples/`` into it with
their placeholder paths replaced, keeping comments and layout.
"""

from __future__ import annotations

import dataclasses
import pathlib

# Download links
#
# Paste the real download URLs in here. Leave a value as "" to leave
# that button showing a "link not set up yet" message instead of opening
# a page.
DOWNLOAD_LINKS: dict[str, str] = {
    "odm_matrix": "",
    "timetable_cif": "https://raildata.org.uk/dashboard/dataProduct/P-dbd92416-2f09-4f72-ad42-d53bbfec50f3/overview",
    "loading_northern": "https://raildata.org.uk/dataProduct/P-05429b52-0bed-4a7a-84d8-64410e356f4e/overview",
    "loading_awc": "https://raildata.org.uk/dataProduct/P-f1c26213-17f5-41dc-8740-08bb92d95e32/overview",
    "loading_emr": "https://raildata.org.uk/dataProduct/P-434dfc7c-9799-41ba-aa5d-1af8eb5b1a47/overview",
    "loading_tpe": "https://raildata.org.uk/dataProduct/P-77d7c8a4-e33b-4784-835c-7b8d068bcf3e/overview",
    "tpe_capacity": "https://raildata.org.uk/dashboard/dataProduct/P-1b4f5005-debf-443c-9298-c97147f9722f/overview",
    "nwr_track_model": "https://raildata.org.uk/dataProduct/P-9b4e960e-8bb6-438b-9722-34ae5768a48f/overview",
    "station_list": "https://docs.google.com/spreadsheets/d/1v1xoqMcFZnnwEAuDucfdqipFQLE1NhJh0PDbWnLO7nY/edit?usp=share_link",
    "table_1410": "https://dataportal.orr.gov.uk/media/1907/table-1410-passenger-entries-and-exits-and-interchanges-by-station.ods",
    "valid_stations": "https://docs.google.com/spreadsheets/d/1pGBT-sxEe8HFexA74jvXQRkwCkcQ_wSpNElt1MSBRsM/edit?usp=sharing",
}


@dataclasses.dataclass(frozen=True)
class DataSource:
    """One input file or folder, and where it should live."""

    key: str
    label: str
    description: str
    # Path relative to the chosen data-folder root.
    relative_path: str
    is_dir: bool
    # Human-readable list of commands that read this source.
    used_by: tuple[str, ...]

    @property
    def download_url(self) -> str:
        """Return the download link, or ``""`` if there is none."""
        return DOWNLOAD_LINKS.get(self.key, "")


DATA_SOURCES: list[DataSource] = [
    DataSource(
        key="odm_matrix",
        label="ODM",
        description="ORR's annual origin-destination matrix: journeys between every pair.",
        relative_path="ODM_for_RDM_2024-25.csv",
        is_dir=False,
        used_by=("OD matrix", "Loading analysis"),
    ),
    DataSource(
        key="timetable_cif",
        label="Timetable (CIF)",
        description="Network Rail's full CIF schedule export as newline-delimited JSON.",
        relative_path="timetable-data/CIF_ALL_FULL_DAILY_toc-full.json",
        is_dir=False,
        used_by=("Timetable",),
    ),
    DataSource(
        key="loading_northern",
        label="Northern loading counts",
        description="Northern's on-train counts, a folder of period files.",
        relative_path="northern-loading-data",
        is_dir=True,
        used_by=("Loading analysis", "Hindcast"),
    ),
    DataSource(
        key="loading_awc",
        label="Avanti West Coast loading counts",
        description="Avanti West Coast's on-train counts (one CSV).",
        relative_path="awc-loading-data.csv",
        is_dir=False,
        used_by=("Loading analysis",),
    ),
    DataSource(
        key="loading_emr",
        label="EMR loading counts",
        description="East Midlands Railway's on-train counts (one CSV).",
        relative_path="emr-loading-data.csv",
        is_dir=False,
        used_by=("Loading analysis",),
    ),
    DataSource(
        key="loading_tpe",
        label="TPE loading counts",
        description="TransPennine Express's on-train counts (one CSV).",
        relative_path="tpe-loading-data.csv",
        is_dir=False,
        used_by=("Loading analysis",),
    ),
    DataSource(
        key="nwr_track_model",
        label="Network Rail track model",
        description="Track centre-lines shapefile, with its .dbf, .shx and .prj files.",
        relative_path="NWR_TrackModel/NWR_TrackCentreLines.shp",
        is_dir=False,
        used_by=("Track distance", "Loading analysis", "Service proposal", "Discovery"),
    ),
    DataSource(
        key="station_list",
        label="Station list",
        description="stations.csv: CRS code, TIPLOC, latitude and longitude of every station.",
        relative_path="stations.csv",
        is_dir=False,
        used_by=("Track distance", "Loading analysis", "Service proposal", "Discovery"),
    ),
    DataSource(
        key="table_1410",
        label="ORR table 1410",
        description="Entries, exits and interchanges by station.",
        relative_path=("table-1410-passenger-entries-and-exits-and-interchanges-by-station.csv"),
        is_dir=False,
        used_by=("Loading analysis", "Service proposal", "Discovery"),
    ),
    DataSource(
        key="valid_stations",
        label="Valid station list",
        description="The CRS codes to keep when filtering the ODM and the timetable.",
        relative_path="valid_stations.csv",
        is_dir=False,
        used_by=("OD matrix", "Timetable"),
    ),
    DataSource(
        key="bplan",
        label="BPLAN Geography",
        description="Network Rail's BPLAN Geography extract (timing links and points).",
        relative_path="bplan-geography-extract.txt",
        is_dir=False,
        used_by=("BPLAN geography",),
    ),
    DataSource(
        key="tag_databook",
        label="TAG Data Book",
        description="DfT's TAG Data Book workbook, for the economic appraisal.",
        relative_path="TAG-data-book.xlsm",
        is_dir=False,
        used_by=("Scenario studio", "Service discovery"),
    ),
]

# The templates in examples/ use these two placeholders for the data
# folder and the output folder.
_DATA_PLACEHOLDER = "/path/to/datasources"
_OUTPUT_PLACEHOLDER = '"outputs/'


def target_path(base_dir: pathlib.Path, source_key: str) -> pathlib.Path:
    """Where ``source_key`` should live under a chosen data folder."""
    for source in DATA_SOURCES:
        if source.key == source_key:
            return base_dir / source.relative_path
    raise KeyError(source_key)


def create_structure(base_dir: pathlib.Path) -> list[pathlib.Path]:
    """Create the data folder and its subfolders.

    Only folders are made; the data files themselves have to be
    downloaded. Folders that already exist are left alone.
    """
    from orta.config import COMMANDS  # noqa: PLC0415

    folders = [base_dir, base_dir / "output"]
    folders += [base_dir / "output" / command for command in COMMANDS]
    for source in DATA_SOURCES:
        rel = pathlib.Path(source.relative_path)
        folders.append(base_dir / rel if source.is_dir else base_dir / rel.parent)
    created = [f for f in dict.fromkeys(folders) if not f.exists()]
    for folder in created:
        folder.mkdir(parents=True, exist_ok=True)
    return created


def write_configs(base_dir: pathlib.Path, template_dir: pathlib.Path) -> list[pathlib.Path]:
    """Copy the templates into ``base_dir/configs``.

    The data and output placeholders are replaced with the chosen
    folder, and comments and layout are kept. The templates themselves
    are not touched. Returns the files written.
    """
    dest_dir = base_dir / "configs"
    dest_dir.mkdir(parents=True, exist_ok=True)
    data = base_dir.as_posix()
    output = (base_dir / "output").as_posix()
    written = []
    for src in sorted(template_dir.glob("*.yml")):
        if src.name.startswith("."):
            continue
        text = src.read_text(encoding="utf-8")
        text = text.replace(_DATA_PLACEHOLDER, data)
        text = text.replace(_OUTPUT_PLACEHOLDER, f'"{output}/')
        dest = dest_dir / src.name
        dest.write_text(text, encoding="utf-8")
        written.append(dest)
    return written
