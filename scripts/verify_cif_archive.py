"""Check the fixed-width CIF reader against the same extract's JSON.

Network Rail publishes one daily extract in both formats. Parsing both
and comparing them record by record is the only way to establish that
:mod:`ortla.cif_archive` reads the archive extracts correctly,
since those are published in fixed width only and have no JSON
counterpart.

Usage::

    python3 scripts/verify_cif_archive.py <extract.CIF> <extract.json> \
        [--limit N]

Reports, per field, how many schedules disagree, and exits non-zero on
any mismatch in a field the model reads.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from ortla import cif_archive

# The fields the model actually reads, from
# ``timetable._service_summary`` and ``timetable._call_rows``. A
# disagreement in any of these is a failure; the rest are reported but
# tolerated.
SCHEDULE_FIELDS = [
    "CIF_train_uid",
    "CIF_stp_indicator",
    "atoc_code",
    "schedule_days_runs",
    "schedule_start_date",
    "schedule_end_date",
    "train_status",
]
SEGMENT_FIELDS = [
    "signalling_id",
    "CIF_train_category",
    "CIF_power_type",
    "CIF_timing_load",
    "CIF_speed",
    "CIF_headcode",
    "CIF_train_service_code",
]
LOCATION_FIELDS = [
    "location_type",
    "tiploc_code",
    "tiploc_instance",
    "arrival",
    "departure",
    "pass",
    "public_arrival",
    "public_departure",
    "platform",
]


def iter_json_schedules(path: pathlib.Path):
    with path.open() as handle:
        for line in handle:
            if '"JsonScheduleV1"' not in line:
                continue
            yield json.loads(line)["JsonScheduleV1"]


def iter_json_tiplocs(path: pathlib.Path):
    with path.open() as handle:
        for line in handle:
            if '"TiplocV1"' not in line:
                continue
            yield json.loads(line)["TiplocV1"]


def blank_to_none(value):
    """Treat blanks and None alike, as the two formats differ on them.

    The JSON feed carries a field the CIF leaves blank as a run of
    spaces (``"    "`` for an absent signalling id), where the
    fixed-width reader returns None.
    """
    if value is None:
        return None
    return str(value).strip() or None


def compare_tiplocs(cif_path: pathlib.Path, json_path: pathlib.Path) -> dict:
    counts: dict[str, int] = collections.Counter()
    from_json = {r["tiploc_code"]: r for r in iter_json_tiplocs(json_path)}
    seen = 0
    for record in cif_archive.iter_tiploc_records(cif_path):
        seen += 1
        other = from_json.get(record["tiploc_code"])
        if other is None:
            counts["missing_in_json"] += 1
            continue
        for field in ("crs_code", "tps_description", "stanox", "nalco"):
            if blank_to_none(record[field]) != blank_to_none(other.get(field)):
                counts[f"tiploc.{field}"] += 1
    counts["n_cif"] = seen
    counts["n_json"] = len(from_json)
    return counts


def schedule_key(record: dict) -> tuple:
    """Identify a schedule across the two formats.

    Not position: the JSON feed orders a UID's short-term cancellation
    before its permanent schedule where the fixed-width file orders it
    after, so a positional walk goes out of step on the first such pair.
    """
    return (
        record["CIF_train_uid"],
        record["CIF_stp_indicator"],
        record["schedule_start_date"],
        record["schedule_end_date"],
    )


def compare_schedules(
    cif_path: pathlib.Path, json_path: pathlib.Path, limit: int
) -> tuple[dict, dict]:
    counts: dict[str, int] = collections.Counter()
    examples: dict[str, tuple] = {}

    def note(field, mine, theirs, uid=""):
        counts[field] += 1
        examples.setdefault(field, (f"{uid} {mine}", theirs))

    from_json = {}
    for n, record in enumerate(iter_json_schedules(json_path)):
        if n >= limit:
            break
        from_json[schedule_key(record)] = record

    for n, mine in enumerate(cif_archive.iter_schedule_records(cif_path)):
        if n >= limit:
            break
        theirs = from_json.get(schedule_key(mine))
        if theirs is None:
            # The two extracts are taken days apart, so each holds
            # schedules the other does not. Counted, not treated as a
            # parse failure.
            counts["unmatched_schedule"] += 1
            continue
        counts["n_compared"] += 1

        for field in SCHEDULE_FIELDS:
            if blank_to_none(mine[field]) != blank_to_none(theirs.get(field)):
                note(field, mine[field], theirs.get(field), mine["CIF_train_uid"])

        my_seg = mine["schedule_segment"]
        their_seg = theirs.get("schedule_segment") or {}
        for field in SEGMENT_FIELDS:
            if blank_to_none(my_seg[field]) != blank_to_none(their_seg.get(field)):
                note(
                    f"segment.{field}", my_seg[field], their_seg.get(field), mine["CIF_train_uid"]
                )

        my_locs = my_seg["schedule_location"]
        their_locs = their_seg.get("schedule_location") or []
        if len(my_locs) != len(their_locs):
            note("n_locations", len(my_locs), len(their_locs), mine["CIF_train_uid"])
            continue
        for a, b in zip(my_locs, their_locs, strict=True):
            for field in LOCATION_FIELDS:
                if blank_to_none(a.get(field)) != blank_to_none(b.get(field)):
                    note(f"location.{field}", a.get(field), b.get(field), mine["CIF_train_uid"])

    return counts, examples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cif", type=pathlib.Path)
    parser.add_argument("json_feed", type=pathlib.Path)
    parser.add_argument("--limit", type=int, default=50_000)
    parser.add_argument("--skip-tiplocs", action="store_true")
    args = parser.parse_args()

    print(f"fixed-width detected: {cif_archive.is_fixed_width_cif(args.cif)}")
    print(f"json detected as fixed width: {cif_archive.is_fixed_width_cif(args.json_feed)}")

    if not args.skip_tiplocs:
        tiploc_counts = compare_tiplocs(args.cif, args.json_feed)
        print("\nTIPLOC reference:")
        for key, value in sorted(tiploc_counts.items()):
            print(f"  {key}: {value:,}")

    counts, examples = compare_schedules(args.cif, args.json_feed, args.limit)
    print(
        f"\nSchedules compared: {counts.pop('n_compared', 0):,} "
        f"(unmatched across the two extracts: "
        f"{counts.pop('unmatched_schedule', 0):,})"
    )
    if not counts:
        print("  no disagreements in any compared field")
        return 0
    print("  disagreements:")
    for key, value in sorted(counts.items(), key=lambda kv: -kv[1]):
        mine, theirs = examples.get(key, ("", ""))
        print(f"  {key}: {value:,}   e.g. ours={mine!r} json={theirs!r}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
