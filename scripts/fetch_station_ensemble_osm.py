"""Build the point files ``ortla station-ensemble`` reads from OpenStreetMap and Wikidata.

    python scripts/fetch_station_ensemble_osm.py osm PBF_DIR OUT_DIR
    python scripts/fetch_station_ensemble_osm.py beds PBF_DIR OUT_CSV
    python scripts/fetch_station_ensemble_osm.py wikidata OUT_CSV

``osm`` filters Geofabrik's England, Scotland and Wales extracts (``*-latest.osm.pbf``, from
download.geofabrik.de/europe/united-kingdom) with osmium into one CSV of points per feature set
(``lat,lon,name,tag,iata``; a way or relation is the mean of its vertices). ``beds`` writes
the places to stay (``@type,@lat,@lon,tourism``). ``wikidata`` asks the Wikidata query service
for every item with coordinates and an English Wikipedia article, tile by tile, with its
sitelink count; it pauses a minute between tiles, so it takes several hours.

Needs the ``osmium`` command (osmium-tool) for the first two.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request

FEATURES = {
    "malls": ["nwr/shop=mall"],
    "retail_parks": ["nwr/landuse=retail"],
    "ferry": ["nwr/amenity=ferry_terminal"],
    "airports": ["nwr/aeroway=aerodrome"],
    "places": ["n/place=city,town"],
    "motorway_j": ["n/highway=motorway_junction"],
    "castles": ["nwr/historic=castle"],
    "beaches": ["nwr/natural=beach"],
    "peaks": ["n/natural=peak"],
    "metro": ["nwr/station=subway,light_rail"],
    "bus_stations": ["nwr/amenity=bus_station"],
    "offices": ["nwr/building=office", "nwr/office"],
    "venues": ["nwr/amenity=theatre,cinema,nightclub,conference_centre,events_venue"],
    "schools": ["nwr/amenity=school"],
    "supermarkets": ["nwr/shop=supermarket"],
    "parking": ["nwr/park_ride=yes,bus,train,tram"],
}
# The first of these a feature has is written as its tag.
TAG_KEYS = ("place", "iata", "amenity", "station", "railway", "public_transport")
BEDS = ["nwr/tourism=hotel,guest_house,hostel,camp_site,caravan_site,apartment,chalet,motel"]
USER_AGENT = "ortla-station-ensemble/1.0"
WIKIDATA_QUERY = """SELECT ?item ?coord ?sl ?title WHERE {
  SERVICE wikibase:box { ?item wdt:P625 ?coord .
    bd:serviceParam wikibase:cornerSouthWest "Point(%s %s)"^^geo:wktLiteral ;
                    wikibase:cornerNorthEast "Point(%s %s)"^^geo:wktLiteral . }
  ?article schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> ; schema:name ?title .
  ?item wikibase:sitelinks ?sl .
}"""


def _extracts(pbf_dir: pathlib.Path) -> list[pathlib.Path]:
    return sorted(p for p in pbf_dir.glob("*-latest.osm.pbf") if not p.name.startswith("._"))


def _points(pbf: pathlib.Path, expressions: list[str], work: pathlib.Path) -> list[dict]:
    """Every feature in ``pbf`` matching ``expressions``, as its centre and properties."""
    filtered, lines = work / "filtered.pbf", work / "features.geojsonseq"
    osmium = shutil.which("osmium")
    if osmium is None:
        sys.exit("osmium is not installed (osmium-tool)")
    # The arguments are our own feature expressions and file paths.
    subprocess.run([osmium, "tags-filter", "-O", "-o", filtered, pbf, *expressions], check=True)  # noqa: S603
    subprocess.run(  # noqa: S603
        [osmium, "export", "-O", "-f", "geojsonseq", "-o", lines, filtered],
        check=True,
        stderr=subprocess.DEVNULL,
    )
    out = []
    for raw in lines.read_text().splitlines():
        line = raw.strip().lstrip("\x1e")
        if not line:
            continue
        feature = json.loads(line)
        coords = (feature.get("geometry") or {}).get("coordinates")
        if not coords:
            continue
        flat, stack = [], [coords]
        while stack:
            item = stack.pop()
            if isinstance(item[0], int | float):
                flat.append(item)
            else:
                stack.extend(item)
        out.append({
            "lon": sum(c[0] for c in flat) / len(flat),
            "lat": sum(c[1] for c in flat) / len(flat),
            "type": feature.get("id", "/").split("/")[0],
            "properties": feature.get("properties", {}),
        })  # fmt: skip
    return out


def osm(pbf_dir: pathlib.Path, out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "_work"
    work.mkdir(exist_ok=True)
    for name, expressions in FEATURES.items():
        rows = {}
        for pbf in _extracts(pbf_dir):
            for p in _points(pbf, expressions, work):
                pr = p["properties"]
                label = str(pr.get("name", "")).replace(",", " ").replace("\n", " ")
                tag = next((pr[k] for k in TAG_KEYS if pr.get(k)), "")
                key = (round(p["lat"], 5), round(p["lon"], 5), label)
                rows[key] = (p["lat"], p["lon"], label, tag, pr.get("iata", ""))
        with (out_dir / f"{name}.csv").open("w") as fh:
            fh.write("lat,lon,name,tag,iata\n")
            fh.writelines(
                f"{lat},{lon},{label},{tag},{iata}\n"
                for lat, lon, label, tag, iata in rows.values()
            )
        print(name, len(rows), flush=True)


def beds(pbf_dir: pathlib.Path, out_csv: pathlib.Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    work = out_csv.parent / "_work"
    work.mkdir(exist_ok=True)
    with out_csv.open("w") as fh:
        fh.write("@type,@lat,@lon,tourism\n")
        for pbf in _extracts(pbf_dir):
            for p in _points(pbf, BEDS, work):
                fh.write(
                    f"{p['type']},{p['lat']},{p['lon']},{p['properties'].get('tourism', '')}\n"
                )


def wikidata(out_csv: pathlib.Path) -> None:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json"}
    queue = [(lat, lon, 2.0) for lat in range(49, 61, 2) for lon in range(-8, 2, 2)]
    with out_csv.open("w") as fh:
        fh.write("lon,lat,sitelinks,title\n")
        while queue:
            lat, lon, step = queue.pop(0)
            query = WIKIDATA_QUERY % (lon, lat, lon + step, lat + step)
            url = "https://query.wikidata.org/sparql?" + urllib.parse.urlencode({"query": query})
            request = urllib.request.Request(url, headers=headers)  # noqa: S310 (https)
            try:
                with urllib.request.urlopen(request, timeout=90) as r:  # noqa: S310
                    found = json.load(r)["results"]["bindings"]
            except OSError as error:
                # A tile too big for the time limit is split in four; a rate limit is retried.
                if "429" not in str(error) and step > 0.25:  # noqa: PLR2004
                    half = step / 2
                    queue += [(lat, lon, half), (lat + half, lon, half), (lat, lon + half, half),
                              (lat + half, lon + half, half)]  # fmt: skip
                else:
                    queue.append((lat, lon, step))
                time.sleep(65)
                continue
            for b in found:
                x, y = b["coord"]["value"].removeprefix("Point(").removesuffix(")").split()
                title = b["title"]["value"].replace(",", " ").replace('"', "")
                fh.write(f"{x},{y},{b['sl']['value']},{title}\n")
            fh.flush()
            print(lat, lon, step, len(found), "tiles left", len(queue), flush=True)
            time.sleep(65)


if __name__ == "__main__":
    command, *args = sys.argv[1:] or ["-h"]
    if command == "osm" and len(args) == 2:
        osm(pathlib.Path(args[0]), pathlib.Path(args[1]))
    elif command == "beds" and len(args) == 2:
        beds(pathlib.Path(args[0]), pathlib.Path(args[1]))
    elif command == "wikidata" and len(args) == 1:
        wikidata(pathlib.Path(args[0]))
    else:
        sys.exit(__doc__)
