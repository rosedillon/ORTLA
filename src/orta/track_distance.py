"""Distance between stations along the track.

Network Rail's track map (a shapefile of every stretch of line) becomes
a graph: each stretch of track is an edge weighted by its length, and
the points where stretches meet are its nodes. The shortest track route
between two stations is a shortest-path search on that graph (Dijkstra's
algorithm, from networkx for a single route and from scipy for every
station at once).

Stations aren't listed as points on the track map, so for each station
we take its latitude/longitude, project it onto the same coordinate
system as the track, and snap it to the nearest node in the graph (using
a KD-tree for a fast nearest-point lookup). That node stands in for the
station when we route.
"""

from __future__ import annotations

import itertools
import logging
from typing import TYPE_CHECKING

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from scipy.constants import kilo, mile
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import KDTree
from shapely.geometry import Point
from tqdm import tqdm

if TYPE_CHECKING:
    import pathlib

    from orta.config import DistanceConfig, TrackCalculatorConfig

logger = logging.getLogger(__name__)

_MIN_LINE_COORDS = 2
# How many nearby track nodes to consider when snapping a station,
# before giving up and taking the nearest one regardless of
# connectivity.
_SNAP_CANDIDATES = 12
# Failing those, the nearest main-network node is taken if it is this
# close. Spondon sits beside a 26-node fragment, its nearest main node
# 710 m away; the Isle of Wight's stations, a separate network, are
# kilometres from it.
_SNAP_MAX_METRES = 1000.0
_MIN_STATIONS = 2
# A routed distance outside [_PLAUSIBLE_LOW, _PLAUSIBLE_HIGH] x the
# straight line is a gap or a mis-snap in the track model, not a route;
# see TrackDistanceCalculator._plausible_metres. _CIRCUITY is the median
# ratio of route to straight line over the 9,900 pairs the hindcast
# cases touch.
_PLAUSIBLE_LOW = 0.9
_PLAUSIBLE_HIGH = 3.0
_PLAUSIBLE_MIN_METRES = 800.0
_CIRCUITY = 1.2


def convert_metres(distance_metres: float, unit: str = "km") -> float:
    """Convert a distance in metres to the requested unit.

    Parameters
    ----------
    distance_metres : float
        Distance in metres.
    unit : str, optional
        Target unit: ``'metres'``, ``'miles'``, or ``'km'`` (default).

    Returns
    -------
    float
        Converted distance.
    """
    if unit == "metres":
        return distance_metres
    if unit == "miles":
        return distance_metres / mile
    return distance_metres / kilo


class TrackDistanceCalculator:
    """Calculate shortest track distances between UK rail stations.

    Building one of these is the expensive bit (it reads the track
    shapefile and builds the graph), so create it once and reuse it for
    many lookups.
    """

    def __init__(self, config: TrackCalculatorConfig) -> None:
        """Build the routing graph and load station locations.

        Parameters
        ----------
        config : TrackCalculatorConfig
            Track shapefile, station dataset and coordinate-system
            settings.
        """
        self.config = config

        logger.info("Loading track shapefile from %s ...", config.track_shapefile)
        self.tracks = gpd.read_file(config.track_shapefile, engine="pyogrio")
        # Remember routes we've already worked out so we don't repeat
        # the search.
        self._cache: dict[tuple[str, str], tuple[float, list, list[tuple]]] = {}
        self._reported_implausible: set[tuple[str, str]] = set()

        self._build_graph()
        self._load_stations()

    def _build_graph(self) -> None:
        """Turn the track shapefile into a routable graph."""
        logger.info("Building routing graph...")
        self.graph = nx.Graph()

        for row in tqdm(
            self.tracks.itertuples(),
            total=len(self.tracks),
            desc="Building graph",
        ):
            geom = row.geometry
            if geom is None:
                continue

            # A row's geometry is either a single line or several lines
            # together.
            if geom.geom_type == "LineString":
                lines = [geom]
            elif geom.geom_type == "MultiLineString":
                lines = list(geom.geoms)
            else:
                continue

            for line in lines:
                coords = list(line.coords)
                if len(coords) < _MIN_LINE_COORDS:
                    continue

                # We only use each line's two end points as graph nodes,
                # joined by one edge whose weight is the line's real
                # length. Lines meet end-to-end on the network, so this
                # is enough to route across it.
                start_node = coords[0]
                end_node = coords[-1]
                weight = line.length

                # If two lines connect the same pair of points, keep the
                # shorter.
                if self.graph.has_edge(start_node, end_node):
                    if weight < self.graph[start_node][end_node]["weight"]:
                        self.graph[start_node][end_node]["weight"] = weight
                        self.graph[start_node][end_node]["geometry"] = line
                else:
                    self.graph.add_edge(
                        start_node,
                        end_node,
                        weight=weight,
                        geometry=line,
                    )

        logger.info(
            "Graph built with %d nodes and %d edges.",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )

        # A KD-tree over the node coordinates lets us find the node
        # nearest to any point (a station) quickly, instead of checking
        # every node.
        self._nodes = list(self.graph.nodes())
        self._nodes_array = np.array(self._nodes)
        self._kdtree = KDTree(self._nodes_array)
        # The same graph as a sparse matrix, for scipy's C Dijkstra in
        # distances_from (networkx's runs in Python, 15x slower).
        self._node_index = {node: i for i, node in enumerate(self._nodes)}
        edges = [
            (self._node_index[a], self._node_index[b], d["weight"])
            for a, b, d in self.graph.edges(data=True)
        ]
        u, v, w = np.array(edges).reshape(-1, 3).T
        n = len(self._nodes)
        self._matrix = csr_matrix((w, (u.astype(int), v.astype(int))), shape=(n, n))

    def _load_stations(self) -> None:
        """Load stations and snap each to its nearest graph node."""
        stations_cfg = self.config.stations
        crs_systems = self.config.coordinate_systems

        logger.info("Loading station coordinates from %s ...", stations_cfg.path)
        try:
            stations_df = pd.read_csv(
                stations_cfg.path,
                usecols=[stations_cfg.crs, stations_cfg.lat, stations_cfg.lon],
            )
            stations_df = stations_df.dropna()

            # Stations come as latitude/longitude; the track map is in
            # metres, so re-project the station points into the same
            # (metre-based) system.
            geometry = gpd.points_from_xy(
                stations_df[stations_cfg.lon], stations_df[stations_cfg.lat]
            )
            stations_gdf = gpd.GeoDataFrame(
                stations_df, geometry=geometry, crs=crs_systems.input_crs
            )
            stations_gdf = stations_gdf.to_crs(crs_systems.working_crs)

            self.station_locations: dict[str, tuple[float, float]] = {}
            for row in stations_gdf.itertuples():
                crs = str(getattr(row, stations_cfg.crs)).strip().upper()
                self.station_locations[crs] = (row.geometry.x, row.geometry.y)

            # For each station, find the closest track node; that's
            # where the station "sits" on the network for routing
            # purposes.
            #
            # Nearest alone is not enough. The track model contains
            # small disconnected fragments (sidings, stubs, digitising
            # artefacts), and a station sitting beside one snaps onto it
            # and is then unreachable from the whole network: Newsham
            # landed on a two-node fragment and lost every route, which
            # looks in the results like a station with no demand rather
            # than a station with no path. So prefer the nearest node
            # that is on the main network, and only fall back to the
            # unconditional nearest if none of the candidates is.
            self._station_nodes: dict[str, tuple] = {}
            coords = np.array(list(self.station_locations.values()))
            main_component = (
                max(nx.connected_components(self.graph), key=len) if self.graph else set()
            )
            neighbours = min(_SNAP_CANDIDATES, len(self._nodes))
            _, indices = self._kdtree.query(coords, k=neighbours)
            indices = np.atleast_2d(indices)
            main_nodes = [n for n in self._nodes if n in main_component]
            main_tree = KDTree(np.array(main_nodes)) if main_nodes else None
            stranded = []
            for code, xy, row in zip(self.station_locations, coords, indices, strict=True):
                nearest = self._nodes[row[0]]
                chosen = next(
                    (self._nodes[i] for i in row if self._nodes[i] in main_component),
                    nearest,
                )
                if chosen not in main_component and main_tree is not None:
                    metres, i = main_tree.query(xy)
                    if metres <= _SNAP_MAX_METRES:
                        chosen = main_nodes[int(i)]
                if chosen != nearest:
                    stranded.append(code)
                self._station_nodes[code] = chosen
            if stranded:
                logger.info(
                    "Snapped %d station(s) past a disconnected track fragment onto "
                    "the main network: %s.",
                    len(stranded),
                    ", ".join(sorted(stranded)),
                )

            logger.info("Loaded %d station locations.", len(self.station_locations))

        except (OSError, KeyError, ValueError) as e:
            # If the station file can't be read we carry on with no
            # stations; any later lookup then raises a clear "not found"
            # error.
            logger.warning("Failed to load station locations. %s", e)
            self.station_locations = {}
            self._station_nodes = {}

    def _route(self, crs_origin: str, crs_dest: str) -> tuple[float, list, list[tuple]]:
        """Find the shortest track route between two stations.

        Gives the metres, the route geometry and the edge keys.
        """
        key = (crs_origin, crs_dest)
        if key in self._cache:
            return self._cache[key]

        if crs_origin not in self._station_nodes:
            raise ValueError(f"Origin CRS '{crs_origin}' not found in station locations.")
        if crs_dest not in self._station_nodes:
            raise ValueError(f"Destination CRS '{crs_dest}' not found in station locations.")

        node_o = self._station_nodes[crs_origin]
        node_d = self._station_nodes[crs_dest]

        try:
            path_nodes = nx.shortest_path(
                self.graph, source=node_o, target=node_d, weight="weight"
            )
        except nx.NetworkXNoPath as exc:
            raise ValueError(
                f"No path found on the track network between {crs_origin} and {crs_dest}."
            ) from exc

        # Walk the path node-by-node, adding up edge lengths and
        # collecting the line geometry of each edge (so we can draw the
        # route later). Each edge's key is direction-independent (sorted
        # node pair) so it's a stable identity for aggregating a metric
        # onto physical track segments shared by many different routes.
        pairs = list(itertools.pairwise(path_nodes))
        edges = [self.graph[u][v] for u, v in pairs]
        edge_keys = [tuple(sorted((u, v))) for u, v in pairs]
        distance_metres = sum(e["weight"] for e in edges)
        route_lines = [e["geometry"] for e in edges]

        # Distance is the same in both directions, so cache the reverse
        # too.
        self._cache[key] = (distance_metres, route_lines, edge_keys)
        reverse_key = (crs_dest, crs_origin)
        if reverse_key not in self._cache:
            self._cache[reverse_key] = (distance_metres, route_lines, edge_keys)

        return distance_metres, route_lines, edge_keys

    def _plausible_metres(self, crs_origin: str, crs_dest: str, metres: float) -> float:
        """``metres``, or a straight-line estimate if it is impossible.

        The track model is missing links and snaps some stations onto
        the wrong stretch of line. Of the 9,900 pairs the hindcast cases
        touch, 488 route *shorter* than the straight line between the
        stations and 530 at over twice it; Headbolt Lane to Kirkby, a
        mile apart, routes at 39.5 miles because the 2023 link through
        Kirkby is not joined up. A gravity model takes those distances
        at face value. Where the route is below 0.9 or above 3 times the
        straight line (and the stations are more than
        :data:`_PLAUSIBLE_MIN_METRES` apart), this returns the straight
        line x :data:`_CIRCUITY`, the median ratio across those pairs.
        """
        a = self.station_locations.get(crs_origin)
        b = self.station_locations.get(crs_dest)
        if a is None or b is None:
            return metres
        crow = float(np.hypot(a[0] - b[0], a[1] - b[1]))
        if crow < _PLAUSIBLE_MIN_METRES:
            return metres
        if metres < _PLAUSIBLE_LOW * crow or metres > _PLAUSIBLE_HIGH * crow:
            key = (crs_origin, crs_dest)
            if key not in self._reported_implausible:
                self._reported_implausible.add(key)
                logger.debug(
                    "Implausible route %s-%s: %.1f km against %.1f km straight; using %.1f km.",
                    crs_origin,
                    crs_dest,
                    metres / 1000,
                    crow / 1000,
                    crow * _CIRCUITY / 1000,
                )
            return crow * _CIRCUITY
        return metres

    def calculate_distance(
        self,
        crs_origin: str,
        crs_dest: str,
        unit: str = "km",
        *,
        plausible: bool = False,
    ) -> float:
        """Calculate the track distance between two stations.

        Parameters
        ----------
        crs_origin : str
            Origin station CRS code.
        crs_dest : str
            Destination station CRS code.
        unit : str, optional
            Output unit: ``'km'`` (default), ``'miles'``, or
            ``'metres'``.
        plausible : bool, optional
            Replace an impossible routed distance with a straight-line
            estimate (see :meth:`_plausible_metres`). Off by default, so
            the distance command reports the route as the track model
            has it.

        Returns
        -------
        float
            Track distance in the requested unit.
        """
        crs_origin = str(crs_origin).strip().upper()
        crs_dest = str(crs_dest).strip().upper()
        distance_metres, _, _ = self._route(crs_origin, crs_dest)
        if plausible:
            distance_metres = self._plausible_metres(crs_origin, crs_dest, distance_metres)
        return convert_metres(distance_metres, unit)

    def distances_from(
        self,
        crs_origin: str,
        unit: str = "km",
        *,
        plausible: bool = False,
    ) -> dict[str, float]:
        """Track distances from one origin to every known station.

        Runs a single shortest-path tree (one Dijkstra) from the origin,
        which is far cheaper than calling :meth:`calculate_distance`
        pairwise when you need many destinations from the same origin.

        Parameters
        ----------
        crs_origin : str
            Origin station CRS code.
        unit : str, optional
            Output unit: ``'km'`` (default), ``'miles'``, or
            ``'metres'``.
        plausible : bool, optional
            Replace a routed distance the track model cannot have got
            right with a straight-line estimate (see
            :meth:`_plausible_metres`). Off by default.

        Returns
        -------
        dict[str, float]
            Mapping of destination CRS to track distance. Stations that
            are unreachable from the origin are omitted.

        Raises
        ------
        ValueError
            If ``crs_origin`` is not in the station locations.
        """
        crs_origin = str(crs_origin).strip().upper()
        if crs_origin not in self._station_nodes:
            raise ValueError(f"Origin CRS '{crs_origin}' not found in station locations.")

        source = self._node_index[self._station_nodes[crs_origin]]
        lengths = dijkstra(self._matrix, directed=False, indices=source)

        distances: dict[str, float] = {}
        for crs, node in self._station_nodes.items():
            metres = float(lengths[self._node_index[node]])
            if np.isfinite(metres):
                if plausible and crs != crs_origin:
                    metres = self._plausible_metres(crs_origin, crs, metres)
                distances[crs] = convert_metres(metres, unit)
        return distances

    def calculate_and_visualise(
        self,
        crs_origin: str,
        crs_dest: str,
        output_file: str | None = "route.html",
        unit: str = "metres",
    ) -> tuple[float, list]:
        """Calculate distance and optionally save an interactive map.

        Parameters
        ----------
        crs_origin : str
            Origin station CRS code.
        crs_dest : str
            Destination station CRS code.
        output_file : str | None, optional
            Path for the HTML map. ``None`` to skip.
        unit : str, optional
            Output unit for the returned distance.

        Returns
        -------
        tuple[float, list]
            Converted distance and route geometries.
        """
        crs_origin = str(crs_origin).strip().upper()
        crs_dest = str(crs_dest).strip().upper()

        distance_metres, route_lines, _ = self._route(crs_origin, crs_dest)

        if output_file:
            self.save_map(
                route_lines_by_leg=[route_lines],
                station_crs_codes=[crs_origin, crs_dest],
                output_file=output_file,
            )

        return convert_metres(distance_metres, unit), route_lines

    def save_map(
        self,
        route_lines_by_leg: list,
        station_crs_codes: list[str],
        output_file: str | pathlib.Path | None = None,
    ) -> None:
        """Save an interactive HTML map of the route.

        Parameters
        ----------
        route_lines_by_leg : list
            Route line geometries grouped by leg.
        station_crs_codes : list[str]
            Station CRS codes to mark on the map.
        output_file : str | pathlib.Path | None, optional
            Output HTML file path. Uses the config map default if
            ``None``.
        """
        map_cfg = self.config.map
        crs_systems = self.config.coordinate_systems

        if output_file is None:
            output_file = map_cfg.default_output
        if output_file is None:
            raise ValueError("No output file given and no map.default_output configured.")

        all_lines = [line for leg in route_lines_by_leg for line in leg]
        route_gdf = gpd.GeoDataFrame(geometry=all_lines, crs=crs_systems.working_crs)

        station_points = [Point(self.station_locations[crs]) for crs in station_crs_codes]
        stations_gdf = gpd.GeoDataFrame(
            {"Station": station_crs_codes},
            geometry=station_points,
            crs=crs_systems.working_crs,
        )

        route_gdf_wgs84 = route_gdf.to_crs(crs_systems.input_crs)
        stations_gdf_wgs84 = stations_gdf.to_crs(crs_systems.input_crs)

        explore_map = route_gdf_wgs84.explore(
            color=map_cfg.route_colour,
            weight=map_cfg.route_weight,
            name="Track Route",
        )
        explore_map = stations_gdf_wgs84.explore(
            m=explore_map,
            color=map_cfg.station_colour,
            marker_type=map_cfg.station_marker_type,
            name="Stations",
        )

        explore_map.save(str(output_file))
        logger.info("Interactive map saved to %s", output_file)


def export_route_geometry(
    calc: TrackDistanceCalculator,
    route_lines_by_leg: list,
    station_crs_codes: list[str],
    output_path: pathlib.Path,
) -> None:
    """Write the route lines and station points to GeoJSON.

    Written in the working (projected) CRS, so a plot with equal aspect
    keeps its shape. Failures are logged, not raised, so they never
    break a distance run.

    Parameters
    ----------
    calc : TrackDistanceCalculator
        Calculator holding station locations and CRS settings.
    route_lines_by_leg : list
        Route line geometries grouped by leg.
    station_crs_codes : list[str]
        Station CRS codes along the route.
    output_path : pathlib.Path
        Folder to write ``route_lines.geojson`` and
        ``route_stations.geojson``.
    """
    working_crs = calc.config.coordinate_systems.working_crs
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        lines = [line for leg in route_lines_by_leg for line in leg]
        if lines:
            gpd.GeoDataFrame(geometry=lines, crs=working_crs).to_file(
                output_path / "route_lines.geojson", driver="GeoJSON"
            )
        known = [c for c in station_crs_codes if c in calc.station_locations]
        if known:
            points = [Point(calc.station_locations[c]) for c in known]
            gpd.GeoDataFrame({"crs": known}, geometry=points, crs=working_crs).to_file(
                output_path / "route_stations.geojson", driver="GeoJSON"
            )
        logger.info("Route geometry written to %s", output_path)
    except (OSError, ValueError) as exc:
        logger.warning("Could not export route geometry: %s", exc)


def run_distance(config: DistanceConfig) -> float:
    """Route through a list of stations and report the distance.

    Parameters
    ----------
    config : DistanceConfig
        Calculator settings plus the station codes to route through.

    Returns
    -------
    float
        Total track distance through all stations, in ``config.units``.
    """
    calc = TrackDistanceCalculator(config.calculator)

    stations = [s.strip().upper() for s in config.station_codes]
    if len(stations) < _MIN_STATIONS:
        raise ValueError("At least two station CRS codes are required (origin and destination).")

    output_file = config.map_output

    # Route each consecutive pair (a "leg") and add the legs up for the
    # total.
    total_metres = 0.0
    all_leg_lines = []
    leg_records = []
    for i in range(len(stations) - 1):
        leg_metres, route_lines = calc.calculate_and_visualise(
            stations[i], stations[i + 1], output_file=None, unit="metres"
        )
        all_leg_lines.append(route_lines)
        total_metres += leg_metres
        leg_records.append(
            {
                "leg": i + 1,
                "origin": stations[i],
                "destination": stations[i + 1],
                "distance_km": convert_metres(leg_metres, "km"),
                "distance_miles": convert_metres(leg_metres, "miles"),
                "cumulative_km": convert_metres(total_metres, "km"),
                "cumulative_miles": convert_metres(total_metres, "miles"),
            }
        )
        if len(stations) > _MIN_STATIONS:
            logger.info(
                "Leg %d: %s -> %s: %.2f km (%.2f miles)",
                i + 1,
                stations[i],
                stations[i + 1],
                convert_metres(leg_metres, "km"),
                convert_metres(leg_metres, "miles"),
            )

    if output_file:
        calc.save_map(all_leg_lines, stations, output_file)

    # Persist the route geometry and a small per-leg distance table.
    export_route_geometry(calc, all_leg_lines, stations, config.output_path)
    config.output_path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(leg_records).to_csv(
        config.output_path / "route_legs.csv", index=False
    )

    route_str = " -> ".join(stations)
    logger.info(
        "Total track distance (%s): %.2f km (%.2f miles)",
        route_str,
        convert_metres(total_metres, "km"),
        convert_metres(total_metres, "miles"),
    )
    return convert_metres(total_metres, config.units)
