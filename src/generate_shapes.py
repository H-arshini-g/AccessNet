import argparse
import os
import pandas as pd
import osmnx as ox
import networkx as nx
from shapely.geometry import Point
import geopandas as gpd


def generate_shapes(gtfs_folder, out_file="shapes.txt", network_type="walk"):
    stops = pd.read_csv(os.path.join(gtfs_folder, "stops.txt"))
    stop_times = pd.read_csv(os.path.join(gtfs_folder, "stop_times.txt"))

    # Build polygon around stops (convex hull + buffer ~500m)
    gdf = gpd.GeoDataFrame(
        stops,
        geometry=gpd.points_from_xy(stops["stop_lon"], stops["stop_lat"]),
        crs="EPSG:4326"
    )
    area = gdf.unary_union.convex_hull.buffer(0.005)  # buffer in degrees (~500m)

    print("Downloading OSM network only around stops…")
    G = ox.graph_from_polygon(area, network_type=network_type)
    G = ox.project_graph(G)

    shape_rows = []
    shape_id_counter = 0

    for trip_id, group in stop_times.groupby("trip_id"):
        group = group.sort_values("stop_sequence")
        stop_list = list(group["stop_id"].values)

        for i in range(len(stop_list) - 1):
            s1 = stops.loc[stops["stop_id"] == stop_list[i]].iloc[0]
            s2 = stops.loc[stops["stop_id"] == stop_list[i + 1]].iloc[0]

            # Find nearest graph nodes
            u = ox.distance.nearest_nodes(G, s1["stop_lon"], s1["stop_lat"])
            v = ox.distance.nearest_nodes(G, s2["stop_lon"], s2["stop_lat"])

            try:
                path = nx.shortest_path(G, u, v, weight="length")
                coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in path]

                for seq, (lat, lon) in enumerate(coords, start=1):
                    shape_rows.append({
                        "shape_id": f"shape_{shape_id_counter}",
                        "shape_pt_lat": lat,
                        "shape_pt_lon": lon,
                        "shape_pt_sequence": seq
                    })
                shape_id_counter += 1
            except nx.NetworkXNoPath:
                print(f"No path between {s1['stop_id']} and {s2['stop_id']}")

    shapes = pd.DataFrame(shape_rows)
    shapes.to_csv(os.path.join(gtfs_folder, out_file), index=False)
    print(f"Synthetic shapes.txt saved to {os.path.join(gtfs_folder, out_file)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gtfs", required=True, help="Path to GTFS folder")
    parser.add_argument("--out", default="shapes.txt", help="Output shapes.txt name")
    parser.add_argument("--network", default="walk", choices=["walk", "drive", "bike"],
                        help="OSM network type for routing")
    args = parser.parse_args()

    generate_shapes(args.gtfs, args.out, args.network)
