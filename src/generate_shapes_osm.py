import argparse
import os
import pandas as pd
import osmnx as ox
import networkx as nx

def generate_shapes_osm(gtfs_folder, out_file="shapes.txt", network_type="walk", dist=1000):
    # Load GTFS stops and stop_times
    stops = pd.read_csv(os.path.join(gtfs_folder, "stops.txt"))
    stop_times = pd.read_csv(os.path.join(gtfs_folder, "stop_times.txt"))

    # Build small OSM graph around the mean of all stops
    center_lat = stops["stop_lat"].mean()
    center_lon = stops["stop_lon"].mean()
    print(f"Downloading OSM network around {center_lat:.4f}, {center_lon:.4f}…")

    G = ox.graph_from_point((center_lat, center_lon), dist=dist, network_type=network_type)

    shape_rows = []
    shape_id_counter = 0

    for trip_id, group in stop_times.groupby("trip_id"):
        group = group.sort_values("stop_sequence")
        stop_list = list(group["stop_id"].values)

        coords_list = []
        for i in range(len(stop_list) - 1):
            s1 = stops.loc[stops["stop_id"] == stop_list[i]].iloc[0]
            s2 = stops.loc[stops["stop_id"] == stop_list[i + 1]].iloc[0]

            # Find nearest OSM nodes
            u = ox.distance.nearest_nodes(G, s1["stop_lon"], s1["stop_lat"])
            v = ox.distance.nearest_nodes(G, s2["stop_lon"], s2["stop_lat"])

            try:
                path = nx.shortest_path(G, u, v, weight="length")
                coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in path]
                coords_list.extend(coords)
            except nx.NetworkXNoPath:
                print(f"No OSM path between {s1['stop_id']} and {s2['stop_id']}")
                coords_list.append((s1["stop_lat"], s1["stop_lon"]))
                coords_list.append((s2["stop_lat"], s2["stop_lon"]))

        # Save shape rows
        for seq, (lat, lon) in enumerate(coords_list, start=1):
            shape_rows.append({
                "shape_id": f"shape_{shape_id_counter}",
                "shape_pt_lat": lat,
                "shape_pt_lon": lon,
                "shape_pt_sequence": seq
            })
        shape_id_counter += 1

    # Save shapes.txt
    shapes = pd.DataFrame(shape_rows)
    shapes.to_csv(os.path.join(gtfs_folder, out_file), index=False)
    print(f"✅ Road-following shapes.txt saved to {os.path.join(gtfs_folder, out_file)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gtfs", required=True, help="Path to GTFS folder")
    parser.add_argument("--out", default="shapes.txt", help="Output shapes.txt name")
    parser.add_argument("--network", default="walk", choices=["walk", "drive", "bike"])
    parser.add_argument("--dist", type=int, default=1000, help="Search radius (meters)")
    args = parser.parse_args()

    generate_shapes_osm(args.gtfs, args.out, args.network, args.dist)
