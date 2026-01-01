import argparse
import os
import networkx as nx
import pickle
import pandas as pd


def build_graph(gtfs_folder, out_path):
    stops = pd.read_csv(os.path.join(gtfs_folder, "stops.txt"))

    G = nx.DiGraph()

    # Add stops as nodes
    for _, row in stops.iterrows():
        G.add_node(row["stop_id"], lat=row["stop_lat"], lon=row["stop_lon"], name=row.get("stop_name", ""))

    shapes_file = os.path.join(gtfs_folder, "shapes.txt")

    if os.path.exists(shapes_file):
        print("Using shapes.txt for route geometry...")
        shapes = pd.read_csv(shapes_file)

        for shape_id, group in shapes.groupby("shape_id"):
            group = group.sort_values("shape_pt_sequence")
            coords = list(zip(group["shape_pt_lat"], group["shape_pt_lon"]))

            for i in range(len(coords) - 1):
                u = f"{shape_id}_{i}"
                v = f"{shape_id}_{i+1}"
                lat1, lon1 = coords[i]
                lat2, lon2 = coords[i + 1]

                # add synthetic intermediate nodes for geometry
                G.add_node(u, lat=lat1, lon=lon1)
                G.add_node(v, lat=lat2, lon=lon2)
                G.add_edge(u, v, shape_id=shape_id)
    else:
        print("No shapes.txt found, using straight lines...")
        stop_times = pd.read_csv(os.path.join(gtfs_folder, "stop_times.txt"))

        for trip_id, group in stop_times.groupby("trip_id"):
            group = group.sort_values("stop_sequence")
            stop_list = list(group["stop_id"].values)

            for i in range(len(stop_list) - 1):
                u, v = stop_list[i], stop_list[i + 1]
                G.add_edge(u, v, trip_id=trip_id)

    with open(out_path, "wb") as f:
        pickle.dump(G, f)
    print(f"Graph saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gtfs", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    build_graph(args.gtfs, args.out)
