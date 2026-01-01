# src/generate_shapes_osm_cached.py
import argparse, os, pickle, time
import pandas as pd
import osmnx as ox
import networkx as nx

def generate_shapes_osm(gtfs_folder, out_file="shapes.txt", network_type="walk", dist=3000, cache="data/outputs/osm_graph.pkl"):
    stops = pd.read_csv(os.path.join(gtfs_folder,"stops.txt"))
    stop_times = pd.read_csv(os.path.join(gtfs_folder,"stop_times.txt"))

    # center and caching
    center_lat, center_lon = stops["stop_lat"].mean(), stops["stop_lon"].mean()
    if os.path.exists(cache):
        print("Loading cached OSM graph:", cache)
        with open(cache,"rb") as f:
            G = pickle.load(f)
    else:
        print(f"Downloading OSM graph around {center_lat:.4f},{center_lon:.4f} dist={dist}m ...")
        G = ox.graph_from_point((center_lat, center_lon), dist=dist, network_type=network_type)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache,"wb") as f:
            pickle.dump(G,f)
        print("Saved OSM graph to cache.")

    rows = []
    shape_id_counter = 0

    for trip_id, group in stop_times.groupby("trip_id"):
        group = group.sort_values("stop_sequence")
        stop_list = list(group["stop_id"].values)
        coords_for_trip = []
        for i in range(len(stop_list)-1):
            s1 = stops[stops["stop_id"]==stop_list[i]].iloc[0]
            s2 = stops[stops["stop_id"]==stop_list[i+1]].iloc[0]
            # nearest nodes (ox expects lon,lat)
            node1 = ox.distance.nearest_nodes(G, s1["stop_lon"], s1["stop_lat"])
            node2 = ox.distance.nearest_nodes(G, s2["stop_lon"], s2["stop_lat"])
            try:
                path = nx.shortest_path(G, node1, node2, weight="length")
                seg_coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in path]
            except Exception as e:
                print("No path, fallback to straight:", s1["stop_id"], s2["stop_id"])
                seg_coords = [(s1["stop_lat"], s1["stop_lon"]), (s2["stop_lat"], s2["stop_lon"])]
            # append, but avoid duplicating the connecting node
            if coords_for_trip and coords_for_trip[-1] == seg_coords[0]:
                coords_for_trip.extend(seg_coords[1:])
            else:
                coords_for_trip.extend(seg_coords)
        # write coords_for_trip to shape rows
        for seq, (lat,lon) in enumerate(coords_for_trip, start=1):
            rows.append({
                "shape_id": f"shape_{shape_id_counter}",
                "shape_pt_lat": lat,
                "shape_pt_lon": lon,
                "shape_pt_sequence": seq
            })
        shape_id_counter += 1

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(gtfs_folder, out_file), index=False)
    print("Shapes generated:", len(df), "rows ->", os.path.join(gtfs_folder, out_file))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gtfs", required=True)
    parser.add_argument("--out", default="shapes.txt")
    parser.add_argument("--dist", type=int, default=3000)
    args = parser.parse_args()
    generate_shapes_osm(args.gtfs, args.out, dist=args.dist)
