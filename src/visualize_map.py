import argparse
import pickle
import os
import pandas as pd
import folium
import matplotlib.cm as cm
import matplotlib.colors as colors


def visualize_map(graph_file, preds_file, out_html, gtfs_dir="data/gtfs"):
    # Load graph
    with open(graph_file, "rb") as f:
        G = pickle.load(f)

    # Load predictions
    preds = pd.read_csv(preds_file)

    if "time_step" in preds.columns:
        # LSTM/GRU predictions → take last timestep
        last_row = preds.iloc[-1]
        stop_cols = [c for c in preds.columns if c.startswith("stop_")]
        demand = {stop_id: last_row[c] for stop_id, c in zip(G.nodes(), stop_cols)}
    else:
        # EMA predictions → average over time
        demand = preds.groupby("stop_id")["ema_pred"].mean().to_dict()

    # Map center = average stop location
    lats = [G.nodes[n]["lat"] for n in G.nodes()]
    lons = [G.nodes[n]["lon"] for n in G.nodes()]
    center = [sum(lats) / len(lats), sum(lons) / len(lons)]

    m = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap")

    # Normalize demand for coloring
    max_val = max(demand.values()) if demand else 1.0
    norm = colors.Normalize(vmin=0, vmax=max_val)
    cmap = cm.get_cmap("viridis")

    # Plot stops
    for node, data in G.nodes(data=True):
        d = demand.get(node, 0)
        color = colors.to_hex(cmap(norm(d)))
        folium.CircleMarker(
            location=[data["lat"], data["lon"]],
            radius=6,
            color=color,
            fill=True,
            fill_opacity=0.9,
            popup=f"Stop {node}<br>Predicted demand: {d:.2f}"
        ).add_to(m)

    # Try to use GTFS shapes.txt
    shapes_path = os.path.join(gtfs_dir, "shapes.txt")
    if os.path.exists(shapes_path):
        print("Drawing routes from shapes.txt")
        shapes = pd.read_csv(shapes_path)
        for shape_id, group in shapes.groupby("shape_id"):
            coords = list(zip(group["shape_pt_lat"], group["shape_pt_lon"]))
            folium.PolyLine(
                coords,
                color="blue",
                weight=3,
                opacity=0.6,
                popup=f"Route shape {shape_id}"
            ).add_to(m)
    else:
        print("No shapes.txt found → drawing straight edges")
        for u, v in G.edges():
            lat1, lon1 = G.nodes[u]["lat"], G.nodes[u]["lon"]
            lat2, lon2 = G.nodes[v]["lat"], G.nodes[v]["lon"]
            folium.PolyLine(
                [(lat1, lon1), (lat2, lon2)],
                color="blue",
                weight=3,
                opacity=0.6
            ).add_to(m)

    # Save map
    m.save(out_html)
    print(f"Interactive map saved to {out_html}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True)
    parser.add_argument("--preds", required=True)
    parser.add_argument("--out", default="map.html")
    parser.add_argument("--gtfs", default="data/gtfs")
    args = parser.parse_args()

    visualize_map(args.graph, args.preds, args.out, args.gtfs)
