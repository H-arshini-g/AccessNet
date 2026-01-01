import argparse
import pandas as pd
import folium
import matplotlib.cm as cm
import matplotlib.colors as colors


def plot_shapes(stops_file, shapes_file, preds_file, out_html):
    # Load stops, shapes, predictions
    stops = pd.read_csv(stops_file)
    shapes = pd.read_csv(shapes_file)
    preds = pd.read_csv(preds_file)

    # Demand per stop
    if "time_step" in preds.columns:
        last_row = preds.iloc[-1]
        stop_cols = [c for c in preds.columns if c.startswith("stop_")]
        demand = {stop_id: last_row[c] for stop_id, c in zip(stops["stop_id"], stop_cols)}
    else:
        demand = preds.groupby("stop_id")["ema_pred"].mean().to_dict()

    center = [stops["stop_lat"].mean(), stops["stop_lon"].mean()]
    m = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap")

    # Normalize for colors
    max_val = max(demand.values()) if demand else 1.0
    norm = colors.Normalize(vmin=0, vmax=max_val)
    cmap = cm.get_cmap("viridis")

    # Plot stops
    for _, row in stops.iterrows():
        d = demand.get(row["stop_id"], 0)
        color = colors.to_hex(cmap(norm(d)))
        folium.CircleMarker(
            location=[row["stop_lat"], row["stop_lon"]],
            radius=6,
            color=color,
            fill=True,
            fill_opacity=0.9,
            popup=f"Stop {row['stop_id']}<br>Predicted demand: {d:.2f}"
        ).add_to(m)

    # Plot shapes with segment-based demand coloring
    for shape_id, group in shapes.groupby("shape_id"):
        group = group.sort_values("shape_pt_sequence").reset_index(drop=True)

        # Go through consecutive points in the shape
        for i in range(len(group) - 1):
            lat1, lon1 = group.loc[i, ["shape_pt_lat", "shape_pt_lon"]]
            lat2, lon2 = group.loc[i + 1, ["shape_pt_lat", "shape_pt_lon"]]

            # Approximate demand: nearest stop to this segment
            nearest_stop = stops.iloc[
                ((stops["stop_lat"] - lat1).abs() + (stops["stop_lon"] - lon1).abs()).argmin()
            ]
            seg_demand = demand.get(nearest_stop["stop_id"], 0)

            color = colors.to_hex(cmap(norm(seg_demand)))

            folium.PolyLine(
                [(lat1, lon1), (lat2, lon2)],
                color=color,
                weight=6,
                opacity=0.9,
                popup=f"Segment near {nearest_stop['stop_id']}<br>Demand: {seg_demand:.2f}"
            ).add_to(m)

    # Add legend
    legend_html = """
    <div style="
        position: fixed;
        bottom: 50px;
        left: 50px;
        width: 240px;
        height: 130px;
        background-color: white;
        border:2px solid grey;
        z-index:9999;
        font-size:14px;
        padding: 10px;
    ">
    <b>Accessibility Legend</b><br>
    <span style="color:blue;">&#9632;</span> Low accessibility (low demand)<br>
    <span style="color:green;">&#9632;</span> Medium accessibility<br>
    <span style="color:yellow;">&#9632;</span> High accessibility (high demand)<br>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Save map
    m.save(out_html)
    print(f"✅ Color-coded map with legend saved to {out_html}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stops", default="data/gtfs/stops.txt")
    parser.add_argument("--shapes", default="data/gtfs/shapes.txt")
    parser.add_argument("--preds", required=True)
    parser.add_argument("--out", default="map.html")
    args = parser.parse_args()

    plot_shapes(args.stops, args.shapes, args.preds, args.out)
