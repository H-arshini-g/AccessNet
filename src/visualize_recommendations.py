# src/visualize_recommendations.py
import argparse
import pandas as pd
import folium
import os

def visualize_recommendations(stops_file="data/gtfs/stops.txt",
                              recs_file="data/outputs/recommendations.csv",
                              out_html="data/outputs/map_recommendations.html",
                              show_only_low=False):
    stops = pd.read_csv(stops_file)
    recs = pd.read_csv(recs_file)

    # Merge to ensure lat/lon present
    df = pd.merge(recs, stops, left_on="Stop ID", right_on="stop_id", how="left")

    # Map center
    center = [df['stop_lat'].astype(float).mean(), df['stop_lon'].astype(float).mean()]
    m = folium.Map(location=center, zoom_start=13, tiles="OpenStreetMap")

    # Color mapping
    color_map = {
        "Low": "#e74c3c",
        "Medium": "#f39c12",
        "High": "#2ecc71",
        "Unknown": "#95a5a6"
    }

    # Add legend
    legend_html = """
     <div style="
         position: fixed;
         bottom: 50px;
         left: 50px;
         width: 260px;
         background-color: white;
         border:2px solid grey;
         z-index:9999;
         font-size:14px;
         padding: 10px;
     ">
       <b>Accessibility Legend</b><br>
       <span style="color:#2ecc71;">&#9632;</span> High accessibility<br>
       <span style="color:#f39c12;">&#9632;</span> Medium accessibility<br>
       <span style="color:#e74c3c;">&#9632;</span> Low accessibility<br>
       <span style="color:#95a5a6;">&#9632;</span> Unknown / No data<br>
       <hr style="margin:8px 0;">
       <a href="/data/outputs/recommendations.csv" target="_blank">Download recommendations.csv</a>
     </div>
     """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Plot stops
    for _, row in df.iterrows():
        level = row.get("Level", "Unknown")
        color = color_map.get(level, "#95a5a6")
        lat = float(row['stop_lat'])
        lon = float(row['stop_lon'])
        stop_id = row["Stop ID"]
        stop_name = row.get("Stop Name", stop_id)
        score = row.get("Accessibility Score", "N/A")
        rec_text = row.get("Recommendation", "")

        # Optionally show only low accessibility stops
        if show_only_low and level != "Low":
            continue

        # Emphasize low-access stops with larger ring
        if level == "Low":
            folium.CircleMarker(
                location=[lat, lon],
                radius=14,
                color=color,
                weight=2,
                fill=True,
                fill_opacity=0.18
            ).add_to(m)

        # Core marker
        folium.CircleMarker(
            location=[lat, lon],
            radius=7,
            color=color,
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.95,
            popup=folium.Popup(f"<b>{stop_id} — {stop_name}</b><br>"
                               f"Score: {score}<br>"
                               f"Level: {level}<br>"
                               f"Suggestion: {rec_text}", max_width=300)
        ).add_to(m)

    # Save
    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    m.save(out_html)
    print(f"Map with recommendations saved to {out_html}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stops", default="data/gtfs/stops.txt")
    parser.add_argument("--recs", default="data/outputs/recommendations.csv")
    parser.add_argument("--out", default="data/outputs/map_recommendations.html")
    parser.add_argument("--only-low", action="store_true", help="Show only low accessibility stops")
    args = parser.parse_args()
    visualize_recommendations(args.stops, args.recs, args.out, args.only_low)
