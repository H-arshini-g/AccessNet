# scripts/generate_synthetic_tickets.py
"""
Generate synthetic ticket/passenger events aligned with GTFS stops.

Usage:
  python scripts/generate_synthetic_tickets.py --gtfs data/gtfs --out data/tickets/synth_tickets.csv --rows 50000

Output format:
  timestamp,route_id,stop_id
"""

import argparse
import pandas as pd
import random
import os
from datetime import datetime, timedelta

def generate_synthetic_tickets(gtfs_dir, out_file, rows=50000):
    # Load stops and routes
    stops = pd.read_csv(os.path.join(gtfs_dir, "stops.txt"), dtype=str)
    routes = pd.read_csv(os.path.join(gtfs_dir, "routes.txt"), dtype=str)
    trips = pd.read_csv(os.path.join(gtfs_dir, "trips.txt"), dtype=str)

    stop_ids = stops["stop_id"].astype(str).tolist()
    route_ids = routes["route_id"].astype(str).tolist()

    if not stop_ids or not route_ids:
        raise ValueError("Stops or routes are missing or empty in GTFS directory.")

    # Pick some random routes and link to stops using trips if available
    route_to_stops = {}
    stop_times_path = os.path.join(gtfs_dir, "stop_times.txt")
    if os.path.exists(stop_times_path):
        stop_times = pd.read_csv(stop_times_path, dtype=str)
        stop_times["stop_id"] = stop_times["stop_id"].astype(str)
        stop_times["trip_id"] = stop_times["trip_id"].astype(str)
        for trip_id, group in stop_times.groupby("trip_id"):
            if trip_id in trips["trip_id"].values:
                r_id = trips.loc[trips["trip_id"] == trip_id, "route_id"].values[0]
                if r_id not in route_to_stops:
                    route_to_stops[r_id] = []
                route_to_stops[r_id].extend(group["stop_id"].tolist())

    # If route_to_stops is empty, fallback to all stops
    if not route_to_stops:
        for r in route_ids:
            route_to_stops[r] = random.sample(stop_ids, min(20, len(stop_ids)))

    # Synthetic generation logic
    start_time = datetime(2025, 1, 1, 6, 0, 0)
    records = []
    for i in range(rows):
        route_id = random.choice(route_ids)
        possible_stops = route_to_stops.get(route_id, stop_ids)
        stop_id = random.choice(possible_stops)
        t = start_time + timedelta(seconds=random.randint(0, 3600 * 12))  # within 6 AM–6 PM window
        records.append([t.strftime("%Y-%m-%d %H:%M:%S"), route_id, stop_id])

    df = pd.DataFrame(records, columns=["timestamp", "route_id", "stop_id"])
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    df.to_csv(out_file, index=False)
    print(f"✅ Synthetic tickets generated: {len(df)} rows → {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gtfs", default="data/gtfs", help="Path to GTFS directory")
    parser.add_argument("--out", default="data/tickets/synth_tickets.csv", help="Output CSV path")
    parser.add_argument("--rows", type=int, default=50000, help="Number of ticket events to generate")
    args = parser.parse_args()
    generate_synthetic_tickets(args.gtfs, args.out, args.rows)
