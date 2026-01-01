# scripts/generate_synthetic_gtfs.py
import os, csv, math
import pandas as pd

# center near previous area
center_lat = 12.9747
center_lon = 77.5964

# create N stops distributed along a rough corridor with small random offsets
N = 12
radius_km = 2.0  # spread ~2 km
stops = []
for i in range(N):
    # spread along bearing ~80 degrees (east) with slight north drift
    frac = i / (N - 1)
    lat = center_lat + (frac - 0.5) * 0.017  # ~1.9 km lat range
    lon = center_lon + (frac - 0.5) * 0.025  # ~2.8 km lon range
    stops.append({
        "stop_id": f"S{i+1}",
        "stop_name": f"Stop {i+1}",
        "stop_lat": round(lat, 6),
        "stop_lon": round(lon, 6)
    })

# trips and stop_times: create two trips (morning & evening) that stop at all stops
trips = []
stop_times = []
for t_idx, trip_id in enumerate(["T1","T2","T3","T4"]):
    trips.append({"route_id":"R1","service_id":"WD","trip_id":trip_id,"trip_headsign":"Synthetic"})
    # schedule times: start at 07:00 + offset per trip
    start_hour = 7 + t_idx*2
    mins = 0
    for seq, s in enumerate(stops, start=1):
        arrival = f"{start_hour:02d}:{mins:02d}:00"
        departure = f"{start_hour:02d}:{(mins+2)%60:02d}:00"
        stop_times.append({
            "trip_id":trip_id,
            "arrival_time":arrival,
            "departure_time":departure,
            "stop_id":s["stop_id"],
            "stop_sequence":seq
        })
        mins += 5  # 5 minutes between stops

# write files to data/gtfs (create folder)
out = "data/gtfs"
os.makedirs(out, exist_ok=True)

stops_df = pd.DataFrame(stops)
stops_df.to_csv(os.path.join(out,"stops.txt"), index=False)

trips_df = pd.DataFrame(trips)
trips_df.to_csv(os.path.join(out,"trips.txt"), index=False)

stop_times_df = pd.DataFrame(stop_times)
stop_times_df.to_csv(os.path.join(out,"stop_times.txt"), index=False)

# create a minimal routes.txt
routes_df = pd.DataFrame([{"route_id":"R1","route_short_name":"R1","route_long_name":"Synthetic Route","route_type":3}])
routes_df.to_csv(os.path.join(out,"routes.txt"), index=False)

print("Synthetic GTFS written to data/gtfs with:")
print(f" - stops: {len(stops_df)}")
print(f" - trips: {len(trips_df)}")
print(f" - stop_times: {len(stop_times_df)}")
