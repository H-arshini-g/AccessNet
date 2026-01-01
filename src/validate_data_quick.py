# src/validate_data_quick.py
import argparse
from pathlib import Path
import pandas as pd
import sys

def load_csv(path):
    try:
        return pd.read_csv(path, dtype=str)
    except Exception as e:
        print("ERROR reading", path, ":", e)
        return None

def check_required(df, required, name):
    missing = [c for c in required if c not in (df.columns if df is not None else [])]
    return missing

def main(args):
    gtfs_dir = Path(args.gtfs)
    tickets_file = Path(args.tickets)
    outdir = Path("data/outputs")
    outdir.mkdir(parents=True, exist_ok=True)

    print("Validating GTFS directory:", gtfs_dir)
    stops = load_csv(gtfs_dir / "stops.txt")
    if stops is None:
        print(" stops.txt missing or unreadable.")
    else:
        req = ["stop_id","stop_lat","stop_lon"]
        missing = check_required(stops, req, "stops")
        print(" stops.txt rows:", len(stops), "missing columns:", missing)
        if "stop_lat" in stops.columns:
            bad_coords = stops[pd.to_numeric(stops['stop_lat'], errors='coerce').isna() |
                                pd.to_numeric(stops['stop_lon'], errors='coerce').isna()]
            print("  bad coord rows:", len(bad_coords))
            if len(bad_coords)>0:
                bad_coords.head(20).to_csv(outdir/"validate_bad_coords.csv", index=False)
                print("   sample bad coords ->", outdir/"validate_bad_coords.csv")
    # shapes
    shapes = load_csv(gtfs_dir / "shapes.txt")
    if shapes is not None:
        req = ["shape_id","shape_pt_lat","shape_pt_lon","shape_pt_sequence"]
        missing = check_required(shapes, req, "shapes")
        print(" shapes.txt rows:", len(shapes), "missing columns:", missing)

    # stop_times
    st = load_csv(gtfs_dir / "stop_times.txt")
    if st is not None:
        req = ["trip_id","arrival_time","departure_time","stop_id","stop_sequence"]
        missing = check_required(st, req, "stop_times")
        print(" stop_times.txt rows:", len(st), "missing columns:", missing)
        # detect if times seem to use 24+ hour format (GTFS allows beyond 24:00)
        if "arrival_time" in st.columns:
            bad_times = st[pd.to_datetime(st['arrival_time'], errors='coerce').isna()]
            print("  bad arrival_time parse count:", len(bad_times))
            if len(bad_times)>0:
                bad_times.head(20).to_csv(outdir/"validate_bad_times.csv", index=False)
                print("   sample bad times ->", outdir/"validate_bad_times.csv")

    # trips
    trips = load_csv(gtfs_dir / "trips.txt")
    if trips is not None:
        req = ["trip_id","route_id","service_id"]
        missing = check_required(trips, req, "trips")
        print(" trips.txt rows:", len(trips), "missing columns:", missing)

    # calendar
    cal = load_csv(gtfs_dir / "calendar.txt")
    if cal is not None:
        req = ["service_id","start_date","end_date"]
        missing = check_required(cal, req, "calendar")
        print(" calendar.txt rows:", len(cal), "missing columns:", missing)

    # check tickets file presence
    print("\nChecking tickets file:", tickets_file)
    if tickets_file.exists():
        tickets = load_csv(tickets_file)
        print(" tickets rows:", len(tickets))
        req = ["timestamp","route_id","stop_id"]
        missing = check_required(tickets, req, "tickets")
        print(" tickets missing columns:", missing)
        if "timestamp" in tickets.columns:
            bad_ts = tickets[pd.to_datetime(tickets['timestamp'], errors='coerce').isna()]
            print("  bad timestamps:", len(bad_ts))
            if len(bad_ts)>0:
                bad_ts.head(20).to_csv(outdir/"validate_bad_timestamps.csv", index=False)
                print("   sample bad timestamps ->", outdir/"validate_bad_timestamps.csv")
        # check stop membership
        if stops is not None and "stop_id" in tickets.columns:
            sset = set(stops['stop_id'].astype(str))
            tickets['stop_id_clean'] = tickets['stop_id'].astype(str).str.strip()
            missing_stops = tickets[~tickets['stop_id_clean'].isin(sset)]
            print("  ticket rows referencing unknown stops:", len(missing_stops))
            if len(missing_stops)>0:
                missing_stops.head(50).to_csv(outdir/"validate_missing_stop_refs.csv", index=False)
                print("   sample missing refs ->", outdir/"validate_missing_stop_refs.csv")
    else:
        print(" tickets file NOT found at", tickets_file)
        print("  NOTE: GTFS fare files (fare_rules.txt / fare_attributes.txt) are NOT passenger events. They describe fares only.")

    print("\nValidation complete. Check data/outputs for sample problem files if any.")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--gtfs", default="data/gtfs")
    p.add_argument("--tickets", default="data/tickets/tickets.csv")
    args = p.parse_args()
    main(args)
