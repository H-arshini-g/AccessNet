# src/normalize_gtfs.py
"""
Normalize a real GTFS directory into the canonical format expected by the project.

Usage:
    python src/normalize_gtfs.py --in data/gtfs_real --out data/gtfs

It will copy/transform:
 - stops.txt -> ensure columns: stop_id, stop_name, stop_lat, stop_lon
 - routes.txt -> ensure: route_id, route_short_name, route_long_name, route_type
 - trips.txt  -> ensure: route_id, service_id, trip_id, trip_headsign, shape_id (if present)
 - stop_times.txt -> ensure: trip_id, arrival_time, departure_time, stop_id, stop_sequence
 - shapes.txt -> ensure: shape_id, shape_pt_lat, shape_pt_lon, shape_pt_sequence
 - calendar.txt -> reorder to service_id,monday,...,end_date
The script is conservative: it won't delete original files.
"""
import argparse
from pathlib import Path
import pandas as pd
import os

def read_csv_if_exists(path):
    if path.exists():
        return pd.read_csv(path, dtype=str)
    return None

def write_csv(df, path):
    os.makedirs(path.parent, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  WROTE {path} rows={len(df)}")

def normalize_stops(indir, outdir):
    p = indir / "stops.txt"
    df = read_csv_if_exists(p)
    if df is None:
        print("  skips stops.txt (not found)")
        return
    # canonical columns
    # attempt to detect columns (your real data had stop_name,zone_id,stop_id,stop_lat,stop_lon)
    # map or rename if needed
    colmap = {}
    if "stop_id" not in df.columns and "stop_id" not in df.columns:
        # try to find a column containing 'stop_id' like numeric id col
        for c in df.columns:
            if c.lower().endswith("stop_id") or c.lower() == "id":
                colmap[c] = "stop_id"
    # ensure stop_name exists
    if "stop_name" not in df.columns:
        for c in df.columns:
            if "name" in c.lower():
                colmap[c] = "stop_name"
                break
    # lat/lon detection
    if "stop_lat" not in df.columns:
        for c in df.columns:
            if c.lower().endswith("lat"):
                colmap[c] = "stop_lat"
                break
    if "stop_lon" not in df.columns:
        for c in df.columns:
            if c.lower().endswith("lon") or c.lower().endswith("lng"):
                colmap[c] = "stop_lon"
                break
    df = df.rename(columns=colmap)
    # Keep only canonical columns (others ignored)
    canonical = ["stop_id","stop_name","stop_lat","stop_lon"]
    for c in canonical:
        if c not in df.columns:
            df[c] = ""
    # cleaning
    df['stop_id'] = df['stop_id'].astype(str).str.strip()
    df['stop_name'] = df['stop_name'].astype(str).str.strip()
    df['stop_lat'] = pd.to_numeric(df['stop_lat'], errors='coerce')
    df['stop_lon'] = pd.to_numeric(df['stop_lon'], errors='coerce')
    df = df.dropna(subset=['stop_lat','stop_lon'])
    out = outdir / "stops.txt"
    write_csv(df[canonical], out)

def normalize_routes(indir, outdir):
    p = indir / "routes.txt"
    df = read_csv_if_exists(p)
    if df is None:
        print("  skips routes.txt (not found)")
        return
    # Map columns: real: route_long_name,route_short_name,agency_id,route_type,route_id
    colmap = {}
    if "route_id" not in df.columns:
        for c in df.columns:
            if c.lower().endswith("route_id") or c.lower() == "id":
                colmap[c] = "route_id"
    if "route_short_name" not in df.columns:
        for c in df.columns:
            if "short" in c.lower():
                colmap[c] = "route_short_name"
    if "route_long_name" not in df.columns:
        for c in df.columns:
            if "long" in c.lower() or "name" in c.lower():
                # be conservative; prefer 'route_long_name' column if present
                colmap[c] = "route_long_name"
                break
    if "route_type" not in df.columns:
        for c in df.columns:
            if "route_type" in c.lower() or c.lower() in ("type",):
                colmap[c] = "route_type"
                break
    df = df.rename(columns=colmap)
    canonical = ["route_id","route_short_name","route_long_name","route_type"]
    for c in canonical:
        if c not in df.columns:
            df[c] = ""
    write_csv(df[canonical], outdir / "routes.txt")

def normalize_trips(indir, outdir):
    p = indir / "trips.txt"
    df = read_csv_if_exists(p)
    if df is None:
        print("  skips trips.txt (not found)")
        return
    # ensure columns
    colmap = {}
    if "trip_id" not in df.columns:
        for c in df.columns:
            if c.lower().endswith("trip_id") or c.lower()=="trip":
                colmap[c] = "trip_id"
    if "route_id" not in df.columns:
        for c in df.columns:
            if c.lower().endswith("route_id") or c.lower()=="route":
                colmap[c] = "route_id"
    if "service_id" not in df.columns:
        for c in df.columns:
            if c.lower().endswith("service_id"):
                colmap[c] = "service_id"
    if "trip_headsign" not in df.columns:
        for c in df.columns:
            if "headsign" in c.lower():
                colmap[c] = "trip_headsign"
    if "shape_id" not in df.columns:
        for c in df.columns:
            if "shape" in c.lower() and "id" in c.lower():
                colmap[c] = "shape_id"
    df = df.rename(columns=colmap)
    canonical = ["route_id","service_id","trip_id","trip_headsign","direction_id","shape_id"]
    for c in canonical:
        if c not in df.columns:
            df[c] = ""
    write_csv(df[canonical], outdir / "trips.txt")

def normalize_stop_times(indir, outdir):
    p = indir / "stop_times.txt"
    df = read_csv_if_exists(p)
    if df is None:
        print("  skips stop_times.txt (not found)")
        return
    colmap = {}
    # map arrival/departure/stop_id/sequence
    for c in df.columns:
        lc = c.lower()
        if "arrival" in lc and "time" in lc:
            colmap[c] = "arrival_time"
        if "departure" in lc and "time" in lc:
            colmap[c] = "departure_time"
        if lc.endswith("stop_id") and "stop_id" not in colmap.values():
            colmap[c] = "stop_id"
        if "stop_sequence" in lc or lc.endswith("sequence"):
            colmap[c] = "stop_sequence"
    df = df.rename(columns=colmap)
    canonical = ["trip_id","arrival_time","departure_time","stop_id","stop_sequence","stop_headsign","pickup_type","drop_off_type"]
    for c in canonical:
        if c not in df.columns:
            df[c] = ""
    # ensure sequence numeric
    df['stop_sequence'] = pd.to_numeric(df['stop_sequence'], errors='coerce').fillna(0).astype(int)
    write_csv(df[["trip_id","arrival_time","departure_time","stop_id","stop_sequence","stop_headsign"]], outdir / "stop_times.txt")

def normalize_shapes(indir, outdir):
    p = indir / "shapes.txt"
    df = read_csv_if_exists(p)
    if df is None:
        print("  skips shapes.txt (not found)")
        return
    colmap = {}
    for c in df.columns:
        lc = c.lower()
        if "shape_id" in lc:
            colmap[c] = "shape_id"
        elif lc.endswith("lat"):
            colmap[c] = "shape_pt_lat"
        elif lc.endswith("lon") or lc.endswith("lng"):
            colmap[c] = "shape_pt_lon"
        elif "sequence" in lc:
            colmap[c] = "shape_pt_sequence"
        elif "dist" in lc:
            colmap[c] = "shape_dist_traveled"
    df = df.rename(columns=colmap)
    canonical = ["shape_id","shape_pt_lat","shape_pt_lon","shape_pt_sequence","shape_dist_traveled"]
    for c in canonical:
        if c not in df.columns:
            df[c] = ""
    df['shape_pt_lat'] = pd.to_numeric(df['shape_pt_lat'], errors='coerce')
    df['shape_pt_lon'] = pd.to_numeric(df['shape_pt_lon'], errors='coerce')
    df['shape_pt_sequence'] = pd.to_numeric(df['shape_pt_sequence'], errors='coerce').fillna(0).astype(int)
    write_csv(df[canonical], outdir / "shapes.txt")

def normalize_calendar(indir, outdir):
    p = indir / "calendar.txt"
    df = read_csv_if_exists(p)
    if df is None:
        print("  skips calendar.txt (not found)")
        return
    # We expect service_id,monday,tuesday,...,end_date
    # Real data had monday,...,end_date,service_id (service_id moved to end)
    cols = [c.lower() for c in df.columns]
    if 'service_id' in cols and cols.index('service_id') != 0:
        # move service_id to front
        cols_new = ['service_id'] + [c for c in df.columns if c.lower() != 'service_id']
        df = df[cols_new]
    # ensure canonical column names exist
    canonical = ["service_id","monday","tuesday","wednesday","thursday","friday","saturday","sunday","start_date","end_date"]
    for c in canonical:
        if c not in df.columns:
            df[c] = ""
    write_csv(df[canonical], outdir / "calendar.txt")

def main(args):
    indir = Path(args.indir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print("Normalizing GTFS from", indir, "→", outdir)
    normalize_stops(indir, outdir)
    normalize_routes(indir, outdir)
    normalize_trips(indir, outdir)
    normalize_stop_times(indir, outdir)
    normalize_shapes(indir, outdir)
    normalize_calendar(indir, outdir)
    print("Done. Inspect files in", outdir)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="indir", default="data/gtfs_real")
    p.add_argument("--out", dest="outdir", default="data/gtfs")
    args = p.parse_args()
    main(args)
