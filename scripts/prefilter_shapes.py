# scripts/prefilter_shapes.py
# Usage: python scripts/prefilter_shapes.py --shapes data/gtfs/shapes.txt --stops data/gtfs/stops.txt --origin 20558 --dest 29374 --out data/gtfs/shapes_filtered.txt --margin-km 2

import argparse
import pandas as pd
from shapely.geometry import LineString, Point, box
from shapely.ops import unary_union
import math

def km_to_deg(km):
    # approx conversion at mid-latitude; safe for a bbox margin
    return km / 111.0

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", required=True)
    p.add_argument("--stops", required=True)
    p.add_argument("--origin", required=True)
    p.add_argument("--dest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--margin-km", type=float, default=2.0)
    args = p.parse_args()

    shapes = pd.read_csv(args.shapes, dtype=str)
    stops = pd.read_csv(args.stops, dtype=str)

    o = stops[stops["stop_id"] == str(args.origin)].iloc[0]
    d = stops[stops["stop_id"] == str(args.dest)].iloc[0]
    o_lat, o_lon = float(o["stop_lat"]), float(o["stop_lon"])
    d_lat, d_lon = float(d["stop_lat"]), float(d["stop_lon"])

    mid_lat = (o_lat + d_lat) / 2.0
    margin_deg = km_to_deg(args.margin_km)

    # bbox around origin-dest with margin
    minx = min(o_lon, d_lon) - margin_deg
    maxx = max(o_lon, d_lon) + margin_deg
    miny = min(o_lat, d_lat) - margin_deg
    maxy = max(o_lat, d_lat) + margin_deg
    bbox = box(minx, miny, maxx, maxy)

    kept_ids = set()
    # group by shape_id and build lines
    for sid, grp in shapes.groupby("shape_id"):
        grp = grp.sort_values("shape_pt_sequence")
        pts = list(zip(grp["shape_pt_lon"].astype(float), grp["shape_pt_lat"].astype(float)))
        if len(pts) < 2:
            continue
        ln = LineString(pts)
        if ln.intersects(bbox):
            kept_ids.add(sid)

    # write filtered shapes rows where shape_id in kept_ids
    filtered = shapes[shapes["shape_id"].isin(kept_ids)].copy()
    # optional: reduce precision to speed read/write
    filtered["shape_pt_lat"] = filtered["shape_pt_lat"].astype(float).round(6)
    filtered["shape_pt_lon"] = filtered["shape_pt_lon"].astype(float).round(6)
    filtered.to_csv(args.out, index=False)

    print(f"WROTE {args.out} rows={len(filtered)} kept shape_ids={len(kept_ids)}")

if __name__ == "__main__":
    main()
