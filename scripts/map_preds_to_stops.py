# scripts/map_preds_to_stops.py
import pandas as pd
import numpy as np
import sys

PRED = "data/outputs/lstm_preds.csv"
STOPS = "data/gtfs/stops.txt"   # normalized GTFS stops
OUT = "data/outputs/lstm_preds_mapped.csv"

# load preds
dfp = pd.read_csv(PRED)
cols = [c for c in dfp.columns if c.startswith("stop_")]
if not cols:
    print("No stop_ columns detected — leaving file untouched.")
    dfp.to_csv(OUT, index=False)
    print("Wrote:", OUT)
    sys.exit(0)

# load stops
stops = pd.read_csv(STOPS)
# We have to assume the feature ordering matches the stops order used by features.py.
# features.py likely used stops['stop_id'] order or stops_parsed.csv in outputs.
# Attempt to pick stops order from data/outputs/stops_parsed.csv if it exists:
try:
    sp = pd.read_csv("data/outputs/stops_parsed.csv")
    stop_ids = sp["stop_id"].astype(str).tolist()
    print("Using data/outputs/stops_parsed.csv for stop ordering.")
except Exception:
    stop_ids = stops["stop_id"].astype(str).tolist()
    print("Using data/gtfs/stops.txt for stop ordering.")

if len(stop_ids) < len(cols):
    print("Warning: number of stops in GTFS is smaller than model outputs.")
    # if mismatch, truncate or repeat as fallback
    stop_ids = stop_ids[:len(cols)]

# construct mapped dataframe: each row -> time_step, then many rows (stop_id, pred)
out_rows = []
for _, row in dfp.iterrows():
    t = row.get("time_step", None)
    for i, c in enumerate(cols):
        sid = stop_ids[i] if i < len(stop_ids) else f"STOP_{i}"
        out_rows.append({"time_step": t, "stop_col": c, "stop_id": sid, "pred": row[c]})

out = pd.DataFrame(out_rows)
out.to_csv(OUT, index=False)
print("Wrote mapped preds to:", OUT)
