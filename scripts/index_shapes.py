# scripts/index_shapes.py
import pandas as pd, json, sys
shapes = pd.read_csv(sys.argv[1], dtype=str)
out = {}
for sid, g in shapes.groupby("shape_id"):
    lats = g["shape_pt_lat"].astype(float).values
    lons = g["shape_pt_lon"].astype(float).values
    out[sid] = [float(lats.min()), float(lats.max()), float(lons.min()), float(lons.max())]
with open(sys.argv[2], "w") as f:
    json.dump(out, f)
print("Wrote", sys.argv[2])
