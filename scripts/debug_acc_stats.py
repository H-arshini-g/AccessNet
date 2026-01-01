# scripts/debug_acc_stats.py
import pandas as pd, numpy as np, sys
preds = pd.read_csv("data/outputs/lstm_preds_mapped.csv")
print("Columns:", preds.columns.tolist())
if "ema_pred" in preds.columns:
    vals = preds["ema_pred"].dropna().astype(float).values
elif "pred" in preds.columns:
    vals = preds.groupby("stop_id")["pred"].mean().values
else:
    cols = [c for c in preds.columns if c.startswith("stop_")]
    vals = []
    for c in cols:
        vals.extend(pd.to_numeric(preds[c], errors="coerce").dropna().astype(float).tolist())
    vals = np.array(vals) if vals else np.array([])
print("n", len(vals))
if len(vals):
    print("min", vals.min(), "max", vals.max(), "mean", vals.mean(), "std", vals.std())
    print("pctiles:", {p: float(np.percentile(vals,p)) for p in [0,1,5,25,50,75,95,99,100]})
else:
    print("No numeric preds found.")
