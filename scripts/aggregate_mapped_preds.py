# scripts/aggregate_mapped_preds.py
import pandas as pd
infile = "data/outputs/lstm_preds_mapped.csv"
outfile = "data/outputs/lstm_preds_agg.csv"
df = pd.read_csv(infile)
# average prediction per stop_id
agg = df.groupby("stop_id")["pred"].mean().reset_index()
agg = agg.rename(columns={"pred": "ema_pred"})
agg.to_csv(outfile, index=False)
print("Wrote:", outfile)
