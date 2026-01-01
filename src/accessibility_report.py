# src/accessibility_report.py
import pandas as pd
import os

def generate_accessibility_report(preds_file, stops_file, out_file="data/outputs/recommendations.csv"):
    # Load predictions and stops
    preds = pd.read_csv(preds_file)
    stops = pd.read_csv(stops_file)

    # Auto-detect prediction type
    if "ema_pred" in preds.columns:
        pred_type = "EMA"
        accessibility = preds.groupby("stop_id")["ema_pred"].mean().reset_index()
        accessibility.rename(columns={"ema_pred": "accessibility_score"}, inplace=True)
    elif any(c.startswith("stop_") for c in preds.columns):
        pred_type = "LSTM/GRU"
        last_row = preds.iloc[-1]
        stop_cols = [c for c in preds.columns if c.startswith("stop_")]
        accessibility = pd.DataFrame({
            "stop_id": stops["stop_id"],
            "accessibility_score": [last_row[c] for c in stop_cols]
        })
    else:
        raise ValueError("Could not detect prediction type. File must contain 'ema_pred' or 'stop_' columns.")

    # Merge with stops data
        # Merge with stops data
    df = pd.merge(stops, accessibility, on="stop_id", how="left")

    # Normalize accessibility scores (0–1 scale)
    if df["accessibility_score"].max() > 1.0:
        df["accessibility_score"] = (
            df["accessibility_score"] - df["accessibility_score"].min()
        ) / (df["accessibility_score"].max() - df["accessibility_score"].min())

    # Thresholds for accessibility levels
    low_thresh = 0.4
    med_thresh = 0.7


    # Recommendations logic
    recommendations = []
    for _, row in df.iterrows():
        score = row["accessibility_score"]
        stop_id = row["stop_id"]
        stop_name = row.get("stop_name", f"Stop {stop_id}")

        if pd.isna(score):
            rec = "⚠️ No data available"
            level = "Unknown"
        elif score < low_thresh:
            level = "Low"
            rec = "🚧 Needs ramp, elevator, or better connecting services"
        elif score < med_thresh:
            level = "Medium"
            rec = "🛠 Improve signage, pedestrian paths, or bus links"
        else:
            level = "High"
            rec = "✅ Accessible — maintain regularly"

        recommendations.append({
            "Stop ID": stop_id,
            "Stop Name": stop_name,
            "Accessibility Score": f"{score:.2f}" if pd.notna(score) else "N/A",
            "Level": level,
            "Recommendation": rec
        })

    # Save output
    rec_df = pd.DataFrame(recommendations)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    rec_df.to_csv(out_file, index=False, encoding="utf-8-sig")


    # Print summary
    print(f"\n📊 Accessibility Report ({pred_type} Predictions)")
    print("-----------------------------------------------------")
    for _, r in rec_df.iterrows():
        print(f"{r['Stop ID']} ({r['Stop Name']}): {r['Accessibility Score']} → {r['Level']} | {r['Recommendation']}")
    print("-----------------------------------------------------")
    print(f"✅ Recommendations saved to: {out_file}\n")

    # Return dataframe (optional)
    return rec_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate accessibility improvement suggestions for planners.")
    parser.add_argument("--preds", required=True, help="Path to predictions file (e.g., lstm_preds.csv or ema_preds.csv)")
    parser.add_argument("--stops", default="data/gtfs/stops.txt", help="Path to GTFS stops.txt file")
    parser.add_argument("--out", default="data/outputs/recommendations.csv", help="Output recommendations CSV path")
    args = parser.parse_args()

    generate_accessibility_report(args.preds, args.stops, args.out)
