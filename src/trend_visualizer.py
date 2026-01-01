import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from datetime import datetime

def trend_visualizer(tickets_file, preds_file, stops_file, out_dir):
    print("\n📈 Generating accessibility trend visualizations...")

    # --- 1️⃣ Load data ---
    tickets = pd.read_csv(tickets_file)
    preds = pd.read_csv(preds_file)
    stops = pd.read_csv(stops_file)

    os.makedirs(out_dir, exist_ok=True)

    # --- 2️⃣ Convert timestamps and create daily/weekly aggregates ---
    tickets["timestamp"] = pd.to_datetime(tickets["timestamp"], errors="coerce")
    tickets["date"] = tickets["timestamp"].dt.date
    daily_counts = tickets.groupby(["stop_id", "date"]).size().reset_index(name="passenger_count")

    # --- 3️⃣ Compute average passenger trends per stop ---
    trend = daily_counts.groupby(["stop_id", "date"])["passenger_count"].mean().reset_index()

    # --- 4️⃣ Get accessibility predictions from LSTM output ---
    if "ema_pred" in preds.columns:
        acc = preds.groupby("stop_id")["ema_pred"].mean().reset_index()
    else:
        last_row = preds.iloc[-1]
        stop_cols = [c for c in preds.columns if c.startswith("stop_")]
        acc = pd.DataFrame({
            "stop_id": stops["stop_id"],
            "predicted_accessibility": [last_row[c] for c in stop_cols]
        })

    # --- 5️⃣ Normalize passenger count per stop ---
    trend["norm_count"] = trend.groupby("stop_id")["passenger_count"].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6)
    )

    # --- 6️⃣ Join with accessibility predictions ---
    merged = pd.merge(trend, acc, on="stop_id", how="left")

    # --- 7️⃣ Plot for each stop ---
    for stop_id in merged["stop_id"].unique():
        df_stop = merged[merged["stop_id"] == stop_id].sort_values("date")

        plt.figure(figsize=(8, 4))
        plt.plot(df_stop["date"], df_stop["norm_count"], label="Historical Passenger Trend", linewidth=2)
        plt.axhline(y=df_stop["predicted_accessibility"].mean(), color="red",
                    linestyle="--", label="Predicted Accessibility (Model Output)")

        plt.title(f"Stop {stop_id}: Historical vs Predicted Accessibility Trend")
        plt.xlabel("Date")
        plt.ylabel("Normalized Demand / Accessibility")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        out_path = os.path.join(out_dir, f"trend_{stop_id}.png")
        plt.savefig(out_path)
        plt.close()

        print(f"✅ Trend chart saved for {stop_id}: {out_path}")

    print("\n📊 All trend visualizations generated successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize historical and predicted accessibility trends")
    parser.add_argument("--tickets", default="data/tickets/tickets.csv")
    parser.add_argument("--preds", default="data/outputs/lstm_preds.csv")
    parser.add_argument("--stops", default="data/gtfs/stops.txt")
    parser.add_argument("--out", default="data/outputs/trends")
    args = parser.parse_args()

    trend_visualizer(args.tickets, args.preds, args.stops, args.out)
