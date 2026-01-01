import argparse
import os
import pandas as pd


def generate_simple_shapes(gtfs_folder, out_file="shapes.txt"):
    stops = pd.read_csv(os.path.join(gtfs_folder, "stops.txt"))
    stop_times = pd.read_csv(os.path.join(gtfs_folder, "stop_times.txt"))

    shape_rows = []
    shape_id_counter = 0

    for trip_id, group in stop_times.groupby("trip_id"):
        group = group.sort_values("stop_sequence")
        for seq, stop_id in enumerate(group["stop_id"], start=1):
            stop = stops.loc[stops["stop_id"] == stop_id].iloc[0]
            shape_rows.append({
                "shape_id": f"shape_{shape_id_counter}",
                "shape_pt_lat": stop["stop_lat"],
                "shape_pt_lon": stop["stop_lon"],
                "shape_pt_sequence": seq
            })
        shape_id_counter += 1

    shapes = pd.DataFrame(shape_rows)
    shapes.to_csv(os.path.join(gtfs_folder, out_file), index=False)
    print(f"Simplified shapes.txt saved to {os.path.join(gtfs_folder, out_file)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gtfs", required=True, help="Path to GTFS folder")
    parser.add_argument("--out", default="shapes.txt", help="Output shapes.txt name")
    args = parser.parse_args()

    generate_simple_shapes(args.gtfs, args.out)
