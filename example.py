import pandas as pd

stops = pd.read_csv("data/gtfs/stops.txt")
print("Lat range:", stops["stop_lat"].min(), "to", stops["stop_lat"].max())
print("Lon range:", stops["stop_lon"].min(), "to", stops["stop_lon"].max())
