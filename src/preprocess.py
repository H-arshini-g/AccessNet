# src/preprocess.py
import pandas as pd
import argparse
import os
from utils import ensure_dir, parse_time
from datetime import datetime

# This script reads GTFS files and (optionally) ticket validation CSV(s). It aggregates ticket validations into
# passenger counts per route-stop per time-window (default 10 minutes) and exports a timeseries CSV used by models.


def read_gtfs(gtfs_dir):
    stops = pd.read_csv(os.path.join(gtfs_dir, 'stops.txt'))
    stop_times = pd.read_csv(os.path.join(gtfs_dir, 'stop_times.txt'))
    trips = pd.read_csv(os.path.join(gtfs_dir, 'trips.txt'))
    routes = pd.read_csv(os.path.join(gtfs_dir, 'routes.txt'))
    return stops, stop_times, trips, routes


def aggregate_tickets(ticket_folder, window_minutes=10, out_path='data/outputs/timeseries.csv'):
    # Expect CSV files with at least: timestamp, route_id, stop_id
    import glob
    files = glob.glob(os.path.join(ticket_folder, '*.csv'))
    if not files:
        raise FileNotFoundError('No ticket CSVs found in ' + ticket_folder)
    df_list = []
    for f in files:
        df = pd.read_csv(f, parse_dates=['timestamp'])
        df_list.append(df)
    df = pd.concat(df_list, ignore_index=True)
    df['time_window'] = df['timestamp'].dt.floor(f'{window_minutes}T')
    grouped = df.groupby(['route_id', 'stop_id', 'time_window']).size().reset_index(name='count')
    grouped.to_csv(out_path, index=False)
    return out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gtfs', required=True)
    parser.add_argument('--tickets', required=False)
    parser.add_argument('--out', default='data/outputs')
    args = parser.parse_args()
    ensure_dir(args.out)
    print('Reading GTFS...')
    stops, stop_times, trips, routes = read_gtfs(args.gtfs)
    stops.to_csv(os.path.join(args.out, 'stops_parsed.csv'), index=False)
    print('GTFS parsed.')
    if args.tickets:
        print('Aggregating tickets...')
        ts_path = aggregate_tickets(args.tickets, out_path=os.path.join(args.out, 'timeseries.csv'))
        print('Timeseries saved to', ts_path)
    else:
        print('No tickets folder provided — skipping timeseries aggregation.')