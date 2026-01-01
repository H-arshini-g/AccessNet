# src/ema_baseline.py
import pandas as pd
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--timeseries', required=True)
    parser.add_argument('--out', default='data/outputs/ema_preds.csv')
    args = parser.parse_args()

    df = pd.read_csv(args.timeseries, parse_dates=['time_window'])
    out_rows = []

    for (route, stop), g in df.groupby(['route_id', 'stop_id']):
        series = g.set_index('time_window')['count'].sort_index()
        if len(series) < 2:
            continue
        ema = series.ewm(span=3, adjust=False).mean()
        preds = ema.shift(1).dropna()
        for t, p in preds.items():
            out_rows.append({'route_id': route, 'stop_id': stop, 'time_window': t, 'ema_pred': float(p)})

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(args.out, index=False)
    print('EMA predictions saved to', args.out)
    print(out_df.head())
