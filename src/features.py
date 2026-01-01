# src/features.py
import pandas as pd
import numpy as np
import argparse
from utils import ensure_dir


def create_lagged(X, lags=6):
    """
    Create lagged features for time series forecasting.

    Parameters:
    -----------
    X : DataFrame
        Columns: ['time_window','route_id','stop_id','count']
    lags : int
        Number of past time steps to include.

    Returns:
    --------
    list of (sequence, target, time)
    """
    X = X.sort_values('time_window')
    pivot = X.pivot_table(
        values='count',
        index='time_window',
        columns=['route_id', 'stop_id'],
        fill_value=0
    )
    sequences = []
    times = pivot.index
    arr = pivot.values
    for i in range(lags, arr.shape[0]):
        seq = arr[i - lags:i]
        target = arr[i]
        sequences.append((seq, target, times[i]))
    return sequences


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tickets', required=True)
    parser.add_argument('--out', default='data/outputs/features.npz')
    parser.add_argument('--lags', type=int, default=6)
    args = parser.parse_args()

    df = pd.read_csv(args.tickets, parse_dates=['time_window'])
    seqs = create_lagged(df, lags=args.lags)

    X = np.array([s[0] for s in seqs])
    y = np.array([s[1] for s in seqs])
    times = np.array([s[2] for s in seqs])

    ensure_dir('/'.join(args.out.split('/')[:-1]))
    np.savez_compressed(args.out, X=X, y=y, times=times)
    print('Saved features to', args.out)
    print('X shape:', X.shape, 'y shape:', y.shape)
