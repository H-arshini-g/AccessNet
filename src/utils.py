# src/utils.py
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

TIME_FMT = "%H:%M:%S"

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def parse_time(s):
    # GTFS times sometimes exceed 24:00:00; handle that
    if pd.isna(s):
        return None
    parts = s.split(":")
    h = int(parts[0])
    m = int(parts[1])
    sec = int(parts[2])
    h = h % 48  # keep sensible
    return timedelta(hours=h, minutes=m, seconds=sec)

def save_pickle(obj, path):
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path):
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)