# tools/debug_candidates.py
import argparse, pickle, os
import numpy as np
from collections import OrderedDict
import pandas as pd
import math
try:
    from scipy.spatial import cKDTree as KDTree
except Exception:
    KDTree = None

import networkx as nx
import osmnx as ox

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1); dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def path_length_m(G, path):
    total=0.0
    for u,v in zip(path[:-1], path[1:]):
        if G.has_edge(u,v):
            total += float(G[u][v].get('length', G[u][v].get('weight',1.0)))
        elif G.has_edge(v,u):
            total += float(G[v][u].get('length', G[v][u].get('weight',1.0)))
        else:
            total += 1.0
    return total

def nearest_stop(tree, stop_coords, stop_ids, lat, lon):
    if tree is None:
        diffs = ((stop_coords[:,0] - lat)**2 + (stop_coords[:,1] - lon)**2)
        idx = int(diffs.argmin())
        return stop_ids[idx], stop_coords[idx]
    _, idx = tree.query([lat, lon])
    return stop_ids[int(idx)], stop_coords[int(idx)]

def compute_shortest_candidates(G, orig_node, dest_node, k=20, max_seconds=6, max_iter=400):
    try:
        gen = nx.shortest_simple_paths(G, orig_node, dest_node, weight='weight')
    except Exception:
        return [nx.shortest_path(G, orig_node, dest_node, weight='weight')]
    paths=[]
    import time
    start=time.time(); it=0
    while True:
        try:
            p=next(gen)
        except StopIteration:
            break
        except Exception:
            break
        paths.append(list(p))
        it+=1
        if len(paths)>=k or it>=max_iter or (time.time()-start)>max_seconds:
            break
    return paths

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--stops", required=True)
    parser.add_argument("--preds", required=True)
    parser.add_argument("--origin", required=True)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--k", type=int, default=24)
    parser.add_argument("--max-detour-pct", type=float, default=0.3)
    args=parser.parse_args()

    stops = pd.read_csv(args.stops, dtype=str)
    stop_coords = np.column_stack((stops["stop_lat"].astype(float).values, stops["stop_lon"].astype(float).values))
    stop_ids = stops["stop_id"].astype(str).values.tolist()
    kd = KDTree(stop_coords) if KDTree is not None else None

    with open(args.cache, "rb") as f:
        G = pickle.load(f)

    # resolve origin/dest nodes
    orow = stops[stops["stop_name"]==args.origin]
    if orow.empty: orow = stops[stops["stop_id"]==args.origin]
    drow = stops[stops["stop_name"]==args.dest]
    if drow.empty: drow = stops[stops["stop_id"]==args.dest]
    if orow.empty or drow.empty:
        print("Origin/dest not found in stops file. Ensure you passed exact names/ids.")
        raise SystemExit(1)
    o_lat, o_lon = float(orow.iloc[0]["stop_lat"]), float(orow.iloc[0]["stop_lon"])
    d_lat, d_lon = float(drow.iloc[0]["stop_lat"]), float(drow.iloc[0]["stop_lon"])

    orig_node = ox.distance.nearest_nodes(G, o_lon, o_lat)
    dest_node = ox.distance.nearest_nodes(G, d_lon, d_lat)

    Gs = nx.Graph()
    Gs.add_nodes_from((n, dict(G.nodes[n])) for n in G.nodes())
    for u,v,data in G.edges(data=True):
        w = data.get('length', data.get('weight',1.0))
        Gs.add_edge(u,v,weight=w)

    cand = compute_shortest_candidates(Gs, orig_node, dest_node, k=args.k)
    print(f"Found {len(cand)} candidates")
    lengths=[path_length_m(Gs,p) for p in cand]
    best_len=min(lengths) if lengths else 0
    filtered=[]
    for p,L in zip(cand,lengths):
        if L <= best_len*(1.0+args.max_detour_pct):
            filtered.append((p,L))
    print("Filtered to", len(filtered), "candidates (detour filter)")
    # load predictions for access score
    preds = pd.read_csv(args.preds)
    # build simple access map fallback (mean per stop_id if available)
    acc_map={}
    if "stop_id" in preds.columns and ("ema_pred" in preds.columns or "pred" in preds.columns):
        if "ema_pred" in preds.columns:
            acc_map = preds.groupby("stop_id")["ema_pred"].mean().to_dict()
        else:
            acc_map = preds.groupby("stop_id")["pred"].mean().to_dict()

    def path_access(p):
        matched=[]
        for n in p:
            node = G.nodes[n]
            if "y" not in node or "x" not in node: continue
            sid, sc = nearest_stop(kd, stop_coords, stop_ids, node["y"], node["x"])
            matched.append(sid)
        unique = list(OrderedDict.fromkeys(matched))
        scores = [acc_map.get(sid, 0.0) for sid in unique]
        return float(sum(scores)/len(scores)) if scores else 0.0

    rows=[]
    for i,(p,L) in enumerate(filtered):
        acc=path_access(p)
        # connectivity fraction
        near=0; total=0; first=None; last=None
        for n in p:
            node=G.nodes[n]
            if "y" not in node or "x" not in node: continue
            total+=1
            if first is None: first=(node["y"], node["x"])
            last=(node["y"], node["x"])
            sid, sc = nearest_stop(kd, stop_coords, stop_ids, node["y"], node["x"])
            if haversine_km(node["y"], node["x"], sc[0], sc[1]) <= 0.15:
                near+=1
        frac = (near/total) if total>0 else 0.0
        cm1 = haversine_km(o_lat, o_lon, first[0], first[1])*1000.0 if first else 999999.0
        cm2 = haversine_km(d_lat, d_lon, last[0], last[1])*1000.0 if last else 999999.0
        connector = cm1+cm2
        rows.append((i, int(L), round(acc,4), round(frac,4), int(connector)))
    # print table
    print("idx | len_m | acc | conn_frac | connector_m")
    for r in rows:
        print("{:>3} | {:>6} | {:>5} | {:>8} | {:>10}".format(*r))
    print("\nPick indices from left column and tell me which index is the long connected route vs floating short route.")
