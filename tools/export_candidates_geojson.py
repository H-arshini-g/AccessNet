# tools/export_candidates_geojson.py
import argparse, pickle, os, math, json
import pandas as pd
import numpy as np
import networkx as nx
import osmnx as ox
try:
    from scipy.spatial import cKDTree as KDTree
except Exception:
    KDTree = None

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

def nearest_stop(tree, coords, stop_ids, lat, lon):
    if tree is None:
        diffs = ((coords[:,0] - lat)**2 + (coords[:,1] - lon)**2)
        idx = int(diffs.argmin())
        return stop_ids[idx], coords[idx]
    _, idx = tree.query([lat, lon])
    return stop_ids[int(idx)], coords[int(idx)]

def compute_candidates(G, orig_node, dest_node, k=24, max_seconds=6, max_iter=500):
    try:
        gen = nx.shortest_simple_paths(G, orig_node, dest_node, weight='weight')
    except Exception:
        try:
            return [nx.shortest_path(G, orig_node, dest_node, weight='weight')]
        except Exception:
            return []
    paths=[]
    import time
    start=time.time(); it=0
    for p in gen:
        paths.append(list(p))
        it+=1
        if len(paths)>=k or it>=max_iter or (time.time()-start)>max_seconds:
            break
    return paths

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--stops", required=True)
    p.add_argument("--preds", required=True)
    p.add_argument("--origin", required=True)
    p.add_argument("--dest", required=True)
    p.add_argument("--k", type=int, default=24)
    p.add_argument("--out", default="data/outputs/candidates.geojson")
    args=p.parse_args()

    stops = pd.read_csv(args.stops, dtype=str)
    stop_coords = np.column_stack((stops["stop_lat"].astype(float).values, stops["stop_lon"].astype(float).values))
    stop_ids = stops["stop_id"].astype(str).values.tolist()
    kd = KDTree(stop_coords) if KDTree is not None else None

    with open(args.cache,"rb") as f:
        G = pickle.load(f)

    # resolve origin/dest coords
    orow = stops[stops["stop_name"]==args.origin]
    if orow.empty: orow = stops[stops["stop_id"]==args.origin]
    drow = stops[stops["stop_name"]==args.dest]
    if drow.empty: drow = stops[stops["stop_id"]==args.dest]
    if orow.empty or drow.empty:
        raise SystemExit("Origin/dest not found in stops file")

    o_lat, o_lon = float(orow.iloc[0]["stop_lat"]), float(orow.iloc[0]["stop_lon"])
    d_lat, d_lon = float(drow.iloc[0]["stop_lat"]), float(drow.iloc[0]["stop_lon"])

    orig_node = ox.distance.nearest_nodes(G, o_lon, o_lat)
    dest_node = ox.distance.nearest_nodes(G, d_lon, d_lat)

    # build a simple graph copy for candidate generation
    Gs = nx.Graph()
    Gs.add_nodes_from((n, dict(G.nodes[n])) for n in G.nodes())
    for u,v,data in G.edges(data=True):
        w = data.get('length', data.get('weight',1.0))
        Gs.add_edge(u,v,weight=w)

    candidates = compute_candidates(Gs, orig_node, dest_node, k=args.k)

    # load preds -> simple stop acc map (mean)
    preds = pd.read_csv(args.preds)
    acc_map = {}
    if "stop_id" in preds.columns and ("ema_pred" in preds.columns or "pred" in preds.columns):
        if "ema_pred" in preds.columns:
            acc_map = preds.groupby("stop_id")["ema_pred"].mean().to_dict()
        else:
            acc_map = preds.groupby("stop_id")["pred"].mean().to_dict()

    features=[]
    for i, p in enumerate(candidates):
        coords=[]
        matched=[]
        for n in p:
            node = G.nodes[n]
            if "y" not in node or "x" not in node: continue
            coords.append([float(node["x"]), float(node["y"])])  # lon,lat
            sid, sc = nearest_stop(kd, stop_coords, stop_ids, node["y"], node["x"])
            matched.append(sid)
        length_m = int(path_length_m(Gs,p))
        unique = list(dict.fromkeys(matched))
        acc_score = float(sum((acc_map.get(s,0.0) for s in unique))/len(unique)) if unique else 0.0
        # compute connectivity fraction
        near=0; total=0
        first=None; last=None
        for n in p:
            node = G.nodes[n]
            if "y" not in node or "x" not in node: continue
            if first is None: first=(node["y"], node["x"])
            last=(node["y"], node["x"])
            total+=1
            sid, sc = nearest_stop(kd, stop_coords, stop_ids, node["y"], node["x"])
            if haversine_km(node["y"], node["x"], sc[0], sc[1]) <= 0.15:
                near+=1
        conn_frac = (near/total) if total>0 else 0.0
        connector_m = int((haversine_km(o_lat,o_lon, first[0], first[1]) + haversine_km(d_lat,d_lon, last[0], last[1]))*1000.0) if first and last else 999999
        feat = {
            "type":"Feature",
            "geometry": {"type":"LineString", "coordinates": coords},
            "properties": {"idx": i, "length_m": length_m, "acc": round(acc_score,4), "conn_frac": round(conn_frac,4), "connector_m": connector_m}
        }
        features.append(feat)

    geo = {"type":"FeatureCollection", "features": features}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(geo, f, indent=2)
    print("WROTE", args.out, "candidates:", len(features))
