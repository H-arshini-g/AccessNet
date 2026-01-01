# src/plot_alternatives.py
"""
Robust plot_alternatives.py (updated)
- Auto-adjust OSM radius for far origins/destinations
- Auto-stretch visualization for color mapping using p90-based stretch
- CLI flags added: --stretch-target and --stretch-max
- Popup shows Crowd label (Busy/Medium/Calm) instead of raw passenger number
- Preserves previous functionality (shapes overlay, caching, recs CSV, etc.)
"""
import argparse
import pandas as pd
import folium
import osmnx as ox
import networkx as nx
import matplotlib.cm as cm
import matplotlib.colors as colors
import os
import pickle
import math
from statistics import mean
import numpy as np
from collections import OrderedDict
import time
try:
    from scipy.spatial import cKDTree as KDTree
except Exception:
    KDTree = None

# ---- helpers ----
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1); dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def path_length(G, path):
    if not path or len(path) < 2: return float("inf")
    total = 0.0
    for a,b in zip(path[:-1], path[1:]):
        if G.has_edge(a,b):
            total += G[a][b].get("weight", G[a][b].get("length", 1.0))
        elif G.has_edge(b,a):
            total += G[b][a].get("weight", G[b][a].get("length", 1.0))
        else:
            total += 1.0
    return total

def compute_alternatives(G_simple, orig_node, dest_node, k=3, expand_factor=4, max_seconds=6, max_iter=200):
    paths = []
    k_req = max(k, int(k * expand_factor))
    try:
        gen = nx.shortest_simple_paths(G_simple, orig_node, dest_node, weight="weight")
    except Exception:
        try:
            return [nx.shortest_path(G_simple, orig_node, dest_node, weight="weight")]
        except Exception:
            return []
    start = time.time()
    iters = 0
    for p in gen:
        paths.append(list(p))
        iters += 1
        if len(paths) >= k_req:
            break
        if iters >= max_iter:
            break
        if (time.time() - start) > max_seconds:
            break
    return paths

def shapes_near_bbox(shapes_df, bbox):
    lat_min, lat_max, lon_min, lon_max = bbox
    cond = (shapes_df["shape_pt_lat"].astype(float) >= lat_min) & (shapes_df["shape_pt_lat"].astype(float) <= lat_max) & \
           (shapes_df["shape_pt_lon"].astype(float) >= lon_min) & (shapes_df["shape_pt_lon"].astype(float) <= lon_max)
    return set(shapes_df.loc[cond, "shape_id"].unique().tolist())

def downsample_coords(coords, max_points=500):
    if len(coords) <= max_points: return coords
    step = max(1, int(len(coords) / max_points))
    return coords[::step]

def write_recommendations_for_stops(recs_out, stop_ids, acc_map, stops_df):
    rows = []
    for sid in stop_ids:
        pred = acc_map.get(sid, None)
        row = {"stop_id": sid, "pred": pred if pred is not None else ""}
        if pred is None:
            row["rec"] = ""
        elif pred < 0.33:
            row["rec"] = "Busy: consider ramps, lifts, wider doors, priority boarding"
        elif pred < 0.66:
            row["rec"] = "Moderate: signage, boarding assistance"
        else:
            row["rec"] = "Good: maintain accessibility"
        srow = stops_df[stops_df["stop_id"] == sid]
        row["stop_name"] = srow.iloc[0]["stop_name"] if (not srow.empty and "stop_name" in srow.columns) else ""
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(recs_out, index=False)
    return df

# ---- load predictions / normalize ----
def load_accessibility(preds_file, stops_df, transform=None, clip_pct=None, debug=False):
    preds = pd.read_csv(preds_file)
    if debug: print(f"DEBUG: loading preds from {preds_file}")

    raw_map = {}
    acc_map = {}

    # long format
    if {"time_step","stop_col","stop_id","pred"}.issubset(preds.columns):
        if debug: print("DEBUG: using long format stop_id/pred grouping")
        grouped = preds.groupby("stop_id")["pred"].mean().to_dict()
        raw_map = {str(k): float(v) for k,v in grouped.items()}

    elif "ema_pred" in preds.columns and "stop_id" in preds.columns:
        grouped = preds.groupby("stop_id")["ema_pred"].mean().to_dict()
        raw_map = {str(k): float(v) for k,v in grouped.items()}
        if debug: print("DEBUG: using ema_pred grouping")

    elif {"stop_id","pred"}.issubset(preds.columns):
        grouped = preds.groupby("stop_id")["pred"].mean().to_dict()
        raw_map = {str(k): float(v) for k,v in grouped.items()}

    elif any(c.startswith("stop_") for c in preds.columns):
        stop_cols = [c for c in preds.columns if c.startswith("stop_")]
        for sid, c in zip(stops_df["stop_id"], stop_cols):
            col = preds[c]
            nums = pd.to_numeric(col, errors="coerce").dropna()
            if len(nums) > 0:
                raw_map[str(sid)] = float(nums.mean())
            else:
                try:
                    last_val = float(pd.to_numeric(col.iloc[-1], errors="coerce"))
                    raw_map[str(sid)] = last_val
                except Exception:
                    raw_map[str(sid)] = 0.0
        if debug: print("DEBUG: constructed raw_map from stop_* columns (robust mode)")

    else:
        raise ValueError("Predictions file must contain recognizable columns (long format, 'ema_pred', 'stop_id/pred' or stop_* columns)")

    vals = np.array([v for v in raw_map.values() if pd.notna(v)], dtype=float)
    if vals.size == 0:
        acc_map = {k: 0.0 for k in raw_map.keys()}
        return acc_map, raw_map

    if clip_pct:
        lo = np.percentile(vals, (100 - clip_pct) / 2.0)
        hi = np.percentile(vals, 100 - (100 - clip_pct) / 2.0)
        for k in list(raw_map.keys()):
            v = raw_map.get(k, 0.0)
            raw_map[k] = max(min(v, hi), lo)

    if transform == "log1p":
        for k in list(raw_map.keys()):
            v = raw_map.get(k, 0.0)
            raw_map[k] = math.log1p(max(0.0, v))

    vals2 = np.array([v for v in raw_map.values() if pd.notna(v)], dtype=float)
    if vals2.size == 0:
        acc_map = {k: 0.0 for k in raw_map.keys()}
        return acc_map, raw_map

    vmin, vmax = float(vals2.min()), float(vals2.max())
    if vmax == vmin:
        acc_map = {k: 1.0 for k in raw_map.keys()}
    else:
        for k in list(raw_map.keys()):
            v = raw_map.get(k, 0.0)
            if pd.isna(v):
                acc_map[k] = 0.0
            else:
                acc_map[k] = float((v - vmin) / (vmax - vmin))

    if debug:
        arr = np.array(list(acc_map.values()))
        print(f"DEBUG: accessibility raw stats -> n={len(arr)}, min={arr.min()}, max={arr.max()}, mean={arr.mean()}")

    return acc_map, raw_map

# ---- matching + scoring helpers ----
def path_accessibility_score(path_nodes, G, tree, stop_ids_arr, stops_coords, acc_map):
    matched = []
    for n in path_nodes:
        node_y = G.nodes[n].get("y", None); node_x = G.nodes[n].get("x", None)
        if node_y is None or node_x is None: continue
        if tree is not None:
            _, idx = tree.query([node_y, node_x])
        else:
            dists = ((stops_coords[:,0]-node_y)**2 + (stops_coords[:,1]-node_x)**2)
            idx = int(dists.argmin())
        matched.append(stop_ids_arr[idx])
    if not matched: return 0.0
    unique = list(OrderedDict.fromkeys(matched))
    scores = [acc_map.get(sid, 0.0) for sid in unique]
    return mean(scores) if scores else 0.0

def get_node_access_values(path_nodes, G, tree, stop_ids_arr, stops_coords, acc_map):
    node_vals = []
    for n in path_nodes:
        node_y = G.nodes[n].get("y", None); node_x = G.nodes[n].get("x", None)
        if node_y is None or node_x is None:
            node_vals.append(0.0); continue
        if tree is not None:
            _, idx = tree.query([node_y, node_x])
        else:
            dists = ((stops_coords[:,0]-node_y)**2 + (stops_coords[:,1]-node_x)**2)
            idx = int(dists.argmin())
        sid = stop_ids_arr[idx]
        node_vals.append(acc_map.get(sid, 0.0))
    return node_vals

# ---- main plotting ----
def plot_alternatives(stops_file, shapes_file, preds_file, out_html,
                      k=3, dist=800, cache_path="data/outputs/osm_graph.pkl",
                      origin=None, dest=None, accessible_only=False, threshold=0.5,
                      tiles_provider="osm", shape_proximity_km=3.0,
                      clip_pct=None, transform=None, stop_filter="bbox", show_stops_km=0.6,
                      recs_out=None, show_stop_preds=False, wheelchair=False, debug=False,
                      max_detour_pct=0.35, color_gamma=1.0,
                      connectivity_weight=0.6, connector_penalty_weight=0.35,
                      connector_threshold_m=1000, preferred_candidate=None,
                      stretch_target=0.35, stretch_max=80.0):
    print("Loading stops and shapes...")
    stops = pd.read_csv(stops_file, dtype=str)
    shapes = pd.read_csv(shapes_file) if shapes_file and os.path.exists(shapes_file) else None

    if wheelchair:
        if "wheelchair_boarding" in stops.columns:
            stops = stops[stops["wheelchair_boarding"].astype(str).isin(["1","2","true","True","yes","Yes"])]
            if debug: print(f"DEBUG: wheelchair filter, stops left: {len(stops)}")
        else:
            if debug: print("DEBUG: wheelchair requested but no wheelchair column found")

    # load accessibility (normalized and raw)
    acc_map, raw_map = load_accessibility(preds_file, stops, transform=transform, clip_pct=clip_pct, debug=debug)

    # resolve origin/dest
    if origin is None or dest is None:
        origin = stops.iloc[0]["stop_id"]
        dest = stops.iloc[-1]["stop_id"]
    try:
        if origin in stops["stop_id"].values:
            o_row = stops[stops["stop_id"] == origin].iloc[0]
        else:
            o_row = stops[stops["stop_name"] == origin].iloc[0]
        if dest in stops["stop_id"].values:
            d_row = stops[stops["stop_id"] == dest].iloc[0]
        else:
            d_row = stops[stops["stop_name"] == dest].iloc[0]
    except Exception:
        raise

    o_lat, o_lon = float(o_row["stop_lat"]), float(o_row["stop_lon"])
    d_lat, d_lon = float(d_row["stop_lat"]), float(d_row["stop_lon"])

    # auto-adjust OSM radius
    od_km = haversine_km(o_lat, o_lon, d_lat, d_lon)
    min_needed = int(od_km * 1000.0) + 2000
    if min_needed > dist:
        if debug: print(f"DEBUG: auto-adjusting dist {dist} -> {min_needed} (od_km={od_km:.2f})")
        dist = min_needed

    stop_coords = np.column_stack((stops["stop_lat"].astype(float).values, stops["stop_lon"].astype(float).values))
    stop_ids_arr = stops["stop_id"].astype(str).values
    if KDTree is not None and len(stop_coords) > 0:
        tree = KDTree(stop_coords)
    else:
        tree = None
        if debug: print("DEBUG: KDTree not available")

    center = [stops["stop_lat"].astype(float).mean(), stops["stop_lon"].astype(float).mean()]
    m = folium.Map(location=center, zoom_start=13, tiles=None)

    def get_tile_layer_spec(provider="osm"):
        if provider == "carto_light":
            url = "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
            opts = {"minZoom": 0, "maxZoom": 20, "attribution": '&copy; OpenStreetMap contributors &copy; CARTO'}
        elif provider == "stamen_toner":
            url = "https://stamen-tiles.a.ssl.fastly.net/toner/{z}/{x}/{y}.png"
            opts = {"minZoom": 0, "maxZoom": 20, "attribution": 'Map tiles by Stamen'}
        elif provider == "stamen_terrain":
            url = "https://stamen-tiles.a.ssl.fastly.net/terrain/{z}/{x}/{y}.jpg"
            opts = {"minZoom": 0, "maxZoom": 18, "attribution": 'Map tiles by Stamen'}
        else:
            url = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            opts = {"minZoom": 0, "maxZoom": 19, "attribution": '&copy; OpenStreetMap contributors'}
        return url, opts

    tile_url, tile_opts = get_tile_layer_spec(tiles_provider)
    folium.TileLayer(tiles=tile_url, attr=tile_opts.get("attribution", ""), name=tiles_provider,
                     max_zoom=tile_opts.get("maxZoom", 19)).add_to(m)

    # pick stops slice between origin and dest
    origin_idx = stops[stops["stop_id"] == origin].index[0] if origin in stops["stop_id"].values else stops[stops["stop_name"] == origin].index[0]
    dest_idx = stops[stops["stop_id"] == dest].index[0] if dest in stops["stop_id"].values else stops[stops["stop_name"] == dest].index[0]
    slice_start, slice_end = min(origin_idx, dest_idx), max(origin_idx, dest_idx)
    stops_visible = stops.iloc[slice_start:slice_end+1].reset_index(drop=True)
    if debug: print(f"DEBUG: Stops slice length (before filtering): {len(stops_visible)}")

    # load/calc OSM graph
    if os.path.exists(cache_path):
        if debug: print(f"Loading cached OSM graph from {cache_path} ...")
        with open(cache_path, "rb") as f:
            G = pickle.load(f)
    else:
        print(f"Downloading OSM network around {center[0]:.4f}, {center[1]:.4f} with dist={dist}m ...")
        G = ox.graph_from_point((center[0], center[1]), dist=dist, network_type="walk")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(G, f)
        if debug: print(f"Saved OSM graph to {cache_path}")

    orig_node = ox.distance.nearest_nodes(G, o_lon, o_lat)
    dest_node = ox.distance.nearest_nodes(G, d_lon, d_lat)

    # pruned simple graph around midpoint
    def bbox_nodes_within(G, center_lat, center_lon, radius_m):
        deg = radius_m / 111000.0
        lat_min, lat_max = center_lat - deg, center_lat + deg
        lon_min, lon_max = center_lon - deg, center_lon + deg
        nodes = [n for n, d in G.nodes(data=True) if ("y" in d and "x" in d and lat_min <= d["y"] <= lat_max and lon_min <= d["x"] <= lon_max)]
        return set(nodes)

    buf_m = max(1000, int((od_km * 1000) / 2 + 1000))
    keep_nodes = bbox_nodes_within(G, (o_lat + d_lat) / 2.0, (o_lon + d_lon) / 2.0, buf_m)

    G_simple = nx.Graph()
    G_simple.add_nodes_from((n, dict(G.nodes[n])) for n in G.nodes())
    for u, v, data in G.edges(data=True):
        if u not in keep_nodes or v not in keep_nodes:
            continue
        w = data.get("length", data.get("weight", 1.0))
        if G_simple.has_edge(u, v):
            if w < G_simple[u][v]["weight"]:
                G_simple[u][v]["weight"] = w
        else:
            G_simple.add_edge(u, v, weight=w)

    # compute candidate paths
    candidates = compute_alternatives(G_simple, orig_node, dest_node, k=k, expand_factor=4, max_seconds=6, max_iter=400)
    if debug: print(f"DEBUG: candidates found {len(candidates)}")

    # match stops from all candidates for markers
    matched_stop_ids_candidates = []
    kd = KDTree(stop_coords) if KDTree is not None and len(stop_coords) > 0 else None
    for p in candidates:
        for n in p:
            ny = G.nodes[n].get("y", None); nx_ = G.nodes[n].get("x", None)
            if ny is None or nx_ is None: continue
            if kd is not None:
                _, idx = kd.query([ny, nx_])
            else:
                dists = ((stop_coords[:,0]-ny)**2 + (stop_coords[:,1]-nx_)**2)
                idx = int(dists.argmin())
            matched_stop_ids_candidates.append(stop_ids_arr[idx])
    matched_unique_candidates = list(OrderedDict.fromkeys(matched_stop_ids_candidates))
    if debug: print(f"DEBUG: matched stops from candidates: {len(matched_unique_candidates)}")

    # filter candidates by detour ratio and compute lengths & scores
    if not candidates:
        draw_paths = []
        path_scores = []
    else:
        lengths = [path_length(G_simple, p) for p in candidates]
        best_len = min(lengths)
        filtered = []
        for p,L in zip(candidates,lengths):
            if L <= best_len * (1.0 + max_detour_pct):
                filtered.append((p,L))
        if len(filtered)==0:
            filtered = sorted(zip(candidates,lengths), key=lambda x:x[1])[:k]
        filtered_paths = [p for p,_ in filtered]

        path_scores_all = [path_accessibility_score(p, G, kd, stop_ids_arr, stop_coords, acc_map) for p in filtered_paths]

        connector_counts = []
        connectivity_fracs = []
        connector_meters = []
        connector_far_flags = []
        for p in filtered_paths:
            near_count = 0
            total_nodes = 0
            first_coord = None
            last_coord = None
            for n in p:
                ny = G.nodes[n].get("y", None); nx_ = G.nodes[n].get("x", None)
                if ny is None or nx_ is None: continue
                total_nodes += 1
                if first_coord is None: first_coord = (ny, nx_)
                last_coord = (ny, nx_)
                if kd is not None:
                    _, idx = kd.query([ny, nx_])
                    s_lat, s_lon = stop_coords[idx]
                    if haversine_km(ny, nx_, s_lat, s_lon) <= 0.15:
                        near_count += 1
                else:
                    near_count += 1
            frac = (near_count / total_nodes) if total_nodes>0 else 0.0
            connectivity_fracs.append(frac)
            connector_counts.append(frac)
            cm1 = 0.0; cm2 = 0.0
            if first_coord is not None:
                cm1 = haversine_km(o_lat, o_lon, first_coord[0], first_coord[1]) * 1000.0
            if last_coord is not None:
                cm2 = haversine_km(d_lat, d_lon, last_coord[0], last_coord[1]) * 1000.0
            connector_meters.append(cm1 + cm2)
            connector_far_flags.append((cm1 + cm2) > connector_threshold_m)

        combined = []
        for acc, conn_frac, cmet in zip(path_scores_all, connectivity_fracs, connector_meters):
            penalty_norm = (cmet / (best_len + 1e-9))
            comb = float(connectivity_weight) * float(conn_frac) + (1.0 - float(connectivity_weight)) * float(acc) - float(connector_penalty_weight) * float(penalty_norm)
            combined.append(comb)

        if preferred_candidate is not None:
            try:
                pref_idx = int(preferred_candidate)
                if 0 <= pref_idx < len(filtered_paths):
                    combined[pref_idx] = max(combined) + 1.0
                    if debug: print(f"DEBUG: preferred_candidate={pref_idx} forced to top")
            except Exception:
                pass

        ranked_idx = sorted(range(len(filtered_paths)),
                            key=lambda i: (combined[i], connectivity_fracs[i], -lengths[i]),
                            reverse=True)

        draw_paths = [filtered_paths[i] for i in ranked_idx][:k]
        path_scores = [path_scores_all[i] for i in ranked_idx][:k]

        if debug:
            print(f"DEBUG: best_len={best_len:.1f}, kept draw_paths={len(draw_paths)} (max_detour_pct={max_detour_pct})")
            print(f"DEBUG: connector counts: {['{:.3f}'.format(x) for x in connector_counts]}")
            print(f"DEBUG: combined_scores (top): {[combined[i] for i in ranked_idx][:min(6,len(ranked_idx))]}")
            print(f"DEBUG: path_scores: {path_scores}")

    # shapes overlay (before routes)
    if shapes is not None:
        if debug: print("Processing GTFS shapes with proximity filter (shapes drawn before routes)...")
        dist_km = max(1.0, dist / 1000.0)
        delta_deg = dist_km / 111.0
        lat_min = min(o_lat, d_lat) - delta_deg
        lat_max = max(o_lat, d_lat) + delta_deg
        lon_min = min(o_lon, d_lon) - delta_deg
        lon_max = max(o_lon, d_lon) + delta_deg
        bbox = (lat_min, lat_max, lon_min, lon_max)

        nearby_shape_ids = shapes_near_bbox(shapes, bbox)
        if nearby_shape_ids:
            small_shapes = shapes[shapes["shape_id"].isin(nearby_shape_ids)].copy()
            if "shape_pt_sequence" in small_shapes.columns:
                small_shapes["shape_pt_sequence"] = small_shapes["shape_pt_sequence"].astype(float)
                grouped = small_shapes.sort_values("shape_pt_sequence").groupby("shape_id")
            else:
                grouped = small_shapes.groupby("shape_id")

            drawn = 0
            for shape_id, group in grouped:
                group = group.sort_values("shape_pt_sequence").reset_index(drop=True)
                lats = group["shape_pt_lat"].astype(float).values
                lons = group["shape_pt_lon"].astype(float).values
                if len(lats) < 2: continue
                pts = list(zip(lats, lons))
                def sq(a,b): return (a[0]-b[0])**2 + (a[1]-b[1])**2
                o_idx = min(range(len(pts)), key=lambda i: sq(pts[i], (o_lat, o_lon)))
                d_idx = min(range(len(pts)), key=lambda i: sq(pts[i], (d_lat, d_lon)))
                dist_o = haversine_km(lats[o_idx], lons[o_idx], o_lat, o_lon)
                dist_d = haversine_km(lats[d_idx], lons[d_idx], d_lat, d_lon)
                if dist_o <= shape_proximity_km and dist_d <= shape_proximity_km:
                    s_i, e_i = min(o_idx, d_idx), max(o_idx, d_idx)
                    seg = pts[s_i:e_i+1]
                    seg_ds = downsample_coords(seg, max_points=500)
                    folium.PolyLine(seg_ds, color="#2c3e91", weight=4, opacity=0.55).add_to(m)
                    drawn += 1
            if debug and drawn==0:
                print("DEBUG: no shapes matched strict proximity; nothing drawn.")

    # draw alternative thin dashed lines
    alt_hex = "#4f6b82"
    try:
        primary_cmap = getattr(cm, "colormaps", None)
        if primary_cmap is not None:
            cmap_main = cm.colormaps.get_cmap("RdYlGn")
        else:
            cmap_main = cm.get_cmap("RdYlGn")
    except Exception:
        cmap_main = cm.get_cmap("RdYlGn")

    for i, path in enumerate(draw_paths):
        if i == 0: continue
        coords = []
        for n in path:
            ny = G.nodes[n].get("y", None); nx_ = G.nodes[n].get("x", None)
            if ny is None or nx_ is None: continue
            coords.append((ny, nx_))
        if len(coords) < 2: continue
        folium.PolyLine(coords, color=alt_hex, weight=5, opacity=0.55, dash_array="8,6").add_to(m)

    # MAIN: draw primary path with colored segments using auto-stretch
    if draw_paths:
        primary = draw_paths[0]
        pscore = path_scores[0] if len(path_scores)>0 else 0.0
        coords = []
        for n in primary:
            ny = G.nodes[n].get("y", None); nx_ = G.nodes[n].get("x", None)
            if ny is None or nx_ is None: continue
            coords.append((ny, nx_))

        # connectors visual
        try:
            orig_node_coords = (G.nodes[orig_node].get("y"), G.nodes[orig_node].get("x"))
            dest_node_coords = (G.nodes[dest_node].get("y"), G.nodes[dest_node].get("x"))
            folium.PolyLine([(o_lat,o_lon), orig_node_coords], color="#1f77b4", weight=2, opacity=0.9).add_to(m)
            folium.PolyLine([(d_lat,d_lon), dest_node_coords], color="#e74c3c", weight=2, opacity=0.9).add_to(m)
        except Exception:
            pass

        node_vals = get_node_access_values(primary, G, KDTree(stop_coords) if KDTree is not None and len(stop_coords)>0 else None, stop_ids_arr, stop_coords, acc_map)
        if len(node_vals) != len(coords):
            node_vals = [pscore for _ in coords]

        # Compute auto-stretch so small values become visible:
        all_acc_vals = np.array([v for v in acc_map.values()]) if acc_map else np.array([0.0])
        p90 = float(np.percentile(all_acc_vals, 90)) if all_acc_vals.size > 0 else 0.0
        # target: map p90 to stretch_target (e.g. 0.35), compute factor; cap by stretch_max
        if p90 <= 0:
            stretch_factor = 1.0
        else:
            desired = float(max(1e-9, stretch_target))
            factor = desired / max(1e-9, p90)
            stretch_factor = min(float(stretch_max), max(1.0, factor))
        if debug: print(f"DEBUG: path coloring stretch_factor={stretch_factor:.3f} (p90={p90:.6f}, target={stretch_target}, max={stretch_max})")

        if len(coords) >= 2 and node_vals:
            folium.PolyLine(coords, color="#000000", weight=12, opacity=0.12).add_to(m)
            g = max(0.01, min(5.0, float(color_gamma)))
            for i_seg in range(len(coords)-1):
                a = node_vals[i_seg]; b = node_vals[i_seg+1]
                seg_score = (a + b)/2.0
                seg_col_val = max(0.0, min(1.0, seg_score * stretch_factor))
                seg_col_val = seg_col_val ** g
                seg_color = colors.to_hex(cmap_main(seg_col_val))
                seg_coords = [coords[i_seg], coords[i_seg+1]]
                folium.PolyLine(seg_coords, color=seg_color, weight=9, opacity=0.95).add_to(m)
            top_val = max(0.0, min(1.0, pscore * stretch_factor)) ** g
            folium.PolyLine(coords, color=colors.to_hex(cmap_main(top_val)), weight=3, opacity=1.0).add_to(m)
        else:
            top_val = max(0.0, min(1.0, pscore * stretch_factor)) ** float(color_gamma)
            folium.PolyLine(coords, color=colors.to_hex(cmap_main(top_val)), weight=9, opacity=1.0).add_to(m)

    # show stop markers derived from matched candidates
    if show_stop_preds and matched_unique_candidates:
        stops_to_show = stops[stops["stop_id"].isin(matched_unique_candidates)]
        if debug: print(f"DEBUG: Stops drawn after candidate match: {len(stops_to_show)}")
        try:
            cmap_stops = cm.colormaps.get_cmap("RdYlGn")
        except Exception:
            cmap_stops = cm.get_cmap("RdYlGn")

        # compute same stretch factor for markers (so colors match path)
        acc_vals = np.array([v for v in acc_map.values()]) if acc_map else np.array([0.0])
        p90 = float(np.percentile(acc_vals, 90)) if acc_vals.size > 0 else 0.0
        if p90 <= 0:
            stretch_factor_markers = 1.0
        else:
            desired = float(max(1e-9, stretch_target))
            factor = desired / max(1e-9, p90)
            stretch_factor_markers = min(float(stretch_max), max(1.0, factor))

        g = max(0.01, min(5.0, float(color_gamma)))
        for _, row in stops_to_show.iterrows():
            sid = str(row["stop_id"])
            d = acc_map.get(sid, 0.0)
            val = max(0.0, min(1.0, d * stretch_factor_markers))
            val = val ** g
            color = colors.to_hex(cmap_stops(val))
            # crowd label only
            if d is None:
                label = "Unknown"
            else:
                label = "Calm" if d >= 0.66 else ("Medium" if d >= 0.33 else "Busy")
            popup_html = f"<b>{row.get('stop_name', sid)}</b> ({sid})<br>Accessibility: {d:.3f}<br>Crowd: {label}"
            folium.CircleMarker(
                location=[float(row["stop_lat"]), float(row["stop_lon"])],
                radius=6, color=color, fill=True, fill_opacity=0.95,
                popup=popup_html
            ).add_to(m)

    # origin/destination markers with label
    def crowd_label(pred, thresholds=(0.33, 0.66)):
        if pred is None:
            return "Unknown"
        if pred < thresholds[0]:
            return "Busy"
        elif pred < thresholds[1]:
            return "Medium"
        else:
            return "Calm"

    o_pred = acc_map.get(str(origin), None) if str(origin) in acc_map else None
    d_pred = acc_map.get(str(dest), None) if str(dest) in acc_map else None

    folium.CircleMarker(location=[o_lat, o_lon], radius=12, color="#1f77b4", weight=3,
                        fill=True, fill_color="#1f77b4", fill_opacity=0.18).add_to(m)
    folium.CircleMarker(location=[o_lat, o_lon], radius=6, color="#ffffff", weight=2,
                        fill=True, fill_color="#1f77b4", fill_opacity=1.0,
                        popup=f"{origin} <br>Crowd: {crowd_label(o_pred)}").add_to(m)

    folium.CircleMarker(location=[d_lat, d_lon], radius=18, color=None, fill=True,
                        fill_color="#e74c3c", fill_opacity=0.22).add_to(m)
    folium.CircleMarker(location=[d_lat, d_lon], radius=12, color="#e74c3c", weight=2, fill=False,
                        popup=f"{dest} <br>Crowd: {crowd_label(d_pred)}").add_to(m)
    folium.CircleMarker(location=[d_lat, d_lon], radius=5, color="#e74c3c", weight=1, fill=True, fill_color="#e74c3c", fill_opacity=1.0).add_to(m)

    # write recommendations CSV for primary path stops (if requested)
    if recs_out and draw_paths:
        primary_nodes = draw_paths[0]
        matched_stop_ids = []
        kd_local = KDTree(stop_coords) if KDTree is not None and len(stop_coords)>0 else None
        for n in primary_nodes:
            ny = G.nodes[n].get("y", None); nx_ = G.nodes[n].get("x", None)
            if ny is None or nx_ is None: continue
            if kd_local is not None:
                _, idx = kd_local.query([ny, nx_])
            else:
                dists = ((stop_coords[:,0]-ny)**2 + (stop_coords[:,1]-nx_)**2)
                idx = int(dists.argmin())
            matched_stop_ids.append(stop_ids_arr[idx])
        matched_unique = list(OrderedDict.fromkeys(matched_stop_ids))
        origin_id = origin if origin in stops["stop_id"].values else stops[stops["stop_name"] == origin]["stop_id"].iloc[0]
        dest_id = dest if dest in stops["stop_id"].values else stops[stops["stop_name"] == dest]["stop_id"].iloc[0]
        if str(origin_id) not in matched_unique: matched_unique.insert(0, str(origin_id))
        if str(dest_id) not in matched_unique: matched_unique.append(str(dest_id))
        write_recommendations_for_stops(recs_out, matched_unique, acc_map, stops)
        if debug: print(f"DEBUG: Writing recommendations to {recs_out}")

    # legend element
    legend = f"""
    <div style="position: fixed; bottom: 40px; left: 40px; width: 360px;
                background-color: white; border:2px solid gray; border-radius:6px;
                z-index:9999; font-size:14px; padding:10px;">
      <b>Legend</b><br>
      <span style='color:#2ecc71;'>&#9632;</span> High accessibility<br>
      <span style='color:#f39c12;'>&#9632;</span> Medium accessibility<br>
      <span style='color:#e74c3c;'>&#9632;</span> Low accessibility<br>
      <br><b>Paths</b><br>
      <div style="display:flex; align-items:center">
        <div style="width:140px; height:12px; border:1px solid #aaa; background: linear-gradient(to right, #1a9850, #fee08b, #d73027); margin-right:8px;"></div>
        <div>Best route (green → red)</div>
      </div>
      <div style="margin-top:6px;">
        <span style='color:gray;'>— —</span> Other routes (dashed, lower accessibility)<br>
      </div>
        <br><a href="/data/outputs/recommendations.csv" target="_blank">Download recommendations.csv</a>

    </div>
    """
    m.get_root().html.add_child(folium.Element(legend))

    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    m.save(out_html)
    print(f"Map saved to {out_html}")


# ---- CLI ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stops", default="data/gtfs/stops.txt")
    parser.add_argument("--shapes", default="data/gtfs/shapes.txt")
    parser.add_argument("--preds", required=True)
    parser.add_argument("--out", default="data/outputs/map_alternatives.html")
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument("--dist", type=int, default=2500)
    parser.add_argument("--cache", dest="cache_path", default="data/outputs/osm_graph.pkl", help="OSM graph cache path")
    parser.add_argument("--cache-path", dest="cache_path", help="Alias for --cache", default=None)
    parser.add_argument("--origin")
    parser.add_argument("--dest")
    parser.add_argument("--accessible-only", action="store_true", help="Show only routes >= threshold")
    parser.add_argument("--threshold", type=float, default=0.5, help="Accessibility threshold (0..1)")
    parser.add_argument("--tiles", default="osm", help="Tile provider: osm|carto_light|stamen_toner|stamen_terrain")
    parser.add_argument("--shape-proximity-km", type=float, default=1.2, help="Max km for shape points to be near origin and dest")
    parser.add_argument("--clip-pct", type=float, default=80.0, help="Clip percentile for preds (0..100) to remove outliers")
    parser.add_argument("--transform", type=str, default="log1p", help="transform to apply to preds (e.g. log1p)")
    parser.add_argument("--stop-filter", default="near_route", help="Which stops to show: bbox|near_route")
    parser.add_argument("--show-stops-km", type=float, default=0.4, help="When showing stops, use this km margin")
    parser.add_argument("--recs-out", default="data/outputs/recommendations.csv", help="Write recommendations CSV for stops along route")
    parser.add_argument("--show-stop-preds", action="store_true", help="Show per-stop prediction markers")
    parser.add_argument("--wheelchair", action="store_true", help="Filter stops to wheelchair-accessible (if column exists)")
    parser.add_argument("--debug", action="store_true", help="Debug prints")
    parser.add_argument("--max-detour-pct", type=float, default=0.30, help="Max detour allowed for alternatives (fraction, e.g. 0.30)")
    parser.add_argument("--color-gamma", type=float, default=0.6, help="Gamma for color mapping (green intensity). 0.6 -> boost green")
    parser.add_argument("--connectivity-weight", type=float, default=0.75, help="Weight for connectivity vs accessibility (0..1)")
    parser.add_argument("--connector-penalty-weight", type=float, default=0.00, help="Penalty weight for origin/dest connector distance (0..1)")
    parser.add_argument("--connector-threshold-m", type=float, default=1000.0, help="Distance (m) above which connector is considered 'far' (for debug/filters)")
    parser.add_argument("--preferred-candidate", type=int, default=None, help="(debug) force preferred candidate index to top")
    parser.add_argument("--stretch-target", type=float, default=0.35, help="Visualization stretch target (p90 -> target). 0.35 is sensible default.")
    parser.add_argument("--stretch-max", type=float, default=80.0, help="Cap for stretch amplification factor (avoid extreme amplification)")
    args = parser.parse_args()

    cache_path = args.cache_path if args.cache_path is not None else "data/outputs/osm_graph.pkl"

    plot_alternatives(args.stops, args.shapes, args.preds, args.out, args.k, args.dist, cache_path,
                      origin=args.origin, dest=args.dest,
                      accessible_only=args.accessible_only, threshold=args.threshold,
                      tiles_provider=args.tiles, shape_proximity_km=args.shape_proximity_km,
                      clip_pct=args.clip_pct, transform=args.transform, stop_filter=args.stop_filter,
                      show_stops_km=args.show_stops_km, recs_out=args.recs_out, show_stop_preds=args.show_stop_preds,
                      wheelchair=args.wheelchair, debug=args.debug, max_detour_pct=args.max_detour_pct,
                      color_gamma=args.color_gamma,
                      connectivity_weight=args.connectivity_weight,
                      connector_penalty_weight=args.connector_penalty_weight,
                      connector_threshold_m=args.connector_threshold_m,
                      preferred_candidate=args.preferred_candidate,
                      stretch_target=args.stretch_target, stretch_max=args.stretch_max)
