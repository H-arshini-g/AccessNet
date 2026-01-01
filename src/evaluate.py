# src/plot_alternatives.py
"""
Plot transit route alternatives with accessibility scores and GTFS shapes.
Updated to:
 - show colored stop markers (calm/moderate/busy) with optional popups
 - include origin & destination markers (flag/play icons)
 - draw GTFS shapes *before* routes so routes appear on top
 - write recommendations.csv for visible stops (includes origin & dest)
 - performance + robustness improvements
"""
import argparse
import os
import math
import pickle
from statistics import mean

# Force non-interactive backend early to avoid slow matplotlib UI backend init
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as colors

import pandas as pd
import folium
import osmnx as ox
import networkx as nx

# utility
def safe_print(*a, **k):
    print(*a, **k, flush=True)

# tile provider helper
def get_tile_layer_spec(provider="osm"):
    if provider in ("carto_light", "carto"):
        url = "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
        opts = {"minZoom": 0, "maxZoom": 20, "attribution": '&copy; OpenStreetMap contributors &copy; CARTO'}
    elif provider in ("stamen_toner", "toner"):
        url = "https://stamen-tiles.a.ssl.fastly.net/toner/{z}/{x}/{y}.png"
        opts = {"minZoom": 0, "maxZoom": 20, "attribution": 'Map tiles by Stamen'}
    elif provider in ("stamen_terrain", "terrain"):
        url = "https://stamen-tiles.a.ssl.fastly.net/terrain/{z}/{x}/{y}.jpg"
        opts = {"minZoom": 0, "maxZoom": 18, "attribution": 'Map tiles by Stamen'}
    else:
        url = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        opts = {"minZoom": 0, "maxZoom": 19, "attribution": '&copy; OpenStreetMap contributors'}
    return url, opts

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1); dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def downsample_coords(coords, max_points=500):
    if len(coords) <= max_points:
        return coords
    step = max(1, int(len(coords) / max_points))
    return coords[::step]

def load_accessibility(preds_file, stops_df, transform=None, clip_pct=95, debug=False):
    """
    Loads predictions in multiple possible formats and returns a dict stop_id -> normalized_score (0..1)
    Supports:
      - long format with columns ['stop_id','pred'] OR ['stop_id','ema_pred']
      - wide last-row format with columns stop_<id>...
    Normalizes after optional transform (e.g. 'log1p').
    """
    safe_print(f"DEBUG: loading preds from {preds_file}")
    preds = pd.read_csv(preds_file)
    # long ema_pred
    if "ema_pred" in preds.columns and "stop_id" in preds.columns:
        grouped = preds.groupby("stop_id")["ema_pred"].mean()
        acc_raw = {str(k): float(v) for k,v in grouped.to_dict().items()}
    # long pred
    elif {"stop_id", "pred"}.issubset(preds.columns):
        grouped = preds.groupby("stop_id")["pred"].mean()
        acc_raw = {str(k): float(v) for k,v in grouped.to_dict().items()}
    else:
        # try wide stop_ columns mapping to stops_df order
        stop_cols = [c for c in preds.columns if str(c).startswith("stop_")]
        if stop_cols:
            last_row = preds.iloc[-1]
            acc_raw = {}
            # Map stops_df order to columns in case they correspond
            for sid, col in zip(stops_df["stop_id"], stop_cols):
                val = last_row[col]
                acc_raw[str(sid)] = float(val) if pd.notna(val) else None
        else:
            raise ValueError("Predictions must contain 'stop_id'/'pred' or 'ema_pred' or wide stop_* columns")

    # collect numeric values for normalization
    vals = [v for v in acc_raw.values() if v is not None and (not pd.isna(v))]
    if not vals:
        safe_print("DEBUG: no numeric preds found; returning zeros")
        return {str(sid): 0.0 for sid in stops_df["stop_id"]}

    # optional percentile clipping (clip_pct is upper percentile; lower percentile = 100-clip_pct)
    lo = min(vals); hi = max(vals)
    if clip_pct and clip_pct < 100:
        lo = float(pd.Series(vals).quantile((100-clip_pct)/100.0))
        hi = float(pd.Series(vals).quantile(clip_pct/100.0))
        if debug: safe_print(f"DEBUG: clipped preds to percentiles -> lo={lo}, hi={hi}")

    def transform_val(x):
        v = x
        if transform == "log1p":
            # ensure non-negative input for log1p by shifting if lo < 0
            shift = 0.0
            if lo < 0:
                shift = -lo
            v = math.log1p(max(0.0, x + shift))
        return v

    transformed = {}
    for k, v in acc_raw.items():
        if v is None or pd.isna(v):
            transformed[k] = None
        else:
            transformed[k] = transform_val(float(v))

    tvs = [t for t in transformed.values() if t is not None]
    if not tvs:
        return {str(sid): 0.0 for sid in stops_df["stop_id"]}

    vmin = min(tvs); vmax = max(tvs)
    if vmax == vmin:
        # everything same -> return 1.0
        return {str(sid): 1.0 for sid in stops_df["stop_id"]}

    normed = {}
    for k, t in transformed.items():
        if t is None:
            normed[k] = 0.0
        else:
            normed[k] = max(0.0, min(1.0, (t - vmin) / (vmax - vmin)))
    safe_print(f"DEBUG: accessibility mapped n={len(normed)} min={min(tvs):.6f} max={max(tvs):.6f} mean={sum(tvs)/len(tvs):.6f}")
    return normed

def simplify_graph_for_routing(G):
    Gs = nx.Graph()
    for u, v, data in G.edges(data=True):
        try:
            length = float(data.get("length", data.get("weight", 1.0)))
        except Exception:
            length = 1.0
        if Gs.has_edge(u, v):
            if length < Gs[u][v]["weight"]:
                Gs[u][v]["weight"] = length
        else:
            Gs.add_edge(u, v, weight=length)
    return Gs

def path_edge_length(G, path):
    if not path or len(path) < 2:
        return 0.0
    total = 0.0
    for u, v in zip(path, path[1:]):
        try:
            ed = G.get_edge_data(u, v)
            if ed is None:
                ed = G.get_edge_data(v, u)
            # MultiGraph -> dict of keys -> data
            if isinstance(ed, dict) and any(isinstance(vv, dict) for vv in ed.values()):
                # ed is {key: {attrs}}
                lens = []
                for kk, vv in ed.items():
                    if isinstance(vv, dict):
                        if "length" in vv: lens.append(float(vv["length"]))
                        elif "weight" in vv: lens.append(float(vv["weight"]))
                if lens:
                    total += min(lens)
                else:
                    total += 1.0
            elif isinstance(ed, dict):
                # single dict
                total += float(ed.get("length", ed.get("weight", 1.0)))
            else:
                total += 1.0
        except Exception:
            total += 1.0
    return total

def compute_alternatives(G, orig_node, dest_node, k=3):
    # convert to simplified Graph for k-shortest path generation
    Gs = simplify_graph_for_routing(G)
    paths = []
    try:
        gen = nx.shortest_simple_paths(Gs, orig_node, dest_node, weight="weight")
        for p in gen:
            paths.append(list(p))
            if len(paths) >= k:
                break
    except Exception:
        # fallback to single shortest path on original graph
        try:
            sp = nx.shortest_path(G, orig_node, dest_node, weight="length")
            paths = [list(sp)]
        except Exception:
            paths = []
    return paths

def stops_near_route(G, route_nodes, stops_df, km_radius=0.5, max_matches=None):
    # route_nodes -> list of node ids
    route_coords = [(G.nodes[n].get("y"), G.nodes[n].get("x")) for n in route_nodes if "y" in G.nodes[n] and "x" in G.nodes[n]]
    if not route_coords:
        return pd.DataFrame([])
    matches = []
    for _, s in stops_df.iterrows():
        try:
            slat = float(s["stop_lat"]); slon = float(s["stop_lon"])
        except Exception:
            continue
        mind = min(haversine_km(slat, slon, rlat, rlon) for (rlat, rlon) in route_coords)
        if mind <= km_radius:
            matches.append((mind, s))
    matches.sort(key=lambda x: x[0])
    rows = [m[1] for m in matches]
    df = pd.DataFrame(rows)
    if max_matches:
        df = df.head(max_matches)
    return df

def shapes_near_bbox(shapes_df, bbox):
    lat_min, lat_max, lon_min, lon_max = bbox
    cond = (shapes_df["shape_pt_lat"].astype(float) >= lat_min) & (shapes_df["shape_pt_lat"].astype(float) <= lat_max) & \
           (shapes_df["shape_pt_lon"].astype(float) >= lon_min) & (shapes_df["shape_pt_lon"].astype(float) <= lon_max)
    return set(shapes_df.loc[cond, "shape_id"].unique().tolist())

def path_accessibility_score(path_nodes, G, stops_df, acc_map):
    stop_lats = stops_df["stop_lat"].astype(float).values
    stop_lons = stops_df["stop_lon"].astype(float).values
    stop_ids = stops_df["stop_id"].astype(str).values.tolist()
    matched = []
    for n in path_nodes:
        node = G.nodes.get(n, {})
        node_y = node.get("y", None); node_x = node.get("x", None)
        if node_y is None or node_x is None:
            continue
        diffs = (abs(stop_lats - node_y) + abs(stop_lons - node_x))
        try:
            idx = int(diffs.argmin())
            matched.append(stop_ids[idx])
        except Exception:
            continue
    if not matched:
        return 0.0
    unique = list(dict.fromkeys(matched))
    scores = [acc_map.get(sid, 0.0) for sid in unique]
    return mean(scores) if scores else 0.0

def plot_alternatives(stops_file, shapes_file, preds_file, out_html,
                      k=3, dist=800, cache="data/outputs/osm_graph.pkl",
                      origin=None, dest=None,
                      tiles="osm", shape_proximity_km=3.0,
                      clip_pct=95, transform=None,
                      stop_filter="none", show_stops_km=0.6,
                      recs_out=None, show_stop_preds=False,
                      debug=False, max_detour_pct=0.25):
    safe_print("Loading stops and shapes...")
    stops = pd.read_csv(stops_file, dtype=str)
    shapes = pd.read_csv(shapes_file) if shapes_file and os.path.exists(shapes_file) else None

    if origin is None or dest is None:
        origin = stops.iloc[0]["stop_id"]
        dest = stops.iloc[-1]["stop_id"]
    if origin not in stops["stop_id"].values or dest not in stops["stop_id"].values:
        raise ValueError("Origin or destination not found in stops file")
    o_row = stops[stops["stop_id"] == origin].iloc[0]
    d_row = stops[stops["stop_id"] == dest].iloc[0]
    o_lat, o_lon = float(o_row["stop_lat"]), float(o_row["stop_lon"])
    d_lat, d_lon = float(d_row["stop_lat"]), float(d_row["stop_lon"])

    # load and normalize accessibility preds
    acc_map = load_accessibility(preds_file, stops, transform=transform, clip_pct=clip_pct, debug=debug)

    # map center roughly at stops mean (keeps map centered on city)
    center_lat = float(stops["stop_lat"].astype(float).mean())
    center_lon = float(stops["stop_lon"].astype(float).mean())

    # folium map + tile layer
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles=None)
    tile_url, tile_opts = get_tile_layer_spec(tiles)
    folium.TileLayer(tiles=tile_url, attr=tile_opts.get("attribution", ""), name=tiles, max_zoom=tile_opts.get("maxZoom", 19)).add_to(m)

    # OSM graph cache load / build
    G = None
    if cache and os.path.exists(cache):
        try:
            safe_print(f"Loading cached OSM graph from {cache} ...")
            with open(cache, "rb") as f:
                G = pickle.load(f)
        except Exception:
            safe_print("Cache exists but failed to load; will rebuild graph.")
            G = None

    if G is None:
        mid_lat = (o_lat + d_lat) / 2.0
        mid_lon = (o_lon + d_lon) / 2.0
        od_km = haversine_km(o_lat, o_lon, d_lat, d_lon)
        radius_m = max(dist, int((od_km * 1000) / 2 + 1500))
        safe_print(f"DEBUG: building graph with center {mid_lat:.6f},{mid_lon:.6f} dist={radius_m}m (od_km={od_km:.3f})")
        G = ox.graph_from_point((mid_lat, mid_lon), dist=radius_m, network_type="walk", simplify=True)
        if cache:
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            with open(cache, "wb") as f:
                pickle.dump(G, f)
            safe_print(f"DEBUG: saved graph cache to {cache}")

    # nearest nodes for origin/destination (osmnx supports two arg forms across versions)
    try:
        orig_node = ox.distance.nearest_nodes(G, o_lon, o_lat)
        dest_node = ox.distance.nearest_nodes(G, d_lon, d_lat)
    except Exception:
        try:
            # fallback: pass tuple (lat,lon) if older osmnx expects that
            orig_node = ox.distance.nearest_nodes(G, (o_lat, o_lon))
            dest_node = ox.distance.nearest_nodes(G, (d_lat, d_lon))
        except Exception as e:
            raise RuntimeError("Could not find nearest OSM nodes for origin/destination") from e

    # debug nearest distances
    try:
        on = G.nodes[orig_node]; dn = G.nodes[dest_node]
        origin_node_dist_km = haversine_km(o_lat, o_lon, on.get("y"), on.get("x"))
        dest_node_dist_km = haversine_km(d_lat, d_lon, dn.get("y"), dn.get("x"))
        safe_print(f"DEBUG: nearest nodes distances: origin {origin_node_dist_km:.3f} km dest {dest_node_dist_km:.3f} km")
    except Exception:
        pass

    # compute alternative paths
    raw_paths = compute_alternatives(G, orig_node, dest_node, k=k)
    if not raw_paths:
        safe_print("No path(s) found between origin and destination in graph.")
        raw_paths = []
    path_lengths = [path_edge_length(G, p) for p in raw_paths]
    if debug:
        safe_print("DEBUG: Found", len(raw_paths), "path(s). path_lengths:", path_lengths)

    # compute accessibility score per path
    path_scores = [path_accessibility_score(p, G, stops, acc_map) for p in raw_paths]
    if debug:
        safe_print("DEBUG: path_scores:", path_scores)

    # rank by score
    ranked = sorted(list(zip(raw_paths, path_scores, path_lengths)), key=lambda x: x[1], reverse=True)
    draw_paths = [r[0] for r in ranked]
    draw_scores = [r[1] for r in ranked]

    # prune by detour percent
    if draw_paths and max_detour_pct and path_lengths:
        best_len = path_lengths[0] if path_lengths else (path_edge_length(G, draw_paths[0]) if draw_paths else None)
        if best_len and best_len > 0:
            kept = []
            for p, s, L in ranked:
                if L is None:
                    kept.append((p, s, L))
                else:
                    if (L - best_len) / best_len <= max_detour_pct:
                        kept.append((p, s, L))
            if kept:
                draw_paths = [p for p, s, L in kept]
                draw_scores = [s for p, s, L in kept]

    # slice stops between origin & destination by index to be conservative
    origin_idx = stops[stops["stop_id"] == origin].index[0]
    dest_idx = stops[stops["stop_id"] == dest].index[0]
    slice_start, slice_end = min(origin_idx, dest_idx), max(origin_idx, dest_idx)
    stops_slice = stops.iloc[slice_start:slice_end+1].reset_index(drop=True)

    # prepare colormap and normalization once (avoid repeated expensive calls)
    max_val = 1.0  # acc_map is already normalized 0..1
    norm = colors.Normalize(vmin=0.0, vmax=max_val)
    cmap_line = cm.get_cmap("RdYlGn")  # gradient for route color (green->red)

    # decide which stops to draw (near routes or bbox or none)
    stops_to_draw_df = stops_slice
    if stop_filter == "near_route" and draw_paths:
        all_route_nodes = []
        for p in draw_paths:
            all_route_nodes.extend(p)
        stops_near = stops_near_route(G, all_route_nodes, stops_slice, km_radius=show_stops_km)
        if not stops_near.empty:
            stops_to_draw_df = stops_near
    elif stop_filter == "bbox" and shapes is not None:
        lat_min = min(o_lat, d_lat) - (show_stops_km / 111.0)
        lat_max = max(o_lat, d_lat) + (show_stops_km / 111.0)
        lon_min = min(o_lon, d_lon) - (show_stops_km / 111.0)
        lon_max = max(o_lon, d_lon) + (show_stops_km / 111.0)
        cond = (stops_slice["stop_lat"].astype(float) >= lat_min) & (stops_slice["stop_lat"].astype(float) <= lat_max) & \
               (stops_slice["stop_lon"].astype(float) >= lon_min) & (stops_slice["stop_lon"].astype(float) <= lon_max)
        stops_to_draw_df = stops_slice.loc[cond]

    # draw stops (colored markers with white border)
    for _, row in stops_to_draw_df.iterrows():
        sid = str(row["stop_id"])
        dval = acc_map.get(sid, 0.0)
        # classification thresholds (tunable)
        if dval >= 0.66:
            level = "Calm / High accessibility"
            color = "#2ecc71"   # green
        elif dval >= 0.33:
            level = "Moderate accessibility"
            color = "#f39c12"   # orange
        else:
            level = "Busy / Low accessibility"
            color = "#e74c3c"   # red

        popup = f"<b>{row.get('stop_name', sid)}</b><br>{level}<br>Score: {dval:.3f}" if show_stop_preds else None
        folium.CircleMarker(
            location=[float(row["stop_lat"]), float(row["stop_lon"])],
            radius=6, color="white", weight=1.5,
            fill=True, fill_color=color, fill_opacity=0.95,
            popup=popup
        ).add_to(m)

    # --- DRAW SHAPES FIRST (so they don't obscure colored path) ---
    if shapes is not None:
        safe_print("Processing GTFS shapes with stricter proximity filter (shapes drawn before routes)...")
        dist_km = max(1.0, shape_proximity_km)
        delta_deg = dist_km / 111.0
        lat_min = min(o_lat, d_lat) - delta_deg
        lat_max = max(o_lat, d_lat) + delta_deg
        lon_min = min(o_lon, d_lon) - delta_deg
        lon_max = max(o_lon, d_lon) + delta_deg
        bbox = (lat_min, lat_max, lon_min, lon_max)

        nearby_shape_ids = shapes_near_bbox(shapes, bbox)
        if nearby_shape_ids:
            small_shapes = shapes[shapes["shape_id"].isin(nearby_shape_ids)].copy()
            grouped = small_shapes.groupby("shape_id") if "shape_pt_sequence" not in small_shapes.columns else small_shapes.sort_values("shape_pt_sequence").groupby("shape_id")
            drawn = 0
            for shape_id, group in grouped:
                group = group.sort_values("shape_pt_sequence").reset_index(drop=True) if "shape_pt_sequence" in group.columns else group.reset_index(drop=True)
                lats = group["shape_pt_lat"].astype(float).values
                lons = group["shape_pt_lon"].astype(float).values
                if len(lats) < 2:
                    continue
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
                    folium.PolyLine(seg_ds, color="#2c3e91", weight=4, opacity=0.45).add_to(m)
                    drawn += 1
            if drawn == 0:
                safe_print("No shapes matched strict proximity; falling back to drawing nearby shape segments (looser filter).")
                for shape_id, group in grouped:
                    group = group.sort_values("shape_pt_sequence").reset_index(drop=True) if "shape_pt_sequence" in group.columns else group.reset_index(drop=True)
                    lats = group["shape_pt_lat"].astype(float).values
                    lons = group["shape_pt_lon"].astype(float).values
                    if len(lats) < 2: continue
                    pts = list(zip(lats, lons))
                    def sq(a,b): return (a[0]-b[0])**2 + (a[1]-b[1])**2
                    o_idx = min(range(len(pts)), key=lambda i: sq(pts[i], (o_lat, o_lon)))
                    d_idx = min(range(len(pts)), key=lambda i: sq(pts[i], (d_lat, d_lon)))
                    s_i, e_i = min(o_idx, d_idx), max(o_idx, d_idx)
                    seg = pts[s_i:e_i+1]
                    seg_ds = downsample_coords(seg, max_points=500)
                    folium.PolyLine(seg_ds, color="#2c3e91", weight=3, opacity=0.35).add_to(m)
                safe_print("Fallback shapes drawn (nearby).")
            else:
                safe_print(f"Drawn {drawn} shapes near both origin & destination (<= {shape_proximity_km} km).")
        else:
            safe_print("No shapes in bbox; skipping shapes overlay.")

    # --- DRAW PATHS AFTER SHAPES (so routes appear on top)
    for i, path in enumerate(draw_paths):
        score = draw_scores[i] if i < len(draw_scores) else 0.0
        coords = []
        for n in path:
            node = G.nodes.get(n, {})
            if "y" in node and "x" in node:
                coords.append((node["y"], node["x"]))
        if not coords:
            continue
        hexc = colors.to_hex(cmap_line(score))
        if i == 0:
            # outline + colored line
            folium.PolyLine(coords, color="#0f1724", weight=10, opacity=0.18).add_to(m)
            folium.PolyLine(coords, color=hexc, weight=8, opacity=0.95, popup=f"Primary route (score {score:.3f})").add_to(m)
        else:
            folium.PolyLine(coords, color=hexc, weight=5, opacity=0.75, dash_array="6,6", popup=f"Alt route (score {score:.3f})").add_to(m)

    # origin/dest markers (no popup text requested)
    folium.CircleMarker(location=[o_lat, o_lon], radius=14, color=None, fill=True, fill_color="#1f77b4", fill_opacity=0.22).add_to(m)
    folium.Marker(location=[o_lat, o_lon], icon=folium.Icon(color="blue", icon="flag", prefix="fa")).add_to(m)
    folium.CircleMarker(location=[d_lat, d_lon], radius=16, color=None, fill=True, fill_color="#e74c3c", fill_opacity=0.18).add_to(m)
    folium.Marker(location=[d_lat, d_lon], icon=folium.Icon(color="red", icon="play", prefix="fa")).add_to(m)

    # legend + link to recs file if present
    link_html = f'<a href="{os.path.basename(recs_out)}" target="_blank">Download recommendations.csv</a>' if recs_out else ""
    legend = f"""
    <div style="position: fixed; bottom: 40px; left: 40px; width: 340px;
                background-color: white; border:2px solid gray; border-radius:6px;
                z-index:9999; font-size:14px; padding:10px;">
      <b>Legend</b><br>
      <span style='color:#2ecc71;'>&#9632;</span> High accessibility<br>
      <span style='color:#f39c12;'>&#9632;</span> Medium accessibility<br>
      <span style='color:#e74c3c;'>&#9632;</span> Low accessibility<br>
      <br><b>Paths</b><br>
      <span style='font-weight:700;color:#2ecc71;'>●</span> Best accessible route (green → red)<br>
      <span style='color:gray;'>—</span> Other routes (dashed=lower accessibility)<br>
      <br>{link_html}
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend))

    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    m.save(out_html)
    safe_print(f"Map saved to {out_html}")

    # write recommendations for visible stops (include origin/dest)
    if recs_out:
        all_recs = []
        if draw_paths:
            primary = draw_paths[0]
            matched_stops = stops_near_route(G, primary, stops, km_radius=show_stops_km)
            # ensure origin & dest included
            if origin not in matched_stops["stop_id"].astype(str).values:
                matched_stops = pd.concat([matched_stops, stops[stops["stop_id"] == origin]], ignore_index=True)
            if dest not in matched_stops["stop_id"].astype(str).values:
                matched_stops = pd.concat([matched_stops, stops[stops["stop_id"] == dest]], ignore_index=True)
            matched_stops = matched_stops.drop_duplicates(subset=["stop_id"])
            for _, r in matched_stops.iterrows():
                sid = str(r["stop_id"])
                pred = float(acc_map.get(sid, 0.0))
                if pred >= 0.66:
                    rec = "Good: maintain accessibility"
                elif pred >= 0.33:
                    rec = "Moderate: signage, boarding assistance"
                else:
                    rec = "Busy: consider ramps, lifts, wider doors, priority boarding"
                all_recs.append({"stop_id": sid, "pred": pred, "rec": rec, "stop_name": r.get("stop_name", "")})
        if not all_recs:
            # fallback include origin/dest
            for s in [origin, dest]:
                r = stops[stops["stop_id"] == s].iloc[0]
                pred = float(acc_map.get(str(s), 0.0))
                rec = "Good: maintain accessibility" if pred >= 0.66 else ("Moderate: signage, boarding assistance" if pred >= 0.33 else "Busy: consider ramps, lifts, wider doors, priority boarding")
                all_recs.append({"stop_id": str(s), "pred": pred, "rec": rec, "stop_name": r.get("stop_name", "")})
        recs_df = pd.DataFrame(all_recs)
        recs_df.to_csv(recs_out, index=False)
        safe_print(f"DEBUG: Writing recommendations to {recs_out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stops", required=True)
    p.add_argument("--shapes", required=False)
    p.add_argument("--preds", required=True)
    p.add_argument("--out", default="data/outputs/map_alternatives.html")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--dist", type=int, default=800)
    p.add_argument("--cache", default="data/outputs/osm_graph.pkl")
    p.add_argument("--origin", required=True)
    p.add_argument("--dest", required=True)
    p.add_argument("--tiles", default="osm", help="osm|carto_light|stamen_toner|stamen_terrain")
    p.add_argument("--shape-proximity-km", type=float, default=3.0)
    p.add_argument("--clip-pct", type=float, default=95.0)
    p.add_argument("--transform", type=str, default=None, help="log1p or None")
    p.add_argument("--stop-filter", type=str, default="none", choices=["none", "near_route", "bbox"])
    p.add_argument("--show-stops-km", type=float, default=0.6)
    p.add_argument("--recs-out", type=str, default=None)
    p.add_argument("--show-stop-preds", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--max-detour-pct", type=float, default=0.25)
    args = p.parse_args()

    plot_alternatives(
        stops_file=args.stops,
        shapes_file=args.shapes,
        preds_file=args.preds,
        out_html=args.out,
        k=args.k, dist=args.dist, cache=args.cache,
        origin=args.origin, dest=args.dest,
        tiles=args.tiles, shape_proximity_km=args.shape_proximity_km,
        clip_pct=args.clip_pct, transform=args.transform,
        stop_filter=args.stop_filter, show_stops_km=args.show_stops_km,
        recs_out=args.recs_out, show_stop_preds=args.show_stop_preds,
        debug=args.debug, max_detour_pct=args.max_detour_pct
    )
