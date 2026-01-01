# src/app.py
import os
import sys
import subprocess
import time
from pathlib import Path
from flask import (
    Flask, render_template_string, request, redirect, url_for,
    send_from_directory, flash
)
import pandas as pd

BASE = Path(__file__).resolve().parents[1]  # project root
GTFS_STOPS = BASE / "data" / "gtfs" / "stops.txt"
DEFAULT_PREDS = BASE / "data" / "outputs" / "lstm_preds_mapped.csv"
OUT_MAP = BASE / "data" / "outputs" / "map_app.html"
PLOT_SCRIPT = BASE / "src" / "plot_alternatives.py"

app = Flask(__name__)
app.secret_key = "dev-secret"

# Template: updated to include stretch controls
TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Accessible Route Finder — Demo</title>
  <link href="https://cdnjs.cloudflare.com/ajax/libs/bootswatch/5.3.0/flatly/bootstrap.min.css" rel="stylesheet">
  <style>
    body { background: #f5f8fb; }
    .sidebar { width: 360px; padding: 20px; }
    .map-area { flex: 1; padding: 18px; position:relative; }
    .container-flex { display:flex; gap:16px; align-items:stretch; min-height:78vh; }
    .card { border-radius: 12px; box-shadow: 0 6px 18px rgba(12,30,60,0.07); }
    iframe { width:100%; height:75vh; border-radius:10px; border:1px solid rgba(0,0,0,0.06); }
    .compact-legend { position: absolute; bottom: 16px; right: 16px; z-index:9999; background: rgba(255,255,255,0.95); padding:8px; border-radius:8px; box-shadow:0 6px 18px rgba(0,0,0,0.06); font-size:13px; }
  </style>
</head>
<body>
  <nav class="navbar navbar-expand-lg navbar-light bg-white card" style="margin:12px;border-radius:12px;">
    <div class="container-fluid">
      <a class="navbar-brand" href="#" style="font-weight:700;color:#1f5fa6">Accessible Route Finder</a>
    </div>
  </nav>

  <div class="container-flex" style="margin:12px 18px;">
    <div class="sidebar card">
      <h5>Route controls</h5>
      <form id="route-form" method="post" action="{{ url_for('show') }}">
        <div class="mb-3">
          <label class="form-label">Origin (type stop id OR name)</label>
          <input name="origin" class="form-control" list="stops-list" placeholder="Type stop id or name" value="{{ default_origin }}">
        </div>

        <div class="mb-3">
          <label class="form-label">Destination (type stop id OR name)</label>
          <input name="dest" class="form-control" list="stops-list" placeholder="Type stop id or name" value="{{ default_dest }}">
        </div>

        <datalist id="stops-list">
          {% for sid, sname in stops %}
            <option value="{{ sname if sname else sid }}">{{ sid }}{% if sname %} — {{ sname }}{% endif %}</option>
          {% endfor %}
        </datalist>

        <div class="mb-3">
          <label class="form-label">Accessibility threshold <span id="thval" style="font-weight:700">{{ threshold }}</span></label>
          <input name="threshold" id="threshold" type="range" min="0" max="1" step="0.01" value="{{ threshold }}" class="form-range" oninput="document.getElementById('thval').innerText = this.value;">
        </div>

        <div class="form-check mb-3">
          <input class="form-check-input" type="checkbox" name="accessible_only" id="accessible_only" {% if accessible_only %}checked{% endif %}>
          <label class="form-check-label" for="accessible_only">Wheelchair only (filter stops to wheelchair_boarding)</label>
        </div>

        <div class="form-check mb-3">
          <input class="form-check-input" type="checkbox" name="show_stop_preds" id="show_stop_preds" {% if show_stop_preds %}checked{% else %}checked{% endif %}>
          <label class="form-check-label" for="show_stop_preds">Show stop markers (busyness dots)</label>
        </div>

        <div class="mb-3">
          <label class="form-label">Color gamma (green intensity) <span id="gval" style="font-weight:700">{{ color_gamma }}</span></label>
          <input name="color_gamma" id="color_gamma" type="range" min="0.2" max="2.5" step="0.01" value="{{ color_gamma }}" class="form-range" oninput="document.getElementById('gval').innerText = this.value;">
        </div>

        <div class="mb-3">
          <label class="form-label">Visualization stretch target (p90) <span id="stval" style="font-weight:700">{{ stretch_target }}</span></label>
          <input name="stretch_target" id="stretch_target" type="range" min="0.05" max="0.8" step="0.01" value="{{ stretch_target }}" class="form-range" oninput="document.getElementById('stval').innerText = this.value;">
          <div class="small-note">Controls auto-amplification of small accessibility values for color mapping.</div>
        </div>

        <div class="mb-3">
          <label class="form-label">Stretch cap (max factor) <span id="scval" style="font-weight:700">{{ stretch_max }}</span></label>
          <input name="stretch_max" id="stretch_max" type="range" min="1" max="200" step="1" value="{{ stretch_max }}" class="form-range" oninput="document.getElementById('scval').innerText = this.value;">
        </div>

        <div class="d-flex gap-2 mb-2">
          <button class="btn btn-primary" id="show-btn" type="submit">Show routes</button>
          <a href="{{ url_for('serve_map') }}" target="_blank" class="btn btn-outline-primary">Open full map</a>
        </div>

        {% with messages = get_flashed_messages() %}
          {% if messages %}
            <div class="alert alert-warning mt-2">
              {% for msg in messages %}{{ msg }}<br>{% endfor %}
            </div>
          {% endif %}
        {% endwith %}
      </form>
    </div>

    <div class="map-area card">
      {% if map_ready %}
        <iframe src="{{ url_for('serve_map') }}?ts={{ timestamp }}" frameborder="0" id="map-frame"></iframe>
      {% else %}
        <div style="padding:40px;text-align:center;color:#666">
          <h4>Interactive map</h4>
          <p style="max-width:560px;margin:12px auto">Use the controls on the left to compute accessible routes. When ready the map will appear here.</p>
        </div>
      {% endif %}
      <div class="compact-legend">
        <div style="font-weight:700;margin-bottom:6px">Legend</div>
        <div><span style="color:#2ecc71">&#9632;</span> High</div>
        <div><span style="color:#f39c12">&#9632;</span> Medium</div>
        <div><span style="color:#e74c3c">&#9632;</span> Low</div>
      </div>
    </div>
  </div>

  <footer class="text-center small" style="margin:12px;">
    <div class="card" style="padding:10px;border-radius:10px;">
      Built for Enhancing public transport accessibility — demo.
    </div>
  </footer>

  <script>
    const form = document.getElementById("route-form");
    form?.addEventListener('submit', () => {
      document.getElementById('show-btn').disabled = true;
    });
  </script>
</body>
</html>
"""

def load_stops_list():
    if not GTFS_STOPS.exists():
        return []
    try:
        df = pd.read_csv(GTFS_STOPS, dtype=str)
        ids = df['stop_id'].astype(str).tolist()
        names = df['stop_name'].astype(str).tolist() if 'stop_name' in df.columns else ['']*len(ids)
        return list(zip(ids, names))
    except Exception:
        return []

@app.route("/", methods=["GET"])
def index():
    stops = load_stops_list()
    default_origin = stops[0][1] if stops and stops[0][1] else (stops[0][0] if stops else "")
    default_dest = stops[-1][1] if stops and stops[-1][1] else (stops[-1][0] if stops else "")
    # sensible UI defaults to match earlier best CLI
    return render_template_string(TEMPLATE,
                                  stops=stops,
                                  default_origin=default_origin,
                                  default_dest=default_dest,
                                  threshold=0.5,
                                  accessible_only=False,
                                  map_ready=False,
                                  timestamp=int(time.time()),
                                  show_stop_preds=True,
                                  color_gamma=0.6,
                                  stretch_target=0.35,
                                  stretch_max=80)

@app.route("/show", methods=["POST"])
def show():
    origin = request.form.get("origin", "").strip()
    dest = request.form.get("dest", "").strip()
    threshold = request.form.get("threshold", "0.50")
    accessible_only = True if request.form.get("accessible_only") else False
    show_stop_preds = True if request.form.get("show_stop_preds") else False
    color_gamma = request.form.get("color_gamma", "0.6")
    stretch_target = request.form.get("stretch_target", "0.35")
    stretch_max = request.form.get("stretch_max", "80")

    stops = load_stops_list()

    if origin == "" or dest == "":
        flash("Please provide both origin and destination (type stop id or name).")
        return redirect(url_for("index"))

    # resolve typed name -> close match (prefer exact name match)
    resolved_origin = origin
    resolved_dest = dest

    lower_names = {s.lower(): sid for sid, s in stops if s}
    lower_ids = {sid.lower(): sid for sid in [sid for sid, _ in stops]}
    if origin.lower() in lower_names:
        resolved_origin = origin
    elif origin.lower() in lower_ids:
        resolved_origin = lower_ids[origin.lower()]
    else:
        matches = [s for sid, s in stops if s and origin.lower() in s.lower()]
        if matches:
            resolved_origin = matches[0]
        else:
            flash("Origin not found in stops list. Use the datalist or enter an exact stop name/id.")
            return redirect(url_for("index"))

    if dest.lower() in lower_names:
        resolved_dest = dest
    elif dest.lower() in lower_ids:
        resolved_dest = lower_ids[dest.lower()]
    else:
        matches = [s for sid, s in stops if s and dest.lower() in s.lower()]
        if matches:
            resolved_dest = matches[0]
        else:
            flash("Destination not found in stops list. Use the datalist or enter an exact stop name/id.")
            return redirect(url_for("index"))

    cmd = [
        sys.executable,
        str(PLOT_SCRIPT),
        "--stops", str(GTFS_STOPS),
        "--shapes", str(BASE / "data" / "gtfs" / "shapes_filtered.txt"),
        "--preds", str(DEFAULT_PREDS),
        "--out", str(OUT_MAP),
        "--origin", str(resolved_origin),
        "--dest", str(resolved_dest),
        "--k", "12",
        "--dist", "2500",
        "--tiles", "osm",
        "--shape-proximity-km", "1.2",
        "--clip-pct", "80",
        "--transform", "log1p",
        "--stop-filter", "near_route",
        "--show-stops-km", "0.4",
        "--recs-out", str(BASE / "data" / "outputs" / "recommendations.csv"),
        "--max-detour-pct", "0.30",
        "--color-gamma", str(color_gamma),
        "--connectivity-weight", "0.75",
        "--connector-penalty-weight", "0.00",
        "--cache", str(BASE / "data" / "outputs" / "osm_graph.pkl"),
        "--stretch-target", str(stretch_target),
        "--stretch-max", str(stretch_max)
    ]
    if accessible_only:
        cmd.append("--wheelchair")
    if show_stop_preds:
        cmd.append("--show-stop-preds")
    # include debug by default to get console info (optional: remove for production)
    cmd.append("--debug")

    try:
        subprocess.check_call(cmd, cwd=str(BASE))
    except subprocess.CalledProcessError as e:
        flash(f"Plotting failed (exit {e.returncode}). See server console for details.")
        return redirect(url_for("index"))
    except FileNotFoundError:
        flash("Plotting script not found. Please ensure src/plot_alternatives.py exists.")
        return redirect(url_for("index"))
    except Exception as ex:
        flash(f"Unexpected error while running plot script: {ex}")
        return redirect(url_for("index"))

    ts = int(time.time())
    return redirect(url_for("index_with_map", ts=ts, origin=resolved_origin, dest=resolved_dest))

@app.route("/index_with_map")
def index_with_map():
    stops = load_stops_list()
    default_origin = request.args.get("origin", stops[0][1] if stops and stops[0][1] else (stops[0][0] if stops else ""))
    default_dest = request.args.get("dest", stops[-1][1] if stops and stops[-1][1] else (stops[-1][0] if stops else ""))
    ts = request.args.get("ts", int(time.time()))
    return render_template_string(TEMPLATE,
                                  stops=stops,
                                  default_origin=default_origin,
                                  default_dest=default_dest,
                                  threshold=0.5,
                                  accessible_only=False,
                                  map_ready=True,
                                  timestamp=ts,
                                  show_stop_preds=True,
                                  color_gamma=0.6,
                                  stretch_target=0.35,
                                  stretch_max=80)

@app.route("/map")
def serve_map():
    out_dir = OUT_MAP.parent
    if not OUT_MAP.exists():
        return "Map not found. Generate one from the controls first.", 404
    return send_from_directory(str(out_dir), OUT_MAP.name)

@app.route("/data/outputs/recommendations.csv")
def download_recommendations():
    rec = BASE / "data" / "outputs" / "recommendations.csv"
    if not rec.exists():
        return "No recommendations found", 404
    return send_from_directory(str(rec.parent), rec.name, as_attachment=True)

if __name__ == "__main__":
    os.makedirs(OUT_MAP.parent, exist_ok=True)
    print("Starting web app. Open http://127.0.0.1:5000/")
    app.run(debug=True, port=5000)
