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
DEFAULT_PREDS = BASE / "data" / "outputs" / "lstm_preds.csv"
DEFAULT_PREDS_ALT = BASE / "data" / "outputs" / "lstm_preds_mapped.csv"
DEFAULT_TICKETS = BASE / "data" / "tickets" / "tickets.csv"
OUT_MAP = BASE / "data" / "outputs" / "map_app.html"
PLOT_SCRIPT = BASE / "src" / "plot_alternatives.py"

app = Flask(__name__)
app.secret_key = "dev-secret"

# Template: modernized with animated background, glass cards, top progress bar, dark mode toggle
TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Accessible Route Finder — Demo</title>

  <!-- Bootswatch CDN (keeps existing look) -->
  <link href="https://cdnjs.cloudflare.com/ajax/libs/bootswatch/5.3.0/flatly/bootstrap.min.css" rel="stylesheet">

  <style>
    /* Animated gradient background */
    :root {
      --card-bg: rgba(255,255,255,0.82);
      --card-border: rgba(20,40,80,0.06);
      --accent: #1f7be1;
      --muted: #6b7280;
    }
    [data-theme="dark"] {
      --card-bg: rgba(18,24,36,0.6);
      --card-border: rgba(255,255,255,0.06);
      --accent: #60a5fa;
      --muted: #9aa6bf;
    }

    body {
      margin: 0; font-family: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
      min-height: 100vh;
      background: linear-gradient(120deg, rgba(18,72,180,0.08), rgba(18,200,180,0.03));
      background-size: 400% 400%;
      animation: bgShift 18s ease-in-out infinite;
      color: #222;
      transition: background 0.25s ease;
    }
    @keyframes bgShift {
      0% { background-position: 0% 50%; }
      50% { background-position: 100% 50%; }
      100% { background-position: 0% 50%; }
    }

    /* Container layout */
    .container-flex { display:flex; gap:20px; align-items:stretch; min-height:78vh; padding:18px; }
    .sidebar { width: 360px; padding:20px; border-radius:12px; background:var(--card-bg); border:1px solid var(--card-border); box-shadow: 0 12px 30px rgba(12,24,48,0.06); }
    .map-area { flex:1; padding:18px; border-radius:12px; background:var(--card-bg); border:1px solid var(--card-border); box-shadow: 0 12px 30px rgba(12,24,48,0.06); position:relative; overflow:hidden; }
    .card-compact { border-radius:10px; padding:12px; background:transparent; }

    nav.navbar {
      margin: 12px; border-radius:14px; padding:10px 20px;
      background: linear-gradient(90deg, var(--accent), #3db5ff);
      color: #fff; box-shadow: 0 10px 30px rgba(20,40,80,0.08);
    }
    nav .navbar-brand { font-weight:700; color: #fff; }

    h5 { margin-top:0; font-weight:700; color: #16325c; }
    [data-theme="dark"] h5 { color: #e6eefc; }

    /* Inputs micro-motion on focus */
    input[type="text"], .form-range {
      transition: box-shadow .18s ease, transform .12s ease;
    }
    input[type="text"]:focus, input[type="text"].focus {
      box-shadow: 0 8px 20px rgba(31,123,225,0.16);
      transform: translateY(-1px);
      outline: none;
      border-color: rgba(31,123,225,0.6);
    }

    /* Buttons */
    .btn-primary {
      background: linear-gradient(180deg, #2b8af6, #1868d4);
      border: none;
      box-shadow: 0 8px 18px rgba(27,90,191,0.12);
      transition: transform .12s ease, box-shadow .12s ease;
    }
    .btn-primary:hover { transform: translateY(-2px); box-shadow: 0 14px 28px rgba(27,90,191,0.16); }

    /* Loader / traffic & progress bar */
    #top-progress {
      position: fixed; top: 0; left: 0; right: 0; height: 5px; z-index: 99999;
      background: linear-gradient(90deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
      display:none;
    }
    #top-progress .bar {
      height:100%; width:0%; background: linear-gradient(90deg, #60c470, #f7b267);
      box-shadow: 0 3px 10px rgba(0,0,0,0.08);
      transition: width 0.45s cubic-bezier(.22,.9,.32,1);
    }

    .loader-overlay {
      position:absolute; inset:0; display:none; align-items:center; justify-content:center; z-index: 9998;
      background: linear-gradient(180deg, rgba(255,255,255,0.55), rgba(255,255,255,0.45));
      backdrop-filter: blur(3px);
    }
    [data-theme="dark"] .loader-overlay {
      background: linear-gradient(180deg, rgba(6,12,22,0.55), rgba(6,12,22,0.45));
    }

    .traffic {
      width:72px; background: rgba(0,0,0,0.08); padding:10px; border-radius:12px; display:flex; flex-direction:column; align-items:center;
      box-shadow: 0 10px 30px rgba(10,20,40,0.06);
    }
    .traffic .light { width:36px; height:36px; border-radius:50%; background:#333; margin:6px 0; opacity:0.28; transition: all .18s ease; }
    .traffic .light.red.on { background:#e9573f; opacity:1; box-shadow:0 6px 16px rgba(233,87,63,0.22); }
    .traffic .light.amber.on { background:#f6b042; opacity:1; box-shadow:0 6px 16px rgba(246,176,66,0.18); }
    .traffic .light.green.on { background:#37c47c; opacity:1; box-shadow:0 6px 16px rgba(55,196,124,0.2); }

    /* Hide the small map legend from the folium map with CSS override inside iframe (can't access iframe DOM directly),
       but we can overlay a cover at the bottom-right of the map area so the small legend is visually hidden here. */
    .legend-cover {
      position:absolute; right: 18px; bottom: 18px; width: 120px; height: 120px; pointer-events:none; z-index:9997;
      /* transparent so map still visible under it; small legend will be covered visually */
      background: linear-gradient(180deg, rgba(255,255,255,0.00), rgba(255,255,255,0.00));
    }
    [data-theme="dark"] .legend-cover { background: linear-gradient(180deg, rgba(0,0,0,0.00), rgba(0,0,0,0.00)); }

    /* footer */
    footer { margin: 12px; padding:10px 14px; border-radius:10px; background:var(--card-bg); border:1px solid var(--card-border); text-align:center; }

    /* compact legend in sidebar */
    .sidebar .compact-legend-item { display:flex; gap:10px; align-items:center; margin-top:10px; }
    .legend-dot { width:12px; height:12px; border-radius:3px; display:inline-block; }
    .muted { color: var(--muted); }

    /* dark-mode toggle */
    .theme-toggle { cursor:pointer; padding:8px 10px; border-radius:8px; background: rgba(255,255,255,0.06); display:inline-flex; align-items:center; gap:8px; }
    [data-theme="dark"] .theme-toggle { background: rgba(255,255,255,0.04); }

    /* responsive */
    @media (max-width: 980px) {
      .container-flex { flex-direction:column; padding:12px; }
      .sidebar { width: auto; order:2; }
      .map-area { order:1; }
    }
  </style>
</head>
<body id="body-root" data-theme="light">
  <!-- Top progress bar (animated while plotting) -->
  <div id="top-progress"><div class="bar" id="top-progress-bar"></div></div>

  <nav class="navbar">
    <div class="container-fluid">
      <a class="navbar-brand" href="#">Accessible Route Finder</a>
      <div style="float:right;">
        <button id="theme-btn" class="btn btn-sm btn-light theme-toggle" title="Toggle dark mode">🌙 <span id="theme-label" style="font-weight:600;margin-left:6px">Dark</span></button>
      </div>
    </div>
  </nav>

  <div class="container-flex">
    <div class="sidebar">
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

        <div class="form-check mb-3">
          <input class="form-check-input" type="checkbox" name="show_stop_preds" id="show_stop_preds" checked>
          <label class="form-check-label" for="show_stop_preds">Show stop markers (busyness dots)</label>
        </div>

        <div class="d-flex gap-2 mb-3">
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

        <div style="margin-top:14px;">
          <b>Legend</b>
          <div class="compact-legend-item"><span class="legend-dot" style="background:#2ecc71"></span><div class="muted">High accessibility</div></div>
          <div class="compact-legend-item"><span class="legend-dot" style="background:#f39c12"></span><div class="muted">Medium accessibility</div></div>
          <div class="compact-legend-item"><span class="legend-dot" style="background:#e74c3c"></span><div class="muted">Low accessibility</div></div>

          <div style="margin-top:12px;"><a href="{{ url_for('download_recommendations') }}">Download recommendations.csv</a></div>
        </div>

      </form>
    </div>

    <div class="map-area">
      {% if map_ready %}
        <!-- loader overlay (traffic lights) -->
        <div id="map-loader" class="loader-overlay">
          <div style="display:flex;gap:18px;align-items:center">
            <div class="traffic">
              <div class="light red" id="light-red"></div>
              <div class="light amber" id="light-amber"></div>
              <div class="light green" id="light-green"></div>
            </div>
            <div style="font-weight:700;color:var(--muted)">Computing accessible routes…</div>
          </div>
        </div>

        <iframe src="{{ url_for('serve_map') }}?ts={{ timestamp }}" frameborder="0" id="map-frame" style="width:100%;height:75vh;border-radius:8px;border:1px solid var(--card-border);" onload="onMapLoaded()"></iframe>
        <!-- We place a transparent cover to hide the small map legend added by the Folium HTML -->
        <div class="legend-cover" aria-hidden="true"></div>
      {% else %}
        <div style="padding:40px;text-align:center;color:#666">
          <h4>Interactive map</h4>
          <p style="max-width:560px;margin:12px auto">Use the controls on the left to compute accessible routes. When ready the map will appear here.</p>
        </div>
      {% endif %}
    </div>
  </div>

  <footer>
    Built for Enhancing public transport accessibility.
  </footer>

  <script>
    // Dark mode toggle with persistence
    (function(){
      const root = document.getElementById('body-root');
      const btn = document.getElementById('theme-btn');
      const label = document.getElementById('theme-label');
      const stored = localStorage.getItem('arf_theme') || 'light';
      root.setAttribute('data-theme', stored);
      label.innerText = stored === 'dark' ? 'Light' : 'Dark';

      btn.addEventListener('click', () => {
        const current = root.getAttribute('data-theme');
        const next = current === 'dark' ? 'light' : 'dark';
        root.setAttribute('data-theme', next);
        localStorage.setItem('arf_theme', next);
        label.innerText = next === 'dark' ? 'Light' : 'Dark';
      });
    })();

    // Top progress bar + loader animation while plotting
    let progressTimer = null;
    function startProgress() {
      document.getElementById('top-progress').style.display = 'block';
      const bar = document.getElementById('top-progress-bar');
      bar.style.width = '6%';
      let pct = 6;
      if (progressTimer) clearInterval(progressTimer);
      progressTimer = setInterval(() => {
        pct = Math.min(92, pct + Math.random() * 12);
        bar.style.width = pct + '%';
      }, 600);
      // show loader overlay
      const loader = document.getElementById('map-loader');
      if (loader) loader.style.display = 'flex';
      startTraffic();
    }
    function finishProgress() {
      const bar = document.getElementById('top-progress-bar');
      if (progressTimer) clearInterval(progressTimer);
      bar.style.width = '100%';
      setTimeout(() => {
        document.getElementById('top-progress').style.display = 'none';
        bar.style.width = '0%';
      }, 420);
      // hide loader overlay
      const loader = document.getElementById('map-loader');
      if (loader) loader.style.display = 'none';
      stopTraffic();
    }

    // traffic-light animation used by loader
    let trafficInterval = null;
    function startTraffic() {
      const red = document.getElementById("light-red");
      const amber = document.getElementById("light-amber");
      const green = document.getElementById("light-green");
      if (!red || !amber || !green) return;
      let state = 0;
      function step() {
        red.classList.remove("on"); amber.classList.remove("on"); green.classList.remove("on");
        if (state === 0) red.classList.add("on");
        else if (state === 1) amber.classList.add("on");
        else green.classList.add("on");
        state = (state + 1) % 3;
      }
      step();
      if (trafficInterval) clearInterval(trafficInterval);
      trafficInterval = setInterval(step, 650);
    }
    function stopTraffic() {
      if (trafficInterval) clearInterval(trafficInterval);
      trafficInterval = null;
      ['light-red','light-amber','light-green'].forEach(id => {
        const el = document.getElementById(id); if (el) el.classList.remove('on');
      });
    }

    // When form is submitted - start top progress and loader
    const form = document.getElementById('route-form');
    form?.addEventListener('submit', () => {
      // start UI progress while plotting runs (server-side)
      startProgress();
      // disable button to avoid duplicate submissions
      document.getElementById('show-btn').disabled = true;
    });

    // called when iframe finishes loading
    function onMapLoaded() {
      // small delay to let assets settle
      setTimeout(() => {
        finishProgress();
        document.getElementById('show-btn').disabled = false;
      }, 350);
    }

    // If user lands on page with map_ready==true (i.e. we are redirecting to index_with_map),
    // show loader immediately until iframe onload fires.
    document.addEventListener('DOMContentLoaded', () => {
      const mapReady = {{ 'true' if map_ready else 'false' }};
      if (mapReady) {
        // show overlay/progress until iframe onload
        startProgress();
      }
    });

    // small UX: focus input selects text
    document.querySelectorAll('input[type="text"]').forEach(inp => {
      inp.addEventListener('focus', e => { e.target.select(); e.target.classList.add('focus'); });
      inp.addEventListener('blur', e => e.target.classList.remove('focus'));
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
    return render_template_string(TEMPLATE,
                                  stops=stops,
                                  default_origin=default_origin,
                                  default_dest=default_dest,
                                  map_ready=False,
                                  timestamp=int(time.time()))

@app.route("/show", methods=["POST"])
def show():
    origin = request.form.get("origin", "").strip()
    dest = request.form.get("dest", "").strip()
    accessible_only = True if request.form.get("accessible_only") else False
    show_stop_preds = True if request.form.get("show_stop_preds") else False
    color_gamma = request.form.get("color_gamma", "0.6")
    stretch_target = request.form.get("stretch_target", "0.35")
    stretch_max = request.form.get("stretch_max", "80")

    stops = load_stops_list()
    if not stops:
        flash("Stops list not found. Check data/gtfs/stops.txt")
        return redirect(url_for("index"))

    if origin == "" or dest == "":
        flash("Please provide both origin and destination (type stop id or name).")
        return redirect(url_for("index"))

    # resolve typed name -> close match (prefer exact name match)
    resolved_origin = origin
    resolved_dest = dest

    lower_names = {s.lower(): sid for sid, s in stops if s}
    lower_ids = {sid.lower(): sid for sid, _ in stops}
    if origin.lower() in lower_names:
        resolved_origin = origin  # plot script accepts stop_name
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

    preds_to_use = DEFAULT_PREDS_ALT if DEFAULT_PREDS_ALT.exists() else DEFAULT_PREDS

    cmd = [
        sys.executable,
        str(PLOT_SCRIPT),
        "--stops", str(GTFS_STOPS),
        "--shapes", str(BASE / "data" / "gtfs" / "shapes_filtered.txt"),
        "--preds", str(preds_to_use),
        "--out", str(OUT_MAP),
        "--origin", str(resolved_origin),
        "--dest", str(resolved_dest),
        "--k", "6",
        "--dist", "2000",
        "--max-detour-pct", "0.30",
        "--color-gamma", str(color_gamma),
        "--show-stop-preds"
    ]

    # pass stretch params (kept to allow fine tuning but default UI removed)
    cmd += ["--stretch-target", str(stretch_target), "--stretch-max", str(stretch_max)]

    if accessible_only:
        cmd.append("--wheelchair")
    if show_stop_preds:
        cmd.append("--show-stop-preds")

    # Always use cache path from project outputs (so we don't download repeatedly)
    cmd += ["--cache", str(BASE / "data" / "outputs" / "osm_graph.pkl")]

    # run plotting synchronously to keep behavior consistent
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
                                  map_ready=True,
                                  timestamp=ts)

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
