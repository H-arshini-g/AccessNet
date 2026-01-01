# scripts/ensure_tiles.py
import sys, pathlib

def ensure_tiles(html_path: pathlib.Path):
    text = html_path.read_text(encoding="utf8")
    # quick check: if a tileLayer is present, do nothing
    if "L.tileLayer(" in text:
        print("Tile layer already present — no change.")
        return
    # create a snippet to insert just before the final "addTo(map_...);" of other layers
    injection = """
// --- injected tile layers fallback (added by ensure_tiles.py) ---
var tile_carto = L.tileLayer(
  "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
  {attribution: '&copy; OpenStreetMap contributors &copy; CARTO', maxZoom: 19, subdomains:"abcd"}
).addTo(map_083b45174d8a7e790c80808f39df18b4);
var tile_osm = L.tileLayer(
  "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
  {attribution: '&copy; OpenStreetMap contributors', maxZoom: 19, subdomains:"abc"}
);
// --- end injected ---
"""
    # naive insertion: append injection before the closing script tag
    if "</script>" in text:
        text = text.replace("</script>", injection + "\n</script>", 1)
        html_path.write_text(text, encoding="utf8")
        print("Injected fallback tile layers into", html_path)
    else:
        print("No </script> tag found, cannot inject.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python scripts/ensure_tiles.py data/outputs/map_shapes_lstm.html")
        sys.exit(1)
    html = pathlib.Path(sys.argv[1])
    ensure_tiles(html)
