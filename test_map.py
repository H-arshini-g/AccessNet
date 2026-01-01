# test_map.py
import folium, os
os.makedirs("data/outputs", exist_ok=True)
m = folium.Map(location=[12.97556, 77.55562], zoom_start=13, tiles=None)
folium.TileLayer(
    tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
    attr='&copy; CartoDB | &copy; OpenStreetMap contributors',
    name="CartoDB Positron",
    max_zoom=19,
    subdomains=["a","b","c","d"],
    detect_retina=True
).add_to(m)
folium.TileLayer(
    tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    attr='&copy; OpenStreetMap contributors',
    name="OpenStreetMap",
    max_zoom=19,
    subdomains=["a","b","c"]
).add_to(m)
folium.LayerControl(collapsed=True).add_to(m)
m.save("data/outputs/test_map.html")
print("Saved: data/outputs/test_map.html")
