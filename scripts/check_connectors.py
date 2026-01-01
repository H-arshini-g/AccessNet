# scripts/check_connectors.py
import pickle, math
G = pickle.load(open("data/outputs/osm_graph.pkl","rb"))
# edit to the origin/dest you used
o_lat,o_lon = 12.8600,77.5717   # replace with your o_lat,o_lon printed earlier
d_lat,d_lon = 12.9490,77.5920   # replace with your d_lat,d_lon
def hav(lat1,lon1,lat2,lon2):
    R=6371.0
    import math
    a = math.sin(math.radians(lat2-lat1)/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(math.radians(lon2-lon1)/2)**2
    return 2*R*math.asin(min(1,math.sqrt(a)))*1000.0
import osmnx as ox
orig_node = ox.distance.nearest_nodes(G, o_lon, o_lat)
dest_node = ox.distance.nearest_nodes(G, d_lon, d_lat)
on = G.nodes[orig_node]; dn = G.nodes[dest_node]
print("orig_node", orig_node, "node coords", on.get("y"), on.get("x"), "dist_m", hav(o_lat,o_lon,on.get("y"),on.get("x")))
print("dest_node", dest_node, "node coords", dn.get("y"), dn.get("x"), "dist_m", hav(d_lat,d_lon,dn.get("y"),dn.get("x")))
