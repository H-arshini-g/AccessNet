# scripts/inspect_graph.py
import pickle, os
p = "data/outputs/osm_graph.pkl"
if not os.path.exists(p):
    print("No osm_graph.pkl at", p); raise SystemExit(1)

with open(p, "rb") as f:
    G = pickle.load(f)

ys = [d.get("y") for n,d in G.nodes(data=True) if d.get("y") is not None]
xs = [d.get("x") for n,d in G.nodes(data=True) if d.get("x") is not None]
print("Nodes:", G.number_of_nodes(), "Edges:", G.number_of_edges())
if ys and xs:
    print("lat range:", min(ys), "->", max(ys))
    print("lon range:", min(xs), "->", max(xs))
    print("center approx:", (sum(ys)/len(ys), sum(xs)/len(xs)))
else:
    print("Graph nodes have no x/y coordinates.")
