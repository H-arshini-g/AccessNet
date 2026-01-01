# src/visualize.py
import argparse
import pickle
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


def visualize(graph_file, preds_file, out_file=None):
    # Load graph
    with open(graph_file, "rb") as f:
        G = pickle.load(f)

    # Load predictions
    preds = pd.read_csv(preds_file)

    # If predictions are from LSTM/GRU, reshape them
    if "time_step" in preds.columns:
        # Take the last time step as "current"
        last_row = preds.iloc[-1]
        stop_cols = [c for c in preds.columns if c.startswith("stop_")]
        demand = {stop_id: last_row[c] for stop_id, c in zip(G.nodes(), stop_cols)}
    else:
        # EMA-style predictions
        demand = preds.groupby("stop_id")["ema_pred"].mean().to_dict()

    # Node positions (lat/lon)
    pos = {n: (G.nodes[n]["lon"], G.nodes[n]["lat"]) for n in G.nodes()}

    # Normalize colors
    max_val = max(demand.values()) if demand else 1.0
    node_colors = [demand.get(n, 0) / max_val for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(8, 6))
    nodes = nx.draw_networkx_nodes(
        G, pos, node_size=200, node_color=node_colors,
        cmap=plt.cm.viridis, ax=ax
    )
    nx.draw_networkx_edges(G, pos, edge_color="gray", ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_color="white", ax=ax)

    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                               norm=plt.Normalize(vmin=0, vmax=max_val))
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Predicted Demand")

    if out_file:
        plt.savefig(out_file, dpi=300)
        print(f"Visualization saved to {out_file}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True)
    parser.add_argument("--preds", required=True,
                        help="CSV file (ema_preds.csv or lstm_preds.csv)")
    parser.add_argument("--out", help="Optional: save figure instead of showing")
    args = parser.parse_args()

    visualize(args.graph, args.preds, args.out)
