# src/sidewalks.py
# Uses osmnx to download OSM pedestrian ways in bounding box and compute sidewalk walkability score per segment
import osmnx as ox
import pandas as pd
import argparse
from utils import ensure_dir


def compute_score(tags):
    # tags: dict of OSM tags for a way
    # X1 surface type (1-5), X2 smoothness (1-5), X3 lightness (0/1)
    surface = tags.get('surface', '')
    if 'asphalt' in surface or 'concrete' in surface:
        x1 = 5
    elif 'paving_stones' in surface or 'pavers' in surface:
        x1 = 4
    elif 'gravel' in surface:
        x1 = 2
    else:
        x1 = 3
    smooth = 4 if 'smoothness' in tags and tags['smoothness'] in ['good','excellent'] else 3
    light = 1 if tags.get('lit', 'no') in ['yes', '1', 'true'] else 0
    # simple weights
    alphas = [0.5, 0.4, 0.1]
    score = (alphas[0]*x1 + alphas[1]*smooth + alphas[2]*light) / sum(alphas)
    return score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bbox', required=True,
                        help='LAT_MIN,LON_MIN,LAT_MAX,LON_MAX')
    parser.add_argument('--out', default='data/outputs/sidewalks.csv')
    args = parser.parse_args()
    latmin, lonmin, latmax, lonmax = map(float, args.bbox.split(','))
    print('Downloading OSM pedestrian ways for bbox...')
    g = ox.graph_from_bbox(latmax, latmin, lonmax, lonmin, network_type='walk')
    # convert to GeoDataFrame of edges
    edges = ox.graph_to_gdfs(g, nodes=False, edges=True)
    edges['walkability_score'] = edges['osmid'].apply(lambda x: 3.0)  # default
    for idx, row in edges.iterrows():
        tags = row.get('highway', {})
        # edges store tags inside 'highway' column as string or list; fallback compute
        try:
            score = compute_score(row)
        except Exception:
            score = 3.0
        edges.at[idx, 'walkability_score'] = score
    edges[['u','v','geometry','walkability_score']].to_file('data/outputs/sidewalks.geojson', driver='GeoJSON')
    edges[['u','v','walkability_score']].to_csv(args.out, index=False)
    print('Sidewalks saved to', args.out)