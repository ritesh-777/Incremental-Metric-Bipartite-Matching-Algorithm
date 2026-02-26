import csv
import random
import argparse
import os

def load_vertices(vertices_csv_path):
    verts = []
    with open(vertices_csv_path, newline='', encoding='utf-8') as f:
    #with open(vertices_csv_path, newline='') as f:
        rdr = csv.DictReader(f)
        if 'vertices' not in rdr.fieldnames:
            raise ValueError("Input CSV must contain a 'vertices' column")
        for row in rdr:
            v = row['vertices']
            if v is not None and v != '':
                verts.append(v)
    if not verts:
        raise ValueError("No vertex ids found in 'vertices' column")
    return verts

def make_samples(verts, k=10000, replace=True):
    if replace:
        return random.choices(verts, k=k)
    # without replacement
    if len(verts) < k:
        raise ValueError(f"Not enough unique vertices ({len(verts)}) for sampling {k} without replacement")
    return random.sample(verts, k)

def write_pair_csv(path, servers, requests):
    with open(path, 'w', newline='', encoding='utf-8') as f:
    #with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['servers', 'requests'])
        for s, r in zip(servers, requests):
            writer.writerow([s, r])

def main(vertices_csv='vertices.csv', out_dir='.', n=5, k=10000, replace=True, seed=None):
    if seed is not None:
        random.seed(seed)
    verts = load_vertices(vertices_csv)
    os.makedirs(out_dir, exist_ok=True)
    for i in range(1, n+1):
        servers = make_samples(verts, k=k, replace=replace)
        requests = make_samples(verts, k=k, replace=replace)
        fname = f"{i}_Point_0.001000_CityRoad_Graph_1dim.csv"
        out_path = os.path.join(out_dir, fname)
        write_pair_csv(out_path, servers, requests)
        print(f"Wrote {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create n server/request CSVs sampled from vertices.csv")
    parser.add_argument('-v', '--vertices', default='vertices.csv', help='path to vertices.csv')
    parser.add_argument('-o', '--outdir', default='.', help='output directory')
    parser.add_argument('-n', type=int, default=5, help='number of CSV files to create')
    parser.add_argument('-k', type=int, default=10000, help='number of samples per column')
    parser.add_argument('--no-replace', dest='replace', action='store_false', help='sample without replacement')
    parser.add_argument('--seed', type=int, default=None, help='random seed (optional)')
    args = parser.parse_args()

    main(vertices_csv=args.vertices, out_dir=args.outdir, n=args.n, k=args.k, replace=args.replace, seed=args.seed)