import networkx as nx
from pyproj import Transformer
import math
import heapq
import json
import sys
import collections
import time
import csv
import os
import random
import pickle
import numpy as np
import torch


#from RouteVar import *
#from Stop import *
#from Path import *

class Graph:
    def __init__(self):
        self.G = nx.MultiDiGraph()
        self.times_all = {}
        self.shortest_path_time = {}
        self.shortest_adj = {}
        self.shortest_cnt = {}
        self.stop_info = {}
        self.node_order = {}
        self.order_of = {}

    def make_graph_demo(self, file=None, vertices_file=None, edges_file=None, sample_n=None, replace=True, seed=None):
        """
        Populate self.G from CSV files.
        - vertices_file: path to vertices.csv (expects column 'vertices')
        - edges_file: path to edges_with_cost.csv (expects columns 'vertex_1','vertex_2','length')
        - sample_n: if int, randomly sample that many rows from vertices.csv (default: None => use all)
        - replace: sampling with replacement when True (default True)
        - seed: optional random seed for reproducibility
        If paths are None, assumes files are in the same directory as this file.
        """
        if seed is not None:
            random.seed(seed)

        # Resolve default paths next to this file
        base_dir = os.path.dirname('./')
        if vertices_file is None:
            vertices_file = os.path.join(base_dir, 'vertices.csv')
        if edges_file is None:
            edges_file = os.path.join(base_dir, 'edges_with_cost.csv')

        all_nodes = []
        
        # Read vertices
        try:
            with open(vertices_file, newline='', encoding='utf-8') as vf:
                reader = csv.DictReader(vf)
                if 'vertices' not in reader.fieldnames:
                    raise ValueError(f"'vertices' column not found in {vertices_file}")
                for row in reader:
                    val = row['vertices']
                    if val is None or val == '':
                        continue
                    # keep as string (IDs in edges file are strings too)
                    all_nodes.append(val)
        except FileNotFoundError:
            raise FileNotFoundError(f"Vertices file not found: {vertices_file}")
        except Exception as e:
            raise

        # choose nodes: either all or a random sample
        if sample_n is None:
            nodes = all_nodes
        else:
            if replace:
                nodes = random.choices(all_nodes, k=sample_n)
            else:
                if sample_n > len(all_nodes):
                    raise ValueError(f"sample_n ({sample_n}) > available unique vertices ({len(all_nodes)}) when replace=False")
                nodes = random.sample(all_nodes, k=sample_n)

        # add nodes to graph and build a set for fast membership checks
        self.G.add_nodes_from(nodes)
        selected_nodes = set(nodes)

        # ensure times_all dict exists
        if not hasattr(self, 'times_all') or self.times_all is None:
            self.times_all = {}

        # Read edges and add to graph
        try:
            with open(edges_file, newline='', encoding='utf-8') as ef:
                reader = csv.DictReader(ef)
                for required in ('vertex_1', 'vertex_2', 'length'):
                    if required not in reader.fieldnames:
                        raise ValueError(f"'{required}' column not found in {edges_file}")

                for row in reader:
                    u = row['vertex_1']
                    v = row['vertex_2']
                    if u is None or v is None or u == '' or v == '':
                        continue
                    # only include edges where both endpoints are in the selected node set
                    if u not in selected_nodes or v not in selected_nodes:
                        continue
                    try:
                        w = float(row['length'])
                    except:
                        # skip invalid weights
                        continue

                    # add directed edges both ways (as the demo did)
                    self.G.add_edge(u, v, weight=w, shortcut_node=0)
                    self.G.add_edge(v, u, weight=w, shortcut_node=0)

                    # update times_all for Dijkstra usage
                    if u not in self.times_all:
                        self.times_all[u] = {}
                    if v not in self.times_all:
                        self.times_all[v] = {}
                    self.times_all[u][v] = w
                    self.times_all[v][u] = w
        except FileNotFoundError:
            raise FileNotFoundError(f"Edges file not found: {edges_file}")
        except Exception as e:
            raise


    '''
    def make_graph_demo(self):
        nodes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
        self.G.add_nodes_from(nodes)

        # Define edges with weights
        edges = [
            ('A', 'C', 5), ('A', 'B', 3),
            ('B', 'C', 3), ('B', 'D', 5),
            ('C', 'D', 2), ('C', 'J', 2),
            ('D', 'E', 7), ('D', 'J', 4), 
            ('E', 'F', 6), ('E', 'J', 3),
            ('F', 'H', 2), 
            ('G', 'H', 3), ('G', 'F', 4),
            ('H', 'I', 3), ('H', 'J', 2), 
            ('I', 'J', 4), ('I', 'G', 5), 
            ('J', 'K', 3),
            ('K', 'A', 3), ('K', 'I', 6)
        ]

        # Add edges with weights and shortcut_node attribute
        for u, v, w in edges:
            self.G.add_edge(u, v, weight=w, shortcut_node=0)
            self.G.add_edge(v, u, weight=w, shortcut_node=0)
            if u not in self.times_all:
                self.times_all[u] = {}
            if v not in self.times_all:
                self.times_all[v] = {}
            self.times_all[u][v] = w
            self.times_all[v][u] = w

    '''

    def dijkstra(self, start_vertex):
        vertices = list(self.G.nodes())

        visited = set()
        parents = {}
        adj = {v: [] for v in vertices}
        exist = {v: 0 for v in vertices}
        D = {v: float('inf') for v in vertices}

        parents[start_vertex] = start_vertex
        D[start_vertex] = 0
        exist[start_vertex] = 1
        pq = [(0, start_vertex)]
        while pq:
            current_cost, current_vertex = heapq.heappop(pq)
            if current_vertex in visited:
                continue
            visited.add(current_vertex)

            current_neighbors = list(self.G.neighbors(current_vertex))
            for neighbor in current_neighbors:
                old_cost = D[neighbor]
                new_cost = D[current_vertex] + self.times_all[current_vertex][neighbor]
                if new_cost < old_cost:
                    parents[neighbor] = current_vertex
                    exist[neighbor] = exist[current_vertex]
                    if neighbor not in adj[current_vertex]:
                        adj[current_vertex].append(neighbor)
                    D[neighbor] = new_cost
                    heapq.heappush(pq, (new_cost, neighbor))

        self.shortest_adj[start_vertex] = adj
        self.shortest_cnt[start_vertex] = exist
        return D, parents
    
    def dijkstra_all_pairs(self):
        vertices = list(self.G.nodes())
        num_sources = len(vertices)
        t1 = time.time()
        for i, u in enumerate(vertices[:num_sources], start=1):
            print(i)
            # compute Dijkstra once per source u
            dist, parents = self.dijkstra(u)
            for v in vertices:
                p = self.get_path(v, parents)
        t2 = time.time()
        total_calls = num_sources * len(vertices)
        print(t2 - t1, (t2 - t1) / total_calls if total_calls else 0)

    def get_shortest_path_dijkstra(self, start_stop, end_stop):
        shortest_time, parents = self.dijkstra(start_stop)
        path = self.get_path(end_stop, parents)
        if not path:
            print(f'No path found from {start_stop} to {end_stop}!')
            return 0, path
        # print(shortest_time[end_stop], path) 
        return shortest_time[end_stop], path  
    
    def get_shortest_paths_from_source_dijkstra(self, start_stop):
        """
        Return single-source shortest-paths from start_stop to every vertex.

        Returns:
          - D: dict {vertex: distance}
          - paths: dict {vertex: [ordered node ids along shortest path]} (empty list for unreachable)
        """
        D, parents = self.dijkstra(start_stop)

        paths = {}
        for v in self.G.nodes():
            p = self.get_path(v, parents)
            paths[v] = p

        return D#, paths

    def get_path(self, current_vertex, parents):
        path = []
        if parents.get(current_vertex, 0) == 0:
            return path
        while parents[current_vertex] != current_vertex:
            path.append(current_vertex)
            current_vertex = parents[current_vertex]
        path.append(current_vertex)
        return list(reversed(path))
    
    def convert_pickle_to_csv(self, pickle_path, nodes_csv="nodes.csv", dists_csv="dists.csv"):
        # Load the pickle
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)

        nodes = data["nodes"]
        INF = data["INF"]
        dists = np.asarray(data["dists"], dtype=np.float32)

        # Save nodes.csv
        with open(nodes_csv, "w", newline="") as f:
            writer = csv.writer(f)
            for node in nodes:
                writer.writerow([node])

        i = 0

        # Save dists.csv with header row
        with open(dists_csv, "w", newline="") as f:
            writer = csv.writer(f)

            # header: empty cell + node IDs
            writer.writerow([""] + nodes)

            # each row: node ID + row of distances
            for node, row in zip(nodes, dists):
                row_values = [INF if x >= INF else float(x) for x in row]
                print(f"Writing row {i} for node {node}")
                i += 1
                writer.writerow([node] + row_values)

        print(f"Saved {nodes_csv} and {dists_csv}")


    def convert_pickle_to_binary(self, pickle_path,
                             dists_bin_path="dists.bin",
                             nodes_txt_path="nodes.txt",
                             meta_json_path="meta.json"):
        """
        Converts first_n_dists.pkl into compact binary format:
        - dists.bin    : raw float32 matrix (n*n)
        - nodes.txt    : node IDs, one per line
        - meta.json    : metadata (n and INF)
        """

        # Load the pickle
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)

        nodes = data["nodes"]
        INF = float(data["INF"])
        dists = np.asarray(data["dists"], dtype=np.float32)

        n = dists.shape[0]
        print(f"Loaded matrix: {n} x {n}")

        # Save binary matrix
        dists.tofile(dists_bin_path)   # raw float32 (row-major)
        print(f"Saved distances to {dists_bin_path}")

        # Save nodes
        with open(nodes_txt_path, "w") as f:
            for node in nodes:
                f.write(str(node) + "\n")
        print(f"Saved nodes to {nodes_txt_path}")

        # Save metadata
        meta = {"n": n, "INF": INF}
        with open(meta_json_path, "w") as f:
            json.dump(meta, f)
        print(f"Saved metadata to {meta_json_path}")

        return dists_bin_path, nodes_txt_path, meta_json_path
    
    def python_sanity_checks(self, pickle_path="first_n_dists.pkl",
                         nodes_txt="nodes.txt",
                         dists_bin="dists.bin",
                         meta_json="meta.json",
                         src="2477780932",
                         tgt="8259183804"):
        # load pickle (original)
        with open(pickle_path, "rb") as f:
            pk = pickle.load(f)
        pk_nodes = list(pk["nodes"])
        pk_INF = float(pk["INF"])
        pk_dists = np.asarray(pk["dists"], dtype=np.float32)
        print("pickle: n=", pk_dists.shape, " INF=", pk_INF)

        # load nodes.txt
        with open(nodes_txt, "r") as f:
            nodes = [line.rstrip("\n\r") for line in f]
        print("nodes.txt: count=", len(nodes))

        # load meta
        with open(meta_json, "r") as f:
            meta = json.load(f)
        n = int(meta["n"])
        INF = float(meta["INF"])
        print("meta:", meta)

        # read binary matrix
        d = np.fromfile(dists_bin, dtype=np.float32)
        if d.size != n*n:
            print("WARNING: dists.bin size mismatch: got", d.size, "expected", n*n)
        d = d.reshape((n, n))

        # check indices
        try:
            i = nodes.index(src)
            j = nodes.index(tgt)
            print("src idx", i, "tgt idx", j)
            print("value from binary:", d[i, j])
        except ValueError as e:
            print("Node missing in nodes.txt:", e)

        # compare to pickle (if these nodes in pickle)
        if src in pk_nodes and tgt in pk_nodes:
            i2 = pk_nodes.index(src)
            j2 = pk_nodes.index(tgt)
            print("pickle indices", i2, j2, "pickle value:", pk_dists[i2, j2])
        else:
            print("One of nodes missing in original pickle list")

        # show raw node strings (helpful if whitespace)
        for idx in [i, j]:
            print("nodes.txt[{}]|{}|".format(idx, nodes[idx]))
            print("pickle_nodes[{}]|{}|".format(idx, pk_nodes[idx] if idx < len(pk_nodes) else "n/a"))


    
    def compute_and_save_first_n_dists_pickle(self, n=None, out_pickle='first_n_dists.pkl'):
        """
        Compute distances between the first n nodes (by ordering from list(self.G.nodes()))
        for all n x n pairs using Dijkstra and save the result as a pickle.

        Pickle contents (dict):
          - 'nodes': list of node ids (length n)
          - 'INF': float sentinel used for unreachable entries
          - 'dists': numpy.ndarray shape (n,n) dtype float32 with distances

        Returns: out_pickle path
        """
        if n is None:
            n = len(self.G.nodes())
        if n <= 0:
            raise ValueError("n must be > 0")

        vertices = list(self.G.nodes())
        n = min(n, len(vertices))
        nodes = vertices[:n]

        INF = np.finfo(np.float32).max / 4.0
        dists = np.full((n, n), INF, dtype=np.float32)

        for i, src in enumerate(nodes):
            D, _ = self.dijkstra(src)
            for j, tgt in enumerate(nodes):
                val = D.get(tgt, float('inf'))
                if val == float('inf') or val is None:
                    dists[i, j] = INF
                else:
                    dists[i, j] = np.float32(val)
                print(f"Computed dist from the {i}-th source {src} to {j}-th target {tgt}: {dists[i, j]}")

        data = {'nodes': nodes, 'INF': float(INF), 'dists': dists}
        with open(out_pickle, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Saved distances pickle to '{out_pickle}'")
        return out_pickle
    
    def load_first_n_dists_pickle(self, pickle_path):
        """
        Load distances pickle produced by compute_and_save_first_n_dists_pickle.

        Returns:
          - nodes: list of node ids (length n)
          - INF: float sentinel used for unreachable entries
          - dists: numpy.ndarray shape (n,n) dtype float32
        """
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)

        nodes = data.get('nodes')
        INF = data.get('INF')
        dists = data.get('dists')

        if nodes is None or dists is None or INF is None:
            raise ValueError("Pickle missing required keys: 'nodes','dists','INF'")

        dists = np.asarray(dists, dtype=np.float32)
        return nodes, float(INF), dists

    def get_distance_from_pickle(self, pickle_path, src_node, tgt_node):
        """
        Convenience: return scalar distance between src_node and tgt_node using the saved pickle.
        Returns float('inf') if either node not in stored list or distance is sentinel INF.
        """
        nodes, INF, dists = self.load_first_n_dists_pickle(pickle_path)
        id_to_idx = {v: i for i, v in enumerate(nodes)}
        if src_node not in id_to_idx or tgt_node not in id_to_idx:
            return float('inf')
        i = id_to_idx[src_node]
        j = id_to_idx[tgt_node]
        val = float(dists[i, j])
        return float('inf') if val >= INF else val
    
    def load_first_n_cache(self, pickle_path):
        """
        Load distances pickle into memory once and keep it cached for repeated queries.
        """
        # if already loaded same path, do nothing
        if getattr(self, '_first_n_cache_path', None) == pickle_path and getattr(self, '_first_n_cache', None) is not None:
            return
        nodes, INF, dists = self.load_first_n_dists_pickle(pickle_path)
        # keep in memory (numpy array stays in RAM)
        self._first_n_cache = {'nodes': nodes, 'INF': INF, 'dists': dists}
        self._first_n_id_to_idx = {v: i for i, v in enumerate(nodes)}
        self._first_n_cache_path = pickle_path

    def get_distance_from_cache(self, src_node, tgt_node, pickle_path=None):
        """
        Query pairwise distance using the in-memory cache.
        If pickle_path is provided and not loaded, it will be loaded once.
        Returns float('inf') when nodes are missing or unreachable.
        """
        if pickle_path:
            if getattr(self, '_first_n_cache_path', None) != pickle_path or getattr(self, '_first_n_cache', None) is None:
                self.load_first_n_cache(pickle_path)
        if getattr(self, '_first_n_cache', None) is None:
            raise RuntimeError("No distance cache loaded. Call load_first_n_cache(pickle_path) first or pass pickle_path here.")
        id_to_idx = self._first_n_id_to_idx
        INF = self._first_n_cache['INF']
        dists = self._first_n_cache['dists']
        if src_node not in id_to_idx or tgt_node not in id_to_idx:
            return float('inf')
        i = id_to_idx[src_node]
        j = id_to_idx[tgt_node]
        val = float(dists[i, j])
        return float('inf') if val >= INF else val

    def clear_first_n_cache(self):
        """Free the in-memory distances cache."""
        if hasattr(self, '_first_n_cache'):
            del self._first_n_cache
        if hasattr(self, '_first_n_id_to_idx'):
            del self._first_n_id_to_idx
        if hasattr(self, '_first_n_cache_path'):
            del self._first_n_cache_path


    def load_first_n_cache_to_gpu(self, pickle_path, device=None):
        """
        Load the previously saved first_n pickle into GPU (or specified device) as a torch tensor.
        Keeps mappings in self._first_n_id_to_idx and stores tensor in self._first_n_cache_gpu.
        """
        # ensure numpy cache present in memory
        self.load_first_n_cache(pickle_path)

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)

        # numpy array from earlier load
        dists_np = self._first_n_cache['dists']
        # create torch tensor on device (copy happens when moving to device)
        # ensure dtype float32
        if dists_np.dtype != np.float32:
            dists_np = dists_np.astype(np.float32, copy=False)
        tensor = torch.from_numpy(dists_np)  # on CPU
        tensor = tensor.to(device)           # copy to device
        # store
        self._first_n_cache_gpu = tensor
        self._first_n_cache_gpu_device = device
        # keep INF accessible
        self._first_n_cache_INF = float(self._first_n_cache['INF'])
        return tensor

    def get_distances_from_gpu_cache(self, req_node_ids, server_node_ids=None):
        """
        Query distances from the in-GPU cache.
        - req_node_ids: iterable of node ids (rows)
        - server_node_ids: optional iterable of node ids (columns). If None, return full row(s).
        Returns: torch.Tensor on the same device as the cached tensor with shape (len(reqs), len(servers or all_cols))
        """
        if not hasattr(self, '_first_n_cache_gpu'):
            raise RuntimeError("GPU cache not loaded. Call load_first_n_cache_to_gpu(pickle_path) first.")

        id_to_idx = self._first_n_id_to_idx
        gpu_mat = self._first_n_cache_gpu
        INF = self._first_n_cache_INF
        device = self._first_n_cache_gpu_device

        # column indices
        if server_node_ids is None:
            col_idx = None
        else:
            col_idx = []
            for vid in server_node_ids:
                if vid in id_to_idx:
                    col_idx.append(id_to_idx[vid])
                else:
                    # mark missing as -1 (will be handled later)
                    col_idx.append(None)

        rows = []
        full_col_count = gpu_mat.size(1)
        for vid in req_node_ids:
            if vid in id_to_idx:
                row = gpu_mat[id_to_idx[vid]]
            else:
                row = torch.full((full_col_count,), INF, dtype=gpu_mat.dtype, device=device)
            if col_idx is None:
                rows.append(row)
            else:
                # build selected columns, handle missing cols by INF
                sel = []
                for c in col_idx:
                    if c is None:
                        sel.append(torch.tensor(INF, dtype=gpu_mat.dtype, device=device))
                    else:
                        sel.append(row[c])
                rows.append(torch.stack(sel))
        result = torch.stack(rows, dim=0)
        return result
    
    def get_distances_from_gpu_cache_parallel(self, req_node_ids, server_node_ids=None):
        """
        Vectorized GPU lookup of precomputed distances.
        - req_node_ids: iterable of node ids
        - server_node_ids: iterable of node ids or None (return full rows)
        Returns: torch.Tensor (B, S) on the same device as the cached tensor.
        """
        if not hasattr(self, '_first_n_cache_gpu'):
            raise RuntimeError("GPU cache not loaded. Call load_first_n_cache_to_gpu(pickle_path) first.")

        id_to_idx = self._first_n_id_to_idx
        gpu_mat = self._first_n_cache_gpu
        INF = self._first_n_cache_INF
        device = self._first_n_cache_gpu_device

        # build row indices tensor (invalid -> -1)
        row_idxs = [id_to_idx.get(vid, -1) for vid in req_node_ids]
        row_idx_t = torch.tensor(row_idxs, dtype=torch.long, device=device)

        if server_node_ids is None:
            # return full rows
            full_cols = gpu_mat.size(1)
            # clamp for indexing then mask invalid rows
            clamped_rows = row_idx_t.clamp(min=0)
            rows = gpu_mat[clamped_rows]               # (B, full_cols)
            if (row_idx_t < 0).any():
                rows = rows.clone()
                rows[row_idx_t < 0] = INF
            return rows

        # build column indices tensor (invalid -> -1)
        col_idxs = [id_to_idx.get(vid, -1) for vid in server_node_ids]
        col_idx_t = torch.tensor(col_idxs, dtype=torch.long, device=device)

        # clamp for safe indexing
        clamped_rows = row_idx_t.clamp(min=0)
        clamped_cols = col_idx_t.clamp(min=0)

        # gather rows then select columns (all on GPU, vectorized)
        rows = gpu_mat[clamped_rows]                 # (B, full_cols)
        selected = rows[:, clamped_cols]            # (B, S)

        # mask invalid rows or columns to INF
        if (row_idx_t < 0).any() or (col_idx_t < 0).any():
            invalid_rows_mask = (row_idx_t < 0).unsqueeze(1)    # (B,1)
            invalid_cols_mask = (col_idx_t < 0).unsqueeze(0)    # (1,S)
            invalid_mask = invalid_rows_mask | invalid_cols_mask
            selected = selected.clone()
            selected[invalid_mask] = INF

        return selected

    def clear_first_n_cache_gpu(self):
        """Free the GPU cache tensor."""
        if hasattr(self, '_first_n_cache_gpu'):
            del self._first_n_cache_gpu
        if hasattr(self, '_first_n_cache_gpu_device'):
            del self._first_n_cache_gpu_device
        if hasattr(self, '_first_n_cache_INF'):
            del self._first_n_cache_INF
  
    # Contraction Hieracrhies
    def get_node_order(self):
        node_pq = []
        for v in list(self.G.nodes()):
            val = len(list(self.G.neighbors(v))) + len(list(self.G.predecessors(v)))
            heapq.heappush(node_pq, (val, v))
        i = 1
        while node_pq:
            _, v = heapq.heappop(node_pq)
            self.node_order[i] = v
            self.order_of[v] = i
            i += 1

    def local_dijkstra_without_v(self, u, v, P_max):
        vertices = list(self.G.nodes)
        visited = set()
        pq = [(0, u)]
        D = {v: float('inf') for v in vertices}
        visited.add(v)
        D[u] = 0
        while pq:
            cost, n = heapq.heappop(pq)
            if n in visited:
                continue
            if cost > P_max:
                break
            visited.add(n)

            for neighbor in list(self.G.neighbors(n)):
                if neighbor in self.order_of:
                    continue
                old_cost = D[neighbor]
                new_cost = D[n] + self.times_all[n][neighbor]
                if new_cost < old_cost:
                    D[neighbor] = new_cost
                    heapq.heappush(pq, (new_cost, neighbor))
        return D

    def edge_difference(self, v):
        dif = - len(list(self.G.neighbors(v))) - len(list(self.G.predecessors(v)))
        for u in list(self.G.predecessors(v)):
            if u in self.order_of:
                continue
            P = {}
            for w in list(self.G.neighbors(v)):
                if w in self.order_of:
                    continue
                P[w] = self.times_all[u][v] + self.times_all[v][w]
            if not P:
                continue
            P_max = max(P.values())

            D = self.local_dijkstra_without_v(u, v, P_max)
            
            for w in list(self.G.neighbors(v)):
                if w in self.order_of:
                    continue
                if D[w] > P[w]:
                    dif += 1
        return dif
    
    def get_node_order_edge_difference(self):
        node_pq = []
        for v in list(self.G.nodes()):
            dif = self.edge_difference(v)
            heapq.heappush(node_pq, (dif, v))
        return node_pq
    
    def preprocess(self):
        node_pq = self.get_node_order_edge_difference()
        order = 0
        while node_pq:
            # Calculate edge difference again to update the pq and get the next node
            _, v = heapq.heappop(node_pq)

            new_dif = self.edge_difference(v)
            if node_pq and new_dif > node_pq[0][0]:
                heapq.heappush(node_pq, (new_dif, v))
                continue

            order += 1
            if order % 500 == 0:
                total = len(self.G.nodes())
                print(f"..........Contracting {order}/{total} nodes..........")
            self.order_of[v] = order
            self.node_order[order] = v

            for u in list(self.G.predecessors(v)):
                if u in self.order_of:
                    continue
                P = {}
                for w in list(self.G.neighbors(v)):
                    if w in self.order_of:
                        continue
                    P[w] = self.times_all[u][v] + self.times_all[v][w]
                if not P:
                    continue
                P_max = max(P.values())

                D = self.local_dijkstra_without_v(u, v, P_max)

                for w in list(self.G.neighbors(v)):
                    if w in self.order_of:
                        continue
                    
                    if D[w] > P[w]:
                        if self.G.has_edge(u, w):
                            self.G.get_edge_data(u, w)[0]['shortcut_node'] = v
                        else:
                            self.G.add_edge(u, w, shortcut_node=v)
                        self.times_all[u][w] = P[w]
        print('Preprocess Done!')
        # mark in-memory preprocessing done
        self._is_preprocessed = True

    def bidirectional_dijkstra(self, source_node, target_node):
        vertices = list(self.G.nodes())
        visited_start = set()
        visited_end = set()
        parents1 = {}
        parents2 = {}
        dist1 = {v: float('inf') for v in vertices}
        dist2 = {v: float('inf') for v in vertices}

        parents1[source_node] = source_node
        parents2[target_node] = target_node
        dist1[source_node] = 0
        dist2[target_node] = 0
        pq_start = [(0, source_node)]
        pq_end = [(0, target_node)]
        while pq_start or pq_end:
            if pq_start:
                _, current_vertex = heapq.heappop(pq_start)
                if current_vertex in visited_start:
                    continue
                visited_start.add(current_vertex)

                for neighbor in self.G.neighbors(current_vertex):
                    if self.order_of[neighbor] <= self.order_of[current_vertex]:
                        continue

                    new_cost = dist1[current_vertex] + self.times_all[current_vertex][neighbor]
                    if new_cost < dist1[neighbor]:
                        parents1[neighbor] = current_vertex
                        dist1[neighbor] = new_cost
                        heapq.heappush(pq_start, (new_cost, neighbor))
            if pq_end:
                _, current_vertex = heapq.heappop(pq_end)
                if current_vertex in visited_end:
                    continue
                visited_end.add(current_vertex)

                for neighbor in self.G.predecessors(current_vertex):
                    if self.order_of[neighbor] <= self.order_of[current_vertex]:
                        continue

                    new_cost = dist2[current_vertex] + self.times_all[neighbor][current_vertex]
                    if new_cost < dist2[neighbor]:
                        parents2[neighbor] = current_vertex
                        dist2[neighbor] = new_cost
                        heapq.heappush(pq_end, (new_cost, neighbor))

        L = [v for v in self.G.nodes if dist1[v] != float('inf') and dist2[v] != float('inf')]
        if not L:
            return 0, []

        shortest_time = math.inf
        common_node = 0
        for v in L:
            if shortest_time > dist1[v] + dist2[v]:
                shortest_time = dist1[v] + dist2[v]
                common_node = v

        def generate_shortcut(start_node, end_node):
            shortcut_node = self.G.get_edge_data(start_node, end_node)[0]['shortcut_node']
            if shortcut_node != 0:
                return generate_shortcut(start_node, shortcut_node) + [shortcut_node] + generate_shortcut(shortcut_node, end_node)
            else:
                return []

        shortest_path = []
        path1 = []
        cur_node = common_node

        while parents1[cur_node] != cur_node:
            tmp_node = parents1[cur_node]
            path = []
            if self.G.get_edge_data(tmp_node, cur_node)[0]['shortcut_node'] != 0:
                path = generate_shortcut(tmp_node, cur_node)
            path1 = path + path1
            path1 = [tmp_node] + path1
            cur_node = tmp_node

        cur_node = common_node
        path2 = []
        while parents2[cur_node] != cur_node:
            path2.append(cur_node)
            tmp_node = parents2[cur_node]
            path = []
            if self.G.get_edge_data(cur_node, tmp_node)[0]['shortcut_node'] != 0:
                path = generate_shortcut(cur_node, tmp_node)
            path2 += path
            cur_node = tmp_node
        path2.append(cur_node)
        
        shortest_path = path1 + path2
        return shortest_time, shortest_path
    

    def save_preprocess(self, path):
        """
        Save preprocessing state to path (binary pickle).
        Stores: graph (self.G), times_all, order_of, node_order.
        """
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        data = {
            'G': self.G,
            'times_all': self.times_all,
            'order_of': self.order_of,
            'node_order': self.node_order
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load_preprocess(self, path):
        """
        Load preprocessing state from path. Overwrites self.G, self.times_all, self.order_of, self.node_order.
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        # minimal validation
        self.G = data.get('G', self.G)
        self.times_all = data.get('times_all', self.times_all)
        self.order_of = data.get('order_of', self.order_of)
        self.node_order = data.get('node_order', self.node_order)
        # mark as preprocessed in memory
        self._is_preprocessed = True

    def get_shortest_path_CH(self, source_node, target_node, cache_path=None, force_recompute=False):
        """
        If cache_path is provided and exists -> load preprocessing from it.
        Otherwise run preprocess() and if cache_path provided, save result to it.
        Set force_recompute=True to force running preprocess() even if cached/in-memory.
        """
        if not force_recompute:
            if getattr(self, '_is_preprocessed', False):
                # already preprocessed in memory — do nothing
                pass
            elif cache_path and os.path.exists(cache_path):
                self.load_preprocess(cache_path)
                print(f"Loaded preprocessed state from {cache_path}")
            else:
                t1 = time.time()
                self.preprocess()
                t2 = time.time()
                print("Preprocessing time:", t2 - t1)
                if cache_path:
                    try:
                        self.save_preprocess(cache_path)
                        print(f"Saved preprocessed state to {cache_path}")
                    except Exception as e:
                        print(f"Failed to save preprocess cache: {e}")
        else:
            t1 = time.time()
            self.preprocess()
            t2 = time.time()
            print("Preprocessing time:", t2 - t1)
            if cache_path:
                try:
                    self.save_preprocess(cache_path)
                    print(f"Saved preprocessed state to {cache_path}")
                except Exception as e:
                    print(f"Failed to save preprocess cache: {e}")

        # run query
        t_start = time.time()
        t, p = self.bidirectional_dijkstra(source_node, target_node)
        t_end = time.time()
        print('Query time: ', t_end - t_start)
        if not p:
            print(f'No path found from {source_node} to {target_node}!')
            return 0, p
        print(t, p)
        return t, p
    
    def get_shortest_paths_from_source_CH(self, start_node, ensure_preprocessed=False):
        """
        Return single-source shortest-paths from start_node to every vertex.
        - ensure_preprocessed: if True, ensure preprocess() has run (useful if you rely on CH for other queries).
        Returns:
          - D: dict {vertex: distance}
          - paths: dict {vertex: [ordered node ids along shortest path]} (empty list for unreachable)
        Notes:
          - CH is a point-to-point acceleration structure; for full single-source results we run Dijkstra once.
        """
        if ensure_preprocessed and not getattr(self, '_is_preprocessed', False):
            # run preprocess once (will set _is_preprocessed)
            self.preprocess()

        D, parents = self.dijkstra(start_node)

        paths = {}
        for v in self.G.nodes():
            paths[v] = self.get_path(v, parents)

        return D#, paths
    
    def CH_all_pairs(self):
        t1 = time.time()
        self.preprocess()
        t2 = time.time()
        print('Preprocess time: ', t2 - t1)
        vertices = list(self.G.nodes)
        num_sources = len(vertices)   # run at most 15 sources or fewer if graph is smaller
        t_start = time.time()
        for i, u in enumerate(vertices[:num_sources], start=1):
            print(i)
            for v in vertices:
                t, p = self.bidirectional_dijkstra(u, v)
        t_end = time.time()
        total_calls = num_sources * len(vertices)
        print(t_end - t_start, (t_end - t_start) / total_calls)