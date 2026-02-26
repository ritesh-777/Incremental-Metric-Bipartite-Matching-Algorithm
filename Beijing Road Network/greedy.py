import csv
import time
import argparse
import torch

# ---------------- Greedy classes ----------------

class PointTorch:
    def __init__(self, coordinates, device):
        # Store as 1D tensor on GPU/CPU
        self.coords = torch.tensor(coordinates, dtype=torch.float32, device=device)

    def __init__(self, coordinates=None, device=None, node_id=None, graph=None):
        """
        If node_id and graph are provided, distance() will use graph.get_distance_from_cache(node_id, other_id).
        Otherwise falls back to L1 distance on coordinates.
        """
        self.device = device
        if node_id is not None and graph is not None:
            self.node_id = str(node_id)
            self.graph = graph
            self.coords = None
        else:
            self.node_id = None
            self.graph = None
            # coordinates must be provided in this case
            self.coords = torch.tensor(coordinates, dtype=torch.float32, device=device)

    
    def distance_cpu(self, other: "PointTorch") -> float:
        # If both points carry node ids and a graph with loaded cache, use cached distances
        if self.node_id is not None and getattr(self, 'graph', None) is not None and \
           other.node_id is not None:
            try:
                # graph.get_distance_from_cache returns float('inf') for missing/unreachable
                return float(self.graph.get_distance_from_cache(self.node_id, other.node_id))
            except Exception:
                # fall back to coordinate L1 if cache/lookup fails
                pass
        '''
        # Fallback: L1 (Manhattan) distance using PyTorch on device
        diff = self.coords - other.coords
        return torch.sum(diff.abs()).item()  # return as Python float
        '''


class GreedyTorch:
    def __init__(self, server_locations, device):
        self.device = device
        self.servers = list(server_locations)
        self.server_assigned = [False] * len(self.servers)
        self.requests = []
        self.match_M = []

    def processRequest(self, r: PointTorch) -> None:
        self.requests.append(r)
        k = len(self.requests) - 1
        self.match_M.append(-1)

        closest_server = -1
        min_distance = float("inf")

        # Loop over unassigned servers
        for s in range(len(self.servers)):
            if not self.server_assigned[s]:
                dist = self.servers[s].distance_cpu(r)  # GPU-powered distance
                if dist < min_distance:
                    min_distance = dist
                    closest_server = s

        if closest_server != -1:
            self.match_M[k] = closest_server
            self.server_assigned[closest_server] = True

    def getTotalCost(self, verbose: bool = False) -> float:
        total = 0.0
        for k, req in enumerate(self.requests):
            s = self.match_M[k]
            if s != -1:
                total += self.servers[s].distance_cpu(req)
        if(verbose):
            print(f"Cost after matching request {len(self.requests)}: ", float(total))
        return total


# ---------------- Main experiment ----------------

def run_experiment(input_file, output_file):
    # Choose device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    servers = []
    requests = []
    with open(input_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sx, sy = float(row["server_x"]), float(row["server_y"])
            rx, ry = float(row["request_x"]), float(row["request_y"])
            servers.append(PointTorch([sx, sy], device))
            requests.append(PointTorch([rx, ry], device))

    greedy = GreedyTorch(servers, device)

    with open(output_file, "w", newline='') as outf:
        writer = csv.writer(outf)
        writer.writerow(["request_id", "cost", "execution_time"])

        for i, req in enumerate(requests, start=1):
            t0 = time.perf_counter()
            greedy.processRequest(req)
            t1 = time.perf_counter()
            exec_time = t1 - t0
            total_cost = greedy.getTotalCost()
            writer.writerow([i, f"{total_cost:.9f}", f"{exec_time:.9f}"])

    print(f"Done. Results written to {output_file}")


# ---------------- CLI ----------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Greedy matching experiment with PyTorch GPU distance")
    parser.add_argument("input_file", help="CSV with columns server_x,server_y,request_x,request_y")
    parser.add_argument("--output_file", default="results.csv", help="Output CSV filename")
    args = parser.parse_args()

    run_experiment(args.input_file, args.output_file)
