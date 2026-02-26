# online_matching_gpu_points.py
# GPU-only PyTorch implementation of your batch-by-batch online matching.
# Distances are computed from d-dimensional points (Euclidean).
# The multi-level scaled distances d_i follow the formula:
#   d_0(a,b) = ceil( (2 * d(a,b) * n) / (eps * omega) )
#   d_i(a,b) = ceil( d_{i-1}(a,b) / ( 2 * (1+eps) * 2^n * phi_{i-1} ) )   for i > 0
#   where phi_i = 3^i * delta
#
# Usage: python online_matching_gpu_points.py
# Requires PyTorch with CUDA.

#from Graph import *

import torch
import math
#import time
import numpy as np
import ot
from typing import Optional, Callable
import gc
from scipy.optimize import linear_sum_assignment

INF = 10**8

class OnlineMatchingGPU:
    def __init__(
        self,
        server_points: torch.Tensor,   # shape (S, dim)
        omega_init: float,
        delta: float,
        graph: None,
        device: Optional[torch.device] = None,
        omega_validity_check: Optional[Callable] = None,
    ):
        """
        server_points: (S, dim) float tensor (CPU or CUDA) of server coordinates
        mu: max level index (levels 0..mu)
        omega_init: initial guess for omega
        y_max_levels: iterable with length mu+1 of thresholds per level
        eps: epsilon in formulas
        delta: delta used in phi_i = 3^i * delta
        device: torch.device (defaults to cuda if available)
        omega_validity_check: optional function to check omega; if None, default conservative check used
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        

        # store servers as points on device
        assert server_points.ndim == 2
        self.server_points = server_points#.to(self.device).int()
        self.S = self.server_points.shape[0]
        self.dim = self.server_points.shape[1]

        self.graph = graph

        '''
        # -------------------------------
        # preprocess server requests shortest paths
        self.graph = Graph()
        self.graph.make_graph_demo()
        self.graph.load_first_n_cache_to_gpu('first_n_dists.pkl', device=device)
        #self.graph.load_preprocessed(cache_path='cache/preprocessed.pkl')   
        '''

        #self.file = open(f'{self.S}.txt', 'w')
        #self.str = ""

        # Defined for Hungarian Matching
        self.hungarian_cost_pending = False
        self.Hungarian_cost_matrix = None
        self.Hungarian_matching = None


        # parameters for distance scaling
        self.delta = float(delta)
        self.eps = float(self.epsilon())
        

        self.mu = int(self.mu())
        self.L = self.mu + 1
        self.omega = float(omega_init)

        self.y_max_levels = torch.as_tensor(self.y_max_levels(), device=self.device, dtype=torch.float32)
        assert self.y_max_levels.numel() == self.L, "y_max_levels length must be mu+1"


        # precompute phi_i values as floats on host (small list)
        # phi_i = 3^i * delta
        self.phi = [ (3 ** i) * self.delta for i in range(self.L) ]

        # accumulate base distances (R, S) on device as requests arrive
        self.distances_all = None
        self.R = 0

        # dynamic algorithmic state (allocated when R grows)
        self._alloc_empty_state()

        # omega check hook
        if omega_validity_check is None:
            self.omega_validity_check = self._default_omega_check
        else:
            self.omega_validity_check = omega_validity_check

        '''
        # Timers
        self.time_finding_admissible_edges = 0
        self.time_prev_admissible_edges = 0
        self.time_using_randomization = 0
        self.time_prev_rendomization = 0
        self.time_updating_servers = 0
        self.time_to_find_matching = 0
        self.time_updating_requests = 0
        self.time_update_duals = 0
        self.time_omega_and_reset = 0
        '''

    def _alloc_empty_state(self):
        """Initialize empty state."""
        device = self.device
        S = self.S
        L = self.L

        self.y_srv = torch.zeros((L, S), dtype=torch.int32, device=device)
        self.y_req = None
        self.slack = None

        self.matched_req = torch.empty((0,), dtype=torch.long, device=device)
        self.matched_level = torch.empty((0,), dtype=torch.long, device=device)
        self.request_level = torch.empty((0,), dtype=torch.long, device=device)

        self.matched_srv = -torch.ones((S,), dtype=torch.long, device=device)
        self.srv_matched_level = -torch.ones((S,), dtype=torch.long, device=device)

        self.R = 0
        self.distances_all = None

    def _ensure_capacity_for_new_requests(self, new_R_total: int):
        """Grow internal arrays for new_R_total requests if needed."""
        if new_R_total <= self.R:
            return
        device = self.device
        S = self.S
        L = self.L
        old_R = self.R
        add = new_R_total - old_R

        if self.y_req is None:
            self.y_req = torch.zeros((L, new_R_total), dtype=torch.int32, device=device)
        else:
            new_y = torch.zeros((L, new_R_total), dtype=torch.int32, device=device)
            new_y[:, :old_R] = self.y_req
            self.y_req = new_y

        if self.slack is None:
            self.slack = torch.zeros((L, new_R_total, S), dtype=torch.int32, device=device)
        else:
            new_slack = torch.zeros((L, new_R_total, S), dtype=torch.int32, device=device)
            new_slack[:, :old_R, :] = self.slack
            self.slack = new_slack

        if self.matched_req.numel() == 0:
            self.matched_req = -torch.ones((new_R_total,), dtype=torch.long, device=device)
            self.matched_level = -torch.ones((new_R_total,), dtype=torch.long, device=device)
            self.request_level = torch.zeros((new_R_total,), dtype=torch.long, device=device)
        else:
            mr = -torch.ones((new_R_total,), dtype=torch.long, device=device)
            mr[:old_R] = self.matched_req
            self.matched_req = mr
            ml = -torch.ones((new_R_total,), dtype=torch.long, device=device)
            ml[:old_R] = self.matched_level
            self.matched_level = ml
            rl = torch.zeros((new_R_total,), dtype=torch.long, device=device)
            rl[:old_R] = self.request_level
            self.request_level = rl

        self.R = new_R_total

    # -------------------------------
    # Euclidean distance computation
    # -------------------------------
    def _euclidean_distances(self, req_points: torch.Tensor):
        """Compute Euclidean distances between req_points (B,d) and self.server_points (S,d).
           Returns tensor (B,S) on device (float).
        """
        # both on device
        rp = req_points.to(self.device).float()
        sp = self.server_points  # (S,d)
        # Computes ((rp[:,None,:] - sp[None,:,:])**2).sum(dim=2).sqrt()
        # vectorized
        diff = rp.unsqueeze(1) - sp.unsqueeze(0)    # (B, S, d)
        d2 = (diff * diff).sum(dim=2)
        return torch.sqrt(d2)
    
    # -------------------------------
    # Shortest path distance computation
    # -------------------------------
    def _shortest_path_distances(self, req_points: torch.Tensor):
        """Compute shortest-path distances from each request (given as vertex ids) to all servers (vertex ids).
           Expects:
             - req_points: 1D tensor of vertex ids (or shape (B,1))
             - self.server_points either:
                 * 1D tensor of server vertex ids (preferred), or
                 * absent, in which case self.server_node_ids or graph cache is used.
           Requires: graph cache loaded to GPU via Graph.load_first_n_cache_to_gpu(...)
           Returns: torch.Tensor (B,S) on self.device
        """
        if not (hasattr(self, 'graph') and getattr(self.graph, '_first_n_cache_gpu', None) is not None):
            raise RuntimeError("Graph GPU cache not loaded. Call graph.load_first_n_cache_to_gpu(...) before using id-based shortest-path distances.")
        #print(f"first few req_points: {req_points[:10]} ...")
        #print(f"first few server_points: {self.server_points[:10]} ...")

        # Normalize req_points -> python list of ids
        if isinstance(req_points, torch.Tensor):
            rp = req_points.view(-1)
            if rp.dtype.is_floating_point:
                req_list = [str(int(x.item())) for x in rp]
            elif rp.dtype in (torch.long, torch.int32, torch.int64, torch.int16, torch.int8):
                req_list = [str(int(x.item())) for x in rp]
            else:
                # fallback: bring to cpu numpy then to python ints
                req_list = [str(int(x)) for x in rp.cpu().numpy().reshape(-1)]
        else:
            # assume iterable of ints
            req_list = list(req_points)
            #req_list = [str(x) for x in req_list]

        # Resolve server ids (must be vertex ids, not coordinates)
        server_ids = None
        # 1) explicit mapping provided
        if getattr(self, 'server_node_ids', None) is not None:
            #server_ids = list(self.server_node_ids)
            server_ids = [str(x) for x in self.server_node_ids]
        else:
            # 2) if self.server_points is present and is a 1D tensor of ids, use it
            if getattr(self, 'server_points', None) is not None:
                sp = self.server_points
                if isinstance(sp, torch.Tensor):
                    sp_flat = sp.view(-1)
                    if sp_flat.dtype.is_floating_point or sp_flat.dtype in (torch.long, torch.int64, torch.int32):
                        server_ids = [str(int(x.item())) for x in sp_flat]
            # 3) fallback: infer from graph first-n cache if available
            if server_ids is None and getattr(self.graph, '_first_n_cache', None) is not None:
                nodes = self.graph._first_n_cache['nodes']
                if len(nodes) >= self.S:
                    #server_ids = list(nodes[: self.S])
                    server_ids = [str(x) for x in nodes[: self.S]]

        if server_ids is None:
            raise RuntimeError("Cannot resolve server vertex ids. Provide self.server_node_ids or set self.server_points to a 1D tensor of server ids, or load graph first-n cache.")

        #print("_shortest_path_distances: Querying graph GPU cache for requests:", req_list[:10], "..., servers:", server_ids[:10], "...")
        # Query graph GPU cache (returns tensor on cache device)
        dist_tensor = self.graph.get_distances_from_gpu_cache_parallel(req_list, server_node_ids=server_ids)
        #print("dist_tensor :", dist_tensor)

        # Ensure the returned tensor is on our target device
        if dist_tensor.device != self.device:
            dist_tensor = dist_tensor.to(self.device)
        '''
        # Optional safe printing for debugging (small sample)
        try:
            cpu_view = dist_tensor.detach().cpu()
            r = min(3, cpu_view.shape[0]); c = min(6, cpu_view.shape[1])
            print(f"_shortest_path_distances: shape={tuple(cpu_view.shape)}, sample {r}x{c}:\n", cpu_view[:r, :c].numpy())
        except Exception:
            pass
        input("Press Enter to continue...")
        '''
        return dist_tensor
    # -------------------------------
    # L1 distance computation
    # -------------------------------
    def _l1_distances(self, req_points: torch.Tensor):
        """Compute L1 (Manhattan) distances between req_points (B,d) and self.server_points (S,d).
            Returns tensor (B,S) on device (float).
        """
        # both on device
        rp = req_points.to(self.device).float()
        sp = self.server_points  # (S,d)
        # Computes (|rp[:,None,:] - sp[None,:,:]|).sum(dim=2)
        # vectorized
        diff = rp.unsqueeze(1) - sp.unsqueeze(0)    # (B, S, d)
        d1 = diff.abs().sum(dim=2)
        return d1

    # -------------------------------
    # Distance scaling as per formula you provided
    # -------------------------------
    def _compute_d_l_for_level(self, level: int, base_distances: torch.Tensor) -> torch.Tensor:
        """
        base_distances: (B, S) float tensor (Euclidean distances)
        level: 0..mu
        returns: int32 tensor (B, S) computed by recursion in the formula.
        Implements:
          d_0 = ceil( (2 * d * n) / (eps * omega) )
          d_i = ceil( d_{i-1} / ( 2 * (1+eps) * 2^n * phi_{i-1} ) ) for i>0
        where phi_{i-1} = 3^{i-1} * delta, and n = S (number of servers).
        """
        # cast to float on device
        device = self.device
        base = base_distances.to(device).float()
        n_val = float(self.S)   # use number of servers for n
        eps = float(self.eps)
        omega = float(self.omega)
        delta = float(self.delta)

        # compute d0
        # d0 = ceil( (2 * base * n) / (eps * omega) )
        numerator = 2.0 * base * n_val   # (B,S)
        denom0 = eps * omega
        d_prev = torch.ceil(numerator / denom0).to(dtype=torch.int32, device=device)

        if level == 0:
            return d_prev

        # compute subsequent levels iteratively (vectorized across B,S)
        # Note: phi_{i-1} = 3^{i-1} * delta
        for i in range(1, level + 1):
            phi_prev = (3 ** (i - 1)) * delta
            denom = 2.0 * ((1.0 + eps) ** 2.0) * (n_val ** float(phi_prev))
            # divide integer d_prev by denom, use ceil
            d_prev = torch.ceil(d_prev.to(dtype=torch.float32) / denom).to(dtype=torch.int32)
        return d_prev
    
    # -------------------------------
    # Define big_phi function
    # -------------------------------
    def big_phi(self, level: int) -> float:
        return math.ceil(((3**level - 1) / 2) * self.delta)
    
    # ---------------------------
    # Compute small_phi
    # ---------------------------
    def phi(self, i: int) -> float:
        return (3 ** i) * self.delta
        
    # ----------------------------
    # Compute small_phi for each level
    # ----------------------------
    def phi_list(self) -> list[float]:
        return [self.phi(i, self.delta) for i in range(self.mu)]

    # ----------------------------
    # Compute the value of mu
    # ----------------------------
    def mu(self) -> int:
        return math.ceil(float(math.ceil(math.log(2.0 / self.delta) / math.log(3.0)) + 1))

    # ----------------------------
    # Compute epsilon
    # ----------------------------
    def epsilon(self) -> float:
        return 1.0 / (math.log(1.0 / self.delta) / math.log(3.0))

    # ----------------------------
    # Compute the value of y_max
    # ----------------------------
    def y_max(self, level: int) -> float:
        return (30 / self.eps) * (self.S ** self.phi(level))

    # ----------------------------
    # Compute y_max for each level
    # ----------------------------
    def y_max_levels(self) -> list[float]:
        return [self.y_max(i) for i in range(self.mu+1)]

    # -------------------------------
    # default omega check (conservative)
    # -------------------------------
    def _default_omega_check(self) -> bool:
        """Default check: number of requests at or above any level should not exceed S."""
        if self.R == 0:
            return True
        for l in range(self.L):
            if int((self.request_level >= l).sum().item()) > (self.S ** (1-self.big_phi(l))):
                return False
        return True

    # -------------------------------
    # Public: add a batch of request points (B, dim). Batch processed immediately.
    # -------------------------------
    def add_batch(self, request_points: torch.Tensor, verbose: bool = False):
        """
        request_points: (B, dim) float tensor
        """

        assert request_points.ndim == 2 and request_points.shape[1] == self.dim
        device = self.device
        #print(f"Adding first few request points: {request_points[:10]} ...")

        # compute Euclidean distances (B,S) on device
        #dists = self._euclidean_distances(request_points)  # (B,S) floats on device
        #dists = self._l1_distances(request_points)  # (B,S) floats on device
        dists = self._shortest_path_distances(request_points)  # (B,S) floats on device
        #print(request_points)
        #print(self.slack)
        #input()

        # append to distances_all
        if self.distances_all is None:
            self.distances_all = dists.clone()
        else:
            self.distances_all = torch.cat((self.distances_all, dists), dim=0)

        new_R = self.distances_all.shape[0]
        self._ensure_capacity_for_new_requests(new_R)

        # initialize new requests: unmatched, level 0
        start = new_R - dists.shape[0]
        end = new_R
        self.matched_req[start:end] = -1
        self.matched_level[start:end] = -1
        self.request_level[start:end] = 0

        # slack[0, start:end, :] += d0 for these new requests
        d0 = self._compute_d_l_for_level(0, dists)  # int32 (B,S)
        #print (start, end)
        #print (d0)
        #input()
        self.slack[0, start:end, :] += d0
        #print (self.slack)
        #input()

        if verbose:
            print(f"Added batch size {dists.shape[0]}, total R={self.R}")

        '''
        # Resetting timers
        self.time_finding_admissible_edges = 0
        self.time_prev_admissible_edges = 0
        self.time_using_randomization = 0
        self.time_prev_rendomization = 0
        self.time_updating_servers = 0
        self.time_to_find_matching = 0
        self.time_updating_requests = 0
        self.time_update_duals = 0
        self.time_omega_and_reset = 0
        '''
        # Immediately process all accumulated requests (including these new ones)
        self._process_all_requests_until_stable(verbose=verbose)

    # -------------------------------
    # Core processing (same high level as before)
    # -------------------------------
    def _process_all_requests_until_stable(self, verbose: bool = False):
        device = self.device
        S = self.S
        L = self.L



        #count = 0

        #with open(f'{self.S}.txt', 'w') as f:
        #    f.write('')

        #f.close()

        # try until omega valid (doubling & restart on invalidity)
        while True:

            #print(self.slack)

            need_restart = False

            #self.time_prev_admissible_edges = 0
            #self.time_prev_rendomization = 0

            # iterate levels
            for l in range(L):
                if verbose:
                    print(f"Processing level {l} with omega={self.omega}")
                # iteration loop inside a level until no more free requests at level l can change state
                #ITER_CAP = max(4096, self.R * 4 + 1000)
                #print(self.R)
                #iter_count = 0

                while True:
                    """
                    iter_count += 1
                    if iter_count > ITER_CAP:
                        if verbose:
                            print(f"Reached iteration cap at level {l}")
                        break
                    """
                    #start_time_finding_admissible_edges = time.perf_counter()
                    # identify free requests at this level
                    free_mask = (self.matched_req == -1) & (self.request_level == l)
                    if int(free_mask.sum().item()) == 0:
                        break   # nothing to do at this level

                    
                        #f.write('This is the first line.\n')
                        #f.write('This is the second line.\n')
                        #f.write(str(123) + '\n') # Convert non-string data to string

                    if(self.S >= self.R) :
                        free_req_mask = (self.matched_req == -1)        # free_req_mask: (R,), boolean
                        free_srv_mask = (self.matched_srv == -1)        # free_srv_mask: (S,), boolean
                        if int(free_srv_mask.sum().item()) <= 400 :
                            # Select submatrix for free requests × free servers
                            distances_free = self.distances_all[free_req_mask][:, free_srv_mask]
                            # Convert to numpy for POT (ensure on CPU)
                            #print(distances_free)
                            self.Hungarian_cost_matrix = distances_free.detach().cpu().numpy()
                            
                            # Save mapping from submatrix rows/cols -> global indices for cost calc
                            self._hungarian_row_idx = torch.nonzero(free_req_mask, as_tuple=False).squeeze(1).cpu().numpy()
                            self._hungarian_col_idx = torch.nonzero(free_srv_mask, as_tuple=False).squeeze(1).cpu().numpy()

                            self.hungarian_cost_pending = True
                            break
                        #self.str = self.str + str(free_mask.sum().item()) + ', '
                        
                        #print(free_mask.sum().item(), end=", ")
                        #count+=1
                        #if (count == 30) :
                            #count = 0
                            #self.str = self.str + '\n'


                    # server availability mask: free servers OR servers matched at level >= l
                    srv_free_mask = (self.matched_srv == -1)   # (S,)
                    srv_matched_ge_l = (self.srv_matched_level >= l)
                    server_available = srv_free_mask | srv_matched_ge_l   # (S,)

                    # admissible: slack == 1 at level l AND server_available
                    slack_l = self.slack[l]  # (R,S) int32
                    admissible = (slack_l == -1) & server_available.unsqueeze(0)  # (R,S)

                    #print(admissible)
                    

                    # eligible positions for free requests
                    admissible_free = admissible & free_mask.unsqueeze(1)  # (R,S)

                    #end_time_finding_admissible_edges = time.perf_counter()
                    #self.time_finding_admissible_edges += (end_time_finding_admissible_edges - start_time_finding_admissible_edges)
                    #self.time_prev_admissible_edges += (end_time_finding_admissible_edges - start_time_finding_admissible_edges)


                    #start_time_update_duals = time.perf_counter()
                    

                    # CASE A: free requests with NO eligible server => slack[row,:] -=1 ; y_req[l,row]+=1 ; possibly push
                    # no eligible servers -> slack-- and y_req++
                    # has_eligible per row
                    has_eligible = admissible_free.any(dim=1)
                    is_eligible_mask = free_mask & has_eligible
                    is_eligible_idxs = torch.nonzero(is_eligible_mask, as_tuple=False).squeeze(1)
                    #print(is_eligible_idxs)
                    no_eligible_mask = free_mask & (~has_eligible)
                    no_eligible_idxs = torch.nonzero(no_eligible_mask, as_tuple=False).squeeze(1)
                    if no_eligible_idxs.numel() > 0 and not is_eligible_idxs.numel():
                        # Select the rows of free servers and columns of eligible servers
                        #print(self.slack[l])
                        slack_subset = self.slack[l, no_eligible_idxs, :][:, server_available]
                        # Choose the minimum slack value to increment
                        slack_min_value = slack_subset.min()
                        #print(slack_min_value)
                        #print(no_eligible_idxs)
                        #input()
                        #print(slack_min_value)
                        # decrease slack row by slack_min_value
                        #print(self.slack)
                        self.slack[l, no_eligible_idxs, :] -= (slack_min_value+1)
                        #print(self.slack)
                        # increment request dual at this level by slack_min_value
                        self.y_req[l, no_eligible_idxs] += (slack_min_value+1)

                        # push those that cross y_max[l] to next level (if any)
                        if l < (L - 1):
                            # crossing condition: y_req[l, r] > y_max_levels[l]
                            crossing_local_mask = (self.y_req[l, no_eligible_idxs].float() > self.y_max_levels[l])
                            if crossing_local_mask.any():
                                local_indices = torch.nonzero(crossing_local_mask, as_tuple=False).squeeze(1)
                                to_push = no_eligible_idxs[local_indices]
                                # increment their level
                                self.request_level[to_push] += 1
                                # compute d_{l+1} for these requests and add to slack
                                base_dists = self.distances_all[to_push, :]
                                d_next = self._compute_d_l_for_level(l + 1, base_dists)
                                self.slack[l + 1, to_push, :] += d_next
                                # their y_req at new level remains as initialized (0)
                        else:
                            # if at highest level mu and they cross threshold, we keep them at mu (no further push)
                            pass

                        #end_time_update_duals = time.perf_counter()
                        #self.time_update_duals += (end_time_update_duals - start_time_update_duals)
                        
                        #input()
                        continue

                    # Recompute free_mask because some were pushed
                    free_mask = (self.matched_req == -1) & (self.request_level == l)
                    if int(free_mask.sum().item()) == 0:
                        break

                    #print(free_mask)

                    # CASE B: free requests that have eligible servers -> random choices & conflict resolution
                    allowed_mask = admissible_free  # (R,S)

                    if not allowed_mask.any():
                        # nothing to do this iteration (safe-break)
                        break

                    #start_time_finding_admissible_edges = time.perf_counter()
                    #start_time_using_randomization = time.perf_counter()
                    # vectorized random selection among eligible servers per free request
                    # Per free row: choose one eligible server uniformly at random
                    Rdim = self.R
                    # random matrix for per-row choice
                    U = torch.rand((Rdim, S), device=device)
                    # mask invalid positions with -1 (so argmax picks among valid ones uniformly)
                    U_masked = torch.where(allowed_mask, U, torch.full_like(U, -1.0))
                    chosen_server_per_row = torch.argmax(U_masked, dim=1)  # (R,)
                    row_has_choice = allowed_mask.any(dim=1)  # (R,)
                    chosen_server_per_row = torch.where(row_has_choice, chosen_server_per_row, -torch.ones_like(chosen_server_per_row))

                    # Build chosen_mask: (R,S) True where row r chose server s
                    server_ids = torch.arange(S, device=device).unsqueeze(0)  # (1,S)
                    chosen_mask = (chosen_server_per_row.unsqueeze(1) == server_ids) & row_has_choice.unsqueeze(1)

                    # For each server column, pick uniformly one chooser (among rows that chose it)
                    V = torch.rand((Rdim, S), device=device)
                    V_masked = torch.where(chosen_mask, V, torch.full_like(V, -1.0))
                    chosen_request_per_server = torch.argmax(V_masked, dim=0)  # (S,)
                    col_has_chooser = chosen_mask.any(dim=0)
                    chosen_request_per_server = torch.where(col_has_chooser, chosen_request_per_server, -torch.ones_like(chosen_request_per_server))
                    #end_time_using_randomization = time.perf_counter()
                    #self.time_using_randomization += (end_time_using_randomization - start_time_using_randomization)
                    #self.time_prev_rendomization += (end_time_using_randomization - start_time_using_randomization)

                    # Now finalize matches for servers that have chooser
                    servers_selected = torch.nonzero(col_has_chooser, as_tuple=False).squeeze(1)
                    if servers_selected.numel() == 0:
                        # nothing selected this iteration
                        continue
                    #end_time_finding_admissible_edges = time.perf_counter()
                    #self.time_finding_admissible_edges += (end_time_finding_admissible_edges - start_time_finding_admissible_edges) 
                    #self.time_prev_admissible_edges += (end_time_finding_admissible_edges - start_time_finding_admissible_edges) 

                    #start_time_to_find_matching = time.perf_counter()
                    requests_selected = chosen_request_per_server[servers_selected]  # global request indices

                    # Save previous matched requests for impacted servers (to free them if they exist)
                    prev_req = self.matched_srv[servers_selected]  # may be -1
                    prev_mask = (prev_req >= 0)

                    # Assign new matches
                    self.matched_req[requests_selected] = servers_selected
                    self.matched_level[requests_selected] = l
                    self.matched_srv[servers_selected] = requests_selected
                    self.srv_matched_level[servers_selected] = l
                    

                    # Previous matched requests (if any) become free (unmatched), keep their request_level as is
                    if prev_mask.any():
                        freed_srv = torch.nonzero(prev_mask, as_tuple=False).squeeze(1)
                        prev_req_idxs = prev_req[prev_mask]
                        self.matched_req[prev_req_idxs] = -1
                        #self.matched_level[prev_req_idxs] = -1
                        # they will be processed in next iterations (they are free at their current levels)

                    #end_time_to_find_matching = time.perf_counter()
                    #self.time_to_find_matching = end_time_to_find_matching - start_time_to_find_matching

                    #start_time_update_duals = time.perf_counter()
                    # Dual / slack updates:
                    # For each newly matched pair (r -> s):
                    #   slack[l, r, :] += 1
                    #   y_srv[l, s] -= 1
                    self.slack[l, :, servers_selected] += 1
                    self.y_srv[l, servers_selected] -= 1
                    #print(self.slack)
                    #print(self.distances_all)
                    #end_time_update_duals = time.perf_counter()
                    #self.time_update_duals = end_time_update_duals - start_time_update_duals


                    # Loop continues: there may be more free requests at this level
                # end while iterating within level

                if self.hungarian_cost_pending :
                    break

                # After finishing pushes for this level, verify omega validity
                if not self.omega_validity_check():
                    # invalid: double omega and restart entire processing for all requests
                    if verbose:
                        print(f"Omega {self.omega} invalid at level {l}. Doubling and restarting.")
                    self.omega *= 2.0
                    need_restart = True
                    '''
                    if(self.S==self.R) :
                        #with open(f'{self.S}.txt', 'a') as f:
                        #    f.write('\n\n\n Omega goes to : '+str(self.omega)+'\n\n\n')
                        #self.str = self.str + '\n\n\n Omega goes to : '+str(self.omega)+'\n\n\n'
                        print("\nOmega doubled to : ", self.omega)
                        #f.close()
                    '''
                    # reset run state but keep distances_all
                    #start_time_omega_and_reset = time.perf_counter()
                    self.matched_req[:] = -1
                    self.matched_level[:] = -1
                    self.matched_srv[:] = -1
                    self.srv_matched_level[:] = -1
                    self.y_srv[:] = 0
                    if self.y_req is not None:
                        self.y_req[:] = 0
                    self.request_level[:] = 0
                    if self.slack is not None:
                        self.slack[:] = 0

                    # recompute slack[0] from base distances for all requests
                    if self.R > 0:
                        d0_all = self._compute_d_l_for_level(0, self.distances_all)
                        self.slack[0, :self.R, :] += d0_all

                    #end_time_omega_and_reset = time.perf_counter()
                    #self.time_omega_and_reset += (end_time_omega_and_reset - start_time_omega_and_reset)

                    break  # break level loop => outer while will restart
                # else proceed to next level

            if self.hungarian_cost_pending :
                break

            if need_restart:
                # restart processing all requests with doubled omega
                continue
            else:
                # success under current omega
                if verbose:
                    matched_total = int((self.matched_req >= 0).sum().item())
                    print(f"Processing complete under omega={self.omega}. Total matched: {matched_total}/{self.R}")
                return  # done processing for this batch (and overall accumulated requests)
            
        if self.hungarian_cost_pending :
            #self.compute_Hungarian()
            self.compute_Hungarian_cost_only()
            #self.hungarian_cost_pending = True
            if verbose:
                print("Moved to Hungarian matching for final assignment.")

    '''
    def compute_Hungarian(self) :
        r = np.ones(self.Hungarian_cost_matrix.shape[0])# / self.Hungarian_cost_matrix.shape[0]  # uniform supply
        s = np.ones(self.Hungarian_cost_matrix.shape[1])# / self.Hungarian_cost_matrix.shape[1]  # uniform demand
        print(r.shape, s.shape)
        self.Hungarian_matching = ot.emd(r, s, self.Hungarian_cost_matrix)
    '''
    '''
    def compute_Hungarian(self):
        C = self.Hungarian_cost_matrix
        row_ind, col_ind = linear_sum_assignment(C)  # works for rectangular (n,m)
        matching = np.zeros_like(C)
        matching[row_ind, col_ind] = 1
        self.Hungarian_matching = matching
        #self.Hungarian_cost = float((matching * C).sum())
        #return matching
    '''

    def compute_Hungarian_cost_only(self):
        """
        Compute Hungarian on the stored Hungarian_cost_matrix and store:
        - self.Hungarian_matching (binary matrix same shape)
        - self.Hungarian_cost_sum (float)
        - self.Hungarian_matching_pairs (list of (global_req, global_srv))
        This function intentionally DOES NOT modify matched_req / matched_srv.
        """
        if not getattr(self, "hungarian_cost_pending", False):
            # nothing to do
            self.Hungarian_matching = None
            self.Hungarian_cost_sum = 0.0
            self.Hungarian_matching_pairs = []
            return

        C = self.Hungarian_cost_matrix
        if C is None or C.size == 0:
            self.Hungarian_matching = np.zeros_like(C, dtype=np.int32)
            self.Hungarian_cost_sum = 0.0
            self.Hungarian_matching_pairs = []
            # keep the pending flag as-is; recompute the next time if still needed
            return

        row_ind, col_ind = linear_sum_assignment(C)   # indices into C
        matching = np.zeros_like(C, dtype=np.int32)
        matching[row_ind, col_ind] = 1
        self.Hungarian_matching = matching

        # Map back to global indices for cost accounting & diagnostics
        global_rows = self._hungarian_row_idx[row_ind] if hasattr(self, "_hungarian_row_idx") else row_ind
        global_cols = self._hungarian_col_idx[col_ind] if hasattr(self, "_hungarian_col_idx") else col_ind

        # compute cost sum reliably from your authoritative distances_all structure
        # ensure distances_all is accessible as a numpy or torch array
        # convert to numpy if needed
        if hasattr(self.distances_all, "detach"):
            distances_np = self.distances_all.detach().cpu().numpy()
        else:
            distances_np = np.asarray(self.distances_all)

        pairs = []
        cost_sum = 0.0
        for r_global, s_global in zip(global_rows, global_cols):
            r = int(r_global); s = int(s_global)
            pairs.append((r, s))
            cost_sum += float(distances_np[r, s])

        self.Hungarian_matching_pairs = pairs
        self.Hungarian_cost_sum = cost_sum

        # Keep the 'hungarian_cost_pending' flag TRUE if you want the cost to be recomputed
        # every time get_matching_cost() is called until you explicitly clear it.
        # To force recompute only once, set:
        # self.hungarian_cost_pending = False

    '''        
    def dump_to_file(self) :
        #print(self.str)
        #self.file.write(self.str)
        #self.file.close()
        pass
    '''    

            
    '''
    def show_time(self) :
        print()
        print("Finding admissible edges : ", self.time_finding_admissible_edges)
        print("Randomization part : ", self.time_using_randomization)
        print("Updating duals : ", self.time_update_duals)
        print("Computing matching : ", self.time_to_find_matching)
        print("During omega double and reset : ", self.time_omega_and_reset)
        print()
    '''

    # -------------------------------
    # Compute matching cost
    # -------------------------------
    '''
    def get_matching_cost(self,  verbose: bool = False):
        """Compute total matching cost directly from tensors on GPU."""
        if self.R == 0:
            return 0.0

        # mask of matched requests
        mask = self.matched_req >= 0
        if not mask.any():
            return 0.0

        req_ids = torch.nonzero(mask, as_tuple=False).squeeze(1)   # matched request indices
        srv_ids = self.matched_req[req_ids]                        # corresponding servers

        # gather distances in one vectorized op
        dists = self.distances_all[req_ids, srv_ids]

        

        matching_cost = float(dists.sum().item())

        #print(total_cost)

        if self.move_to_Hungarian :
            
            #matches = [(i, j, self.Hungarian_matching[i, j]) 
            #        for i in range(self.Hungarian_matching.shape[0]) 
            #        for j in range(self.Hungarian_matching.shape[1]) 
            #        if self.Hungarian_matching[i, j] > 0]
            
            #print(self.Hungarian_matching)
            #print(self.Hungarian_cost_matrix)
            #print((self.Hungarian_matching*self.Hungarian_cost_matrix).sum())
            matching_cost = matching_cost + float((self.Hungarian_matching*self.Hungarian_cost_matrix).sum())
        
        if verbose :
            print(f"Cost after batch {self.R}: ", matching_cost)

        return matching_cost
    '''
    
    def get_matching_cost(self,  verbose: bool = False):
        """
        Computes the cost of the *committed* global matching (from matched_req)
        plus the cost-only Hungarian contribution (if pending), without altering state.
        """
        # 1) cost of committed matches
        cost_committed = 0.0
        if hasattr(self, "matched_req"):
            # iterate over all requests that are matched
            # use distances_all to compute cost
            if hasattr(self.distances_all, "detach"):
                distances_np = self.distances_all.detach().cpu().numpy()
            else:
                distances_np = np.asarray(self.distances_all)
            for r_idx, s_val in enumerate(self.matched_req):
                s = int(s_val)
                if s >= 0:
                    cost_committed += float(distances_np[r_idx, s])

        # 2) cost-only Hungarian (recompute if pending)
        if getattr(self, "hungarian_cost_pending", False):
            self.compute_Hungarian_cost_only()

        cost_hungarian = float(getattr(self, "Hungarian_cost_sum", 0.0))

        if verbose:
            print(f"Matching cost after batch {self.R}: committed={cost_committed}, hungarian={cost_hungarian}, total={cost_committed + cost_hungarian}")

        return cost_committed + cost_hungarian
    
    # ---------------------------------
    # Resets python variables and GPU
    # --------------------------------- 
    def reset(self, clear_GPU: bool = True):
        """
        Reset all internal state and free GPU memory.
        If clear_cache=True, also clear PyTorch CUDA caching allocator.
        """
        '''
        # Drop references to big tensors
        self.distances_all = None
        self.y_req = None
        self.slack = None
        self.matched_req = None
        self.matched_level = None
        self.request_level = None
        self.matched_srv = None
        self.srv_matched_level = None
        self.y_srv = None

        # Reset counters
        self.R = 0
        '''
        
        # Delete big tensors
        del self.distances_all
        del self.slack
        del self.y_req
        del self.y_srv
        del self.matched_req
        del self.matched_srv
        del self.matched_level
        del self.request_level
        del self.srv_matched_level

        # Force Python garbage collection
        gc.collect()

        # Free cached CUDA memory (optional but usually helpful)
        if clear_GPU:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_accumulated_memory_stats()
            torch.cuda.ipc_collect()

        #if clear_cache and self.device.type == "cuda":
        #    torch.cuda.empty_cache()


    # -------------------------------
    # Helper
    # -------------------------------
    def get_matches(self):
        """Return matched_req (tensor, R) with server idx or -1."""
        return self.matched_req
    



# -------------------------------
# Demo for point-based input and new d_i formula
# -------------------------------
def demo():
    torch.manual_seed(7)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # servers: S=3 in 1D (for clarity)
    server_pts = torch.tensor([[0.0], [4.0], [10.0]])  # shape (3,1)
    delta = 0.001
    omega0 = 1.0

    OM = OnlineMatchingGPU(server_pts, omega0, delta, device=device)

    # batch1: two request points (1D)
    batch1 = torch.tensor([[1.0], [6.0]])
    print("Adding batch1 ...")
    OM.add_batch(batch1, verbose=True)
    print("Matches after batch1:", OM.get_matches().cpu().numpy())

    # batch2
    batch2 = torch.tensor([[9.0]])
    print("Adding batch2 ...")
    OM.add_batch(batch2, verbose=True)
    print("Matches after batch2:", OM.get_matches().cpu().numpy())


def run_online_matching(num_servers: int, dim: int, batch_size: int, delta: float = 0.001):
    torch.cuda.set_device(0)
    torch.manual_seed(7)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Setup servers (uniform random for demo) ---
    server_pts = torch.rand((num_servers, dim), device=device) * 10.0  # random in [0,10)
    
    # --- Compute parameters ---
    omega0 = 1.0

    OM = OnlineMatchingGPU(server_pts, omega0, delta, device=device)

    print(f"Initialized with {num_servers} servers in {dim}D, batch_size={batch_size}")

    # --- Generate and process request batches until we reach num_servers requests ---
    total_requests = 0
    batch_id = 1
    while total_requests < num_servers:
        # how many requests left to reach num_servers
        remaining = num_servers - total_requests
        current_batch_size = min(batch_size, remaining)

        # generate current batch randomly
        batch = torch.rand((current_batch_size, dim), device=device) * 10.0

        print(f"\nAdding batch{batch_id} with {current_batch_size} requests...")
        OM.add_batch(batch, verbose=True)
        #print("Matches after batch:", OM.get_matches().cpu().numpy())
        print("Cost after batch:", OM.get_matching_cost())

        total_requests += current_batch_size
        batch_id += 1

if __name__ == "__main__":
    #demo()
    run_online_matching(1000, 700, 200)
