import os
import time
import torch
import pandas as pd
from PushRelabelBatch import OnlineMatchingGPU   # assumes your class is here
from greedy import GreedyTorch, PointTorch   # import our GPU-based Greedy


def run_experiments_greedy(
    master_folder, 
    subfolders, 
    num_datasets,
    delta, 
    dim, 
    cuda_device="cuda:0", 
    batch_size=1
):
    """
    master_folder: str, path to PlotData
    subfolders: list of str, e.g. ["1000", "2000"]
    num_datasets: int, number of datasets to process in each subfolder
    dim: int, dimension of servers/requests (e.g., 2 for 2D)
    cuda_device: str, e.g. "cuda:0" or "cuda:1"
    batch_size: int, number of requests per batch
    """

    device = torch.device(cuda_device)

    for sub in subfolders:
        folder_path = os.path.join(master_folder, sub)

        for ds_id in range(1, num_datasets + 1):
            # -------------------------------
            # build file names
            input_file = f"{ds_id}_Point_{delta:.6f}_MNIST_L1_{dim}dim.csv"
            output_file = f"{ds_id}_Greedy_{delta:.6f}_MNIST_L1_{dim}dim.csv"

            input_path = os.path.join(folder_path, input_file)
            output_path = os.path.join(folder_path, output_file)

            if not os.path.exists(input_path):
                print(f"Skipping missing {input_path}")
                continue

            print(f"Processing {input_path} on {cuda_device} ...")

            # -------------------------------
            # read dataset
            df = pd.read_csv(input_path, header=0)
            data = df.values
            servers_np = data[:, :dim]
            requests_np = data[:, dim:]

            # Convert servers to PointTorch
            servers = [PointTorch(coords, device) for coords in servers_np]
            greedy = GreedyTorch(servers, device)

            results = []

            # -------------------------------
            # process requests in batches
            for start in range(0, len(requests_np), batch_size):
                end = min(start + batch_size, len(requests_np))
                batch = requests_np[start:end]

                start_time = time.perf_counter()
                for req_coords in batch:
                    req_point = PointTorch(req_coords, device)
                    greedy.processRequest(req_point)
                cost = greedy.getTotalCost(verbose=True)
                elapsed = time.perf_counter() - start_time
                results.append([end, cost, elapsed])
                print()
                print("Time to match last request is : ", elapsed)
                print()

            # -------------------------------
            # save results with header
            out_df = pd.DataFrame(results, columns=["num_requests_matched", "cost", "execution_time"])
            out_df.to_csv(output_path, index=False)

            print(f"Saved results to {output_path}")




def run_experiments_pr(
    master_folder, 
    subfolders, 
    num_datasets, 
    dim, 
    num_servers,
    cuda_device="cuda:1", 
    omega_init=200.0, 
    delta=0.001, 
    batch_size=1
):
    """
    master_folder: str, path to PlotData
    subfolders: list of str, e.g. ["1000", "2000"]
    num_datasets: int, number of datasets to process in each subfolder
    dim: int, dimension of servers/requests (e.g., 2 for 2D)
    cuda_device: str, e.g. "cuda:0" or "cuda:1"
    mu, omega_init, delta: parameters for algorithm
    batch_size: int, number of requests per batch
    """

    device = torch.device(cuda_device)

    for sub in subfolders:
        folder_path = os.path.join(master_folder, sub)

        for ds_id in range(1, num_datasets + 1):
            # -------------------------------
            # build file names
            input_file = f"{ds_id}_Point_{delta:.6f}_MNIST_L1_{dim}dim.csv"
            output_file = f"{ds_id}_PRPR_{delta:.6f}_MNIST_L1_{dim}dim.csv"

            input_path = os.path.join(folder_path, input_file)
            output_path = os.path.join(folder_path, output_file)

            if not os.path.exists(input_path):
                print(f"Skipping missing {input_path}")
                continue

            print(f"Processing {input_path} on {cuda_device} ...")

            # -------------------------------
            # read dataset (skip header row)
            df = pd.read_csv(input_path, header=0, nrows=num_servers)   # skip first row (header)
            data = df.values
            servers = data[:, :dim]
            servers = torch.tensor(data[:, :dim], dtype=torch.float32, device=device)
            requests = data[:, dim:]

            # -------------------------------
            # run algorithm
            OM = OnlineMatchingGPU(servers, omega_init=omega_init, delta=delta, device=device)

            results = []
            #start_time = time.time()

            # process requests in batches
            for start in range(0, len(requests), batch_size):
                end = min(start + batch_size, len(requests))
                batch = torch.tensor(requests[start:end], dtype=torch.float32, device=device)
                start_time = time.perf_counter()
                OM.add_batch(batch)
                #OM.process_current_batch()

                cost = OM.get_matching_cost(verbose=True)
                elapsed = time.perf_counter() - start_time         # <-------- Time in seconds
                print()
                print("Time to match this batch is : ", elapsed)
                print()
                results.append([end, cost, elapsed])

            # -------------------------------
            # save results with header
            out_df = pd.DataFrame(results, columns=["num_requests_matched", "cost", "execution_time"])
            out_df.to_csv(output_path, index=False)

            print(f"Saved results to {output_path}")
            OM.reset()


master = "PlotData"
subfolders = ["10000"]
num_datasets = 10
"""
run_experiments_pr(
    master_folder=master,
    subfolders=subfolders,
    num_datasets=num_datasets, # number of sets to read of size 10000
    dim=784,                  # manually set dimension
    num_servers=10000,       # number of rows to read
    cuda_device="cuda:0",   # choose device
    omega_init=1.0,
    delta=0.001,
    batch_size=200
)
"""


run_experiments_greedy(
        master_folder=master,
        subfolders=subfolders,
        num_datasets=num_datasets,
        delta=0.001,
        dim=784,
        cuda_device="cuda:0",
        batch_size=1
    )
