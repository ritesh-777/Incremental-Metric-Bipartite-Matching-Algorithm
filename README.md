# Efficient Algorithms for Incremental Metric Bipartite Matching - Implementation

This repository contains implementations and experimental results for three online matching algorithms evaluated on three different datasets: Beijing Road Network, Synthetic, MNIST, and Taxi.

## Repository Structure

```
main/
├── Beijing Road Network/
├── Synthetic/
├── MNIST/
├── Taxi/
└── README.md
```

Each dataset folder contains algorithm implementations, plotting utilities, and experimental results in the `PlotData/` subfolder.

## Algorithms Implemented

1. **Batch Incremental Push-Relabel (PRPR)**: GPU-accelerated implementation using PyTorch
2. **Greedy Algorithm**: Greedy Matching
3. **Quadtree Algorithm (QT)**: CPU-based spatial partitioning approach (C++)
3. **OnlineOptimal Algorithm**: CPU-based spatial optimal online matching algorithm (C++)

## Requirements

### Python Dependencies
```bash
pip install torch pandas matplotlib numpy pot pyproj openai rtree
```

### C++ Dependencies
- g++ compiler with C++20 support
- OpenMP for parallel processing

## Dataset Descriptions

### Synthetic Dataset
- **Description**: Randomly generated 2D point datasets with configurable parameters
- **Parameters**: 
  - `delta`: Distance scaling parameter (default: 0.001)
  - `n`: Number of server/request pairs
  - `dimensions`: Coordinate dimensions (default: 2)

### Beijing Road Network 
- **Description**: Road network of Beijing city
- **Source**: https://github.com/idea-iitd/NeuroMLR?tab=readme-ov-file
- **Preprocessing**: Shortest path computed for each pair of location nodes and stored as first_n_dists.pkl to be used by Python (GPU-accelerated implementation) and as dists.bin to be used by C++ (Sequential implementaion) 
- **Format**: Server and request points sampled from the location nodes

### MNIST Dataset 
- **Description**: MNIST digit images treated as high-dimensional points
- **Source**: http://yann.lecun.com/exdb/mnist/
- **Preprocessing**: Images flattened to 784-dimensional vectors and normalized
- **Format**: Server and request points sampled from the MNIST training set

### Taxi Dataset
- **Description**: NYC taxi pickup/dropoff location data
- **Source**: https://www.kaggle.com/datasets/yasserh/nyc-taxi-trip-duration
- **Format**: Latitude/longitude coordinates converted to 2D points
- **Processing**: Pickup locations serve as servers, dropoff locations as requests

## Usage Instructions

### 1. Synthetic Dataset

#### Running C++ Algorithms (Quadtree)
```bash
cd Synthetic/
g++ -fopenmp -std=c++20 -lpthread driver.cpp -o driver
./driver
```

#### Running Python Algorithms (PRPR & Greedy)
```bash
cd Synthetic/
# Run Push-Relabel algorithm
python driver.py

# Run Greedy algorithm (modify driver.py to uncomment greedy section)
python driver.py
```

#### Configuration Parameters (driver.py)
- `master_folder`: Path to data directory ("PlotData")
- `subfolders`: List of dataset sizes (e.g., ["10000"])
- `num_datasets`: Number of experimental instances (default: 10)
- `delta`: Algorithm parameter (default: 0.001)
- `batch_size`: Requests processed per batch (default: 200)
- `omega_init`: Initial omega value for PRPR (default: 1.0)

#### Generating Plots
```bash
# Generate performance comparison plots
python plotting_Synt.py

# Generate plots with standard deviation
python plot_var_Synt.py
```

### 2. MNIST Dataset

#### Running Experiments
```bash
cd MNIST/
# Run Push-Relabel algorithm
python driver.py

# Modify driver.py to switch between algorithms
```

#### Generating Plots
```bash
# Generate performance comparison plots  
python plotting_MNIST.py

# Generate variance analysis plots
python plot_var_MNIST.py
```

### 3. Taxi Dataset

#### Running Experiments
```bash
cd Taxi/
# C++ algorithms
g++ -fopenmp -std=c++20 -lpthread driver.cpp -o driver
./driver

# Python algorithms
python driver.py
```

#### Generating Plots
```bash
# Generate performance comparison plots
python plotting_Taxi.py

# Generate variance analysis plots  
python plot_var_Taxi.py
```

## Key Files Description

### Algorithm Implementations

- **`PushRelabelBatch.py`**: GPU-accelerated Push-Relabel implementation with batch processing
- **`PushRelabel.h`**: Sequential Push-Relabel implementation (C++ header)
- **`greedy.py`**: Greedy Matching Algorithm
- **`QT_Algo.h`**: Quadtree-based matching algorithm (C++ header)
- **`OnlineOptimal.h`**: Optimal algorithm for online matching (C++ header)
- **`common_structures.h`**: Shared data structures and utilities (C++)

### Data Handling

- **`DataReader.cpp`**: C++ utility for reading CSV datasets
- **`driver.cpp`**: C++ experimental driver
- **`driver.py`**: Python experimental driver with GPU support

### Visualization

- **`plotting_*.py`**: Generate performance comparison plots across algorithms
- **`plot_var_*.py`**: Generate plots with a shadow of the standard deviation

## Output Format

### Results Files
Each algorithm generates CSV files with the following format:
- **Columns**: `n`, `cost`, `execution_time` (or `run_time`)
- **Naming Convention**: `{instance_id}_{algorithm}_{delta}_{dataset_type}_{dimensions}dim.csv`

### Plots Generated
1. **Cost vs. Number of Requests**: Comparative performance analysis
2. **Runtime vs. Number of Requests**: Execution time comparison
3. **Statistical Analysis**: Mean performance with standard deviation bands

## Experimental Parameters

### Default Configuration
- **Delta**: 0.001 (distance scaling parameter)
- **Batch Size**: 200 requests per batch
- **Instances**: 10 independent runs per configuration
- **Dimensions**: 2D for Synthetic/Taxi, 784D for MNIST
- **Problem Sizes**: 1000 to 10000 servers/requests

### GPU Configuration
- **Device**: CUDA-enabled GPU (falls back to CPU if unavailable)
- **Memory Management**: Automatic GPU memory cleanup between experiments
- **Precision**: 32-bit floating point

## Notes

1. **CUDA Compatibility**: Ensure PyTorch CUDA support is properly installed for GPU acceleration
2. **Memory Requirements**: Large datasets may require significant GPU memory
3. **Compilation**: C++ code requires OpenMP support for parallel execution
4. **Data Location**: Ensure `PlotData/` directories contain the required CSV datasets