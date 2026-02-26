import pickle
import networkx as nx
import os
import csv

file_path = './beijing_data/map/graph_with_haversine.pkl'
# --- 1. Load the MultiGraph data from the .pkl file ---
try:
    with open(file_path, 'rb') as f:
        # The 'rb' mode is essential for reading binary files
        multigraph_data = pickle.load(f)

    # Check if the loaded data is a networkx MultiGraph (or MultiDiGraph)
    if isinstance(multigraph_data, (nx.MultiGraph, nx.MultiDiGraph)):
        G = multigraph_data
        print(f"Successfully loaded a {type(G).__name__} from the .pkl file.")
        print("-" * 30)

        # --- 2. Print various aspects of the graph data ---

        print(f"Graph type: {type(G).__name__}")
        print(f"Number of nodes: {G.number_of_nodes()}")
        print(f"Number of edges: {G.number_of_edges()}")
        print("-" * 30)

        print("Nodes (first 10):")
        # Display the first 10 nodes for large graphs
        for i, node in enumerate(G.nodes(data=True)):
            if i >= 10:
                break
            print(node)
        print("-" * 30)

        # Export node IDs to CSV (single column with header "vertices")
        output_csv = os.path.join(os.path.dirname(__file__), 'vertices.csv')
        try:
            with open(output_csv, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['vertices', 'x', 'y'])
                for i, node in enumerate(G.nodes(data=True)):
                    writer.writerow([str(node[0]),str(node[1]['x']),str(node[1]['y'])])
            print(f"Wrote {G.number_of_nodes()} node IDs to '{output_csv}'.")
        except Exception as e:
            print(f"Failed to write CSV: {e}")

        print("Edges (first 10):")
        # Display the first 10 edges with their data
        for i, edge in enumerate(G.edges(data=True)):
            if i >= 10:
                break
            print(edge[0], edge[1], edge[2]['length'], edge[2]['haversine'])
        print("-" * 30)

        # Export edges along with node IDs and distances to CSV (single column with header "edges_with_cost")
        output_csv = os.path.join(os.path.dirname(__file__), 'edges_with_cost.csv')
        try:
            with open(output_csv, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['vertex_1', 'vertex_2', 'length', 'haversine'])
                for i, edge in enumerate(G.edges(data=True)):
                    writer.writerow([str(edge[0]),str(edge[1]),str(edge[2]['length']),str(edge[2]['haversine'])])
            print(f"Wrote {G.number_of_nodes()} edges with distances to '{output_csv}'.")
        except Exception as e:
            print(f"Failed to write CSV: {e}")

        # Print all graph attributes
        #print(f"Graph attributes: {G.graph}")
        print(f"Node attributes: {G.nodes[list(G.nodes)[0]] if G.nodes else 'No nodes'}")

    else:
        print(f"The loaded object is not a networkx graph, it is a {type(multigraph_data).__name__}.")
        print("Raw data sample (first 100 characters):")
        print(str(multigraph_data)[:100])

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except pickle.UnpicklingError:
    print(f"Error: Could not unpickle the file. Ensure it is a valid Python pickle file.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")