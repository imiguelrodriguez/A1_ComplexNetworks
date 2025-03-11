import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


class NetworkAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.G = self._load_network()

        # Compute degree-related statistics
        self.degree_sequence = [d for _, d in self.G.degree()]
        self.min_degree = min(self.degree_sequence)
        self.max_degree = max(self.degree_sequence)
        self.avg_degree = np.mean(self.degree_sequence)

    def _load_network(self):
        G = nx.read_pajek(self.file_path)

        if nx.is_directed(G):
            G = G.to_undirected()

        if G.is_multigraph():
            G = nx.Graph(G)  # Convert multigraph to simple graph

        return G

    def extract_microscopic_characterizations(self):
        """Extracts and prints various microscopic network properties."""
        num_nodes = self.G.number_of_nodes()
        num_edges = self.G.number_of_edges()

        # Compute clustering and assortativity
        avg_clustering = nx.average_clustering(self.G)
        assortativity = nx.degree_assortativity_coefficient(self.G)

        # Compute path length and diameter (if graph is connected)
        if nx.is_connected(self.G):
            avg_path_length = nx.average_shortest_path_length(self.G)
            diameter = nx.diameter(self.G)
        else:
            avg_path_length = None
            diameter = None

        # Print results
        print(f"Number of nodes: {num_nodes}")
        print(f"Number of edges: {num_edges}")
        print(f"Minimum degree: {self.min_degree}")
        print(f"Maximum degree: {self.max_degree}")
        print(f"Average degree: {self.avg_degree:.2f}")
        print(f"Average clustering coefficient: {avg_clustering:.4f}")
        print(f"Assortativity (degree correlation): {assortativity:.4f}")

        if avg_path_length is not None:
            print(f"Average path length: {avg_path_length:.4f}")
            print(f"Diameter: {diameter}")
        else:
            print("Graph is disconnected. Average path length and diameter are not defined.")

    def plot_histograms(self):
        """Plots the degree distribution in both linear and log-log scales."""
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # Linear Scale Histogram
        axs[0].hist(self.degree_sequence, bins=30, edgecolor='black')
        axs[0].set_title("Degree Distribution (Linear Scale)")
        axs[0].set_xlabel("Degree")
        axs[0].set_ylabel("Frequency")

        # Log-Log Scale Histogram with Logarithmic Binning
        bins = np.logspace(np.log10(self.min_degree), np.log10(self.max_degree), num=20)
        axs[1].hist(self.degree_sequence, bins=bins, edgecolor='black', log=True)
        axs[1].set_xscale("log")
        axs[1].set_title("Degree Distribution (Log-Log Scale)")
        axs[1].set_xlabel("Degree (log scale)")
        axs[1].set_ylabel("Frequency (log scale)")

        plt.tight_layout()
        plt.show()
