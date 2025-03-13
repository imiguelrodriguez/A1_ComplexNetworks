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

    def extract_macroscopic_features(self):
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
        """Plots the degree distribution using a combination of histograms and scatter plots 
        in both linear and log-log scales, including trend lines."""
        
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))  # Two plots side by side
        
        degree_set = set(self.degree_sequence)

        ### Compute degree frequencies ###
        from collections import Counter
        degree_counts = Counter(self.degree_sequence)
        degree = sorted(degree_counts.keys())  # Unique degrees
        degree_count = [degree_counts[d] for d in degree]  # Count of each degree
        
        ### Histogram and Scatter (Linear Scale) ###
        axs[0].hist(self.degree_sequence, bins=len(degree_set), edgecolor='black', alpha=0.6, label="Histogram")
        axs[0].scatter(degree, degree_count, color="blue", label="Scatter Data", zorder=3)
        axs[0].plot(degree, degree_count, color="blue", linestyle="-", alpha=0.7)  # Trend line
        
        axs[0].set_title("Degree Distribution (Linear Scale)", fontsize=15)
        axs[0].set_xlabel("$k$", fontsize=13)
        axs[0].set_ylabel("$P(k)$", fontsize=13)
        axs[0].set_xlim(left=self.min_degree - 1, right=self.max_degree + 1)
        axs[0].legend()

        ### Histogram and Scatter (Log-Log Scale) ###
        bins = np.logspace(np.log10(self.min_degree), np.log10(self.max_degree), num=40)
        axs[1].hist(self.degree_sequence, bins=bins, edgecolor='black', alpha=0.6, label="Histogram", log=True)
        axs[1].scatter(degree, degree_count, color="red", label="Scatter Data", zorder=3)
        axs[1].plot(degree, degree_count, color="red", linestyle="-", alpha=0.7)  # Trend line
        
        axs[1].set_xscale("log")
        axs[1].set_yscale("log")
        axs[1].set_title("Degree Distribution (Log-Log Scale)", fontsize=15)
        axs[1].set_xlabel("$k$", fontsize=13)
        axs[1].set_ylabel("$P(k)$", fontsize=13)
        axs[1].legend()

        # Adjust layout and display
        plt.tight_layout()
        plt.show()

    def extract_microscopic_features(self):
        # Compute centralities
        betweenness = nx.betweenness_centrality(self.G)
        degree_centrality = nx.degree_centrality(self.G)
        eigenvector = nx.eigenvector_centrality(self.G, max_iter=1000)

        # Get top 5 nodes for each centrality measure
        top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]
        top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        top_eigenvector = sorted(eigenvector.items(), key=lambda x: x[1], reverse=True)[:5]

        # Print results
        print("\n Top 5 Nodes by Betweenness Centrality:")
        for node, value in top_betweenness:
            print(f"   {node}: {value:.4f}")

        print("\n Top 5 Nodes by Degree Centrality:")
        for node, value in top_degree:
            print(f"   {node}: {value:.4f}")

        print("\n Top 5 Nodes by Eigenvector Centrality:")
        for node, value in top_eigenvector:
            print(f"   {node}: {value:.4f}")

        # Compare rankings
        betweenness_nodes = [node for node, _ in top_betweenness]
        degree_nodes = [node for node, _ in top_degree]
        eigenvector_nodes = [node for node, _ in top_eigenvector]

        overlap = set(betweenness_nodes) & set(degree_nodes) & set(eigenvector_nodes)
        print("\n Nodes appearing in all three centrality rankings:", overlap if overlap else "None")