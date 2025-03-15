from collections import Counter

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson


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
    

    def extract_microscopic_features(self):
        # Compute centralities
        betweenness = nx.betweenness_centrality(self.G)
        degree_centrality = nx.degree_centrality(self.G)
        eigenvector = nx.eigenvector_centrality(self.G, max_iter=1000)
        pagerank = nx.pagerank(self.G)

        # Get top 5 nodes for each centrality measure
        top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]
        top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        top_eigenvector = sorted(eigenvector.items(), key=lambda x: x[1], reverse=True)[:5]
        top_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:5]

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

        print("\n Top 5 Nodes by PageRank:")
        for node, value in top_pagerank:
            print(f"   {node}: {value:.4f}")

        # Compare rankings
        betweenness_nodes = {node for node, _ in top_betweenness}
        degree_nodes = {node for node, _ in top_degree}
        eigenvector_nodes = {node for node, _ in top_eigenvector}
        pagerank_nodes = {node for node, _ in top_pagerank}

        overlap = betweenness_nodes & degree_nodes & eigenvector_nodes & pagerank_nodes
        print("\n Nodes appearing in all four centrality rankings:", overlap if overlap else "None")


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

    def plot_histograms(self, plot_theory=False):
        """Plots the degree distribution with an optional theoretical Poisson distribution."""

        fig, axs = plt.subplots(1, 2, figsize=(14, 6))  # Two side-by-side plots

        ### Compute degree frequencies ###
        degree_counts = Counter(self.degree_sequence)
        degree = sorted(degree_counts.keys())  # Unique degrees
        degree_count = np.array([degree_counts[d] for d in degree])  # Count of each degree

        ### Compute empirical probability ###
        P_k = degree_count / sum(degree_count)  # Normalize to get probabilities

        ### Scatter Plot (Linear Scale) ###
        axs[0].bar(degree, P_k, color="gray", alpha=0.6, label="Network")
        axs[0].set_title("Degree Distribution (Linear Scale)", fontsize=15)
        axs[0].set_xlabel("$k$", fontsize=14)
        axs[0].set_ylabel("$P(k)$", fontsize=14)
        axs[0].set_xlim(left=self.min_degree - 1, right=self.max_degree + 1)
        axs[0].tick_params(axis='both', which='major', labelsize=12)
        axs[0].grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

        if plot_theory:
            # Compute theoretical Poisson distribution
            k_values = np.arange(min(degree), max(degree) + 1)
            theoretical_probs = poisson.pmf(k_values, mu=np.mean(self.degree_sequence))

            # Overlay theoretical distribution
            axs[0].plot(k_values, theoretical_probs, color="blue", lw=2, label="Theory")

        axs[0].legend()

        ### Scatter Plot (Log-Log Scale) ###
        axs[1].scatter(degree, P_k, color="red", marker='o', alpha=0.8, zorder=3)
        axs[1].set_xscale("log")
        axs[1].set_yscale("log")
        axs[1].set_title("Degree Distribution (Log-Log Scale)", fontsize=15)
        axs[1].set_xlabel("$k$", fontsize=14)
        axs[1].set_ylabel("$P(k)$", fontsize=14)
        axs[1].tick_params(axis='both', which='major', labelsize=12)
        axs[1].grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

        # Adjust layout and display
        plt.tight_layout()
        plt.show()


    def fit_CCDF(self):
        """Fits and plots the Complementary Cumulative Distribution Function (CCDF) with a power-law fit."""
        
        G = self.G
        degree_sequence = [G.degree(node) for node in G.nodes()]
        from collections import Counter

        # Compute degree frequencies
        degree_counts = Counter(degree_sequence)
        min_degree = min(degree_sequence)
        max_degree = max(degree_sequence)

        degrees = list(range(min_degree, max_degree + 1))
        degree_count = [degree_counts.get(k, 0) for k in degrees]

        # Remove zero-frequency degrees
        degrees = [degrees[i] for i in range(len(degrees)) if degree_count[i] != 0]
        degree_count = [degree_count[i] for i in range(len(degree_count)) if degree_count[i] != 0]

        # Compute CCDF
        cdf = np.cumsum(degree_count) / G.number_of_nodes()  
        ccdf = 1 - cdf  

        # Prepare log-log fitting
        log_degree_fit = np.log(degrees)[:-1]  # Exclude last point (log(0) is undefined)
        log_ccdf_fit = np.log(ccdf)[:-1]

        # Fit power-law (log-log scale)
        m, b = np.polyfit(log_degree_fit, log_ccdf_fit, 1)  
        theoretical = [np.exp(b) * k ** m for k in degrees]

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))  

        ax.scatter(degrees, ccdf, color='blue', marker='o', alpha=0.8)  # Scatter points
        ax.plot(degrees, theoretical, color='black', linestyle='--', linewidth=2, label=r'$\gamma-1=%.2f$' % (-m))  # Theoretical line

        ax.set_yscale('log')
        ax.set_xscale('log')

        ax.set_xlabel('$k$', fontsize=15)
        ax.set_ylabel('$CCDF(k)$', fontsize=15)

        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.legend(loc='best', fontsize=12)  

        plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)  
        plt.tight_layout()
        plt.show()
