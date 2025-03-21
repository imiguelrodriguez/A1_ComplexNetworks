from collections import Counter
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson


class NetworkAnalyzer:
    def __init__(self, G=None, file_path=None, positions_file=None, positions=None):
        if file_path:
            self.file_path = file_path
            self.G = self._load_network()
            self.G = nx.relabel_nodes(self.G, {node: int(node) for node in self.G.nodes()})
        else:
            self.G = G

        # Compute degree-related statistics
        self.degree_sequence = [d for _, d in self.G.degree()]
        self.min_degree = min(self.degree_sequence)
        self.max_degree = max(self.degree_sequence)
        self.avg_degree = np.mean(self.degree_sequence)
        self.betweenness_centrality = None
        self.degree_centrality = None
        self.eigenvector_centrality = None
        self.pagerank_centrality = None
        self.closeness_centrality = None
        self.katz_centrality = None
        if positions_file:
            self.set_positions(positions_file)
        else:
            self.positions = positions

    def _load_network(self):
        G = nx.read_pajek(self.file_path)


        if nx.is_directed(G):
            G = G.to_undirected()

        if G.is_multigraph():
            G = nx.Graph(G)  # Convert multigraph to simple graph

        return G

    def set_positions(self, positions_file=None, positions=None):
        if positions_file:
            try:
                df = pd.read_csv(positions_file, sep="\t")  # Adjust separator if needed

                # Convert to dictionary
                p = df.set_index("Node")[["x", "y"]].to_dict(orient="index")

                # Convert {0: {"x": 0.2109, "y": 0.0554}} -> {0: (0.2109, 0.0554)}
                self.positions = {int(k): (v["x"], v["y"]) for k, v in p.items()}
            except Exception as e:
                print(e)
        elif positions:
            self.positions = positions
        else:
            print("You must specify the positions!")

    def extract_microscopic_features(self):
        # Compute centralities
        self.betweenness_centrality = nx.betweenness_centrality(self.G)
        self.degree_centrality = nx.degree_centrality(self.G)
        self.eigenvector_centrality = nx.eigenvector_centrality(self.G, max_iter=1000)
        self.pagerank_centrality = nx.pagerank(self.G)
        self.closeness_centrality = nx.closeness_centrality(self.G)

        # We calculate the largest eigenvalue of the adjacency matrix
        largest_eigenvalue = max(np.linalg.eigvals(nx.to_numpy_array(self.G)).real)

        # We choose a safe alpha (slightly less than 1/largest_eigenvalue)
        alpha_safe = 0.85 / largest_eigenvalue

        self.katz_centrality = nx.katz_centrality(self.G, alpha=alpha_safe, max_iter=5000)

        # Get top 5 nodes for each centrality measure
        top_betweenness = sorted(self.betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        top_degree = sorted(self.degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        top_eigenvector = sorted(self.eigenvector_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        top_pagerank = sorted(self.pagerank_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        top_closeness = sorted(self.closeness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        top_katz = sorted(self.katz_centrality.items(), key=lambda x: x[1], reverse=True)[:5]

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

        print("\n Top 5 Nodes by PageRank Centrality:")
        for node, value in top_pagerank:
            print(f"   {node}: {value:.4f}")

        print("\n Top 5 Nodes by Closeness Centrality:")
        for node, value in top_closeness:
            print(f"   {node}: {value:.4f}")

        print("\n Top 5 Nodes by Katz Centrality:")
        for node, value in top_katz:
            print(f"   {node}: {value:.4f}")

        # Compare rankings
        betweenness_nodes = {node for node, _ in top_betweenness}
        degree_nodes = {node for node, _ in top_degree}
        eigenvector_nodes = {node for node, _ in top_eigenvector}
        pagerank_nodes = {node for node, _ in top_pagerank}
        closeness_nodes = {node for node, _ in top_closeness}
        katz_nodes = {node for node, _ in top_katz}

        overlap = betweenness_nodes & degree_nodes & eigenvector_nodes & pagerank_nodes & closeness_nodes & katz_nodes
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


    def plot_spearman_centrality_correlation(self):

        G = self.G

        centralities = {
            "Degree": self.degree_centrality,
            "Betweenness": self.betweenness_centrality,
            "Eigenvector": self.eigenvector_centrality,
            "PageRank": self.pagerank_centrality,
            "Closeness": self.closeness_centrality,
            "Katz": self.katz_centrality,
        }

        df = pd.DataFrame(centralities)
        correlation_matrix = df.corr(method='spearman')

        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title("Spearman Correlation Matrix of Centrality Measures", fontsize=15)
        plt.show()

    def plot_with_positions(self, r=None):
        if self.positions:

            # Find connected components
            components = list(nx.connected_components(self.G))
            num_components = len(components)

            # Generate unique colors for each component
            cmap = plt.colormaps.get_cmap("tab10")  # No need for num_components

            # Generate colors
            colors = [cmap(i / max(1, num_components - 1)) for i in range(num_components)]
            plt.figure(figsize=(10, 7))
            for i, component in enumerate(components):
                subgraph = self.G.subgraph(component)  # Extract subgraph
                nx.draw(
                    subgraph,
                    pos=self.positions,
                    node_color=[colors[i] for _ in subgraph.nodes],  # Assign color to each node
                    edge_color="gray",
                    with_labels=False,
                    node_size=50
                )
            if r:
                plt.title(f"r={r:.2f} CC={len(list(nx.connected_components(self.G)))}")
            else:
                plt.title(f"CC={len(list(nx.connected_components(self.G)))}")
            plt.show()

        else:
            print("Positions have not been specified!")