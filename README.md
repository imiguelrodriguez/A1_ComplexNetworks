# A1_ComplexNetworks

This repository contains an assignment for the **Complex Networks** subject in the **MESSIA master's program at URV**. The project focuses on analyzing complex networks by computing macroscopic features such as connectivity, clustering, 
Furthermore, it provides characterization of networks by guessing the models that were used to generate them.

## Contents

- `networks/`: Directory containing network data files.
- `networks_analyzer.ipynb`: Jupyter Notebook for analyzing the provided networks.
- `utils.py`: Python utility functions to support network analysis tasks.

## Requirements

- Python 3.x
- Jupyter Notebook
- NetworkX
- Matplotlib
- NumPy

## Usage

1. **Clone the repository:**

   ```bash
   git clone https://github.com/imiguelrodriguez/A1_ComplexNetworks.git
   cd A1_ComplexNetworks

2. Run the `networks_analyzer.ipynb` notebook.

## Example

```python
import NetworkAnalyzer
net = NetworkAnalyzer(file_path="example.net")

# Extract macroscopic and microcropic features from the network
net.extract_macroscopic_features()
net.extract_microscopic_features()

# Plot degree distribution
net.plot_histograms()

# Plot CCDF
net.fit_CCDF()

# Plot correlations between centrality measures
net.plot_spearman_centrality_correlation()
```
   
## Features
* **Network Analysis**: Compute key metrics such as the number of nodes and edges, degree distribution, clustering coefficient, assortativity, average path length, and diameter. 
* **Visualization**: Generate plots to visualize network structures and their properties. 
* **Utility Functions**: Reusable functions for common tasks in network analysis.

## Acknowledgments
* Developed as part of the Complex Networks subject in the MESSIA master's program at URV. 
* Developed by [Ignacio Miguel Rodríguez](https://github.com/imiguelrodriguez) and [Daniel Arias Cámara](https://github.com/Danie1Arias)
