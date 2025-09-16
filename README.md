# synthetic-data-generator

SyntheticDataGen is a Python library for generating synthetic datasets with rich, controllable properties. The framework supports class-conditional Gaussian mixtures with multiple modes per class, label noise, outliers, and anisotropic covariances. Additionally, it provides high-dimensional embedding methods to map low-dimensional data into higher dimensions while preserving neighborhood structure.

This repository is ideal for researchers and practitioners who need reproducible, customizable synthetic datasets for benchmarking classifiers, visualization, or testing optimization algorithms.

## Features

- Class-conditional Gaussian mixture generator
- Multiple modes per class
- Control over cluster separation, overlap, and covariance
- Support for anisotropic clusters and label noise
- Heavy-tailed noise and outlier generation
- High-dimensional embeddings: linear and smooth nonlinear methods
- Visualization tools (2D scatter, 3D scatter, PCA, t-SNE)
- Reproducible via RNG seed

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/diyarino/synthetic-data-generator.git
cd synthetic-data-generator
pip install -r requirements.txt
````

Dependencies:

* Python >= 3.8
* NumPy
* Matplotlib
* scikit-learn

## Usage

### Programmatic API

```python
from synthetic_dataset_generator import SyntheticDatasetGenerator, HighDimEmbedder, Visualizer

# Generate a 2D synthetic dataset
gen = SyntheticDatasetGenerator(seed=0)
X2d, y = gen.generate(n_classes=8, samples_per_class=300, modes_per_class=2)

# Embed into higher-dimensional space
embedder = HighDimEmbedder(seed=0)
X_high = embedder.embed(X2d=X2d[:, :2], method='smooth', target_dim=32)

# Visualize
Visualizer.plot_2d_scatter(X2d, y)
Visualizer.plot_3d_first3(X_high, y)
Visualizer.compare_pca_tsne(X_high, y)
```

### Command Line Interface

```bash
python main.py --n_classes 8 --samples_per_class 300 --modes_per_class 2 --radius 3.0 --embed_method smooth --target_dim 32
```

Use `--help` for a full list of options.

## License

MIT License


