# -*- coding: utf-8 -*-
"""
Python script to generate class-conditional synthetic datasets with
rich, controllable properties for benchmarking optimization and training behavior.

Created on Tue Sep 16 07:14:59 2025

@author: Diyar Altinses, M.Sc.

Features
--------
- Gaussian-mixture class-conditional generator with controls for:
  - number of classes and modes per class
  - cluster separation (radius), overlap, covariance scale
  - anisotropy (per-cluster covariance ellipses), imbalance, label noise
  - outliers and heavy-tailed noise
- High-dimensional embedder to map 2D data into higher-dimensional spaces while preserving
  neighborhood relationships (``tones`` / structure). Multiple strategies provided:
  - linear orthonormal expansion
  - nonlinear smooth lift using basis functions (sin / poly) applied to coordinates
  - preserves local geometry by making extra-dim signals smooth functions of the 2D coords
- Visualization helpers (2D scatter, 3D scatter of first 3 dims, PCA / t-SNE reconstructions)
- CLI and programmatic API, deterministic RNG via seed

Usage
-----
As a module::

    from synthetic_dataset_generator import SyntheticDatasetGenerator, HighDimEmbedder, Visualizer
    gen = SyntheticDatasetGenerator(seed=0)
    X, y = gen.generate(n_classes=4, samples_per_class=300, ...)
    embedder = HighDimEmbedder(seed=0)
    X_high = embedder.embed(X2d=X[:, :2], method='smooth', target_dim=32)
    Visualizer.plot_pair(X, y)

As a script (help):

    python synthetic_dataset_generator.py --help

Dependencies
------------
- numpy
- matplotlib
- scikit-learn

"""

from __future__ import annotations
import argparse
import json
import math
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple, List

import numpy as np
from numpy.random import Generator, PCG64
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (register 3d projection)

from distutils.spawn import find_executable

# ---------------------------
# Utilities
# ---------------------------

def _ensure_rng(seed: Optional[int]) -> Generator:
    """Return a numpy random Generator for deterministic behavior when seed provided."""
    return np.random.default_rng(seed)


# ---------------------------
# Configurations
# ---------------------------

def configure_plt(check_latex = True):
        """
        Set Font sizes for plots.
    
        Parameters
        ----------
        check_latex : bool, optional
            Use LaTex-mode (if available). The default is True.
    
        Returns
        -------
        None.
    
        """
        
        if check_latex:
            
            if find_executable('latex'):
                plt.rc('text', usetex=True)
            else:
                plt.rc('text', usetex=False)
        plt.rc('font',family='Times New Roman')
        plt.rcParams.update({'figure.max_open_warning': 0})
        
        small_size = 13
        small_medium = 14
        medium_size = 16
        big_medium = 18
        big_size = 20
        
        plt.rc('font', size = small_size)          # controls default text sizes
        plt.rc('axes', titlesize = big_medium)     # fontsize of the axes title
        plt.rc('axes', labelsize = medium_size)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize = small_size)    # fontsize of the tick labels
        plt.rc('ytick', labelsize = small_size)    # fontsize of the tick labels
        plt.rc('legend', fontsize = small_medium)    # legend fontsize
        plt.rc('figure', titlesize = big_size)  # fontsize of the figure title
        
        plt.rc('grid', c='0.5', ls='-', lw=0.5)
        plt.grid(True)
        plt.tight_layout()
        plt.close()


# ---------------------------
# Dataset generator
# ---------------------------

@dataclass
class SyntheticDatasetGenerator:
    """Generator for class-conditional synthetic datasets.

    Parameters
    ----------
    seed:
        Optional random seed for reproducibility.
    base_dim:
        Base dimensionality of generated signal (2 by default). Extra irrelevant dims can be
        appended later by the embedder.
    """

    seed: Optional[int] = None
    base_dim: int = 2
    rng: Generator = field(init=False)

    def __post_init__(self):
        self.rng = _ensure_rng(self.seed)

    def generate(self,
                 n_classes: int = 4,
                 samples_per_class: int = 300,
                 modes_per_class: int = 1,
                 radius: float = 5.0,
                 cov_scale: float = 0.5,
                 overlap: float = 0.2,
                 anisotropy: bool = False,
                 imbalance: Optional[Sequence[float]] = None,
                 label_noise: float = 0.0,
                 outlier_fraction: float = 0.0,
                 heavy_tail: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic dataset.

        Returns
        -------
        X : np.ndarray, shape (N, base_dim)
            Generated features (may be extended later by embedder).
        y : np.ndarray, shape (N,)
            Integer class labels in 0..n_classes-1.

        Notes
        -----
        - `modes_per_class` allows each class to be modeled as several Gaussian components.
        - `overlap` adds jitter to class means to increase mixing.
        """
        if self.base_dim < 2:
            raise ValueError("base_dim must be >= 2")

        # Determine counts per class when imbalance provided
        total_samples = int(samples_per_class * n_classes)
        if imbalance is None:
            counts = [int(samples_per_class)] * n_classes
        else:
            rel = np.array(imbalance, dtype=float)
            rel = rel / rel.sum()
            counts = np.round(rel * total_samples).astype(int).tolist()

        X_list: List[np.ndarray] = []
        y_list: List[np.ndarray] = []

        # angles around circle to place class means
        angles = np.linspace(0, 2 * math.pi, n_classes, endpoint=False)

        for cls in range(n_classes):
            cls_count = counts[cls]
            # If modes_per_class > 1, split samples among modes
            mode_counts = np.full(modes_per_class, cls_count // modes_per_class)
            mode_counts[: cls_count % modes_per_class] += 1

            for m in range(modes_per_class):
                if mode_counts[m] == 0:
                    continue

                # Base mean on a circle; add overlap jitter
                angle = angles[cls] + self.rng.normal(scale=0.02)
                base_mean = np.array([math.cos(angle), math.sin(angle)]) * radius
                mean_jitter = self.rng.normal(scale=overlap * radius * 0.3, size=self.base_dim)
                mean = base_mean[: self.base_dim] + mean_jitter

                # Covariance: isotropic or diagonal anisotropic
                if anisotropy:
                    scales = np.abs(self.rng.normal(loc=1.0, scale=0.5, size=self.base_dim)) * cov_scale
                    cov = np.diag(scales ** 2)
                else:
                    cov = np.eye(self.base_dim) * (cov_scale ** 2)

                samples = self.rng.multivariate_normal(mean, cov, size=mode_counts[m])
                X_list.append(samples)
                y_list.append(np.full(len(samples), cls, dtype=int))

        X = np.vstack(X_list)
        y = np.concatenate(y_list)

        # Add heavy-tailed noise to some samples if requested
        if heavy_tail and outlier_fraction > 0.0:
            n_out = int(len(X) * outlier_fraction)
            idx = self.rng.choice(len(X), size=n_out, replace=False)
            # Cauchy-like noise (student-t with low df)
            noise = self.rng.standard_t(df=2.0, size=(n_out, self.base_dim)) * cov_scale * 3.0
            X[idx] += noise

        # Add label noise
        if label_noise > 0.0:
            nflip = int(len(y) * label_noise)
            if nflip > 0:
                flip_idx = self.rng.choice(len(y), size=nflip, replace=False)
                y[flip_idx] = self.rng.integers(0, n_classes, size=nflip)

        # Shuffle dataset
        perm = self.rng.permutation(len(y))
        X = X[perm]
        y = y[perm]

        return X, y


# ---------------------------
# High-dim embedder
# ---------------------------

@dataclass
class HighDimEmbedder:
    """Embed 2D data into a higher-dimensional space while preserving relationships.

    Methods
    -------
    - 'linear': Expand via orthonormal linear transform and padding with small noise.
    - 'smooth': Append smooth nonlinear basis functions (sin, cos, polynomials) of the 2D coords;
      these functions are low-frequency so local neighborhoods remain coherent.

    Important
    ---------
    This class expects input X2d with shape (N, 2). If original X has more dims, you may
    pass only the first 2 columns (the canonical coordinates) into the embedder; the rest
    will be treated as irrelevant noise.
    """

    seed: Optional[int] = None
    rng: Generator = field(init=False)

    def __post_init__(self):
        self.rng = _ensure_rng(self.seed)

    def embed(self, X2d: np.ndarray, method: str = "smooth", target_dim: int = 32,
              nonlinear_scale: float = 0.5, orthonormal: bool = True) -> np.ndarray:
        """Return high-dimensional embedding of X2d.

        Parameters
        ----------
        X2d:
            Array of shape (N, 2) containing the 2D coordinates that must be preserved.
        method:
            One of {'linear', 'smooth'}.
        target_dim:
            Desired output dimensionality (>=2).
        nonlinear_scale:
            Scale for nonlinear basis contributions (only for 'smooth').
        orthonormal:
            If True (for 'linear'), extend by an orthonormal basis for stability.

        Returns
        -------
        X_high:
            Array of shape (N, target_dim) whose first two coordinates are related to X2d
            and additional coordinates are smooth functions of X2d so local relationships
            are preserved.
        """
        if X2d.ndim != 2 or X2d.shape[1] < 2:
            raise ValueError("X2d must be a 2D array with at least 2 columns")
        if target_dim < 2:
            raise ValueError("target_dim must be >= 2")

        N = X2d.shape[0]
        if method == "linear":
            # Start with the 2D coordinates, optionally whiten them, then append orthonormal random basis
            X_centered = X2d - X2d.mean(axis=0, keepdims=True)
            # Optionally whiten (PCA) to stabilize scales
            pca = PCA(n_components=2)
            X2 = pca.fit_transform(X_centered)
            X_high = np.zeros((N, target_dim))
            X_high[:, :2] = X2

            if target_dim > 2:
                # create random orthonormal basis for remaining dims
                R = self.rng.normal(size=(target_dim - 2, target_dim - 2))
                # QR decomposition yields orthonormal columns in Q
                Q, _ = np.linalg.qr(R)
                # Fill remaining dims with small projected noise from original 2D coords to keep relation
                proj = (X2 @ self.rng.normal(size=(2, target_dim - 2))) * 0.1
                X_high[:, 2:] = proj @ Q
            return X_high

        elif method == "smooth":
            # Construct a set of smooth basis functions of the 2D coords.
            x = X2d[:, 0]
            y = X2d[:, 1]
            features: List[np.ndarray] = [x, y]

            # Low-frequency sin/cos basis (preserves neighborhoods because they're smooth)
            freqs = np.linspace(0.5, 2.5, num=max(1, (target_dim - 2) // 4))
            for f in freqs:
                features.append(np.sin(f * x) * nonlinear_scale)
                features.append(np.cos(f * x) * nonlinear_scale)
                features.append(np.sin(f * y) * nonlinear_scale)
                features.append(np.cos(f * y) * nonlinear_scale)

            # Polynomial interactions up to degree 2
            features.append((x ** 2) * nonlinear_scale)
            features.append((y ** 2) * nonlinear_scale)
            features.append((x * y) * nonlinear_scale)

            # Stack and if needed add small Gaussian noise and random linear combinations to reach target_dim
            F = np.vstack([f.reshape(N) for f in features]).T
            # Normalize features
            F = (F - F.mean(axis=0, keepdims=True)) / (F.std(axis=0, keepdims=True) + 1e-12)

            if F.shape[1] >= target_dim:
                # reduce dimensionality deterministically via PCA to target_dim
                pca = PCA(n_components=target_dim)
                X_high = pca.fit_transform(F)
            else:
                # expand by appending small projections of F with random orthonormal mixing
                X_high = np.zeros((N, target_dim))
                X_high[:, : F.shape[1]] = F
                remaining = target_dim - F.shape[1]
                if remaining > 0:
                    # random mixing of existing features
                    mix = self.rng.normal(scale=0.1, size=(F.shape[1], remaining))
                    X_high[:, F.shape[1]:] = F @ mix

            # Finally, ensure the first two components are closely tied to original coords (up to scaling)
            # by linearly projecting back a mixture of x,y onto the first two dims.
            X_high[:, :2] = (X2d - X2d.mean(axis=0)) / (X2d.std(axis=0) + 1e-12)

            return X_high

        else:
            raise ValueError(f"Unknown embedding method: {method}")


# ---------------------------
# Visualization
# ---------------------------

class Visualizer:
    """Visualization helpers for 2D and high-dimensional synthetic data."""

    @staticmethod
    def plot_2d_scatter(X: np.ndarray, y: np.ndarray, title: str = "2D_Scatter") -> None:
        """Scatter plot of first two features colored by label."""
        plt.figure(figsize=(6, 3))
        plt.scatter(X[:, 0], X[:, 1], c=y, s=10, cmap="tab10", alpha=0.9)
        plt.xlabel("dim 0")
        plt.ylabel("dim 1")
        # plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(title + '.png', dpi = 300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_3d_first3(X: np.ndarray, y: np.ndarray, title: str = "3D (first 3 dims)") -> None:
        """3D scatter of the first three dimensions of X."""
        if X.shape[1] < 3:
            raise ValueError("X must have at least 3 dimensions for 3D plotting")
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        p = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, s=12, cmap="tab10", depthshade=True)
        ax.set_xlabel("dim 0")
        ax.set_ylabel("dim 1")
        ax.set_zlabel("dim 2")
        # plt.title(title)
        plt.tight_layout()
        plt.savefig(title + '.png', dpi = 300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def compare_pca_tsne(X_high: np.ndarray, y: np.ndarray, perplexity: float = 30.0) -> None:
        """Compute PCA(2) and t-SNE(2) on high-dimensional data and plot side-by-side.

        This shows whether neighborhood relationships were preserved after embedding.
        """
        # PCA projection
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_high)

        # t-SNE (slow for large N) — keep deterministic by fixing random_state
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=0)
        X_tsne = tsne.fit_transform(X_high)

        fig = plt.figure(figsize=(3, 3))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, s=8, cmap="tab10")
        # plt.title("PCA (2) of high-dim data")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('PCA.png', dpi = 300, bbox_inches='tight')
        plt.show()
        
        fig = plt.figure(figsize=(3, 3))
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, s=8, cmap="tab10")
        # plt.title("t-SNE (2) of high-dim data")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('Tsne.png', dpi = 300, bbox_inches='tight')
        plt.show()


# ---------------------------
# CLI / Example
# ---------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Synthetic dataset generator and high-dim embedder")
    p.add_argument("--seed", type=int, default=0, help="RNG seed")
    p.add_argument("--n_classes", type=int, default=8)
    p.add_argument("--samples_per_class", type=int, default=300)
    p.add_argument("--modes_per_class", type=int, default=2)
    p.add_argument("--radius", type=float, default=3.0)
    p.add_argument("--cov_scale", type=float, default=0.5)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--anisotropy", action="store_true")
    p.add_argument("--label_noise", type=float, default=0.01)
    p.add_argument("--outlier_fraction", type=float, default=0.02)
    p.add_argument("--heavy_tail", action="store_true")
    p.add_argument("--target_dim", type=int, default=32)
    p.add_argument("--embed_method", choices=["linear", "smooth"], default="smooth")
    p.add_argument("--save", type=str, default=None, help="Optional path to save dataset as .npz")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    configure_plt()
    parser = _build_parser()
    args = parser.parse_args(argv)

    gen = SyntheticDatasetGenerator(seed=args.seed)
    X2d, y = gen.generate(n_classes=args.n_classes,
                          samples_per_class=args.samples_per_class,
                          modes_per_class=args.modes_per_class,
                          radius=args.radius,
                          cov_scale=args.cov_scale,
                          overlap=args.overlap,
                          anisotropy=args.anisotropy,
                          label_noise=args.label_noise,
                          outlier_fraction=args.outlier_fraction,
                          heavy_tail=args.heavy_tail)

    embedder = HighDimEmbedder(seed=args.seed)
    X_high = embedder.embed(X2d=X2d[:, :2], method=args.embed_method, target_dim=args.target_dim)
    
    # Visualize
    Visualizer.plot_2d_scatter(X2d, y, title="Original 2D Synthetic Data")
    if args.target_dim >= 3:
        Visualizer.plot_3d_first3(X_high, y, title=f"Embedded data (first 3 dims) - {args.embed_method}")
    Visualizer.compare_pca_tsne(X_high, y)

    if args.save:
        np.savez_compressed(args.save, X2d=X2d, X_high=X_high, y=y)
        print(f"Saved dataset to {args.save}")


if __name__ == "__main__":
    main()
