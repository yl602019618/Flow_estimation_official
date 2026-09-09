"""Compact diagnostic figures for the unit-ball benchmark."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .geometry import SphereAcquisition, coverage_tensor


def plot_sensors(acquisition: SphereAcquisition, path: Path) -> None:
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    p = acquisition.points
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=14, c=np.arange(len(p)), cmap="viridis")
    ax.set(xlabel=r"$\xi_1$", ylabel=r"$\xi_2$", zlabel=r"$\xi_3$",
           title=f"Fibonacci sphere: {len(p)} transducers")
    ax.set_box_aspect((1, 1, 1))
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def plot_coverage(acquisition: SphereAcquisition, path: Path) -> None:
    axis = np.linspace(-.75, .75, 41)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    points = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))
    points = points[np.linalg.norm(points, axis=1) <= .75]
    eigenvalues = np.linalg.eigvalsh(coverage_tensor(acquisition, points))
    score = eigenvalues[:, 0] / np.mean(eigenvalues, axis=1)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    image = ax.tricontourf(points[:, 0], points[:, 1], score, 24, cmap="viridis")
    fig.colorbar(image, ax=ax, label="normalized minimum eigenvalue")
    ax.set(xlabel=r"$\xi_1$", ylabel=r"$\xi_2$", title=r"Local angular coverage at $\xi_3=0$")
    ax.set_aspect("equal"); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def plot_spectrum(singular: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(np.arange(1, len(singular)+1), singular/singular[0], lw=1.5)
    ax.axhline(1e-8, color="k", ls="--", lw=1)
    ax.set(xlabel="mode", ylabel=r"$s_k/s_1$", title="Longitudinal ray singular spectrum")
    ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def plot_fit(observed: np.ndarray, predicted: np.ndarray, path: Path) -> None:
    scale = 1e9
    lo = min(observed.min(), predicted.min()) * scale
    hi = max(observed.max(), predicted.max()) * scale
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(observed*scale, predicted*scale, s=4, alpha=.4)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set(xlabel="observed reciprocal delay (ns)", ylabel="predicted (ns)",
           title="Travel-time fit (noise 10 ns)")
    ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def plot_slices(points: np.ndarray, planes: np.ndarray, truth: np.ndarray,
                estimates: dict[str, np.ndarray], path: Path) -> None:
    names = ["truth", "clean", "noise 10 ns"]
    fields = [truth, estimates["clean"], estimates["noise_10ns"]]
    plane_names = [r"$\xi_3=0$", r"$\xi_2=0$", r"$\xi_1=0$"]
    coordinate_pairs = [(0, 1), (0, 2), (1, 2)]
    fig, axes = plt.subplots(3, 3, figsize=(10, 9), constrained_layout=True)
    maximum = max(np.max(np.linalg.norm(field, axis=1)) for field in fields)
    for row, (plane_name, pair) in enumerate(zip(plane_names, coordinate_pairs)):
        selected = planes == row
        for column, (name, field) in enumerate(zip(names, fields)):
            speed = np.linalg.norm(field[selected], axis=1)
            image = axes[row, column].tricontourf(points[selected, pair[0]],
                                                  points[selected, pair[1]], speed, 31,
                                                  vmin=0, vmax=maximum, cmap="viridis")
            axes[row, column].set_aspect("equal")
            axes[row, column].set_title(f"{name}, {plane_name}")
            axes[row, column].set_xlabel(rf"$\xi_{pair[0]+1}$")
            axes[row, column].set_ylabel(rf"$\xi_{pair[1]+1}$")
    fig.colorbar(image, ax=axes, shrink=.7, label="m/s")
    fig.savefig(path, dpi=180); plt.close(fig)


def plot_ablation(rows: list[dict], path: Path) -> None:
    n = [row["n_transducers"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].plot(n, [row["minimum_normalized_eigenvalue"] for row in rows], "o-")
    axes[0].axhline(.3, color="k", ls="--", lw=1)
    axes[0].set(xlabel="transducers", ylabel="minimum normalized eigenvalue", title="Coverage")
    axes[1].semilogy(n, [row["ray_condition_number"] for row in rows], "o-")
    axes[1].set(xlabel="transducers", ylabel="ray-matrix condition number", title="Stability")
    for ax in axes: ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)
