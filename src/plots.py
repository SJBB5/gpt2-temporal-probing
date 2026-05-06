"""
plots.py — all matplotlib visualisation functions
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA

from src.geometry import fit_circle_algebraic

# Category -> plot colour (shared across all plots)
CAT_COLORS = {
    "ancient_bc":   "#6c3483",
    "ancient_ad":   "#a569bd",
    "medieval":     "#2980b9",
    "early_modern": "#27ae60",
    "modern":       "#e74c3c",
}


def _cat_colors(categories: list[str]) -> list[str]:
    return [CAT_COLORS.get(c, "#7f8c8d") for c in categories]


def plot_pca_overview(
    acts,
    labels: list[str],
    categories: list[str],
    title: str,
    path: Path,
) -> None:
    """2-D and 3-D PCA scatter, coloured by historical category."""
    X    = acts.numpy()
    N    = X.shape[0]
    n_pc = min(3, N - 1, X.shape[1])

    pca    = PCA(n_components=n_pc)
    coords = pca.fit_transform(X)
    var    = pca.explained_variance_ratio_
    colors = _cat_colors(categories)

    ncols = 2 if n_pc >= 3 else 1
    fig   = plt.figure(figsize=(7 * ncols, 6))
    fig.suptitle(title, fontsize=10)

    # 2-D scatter
    ax = fig.add_subplot(1, ncols, 1)
    for i in range(N):
        ax.scatter(coords[i, 0], coords[i, 1],
                   color=colors[i], s=90, edgecolors="k", linewidths=0.5, zorder=3)
        ax.annotate(labels[i], (coords[i, 0], coords[i, 1]),
                    fontsize=7, alpha=0.85, ha="center", va="bottom")
    ax.set_xlabel(f"PC1  ({var[0]:.1%})")
    ax.set_ylabel(f"PC2  ({var[1]:.1%})" if n_pc >= 2 else "")
    ax.set_title("PC1 vs PC2")
    ax.grid(alpha=0.3)
    for cat in dict.fromkeys(categories):
        ax.scatter([], [], color=CAT_COLORS.get(cat, "#7f8c8d"), label=cat, s=60)
    ax.legend(fontsize=7)

    # 3-D scatter
    if n_pc >= 3:
        ax3 = fig.add_subplot(1, 2, 2, projection="3d")
        for i in range(N):
            ax3.scatter(coords[i, 0], coords[i, 1], coords[i, 2],
                        color=colors[i], s=60, edgecolors="k", linewidths=0.3)
            ax3.text(coords[i, 0], coords[i, 1], coords[i, 2], labels[i], fontsize=6)
        ax3.set_xlabel("PC1"); ax3.set_ylabel("PC2"); ax3.set_zlabel("PC3")
        ax3.set_title(f"3-D  ({var[:3].sum():.1%} var)")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved -> {path.name}")


def plot_year_linearity(
    acts,
    items: list,
    path: Path,
    layer: int,
) -> None:
    """
    Plot PC1 / PC2 / PC3 vs. the numeric year value.
    A high Pearson |r| means the model linearly encodes temporal position,
    consistent with Gurnee & Tegmark.
    """
    X       = acts.numpy()
    numeric = np.array([it[2] for it in items], dtype=float)
    lbls    = [it[0] for it in items]

    n_pc   = min(3, X.shape[0] - 1, X.shape[1])
    pca    = PCA(n_components=n_pc)
    coords = pca.fit_transform(X)
    var    = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, n_pc, figsize=(5 * n_pc, 4))
    if n_pc == 1:
        axes = [axes]

    for pc, ax in enumerate(axes):
        sc = ax.scatter(numeric, coords[:, pc], c=numeric,
                        cmap="plasma", s=80, edgecolors="k", linewidths=0.5)
        for i, lbl in enumerate(lbls):
            ax.annotate(lbl, (numeric[i], coords[i, pc]), fontsize=7, alpha=0.85)
        r = float(np.corrcoef(numeric, coords[:, pc])[0, 1])
        ax.set_xlabel("Year (numeric)")
        ax.set_ylabel(f"PC{pc+1}  ({var[pc]:.1%} var)")
        ax.set_title(f"PC{pc+1} vs Year   r = {r:.3f}")
        ax.grid(alpha=0.3)
        plt.colorbar(sc, ax=ax, label="Year")

    fig.suptitle(f"Linear Decodability of Year Value  (Layer {layer})", fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved -> {path.name}")


def _plot_geometric_analysis(
    coords: np.ndarray, var: np.ndarray,
    angles: np.ndarray,
    year_nums: np.ndarray, year_labels: list,
    layer: int,
    year_categories: list,
    output_dir: Path,
) -> None:
    """Two-plot geometric structure: PC2 vs PC3 with fitted circle, and angular position vs year."""
    cat_colors_pts = [CAT_COLORS.get(c, "#7f8c8d") for c in year_categories]

    # -- Plot 1: PC2 vs PC3 with fitted circle, all categories -----------------
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle(f"Phase 2 - PC2 vs PC3  (Layer {layer})", fontsize=11)

    ax.scatter(coords[:, 1], coords[:, 2], c=cat_colors_pts,
               s=80, edgecolors="k", linewidths=0.5, zorder=3)
    for i, lbl in enumerate(year_labels):
        ax.annotate(lbl, (coords[i, 1], coords[i, 2]), fontsize=6, alpha=0.8)

    try:
        cx, cy, r, rmse = fit_circle_algebraic(coords[:, 1:3])
        th = np.linspace(-np.pi, np.pi, 300)
        ax.plot(cx + r * np.cos(th), cy + r * np.sin(th),
                "k--", alpha=0.45, linewidth=1.5, label=f"Circle (RMSE={rmse:.3f})")
        ax.scatter([cx], [cy], color="black", marker="+", s=120, zorder=6)
    except Exception:
        pass

    for cat in dict.fromkeys(year_categories):
        ax.scatter([], [], color=CAT_COLORS.get(cat, "#7f8c8d"), label=cat, s=60)
    ax.legend(fontsize=7)
    ax.set_xlabel(f"PC2 ({var[1]:.1%})")
    ax.set_ylabel(f"PC3 ({var[2]:.1%})")
    ax.set_title("PC2 vs PC3 + fitted circle")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    p1 = output_dir / f"phase2_pc2pc3_layer{layer}.png"
    plt.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {p1.name}")

    # -- Plot 2: Angular position on fitted circle vs year ----------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    deg = np.degrees(angles)
    ax.scatter(year_nums, deg, c=cat_colors_pts,
               s=80, edgecolors="k", linewidths=0.5, zorder=3)
    for i, lbl in enumerate(year_labels):
        ax.annotate(lbl, (year_nums[i], deg[i]), fontsize=6, alpha=0.8)
    r_ang = float(np.corrcoef(year_nums, deg)[0, 1])

    for cat in dict.fromkeys(year_categories):
        ax.scatter([], [], color=CAT_COLORS.get(cat, "#7f8c8d"), label=cat, s=60)
    ax.legend(fontsize=7)
    ax.set_xlabel("Year (numeric)")
    ax.set_ylabel("Angle on fitted circle (deg)")
    ax.set_title(f"Angular position vs year  (r = {r_ang:+.3f})")
    ax.grid(alpha=0.3)

    fig.suptitle(f"Phase 2 - Angular Position  (Layer {layer})", fontsize=11)
    plt.tight_layout()
    p2 = output_dir / f"phase2_angular_layer{layer}.png"
    plt.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {p2.name}")
