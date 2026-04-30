import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
import json
import csv
from tqdm import tqdm

from sae_lens import SAE
from config import OUTPUT_DIR, DEVICE, SAE_RELEASE
from src.model import load_model, extract_activations
from src.metrics import fit_and_score_circle, cluster_quality_check
from src.geometry import fit_circle_algebraic

_DATA_PATH = Path(__file__).parent.parent / "data" / "prompts.json"


def _get_sae_reconstruction(model, prompts, layer, device):
    """
    Extract residual stream activations at given layer for all prompts,
    pass through SAE encoder then decoder, return reconstruction as
    numpy array (N x 768).
    """
    hook = f"blocks.{layer}.hook_resid_pre"
    sae_id = f"blocks.{layer}.hook_resid_pre"
    sae, _, _ = SAE.from_pretrained(release=SAE_RELEASE, sae_id=sae_id, device=device)
    sae.eval()
    W_dec = sae.W_dec.detach().cpu()

    rows = []
    for _, prompt, *_ in tqdm(prompts, desc=f"  L{layer} acts", leave=False):
        tokens = model.to_tokens(prompt, prepend_bos=True)
        import torch
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook, device=device)
        rows.append(cache[hook][0, -1, :].cpu())

    import torch
    acts = torch.stack(rows)
    with torch.no_grad():
        feat_acts = sae.encode(acts.to(device)).cpu()
    recon = (feat_acts @ W_dec).numpy()
    return recon


def _plot_pc2_pc3(ax, coords, labels, n_categories, title, annotate_every=1,
                  colorbar_label=""):
    """
    Plot PC2 vs PC3 scatter on given axes object.
    Returns (angular_r, rmse).
    """
    labels_arr = np.array(labels)

    sc = ax.scatter(coords[:, 1], coords[:, 2], c=labels_arr,
                    cmap="plasma", vmin=0.5, vmax=n_categories + 0.5,
                    s=60, zorder=3)
    plt.colorbar(sc, ax=ax, label=colorbar_label)

    angular_r = 0.0
    rmse = 999.0
    cx, cy, radius = 0.0, 0.0, 1.0

    try:
        result = fit_and_score_circle(coords, labels)
        angular_r = result["angular_r"]
        rmse = result["rmse"]
        cx = result["cx"]
        cy = result["cy"]
        radius = result["radius"]

        theta = np.linspace(0, 2 * np.pi, 300)
        ax.plot(cx + radius * np.cos(theta), cy + radius * np.sin(theta),
                "--", color="black", linewidth=1.5, zorder=4)
        ax.plot(cx, cy, "k+", markersize=10, markeredgewidth=1.5,
                zorder=5)
    except Exception:
        angular_r = 0.0
        rmse = 999.0

    # Annotate every annotate_every-th unique label
    seen = {}
    for i, lbl in enumerate(labels):
        if lbl not in seen:
            seen[lbl] = (coords[i, 1], coords[i, 2])

    unique_sorted = sorted(seen.keys())
    for k, lbl in enumerate(unique_sorted):
        if (lbl - 1) % annotate_every == 0 or lbl == unique_sorted[-1]:
            x, y = seen[lbl]
            ax.annotate(str(lbl), (x, y), fontsize=6, ha="center",
                        va="bottom", alpha=0.85)

    ax.set_title(f"{title}\nRMSE={rmse:.3f}  angular_r={angular_r:.3f}")
    return angular_r, rmse


def run_phase6() -> None:
    print("\n" + "=" * 65)
    print("PHASE 6  -  Days of the Month — Geometric Structure")
    print("=" * 65)

    OUTPUT_DIR.mkdir(exist_ok=True)
    phase_dir = OUTPUT_DIR / "phase6"
    phase_dir.mkdir(parents=True, exist_ok=True)

    # -- 1. Load data ----------------------------------------------------------
    with open(_DATA_PATH) as f:
        data = json.load(f)

    dom_raw = data["days_of_month"]
    dom_prompts = [(e["label"], e["prompt"], e["day"]) for e in dom_raw]
    days_of_month_labels = data["days_of_month_labels"]

    days_raw = data["days_of_week"]
    dow_prompts = [(e["label"], e["prompt"], e["day"]) for e in days_raw]
    days_of_week_labels = data["days_of_week_labels"]

    months_raw = data["months_of_year"]
    moy_prompts = [(e["label"], e["prompt"], e["month"]) for e in months_raw]
    months_of_year_labels = data["months_of_year_labels"]

    print(f"  Loaded {len(dom_prompts)} days-of-month prompts")

    # -- 2. Load model once ----------------------------------------------------
    model = load_model()
    print("Model loaded. Running geometry analysis across layers...")

    # -- 3. Cluster quality check at layer 7 -----------------------------------
    print("\n[Cluster quality check at layer 7]")
    reconstruction = _get_sae_reconstruction(model, dom_prompts, 7, DEVICE)
    pca = PCA(n_components=3)
    coords = pca.fit_transform(reconstruction)
    ratio = cluster_quality_check(coords, days_of_month_labels, n_per_cluster=3)
    print(f"  Intra/inter ratio = {ratio:.4f}")
    if ratio < 0.3:
        print("  Cluster quality: Excellent — frames are consistent")
    elif ratio < 0.6:
        print("  Cluster quality: Good — minor within-day variance")
    elif ratio < 0.8:
        print("  Cluster quality: Moderate — some frame noise")
    else:
        print("  Cluster quality: Poor — WARNING: high frame variance")

    # -- 4. Geometry analysis across target layers -----------------------------
    print("\n[Geometry analysis across target layers]")
    TARGET_LAYERS = [4, 6, 7, 8, 10]

    results = {}
    for layer in TARGET_LAYERS:
        reconstruction = _get_sae_reconstruction(model, dom_prompts, layer, DEVICE)
        pca = PCA(n_components=5)
        coords = pca.fit_transform(reconstruction)
        var = pca.explained_variance_ratio_
        try:
            result = fit_and_score_circle(coords, days_of_month_labels)
            angular_r = result["angular_r"]
            rmse = result["rmse"]
        except Exception:
            angular_r = 0.0
            rmse = 999.0
        results[layer] = {
            "angular_r": angular_r, "rmse": rmse,
            "coords": coords, "var": var
        }
        print(f"  Layer {layer}: angular_r={angular_r:.3f}, RMSE={rmse:.3f}")

    # -- 5. Main geometry plot -------------------------------------------------
    print("\n[Saving main geometry plot]")
    fig, axes = plt.subplots(3, 5, figsize=(25, 12))
    fig.suptitle("Days of the Month — Geometric Structure in GPT-2 Residual Stream",
                 fontsize=14)

    frame_colors = {1: "steelblue", 2: "crimson", 3: "seagreen"}
    frame_labels_list = [e["frame"] for e in dom_raw]

    for col, layer in enumerate(TARGET_LAYERS):
        r = results[layer]
        coords = r["coords"]
        var = r["var"]
        angular_r = r["angular_r"]
        rmse = r["rmse"]

        # Row 1 — PC1 vs PC2
        ax = axes[0, col]
        sc = ax.scatter(coords[:, 0], coords[:, 1],
                        c=days_of_month_labels, cmap="hsv",
                        vmin=0.5, vmax=31.5, s=40, zorder=3)
        plt.colorbar(sc, ax=ax, label="Day")
        ax.set_xlabel(f"PC1 ({var[0]:.1%})")
        ax.set_ylabel(f"PC2 ({var[1]:.1%})")
        ax.set_title(f"Layer {layer} — PC1 vs PC2")

        # Row 2 — PC2 vs PC3 with circle
        ax = axes[1, col]
        _plot_pc2_pc3(ax, coords, days_of_month_labels, n_categories=31,
                      title=f"Layer {layer} — PC2 vs PC3", annotate_every=5,
                      colorbar_label="Day of month")
        ax.set_xlabel(f"PC2 ({var[1]:.1%})")
        ax.set_ylabel(f"PC3 ({var[2]:.1%})")

        # Row 3 — Angle vs day number
        ax = axes[2, col]
        try:
            result = fit_and_score_circle(coords, days_of_month_labels)
            angles_rad = result["angles"]
            angles_deg = np.degrees(angles_rad)
            valid = True
        except Exception:
            angles_deg = np.zeros(len(days_of_month_labels))
            valid = False

        for frame_num, color in frame_colors.items():
            idxs = [i for i, f in enumerate(frame_labels_list) if f == frame_num]
            day_nums = [days_of_month_labels[i] for i in idxs]
            ang = [angles_deg[i] for i in idxs]
            label_str = f"Frame {frame_num}" if col == 0 else None
            ax.scatter(day_nums, ang, c=color, s=25, alpha=0.7,
                       label=label_str, zorder=3)

        if col == 0:
            ax.legend(fontsize=7, loc="best")

        ax.set_xlabel("Day number")
        ax.set_ylabel("Angle (degrees)")
        if valid and angular_r > 0:
            ax.set_title(f"Layer {layer} — Angle vs Day (r={angular_r:.3f})")
        else:
            ax.set_title(f"Layer {layer} — No circle detected")
        ax.set_xticks([1, 6, 11, 16, 21, 26, 31])

    plt.tight_layout()
    out_path = phase_dir / "phase6_days_of_month_geometry.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("  Saved -> phase6_days_of_month_geometry.png")

    # -- 6. Best-layer focused plot --------------------------------------------
    print("\n[Saving best-layer focused plot]")
    best_layer = max(TARGET_LAYERS, key=lambda l: results[l]["angular_r"])
    best_r = results[best_layer]["angular_r"]
    best_rmse = results[best_layer]["rmse"]
    best_coords = results[best_layer]["coords"]

    if best_r == 0.0:
        best_layer = 7

    fig, ax = plt.subplots(figsize=(8, 7))

    sc = ax.scatter(best_coords[:, 1], best_coords[:, 2],
                    c=days_of_month_labels, cmap="hsv",
                    vmin=0.5, vmax=31.5, s=120,
                    edgecolors="k", linewidths=0.5, zorder=3)
    plt.colorbar(sc, ax=ax, label="Day of Month")

    try:
        result = fit_and_score_circle(best_coords, days_of_month_labels)
        cx = result["cx"]
        cy = result["cy"]
        radius = result["radius"]
        theta = np.linspace(0, 2 * np.pi, 300)
        ax.plot(cx + radius * np.cos(theta), cy + radius * np.sin(theta),
                "--", color="black", linewidth=2, alpha=0.7, zorder=4)
        ax.plot(cx, cy, "k+", markersize=12, markeredgewidth=2, zorder=5)
        title_str = (f"Days of the Month — PC2 vs PC3 (Layer {best_layer})\n"
                     f"RMSE={best_rmse:.3f}  angular_r={best_r:.3f}")
    except Exception:
        title_str = (f"Days of the Month — PC2 vs PC3 (Layer {best_layer})\n"
                     "No circular structure detected")

    # Label every day
    seen = {}
    for i, lbl in enumerate(days_of_month_labels):
        if lbl not in seen:
            seen[lbl] = (best_coords[i, 1], best_coords[i, 2])
    for lbl, (x, y) in seen.items():
        ax.annotate(str(lbl), (x, y), fontsize=7, ha="center",
                    va="bottom", alpha=0.85)

    ax.set_xlabel("PC2")
    ax.set_ylabel("PC3")
    ax.set_title(title_str)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path = phase_dir / "phase6_days_of_month_best_layer.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("  Saved -> phase6_days_of_month_best_layer.png")

    # -- 7. Three-concept comparison plot --------------------------------------
    print("\n[Saving cyclic concepts comparison plot]")

    # Get days of week at layer 7
    dow_recon = _get_sae_reconstruction(model, dow_prompts, 7, DEVICE)
    dow_pca = PCA(n_components=5)
    dow_coords = dow_pca.fit_transform(dow_recon)
    try:
        dow_result = fit_and_score_circle(dow_coords, days_of_week_labels)
        dow_r = dow_result["angular_r"]
    except Exception:
        dow_r = 0.0

    # Get months of year at layer 7
    moy_recon = _get_sae_reconstruction(model, moy_prompts, 7, DEVICE)
    moy_pca = PCA(n_components=5)
    moy_coords = moy_pca.fit_transform(moy_recon)
    try:
        moy_result = fit_and_score_circle(moy_coords, months_of_year_labels)
        moy_r = moy_result["angular_r"]
    except Exception:
        moy_r = 0.0

    day_abbrevs = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu",
                   5: "Fri", 6: "Sat", 7: "Sun"}
    month_abbrevs = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
                     5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
                     9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Cyclic Temporal Concepts — Circular Geometry in GPT-2",
                 fontsize=13)

    # Panel 1 — Days of week
    ax = axes[0]
    _plot_pc2_pc3(ax, dow_coords, days_of_week_labels, n_categories=7,
                  title="Days of Week — Layer 7", annotate_every=1,
                  colorbar_label="Day of week")
    ax.set_xlabel("PC2")
    ax.set_ylabel("PC3")
    # Overwrite annotations with day abbreviations
    for txt in list(ax.texts):
        txt.remove()
    seen_dow = {}
    for i, lbl in enumerate(days_of_week_labels):
        if lbl not in seen_dow:
            seen_dow[lbl] = (dow_coords[i, 1], dow_coords[i, 2])
    for lbl, (x, y) in seen_dow.items():
        ax.annotate(day_abbrevs.get(lbl, str(lbl)), (x, y),
                    fontsize=8, ha="center", va="bottom", alpha=0.85)

    # Panel 2 — Months of year
    ax = axes[1]
    _plot_pc2_pc3(ax, moy_coords, months_of_year_labels, n_categories=12,
                  title="Months of Year — Layer 7", annotate_every=1,
                  colorbar_label="Month")
    ax.set_xlabel("PC2")
    ax.set_ylabel("PC3")
    for txt in list(ax.texts):
        txt.remove()
    seen_moy = {}
    for i, lbl in enumerate(months_of_year_labels):
        if lbl not in seen_moy:
            seen_moy[lbl] = (moy_coords[i, 1], moy_coords[i, 2])
    for lbl, (x, y) in seen_moy.items():
        ax.annotate(month_abbrevs.get(lbl, str(lbl)), (x, y),
                    fontsize=8, ha="center", va="bottom", alpha=0.85)

    # Panel 3 — Days of month
    ax = axes[2]
    _plot_pc2_pc3(ax, best_coords, days_of_month_labels, n_categories=31,
                  title=f"Days of Month — Layer {best_layer}", annotate_every=5,
                  colorbar_label="Day of month")
    ax.set_xlabel("PC2")
    ax.set_ylabel("PC3")

    plt.tight_layout()
    out_path = phase_dir / "phase6_cyclic_concepts_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("  Saved -> phase6_cyclic_concepts_comparison.png")

    # -- 8. Save results CSV ---------------------------------------------------
    csv_path = phase_dir / "phase6_days_of_month_geometry.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "layer", "angular_r", "rmse", "pc1_var", "pc2_var", "pc3_var"
        ])
        writer.writeheader()
        for layer in TARGET_LAYERS:
            r = results[layer]
            writer.writerow({
                "layer":    layer,
                "angular_r": round(r["angular_r"], 4),
                "rmse":      round(r["rmse"], 4),
                "pc1_var":   round(float(r["var"][0]), 4),
                "pc2_var":   round(float(r["var"][1]), 4),
                "pc3_var":   round(float(r["var"][2]), 4),
            })
    print("  Saved -> phase6_days_of_month_geometry.csv")

    # -- 9. Print summary table ------------------------------------------------
    print("\n=== PHASE 6 SUMMARY — Days of the Month ===")
    print(f"{'Layer':<6}| {'RMSE':<8}| {'angular_r':<10}| Structure")
    print(f"{'------':<6}|{'---------':<9}|{'-----------':<11}|----------")
    for layer in TARGET_LAYERS:
        r = results[layer]
        ar = r["angular_r"]
        if ar > 0.7:
            structure = "strong circle"
        elif ar > 0.4:
            structure = "moderate circle"
        elif ar > 0.2:
            structure = "weak circle"
        else:
            structure = "no clear circle"
        print(f"{layer:<6}| {r['rmse']:<8.4f}| {ar:<10.4f}| {structure}")

    print()
    print(f"Best layer: {best_layer} (angular_r = {best_r:.3f})")
    print()
    print("=== COMPARISON WITH KNOWN CIRCULAR CONCEPTS ===")
    print(f"Days of week  (layer 7): angular_r = {dow_r:.3f}")
    print(f"Months of year (layer 7): angular_r = {moy_r:.3f}")
    print(f"Days of month  (layer {best_layer}): angular_r = {best_r:.3f}")

    print("\n" + "=" * 65)
    print("PHASE 6  COMPLETE")
    print(f"  Outputs saved to: {OUTPUT_DIR.resolve()}")
    print("=" * 65)
