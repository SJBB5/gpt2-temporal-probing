import csv
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from scipy.stats import f_oneway
from tqdm import tqdm
from sae_lens import SAE

from config import OUTPUT_DIR, DEVICE, SAE_RELEASE, N_DISC
from src.model import load_model
from src.metrics import fit_and_score_circle, cluster_quality_check
from src.geometry import fit_circle_algebraic  # noqa: F401 — imported per spec

_DATA_PATH = Path(__file__).parent.parent / "data" / "prompts.json"


def _extract_acts_at_layer(model, layer: int, items: list) -> torch.Tensor:
    """Extract last-token residual stream activations at the given layer."""
    hook = f"blocks.{layer}.hook_resid_pre"
    rows = []
    for _, prompt, _ in tqdm(items, desc=f"  L{layer} acts", leave=False):
        tokens = model.to_tokens(prompt, prepend_bos=True)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook, device=DEVICE)
        rows.append(cache[hook][0, -1, :].cpu())
    return torch.stack(rows)


def _circular_ablation_scan_at_layer(
    model, layer: int, items: list, labels: list
) -> dict:
    """
    Run the circular superposition ablation scan at one layer.
    Returns dict with: layer, baseline_angular_r, baseline_rmse, max_drop, top_feature.
    """
    sae_id = f"blocks.{layer}.hook_resid_pre"
    print(f"  Loading SAE for layer {layer} ...")
    sae, _, _ = SAE.from_pretrained(release=SAE_RELEASE, sae_id=sae_id, device=DEVICE)
    sae.eval()
    W_dec = sae.W_dec.detach().cpu()

    acts = _extract_acts_at_layer(model, layer, items)
    with torch.no_grad():
        feat_acts = sae.encode(acts.to(DEVICE)).cpu()

    n_pts = len(labels)
    n_comp = min(5, n_pts - 1, acts.shape[1])

    def _score(fa: torch.Tensor):
        recon = (fa @ W_dec).numpy()
        n_c = min(n_comp, recon.shape[0] - 1, recon.shape[1])
        if n_c < 3:
            return 0.0, 999.0
        try:
            coords = PCA(n_components=n_c).fit_transform(recon)
            result = fit_and_score_circle(coords, labels)
            return result["angular_r"], result["rmse"]
        except Exception:
            return 0.0, 999.0

    baseline_r, baseline_rmse = _score(feat_acts)

    # ANOVA-based discriminative feature selection
    unique_labels = sorted(set(labels))
    label_idx = {l: [i for i, lbl in enumerate(labels) if lbl == l] for l in unique_labels}

    freq = (feat_acts > 0).float().sum(0)
    active_idx = freq.ge(2).nonzero(as_tuple=True)[0].tolist()
    print(f"    Active features (fire on >= 2 tokens): {len(active_idx)}")

    if not active_idx:
        print(f"    Layer {layer}: no active features, skipping.")
        return dict(layer=layer, baseline_angular_r=baseline_r,
                    baseline_rmse=baseline_rmse, max_drop=float("nan"), top_feature=-1)

    f_scores = np.zeros(len(active_idx))
    for k, fi in enumerate(active_idx):
        groups = [feat_acts[label_idx[l], fi].numpy() for l in unique_labels]
        try:
            F, _ = f_oneway(*groups)
            f_scores[k] = float(F) if np.isfinite(F) else 0.0
        except Exception:
            pass

    n_keep = min(N_DISC, len(active_idx))
    top_local = np.argsort(f_scores)[-n_keep:][::-1]
    top_global = [active_idx[i] for i in top_local]
    print(f"    Top-{n_keep} discriminative features selected  "
          f"(max F = {f_scores[top_local[0]]:.2f})")

    scan = []
    for fi in tqdm(top_global, desc=f"  L{layer} scan", leave=False):
        fa_abl = feat_acts.clone()
        fa_abl[:, fi] = 0.0
        r_abl, _ = _score(fa_abl)
        drop = baseline_r - r_abl
        scan.append((fi, drop))

    scan.sort(key=lambda x: x[1], reverse=True)
    max_drop = scan[0][1]
    top_feat = scan[0][0]

    print(f"    Layer {layer}: baseline_r={baseline_r:.4f}  "
          f"max_drop={max_drop:+.4f}  top_feat={top_feat}")

    return dict(layer=layer, baseline_angular_r=baseline_r, baseline_rmse=baseline_rmse,
                max_drop=max_drop, top_feature=top_feat)


def _geometry_plot(
    model, layer: int, items: list, labels: list,
    concept_name: str, item_labels_list: list, short_name: str
) -> None:
    """Generate the PC2/PC3 circle geometry validation plot at the given layer."""
    sae_id = f"blocks.{layer}.hook_resid_pre"
    print(f"  Loading SAE for geometry plot (layer {layer}) ...")
    sae, _, _ = SAE.from_pretrained(release=SAE_RELEASE, sae_id=sae_id, device=DEVICE)
    sae.eval()
    W_dec = sae.W_dec.detach().cpu()

    acts = _extract_acts_at_layer(model, layer, items)
    with torch.no_grad():
        feat_acts = sae.encode(acts.to(DEVICE)).cpu()
    recon = (feat_acts @ W_dec).numpy()

    n_pts = len(labels)
    n_c = min(5, n_pts - 1, recon.shape[1])
    coords = PCA(n_components=n_c).fit_transform(recon)

    try:
        result = fit_and_score_circle(coords, labels)
    except Exception:
        result = dict(rmse=999.0, angular_r=0.0, angles=np.zeros(n_pts),
                      cx=0.0, cy=0.0, radius=1.0)

    cx       = result["cx"]
    cy       = result["cy"]
    radius   = result["radius"]
    angles   = result["angles"]
    angular_r = result["angular_r"]
    label_max = max(labels)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"{concept_name} — Layer {layer} geometry (SAE reconstruction)", fontsize=12)

    # Left: PC2 vs PC3 scatter with fitted circle
    ax = axes[0]
    sc = ax.scatter(coords[:, 1], coords[:, 2],
                    c=labels, cmap="hsv", vmin=0.5, vmax=label_max + 0.5,
                    s=60, zorder=3)
    plt.colorbar(sc, ax=ax, label=f"{concept_name.split()[0]} number")

    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(cx + radius * np.cos(theta), cy + radius * np.sin(theta),
            "--", color="gray", linewidth=1.5, label=f"circle r={radius:.2f}")

    for i, lbl in enumerate(item_labels_list):
        ax.annotate(lbl, (coords[i, 1], coords[i, 2]),
                    fontsize=5, ha="center", va="bottom", alpha=0.75)

    ax.set_xlabel("PC2")
    ax.set_ylabel("PC3")
    ax.set_title(f"PC2 vs PC3 scatter\nCircle RMSE = {result['rmse']:.4f}")
    ax.legend(fontsize=8)
    ax.set_aspect("equal", adjustable="box")

    # Right: angular position vs label number
    ax = axes[1]
    ax.scatter(labels, angles, c=labels, cmap="hsv",
               vmin=0.5, vmax=label_max + 0.5, s=60)
    ax.set_xlabel(f"{concept_name.split()[0]} number")
    ax.set_ylabel("Angle (radians)")
    ax.set_title(f"Angular position vs label\nangular_r = {angular_r:.4f}")

    fig.tight_layout()
    p = OUTPUT_DIR / f"phase5_{short_name}_geometry_layer{layer}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Saved -> {p.name}")


def run_phase5() -> None:
    print("\n" + "=" * 65)
    print("PHASE 5  -  Circular Superposition Depth Scan")
    print("  (Days of Week & Months of Year)")
    print("=" * 65)

    OUTPUT_DIR.mkdir(exist_ok=True)

    # -- 1. Load data ----------------------------------------------------------
    with open(_DATA_PATH) as f:
        data = json.load(f)

    days_raw    = data["days_of_week"]
    days_items  = [(e["label"], e["prompt"], e["day"]) for e in days_raw]
    days_labels = [e["day"] for e in days_raw]
    days_item_labels = [e["label"] for e in days_raw]

    months_raw    = data["months_of_year"]
    months_items  = [(e["label"], e["prompt"], e["month"]) for e in months_raw]
    months_labels = [e["month"] for e in months_raw]
    months_item_labels = [e["label"] for e in months_raw]

    print(f"  Loaded {len(days_items)} day prompts, {len(months_items)} month prompts")

    # -- 2. Load model ---------------------------------------------------------
    model = load_model()

    # -- 3. Cluster quality check at layer 7 -----------------------------------
    print("\n[Cluster quality check at layer 7]")

    for concept, items, labels, n_per in [
        ("Days",   days_items,   days_labels,   3),
        ("Months", months_items, months_labels, 3),
    ]:
        acts = _extract_acts_at_layer(model, 7, items)
        n_c = min(3, len(labels) - 1, acts.shape[1])
        coords = PCA(n_components=n_c).fit_transform(acts.numpy())
        ratio = cluster_quality_check(coords, labels, n_per)
        print(f"  {concept} intra/inter ratio = {ratio:.4f}")
        if ratio > 0.8:
            print(f"  WARNING: frames may be too noisy ({concept.lower()})")

    # -- 4. Geometry visualization at layer 7 ----------------------------------
    print("\n[Geometry visualization at layer 7]")
    _geometry_plot(model, 7, days_items,   days_labels,   "Days of Week",   days_item_labels,   "days")
    _geometry_plot(model, 7, months_items, months_labels, "Months of Year", months_item_labels, "months")

    # -- 5. Circular superposition depth scan ----------------------------------
    print("\n[Circular superposition depth scan]")

    days_results = []
    print("\n--- Days of Week ---")
    for layer in range(12):
        print(f"\n--- Layer {layer} (days) ---")
        days_results.append(
            _circular_ablation_scan_at_layer(model, layer, days_items, days_labels)
        )

    months_results = []
    print("\n--- Months of Year ---")
    for layer in range(12):
        print(f"\n--- Layer {layer} (months) ---")
        months_results.append(
            _circular_ablation_scan_at_layer(model, layer, months_items, months_labels)
        )

    # -- 6. Save results CSV ---------------------------------------------------
    def _fmt(v):
        if isinstance(v, float) and np.isnan(v):
            return ""
        return round(v, 4) if isinstance(v, float) else v

    for results, fname in [
        (days_results,   "phase5_days_layer_summary.csv"),
        (months_results, "phase5_months_layer_summary.csv"),
    ]:
        p = OUTPUT_DIR / fname
        with open(p, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "layer", "baseline_angular_r", "baseline_rmse",
                "max_drop", "top_feature_index",
            ])
            writer.writeheader()
            for r in results:
                writer.writerow({
                    "layer":              r["layer"],
                    "baseline_angular_r": _fmt(r["baseline_angular_r"]),
                    "baseline_rmse":      _fmt(r["baseline_rmse"]),
                    "max_drop":           _fmt(r["max_drop"]),
                    "top_feature_index":  r["top_feature"],
                })
        print(f"  Saved -> {fname}")

    # -- 7. Depth scan plot ----------------------------------------------------
    layers = list(range(12))

    def _safe(results, key):
        return [r[key] if not (isinstance(r[key], float) and np.isnan(r[key]))
                else np.nan for r in results]

    days_max_drops   = _safe(days_results,   "max_drop")
    days_base_rs     = _safe(days_results,   "baseline_angular_r")
    months_max_drops = _safe(months_results, "max_drop")
    months_base_rs   = _safe(months_results, "baseline_angular_r")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Circular Superposition Depth Scan (Phase 5)", fontsize=12)

    for ax, drops, title in [
        (axes[0, 0], days_max_drops,   "Days of Week — Max angular_r drop"),
        (axes[0, 1], months_max_drops, "Months of Year — Max angular_r drop"),
    ]:
        ax.plot(layers, drops, "o-", color="steelblue", linewidth=2, markersize=6)
        ax.axhline(0.10, color="crimson",    linestyle="--", linewidth=1.1,
                   label="load-bearing (0.10)")
        ax.axhline(0.02, color="darkorange", linestyle="--", linewidth=1.1,
                   label="moderate (0.02)")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Max single-feature drop in angular_r")
        ax.set_title(title)
        ax.set_xticks(layers)
        ax.legend(fontsize=8)

    for ax, base_rs, title in [
        (axes[1, 0], days_base_rs,   "Days of Week — Baseline angular_r"),
        (axes[1, 1], months_base_rs, "Months of Year — Baseline angular_r"),
    ]:
        ax.plot(layers, base_rs, "s-", color="seagreen", linewidth=2, markersize=6)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Baseline angular_r")
        ax.set_title(title)
        ax.set_xticks(layers)
        ax.set_ylim(0, 1)

    fig.tight_layout()
    p = OUTPUT_DIR / "phase5_circular_superposition_depth.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Saved -> {p.name}")

    # -- 8. Comparison plot with Phase 3/4 -------------------------------------
    try:
        p3_csv = OUTPUT_DIR / "phase3_layer_summary.csv"
        p4_csv = OUTPUT_DIR / "phase4_layer_summary.csv"

        with open(p3_csv, newline="") as f:
            p3_rows = {int(r["layer"]): r for r in csv.DictReader(f)}
        with open(p4_csv, newline="") as f:
            p4_rows = {int(r["layer"]): r for r in csv.DictReader(f)}

        def _csv_float(rows, layer, key):
            v = rows.get(layer, {}).get(key, "") or ""
            try:
                return float(v)
            except (ValueError, TypeError):
                return np.nan

        p3_max_drops = [_csv_float(p3_rows, l, "max_drop") for l in layers]
        p3_base_rs   = [_csv_float(p3_rows, l, "base_r")   for l in layers]
        p4_max_drops = [_csv_float(p4_rows, l, "max_drop") for l in layers]
        p4_base_rs   = [_csv_float(p4_rows, l, "base_r")   for l in layers]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("All Concepts — Superposition Depth Comparison", fontsize=12)

        ax = axes[0]
        ax.plot(layers, p3_max_drops,    "--", color="gray",      linewidth=2,
                label="Temporal (original)")
        ax.plot(layers, p4_max_drops,    "-",  color="steelblue", linewidth=2,
                label="Temporal (clean)")
        ax.plot(layers, days_max_drops,  "-",  color="crimson",   linewidth=2,
                label="Days of week")
        ax.plot(layers, months_max_drops, "-", color="seagreen",  linewidth=2,
                label="Months of year")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Max single-feature drop")
        ax.set_title("Signal concentration by layer\n(higher = less superposition)")
        ax.set_xticks(layers)
        ax.legend(fontsize=9)

        ax = axes[1]
        ax.plot(layers, p3_base_rs,    "--", color="gray",      linewidth=2,
                label="Temporal (original)")
        ax.plot(layers, p4_base_rs,    "-",  color="steelblue", linewidth=2,
                label="Temporal (clean)")
        ax.plot(layers, days_base_rs,  "-",  color="crimson",   linewidth=2,
                label="Days of week")
        ax.plot(layers, months_base_rs, "-", color="seagreen",  linewidth=2,
                label="Months of year")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Baseline signal strength")
        ax.set_title("Baseline signal by layer")
        ax.set_xticks(layers)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=9)

        fig.tight_layout()
        p = OUTPUT_DIR / "phase5_all_concepts_comparison.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"  Saved -> {p.name}")

    except FileNotFoundError as e:
        print(f"  Skipping comparison plot (phase3/4 CSVs not found): {e}")

    # -- 9. Print summary tables -----------------------------------------------
    summary_layers = [0, 3, 6, 9, 11]
    for concept, results in [("Days of Week", days_results), ("Months of Year", months_results)]:
        by_layer = {r["layer"]: r for r in results}
        print(f"\n{concept} summary:")
        print(f"  {'Layer':>5}  {'base_r':>8}  {'rmse':>8}  {'max_drop':>10}  {'top_feat':>9}")
        print("  " + "-" * 52)
        for layer in summary_layers:
            r = by_layer.get(layer, {})
            base_r   = r.get("baseline_angular_r", float("nan"))
            rmse     = r.get("baseline_rmse",      float("nan"))
            max_drop = r.get("max_drop",            float("nan"))
            top_feat = r.get("top_feature",         -1)
            md_str = f"{max_drop:+10.4f}" if not (isinstance(max_drop, float) and np.isnan(max_drop)) else f"{'nan':>10}"
            print(f"  {layer:>5}  {base_r:>8.4f}  {rmse:>8.4f}  {md_str}  {top_feat:>9}")

    print("\n" + "=" * 65)
    print("PHASE 5  COMPLETE")
    print(f"  Outputs saved to: {OUTPUT_DIR.resolve()}")
    print("=" * 65)
