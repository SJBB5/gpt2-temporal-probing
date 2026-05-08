import csv
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from config import OUTPUT_DIR, DEVICE, SAE_RELEASE, PHASE4_LAYERS
from src.model import load_model, extract_activations
from src.plots import plot_year_linearity
from src.data_loading import MATCHED_CLEAN_ITEMS, MATCHED_CLEAN_CATEGORIES
import phases.phase3 as _phase3_mod
from phases.phase3 import _ablation_scan_at_layer

COMPARE_LAYERS = [0, 1, 4, 8, 11]


def run_phase4() -> None:
    print("\n" + "=" * 65)
    print("PHASE 4  -  Clean Prompts Superposition Depth Scan")
    print("  (no BC/AD tokens)")
    print(f"  layers={PHASE4_LAYERS}")
    print("=" * 65)

    phase_dir = OUTPUT_DIR / "phase4"
    phase_dir.mkdir(parents=True, exist_ok=True)

    # -- 1. Load model ---------------------------------------------------------
    print(f"  Using {len(MATCHED_CLEAN_ITEMS)} matched clean prompts")
    model = load_model()

    # -- 2. Linearity check on clean data at layer 4 --------------------------
    print("\n[Linearity]  Extracting activations at layer 4 (clean) ...")
    acts_l4, _ = extract_activations(model, MATCHED_CLEAN_ITEMS)
    plot_year_linearity(
        acts_l4, MATCHED_CLEAN_ITEMS,
        path=phase_dir / "phase4_clean_year_linearity.png",
        layer=4,
    )
    acts_np = acts_l4.numpy()
    numeric = np.array([it[2] for it in MATCHED_CLEAN_ITEMS], dtype=float)
    n_pc = min(3, acts_np.shape[0] - 1, acts_np.shape[1])
    coords = PCA(n_components=n_pc).fit_transform(acts_np)
    for pc in range(n_pc):
        r = float(np.corrcoef(numeric, coords[:, pc])[0, 1])
        print(f"    PC{pc+1} vs year:  r = {r:+.3f}")

    # -- 4. Monkey-patch phase3 module globals to use clean data ---------------
    _orig_year_items = _phase3_mod.YEAR_ITEMS
    _orig_all_cats   = _phase3_mod.ALL_CATS

    _phase3_mod.YEAR_ITEMS = MATCHED_CLEAN_ITEMS
    _phase3_mod.ALL_CATS   = MATCHED_CLEAN_CATEGORIES

    # -- 5. Run superposition depth scan on clean data -------------------------
    clean_results = []
    for layer in PHASE4_LAYERS:
        print(f"\n--- Layer {layer} (clean) ---")
        clean_results.append(_ablation_scan_at_layer(model, layer))

    # -- 6. Restore originals and run comparison layers for original data ------
    _phase3_mod.YEAR_ITEMS = _orig_year_items
    _phase3_mod.ALL_CATS   = _orig_all_cats

    print("\n" + "=" * 65)
    print("  Running original data scan at comparison layers ...")
    orig_results = {}
    for layer in COMPARE_LAYERS:
        print(f"\n--- Layer {layer} (original) ---")
        orig_results[layer] = _ablation_scan_at_layer(model, layer)

    # -- 7. Comparison plot identical in layout to phase3 ---------------------
    layers_valid = [r["layer"] for r in clean_results if not np.isnan(r["max_drop"])]
    max_drops    = [r["max_drop"] for r in clean_results if not np.isnan(r["max_drop"])]
    base_rs      = [r["baseline_r"] for r in clean_results if not np.isnan(r["baseline_r"])]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle("Clean Prompts Only (no BC/AD tokens)", fontsize=11)

    ax = axes[0]
    ax.plot(layers_valid, max_drops, "o-", color="steelblue", linewidth=2, markersize=6)
    ax.axhline(0.10, color="crimson",    linestyle="--", linewidth=1.1,
               label="load-bearing threshold (0.10)")
    ax.axhline(0.02, color="darkorange", linestyle="--", linewidth=1.1,
               label="moderate threshold (0.02)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Max single-feature drop in |PC1-year r|")
    ax.set_title("Concentration of temporal signal\n(higher = less superposition)")
    ax.set_xticks(layers_valid)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(layers_valid, base_rs, "s-", color="seagreen", linewidth=2, markersize=6)
    ax.set_xlabel("Layer")
    ax.set_ylabel("|PC1-year r| (full SAE recon)")
    ax.set_title("Temporal axis strength by layer")
    ax.set_xticks(layers_valid)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    p = phase_dir / "phase4_clean_superposition_depth.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"\n  Saved -> {p.name}")

    # -- 8. Side-by-side summary table ----------------------------------------
    clean_by_layer = {r["layer"]: r for r in clean_results}

    print("\n" + "=" * 65)
    print("PHASE 4  COMPARISON  (clean vs original)")
    header = (
        f"  {'Layer':>5}  {'Clean base|r|':>14}  {'Orig base|r|':>12}  "
        f"{'Clean max_drop':>14}  {'Orig max_drop':>13}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for layer in COMPARE_LAYERS:
        cr  = clean_by_layer.get(layer, {})
        orr = orig_results.get(layer, {})
        c_r  = cr.get("baseline_r", float("nan"))
        o_r  = orr.get("baseline_r", float("nan"))
        c_md = cr.get("max_drop",    float("nan"))
        o_md = orr.get("max_drop",   float("nan"))
        print(
            f"  {layer:>5}  {c_r:>14.4f}  {o_r:>12.4f}  "
            f"  {c_md:>+13.4f}  {o_md:>+13.4f}"
        )

    # -- CSV output: phase4 ---------------------------------------------------
    p4_csv = phase_dir / "phase4_layer_summary.csv"
    with open(p4_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "layer", "base_r", "max_drop", "top_feature_index",
            "n_load_bearing", "n_moderate", "n_negative",
        ])
        writer.writeheader()
        for r in clean_results:
            n_neg = int((r["all_drops"] < 0).sum()) if len(r["all_drops"]) > 0 else 0
            writer.writerow({
                "layer":             r["layer"],
                "base_r":            round(r["baseline_r"], 4),
                "max_drop":          round(r["max_drop"], 4),
                "top_feature_index": r["top_feature"],
                "n_load_bearing":    r["n_load"],
                "n_moderate":        r["n_moderate"],
                "n_negative":        n_neg,
            })
    print(f"  Saved -> phase4_layer_summary.csv")

    # -- CSV output: comparison -----------------------------------------------
    p3_csv = OUTPUT_DIR / "phase3" / "phase3_layer_summary.csv"
    p_cmp  = phase_dir / "phase3_vs_phase4_comparison.csv"
    with open(p3_csv, newline="") as f3, open(p4_csv, newline="") as f4:
        p3_rows = {int(r["layer"]): r for r in csv.DictReader(f3)}
        p4_rows = {int(r["layer"]): r for r in csv.DictReader(f4)}
    with open(p_cmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "layer", "orig_base_r", "clean_base_r",
            "orig_max_drop", "clean_max_drop",
            "orig_top_feature", "clean_top_feature",
        ])
        writer.writeheader()
        for layer in sorted(p4_rows):
            p3 = p3_rows.get(layer, {})
            p4 = p4_rows[layer]
            writer.writerow({
                "layer":             layer,
                "orig_base_r":       p3.get("base_r", ""),
                "clean_base_r":      p4["base_r"],
                "orig_max_drop":     p3.get("max_drop", ""),
                "clean_max_drop":    p4["max_drop"],
                "orig_top_feature":  p3.get("top_feature_index", ""),
                "clean_top_feature": p4["top_feature_index"],
            })
    print(f"  Saved -> phase3_vs_phase4_comparison.csv")

    print("\n" + "=" * 65)
    print("PHASE 4  COMPLETE")
    print(f"  Outputs saved to: {OUTPUT_DIR.resolve()}")
    print("=" * 65)
