import csv
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import f_oneway
from sklearn.decomposition import PCA
from sae_lens import SAE
from transformer_lens import HookedTransformer

from config import DEVICE, OUTPUT_DIR, SAE_RELEASE, N_DISC, PHASE3_LAYERS
from src.data_loading import (
    PHASE3_YEAR_ITEMS      as YEAR_ITEMS,
    PHASE3_YEAR_CATEGORIES as ALL_CATS,
)
from src.model import load_model


def _ablation_scan_at_layer(
    model: HookedTransformer,
    layer: int,
) -> dict:
    """
    For a single layer, run the full cluster ablation scan and return summary
    statistics used to trace how superposition evolves with depth.

    Returns a dict with keys:
        layer, baseline_r, max_drop, top_feature,
        n_load, n_moderate, n_negligible, all_drops (np.ndarray)
    """
    hook   = f"blocks.{layer}.hook_resid_pre"
    sae_id = f"blocks.{layer}.hook_resid_pre"

    # Load SAE for this layer
    print(f"  Loading SAE for layer {layer} ...")
    sae, _, _ = SAE.from_pretrained(
        release=SAE_RELEASE, sae_id=sae_id, device=DEVICE,
    )
    sae.eval()

    # Extract activations and encode through SAE
    rows = []
    for _, prompt, _ in tqdm(YEAR_ITEMS, desc=f"  L{layer} acts", leave=False):
        tokens = model.to_tokens(prompt, prepend_bos=True)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook, device=DEVICE)
        rows.append(cache[hook][0, -1, :].cpu())
    acts = torch.stack(rows)                    # (N_years, 768)

    with torch.no_grad():
        feat_acts_year = sae.encode(acts.to(DEVICE)).cpu()

    # Active features
    freq       = (feat_acts_year > 0).float().sum(0)
    active_idx = freq.ge(2).nonzero(as_tuple=True)[0].tolist()
    print(f"    Active features (fire on >= 2 tokens): {len(active_idx)}")

    if not active_idx:
        print(f"    No active features at layer {layer} — skipping.")
        return dict(layer=layer, baseline_r=float("nan"), max_drop=float("nan"),
                    top_feature=-1, n_load=0, n_moderate=0, n_negligible=0,
                    all_drops=np.array([]))

    # ANOVA F-score per active feature
    cats    = sorted(set(ALL_CATS))
    cat_idx = {c: [i for i, cat in enumerate(ALL_CATS) if cat == c] for c in cats}
    f_scores = np.zeros(len(active_idx))
    for k, fi in enumerate(active_idx):
        groups = [feat_acts_year[cat_idx[c], fi].numpy()
                  for c in cats if len(cat_idx[c]) > 0]
        if len(groups) >= 2:
            try:
                F, _ = f_oneway(*groups)
                f_scores[k] = float(F) if np.isfinite(F) else 0.0
            except Exception:
                pass

    # Select top-N_DISC by F-score
    n_keep      = min(N_DISC, len(active_idx))
    top_local   = np.argsort(f_scores)[-n_keep:][::-1]
    top_feats   = [active_idx[i] for i in top_local]
    print(f"    Top-{n_keep} features by ANOVA F-score  (max F = {f_scores[top_local[0]]:.2f})")
    W_dec        = sae.W_dec.detach().cpu()
    year_numeric = np.array([it[2] for it in YEAR_ITEMS], dtype=float)

    def _absr(fa: torch.Tensor) -> float:
        recon  = (fa @ W_dec).numpy()
        coords = PCA(n_components=1).fit_transform(recon)
        return abs(float(np.corrcoef(year_numeric, coords[:, 0])[0, 1]))

    baseline_r = _absr(feat_acts_year)

    scan = []
    for fi in tqdm(top_feats, desc=f"  L{layer} scan", leave=False):
        fa_abl = feat_acts_year.clone()
        fa_abl[:, fi] = 0.0
        drop = baseline_r - _absr(fa_abl)
        scan.append((fi, drop))

    scan.sort(key=lambda x: x[1], reverse=True)
    drops    = np.array([d for _, d in scan])
    top_feat = scan[0][0]
    n_load   = int((drops > 0.10).sum())
    n_mod    = int(((drops > 0.02) & (drops <= 0.10)).sum())
    n_neg    = int((drops <= 0.02).sum())

    print(f"    Layer {layer}: baseline |r|={baseline_r:.4f}  "
          f"max_drop={drops[0]:+.4f}  top_feat={top_feat}  "
          f"load={n_load}  mod={n_mod}  neg={n_neg}")

    return dict(layer=layer, baseline_r=baseline_r, max_drop=drops[0],
                top_feature=top_feat, n_load=n_load, n_moderate=n_mod,
                n_negligible=n_neg, all_drops=drops)


def run_phase3() -> None:
    print("\n" + "=" * 65)
    print("PHASE 3  -  Superposition Depth Scan")
    print(f"  layers={PHASE3_LAYERS}")
    print("=" * 65)

    model = load_model()
    results = []
    for layer in PHASE3_LAYERS:
        print(f"\n--- Layer {layer} ---")
        results.append(_ablation_scan_at_layer(model, layer))

    # -- Summary table --------------------------------------------------------
    print("\n" + "=" * 65)
    print("PHASE 3  SUMMARY")
    print(f"  {'Layer':>6}  {'base |r|':>9}  {'max drop':>9}  "
          f"{'top feat':>9}  {'load':>5}  {'mod':>5}  {'neg':>5}")
    print("  " + "-" * 58)
    for r in results:
        print(f"  {r['layer']:>6}  {r['baseline_r']:>9.4f}  {r['max_drop']:>+9.4f}  "
              f"  {r['top_feature']:>8}  {r['n_load']:>5}  "
              f"{r['n_moderate']:>5}  {r['n_negligible']:>5}")

    # -- CSV output -----------------------------------------------------------
    phase_dir = OUTPUT_DIR / "phase3"
    phase_dir.mkdir(parents=True, exist_ok=True)
    p_csv = phase_dir / "phase3_layer_summary.csv"
    with open(p_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "layer", "base_r", "max_drop", "top_feature_index",
            "n_load_bearing", "n_moderate", "n_negative",
        ])
        writer.writeheader()
        for r in results:
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
    print(f"  Saved -> phase3_layer_summary.csv")

    # -- Plot 1: max single-feature drop vs layer -----------------------------
    layers_valid = [r["layer"] for r in results if not np.isnan(r["max_drop"])]
    max_drops    = [r["max_drop"] for r in results if not np.isnan(r["max_drop"])]
    base_rs      = [r["baseline_r"] for r in results if not np.isnan(r["baseline_r"])]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    ax = axes[0]
    ax.plot(layers_valid, max_drops, "o-", color="steelblue", linewidth=2, markersize=6)
    ax.axhline(0.10, color="crimson",   linestyle="--", linewidth=1.1,
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
    p = phase_dir / "phase3_superposition_depth.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"\n  Saved -> {p.name}")

    # -- Plot 2: per-layer drop distributions (violin) ------------------------
    valid = [r for r in results if len(r["all_drops"]) > 0]
    if valid:
        fig, ax = plt.subplots(figsize=(max(10, len(valid) * 1.1), 4))
        data  = [r["all_drops"] for r in valid]
        lbls  = [str(r["layer"]) for r in valid]
        parts = ax.violinplot(data, positions=range(len(valid)),
                              showmedians=True, showextrema=True)
        for pc in parts["bodies"]:
            pc.set_facecolor("steelblue")
            pc.set_alpha(0.6)
        ax.set_xticks(range(len(valid)))
        ax.set_xticklabels([f"Layer {l}" for l in lbls], rotation=45, ha="right")
        ax.axhline(0.10, color="crimson",   linestyle="--", linewidth=1.1,
                   label="load-bearing (0.10)")
        ax.axhline(0.02, color="darkorange", linestyle="--", linewidth=1.1,
                   label="moderate (0.02)")
        ax.set_ylabel("Drop in |PC1-year r|")
        ax.set_title("Distribution of single-feature ablation drops by layer\n"
                     "(narrowing = more distributed / more superposition)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p2 = phase_dir / "phase3_drop_distributions.png"
        fig.savefig(p2, dpi=150)
        plt.close(fig)
        print(f"  Saved -> {p2.name}")

    print("\n" + "=" * 65)
    print("PHASE 3  COMPLETE")
    print(f"  Outputs saved to: {OUTPUT_DIR.resolve()}")
    print("=" * 65)
