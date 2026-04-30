import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA

from config import DEVICE, OUTPUT_DIR, LAYER, CORR_THRESHOLD
from src.data_loading import YEAR_ITEMS, BATTLE_ITEMS, ALL_ITEMS, ALL_CATS
from src.model import load_model, load_sae, extract_activations, get_feature_acts
from src.clustering import find_clusters
from src.metrics import separability_index, epsilon_mixture_index
from src.plots import plot_pca_overview, plot_year_linearity


def run_phase1() -> None:
    print("=" * 65)
    print("PHASE 1  -  Feature Clustering & Irreducibility Scoring")
    print(f"  model=gpt2   layer={LAYER}   device={DEVICE}")
    print("=" * 65)

    # -- 1. Load model + SAE ---------------------------------------------------
    model = load_model()
    sae   = load_sae()

    # -- 2. Extract residual-stream activations --------------------------------
    print("\n[1]  Extracting residual-stream activations ...")
    year_acts,   year_labels   = extract_activations(model, YEAR_ITEMS)
    battle_acts, battle_labels = extract_activations(model, BATTLE_ITEMS)
    all_acts   = torch.cat([year_acts, battle_acts], dim=0)
    all_labels = year_labels + battle_labels
    print(f"  years: {year_acts.shape}  |  battles: {battle_acts.shape}  |  total: {all_acts.shape}")

    # -- 3. Baseline: PCA directly on raw residual stream ---------------------
    print("\n[2]  Baseline - PCA on raw residual stream ...")
    plot_pca_overview(
        all_acts, all_labels, ALL_CATS,
        title=f"All Historical Tokens - Raw Residual Stream  (Layer {LAYER})",
        path=OUTPUT_DIR / "phase1_all_pca.png",
    )
    plot_year_linearity(
        year_acts, YEAR_ITEMS,
        path=OUTPUT_DIR / "phase1_year_linearity.png",
        layer=LAYER,
    )
    # Report PC correlations in console too
    acts_np = year_acts.numpy()
    numeric = np.array([it[2] for it in YEAR_ITEMS], dtype=float)
    n_pc    = min(3, acts_np.shape[0] - 1, acts_np.shape[1])
    coords  = PCA(n_components=n_pc).fit_transform(acts_np)
    for pc in range(n_pc):
        r = float(np.corrcoef(numeric, coords[:, pc])[0, 1])
        print(f"    PC{pc+1} vs year:  r = {r:+.3f}")

    # -- 4. SAE feature activations --------------------------------------------
    print("\n[3]  Running SAE encoder ...")
    feat_acts   = get_feature_acts(sae, all_acts)
    mean_active = (feat_acts > 0).float().sum(1).mean().item()
    print(f"  feature acts: {feat_acts.shape}  |  mean active / token: {mean_active:.1f}")

    # -- 5. Feature clustering -------------------------------------------------
    print("\n[4]  Clustering discriminative features by epoch-profile correlation ...")
    clusters, f_scores = find_clusters(feat_acts, ALL_CATS, corr_threshold=CORR_THRESHOLD)
    if not clusters:
        print("  No clusters at threshold 0.5 - retrying at 0.3 ...")
        clusters, f_scores = find_clusters(feat_acts, ALL_CATS, corr_threshold=0.3)
    print(f"  Clusters found: {len(clusters)}")

    # Report the 5 most discriminative individual features
    top5_feat = np.argsort(f_scores)[-5:][::-1]
    print("  Top-5 discriminative features by ANOVA F-score:")
    for rank, fi in enumerate(top5_feat):
        print(f"    #{rank+1}  feature {fi:>6}   F = {f_scores[fi]:.2f}")

    # -- 6. Score clusters -----------------------------------------------------
    print("\n[5]  Scoring clusters for irreducibility ...")
    W_dec = sae.W_dec.detach().cpu()          # (d_sae, d_model)

    print(f"  {'ID':>4}  {'|f|':>5}  {'S(f)':>7}  {'M_eps':>7}  {'Irred.':>8}")
    print("  " + "-" * 37)

    scored = []
    for cid, feat_indices in enumerate(clusters):
        fi    = torch.tensor(feat_indices, dtype=torch.long)
        recon = feat_acts[:, fi] @ W_dec[fi, :]          # (N, d_model)

        S     = separability_index(recon, ALL_CATS)
        M_eps = epsilon_mixture_index(recon)
        irred = S * (1.0 - M_eps)

        scored.append(dict(
            id=cid, features=feat_indices, n=len(feat_indices),
            S=S, M_eps=M_eps, irred=irred, recon=recon,
        ))
        print(f"  {cid:>4}  {len(feat_indices):>5}  {S:>7.3f}  {M_eps:>7.3f}  {irred:>8.3f}")

    scored.sort(key=lambda x: x["irred"], reverse=True)

    # -- 7. Visualise top-3 clusters -------------------------------------------
    print("\n[6]  Visualising top-3 clusters by irreducibility ...")
    for rank, entry in enumerate(scored[:3]):
        plot_pca_overview(
            entry["recon"],
            all_labels,
            ALL_CATS,
            title=(
                f"Cluster {entry['id']}  (rank {rank+1}/{len(scored)})  "
                f"S={entry['S']:.3f}  M_eps={entry['M_eps']:.3f}  "
                f"irred={entry['irred']:.3f}"
            ),
            path=OUTPUT_DIR / f"phase1_cluster{entry['id']}_rank{rank+1}.png",
        )

    # -- 8. Feature ablation experiment ----------------------------------------
    print("\n[7]  Feature ablation experiment ...")
    KEY_FEATURES = {
        1772:  "AD token detector (orthographic)",
        16307: "BC token detector (orthographic)",
        8113:  "Historical date marker (context-gated)",
        22008: "Modern year/date context",
    }

    # Baseline: PC1-year correlation from full SAE reconstruction
    year_feat_acts = get_feature_acts(sae, year_acts)
    year_numeric   = np.array([it[2] for it in YEAR_ITEMS], dtype=float)
    W_dec_full     = sae.W_dec.detach().cpu()

    def _pc1_year_absr(fa: torch.Tensor) -> float:
        """Absolute PC1-year correlation (sign-invariant: PCA direction is arbitrary)."""
        recon  = (fa @ W_dec_full).numpy()
        coords = PCA(n_components=1).fit_transform(recon)
        return abs(float(np.corrcoef(year_numeric, coords[:, 0])[0, 1]))

    baseline_r = _pc1_year_absr(year_feat_acts)
    print(f"  Baseline |PC1-year r| (full SAE recon): {baseline_r:.4f}")
    print()
    print(f"  {'Feature':<8}  {'Description':<42}  {'|r| ablated':>11}  {'drop':>8}")
    print("  " + "-" * 73)

    ablation_results = []
    for fi, desc in KEY_FEATURES.items():
        fa_abl = year_feat_acts.clone()
        fa_abl[:, fi] = 0.0
        r_abl = _pc1_year_absr(fa_abl)
        drop  = baseline_r - r_abl        # positive = feature was helping
        ablation_results.append((fi, desc, r_abl, drop))
        print(f"  {fi:<8}  {desc:<42}  {r_abl:>11.4f}  {drop:>+8.4f}")

    # All four ablated simultaneously
    fa_all = year_feat_acts.clone()
    for fi in KEY_FEATURES:
        fa_all[:, fi] = 0.0
    r_all    = _pc1_year_absr(fa_all)
    drop_all = baseline_r - r_all
    print("  " + "-" * 73)
    print(f"  {'all 4':<8}  {'all four ablated':<42}  {r_all:>11.4f}  {drop_all:>+8.4f}")

    print()
    print("  Interpretation guide (drop = baseline |r| - ablated |r|):")
    print("    drop < 0.02  -> feature contributes negligibly (encoding is distributed)")
    print("    drop 0.02-0.1 -> modest contribution")
    print("    drop > 0.1   -> feature is load-bearing for temporal axis")

    # -- 8b. Full cluster ablation scan ----------------------------------------
    if scored:
        print("\n[8]  Full cluster ablation scan (all features individually) ...")
        cluster_feats = scored[0]["features"]   # the single top cluster
        scan_results  = []
        for fi in tqdm(cluster_feats, desc="  ablation scan", leave=False):
            fa_abl = year_feat_acts.clone()
            fa_abl[:, fi] = 0.0
            r_abl = _pc1_year_absr(fa_abl)
            scan_results.append((fi, r_abl, baseline_r - r_abl))

        scan_results.sort(key=lambda x: x[2], reverse=True)   # sort by drop desc

        print(f"  {'Rank':<5}  {'Feature':<8}  {'|r| ablated':>11}  {'drop':>8}")
        print("  " + "-" * 37)
        for rank, (fi, r_abl, drop) in enumerate(scan_results):
            marker = "  <-- load-bearing" if drop > 0.1 else ""
            print(f"  {rank+1:<5}  {fi:<8}  {r_abl:>11.4f}  {drop:>+8.4f}{marker}")

        drops        = np.array([d for _, _, d in scan_results])
        n_load       = int((drops > 0.10).sum())
        n_moderate   = int(((drops > 0.02) & (drops <= 0.10)).sum())
        n_negligible = int((drops <= 0.02).sum())
        print()
        print(f"  Summary: {n_load} load-bearing (>0.10)  |  "
              f"{n_moderate} moderate (0.02-0.10)  |  "
              f"{n_negligible} negligible (<0.02)")

        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.hist(drops, bins=20, color="steelblue", edgecolor="white", linewidth=0.5)
        ax.axvline(0.10, color="crimson",  linestyle="--", linewidth=1.2,
                   label="load-bearing threshold (0.10)")
        ax.axvline(0.02, color="darkorange", linestyle="--", linewidth=1.2,
                   label="moderate threshold (0.02)")
        ax.set_xlabel("Drop in |PC1-year r| when feature ablated")
        ax.set_ylabel("Number of features")
        ax.set_title(f"Single-feature ablation across {len(cluster_feats)} cluster features\n"
                     f"baseline |r| = {baseline_r:.4f}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        hist_path = OUTPUT_DIR / "phase1_ablation_histogram.png"
        fig.savefig(hist_path, dpi=150)
        plt.close(fig)
        print(f"  Saved -> {hist_path.name}")

    # -- 9. Summary ------------------------------------------------------------
    print("\n" + "=" * 65)
    print("PHASE 1  COMPLETE")
    if scored:
        b = scored[0]
        print(f"  Best cluster: #{b['id']}  ({b['n']} features)")
        print(f"    Separability   S(f) = {b['S']:.3f}")
        print(f"    eps-mixture      M_eps  = {b['M_eps']:.3f}")
        print(f"    Irreducibility      = {b['irred']:.3f}")
    print(f"  Outputs saved to: {OUTPUT_DIR.resolve()}")
    print("=" * 65)
