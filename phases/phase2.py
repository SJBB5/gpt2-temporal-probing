import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from config import DEVICE, OUTPUT_DIR, LAYER, PHASE2_LAYERS
from src.data_loading import YEAR_ITEMS, YEAR_CATEGORIES
from src.model import load_model
from src.metrics import circularity_score
from src.geometry import fit_circle_algebraic, _extract_layer_acts
from src.plots import _plot_geometric_analysis


def run_phase2() -> None:
    phase_dir = OUTPUT_DIR / "phase2"
    phase_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print("PHASE 2  -  Geometric Analysis")
    print("=" * 65)

    model       = load_model()
    year_nums   = np.array([it[2] for it in YEAR_ITEMS], dtype=float)
    year_labels = [it[0] for it in YEAR_ITEMS]

    # -- 1. Extract activations at each analysis layer -------------------------
    print("\n[1]  Extracting year activations across layers ...")
    layer_data: dict = {}
    for layer in PHASE2_LAYERS:
        acts   = _extract_layer_acts(model, layer)
        n_pcs  = min(5, acts.shape[0] - 1, acts.shape[1])
        pca    = PCA(n_components=n_pcs)
        coords = pca.fit_transform(acts)
        var    = pca.explained_variance_ratio_
        layer_data[layer] = dict(acts=acts, pca=pca, coords=coords, var=var)
        r1 = float(np.corrcoef(year_nums, coords[:, 0])[0, 1])
        print(f"  Layer {layer:>2}:  PC1 r={r1:+.3f}  "
              f"var(PC1+PC2+PC3)={var[:3].sum():.1%}")

    # -- 2. Select layer with strongest PC1-year correlation -------------------
    print("\n[2]  Selecting analysis layer by PC1-year correlation ...")
    best_layer, best_r1 = LAYER, -np.inf
    for layer, ld in layer_data.items():
        r1 = float(np.corrcoef(year_nums, ld["coords"][:, 0])[0, 1])
        if abs(r1) > abs(best_r1):
            best_layer, best_r1 = layer, r1
    print(f"  Best: layer {best_layer}  (PC1 r = {best_r1:+.4f})")

    # -- 3. Detailed geometric analysis at best layer --------------------------
    print(f"\n[3]  Detailed geometric analysis at layer {best_layer} ...")
    ld     = layer_data[best_layer]
    acts   = ld["acts"]
    coords = ld["coords"]
    var    = ld["var"]

    # Detrend: remove the dominant axis (PC1) to expose residual structure
    pc1_dir   = ld["pca"].components_[0]
    proj_pc1  = (acts @ pc1_dir[:, None]) * pc1_dir[None, :]
    acts_detr = acts - proj_pc1
    n_pcs_d   = min(3, acts_detr.shape[0] - 1, acts_detr.shape[1])
    pca_d     = PCA(n_components=n_pcs_d)
    coords_d  = pca_d.fit_transform(acts_detr)
    var_d     = pca_d.explained_variance_ratio_

    # Circularity scores
    circ_raw  = circularity_score(coords[:, 1:3])
    circ_detr = circularity_score(coords_d[:, :2])
    print(f"  Circularity (raw PC2 vs PC3):         {circ_raw:.4f}")
    print(f"  Circularity (detrended PC1 vs PC2):   {circ_detr:.4f}")

    # Circle fit + arc angles for BC and AD subsets
    bc_mask = [i for i, c in enumerate(YEAR_CATEGORIES) if c == "ancient_bc"]
    ad_mask = [i for i, c in enumerate(YEAR_CATEGORIES) if c == "ancient_ad"]
    cx, cy, r, rmse = fit_circle_algebraic(coords[:, 1:3])
    angles  = np.arctan2(coords[:, 2] - cy, coords[:, 1] - cx)
    bc_ang  = np.degrees(angles[bc_mask])
    ad_ang  = np.degrees(angles[ad_mask])
    arc_sep = float(abs(bc_ang.mean() - ad_ang.mean()))
    arc_sep = min(arc_sep, 360 - arc_sep)
    print(f"  Circle fit (PC2 vs PC3):  cx={cx:.3f}  cy={cy:.3f}  "
          f"r={r:.3f}  RMSE={rmse:.4f}")
    print(f"  BC arc:  mean={bc_ang.mean():+.1f} deg  std={bc_ang.std():.1f} deg")
    print(f"  AD arc:  mean={ad_ang.mean():+.1f} deg  std={ad_ang.std():.1f} deg")
    print(f"  BC-AD arc separation: {arc_sep:.1f} deg  "
          f"({'opposite sides' if arc_sep > 120 else 'overlapping'})")

    # Within-group temporal ordering
    for mask, label in [(bc_mask, "BC"), (ad_mask, "AD")]:
        yrs_g = year_nums[mask]
        r_g   = float(np.corrcoef(yrs_g, coords[mask, 0])[0, 1]) if len(mask) > 2 else float("nan")
        print(f"  Within-{label} PC1 vs year r:      {r_g:+.3f}")

    # -- 4. Visualise ----------------------------------------------------------
    print("\n[4]  Saving plots ...")
    _plot_geometric_analysis(
        coords, var, angles, bc_mask, ad_mask,
        coords_d, var_d, year_nums, year_labels,
        best_layer,
        year_categories=YEAR_CATEGORIES,
        output_dir=phase_dir,
    )

    print("\n" + "=" * 65)
    print("PHASE 2  COMPLETE")
    print(f"  Outputs saved to: {OUTPUT_DIR.resolve()}")
    print("=" * 65)
