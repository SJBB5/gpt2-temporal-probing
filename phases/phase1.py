import numpy as np
from sklearn.decomposition import PCA

from config import DEVICE, OUTPUT_DIR, LAYER
from src.data_loading import YEAR_ITEMS, YEAR_CATEGORIES
from src.model import load_model, extract_activations
from src.plots import plot_pca_overview, plot_year_linearity


def run_phase1() -> None:
    print("=" * 65)
    print("PHASE 1  -  PCA Overview & Linear Decodability")
    print(f"  model=gpt2   layer={LAYER}   device={DEVICE}")
    print("=" * 65)

    phase_dir = OUTPUT_DIR / "phase1"
    phase_dir.mkdir(parents=True, exist_ok=True)

    # -- 1. Load model ---------------------------------------------------------
    model = load_model()

    # -- 2. Extract residual-stream activations --------------------------------
    print("\n[1]  Extracting residual-stream activations ...")
    year_acts, year_labels = extract_activations(model, YEAR_ITEMS)
    print(f"  year acts: {year_acts.shape}")

    # -- 3. PCA on raw residual stream ----------------------------------------
    print("\n[2]  PCA on raw residual stream ...")
    plot_pca_overview(
        year_acts, year_labels, YEAR_CATEGORIES,
        title=f"All Historical Tokens - Raw Residual Stream  (Layer {LAYER})",
        path=phase_dir / "phase1_all_pca.png",
    )
    plot_year_linearity(
        year_acts, YEAR_ITEMS,
        path=phase_dir / "phase1_year_linearity.png",
        layer=LAYER,
    )
    acts_np = year_acts.numpy()
    numeric = np.array([it[2] for it in YEAR_ITEMS], dtype=float)
    n_pc    = min(3, acts_np.shape[0] - 1, acts_np.shape[1])
    coords  = PCA(n_components=n_pc).fit_transform(acts_np)
    for pc in range(n_pc):
        r = float(np.corrcoef(numeric, coords[:, pc])[0, 1])
        print(f"    PC{pc+1} vs year:  r = {r:+.3f}")

    # -- 4. Summary ------------------------------------------------------------
    print("\n" + "=" * 65)
    print("PHASE 1  COMPLETE")
    print(f"  Outputs saved to: {OUTPUT_DIR.resolve()}")
    print("=" * 65)
