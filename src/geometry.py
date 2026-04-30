import numpy as np
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

from config import DEVICE
from src.data_loading import YEAR_ITEMS


def fit_circle_algebraic(points: np.ndarray) -> tuple[float, float, float, float]:
    """
    Algebraic least-squares circle fit (Coope 1993).
    Returns (cx, cy, radius, RMSE).
    """
    X, Y = points[:, 0], points[:, 1]
    A    = np.column_stack([2 * X, 2 * Y, np.ones(len(X))])
    b    = X ** 2 + Y ** 2
    res, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = float(res[0]), float(res[1])
    r      = float(np.sqrt(res[2] + cx ** 2 + cy ** 2))
    radii  = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    rmse   = float(np.sqrt(np.mean((radii - r) ** 2)))
    return cx, cy, r, rmse


def _extract_layer_acts(model: HookedTransformer, layer: int) -> np.ndarray:
    """Extract last-token residual stream activations for all YEAR_ITEMS."""
    hook = f"blocks.{layer}.hook_resid_pre"
    rows = []
    for _, prompt, _ in tqdm(YEAR_ITEMS, desc=f"  L{layer}", leave=False):
        tokens = model.to_tokens(prompt, prepend_bos=True)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook, device=DEVICE)
        rows.append(cache[hook][0, -1, :].cpu().numpy())
    return np.stack(rows)    # (N_years, 768)
