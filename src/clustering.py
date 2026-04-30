import numpy as np
import torch
from scipy.stats import f_oneway

from config import CORR_THRESHOLD, MIN_CLUSTER_SZ, N_DISC


def find_clusters(
    feat_acts: torch.Tensor,
    categories: list[str],
    corr_threshold: float = CORR_THRESHOLD,
    min_size: int = MIN_CLUSTER_SZ,
    n_disc: int = N_DISC,
) -> tuple[list[list[int]], np.ndarray]:
    """
    Find SAE feature clusters discriminative for historical epoch categories.

    Selects the top-n_disc features by ANOVA F-score across epoch categories,
    then clusters by Pearson correlation of epoch-mean activation profiles
    (adapted from Engels et al. for small datasets).

    Returns
    -------
    clusters   : list of clusters, each a list of global feature indices
    f_scores   : (d_sae,) ANOVA F-score for every feature (for reporting)
    """
    N, d_sae = feat_acts.shape
    cats    = sorted(set(categories))
    cat_idx = {c: [i for i, cat in enumerate(categories) if cat == c] for c in cats}

    # 1. Active features (fire on >= 2 tokens)
    freq       = (feat_acts > 0).float().sum(0)
    active_idx = freq.ge(2).nonzero(as_tuple=True)[0].tolist()
    print(f"    Active features (fire on >= 2 tokens): {len(active_idx)} / {d_sae}")

    if len(active_idx) < 2:
        return [], np.zeros(d_sae)

    # 2. ANOVA F-score per active feature
    f_scores_active = np.zeros(len(active_idx))
    for k, fi in enumerate(active_idx):
        groups = [feat_acts[cat_idx[c], fi].numpy()
                  for c in cats if len(cat_idx[c]) > 0]
        if len(groups) >= 2:
            try:
                F, _ = f_oneway(*groups)
                f_scores_active[k] = float(F) if np.isfinite(F) else 0.0
            except Exception:
                pass

    f_scores_full = np.zeros(d_sae)
    for k, fi in enumerate(active_idx):
        f_scores_full[fi] = f_scores_active[k]

    # 3. Keep top-n_disc discriminative features
    n_keep     = min(n_disc, len(active_idx))
    top_local  = np.argsort(f_scores_active)[-n_keep:][::-1]
    top_global = [active_idx[i] for i in top_local]
    print(f"    Top-{n_keep} discriminative features selected  "
          f"(max F = {f_scores_active[top_local[0]]:.2f})")

    # 4. Epoch-mean activation profile for each selected feature
    profiles = np.array([
        [feat_acts[cat_idx[c], fi].mean().item() for c in cats]
        for fi in top_global
    ])   # (n_keep, n_cats)

    stds       = profiles.std(1, keepdims=True) + 1e-8
    profiles_z = (profiles - profiles.mean(1, keepdims=True)) / stds
    corr       = (profiles_z @ profiles_z.T) / profiles.shape[1]

    # 5. Build graph and find connected components
    adj = np.abs(corr) > corr_threshold
    np.fill_diagonal(adj, False)

    visited, clusters = np.zeros(n_keep, bool), []
    for start in range(n_keep):
        if visited[start]:
            continue
        component, queue = [], [start]
        while queue:
            node = queue.pop(0)
            if visited[node]:
                continue
            visited[node] = True
            component.append(top_global[node])
            queue.extend(np.where(adj[node])[0].tolist())
        if len(component) >= min_size:
            clusters.append(component)

    return clusters, f_scores_full
