import numpy as np
from sklearn.decomposition import PCA

from src.geometry import fit_circle_algebraic


def fit_and_score_circle(coords: np.ndarray, labels: list) -> dict:
    """
    Given coords (N x 3+ numpy array of PCA components) and numeric labels
    (day numbers 1-7 or month numbers 1-12), fit a circle to PC2/PC3 and
    compute the circular-linear correlation between angular position and label.

    Returns dict with keys:
      rmse, angular_r, angles, cx, cy, radius
    """
    points_2d = np.column_stack([coords[:, 1], coords[:, 2]])
    try:
        cx, cy, radius, rmse = fit_circle_algebraic(points_2d)
    except Exception:
        return dict(rmse=999.0, angular_r=0.0, angles=np.zeros(len(labels)),
                    cx=0.0, cy=0.0, radius=0.0)

    angles = np.arctan2(coords[:, 2] - cy, coords[:, 1] - cx)

    labels_arr = np.array(labels)

    rxs = float(np.corrcoef(np.sin(angles), labels_arr)[0, 1])
    rxc = float(np.corrcoef(np.cos(angles), labels_arr)[0, 1])
    rcs = float(np.corrcoef(np.sin(angles), np.cos(angles))[0, 1])

    denom = 1.0 - rcs ** 2
    if abs(denom) < 1e-8:
        angular_r = 0.0
    else:
        val = (rxc ** 2 + rxs ** 2 - 2 * rxc * rxs * rcs) / denom
        angular_r = float(np.sqrt(max(val, 0.0)))

    return dict(rmse=rmse, angular_r=angular_r, angles=angles, cx=cx, cy=cy, radius=radius)


def cluster_quality_check(coords: np.ndarray, labels: list, n_per_cluster: int) -> float:
    """
    Check whether prompts from the same day/month cluster together in PCA.
    Returns mean intra-cluster distance / mean inter-cluster distance.
    Lower ratio = better clustering (same-day prompts are closer together).
    """
    coords3 = coords[:, :3]
    unique_labels = sorted(set(labels))
    groups = {l: [i for i, lbl in enumerate(labels) if lbl == l] for l in unique_labels}

    intra_dists = []
    for idxs in groups.values():
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                intra_dists.append(float(np.linalg.norm(coords3[idxs[i]] - coords3[idxs[j]])))

    inter_dists = []
    label_list = unique_labels
    for i in range(len(label_list)):
        for j in range(i + 1, len(label_list)):
            for ii in groups[label_list[i]]:
                for jj in groups[label_list[j]]:
                    inter_dists.append(float(np.linalg.norm(coords3[ii] - coords3[jj])))

    if not intra_dists or not inter_dists:
        return 1.0
    return float(np.mean(intra_dists) / (np.mean(inter_dists) + 1e-8))


def circularity_score(points_2d: np.ndarray) -> float:
    """
    1 - std(radial distances) / mean(radial distances) from the fitted circle.
    Score of 1.0 = perfect circle; 0.0 = no circular structure.
    """
    try:
        cx, cy, _, _ = fit_circle_algebraic(points_2d)
        radii = np.sqrt((points_2d[:, 0] - cx) ** 2 + (points_2d[:, 1] - cy) ** 2)
        return float(np.clip(1.0 - radii.std() / (radii.mean() + 1e-8), 0.0, 1.0))
    except Exception:
        return 0.0
