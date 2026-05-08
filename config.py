import torch
from pathlib import Path

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR  = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Primary analysis layer — used by phase1 (PCA overview) and phase4 (linearity plot).
# Also sets HOOK_NAME used by src/model.py extract_activations().
MODEL_NAME  = "gpt2"
LAYER       = 4
HOOK_NAME   = f"blocks.{LAYER}.hook_resid_pre"

# SAE release identifier — used by phases 3, 4, 5, 6 when loading per-layer SAEs.
SAE_RELEASE = "gpt2-small-res-jb"

# Max features selected by ANOVA F-score for ablation — used by phases 3, 4, 5.
N_DISC         = 100

# Layers scanned per phase. Edit these to restrict or extend the depth scan.
PHASE2_LAYERS = [4, 6, 8, 10]   # layers compared in phase2 geometric analysis
PHASE3_LAYERS = list(range(12)) # original BC/AD superposition depth scan
PHASE4_LAYERS = list(range(12)) # clean-prompt superposition depth scan
PHASE5_LAYERS = list(range(12)) # circular superposition scan (days/months)
PHASE6_LAYERS = [4, 6, 7, 8, 10]  # days-of-month geometry analysis layers
