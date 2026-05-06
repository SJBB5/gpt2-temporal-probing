import torch
from pathlib import Path

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR  = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_NAME  = "gpt2"
LAYER       = 4
HOOK_NAME   = f"blocks.{LAYER}.hook_resid_pre"

SAE_RELEASE = "gpt2-small-res-jb"
SAE_ID      = f"blocks.{LAYER}.hook_resid_pre"

CORR_THRESHOLD = 0.5
MIN_CLUSTER_SZ = 3
N_DISC         = 100

PHASE2_LAYERS = [4, 6, 8, 10]
PHASE3_LAYERS = list(range(12))
PHASE4_LAYERS = list(range(12))
PHASE5_LAYERS = list(range(12))
