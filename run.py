# Repository structure:
#   run.py                      — entry point, dispatches to phase runners
#   config.py                   — global constants (device, layers, thresholds)
#   data/prompts.json           — all prompt datasets
#   src/data_loading.py         — parses prompts.json into typed tuples
#   src/model.py                — GPT-2 + SAE loading and activation extraction
#   src/clustering.py           — discriminative SAE feature clustering
#   src/metrics.py              — separability and mixture scoring
#   src/geometry.py             — algebraic circle fitting
#   src/plots.py                — all matplotlib output functions
#   phases/phase1.py            — feature clustering + irreducibility at layer 4
#   phases/phase2.py            — geometric structure (circular encoding)
#   phases/phase3.py            — superposition depth scan, original dataset
#   phases/phase4.py            — superposition depth scan, clean dataset

import argparse
from phases.phase1 import run_phase1
from phases.phase2 import run_phase2
from phases.phase3 import run_phase3
from phases.phase4 import run_phase4
from phases.phase5 import run_phase5
from phases.phase6 import run_phase6

parser = argparse.ArgumentParser()
parser.add_argument("--phase", choices=["1", "2", "3", "4", "5", "6", "all"], default="all")
args = parser.parse_args()

if args.phase in ("1", "all"): run_phase1()
if args.phase in ("2", "all"): run_phase2()
if args.phase in ("3", "all"): run_phase3()
if args.phase in ("4", "all"): run_phase4()
if args.phase in ("5", "all"): run_phase5()
if args.phase in ("6", "all"): run_phase6()
