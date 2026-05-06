import json
from pathlib import Path

_DATA_PATH = Path(__file__).parent.parent / "data" / "prompts.json"

with open(_DATA_PATH) as _f:
    _prompts = json.load(_f)

# ── Phase 1 & 2 ───────────────────────────────────────────────────────────────
_p12 = _prompts["phase1_2"]

YEAR_ITEMS      = [(e["label"], e["prompt"], e["year"]) for e in _p12["year_items"]]
YEAR_CATEGORIES = _p12["year_categories"]

# ── Phase 3 ───────────────────────────────────────────────────────────────────
_p3 = _prompts["phase3"]

PHASE3_YEAR_ITEMS      = [(e["label"], e["prompt"], e["year"]) for e in _p3["year_items"]]
PHASE3_YEAR_CATEGORIES = _p3["year_categories"]

# ── Phase 4 ───────────────────────────────────────────────────────────────────
_p4 = _prompts["phase4"]

MATCHED_CLEAN_ITEMS      = [(e["label"], e["prompt"], e["year"]) for e in _p4["year_items"]]
MATCHED_CLEAN_CATEGORIES = _p4["year_categories"]
