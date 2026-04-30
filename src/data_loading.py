import json
from pathlib import Path

_DATA_PATH = Path(__file__).parent.parent / "data" / "prompts.json"

with open(_DATA_PATH) as _f:
    _prompts = json.load(_f)

BC_AD_PAIRS = [
    (e["year"], e["bc_prompt"], e["ad_prompt"])
    for e in _prompts["bc_ad_pairs"]
]

YEAR_ITEMS = (
    [(f"{yr}BC", prompt_bc, -yr) for yr, prompt_bc, _ in BC_AD_PAIRS] +
    [(f"{yr}AD", prompt_ad,  yr) for yr, _, prompt_ad in BC_AD_PAIRS] +
    [(e["label"], e["prompt"], e["year"]) for e in _prompts["extra_year_items"]]
)

YEAR_CATEGORIES = (
    ["ancient_bc"]   * len(BC_AD_PAIRS) +
    ["ancient_ad"]   * len(BC_AD_PAIRS) +
    ["medieval"]     * 7  +
    ["early_modern"] * 7  +
    ["modern"]       * 10
)
assert len(YEAR_CATEGORIES) == len(YEAR_ITEMS), (
    f"YEAR_CATEGORIES length {len(YEAR_CATEGORIES)} != YEAR_ITEMS length {len(YEAR_ITEMS)}"
)

BATTLE_EUROPE  = [(e["label"], e["prompt"], e["year"]) for e in _prompts["battle_europe"]]
BATTLE_AMERICA = [(e["label"], e["prompt"], e["year"]) for e in _prompts["battle_america"]]
BATTLE_EAST    = [(e["label"], e["prompt"], e["year"]) for e in _prompts["battle_east"]]

BATTLE_ITEMS = BATTLE_EUROPE + BATTLE_AMERICA + BATTLE_EAST

ALL_ITEMS = YEAR_ITEMS + BATTLE_ITEMS
ALL_CATS  = (
    YEAR_CATEGORIES
    + ["battle_europe"]  * len(BATTLE_EUROPE)
    + ["battle_america"] * len(BATTLE_AMERICA)
    + ["battle_east"]    * len(BATTLE_EAST)
)
