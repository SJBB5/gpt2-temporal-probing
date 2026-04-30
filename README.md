# GPT-2 Temporal Probing

Mechanistic interpretability project investigating how GPT-2-small encodes structured temporal concepts in its residual stream using Sparse Autoencoders (SAEs).

---

## Research Question

**Do transformers encode structured temporal concepts — historical years, days of the week, months of the year, and days of the month — as geometric objects in the residual stream, and if so, what is the shape of that geometry (linear axis, circle, or superposition of both)?**

A secondary question: is any such geometry implemented via a small number of load-bearing SAE features (concentrated) or spread across many features (superposed)?

### Relation to existing literature

- **Gurnee & Tegmark (2023)** showed that LLMs linearly encode space and time. We extend this by asking whether the encoding is *purely* linear or has additional geometric structure (circular components), and how that structure changes layer-by-layer.
- **Engels et al. (2024)** showed that GPT-2 encodes days of the week and months of the year as circles in the residual stream — the first demonstration of non-linear (multi-dimensional) features in SAEs. We replicate their circular findings and extend to a novel concept: **days of the month (1–31)**, which has never been tested.
- **Elhage et al. (2022)** — superposition theory predicts that a model under embedding pressure will store multiple features in shared dimensions. We operationalise this as a depth scan: does temporal information concentrate in a single SAE feature at some layers, or is it always distributed?

---

## Method

**Probe point:** Last-token residual stream at `blocks.{layer}.hook_resid_pre` for each prompt.
**SAE:** Joseph Bloom's `gpt2-small-res-jb` release, loaded per-layer.
**Geometry:** PCA on SAE reconstructions; algebraic circle fit (Coope 1993) to PC2/PC3 plane; circular-linear correlation (`angular_r`) to measure how well angular position on the fitted circle predicts the label.
**Superposition probe:** Single-feature ablation scan — zero out one SAE feature at a time and measure drop in |PC1-label correlation|. Large drop = load-bearing; small drops across all features = distributed / superposed.

The clustering approach (epoch-profile correlation graph on top ANOVA-discriminative features) is adapted from Engels et al. for our smaller dataset.

---

## Experiments Run & Results

### Phase 1 — Feature Clustering & Irreducibility at Layer 4

Extracted activations for historical year prompts (BC/AD pairs, extra events) and battle prompts. Encoded through SAE, clustered discriminative features by Pearson correlation of epoch-mean profiles. Scored clusters on separability S(f) and ε-mixture M_eps (fraction of variance in PC1 — low = spread across multiple dims).

Key ablation result: no single feature is strongly load-bearing for the PC1–year correlation. Drops are mostly small (<0.02), consistent with distributed encoding.

**Mind change:** Initially expected a few dominant features. The ablation histogram showed the temporal axis is distributed across many features, pointing toward superposition rather than a single clean circuit.

**Outputs:** `phase1_all_pca.png`, `phase1_year_linearity.png`, `phase1_cluster*.png`, `phase1_ablation_histogram.png`

---

### Phase 2 — Geometric Structure of the Year Axis

Extracted year activations at layers 4, 6, 8, 10. Layer 4 had the strongest PC1–year correlation. Detailed analysis at layer 4: raw PCA shows a strong linear axis (PC1 ≈ year), and PC2/PC3 shows a circular component with BC and AD items occupying opposite arcs. Detrended PCA (PC1 removed) exposes this residual circularity. The BC–AD arc separation is well above 120°, consistent with the model treating BC and AD as directionally distinct despite both being "ancient history."

**Mind change:** The linear-only picture from Phase 1 is incomplete — there is genuine multi-dimensional structure in the year representation.

**Outputs:** `phase2_geometric_layer4.png`

---

### Phase 3 — Superposition Depth Scan (Original Prompts)

Ran the full cluster ablation scan independently at every layer (0–11). Max single-feature drop and distribution shape tracked across depth. The signal remains distributed at most layers; no layer shows a single dominant feature, confirming superposition.

**Outputs:** `phase3_superposition_depth.png`, `phase3_drop_distributions.png`, `phase3_layer_summary.csv`

---

### Phase 4 — Clean Prompts Depth Scan

Repeated Phase 3 with prompts that avoid BC/AD tokens (to test whether Phase 3 results were driven by orthographic features like the literal string "BC"). Results are qualitatively similar — the temporal axis persists without BC/AD tokens, and encoding remains distributed. Confirmed that the structure is semantic, not orthographic.

**Outputs:** `phase4_clean_superposition_depth.png`, `phase4_clean_year_linearity.png`, `phase4_layer_summary.csv`, `phase3_vs_phase4_comparison.csv`

---

### Phase 5 — Circular Concepts Depth Scan (Days of Week & Months)

Replicated Engels et al.'s circular geometry for days of the week and months of the year, then ran circular superposition ablation scans across all 12 layers for both concepts.

Results at layer 7:
- Days of week: `angular_r = 0.662`
- Months of year: `angular_r = 0.748`

Both concepts show clear circular structure in PC2/PC3. The circular signal is strongest around layers 6–8.

**Outputs:** `phase5_days_geometry_layer7.png`, `phase5_months_geometry_layer7.png`, `phase5_circular_superposition_depth.png`, `phase5_all_concepts_comparison.png`, `phase5_days_layer_summary.csv`, `phase5_months_layer_summary.csv`

---

### Phase 6 — Days of the Month (Geometry Only, Novel Concept)

First study of whether GPT-2 encodes days-of-the-month (1–31) as a circle. 93 prompts across 3 linguistic frames (simple statement, wraparound sequential, positional). Purely geometric — no ablation.

Results across layers 4, 6, 7, 8, 10:

| Layer | angular_r | RMSE   | Structure       |
|-------|-----------|--------|-----------------|
| 4     | **0.495** | 2.651  | moderate circle |
| 6     | 0.424     | 2.792  | moderate circle |
| 7     | 0.379     | 3.481  | weak circle     |
| 8     | 0.403     | 4.114  | moderate circle |
| 10    | 0.419     | 6.530  | moderate circle |

Comparison at their respective best layers:
- Days of week (L7): `angular_r = 0.662`
- Months of year (L7): `angular_r = 0.748`
- **Days of month (L4): `angular_r = 0.495`**

Days of the month show *moderate* circular geometry — weaker than the 7-day and 12-month cycles but well above noise. The cluster quality check flagged high within-day variance (intra/inter ratio = 1.42), suggesting the 3 prompt frames pull the same-day representations apart more than for simpler cyclic concepts. This is expected: "1st" and "first" and "day after 31st" activate different surface-level features.

**Mind change:** A weaker circle for 31 days was expected (less redundancy, more noise), but the moderate signal is still a positive finding — the model does appear to represent day-of-month position geometrically, not just as an ordinal lookup.

**Outputs:** `phase6_days_of_month_geometry.png`, `phase6_days_of_month_best_layer.png`, `phase6_cyclic_concepts_comparison.png`, `phase6_days_of_month_geometry.csv`

---

## Remaining Experiments

1. **Phase 6 superposition scan** — run the circular ablation scan for days-of-month across all 12 layers, matching the Phase 5 methodology. This will reveal whether the circular signal is carried by a small number of features (concentrated) or is superposed.

2. **Prompt robustness for days of month** — the high frame variance (ratio 1.42) suggests frame 3 ("The 15th is the fifteenth day") may be degrading the signal. Running geometry on frame 1 alone ("Today is the Nth of the month") would test whether cleaner prompts yield a stronger circle.

3. **Cross-concept superposition** — do the same SAE features that carry day-of-week circularity also contribute to day-of-month or month-of-year circularity? A feature overlap analysis would test whether temporal cycles share a representational substrate.

4. **Helical structure** — historical years show both a linear axis (PC1 ≈ year) and a circular component (PC2/PC3). A full 3-D PCA visualization could confirm a helix, which would be the clearest evidence of a single unified temporal representation.

---

## How Results Will Inform the Conclusion

- If the days-of-month ablation scan (remaining experiment 1) shows concentrated circular signal (large max-drop at a single feature), it would suggest the 31-day cycle is represented more sparsely than the 7-day or 12-month cycles — perhaps because it's less culturally salient. If distributed, it matches the pattern seen for years.
- If single-frame prompts yield `angular_r > 0.6` for days of month, the moderate result in Phase 6 was a prompt-noise artifact, not a weaker representational geometry.
- If cross-concept feature overlap is high, it supports a unified "temporal position" circuit. If low, each cyclic concept has its own dedicated features.
- The helix test will be the cleanest statement about whether years are encoded in a fundamentally higher-dimensional structure than a simple linear axis.

---

## Roadblocks

- **SAE loading per layer is slow on CPU** — each phase that scans all 12 layers (phases 3–5) is I/O-heavy. Phase 6 was scoped to 5 layers for this reason.
- **High frame variance for days of month** — 31 distinct days × 3 frames creates more noise than 7 days × 3 frames. The intra/inter ratio of 1.42 means that within-day spread is larger than between-day spread in the first 3 PCA dimensions, which inflates RMSE and depresses `angular_r`. This is a data quality concern, not a modeling one.
- **No ground-truth circular signal strength** — `angular_r` is sensitive to prompt framing. Without a frame-invariant baseline, it is hard to say definitively whether days-of-month have a "weaker" circle than months-of-year or just noisier prompts.

---

## Codebase Map

```
gpt2-temporal-probing/
│
├── run.py                      Entry point. --phase {1-6, all}
├── config.py                   Global constants: DEVICE, layers, thresholds, SAE_RELEASE
│
├── data/
│   └── prompts.json            All prompt datasets:
│                                 bc_ad_pairs, extra_year_items     — historical years
│                                 battle_europe/america/east        — named battles
│                                 clean_year_items                  — no BC/AD tokens
│                                 boundary_probes                   — year boundary tests
│                                 geographic_cities                 — spatial control
│                                 numeric_magnitude                 — numeric control
│                                 days_of_week + _labels            — 7-day cycle (Engels replication)
│                                 months_of_year + _labels          — 12-month cycle (Engels replication)
│                                 days_of_month + _labels           — 31-day cycle (novel, Phase 6)
│
├── src/
│   ├── data_loading.py         Parses prompts.json into typed tuples;
│                               defines YEAR_ITEMS, BATTLE_ITEMS, ALL_ITEMS, YEAR_CATEGORIES
│   ├── model.py                load_model() / load_sae() / extract_activations() / get_feature_acts()
│   ├── clustering.py           find_clusters(): ANOVA-select + epoch-profile correlation graph
│   ├── metrics.py              separability_index(), epsilon_mixture_index(),
│                               fit_and_score_circle(), cluster_quality_check(), circularity_score()
│   ├── geometry.py             fit_circle_algebraic() (Coope 1993), _extract_layer_acts()
│   └── plots.py                plot_pca_overview(), plot_year_linearity(), _plot_geometric_analysis()
│
├── phases/
│   ├── phase1.py               Feature clustering + irreducibility at layer 4
│   ├── phase2.py               Geometric analysis of year axis across layers
│   ├── phase3.py               Superposition depth scan, original prompts (all 12 layers)
│   ├── phase4.py               Superposition depth scan, clean prompts (no BC/AD tokens)
│   ├── phase5.py               Circular geometry + superposition scan for days-of-week & months
│   └── phase6.py               Circular geometry for days-of-month (novel concept, 5 layers)
│
└── outputs/                    All PNGs and CSVs written here
```

**Key entry points for reading the code:**
- `src/metrics.py:69` — `fit_and_score_circle()` is the core geometric measurement used in phases 5–6
- `src/clustering.py:8` — `find_clusters()` is the core feature-selection step used in phases 1–4
- `phases/phase5.py:34` — `_circular_ablation_scan_at_layer()` is the superposition measurement for cyclic concepts
- `phases/phase6.py:100` — `run_phase6()` is the novel days-of-month analysis

---

## Installation & Usage

```bash
uv venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt

python run.py --phase 1   # feature clustering
python run.py --phase 2   # geometric analysis (years)
python run.py --phase 3   # superposition depth scan
python run.py --phase 4   # clean prompts scan
python run.py --phase 5   # circular concepts (days of week, months)
python run.py --phase 6   # days of month (novel)
python run.py --phase all # run everything
```

All outputs are written to `outputs/`.

---

## References

- Engels, J. et al. (2024). *Not All Language Model Features Are Linear.* NeurIPS.
- Gurnee, W. & Tegmark, M. (2023). *Language Models Represent Space and Time.* ICLR.
- Elhage, N. et al. (2022). *Toy Models of Superposition.* Anthropic.
- Coope, I. D. (1993). *Circle fitting by linear and nonlinear least squares.* J. Optim. Theory Appl.
