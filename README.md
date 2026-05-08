# GPT-2 Temporal Probing

Mechanistic interpretability project investigating how GPT-2 Small encodes structured temporal concepts in its residual stream using Sparse Autoencoders (SAEs).

---

## Writeup

- **Compiled PDF**: `paper/main.pdf`
- **LaTeX source**: `paper/main.tex`
- **Bibliography**: `paper/references.bib`
- **Figures**: all figures are in `outputs/` (organised by phase) and compiled into the PDF automatically via `\graphicspath`

---

## Source Code

All source code is in the root directory and subdirectories:

- `run.py` — entry point; run any phase with `python run.py --phase <1-6>`
- `config.py` — shared configuration (model, layers, hyperparameters)
- `phases/` — one file per analysis phase (`phase1.py` through `phase6.py`)
- `src/` — shared utilities: `model.py`, `metrics.py`, `plots.py`, `geometry.py`, `data_loading.py`

---

## Data

- `data/prompts.json` — all prompts used across phases (historical year corpus, cyclic concept datasets)

No large external datasets are used. GPT-2 Small weights and SAEs are downloaded automatically via `transformer_lens` and `sae_lens` on first run.

---

## Experimental Results

All outputs are in `outputs/`, organised by phase:

| Folder | Contents |
|--------|----------|
| `outputs/phase1/` | PCA overview and year linearity plots |
| `outputs/phase2/` | PC2/PC3 geometry and angular correlation plots |
| `outputs/phase3/` | BC/AD corpus superposition depth scan |
| `outputs/phase4/` | Clean corpus superposition depth scan |
| `outputs/phase5/` | Cyclic concept circular superposition depth scan |
| `outputs/phase6/` | Days-of-month geometry analysis |

Each phase also saves a CSV summary of numerical results alongside the figures.

---

## Repository Structure

```
gpt2-temporal-probing/
├── run.py                        # Entry point — dispatches to phase runners
├── config.py                     # Shared configuration (model, layers, hyperparameters)
├── pyproject.toml                # uv/pip dependency specification
├── requirements.txt              # Fallback dependency list
│
├── phases/                       # One file per analysis phase
│   ├── phase1.py                 # PCA overview and linear decodability
│   ├── phase2.py                 # Geometric structure (PC2/PC3 circle fit)
│   ├── phase3.py                 # Superposition depth scan (BC/AD corpus)
│   ├── phase4.py                 # Superposition depth scan (clean corpus)
│   ├── phase5.py                 # Circular superposition scan (days/months)
│   └── phase6.py                 # Days-of-month geometry analysis
│
├── src/                          # Shared utilities
│   ├── data_loading.py           # Parses prompts.json into typed tuples
│   ├── model.py                  # GPT-2 loading and activation extraction
│   ├── metrics.py                # Circle fitting and cluster quality scoring
│   ├── geometry.py               # Algebraic circle fit (Coope 1993)
│   └── plots.py                  # All matplotlib output functions
│
├── data/
│   └── prompts.json              # All prompts (historical year corpus + cyclic datasets)
│
├── outputs/                      # All experimental results (figures + CSVs)
│   ├── phase1/
│   ├── phase2/
│   ├── phase3/
│   ├── phase4/
│   ├── phase5/
│   └── phase6/
│
└── paper/                        # Writeup
    ├── main.tex                  # LaTeX source
    ├── main.pdf                  # Compiled PDF
    └── references.bib            # Bibliography
```

---

## Reproducing Results

Install dependencies with `uv`:

```bash
uv sync
```

Run all phases:

```bash
uv run python run.py
```

Run a specific phase:

```bash
uv run python run.py --phase 3
```
