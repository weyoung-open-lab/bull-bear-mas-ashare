# Trained Models — Retraining Instructions

CatBoost binary model files (`*.cbm`) are excluded from this repository (each
file is 5–20 MB and they total ~100 MB across the Walk-Forward suite). To
reproduce the headline results you need to retrain them locally:

## 1. Place the dataset

Set the dataset path in `config.py` so `src.data.load_dataset()` returns the
A-share panel with all 17 G1-G4 features and the `r_future_5` column. The
column schema is documented in `src/features.py`.

## 2. Train Alpha + 3 strategy agents (legacy + reused as Alpha)

```bash
python -m strategy_debate.experiments.run_phase1_train_agents
```

This produces:
- `strategy_debate/results/models/trend_agent.cbm`   (Alpha Agent for paper)
- `strategy_debate/results/models/momentum_agent.cbm`
- `strategy_debate/results/models/strength_agent.cbm`
- corresponding `*_medians.csv` files

## 3. Train Bear Agent D1 (main paper system)

```bash
python -m bull_bear.experiments.step1_train_bear      # Bear V1 (G4 features)
python -m bull_bear.experiments.step3_parallel        # D1 / D2 / D3 — D1 wins
```

Output: `bull_bear/results/models/bear_D1_agent.cbm` + medians.

## 4. Train Walk-Forward agents (W1-W4)

```bash
python -m bull_bear.experiments.step6_final_validation
```

Trains Alpha + Bear D1 on three additional rolling-windows (2016-2018,
2016-2019, 2016-2020). Output: `bull_bear/results/models/walkforward/*.cbm`.

W5 (the main 2016-2021 train window) reuses the agents from steps 2-3.

## 5. Estimated training time

| Step | Files produced | Time on modern desktop CPU |
|---|---|---|
| 2 (3 strategy agents) | 3 × 5 MB | ~30 s |
| 3 (Bear D1) | 1 × 8 MB | ~30 s |
| 4 (Walk-Forward W1-W4) | 6 × 8 MB | ~3 min |
| **Total** | ~70 MB models | **~5 min** |

All scripts are deterministic given `random_seed=42` in
`bull_bear/config_bb.py::CATBOOST_PARAMS`.
