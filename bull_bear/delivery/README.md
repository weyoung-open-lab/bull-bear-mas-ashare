# BBAQ-MAS: Bull-Bear Adversarial Quantitative Multi-Agent System

> A multi-agent stock-selection framework for the Chinese A-share market, combining a Trend α-Agent, an error-informed adversarial Bear-Agent, a Reversal-Agent, and a regime-aware Circuit-Breaker Agent.

## Final System Performance

| Metric | Value |
|---|---|
| RankICIR (test 2023-01-03 → 2026-01-30) | **0.8094** (test-ref) / 0.8011 (val-honest) |
| Top-5% Sharpe (with Agent 4 / F3-v2) | **+1.894** |
| MaxDD (with Agent 4 / F3-v2) | **-27.27%** |
| AnnRet (with Agent 4 / F3-v2) | **+21.26%** |
| Walk-forward win-rate vs Trend (7 yrs) | **7/7** |
| Walk-forward win-rate vs D1c on RIC | **7/7** |
| Walk-forward win-rate vs D2f on Sharpe | **7/7** (D1c track) |
| Bootstrap p-value (D2f − Trend, n=1000) | **p < 0.001** |
| Bootstrap p-value (D2f − D1c on RIC)    | **p < 0.001** |
| Bootstrap p-value (D1c − D2f on Sharpe) | **p < 0.001** |

## System Architecture

- **α-Agent (Alpha)** — CatBoost-regressor, G1-G4 features (21 dim), target = forward 5-day cross-sectional return.
- **Bear-Agent** — CatBoost-regressor, G1+G3 features, target = forward 5-day max-drawdown (`max_drawdown_5d`). Error-Informed training applies AdaBoost-style sample re-weighting `w_i = 1 + λ · |alpha_rank_i − actual_rank_i|` with **λ = 3.0** (optimal across both fixed- and adaptive-α tracks).
- **Reversal-Agent** — CatBoost-regressor, short-term reversal features (`rev_mkt_excess_1d`, `ret_1d`, `rev_zscore_1d`, `rev_ret_3d_minus_1d`, `rev_ret_2d`), target = forward 5-day return. Near-orthogonal to α and Bear (ρ < 0.14).
- **Regime-Agent** — CatBoost-classifier on macro/market features. Outputs P_bear, P_sideway, P_bull which control α(t) and γ(t).
- **Circuit-Breaker (Agent 4 / F3-v2)** — Portfolio-level exposure modulator. Cuts gross exposure to **0.5** when market 1-day return < −3% (crash) and to **0.7** under bear-regime; otherwise 1.0. Triggers on 262/733 (35.7%) of test days.

## Conviction Formula

```
c(i, t) = s_alpha(i, t) − α(t) · s_bear(i, t) + γ(t) · s_reversal(i, t)

α(t) = 0.50 + 0.15·1[bear] − 0.15·1[bull]     ∈ [0.35, 0.65]
γ(t) = 0.40 + 0.15·1[bull] − 0.15·1[bear]     ∈ [0.25, 0.55]
```

Top-5% by `c(i, t)`, equal-weighted within the top bucket, held for 5 trading days, 0.30% round-trip transaction cost.

## Directory Layout

```
delivery/
├── paper/                              ← LaTeX manuscript + figures
│   ├── bull_bear_paper.pdf             ← compiled paper
│   ├── data_bundle.md                  ← per-section data appendix
│   ├── catboost_selection_rationale.md
│   ├── model_selection_report.md
│   ├── figures/                        ← 17 paper figures (PNG)
│   └── sections/                       ← bull_bear_paper.tex + cls/bst + ELS templates
│
├── models/                             ← trained CatBoost binaries
│   ├── alpha_agent.cbm
│   ├── bear_D2_lambda3.cbm             ← canonical Bear (λ=3.0)
│   ├── reversal_agent.cbm
│   ├── regime_agent/                   ← regime classifier + SRD config
│   ├── bear_D2_lambda_sweep/           ← λ ∈ {0.5, 1, 2, 3, 5, 10} for §5.7 reproduction
│   └── walkforward/                    ← α/Bear/Reversal models for W1-W5 windows
│
├── results/                            ← CSV outputs grouped by experiment role
│   ├── ablation/                       ← B0 → D2f+A4 evolution (§5.5–§5.7, §5.12)
│   ├── validation/                     ← walk-forward, bootstrap, quintile (§5.9–§5.11)
│   ├── robustness/                     ← Step-17 parameter sensitivity (§5.13)
│   └── model_selection/                ← 14-model comparison, feature ablation, SRD (§5.2–§5.4)
│
├── src/                                ← reproducibility code
│   ├── features/                       ← G1-G5 feature pipeline + Bear targets
│   ├── agents/                         ← α / Bear-D1 / Bear-D2 / Reversal / Regime / Wave training
│   ├── backtest/                       ← RankICIR, Sharpe, MaxDD, walk-forward, Agent-4
│   ├── pipeline.py                     ← end-to-end ablation runner
│   └── pipeline_validation.py          ← walk-forward / bootstrap orchestrator
│
├── requirements.txt
└── README.md                           ← this file
```

## Key Experiment Results — Quick Index

| Paper section | What it shows | Files |
|---|---|---|
| §5.1 | Setup (733 days, 3,876 stocks, 0.30% cost) | — |
| §5.2 | 14-model comparison, BCE vs MSE; CatBoost-reg wins | `results/model_selection/main_compare_*`, `binary_vs_regression.csv` |
| §5.3 | Feature-group ablation; G1–G4 (21 dim) optimal | `results/model_selection/feature_ablation.csv` |
| §5.4 | SHAP Regime Divergence; CatBoost SRD(bear, bull) = 0.6935 | `results/model_selection/srd_matrix_*.csv` |
| §5.5 | Main ablation B0 → D2f+A4 (10 rows) | `results/ablation/final_ablation.csv`, `step8_d2_peak.csv`, `step13_gamma_adaptive.csv`, `step16_agent4_practical.csv` |
| §5.6 | Y(T−α·D1) beats X(T+α·M1) by +1,777 bp at α=0.5 | `results/ablation/mechanism_validation.csv` |
| §5.7 | λ peak at 3.0 (RankICIR 0.7505) | `results/ablation/step8_d2_peak.csv` |
| §5.8 | Reversal standalone 0.3215; γ peak 0.40 | `results/validation/step11_reversal_standalone.csv`, `step12_gamma_peak.csv` |
| §5.9 | Walk-forward 7-year (Trend / D1c / D2c / D2f) | `results/validation/step15_walkforward_d1c_vs_d2f.csv`, `rolling_walkforward.csv` |
| §5.10 | Bootstrap n=1000, all comparisons p < 0.001 | `results/validation/bootstrap_test.csv`, `step12_bootstrap_d2f.csv`, `step15_bootstrap_d1c_vs_d2f.csv` |
| §5.11 | Bear quintile monotone MaxDD (Q5−Q1 = +2.37 pp) | `results/validation/bear_quintile_analysis.csv` + `results/ablation/simple_baseline_comparison.csv` |
| §5.12 | Agent 4 / F3-v2: Sharpe +1.894, MaxDD −27.27% | `results/ablation/step16_agent4_practical.csv` |
| §5.13 | Parameter sensitivity (max \|Δ\| = 76 bp on λ) | `results/robustness/step17_*.csv` |

The full per-section data appendix lives in [`paper/data_bundle.md`](paper/data_bundle.md).

## Reproduction Hints

1. **Environment** — `pip install -r requirements.txt` (CatBoost, LightGBM, XGBoost, pandas, numpy, scikit-learn, matplotlib).
2. **Single-shot ablation** — `python src/pipeline.py` walks B0 → D1 → D1c → D2c → D2f and emits the canonical RankICIR / Sharpe / MaxDD per row.
3. **Walk-forward / bootstrap** — `python src/pipeline_validation.py` re-trains W1–W5 and produces `step15_walkforward_d1c_vs_d2f.csv` + `step15_bootstrap_d1c_vs_d2f.csv`.
4. **Robustness** — `python src/backtest/step17_robustness.py` reproduces validation-best vs canonical config and the 7-point sensitivity neighbourhood.
5. **Agent 4** — `python src/backtest/agent4_circuit_breaker.py` re-runs the F1 / F2 / F3 grid on D2f cross-section predictions.

The `models/` directory contains every CatBoost binary needed to re-evaluate the canonical configuration without re-training. Re-training from scratch takes ~3.5 minutes per agent on a single CPU.

## Target Journal

**Expert Systems with Applications** (ESWA, SCI Q1, JCR Impact Factor 7.5).
