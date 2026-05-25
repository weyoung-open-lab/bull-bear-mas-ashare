# Why CatBoost — Model Selection Evidence Chain

> Extracted from prior model-comparison experiments
> (see `results/main_compare_*` and `results/FINAL_SUMMARY.md`).

## TL;DR

CatBoost-reg was chosen as the **Alpha Agent / Bear Agent backbone** on the
basis of three independent experiments run before the adversarial framework
was designed:

1. **Cross-family comparison (14 models)** — CatBoost-reg achieves the
   **highest RankICIR (0.376)** among all 14 model families on the
   2023-2026 test panel.
2. **Loss-function comparison (BCE vs MSE)** — CatBoost gains the **largest
   absolute lift** when switching from binary BCE to regression MSE
   (+0.320 RankICIR, the largest single jump in the entire benchmark).
3. **Regime-divergence signal** — CatBoost-reg + G1+G2+G3+G4 features
   produces **SRD(bear, bull) = 0.694**, the strongest regime-explanation
   signal across all configurations and inside the ESWA-prespecified
   $[0.3, 0.7]$ range.

The decision was not driven by Top-5\% portfolio Sharpe (where TabNet-reg
scored higher at 1.42), because the paper's primary target metric is
**RankICIR** — cross-sectional ranking quality of the whole universe, not
the thin tail of a low-turnover portfolio.

---

## Experiment 1 — 14-Model Cross-Family Comparison

**Setup**: 11 ML models (linear, GBDT, tabular DL, sequence DL) + 3 factor
baselines, all trained on the full 7.17 M-row panel (2016-2021 train,
2023-2026 test, 733 trading days), all with the regression MSE objective
on `r_future_5`. Results sorted by RankICIR (the primary cross-sectional
ranking metric).

| Rank | Model | Family | RankICIR | Top-5\% Sharpe | Fit+Predict (sec) |
|---:|---|---|---:|---:|---:|
| 1 | **CatBoost-reg** | **gbdt** | **0.3763** | 0.66 | **41.97** |
| 2 | LightGBM-shallow-reg | gbdt | 0.3296 | 0.67 | 43.84 |
| 3 | LightGBM-conservative-reg | gbdt | 0.3251 | 0.43 | 85.42 |
| 4 | LightGBM-std-reg | gbdt | 0.3173 | 0.47 | 56.02 |
| 5 | XGBoost-reg | gbdt | 0.3025 | 0.70 | 27.40 |
| 6 | Ridge | linear | 0.3001 | 0.84 | 6.43 |
| 7 | RandomForest-reg | gbdt | 0.2952 | 0.68 | 105.02 |
| 8 | TCN-reg | sequence | 0.2018 | 0.93 | 58.21 |
| 9 | FT-Transformer-reg | tabular_dl | 0.1942 | 0.53 | 636.64 |
| 10 | ALSTM-reg | sequence | 0.1775 | 0.51 | 50.18 |
| 11 | Momentum-5d | factor | -0.2503 | -2.33 | 0.04 |
| 12 | TabNet-reg | tabular_dl | -0.2987 | **1.42** | 1370.70 |
| 13 | EMA-slope | factor | -0.4209 | -2.27 | 0.03 |
| 14 | Rel-Strength | factor | -0.4322 | -1.96 | 0.04 |

**Key observations:**

- **CatBoost-reg leads RankICIR by +14% relative** over the second-place
  model (LightGBM-shallow-reg 0.330) — a meaningful gap, not a tie.
- **All five GBDT models cluster in the top 7 by RankICIR**; GBDT as a
  family is the clear winner over linear, tabular DL, and sequence DL on
  this panel.
- **TabNet-reg's RankICIR is negative (-0.30) despite a 1.42 Top-5\%
  Sharpe**. It picks a thin tail extremely well but mis-ranks the broad
  cross-section, making it unsuitable as the "ranker" backbone that the
  adversarial system requires.
- **CatBoost-reg's inference time (41.97 sec for 7.17M-row panel) is
  competitive** with LightGBM and dramatically faster than tabular DL
  (TabNet 1370 s, FT-Transformer 637 s).

Source: `results/main_compare_20260506_225947_full_reg/metrics_summary.csv`

---

## Experiment 2 — Binary BCE vs Regression MSE Objective

**Setup**: same 11 ML models, trained both with binary cross-entropy
(label = $\mathbb{1}\{r^{(5d)} > 1\%\}$) and with regression MSE on
$r^{(5d)}$. The objective comparison is done within each model family so
the gain can be attributed to the loss function, not the architecture.

| Model | RankICIR (BCE) | RankICIR (MSE) | **Δ RankICIR** |
|---|---:|---:|---:|
| Linear (Ridge) | 0.241 | 0.300 | +0.059 |
| LGBM-std | 0.060 | 0.317 | +0.257 |
| LGBM-shallow | 0.043 | 0.330 | +0.287 |
| LGBM-cons | 0.040 | 0.325 | +0.285 |
| XGBoost | 0.061 | 0.302 | +0.241 |
| **CatBoost** | **0.057** | **0.376** | **+0.320** ★ |
| RandomForest | -0.012 | 0.295 | +0.307 |
| TabNet | -0.037 | -0.299 | -0.261 |
| FT-Transformer | -0.029 | 0.194 | +0.223 |
| ALSTM | -0.007 | 0.177 | +0.184 |
| TCN | -0.001 | 0.202 | +0.203 |

**Key observations:**

- **CatBoost gains the largest absolute lift (+0.320)** when switching from
  BCE to MSE — larger than any other model family.
- Switching the loss function turns CatBoost from "below average" (0.057,
  rank 6 among BCE models) to "field leader" (0.376, rank 1 among MSE
  models).
- This suggests CatBoost's regularised tree growth is particularly
  well-suited to the smooth regression target $r^{(5d)}$ relative to the
  step-function BCE target.

Source: `results/binary_vs_regression.csv`

---

## Experiment 3 — SRD Regime-Divergence Signal

**Setup**: train CatBoost-reg and LightGBM-shallow-reg on G1+G2+G3+G4
(21 features, no macro-regime features). Compute regime-conditioned SHAP
importance for the three regimes (bear / sideway / bull) and the pairwise
SHAP Regime Divergence
$\text{SRD}(r_1, r_2) = 1 - \rho_{\text{Spearman}}(\text{rank}_{r_1}, \text{rank}_{r_2})$.

| SRD pair | LightGBM-shallow-reg + G1234 | **CatBoost-reg + G1234** |
|---|---:|---:|
| bear ↔ bull | 0.291 | **0.694** ★ |
| bear ↔ sideway | — | 0.488 |
| bull ↔ sideway | — | 0.543 |

**Key observation**: CatBoost-reg + G1234 produces an SRD(bear, bull) of
**0.694**, more than **double** what LightGBM-shallow-reg produces on the
identical feature set (0.291). This is the strongest regime-explanation
signal across all configurations tested, and lands at the upper bound of
the pre-specified $[0.3, 0.7]$ range.

For the Bear Agent — whose explicit purpose is to expose a different
selection logic from the Alpha Agent — using a model family that
**amplifies** rather than smooths over regime structure is the natural
choice.

Source: `results/regime_20260507_013443_final_g1234_cat/srd_matrix.csv`,
`results/FINAL_SUMMARY.md` Tables 5–6.

---

## Why Not the Alternatives

| Candidate | Best metric | Why rejected |
|---|---|---|
| **LightGBM-shallow-reg** | RankICIR 0.330 | Second-place RankICIR (−4.7 pp vs CatBoost); SRD signal only 0.291, so the Bear Agent's regime-divergence story would be weak. |
| **TabNet-reg** | Sharpe 1.42 | RankICIR -0.30 — fails the primary cross-sectional ranking objective. Also 33× slower inference (1370 s vs 42 s). |
| **TCN-reg / ALSTM-reg** | Sharpe 0.93 / 0.51 | RankICIR < 0.21 — sequence DL fails to leverage the cross-sectional structure of A-share data on panel sizes this large. |
| **Ridge** | RankICIR 0.300 | Linear ceiling, no native feature-interaction modelling, no regime sensitivity. |
| **Factor baselines (Momentum, EMA, Rel-Strength)** | -- | All produce **negative** RankICIR in the 2023-2026 test period (-0.25 to -0.43), confirming the post-2023 A-share regime is reversal-heavy and trend baselines have collapsed. |
| **FT-Transformer-reg** | RankICIR 0.194 | Weaker RankICIR than every GBDT; 637 s inference is 15× slower than CatBoost; no benefit. |
| **RandomForest-reg** | RankICIR 0.295 | Weaker RankICIR than every boosting variant; 2.5× slower (105 s) than CatBoost. |

---

## Operational Properties of CatBoost That Made It the Bull-Bear Backbone

Beyond the headline metric, three engineering properties influenced the
choice:

1. **Native handling of missing values & categorical columns** —
   `所属行业` (industry, 28 categories) and `macro_regime_3` are encoded
   natively without manual one-hot expansion.
2. **Deterministic training under fixed seed** —
   `random_seed=42` produces bit-identical models across re-runs on the
   same hardware. This was essential for the walk-forward, bootstrap, and
   $X$ vs $Y$ mechanism experiments where every result must be exactly
   reproducible.
3. **TreeSHAP exact attribution** — CatBoost natively supports exact
   TreeSHAP attribution in `get_feature_importance(type="ShapValues")`,
   which underlies the Bear quintile independence verification and the
   per-regime SHAP analysis in the legacy strategy-debate suite.

Together with the RankICIR lead and the strongest SRD signal, these
properties make CatBoost-reg the natural backbone for both the Alpha
Agent and the Bear Agent of the adversarial system.

---

## Citation Within the Paper

The Bull-Bear paper Section 2.2 cites CatBoost via
$\text{prokhorenkova2018catboost}$ (NeurIPS 2018) as the de-facto baseline
for cross-sectional ranking. Sections 4.3 and 4.4 specify the same
hyper-parameters (depth 6, learning rate 0.05, 300 trees, RMSE loss,
random seed 42) for both Alpha and Bear agents, so that any observed
performance difference can be attributed to the **training target**, not
the model architecture.
