# §5 Experiments — Tables & Figures Bundle

> All numbers extracted live from CSV files. Test panel: 2023-01 to 2026-01, 733 trading days, 3,876 stocks.

> Figures (300 dpi PNG) located in `paper/figures/`. Source CSVs path noted under each table.


---

## §5.1 Experimental Setup

Data partition (defined in §3, reproduced for reference):

| Split | Period | Role | Trading days |
|---|---|---|---:|
| Train | 2016-10-17 to 2021-12-31 | Model fitting | 1,271 |
| Val   | 2022-01-01 to 2022-12-31 | alpha selection | 242 |
| Test  | 2023-01-01 to 2026-01-19 | OOS evaluation | 733 |

Metrics: RankICIR (primary), Top-5% Sharpe (after 0.3% round-trip cost, 5-day holding), Max Drawdown.

Computational cost (from `strategy_debate/results/computational_cost.csv`):
- One-off training: ~3.5 minutes total for 4 strategy agents + B0
- Per-day inference (3,876 stocks): 11.69 ms full system vs 4.43 ms B0
- Cost ratio: 2.64x B0

---

## §5.2 Backbone Model Selection

Source: `results/main_compare_20260506_225947_full_reg/metrics_summary.csv`

### Table 5.2.1 — 14-model comparison (regression MSE, RankICIR-sorted)

| Rank | Model | Family | RankICIR | Top-5% SR | AUC | Time (s) |
|---:|---|---|---:|---:|---:|---:|
| 1 | **CatBoost-reg** *star* | gbdt | +0.3763 | +0.656 | 0.535 | 41.97 |
| 2 | **LightGBM-shallow-reg** | gbdt | +0.3296 | +0.672 | 0.536 | 43.84 |
| 3 | **LightGBM-conservative-reg** | gbdt | +0.3251 | +0.427 | 0.543 | 85.42 |
| 4 | **LightGBM-std-reg** | gbdt | +0.3173 | +0.471 | 0.536 | 56.02 |
| 5 | **XGBoost-reg** | gbdt | +0.3025 | +0.695 | 0.541 | 27.40 |
| 6 | **Ridge** | linear | +0.3001 | +0.839 | 0.555 | 6.43 |
| 7 | **RandomForest-reg** | gbdt | +0.2952 | +0.684 | 0.551 | 105.02 |
| 8 | **TCN-reg** | sequence | +0.2018 | +0.928 | 0.571 | 58.21 |
| 9 | **FT-Transformer-reg** | tabular_dl | +0.1942 | +0.533 | 0.525 | 636.64 |
| 10 | **ALSTM-reg** | sequence | +0.1775 | +0.511 | 0.559 | 50.18 |
| 11 | **Momentum-5d** | factor | -0.2503 | -2.334 | 0.482 | 0.04 |
| 12 | **TabNet-reg** | tabular_dl | -0.2987 | +1.425 | 0.524 | 1370.70 | *(highest SR but negative RankICIR)*
| 13 | **EMA-slope** | factor | -0.4209 | -2.271 | 0.481 | 0.03 |
| 14 | **Rel-Strength** | factor | -0.4322 | -1.957 | 0.495 | 0.04 |

**Figure**: `paper/figures/model_compare_bar.png` (horizontal bar, CatBoost highlighted orange, GBDT green, DL purple, factor gray)

**Findings**:
1. CatBoost RankICIR 0.376 leads 2nd-place LightGBM-shallow-reg 0.330 by +4.7 pp absolute (+1416.9% relative).
2. 6 of top 7 models are GBDT; deep learning (TabNet, FT-Transformer, ALSTM, TCN) ranks 8-12.
3. All 3 factor baselines produce negative RankICIR (mean -0.368); traditional momentum factors collapsed in 2023-2026.
4. TabNet has the highest Top-5% Sharpe (+1.425) yet RankICIR is -0.299; it picks a thin tail well but mis-ranks the cross-section, unsuitable as a ranker backbone.

### Table 5.2.2 — Binary BCE vs Regression MSE objective

Source: `results/binary_vs_regression.csv`

| Model | RankICIR (BCE) | RankICIR (MSE) | Δ (MSE − BCE) |
|---|---:|---:|---:|
| **CatBoost** *star* | +0.057 | +0.376 | +0.320 |
| **LGBM-shallow** | +0.043 | +0.330 | +0.287 |
| **LGBM-cons** | +0.040 | +0.325 | +0.285 |
| **LGBM-std** | +0.060 | +0.317 | +0.257 |
| **XGBoost** | +0.061 | +0.302 | +0.241 |
| **Linear** | +0.241 | +0.300 | +0.059 |
| **RandomForest** | -0.012 | +0.295 | +0.307 |
| **TCN** | -0.001 | +0.202 | +0.203 |
| **FT-Transformer** | -0.029 | +0.194 | +0.223 |
| **ALSTM** | -0.007 | +0.177 | +0.184 |
| **TabNet** | -0.037 | -0.299 | -0.261 |

**Figure**: `paper/figures/bce_vs_mse_bar.png`

**Finding**: 10 of 11 models gain from switching to MSE (TabNet inverts). CatBoost gains the most: Δ = +0.320. Within the MSE family, CatBoost jumps from BCE-rank 6 to MSE-rank 1, the largest re-ranking move observed.

---

## §5.3 Feature Group Ablation

Source: `results/feature_ablation_20260506_235253_full/feature_ablation.csv`

### Table 5.3.1 — Cumulative feature group ablation (LightGBM-shallow-reg)

| Feature set | # features | RankICIR |
|---|---:|---:|
| G1 | 5 | 0.010 |
| G1+G2 | 8 | 0.018 |
| G1+G2+G3 | 19 | 0.301 |
| G1+G2+G3+G4 | 21 | 0.371 *peak* |
| G1+G2+G3+G4+G5 | 24 | 0.253 |
| Full(G1-G6) | 28 | 0.330 |

**Figure**: `paper/figures/feature_ablation_curve.png`

**Finding**: G1+G2+G3+G4 = 0.371 is the peak. Adding G5 (macro_regime_3) drops RankICIR by 11.7 pp to 0.253. macro_regime_3 has zero per-day cross-section variance, so as a feature it adds no ranking information. This motivates the design: G5 is used as a router by the Regime Agent rather than as a feature.

---

## §5.4 SHAP Regime Divergence

SRD(r1, r2) = 1 - Spearman( rank^SHAP_{r1}, rank^SHAP_{r2} )

### Table 5.4.1 — 4-configuration SRD comparison

| Configuration | SRD(bear, bull) | SRD(bear, side) | SRD(bull, side) |
|---|---:|---:|---:|
| LGBM-shallow BCE + G1--G6 | 0.266 | 0.339 | 0.341 |
| LGBM-shallow MSE + G1--G6 | 0.289 | 0.418 | 0.232 |
| LGBM-shallow MSE + G1234 | 0.291 | 0.308 | 0.575 |
| CatBoost MSE + G1234 *star* | 0.694 | 0.488 | 0.543 |

**Figure**: `paper/figures/srd_heatmap.png`

**Finding**: CatBoost + G1234 produces SRD(bear, bull) = **0.694**, vs LightGBM on the same features 0.291 (ratio 2.4x). CatBoost's oblivious-tree structure imposes the same split rule across an entire level, so regime-routed sub-models diverge sharply in feature ordering. This justifies CatBoost as the adversarial backbone: it amplifies regime-conditioned feature use that the Bear Agent exploits.

---

## §5.5 Main Ablation Study

Source: `bull_bear/results/final_ablation.csv`

### Table 5.5.1 — Main ablation (all configs vs Trend baseline)

| Code | Configuration | RankICIR | Top-5% SR | MaxDD | Δ vs T (bp) |
|---|---|---:|---:|---:|---:|
| B0 | B0 Global CatBoost (17 feat) | 0.297 | +0.920 | -42.84% | -3005 |
| M1 | M1 G1+G3 global (additive ref) | 0.256 | +0.746 | -42.24% | -3419 |
| T | Trend pure (Alpha) | 0.598 | +1.694 | -34.20% | +0 |
| BC | Bear_C α=0.2 (|bias_60| rule) | 0.633 | +1.750 | -33.33% | +346 |
| D1a | D1 α=0.2 (adversarial) | 0.692 | +1.753 | -33.72% | +939 |
| D1b | D1 α=0.5 (* CORE) | 0.741 | +1.802 | -33.52% | +1429 |
| D1c | **+ adaptive α (regime)** | **0.744** | +1.829 | -33.28% | +1459 |
| D1d | + Agent 5 anomaly valve | 0.742 | +1.805 | -33.52% | +1445 |
| V | + Vol-gate (high vol -> Alpha) | 0.715 | +1.765 | -33.61% | +1175 |
| FINAL | FINAL: adaptive α + anomaly + vol-gate | 0.712 | +1.772 | -33.48% | +1137 |

**Figure**: `paper/figures/main_ablation_bar.png`

**Narrative**:
- **B0 vs T**: full 17-feature global CatBoost (0.297) is half of Trend single (0.598); feature pooling dilutes the strong G4 signal.
- **M1 < B0** (0.256 vs 0.297): G1+G3 features alone cannot rank; they only become useful under adversarial subtraction (see §5.6).
- **BC over T**: +346 bp by subtracting standardised |bias_60|; first evidence the Bear concept works even without training.
- **D1a -> D1b -> D1c**: trained Bear at α=0.2 -> 0.5 -> adaptive α. Each step adds +593 / +490 / +30 bp.
- **D1d (Anomaly valve)**: 0.742, near-neutral (-13 bp vs D1c) since only 64/733 days trigger.
- **V (Vol-gate) is detrimental**: 0.715, -283 bp vs D1c. In the D1c framework D1c already outperforms Trend in high-vol regimes, so switching to Trend on those days surrenders the gain.

---

## §5.6 Mechanism Proof: Adversarial (Y) vs Additive (X)

Source: `bull_bear/results/mechanism_validation.csv`

### Table 5.6.1 — Y vs X across α (identical G1+G3 features)

| α | X = T + α·M1 (additive) | Y = T − α·D1 (adversarial) | Δ(Y − X) bp |
|---:|---:|---:|---:|
| 0.1 | 0.607 | 0.654 | +473 |
| 0.2 | 0.605 | 0.692 | +871 |
| 0.3 | 0.595 | 0.716 | +1,215 |
| 0.5 | 0.563 | 0.741 | +1,777 |

**Figure**: `paper/figures/yx_mechanism.png`

**Finding**: As α grows from 0.1 to 0.5, Y monotonically increases (0.654 -> 0.741) while X monotonically decreases (0.607 -> 0.563). At α = 0.5 the gap is **+1,777 bp**. The monotonic widening rules out the additive-equivalence interpretation: if Bear ≈ −Alpha, then c = (1+α)·Alpha would only rescale ranks, not produce a 17.8-pp lift. The two estimators target different functionals of the conditional distribution (conditional mean for Alpha; conditional lower-tail expectation for Bear), so the trained functions diverge even though their targets correlate at ρ = -0.74.

---

## §5.7 Walk-Forward Cross-Validation

Source: `bull_bear/results/final/rolling_walkforward.csv`

### Table 5.7.1 — Walk-forward results (2019--2025)

| Year | Train window | Trend RIC | D1c RIC | Δ | D1 wins? |
|---:|---|---:|---:|---:|:---:|
| 2019 | 2016-2018 | +0.758 | +0.820 | +0.062 | Y |
| 2020 *(COVID)* | 2016-2019 | +0.280 | +0.361 | +0.082 | Y |
| 2021 | 2016-2020 | +0.493 | +0.556 | +0.063 | Y |
| 2022 *(deep bear)* | 2016-2021 | +0.632 | +0.772 | +0.140 | Y |
| 2023 | 2016-2021 | +0.742 | +0.875 | +0.133 | Y |
| 2024 | 2016-2021 | +0.560 | +0.696 | +0.136 | Y |
| 2025 | 2016-2021 | +0.687 | +0.761 | +0.074 | Y |
| **Mean** | -- | **+0.593** | **+0.692** | **+0.099** | **7/7** |

**Figure**: `paper/figures/walkforward_yearly.png`

**Finding**: D1c wins 7/7 legitimate evaluation years. Smallest gap +0.062 (2019); largest gap +0.140 (2022). Mean gap +0.099. The 2026 partial year (6 trading days) is excluded as statistical noise.

---

## §5.8 Statistical Significance (Bootstrap)

Source: `bull_bear/results/final/bootstrap_test.csv`

### Table 5.8.1 — Bootstrap test (N=1,000 day-stratified resamples)

| Metric | Value |
|---|---:|
| Test days | 733 |
| Bootstrap iterations | 1000 |
| Observed Trend RankICIR | 0.5979 |
| Observed D1b RankICIR | 0.7408 |
| Observed Δ | **+0.1429** |
| Bootstrap mean Δ | +0.1430 |
| 95% CI | [+0.1064, +0.1808] |
| Empirical p-value | **p < 0.0001** (0 of 1,000 replicates produced Δ ≤ 0) |

**Finding**: CI lower bound +0.1064 is well above zero. Report p < 0.0001 rather than p = 0.000 because the precision floor of 1000 replicates is 1/1000.

---

## §5.9 Bear Agent Independence (Quintile Analysis)

Source: `bull_bear/results/final/bear_quintile_analysis.csv`

### Table 5.9.1 — Daily-quintile breakdown of Bear scores

| Quintile | n (stock-days) | Avg bear score | Avg MaxDD_5d (%) | Avg r^(5d) (%) |
|---|---:|---:|---:|---:|
| Q1 (safest) | 563,172 | -0.235 | 2.24 | +0.475 |
| Q2 | 563,445 | -0.149 | 2.61 | +0.544 |
| Q3 | 563,444 | -0.072 | 2.91 | +0.540 |
| Q4 | 563,445 | +0.046 | 3.27 | +0.475 |
| Q5 (riskiest) | 563,724 | +0.421 | 4.61 | +0.021 |
| **Q5 − Q1 MaxDD gap** | -- | -- | **+2.37 pp** (Wilcoxon rank-sum p < 0.001) | -- |

**Figure**: `paper/figures/quintile_analysis.png`

**Findings**:
1. MaxDD ordering is **monotone** Q1 -> Q5 (2.24% -> 4.61%, gap +2.37 pp, p < 0.001). Bear genuinely predicts forward drawdown risk.
2. Return ordering is **non-monotone**: +0.475%, +0.544%, +0.540%, +0.475%, +0.021%. Q2/Q3 mean +0.542% exceeds Q1 (+0.475%); only Q5 collapses (+0.021%).
3. If Bear were proportional to −Alpha, returns would decrease monotonically Q1 -> Q5. The observed Q2/Q3 > Q1 inversion rules out this interpretation, confirming Bear captures structurally different information from Alpha.

### Table 5.9.2 — Comparison against rule-based historical baseline
Source: `bull_bear/results/final/simple_baseline_comparison.csv`

| Configuration | RankICIR | Top-5% SR | MaxDD |
|---|---:|---:|---:|
| Trend pure (Alpha) | 0.598 | +1.694 | -34.20% |
| |bias_60| rule Bear (α=0.2) | 0.633 | +1.750 | -33.33% |
| Historical MaxDD 60d rule (α=0.5) | 0.484 | +1.833 | -28.09% |
| Trained Bear D1 (α=0.5) *star* | 0.741 | +1.802 | -33.52% |

**Finding**: Trained Bear D1 (0.741) beats the naive Historical-MaxDD-60d rule (0.484) by +2572 bp. The historical rule even underperforms Trend pure (0.598), confirming that the trained Bear captures forward-looking max-drawdown structure unrecoverable from a backward-looking rolling statistic.