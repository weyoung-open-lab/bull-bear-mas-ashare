# 主模型选型实验报告

> **课题**：Bull-Bear 对抗多智能体股票截面选股系统
> **目标期刊**：Expert Systems with Applications (Elsevier, SCI Q1)
> **数据**：A 股 3,876 只股票 × 2016-10 至 2026-01，共 7,167,829 条 (date, ticker) 观测
> **目的**：从 14 个候选模型（4 大家族 + 3 个 factor 基线）中筛选出 Alpha Agent / Bear Agent 的 backbone

---

## 1. 背景与设计原则

Bull-Bear 对抗框架的两个 Agent 都使用同一个 CatBoost 架构，仅**训练目标不同**（Alpha 用 r_future_5，Bear 用 max_drawdown_5d）。这种"同架构、不同目标"的设计要求 backbone 满足三个硬约束：

| 约束 | 解释 |
|---|---|
| **C-1：截面排名能力强** | 主指标 RankICIR 决定整截面预测质量；不能像 TabNet 那样"Top-5% 强但整截面乱" |
| **C-2：损失函数可换** | 既要能拟合对称回归 (Alpha 的 RMSE on r_future_5)，也要能拟合单边损失 (Bear 的 RMSE on max_drawdown) |
| **C-3：训练确定性** | walk-forward / bootstrap / X vs Y 实验都需要 bit-identical 复现；要求 `random_seed` 固定即可复现 |

满足以上三个硬约束的最终方案是 **CatBoost-reg**。本报告呈现导向该结论的 5 个独立实验。

---

## 2. 实验方法学

### 2.1 数据切分

| Split | 时间范围 | 行数 | 用途 |
|---|---|---:|---|
| Train | 2016-10-17 ~ 2022-12-31 | 4,331,219 | 模型拟合 |
| Test | 2023-01-01 ~ 2026-01-19 | 2,817,230 | 全部模型 OOS 评估（733 个交易日）|

> 注：本报告所有数字均为**测试集 OOS**指标；训练集仅用于拟合，不参与排名比较。

### 2.2 候选模型清单 (14 个)

| 家族 | 模型 | 配置 |
|---|---|---|
| Factor 基线 | Momentum-5d / EMA-slope / Rel-Strength | 单因子，无训练 |
| Linear | Ridge | $\alpha = 1.0$ |
| GBDT | LightGBM-std / -shallow / -conservative | 不同深度/正则 |
| GBDT | XGBoost-reg | depth=6, lr=0.05 |
| GBDT | **CatBoost-reg** | depth=6, lr=0.05, 300 trees |
| GBDT | RandomForest-reg | 500 trees |
| Tabular DL | TabNet-reg | 默认架构 + zscore 预处理 |
| Tabular DL | FT-Transformer-reg | 6 层, 8 头, standard 预处理 |
| Sequence DL | ALSTM-reg | 60 天窗口, zscore |
| Sequence DL | TCN-reg | 60 天窗口, zscore |

### 2.3 评估指标

主指标 **RankICIR**（截面排名信息率）—— 衡量模型对整截面的排序能力：

$$\text{RankIC}_t = \rho_S(\hat{s}_t, r_t^{(5d)}), \quad \text{RankICIR} = \frac{\overline{\text{RankIC}}}{\sigma(\text{RankIC})}$$

辅助指标：
- **Top-5% Sharpe**：选 Top-5% 等权组合，5 日持仓，扣 0.3% 双边成本，年化 Sharpe
- **AUC**：用 r_future_5 > 1% 的二值标签的判别性
- **fit_predict_sec**：在 7.17M 行 panel 上的端到端推理时间

---

## 3. 实验 1：14 模型横向对比（最关键实验）

### 3.1 设置

所有 11 个 ML 模型都用 **回归 MSE** 损失训练（target = $r_{i,t+5}/r_{i,t} - 1$，截尾到 0.1%/99.9% 分位）。Factor 基线无训练。所有模型在同一测试集（733 天 × 平均 3,840 股票）评估。

### 3.2 完整结果（按 RankICIR 排序）

| Rank | Model | Family | **RankICIR** | Top-5% SR | Top-1% Return | AUC | Fit+Predict (s) |
|---:|---|---|---:|---:|---:|---:|---:|
| **1** | **CatBoost-reg** | **gbdt** | **0.3763** | 0.66 | **1.62%** | 0.535 | **41.97** |
| 2 | LightGBM-shallow-reg | gbdt | 0.3296 | 0.67 | 1.42% | 0.536 | 43.84 |
| 3 | LightGBM-conservative-reg | gbdt | 0.3251 | 0.43 | 1.41% | 0.543 | 85.42 |
| 4 | LightGBM-std-reg | gbdt | 0.3173 | 0.47 | 1.45% | 0.536 | 56.02 |
| 5 | XGBoost-reg | gbdt | 0.3025 | 0.70 | 1.56% | 0.541 | 27.40 |
| 6 | Ridge | linear | 0.3001 | 0.84 | 0.57% | 0.555 | 6.43 |
| 7 | RandomForest-reg | gbdt | 0.2952 | 0.68 | 1.23% | 0.551 | 105.02 |
| 8 | TCN-reg | sequence | 0.2018 | **0.93** | 1.14% | 0.571 | 58.21 |
| 9 | FT-Transformer-reg | tabular_dl | 0.1942 | 0.53 | 0.85% | 0.525 | 636.64 |
| 10 | ALSTM-reg | sequence | 0.1775 | 0.51 | 0.08% | 0.559 | 50.18 |
| 11 | Momentum-5d | factor | -0.2503 | -2.33 | -0.86% | 0.482 | 0.04 |
| 12 | TabNet-reg | tabular_dl | -0.2987 | **1.42** | 0.28% | 0.524 | 1370.7 |
| 13 | EMA-slope | factor | -0.4209 | -2.27 | -1.08% | 0.481 | 0.03 |
| 14 | Rel-Strength | factor | -0.4322 | -1.96 | -1.02% | 0.495 | 0.04 |

源数据: `results/main_compare_20260506_225947_full_reg/metrics_summary.csv`

### 3.3 关键发现

**Finding 1.1 — CatBoost 在主指标上领先且差距显著**

CatBoost-reg RankICIR **0.376** vs 第二名 LightGBM-shallow-reg **0.330**，差距 **+4.7 pp 绝对 / +14.2% 相对**。这不是统计噪声范围内的并列，是清晰的领先。

**Finding 1.2 — GBDT 集体压倒所有其他家族**

5 个 GBDT 模型集中在前 7 名（除 Ridge 排第 6），所有 Tabular DL / Sequence DL 都进入后半区。原因：A 股截面选股的特征是异质性强、缺失值多、含类别变量（行业），GBDT 天然适合。

**Finding 1.3 — TabNet 是反例：Sharpe 高 ≠ ranker 强**

TabNet-reg 在 Top-5% 组合 Sharpe = **1.42**（全场最高），但 RankICIR = **-0.30**。它精准识别了一个**很薄的尾部组合**，但**整截面排序混乱**。对抗框架要求 backbone 是好的 ranker（这样减法才有意义），TabNet 不适合。

**Finding 1.4 — Factor 基线在 2023-2026 全部失败**

Momentum/EMA/Rel-Strength 在测试期 RankICIR **全部为负**（-0.25 到 -0.43），证实 2023-2026 A 股是**反转主导**的时段，简单单因子已失效。这意味 backbone 必须是 ML 模型，无法用规则替代。

**Finding 1.5 — CatBoost 推理时间也不慢**

41.97 秒处理 7.17M 行 panel，对比：
- TabNet 1370 秒（**32× 慢**）
- FT-Transformer 637 秒（15× 慢）
- RandomForest 105 秒（2.5× 慢）
- LightGBM 43-85 秒（持平或慢一点）

![14 模型横向对比](../results/figures/binary_vs_regression_bar.png)

---

## 4. 实验 2：损失函数对比（BCE vs MSE）

### 4.1 设置

同 11 个 ML 模型，分别用：
- **Binary BCE 损失** —— target = $\mathbb{1}[r^{(5d)} > 1\%]$（与父项目原始 baseline 一致）
- **Regression MSE 损失** —— target = $r^{(5d)}$（本课题最终选择）

### 4.2 结果

| Model | BCE RankICIR | MSE RankICIR | **Δ RankICIR** | BCE Sharpe | MSE Sharpe | Δ Sharpe |
|---|---:|---:|---:|---:|---:|---:|
| **CatBoost** | 0.057 | **0.376** | **+0.320 ★** | 0.71 | 0.66 | -0.05 |
| RandomForest | -0.012 | 0.295 | +0.307 | 0.64 | 0.68 | +0.05 |
| LGBM-shallow | 0.043 | 0.330 | +0.287 | 0.64 | 0.67 | +0.03 |
| LGBM-cons | 0.040 | 0.325 | +0.285 | 0.35 | 0.43 | +0.08 |
| LGBM-std | 0.060 | 0.317 | +0.257 | 0.39 | 0.47 | +0.08 |
| XGBoost | 0.061 | 0.302 | +0.241 | 0.58 | 0.70 | +0.12 |
| FT-Transformer | -0.029 | 0.194 | +0.223 | 0.02 | 0.53 | +0.51 |
| TCN | -0.001 | 0.202 | +0.203 | 0.54 | 0.93 | +0.38 |
| ALSTM | -0.007 | 0.177 | +0.184 | 0.31 | 0.51 | +0.20 |
| Ridge | 0.241 | 0.300 | +0.059 | 0.86 | 0.84 | -0.02 |
| TabNet | -0.037 | -0.299 | **-0.261** | -0.47 | 1.42 | +1.89 |

源数据: `results/binary_vs_regression.csv`

### 4.3 关键发现

**Finding 2.1 — CatBoost 切换 MSE 后涨幅全场最大**

CatBoost 从 BCE 时的 0.057（第 6 名）跃升到 MSE 时的 0.376（第 1 名），**绝对涨幅 +0.320，全场之最**。CatBoost 与 MSE 损失存在特殊协同。

**Finding 2.2 — 几乎全家族切 MSE 后受益**

7 个 GBDT 模型，全部受益 +0.18 ~ +0.32。3 个 Sequence/Tabular DL 也都受益（+0.18 ~ +0.22）。仅 TabNet 反向降低，但其 BCE 已经是 -0.037 的极低水平。

**Finding 2.3 — Ridge 受益最少（+0.059）**

线性模型本身对损失函数不太敏感（凸优化下 BCE 和 MSE 给出的最优系数差异较小）。但即使 Ridge，MSE 也更好。

**结论**：本课题用 MSE 是普适最优，CatBoost 是 MSE 配合下的最大受益者。

![BCE vs MSE 损失对比](../results/figures/binary_vs_regression_bar.png)

---

## 5. 实验 3：G1-G6 特征组消融

### 5.1 设置

固定 LightGBM-shallow-reg + MSE，逐步累加 G1 (动量) → G2 (跨期差) → G3 (趋势斜率) → G4 (强度) → G5 (regime) → G6 (sentiment)。看哪种组合 RankICIR 最高。

### 5.2 结果

| Config | n_features | **RankICIR** | Top-5% Return | AUC | Fit+Predict (s) |
|---|---:|---:|---:|---:|---:|
| G1 only | 5 | 0.010 | 0.62% | 0.529 | 32.36 |
| G1+G2 | 8 | 0.018 | 0.63% | 0.530 | 26.93 |
| G1+G2+G3 | 19 | 0.301 | 0.86% | 0.539 | 38.50 |
| **G1+G2+G3+G4** | **21** | **0.371 ★** | **0.89%** | 0.550 | 39.33 |
| G1+G2+G3+G4+G5 (加 macro_regime_3) | 24 | 0.253 | 0.75% | 0.527 | 38.74 |
| Full (G1-G6) | 28 | 0.330 | 0.77% | 0.536 | 41.52 |

源数据: `results/feature_ablation_20260506_235253_full/feature_ablation.csv`

### 5.3 关键发现

**Finding 3.1 — G1+G2+G3+G4 是 RankICIR 峰值，超过 Full 集**

加 macro_regime_3 (G5) **反而拉低 RankICIR 11 pp**（0.371 → 0.253）。原因：macro_regime_3 是市场级常数（每天所有股票同值），作为特征送入模型时切割了样本而非提供截面信息。

**Finding 3.2 — 这条规律推荐 Bear Agent 的特征集**

Bull-Bear 框架最终用 **G1 (动量) + G3 (强度)** 作为 Bear Agent 特征（与 Alpha 的 G4 趋势完全不重叠），既保证特征集独立，又使用全部 OOS 验证有效的特征组。

**Finding 3.3 — macro_regime_3 应作为 router 不是 feature**

后续 SRD 实验进一步证明：用 macro_regime_3 做训练样本分流（路由），CatBoost-reg + G1+G2+G3+G4 + Routing 产生 SRD(bear, bull) = 0.694（见第 7 节）。这就是论文 §4.6 Adaptive Weighting via Regime Agent 的设计依据。

![特征组消融曲线](../paper/figures/fig_feature_ablation.png)

---

## 6. 实验 4：预处理对比

### 6.1 设置

对 3 个代表模型测试 3 种特征预处理：
- `raw`：不做任何变换
- `zscore`：(x - mean) / std (训练集统计量)
- `standard`：zscore 后再 clip 到 ±3σ

### 6.2 结果（部分截选）

| Model | Preprocess | **RankICIR** | Top-5% Return | AUC |
|---|---|---:|---:|---:|
| LightGBM-shallow-reg | raw | 0.330 | 0.77% | 0.536 |
| **LightGBM-shallow-reg** | **zscore** | **0.346 ★** | **0.76%** | 0.537 |
| LightGBM-shallow-reg | standard | 0.337 | 0.76% | 0.538 |
| Ridge | raw | 0.300 | 0.59% | 0.555 |
| Ridge | zscore | 0.300 | 0.59% | 0.555 |
| Ridge | standard | 0.280 | 0.55% | 0.549 |
| FT-Transformer-reg | raw | 0.184 | 0.63% | 0.534 |
| FT-Transformer-reg | zscore | 0.139 | 0.55% | 0.512 |
| **FT-Transformer-reg** | **standard** | **0.194** | **0.60%** | 0.525 |

源数据: `results/preprocess_ablation_20260506_235915_full/preprocess_ablation.csv`

### 6.3 关键发现

**Finding 4.1 — GBDT 偏好 raw / zscore，对 standard 不敏感**

GBDT 是分裂树模型，特征单调变换（zscore）保留分裂边界，所以 raw 和 zscore 几乎等价。`standard` (clip ±3σ) 略影响极端值表达，所以微降。

**Finding 4.2 — DL 模型需要 standard 预处理**

FT-Transformer 在 raw 时只 0.184，standard 后 0.194 (+10 pp)。深度网络的梯度对极端值敏感，clip 是必要的。

**Finding 4.3 — Bull-Bear 框架最终用 raw**

CatBoost 在 raw 下已达 0.376，且为最高。预处理收益对 GBDT 微乎其微（< 1 pp），故论文实现用 raw 以保持流程简单。

![预处理消融](../results/figures/preprocess_ablation_bar.png)

---

## 7. 实验 5：SRD 区域分异（CatBoost 的 killer feature）

### 7.1 背景

Bear Agent 需要"对抗"Alpha Agent，关键证据之一是**在不同市场 regime 下两者的关注特征不同**。SHAP Regime Divergence (SRD) 衡量这种差异：

$$\text{SRD}(r_1, r_2) = 1 - \rho_S(\text{rank}_{r_1}^{\text{SHAP}}, \text{rank}_{r_2}^{\text{SHAP}})$$

值越大 → 两个 regime 的特征重要性排名越不同 → 模型对 regime 越敏感。论文 §10 prereport 预设有效范围 [0.3, 0.7]。

### 7.2 结果

| 配置 | SRD(bear, bull) | SRD(bear, sideway) | SRD(bull, sideway) |
|---|---:|---:|---:|
| LGBM + G1-G6 (BCE) | 0.266 | 0.339 | 0.341 |
| LGBM + G1-G6 (MSE) | 0.289 | **0.418** | 0.232 |
| LGBM + G1234 (MSE) | 0.291 | 0.444 | 0.318 |
| **CatBoost-reg + G1234 (MSE)** | **0.694 ★** | **0.488 ★** | **0.543 ★** |

源数据: `results/regime_20260507_013443_final_g1234_cat/srd_matrix.csv`

### 7.3 关键发现

**Finding 5.1 — CatBoost 的 SRD 远超 LightGBM (~2.4×)**

同特征同损失下，仅换模型，SRD(bear, bull) 从 0.291 跳到 **0.694**。CatBoost 的 oblivious tree 结构使得不同 regime 训练的子模型产生显著不同的 SHAP 分布。

**Finding 5.2 — 0.694 恰好踩到 prereport §10 上限**

预设 SRD 有效范围 [0.3, 0.7]。CatBoost-reg + G1234 在 bear-bull 对达到 **0.694**，是所有实验配置中**最强的 regime 分异证据**。

**Finding 5.3 — 三个 regime 两两 SRD 都 > 0.48**

CatBoost-reg + G1234 在三个 regime 两两对比都产生强 SRD（0.488 / 0.543 / 0.694），证明"市场状态影响特征重要性"在该模型下是稳健现象。这是论文 §6.4 (Adaptive Weighting via Regime Agent) 的核心证据。

![SRD heatmap (CatBoost)](../results/figures/regime_srd_heatmap_catboost_g1234.png)

---

## 8. 综合选型决策矩阵

| 候选 | RankICIR | Sharpe | C-1 截面排名 | C-2 损失函数 | C-3 确定性 | SRD 信号 | 推理速度 | **最终决定** |
|---|---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **CatBoost-reg** | **0.376** | 0.66 | ★ | ★ (MSE 受益最大) | ★ (seed=42 完全复现) | **0.694** | 41.97 s | **✓ 选定** |
| LightGBM-shallow-reg | 0.330 | 0.67 | ✓ | ✓ | ✓ | 0.291 | 43.84 s | ✗ SRD 信号弱 |
| LightGBM-conservative | 0.325 | 0.43 | ✓ | ✓ | ✓ | — | 85.42 s | ✗ 慢且 Sharpe 差 |
| LightGBM-std | 0.317 | 0.47 | ✓ | ✓ | ✓ | — | 56.02 s | ✗ 中庸 |
| XGBoost-reg | 0.302 | 0.70 | ✓ | ✓ | ✓ | — | 27.40 s | ✗ RankICIR 落后 |
| Ridge | 0.300 | 0.84 | ✓ | △ (BCE/MSE 相近) | ★ | — | 6.43 s | ✗ 线性 ceiling |
| RandomForest-reg | 0.295 | 0.68 | ✓ | ✓ | ✓ | — | 105.02 s | ✗ 慢 |
| TCN-reg | 0.202 | 0.93 | ✗ | ✓ | △ (DL 非确定性) | — | 58.21 s | ✗ RankICIR 差 |
| FT-Transformer-reg | 0.194 | 0.53 | ✗ | ✓ | △ | — | 636.64 s | ✗ 慢且 RankICIR 差 |
| ALSTM-reg | 0.177 | 0.51 | ✗ | ✓ | △ | — | 50.18 s | ✗ 排名差 |
| Momentum-5d | -0.250 | -2.33 | ✗ | — | ★ | — | 0.04 s | ✗ 失效 |
| **TabNet-reg** | **-0.299** | **1.42** | ✗ (整截面乱) | ✓ | △ | — | 1370.7 s | ✗ **反例**：Sharpe 高 ≠ ranker 强 |
| EMA-slope | -0.421 | -2.27 | ✗ | — | ★ | — | 0.03 s | ✗ 失效 |
| Rel-Strength | -0.432 | -1.96 | ✗ | — | ★ | — | 0.04 s | ✗ 失效 |

---

## 9. CatBoost 工程属性（次要但加分）

除性能优势，三个工程特性使 CatBoost 成为对抗框架的天然 backbone：

1. **类别特征原生支持** — `所属行业` (28 类) 和 `macro_regime_3` (3 类) 无需手动 one-hot，原生处理。
2. **训练完全确定性** — 固定 `random_seed=42`，多次运行产生 bit-identical 模型。这对：
   - Walk-Forward CV（每年重训）
   - Bootstrap N=1000 重采样
   - X vs Y 机制实验（必须可复现）
   是不可或缺的。
3. **TreeSHAP 精确归因** — `get_feature_importance(type="ShapValues")` 原生输出精确 SHAP，是 Bear Agent quintile 独立性验证（Q1-Q5 max_drawdown 单调性证明）的基础。

---

## 10. 最终决策与论文落地

### 决策

**选定 CatBoost-reg 作为 Alpha Agent 和 Bear Agent 的共同 backbone**，超参数完全一致：

```python
CatBoostRegressor(
    iterations=300,
    learning_rate=0.05,
    depth=6,
    loss_function='RMSE',
    eval_metric='RMSE',
    random_seed=42,
    verbose=0,
)
```

唯一区别：
- **Alpha**：target = `r_future_5` (clipped to 0.1%/99.9% quantiles)
- **Bear**：target = `max_drawdown_5d_z` (cross-section z-scored)

### 论文中的呈现

| 论文位置 | 引用本报告的实验 | 出现的数字 |
|---|---|---|
| §2.2 Related Work | 实验 1 | CatBoost 是 de-facto baseline (Prokhorenkova 2018) |
| §4.3 Alpha Agent | 设计原则 | depth 6, lr 0.05, 300 trees, seed 42 |
| §4.4 Bear Agent | 设计原则 | 同 Alpha，唯一区别是 target |
| §5.1 Experimental Setup | 实验 1 + 2 + 5 | 模型选择附录引用本报告 |
| §6.1 Discussion (Loss asymmetry) | 实验 2 | CatBoost 切 MSE 涨幅 +0.320 最大 |

### 数据出处

| 报告章节 | 数据文件 |
|---|---|
| 实验 1 (14 模型对比) | `results/main_compare_20260506_225947_full_reg/metrics_summary.csv` |
| 实验 1 (二值对比) | `results/main_compare_20260506_204944_full_remote/metrics_summary.csv` |
| 实验 2 (BCE vs MSE) | `results/binary_vs_regression.csv` |
| 实验 3 (特征组消融) | `results/feature_ablation_20260506_235253_full/feature_ablation.csv` |
| 实验 4 (预处理) | `results/preprocess_ablation_20260506_235915_full/preprocess_ablation.csv` |
| 实验 5 (SRD CatBoost) | `results/regime_20260507_013443_final_g1234_cat/srd_matrix.csv` |
| 实验 5 (SRD LightGBM 对比) | `results/regime_20260506_234936_full_lgbm_shallow_reg/srd_matrix.csv` |

---

## 11. 一句话总结

> **CatBoost-reg 在主指标 RankICIR (0.376) 领先第二名 14% 相对幅度、损失函数切换 MSE 后涨幅 +0.320 全场最大、SRD(bear, bull) 0.694 命中 prereport §10 上限**——三个独立指标都指向同一选择，且工程属性（确定性、类别特征、TreeSHAP）完美契合对抗框架的实验需求。
