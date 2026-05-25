# 论文主对比 SUMMARY

- 跑批目录: `main_compare_20260506_204944_full_remote`
- 训练终止日: 2023-01-01；测试 2023-01 ~ 2026-01（含牛市/熊市/震荡 3 种 regime，733 个交易日）
- 截面模型测试集: **2,817,230** 样本（733 天 × 3,876 只 A 股，去 NaN 后）
- 序列模型测试集: **387,600** 样本（每股票上限 100 个 10 日窗，避免 8.5M 窗口爆显存）
- 模型数量: 14（factor 3 + linear 1 + gbdt 6 + tabular_dl 2 + sequence 2）

## 关键发现

1. **RankICIR 冠军：LogisticRegression**（0.241），显著领先所有 GBDT 与 DL 变体。线性可分性在大样本上展现出强势 baseline。
2. **AUC 冠军：ALSTM**（0.575），但 RankIC≈0；
   说明序列 DL 学到了二分类边界但未学到排序信号——典型「分类 vs 排名」错位。
3. **Top-5% 多头 Sharpe 冠军：LogisticRegression**（0.856），年化收益 13.4%，最大回撤 -41.8%。
4. **Top-1% 多头 Sharpe 冠军：LightGBM-shallow**（0.676），窄多头组合上浅树 LightGBM 与 CatBoost 反超 LogisticRegression，说明在分布尾部 GBDT 的非线性优势开始体现。
5. **传统因子基线全部强烈负向**（Top-5% Sharpe 均值 -2.19），印证 2023+ 测试期 A 股短期动量反转的市场特性，也是论文 Regime-Conditioned 创新（§5）天然的铺垫论点。

## 论文 §10 预期区间对照

| 指标 | 预期区间 | 实测最佳 | 是否达标 |
|---|---|---|---|
| AUC | 0.54 – 0.62 | 0.575 (ALSTM) | ✓ |
| IC 均值 | 0.02 – 0.06 | 0.025 | ✓ |
| RankIC 均值 | 0.03 – 0.08 | 0.035 | ✓ |
| RankICIR | 0.4 – 1.2 | 0.241 | ⚠ 未达 0.4 下限 |
| Top-1% 5d 收益 | 1.5% – 4% | 1.11% | ⚠ 略低 |
| Top-5% Sharpe | 0.5 – 1.8 | 0.86 | ✓ |

结论：分类指标和 Sharpe 已落入合理区间；但 RankICIR 未达预报告 §10 的下限 0.4，意味着横截面排序稳定性还有显著提升空间——这正是论文 §5 提出的 **Regime-Conditioned SHAP Ensemble** 期望解决的核心问题。

## 输出文件

- [table1_metrics.csv](table1_metrics.csv) — 论文 Table 1（主对比）
- [table2_backtest.csv](table2_backtest.csv) — 论文 Table 2（回测）
- [figures/nav_top5pct.png](figures/nav_top5pct.png) — Top-5% 净值曲线（预报告 Figure 1）
- [figures/sharpe_bar.png](figures/sharpe_bar.png) — Sharpe 横向对比
- [figures/rankic_box.png](figures/rankic_box.png) — 日 RankIC 分布箱线图
- [predictions/](predictions/) — 14 个模型的逐样本预测，供 SHAP / Regime 分析直接复用