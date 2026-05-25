"""Step 3 — 三个并行 Bear Agent 方向。

D1: G1+G3 特征 (8) + max_drawdown_5d_z 目标   (与 Alpha 特征不重叠)
D2: G4 特征 (5)     + path_vol_5d_z 目标       (同特征，不同目标)
D3: G4 特征 (5)     + disappointment_z 目标    (直接预测 Alpha 错位)

对每个方向：
  1. 训练 + 验证集 Spearman
  2. 测试集预测 -> α 网格 {0.1, 0.2, 0.3, 0.5}
  3. 与 Bear_C |bias_60| 规则组合（mean of z-scores）
  4. 特征重要性 top-5

验收：任一 Di 同时满足
  • corr(Di_pred, Alpha) < 0.1
  • best α 时 RankICIR > 0.6325 (Bear_C 基线)
  • val Spearman > 0.15

输出：
  models/bear_D1_agent.cbm, bear_D2_agent.cbm, bear_D3_agent.cbm
  results/step3_parallel_results.csv
  results/target_correlation_matrix.csv
  results/feature_importance_D123.csv
"""

from __future__ import annotations

import sys
from pathlib import Path
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.stats import pearsonr, spearmanr

from bull_bear.config_bb import (
    ALPHA_AGENT_PATH, ALPHA_GRID, BB_MODELS, BB_RESULTS,
    BEAR_FEATURES_D1, BEAR_FEATURES_D2, BEAR_FEATURES_D3,
    DATE_COL, P10_ALPHA,
    TARGET_RET_COL, TEST_END, TEST_START, TICKER_COL,
    TRAIN_END, TRAIN_START, VAL_END, VAL_START,
)
from bull_bear.src.bear_agent import BearAgent
from bull_bear.src.bear_target import (
    build_disappointment, build_max_drawdown_5d, build_path_vol_5d,
    cross_section_zscore,
)
from src.data import load_dataset
from bull_bear.src.metrics_utils import evaluate_full


ALPHA_FEATURES = [
    "ma60_slope", "ema180_slope",
    "bias_60", "bias_60_vr", "ma180_slope",
]
BEAR_C_REF = 0.6325
TREND_PURE_REF = 0.5979


def safe_pearson(a, b) -> float:
    """Pearson corr ignoring NaN; returns NaN if too few finite pairs."""
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 100:
        return float("nan")
    r, _ = pearsonr(a[m], b[m])
    return float(r)


def safe_spearman(a, b) -> float:
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 100:
        return float("nan")
    rho, _ = spearmanr(a[m], b[m])
    return float(rho)


def zscore_daily(panel: pd.DataFrame, col: str) -> np.ndarray:
    """每日截面 z-score，NaN -> 0。"""
    out = np.zeros(len(panel), dtype="float64")
    for d, g in panel.groupby(DATE_COL):
        v = g[col].to_numpy(dtype="float64")
        mu, sd = np.nanmean(v), np.nanstd(v, ddof=0)
        if sd > 1e-9 and np.isfinite(mu):
            z = (v - mu) / sd
            z = np.where(np.isfinite(z), z, 0.0)
        else:
            z = np.zeros_like(v)
        out[g.index.to_numpy()] = z
    return np.where(np.isfinite(out), out, 0.0).astype("float32")


def main() -> None:
    print("=" * 80)
    print("Step 3 — three parallel Bear Agent directions")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 0. 加载 + 切分
    # ------------------------------------------------------------------
    print("\n[0/6] load + pre-sort + alpha predict ...")
    df = load_dataset().dropna(subset=[TARGET_RET_COL]).reset_index(drop=True)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    # Alpha 预测（全 panel，用于 disappointment 目标）
    alpha = CatBoostRegressor(); alpha.load_model(str(ALPHA_AGENT_PATH))
    medians_path = Path(str(ALPHA_AGENT_PATH).replace(".cbm", "_medians.csv"))
    alpha_medians = pd.read_csv(medians_path, index_col=0).iloc[:, 0]
    X_alpha_full = df[ALPHA_FEATURES].astype("float32").fillna(alpha_medians)
    df["alpha_pred"] = alpha.predict(X_alpha_full).astype("float32")

    # ------------------------------------------------------------------
    # 1. 构造三个目标变量（注意：这些函数会 sort_values+reset_index）
    # ------------------------------------------------------------------
    print("\n[1/6] build three target variables ...")
    df = build_max_drawdown_5d(df, ret_col="ret_1d", window=5)
    df = build_path_vol_5d(df, ret_col="ret_1d", window=5)
    df = build_disappointment(df, alpha_pred_col="alpha_pred",
                                  actual_col=TARGET_RET_COL)
    # 截面 z-score (per-day)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = cross_section_zscore(df, "max_drawdown_5d")
    df = cross_section_zscore(df, "path_vol_5d")
    df = cross_section_zscore(df, "disappointment")

    # 重要：所有 sort+reset_index 完成后才创建 mask（避免 mask 指向旧索引位置）
    mask_tr = (df[DATE_COL] >= pd.Timestamp(TRAIN_START)) & (df[DATE_COL] <= pd.Timestamp(TRAIN_END))
    mask_va = (df[DATE_COL] >= pd.Timestamp(VAL_START))  & (df[DATE_COL] <= pd.Timestamp(VAL_END))
    mask_te = (df[DATE_COL] >= pd.Timestamp(TEST_START)) & (df[DATE_COL] <= pd.Timestamp(TEST_END))

    # 目标分布 + 互相关诊断
    tr_full = df.loc[mask_tr]
    md = tr_full["max_drawdown_5d"].to_numpy("float64")
    pv = tr_full["path_vol_5d"].to_numpy("float64")
    dp = tr_full["disappointment"].to_numpy("float64")
    rf = tr_full[TARGET_RET_COL].to_numpy("float64")

    print("\n  training-set target distributions (n = "
          f"{(~np.isnan(md)).sum():,} after target NaN):")
    for name, arr in [("max_drawdown_5d", md), ("path_vol_5d", pv),
                       ("disappointment", dp)]:
        a = arr[~np.isnan(arr)]
        print(f"    {name:18s}  mean={a.mean():+.5f}  std={a.std(ddof=0):.5f}  "
              f"q10={np.quantile(a, 0.10):+.4f}  q50={np.quantile(a, 0.50):+.4f}  "
              f"q90={np.quantile(a, 0.90):+.4f}")

    corr_mat = pd.DataFrame({
        "target": ["max_drawdown_5d", "path_vol_5d", "disappointment"],
        "vs_r_future_5":  [safe_pearson(md, rf), safe_pearson(pv, rf),
                            safe_pearson(dp, rf)],
        "vs_max_drawdown": [1.0, safe_pearson(md, pv), safe_pearson(md, dp)],
        "vs_path_vol":     [safe_pearson(md, pv), 1.0, safe_pearson(pv, dp)],
        "vs_disappoint":   [safe_pearson(md, dp), safe_pearson(pv, dp), 1.0],
    })
    print("\n  target pairwise correlations (Pearson, training set):")
    print(corr_mat.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    corr_mat.to_csv(BB_RESULTS / "target_correlation_matrix.csv",
                     index=False, encoding="utf-8-sig")

    # 重点验证
    print(f"\n  Key orthogonality (target vs r_future_5d):")
    print(f"    max_drawdown_5d: {safe_pearson(md, rf):+.4f}  (expected ~ -0.3 to 0)")
    print(f"    path_vol_5d:     {safe_pearson(pv, rf):+.4f}  (expected <|0.3|)")
    print(f"    disappointment:  {safe_pearson(dp, rf):+.4f}  (expected <|0.3|)")

    train_df = df.loc[mask_tr].reset_index(drop=True)
    val_df   = df.loc[mask_va].reset_index(drop=True)
    test_df  = df.loc[mask_te].sort_values([DATE_COL, TICKER_COL]).reset_index(drop=True)
    print(f"\n  rows: train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

    # ------------------------------------------------------------------
    # 2-4. 训练 D1 / D2 / D3
    # ------------------------------------------------------------------
    bear_specs = [
        ("D1", BEAR_FEATURES_D1, "max_drawdown_5d_z", "max_drawdown_5d"),
        ("D2", BEAR_FEATURES_D2, "path_vol_5d_z",     "path_vol_5d"),
        ("D3", BEAR_FEATURES_D3, "disappointment_z",  "disappointment"),
    ]
    trained_bears: dict[str, BearAgent] = {}
    val_spearman: dict[str, float] = {}
    feature_importance_rows = []

    for label, features, tgt_z, tgt_raw in bear_specs:
        print(f"\n[2-4] train Bear {label} — features={len(features)}  target={tgt_raw} ...")
        bear = BearAgent(features=features, name=f"bear_{label}")
        bear.train(train_df, target_col=tgt_z, df_val=val_df, save=True)
        sp = bear.validate_spearman(val_df, target_col=tgt_raw)
        val_spearman[label] = sp
        print(f"   val Spearman(pred, {tgt_raw}) = {sp:+.4f}  "
              f"{'OK (>0.15)' if sp > 0.15 else 'WEAK'}")
        trained_bears[label] = bear

        # 特征重要性 top-5
        fi = bear.model.get_feature_importance()
        fi_df = pd.DataFrame({"feature": features, "importance": fi}).sort_values(
            "importance", ascending=False)
        print(f"   top-5 features:")
        for _, r in fi_df.head(5).iterrows():
            print(f"     {r['feature']:22s}  {r['importance']:>6.2f}")
            feature_importance_rows.append({
                "direction": label, "feature": r["feature"],
                "importance": float(r["importance"]),
            })

    pd.DataFrame(feature_importance_rows).to_csv(
        BB_RESULTS / "feature_importance_D123.csv",
        index=False, encoding="utf-8-sig",
    )

    # ------------------------------------------------------------------
    # 5. 测试集预测 + 正交性诊断
    # ------------------------------------------------------------------
    print("\n[5/6] test-set predictions + orthogonality vs Alpha and |bias_60| ...")
    test_df["alpha_pred"] = alpha.predict(
        test_df[ALPHA_FEATURES].astype("float32").fillna(alpha_medians)
    ).astype("float32")
    test_df["bull_score_z"] = zscore_daily(test_df, "alpha_pred")
    test_df["abs_bias60"] = test_df["bias_60"].abs()
    test_df["bear_C_z"] = zscore_daily(test_df, "abs_bias60")

    bear_preds: dict[str, np.ndarray] = {}
    bear_z: dict[str, np.ndarray] = {}
    for label, bear in trained_bears.items():
        pred = bear.predict_panel(test_df).astype("float32")
        bear_preds[label] = pred
        test_df[f"bear_{label}_raw"] = pred
        bear_z[label] = zscore_daily(test_df, f"bear_{label}_raw")
        test_df[f"bear_{label}_z"] = bear_z[label]
        corr_alpha = safe_pearson(pred, test_df["alpha_pred"].to_numpy("float64"))
        corr_b60   = safe_pearson(pred, test_df["abs_bias60"].to_numpy("float64"))
        sp_alpha   = safe_spearman(pred, test_df["alpha_pred"].to_numpy("float64"))
        sp_b60     = safe_spearman(pred, test_df["abs_bias60"].to_numpy("float64"))
        print(f"\n   Bear {label}:  corr vs Alpha   = P={corr_alpha:+.4f}  S={sp_alpha:+.4f}  "
              f"(target |P|<0.1)")
        print(f"            corr vs |bias_60| = P={corr_b60:+.4f}  S={sp_b60:+.4f}")

    # ------------------------------------------------------------------
    # 6. α 网格 + 组合
    # ------------------------------------------------------------------
    print("\n[6/6] alpha sweep + Bear_C combination ...")
    meta = test_df[[DATE_COL, TICKER_COL, TARGET_RET_COL]].reset_index(drop=True)
    bull_z = test_df["bull_score_z"].to_numpy("float32")
    bear_C_z_arr = test_df["bear_C_z"].to_numpy("float32")

    rows = []

    # Trend pure
    m_pure = evaluate_full(meta, test_df["alpha_pred"].to_numpy("float32"))
    rows.append({"config": "Trend pure (Alpha)", "alpha": np.nan,
                  "rankicir": float(m_pure["rankicir"]),
                  "sharpe":   float(m_pure["top5pct_sharpe"]),
                  "maxdd":    float(m_pure["top5pct_max_dd"]),
                  "vs_BearC_bp": float(m_pure["rankicir"] - BEAR_C_REF) * 10000})

    # Bear_C
    conv_C = (bull_z - P10_ALPHA * bear_C_z_arr).astype("float32")
    m_C = evaluate_full(meta, conv_C)
    rows.append({"config": "Bear_C α=0.2 (|bias_60| rule)", "alpha": P10_ALPHA,
                  "rankicir": float(m_C["rankicir"]),
                  "sharpe":   float(m_C["top5pct_sharpe"]),
                  "maxdd":    float(m_C["top5pct_max_dd"]),
                  "vs_BearC_bp": 0.0})

    # D1 / D2 / D3 α 网格
    for label in ("D1", "D2", "D3"):
        bz = bear_z[label]
        for a in ALPHA_GRID:
            conv = (bull_z - a * bz).astype("float32")
            m = evaluate_full(meta, conv)
            rows.append({"config": f"Bear_{label} α={a:.1f}", "alpha": float(a),
                          "rankicir": float(m["rankicir"]),
                          "sharpe":   float(m["top5pct_sharpe"]),
                          "maxdd":    float(m["top5pct_max_dd"]),
                          "vs_BearC_bp": (float(m["rankicir"]) - BEAR_C_REF) * 10000})

    # D? + Bear_C combined（mean of z-scores）
    for label in ("D1", "D2", "D3"):
        combo_bear = 0.5 * (bear_C_z_arr + bear_z[label])
        for a in ALPHA_GRID:
            conv = (bull_z - a * combo_bear).astype("float32")
            m = evaluate_full(meta, conv)
            rows.append({"config": f"Bear_{label}+Bear_C combined α={a:.1f}",
                          "alpha": float(a),
                          "rankicir": float(m["rankicir"]),
                          "sharpe":   float(m["top5pct_sharpe"]),
                          "maxdd":    float(m["top5pct_max_dd"]),
                          "vs_BearC_bp": (float(m["rankicir"]) - BEAR_C_REF) * 10000})

    df_out = pd.DataFrame(rows)
    out_csv = BB_RESULTS / "step3_parallel_results.csv"
    df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # 输出
    print()
    line = "+" + "-" * 44 + "+" + "-" * 7 + "+" + "-" * 10 + "+" + "-" * 9 + "+" + "-" * 10 + "+" + "-" * 12 + "+"
    print(line)
    print(f"| {'Config':42s} | {'α':>5s} | {'RankICIR':>8s} | {'SR':>7s} | "
          f"{'MaxDD':>8s} | {'Δ vs C bp':>10s} |")
    print(line)
    for r in rows:
        a_str = "—" if not np.isfinite(r["alpha"]) else f"{r['alpha']:.1f}"
        print(f"| {r['config']:42s} | {a_str:>5s} | "
              f"{r['rankicir']:>8.4f} | {r['sharpe']:>+7.3f} | "
              f"{r['maxdd']*100:>+7.2f}% | {r['vs_BearC_bp']:>+9.1f} |")
    print(line)

    # ------------------------------------------------------------------
    # VERDICT
    # ------------------------------------------------------------------
    print("\n=== VERDICT ===")

    # 找各 direction 最佳 α（仅 standalone，不含 combined）
    for label in ("D1", "D2", "D3"):
        sub = [r for r in rows if r["config"].startswith(f"Bear_{label} α")]
        best = max(sub, key=lambda r: r["rankicir"])
        corr_alpha = safe_pearson(bear_preds[label],
                                    test_df["alpha_pred"].to_numpy("float64"))
        sp = val_spearman[label]
        chk1 = abs(corr_alpha) < 0.1
        chk2 = best["rankicir"] > BEAR_C_REF
        chk3 = sp > 0.15
        verdict = "PASS" if (chk1 and chk2 and chk3) else "FAIL"
        print(f"  {label}:  best α={best['alpha']:.1f}  RankICIR={best['rankicir']:.4f}  "
              f"vs Bear_C 0.6325: Δ={best['vs_BearC_bp']:+.1f} bp")
        print(f"       [{'OK' if chk1 else 'NO'}] |corr_alpha|<0.1 ({corr_alpha:+.3f})  "
              f"[{'OK' if chk2 else 'NO'}] beats Bear_C  "
              f"[{'OK' if chk3 else 'NO'}] val Spearman>0.15 ({sp:+.3f})")
        print(f"       Overall: {verdict}")

        # 组合最佳
        sub_combo = [r for r in rows if r["config"].startswith(f"Bear_{label}+Bear_C")]
        best_combo = max(sub_combo, key=lambda r: r["rankicir"])
        print(f"       combined Bear_C+{label} best α={best_combo['alpha']:.1f}  "
              f"RankICIR={best_combo['rankicir']:.4f}  Δ vs C={best_combo['vs_BearC_bp']:+.1f} bp")

    # 全局最优
    best_overall = max(rows, key=lambda r: r["rankicir"])
    print(f"\n  Overall best: {best_overall['config']}  "
          f"RankICIR={best_overall['rankicir']:.4f}  "
          f"Δ vs Bear_C={best_overall['vs_BearC_bp']:+.1f} bp")

    print(f"\nOutputs:")
    for p in [out_csv,
              BB_RESULTS / "target_correlation_matrix.csv",
              BB_RESULTS / "feature_importance_D123.csv",
              BB_MODELS / "bear_D1_agent.cbm",
              BB_MODELS / "bear_D2_agent.cbm",
              BB_MODELS / "bear_D3_agent.cbm"]:
        if p.exists():
            print(f"  -> {p.relative_to(BB_RESULTS.parent.parent)}")


if __name__ == "__main__":
    main()
