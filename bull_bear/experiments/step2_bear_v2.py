"""Step 2 — Bear Agent V2：G4 + abs/sq 非线性派生特征。

四个任务：
  Task 1: 加入派生特征 + 验证正交性
            corr(|bias_60|, bias_60) 应接近 0
            corr(|bias_60|, Alpha) 应 < 0.1
  Task 2: 重训 Bear Agent V2（target 同 step1: max_drawdown_5d_z）
            打印 feature importance
  Task 3: 对抗仲裁 α 网格对比
            Trend pure / Bear_C / Bear_A (step1) / Bear_V2
  Task 4: 验收
            Bear_V2 最佳 α > Bear_C 0.6325 (训练击败规则)
            或承认规则即终态

输出：
  bull_bear/results/models/bear_agent_v2.cbm
  bull_bear/results/step2_comparison.csv
  bull_bear/results/bear_feature_importance.csv
  bull_bear/results/predictions_step2.parquet
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
    BEAR_FEATURES_V2, DATE_COL, P10_ALPHA,
    TARGET_RET_COL, TEST_END, TEST_START, TICKER_COL,
    TRAIN_END, TRAIN_START, VAL_END, VAL_START,
)
from bull_bear.src.bear_agent import BearAgent
from bull_bear.src.bear_target import build_max_drawdown_5d, cross_section_zscore
from bull_bear.src.feature_engineering import add_bear_v2_features
from src.data import load_dataset
from bull_bear.src.metrics_utils import evaluate_full


ALPHA_FEATURES = [
    "ma60_slope", "ema180_slope",
    "bias_60", "bias_60_vr", "ma180_slope",
]
BEAR_C_REF = 0.6325    # step1 验证后的真实基线
TREND_PURE_REF = 0.5979


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
    print("Step 2 — Bear Agent V2 with abs/sq augmented features")
    print("=" * 80)

    # =============================================================
    # 0. 加载 + 派生特征 + 切分
    # =============================================================
    print("\n[0/4] load + derive features + 3-way split ...")
    df = load_dataset().dropna(subset=[TARGET_RET_COL]).reset_index(drop=True)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = add_bear_v2_features(df)
    mask_tr = (df[DATE_COL] >= pd.Timestamp(TRAIN_START)) & (df[DATE_COL] <= pd.Timestamp(TRAIN_END))
    mask_va = (df[DATE_COL] >= pd.Timestamp(VAL_START))  & (df[DATE_COL] <= pd.Timestamp(VAL_END))
    mask_te = (df[DATE_COL] >= pd.Timestamp(TEST_START)) & (df[DATE_COL] <= pd.Timestamp(TEST_END))
    print(f"   panel rows: {len(df):,}   v2 features = {BEAR_FEATURES_V2}")

    # =============================================================
    # 1. Task 1 — 正交性验证
    # =============================================================
    print("\n[1/4] Task 1: orthogonality checks on test panel ...")
    # 先在测试集上做 Alpha 预测以做正交性比较
    test_df = df.loc[mask_te].sort_values([DATE_COL, TICKER_COL]).reset_index(drop=True)
    alpha = CatBoostRegressor(); alpha.load_model(str(ALPHA_AGENT_PATH))
    medians_path = Path(str(ALPHA_AGENT_PATH).replace(".cbm", "_medians.csv"))
    alpha_medians = pd.read_csv(medians_path, index_col=0).iloc[:, 0]
    X_alpha_test = test_df[ALPHA_FEATURES].astype("float32").fillna(alpha_medians)
    test_df["alpha_pred"] = alpha.predict(X_alpha_test).astype("float32")

    # 计算相关性（在测试集上）
    def safe_pearson(a, b) -> float:
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 100:
            return float("nan")
        r, _ = pearsonr(a[m], b[m])
        return float(r)

    a_b60   = test_df["bias_60"].to_numpy("float64")
    a_abs60 = test_df["abs_bias_60"].to_numpy("float64")
    a_sq60  = test_df["bias_60_sq"].to_numpy("float64")
    alpha_s = test_df["alpha_pred"].to_numpy("float64")

    print(f"\n   On test panel ({len(test_df):,} rows):")
    print(f"     corr(|bias_60|, bias_60)   = {safe_pearson(a_abs60, a_b60):+.4f}  "
          f"(expected ~ 0)")
    print(f"     corr(bias_60_sq, bias_60)  = {safe_pearson(a_sq60, a_b60):+.4f}  "
          f"(expected ~ 0)")
    print(f"     corr(|bias_60|, Alpha)     = {safe_pearson(a_abs60, alpha_s):+.4f}  "
          f"(expected < 0.1)")
    print(f"     corr(bias_60_sq, Alpha)    = {safe_pearson(a_sq60, alpha_s):+.4f}  "
          f"(expected < 0.1)")
    print(f"     corr(|ma60_slope|, Alpha)  = "
          f"{safe_pearson(test_df['abs_ma60_slope'].to_numpy('float64'), alpha_s):+.4f}")
    print(f"     corr(|ema180_slope|, Alpha)= "
          f"{safe_pearson(test_df['abs_ema180_slope'].to_numpy('float64'), alpha_s):+.4f}")

    # =============================================================
    # 2. Task 2 — 训练 Bear V2
    # =============================================================
    print("\n[2/4] Task 2: build max_drawdown_5d target + train Bear V2 ...")
    df_full = build_max_drawdown_5d(df, ret_col="ret_1d", window=5)
    df_full[DATE_COL] = pd.to_datetime(df_full[DATE_COL])
    df_full = cross_section_zscore(df_full, "max_drawdown_5d")
    train_df = df_full.loc[mask_tr].copy().reset_index(drop=True)
    val_df = df_full.loc[mask_va].copy().reset_index(drop=True)

    bear_v2 = BearAgent(features=BEAR_FEATURES_V2, name="bear_v2")
    bear_v2.train(train_df, target_col="max_drawdown_5d_z", df_val=val_df, save=True)
    val_sp_v2 = bear_v2.validate_spearman(val_df, target_col="max_drawdown_5d")
    print(f"   val Spearman(Bear_V2 pred, raw max_drawdown_5d) = {val_sp_v2:+.4f}  "
          f"(step1 was +0.1555; expect higher)")

    # Feature importance
    print("\n   feature importance (PredictionValuesChange):")
    fi = bear_v2.model.get_feature_importance()
    fi_df = pd.DataFrame({"feature": BEAR_FEATURES_V2, "importance": fi})
    fi_df = fi_df.sort_values("importance", ascending=False).reset_index(drop=True)
    for _, r in fi_df.iterrows():
        print(f"     {r['feature']:18s}  {r['importance']:>6.2f}")
    fi_df.to_csv(BB_RESULTS / "bear_feature_importance.csv",
                  index=False, encoding="utf-8-sig")
    top3 = list(fi_df.head(3)["feature"])
    print(f"   top-3: {top3}")
    nonlinear_in_top3 = any(f.startswith("abs_") or f.endswith("_sq") for f in top3)
    print(f"   non-linear features in top-3: "
          f"{'YES' if nonlinear_in_top3 else 'NO (V2 features did not dominate)'}")

    # =============================================================
    # 3. Task 3 — 对抗仲裁 α 网格
    # =============================================================
    print("\n[3/4] Task 3: alpha sweep on test panel ...")
    # 测试集预测
    test_df_full = df_full.loc[mask_te].sort_values([DATE_COL, TICKER_COL]).reset_index(drop=True)
    assert len(test_df_full) == len(test_df), "test panel size mismatch"
    bear_v2_raw = bear_v2.predict_panel(test_df_full).astype("float32")
    test_df["bear_v2_raw"] = bear_v2_raw
    test_df["bear_v2_z"] = zscore_daily(test_df, "bear_v2_raw")

    # bull_score = z(alpha_pred)
    test_df["bull_score"] = zscore_daily(test_df, "alpha_pred")
    # Bear_C = z(|bias_60|)
    test_df["abs_bias60"] = test_df["bias_60"].abs()
    test_df["bear_C_z"] = zscore_daily(test_df, "abs_bias60")

    bull_z   = test_df["bull_score"].to_numpy("float32")
    bear_v2_z = test_df["bear_v2_z"].to_numpy("float32")
    bear_C_z = test_df["bear_C_z"].to_numpy("float32")

    meta = test_df[[DATE_COL, TICKER_COL, TARGET_RET_COL]].reset_index(drop=True)

    # 关键诊断：Bear_V2 与 |bias_60|，与 Alpha 的相关性
    corr_v2_bC = safe_pearson(bear_v2_raw.astype("float64"),
                                 test_df["abs_bias60"].to_numpy("float64"))
    corr_v2_alpha = safe_pearson(bear_v2_raw.astype("float64"), alpha_s)
    sp_v2_bC = float(spearmanr(bear_v2_raw, test_df["abs_bias60"].to_numpy("float64"))[0])
    sp_v2_alpha = float(spearmanr(bear_v2_raw, alpha_s)[0])
    print(f"\n   corr(Bear_V2 pred, |bias_60|):  Pearson={corr_v2_bC:+.4f}  "
          f"Spearman={sp_v2_bC:+.4f}  (target Pearson > 0.3)")
    print(f"   corr(Bear_V2 pred, Alpha):       Pearson={corr_v2_alpha:+.4f}  "
          f"Spearman={sp_v2_alpha:+.4f}  (target |Pearson| < 0.1)")

    # 评估表
    rows = []
    pred_cols = {}

    m_pure = evaluate_full(meta, test_df["alpha_pred"].to_numpy("float32"))
    rows.append({"config": "Trend pure (Alpha)", "alpha": np.nan,
                  "rankicir": float(m_pure["rankicir"]),
                  "sharpe":   float(m_pure["top5pct_sharpe"]),
                  "maxdd":    float(m_pure["top5pct_max_dd"])})
    pred_cols["pred_trend_pure"] = test_df["alpha_pred"].to_numpy("float32")

    # Bear_C α=0.2
    conv_C = (bull_z - P10_ALPHA * bear_C_z).astype("float32")
    m_C = evaluate_full(meta, conv_C)
    rows.append({"config": "Bear_C α=0.2 (|bias_60| rule)", "alpha": P10_ALPHA,
                  "rankicir": float(m_C["rankicir"]),
                  "sharpe":   float(m_C["top5pct_sharpe"]),
                  "maxdd":    float(m_C["top5pct_max_dd"])})
    pred_cols["pred_bear_C"] = conv_C

    # Bear_A step1 best (α=0.5) — reference only
    rows.append({"config": "Bear_A α=0.5 (G4-only, step1 ref)", "alpha": 0.5,
                  "rankicir": 0.6002, "sharpe": 1.691,
                  "maxdd": -0.3443})

    # Bear_V2 α grid
    for a in ALPHA_GRID:
        conv_v2 = (bull_z - a * bear_v2_z).astype("float32")
        m_v2 = evaluate_full(meta, conv_v2)
        rows.append({"config": f"Bear_V2 α={a:.1f} (G4+abs/sq)", "alpha": float(a),
                      "rankicir": float(m_v2["rankicir"]),
                      "sharpe":   float(m_v2["top5pct_sharpe"]),
                      "maxdd":    float(m_v2["top5pct_max_dd"])})
        pred_cols[f"pred_bear_V2_a{a}"] = conv_v2

    df_out = pd.DataFrame(rows)
    df_out["bear_C_ref"] = BEAR_C_REF
    df_out["trend_pure_ref"] = TREND_PURE_REF
    out_csv = BB_RESULTS / "step2_comparison.csv"
    df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")

    pred_df = meta.copy()
    for c, arr in pred_cols.items():
        pred_df[c] = arr
    pred_df["bull_score_z"] = bull_z
    pred_df["bear_v2_z"] = bear_v2_z
    pred_df["bear_C_z"] = bear_C_z
    pred_df["bear_v2_raw"] = bear_v2_raw
    pred_df.to_parquet(BB_RESULTS / "predictions_step2.parquet", index=False)

    # 控制台对比表
    print()
    line = "+" + "-" * 40 + "+" + "-" * 7 + "+" + "-" * 11 + "+" + "-" * 10 + "+" + "-" * 11 + "+"
    print(line)
    print(f"| {'Config':38s} | {'α':>5s} | {'RankICIR':>9s} | "
          f"{'SR':>8s} | {'MaxDD':>9s} |")
    print(line)
    for r in rows:
        a_str = "—" if not np.isfinite(r["alpha"]) else f"{r['alpha']:.1f}"
        delta = (r["rankicir"] - TREND_PURE_REF) * 10000
        print(f"| {r['config']:38s} | {a_str:>5s} | {r['rankicir']:>9.4f} | "
              f"{r['sharpe']:>+8.3f} | {r['maxdd']*100:>+8.2f}% |  Δ={delta:+.0f} bp")
    print(line)

    # =============================================================
    # 4. Task 4 — 验收
    # =============================================================
    bear_v2_rows = [r for r in rows if r["config"].startswith("Bear_V2")]
    best_v2 = max(bear_v2_rows, key=lambda r: r["rankicir"])

    print("\n=== VERDICT ===")
    print(f"  Bear_C (rule)        RankICIR = {BEAR_C_REF:.4f}")
    print(f"  Bear_V2 best (α={best_v2['alpha']:.1f}) = {best_v2['rankicir']:.4f}")
    delta_bp = (best_v2["rankicir"] - BEAR_C_REF) * 10000
    print(f"  Δ Bear_V2 - Bear_C = {delta_bp:+.1f} bp")

    # 验收条件
    check_a = best_v2["rankicir"] > BEAR_C_REF
    check_b = corr_v2_bC > 0.3
    check_c = abs(corr_v2_alpha) < 0.1
    check_d = nonlinear_in_top3
    print(f"\n  Acceptance criteria:")
    print(f"    [{'PASS' if check_a else 'FAIL'}]  Bear_V2 best > Bear_C 0.6325")
    print(f"    [{'PASS' if check_b else 'FAIL'}]  corr(Bear_V2, |bias_60|) > 0.3 "
          f"(got {corr_v2_bC:+.3f})")
    print(f"    [{'PASS' if check_c else 'FAIL'}]  |corr(Bear_V2, Alpha)| < 0.1 "
          f"(got {corr_v2_alpha:+.3f})")
    print(f"    [{'PASS' if check_d else 'FAIL'}]  abs/sq features in top-3 "
          f"(top-3 = {top3})")

    if not check_a:
        print("\n  -> Bear_V2 did NOT beat the rule.")
        print(f"     Recommend final system: Bear_C (rule) α={P10_ALPHA:.1f}, "
              f"RankICIR={BEAR_C_REF:.4f}.")
        print(f"     Paper framing: 'distance-from-equilibrium risk measure |bias_60|,")
        print(f"     cross-sectionally standardized — captures reversal risk orthogonal")
        print(f"     to the directional trend signal.'")
    else:
        print("\n  -> Bear_V2 BEATS the rule. Training Bear Agent justified.")

    print(f"\nOutputs:")
    for p_ in [out_csv,
                BB_RESULTS / "bear_feature_importance.csv",
                BB_RESULTS / "predictions_step2.parquet",
                BB_MODELS / "bear_v2_agent.cbm"]:
        if p_.exists():
            print(f"  -> {p_.relative_to(BB_RESULTS.parent.parent)}")


if __name__ == "__main__":
    main()
