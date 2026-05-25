"""Step 4 — 机制验证：对抗框架 vs 直接特征集成。

实验一：训练 M1（与 D1 相同特征集，但目标是正向 r_future_5d）
        看 G1+G3 单独的预测力
        若 M1 < 0.3 → 对抗框架真正让弱特征参与；若 M1 > 0.5 → G1+G3 本身就强
        B0 = 0.297（17 特征全局），Trend = 0.598（G4），D1 对抗 = 0.741

实验二：对比 X (additive) vs Y (subtract D1)
        X = z(Trend) + 0.5 × z(M1_pred)
        Y = z(Trend) - 0.5 × z(D1_pred)
        若 X ≈ Y (差 <100 bp) → 对抗框架本质是特征集成
        若 Y >> X (差 >200 bp) → 对抗框架有非线性增益

实验三：D1_v2（特征加入 G2）：观察特征扩展能否进一步提升

输出：bull_bear/results/mechanism_validation.csv
"""

from __future__ import annotations

import sys
from pathlib import Path
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))

import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from scipy.stats import pearsonr

from bull_bear.config_bb import (
    ALPHA_AGENT_PATH, ALPHA_GRID, BB_MODELS, BB_RESULTS,
    BEAR_FEATURES_D1, CATBOOST_PARAMS, DATE_COL, P10_ALPHA,
    TARGET_RET_COL, TEST_END, TEST_START, TICKER_COL,
    TRAIN_END, TRAIN_START, VAL_END, VAL_START,
)
from bull_bear.src.bear_agent import BearAgent
from bull_bear.src.bear_target import build_max_drawdown_5d, cross_section_zscore
from src.data import load_dataset
from bull_bear.src.metrics_utils import evaluate_full


ALPHA_FEATURES = [
    "ma60_slope", "ema180_slope", "bias_60", "bias_60_vr", "ma180_slope",
]

# D1 用的特征集（动量+强度+部分趋势）：参考 BEAR_FEATURES_D1
M1_FEATURES = list(BEAR_FEATURES_D1)   # 与 D1 完全相同

# D1_v2：D1 + parent G2 缺失项（ret_3d_minus_10d, ret_1d_minus_3d；ret_1d_minus_5d 已在 D1）
# 也补一些常用的 momentum 特征
D1_V2_FEATURES = list(BEAR_FEATURES_D1) + [
    "ret_3d_minus_10d", "ret_1d_minus_3d",
    "ret_10d", "momentum_change",
]   # 12 features


BEAR_C_REF = 0.6325
TREND_PURE_REF = 0.5979
D1_BEST_REF = 0.7408   # step3 D1 α=0.5


def zscore_daily(panel: pd.DataFrame, col: str) -> np.ndarray:
    """每日截面 z-score；NaN -> 0。"""
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


def train_catboost_regressor(df_train: pd.DataFrame, features: list[str],
                                target_col: str, name: str) -> tuple:
    """通用 CatBoost 训练（与 Alpha / Bear Agent 同配置）。"""
    X = df_train[features].astype("float32")
    medians = X.median()
    X = X.fillna(medians)
    y = df_train[target_col].astype("float32")
    # 标签裁剪到 0.1-99.9 分位
    lo, hi = y.quantile(0.001), y.quantile(0.999)
    y = y.clip(lo, hi).to_numpy()
    print(f"   training {name}: n_features={len(features)}  n_rows={len(df_train):,}  target={target_col}")
    t0 = time.time()
    m = CatBoostRegressor(**CATBOOST_PARAMS)
    m.fit(Pool(X, y), verbose=False)
    print(f"   {name} trained in {time.time()-t0:.1f}s")
    return m, medians


def main() -> None:
    print("=" * 80)
    print("Step 4 — mechanism validation: adversarial vs additive ensemble")
    print("=" * 80)

    # =============================================================
    # 0. 数据 + Alpha 预测
    # =============================================================
    print("\n[0/5] load + split + Alpha predict ...")
    df = load_dataset().dropna(subset=[TARGET_RET_COL]).reset_index(drop=True)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    alpha = CatBoostRegressor(); alpha.load_model(str(ALPHA_AGENT_PATH))
    medians_alpha = pd.read_csv(
        str(ALPHA_AGENT_PATH).replace(".cbm", "_medians.csv"),
        index_col=0).iloc[:, 0]
    df["alpha_pred"] = alpha.predict(
        df[ALPHA_FEATURES].astype("float32").fillna(medians_alpha)
    ).astype("float32")

    # 构造 max_drawdown_5d target（D1_v2 需要）
    df = build_max_drawdown_5d(df, ret_col="ret_1d", window=5)
    df = cross_section_zscore(df, "max_drawdown_5d")
    # 重要：mask 必须在 sort+reset_index 之后才创建
    mask_tr = (df[DATE_COL] >= pd.Timestamp(TRAIN_START)) & (df[DATE_COL] <= pd.Timestamp(TRAIN_END))
    mask_va = (df[DATE_COL] >= pd.Timestamp(VAL_START))  & (df[DATE_COL] <= pd.Timestamp(VAL_END))
    mask_te = (df[DATE_COL] >= pd.Timestamp(TEST_START)) & (df[DATE_COL] <= pd.Timestamp(TEST_END))
    train_df = df.loc[mask_tr].reset_index(drop=True)
    val_df   = df.loc[mask_va].reset_index(drop=True)
    test_df  = df.loc[mask_te].sort_values([DATE_COL, TICKER_COL]).reset_index(drop=True)
    print(f"   train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

    # =============================================================
    # 1. Experiment 1 — M1 全局模型 (G1+G3 → r_future_5d)
    # =============================================================
    print("\n[1/5] Experiment 1: M1 global model on G1+G3 features ...")
    m1_model, m1_medians = train_catboost_regressor(
        train_df, M1_FEATURES, TARGET_RET_COL, "M1")
    # Test 预测
    test_df["m1_pred"] = m1_model.predict(
        test_df[M1_FEATURES].astype("float32").fillna(m1_medians)
    ).astype("float32")
    meta = test_df[[DATE_COL, TICKER_COL, TARGET_RET_COL]].reset_index(drop=True)
    m_m1 = evaluate_full(meta, test_df["m1_pred"].to_numpy("float32"))
    print(f"\n   M1 G1+G3 global   RankICIR = {m_m1['rankicir']:+.4f}  "
          f"(B0=0.297, Trend=0.598)")
    if m_m1["rankicir"] < 0.30:
        m1_strength = "WEAK (D1 adversarial gain truly compensates)"
    elif m_m1["rankicir"] < 0.50:
        m1_strength = "MODEST (some standalone power; D1 lift partly intrinsic)"
    else:
        m1_strength = "STRONG (G1+G3 alone is already powerful; B0 dilution problem)"
    print(f"   -> {m1_strength}")

    # =============================================================
    # 2. Experiment 2 — 加性 X vs 减性 Y 对比
    # =============================================================
    print("\n[2/5] Experiment 2: additive X vs subtractive Y ...")
    # 加载 step3 训练的 D1 模型
    bear_d1 = BearAgent(features=BEAR_FEATURES_D1, name="bear_D1")
    bear_d1.load(BB_MODELS / "bear_D1_agent.cbm")
    test_df["d1_pred"] = bear_d1.predict_panel(test_df).astype("float32")

    # 准备 z-scores
    test_df["bull_z"] = zscore_daily(test_df, "alpha_pred")
    test_df["m1_z"]   = zscore_daily(test_df, "m1_pred")
    test_df["d1_z"]   = zscore_daily(test_df, "d1_pred")

    bull_z = test_df["bull_z"].to_numpy("float32")
    m1_z = test_df["m1_z"].to_numpy("float32")
    d1_z = test_df["d1_z"].to_numpy("float32")

    # 相关性诊断
    def safe_p(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        return float(pearsonr(a[m], b[m])[0]) if m.sum() > 100 else float("nan")
    print(f"   corr(M1_pred, Alpha)     = {safe_p(m1_z, bull_z):+.4f}")
    print(f"   corr(D1_pred, Alpha)     = {safe_p(d1_z, bull_z):+.4f}")
    print(f"   corr(M1_pred, D1_pred)   = {safe_p(m1_z, d1_z):+.4f}  "
          f"(若 negative 强相关 → M1 ≈ -D1)")

    # X: 加性  vs  Y: 减性，各跑 α=0.5 + α grid
    rows_xy = []
    rows_xy.append({"config": "Trend pure", "alpha": np.nan,
                     "rankicir": TREND_PURE_REF,
                     "sharpe": 1.694, "maxdd": -0.3420,
                     "note": "ref"})
    rows_xy.append({"config": "M1 G1+G3 global (positive target)", "alpha": np.nan,
                     "rankicir": float(m_m1["rankicir"]),
                     "sharpe":   float(m_m1["top5pct_sharpe"]),
                     "maxdd":    float(m_m1["top5pct_max_dd"]),
                     "note": "standalone M1"})
    rows_xy.append({"config": "D1 standalone (negative target)", "alpha": np.nan,
                     "rankicir": float(evaluate_full(meta, -d1_z)["rankicir"]),
                     "sharpe":   float(evaluate_full(meta, -d1_z)["top5pct_sharpe"]),
                     "maxdd":    float(evaluate_full(meta, -d1_z)["top5pct_max_dd"]),
                     "note": "−D1 alone (signed)"})

    for a in ALPHA_GRID:
        # X 加性：Trend + α × M1
        x_score = (bull_z + a * m1_z).astype("float32")
        m_x = evaluate_full(meta, x_score)
        # Y 减性：Trend − α × D1
        y_score = (bull_z - a * d1_z).astype("float32")
        m_y = evaluate_full(meta, y_score)
        rows_xy.append({"config": f"X = Trend + α·M1 (additive)",
                          "alpha": float(a),
                          "rankicir": float(m_x["rankicir"]),
                          "sharpe":   float(m_x["top5pct_sharpe"]),
                          "maxdd":    float(m_x["top5pct_max_dd"]),
                          "note": "additive ensemble"})
        rows_xy.append({"config": f"Y = Trend - α·D1 (adversarial)",
                          "alpha": float(a),
                          "rankicir": float(m_y["rankicir"]),
                          "sharpe":   float(m_y["top5pct_sharpe"]),
                          "maxdd":    float(m_y["top5pct_max_dd"]),
                          "note": "adversarial subtract"})

    # =============================================================
    # 3. Experiment 3 — D1_v2: 扩展特征集
    # =============================================================
    print("\n[3/5] Experiment 3: D1_v2 with extended features (G1+G2+G3 ext) ...")
    print(f"   D1_v2 features ({len(D1_V2_FEATURES)}): {D1_V2_FEATURES}")
    bear_v2 = BearAgent(features=D1_V2_FEATURES, name="bear_D1_v2")
    bear_v2.train(train_df, target_col="max_drawdown_5d_z",
                   df_val=val_df, save=True)
    val_sp = bear_v2.validate_spearman(val_df, target_col="max_drawdown_5d")
    print(f"   D1_v2 val Spearman(pred, max_drawdown_5d) = {val_sp:+.4f}")
    test_df["d1_v2_pred"] = bear_v2.predict_panel(test_df).astype("float32")
    test_df["d1_v2_z"] = zscore_daily(test_df, "d1_v2_pred")
    d1_v2_z = test_df["d1_v2_z"].to_numpy("float32")
    corr_v2_alpha = safe_p(d1_v2_z, bull_z)
    print(f"   corr(D1_v2, Alpha) = {corr_v2_alpha:+.4f}")

    rows_v2 = []
    for a in ALPHA_GRID:
        conv = (bull_z - a * d1_v2_z).astype("float32")
        m = evaluate_full(meta, conv)
        rows_v2.append({"config": f"D1_v2 α={a:.1f} (G1+G2+G3 ext)",
                         "alpha": float(a),
                         "rankicir": float(m["rankicir"]),
                         "sharpe":   float(m["top5pct_sharpe"]),
                         "maxdd":    float(m["top5pct_max_dd"]),
                         "note": "D1 + G2 feature extension"})

    # =============================================================
    # 4. 汇总
    # =============================================================
    print("\n[4/5] consolidate + write CSV ...")
    rows_all = rows_xy + rows_v2
    df_out = pd.DataFrame(rows_all)
    df_out["M1_standalone"] = float(m_m1["rankicir"])
    df_out["bear_C_ref"] = BEAR_C_REF
    df_out["d1_best_ref"] = D1_BEST_REF
    out_csv = BB_RESULTS / "mechanism_validation.csv"
    df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # =============================================================
    # 5. 打印 + 总结
    # =============================================================
    print("\n[5/5] full comparison table ...")
    line = "+" + "-" * 42 + "+" + "-" * 7 + "+" + "-" * 11 + "+" + "-" * 10 + "+" + "-" * 11 + "+"
    print(line)
    print(f"| {'Config':40s} | {'α':>5s} | {'RankICIR':>9s} | "
          f"{'SR':>8s} | {'MaxDD':>9s} |")
    print(line)
    for r in rows_all:
        a_str = "—" if not np.isfinite(r["alpha"]) else f"{r['alpha']:.1f}"
        print(f"| {r['config']:40s} | {a_str:>5s} | "
              f"{r['rankicir']:>9.4f} | {r['sharpe']:>+8.3f} | "
              f"{r['maxdd']*100:>+8.2f}% |")
    print(line)

    # X vs Y 配对差距
    print("\n=== X (additive) vs Y (adversarial) per α ===")
    for a in ALPHA_GRID:
        x = next(r for r in rows_xy if r["config"].startswith("X = ") and r["alpha"] == a)
        y = next(r for r in rows_xy if r["config"].startswith("Y = ") and r["alpha"] == a)
        delta = (y["rankicir"] - x["rankicir"]) * 10000
        print(f"  α={a:.1f}:  X={x['rankicir']:.4f}  Y={y['rankicir']:.4f}  "
              f"Δ(Y-X) = {delta:+.1f} bp")

    # D1 vs D1_v2 best
    best_v2 = max(rows_v2, key=lambda r: r["rankicir"])
    print(f"\n=== D1_v2 best ===")
    print(f"  α={best_v2['alpha']:.1f}  RankICIR={best_v2['rankicir']:.4f}  "
          f"Δ vs D1 best ({D1_BEST_REF:.4f}) = {(best_v2['rankicir']-D1_BEST_REF)*10000:+.1f} bp")

    # 全局最优
    best_all = max(rows_all, key=lambda r: r["rankicir"])
    print(f"\n=== Overall best across all configs ===")
    print(f"  {best_all['config']}  α={best_all['alpha']}  "
          f"RankICIR={best_all['rankicir']:.4f}  "
          f"SR={best_all['sharpe']:+.3f}  MaxDD={best_all['maxdd']*100:+.2f}%")

    # 总结结论
    print("\n=== MECHANISM VERDICT ===")
    print(f"  M1 G1+G3 standalone RankICIR = {m_m1['rankicir']:+.4f}")
    print(f"    (B0 17-feat global = 0.297, Trend G4 only = 0.598, D1 adversarial = 0.741)")
    if m_m1["rankicir"] < 0.30:
        print(f"  -> G1+G3 alone is WEAK (<0.30). "
              f"Adversarial subtraction is genuinely making weak features useful.")
    elif m_m1["rankicir"] > 0.50:
        print(f"  -> G1+G3 alone is STRONG (>0.50). "
              f"B0 (0.297) reveals feature dilution problem in full-set training.")
    else:
        print(f"  -> G1+G3 alone is MODEST. Adversarial framework adds some non-linear value.")

    # X vs Y 最大 |gap|
    max_gap = 0.0
    best_pair = None
    for a in ALPHA_GRID:
        x = next(r for r in rows_xy if r["config"].startswith("X = ") and r["alpha"] == a)
        y = next(r for r in rows_xy if r["config"].startswith("Y = ") and r["alpha"] == a)
        gap = abs(y["rankicir"] - x["rankicir"]) * 10000
        if gap > max_gap:
            max_gap = gap
            best_pair = (a, x, y)
    a, x, y = best_pair
    delta = (y["rankicir"] - x["rankicir"]) * 10000
    print(f"\n  Max |X-Y| at α={a:.1f}:  Y={y['rankicir']:.4f}  X={x['rankicir']:.4f}  "
          f"Δ={delta:+.1f} bp")
    if max_gap < 100:
        print(f"  -> X ≈ Y across alphas (gap <100 bp).  Adversarial framework is "
              f"ESSENTIALLY a re-encoded additive ensemble.")
    elif delta > 200:
        print(f"  -> Y >> X (>200 bp).  Adversarial framework adds non-linear value "
              f"beyond simple additive ensembling.")
    else:
        print(f"  -> Mixed: gap between {max_gap:.0f} bp. Adversarial has marginal extra value.")

    print(f"\nOutputs:")
    for p in [out_csv, BB_MODELS / "bear_D1_v2_agent.cbm"]:
        if p.exists():
            print(f"  -> {p.relative_to(BB_RESULTS.parent.parent)}")


if __name__ == "__main__":
    main()
