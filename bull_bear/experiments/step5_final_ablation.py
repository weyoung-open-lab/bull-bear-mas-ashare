"""Step 5 — 最终系统集成 + 完整消融表 + 6 年跨窗对比。

主窗口 (2023-2026) 消融：
  B0  全局 CatBoost                  0.297 (ref)
  M1  G1+G3 全局（加法失败参照）       0.256 (from step4)
  Trend pure (Alpha)                  0.598
  Bear_C (|bias_60| 规则)              0.633
  D1 α=0.2                             0.692 (from step3)
  D1 α=0.5 ★核心                       0.741
  D1 + 自适应 α(regime)
  D1 + Agent 5 异常安全阀（异常日改回 Alpha）
  D1 + Vol-gate（高 vol(t-1) 日改回 Alpha）  ← 最终系统

跨窗口逐年（2020-2025）：
  Window B (2020-2022): 跨周期重训 Bear D1 (2016-2018 train)
  Window A (2023-2025): 主模型
  对每年算 Trend pure vs D1 α=0.5 的 RankICIR

输出：
  bull_bear/results/final_ablation.csv
  bull_bear/results/yearly_comparison.csv
  bull_bear/results/models/bear_D1_agent_B.cbm
"""

from __future__ import annotations

import sys
from pathlib import Path
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))

import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from bull_bear.config_bb import (
    ALPHA_AGENT_PATH, BB_MODELS, BB_RESULTS,
    BEAR_FEATURES_D1, DATE_COL,
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

# Window B (跨周期)
WIN_B_TRAIN_START = "2016-01-01"
WIN_B_TRAIN_END   = "2018-12-31"
WIN_B_TEST_START  = "2020-01-01"
WIN_B_TEST_END    = "2022-12-31"

ALPHA_AGENT_B_PATH = (Path(__file__).resolve().parents[2]
                       / "strategy_debate/results/models/cross_period/trend_agent_B.cbm")

# 自适应 α 映射
ALPHA_BY_REGIME = {"bear": 0.65, "sideway": 0.50, "bull": 0.35}

D1_ALPHA = 0.5


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


def yearly_rankicir(meta: pd.DataFrame, pred: np.ndarray) -> dict[int, dict]:
    """对每个日历年子集计算 RankICIR + SR + MaxDD。"""
    out: dict[int, dict] = {}
    years = meta[DATE_COL].dt.year
    for y in sorted(years.unique()):
        mask = (years == y).to_numpy()
        if mask.sum() < 100:
            continue
        sub_meta = meta[mask].reset_index(drop=True)
        sub_pred = pred[mask]
        m = evaluate_full(sub_meta, sub_pred)
        out[int(y)] = {"rankicir": float(m["rankicir"]),
                        "sharpe":   float(m["top5pct_sharpe"]),
                        "maxdd":    float(m["top5pct_max_dd"]),
                        "n_rows":   int(mask.sum())}
    return out


def main() -> None:
    print("=" * 80)
    print("Step 5 — final system ablation + 6-year cross-window comparison")
    print("=" * 80)

    # =============================================================
    # 0. 加载 + targets + masks
    # =============================================================
    print("\n[0/6] load + build targets + masks ...")
    df = load_dataset().dropna(subset=[TARGET_RET_COL]).reset_index(drop=True)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = build_max_drawdown_5d(df, ret_col="ret_1d", window=5)
    df = cross_section_zscore(df, "max_drawdown_5d")
    # masks AFTER target build
    mask_tr_A = (df[DATE_COL] >= pd.Timestamp(TRAIN_START)) & (df[DATE_COL] <= pd.Timestamp(TRAIN_END))
    mask_va_A = (df[DATE_COL] >= pd.Timestamp(VAL_START))  & (df[DATE_COL] <= pd.Timestamp(VAL_END))
    mask_te_A = (df[DATE_COL] >= pd.Timestamp(TEST_START)) & (df[DATE_COL] <= pd.Timestamp(TEST_END))
    mask_tr_B = (df[DATE_COL] >= pd.Timestamp(WIN_B_TRAIN_START)) & (df[DATE_COL] <= pd.Timestamp(WIN_B_TRAIN_END))
    mask_te_B = (df[DATE_COL] >= pd.Timestamp(WIN_B_TEST_START))  & (df[DATE_COL] <= pd.Timestamp(WIN_B_TEST_END))

    train_A = df.loc[mask_tr_A].reset_index(drop=True)
    train_B = df.loc[mask_tr_B].reset_index(drop=True)
    test_A  = df.loc[mask_te_A].sort_values([DATE_COL, TICKER_COL]).reset_index(drop=True)
    test_B  = df.loc[mask_te_B].sort_values([DATE_COL, TICKER_COL]).reset_index(drop=True)
    print(f"   Window A: train={len(train_A):,}  test={len(test_A):,}  days={test_A[DATE_COL].nunique()}")
    print(f"   Window B: train={len(train_B):,}  test={len(test_B):,}  days={test_B[DATE_COL].nunique()}")

    # =============================================================
    # 1. 加载 Alpha 模型 (Window A + Window B)
    # =============================================================
    print("\n[1/6] load Alpha agents (both windows) + Bear D1 agent (Window A) ...")
    alpha_A = CatBoostRegressor(); alpha_A.load_model(str(ALPHA_AGENT_PATH))
    med_A = pd.read_csv(
        str(ALPHA_AGENT_PATH).replace(".cbm", "_medians.csv"),
        index_col=0).iloc[:, 0]
    alpha_B = CatBoostRegressor(); alpha_B.load_model(str(ALPHA_AGENT_B_PATH))
    med_B = pd.read_csv(
        str(ALPHA_AGENT_B_PATH).replace(".cbm", "_medians.csv"),
        index_col=0).iloc[:, 0]

    bear_A = BearAgent(features=BEAR_FEATURES_D1, name="bear_D1")
    bear_A.load(BB_MODELS / "bear_D1_agent.cbm")

    # =============================================================
    # 2. 重训 Bear D1 (Window B: 2016-2018)
    # =============================================================
    print("\n[2/6] retrain Bear D1 on Window B (2016-2018) ...")
    bear_B = BearAgent(features=BEAR_FEATURES_D1, name="bear_D1_B")
    t0 = time.time()
    bear_B.train(train_B, target_col="max_drawdown_5d_z",
                  df_val=None, save=False)
    bear_B.save(BB_MODELS / "bear_D1_agent_B.cbm")
    print(f"   trained in {time.time()-t0:.1f}s")

    # =============================================================
    # 3. 主窗口预测 + 消融表
    # =============================================================
    print("\n[3/6] main window (2023-2026) predictions + ablation ...")
    test_A["alpha_pred"] = alpha_A.predict(
        test_A[ALPHA_FEATURES].astype("float32").fillna(med_A)
    ).astype("float32")
    test_A["d1_pred"] = bear_A.predict_panel(test_A).astype("float32")

    test_A["bull_z"] = zscore_daily(test_A, "alpha_pred")
    test_A["d1_z"]   = zscore_daily(test_A, "d1_pred")
    bull_z = test_A["bull_z"].to_numpy("float32")
    d1_z   = test_A["d1_z"].to_numpy("float32")
    meta_A = test_A[[DATE_COL, TICKER_COL, TARGET_RET_COL]].reset_index(drop=True)

    # Vol-gate 数据
    train_daily_vol = train_A.groupby(DATE_COL)["vol20"].mean()
    tau = float(train_daily_vol.quantile(0.75))
    all_daily_vol = df.groupby(DATE_COL)["vol20"].mean().sort_index()
    all_daily_vol = all_daily_vol[~all_daily_vol.index.duplicated()]
    prev_day_vol = all_daily_vol.shift(1)
    vol_gate_high = test_A[DATE_COL].map(lambda d: prev_day_vol.get(d, np.nan) > tau).to_numpy()
    vol_gate_high = np.where(np.isfinite(vol_gate_high.astype("float64")), vol_gate_high, False).astype(bool)
    print(f"   Vol-gate threshold (train p75) = {tau:.5f}  triggers "
          f"{int(vol_gate_high.sum()/len(test_A)*test_A[DATE_COL].nunique())} days approx "
          f"(~{vol_gate_high.sum()/len(test_A)*100:.1f}% rows)")

    # Anomaly flag（来自 strategy_debate predictions_R6.parquet）
    r6 = pd.read_parquet(
        Path(__file__).resolve().parents[2]
        / "strategy_debate/results/predictions_R6.parquet"
    )
    r6[DATE_COL] = pd.to_datetime(r6[DATE_COL])
    anomaly_dates = set(r6.loc[r6["anomaly"], DATE_COL].dt.normalize().unique())
    is_anomaly = test_A[DATE_COL].dt.normalize().isin(anomaly_dates).to_numpy(dtype=bool)
    print(f"   Anomaly days: {len(anomaly_dates)}  "
          f"covering {int(is_anomaly.sum()/len(test_A)*test_A[DATE_COL].nunique())} test days approx")

    # 自适应 α
    regime_daily = df.groupby(DATE_COL)["macro_regime_3"].first()
    regime_map = test_A[DATE_COL].map(regime_daily).to_numpy()
    adaptive_alpha = np.array([ALPHA_BY_REGIME.get(r, 0.50) for r in regime_map], dtype="float32")
    print(f"   regime distribution on test:")
    for r in ("bear", "sideway", "bull"):
        cnt = int((regime_map == r).sum())
        print(f"     {r:8s}  {cnt:>8,} rows  α={ALPHA_BY_REGIME[r]}")

    # ===== 构造各 conviction =====
    def evaluate(arr, label):
        m = evaluate_full(meta_A, arr.astype("float32"))
        return {"config": label,
                "rankicir": float(m["rankicir"]),
                "sharpe":   float(m["top5pct_sharpe"]),
                "maxdd":    float(m["top5pct_max_dd"])}

    rows = []
    # baselines (已知 references)
    rows.append({"config": "B0 Global CatBoost (17 feat)",
                  "rankicir": 0.2974, "sharpe": 0.92, "maxdd": -0.4284})
    rows.append({"config": "M1 G1+G3 global (additive ref)",
                  "rankicir": 0.2560, "sharpe": 0.746, "maxdd": -0.4224})
    rows.append(evaluate(test_A["alpha_pred"].to_numpy("float32"),
                          "Trend pure (Alpha)"))

    # Bear_C 规则
    test_A["abs_bias60"] = test_A["bias_60"].abs()
    bear_C_z = zscore_daily(test_A, "abs_bias60")
    rows.append(evaluate(bull_z - 0.2 * bear_C_z,
                          "Bear_C α=0.2 (|bias_60| rule)"))

    # D1 α=0.2 / α=0.5
    rows.append(evaluate(bull_z - 0.2 * d1_z, "D1 α=0.2 (adversarial)"))
    rows.append(evaluate(bull_z - 0.5 * d1_z, "D1 α=0.5 (* CORE)"))

    # + 自适应 α (regime-conditioned)
    rows.append(evaluate(bull_z - adaptive_alpha * d1_z,
                          "+ adaptive α (regime)"))

    # + Agent 5 异常安全阀（异常日改用 alpha pure）
    conv_d1_core = bull_z - 0.5 * d1_z
    conv_anom = np.where(is_anomaly, test_A["alpha_pred"].to_numpy("float32"),
                          conv_d1_core).astype("float32")
    rows.append(evaluate(conv_anom, "+ Agent 5 anomaly valve"))

    # + Vol-gate (final system)
    conv_gate = np.where(vol_gate_high, test_A["alpha_pred"].to_numpy("float32"),
                          conv_d1_core).astype("float32")
    rows.append(evaluate(conv_gate, "+ Vol-gate (high vol -> Alpha)"))

    # 全合并：自适应 α + anomaly + vol-gate
    conv_adapt = bull_z - adaptive_alpha * d1_z
    conv_full = np.where(
        is_anomaly | vol_gate_high,
        test_A["alpha_pred"].to_numpy("float32"),
        conv_adapt,
    ).astype("float32")
    rows.append(evaluate(conv_full, "FINAL: adaptive α + anomaly + vol-gate"))

    df_ab = pd.DataFrame(rows)
    out_ab = BB_RESULTS / "final_ablation.csv"
    df_ab.to_csv(out_ab, index=False, encoding="utf-8-sig")

    # ----- print main ablation -----
    print()
    line = "+" + "-" * 44 + "+" + "-" * 11 + "+" + "-" * 10 + "+" + "-" * 11 + "+" + "-" * 14 + "+"
    print(line)
    print(f"| {'Config':42s} | {'RankICIR':>9s} | {'SR':>8s} | "
          f"{'MaxDD':>9s} | {'Δ vs Trend':>12s} |")
    print(line)
    ric_trend = 0.5979
    for r in rows:
        delta = (r["rankicir"] - ric_trend) * 10000
        print(f"| {r['config']:42s} | {r['rankicir']:>9.4f} | "
              f"{r['sharpe']:>+8.3f} | {r['maxdd']*100:>+8.2f}% | "
              f"{delta:>+9.0f} bp |")
    print(line)

    # =============================================================
    # 4. Window B (2020-2022) 预测
    # =============================================================
    print("\n[4/6] Window B (2020-2022) predictions ...")
    test_B["alpha_pred"] = alpha_B.predict(
        test_B[ALPHA_FEATURES].astype("float32").fillna(med_B)
    ).astype("float32")
    test_B["d1_pred"] = bear_B.predict_panel(test_B).astype("float32")
    test_B["bull_z"] = zscore_daily(test_B, "alpha_pred")
    test_B["d1_z"]   = zscore_daily(test_B, "d1_pred")
    meta_B = test_B[[DATE_COL, TICKER_COL, TARGET_RET_COL]].reset_index(drop=True)
    bull_B = test_B["bull_z"].to_numpy("float32")
    d1_B   = test_B["d1_z"].to_numpy("float32")
    conv_B_core = bull_B - 0.5 * d1_B

    # =============================================================
    # 5. 6 年 yearly comparison
    # =============================================================
    print("\n[5/6] 6-year yearly comparison (Trend pure vs D1 α=0.5) ...")
    yearly_rows = []
    # Window A: 2023/2024/2025
    yA_trend = yearly_rankicir(meta_A, test_A["alpha_pred"].to_numpy("float32"))
    yA_d1    = yearly_rankicir(meta_A, conv_d1_core)
    # Window B: 2020/2021/2022
    yB_trend = yearly_rankicir(meta_B, test_B["alpha_pred"].to_numpy("float32"))
    yB_d1    = yearly_rankicir(meta_B, conv_B_core)

    for y in (2020, 2021, 2022):
        t = yB_trend.get(y, {})
        d = yB_d1.get(y, {})
        if not t or not d:
            continue
        yearly_rows.append({
            "year": y, "window": "B (2020-2022)",
            "trend_rankicir": t["rankicir"],
            "trend_sharpe":   t["sharpe"],
            "trend_maxdd":    t["maxdd"],
            "d1_rankicir": d["rankicir"],
            "d1_sharpe":   d["sharpe"],
            "d1_maxdd":    d["maxdd"],
            "delta_ric":   d["rankicir"] - t["rankicir"],
            "d1_wins":     d["rankicir"] > t["rankicir"],
        })
    for y in (2023, 2024, 2025):
        t = yA_trend.get(y, {})
        d = yA_d1.get(y, {})
        if not t or not d:
            continue
        yearly_rows.append({
            "year": y, "window": "A (2023-2025)",
            "trend_rankicir": t["rankicir"],
            "trend_sharpe":   t["sharpe"],
            "trend_maxdd":    t["maxdd"],
            "d1_rankicir": d["rankicir"],
            "d1_sharpe":   d["sharpe"],
            "d1_maxdd":    d["maxdd"],
            "delta_ric":   d["rankicir"] - t["rankicir"],
            "d1_wins":     d["rankicir"] > t["rankicir"],
        })

    df_yr = pd.DataFrame(yearly_rows)
    out_yr = BB_RESULTS / "yearly_comparison.csv"
    df_yr.to_csv(out_yr, index=False, encoding="utf-8-sig")

    # print yearly table
    print()
    print(f"  {'year':>5s} | {'window':>15s} | {'Trend RIC':>10s}  {'D1 RIC':>10s}  "
          f"{'Δ':>10s}  {'D1 SR':>8s}  {'D1 MaxDD':>9s}  {'D1 wins':>9s}")
    print("  " + "-" * 92)
    n_wins = 0
    n_years = 0
    for r in yearly_rows:
        win_marker = "YES" if r["d1_wins"] else "no"
        if r["d1_wins"]:
            n_wins += 1
        n_years += 1
        print(f"  {r['year']:>5d} | {r['window']:>15s} | "
              f"{r['trend_rankicir']:>+10.4f}  {r['d1_rankicir']:>+10.4f}  "
              f"{r['delta_ric']:>+10.4f}  "
              f"{r['d1_sharpe']:>+8.3f}  {r['d1_maxdd']*100:>+8.2f}%  "
              f"{win_marker:>9s}")
    print("  " + "-" * 92)
    print(f"\n  D1 wins {n_wins}/{n_years} years")

    # =============================================================
    # 6. 输出
    # =============================================================
    print("\n[6/6] outputs ...")
    for p in [out_ab, out_yr, BB_MODELS / "bear_D1_agent_B.cbm"]:
        if p.exists():
            print(f"  -> {p.relative_to(BB_RESULTS.parent.parent)}")


if __name__ == "__main__":
    main()
