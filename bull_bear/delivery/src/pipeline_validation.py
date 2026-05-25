"""Step 6 — 审稿人级最终验证：四个并行任务。

Task 1: Walk-Forward 滚动重训（消除 2020/2021 已在 train 的问题）
  W1: train 2016-2018  -> test 2019
  W2: train 2016-2019  -> test 2020
  W3: train 2016-2020  -> test 2021
  W4: train 2016-2021  -> test 2022 (α=0.5 selection year, val)
  W5: train 2016-2021  -> test 2023-2025 (canonical hold-out)
  对每个窗口训练独立 Alpha + Bear，年度 RankICIR 对比

Task 2: Bootstrap N=1000 显著性检验（D1 vs Trend pure，测试集 2023-2025）
  按天 with-replacement，分别算 D1 / Trend RankICIR，求 delta
  p-value = P(delta <= 0)

Task 3: Bear D1 Quintile 分析（验证 Bear 不是 -Alpha）
  按 bear_score 分 5 组，统计每组 avg max_drawdown_5d / avg r_future_5

Task 4: 历史 MaxDD 60d 规则 Bear 基线
  historical_maxdd_60d = rolling 60-day max(|min(0, ret_1d)|)
  对比 Trained Bear D1 vs Historical 规则

输出：
  bull_bear/results/final/rolling_walkforward.csv
  bull_bear/results/final/bootstrap_test.csv
  bull_bear/results/final/bear_quintile_analysis.csv
  bull_bear/results/final/simple_baseline_comparison.csv
  bull_bear/results/models/walkforward/{alpha,bear}_{W1,W2,W3}.cbm
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from scipy.stats import spearmanr

from bull_bear.config_bb import (
    ALPHA_AGENT_PATH, BB_MODELS, BB_RESULTS,
    BEAR_FEATURES_D1, CATBOOST_PARAMS, DATE_COL,
    TARGET_RET_COL, TICKER_COL,
)
from bull_bear.src.bear_agent import BearAgent
from bull_bear.src.bear_target import build_max_drawdown_5d, cross_section_zscore
from src.data import load_dataset
from bull_bear.src.metrics_utils import evaluate_full


ALPHA_FEATURES = [
    "ma60_slope", "ema180_slope", "bias_60", "bias_60_vr", "ma180_slope",
]
D1_ALPHA = 0.5

WALKFORWARD_DIR = BB_MODELS / "walkforward"
FINAL_DIR = BB_RESULTS / "final"
FINAL_DIR.mkdir(parents=True, exist_ok=True)
WALKFORWARD_DIR.mkdir(parents=True, exist_ok=True)


def zscore_daily(panel: pd.DataFrame, col: str) -> np.ndarray:
    """每日截面 z-score; NaN -> 0."""
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


def train_alpha(df_train: pd.DataFrame, name: str,
                  save_path: Path) -> tuple[CatBoostRegressor, pd.Series]:
    """训练 Alpha (Trend G4)，固定 300 轮。"""
    X = df_train[ALPHA_FEATURES].astype("float32")
    medians = X.median()
    X = X.fillna(medians)
    y_raw = df_train[TARGET_RET_COL].astype("float32")
    lo, hi = y_raw.quantile(0.001), y_raw.quantile(0.999)
    y = y_raw.clip(lo, hi).to_numpy()
    m = CatBoostRegressor(**CATBOOST_PARAMS)
    m.fit(Pool(X, y), verbose=False)
    m.save_model(str(save_path))
    medians.to_csv(str(save_path).replace(".cbm", "_medians.csv"))
    print(f"   [Alpha-{name}]  saved -> {save_path.name}")
    return m, medians


def train_bear(df_train: pd.DataFrame, name: str,
                save_path: Path) -> BearAgent:
    """训练 Bear D1 (G1+G3, max_drawdown_5d_z)."""
    bear = BearAgent(features=BEAR_FEATURES_D1, name=f"bear_{name}")
    bear.train(df_train, target_col="max_drawdown_5d_z", df_val=None, save=False)
    bear.save(save_path)
    return bear


def yearly_rankicir(meta: pd.DataFrame, pred: np.ndarray) -> dict[int, float]:
    """对每个日历年子集计算 RankICIR."""
    out: dict[int, float] = {}
    years = meta[DATE_COL].dt.year
    for y in sorted(years.unique()):
        mask = (years == y).to_numpy()
        if mask.sum() < 100:
            continue
        sub_meta = meta[mask].reset_index(drop=True)
        sub_pred = pred[mask]
        m = evaluate_full(sub_meta, sub_pred)
        out[int(y)] = float(m["rankicir"])
    return out


def daily_ic(meta: pd.DataFrame, pred: np.ndarray) -> dict[pd.Timestamp, float]:
    """Per-day Spearman IC."""
    df = meta.copy()
    df["pred"] = pred
    out = {}
    for d, g in df.groupby(DATE_COL):
        if len(g) < 5:
            continue
        x = g["pred"].to_numpy(dtype="float64")
        y = g[TARGET_RET_COL].to_numpy(dtype="float64")
        if np.std(x) == 0 or np.std(y) == 0:
            continue
        rho, _ = spearmanr(x, y)
        out[pd.Timestamp(d)] = float(rho)
    return out


def predict_test(window_name: str, alpha_model, alpha_medians,
                   bear_model: BearAgent, test_df: pd.DataFrame) -> tuple:
    """对 test_df 给出 alpha_pred / bull_z / d1_pred / d1_z / conviction."""
    X_a = test_df[ALPHA_FEATURES].astype("float32").fillna(alpha_medians)
    alpha_pred = alpha_model.predict(X_a).astype("float32")
    bear_pred = bear_model.predict_panel(test_df).astype("float32")
    df = test_df.copy().reset_index(drop=True)
    df["alpha_pred"] = alpha_pred
    df["bear_pred"] = bear_pred
    df["bull_z"] = zscore_daily(df, "alpha_pred")
    df["bear_z"] = zscore_daily(df, "bear_pred")
    conviction = df["bull_z"].to_numpy("float32") - D1_ALPHA * df["bear_z"].to_numpy("float32")
    return df, alpha_pred, bear_pred, conviction


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    print("=" * 80)
    print("Step 6 — final reviewer-grade validation: 4 parallel tasks")
    print("=" * 80)

    # ---- 数据加载 + 目标 ----
    print("\n[0/5] load + build max_drawdown_5d target ...")
    t0 = time.time()
    df = load_dataset().dropna(subset=[TARGET_RET_COL]).reset_index(drop=True)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = build_max_drawdown_5d(df, ret_col="ret_1d", window=5)
    df = cross_section_zscore(df, "max_drawdown_5d")
    print(f"   panel: {len(df):,} rows, dates {df[DATE_COL].min().date()}-{df[DATE_COL].max().date()}")
    print(f"   loaded + targets in {time.time()-t0:.1f}s")

    # ---- 5 个滚动窗口定义 ----
    windows = [
        ("W1", "2016-01-01", "2018-12-31", "2019-01-01", "2019-12-31"),
        ("W2", "2016-01-01", "2019-12-31", "2020-01-01", "2020-12-31"),
        ("W3", "2016-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
        ("W4", "2016-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
        ("W5", "2016-01-01", "2021-12-31", "2023-01-01", "2026-01-31"),
    ]

    # =================================================================
    # Task 1: Walk-Forward 训练 + 评估
    # =================================================================
    print("\n[1/5] TASK 1 — Walk-Forward 5-window evaluation ...")
    walkforward_rows = []
    cached_predictions = {}    # cache for tasks 2-4

    for name, tr_s, tr_e, te_s, te_e in windows:
        print(f"\n   === Window {name}: train {tr_s[:4]}-{tr_e[:4]}  test {te_s[:7]}-{te_e[:7]} ===")
        m_tr = (df[DATE_COL] >= pd.Timestamp(tr_s)) & (df[DATE_COL] <= pd.Timestamp(tr_e))
        m_te = (df[DATE_COL] >= pd.Timestamp(te_s)) & (df[DATE_COL] <= pd.Timestamp(te_e))
        train_df = df.loc[m_tr].reset_index(drop=True)
        test_df  = df.loc[m_te].sort_values([DATE_COL, TICKER_COL]).reset_index(drop=True)
        print(f"     train rows={len(train_df):,}  test rows={len(test_df):,}  "
              f"test days={test_df[DATE_COL].nunique()}")

        # 加载或训练 Alpha
        alpha_path = WALKFORWARD_DIR / f"alpha_{name}.cbm"
        bear_path  = WALKFORWARD_DIR / f"bear_{name}.cbm"
        if name == "W5":
            # 复用既有
            alpha = CatBoostRegressor(); alpha.load_model(str(ALPHA_AGENT_PATH))
            alpha_medians = pd.read_csv(
                str(ALPHA_AGENT_PATH).replace(".cbm", "_medians.csv"),
                index_col=0).iloc[:, 0]
            bear = BearAgent(features=BEAR_FEATURES_D1, name="bear_D1")
            bear.load(BB_MODELS / "bear_D1_agent.cbm")
            print(f"     [reuse] existing W5 models")
        elif name == "W4":
            # W4 训练数据与 W5 相同（2016-2021），可复用 W5 模型
            alpha = CatBoostRegressor(); alpha.load_model(str(ALPHA_AGENT_PATH))
            alpha_medians = pd.read_csv(
                str(ALPHA_AGENT_PATH).replace(".cbm", "_medians.csv"),
                index_col=0).iloc[:, 0]
            bear = BearAgent(features=BEAR_FEATURES_D1, name="bear_D1")
            bear.load(BB_MODELS / "bear_D1_agent.cbm")
            print(f"     [reuse] W5 models (same train set 2016-2021)")
        else:
            # W1 / W2 / W3 需要新训
            t_train = time.time()
            alpha, alpha_medians = train_alpha(train_df, name, alpha_path)
            bear = train_bear(train_df, name, bear_path)
            print(f"     trained Alpha + Bear in {time.time()-t_train:.1f}s")

        # Predict test
        df_pred, alpha_pred, bear_pred, conviction = predict_test(
            name, alpha, alpha_medians, bear, test_df)
        meta = df_pred[[DATE_COL, TICKER_COL, TARGET_RET_COL]].reset_index(drop=True)

        # 年度 RankICIR
        yr_trend = yearly_rankicir(meta, alpha_pred)
        yr_d1    = yearly_rankicir(meta, conviction)

        for y in sorted(yr_trend.keys()):
            t_ric = yr_trend[y]
            d_ric = yr_d1.get(y, float("nan"))
            walkforward_rows.append({
                "year": y,
                "window": name,
                "train_range": f"{tr_s[:4]}-{tr_e[:4]}",
                "trend_rankicir": t_ric,
                "d1_rankicir": d_ric,
                "delta": d_ric - t_ric,
                "d1_wins": (d_ric > t_ric),
            })
            print(f"     {y}:  Trend={t_ric:+.4f}  D1={d_ric:+.4f}  "
                  f"Δ={d_ric-t_ric:+.4f}  {'WIN' if d_ric > t_ric else 'lose'}")

        # 缓存 W5 预测供 task 2/3 用
        if name == "W5":
            cached_predictions["W5"] = {
                "df": df_pred,
                "alpha_pred": alpha_pred,
                "bear_pred": bear_pred,
                "conviction": conviction,
                "meta": meta,
            }

    df_wf = pd.DataFrame(walkforward_rows)
    out_wf = FINAL_DIR / "rolling_walkforward.csv"
    df_wf.to_csv(out_wf, index=False, encoding="utf-8-sig")

    print("\n   === Walk-Forward summary table ===")
    print(f"   {'Year':>5s} | {'Window':>6s} | {'Train':>10s} | "
          f"{'Trend RIC':>10s} | {'D1 RIC':>10s} | {'Δ':>10s} | {'Win':>5s}")
    print("   " + "-" * 70)
    n_wins = 0
    n_total = 0
    for r in walkforward_rows:
        win_str = "YES" if r["d1_wins"] else "no"
        if r["d1_wins"]: n_wins += 1
        n_total += 1
        print(f"   {r['year']:>5d} | {r['window']:>6s} | {r['train_range']:>10s} | "
              f"{r['trend_rankicir']:>+10.4f} | {r['d1_rankicir']:>+10.4f} | "
              f"{r['delta']:>+10.4f} | {win_str:>5s}")
    print(f"\n   D1 wins {n_wins}/{n_total} walk-forward year-evaluations")

    # =================================================================
    # Task 2: Bootstrap 显著性检验
    # =================================================================
    print("\n[2/5] TASK 2 — Bootstrap significance test (N=1000) on 2023-2025 ...")
    w5 = cached_predictions["W5"]
    meta_w5 = w5["meta"]
    trend_pred_w5 = w5["alpha_pred"]
    conv_w5 = w5["conviction"]

    # 限定到 2023-2025（W5 的 test 范围其实是 2023-01-01..2026-01-31）
    # bootstrap 按 days 抽样
    daily_ic_trend = daily_ic(meta_w5, trend_pred_w5)
    daily_ic_d1    = daily_ic(meta_w5, conv_w5)
    common_days = sorted(set(daily_ic_trend) & set(daily_ic_d1))
    print(f"   {len(common_days)} common days for bootstrap")

    ic_t_arr = np.array([daily_ic_trend[d] for d in common_days])
    ic_d_arr = np.array([daily_ic_d1[d]    for d in common_days])

    def rankicir_from_ic(ic_arr):
        return float(np.mean(ic_arr) / (np.std(ic_arr, ddof=0) + 1e-9))

    observed_trend_ric = rankicir_from_ic(ic_t_arr)
    observed_d1_ric    = rankicir_from_ic(ic_d_arr)
    observed_delta = observed_d1_ric - observed_trend_ric
    print(f"   observed Trend RankICIR = {observed_trend_ric:+.4f}")
    print(f"   observed D1    RankICIR = {observed_d1_ric:+.4f}")
    print(f"   observed Δ              = {observed_delta:+.4f}")

    rng = np.random.default_rng(42)
    n_boot = 1000
    boot_deltas = np.zeros(n_boot, dtype="float64")
    n_days = len(common_days)
    for i in range(n_boot):
        idx = rng.integers(0, n_days, size=n_days)
        boot_deltas[i] = rankicir_from_ic(ic_d_arr[idx]) - rankicir_from_ic(ic_t_arr[idx])

    boot_mean = float(boot_deltas.mean())
    boot_ci_low, boot_ci_high = float(np.quantile(boot_deltas, 0.025)), float(np.quantile(boot_deltas, 0.975))
    p_value = float(np.mean(boot_deltas <= 0))

    print(f"\n   bootstrap N=1000:")
    print(f"     mean Δ          = {boot_mean:+.4f}")
    print(f"     95% CI          = [{boot_ci_low:+.4f}, {boot_ci_high:+.4f}]")
    print(f"     p-value         = {p_value:.4f}  (target < 0.01)")

    pd.DataFrame([{
        "n_days": n_days, "n_bootstrap": n_boot,
        "observed_trend_rankicir": observed_trend_ric,
        "observed_d1_rankicir": observed_d1_ric,
        "observed_delta": observed_delta,
        "bootstrap_mean_delta": boot_mean,
        "bootstrap_ci_low": boot_ci_low,
        "bootstrap_ci_high": boot_ci_high,
        "p_value": p_value,
    }]).to_csv(FINAL_DIR / "bootstrap_test.csv", index=False, encoding="utf-8-sig")

    # =================================================================
    # Task 3: Bear Quintile 分析
    # =================================================================
    print("\n[3/5] TASK 3 — Bear D1 Quintile analysis ...")
    df_w5 = w5["df"].copy()
    # 添加 actual max_drawdown_5d (来自 df 大表)
    md_lookup = df.set_index([DATE_COL, TICKER_COL])["max_drawdown_5d"]
    df_w5 = df_w5.merge(
        md_lookup.reset_index().rename(columns={"max_drawdown_5d": "actual_maxdd_5d"}),
        on=[DATE_COL, TICKER_COL], how="left",
    )
    # 每日按 bear_pred 分 5 等分
    df_w5["quintile"] = -1
    for d, g in df_w5.groupby(DATE_COL):
        if len(g) < 5:
            continue
        ranks = g["bear_pred"].rank(method="first", pct=True).to_numpy()
        q = np.ceil(ranks * 5).astype("int8")    # 1..5
        df_w5.loc[g.index, "quintile"] = q

    df_w5 = df_w5[df_w5["quintile"] >= 1].reset_index(drop=True)

    # 每个 quintile 平均 bear / max_drawdown / r_future_5
    q_summary = df_w5.groupby("quintile").agg(
        n_rows=("quintile", "size"),
        avg_bear=("bear_pred", "mean"),
        avg_maxdd_5d=("actual_maxdd_5d", "mean"),
        avg_r_future_5=(TARGET_RET_COL, "mean"),
    ).reset_index()
    # 排名（1=最高）：MaxDD 升序 -> rank 越大越危险；r_future_5 降序 -> rank 越大越差
    q_summary["maxdd_rank"] = q_summary["avg_maxdd_5d"].rank(ascending=True).astype(int)
    q_summary["return_rank"] = q_summary["avg_r_future_5"].rank(ascending=False).astype(int)
    q_summary["label"] = q_summary["quintile"].map({1: "Q1 (safest)",
                                                       2: "Q2",
                                                       3: "Q3",
                                                       4: "Q4",
                                                       5: "Q5 (riskiest)"})
    q_summary.to_csv(FINAL_DIR / "bear_quintile_analysis.csv",
                       index=False, encoding="utf-8-sig")

    print()
    print(f"   {'Quintile':14s}  {'n_rows':>10s}  {'avg_bear':>10s}  "
          f"{'avg_MaxDD':>10s}  {'avg_r_future_5':>14s}  "
          f"{'MaxDD_rank':>10s}  {'Return_rank':>11s}")
    for _, r in q_summary.iterrows():
        print(f"   {r['label']:14s}  {r['n_rows']:>10,}  "
              f"{r['avg_bear']:>+10.4f}  {r['avg_maxdd_5d']:>+10.4f}  "
              f"{r['avg_r_future_5']:>+14.4f}  "
              f"{r['maxdd_rank']:>10d}  {r['return_rank']:>11d}")

    q1_md = float(q_summary[q_summary["quintile"] == 1]["avg_maxdd_5d"].iloc[0])
    q5_md = float(q_summary[q_summary["quintile"] == 5]["avg_maxdd_5d"].iloc[0])
    md_gap = q5_md - q1_md
    print(f"\n   Q5 - Q1 avg MaxDD gap = {md_gap*100:+.2f}%  (target > 1.0%)")

    # =================================================================
    # Task 4: Historical MaxDD 60d 基线
    # =================================================================
    print("\n[4/5] TASK 4 — Historical MaxDD 60d baseline ...")
    # historical_maxdd_60d(i, t) = max over past 60 trading days of max(0, -ret_1d)
    print("   building historical_maxdd_60d ...")
    t0 = time.time()
    df_hist = df.sort_values([TICKER_COL, DATE_COL]).reset_index(drop=True)
    df_hist["worst_1d"] = np.maximum(0.0,
                                       -df_hist["ret_1d"].astype("float64").fillna(0.0))
    # 按 ticker 滚动 60 天 max
    df_hist["historical_maxdd_60d"] = (
        df_hist.groupby(TICKER_COL)["worst_1d"]
        .transform(lambda s: s.rolling(60, min_periods=10).max())
    ).astype("float32")
    print(f"   computed in {time.time()-t0:.1f}s")

    # 测试集 W5 子集
    m_te_5 = ((df_hist[DATE_COL] >= pd.Timestamp("2023-01-01"))
                & (df_hist[DATE_COL] <= pd.Timestamp("2026-01-31")))
    test_hist = df_hist.loc[m_te_5].sort_values([DATE_COL, TICKER_COL]).reset_index(drop=True)
    test_hist["hist_maxdd_z"] = zscore_daily(test_hist, "historical_maxdd_60d")

    # 加 Alpha 预测（来自 cached W5）
    df_w5_sorted = w5["df"][[DATE_COL, TICKER_COL, "alpha_pred", "bull_z"]].copy()
    test_hist = test_hist.merge(df_w5_sorted, on=[DATE_COL, TICKER_COL], how="left")

    bull_z = test_hist["bull_z"].to_numpy("float32")
    hist_z = test_hist["hist_maxdd_z"].to_numpy("float32")

    # bear_simple conviction
    conv_simple = bull_z - 0.5 * hist_z
    meta_h = test_hist[[DATE_COL, TICKER_COL, TARGET_RET_COL]].reset_index(drop=True)
    m_simple = evaluate_full(meta_h, conv_simple.astype("float32"))

    # Reference: Trend pure, |bias_60| rule, D1 trained (W5)
    m_trend  = evaluate_full(meta_h, test_hist["alpha_pred"].to_numpy("float32"))
    # |bias_60| rule
    test_hist["abs_bias60"] = test_hist["bias_60"].abs()
    bear_C_z = zscore_daily(test_hist, "abs_bias60")
    m_C = evaluate_full(meta_h, (bull_z - 0.2 * bear_C_z).astype("float32"))
    # D1 W5
    test_hist_with_D1 = test_hist.merge(
        w5["df"][[DATE_COL, TICKER_COL, "bear_pred"]],
        on=[DATE_COL, TICKER_COL], how="left",
    )
    test_hist_with_D1["d1_z"] = zscore_daily(test_hist_with_D1, "bear_pred")
    d1_z_arr = test_hist_with_D1["d1_z"].to_numpy("float32")
    m_D1 = evaluate_full(meta_h, (bull_z - 0.5 * d1_z_arr).astype("float32"))

    baseline_rows = [
        {"config": "Trend pure (Alpha)", "rankicir": float(m_trend["rankicir"]),
         "sharpe": float(m_trend["top5pct_sharpe"]),
         "maxdd": float(m_trend["top5pct_max_dd"])},
        {"config": "|bias_60| rule Bear (α=0.2)",
         "rankicir": float(m_C["rankicir"]),
         "sharpe": float(m_C["top5pct_sharpe"]),
         "maxdd": float(m_C["top5pct_max_dd"])},
        {"config": "Historical MaxDD 60d rule (α=0.5)",
         "rankicir": float(m_simple["rankicir"]),
         "sharpe": float(m_simple["top5pct_sharpe"]),
         "maxdd": float(m_simple["top5pct_max_dd"])},
        {"config": "Trained Bear D1 (α=0.5)",
         "rankicir": float(m_D1["rankicir"]),
         "sharpe": float(m_D1["top5pct_sharpe"]),
         "maxdd": float(m_D1["top5pct_max_dd"])},
    ]
    pd.DataFrame(baseline_rows).to_csv(
        FINAL_DIR / "simple_baseline_comparison.csv",
        index=False, encoding="utf-8-sig")

    print()
    print(f"   {'Config':38s}  {'RankICIR':>9s}  {'SR':>8s}  {'MaxDD':>9s}")
    for r in baseline_rows:
        print(f"   {r['config']:38s}  {r['rankicir']:>9.4f}  "
              f"{r['sharpe']:>+8.3f}  {r['maxdd']*100:>+8.2f}%")

    d1_minus_hist = (baseline_rows[3]["rankicir"] - baseline_rows[2]["rankicir"]) * 10000
    print(f"\n   D1 trained - Historical MaxDD = {d1_minus_hist:+.1f} bp  "
          f"(target > 50 bp)")

    # =================================================================
    # 汇总
    # =================================================================
    print("\n[5/5] FINAL summary for paper abstract:")
    print()
    print(f"   1. Walk-Forward: D1 wins {n_wins}/{n_total} year-evaluations  "
          f"(target ≥ 5/7)")
    print(f"   2. Bootstrap p-value (D1 > Trend): {p_value:.4f}  "
          f"(target < 0.01)")
    print(f"   3. Q5 - Q1 avg MaxDD gap: {md_gap*100:+.2f}%  (target > 1.0%)")
    print(f"   4. D1 - Historical MaxDD baseline: {d1_minus_hist:+.1f} bp  "
          f"(target > 50 bp)")

    print(f"\nOutputs:")
    for p in [FINAL_DIR / "rolling_walkforward.csv",
              FINAL_DIR / "bootstrap_test.csv",
              FINAL_DIR / "bear_quintile_analysis.csv",
              FINAL_DIR / "simple_baseline_comparison.csv"]:
        if p.exists():
            print(f"  -> {p.relative_to(BB_RESULTS.parent.parent)}")


if __name__ == "__main__":
    main()
