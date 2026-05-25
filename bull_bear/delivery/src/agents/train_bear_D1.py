"""Step 1 — Bull-Bear 对抗框架基础 + Bear Agent 训练。

四个任务：
  Task 1: 复现 P10 基线（bull − 0.2 × z(|bias_60|) = 0.6074）
  Task 2: 构造 max_drawdown_5d 训练目标 + 截面 z-score
  Task 3: 训练 Bear Agent CatBoost
  Task 4: 对抗仲裁对比（Trend pure / Bear_C 规则 / Bear_A 训练 α 网格）

输入：
  parent dataset (含 ret_1d / r_future_5 / 五个 trend 特征 / target)
  strategy_debate/results/models/trend_agent.cbm (Alpha Agent)

输出：
  bull_bear/results/models/bear_agent.cbm + medians
  bull_bear/results/bull_bear_step1.csv
  bull_bear/results/predictions_step1.parquet
  bull_bear/results/bear_target_stats.txt
"""

from __future__ import annotations

import sys
from pathlib import Path
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from bull_bear.config_bb import (
    ALPHA_AGENT_PATH, ALPHA_GRID, BB_MODELS, BB_RESULTS,
    BEAR_FEATURES, DATE_COL, P10_ALPHA, P10_TARGET_RANKICIR,
    TARGET_RET_COL, TEST_END, TEST_START, TICKER_COL,
    TRAIN_END, TRAIN_START, VAL_END, VAL_START,
)
from bull_bear.src.bear_agent import BearAgent
from bull_bear.src.bear_target import build_max_drawdown_5d, cross_section_zscore
from src.data import load_dataset
from bull_bear.src.metrics_utils import evaluate_full


# Alpha Agent 用的 trend 特征集（与 strategy_debate 的 trend_agent.cbm 训练特征一致）
ALPHA_FEATURES = [
    "ma60_slope", "ema180_slope",
    "bias_60", "bias_60_vr", "ma180_slope",
]


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


def quantile_summary(arr: np.ndarray, qs=(0.10, 0.50, 0.90)) -> dict:
    """打印分布摘要。"""
    a = arr[~np.isnan(arr)]
    return {f"q{int(q*100)}": float(np.quantile(a, q)) for q in qs}


def main() -> None:
    print("=" * 80)
    print("Step 1 — Bull-Bear adversarial framework + Bear Agent training")
    print("=" * 80)

    # =============================================================
    # 0. 加载 + 切分
    # =============================================================
    print("\n[0/4] load dataset + 3-way split ...")
    df = load_dataset().dropna(subset=[TARGET_RET_COL]).reset_index(drop=True)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    mask_tr = (df[DATE_COL] >= pd.Timestamp(TRAIN_START)) & (df[DATE_COL] <= pd.Timestamp(TRAIN_END))
    mask_va = (df[DATE_COL] >= pd.Timestamp(VAL_START))  & (df[DATE_COL] <= pd.Timestamp(VAL_END))
    mask_te = (df[DATE_COL] >= pd.Timestamp(TEST_START)) & (df[DATE_COL] <= pd.Timestamp(TEST_END))

    # 注意：build_max_drawdown_5d 需要完整连续序列（含 t+5 的 future return）
    # 所以我们在 split 之前要先在每只 ticker 内构造 target（用全 panel 数据）
    print(f"   panel rows total: {len(df):,}")

    # =============================================================
    # 1. Task 1 — 复现 P10 基线
    # =============================================================
    print("\n[1/4] Task 1: replicate P10 baseline ...")
    test_df = df.loc[mask_te].sort_values([DATE_COL, TICKER_COL]).reset_index(drop=True)
    print(f"   test rows: {len(test_df):,}  days={test_df[DATE_COL].nunique()}")

    # Alpha Agent (Trend) — 复用既有模型
    alpha = CatBoostRegressor()
    alpha.load_model(str(ALPHA_AGENT_PATH))
    # 复用其 medians
    medians_path = Path(str(ALPHA_AGENT_PATH).replace(".cbm", "_medians.csv"))
    alpha_medians = pd.read_csv(medians_path, index_col=0).iloc[:, 0]
    X_alpha_test = test_df[ALPHA_FEATURES].astype("float32").fillna(alpha_medians)
    test_df["bull_score_raw"] = alpha.predict(X_alpha_test).astype("float32")
    test_df["bull_score"] = zscore_daily(test_df, "bull_score_raw")

    # Bear_C 规则：|bias_60| 截面 z-score
    test_df["abs_bias60"] = test_df["bias_60"].abs()
    test_df["bear_score_C"] = zscore_daily(test_df, "abs_bias60")

    # conviction_C = bull - 0.2 * bear_C
    conviction_C = (test_df["bull_score"].to_numpy("float32")
                     - P10_ALPHA * test_df["bear_score_C"].to_numpy("float32"))
    meta = test_df[[DATE_COL, TICKER_COL, TARGET_RET_COL]].reset_index(drop=True)
    m_C = evaluate_full(meta, conviction_C)
    m_pure = evaluate_full(meta, test_df["bull_score_raw"].to_numpy("float32"))
    print(f"\n   Trend pure (Alpha raw)         RankICIR = {m_pure['rankicir']:.4f}")
    print(f"   Bear_C  α={P10_ALPHA:.1f} (P10 replication) "
          f"RankICIR = {m_C['rankicir']:.4f}   (target = {P10_TARGET_RANKICIR:.4f})")
    delta = abs(m_C["rankicir"] - P10_TARGET_RANKICIR) * 10000
    if delta < 5:
        print(f"   Baseline replicated: RankICIR = {m_C['rankicir']:.4f}  "
              f"(P10 target: {P10_TARGET_RANKICIR:.4f})  diff = {delta:.1f} bp  [OK]")
    else:
        print(f"   [WARN] baseline drift {delta:.1f} bp > 5 bp threshold")

    # =============================================================
    # 2. Task 2 — 构造 max_drawdown_5d 目标
    # =============================================================
    print("\n[2/4] Task 2: build max_drawdown_5d on full panel (per-ticker forward chain) ...")
    df_full = build_max_drawdown_5d(df, ret_col="ret_1d", window=5)
    df_full[DATE_COL] = pd.to_datetime(df_full[DATE_COL])

    # 截面 z-score per day
    df_full = cross_section_zscore(df_full, "max_drawdown_5d")

    # 分布诊断
    tr_target = df_full.loc[
        (df_full[DATE_COL] >= pd.Timestamp(TRAIN_START))
         & (df_full[DATE_COL] <= pd.Timestamp(TRAIN_END))
         & df_full["max_drawdown_5d"].notna(),
        "max_drawdown_5d",
    ].to_numpy("float64")
    print(f"\n   training-set max_drawdown_5d distribution:")
    print(f"     n          = {len(tr_target):,}")
    print(f"     mean       = {tr_target.mean():.5f}")
    print(f"     std        = {tr_target.std(ddof=0):.5f}")
    qs = quantile_summary(tr_target)
    print(f"     quantiles  = q10={qs['q10']:.5f}  q50={qs['q50']:.5f}  q90={qs['q90']:.5f}")
    zero_share = float(np.mean(tr_target == 0))
    print(f"     rows with max_dd=0 (uniformly rising)  = {zero_share*100:.1f}%")

    # 保存诊断
    stats_lines = [
        "Bear target diagnosis (training set 2016-2021)",
        f"  n          = {len(tr_target):,}",
        f"  mean       = {tr_target.mean():.6f}",
        f"  std        = {tr_target.std(ddof=0):.6f}",
        f"  q10        = {qs['q10']:.6f}",
        f"  q50        = {qs['q50']:.6f}",
        f"  q90        = {qs['q90']:.6f}",
        f"  zero-share = {zero_share*100:.2f}%  (rows with no drawdown in next 5 days)",
    ]
    (BB_RESULTS / "bear_target_stats.txt").write_text("\n".join(stats_lines),
                                                          encoding="utf-8")

    # =============================================================
    # 3. Task 3 — 训练 Bear Agent
    # =============================================================
    print("\n[3/4] Task 3: train Bear Agent ...")
    train_df_bear = df_full.loc[mask_tr].copy().reset_index(drop=True)
    val_df_bear   = df_full.loc[mask_va].copy().reset_index(drop=True)
    bear = BearAgent(features=BEAR_FEATURES, name="bear")
    bear.train(train_df_bear, target_col="max_drawdown_5d_z",
                df_val=val_df_bear, save=True)

    val_spearman = bear.validate_spearman(val_df_bear, target_col="max_drawdown_5d")
    print(f"   val Spearman(pred, raw max_drawdown_5d) = {val_spearman:+.4f}  "
          f"(target > 0.05)")
    if val_spearman > 0.05:
        print(f"   -> Bear Agent has predictive power on validation (PASS)")
    else:
        print(f"   -> [WARN] Bear Agent val Spearman below 0.05 threshold")

    # =============================================================
    # 4. Task 4 — 对抗仲裁对比
    # =============================================================
    print("\n[4/4] Task 4: bull - α·bear comparison ...")
    # 在测试集上预测 bear_score_A
    test_df_bear = df_full.loc[mask_te].sort_values([DATE_COL, TICKER_COL]).reset_index(drop=True)
    # 确认行序与 test_df 一致
    assert len(test_df_bear) == len(test_df), "test panel size mismatch"
    bear_pred_A = bear.predict_panel(test_df_bear).astype("float32")
    test_df["bear_score_A_raw"] = bear_pred_A
    test_df["bear_score_A"] = zscore_daily(test_df, "bear_score_A_raw")

    rows = []
    pred_cols = {}

    # Trend pure (raw Alpha)
    rows.append({"config": "Trend pure (Alpha raw)",
                 "alpha": np.nan,
                 "rankicir": float(m_pure["rankicir"]),
                 "sharpe":   float(m_pure["top5pct_sharpe"]),
                 "maxdd":    float(m_pure["top5pct_max_dd"])})
    pred_cols["pred_trend_pure"] = test_df["bull_score_raw"].to_numpy("float32")

    # Bear_C α=0.2 (P10 复现)
    rows.append({"config": f"Bear_C α={P10_ALPHA:.1f} (|bias_60| rule)",
                 "alpha": P10_ALPHA,
                 "rankicir": float(m_C["rankicir"]),
                 "sharpe":   float(m_C["top5pct_sharpe"]),
                 "maxdd":    float(m_C["top5pct_max_dd"])})
    pred_cols["pred_bear_C"] = conviction_C

    # Bear_A α grid
    bull_z = test_df["bull_score"].to_numpy("float32")
    bear_A_z = test_df["bear_score_A"].to_numpy("float32")
    for a in ALPHA_GRID:
        conviction_A = (bull_z - a * bear_A_z).astype("float32")
        m_A = evaluate_full(meta, conviction_A)
        rows.append({"config": f"Bear_A α={a:.1f} (trained model)",
                     "alpha": float(a),
                     "rankicir": float(m_A["rankicir"]),
                     "sharpe":   float(m_A["top5pct_sharpe"]),
                     "maxdd":    float(m_A["top5pct_max_dd"])})
        pred_cols[f"pred_bear_A_a{a}"] = conviction_A

    df_out = pd.DataFrame(rows)
    df_out["bear_C_p10_target"] = P10_TARGET_RANKICIR
    out_csv = BB_RESULTS / "bull_bear_step1.csv"
    df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # 预测 parquet
    pred_df = meta.copy()
    for c, arr in pred_cols.items():
        pred_df[c] = arr
    pred_df["bull_score_z"] = bull_z
    pred_df["bear_score_A_z"] = bear_A_z
    pred_df["bear_score_C_z"] = test_df["bear_score_C"].to_numpy("float32")
    pred_df["bear_score_A_raw"] = bear_pred_A
    pred_df.to_parquet(BB_RESULTS / "predictions_step1.parquet", index=False)

    # 控制台打印 + verdict
    print()
    line = "+" + "-" * 40 + "+" + "-" * 7 + "+" + "-" * 11 + "+" + "-" * 10 + "+" + "-" * 11 + "+"
    print(line)
    print(f"| {'Config':38s} | {'α':>5s} | {'RankICIR':>9s} | "
          f"{'SR':>8s} | {'MaxDD':>9s} |")
    print(line)
    ric_pure = m_pure["rankicir"]
    for r in rows:
        a_str = "—" if not np.isfinite(r["alpha"]) else f"{r['alpha']:.1f}"
        delta_bp = (r["rankicir"] - ric_pure) * 10000
        print(f"| {r['config']:38s} | {a_str:>5s} | "
              f"{r['rankicir']:>9.4f} | "
              f"{r['sharpe']:>+8.3f} | "
              f"{r['maxdd']*100:>+8.2f}% |  "
              f"Δ={delta_bp:+.1f} bp")
    print(line)

    # 关键验收：Bear_A best vs Bear_C
    bear_A_rows = [r for r in rows if r["config"].startswith("Bear_A")]
    best_A = max(bear_A_rows, key=lambda r: r["rankicir"])
    bear_C_ric = m_C["rankicir"]
    print("\n=== Verdict ===")
    print(f"  Bear_C (rule)  α={P10_ALPHA:.1f}             = {bear_C_ric:.4f}  "
          f"(P10 target 0.6074)")
    print(f"  Bear_A (trained) best α={best_A['alpha']:.1f}  = {best_A['rankicir']:.4f}  "
          f"Δ vs Bear_C = {(best_A['rankicir']-bear_C_ric)*10000:+.1f} bp")
    if best_A["rankicir"] > bear_C_ric:
        print(f"  -> Bear Agent OUTPERFORMS the |bias_60| rule. "
              f"Training adds value beyond the rule.")
    else:
        print(f"  -> Bear Agent does NOT outperform the rule. "
              f"Training adds no value over |bias_60|.")

    # bear pred 与 |bias_60| 的相关性诊断
    bear_raw = test_df["bear_score_A_raw"].to_numpy("float64")
    abs_b60_z = test_df["bear_score_C"].to_numpy("float64")
    from scipy.stats import pearsonr, spearmanr
    p, _ = pearsonr(bear_raw, abs_b60_z)
    s, _ = spearmanr(bear_raw, abs_b60_z)
    print(f"\n  corr(Bear Agent pred, |bias_60|_z): Pearson={p:+.4f}  Spearman={s:+.4f}")
    if abs(p) < 0.5:
        print(f"  -> Bear learned beyond just |bias_60|.")
    else:
        print(f"  -> Bear strongly correlated with rule (degenerate).")

    print(f"\nOutputs:")
    for p_ in [out_csv,
                BB_RESULTS / "predictions_step1.parquet",
                BB_RESULTS / "bear_target_stats.txt",
                BB_MODELS / "bear_agent.cbm"]:
        if p_.exists():
            print(f"  -> {p_.relative_to(BB_RESULTS.parent.parent)}")


if __name__ == "__main__":
    main()
