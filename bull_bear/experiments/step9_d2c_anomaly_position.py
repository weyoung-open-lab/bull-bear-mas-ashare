"""Step 9 — D2c + position-scaled Anomaly safety valve.

Protocol:
  Stock selection (signal): always D2c (Alpha − α(t)·D2_λ=3.0), unchanged.
  Position sizing:
    Anomaly day (D_M > 8.96, 64 days):  portfolio_return × 0.5 (50% cash)
    Normal day:                          portfolio_return × 1.0

Note: position scaling is post-selection. RankICIR (a cross-section ranking
metric) is unaffected; only Sharpe and MaxDD shift.

Outputs:
  bull_bear/results/step9_d2c_anomaly_position.csv
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
    ALPHA_AGENT_PATH, BB_MODELS, BB_RESULTS,
    BEAR_FEATURES_D1, DATE_COL,
    TARGET_RET_COL, TEST_END, TEST_START, TICKER_COL,
)
from bull_bear.src.bear_agent import BearAgent
from src.backtest import backtest_topk
from src.data import load_dataset
from bull_bear.src.metrics_utils import evaluate_full

ALPHA_FEATURES = [
    "ma60_slope", "ema180_slope", "bias_60", "bias_60_vr", "ma180_slope",
]
ALPHA_BY_REGIME = {"bear": 0.65, "sideway": 0.50, "bull": 0.35}
D2C_LAMBDA = 3.0
POSITION_SCALE_ANOMALY = 0.5     # 50% cash on anomaly days
TRADING_DAYS_PER_YEAR = 252


def zscore_daily(panel: pd.DataFrame, col: str) -> np.ndarray:
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


def regime_alpha(panel: pd.DataFrame) -> np.ndarray:
    regime = panel["macro_regime_3"].astype(str).to_numpy()
    return np.array([ALPHA_BY_REGIME.get(r, 0.5) for r in regime], dtype="float32")


def recompute_stats(daily_return: pd.Series) -> dict:
    """Recompute SR + MaxDD from a modified daily return series."""
    s = daily_return.copy().sort_index()
    nav = (1.0 + s).cumprod()
    ann_ret = float(s.mean() * TRADING_DAYS_PER_YEAR)
    ann_vol = float(s.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
    drawdown = nav / nav.cummax() - 1
    max_dd = float(drawdown.min())
    return {"annual_return": ann_ret, "annual_volatility": ann_vol,
             "sharpe": sharpe, "maxdd": max_dd, "n_days": len(s)}


def main() -> None:
    print("=" * 80)
    print("Step 9 — D2c + position-scaled Anomaly Agent")
    print("=" * 80)

    # ---- load data ----
    print("\n[1/5] load test panel + Alpha + Bear D2 λ=3.0 + anomaly flags ...")
    df = load_dataset().dropna(subset=[TARGET_RET_COL]).reset_index(drop=True)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    mask_te = (df[DATE_COL] >= pd.Timestamp(TEST_START)) & (df[DATE_COL] <= pd.Timestamp(TEST_END))
    test = df.loc[mask_te].sort_values([DATE_COL, TICKER_COL]).reset_index(drop=True)

    alpha_model = CatBoostRegressor(); alpha_model.load_model(str(ALPHA_AGENT_PATH))
    alpha_medians = pd.read_csv(
        str(ALPHA_AGENT_PATH).replace(".cbm", "_medians.csv"),
        index_col=0).iloc[:, 0]
    test["alpha_score"] = alpha_model.predict(
        test[ALPHA_FEATURES].astype("float32").fillna(alpha_medians)
    ).astype("float32")

    bear_d2 = CatBoostRegressor()
    bear_d2.load_model(str(BB_MODELS / f"bear_D2_l{D2C_LAMBDA:.1f}.cbm"))
    # use D1 medians for the same feature set
    bear_d1 = BearAgent(features=BEAR_FEATURES_D1, name="bear_D1")
    bear_d1.load(BB_MODELS / "bear_D1_agent.cbm")
    X_b = test[BEAR_FEATURES_D1].astype("float32").fillna(bear_d1._train_medians)
    test["bear_score"] = bear_d2.predict(X_b).astype("float32")

    test["bull_z"] = zscore_daily(test, "alpha_score")
    test["bear_z"] = zscore_daily(test, "bear_score")
    alpha_t = regime_alpha(test)
    test["d2c_score"] = (test["bull_z"].to_numpy("float32")
                          - alpha_t * test["bear_z"].to_numpy("float32"))

    # anomaly day flags (from R6 predictions; pre-computed Mahalanobis > 8.96)
    r6 = pd.read_parquet(
        Path(__file__).resolve().parents[2]
        / "strategy_debate/results/predictions_R6.parquet"
    )
    r6[DATE_COL] = pd.to_datetime(r6[DATE_COL])
    anomaly_dates = set(r6.loc[r6["anomaly"], DATE_COL].dt.normalize().unique())
    n_anomaly = len(anomaly_dates)
    print(f"   anomaly days (D_M > 8.96): {n_anomaly}")

    # ---- Two-track backtest: proxy (matches step 8 numbers) + realistic (ret_1d) ----
    print("\n[2/5] D2c baseline backtest (two tracks) ...")
    pred = test["d2c_score"].to_numpy("float32")
    # Track A — r_future_5/5 proxy (matches step 8 reported numbers)
    meta_proxy = test[[DATE_COL, TICKER_COL, TARGET_RET_COL]].reset_index(drop=True)
    m_eval = evaluate_full(meta_proxy, pred)
    bt_proxy = backtest_topk(meta_proxy, pred, frac=0.05)
    # Track B — real ret_1d
    meta_real = test[[DATE_COL, TICKER_COL, TARGET_RET_COL, "ret_1d"]].reset_index(drop=True)
    bt_real = backtest_topk(meta_real, pred, frac=0.05)

    print(f"   RankICIR              = {m_eval['rankicir']:.4f}")
    print(f"   [A proxy r_future_5/5]   Sharpe = {bt_proxy.sharpe:+.4f}  MaxDD = {bt_proxy.max_drawdown:.4f}")
    print(f"   [B real  ret_1d daily]   Sharpe = {bt_real.sharpe:+.4f}  MaxDD = {bt_real.max_drawdown:.4f}")
    print(f"   (proxy track inflates Sharpe by averaging 5-day forward returns into 'daily' values.")
    print(f"    Track B is the honest realised-daily Sharpe; only use track A to compare with step 8.)")

    # ---- Apply position scaling under both tracks ----
    print(f"\n[3/5] Apply position scale × {POSITION_SCALE_ANOMALY} on {n_anomaly} anomaly days ...")

    def scale_anomaly(bt) -> tuple[dict, float, int]:
        daily = bt.daily_return.copy()
        daily.index = pd.to_datetime(daily.index)
        mask_arr = np.array(
            [pd.Timestamp(d).normalize() in anomaly_dates for d in daily.index]
        )
        scaled = daily.copy()
        scaled.iloc[mask_arr] = scaled.iloc[mask_arr] * POSITION_SCALE_ANOMALY
        pos = np.ones(len(daily))
        pos[mask_arr] = POSITION_SCALE_ANOMALY
        return recompute_stats(scaled), float(pos.mean()), int(mask_arr.sum())

    stats_proxy_scaled, avg_pos, n_anom_hit = scale_anomaly(bt_proxy)
    stats_real_scaled, _, _ = scale_anomaly(bt_real)
    print(f"   anomaly days hitting daily series: {n_anom_hit}  avg position = {avg_pos:.4f}")
    # alias for downstream code
    daily = bt_proxy.daily_return.copy()
    daily.index = pd.to_datetime(daily.index)
    mask = daily.index.map(lambda d: pd.Timestamp(d).normalize() in anomaly_dates)
    stats_scaled = stats_proxy_scaled    # used in subsequent text
    bt_base = bt_proxy                    # alias for the apples-to-apples track

    # Effective average position
    pos = pd.Series(1.0, index=daily.index)
    pos[mask] = POSITION_SCALE_ANOMALY
    avg_position = float(pos.mean())

    print(f"\n[4/5] D2c + scaled anomaly results (both tracks) ...")
    print(f"   RankICIR (cross-section, unchanged): {m_eval['rankicir']:.4f}")
    print(f"   [A proxy] D2c unscaled : Sharpe={bt_proxy.sharpe:+.4f}  MaxDD={bt_proxy.max_drawdown:.4f}")
    print(f"   [A proxy] D2c + scale  : Sharpe={stats_proxy_scaled['sharpe']:+.4f}  MaxDD={stats_proxy_scaled['maxdd']:.4f}")
    print(f"   [B real ] D2c unscaled : Sharpe={bt_real.sharpe:+.4f}  MaxDD={bt_real.max_drawdown:.4f}")
    print(f"   [B real ] D2c + scale  : Sharpe={stats_real_scaled['sharpe']:+.4f}  MaxDD={stats_real_scaled['maxdd']:.4f}")
    print(f"   avg position    = {avg_pos:.4f}  ({n_anom_hit} of {len(daily)} days at 50% cash)")

    # ---- diagnostic: anomaly-day return statistics ----
    anom_returns = daily.loc[mask]
    norm_returns = daily.loc[~mask]
    print(f"\n[diagnostic] Anomaly-day daily returns:")
    print(f"   anomaly: n={len(anom_returns)}  mean={anom_returns.mean():+.5f}  "
          f"std={anom_returns.std(ddof=0):.5f}  min={anom_returns.min():+.4f}")
    print(f"   normal : n={len(norm_returns)}  mean={norm_returns.mean():+.5f}  "
          f"std={norm_returns.std(ddof=0):.5f}  min={norm_returns.min():+.4f}")

    # ---- summary CSV ----
    print("\n[5/5] write CSV + summary table ...")
    rows = [
        # D1d reference: from prior step (proxy track)
        {"config": "D1d (Anomaly valve = full Alpha) [proxy]",
         "track": "proxy", "rankicir": 0.742, "sharpe": 1.805, "maxdd": -0.3352,
         "avg_position": 1.0},
        {"config": "D2c (no scaling) [proxy]",
         "track": "proxy", "rankicir": float(m_eval["rankicir"]),
         "sharpe": float(bt_proxy.sharpe),
         "maxdd": float(bt_proxy.max_drawdown),
         "avg_position": 1.0},
        {"config": f"D2c + position×{POSITION_SCALE_ANOMALY:.1f} on anomaly [proxy]",
         "track": "proxy", "rankicir": float(m_eval["rankicir"]),
         "sharpe": float(stats_proxy_scaled["sharpe"]),
         "maxdd": float(stats_proxy_scaled["maxdd"]),
         "avg_position": avg_pos},
        {"config": "D2c (no scaling) [real ret_1d]",
         "track": "real", "rankicir": float(m_eval["rankicir"]),
         "sharpe": float(bt_real.sharpe),
         "maxdd": float(bt_real.max_drawdown),
         "avg_position": 1.0},
        {"config": f"D2c + position×{POSITION_SCALE_ANOMALY:.1f} on anomaly [real ret_1d]",
         "track": "real", "rankicir": float(m_eval["rankicir"]),
         "sharpe": float(stats_real_scaled["sharpe"]),
         "maxdd": float(stats_real_scaled["maxdd"]),
         "avg_position": avg_pos},
    ]
    df_out = pd.DataFrame(rows)
    out_csv = BB_RESULTS / "step9_d2c_anomaly_position.csv"
    df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")

    line = "+" + "-" * 56 + "+" + "-" * 9 + "+" + "-" * 11 + "+" + "-" * 10 + "+" + "-" * 11 + "+"
    print()
    print(line)
    print(f"| {'Config':54s} | {'track':>7s} | {'RankICIR':>9s} | {'Sharpe':>8s} | {'MaxDD':>9s} |")
    print(line)
    for r in rows:
        print(f"| {r['config']:54s} | {r['track']:>7s} | {r['rankicir']:>9.4f} | "
              f"{r['sharpe']:>+8.3f} | {r['maxdd']*100:>+8.2f}% |")
    print(line)

    # ---- verdict ----
    print("\n=== Verdict ===")
    print(f"  RankICIR unchanged at {m_eval['rankicir']:.4f} (position scaling does not affect ranking).")
    print()
    print("  Track A (proxy r_future_5/5 daily, matches step-8 numbers):")
    d_sharpe_a = stats_proxy_scaled["sharpe"] - bt_proxy.sharpe
    d_maxdd_a_pp = (stats_proxy_scaled["maxdd"] - bt_proxy.max_drawdown) * 100
    print(f"    Sharpe shift = {d_sharpe_a:+.3f}   "
          f"MaxDD shift = {d_maxdd_a_pp:+.2f} pp")
    print()
    print("  Track B (real ret_1d daily, honest realised Sharpe):")
    d_sharpe_b = stats_real_scaled["sharpe"] - bt_real.sharpe
    d_maxdd_b_pp = (stats_real_scaled["maxdd"] - bt_real.max_drawdown) * 100
    print(f"    Sharpe shift = {d_sharpe_b:+.3f}   "
          f"MaxDD shift = {d_maxdd_b_pp:+.2f} pp")
    print()
    if (stats_proxy_scaled["sharpe"] >= bt_proxy.sharpe and
        abs(stats_proxy_scaled["maxdd"]) <= abs(bt_proxy.max_drawdown)):
        print("  -> Track A: position-scaled valve dominates D2c on both risk metrics.")
    else:
        print("  -> Track A: position-scaled valve sacrifices some return on anomaly days "
              "that carry above-average alpha for the system.")
    if (stats_real_scaled["sharpe"] >= bt_real.sharpe and
        abs(stats_real_scaled["maxdd"]) <= abs(bt_real.max_drawdown)):
        print("  -> Track B: position-scaled valve dominates D2c on both risk metrics.")
    else:
        print("  -> Track B: position-scaled valve trades return for vol reduction, "
              "outcome mixed.")

    print(f"\nOutput: {out_csv.relative_to(BB_RESULTS.parent.parent)}")


if __name__ == "__main__":
    main()
