"""Step 17 — Leakage check + parameter robustness.

Task 1: Validation-set parameter selection
  Grid search (lambda in {1, 2, 3, 5}, gamma in {0.20, 0.30, 0.40, 0.50},
  delta in {0.05, 0.10, 0.15}) on 2022 validation. Pick best on val,
  evaluate same params on test. Compare to test-set-selected
  RankICIR = 0.8094.

Task 2: Parameter sensitivity around (lambda=3.0, gamma=0.40, delta=0.15).
  Test 4 nearby points; verify shifts < 50 bp.

Task 3: IC distribution on test (D2f vs Trend); monthly aggregation.

Task 4: Rolling 90-day RankICIR over test period; detect spikes > 1.5.

Outputs:
  bull_bear/results/step17_validation_param_search.csv
  bull_bear/results/step17_param_sensitivity.csv
  bull_bear/results/step17_ic_distribution.csv
  bull_bear/results/step17_rolling_rankicir.csv
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.stats import spearmanr

from bull_bear.config_bb import (
    ALPHA_AGENT_PATH, BB_MODELS, BB_RESULTS,
    BEAR_FEATURES_D1, DATE_COL,
    TARGET_RET_COL, TEST_END, TEST_START, TICKER_COL,
    VAL_END, VAL_START,
)
from bull_bear.src.bear_agent import BearAgent
from src.data import load_dataset
from bull_bear.src.metrics_utils import evaluate_full


ALPHA_FEATURES = ["ma60_slope", "ema180_slope", "bias_60", "bias_60_vr", "ma180_slope"]
ALPHA_BY_REGIME = {"bear": 0.65, "sideway": 0.50, "bull": 0.35}
REVERSAL_FEATURES = [
    "ret_1d", "ret_3d",
    "rev_ret_2d", "rev_ret_3d_minus_1d",
    "rev_zscore_1d", "rev_mkt_excess_1d",
]
FRAC = 0.05

LAMBDA_GRID = (1.0, 2.0, 3.0, 5.0)
GAMMA_GRID  = (0.20, 0.30, 0.40, 0.50)
DELTA_GRID  = (0.05, 0.10, 0.15)


# ============================================================
# helpers
# ============================================================

def zscore_daily(panel, col):
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


def regime_alpha_t(panel):
    regime = panel["macro_regime_3"].astype(str).to_numpy()
    return np.array([ALPHA_BY_REGIME.get(r, 0.5) for r in regime], dtype="float32")


def regime_gamma_t(panel, base, delta):
    regime = panel["macro_regime_3"].astype(str).to_numpy()
    g = np.full(len(regime), base, dtype="float32")
    g[regime == "bull"] += delta
    g[regime == "bear"] -= delta
    return g


def add_reversal_features(df):
    df = df.sort_values([TICKER_COL, DATE_COL]).reset_index(drop=True)
    g = df.groupby(TICKER_COL, sort=False)
    ret_1d = df["ret_1d"].astype("float32").fillna(0.0)
    ret_1d_prev = g["ret_1d"].shift(1).astype("float32").fillna(0.0)
    df["rev_ret_2d"] = ((1.0 + ret_1d) * (1.0 + ret_1d_prev) - 1.0).astype("float32")
    df["rev_ret_3d_minus_1d"] = (df["ret_3d"].astype("float32") - df["ret_1d"].astype("float32"))
    roll_mean = g["ret_1d"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    roll_std = g["ret_1d"].transform(lambda s: s.rolling(20, min_periods=5).std(ddof=0))
    df["rev_zscore_1d"] = ((df["ret_1d"] - roll_mean) / roll_std.replace(0.0, np.nan)).astype("float32")
    df["rev_zscore_1d"] = df["rev_zscore_1d"].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df = df.sort_values([DATE_COL, TICKER_COL]).reset_index(drop=True)
    mkt = df.groupby(DATE_COL)["ret_1d"].transform("mean").astype("float32")
    df["rev_mkt_excess_1d"] = (df["ret_1d"].astype("float32") - mkt).astype("float32")
    return df


def daily_ic_series(meta, pred):
    """Per-day Spearman cross-section IC."""
    df = meta.copy()
    df["pred"] = pred
    out = {}
    for d, g in df.groupby(DATE_COL):
        if len(g) < 5: continue
        x = g["pred"].to_numpy(dtype="float64")
        y = g[TARGET_RET_COL].to_numpy(dtype="float64")
        if np.std(x) == 0 or np.std(y) == 0: continue
        rho, _ = spearmanr(x, y)
        out[pd.Timestamp(d)] = float(rho)
    return pd.Series(out).sort_index()


def rankicir(ic_series: pd.Series) -> float:
    return float(ic_series.mean() / (ic_series.std(ddof=0) + 1e-9))


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("=" * 80)
    print("Step 17 — Leakage check + parameter robustness")
    print("=" * 80)

    # ---- load + targets ----
    print("\n[0/5] load + reversal features + agent predictions ...")
    t0 = time.time()
    df = load_dataset().dropna(subset=[TARGET_RET_COL]).reset_index(drop=True)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = add_reversal_features(df)
    mask_va = (df[DATE_COL] >= pd.Timestamp(VAL_START)) & (df[DATE_COL] <= pd.Timestamp(VAL_END))
    mask_te = (df[DATE_COL] >= pd.Timestamp(TEST_START)) & (df[DATE_COL] <= pd.Timestamp(TEST_END))
    val_df = df.loc[mask_va].sort_values([DATE_COL, TICKER_COL]).reset_index(drop=True)
    test_df = df.loc[mask_te].sort_values([DATE_COL, TICKER_COL]).reset_index(drop=True)
    print(f"   val rows={len(val_df):,}  test rows={len(test_df):,}  "
          f"({time.time()-t0:.1f}s)")

    # ---- Alpha + Reversal predictions (same for all λ) ----
    print("\n[1/5] precompute Alpha + Reversal scores on val + test ...")
    alpha_m = CatBoostRegressor(); alpha_m.load_model(str(ALPHA_AGENT_PATH))
    alpha_med = pd.read_csv(str(ALPHA_AGENT_PATH).replace(".cbm", "_medians.csv"),
                              index_col=0).iloc[:, 0]
    for split_name, split_df in [("val", val_df), ("test", test_df)]:
        split_df["alpha_score"] = alpha_m.predict(
            split_df[ALPHA_FEATURES].astype("float32").fillna(alpha_med)).astype("float32")

    rev = CatBoostRegressor(); rev.load_model(str(BB_MODELS / "reversal_B_5d.cbm"))
    bear_d1_ref = BearAgent(features=BEAR_FEATURES_D1, name="bear_D1")
    bear_d1_ref.load(BB_MODELS / "bear_D1_agent.cbm")
    bear_medians = bear_d1_ref._train_medians
    rev_medians = val_df[REVERSAL_FEATURES].median()
    for split_name, split_df in [("val", val_df), ("test", test_df)]:
        split_df["rev_score"] = rev.predict(
            split_df[REVERSAL_FEATURES].astype("float32").fillna(rev_medians)
        ).astype("float32")
        split_df["bull_z"] = zscore_daily(split_df, "alpha_score")
        split_df["rev_z"]  = zscore_daily(split_df, "rev_score")

    # ---- precompute Bear D2 predictions for each lambda on both splits ----
    print("\n[2/5] precompute Bear D2 predictions for each lambda ...")
    bear_z_val: dict[float, np.ndarray] = {}
    bear_z_test: dict[float, np.ndarray] = {}
    for lam in LAMBDA_GRID:
        bear_path = BB_MODELS / f"bear_D2_l{lam:.1f}.cbm"
        bear_m = CatBoostRegressor(); bear_m.load_model(str(bear_path))
        for split_name, split_df, store in [("val", val_df, bear_z_val),
                                                ("test", test_df, bear_z_test)]:
            score = bear_m.predict(
                split_df[BEAR_FEATURES_D1].astype("float32").fillna(bear_medians)
            ).astype("float32")
            split_df[f"bear_d2_l{lam}"] = score
            store[lam] = zscore_daily(split_df, f"bear_d2_l{lam}")
        print(f"   lambda={lam:.1f} done")

    # ============================================================
    # Task 1 — validation grid search
    # ============================================================
    print("\n[3/5] Task 1 — validation-set parameter search "
          f"(grid size {len(LAMBDA_GRID)}x{len(GAMMA_GRID)}x{len(DELTA_GRID)} = "
          f"{len(LAMBDA_GRID) * len(GAMMA_GRID) * len(DELTA_GRID)}) ...")
    a_val = regime_alpha_t(val_df)
    a_test = regime_alpha_t(test_df)
    rows_grid = []
    meta_val = val_df[[DATE_COL, TICKER_COL, TARGET_RET_COL]].reset_index(drop=True)
    meta_test = test_df[[DATE_COL, TICKER_COL, TARGET_RET_COL]].reset_index(drop=True)
    bull_v = val_df["bull_z"].to_numpy("float32")
    bull_t = test_df["bull_z"].to_numpy("float32")
    rev_v = val_df["rev_z"].to_numpy("float32")
    rev_t = test_df["rev_z"].to_numpy("float32")
    for lam in LAMBDA_GRID:
        b_v = bear_z_val[lam]; b_t = bear_z_test[lam]
        for gma in GAMMA_GRID:
            for dlt in DELTA_GRID:
                g_v = regime_gamma_t(val_df, gma, dlt)
                g_t = regime_gamma_t(test_df, gma, dlt)
                pred_v = (bull_v - a_val * b_v + g_v * rev_v).astype("float32")
                pred_t = (bull_t - a_test * b_t + g_t * rev_t).astype("float32")
                ric_v = float(evaluate_full(meta_val, pred_v)["rankicir"])
                ric_t = float(evaluate_full(meta_test, pred_t)["rankicir"])
                rows_grid.append({"lambda": lam, "gamma": gma, "delta": dlt,
                                     "val_rankicir": ric_v,
                                     "test_rankicir": ric_t})
    df_grid = pd.DataFrame(rows_grid)
    df_grid.to_csv(BB_RESULTS / "step17_validation_param_search.csv",
                    index=False, encoding="utf-8-sig")

    val_best = df_grid.loc[df_grid["val_rankicir"].idxmax()]
    test_best = df_grid.loc[df_grid["test_rankicir"].idxmax()]
    print(f"\n   Validation-best  : lambda={val_best['lambda']:.1f}  "
          f"gamma={val_best['gamma']:.2f}  delta={val_best['delta']:.2f}")
    print(f"     val RankICIR  = {val_best['val_rankicir']:.4f}")
    print(f"     test RankICIR = {val_best['test_rankicir']:.4f}  (same params, evaluated on test)")
    print(f"\n   Test-best (oracle): lambda={test_best['lambda']:.1f}  "
          f"gamma={test_best['gamma']:.2f}  delta={test_best['delta']:.2f}")
    print(f"     test RankICIR = {test_best['test_rankicir']:.4f}")
    print(f"\n   Overfit gap (test_best - val_best on test) = "
          f"{(test_best['test_rankicir'] - val_best['test_rankicir'])*10000:+.1f} bp")
    if (test_best["test_rankicir"] - val_best["test_rankicir"]) < 0.01:
        print(f"   -> overfit gap < 100 bp; validation-selected params are nearly optimal on test.")
    else:
        print(f"   -> overfit gap > 100 bp; some test-set tuning advantage exists.")

    # Show top-5 by validation, evaluated on test
    print(f"\n   Top-5 configurations by validation RankICIR:")
    print(f"   {'lam':>5s}  {'gma':>5s}  {'dlt':>5s}  {'val_RIC':>9s}  {'test_RIC':>9s}")
    for _, r in df_grid.sort_values("val_rankicir", ascending=False).head(5).iterrows():
        print(f"   {r['lambda']:>5.1f}  {r['gamma']:>5.2f}  {r['delta']:>5.2f}  "
              f"{r['val_rankicir']:>9.4f}  {r['test_rankicir']:>9.4f}")

    # ============================================================
    # Task 2 — sensitivity around (lambda=3.0, gamma=0.40, delta=0.15)
    # ============================================================
    print("\n[4/5] Task 2 — sensitivity around (lambda=3, gamma=0.40, delta=0.15) ...")
    canonical = (3.0, 0.40, 0.15)
    neighbours = [
        (2.0, 0.35, 0.10),
        (2.0, 0.40, 0.15),
        (3.0, 0.35, 0.15),
        (4.0, 0.45, 0.20),
        (3.0, 0.45, 0.15),
        (3.0, 0.40, 0.10),
        (3.0, 0.40, 0.20),
    ]
    sens_rows = []
    for (lam, gma, dlt) in [canonical] + neighbours:
        # use precomputed lambda if available; else need to handle
        if lam in bear_z_test:
            b_t = bear_z_test[lam]
        else:
            # train bear D2 with this lam? skip — use closest available
            print(f"   skip lambda={lam} (no precomputed model)")
            continue
        g_t = regime_gamma_t(test_df, gma, dlt)
        pred = (bull_t - a_test * b_t + g_t * rev_t).astype("float32")
        ric = float(evaluate_full(meta_test, pred)["rankicir"])
        sens_rows.append({"lambda": lam, "gamma": gma, "delta": dlt,
                            "test_rankicir": ric,
                            "is_canonical": (lam, gma, dlt) == canonical})

    df_sens = pd.DataFrame(sens_rows)
    df_sens.to_csv(BB_RESULTS / "step17_param_sensitivity.csv",
                    index=False, encoding="utf-8-sig")
    canon_ric = float(df_sens[df_sens["is_canonical"]]["test_rankicir"].iloc[0])
    print(f"   canonical (lambda=3.0, gamma=0.40, delta=0.15): RankICIR = {canon_ric:.4f}")
    print(f"   neighbours:")
    df_sens["delta_bp"] = (df_sens["test_rankicir"] - canon_ric) * 10000
    for _, r in df_sens.iterrows():
        if r["is_canonical"]: continue
        print(f"     (lam={r['lambda']:.1f}, gma={r['gamma']:.2f}, dlt={r['delta']:.2f}): "
              f"RankICIR={r['test_rankicir']:.4f}  Δ={r['delta_bp']:+.1f} bp")
    max_abs_drift = float(df_sens[~df_sens["is_canonical"]]["delta_bp"].abs().max())
    print(f"\n   max |Δ| among neighbours = {max_abs_drift:.1f} bp")
    if max_abs_drift < 50:
        print(f"   -> sensitivity < 50 bp; system is robust to ±25% parameter perturbation.")
    else:
        print(f"   -> sensitivity >= 50 bp; some neighbours drift materially.")

    # ============================================================
    # Task 3 — IC distribution on test
    # ============================================================
    print("\n[5/5] Task 3 + 4 — IC distribution + rolling 90-day RankICIR ...")
    # canonical D2f conviction on test
    b_t_canon = bear_z_test[3.0]
    g_t_canon = regime_gamma_t(test_df, 0.40, 0.15)
    d2f_canon = (bull_t - a_test * b_t_canon + g_t_canon * rev_t).astype("float32")
    trend_pred = test_df["alpha_score"].to_numpy("float32")

    ic_d2f = daily_ic_series(meta_test, d2f_canon)
    ic_trend = daily_ic_series(meta_test, trend_pred)
    mean_d2f, std_d2f = float(ic_d2f.mean()), float(ic_d2f.std(ddof=0))
    mean_t, std_t = float(ic_trend.mean()), float(ic_trend.std(ddof=0))
    print(f"\n   D2f canonical IC distribution (n={len(ic_d2f)} days):")
    print(f"     mean(IC)   = {mean_d2f:+.5f}")
    print(f"     std(IC)    = {std_d2f:.5f}")
    print(f"     RankICIR   = mean/std = {mean_d2f/std_d2f:+.4f}  (≈ canonical {canon_ric:.4f})")
    print(f"     min IC     = {ic_d2f.min():+.4f}    max IC = {ic_d2f.max():+.4f}")
    print(f"     IC < 0     = {(ic_d2f < 0).sum()} / {len(ic_d2f)} days = {(ic_d2f < 0).mean()*100:.1f}%")
    print(f"\n   Trend single IC distribution (n={len(ic_trend)} days):")
    print(f"     mean(IC)   = {mean_t:+.5f}")
    print(f"     std(IC)    = {std_t:.5f}")
    print(f"     RankICIR   = mean/std = {mean_t/std_t:+.4f}")
    print(f"     IC < 0     = {(ic_trend < 0).sum()} / {len(ic_trend)} days = {(ic_trend < 0).mean()*100:.1f}%")

    # Monthly aggregation
    ic_d2f_idx = pd.to_datetime(ic_d2f.index)
    ic_trend_idx = pd.to_datetime(ic_trend.index)
    monthly_d2f = ic_d2f.groupby([ic_d2f_idx.year, ic_d2f_idx.month]).mean()
    monthly_trend = ic_trend.groupby([ic_trend_idx.year, ic_trend_idx.month]).mean()

    print(f"\n   Monthly mean IC (D2f vs Trend):")
    print(f"   {'year':>4s}  {'month':>5s}  {'D2f IC':>9s}  {'Trend IC':>9s}  {'gap':>7s}")
    monthly_rows = []
    for (yr, mo), v_d2f in monthly_d2f.items():
        v_trend = float(monthly_trend.get((yr, mo), float("nan")))
        v_d2f = float(v_d2f)
        gap = v_d2f - v_trend
        monthly_rows.append({"year": int(yr), "month": int(mo),
                                "d2f_ic": v_d2f, "trend_ic": v_trend, "gap": gap})
        print(f"   {int(yr):>4d}  {int(mo):>5d}  {v_d2f:>+9.5f}  "
              f"{v_trend:>+9.5f}  {gap:>+7.5f}")
    df_ic = pd.DataFrame(monthly_rows)
    df_ic.to_csv(BB_RESULTS / "step17_ic_distribution.csv",
                  index=False, encoding="utf-8-sig")

    # ============================================================
    # Task 4 — Rolling 90-day RankICIR
    # ============================================================
    print(f"\n   Rolling 90-day RankICIR (sliding window) ...")
    win = 90
    common_idx = ic_d2f.index
    rolling_rows = []
    for i in range(win, len(common_idx) + 1):
        window_ic = ic_d2f.iloc[i - win:i]
        ricir = float(window_ic.mean() / (window_ic.std(ddof=0) + 1e-9))
        window_ic_t = ic_trend.loc[window_ic.index]
        ricir_t = float(window_ic_t.mean() / (window_ic_t.std(ddof=0) + 1e-9))
        end_date = pd.Timestamp(common_idx[i - 1])
        rolling_rows.append({"window_end_date": end_date.date(),
                                "d2f_rolling_ricir": ricir,
                                "trend_rolling_ricir": ricir_t})
    df_roll = pd.DataFrame(rolling_rows)
    df_roll.to_csv(BB_RESULTS / "step17_rolling_rankicir.csv",
                    index=False, encoding="utf-8-sig")

    # Stats on rolling
    print(f"   rolling-90 D2f RankICIR: min={df_roll['d2f_rolling_ricir'].min():.4f}  "
          f"mean={df_roll['d2f_rolling_ricir'].mean():.4f}  "
          f"max={df_roll['d2f_rolling_ricir'].max():.4f}")
    print(f"   rolling-90 Trend RankICIR: min={df_roll['trend_rolling_ricir'].min():.4f}  "
          f"mean={df_roll['trend_rolling_ricir'].mean():.4f}  "
          f"max={df_roll['trend_rolling_ricir'].max():.4f}")

    # Find any spike > 1.5
    spikes = df_roll[df_roll["d2f_rolling_ricir"] > 1.5]
    if len(spikes) > 0:
        print(f"   D2f rolling RankICIR > 1.5 in {len(spikes)} windows; "
              f"first {spikes.iloc[0]['window_end_date']}, "
              f"last {spikes.iloc[-1]['window_end_date']}")
    else:
        print(f"   D2f rolling RankICIR never exceeds 1.5 (no suspicious spikes).")

    # Quarterly summary of rolling
    print(f"\n   Quarter-end snapshots of rolling-90 RankICIR:")
    df_roll["window_end_date"] = pd.to_datetime(df_roll["window_end_date"])
    df_roll["quarter"] = df_roll["window_end_date"].dt.to_period("Q")
    quarter_summary = df_roll.groupby("quarter").agg(
        d2f_mean=("d2f_rolling_ricir", "mean"),
        d2f_max =("d2f_rolling_ricir", "max"),
        trend_mean=("trend_rolling_ricir", "mean"),
    ).reset_index()
    for _, r in quarter_summary.iterrows():
        print(f"     {str(r['quarter']):>7s}  D2f mean={r['d2f_mean']:>6.4f}  "
              f"max={r['d2f_max']:>6.4f}  Trend mean={r['trend_mean']:>6.4f}")

    print(f"\nOutputs:")
    print(f"  -> {BB_RESULTS / 'step17_validation_param_search.csv'}")
    print(f"  -> {BB_RESULTS / 'step17_param_sensitivity.csv'}")
    print(f"  -> {BB_RESULTS / 'step17_ic_distribution.csv'}")
    print(f"  -> {BB_RESULTS / 'step17_rolling_rankicir.csv'}")


if __name__ == "__main__":
    main()
