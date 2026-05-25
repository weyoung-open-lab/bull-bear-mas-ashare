"""Step 20 — Holding period sweep + Alpha in-sample vs validation diagnostic.

Two tasks:

  Task A. Alpha in-sample fit quality vs validation OOF quality
    - Load canonical Alpha agent (trained on 2016-2021)
    - Predict on the same 2016-2021 train set (in-sample) and on the
      2022 validation set (OOF)
    - Report per-day RankIC mean + RankICIR for each split

  Task B. Holding-period sweep over D2c (no Reversal) vs D2f (with Reversal)
    For each N in {2, 3, 5} trading days:
      - Build r_future_Nd on the test panel (forward N-day cumulative ret)
      - Score the test set with the canonical Alpha / Bear-D2(λ=3) / Reversal models
      - D2c conviction:  alpha_z − α(t) · bear_z
      - D2f conviction:  alpha_z − α(t) · bear_z + γ(t) · reversal_z
      - Evaluate RankICIR against r_future_Nd
      - Backtest Top-5% with N-day hold and 0.30% round-trip cost
      - Report Sharpe + MaxDD per system

The script is intentionally standalone (no dependence on strategy_debate).

Output:
  bull_bear/results/step20_holding_period.csv
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
from scipy.stats import spearmanr, rankdata

from bull_bear.config_bb import (
    ALPHA_AGENT_PATH, BB_MODELS, BB_RESULTS,
    BEAR_FEATURES_D1, DATE_COL,
    TARGET_RET_COL, TEST_END, TEST_START, TICKER_COL,
    TRAIN_END, TRAIN_START,
    VAL_START, VAL_END,
)
from src.data import load_dataset


ALPHA_FEATURES = ["ma60_slope", "ema180_slope", "bias_60", "bias_60_vr", "ma180_slope"]
REVERSAL_FEATURES = [
    "ret_1d", "ret_3d",
    "rev_ret_2d", "rev_ret_3d_minus_1d",
    "rev_zscore_1d", "rev_mkt_excess_1d",
]
ALPHA_BY_REGIME   = {"bear": 0.65, "sideway": 0.50, "bull": 0.35}
GAMMA_BY_REGIME   = {"bear": 0.25, "sideway": 0.40, "bull": 0.55}
HOLDING_DAYS_GRID = (2, 3, 5)
COST_ROUND_TRIP   = 0.003          # 0.30% per round trip
TRADING_DAYS_YEAR = 252
BEAR_D2_LAMBDA3_PATH = BB_MODELS / "bear_D2_l3.0.cbm"
REVERSAL_B5D_PATH    = BB_MODELS / "reversal_B_5d.cbm"


# ------------------------------------------------------------------ utilities
def zscore_daily(panel: pd.DataFrame, col: str) -> np.ndarray:
    out = np.zeros(len(panel), dtype="float64")
    for _, g in panel.groupby(DATE_COL):
        v = g[col].to_numpy(dtype="float64")
        mu, sd = np.nanmean(v), np.nanstd(v, ddof=0)
        if sd > 1e-9 and np.isfinite(mu):
            z = (v - mu) / sd
            z = np.where(np.isfinite(z), z, 0.0)
        else:
            z = np.zeros_like(v)
        out[g.index.to_numpy()] = z
    return np.where(np.isfinite(out), out, 0.0).astype("float32")


def daily_rankic_series(panel: pd.DataFrame, score_col: str, target_col: str) -> pd.Series:
    rows = []
    for d, g in panel.groupby(DATE_COL):
        if len(g) < 5: continue
        x = g[score_col].to_numpy(dtype="float64")
        y = g[target_col].to_numpy(dtype="float64")
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 5: continue
        x, y = x[mask], y[mask]
        if np.std(x) < 1e-12 or np.std(y) < 1e-12: continue
        rho, _ = spearmanr(x, y)
        if np.isfinite(rho):
            rows.append((pd.Timestamp(d), float(rho)))
    return pd.Series(dict(rows)).sort_index()


def build_r_future_N(df: pd.DataFrame, N: int) -> pd.DataFrame:
    """Forward N-day cumulative return per (ticker, date) from ret_1d.

    r_future_Nd[t] = prod_{k=1..N}(1 + ret_1d[t+k]) - 1
    """
    df = df.sort_values([TICKER_COL, DATE_COL]).reset_index(drop=True)
    g = df.groupby(TICKER_COL, sort=False)["ret_1d"]
    # Build by chaining shifts -- avoids extra dependencies
    one_plus = (1.0 + df["ret_1d"].astype("float64").fillna(0.0))
    one_plus = pd.Series(one_plus.values, index=df.index)
    # roll forward N days within each ticker
    df = df.copy()
    df["_op"] = one_plus
    g2 = df.groupby(TICKER_COL, sort=False)
    prod = pd.Series(1.0, index=df.index, dtype="float64")
    for k in range(1, N + 1):
        prod = prod * g2["_op"].shift(-k).fillna(np.nan)
    df[f"r_future_{N}d"] = (prod - 1.0).astype("float32")
    df = df.drop(columns=["_op"])
    return df


def build_reversal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values([TICKER_COL, DATE_COL]).reset_index(drop=True)
    g = df.groupby(TICKER_COL, sort=False)
    ret_1d = df["ret_1d"].astype("float32").fillna(0.0)
    ret_1d_prev = g["ret_1d"].shift(1).astype("float32").fillna(0.0)
    df["rev_ret_2d"] = ((1.0 + ret_1d) * (1.0 + ret_1d_prev) - 1.0).astype("float32")
    df["rev_ret_3d_minus_1d"] = (df["ret_3d"].astype("float32")
                                  - df["ret_1d"].astype("float32"))
    roll_mean = g["ret_1d"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    roll_std  = g["ret_1d"].transform(lambda s: s.rolling(20, min_periods=5).std(ddof=0))
    df["rev_zscore_1d"] = ((df["ret_1d"] - roll_mean) / roll_std.replace(0.0, np.nan)).astype("float32")
    df["rev_zscore_1d"] = df["rev_zscore_1d"].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df = df.sort_values([DATE_COL, TICKER_COL]).reset_index(drop=True)
    mkt = df.groupby(DATE_COL)["ret_1d"].transform("mean").astype("float32")
    df["rev_mkt_excess_1d"] = (df["ret_1d"].astype("float32") - mkt).astype("float32")
    return df


def regime_array(panel: pd.DataFrame, mapping: dict[str, float], default: float) -> np.ndarray:
    regime = panel["macro_regime_3"].astype(str).to_numpy()
    return np.array([mapping.get(r, default) for r in regime], dtype="float32")


def backtest_topk_Nday(panel: pd.DataFrame, score_col: str, ret_fwd_col: str,
                        N: int, k_frac: float = 0.05) -> dict:
    """N-day-hold Top-K backtest with round-trip cost amortized per holding period.

    Mechanics:
      Each rebalance day, select Top-K% by score, hold for N days.
      Position turns over every N days → per-day cost share = COST / N.
      Daily portfolio return = mean(top_k r_future_Nd) / N - cost_share.

    Returns Sharpe (annualised) and MaxDD on the resulting daily NAV.
    """
    rows = []
    for d, g in panel.groupby(DATE_COL):
        if len(g) < 20:
            rows.append((pd.Timestamp(d), 0.0))
            continue
        s = g[score_col].to_numpy(dtype="float64")
        r = g[ret_fwd_col].to_numpy(dtype="float64")
        mask = np.isfinite(s) & np.isfinite(r)
        if mask.sum() < 5:
            rows.append((pd.Timestamp(d), 0.0))
            continue
        s, r = s[mask], r[mask]
        k = max(1, int(np.ceil(len(s) * k_frac)))
        idx = np.argsort(-s)[:k]                                       # top by score
        avg_fwd = float(np.mean(r[idx]))                                # mean N-day return
        daily_ret = avg_fwd / N - COST_ROUND_TRIP / N                  # amortise cost
        rows.append((pd.Timestamp(d), daily_ret))
    s = pd.Series(dict(rows)).sort_index()
    s = s.dropna()
    if s.empty or s.std() == 0:
        return {"sharpe": 0.0, "maxdd": 0.0, "ann_ret": 0.0}
    sharpe = float(s.mean() / s.std() * np.sqrt(TRADING_DAYS_YEAR))
    ann_ret = float(s.mean() * TRADING_DAYS_YEAR)
    nav = (1.0 + s).cumprod()
    dd = nav / nav.cummax() - 1.0
    return {"sharpe": sharpe, "maxdd": float(dd.min()), "ann_ret": ann_ret}


# ------------------------------------------------------------------ main
def main() -> None:
    print("=" * 80)
    print("Step 20 — Holding-period sweep + Alpha in-sample/OOF diagnostic")
    print("=" * 80)

    t0 = time.time()
    print("\n[0/4] load dataset ...")
    df = load_dataset().dropna(subset=["ret_1d"]).reset_index(drop=True)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    print(f"   rows={len(df):,}   ({time.time()-t0:.1f}s)")

    # build reversal features once on full panel (used for test scoring)
    df = build_reversal_features(df)

    # ---- masks
    mask_tr  = (df[DATE_COL] >= pd.Timestamp(TRAIN_START)) & (df[DATE_COL] <= pd.Timestamp(TRAIN_END))
    mask_val = (df[DATE_COL] >= pd.Timestamp(VAL_START))  & (df[DATE_COL] <= pd.Timestamp(VAL_END))
    mask_te  = (df[DATE_COL] >= pd.Timestamp(TEST_START)) & (df[DATE_COL] <= pd.Timestamp(TEST_END))

    # ============================================================
    # Task A — Alpha in-sample fit vs validation OOF
    # ============================================================
    print("\n[1/4] Task A — Alpha in-sample vs validation OOF ...")
    alpha_model = CatBoostRegressor(); alpha_model.load_model(str(ALPHA_AGENT_PATH))
    medians_path = Path(str(ALPHA_AGENT_PATH).replace(".cbm", "_medians.csv"))
    if medians_path.exists():
        alpha_medians = pd.read_csv(medians_path, index_col=0).iloc[:, 0]
    else:
        # Recompute from the training set (same operation that originally produced the file)
        tr_for_med = df.loc[mask_tr].dropna(subset=[TARGET_RET_COL])
        alpha_medians = tr_for_med[ALPHA_FEATURES].astype("float32").median()
        print(f"   (medians file missing, recomputed from train set: "
              f"{', '.join(f'{c}={v:.4f}' for c, v in alpha_medians.items())})")

    # On train (in-sample): need r_future_5
    tr_full = df.loc[mask_tr].dropna(subset=[TARGET_RET_COL]).reset_index(drop=True)
    X_a_tr = tr_full[ALPHA_FEATURES].astype("float32").fillna(alpha_medians)
    tr_full["alpha_score"] = alpha_model.predict(X_a_tr).astype("float32")
    ic_tr = daily_rankic_series(tr_full, "alpha_score", TARGET_RET_COL)
    ric_tr = float(ic_tr.mean() / ic_tr.std()) if ic_tr.std() > 0 else 0.0

    # On validation 2022 (OOF, model unseen): need r_future_5
    val_full = df.loc[mask_val].dropna(subset=[TARGET_RET_COL]).reset_index(drop=True)
    X_a_val = val_full[ALPHA_FEATURES].astype("float32").fillna(alpha_medians)
    val_full["alpha_score"] = alpha_model.predict(X_a_val).astype("float32")
    ic_val = daily_rankic_series(val_full, "alpha_score", TARGET_RET_COL)
    ric_val = float(ic_val.mean() / ic_val.std()) if ic_val.std() > 0 else 0.0

    # On test (OOF, for reference)
    te_a = df.loc[mask_te].dropna(subset=[TARGET_RET_COL]).reset_index(drop=True)
    X_a_te = te_a[ALPHA_FEATURES].astype("float32").fillna(alpha_medians)
    te_a["alpha_score"] = alpha_model.predict(X_a_te).astype("float32")
    ic_te = daily_rankic_series(te_a, "alpha_score", TARGET_RET_COL)
    ric_te = float(ic_te.mean() / ic_te.std()) if ic_te.std() > 0 else 0.0

    print(f"\n   in-sample (train 2016-2021, n={len(ic_tr)} days):")
    print(f"     mean(RankIC) = {ic_tr.mean():+.4f}   std = {ic_tr.std():.4f}   RankICIR = {ric_tr:+.4f}")
    print(f"   OOF (val 2022,         n={len(ic_val)} days):")
    print(f"     mean(RankIC) = {ic_val.mean():+.4f}   std = {ic_val.std():.4f}   RankICIR = {ric_val:+.4f}")
    print(f"   OOF (test 2023-2026,   n={len(ic_te)} days):")
    print(f"     mean(RankIC) = {ic_te.mean():+.4f}   std = {ic_te.std():.4f}   RankICIR = {ric_te:+.4f}")

    rankic_gap_in_vs_val = float(ic_tr.mean() - ic_val.mean())
    ric_gap_in_vs_val    = float(ric_tr - ric_val)
    print(f"\n   IN-vs-VAL gap:")
    print(f"     mean(RankIC):  in-sample − val = {rankic_gap_in_vs_val:+.4f}")
    print(f"     RankICIR:      in-sample − val = {ric_gap_in_vs_val:+.4f}")

    rows_taskA = [
        {"split": "train_2016-2021_in_sample",
         "n_days": int(len(ic_tr)),
         "mean_rankic": float(ic_tr.mean()),
         "std_rankic":  float(ic_tr.std()),
         "rankicir":    ric_tr},
        {"split": "val_2022_OOF",
         "n_days": int(len(ic_val)),
         "mean_rankic": float(ic_val.mean()),
         "std_rankic":  float(ic_val.std()),
         "rankicir":    ric_val},
        {"split": "test_2023-2026_OOF",
         "n_days": int(len(ic_te)),
         "mean_rankic": float(ic_te.mean()),
         "std_rankic":  float(ic_te.std()),
         "rankicir":    ric_te},
    ]

    # ============================================================
    # Task B — D2c vs D2f at holding periods 2/3/5
    # ============================================================
    print("\n[2/4] Task B prep — load Bear D2 (lambda=3) and Reversal B_5d ...")
    bear  = CatBoostRegressor(); bear.load_model(str(BEAR_D2_LAMBDA3_PATH))
    rev   = CatBoostRegressor(); rev.load_model(str(REVERSAL_B5D_PATH))

    # Score on test panel
    te = df.loc[mask_te].sort_values([DATE_COL, TICKER_COL]).reset_index(drop=True)
    X_a_te = te[ALPHA_FEATURES].astype("float32").fillna(alpha_medians)
    te["alpha_score"] = alpha_model.predict(X_a_te).astype("float32")

    bear_medians = te[BEAR_FEATURES_D1].median()
    X_b_te = te[BEAR_FEATURES_D1].astype("float32").fillna(bear_medians)
    te["bear_score"] = bear.predict(X_b_te).astype("float32")

    rev_medians = te[REVERSAL_FEATURES].median()
    X_r_te = te[REVERSAL_FEATURES].astype("float32").fillna(rev_medians)
    te["rev_score"] = rev.predict(X_r_te).astype("float32")

    te["alpha_z"] = zscore_daily(te, "alpha_score")
    te["bear_z"]  = zscore_daily(te, "bear_score")
    te["rev_z"]   = zscore_daily(te, "rev_score")

    alpha_t = regime_array(te, ALPHA_BY_REGIME, default=0.50)
    gamma_t = regime_array(te, GAMMA_BY_REGIME, default=0.40)

    te["conv_d2c"] = te["alpha_z"].to_numpy("float32") - alpha_t * te["bear_z"].to_numpy("float32")
    te["conv_d2f"] = te["conv_d2c"].to_numpy("float32") + gamma_t * te["rev_z"].to_numpy("float32")

    print(f"   test panel: {len(te):,} rows   ({te[DATE_COL].nunique()} days)")
    print(f"   regime alpha_t   mean={alpha_t.mean():.3f}  range=[{alpha_t.min():.2f}, {alpha_t.max():.2f}]")
    print(f"   regime gamma_t   mean={gamma_t.mean():.3f}  range=[{gamma_t.min():.2f}, {gamma_t.max():.2f}]")

    print("\n[3/4] Task B — sweep holding periods N in {2, 3, 5} ...")
    rows_taskB = []
    print()
    print(f"   {'Hold':<6}{'D2c RIC':>10}{'D2f RIC':>10}{'dRIC(bp)':>11}"
          f"{'D2c SR':>10}{'D2f SR':>10}{'dSR':>10}"
          f"{'D2c MDD':>10}{'D2f MDD':>10}")
    print(f"   {'-'*6}{'-'*10}{'-'*10}{'-'*11}{'-'*10}{'-'*10}{'-'*10}{'-'*10}{'-'*10}")

    for N in HOLDING_DAYS_GRID:
        te_N = build_r_future_N(te, N)
        te_N = te_N.dropna(subset=[f"r_future_{N}d"]).reset_index(drop=True)

        ic_d2c = daily_rankic_series(te_N, "conv_d2c", f"r_future_{N}d")
        ic_d2f = daily_rankic_series(te_N, "conv_d2f", f"r_future_{N}d")
        ric_d2c = float(ic_d2c.mean() / ic_d2c.std()) if ic_d2c.std() > 0 else 0.0
        ric_d2f = float(ic_d2f.mean() / ic_d2f.std()) if ic_d2f.std() > 0 else 0.0

        bt_d2c = backtest_topk_Nday(te_N, "conv_d2c", f"r_future_{N}d", N)
        bt_d2f = backtest_topk_Nday(te_N, "conv_d2f", f"r_future_{N}d", N)

        rows_taskB.append({
            "holding_days":  N,
            "d2c_rankicir":  ric_d2c,
            "d2f_rankicir":  ric_d2f,
            "delta_ric_bp":  (ric_d2f - ric_d2c) * 10000,
            "d2c_sharpe":    bt_d2c["sharpe"],
            "d2f_sharpe":    bt_d2f["sharpe"],
            "delta_sharpe":  bt_d2f["sharpe"] - bt_d2c["sharpe"],
            "d2c_maxdd":     bt_d2c["maxdd"],
            "d2f_maxdd":     bt_d2f["maxdd"],
            "d2c_ann_ret":   bt_d2c["ann_ret"],
            "d2f_ann_ret":   bt_d2f["ann_ret"],
        })

        print(f"   {N:>3}d  "
              f"{ric_d2c:>10.4f}{ric_d2f:>10.4f}{(ric_d2f-ric_d2c)*10000:>+10.1f}  "
              f"{bt_d2c['sharpe']:>+9.3f}{bt_d2f['sharpe']:>+10.3f}{bt_d2f['sharpe']-bt_d2c['sharpe']:>+10.3f}  "
              f"{bt_d2c['maxdd']*100:>+8.2f}%{bt_d2f['maxdd']*100:>+9.2f}%")

    # ============================================================
    # write outputs
    # ============================================================
    print("\n[4/4] writing outputs ...")
    out_taskA = BB_RESULTS / "step20_alpha_in_sample_vs_oof.csv"
    out_taskB = BB_RESULTS / "step20_holding_period.csv"
    pd.DataFrame(rows_taskA).to_csv(out_taskA, index=False, encoding="utf-8-sig")
    pd.DataFrame(rows_taskB).to_csv(out_taskB, index=False, encoding="utf-8-sig")
    print(f"   -> {out_taskA}")
    print(f"   -> {out_taskB}")
    print(f"\n   total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
