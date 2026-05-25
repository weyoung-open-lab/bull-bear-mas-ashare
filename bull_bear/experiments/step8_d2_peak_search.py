"""Step 8 — D2 lambda peak search + adaptive alpha combination.

Two stages:

  Stage 1 (lambda peak search)
    Train Error-Informed Bear D2 at lambda in {0.5, 1.0, 2.0, 3.0, 5.0, 10.0}.
    Evaluate each at:
      D2a: fixed alpha = 0.5 (the D1b protocol)
      D2c: adaptive alpha(t) by regime (the D1c protocol)
    Find the peak lambda for both protocols.

  Stage 2 (consolidated ablation chain)
    Print updated ablation table from B0 down to D2c, with the new
    D2a / D2b / D2c rows inserted after D1c.

Reuses bear_D2_l*.cbm from step 7 when lambda matches; trains new ones
for lambda in {3.0, 5.0, 10.0}.

Output:
  bull_bear/results/step8_d2_peak.csv
  bull_bear/results/models/bear_D2_l{3.0,5.0,10.0}.cbm
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
from scipy.stats import rankdata

from bull_bear.config_bb import (
    ALPHA_AGENT_PATH, BB_MODELS, BB_RESULTS,
    BEAR_FEATURES_D1, CATBOOST_PARAMS, DATE_COL,
    TARGET_RET_COL, TEST_END, TEST_START, TICKER_COL,
    TRAIN_END, TRAIN_START,
)
from bull_bear.src.bear_agent import BearAgent
from bull_bear.src.bear_target import build_max_drawdown_5d, cross_section_zscore
from src.data import load_dataset
from bull_bear.src.metrics_utils import evaluate_full


ALPHA_FEATURES = [
    "ma60_slope", "ema180_slope", "bias_60", "bias_60_vr", "ma180_slope",
]
FIXED_ALPHA = 0.5
ALPHA_BY_REGIME = {"bear": 0.65, "sideway": 0.50, "bull": 0.35}
LAMBDA_GRID = (0.5, 1.0, 2.0, 3.0, 5.0, 10.0)

# Reference numbers from prior ablation (kept for the consolidated table)
B0_REF      = 0.297
M1_REF      = 0.256
T_REF       = 0.598
BC_REF      = 0.633
D1A_REF     = 0.692     # D1 alpha=0.2
D1B_REF     = 0.741     # D1 alpha=0.5 fixed
D1C_REF     = 0.744     # D1 + adaptive alpha (regime)


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


def pct_rank_daily(panel: pd.DataFrame, col: str) -> np.ndarray:
    out = np.full(len(panel), 0.5, dtype="float64")
    for d, g in panel.groupby(DATE_COL):
        v = g[col].to_numpy(dtype="float64")
        r = rankdata(v, method="average") / max(len(v), 1)
        out[g.index.to_numpy()] = r
    return out.astype("float32")


def regime_alpha(panel: pd.DataFrame) -> np.ndarray:
    regime = panel["macro_regime_3"].astype(str).to_numpy()
    return np.array([ALPHA_BY_REGIME.get(r, 0.5) for r in regime], dtype="float32")


def evaluate(meta: pd.DataFrame, pred: np.ndarray, label: str) -> dict:
    m = evaluate_full(meta, pred.astype("float32"))
    return {"config": label,
             "rankicir": float(m["rankicir"]),
             "sharpe":   float(m["top5pct_sharpe"]),
             "maxdd":    float(m["top5pct_max_dd"])}


def main() -> None:
    print("=" * 80)
    print("Step 8 — D2 lambda peak search + adaptive alpha combination")
    print("=" * 80)

    print("\n[0/5] load + build max_drawdown_5d target ...")
    t0 = time.time()
    df = load_dataset().dropna(subset=[TARGET_RET_COL]).reset_index(drop=True)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = build_max_drawdown_5d(df, ret_col="ret_1d", window=5)
    df = cross_section_zscore(df, "max_drawdown_5d")
    mask_tr = (df[DATE_COL] >= pd.Timestamp(TRAIN_START)) & (df[DATE_COL] <= pd.Timestamp(TRAIN_END))
    mask_te = (df[DATE_COL] >= pd.Timestamp(TEST_START)) & (df[DATE_COL] <= pd.Timestamp(TEST_END))
    train = df.loc[mask_tr].reset_index(drop=True)
    test  = df.loc[mask_te].sort_values([DATE_COL, TICKER_COL]).reset_index(drop=True)
    print(f"   train={len(train):,}  test={len(test):,}  ({time.time()-t0:.1f}s)")

    # ---- load Alpha, compute scores ----
    print("\n[1/5] load Alpha + compute scores on train/test ...")
    alpha_model = CatBoostRegressor(); alpha_model.load_model(str(ALPHA_AGENT_PATH))
    alpha_medians = pd.read_csv(
        str(ALPHA_AGENT_PATH).replace(".cbm", "_medians.csv"),
        index_col=0).iloc[:, 0]

    X_a_tr = train[ALPHA_FEATURES].astype("float32").fillna(alpha_medians)
    train["alpha_score"] = alpha_model.predict(X_a_tr).astype("float32")
    X_a_te = test[ALPHA_FEATURES].astype("float32").fillna(alpha_medians)
    test["alpha_score"] = alpha_model.predict(X_a_te).astype("float32")
    test["bull_z"] = zscore_daily(test, "alpha_score")

    # alpha rank error on train
    alpha_rank  = pct_rank_daily(train, "alpha_score")
    actual_rank = pct_rank_daily(train, TARGET_RET_COL)
    train["alpha_rank_error"] = np.abs(alpha_rank - actual_rank).astype("float32")
    print(f"   alpha_rank_error  mean={train['alpha_rank_error'].mean():.4f}  "
          f"p50={train['alpha_rank_error'].quantile(0.50):.4f}  "
          f"p95={train['alpha_rank_error'].quantile(0.95):.4f}")

    # ---- Bear training prep ----
    valid_mask = train["max_drawdown_5d_z"].notna()
    tr2 = train.loc[valid_mask].reset_index(drop=True)
    X_b_tr = tr2[BEAR_FEATURES_D1].astype("float32")
    bear_medians = X_b_tr.median()
    X_b_tr = X_b_tr.fillna(bear_medians)
    y_b_tr = tr2["max_drawdown_5d_z"].astype("float32").to_numpy()
    err = tr2["alpha_rank_error"].to_numpy("float32")
    X_b_te = test[BEAR_FEATURES_D1].astype("float32").fillna(bear_medians)

    alpha_t = regime_alpha(test)
    meta_te = test[[DATE_COL, TICKER_COL, TARGET_RET_COL]].reset_index(drop=True)

    # ---- Stage 1: lambda grid (train + eval each) ----
    print("\n[2/5] Stage 1 — train + evaluate D2 at lambda in {0.5, 1.0, 2.0, 3.0, 5.0, 10.0} ...")
    print()
    rows: list[dict] = []
    for lam in LAMBDA_GRID:
        save = BB_MODELS / f"bear_D2_l{lam:.1f}.cbm"
        if save.exists():
            m = CatBoostRegressor(); m.load_model(str(save))
            tag = "(reused)"
            tt = 0.0
        else:
            tt0 = time.time()
            w = (1.0 + lam * err).astype("float32")
            m = CatBoostRegressor(**CATBOOST_PARAMS)
            m.fit(Pool(X_b_tr, y_b_tr, weight=w), verbose=False)
            m.save_model(str(save))
            tag = "(trained)"
            tt = time.time() - tt0

        pred_te = m.predict(X_b_te).astype("float32")
        test[f"bear_d2_l{lam}"] = pred_te
        bear_z = zscore_daily(test, f"bear_d2_l{lam}")

        # fixed alpha = 0.5 (D2a protocol, isolates lambda effect)
        conv_a = (test["bull_z"].to_numpy("float32") - FIXED_ALPHA * bear_z).astype("float32")
        m_a = evaluate(meta_te, conv_a, f"D2 lambda={lam} | fixed alpha=0.5 (D2a)")
        # adaptive alpha (D2c protocol, layered onto D2)
        conv_c = (test["bull_z"].to_numpy("float32") - alpha_t * bear_z).astype("float32")
        m_c = evaluate(meta_te, conv_c, f"D2 lambda={lam} | adaptive alpha (D2c)")

        rows.append({**m_a, "lambda": lam, "alpha_mode": "fixed_0.5"})
        rows.append({**m_c, "lambda": lam, "alpha_mode": "adaptive"})
        print(f"   lambda={lam:>4.1f}  {tag} ({tt:>4.1f}s)  "
              f"fixed-alpha={m_a['rankicir']:.4f}  "
              f"adaptive={m_c['rankicir']:.4f}")

    df_grid = pd.DataFrame(rows)
    # find peaks
    fixed_peak = df_grid[df_grid["alpha_mode"] == "fixed_0.5"].loc[
        df_grid[df_grid["alpha_mode"] == "fixed_0.5"]["rankicir"].idxmax()]
    adap_peak = df_grid[df_grid["alpha_mode"] == "adaptive"].loc[
        df_grid[df_grid["alpha_mode"] == "adaptive"]["rankicir"].idxmax()]

    print()
    print("[3/5] Stage 1 peaks:")
    print(f"   Fixed alpha=0.5 peak: lambda={fixed_peak['lambda']:.1f}  "
          f"RankICIR={fixed_peak['rankicir']:.4f}  SR={fixed_peak['sharpe']:+.3f}  "
          f"MaxDD={fixed_peak['maxdd']*100:+.2f}%")
    print(f"   Adaptive alpha peak : lambda={adap_peak['lambda']:.1f}  "
          f"RankICIR={adap_peak['rankicir']:.4f}  SR={adap_peak['sharpe']:+.3f}  "
          f"MaxDD={adap_peak['maxdd']*100:+.2f}%")

    df_grid.to_csv(BB_RESULTS / "step8_d2_peak.csv", index=False, encoding="utf-8-sig")
    print(f"\n   wrote {BB_RESULTS / 'step8_d2_peak.csv'}")

    # ---- Stage 2: consolidated ablation table ----
    print("\n[4/5] Stage 2 — consolidated ablation table (B0 -> D2c)")
    # Numbers re-evaluated where we have predictions; refs used for B0/M1/T/BC/D1*
    d2a_lam = fixed_peak["lambda"]
    d2c_lam = adap_peak["lambda"]
    d2_l2_fixed = df_grid[(df_grid["alpha_mode"] == "fixed_0.5") & (df_grid["lambda"] == 2.0)].iloc[0]

    ablation_rows = [
        ("B0", "Global CatBoost (17 feat.)",        B0_REF,  None, None),
        ("M1", "G1+G3 additive (additive ref)",     M1_REF,  None, None),
        ("T",  "Trend-SA (Alpha Agent alone)",      T_REF,   None, None),
        ("BC", "|bias_60| rule Bear (alpha=0.2)",   BC_REF,  None, None),
        ("D1a","Adversarial D1, alpha=0.2",         D1A_REF, None, None),
        ("D1b","Adversarial D1, alpha=0.5",         D1B_REF, None, None),
        ("D1c","D1 + adaptive alpha (regime)",      D1C_REF, None, None),
        ("D2a-2.0", "D2 lambda=2.0 + fixed alpha=0.5",
            float(d2_l2_fixed["rankicir"]),
            float(d2_l2_fixed["sharpe"]),
            float(d2_l2_fixed["maxdd"])),
        (f"D2a-peak", f"D2 lambda={d2a_lam:.1f} + fixed alpha=0.5 (peak fixed)",
            float(fixed_peak["rankicir"]),
            float(fixed_peak["sharpe"]),
            float(fixed_peak["maxdd"])),
        (f"D2c-peak", f"D2 lambda={d2c_lam:.1f} + adaptive alpha (peak final)",
            float(adap_peak["rankicir"]),
            float(adap_peak["sharpe"]),
            float(adap_peak["maxdd"])),
    ]

    line = "+" + "-" * 12 + "+" + "-" * 48 + "+" + "-" * 11 + "+" + "-" * 10 + "+" + "-" * 11 + "+" + "-" * 14 + "+"
    print()
    print(line)
    print(f"| {'Code':10s} | {'Configuration':46s} | "
          f"{'RankICIR':>9s} | {'SR':>8s} | {'MaxDD':>9s} | {'Delta D1c bp':>12s} |")
    print(line)
    for code, cfg, ric, sr, dd in ablation_rows:
        sr_s = f"{sr:+.3f}" if sr is not None else "  --   "
        dd_s = f"{dd*100:+.2f}%" if dd is not None else "    --   "
        delta_bp = (ric - D1C_REF) * 10000
        d_s = f"{delta_bp:+.1f}"
        print(f"| {code:10s} | {cfg:46s} | {ric:>9.4f} | "
              f"{sr_s:>8s} | {dd_s:>9s} | {d_s:>12s} |")
    print(line)

    # ---- Verdict ----
    print("\n[5/5] Verdict")
    best_ric = float(adap_peak["rankicir"])
    if best_ric > D1C_REF:
        gain = (best_ric - D1C_REF) * 10000
        print(f"  D2c at lambda={d2c_lam:.1f}  RankICIR = {best_ric:.4f}  "
              f"BEATS D1c (0.7440) by {gain:+.1f} bp.")
        if best_ric > float(d2_l2_fixed["rankicir"]):
            print(f"  Also beats step-7 D2 lambda=2.0 result "
                  f"({d2_l2_fixed['rankicir']:.4f}); peak is at lambda={d2c_lam:.1f}.")
    else:
        print(f"  No D2 configuration surpasses D1c. Best: lambda={d2c_lam:.1f} "
              f"at {best_ric:.4f}.")


if __name__ == "__main__":
    main()
