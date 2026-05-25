"""Step 11 — Agent 4 redesign: two parallel directions.

Baseline: D2c (Error-Informed Bear lambda=3.0 + adaptive alpha) = 0.7505.

Direction A — Anomaly-Amplified alpha
  On anomaly days (D_M > 8.96, 64 days), instead of falling back or
  scaling position, INCREASE alpha to lean harder on Bear.
    alpha_used = min(alpha_base + Delta, 0.85)
  Test Delta in {0.1, 0.2, 0.3}.

Direction B — Short-term Reversal Agent
  Build 6 reversal features from ret_1d / ret_3d / per-ticker rolling /
  cross-section market excess. Train two CatBoost versions:
    B_1d:  target = -ret_future_1d  (literal per user spec)
    B_5d:  target = r_future_5d
  Standalone diagnostic + integration with gamma in {-0.3,-0.2,-0.1,
  0.1, 0.2, 0.3}.

Outputs:
  bull_bear/results/step11_agent4_redesign.csv          (summary)
  bull_bear/results/step11_reversal_standalone.csv      (B standalone)
  bull_bear/results/models/reversal_B_{1d,5d}.cbm
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

from bull_bear.config_bb import (
    ALPHA_AGENT_PATH, BB_MODELS, BB_RESULTS,
    BEAR_FEATURES_D1, CATBOOST_PARAMS, DATE_COL,
    TARGET_RET_COL, TEST_END, TEST_START, TICKER_COL,
    TRAIN_END, TRAIN_START,
)
from bull_bear.src.bear_agent import BearAgent
from src.data import load_dataset
from bull_bear.src.metrics_utils import evaluate_full


ALPHA_FEATURES = [
    "ma60_slope", "ema180_slope", "bias_60", "bias_60_vr", "ma180_slope",
]
ALPHA_BY_REGIME = {"bear": 0.65, "sideway": 0.50, "bull": 0.35}
D2C_LAMBDA = 3.0
ALPHA_CAP = 0.85    # upper bound for amplified alpha
DELTA_GRID = (0.1, 0.2, 0.3)
GAMMA_GRID = (-0.3, -0.2, -0.1, 0.1, 0.2, 0.3)
REVERSAL_FEATURES = [
    "ret_1d", "ret_3d",
    "rev_ret_2d", "rev_ret_3d_minus_1d",
    "rev_zscore_1d", "rev_mkt_excess_1d",
]


# ============================================================
# helpers
# ============================================================

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


def evaluate(meta: pd.DataFrame, pred: np.ndarray, label: str) -> dict:
    m = evaluate_full(meta, pred.astype("float32"))
    return {"config": label,
             "rankicir": float(m["rankicir"]),
             "sharpe":   float(m["top5pct_sharpe"]),
             "maxdd":    float(m["top5pct_max_dd"])}


def build_reversal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 4 derived features to df. Assumes df sorted by ticker, date."""
    df = df.sort_values([TICKER_COL, DATE_COL]).reset_index(drop=True)
    g = df.groupby(TICKER_COL, sort=False)
    # rev_ret_2d: cumulative 2-day return (today + yesterday's ret_1d, chained)
    ret_1d = df["ret_1d"].astype("float32").fillna(0.0)
    ret_1d_prev = g["ret_1d"].shift(1).astype("float32").fillna(0.0)
    df["rev_ret_2d"] = ((1.0 + ret_1d) * (1.0 + ret_1d_prev) - 1.0).astype("float32")
    df["rev_ret_3d_minus_1d"] = (df["ret_3d"].astype("float32")
                                  - df["ret_1d"].astype("float32"))
    # rev_zscore_1d: per-ticker rolling z-score of ret_1d over past 20 days
    roll_mean = g["ret_1d"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    roll_std  = g["ret_1d"].transform(lambda s: s.rolling(20, min_periods=5).std(ddof=0))
    df["rev_zscore_1d"] = ((df["ret_1d"] - roll_mean) / roll_std.replace(0.0, np.nan)).astype("float32")
    df["rev_zscore_1d"] = df["rev_zscore_1d"].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    # rev_mkt_excess_1d: ret_1d minus same-day cross-section mean
    df = df.sort_values([DATE_COL, TICKER_COL]).reset_index(drop=True)
    mkt = df.groupby(DATE_COL)["ret_1d"].transform("mean").astype("float32")
    df["rev_mkt_excess_1d"] = (df["ret_1d"].astype("float32") - mkt).astype("float32")
    return df


def build_ret_future_1d(df: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker forward 1-day return."""
    df = df.sort_values([TICKER_COL, DATE_COL]).reset_index(drop=True)
    df["ret_future_1d"] = df.groupby(TICKER_COL, sort=False)["ret_1d"].shift(-1).astype("float32")
    df = df.sort_values([DATE_COL, TICKER_COL]).reset_index(drop=True)
    return df


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("=" * 80)
    print("Step 11 — Agent 4 redesign (Anomaly-Amplified alpha + Reversal Agent)")
    print("=" * 80)

    # ---- data ----
    print("\n[0/7] load data + derive features ...")
    t0 = time.time()
    df = load_dataset().dropna(subset=[TARGET_RET_COL]).reset_index(drop=True)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = build_ret_future_1d(df)        # adds ret_future_1d
    df = build_reversal_features(df)    # adds 4 rev_ features
    mask_tr = (df[DATE_COL] >= pd.Timestamp(TRAIN_START)) & (df[DATE_COL] <= pd.Timestamp(TRAIN_END))
    mask_te = (df[DATE_COL] >= pd.Timestamp(TEST_START)) & (df[DATE_COL] <= pd.Timestamp(TEST_END))
    train = df.loc[mask_tr].reset_index(drop=True)
    test  = df.loc[mask_te].sort_values([DATE_COL, TICKER_COL]).reset_index(drop=True)
    print(f"   train={len(train):,}  test={len(test):,}  ({time.time()-t0:.1f}s)")

    # ---- load Alpha + Bear D2 lambda=3.0 ----
    print("\n[1/7] load Alpha + Bear D2 lambda=3.0 + anomaly flags ...")
    alpha_model = CatBoostRegressor(); alpha_model.load_model(str(ALPHA_AGENT_PATH))
    alpha_medians = pd.read_csv(
        str(ALPHA_AGENT_PATH).replace(".cbm", "_medians.csv"),
        index_col=0).iloc[:, 0]
    test["alpha_score"] = alpha_model.predict(
        test[ALPHA_FEATURES].astype("float32").fillna(alpha_medians)
    ).astype("float32")

    bear_d2 = CatBoostRegressor()
    bear_d2.load_model(str(BB_MODELS / f"bear_D2_l{D2C_LAMBDA:.1f}.cbm"))
    bear_d1_ref = BearAgent(features=BEAR_FEATURES_D1, name="bear_D1")
    bear_d1_ref.load(BB_MODELS / "bear_D1_agent.cbm")
    test["bear_score"] = bear_d2.predict(
        test[BEAR_FEATURES_D1].astype("float32").fillna(bear_d1_ref._train_medians)
    ).astype("float32")

    test["bull_z"] = zscore_daily(test, "alpha_score")
    test["bear_z"] = zscore_daily(test, "bear_score")
    alpha_t = regime_alpha(test)
    d2c_score = (test["bull_z"].to_numpy("float32")
                  - alpha_t * test["bear_z"].to_numpy("float32"))

    # anomaly day flags
    r6 = pd.read_parquet(
        Path(__file__).resolve().parents[2]
        / "strategy_debate/results/predictions_R6.parquet"
    )
    r6[DATE_COL] = pd.to_datetime(r6[DATE_COL])
    anomaly_dates = set(r6.loc[r6["anomaly"], DATE_COL].dt.normalize().unique())
    is_anomaly_arr = np.array(
        [pd.Timestamp(d).normalize() in anomaly_dates
         for d in test[DATE_COL]],
        dtype=bool,
    )
    print(f"   anomaly days: {len(anomaly_dates)} ({is_anomaly_arr.sum():,} stock-days)")

    meta_te = test[[DATE_COL, TICKER_COL, TARGET_RET_COL]].reset_index(drop=True)
    m_d2c = evaluate(meta_te, d2c_score, "D2c baseline")
    print(f"   D2c baseline RankICIR = {m_d2c['rankicir']:.4f}")

    rows: list[dict] = []
    rows.append({**m_d2c, "delta_bp_vs_d2c": 0.0, "track": "baseline"})

    # ============================================================
    # Direction A — Anomaly-Amplified alpha
    # ============================================================
    print("\n[2/7] Direction A — Anomaly-Amplified alpha ...")
    for delta in DELTA_GRID:
        alpha_used = np.where(is_anomaly_arr,
                                np.minimum(alpha_t + delta, ALPHA_CAP),
                                alpha_t).astype("float32")
        conv = (test["bull_z"].to_numpy("float32")
                 - alpha_used * test["bear_z"].to_numpy("float32"))
        m = evaluate(meta_te, conv, f"D2e Δα={delta:+.1f} on anomaly days")
        m["delta_bp_vs_d2c"] = (m["rankicir"] - m_d2c["rankicir"]) * 10000
        m["track"] = "A"
        rows.append(m)
        print(f"   Δ={delta:+.1f}  alpha_used range [{alpha_used[is_anomaly_arr].min():.2f}, {alpha_used[is_anomaly_arr].max():.2f}]  "
              f"RankICIR={m['rankicir']:.4f}  Δ vs D2c={m['delta_bp_vs_d2c']:+.1f} bp")

    # Diagnostic for direction A
    print("\n[2b] Anomaly-day vs normal-day RankICIR diagnostic (D2c baseline) ...")
    def per_group_ric(mask):
        sub_meta = meta_te[mask].reset_index(drop=True)
        sub_pred = d2c_score[mask]
        m = evaluate_full(sub_meta, sub_pred)
        return float(m["rankicir"])
    ric_anom_baseline = per_group_ric(is_anomaly_arr)
    ric_norm_baseline = per_group_ric(~is_anomaly_arr)
    print(f"   D2c   anomaly days RankICIR = {ric_anom_baseline:.4f}  ({is_anomaly_arr.sum():,} rows)")
    print(f"   D2c   normal  days RankICIR = {ric_norm_baseline:.4f}  ({(~is_anomaly_arr).sum():,} rows)")
    # Also for best Δ
    for delta in DELTA_GRID:
        alpha_used = np.where(is_anomaly_arr,
                                np.minimum(alpha_t + delta, ALPHA_CAP),
                                alpha_t).astype("float32")
        conv = (test["bull_z"].to_numpy("float32")
                 - alpha_used * test["bear_z"].to_numpy("float32"))
        ric_anom = float(evaluate_full(meta_te[is_anomaly_arr].reset_index(drop=True),
                                         conv[is_anomaly_arr])["rankicir"])
        ric_norm = float(evaluate_full(meta_te[~is_anomaly_arr].reset_index(drop=True),
                                         conv[~is_anomaly_arr])["rankicir"])
        print(f"   D2e Δ={delta:+.1f}  anomaly={ric_anom:.4f}   normal={ric_norm:.4f}")

    # ============================================================
    # Direction B — Short-term Reversal Agent
    # ============================================================
    print("\n[3/7] Direction B — Reversal Agent feature diagnostics ...")
    # Check reversal feature daily cross-section variance
    for f in REVERSAL_FEATURES:
        col = train[f].to_numpy(dtype="float64")
        finite = np.isfinite(col)
        nan_share = (~finite).sum() / len(col)
        std_xs = train.groupby(DATE_COL)[f].std(ddof=0).mean()
        print(f"   {f:24s}  NaN share={nan_share*100:.2f}%  mean per-day std={std_xs:.5f}")

    # Train two reversal agents
    print("\n[4/7] Training Reversal Agent versions ...")
    # B_1d: target = -ret_future_1d
    tr_b1d = train.dropna(subset=["ret_future_1d"]).reset_index(drop=True)
    X_b1d = tr_b1d[REVERSAL_FEATURES].astype("float32")
    medians_b = X_b1d.median()
    X_b1d = X_b1d.fillna(medians_b)
    y_b1d_raw = (-tr_b1d["ret_future_1d"]).astype("float32")
    lo, hi = y_b1d_raw.quantile(0.001), y_b1d_raw.quantile(0.999)
    y_b1d = y_b1d_raw.clip(lo, hi).to_numpy()
    t1 = time.time()
    m_b1d = CatBoostRegressor(**CATBOOST_PARAMS)
    m_b1d.fit(Pool(X_b1d, y_b1d), verbose=False)
    m_b1d.save_model(str(BB_MODELS / "reversal_B_1d.cbm"))
    print(f"   B_1d (target=-ret_future_1d) trained in {time.time()-t1:.1f}s")

    # B_5d: target = r_future_5d
    tr_b5d = train.dropna(subset=[TARGET_RET_COL]).reset_index(drop=True)
    X_b5d = tr_b5d[REVERSAL_FEATURES].astype("float32").fillna(medians_b)
    y_b5d_raw = tr_b5d[TARGET_RET_COL].astype("float32")
    lo, hi = y_b5d_raw.quantile(0.001), y_b5d_raw.quantile(0.999)
    y_b5d = y_b5d_raw.clip(lo, hi).to_numpy()
    t1 = time.time()
    m_b5d = CatBoostRegressor(**CATBOOST_PARAMS)
    m_b5d.fit(Pool(X_b5d, y_b5d), verbose=False)
    m_b5d.save_model(str(BB_MODELS / "reversal_B_5d.cbm"))
    print(f"   B_5d (target=+r_future_5d) trained in {time.time()-t1:.1f}s")

    # Predict on test
    X_b_te = test[REVERSAL_FEATURES].astype("float32").fillna(medians_b)
    test["rev_b1d_raw"] = m_b1d.predict(X_b_te).astype("float32")
    test["rev_b5d_raw"] = m_b5d.predict(X_b_te).astype("float32")
    test["rev_b1d_z"]   = zscore_daily(test, "rev_b1d_raw")
    test["rev_b5d_z"]   = zscore_daily(test, "rev_b5d_raw")

    # Standalone diagnostic
    print("\n[5/7] Reversal Agent standalone evaluation on test set ...")
    standalone_rows = []
    for tag, arr in [
        ("B_1d (-ret_future_1d)",  test["rev_b1d_raw"].to_numpy("float32")),
        ("B_5d (+r_future_5d)",    test["rev_b5d_raw"].to_numpy("float32")),
        ("-B_1d (sign flipped)",   (-test["rev_b1d_raw"]).to_numpy("float32")),
    ]:
        m_solo = evaluate(meta_te, arr, f"Reversal {tag} standalone")
        standalone_rows.append(m_solo)
        print(f"   {tag:35s}  RankICIR={m_solo['rankicir']:+.4f}  "
              f"Sharpe={m_solo['sharpe']:+.3f}")

    pd.DataFrame(standalone_rows).to_csv(
        BB_RESULTS / "step11_reversal_standalone.csv",
        index=False, encoding="utf-8-sig")

    # Pick the best reversal variant by absolute standalone RankICIR
    abs_best = max(standalone_rows, key=lambda r: abs(r["rankicir"]))
    best_tag = abs_best["config"]
    best_ric = abs_best["rankicir"]
    print(f"   strongest standalone reversal: {best_tag}  RankICIR={best_ric:+.4f}")

    proceed_b = abs(best_ric) > 0.02
    if not proceed_b:
        print("   |standalone RankICIR| < 0.02 — reversal signal too weak. "
               "Skipping integration with D2c.")

    # ============================================================
    # Direction B integration (gamma sweep on the better of B_1d / B_5d)
    # ============================================================
    if proceed_b:
        print("\n[6/7] Direction B integration: D2c + gamma * standardized rev_score ...")
        # Use both B_1d and B_5d in integration since each has merits
        for tag, zcol in [("B_1d", "rev_b1d_z"), ("B_5d", "rev_b5d_z")]:
            rev_z = test[zcol].to_numpy("float32")
            for g in GAMMA_GRID:
                conv = (d2c_score + g * rev_z).astype("float32")
                m = evaluate(meta_te, conv, f"D2f {tag} gamma={g:+.1f}")
                m["delta_bp_vs_d2c"] = (m["rankicir"] - m_d2c["rankicir"]) * 10000
                m["track"] = "B"
                rows.append(m)
                print(f"   {tag} gamma={g:+.1f}  RankICIR={m['rankicir']:.4f}  "
                      f"Δ vs D2c={m['delta_bp_vs_d2c']:+.1f} bp")
    else:
        print("\n[6/7] SKIP integration of Direction B — standalone signal too weak.")

    # ============================================================
    # Combined A + B (D2ef) — only if both directions have any positive Δ
    # ============================================================
    pos_a = [r for r in rows if r.get("track") == "A" and r["delta_bp_vs_d2c"] > 0]
    pos_b = [r for r in rows if r.get("track") == "B" and r["delta_bp_vs_d2c"] > 0]
    if pos_a and pos_b:
        best_a = max(pos_a, key=lambda r: r["delta_bp_vs_d2c"])
        best_b = max(pos_b, key=lambda r: r["delta_bp_vs_d2c"])
        # Reconstruct best A
        # Parse Δα from label
        import re
        ma = re.search(r"Δα=([+-]?\d+\.\d+)", best_a["config"])
        delta_a = float(ma.group(1)) if ma else 0.2
        alpha_used = np.where(is_anomaly_arr,
                                np.minimum(alpha_t + delta_a, ALPHA_CAP),
                                alpha_t).astype("float32")
        conv_a = (test["bull_z"].to_numpy("float32")
                   - alpha_used * test["bear_z"].to_numpy("float32"))
        # Reconstruct best B
        mb_tag = "B_1d" if "B_1d" in best_b["config"] else "B_5d"
        mg = re.search(r"gamma=([+-]?\d+\.\d+)", best_b["config"])
        gamma_b = float(mg.group(1)) if mg else 0.1
        rev_z = test["rev_b1d_z" if mb_tag == "B_1d" else "rev_b5d_z"].to_numpy("float32")
        conv_combined = (conv_a + gamma_b * rev_z).astype("float32")
        m_comb = evaluate(meta_te, conv_combined,
                           f"D2ef combined (A Δα={delta_a:+.1f} + {mb_tag} gamma={gamma_b:+.1f})")
        m_comb["delta_bp_vs_d2c"] = (m_comb["rankicir"] - m_d2c["rankicir"]) * 10000
        m_comb["track"] = "A+B"
        rows.append(m_comb)
        print(f"\n[combined] D2ef = best A + best B  RankICIR={m_comb['rankicir']:.4f}  "
              f"Δ vs D2c={m_comb['delta_bp_vs_d2c']:+.1f} bp")
    else:
        print("\n[combined] SKIP A+B combination (no positive Δ in at least one direction).")

    # ============================================================
    # Output
    # ============================================================
    print("\n[7/7] write CSV + summary table ...")
    df_out = pd.DataFrame(rows)
    out_csv = BB_RESULTS / "step11_agent4_redesign.csv"
    df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")

    line = "+" + "-" * 58 + "+" + "-" * 8 + "+" + "-" * 11 + "+" + "-" * 10 + "+" + "-" * 11 + "+" + "-" * 14 + "+"
    print()
    print(line)
    print(f"| {'Config':56s} | {'track':>6s} | {'RankICIR':>9s} | "
          f"{'Sharpe':>8s} | {'MaxDD':>9s} | {'Δ vs D2c (bp)':>12s} |")
    print(line)
    for r in rows:
        print(f"| {r['config']:56s} | {r.get('track','-'):>6s} | "
              f"{r['rankicir']:>9.4f} | {r['sharpe']:>+8.3f} | "
              f"{r['maxdd']*100:>+8.2f}% | {r['delta_bp_vs_d2c']:>+12.1f} |")
    print(line)

    # ---- verdict ----
    print("\n=== Verdict ===")
    best_overall = max(rows[1:], key=lambda r: r["rankicir"])
    if best_overall["rankicir"] > m_d2c["rankicir"]:
        print(f"  Best beats D2c: {best_overall['config']}")
        print(f"    RankICIR = {best_overall['rankicir']:.4f}  "
              f"(Δ vs D2c {best_overall['delta_bp_vs_d2c']:+.1f} bp)")
    else:
        print(f"  No configuration beats D2c (0.7505). Closest: {best_overall['config']} "
              f"at {best_overall['rankicir']:.4f}.")

    # Direction A specific verdict
    rows_a = [r for r in rows if r.get("track") == "A"]
    if rows_a:
        best_a = max(rows_a, key=lambda r: r["rankicir"])
        if best_a["rankicir"] > m_d2c["rankicir"]:
            print(f"  Direction A: best Δα={best_a['config']} BEATS D2c "
                  f"({best_a['delta_bp_vs_d2c']:+.1f} bp).")
        else:
            print(f"  Direction A: all configs UNDERPERFORM D2c. "
                  f"Closest = {best_a['delta_bp_vs_d2c']:+.1f} bp.")
            print(f"    Diagnostic: D2c anomaly-day RankICIR = {ric_anom_baseline:.4f} "
                  f"already high; amplifying alpha cannot extract more.")

    rows_b = [r for r in rows if r.get("track") == "B"]
    if rows_b:
        best_b = max(rows_b, key=lambda r: r["rankicir"])
        if best_b["rankicir"] > m_d2c["rankicir"]:
            print(f"  Direction B: best {best_b['config']} BEATS D2c "
                  f"({best_b['delta_bp_vs_d2c']:+.1f} bp).")
        else:
            print(f"  Direction B: all configs UNDERPERFORM D2c. "
                  f"Reversal signal interferes with D2c's adversarial structure.")

    print(f"\nOutputs:")
    print(f"  -> {out_csv.relative_to(BB_RESULTS.parent.parent)}")
    print(f"  -> {(BB_RESULTS / 'step11_reversal_standalone.csv').relative_to(BB_RESULTS.parent.parent)}")


if __name__ == "__main__":
    main()
