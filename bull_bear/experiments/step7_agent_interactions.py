"""Step 7 — Agent interaction mechanism experiments.

Three parallel innovations on top of the D1c baseline (RankICIR = 0.744):

  D2  Error-Informed Bear
       Sample weight = 1 + λ * |alpha_rank - actual_rank| on train.
       Train Bear with these weights at λ in {0.5, 1.0, 2.0}.

  D3  Disagreement-Conditioned Arbitration
       On train, compute disagreement = |z(alpha) - z(bear_D1)|.
       Train an arbitration CatBoost on rows with disagreement > 0.5,
       using the union of G1+G3+G4 features and target = r_future_5.
       At inference: high-disagree -> arb_score; else D1c conviction.

  D4  Residual Correction
       In-sample D1c residuals on train -> train a correction agent
       on G2 features (cross-period differential features not used elsewhere).
       Final = D1c_score + gamma * standardize(correction_score).

Baseline D1c (alpha - alpha(t)*bear, adaptive alpha by regime) reproduced
on test for direct comparison.

Outputs:
  bull_bear/results/step7_agent_interactions.csv
  bull_bear/results/models/bear_D2_l{0.5,1.0,2.0}.cbm
  bull_bear/results/models/arb_D3.cbm
  bull_bear/results/models/correction_D4.cbm
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


# ---- feature sets ----
ALPHA_FEATURES = [
    "ma60_slope", "ema180_slope", "bias_60", "bias_60_vr", "ma180_slope",
]
# Bear D1's G1+G3 set (re-export)
BEAR_FEATURES = BEAR_FEATURES_D1

# G3 D4 residual-correction features (cross-period diffs not used elsewhere)
G2_RESIDUAL_FEATURES = [
    "ret_3d_minus_10d", "ret_1d_minus_3d", "ret_10d", "momentum_change",
]

# D3 arbitration features (union of Alpha + Bear features, deduped)
ARB_FEATURES = sorted(set(ALPHA_FEATURES + BEAR_FEATURES_D1))

# ---- pipeline params (match the rest of the paper) ----
D1_ALPHA = 0.5
ALPHA_BY_REGIME = {"bear": 0.65, "sideway": 0.50, "bull": 0.35}
D1C_REF = 0.744


# ============================================================
# helpers
# ============================================================

def zscore_daily(panel: pd.DataFrame, col: str) -> np.ndarray:
    """Per-day cross-section z-score with NaN -> 0."""
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
    """Per-day percentile rank in [0,1]."""
    out = np.full(len(panel), 0.5, dtype="float64")
    for d, g in panel.groupby(DATE_COL):
        v = g[col].to_numpy(dtype="float64")
        # rankdata uses fractional ranks; divide by N for percentile
        r = rankdata(v, method="average") / max(len(v), 1)
        out[g.index.to_numpy()] = r
    return out.astype("float32")


def predict_alpha(panel: pd.DataFrame, alpha_model, alpha_medians) -> np.ndarray:
    X = panel[ALPHA_FEATURES].astype("float32").fillna(alpha_medians)
    return alpha_model.predict(X).astype("float32")


def evaluate(meta: pd.DataFrame, pred: np.ndarray, label: str) -> dict:
    m = evaluate_full(meta, pred.astype("float32"))
    return {"config": label,
             "rankicir": float(m["rankicir"]),
             "sharpe": float(m["top5pct_sharpe"]),
             "maxdd": float(m["top5pct_max_dd"])}


def regime_alpha(panel: pd.DataFrame) -> np.ndarray:
    """Map macro_regime_3 to adaptive alpha(t)."""
    regime = panel["macro_regime_3"].astype(str).to_numpy()
    out = np.array([ALPHA_BY_REGIME.get(r, 0.5) for r in regime], dtype="float32")
    return out


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("=" * 80)
    print("Step 7 — Agent interaction experiments (D2 / D3 / D4)")
    print("=" * 80)

    # ---- 0. data + targets ----
    print("\n[0/6] load + build max_drawdown_5d target ...")
    t0 = time.time()
    df = load_dataset().dropna(subset=[TARGET_RET_COL]).reset_index(drop=True)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = build_max_drawdown_5d(df, ret_col="ret_1d", window=5)
    df = cross_section_zscore(df, "max_drawdown_5d")
    # mask AFTER sort+reset inside the target builder
    mask_tr = (df[DATE_COL] >= pd.Timestamp(TRAIN_START)) & (df[DATE_COL] <= pd.Timestamp(TRAIN_END))
    mask_te = (df[DATE_COL] >= pd.Timestamp(TEST_START)) & (df[DATE_COL] <= pd.Timestamp(TEST_END))
    train = df.loc[mask_tr].reset_index(drop=True)
    test  = df.loc[mask_te].sort_values([DATE_COL, TICKER_COL]).reset_index(drop=True)
    print(f"   train={len(train):,}  test={len(test):,} ({test[DATE_COL].nunique()} days)")
    print(f"   built targets in {time.time()-t0:.1f}s")

    # ---- 1. load Alpha + Bear D1 ----
    print("\n[1/6] load Alpha + Bear D1 ...")
    alpha_model = CatBoostRegressor(); alpha_model.load_model(str(ALPHA_AGENT_PATH))
    alpha_medians = pd.read_csv(
        str(ALPHA_AGENT_PATH).replace(".cbm", "_medians.csv"),
        index_col=0).iloc[:, 0]
    bear_d1 = BearAgent(features=BEAR_FEATURES_D1, name="bear_D1")
    bear_d1.load(BB_MODELS / "bear_D1_agent.cbm")

    # in-sample scores on train (Alpha is trained on this; in-sample by design)
    train["alpha_score"] = predict_alpha(train, alpha_model, alpha_medians)
    train["bear_score"]  = bear_d1.predict_panel(train).astype("float32")

    # test set scores (true OOS)
    test["alpha_score"] = predict_alpha(test, alpha_model, alpha_medians)
    test["bear_score"]  = bear_d1.predict_panel(test).astype("float32")
    test["bull_z"]      = zscore_daily(test, "alpha_score")
    test["bear_z"]      = zscore_daily(test, "bear_score")
    meta_test = test[[DATE_COL, TICKER_COL, TARGET_RET_COL]].reset_index(drop=True)

    # D1c baseline reproduction
    alpha_t = regime_alpha(test)
    d1c_score = (test["bull_z"].to_numpy("float32")
                  - alpha_t * test["bear_z"].to_numpy("float32"))
    m_d1c = evaluate(meta_test, d1c_score, "D1c baseline (adaptive alpha)")
    print(f"   D1c baseline reproduction: RankICIR = {m_d1c['rankicir']:.4f} "
          f"(reference {D1C_REF})")

    rows: list[dict] = []
    rows.append({**m_d1c, "delta_bp_vs_d1c": 0.0})

    # ============================================================
    # D2 — Error-Informed Bear
    # ============================================================
    print("\n[2/6] D2 — Error-Informed Bear ...")
    print("   computing per-day alpha rank error on train ...")
    alpha_rank = pct_rank_daily(train, "alpha_score")
    actual_rank = pct_rank_daily(train, TARGET_RET_COL)
    train["alpha_rank_error"] = np.abs(alpha_rank - actual_rank).astype("float32")
    err_mean = float(train["alpha_rank_error"].mean())
    print(f"   alpha_rank_error mean={err_mean:.4f}  p95={train['alpha_rank_error'].quantile(0.95):.4f}")

    # drop rows missing the target before training
    valid_mask = train["max_drawdown_5d_z"].notna()
    tr2 = train.loc[valid_mask].reset_index(drop=True)
    X_tr = tr2[BEAR_FEATURES_D1].astype("float32")
    bear_medians = X_tr.median()
    X_tr = X_tr.fillna(bear_medians)
    y_tr = tr2["max_drawdown_5d_z"].astype("float32").to_numpy()

    # test predictions
    X_te = test[BEAR_FEATURES_D1].astype("float32").fillna(bear_medians)

    for lam in (0.5, 1.0, 2.0):
        t1 = time.time()
        w = (1.0 + lam * tr2["alpha_rank_error"].to_numpy("float32")).astype("float32")
        m = CatBoostRegressor(**CATBOOST_PARAMS)
        m.fit(Pool(X_tr, y_tr, weight=w), verbose=False)
        save = BB_MODELS / f"bear_D2_l{lam:.1f}.cbm"
        m.save_model(str(save))
        bear_d2 = m.predict(X_te).astype("float32")
        test[f"bear_d2_l{lam}"] = bear_d2
        bear_d2_z = zscore_daily(test, f"bear_d2_l{lam}")
        # apply D1c-style adaptive alpha so this is a true apples-to-apples drop-in
        conv = (test["bull_z"].to_numpy("float32")
                 - alpha_t * bear_d2_z.astype("float32"))
        m_d2 = evaluate(meta_test, conv, f"D2 lambda={lam:.1f} (error-weighted Bear)")
        m_d2["delta_bp_vs_d1c"] = (m_d2["rankicir"] - m_d1c["rankicir"]) * 10000
        rows.append(m_d2)
        print(f"   D2 lambda={lam:.1f}  RankICIR={m_d2['rankicir']:.4f}  "
              f"delta vs D1c = {m_d2['delta_bp_vs_d1c']:+.1f} bp  "
              f"({time.time()-t1:.1f}s)")

    # ============================================================
    # D3 — Disagreement-Conditioned Arbitration
    # ============================================================
    print("\n[3/6] D3 — Disagreement-Conditioned Arbitration ...")
    train["alpha_z_tr"] = zscore_daily(train, "alpha_score")
    train["bear_z_tr"]  = zscore_daily(train, "bear_score")
    train["disagree"]   = (train["alpha_z_tr"] - train["bear_z_tr"]).abs().astype("float32")
    # disagreement threshold = 0.5
    thresh = 0.5
    high_mask = train["disagree"] > thresh
    print(f"   high-disagreement rows on train: {int(high_mask.sum()):,} / "
          f"{len(train):,} = {high_mask.mean()*100:.1f}%")

    arb_tr = train.loc[high_mask].reset_index(drop=True)
    X_arb_tr = arb_tr[ARB_FEATURES].astype("float32")
    arb_medians = X_arb_tr.median()
    X_arb_tr = X_arb_tr.fillna(arb_medians)
    # clip target to mitigate outliers (same as Alpha training)
    y_arb_raw = arb_tr[TARGET_RET_COL].astype("float32")
    lo, hi = y_arb_raw.quantile(0.001), y_arb_raw.quantile(0.999)
    y_arb = y_arb_raw.clip(lo, hi).to_numpy()

    t1 = time.time()
    arb = CatBoostRegressor(**CATBOOST_PARAMS)
    arb.fit(Pool(X_arb_tr, y_arb), verbose=False)
    arb.save_model(str(BB_MODELS / "arb_D3.cbm"))
    print(f"   trained arb in {time.time()-t1:.1f}s")

    # test
    test["disagree_te"] = (test["bull_z"] - test["bear_z"]).abs().astype("float32")
    high_te = (test["disagree_te"] > thresh).to_numpy()
    print(f"   high-disagreement on test: {int(high_te.sum()):,} / "
          f"{len(test):,} = {high_te.mean()*100:.1f}%")

    X_arb_te = test[ARB_FEATURES].astype("float32").fillna(arb_medians)
    arb_score = arb.predict(X_arb_te).astype("float32")
    test["arb_score"] = arb_score
    arb_score_z = zscore_daily(test, "arb_score")

    # conditional conviction
    conv_d3 = np.where(high_te, arb_score_z, d1c_score).astype("float32")
    m_d3 = evaluate(meta_test, conv_d3, "D3 disagreement-conditioned (thresh=0.5)")
    m_d3["delta_bp_vs_d1c"] = (m_d3["rankicir"] - m_d1c["rankicir"]) * 10000
    rows.append(m_d3)
    print(f"   D3  RankICIR={m_d3['rankicir']:.4f}  "
          f"delta vs D1c = {m_d3['delta_bp_vs_d1c']:+.1f} bp")

    # ============================================================
    # D4 — Residual Correction
    # ============================================================
    print("\n[4/6] D4 — Residual Correction (G2 features) ...")
    # D1c residuals on TRAIN (in-sample, by design)
    train["bull_z_tr"] = train["alpha_z_tr"]
    train["bear_z_tr_2"] = train["bear_z_tr"]
    alpha_t_train = regime_alpha(train)
    d1c_train_score = (train["bull_z_tr"].to_numpy("float32")
                        - alpha_t_train * train["bear_z_tr_2"].to_numpy("float32"))
    train["d1c_score"] = d1c_train_score
    # per-day rank of D1c and rank of r_future_5
    d1c_rank   = pct_rank_daily(train, "d1c_score")
    train["actual_rank"] = pct_rank_daily(train, TARGET_RET_COL)
    train["residual"] = (train["actual_rank"] - d1c_rank).astype("float32")
    # z-score per day for stable target
    train = cross_section_zscore(train, "residual")    # creates residual_z

    # train correction agent on G2 features
    valid = train["residual_z"].notna()
    X_c_tr = train.loc[valid, G2_RESIDUAL_FEATURES].astype("float32")
    cor_medians = X_c_tr.median()
    X_c_tr = X_c_tr.fillna(cor_medians)
    y_c = train.loc[valid, "residual_z"].astype("float32").to_numpy()

    t1 = time.time()
    cor = CatBoostRegressor(**CATBOOST_PARAMS)
    cor.fit(Pool(X_c_tr, y_c), verbose=False)
    cor.save_model(str(BB_MODELS / "correction_D4.cbm"))
    print(f"   trained correction in {time.time()-t1:.1f}s, "
          f"train RMSE={np.sqrt(np.mean((cor.predict(X_c_tr) - y_c)**2)):.4f}")

    # test
    X_c_te = test[G2_RESIDUAL_FEATURES].astype("float32").fillna(cor_medians)
    cor_score = cor.predict(X_c_te).astype("float32")
    test["cor_score"] = cor_score
    cor_z = zscore_daily(test, "cor_score")

    for gamma in (0.1, 0.2, 0.3, 0.5):
        final = (d1c_score + gamma * cor_z).astype("float32")
        m_d4 = evaluate(meta_test, final, f"D4 gamma={gamma:.1f} (G2 residual correction)")
        m_d4["delta_bp_vs_d1c"] = (m_d4["rankicir"] - m_d1c["rankicir"]) * 10000
        rows.append(m_d4)
        print(f"   D4 gamma={gamma:.1f}  RankICIR={m_d4['rankicir']:.4f}  "
              f"delta vs D1c = {m_d4['delta_bp_vs_d1c']:+.1f} bp")

    # ============================================================
    # Output
    # ============================================================
    print("\n[5/6] write CSV ...")
    df_out = pd.DataFrame(rows)
    out_csv = BB_RESULTS / "step7_agent_interactions.csv"
    df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # console summary
    print("\n[6/6] summary table ...")
    print()
    line = "+" + "-" * 52 + "+" + "-" * 11 + "+" + "-" * 10 + "+" + "-" * 11 + "+" + "-" * 14 + "+"
    print(line)
    print(f"| {'Config':50s} | {'RankICIR':>9s} | {'SR':>8s} | {'MaxDD':>9s} | {'Δ vs D1c (bp)':>12s} |")
    print(line)
    for r in rows:
        print(f"| {r['config']:50s} | {r['rankicir']:>9.4f} | "
              f"{r['sharpe']:>+8.3f} | {r['maxdd']*100:>+8.2f}% | "
              f"{r['delta_bp_vs_d1c']:>+12.1f} |")
    print(line)

    # verdict
    best = max(rows[1:], key=lambda r: r["rankicir"])    # exclude D1c baseline
    print()
    print("=== Verdict ===")
    if best["rankicir"] > m_d1c["rankicir"]:
        print(f"  Best innovation: {best['config']}")
        print(f"    RankICIR = {best['rankicir']:.4f}  (vs D1c {m_d1c['rankicir']:.4f}, "
              f"delta = {best['delta_bp_vs_d1c']:+.1f} bp)")
        print(f"    BEATS D1c baseline.")
    else:
        print(f"  No configuration surpasses D1c baseline.")
        print(f"  Closest: {best['config']}")
        print(f"    RankICIR = {best['rankicir']:.4f}  (vs D1c {m_d1c['rankicir']:.4f}, "
              f"delta = {best['delta_bp_vs_d1c']:+.1f} bp)")

    print(f"\nOutput: {out_csv.relative_to(BB_RESULTS.parent.parent)}")


if __name__ == "__main__":
    main()
