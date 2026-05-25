"""Step 12 — D2f final-system confirmation (4 tasks).

Task 1: gamma peak search beyond 0.3 (test gamma in {0.35, 0.40, 0.50, 0.60}).
Task 2: 7-year Walk-Forward with full retraining (Alpha + Bear D2 + Reversal per window).
Task 3: bootstrap N=1000 of D2f vs Trend on the canonical hold-out.
Task 4: Reversal Agent diagnostics (features, importance, temporal stability,
        cross-agent correlation).

Outputs:
  bull_bear/results/step12_gamma_peak.csv
  bull_bear/results/step12_walkforward_d2f.csv
  bull_bear/results/step12_bootstrap_d2f.csv
  bull_bear/results/step12_reversal_diagnostics.csv
  bull_bear/results/models/walkforward/{bear_D2_W*, reversal_W*}.cbm
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
from scipy.stats import pearsonr, spearmanr, rankdata

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


ALPHA_FEATURES = ["ma60_slope", "ema180_slope", "bias_60", "bias_60_vr", "ma180_slope"]
ALPHA_BY_REGIME = {"bear": 0.65, "sideway": 0.50, "bull": 0.35}
D2C_LAMBDA = 3.0
GAMMA_EXTENDED = (0.30, 0.35, 0.40, 0.50, 0.60)
REVERSAL_FEATURES = [
    "ret_1d", "ret_3d",
    "rev_ret_2d", "rev_ret_3d_minus_1d",
    "rev_zscore_1d", "rev_mkt_excess_1d",
]

WALKFORWARD_DIR = BB_MODELS / "walkforward"
WALKFORWARD_DIR.mkdir(parents=True, exist_ok=True)


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


def evaluate_q(meta: pd.DataFrame, pred: np.ndarray, label: str) -> dict:
    m = evaluate_full(meta, pred.astype("float32"))
    return {"config": label,
             "rankicir": float(m["rankicir"]),
             "sharpe":   float(m["top5pct_sharpe"]),
             "maxdd":    float(m["top5pct_max_dd"])}


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


def daily_ic_series(meta: pd.DataFrame, pred: np.ndarray) -> pd.Series:
    out = {}
    df = meta.copy()
    df["pred"] = pred
    for d, g in df.groupby(DATE_COL):
        if len(g) < 5: continue
        x = g["pred"].to_numpy(dtype="float64")
        y = g[TARGET_RET_COL].to_numpy(dtype="float64")
        if np.std(x) == 0 or np.std(y) == 0: continue
        rho, _ = spearmanr(x, y)
        out[pd.Timestamp(d)] = float(rho)
    return pd.Series(out).sort_index()


def train_alpha_window(df_train: pd.DataFrame, save_path: Path) -> tuple[CatBoostRegressor, pd.Series]:
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
    return m, medians


def train_bear_d2_window(df_train: pd.DataFrame, alpha_model, alpha_medians,
                            save_path: Path) -> tuple[CatBoostRegressor, pd.Series]:
    """Bear D2 with Alpha-error sample weights, λ=3.0."""
    # Compute alpha score on train
    X_a = df_train[ALPHA_FEATURES].astype("float32").fillna(alpha_medians)
    a_score = alpha_model.predict(X_a).astype("float32")
    tmp = df_train[[DATE_COL, TICKER_COL, TARGET_RET_COL]].copy()
    tmp["alpha_score"] = a_score
    alpha_rank = pct_rank_daily(tmp, "alpha_score")
    actual_rank = pct_rank_daily(tmp, TARGET_RET_COL)
    err = np.abs(alpha_rank - actual_rank).astype("float32")
    # Drop rows missing target
    valid = df_train["max_drawdown_5d_z"].notna()
    tr2 = df_train.loc[valid].reset_index(drop=True)
    err2 = err[valid.to_numpy()]
    X = tr2[BEAR_FEATURES_D1].astype("float32")
    medians = X.median()
    X = X.fillna(medians)
    y = tr2["max_drawdown_5d_z"].astype("float32").to_numpy()
    w = (1.0 + D2C_LAMBDA * err2).astype("float32")
    m = CatBoostRegressor(**CATBOOST_PARAMS)
    m.fit(Pool(X, y, weight=w), verbose=False)
    m.save_model(str(save_path))
    medians.to_csv(str(save_path).replace(".cbm", "_medians.csv"))
    return m, medians


def train_reversal_window(df_train: pd.DataFrame, save_path: Path
                            ) -> tuple[CatBoostRegressor, pd.Series]:
    """Reversal B_5d: target = r_future_5d, features = REVERSAL_FEATURES."""
    valid = df_train[TARGET_RET_COL].notna()
    tr = df_train.loc[valid].reset_index(drop=True)
    X = tr[REVERSAL_FEATURES].astype("float32")
    medians = X.median()
    X = X.fillna(medians)
    y_raw = tr[TARGET_RET_COL].astype("float32")
    lo, hi = y_raw.quantile(0.001), y_raw.quantile(0.999)
    y = y_raw.clip(lo, hi).to_numpy()
    m = CatBoostRegressor(**CATBOOST_PARAMS)
    m.fit(Pool(X, y), verbose=False)
    m.save_model(str(save_path))
    medians.to_csv(str(save_path).replace(".cbm", "_medians.csv"))
    return m, medians


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("=" * 80)
    print("Step 12 — D2f final confirmation")
    print("=" * 80)

    # ---- 0. data + targets + features ----
    print("\n[0/6] load + targets + reversal features ...")
    t0 = time.time()
    df = load_dataset().dropna(subset=[TARGET_RET_COL]).reset_index(drop=True)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = build_max_drawdown_5d(df, ret_col="ret_1d", window=5)
    df = cross_section_zscore(df, "max_drawdown_5d")
    df = build_reversal_features(df)
    mask_tr = (df[DATE_COL] >= pd.Timestamp(TRAIN_START)) & (df[DATE_COL] <= pd.Timestamp(TRAIN_END))
    mask_te = (df[DATE_COL] >= pd.Timestamp(TEST_START)) & (df[DATE_COL] <= pd.Timestamp(TEST_END))
    train = df.loc[mask_tr].reset_index(drop=True)
    test  = df.loc[mask_te].sort_values([DATE_COL, TICKER_COL]).reset_index(drop=True)
    print(f"   train={len(train):,}  test={len(test):,}  ({time.time()-t0:.1f}s)")

    # ---- baseline scores on test (D2c + reversal B_5d) ----
    print("\n[1/6] baseline scores on test (D2c + reversal_B_5d) ...")
    alpha = CatBoostRegressor(); alpha.load_model(str(ALPHA_AGENT_PATH))
    alpha_med = pd.read_csv(str(ALPHA_AGENT_PATH).replace(".cbm", "_medians.csv"),
                              index_col=0).iloc[:, 0]
    test["alpha_score"] = alpha.predict(
        test[ALPHA_FEATURES].astype("float32").fillna(alpha_med)
    ).astype("float32")
    bear = CatBoostRegressor(); bear.load_model(str(BB_MODELS / f"bear_D2_l{D2C_LAMBDA:.1f}.cbm"))
    bear_d1 = BearAgent(features=BEAR_FEATURES_D1, name="bear_D1")
    bear_d1.load(BB_MODELS / "bear_D1_agent.cbm")
    test["bear_score"] = bear.predict(
        test[BEAR_FEATURES_D1].astype("float32").fillna(bear_d1._train_medians)
    ).astype("float32")
    rev = CatBoostRegressor(); rev.load_model(str(BB_MODELS / "reversal_B_5d.cbm"))
    rev_medians = test[REVERSAL_FEATURES].median()
    test["rev_score"] = rev.predict(
        test[REVERSAL_FEATURES].astype("float32").fillna(rev_medians)
    ).astype("float32")

    test["bull_z"] = zscore_daily(test, "alpha_score")
    test["bear_z"] = zscore_daily(test, "bear_score")
    test["rev_z"]  = zscore_daily(test, "rev_score")
    alpha_t = regime_alpha(test)
    d2c_score = (test["bull_z"].to_numpy("float32")
                  - alpha_t * test["bear_z"].to_numpy("float32"))
    meta_te = test[[DATE_COL, TICKER_COL, TARGET_RET_COL]].reset_index(drop=True)
    ric_d2c = float(evaluate_full(meta_te, d2c_score)["rankicir"])
    print(f"   D2c baseline RankICIR = {ric_d2c:.4f}")

    # ============================================================
    # Task 1 — gamma peak search
    # ============================================================
    print("\n[2/6] Task 1 — gamma peak search ...")
    rev_z = test["rev_z"].to_numpy("float32")
    rows_gamma = []
    for g in GAMMA_EXTENDED:
        conv = (d2c_score + g * rev_z).astype("float32")
        m = evaluate_q(meta_te, conv, f"D2f gamma={g:.2f}")
        m["gamma"] = g
        m["delta_bp_vs_d2c"] = (m["rankicir"] - ric_d2c) * 10000
        rows_gamma.append(m)
        print(f"   gamma={g:.2f}  RankICIR={m['rankicir']:.4f}  "
              f"Sharpe={m['sharpe']:+.3f}  MaxDD={m['maxdd']*100:+.2f}%  "
              f"Δ vs D2c={m['delta_bp_vs_d2c']:+.1f} bp")

    peak_row = max(rows_gamma, key=lambda r: r["rankicir"])
    gamma_star = float(peak_row["gamma"])
    print(f"\n   peak: gamma={gamma_star:.2f}  RankICIR={peak_row['rankicir']:.4f}")
    pd.DataFrame(rows_gamma).to_csv(
        BB_RESULTS / "step12_gamma_peak.csv", index=False, encoding="utf-8-sig"
    )

    # ============================================================
    # Task 4 — Reversal diagnostics
    # ============================================================
    print(f"\n[3/6] Task 4 — Reversal Agent diagnostics ...")
    # SHAP top 5 via CatBoost feature importance
    fi = rev.get_feature_importance()
    fi_df = pd.DataFrame({"feature": REVERSAL_FEATURES, "importance": fi}).sort_values(
        "importance", ascending=False).reset_index(drop=True)
    print(f"\n   Reversal Agent features (in training order):")
    for f in REVERSAL_FEATURES:
        print(f"     {f}")
    print(f"\n   Feature importance (top 5):")
    for _, r in fi_df.head(5).iterrows():
        print(f"     {r['feature']:24s}  {r['importance']:>6.2f}")

    # Correlation with Alpha and Bear
    rev_arr = test["rev_score"].to_numpy("float64")
    alpha_arr = test["alpha_score"].to_numpy("float64")
    bear_arr  = test["bear_score"].to_numpy("float64")
    def safe_p(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        return float(pearsonr(a[m], b[m])[0]) if m.sum() > 100 else float("nan")
    def safe_s(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        return float(spearmanr(a[m], b[m])[0]) if m.sum() > 100 else float("nan")
    p_ra = safe_p(rev_arr, alpha_arr); s_ra = safe_s(rev_arr, alpha_arr)
    p_rb = safe_p(rev_arr, bear_arr);  s_rb = safe_s(rev_arr, bear_arr)
    print(f"\n   corr(Reversal, Alpha): Pearson={p_ra:+.4f}  Spearman={s_ra:+.4f}")
    print(f"   corr(Reversal, Bear) : Pearson={p_rb:+.4f}  Spearman={s_rb:+.4f}")

    # Standalone RankICIR + yearly time stability
    rev_solo = evaluate_q(meta_te, rev_arr.astype("float32"), "Reversal solo")
    print(f"\n   Reversal solo  RankICIR={rev_solo['rankicir']:+.4f}  "
          f"Sharpe={rev_solo['sharpe']:+.3f}")
    # yearly RankICIR
    print(f"\n   yearly RankICIR (Reversal solo):")
    by_year = []
    for y in sorted(test[DATE_COL].dt.year.unique()):
        mask = (test[DATE_COL].dt.year == y).to_numpy()
        if mask.sum() < 100: continue
        ric_y = float(evaluate_full(
            meta_te[mask].reset_index(drop=True), rev_arr[mask])["rankicir"])
        print(f"     {y}: RankICIR={ric_y:+.4f}")
        by_year.append({"year": int(y), "rankicir": ric_y})

    diag_rows = [
        {"metric": "feature_importance_top5",
         "value": "; ".join(f"{r['feature']}={r['importance']:.2f}" for _, r in fi_df.head(5).iterrows())},
        {"metric": "corr_reversal_alpha_pearson",  "value": f"{p_ra:+.4f}"},
        {"metric": "corr_reversal_alpha_spearman", "value": f"{s_ra:+.4f}"},
        {"metric": "corr_reversal_bear_pearson",   "value": f"{p_rb:+.4f}"},
        {"metric": "corr_reversal_bear_spearman",  "value": f"{s_rb:+.4f}"},
        {"metric": "standalone_rankicir",          "value": f"{rev_solo['rankicir']:+.4f}"},
        {"metric": "yearly_rankicir",
         "value": "; ".join(f"{r['year']}:{r['rankicir']:+.4f}" for r in by_year)},
    ]
    pd.DataFrame(diag_rows).to_csv(
        BB_RESULTS / "step12_reversal_diagnostics.csv", index=False, encoding="utf-8-sig"
    )

    # ============================================================
    # Task 2 — 7-year Walk-Forward
    # ============================================================
    print(f"\n[4/6] Task 2 — 7-year Walk-Forward (D2f at gamma={gamma_star:.2f}) ...")

    windows = [
        ("W1", "2016-01-01", "2018-12-31", "2019-01-01", "2019-12-31"),
        ("W2", "2016-01-01", "2019-12-31", "2020-01-01", "2020-12-31"),
        ("W3", "2016-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
        ("W4", "2016-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
        ("W5", "2016-01-01", "2021-12-31", "2023-01-01", "2026-01-31"),
    ]

    # cached models per-window
    cached = {}
    def get_window_models(name, tr_s, tr_e):
        """Train or load Alpha + Bear D2 + Reversal for window."""
        if name in cached:
            return cached[name]
        m_tr = (df[DATE_COL] >= pd.Timestamp(tr_s)) & (df[DATE_COL] <= pd.Timestamp(tr_e))
        tr_df = df.loc[m_tr].reset_index(drop=True)

        # Alpha: reuse strategy_debate/results/models/cross_period for W1, walkforward/ for W2-W3, ALPHA_AGENT_PATH for W4-W5
        if name == "W1":
            alpha_path = (Path(__file__).resolve().parents[2]
                           / "strategy_debate/results/models/cross_period/trend_agent_B.cbm")
            alpha_m = CatBoostRegressor(); alpha_m.load_model(str(alpha_path))
            alpha_med = pd.read_csv(str(alpha_path).replace(".cbm", "_medians.csv"),
                                       index_col=0).iloc[:, 0]
        elif name == "W2":
            wf_path = WALKFORWARD_DIR / "alpha_W2.cbm"
            if wf_path.exists():
                alpha_m = CatBoostRegressor(); alpha_m.load_model(str(wf_path))
                alpha_med = pd.read_csv(str(wf_path).replace(".cbm", "_medians.csv"),
                                           index_col=0).iloc[:, 0]
            else:
                alpha_m, alpha_med = train_alpha_window(tr_df, wf_path)
        elif name == "W3":
            wf_path = WALKFORWARD_DIR / "alpha_W3.cbm"
            if wf_path.exists():
                alpha_m = CatBoostRegressor(); alpha_m.load_model(str(wf_path))
                alpha_med = pd.read_csv(str(wf_path).replace(".cbm", "_medians.csv"),
                                           index_col=0).iloc[:, 0]
            else:
                alpha_m, alpha_med = train_alpha_window(tr_df, wf_path)
        else:    # W4/W5 share 2016-2021
            alpha_m = CatBoostRegressor(); alpha_m.load_model(str(ALPHA_AGENT_PATH))
            alpha_med = pd.read_csv(str(ALPHA_AGENT_PATH).replace(".cbm", "_medians.csv"),
                                       index_col=0).iloc[:, 0]

        # Bear D2 per window
        bear_path = WALKFORWARD_DIR / f"bear_D2_{name}.cbm"
        if bear_path.exists():
            bear_m = CatBoostRegressor(); bear_m.load_model(str(bear_path))
            bear_med = pd.read_csv(str(bear_path).replace(".cbm", "_medians.csv"),
                                      index_col=0).iloc[:, 0]
        else:
            print(f"     training Bear D2 for {name} ...")
            t1 = time.time()
            bear_m, bear_med = train_bear_d2_window(tr_df, alpha_m, alpha_med, bear_path)
            print(f"       done in {time.time()-t1:.1f}s")

        # Reversal per window
        rev_path = WALKFORWARD_DIR / f"reversal_{name}.cbm"
        if rev_path.exists():
            rev_m = CatBoostRegressor(); rev_m.load_model(str(rev_path))
            rev_med = pd.read_csv(str(rev_path).replace(".cbm", "_medians.csv"),
                                     index_col=0).iloc[:, 0]
        else:
            print(f"     training Reversal for {name} ...")
            t1 = time.time()
            rev_m, rev_med = train_reversal_window(tr_df, rev_path)
            print(f"       done in {time.time()-t1:.1f}s")

        cached[name] = (alpha_m, alpha_med, bear_m, bear_med, rev_m, rev_med)
        return cached[name]

    walkforward_rows = []
    for name, tr_s, tr_e, te_s, te_e in windows:
        print(f"\n   ===== Window {name}: train {tr_s[:7]} -> {tr_e[:7]} ; test {te_s[:7]} -> {te_e[:7]} =====")
        alpha_m, alpha_med, bear_m, bear_med, rev_m, rev_med = get_window_models(name, tr_s, tr_e)
        m_te = (df[DATE_COL] >= pd.Timestamp(te_s)) & (df[DATE_COL] <= pd.Timestamp(te_e))
        te_df = df.loc[m_te].sort_values([DATE_COL, TICKER_COL]).reset_index(drop=True)
        # Predict
        te_df["a_score"] = alpha_m.predict(
            te_df[ALPHA_FEATURES].astype("float32").fillna(alpha_med)).astype("float32")
        te_df["b_score"] = bear_m.predict(
            te_df[BEAR_FEATURES_D1].astype("float32").fillna(bear_med)).astype("float32")
        te_df["r_score"] = rev_m.predict(
            te_df[REVERSAL_FEATURES].astype("float32").fillna(rev_med)).astype("float32")
        te_df["a_z"] = zscore_daily(te_df, "a_score")
        te_df["b_z"] = zscore_daily(te_df, "b_score")
        te_df["r_z"] = zscore_daily(te_df, "r_score")
        a_t = regime_alpha(te_df)
        meta = te_df[[DATE_COL, TICKER_COL, TARGET_RET_COL]].reset_index(drop=True)
        # Per-year evaluation
        d2c_te = (te_df["a_z"].to_numpy("float32")
                  - a_t * te_df["b_z"].to_numpy("float32"))
        d2f_te = (d2c_te + gamma_star * te_df["r_z"].to_numpy("float32")).astype("float32")
        for y in sorted(te_df[DATE_COL].dt.year.unique()):
            mask = (te_df[DATE_COL].dt.year == y).to_numpy()
            if mask.sum() < 100: continue
            sub_meta = meta[mask].reset_index(drop=True)
            trend_ric = float(evaluate_full(sub_meta, te_df["a_score"].to_numpy("float32")[mask])["rankicir"])
            d2c_ric   = float(evaluate_full(sub_meta, d2c_te[mask])["rankicir"])
            d2f_ric   = float(evaluate_full(sub_meta, d2f_te[mask])["rankicir"])
            walkforward_rows.append({
                "year": int(y), "window": name, "train_range": f"{tr_s[:4]}-{tr_e[:4]}",
                "trend_rankicir": trend_ric,
                "d2c_rankicir":   d2c_ric,
                "d2f_rankicir":   d2f_ric,
                "delta_d2f_vs_d2c":   d2f_ric - d2c_ric,
                "delta_d2f_vs_trend": d2f_ric - trend_ric,
                "d2f_beats_trend":    d2f_ric > trend_ric,
                "d2f_beats_d2c":      d2f_ric > d2c_ric,
            })
            print(f"     {y}: Trend={trend_ric:+.4f}  D2c={d2c_ric:+.4f}  D2f={d2f_ric:+.4f}  "
                  f"Δ(D2f-Trend)={d2f_ric-trend_ric:+.4f}  Δ(D2f-D2c)={d2f_ric-d2c_ric:+.4f}")

    df_wf = pd.DataFrame(walkforward_rows)
    df_wf.to_csv(BB_RESULTS / "step12_walkforward_d2f.csv", index=False, encoding="utf-8-sig")

    # Filter out 2026 (statistical noise)
    df_wf_main = df_wf[df_wf["year"] <= 2025].reset_index(drop=True)
    wins_t = int(df_wf_main["d2f_beats_trend"].sum())
    wins_c = int(df_wf_main["d2f_beats_d2c"].sum())
    n_y = len(df_wf_main)
    print(f"\n   Walk-Forward summary (years 2019-2025, excl. 2026):")
    print(f"     D2f beats Trend: {wins_t}/{n_y}")
    print(f"     D2f beats D2c  : {wins_c}/{n_y}")
    print(f"     mean Δ(D2f-Trend) = {df_wf_main['delta_d2f_vs_trend'].mean():+.4f}")
    print(f"     mean Δ(D2f-D2c)   = {df_wf_main['delta_d2f_vs_d2c'].mean():+.4f}")

    # ============================================================
    # Task 3 — Bootstrap D2f vs Trend (and vs D2c) on canonical hold-out
    # ============================================================
    print(f"\n[5/6] Task 3 — Bootstrap N=1000 on test (D2f gamma={gamma_star:.2f}) ...")
    d2f_score = (d2c_score + gamma_star * test["rev_z"].to_numpy("float32")).astype("float32")
    ic_trend = daily_ic_series(meta_te, test["alpha_score"].to_numpy("float32"))
    ic_d2c   = daily_ic_series(meta_te, d2c_score)
    ic_d2f   = daily_ic_series(meta_te, d2f_score)
    common = sorted(set(ic_trend.index) & set(ic_d2c.index) & set(ic_d2f.index))
    print(f"   common days: {len(common)}")
    a_t = np.array([ic_trend[d] for d in common])
    a_c = np.array([ic_d2c[d]   for d in common])
    a_f = np.array([ic_d2f[d]   for d in common])

    def ricir(a):
        return float(np.mean(a) / (np.std(a, ddof=0) + 1e-9))

    obs_d2f_trend = ricir(a_f) - ricir(a_t)
    obs_d2f_d2c   = ricir(a_f) - ricir(a_c)
    print(f"   observed RankICIR: Trend={ricir(a_t):.4f}  D2c={ricir(a_c):.4f}  D2f={ricir(a_f):.4f}")
    print(f"   observed Δ(D2f - Trend) = {obs_d2f_trend:+.4f}")
    print(f"   observed Δ(D2f - D2c)   = {obs_d2f_d2c:+.4f}")

    rng = np.random.default_rng(42)
    n_boot = 1000
    boot_d2f_trend = np.zeros(n_boot); boot_d2f_d2c = np.zeros(n_boot)
    n_days = len(common)
    for i in range(n_boot):
        idx = rng.integers(0, n_days, size=n_days)
        boot_d2f_trend[i] = ricir(a_f[idx]) - ricir(a_t[idx])
        boot_d2f_d2c[i]   = ricir(a_f[idx]) - ricir(a_c[idx])

    def report(boot, obs, name):
        mean = float(boot.mean())
        lo, hi = float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
        p = float(np.mean(boot <= 0))
        print(f"   [{name}]")
        print(f"     observed = {obs:+.4f}  bootstrap mean = {mean:+.4f}")
        print(f"     95% CI = [{lo:+.4f}, {hi:+.4f}]  p-value = {p:.4f}  "
              f"({'< 0.001' if p == 0 else f'= {p:.4f}'})")
        return {"name": name, "observed": obs, "bootstrap_mean": mean,
                "ci_low": lo, "ci_high": hi, "p_value": p, "n_boot": n_boot, "n_days": n_days}

    boot_rows = [
        report(boot_d2f_trend, obs_d2f_trend, "D2f - Trend"),
        report(boot_d2f_d2c,   obs_d2f_d2c,   "D2f - D2c"),
    ]
    pd.DataFrame(boot_rows).to_csv(
        BB_RESULTS / "step12_bootstrap_d2f.csv", index=False, encoding="utf-8-sig"
    )

    # ============================================================
    # Final report
    # ============================================================
    print("\n[6/6] FINAL summary")
    print()
    line = "+" + "-" * 28 + "+" + "-" * 11 + "+" + "-" * 12 + "+"
    print(line)
    print(f"| {'Item':26s} | {'value':>9s} | {'judgment':>10s} |")
    print(line)
    print(f"| {'gamma peak':26s} | {gamma_star:>9.2f} | {'see CSV':>10s} |")
    print(f"| {'D2f peak RankICIR':26s} | "
          f"{peak_row['rankicir']:>9.4f} | "
          f"{('>D2c' if peak_row['rankicir']>ric_d2c else 'leq D2c'):>10s} |")
    print(f"| {'D2f beats Trend yrs':26s} | "
          f"{wins_t:>2d}/{n_y:<6d} | "
          f"{('PASS' if wins_t >= 6 else 'WARN'):>10s} |")
    print(f"| {'D2f beats D2c yrs':26s} | "
          f"{wins_c:>2d}/{n_y:<6d} | "
          f"{('PASS' if wins_c >= 5 else 'WARN'):>10s} |")
    bvtr = boot_rows[0]
    print(f"| {'p (D2f-Trend)':26s} | "
          f"{bvtr['p_value']:>9.4f} | "
          f"{('PASS' if bvtr['p_value'] < 0.01 else 'WARN'):>10s} |")
    print(line)

    # Detailed Walk-Forward table
    print()
    print("Walk-Forward detail:")
    print(f"  {'Year':>5s}  {'Train':>11s}  {'Trend':>9s}  {'D2c':>9s}  {'D2f':>9s}  "
          f"{'D2f-D2c':>9s}  {'D2f-Trend':>10s}")
    for r in walkforward_rows:
        mk = ""
        if r["year"] == 2020: mk = " (COVID)"
        elif r["year"] == 2022: mk = " (deep bear)"
        elif r["year"] == 2026: mk = " (noise, 6d)"
        print(f"  {r['year']:>5d}{mk:>13s}  {r['train_range']:>11s}  "
              f"{r['trend_rankicir']:>+9.4f}  {r['d2c_rankicir']:>+9.4f}  "
              f"{r['d2f_rankicir']:>+9.4f}  "
              f"{r['delta_d2f_vs_d2c']:>+9.4f}  {r['delta_d2f_vs_trend']:>+10.4f}")

    # Early-warning checks
    print("\n=== Warning scan ===")
    warn = []
    for r in walkforward_rows:
        if r["year"] == 2026: continue    # noise
        if r["delta_d2f_vs_d2c"] < -0.01:
            warn.append(f"   YEAR {r['year']}: D2f underperforms D2c by {r['delta_d2f_vs_d2c']*10000:+.0f} bp")
    if wins_c < n_y:
        warn.append(f"   D2f win rate vs D2c = {wins_c}/{n_y} (target {n_y}/{n_y})")
    if not warn:
        print("   no warnings; D2f appears robust across all evaluation years.")
    else:
        for w in warn:
            print(w)


if __name__ == "__main__":
    main()
