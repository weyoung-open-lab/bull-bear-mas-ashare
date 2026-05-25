"""Step 15 — Walk-Forward comparison of D1c (Sharpe-optimal) vs D2f (IC-optimal).

For each window (W1..W5 trainings reused from step 12 + bear_W1..W3 from step 6),
evaluate both D1c and D2f per calendar year (2019-2025). Compute:
  - RankICIR (cross-section)
  - Top-5% Sharpe (proxy r_future_5/5 daily) — matches step 13 numbers

Then bootstrap (N=1000) on the 2023-2025 hold-out (W5 test) for:
  - D1c Sharpe - D2f Sharpe (expect positive, p < 0.05)
  - D2f RankICIR - D1c RankICIR (expect positive, p < 0.001)

Outputs:
  bull_bear/results/step15_walkforward_d1c_vs_d2f.csv
  bull_bear/results/step15_bootstrap_d1c_vs_d2f.csv
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
    TARGET_RET_COL, TICKER_COL,
)
from bull_bear.src.bear_agent import BearAgent
from bull_bear.src.bear_target import build_max_drawdown_5d
from src.backtest import backtest_topk
from src.data import load_dataset
from bull_bear.src.metrics_utils import evaluate_full


ALPHA_FEATURES = ["ma60_slope", "ema180_slope", "bias_60", "bias_60_vr", "ma180_slope"]
ALPHA_BY_REGIME = {"bear": 0.65, "sideway": 0.50, "bull": 0.35}
GAMMA_BASE = 0.40
GAMMA_DELTA = 0.15
REVERSAL_FEATURES = [
    "ret_1d", "ret_3d",
    "rev_ret_2d", "rev_ret_3d_minus_1d",
    "rev_zscore_1d", "rev_mkt_excess_1d",
]
FRAC = 0.05
TRADING_DAYS_PER_YEAR = 252

WALKFORWARD_DIR = BB_MODELS / "walkforward"


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


def add_reversal_features(df: pd.DataFrame) -> pd.DataFrame:
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


def daily_ic(meta, pred):
    df = meta.copy()
    df["pred"] = pred
    out = {}
    for d, g in df.groupby(DATE_COL):
        if len(g) < 5: continue
        x = g["pred"].to_numpy("float64")
        y = g[TARGET_RET_COL].to_numpy("float64")
        if np.std(x) == 0 or np.std(y) == 0: continue
        rho, _ = spearmanr(x, y)
        out[pd.Timestamp(d)] = float(rho)
    return pd.Series(out).sort_index()


def evaluate_q(meta, pred, label):
    m = evaluate_full(meta, pred.astype("float32"))
    return {"config": label,
             "rankicir": float(m["rankicir"]),
             "sharpe":   float(m["top5pct_sharpe"]),
             "maxdd":    float(m["top5pct_max_dd"])}


def load_model_pair(path: Path):
    m = CatBoostRegressor(); m.load_model(str(path))
    med = pd.read_csv(str(path).replace(".cbm", "_medians.csv"),
                        index_col=0).iloc[:, 0]
    return m, med


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("=" * 80)
    print("Step 15 — D1c vs D2f Walk-Forward (Sharpe + RankICIR comparison)")
    print("=" * 80)

    # ---- data + reversal features ----
    print("\n[0/4] load + reversal features ...")
    t0 = time.time()
    df = load_dataset().dropna(subset=[TARGET_RET_COL]).reset_index(drop=True)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = add_reversal_features(df)
    print(f"   panel rows: {len(df):,}  ({time.time()-t0:.1f}s)")

    # ---- agent paths per window ----
    windows = [
        ("W1", "2016-01-01", "2018-12-31", "2019-01-01", "2019-12-31"),
        ("W2", "2016-01-01", "2019-12-31", "2020-01-01", "2020-12-31"),
        ("W3", "2016-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
        ("W4", "2016-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
        ("W5", "2016-01-01", "2021-12-31", "2023-01-01", "2026-01-31"),
    ]

    def get_models(name):
        # alpha
        if name == "W1":
            a_path = (Path(__file__).resolve().parents[2]
                       / "strategy_debate/results/models/cross_period/trend_agent_B.cbm")
        elif name in ("W4", "W5"):
            a_path = ALPHA_AGENT_PATH
        else:
            a_path = WALKFORWARD_DIR / f"alpha_{name}.cbm"
        alpha_m, alpha_med = load_model_pair(a_path)

        # bear D1 (regular, for D1c)
        if name in ("W4", "W5"):
            bear_d1_obj = BearAgent(features=BEAR_FEATURES_D1, name="bear_D1")
            bear_d1_obj.load(BB_MODELS / "bear_D1_agent.cbm")
            bear_d1_m = bear_d1_obj.model
            bear_d1_med = bear_d1_obj._train_medians
        else:
            b_path = WALKFORWARD_DIR / f"bear_{name}.cbm"
            bear_d1_m, bear_d1_med = load_model_pair(b_path)

        # bear D2 (error-informed, lambda=3.0)
        bear_d2_path = WALKFORWARD_DIR / f"bear_D2_{name}.cbm"
        bear_d2_m, bear_d2_med = load_model_pair(bear_d2_path)

        # reversal
        rev_path = WALKFORWARD_DIR / f"reversal_{name}.cbm"
        rev_m, rev_med = load_model_pair(rev_path)
        return alpha_m, alpha_med, bear_d1_m, bear_d1_med, bear_d2_m, bear_d2_med, rev_m, rev_med

    # ---- per-window per-year evaluation ----
    print("\n[1/4] Walk-Forward evaluation per year ...")
    rows = []
    daily_portfolios = {}    # for W5 bootstrap
    daily_ics = {}            # for W5 bootstrap

    for name, tr_s, tr_e, te_s, te_e in windows:
        print(f"\n   ===== Window {name}: test {te_s[:7]} → {te_e[:7]} =====")
        models = get_models(name)
        alpha_m, alpha_med, b1_m, b1_med, b2_m, b2_med, r_m, r_med = models

        m_te = (df[DATE_COL] >= pd.Timestamp(te_s)) & (df[DATE_COL] <= pd.Timestamp(te_e))
        te = df.loc[m_te].sort_values([DATE_COL, TICKER_COL]).reset_index(drop=True)

        # predict
        te["alpha"] = alpha_m.predict(te[ALPHA_FEATURES].astype("float32").fillna(alpha_med)).astype("float32")
        te["bear_d1"] = b1_m.predict(te[BEAR_FEATURES_D1].astype("float32").fillna(b1_med)).astype("float32")
        te["bear_d2"] = b2_m.predict(te[BEAR_FEATURES_D1].astype("float32").fillna(b2_med)).astype("float32")
        te["rev"]     = r_m.predict(te[REVERSAL_FEATURES].astype("float32").fillna(r_med)).astype("float32")

        # z-scores
        te["bull_z"]    = zscore_daily(te, "alpha")
        te["bear_d1_z"] = zscore_daily(te, "bear_d1")
        te["bear_d2_z"] = zscore_daily(te, "bear_d2")
        te["rev_z"]     = zscore_daily(te, "rev")

        a_t = regime_alpha_t(te)
        g_t = regime_gamma_t(te, GAMMA_BASE, GAMMA_DELTA)

        d1c_score = (te["bull_z"].to_numpy("float32") - a_t * te["bear_d1_z"].to_numpy("float32"))
        d2c_score = (te["bull_z"].to_numpy("float32") - a_t * te["bear_d2_z"].to_numpy("float32"))
        d2f_score = (d2c_score + g_t * te["rev_z"].to_numpy("float32")).astype("float32")
        trend_score = te["alpha"].to_numpy("float32")

        # per-year
        for y in sorted(te[DATE_COL].dt.year.unique()):
            mask = (te[DATE_COL].dt.year == y).to_numpy()
            if mask.sum() < 100: continue
            sub_meta = te[[DATE_COL, TICKER_COL, TARGET_RET_COL]].loc[mask].reset_index(drop=True)
            m_T = evaluate_full(sub_meta, trend_score[mask])
            m_d1c = evaluate_full(sub_meta, d1c_score[mask])
            m_d2c = evaluate_full(sub_meta, d2c_score[mask])
            m_d2f = evaluate_full(sub_meta, d2f_score[mask])
            rows.append({
                "year": int(y), "window": name, "train_range": f"{tr_s[:4]}-{tr_e[:4]}",
                "trend_ric": float(m_T["rankicir"]),
                "trend_sr":  float(m_T["top5pct_sharpe"]),
                "d1c_ric":   float(m_d1c["rankicir"]),
                "d1c_sr":    float(m_d1c["top5pct_sharpe"]),
                "d2c_ric":   float(m_d2c["rankicir"]),
                "d2c_sr":    float(m_d2c["top5pct_sharpe"]),
                "d2f_ric":   float(m_d2f["rankicir"]),
                "d2f_sr":    float(m_d2f["top5pct_sharpe"]),
                "sr_gap_d1c_minus_d2f": float(m_d1c["top5pct_sharpe"] - m_d2f["top5pct_sharpe"]),
                "ric_gap_d2f_minus_d1c": float(m_d2f["rankicir"] - m_d1c["rankicir"]),
                "d1c_beats_d2f_sr":  bool(m_d1c["top5pct_sharpe"] > m_d2f["top5pct_sharpe"]),
                "d2f_beats_d1c_ric": bool(m_d2f["rankicir"] > m_d1c["rankicir"]),
            })
            print(f"     {y}: Trend SR={m_T['top5pct_sharpe']:+.3f}  "
                  f"D1c SR={m_d1c['top5pct_sharpe']:+.3f}  "
                  f"D2f SR={m_d2f['top5pct_sharpe']:+.3f}  "
                  f"D1c RIC={m_d1c['rankicir']:.3f}  D2f RIC={m_d2f['rankicir']:.3f}")

        # cache W5 series for bootstrap
        if name == "W5":
            meta_te = te[[DATE_COL, TICKER_COL, TARGET_RET_COL]].reset_index(drop=True)
            bt_d1c = backtest_topk(meta_te, d1c_score, frac=FRAC)
            bt_d2f = backtest_topk(meta_te, d2f_score, frac=FRAC)
            daily_portfolios["d1c"] = bt_d1c.daily_return
            daily_portfolios["d2f"] = bt_d2f.daily_return
            daily_ics["d1c"] = daily_ic(meta_te, d1c_score)
            daily_ics["d2f"] = daily_ic(meta_te, d2f_score)
            daily_ics["trend"] = daily_ic(meta_te, trend_score)

    df_wf = pd.DataFrame(rows)
    df_wf.to_csv(BB_RESULTS / "step15_walkforward_d1c_vs_d2f.csv",
                  index=False, encoding="utf-8-sig")

    # ---- summary table ----
    print("\n[2/4] Walk-Forward summary table ...")
    df_main = df_wf[df_wf["year"] <= 2025].reset_index(drop=True)
    print()
    line = "+" + "-" * 6 + "+" + "-" * 9 + "+" + "-" * 9 + "+" + "-" * 9 + "+" + "-" * 11 + "+" + "-" * 11 + "+" + "-" * 11 + "+" + "-" * 12 + "+" + "-" * 13 + "+"
    print(line)
    print(f"| {'Year':>4s} | {'D1c SR':>7s} | {'D2c SR':>7s} | {'D2f SR':>7s} | "
          f"{'D1c RIC':>9s} | {'D2c RIC':>9s} | {'D2f RIC':>9s} | "
          f"{'SR gap':>10s} | {'IC gap':>11s} |")
    print(line)
    for _, r in df_main.iterrows():
        print(f"| {int(r['year']):>4d} | "
              f"{r['d1c_sr']:>+7.3f} | {r['d2c_sr']:>+7.3f} | {r['d2f_sr']:>+7.3f} | "
              f"{r['d1c_ric']:>9.4f} | {r['d2c_ric']:>9.4f} | {r['d2f_ric']:>9.4f} | "
              f"{r['sr_gap_d1c_minus_d2f']:>+10.3f} | {r['ric_gap_d2f_minus_d1c']:>+11.4f} |")
    # means
    print(line)
    print(f"| {'Mean':>4s} | "
          f"{df_main['d1c_sr'].mean():>+7.3f} | {df_main['d2c_sr'].mean():>+7.3f} | "
          f"{df_main['d2f_sr'].mean():>+7.3f} | "
          f"{df_main['d1c_ric'].mean():>9.4f} | {df_main['d2c_ric'].mean():>9.4f} | "
          f"{df_main['d2f_ric'].mean():>9.4f} | "
          f"{df_main['sr_gap_d1c_minus_d2f'].mean():>+10.3f} | "
          f"{df_main['ric_gap_d2f_minus_d1c'].mean():>+11.4f} |")
    print(line)

    n_y = len(df_main)
    sr_wins = int(df_main["d1c_beats_d2f_sr"].sum())
    ric_wins = int(df_main["d2f_beats_d1c_ric"].sum())
    print(f"\n   D1c SR > D2f SR  in {sr_wins}/{n_y} years")
    print(f"   D2f RIC > D1c RIC in {ric_wins}/{n_y} years")

    # checks
    fail_sr_yr = df_main.loc[~df_main["d1c_beats_d2f_sr"], "year"].tolist()
    fail_ic_yr = df_main.loc[~df_main["d2f_beats_d1c_ric"], "year"].tolist()
    if fail_sr_yr:
        print(f"   WARN: years where D2f Sharpe >= D1c Sharpe: {fail_sr_yr}")
    if fail_ic_yr:
        print(f"   WARN: years where D1c RankICIR >= D2f RankICIR: {fail_ic_yr}")

    # COVID / deep bear callouts
    print(f"\n   2020 (COVID) Sharpe: D1c={df_main[df_main['year']==2020]['d1c_sr'].iloc[0]:+.3f}  "
          f"D2f={df_main[df_main['year']==2020]['d2f_sr'].iloc[0]:+.3f}")
    print(f"   2022 (deep bear) Sharpe: D1c={df_main[df_main['year']==2022]['d1c_sr'].iloc[0]:+.3f}  "
          f"D2f={df_main[df_main['year']==2022]['d2f_sr'].iloc[0]:+.3f}")

    # ---- Bootstrap on W5 test (2023-2025) ----
    print("\n[3/4] Bootstrap N=1000 on W5 test (2023-2026) ...")
    if "d1c" not in daily_ics or "d2f" not in daily_ics:
        print("   missing W5 daily series; skip bootstrap")
        return

    # IC series for RankICIR bootstrap
    ic_d1c = daily_ics["d1c"]
    ic_d2f = daily_ics["d2f"]
    common_ic = sorted(set(ic_d1c.index) & set(ic_d2f.index))
    a_ic_d1c = np.array([ic_d1c[d] for d in common_ic])
    a_ic_d2f = np.array([ic_d2f[d] for d in common_ic])

    # portfolio daily return for Sharpe bootstrap
    pr_d1c = daily_portfolios["d1c"]; pr_d1c.index = pd.to_datetime(pr_d1c.index)
    pr_d2f = daily_portfolios["d2f"]; pr_d2f.index = pd.to_datetime(pr_d2f.index)
    common_pr = sorted(set(pr_d1c.index) & set(pr_d2f.index))
    a_pr_d1c = np.array([pr_d1c[d] for d in common_pr])
    a_pr_d2f = np.array([pr_d2f[d] for d in common_pr])

    def rankicir(a):
        return float(np.mean(a) / (np.std(a, ddof=0) + 1e-9))

    def sharpe(a):
        ann_ret = float(np.mean(a) * TRADING_DAYS_PER_YEAR)
        ann_vol = float(np.std(a, ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR))
        return ann_ret / ann_vol if ann_vol > 0 else float("nan")

    obs_ic_gap = rankicir(a_ic_d2f) - rankicir(a_ic_d1c)
    obs_sr_gap = sharpe(a_pr_d1c) - sharpe(a_pr_d2f)
    print(f"   observed RankICIR: D1c={rankicir(a_ic_d1c):.4f}  D2f={rankicir(a_ic_d2f):.4f}  gap={obs_ic_gap:+.4f}")
    print(f"   observed Sharpe  : D1c={sharpe(a_pr_d1c):+.4f}  D2f={sharpe(a_pr_d2f):+.4f}  gap={obs_sr_gap:+.4f}")

    rng = np.random.default_rng(42)
    n_boot = 1000

    # IC bootstrap
    n_ic = len(common_ic)
    boot_ic = np.zeros(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n_ic, size=n_ic)
        boot_ic[i] = rankicir(a_ic_d2f[idx]) - rankicir(a_ic_d1c[idx])

    # Sharpe bootstrap (independent re-sampling)
    n_pr = len(common_pr)
    boot_sr = np.zeros(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n_pr, size=n_pr)
        boot_sr[i] = sharpe(a_pr_d1c[idx]) - sharpe(a_pr_d2f[idx])

    def stats(boot, name, obs):
        mean = float(boot.mean())
        lo, hi = float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
        p = float(np.mean(boot <= 0))
        marker = "(< 0.001)" if p == 0 else f"(={p:.4f})"
        print(f"   [{name}]")
        print(f"     observed={obs:+.4f}  bootstrap mean={mean:+.4f}")
        print(f"     95% CI=[{lo:+.4f}, {hi:+.4f}]  p-value={p:.4f}  {marker}")
        return {"name": name, "observed": obs, "bootstrap_mean": mean,
                "ci_low": lo, "ci_high": hi, "p_value": p,
                "n_boot": n_boot}

    boot_rows = [
        stats(boot_ic, "D2f RIC - D1c RIC", obs_ic_gap),
        stats(boot_sr, "D1c SR - D2f SR",   obs_sr_gap),
    ]
    pd.DataFrame(boot_rows).to_csv(
        BB_RESULTS / "step15_bootstrap_d1c_vs_d2f.csv",
        index=False, encoding="utf-8-sig")

    # ---- final verdict ----
    print("\n[4/4] verdict")
    print()
    p_ic = boot_rows[0]["p_value"]
    p_sr = boot_rows[1]["p_value"]
    print(f"   D2f vs D1c RankICIR:  p={p_ic:.4f}  "
          f"{('PASS (< 0.001)' if p_ic < 0.001 else 'WARN' if p_ic > 0.05 else 'PASS')}")
    print(f"   D1c vs D2f Sharpe  :  p={p_sr:.4f}  "
          f"{('PASS (< 0.001)' if p_sr < 0.001 else 'PASS' if p_sr < 0.05 else 'WARN')}")

    if p_ic < 0.05 and p_sr < 0.05:
        print()
        print("   Dual-variant paper claim is statistically supported:")
        print(f"     BBAQ-MAS-IC (D2f) > D1c on RankICIR  (p={p_ic:.4f})")
        print(f"     BBAQ-MAS-SR (D1c) > D2f on Sharpe    (p={p_sr:.4f})")
    else:
        print("\n   WARNING: at least one trade-off direction not significant; "
              "the dual-variant claim needs caveats.")

    print(f"\nOutputs:")
    print(f"  -> {BB_RESULTS / 'step15_walkforward_d1c_vs_d2f.csv'}")
    print(f"  -> {BB_RESULTS / 'step15_bootstrap_d1c_vs_d2f.csv'}")


if __name__ == "__main__":
    main()
