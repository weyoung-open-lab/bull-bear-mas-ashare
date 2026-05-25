"""Step 13 — Regime-adaptive gamma + D2f final-system confirmation.

Task 1: gamma adaptive
  gamma(t) = gamma_base + delta * P_bull(t) - delta * P_bear(t)
  gamma_base = 0.40 (step 12 peak)
  test delta in {0.05, 0.10, 0.15}

Task 2: turnover diagnostic
  daily_turnover(t) = |selection(t) symm-diff selection(t-1)| / |union|
  for D1c, D2c, D2f

Task 3: annualised return for each system (proxy track)

Outputs:
  bull_bear/results/step13_gamma_adaptive.csv
  bull_bear/results/step13_turnover_diagnostic.csv
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

from bull_bear.config_bb import (
    ALPHA_AGENT_PATH, BB_MODELS, BB_RESULTS,
    BEAR_FEATURES_D1, DATE_COL,
    TARGET_RET_COL, TEST_END, TEST_START, TICKER_COL,
)
from bull_bear.src.bear_agent import BearAgent
from bull_bear.src.bear_target import build_max_drawdown_5d
from src.backtest import backtest_topk
from src.data import load_dataset
from bull_bear.src.metrics_utils import evaluate_full

ALPHA_FEATURES = ["ma60_slope", "ema180_slope", "bias_60", "bias_60_vr", "ma180_slope"]
ALPHA_BY_REGIME = {"bear": 0.65, "sideway": 0.50, "bull": 0.35}
GAMMA_BASE = 0.40
DELTA_GRID = (0.05, 0.10, 0.15)
REVERSAL_FEATURES = [
    "ret_1d", "ret_3d",
    "rev_ret_2d", "rev_ret_3d_minus_1d",
    "rev_zscore_1d", "rev_mkt_excess_1d",
]
TRADING_DAYS_PER_YEAR = 252
FRAC = 0.05


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


def regime_gamma(panel: pd.DataFrame, gamma_base: float, delta: float) -> np.ndarray:
    """gamma(t) = base + delta * 1[bull] - delta * 1[bear]."""
    regime = panel["macro_regime_3"].astype(str).to_numpy()
    g = np.full(len(regime), gamma_base, dtype="float32")
    g[regime == "bull"] += delta
    g[regime == "bear"] -= delta
    return g


def daily_topk_selection(meta: pd.DataFrame, pred: np.ndarray,
                          frac: float = FRAC) -> dict[pd.Timestamp, set[str]]:
    df = meta[[DATE_COL, TICKER_COL]].copy()
    df["pred"] = pred
    out: dict[pd.Timestamp, set[str]] = {}
    for d, g in df.groupby(DATE_COL, sort=True):
        n = len(g)
        k = max(1, int(np.ceil(n * frac)))
        idx = g["pred"].to_numpy().argsort()[::-1][:k]
        out[pd.Timestamp(d)] = set(g.iloc[idx][TICKER_COL].tolist())
    return out


def daily_turnover(selections: dict[pd.Timestamp, set[str]]) -> tuple[pd.Series, float]:
    """Consecutive-day symmetric difference / mean of sizes."""
    dates = sorted(selections.keys())
    rows = []
    for i in range(1, len(dates)):
        a = selections[dates[i - 1]]
        b = selections[dates[i]]
        denom = (len(a) + len(b)) / 2.0
        t = len(a.symmetric_difference(b)) / 2.0 / max(denom, 1)
        rows.append((dates[i], t))
    s = pd.Series({d: t for d, t in rows}).sort_index()
    annual = float(s.mean() * TRADING_DAYS_PER_YEAR)
    return s, annual


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("=" * 80)
    print("Step 13 — Regime-adaptive gamma + D2f final confirmation")
    print("=" * 80)

    # ---- data ----
    print("\n[0/5] load data + reversal features + targets ...")
    t0 = time.time()
    df = load_dataset().dropna(subset=[TARGET_RET_COL]).reset_index(drop=True)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    # reversal features (only need rev_ret_2d, rev_ret_3d_minus_1d, rev_zscore_1d, rev_mkt_excess_1d)
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
    mask_te = (df[DATE_COL] >= pd.Timestamp(TEST_START)) & (df[DATE_COL] <= pd.Timestamp(TEST_END))
    test = df.loc[mask_te].sort_values([DATE_COL, TICKER_COL]).reset_index(drop=True)
    print(f"   test={len(test):,}  ({time.time()-t0:.1f}s)")

    # ---- agents ----
    print("\n[1/5] load Alpha + Bear D1 + Bear D2 + Reversal ...")
    alpha_m = CatBoostRegressor(); alpha_m.load_model(str(ALPHA_AGENT_PATH))
    alpha_med = pd.read_csv(str(ALPHA_AGENT_PATH).replace(".cbm", "_medians.csv"),
                              index_col=0).iloc[:, 0]
    test["alpha_score"] = alpha_m.predict(
        test[ALPHA_FEATURES].astype("float32").fillna(alpha_med)
    ).astype("float32")

    # Bear D1 (regular) for D1c baseline
    bear_d1 = BearAgent(features=BEAR_FEATURES_D1, name="bear_D1")
    bear_d1.load(BB_MODELS / "bear_D1_agent.cbm")
    test["bear_d1_score"] = bear_d1.predict_panel(test).astype("float32")

    # Bear D2 (error-informed, λ=3) for D2c/D2f
    bear_d2 = CatBoostRegressor()
    bear_d2.load_model(str(BB_MODELS / "bear_D2_l3.0.cbm"))
    test["bear_d2_score"] = bear_d2.predict(
        test[BEAR_FEATURES_D1].astype("float32").fillna(bear_d1._train_medians)
    ).astype("float32")

    # Reversal
    rev = CatBoostRegressor(); rev.load_model(str(BB_MODELS / "reversal_B_5d.cbm"))
    rev_medians = test[REVERSAL_FEATURES].median()
    test["rev_score"] = rev.predict(
        test[REVERSAL_FEATURES].astype("float32").fillna(rev_medians)
    ).astype("float32")

    # cross-section z-scores
    test["bull_z"]    = zscore_daily(test, "alpha_score")
    test["bear_d1_z"] = zscore_daily(test, "bear_d1_score")
    test["bear_d2_z"] = zscore_daily(test, "bear_d2_score")
    test["rev_z"]     = zscore_daily(test, "rev_score")

    alpha_t = regime_alpha(test)
    meta_te = test[[DATE_COL, TICKER_COL, TARGET_RET_COL, "ret_1d"]].reset_index(drop=True)

    # ---- baselines ----
    d1c = (test["bull_z"].to_numpy("float32") - alpha_t * test["bear_d1_z"].to_numpy("float32"))
    d2c = (test["bull_z"].to_numpy("float32") - alpha_t * test["bear_d2_z"].to_numpy("float32"))
    d2f_fixed = (d2c + GAMMA_BASE * test["rev_z"].to_numpy("float32")).astype("float32")

    # ============================================================
    # Task 1 — gamma adaptive
    # ============================================================
    print("\n[2/5] Task 1 — regime-adaptive gamma ...")
    rev_z = test["rev_z"].to_numpy("float32")
    rows: list[dict] = []

    # baselines first
    def eval_record(pred: np.ndarray, label: str, track: str = "") -> dict:
        m_eval = evaluate_full(meta_te[[DATE_COL, TICKER_COL, TARGET_RET_COL]].reset_index(drop=True),
                                 pred)
        bt_proxy = backtest_topk(
            meta_te[[DATE_COL, TICKER_COL, TARGET_RET_COL]].reset_index(drop=True),
            pred, frac=FRAC)
        return {"config": label, "track": track,
                 "rankicir": float(m_eval["rankicir"]),
                 "sharpe":   float(bt_proxy.sharpe),
                 "maxdd":    float(bt_proxy.max_drawdown),
                 "annual_return": float(bt_proxy.annual_return),
                 "annual_vol":    float(bt_proxy.annual_volatility),
                 "_daily": bt_proxy.daily_return.copy()}

    m_d1c = eval_record(d1c, "D1c (Alpha + Bear D1 + adaptive alpha)", "baseline")
    m_d2c = eval_record(d2c, "D2c (D1c + error-informed Bear lambda=3)", "baseline")
    m_d2f_fixed = eval_record(d2f_fixed, f"D2f gamma_fixed=0.40", "baseline")
    rows += [
        {k: v for k, v in m.items() if not k.startswith("_")}
        for m in (m_d1c, m_d2c, m_d2f_fixed)
    ]
    # delta vs D2f fixed
    ric_ref = m_d2f_fixed["rankicir"]
    for r in rows:
        r["delta_bp_vs_d2f_fixed"] = (r["rankicir"] - ric_ref) * 10000

    print(f"   baselines:")
    for r in rows:
        print(f"     {r['config']:48s}  RankICIR={r['rankicir']:.4f}  "
              f"Sharpe={r['sharpe']:+.3f}  MaxDD={r['maxdd']*100:+.2f}%  "
              f"AnnRet={r['annual_return']*100:+.2f}%")

    # Regime composition on test
    reg = test["macro_regime_3"].astype(str)
    n_bull = int((reg == "bull").sum())
    n_bear = int((reg == "bear").sum())
    n_side = int((reg == "sideway").sum())
    print(f"   test regime mix: bull={n_bull/len(test)*100:.1f}%  "
          f"sideway={n_side/len(test)*100:.1f}%  bear={n_bear/len(test)*100:.1f}%")

    # gamma adaptive grid
    print()
    daily_for_d2f_adap = {}    # save daily returns for turnover later
    for delta in DELTA_GRID:
        gamma_t = regime_gamma(test, GAMMA_BASE, delta)
        pred = (d2c + gamma_t * rev_z).astype("float32")
        m = eval_record(pred, f"D2f gamma_adaptive base=0.40 delta=±{delta:.2f}", "adaptive")
        m["delta_bp_vs_d2f_fixed"] = (m["rankicir"] - ric_ref) * 10000
        # cache daily for turnover
        daily_for_d2f_adap[delta] = (pred, m["_daily"])
        rows.append({k: v for k, v in m.items() if not k.startswith("_")})
        gamma_range = f"[{GAMMA_BASE - delta:.2f}, {GAMMA_BASE + delta:.2f}]"
        print(f"   delta={delta:.2f}  γ in {gamma_range}  "
              f"RankICIR={m['rankicir']:.4f}  Sharpe={m['sharpe']:+.3f}  "
              f"MaxDD={m['maxdd']*100:+.2f}%  AnnRet={m['annual_return']*100:+.2f}%  "
              f"Δ vs D2f-fixed = {m['delta_bp_vs_d2f_fixed']:+.1f} bp")

    df_gamma = pd.DataFrame(rows)
    df_gamma.to_csv(BB_RESULTS / "step13_gamma_adaptive.csv", index=False,
                     encoding="utf-8-sig")

    # ============================================================
    # Task 2 — Turnover diagnostic
    # ============================================================
    print("\n[3/5] Task 2 — turnover diagnostic ...")
    sel_d1c = daily_topk_selection(meta_te, d1c)
    sel_d2c = daily_topk_selection(meta_te, d2c)
    sel_d2f = daily_topk_selection(meta_te, d2f_fixed)

    s_d1c, ann_d1c = daily_turnover(sel_d1c)
    s_d2c, ann_d2c = daily_turnover(sel_d2c)
    s_d2f, ann_d2f = daily_turnover(sel_d2f)

    # adaptive
    ann_adapt = {}
    for delta in DELTA_GRID:
        pred_adapt, _ = daily_for_d2f_adap[delta]
        sel = daily_topk_selection(meta_te, pred_adapt)
        s, ann = daily_turnover(sel)
        ann_adapt[delta] = ann

    print(f"   {'Config':50s}  {'Daily TO':>9s}  {'Annual TO':>10s}  {'Sharpe':>7s}  {'AnnRet':>8s}")
    rows_to = [
        ("D1c", s_d1c, ann_d1c, m_d1c),
        ("D2c", s_d2c, ann_d2c, m_d2c),
        ("D2f gamma_fixed=0.40", s_d2f, ann_d2f, m_d2f_fixed),
    ]
    for name, s, ann, m in rows_to:
        print(f"   {name:50s}  {s.mean()*100:>8.2f}%  {ann:>9.2f}x  "
              f"{m['sharpe']:>+7.3f}  {m['annual_return']*100:>+7.2f}%")
    for delta in DELTA_GRID:
        ann = ann_adapt[delta]
        m = next(r for r in rows if r.get("config", "").startswith(f"D2f gamma_adaptive base=0.40 delta=±{delta:.2f}"))
        name = f"D2f adaptive delta={delta:.2f}"
        print(f"   {name:50s}  {ann/TRADING_DAYS_PER_YEAR*100:>8.2f}%  "
              f"{ann:>9.2f}x  {m['sharpe']:>+7.3f}  {m['annual_return']*100:>+7.2f}%")

    # CSV
    to_rows = [
        {"config": "D1c", "daily_turnover_mean": float(s_d1c.mean()),
         "annual_turnover": ann_d1c, "rankicir": m_d1c["rankicir"],
         "sharpe": m_d1c["sharpe"], "annual_return": m_d1c["annual_return"]},
        {"config": "D2c", "daily_turnover_mean": float(s_d2c.mean()),
         "annual_turnover": ann_d2c, "rankicir": m_d2c["rankicir"],
         "sharpe": m_d2c["sharpe"], "annual_return": m_d2c["annual_return"]},
        {"config": "D2f gamma_fixed=0.40", "daily_turnover_mean": float(s_d2f.mean()),
         "annual_turnover": ann_d2f, "rankicir": m_d2f_fixed["rankicir"],
         "sharpe": m_d2f_fixed["sharpe"], "annual_return": m_d2f_fixed["annual_return"]},
    ]
    pd.DataFrame(to_rows).to_csv(BB_RESULTS / "step13_turnover_diagnostic.csv",
                                  index=False, encoding="utf-8-sig")

    # ============================================================
    # Cost-drag analysis
    # ============================================================
    print("\n[4/5] Cost-drag analysis ...")
    # Each system pays cost = annual_turnover × 0.3% (double-side per round-trip)
    # already baked into Sharpe by backtest_topk
    # gross sharpe ≈ (annual_return + cost_drag) / annual_vol
    # cost_drag = 2 * cost_one_side * annual_turnover = 0.3% * ann_turnover
    cost_one_side = 0.0015
    print(f"   {'Config':50s}  {'AnnRet':>8s}  {'CostDrag':>9s}  {'AnnVol':>8s}  "
          f"{'Sharpe(net)':>11s}  {'Sharpe(gross)':>13s}")
    for name, m_obj in [("D1c", m_d1c), ("D2c", m_d2c),
                          ("D2f gamma_fixed=0.40", m_d2f_fixed)]:
        ann_to = {"D1c": ann_d1c, "D2c": ann_d2c,
                    "D2f gamma_fixed=0.40": ann_d2f}[name]
        cost_drag = 2.0 * cost_one_side * ann_to
        gross_ret = m_obj["annual_return"] + cost_drag
        sharpe_gross = gross_ret / m_obj["annual_vol"] if m_obj["annual_vol"] > 0 else float("nan")
        print(f"   {name:50s}  {m_obj['annual_return']*100:>+7.2f}%  "
              f"{cost_drag*100:>+7.2f}%  {m_obj['annual_vol']*100:>+7.2f}%  "
              f"{m_obj['sharpe']:>+11.3f}  {sharpe_gross:>+13.3f}")

    # ============================================================
    # Final summary
    # ============================================================
    print("\n[5/5] FINAL summary table")
    print()
    line = "+" + "-" * 50 + "+" + "-" * 11 + "+" + "-" * 10 + "+" + "-" * 11 + "+" + "-" * 11 + "+" + "-" * 12 + "+"
    print(line)
    print(f"| {'Config':48s} | {'RankICIR':>9s} | {'Sharpe':>8s} | "
          f"{'MaxDD':>9s} | {'AnnRet':>9s} | {'AnnTO':>10s} |")
    print(line)
    for r in rows:
        cfg = r["config"]
        ann_to_lookup = {
            "D1c (Alpha + Bear D1 + adaptive alpha)": ann_d1c,
            "D2c (D1c + error-informed Bear lambda=3)": ann_d2c,
            "D2f gamma_fixed=0.40": ann_d2f,
        }
        ann_to_str = "-"
        for k, v in ann_to_lookup.items():
            if cfg == k:
                ann_to_str = f"{v:.2f}x"
        if cfg.startswith("D2f gamma_adaptive"):
            for delta in DELTA_GRID:
                if f"delta=±{delta:.2f}" in cfg:
                    ann_to_str = f"{ann_adapt[delta]:.2f}x"
        print(f"| {cfg:48s} | {r['rankicir']:>9.4f} | "
              f"{r['sharpe']:>+8.3f} | {r['maxdd']*100:>+8.2f}% | "
              f"{r['annual_return']*100:>+7.2f}% | {ann_to_str:>10s} |")
    print(line)

    # Verdict
    print("\n=== Verdict ===")
    best_adapt = max([r for r in rows if r.get("track") == "adaptive"],
                       key=lambda r: r["rankicir"])
    print(f"  Best adaptive: {best_adapt['config']}")
    print(f"    RankICIR={best_adapt['rankicir']:.4f}  vs D2f-fixed {ric_ref:.4f}  "
          f"Δ={best_adapt['delta_bp_vs_d2f_fixed']:+.1f} bp")
    print(f"    Sharpe={best_adapt['sharpe']:+.3f}  vs D2f-fixed {m_d2f_fixed['sharpe']:+.3f}")
    print(f"    AnnRet={best_adapt['annual_return']*100:+.2f}%  vs D2f-fixed "
          f"{m_d2f_fixed['annual_return']*100:+.2f}%")
    if best_adapt["sharpe"] > m_d2f_fixed["sharpe"]:
        print("  -> Adaptive gamma improves Sharpe.")
    else:
        print("  -> Adaptive gamma does not improve Sharpe; "
              "regime-conditioning of reversal weight has limited value.")

    # Turnover ratio analysis
    print(f"\n  Turnover ratio:")
    print(f"    D2f / D1c = {ann_d2f / ann_d1c:.2f}x")
    print(f"    D2f / D2c = {ann_d2f / ann_d2c:.2f}x")
    # Annual return comparison
    print(f"\n  Annual returns (proxy track):")
    print(f"    D1c = {m_d1c['annual_return']*100:+.2f}%")
    print(f"    D2c = {m_d2c['annual_return']*100:+.2f}%")
    print(f"    D2f = {m_d2f_fixed['annual_return']*100:+.2f}%")
    if m_d2f_fixed["annual_return"] > m_d1c["annual_return"]:
        print("    -> D2f earns more return than D1c; Sharpe drop is from higher volatility.")
    else:
        print("    -> D2f earns less return than D1c; Sharpe drop is from signal quality.")

    print(f"\nOutputs:")
    print(f"  -> {BB_RESULTS / 'step13_gamma_adaptive.csv'}")
    print(f"  -> {BB_RESULTS / 'step13_turnover_diagnostic.csv'}")


if __name__ == "__main__":
    main()
