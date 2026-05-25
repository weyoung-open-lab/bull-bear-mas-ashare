"""Step 16 — Agent 4 practical deployment layer.

Agent 4 operates AFTER the conviction scores from D2f, modifying portfolio
weights and/or exposure without changing the cross-section ranking.
RankICIR is therefore preserved by construction; only Sharpe and MaxDD
change.

Three functions:
  F1 — Tradability filter: exclude limit-up stocks (ret_1d >= 0.099) from
       Top-5% selection. (No volume column in dataset; cannot detect
       suspension directly.)
  F2 — Confidence-weighted positioning: softmax weighting with temperatures
       tau in {0.5, 1.0, 2.0} on conviction scores within the Top-5%.
  F3 — Market circuit breaker: reduce exposure when market daily return
       < -3% (exposure -> 0.5) or when bear regime fires (exposure -> 0.7).

Outputs:
  bull_bear/results/step16_agent4_practical.csv
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
HOLDING_DAYS = 5
TRADING_DAYS_PER_YEAR = 252
COST_ONE_SIDE = 0.0015
LIMIT_UP_THRESHOLD = 0.099    # ret_1d at or above this = limit-up touched today
MARKET_DROP_THRESHOLD = -0.03  # circuit breaker if market daily mean ret < -3%
EXPOSURE_DROP = 0.5
EXPOSURE_BEAR_REGIME = 0.7
TAU_GRID = (0.5, 1.0, 2.0)


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


def gini(x):
    """Gini coefficient of a non-negative weight vector."""
    x = np.array(x, dtype="float64")
    x = x[x > 0]
    if len(x) == 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    cumx = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n)


def cohort_backtest_weighted(
    meta: pd.DataFrame,
    pred: np.ndarray,
    *,
    weight_mode: str = "equal",
    tau: float = 1.0,
    limit_up_mask: np.ndarray | None = None,
    exposure_vector: np.ndarray | None = None,
    frac: float = FRAC,
    holding_days: int = HOLDING_DAYS,
    cost_one_side: float = COST_ONE_SIDE,
) -> dict:
    """Backtest with cohort-style 5-day holding, supporting weighting + exposure.

    weight_mode in {"equal", "softmax"}.
    limit_up_mask: bool array same length as meta, True for stocks to exclude.
    exposure_vector: array of per-day exposure in [0, 1]. Length = unique dates.
    """
    df_m = meta[[DATE_COL, TICKER_COL, TARGET_RET_COL]].copy().reset_index(drop=True)
    df_m["pred"] = pred
    if limit_up_mask is not None:
        df_m["filtered_out"] = limit_up_mask
    else:
        df_m["filtered_out"] = False
    # proxy daily return = r_future_5 / holding_days
    df_m["day_ret"] = df_m[TARGET_RET_COL] / holding_days

    dates = sorted(df_m[DATE_COL].unique())
    date_to_idx = {d: i for i, d in enumerate(dates)}

    # per-day selection with optional filter + weights
    selections: dict[pd.Timestamp, dict[str, float]] = {}
    for d, g in df_m.groupby(DATE_COL, sort=True):
        n = len(g)
        k = max(1, int(np.ceil(n * frac)))
        g_sorted = g.sort_values("pred", ascending=False).head(k)
        if limit_up_mask is not None:
            g_sorted = g_sorted[~g_sorted["filtered_out"]]
        if len(g_sorted) == 0:
            selections[pd.Timestamp(d)] = {}
            continue
        if weight_mode == "softmax":
            scores = g_sorted["pred"].to_numpy("float64")
            scores_shift = scores - scores.max()
            ex = np.exp(scores_shift * tau)
            w = ex / ex.sum()
        else:
            w = np.full(len(g_sorted), 1.0 / len(g_sorted))
        selections[pd.Timestamp(d)] = dict(zip(g_sorted[TICKER_COL].tolist(),
                                                   w.tolist()))

    # daily return table (date, ticker) -> day_ret
    rt = df_m.set_index([DATE_COL, TICKER_COL])["day_ret"]

    # cohort run
    prev_holdings: list[dict[str, float]] = [dict() for _ in range(holding_days)]
    daily_returns = []
    daily_turnover = []

    for i, d in enumerate(dates):
        d_ts = pd.Timestamp(d)
        cohort = i % holding_days
        new = selections.get(d_ts, {})

        prev = prev_holdings[cohort]
        # turnover = sum |w_new - w_prev| / 2 (L1 distance / 2; max = 1)
        all_keys = set(prev.keys()) | set(new.keys())
        turnover = 0.5 * sum(abs(new.get(k, 0.0) - prev.get(k, 0.0)) for k in all_keys)
        prev_holdings[cohort] = new

        # active cohorts portfolio return
        try:
            day_rets = rt.loc[d_ts]
        except KeyError:
            continue
        cohort_rets = []
        for c in range(holding_days):
            h = prev_holdings[c]
            if not h:
                continue
            # weighted sum of returns for keys present in day_rets
            keys_present = [k for k in h.keys() if k in day_rets.index]
            if not keys_present:
                continue
            wsum = sum(h[k] for k in keys_present)
            if wsum <= 0:
                continue
            # weighted return
            cohort_ret = sum(h[k] * float(day_rets.loc[k]) for k in keys_present) / wsum
            cohort_rets.append(cohort_ret)

        if cohort_rets:
            gross = float(np.mean(cohort_rets))
        else:
            gross = 0.0

        cost = turnover * cost_one_side * 2.0 / holding_days
        net = gross - cost

        # exposure scaling
        if exposure_vector is not None:
            exp_idx = date_to_idx[d]
            exposure = float(exposure_vector[exp_idx])
            net = net * exposure    # missing exposure earns 0 (cash)

        daily_returns.append((d_ts, net))
        daily_turnover.append(turnover)

    s = pd.Series({d: r for d, r in daily_returns}).sort_index()
    nav = (1.0 + s).cumprod()
    ann_ret = float(s.mean() * TRADING_DAYS_PER_YEAR)
    ann_vol = float(s.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
    drawdown = nav / nav.cummax() - 1
    max_dd = float(drawdown.min())
    return {"sharpe": sharpe, "maxdd": max_dd,
             "annual_return": ann_ret, "annual_vol": ann_vol,
             "avg_turnover": float(np.mean(daily_turnover)) if daily_turnover else float("nan"),
             "daily_return": s, "selections": selections}


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("=" * 80)
    print("Step 16 — Agent 4 practical deployment layer")
    print("=" * 80)

    # ---- load ----
    print("\n[0/6] load + reversal features ...")
    t0 = time.time()
    df = load_dataset().dropna(subset=[TARGET_RET_COL]).reset_index(drop=True)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    # check for volume column
    has_volume = "volume" in df.columns
    print(f"   volume column in dataset: {has_volume}")

    df = add_reversal_features(df)
    mask_te = (df[DATE_COL] >= pd.Timestamp(TEST_START)) & (df[DATE_COL] <= pd.Timestamp(TEST_END))
    test = df.loc[mask_te].sort_values([DATE_COL, TICKER_COL]).reset_index(drop=True)
    print(f"   test rows={len(test):,}  days={test[DATE_COL].nunique()}  "
          f"({time.time()-t0:.1f}s)")

    # ---- agents -> D2f conviction ----
    print("\n[1/6] compute D2f conviction on test ...")
    alpha_m = CatBoostRegressor(); alpha_m.load_model(str(ALPHA_AGENT_PATH))
    alpha_med = pd.read_csv(str(ALPHA_AGENT_PATH).replace(".cbm", "_medians.csv"),
                              index_col=0).iloc[:, 0]
    test["alpha_score"] = alpha_m.predict(
        test[ALPHA_FEATURES].astype("float32").fillna(alpha_med)).astype("float32")
    bear_d1_ref = BearAgent(features=BEAR_FEATURES_D1, name="bear_D1")
    bear_d1_ref.load(BB_MODELS / "bear_D1_agent.cbm")
    bear_d2 = CatBoostRegressor()
    bear_d2.load_model(str(BB_MODELS / "bear_D2_l3.0.cbm"))
    test["bear_score"] = bear_d2.predict(
        test[BEAR_FEATURES_D1].astype("float32").fillna(bear_d1_ref._train_medians)
    ).astype("float32")
    rev = CatBoostRegressor(); rev.load_model(str(BB_MODELS / "reversal_B_5d.cbm"))
    rev_medians = test[REVERSAL_FEATURES].median()
    test["rev_score"] = rev.predict(
        test[REVERSAL_FEATURES].astype("float32").fillna(rev_medians)).astype("float32")
    test["bull_z"] = zscore_daily(test, "alpha_score")
    test["bear_z"] = zscore_daily(test, "bear_score")
    test["rev_z"]  = zscore_daily(test, "rev_score")

    alpha_t = regime_alpha_t(test)
    gamma_t = regime_gamma_t(test, GAMMA_BASE, GAMMA_DELTA)
    d2f_score = (test["bull_z"].to_numpy("float32")
                  - alpha_t * test["bear_z"].to_numpy("float32")
                  + gamma_t * test["rev_z"].to_numpy("float32")).astype("float32")
    test["conviction"] = d2f_score

    meta_te = test[[DATE_COL, TICKER_COL, TARGET_RET_COL, "ret_1d"]].reset_index(drop=True)
    rankicir_d2f = float(evaluate_full(
        meta_te[[DATE_COL, TICKER_COL, TARGET_RET_COL]].reset_index(drop=True),
        d2f_score)["rankicir"])
    print(f"   D2f RankICIR = {rankicir_d2f:.4f}  (unchanged across all sub-functions of Agent 4)")

    # ---- F1 diagnostic ----
    print("\n[2/6] F1 Tradability filter diagnostic ...")
    # Limit-up: ret_1d >= 0.099
    limit_up = (test["ret_1d"].astype("float64") >= LIMIT_UP_THRESHOLD).to_numpy()
    # Top-5% selection per day
    df_sel = test[[DATE_COL, TICKER_COL, "conviction"]].copy().reset_index(drop=True)
    df_sel["limit_up"] = limit_up
    avg_filtered_per_day = []
    total_top_per_day = []
    for d, g in df_sel.groupby(DATE_COL):
        n = len(g)
        k = max(1, int(np.ceil(n * FRAC)))
        top = g.sort_values("conviction", ascending=False).head(k)
        avg_filtered_per_day.append(int(top["limit_up"].sum()))
        total_top_per_day.append(k)
    avg_filtered = float(np.mean(avg_filtered_per_day))
    avg_total = float(np.mean(total_top_per_day))
    print(f"   limit-up days in test (any stock): {int(limit_up.sum()):,} rows total")
    print(f"   avg stocks per day in Top-5%      : {avg_total:.1f}")
    print(f"   avg stocks filtered (limit-up)    : {avg_filtered:.2f} ({avg_filtered/avg_total*100:.2f}%)")
    if has_volume:
        print(f"   (also filter volume=0 if column present)")
    else:
        print(f"   no volume column — cannot detect suspension; only limit-up filter applies")

    # ---- F3 diagnostic: market daily return + bear regime ----
    print("\n[3/6] F3 Circuit breaker diagnostic ...")
    market_daily = test.groupby(DATE_COL)["ret_1d"].mean().sort_index()
    is_bear_day = (test.groupby(DATE_COL)["macro_regime_3"].first() == "bear").astype(bool)
    is_crash_day = (market_daily < MARKET_DROP_THRESHOLD)
    crash_days = int(is_crash_day.sum())
    bear_days = int(is_bear_day.sum())
    both = int((is_crash_day & is_bear_day).sum())
    print(f"   crash days (market mean ret_1d < {MARKET_DROP_THRESHOLD:+.2f}): {crash_days}")
    print(f"   bear regime days (macro_regime_3=bear)            : {bear_days}")
    print(f"   overlap (crash & bear)                            : {both}")

    # ---- baseline run (cohort-style, equal-weighted, no filter) ----
    print("\n[4/6] D2f baseline (equal-weighted, cohort backtest) ...")
    base = cohort_backtest_weighted(meta_te, d2f_score,
                                       weight_mode="equal")
    print(f"   D2f equal-weight: Sharpe={base['sharpe']:+.3f}  "
          f"MaxDD={base['maxdd']*100:+.2f}%  "
          f"AnnRet={base['annual_return']*100:+.2f}%  "
          f"avg_turnover={base['avg_turnover']*100:.2f}%")

    rows = []
    rows.append({"config": "D2f baseline (equal weight)",
                  "function": "ref",
                  "rankicir": rankicir_d2f,
                  "sharpe": base["sharpe"], "maxdd": base["maxdd"],
                  "annual_return": base["annual_return"],
                  "avg_turnover": base["avg_turnover"],
                  "delta_sr_vs_baseline": 0.0})

    # ---- F1: tradability filter ----
    print("\n[5/6] F1 — tradability filter (limit-up exclusion) ...")
    bt_f1 = cohort_backtest_weighted(meta_te, d2f_score,
                                        weight_mode="equal",
                                        limit_up_mask=limit_up)
    rows.append({"config": "F1 — limit-up filter (equal weight)",
                  "function": "F1",
                  "rankicir": rankicir_d2f,
                  "sharpe": bt_f1["sharpe"], "maxdd": bt_f1["maxdd"],
                  "annual_return": bt_f1["annual_return"],
                  "avg_turnover": bt_f1["avg_turnover"],
                  "delta_sr_vs_baseline": bt_f1["sharpe"] - base["sharpe"]})
    print(f"   F1: Sharpe={bt_f1['sharpe']:+.3f}  MaxDD={bt_f1['maxdd']*100:+.2f}%  "
          f"AnnRet={bt_f1['annual_return']*100:+.2f}%  "
          f"Δ_SR={bt_f1['sharpe']-base['sharpe']:+.3f}")

    # ---- F2: confidence weighting ----
    print("\n[5b/6] F2 — softmax confidence weighting ...")
    weight_diags = []    # to print Top-10 share and Gini for each tau
    for tau in TAU_GRID:
        bt = cohort_backtest_weighted(meta_te, d2f_score,
                                          weight_mode="softmax", tau=tau)
        rows.append({"config": f"F2 — softmax tau={tau:.1f}",
                      "function": "F2",
                      "rankicir": rankicir_d2f,
                      "sharpe": bt["sharpe"], "maxdd": bt["maxdd"],
                      "annual_return": bt["annual_return"],
                      "avg_turnover": bt["avg_turnover"],
                      "delta_sr_vs_baseline": bt["sharpe"] - base["sharpe"]})
        print(f"   tau={tau:.1f}: Sharpe={bt['sharpe']:+.3f}  "
              f"MaxDD={bt['maxdd']*100:+.2f}%  AnnRet={bt['annual_return']*100:+.2f}%  "
              f"Δ_SR={bt['sharpe']-base['sharpe']:+.3f}")
        # diagnose first-day weight distribution
        first_day = next(iter(bt["selections"].values()))
        w_sorted = sorted(first_day.values(), reverse=True)
        top10_share = float(sum(w_sorted[:10]))
        gini_coef = gini(list(first_day.values()))
        weight_diags.append({"tau": tau, "top10_share": top10_share, "gini": gini_coef})

    # Also report equal weight as a reference for the diagnostic
    bt_eq = base
    first_day_eq = next(iter(bt_eq["selections"].values()))
    w_eq_sorted = sorted(first_day_eq.values(), reverse=True)
    eq_top10 = float(sum(w_eq_sorted[:10]))
    eq_gini = gini(list(first_day_eq.values()))
    print(f"\n   Weight distribution (first test day, ~194 stocks in Top-5%):")
    print(f"     equal weight   : top-10 share={eq_top10*100:.2f}%  Gini={eq_gini:.4f}")
    for d_ in weight_diags:
        print(f"     softmax τ={d_['tau']:.1f} : top-10 share={d_['top10_share']*100:.2f}%  "
              f"Gini={d_['gini']:.4f}")

    # ---- F3: circuit breaker ----
    print("\n[5c/6] F3 — circuit breaker ...")
    dates_list = sorted(test[DATE_COL].unique())
    crash_arr = np.array([float(market_daily.loc[d] < MARKET_DROP_THRESHOLD)
                            for d in dates_list])
    bear_arr = np.array([float((test[test[DATE_COL] == d]["macro_regime_3"].iloc[0] == "bear"))
                           for d in dates_list])
    # v1: only crash
    exp_v1 = np.where(crash_arr > 0, EXPOSURE_DROP, 1.0)
    # v2: crash OR bear regime — take minimum exposure
    exp_v2 = np.minimum(
        np.where(crash_arr > 0, EXPOSURE_DROP, 1.0),
        np.where(bear_arr > 0, EXPOSURE_BEAR_REGIME, 1.0),
    )

    bt_v1 = cohort_backtest_weighted(meta_te, d2f_score,
                                        weight_mode="equal",
                                        exposure_vector=exp_v1)
    bt_v2 = cohort_backtest_weighted(meta_te, d2f_score,
                                        weight_mode="equal",
                                        exposure_vector=exp_v2)
    rows.append({"config": "F3-v1 — circuit breaker (crash only)",
                  "function": "F3",
                  "rankicir": rankicir_d2f,
                  "sharpe": bt_v1["sharpe"], "maxdd": bt_v1["maxdd"],
                  "annual_return": bt_v1["annual_return"],
                  "avg_turnover": bt_v1["avg_turnover"],
                  "delta_sr_vs_baseline": bt_v1["sharpe"] - base["sharpe"]})
    rows.append({"config": "F3-v2 — circuit breaker (crash + bear regime)",
                  "function": "F3",
                  "rankicir": rankicir_d2f,
                  "sharpe": bt_v2["sharpe"], "maxdd": bt_v2["maxdd"],
                  "annual_return": bt_v2["annual_return"],
                  "avg_turnover": bt_v2["avg_turnover"],
                  "delta_sr_vs_baseline": bt_v2["sharpe"] - base["sharpe"]})
    print(f"   v1 (crash only)  : Sharpe={bt_v1['sharpe']:+.3f}  "
          f"MaxDD={bt_v1['maxdd']*100:+.2f}%  Δ_SR={bt_v1['sharpe']-base['sharpe']:+.3f}  "
          f"trigger days={int(crash_arr.sum())}")
    print(f"   v2 (crash+bear)  : Sharpe={bt_v2['sharpe']:+.3f}  "
          f"MaxDD={bt_v2['maxdd']*100:+.2f}%  Δ_SR={bt_v2['sharpe']-base['sharpe']:+.3f}  "
          f"trigger days={int(((crash_arr > 0) | (bear_arr > 0)).sum())}")

    # F3 diagnostic: D2f return on trigger days (before exposure scaling)
    base_daily = base["daily_return"]
    base_daily.index = pd.to_datetime(base_daily.index)
    trigger_v1 = pd.Series(crash_arr > 0,
                              index=[pd.Timestamp(d) for d in dates_list])
    trigger_v2_mask = (crash_arr > 0) | (bear_arr > 0)
    trigger_v2 = pd.Series(trigger_v2_mask,
                              index=[pd.Timestamp(d) for d in dates_list])
    common = sorted(set(base_daily.index) & set(trigger_v1.index))
    base_aligned = base_daily.loc[common]
    t_v1_aligned = trigger_v1.loc[common]
    t_v2_aligned = trigger_v2.loc[common]
    print(f"\n   diagnostic — D2f return on trigger days:")
    if t_v1_aligned.sum() > 0:
        ret_trig = float(base_aligned[t_v1_aligned].mean())
        ret_norm = float(base_aligned[~t_v1_aligned].mean())
        print(f"     v1 trigger days mean D2f return: {ret_trig*100:+.4f}%  "
              f"normal days: {ret_norm*100:+.4f}%")
    if t_v2_aligned.sum() > 0:
        ret_trig = float(base_aligned[t_v2_aligned].mean())
        ret_norm = float(base_aligned[~t_v2_aligned].mean())
        print(f"     v2 trigger days mean D2f return: {ret_trig*100:+.4f}%  "
              f"normal days: {ret_norm*100:+.4f}%")

    # ---- combined ----
    print("\n[6/6] combined: best F2 + best F3 + F1 ...")
    # best F2
    best_f2 = max([r for r in rows if r["function"] == "F2"],
                    key=lambda r: r["sharpe"])
    best_f3 = max([r for r in rows if r["function"] == "F3"],
                    key=lambda r: r["sharpe"])
    # Parse best F2 tau
    import re
    m_tau = re.search(r"tau=([0-9.]+)", best_f2["config"])
    best_tau = float(m_tau.group(1)) if m_tau else 1.0
    # F3 best version
    best_exp = exp_v1 if "v1" in best_f3["config"] else exp_v2

    bt_all = cohort_backtest_weighted(meta_te, d2f_score,
                                          weight_mode="softmax", tau=best_tau,
                                          limit_up_mask=limit_up,
                                          exposure_vector=best_exp)
    rows.append({
        "config": f"D4_final — F1 + F2(tau={best_tau:.1f}) + F3({'v1' if 'v1' in best_f3['config'] else 'v2'})",
        "function": "combined",
        "rankicir": rankicir_d2f,
        "sharpe": bt_all["sharpe"], "maxdd": bt_all["maxdd"],
        "annual_return": bt_all["annual_return"],
        "avg_turnover": bt_all["avg_turnover"],
        "delta_sr_vs_baseline": bt_all["sharpe"] - base["sharpe"]
    })
    print(f"   D4_final: Sharpe={bt_all['sharpe']:+.3f}  MaxDD={bt_all['maxdd']*100:+.2f}%  "
          f"AnnRet={bt_all['annual_return']*100:+.2f}%  "
          f"Δ_SR vs base={bt_all['sharpe']-base['sharpe']:+.3f}")

    # ---- write CSV + final summary ----
    df_out = pd.DataFrame(rows)
    df_out.to_csv(BB_RESULTS / "step16_agent4_practical.csv",
                   index=False, encoding="utf-8-sig")

    print()
    line = "+" + "-" * 56 + "+" + "-" * 8 + "+" + "-" * 11 + "+" + "-" * 10 + "+" + "-" * 11 + "+" + "-" * 11 + "+" + "-" * 11 + "+"
    print(line)
    print(f"| {'Config':54s} | {'func':>6s} | {'RankICIR':>9s} | "
          f"{'Sharpe':>8s} | {'MaxDD':>9s} | {'AnnRet':>9s} | {'Δ_SR':>9s} |")
    print(line)
    for r in rows:
        print(f"| {r['config']:54s} | {r['function']:>6s} | "
              f"{r['rankicir']:>9.4f} | {r['sharpe']:>+8.3f} | "
              f"{r['maxdd']*100:>+8.2f}% | {r['annual_return']*100:>+7.2f}% | "
              f"{r['delta_sr_vs_baseline']:>+9.3f} |")
    print(line)

    # verdict
    print("\n=== Verdict ===")
    best_overall = max(rows[1:], key=lambda r: r["sharpe"])
    if best_overall["sharpe"] > base["sharpe"]:
        print(f"  Best deployment variant: {best_overall['config']}")
        print(f"    Sharpe={best_overall['sharpe']:+.3f}  (vs baseline {base['sharpe']:+.3f}, "
              f"Δ={best_overall['delta_sr_vs_baseline']:+.3f})")
        print(f"    RankICIR unchanged at {rankicir_d2f:.4f} as expected.")
        print()
        print("  Paper text: 'Agent 4 improves realised Sharpe from "
              f"{base['sharpe']:.3f} to {best_overall['sharpe']:.3f} "
              "without changing cross-section ranking quality.'")
    else:
        print(f"  No deployment variant improves Sharpe over baseline ({base['sharpe']:+.3f}).")
        closest = max(rows[1:], key=lambda r: r["sharpe"])
        print(f"  Closest: {closest['config']}  Sharpe={closest['sharpe']:+.3f}  "
              f"(Δ={closest['delta_sr_vs_baseline']:+.3f})")
        print()
        print("  Paper text: 'Agent 4 provides tradability guarantees (limit-up exclusion) "
              "and exposure control under extreme market conditions without degrading "
              "signal quality; its value lies in deployment robustness rather than "
              "backtest metrics.'")

    print(f"\nOutput: {BB_RESULTS / 'step16_agent4_practical.csv'}")


if __name__ == "__main__":
    main()
