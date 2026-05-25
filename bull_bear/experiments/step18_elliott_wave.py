"""Step 18 — Elliott Wave market-timing signal integration.

Baseline: D2f + F3-v2 (Sharpe 1.894, MaxDD -27.27%, RankICIR 0.8094).

Run ElliottWaveAnalyzer on a rolling 252-day window of the CSI All-Share
index for each test trading day. Map wave_location strings to numeric
wave_signal in [-1, +1]. Test three integration schemes:

  A — replace F3-v2's bear-regime trigger with wave-based trigger
       (exposure 0.7 when wave_signal < -0.3)
  B — wave-aided alpha: alpha(t) += 0.1 * max(0, -wave_signal[t])
  C — replace Regime Agent with wave-based P_bull/P_bear

Outputs:
  bull_bear/results/step_wave_signal_diag.csv
  bull_bear/results/step_wave_integration.csv
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))
warnings.filterwarnings("ignore")

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

# Import Elliott Wave analyzer from top-level module
sys.path.insert(0, str(_HERE.parents[2]))
from waves_agent import ElliottWaveAnalyzer


ALPHA_FEATURES = ["ma60_slope", "ema180_slope", "bias_60", "bias_60_vr", "ma180_slope"]
ALPHA_BY_REGIME = {"bear": 0.65, "sideway": 0.50, "bull": 0.35}
GAMMA_BASE = 0.40
GAMMA_DELTA = 0.15
REVERSAL_FEATURES = [
    "ret_1d", "ret_3d",
    "rev_ret_2d", "rev_ret_3d_minus_1d",
    "rev_zscore_1d", "rev_mkt_excess_1d",
]
ELLIOTT_WINDOW = 252
ELLIOTT_DEVIATION = 0.05
MARKET_DROP_THRESHOLD = -0.03
EXPOSURE_DROP = 0.5
EXPOSURE_BEAR = 0.7

CSI_PATH = Path(r"D:/project1/pythonProject/QMT/Season & Weather/中证全指历史数据 (2).csv")


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


def regime_alpha_t(panel, alpha_by_regime=ALPHA_BY_REGIME):
    regime = panel["macro_regime_3"].astype(str).to_numpy()
    return np.array([alpha_by_regime.get(r, 0.5) for r in regime], dtype="float32")


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


def map_wave_location_to_signal(loc: str) -> float:
    """Map wave_location text to numeric signal in [-1, +1]."""
    if not loc or loc == "未知":
        return 0.0
    # priority order matters: check most-specific keywords first
    if "C浪末端" in loc or "调整浪末期" in loc or "调整末期" in loc or "准备买入" in loc:
        return +0.8
    if "推动浪末期" in loc or "5浪末端" in loc or "警惕调整" in loc:
        return +0.3
    if "1浪" in loc or "3浪" in loc or "推动浪" in loc:
        return +1.0
    if "A浪" in loc or "C浪下跌" in loc:
        return -0.5
    if "B浪" in loc:
        return -0.2    # B浪反弹仍属于调整结构
    if "2浪" in loc or "4浪" in loc:
        return -0.1
    return 0.0


def evaluate_full_two_track(meta_proxy, meta_real, pred, daily_index):
    """Return RankICIR + proxy Sharpe/MaxDD + real (ret_1d) Sharpe/MaxDD.

    daily_index is a per-day exposure multiplier (1.0 = full position).
    """
    # RankICIR from proxy meta (unchanged regardless of exposure)
    m = evaluate_full(meta_proxy, pred)
    bt_proxy = backtest_topk(meta_proxy, pred, frac=0.05)
    # Apply exposure multiplier to daily returns
    s_proxy = bt_proxy.daily_return.copy()
    s_proxy.index = pd.to_datetime(s_proxy.index)
    s_proxy = s_proxy.sort_index()
    exposure = pd.Series(daily_index).reindex(s_proxy.index).fillna(1.0).to_numpy()
    s_scaled = s_proxy.to_numpy() * exposure
    ann_ret = float(np.mean(s_scaled) * 252)
    ann_vol = float(np.std(s_scaled, ddof=0) * np.sqrt(252))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
    nav = (1.0 + pd.Series(s_scaled, index=s_proxy.index)).cumprod()
    drawdown = nav / nav.cummax() - 1
    max_dd = float(drawdown.min())
    return {"rankicir": float(m["rankicir"]),
             "sharpe":   sharpe,
             "maxdd":    max_dd,
             "annual_return": ann_ret}


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("=" * 80)
    print("Step 18 — Elliott Wave market timing signal integration")
    print("=" * 80)

    # ---- 1. Load CSI All-Share index OHLCV ----
    print("\n[1/6] load CSI All-Share index OHLCV ...")
    idx = pd.read_csv(CSI_PATH, thousands=",")
    idx = idx[["日期", "开盘", "高", "低", "收盘"]].rename(
        columns={"日期": "date", "开盘": "open", "高": "high", "低": "low", "收盘": "close"}
    )
    for c in ["open", "high", "low", "close"]:
        idx[c] = idx[c].astype(str).str.replace(",", "").astype(float)
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.sort_values("date").reset_index(drop=True)
    print(f"   index range: {idx['date'].min().date()} to {idx['date'].max().date()}  "
          f"({len(idx):,} rows)")

    # ---- 2. Run Elliott Wave rolling for each test date ----
    print(f"\n[2/6] running Elliott Wave on rolling {ELLIOTT_WINDOW}-day windows ...")
    t0 = time.time()
    # Test date range
    test_dates = idx[(idx["date"] >= pd.Timestamp(TEST_START))
                       & (idx["date"] <= pd.Timestamp(TEST_END))]["date"].tolist()
    print(f"   test dates: {len(test_dates)}  ({test_dates[0].date()} - {test_dates[-1].date()})")

    wave_signals = {}
    wave_locations = {}
    success_count = 0
    failure_count = 0
    for i, t in enumerate(test_dates):
        # rolling window: dates < t
        win = idx[idx["date"] < t].tail(ELLIOTT_WINDOW)
        if len(win) < ELLIOTT_WINDOW:
            wave_signals[t] = 0.0
            wave_locations[t] = "insufficient_history"
            failure_count += 1
            continue
        try:
            analyzer = ElliottWaveAnalyzer(deviation=ELLIOTT_DEVIATION)
            r = analyzer.analyze(win.reset_index(drop=True))
            pos = r.get("current_position", {})
            loc = str(pos.get("wave_location", "未知"))
            sig = map_wave_location_to_signal(loc)
            wave_signals[t] = sig
            wave_locations[t] = loc
            if loc != "未知":
                success_count += 1
            else:
                failure_count += 1
        except Exception as e:
            wave_signals[t] = 0.0
            wave_locations[t] = f"error:{type(e).__name__}"
            failure_count += 1
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(test_dates) - i - 1)
            print(f"   processed {i+1}/{len(test_dates)} ({elapsed:.1f}s, ETA {eta:.1f}s)  "
                  f"success={success_count}  unknown/error={failure_count}")
    print(f"   total time: {time.time()-t0:.1f}s")
    print(f"   identified valid waves: {success_count}/{len(test_dates)} = "
          f"{success_count/len(test_dates)*100:.1f}%")

    # Diagnostic on wave_signal distribution
    sig_arr = np.array([wave_signals[t] for t in test_dates])
    print(f"\n   wave_signal distribution:")
    for v in (1.0, 0.8, 0.3, 0.0, -0.1, -0.2, -0.5):
        cnt = int(np.sum(np.isclose(sig_arr, v)))
        if cnt > 0:
            print(f"     signal={v:+.1f}: {cnt:>4d} days ({cnt/len(test_dates)*100:.1f}%)")
    print(f"     overall mean = {sig_arr.mean():+.3f}  std = {sig_arr.std(ddof=0):.3f}")

    # Most common wave_location strings
    loc_arr = pd.Series([wave_locations[t] for t in test_dates])
    print(f"\n   top wave_location strings:")
    for loc, c in loc_arr.value_counts().head(10).items():
        print(f"     {loc:40s}  {c:>4d} days ({c/len(test_dates)*100:.1f}%)")

    # save diag
    pd.DataFrame({
        "date": test_dates,
        "wave_signal": [wave_signals[t] for t in test_dates],
        "wave_location": [wave_locations[t] for t in test_dates],
    }).to_csv(BB_RESULTS / "step_wave_signal_diag.csv",
                index=False, encoding="utf-8-sig")

    # ---- 3. Load D2f conviction on test ----
    print("\n[3/6] build D2f conviction on test panel ...")
    df = load_dataset().dropna(subset=[TARGET_RET_COL]).reset_index(drop=True)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = add_reversal_features(df)
    mask_te = (df[DATE_COL] >= pd.Timestamp(TEST_START)) & (df[DATE_COL] <= pd.Timestamp(TEST_END))
    test = df.loc[mask_te].sort_values([DATE_COL, TICKER_COL]).reset_index(drop=True)

    alpha_m = CatBoostRegressor(); alpha_m.load_model(str(ALPHA_AGENT_PATH))
    alpha_med = pd.read_csv(str(ALPHA_AGENT_PATH).replace(".cbm", "_medians.csv"),
                              index_col=0).iloc[:, 0]
    test["alpha_score"] = alpha_m.predict(
        test[ALPHA_FEATURES].astype("float32").fillna(alpha_med)).astype("float32")
    bear_d1 = BearAgent(features=BEAR_FEATURES_D1, name="bear_D1")
    bear_d1.load(BB_MODELS / "bear_D1_agent.cbm")
    bear_d2 = CatBoostRegressor()
    bear_d2.load_model(str(BB_MODELS / "bear_D2_l3.0.cbm"))
    test["bear_score"] = bear_d2.predict(
        test[BEAR_FEATURES_D1].astype("float32").fillna(bear_d1._train_medians)
    ).astype("float32")
    rev = CatBoostRegressor(); rev.load_model(str(BB_MODELS / "reversal_B_5d.cbm"))
    rev_medians = test[REVERSAL_FEATURES].median()
    test["rev_score"] = rev.predict(
        test[REVERSAL_FEATURES].astype("float32").fillna(rev_medians)).astype("float32")

    test["bull_z"] = zscore_daily(test, "alpha_score")
    test["bear_z"] = zscore_daily(test, "bear_score")
    test["rev_z"]  = zscore_daily(test, "rev_score")

    # Map wave_signal per test row
    test["wave_signal"] = test[DATE_COL].map(wave_signals).fillna(0.0).astype("float32")

    # ---- 4. Build baselines ----
    meta_proxy = test[[DATE_COL, TICKER_COL, TARGET_RET_COL]].reset_index(drop=True)

    alpha_t = regime_alpha_t(test)
    gamma_t = regime_gamma_t(test, GAMMA_BASE, GAMMA_DELTA)
    d2f_score = (test["bull_z"].to_numpy("float32")
                  - alpha_t * test["bear_z"].to_numpy("float32")
                  + gamma_t * test["rev_z"].to_numpy("float32")).astype("float32")

    # F3-v2 exposure: crash_day OR bear_regime
    market_daily = test.groupby(DATE_COL)["ret_1d"].mean().sort_index()
    is_bear_regime = (test.groupby(DATE_COL)["macro_regime_3"].first() == "bear")

    def exposure_f3v2():
        exp = pd.Series(1.0, index=market_daily.index)
        exp[market_daily < MARKET_DROP_THRESHOLD] = EXPOSURE_DROP
        exp[is_bear_regime] = exp.combine(
            pd.Series(EXPOSURE_BEAR, index=is_bear_regime[is_bear_regime].index),
            np.minimum, fill_value=1.0)
        return exp

    exp_f3v2 = exposure_f3v2()

    # ---- 5. Three integration methods ----
    print("\n[4/6] evaluate baselines + 3 wave integration variants ...")

    rows = []
    # full exposure baseline
    full_exp = pd.Series(1.0, index=market_daily.index)
    r_d2f = evaluate_full_two_track(meta_proxy, None, d2f_score, full_exp.to_dict())
    rows.append({"config": "D2f baseline (no exposure)", **r_d2f, "track": "ref"})
    # D2f + F3-v2
    r_f3 = evaluate_full_two_track(meta_proxy, None, d2f_score, exp_f3v2.to_dict())
    rows.append({"config": "D2f + F3-v2 (canonical)", **r_f3, "track": "ref"})

    # daily wave_signal series aligned with market_daily dates
    wave_series = pd.Series({pd.Timestamp(d): wave_signals.get(pd.Timestamp(d), 0.0)
                                for d in market_daily.index})

    # --- Method A — replace F3-v2 bear trigger with wave_signal < -0.3 ---
    exp_a = pd.Series(1.0, index=market_daily.index)
    exp_a[market_daily < MARKET_DROP_THRESHOLD] = EXPOSURE_DROP
    exp_a[wave_series < -0.3] = exp_a.combine(
        pd.Series(EXPOSURE_BEAR, index=wave_series[wave_series < -0.3].index),
        np.minimum, fill_value=1.0)
    r_a = evaluate_full_two_track(meta_proxy, None, d2f_score, exp_a.to_dict())
    n_trig_a = int(((market_daily < MARKET_DROP_THRESHOLD) | (wave_series < -0.3)).sum())
    rows.append({"config": f"A) wave-triggered F3 (wave<-0.3) {n_trig_a} trigger days",
                  **r_a, "track": "A"})

    # --- Method B — wave-aided alpha: alpha += 0.1 * max(0, -wave_signal) ---
    # Alpha-by-regime: bear=0.65, sideway=0.50, bull=0.35.
    # Add wave correction.
    wave_signal_arr = test["wave_signal"].to_numpy("float32")
    alpha_t_corrected = (alpha_t + 0.1 * np.maximum(0.0, -wave_signal_arr)).astype("float32")
    alpha_t_corrected = np.minimum(alpha_t_corrected, 0.85)    # safety cap
    d2f_method_b = (test["bull_z"].to_numpy("float32")
                     - alpha_t_corrected * test["bear_z"].to_numpy("float32")
                     + gamma_t * test["rev_z"].to_numpy("float32")).astype("float32")
    r_b = evaluate_full_two_track(meta_proxy, None, d2f_method_b, exp_f3v2.to_dict())
    rows.append({"config": "B) wave-aided alpha (+F3-v2 exposure)",
                  **r_b, "track": "B"})

    # --- Method C — replace Regime Agent with wave_signal ---
    # P_bear_wave = max(0, -wave), P_bull_wave = max(0, +wave)
    P_bear_wave = np.maximum(0.0, -wave_signal_arr)
    P_bull_wave = np.maximum(0.0, +wave_signal_arr)
    alpha_t_wave = (0.5 + 0.15 * P_bear_wave - 0.15 * P_bull_wave).astype("float32")
    gamma_t_wave = (GAMMA_BASE + GAMMA_DELTA * P_bull_wave - GAMMA_DELTA * P_bear_wave).astype("float32")
    d2f_method_c = (test["bull_z"].to_numpy("float32")
                     - alpha_t_wave * test["bear_z"].to_numpy("float32")
                     + gamma_t_wave * test["rev_z"].to_numpy("float32")).astype("float32")
    r_c = evaluate_full_two_track(meta_proxy, None, d2f_method_c, exp_f3v2.to_dict())
    rows.append({"config": "C) wave replaces Regime (+F3-v2 exposure)",
                  **r_c, "track": "C"})

    # ---- 6. Report ----
    print("\n[5/6] results ...")
    line = "+" + "-" * 58 + "+" + "-" * 8 + "+" + "-" * 11 + "+" + "-" * 10 + "+" + "-" * 11 + "+" + "-" * 11 + "+"
    print()
    print(line)
    print(f"| {'Config':56s} | {'track':>6s} | {'RankICIR':>9s} | "
          f"{'Sharpe':>8s} | {'MaxDD':>9s} | {'AnnRet':>9s} |")
    print(line)
    for r in rows:
        print(f"| {r['config']:56s} | {r.get('track', '-'):>6s} | "
              f"{r['rankicir']:>9.4f} | {r['sharpe']:>+8.3f} | "
              f"{r['maxdd']*100:>+8.2f}% | {r['annual_return']*100:>+7.2f}% |")
    print(line)

    pd.DataFrame(rows).to_csv(BB_RESULTS / "step_wave_integration.csv",
                                index=False, encoding="utf-8-sig")

    # ---- verdict ----
    print("\n=== Verdict ===")
    ref = next(r for r in rows if r["config"] == "D2f + F3-v2 (canonical)")
    print(f"   D2f + F3-v2 baseline: RankICIR={ref['rankicir']:.4f}  "
          f"Sharpe={ref['sharpe']:+.3f}  MaxDD={ref['maxdd']*100:+.2f}%")
    wave_rows = [r for r in rows if r.get("track") in ("A", "B", "C")]
    best_sr = max(wave_rows, key=lambda r: r["sharpe"])
    best_ric = max(wave_rows, key=lambda r: r["rankicir"])
    print(f"\n   Best Sharpe under wave variants:  {best_sr['config']}")
    print(f"     Sharpe={best_sr['sharpe']:+.3f}  vs baseline {ref['sharpe']:+.3f}  "
          f"Δ={best_sr['sharpe']-ref['sharpe']:+.3f}")
    print(f"   Best RankICIR under wave variants: {best_ric['config']}")
    print(f"     RankICIR={best_ric['rankicir']:.4f}  vs baseline {ref['rankicir']:.4f}")
    if best_sr["sharpe"] > ref["sharpe"]:
        print("   -> A wave-based variant improves Sharpe over F3-v2.")
    else:
        print("   -> No wave variant improves Sharpe over F3-v2.")

    print(f"\nOutputs:")
    print(f"  -> {BB_RESULTS / 'step_wave_signal_diag.csv'}")
    print(f"  -> {BB_RESULTS / 'step_wave_integration.csv'}")


if __name__ == "__main__":
    main()
