"""Step 19 — Elliott Wave with deviation=0.02, Method B only.

5-minute experiment: rerun the rolling Elliott Wave analyser with a finer
ZigZag deviation threshold (0.02 instead of 0.05) and evaluate Method B
(wave-aided alpha) on the D2f + F3-v2 baseline.

Diagnostic outputs:
  - wave_signal distribution under deviation=0.02
  - mean pivot count per 252-day window
  - RankICIR and Sharpe of Method B
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
DEVIATION = 0.02            # changed from 0.05 to 0.02
MARKET_DROP_THRESHOLD = -0.03
EXPOSURE_DROP = 0.5
EXPOSURE_BEAR = 0.7
CSI_PATH = Path(r"D:/project1/pythonProject/QMT/Season & Weather/中证全指历史数据 (2).csv")


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


def map_wave_location_to_signal(loc: str) -> float:
    if not loc or loc == "未知":
        return 0.0
    if "C浪末端" in loc or "调整浪末期" in loc or "调整末期" in loc or "准备买入" in loc:
        return +0.8
    if "推动浪末期" in loc or "5浪末端" in loc or "警惕调整" in loc:
        return +0.3
    if "1浪" in loc or "3浪" in loc or "推动浪" in loc:
        return +1.0
    if "A浪" in loc or "C浪下跌" in loc:
        return -0.5
    if "B浪" in loc:
        return -0.2
    if "2浪" in loc or "4浪" in loc:
        return -0.1
    return 0.0


def main():
    print("=" * 78)
    print(f"Step 19 — Elliott Wave with deviation={DEVIATION} (Method B only)")
    print("=" * 78)

    # ---- 1. Load CSI All-Share ----
    idx = pd.read_csv(CSI_PATH, thousands=",")
    idx = idx[["日期", "开盘", "高", "低", "收盘"]].rename(
        columns={"日期": "date", "开盘": "open", "高": "high", "低": "low", "收盘": "close"})
    for c in ["open", "high", "low", "close"]:
        idx[c] = idx[c].astype(str).str.replace(",", "").astype(float)
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.sort_values("date").reset_index(drop=True)
    print(f"   CSI rows: {len(idx)} ({idx['date'].min().date()} - {idx['date'].max().date()})")

    # ---- 2. Rolling Elliott Wave with deviation=0.02 ----
    print(f"\n[Wave analysis] rolling {ELLIOTT_WINDOW}-day windows, deviation={DEVIATION} ...")
    test_dates = idx[(idx["date"] >= pd.Timestamp(TEST_START))
                      & (idx["date"] <= pd.Timestamp(TEST_END))]["date"].tolist()
    wave_signals = {}
    wave_locs = {}
    pivot_counts = []
    t0 = time.time()
    for i, t in enumerate(test_dates):
        win = idx[idx["date"] < t].tail(ELLIOTT_WINDOW)
        if len(win) < ELLIOTT_WINDOW:
            wave_signals[t] = 0.0; wave_locs[t] = "insufficient"
            continue
        try:
            analyzer = ElliottWaveAnalyzer(deviation=DEVIATION)
            r = analyzer.analyze(win.reset_index(drop=True))
            pivots = r.get("pivots", [])
            pivot_counts.append(len(pivots))
            pos = r.get("current_position", {})
            loc = str(pos.get("wave_location", "未知"))
            wave_signals[t] = map_wave_location_to_signal(loc)
            wave_locs[t] = loc
        except Exception as e:
            wave_signals[t] = 0.0
            wave_locs[t] = f"error:{type(e).__name__}"
            pivot_counts.append(0)
        if (i + 1) % 100 == 0:
            print(f"   {i+1}/{len(test_dates)} done ({time.time()-t0:.1f}s)")
    print(f"   total time: {time.time()-t0:.1f}s")

    # Diagnostics
    sig_arr = np.array([wave_signals[t] for t in test_dates])
    pivots_arr = np.array(pivot_counts)
    print(f"\n   pivot count per window: mean={pivots_arr.mean():.1f}  "
          f"median={np.median(pivots_arr):.0f}  max={pivots_arr.max()}  min={pivots_arr.min()}")
    print(f"   wave_signal distribution:")
    for v in (1.0, 0.8, 0.3, 0.0, -0.1, -0.2, -0.5):
        cnt = int(np.sum(np.isclose(sig_arr, v)))
        if cnt > 0:
            print(f"     {v:+.1f}: {cnt:>4d} days ({cnt/len(test_dates)*100:.1f}%)")
    print(f"   wave_signal: mean={sig_arr.mean():+.3f}  std={sig_arr.std(ddof=0):.3f}")

    loc_series = pd.Series([wave_locs[t] for t in test_dates])
    print(f"\n   top wave_location strings:")
    for loc, c in loc_series.value_counts().head(10).items():
        print(f"     {loc:40s}  {c:>4d} days ({c/len(test_dates)*100:.1f}%)")

    # Gate check
    if pivots_arr.mean() < 5:
        print(f"\n   GATE FAILED: mean pivot count {pivots_arr.mean():.1f} < 5")
        print("   Elliott Wave still cannot extract enough pivots. STOP.")
        return

    # ---- 3. Method B evaluation ----
    print("\n[Method B evaluation] wave-aided alpha on D2f + F3-v2 ...")
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
    test["wave_signal"] = test[DATE_COL].map(wave_signals).fillna(0.0).astype("float32")

    alpha_t = regime_alpha_t(test)
    gamma_t = regime_gamma_t(test, GAMMA_BASE, GAMMA_DELTA)
    wave_arr = test["wave_signal"].to_numpy("float32")
    # Method B: alpha += 0.1 * max(0, -wave)
    alpha_t_b = np.minimum(alpha_t + 0.1 * np.maximum(0.0, -wave_arr), 0.85).astype("float32")
    d2f_b = (test["bull_z"].to_numpy("float32")
              - alpha_t_b * test["bear_z"].to_numpy("float32")
              + gamma_t * test["rev_z"].to_numpy("float32")).astype("float32")

    # baseline D2f
    d2f_ref = (test["bull_z"].to_numpy("float32")
                - alpha_t * test["bear_z"].to_numpy("float32")
                + gamma_t * test["rev_z"].to_numpy("float32")).astype("float32")

    meta_proxy = test[[DATE_COL, TICKER_COL, TARGET_RET_COL]].reset_index(drop=True)
    # RankICIR
    ric_ref = float(evaluate_full(meta_proxy, d2f_ref)["rankicir"])
    ric_b   = float(evaluate_full(meta_proxy, d2f_b)["rankicir"])

    # F3-v2 exposure
    market_daily = test.groupby(DATE_COL)["ret_1d"].mean().sort_index()
    is_bear = test.groupby(DATE_COL)["macro_regime_3"].first() == "bear"
    exp_f3 = pd.Series(1.0, index=market_daily.index)
    exp_f3[market_daily < MARKET_DROP_THRESHOLD] = EXPOSURE_DROP
    exp_f3[is_bear] = exp_f3.combine(
        pd.Series(EXPOSURE_BEAR, index=is_bear[is_bear].index),
        np.minimum, fill_value=1.0)

    def sharpe_maxdd(meta, pred, exp_map):
        bt = backtest_topk(meta, pred, frac=0.05)
        s = bt.daily_return.copy()
        s.index = pd.to_datetime(s.index)
        exp_series = pd.Series(exp_map).reindex(s.index).fillna(1.0).to_numpy()
        s_arr = s.to_numpy() * exp_series
        ann_ret = float(np.mean(s_arr) * 252)
        ann_vol = float(np.std(s_arr, ddof=0) * np.sqrt(252))
        sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
        nav = (1.0 + pd.Series(s_arr, index=s.index)).cumprod()
        dd = float((nav / nav.cummax() - 1).min())
        return ann_ret, ann_vol, sharpe, dd

    ar_ref, av_ref, sr_ref, dd_ref = sharpe_maxdd(meta_proxy, d2f_ref, exp_f3.to_dict())
    ar_b,   av_b,   sr_b,   dd_b   = sharpe_maxdd(meta_proxy, d2f_b,   exp_f3.to_dict())

    print(f"\n   D2f + F3-v2 (ref) : RankICIR={ric_ref:.4f}  Sharpe={sr_ref:+.4f}  "
          f"MaxDD={dd_ref*100:+.2f}%  AnnRet={ar_ref*100:+.2f}%")
    print(f"   Method B (wave α) : RankICIR={ric_b:.4f}  Sharpe={sr_b:+.4f}  "
          f"MaxDD={dd_b*100:+.2f}%  AnnRet={ar_b*100:+.2f}%")
    print(f"   Δ RankICIR        : {(ric_b - ric_ref)*10000:+.1f} bp")
    print(f"   Δ Sharpe          : {sr_b - sr_ref:+.4f}")
    print(f"   Δ MaxDD           : {(dd_b - dd_ref)*100:+.2f} pp")

    # Verdict
    print()
    if pivots_arr.mean() < 5:
        print(">>> GATE FAILED: pivot count < 5; Elliott Wave technical limit.")
    elif ric_b < ric_ref:
        print(">>> Method B RankICIR < baseline. Elliott Wave cannot integrate without RankICIR loss.")
    elif sr_b <= sr_ref:
        print(">>> Method B Sharpe ≤ baseline. No Sharpe benefit either.")
    else:
        print(">>> Method B improves both RankICIR and Sharpe under deviation=0.02.")

    # Save
    pd.DataFrame([
        {"variant": "D2f + F3-v2 (ref)",
         "rankicir": ric_ref, "sharpe": sr_ref, "maxdd": dd_ref, "annual_return": ar_ref,
         "wave_mean": float(sig_arr.mean()), "wave_std": float(sig_arr.std(ddof=0)),
         "pivot_mean": float(pivots_arr.mean()), "deviation": DEVIATION},
        {"variant": f"Method B wave-aided alpha (deviation={DEVIATION})",
         "rankicir": ric_b, "sharpe": sr_b, "maxdd": dd_b, "annual_return": ar_b,
         "wave_mean": float(sig_arr.mean()), "wave_std": float(sig_arr.std(ddof=0)),
         "pivot_mean": float(pivots_arr.mean()), "deviation": DEVIATION},
    ]).to_csv(BB_RESULTS / "step19_wave_dev02_methodB.csv",
                index=False, encoding="utf-8-sig")
    print(f"\nOutput: {BB_RESULTS / 'step19_wave_dev02_methodB.csv'}")


if __name__ == "__main__":
    main()
