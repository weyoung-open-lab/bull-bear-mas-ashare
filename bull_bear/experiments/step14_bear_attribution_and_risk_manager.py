"""Step 14 — Bear contribution attribution + Risk Manager (Agent 4 redesign).

Part 1: Bear contribution verification (4 configs)
  Config 1: alpha + reversal           (no Bear, no regime gamma)
  Config 2: D2c (alpha + bear + adapt) (no Reversal)         [known 0.7505]
  Config 3: alpha + gamma(t)*reversal  (no Bear, with regime gamma)
  Config 4: D2f full                   (ref 0.8094)

Part 2: Step 14 Risk Manager Agent
  Direction A — Sector concentration control (threshold-based demotion)
    thresholds: 0.20, 0.25, 0.30, 0.35
  Direction B — Signal consistency filter
    adjustment = min(exp(theta * (-alpha_z * bear_z)), 1)
    theta in {0.3, 0.5, 1.0}
  Direction C — best A + best B combined (only if both >= 0.79 RankICIR)

Outputs:
  bull_bear/results/step14_bear_attribution.csv
  bull_bear/results/step14_risk_manager.csv
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
from src.data import load_dataset
from bull_bear.src.metrics_utils import evaluate_full


ALPHA_FEATURES = ["ma60_slope", "ema180_slope", "bias_60", "bias_60_vr", "ma180_slope"]
ALPHA_BY_REGIME = {"bear": 0.65, "sideway": 0.50, "bull": 0.35}
GAMMA_BASE = 0.40
GAMMA_DELTA = 0.15    # the D2f peak adaptive delta
REVERSAL_FEATURES = [
    "ret_1d", "ret_3d",
    "rev_ret_2d", "rev_ret_3d_minus_1d",
    "rev_zscore_1d", "rev_mkt_excess_1d",
]
SECTOR_COL = "所属行业"
FRAC = 0.05


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


def evaluate(meta, pred, label, extra=None):
    m = evaluate_full(meta, pred.astype("float32"))
    out = {"config": label,
            "rankicir": float(m["rankicir"]),
            "sharpe":   float(m["top5pct_sharpe"]),
            "maxdd":    float(m["top5pct_max_dd"])}
    if extra: out.update(extra)
    return out


def sector_concentration_adjust(panel, conviction, sector_col, threshold,
                                  frac=FRAC, demote=0.5):
    """Apply sector-concentration penalty.

    For each day's Top-K selection, identify sectors exceeding `threshold`
    share. Within each such sector, the lowest-conviction excess stocks
    have their conviction multiplied by `demote`. Returns adjusted
    conviction array (same length as input)."""
    adj = conviction.astype("float64").copy()
    df = panel[[DATE_COL, TICKER_COL, sector_col]].copy().reset_index(drop=True)
    df["conv"] = conviction
    df["row"] = np.arange(len(df))
    for d, g in df.groupby(DATE_COL):
        n = len(g)
        k = max(1, int(np.ceil(n * frac)))
        top = g.sort_values("conv", ascending=False).head(k)
        for sec, sub in top.groupby(sector_col):
            ratio = len(sub) / k
            if ratio > threshold:
                max_allowed = int(np.floor(threshold * k))
                excess = len(sub) - max_allowed
                if excess > 0:
                    excess_rows = sub.sort_values("conv", ascending=True).head(excess)["row"].to_numpy()
                    adj[excess_rows] *= demote
    return adj.astype("float32")


def consistency_adjust(conviction, alpha_z, bear_z, theta):
    """Multiplicative penalty for contradictory (Alpha high + Bear high) signals.

    consistency = -(alpha_z * bear_z)
    adjustment   = min(exp(theta * consistency), 1.0)
    -> contradictory rows scaled down; consistent rows unchanged."""
    consistency = -(alpha_z.astype("float64") * bear_z.astype("float64"))
    adjustment = np.minimum(np.exp(theta * consistency), 1.0)
    return (conviction.astype("float64") * adjustment).astype("float32"), \
            consistency.astype("float32"), adjustment.astype("float32")


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("=" * 80)
    print("Step 14 — Bear attribution + Risk Manager")
    print("=" * 80)

    # ---- load ----
    print("\n[0/8] load data + reversal features + agents ...")
    t0 = time.time()
    df = load_dataset().dropna(subset=[TARGET_RET_COL]).reset_index(drop=True)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    # reversal features
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
    print(f"   test rows={len(test):,}  ({time.time()-t0:.1f}s)")

    # agents
    alpha_m = CatBoostRegressor(); alpha_m.load_model(str(ALPHA_AGENT_PATH))
    alpha_med = pd.read_csv(str(ALPHA_AGENT_PATH).replace(".cbm", "_medians.csv"),
                              index_col=0).iloc[:, 0]
    test["alpha_score"] = alpha_m.predict(
        test[ALPHA_FEATURES].astype("float32").fillna(alpha_med)).astype("float32")

    bear_d1 = BearAgent(features=BEAR_FEATURES_D1, name="bear_D1")
    bear_d1.load(BB_MODELS / "bear_D1_agent.cbm")
    bear_d1_med = bear_d1._train_medians
    test["bear_d1_score"] = bear_d1.predict_panel(test).astype("float32")

    bear_d2 = CatBoostRegressor()
    bear_d2.load_model(str(BB_MODELS / "bear_D2_l3.0.cbm"))
    test["bear_d2_score"] = bear_d2.predict(
        test[BEAR_FEATURES_D1].astype("float32").fillna(bear_d1_med)).astype("float32")

    rev = CatBoostRegressor(); rev.load_model(str(BB_MODELS / "reversal_B_5d.cbm"))
    rev_medians = test[REVERSAL_FEATURES].median()
    test["rev_score"] = rev.predict(
        test[REVERSAL_FEATURES].astype("float32").fillna(rev_medians)).astype("float32")

    test["bull_z"]    = zscore_daily(test, "alpha_score")
    test["bear_d2_z"] = zscore_daily(test, "bear_d2_score")
    test["rev_z"]     = zscore_daily(test, "rev_score")

    alpha_t = regime_alpha_t(test)
    gamma_t = regime_gamma_t(test, GAMMA_BASE, GAMMA_DELTA)
    meta_te = test[[DATE_COL, TICKER_COL, TARGET_RET_COL]].reset_index(drop=True)

    bull = test["bull_z"].to_numpy("float32")
    bear = test["bear_d2_z"].to_numpy("float32")
    revs = test["rev_z"].to_numpy("float32")

    # Pre-computed canonical scores
    trend_score = test["alpha_score"].to_numpy("float32")
    d1c_score   = bull - regime_alpha_t(test) * zscore_daily(test, "bear_d1_score")
    d2c_score   = bull - alpha_t * bear
    d2f_score   = (d2c_score + gamma_t * revs).astype("float32")

    # ============================================================
    # Part 1 — Bear contribution attribution
    # ============================================================
    print("\n[1/8] Part 1 — Bear contribution attribution ...")
    bear_attr_rows = []

    # Config 4: D2f full (reference)
    m_d2f = evaluate(meta_te, d2f_score, "Config 4 D2f full (Alpha+Bear+Reversal+γ_adapt)")
    bear_attr_rows.append({**m_d2f, "delta_bp_vs_d2f": 0.0})

    # Config 2: D2c (no Reversal)
    m_d2c = evaluate(meta_te, d2c_score, "Config 2 D2c (Alpha+Bear+α_adapt, no Reversal)")
    m_d2c["delta_bp_vs_d2f"] = (m_d2c["rankicir"] - m_d2f["rankicir"]) * 10000
    bear_attr_rows.append(m_d2c)

    # Config 1: alpha + 0.4*rev (no Bear, no Regime gamma)
    c1 = (bull + GAMMA_BASE * revs).astype("float32")
    m_c1 = evaluate(meta_te, c1, "Config 1 (Alpha + 0.40·Reversal, no Bear, no γ_adapt)")
    m_c1["delta_bp_vs_d2f"] = (m_c1["rankicir"] - m_d2f["rankicir"]) * 10000
    bear_attr_rows.append(m_c1)

    # Config 3: alpha + gamma_t * rev (no Bear, with Regime gamma)
    c3 = (bull + gamma_t * revs).astype("float32")
    m_c3 = evaluate(meta_te, c3, "Config 3 (Alpha + γ_adapt·Reversal, no Bear)")
    m_c3["delta_bp_vs_d2f"] = (m_c3["rankicir"] - m_d2f["rankicir"]) * 10000
    bear_attr_rows.append(m_c3)

    # Trend reference
    m_T = evaluate(meta_te, trend_score, "Trend single (Alpha alone, ref)")
    m_T["delta_bp_vs_d2f"] = (m_T["rankicir"] - m_d2f["rankicir"]) * 10000
    bear_attr_rows.append(m_T)

    print(f"\n  Bear contribution table:")
    print(f"  {'Config':62s}  {'RankICIR':>9s}  {'Sharpe':>7s}  {'MaxDD':>8s}  {'Δ vs D2f (bp)':>13s}")
    for r in bear_attr_rows:
        print(f"  {r['config']:62s}  {r['rankicir']:>9.4f}  {r['sharpe']:>+7.3f}  "
              f"{r['maxdd']*100:>+7.2f}%  {r['delta_bp_vs_d2f']:>+12.1f}")

    pd.DataFrame(bear_attr_rows).to_csv(
        BB_RESULTS / "step14_bear_attribution.csv",
        index=False, encoding="utf-8-sig")

    # Bear marginal contribution (D2f − Config 3) — Bear's pure addition when reversal is already there
    bear_marginal = (m_d2f["rankicir"] - m_c3["rankicir"]) * 10000
    print(f"\n  Bear marginal contribution given Reversal+γ_adapt is present: "
          f"{bear_marginal:+.1f} bp")
    if abs(bear_marginal) > 50:
        print("  -> Bear provides non-trivial independent contribution to D2f.")
    else:
        print("  -> Bear's marginal contribution in D2f is small; "
              "Reversal+γ_adapt already captures most non-Alpha information.")

    # ============================================================
    # Part 2 — Step 14 Risk Manager
    # ============================================================
    print("\n[2/8] Part 2 — Risk Manager: Direction A (Sector Concentration) ...")

    # A1: diagnose current D2f sector distribution
    print("\n  A1 sector diagnostic (D2f Top-5% selections) ...")
    df_diag = test[[DATE_COL, TICKER_COL, SECTOR_COL]].copy()
    df_diag["conv"] = d2f_score
    sector_share_per_day: dict[str, list[float]] = {}
    for d, gd in df_diag.groupby(DATE_COL):
        k = max(1, int(np.ceil(len(gd) * FRAC)))
        top = gd.sort_values("conv", ascending=False).head(k)
        n_top = len(top)
        vc = top[SECTOR_COL].value_counts() / n_top
        for sec, share in vc.items():
            sector_share_per_day.setdefault(sec, []).append(float(share))
    sec_stats = []
    for sec, shares in sector_share_per_day.items():
        arr = np.array(shares)
        sec_stats.append({"sector": sec, "n_days_appeared": len(arr),
                            "mean_share": float(arr.mean()),
                            "max_share":  float(arr.max())})
    sec_stats = sorted(sec_stats, key=lambda r: -r["mean_share"])
    print(f"    top 10 sectors by mean share in D2f Top-5%:")
    print(f"    {'sector':14s}  {'days':>6s}  {'mean':>7s}  {'max':>7s}")
    for r in sec_stats[:10]:
        print(f"    {r['sector']:14s}  {r['n_days_appeared']:>6d}  "
              f"{r['mean_share']*100:>6.2f}%  {r['max_share']*100:>6.2f}%")

    # A2: threshold sweep
    print("\n  A2 threshold sweep ...")
    risk_rows = []
    risk_rows.append({**m_d2f, "track": "ref",
                       "delta_bp_vs_d2f": 0.0,
                       "param": "baseline"})

    for th in (0.20, 0.25, 0.30, 0.35):
        adj_score = sector_concentration_adjust(
            test, d2f_score, SECTOR_COL, threshold=th, frac=FRAC, demote=0.5)
        m = evaluate(meta_te, adj_score, f"D4a sector_threshold={th:.2f}",
                       extra={"track": "A", "param": f"sector_th={th:.2f}"})
        m["delta_bp_vs_d2f"] = (m["rankicir"] - m_d2f["rankicir"]) * 10000
        risk_rows.append(m)
        print(f"    threshold={th:.2f}  RankICIR={m['rankicir']:.4f}  "
              f"Sharpe={m['sharpe']:+.3f}  MaxDD={m['maxdd']*100:+.2f}%  "
              f"Δ_RIC={m['delta_bp_vs_d2f']:+.1f} bp")

    # ============================================================
    # Direction B — Signal consistency
    # ============================================================
    print("\n[3/8] Direction B — Signal consistency filter ...")

    # B2 diagnostic: bottom vs top 10% consistency stocks
    consistency_all = -(bull * bear).astype("float64")
    # daily percentiles
    cons_pct_low10 = np.zeros(len(test), dtype=bool)
    cons_pct_high10 = np.zeros(len(test), dtype=bool)
    for d, gd in test.groupby(DATE_COL):
        ids = gd.index.to_numpy()
        c = consistency_all[ids]
        lo = np.quantile(c, 0.10)
        hi = np.quantile(c, 0.90)
        cons_pct_low10[ids[c <= lo]] = True
        cons_pct_high10[ids[c >= hi]] = True
    rf = test[TARGET_RET_COL].to_numpy("float64")
    mean_low  = float(rf[cons_pct_low10].mean())
    mean_high = float(rf[cons_pct_high10].mean())
    gap_bp = (mean_high - mean_low) * 10000
    print(f"    contradictory (bottom 10% cons) avg r_future_5 = {mean_low*100:+.4f}%")
    print(f"    consistent    (top 10% cons)    avg r_future_5 = {mean_high*100:+.4f}%")
    print(f"    gap = {gap_bp:+.1f} bp  ({'>50' if abs(gap_bp) > 50 else '<50'} threshold)")
    if abs(gap_bp) > 50:
        print(f"    -> contradictory signals empirically have lower forward returns; "
              "direction B has prior support.")
    else:
        print(f"    -> gap below 50 bp; direction B may have limited room.")

    for theta in (0.3, 0.5, 1.0):
        adj_score, _, adj_factor = consistency_adjust(d2f_score, bull, bear, theta)
        m = evaluate(meta_te, adj_score, f"D4b consistency theta={theta:.1f}",
                       extra={"track": "B", "param": f"theta={theta:.1f}"})
        m["delta_bp_vs_d2f"] = (m["rankicir"] - m_d2f["rankicir"]) * 10000
        risk_rows.append(m)
        # diagnostic on adjustment factor
        adj_p5 = float(np.quantile(adj_factor, 0.05))
        adj_p50 = float(np.quantile(adj_factor, 0.50))
        print(f"    theta={theta:.1f}  RankICIR={m['rankicir']:.4f}  "
              f"Sharpe={m['sharpe']:+.3f}  MaxDD={m['maxdd']*100:+.2f}%  "
              f"Δ_RIC={m['delta_bp_vs_d2f']:+.1f} bp  "
              f"adj p5={adj_p5:.3f} p50={adj_p50:.3f}")

    # ============================================================
    # Direction C — best A + best B combined (if both >= 0.79)
    # ============================================================
    print("\n[4/8] Direction C — combined A+B (if both viable) ...")
    a_rows = [r for r in risk_rows if r.get("track") == "A"]
    b_rows = [r for r in risk_rows if r.get("track") == "B"]
    best_a = max(a_rows, key=lambda r: r["rankicir"]) if a_rows else None
    best_b = max(b_rows, key=lambda r: r["rankicir"]) if b_rows else None
    THRESH = 0.79
    if best_a and best_b and best_a["rankicir"] >= THRESH and best_b["rankicir"] >= THRESH:
        # Extract best params
        a_th = float(best_a["param"].split("=")[-1])
        b_theta = float(best_b["param"].split("=")[-1])
        print(f"   combining best A (sector_th={a_th:.2f}) + best B (theta={b_theta:.1f}) ...")
        adj_a = sector_concentration_adjust(test, d2f_score, SECTOR_COL,
                                              threshold=a_th, frac=FRAC, demote=0.5)
        adj_ab, _, _ = consistency_adjust(adj_a, bull, bear, b_theta)
        m_c = evaluate(meta_te, adj_ab,
                         f"D4ab sector_th={a_th:.2f} + theta={b_theta:.1f}",
                         extra={"track": "A+B",
                                "param": f"sector_th={a_th:.2f} | theta={b_theta:.1f}"})
        m_c["delta_bp_vs_d2f"] = (m_c["rankicir"] - m_d2f["rankicir"]) * 10000
        risk_rows.append(m_c)
        print(f"     RankICIR={m_c['rankicir']:.4f}  Sharpe={m_c['sharpe']:+.3f}  "
              f"MaxDD={m_c['maxdd']*100:+.2f}%  Δ={m_c['delta_bp_vs_d2f']:+.1f} bp")
    else:
        print(f"   skip combined: best A={best_a['rankicir'] if best_a else None}, "
              f"best B={best_b['rankicir'] if best_b else None}  "
              f"(need both >= {THRESH})")

    # ============================================================
    # Write CSV + final summary
    # ============================================================
    print("\n[5/8] write CSV + final summary ...")
    pd.DataFrame(risk_rows).to_csv(
        BB_RESULTS / "step14_risk_manager.csv",
        index=False, encoding="utf-8-sig")

    line = "+" + "-" * 54 + "+" + "-" * 8 + "+" + "-" * 11 + "+" + "-" * 10 + "+" + "-" * 11 + "+" + "-" * 14 + "+"
    print()
    print(line)
    print(f"| {'Config':52s} | {'track':>6s} | {'RankICIR':>9s} | "
          f"{'Sharpe':>8s} | {'MaxDD':>9s} | {'Δ vs D2f (bp)':>12s} |")
    print(line)
    for r in risk_rows:
        track = r.get("track", "-")
        print(f"| {r['config']:52s} | {track:>6s} | {r['rankicir']:>9.4f} | "
              f"{r['sharpe']:>+8.3f} | {r['maxdd']*100:>+8.2f}% | "
              f"{r['delta_bp_vs_d2f']:>+12.1f} |")
    print(line)

    # Verdict
    print("\n=== Verdict ===")
    if a_rows:
        best_a = max(a_rows, key=lambda r: r["rankicir"])
        if best_a["rankicir"] >= 0.79 and best_a["sharpe"] > m_d2f["sharpe"]:
            print(f"  Direction A: best = {best_a['config']}")
            print(f"    RankICIR={best_a['rankicir']:.4f} (>= 0.79)  "
                  f"Sharpe={best_a['sharpe']:+.3f} (vs D2f {m_d2f['sharpe']:+.3f}: improved)")
        elif best_a["rankicir"] >= 0.79:
            print(f"  Direction A: RankICIR ok but Sharpe not improved.")
        else:
            print(f"  Direction A: all thresholds drop RankICIR below 0.79; "
                  f"sector concentration control hurts.")
    if b_rows:
        best_b = max(b_rows, key=lambda r: r["rankicir"])
        if best_b["rankicir"] >= 0.79 and best_b["sharpe"] > m_d2f["sharpe"]:
            print(f"  Direction B: best = {best_b['config']}")
            print(f"    RankICIR={best_b['rankicir']:.4f}  Sharpe={best_b['sharpe']:+.3f}")
        elif best_b["rankicir"] >= 0.79:
            print(f"  Direction B: RankICIR ok but Sharpe not improved.")
        else:
            print(f"  Direction B: all theta drop RankICIR below 0.79.")
    # Overall best
    best_overall = max(risk_rows, key=lambda r: r["rankicir"])
    print(f"\n  Best overall: {best_overall['config']}")
    print(f"    RankICIR={best_overall['rankicir']:.4f}  Sharpe={best_overall['sharpe']:+.3f}  "
          f"MaxDD={best_overall['maxdd']*100:+.2f}%  Δ={best_overall['delta_bp_vs_d2f']:+.1f} bp")
    if (best_overall["rankicir"] < 0.79 or
        best_overall["sharpe"] <= m_d2f["sharpe"]):
        print()
        print("  >>> Risk Manager cannot improve D2f without sacrificing signal quality.")
        print("  >>> Recommendation: Agent 4 should be a *monitoring* role only,")
        print("      not a signal modifier. D2f remains the canonical system.")

    print(f"\nOutputs:")
    print(f"  -> {BB_RESULTS / 'step14_bear_attribution.csv'}")
    print(f"  -> {BB_RESULTS / 'step14_risk_manager.csv'}")


if __name__ == "__main__":
    main()
