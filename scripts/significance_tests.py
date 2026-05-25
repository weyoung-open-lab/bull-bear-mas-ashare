"""
Compute empirical statistical significance for the paper:
  1. t-test on daily IC / RankIC series for the top-7 models in Table 1
  2. Permutation test on SRD(bear, bull) for CatBoost-reg + G1-G4
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t as t_dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import DATE_COL, TARGET_RET_COL
from src.metrics import daily_ic


# -------------------- 1. t-test on IC series --------------------

PRED_DIR = ROOT / "results" / "main_compare_20260506_225947_full_reg" / "predictions"

TOP7 = [
    "CatBoost-reg",
    "LightGBM-shallow-reg",
    "LightGBM-conservative-reg",
    "LightGBM-std-reg",
    "XGBoost-reg",
    "Ridge",
    "RandomForest-reg",
]


def t_stat(series: pd.Series) -> tuple[float, float, int]:
    n = len(series)
    if n < 2:
        return float("nan"), float("nan"), n
    mean = float(series.mean())
    std = float(series.std(ddof=1))
    if std == 0:
        return float("inf"), 0.0, n
    t = mean * np.sqrt(n) / std
    p = 2 * (1 - t_dist.cdf(abs(t), df=n - 1))
    return t, p, n


print("=" * 80)
print("1. t-test on daily IC / RankIC series (top-7 models, full test set)")
print("=" * 80)
rows = []
for model in TOP7:
    pf = PRED_DIR / f"{model}.parquet"
    if not pf.exists():
        # Predictions not pulled locally for this model - skip silently
        continue
    df = pd.read_parquet(pf)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    ric = daily_ic(df[[DATE_COL, TARGET_RET_COL]], df["pred"].to_numpy(), kind="spearman")
    ic = daily_ic(df[[DATE_COL, TARGET_RET_COL]], df["pred"].to_numpy(), kind="pearson")
    t_ric, p_ric, n_ric = t_stat(ric)
    t_ic, p_ic, n_ic = t_stat(ic)
    rows.append({
        "model": model,
        "n_days": n_ric,
        "ic_mean": ic.mean(),
        "ic_t": t_ic, "ic_p": p_ic,
        "rankic_mean": ric.mean(),
        "rankic_t": t_ric, "rankic_p": p_ric,
    })
    print(f"  {model:30s} T={n_ric:4d}  RankIC mean={ric.mean():+.4f} t={t_ric:+.2f} p={p_ric:.2e}  "
          f"IC mean={ic.mean():+.4f} t={t_ic:+.2f} p={p_ic:.2e}")

df_sig = pd.DataFrame(rows)
out_csv = ROOT / "results" / "significance_t_tests.csv"
df_sig.to_csv(out_csv, index=False, encoding="utf-8-sig")
print(f"\nsaved -> {out_csv}")
min_abs_t = float(df_sig[["ic_t", "rankic_t"]].abs().min().min())
max_p = float(df_sig[["ic_p", "rankic_p"]].max().max())
print(f"\n  min |t| across IC and RankIC for top-7 = {min_abs_t:.2f}")
print(f"  max p-value across IC and RankIC for top-7 = {max_p:.2e}")


# -------------------- 2. SRD permutation test --------------------

print("\n" + "=" * 80)
print("2. Permutation test on SRD(bear, bull) — CatBoost-reg + G1-G4")
print("=" * 80)

REGIME_DIR = ROOT / "results" / "regime_20260507_013443_final_g1234_cat" / "submodel_shap"
bear = pd.read_csv(REGIME_DIR / "bear_top_features.csv", index_col=0).iloc[:, 0]
bull = pd.read_csv(REGIME_DIR / "bull_top_features.csv", index_col=0).iloc[:, 0]
common = bear.index.intersection(bull.index)
bear = bear.loc[common]
bull = bull.loc[common]

rank_bear = bear.rank(ascending=False).values
rank_bull = bull.rank(ascending=False).values

corr_obs, _ = spearmanr(rank_bear, rank_bull)
srd_obs = 1 - corr_obs
print(f"  feature universe size = {len(common)}")
print(f"  observed Spearman(bear, bull) = {corr_obs:.4f}")
print(f"  observed SRD(bear, bull)     = {srd_obs:.4f}")

n_perm = 10_000
rng = np.random.default_rng(42)
srd_null = np.empty(n_perm)
for i in range(n_perm):
    shuffled = rng.permutation(rank_bull)
    c, _ = spearmanr(rank_bear, shuffled)
    srd_null[i] = 1 - c

p_emp = float(np.mean(srd_null >= srd_obs))
print(f"  permutations: {n_perm}, seed=42")
print(f"  P(SRD >= {srd_obs:.4f}) under null = {p_emp:.4e}")
print(f"  null mean = {srd_null.mean():.4f}, std = {srd_null.std():.4f}")
print(f"  null max  = {srd_null.max():.4f}")
print(f"  null 95th pct = {np.percentile(srd_null, 95):.4f},  99th = {np.percentile(srd_null, 99):.4f}")

# Save
out2 = ROOT / "results" / "significance_srd_permutation.csv"
pd.DataFrame({
    "metric": ["observed_corr", "observed_srd", "n_perm", "p_emp",
               "null_mean", "null_std", "null_max", "null_p95", "null_p99"],
    "value": [corr_obs, srd_obs, n_perm, p_emp,
              float(srd_null.mean()), float(srd_null.std()),
              float(srd_null.max()),
              float(np.percentile(srd_null, 95)),
              float(np.percentile(srd_null, 99))],
}).to_csv(out2, index=False, encoding="utf-8-sig")
print(f"\nsaved -> {out2}")
