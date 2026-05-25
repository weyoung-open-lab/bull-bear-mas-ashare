"""
Bootstrap CI for SRD(bear, bull) under CatBoost-reg + G1+G2+G3+G4.

Fast pipeline:
  1. Train three CatBoost-reg sub-models on bear/sideway/bull training subsets.
  2. For each regime, compute TreeSHAP values on a fixed test sample (N=20,000).
  3. Bootstrap-resample SHAP rows (with replacement) B=1000 times, each time:
        recompute mean(|SHAP|) per feature, rerank, compute SRD between regimes.
  4. Report observed SRD, 95% bootstrap CI, and test SRD > 0.

Total runtime: ~10-15 min on local CPU (no GPU needed for TreeSHAP).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import LABEL_COL, REGIME_COL, TARGET_RET_COL
from src.data import build_label, load_dataset, split_train_test, Preprocessor
from src.features import get_feature_columns
from src.models.gbdt import CatBoostStdReg

# -------- Config --------
FEATURE_GROUPS = ("G1", "G2", "G3", "G4")
SHAP_SAMPLE_PER_REGIME = 20_000
N_BOOTSTRAP = 1000
SEED = 42

OUT_DIR = ROOT / "results" / "srd_bootstrap"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# -------- 1. Load + split data --------
print("[1/5] loading data ...", flush=True)
t0 = time.time()
df = load_dataset()
df = build_label(df)
df = df.dropna(subset=[TARGET_RET_COL]).reset_index(drop=True)
train, test = split_train_test(df)
feat_cols = get_feature_columns(FEATURE_GROUPS)  # G1234 = 21 features, no macro_regime_3
print(f"   features ({len(feat_cols)}): {feat_cols[:6]} +{len(feat_cols)-6}")
print(f"   train: {len(train):,}  test: {len(test):,}  (loaded in {time.time()-t0:.1f}s)")

pre = Preprocessor(mode="raw", feature_cols=feat_cols)
X_train = pre.fit_transform(train)
X_test = pre.transform(test)

# Regression target with quantile clipping (matches paper)
yr = train[TARGET_RET_COL].to_numpy(dtype="float32")
lo, hi = float(np.quantile(yr, 0.001)), float(np.quantile(yr, 0.999))
y_train_reg = np.clip(yr, lo, hi).astype("float32")
print(f"   regression target clipped to [{lo:.4f}, {hi:.4f}]")

regime_train = train[REGIME_COL].astype(str).reset_index(drop=True)
regime_test = test[REGIME_COL].astype(str).reset_index(drop=True)
print("   train regime dist:", regime_train.value_counts().to_dict())
print("   test  regime dist:", regime_test.value_counts().to_dict())


# -------- 2. Train sub-models --------
print(f"\n[2/5] training CatBoost-reg × 3 sub-models ...", flush=True)
submodels: dict[str, object] = {}
for r in ("bear", "sideway", "bull"):
    mask = (regime_train == r).values
    n = int(mask.sum())
    if n == 0:
        continue
    t0 = time.time()
    m = CatBoostStdReg()
    m.fit(X_train.iloc[mask], y_train_reg[mask])
    submodels[r] = m.raw_model
    print(f"   regime={r:7s} trained on {n:,} samples in {time.time()-t0:.1f}s", flush=True)


# -------- 3. Compute SHAP per regime on fixed test sample --------
print(f"\n[3/5] computing TreeSHAP on {SHAP_SAMPLE_PER_REGIME:,} test samples per regime ...", flush=True)
rng = np.random.default_rng(SEED)
shap_per_regime: dict[str, np.ndarray] = {}
for r in submodels:
    mask = (regime_test == r).values
    idx_full = np.where(mask)[0]
    if len(idx_full) == 0:
        continue
    n_sample = min(SHAP_SAMPLE_PER_REGIME, len(idx_full))
    sample_idx = rng.choice(idx_full, size=n_sample, replace=False)
    Xs = X_test.iloc[sample_idx]

    t0 = time.time()
    expl = shap.TreeExplainer(submodels[r])
    sv = expl.shap_values(Xs)
    if isinstance(sv, list):
        sv = sv[1]
    if hasattr(sv, "values"):
        sv = sv.values
    shap_per_regime[r] = np.asarray(np.abs(sv), dtype="float32")  # |SHAP|, shape [N, F]
    print(f"   regime={r:7s} SHAP {shap_per_regime[r].shape} in {time.time()-t0:.1f}s", flush=True)


# -------- 4. Bootstrap SRD distribution --------
print(f"\n[4/5] bootstrap SRD x {N_BOOTSTRAP} ...", flush=True)
t0 = time.time()


def srd_from_abs_shap(abs_shap_a: np.ndarray, abs_shap_b: np.ndarray) -> float:
    imp_a = abs_shap_a.mean(axis=0)
    imp_b = abs_shap_b.mean(axis=0)
    rank_a = pd.Series(imp_a).rank(ascending=False).values
    rank_b = pd.Series(imp_b).rank(ascending=False).values
    rho, _ = spearmanr(rank_a, rank_b)
    return 1.0 - float(rho)


# Observed SRD
srd_observed = {}
pairs = [("bear", "bull"), ("bear", "sideway"), ("bull", "sideway")]
for a, b in pairs:
    if a in shap_per_regime and b in shap_per_regime:
        srd_observed[(a, b)] = srd_from_abs_shap(shap_per_regime[a], shap_per_regime[b])
        print(f"   observed SRD({a:7s}, {b:7s}) = {srd_observed[(a, b)]:.4f}")

# Bootstrap loop
boot_rng = np.random.default_rng(SEED + 1)
srd_boot = {p: np.empty(N_BOOTSTRAP) for p in pairs if p in srd_observed}
for k in range(N_BOOTSTRAP):
    sampled = {}
    for r, mat in shap_per_regime.items():
        n = mat.shape[0]
        idx = boot_rng.integers(0, n, size=n)
        sampled[r] = mat[idx]
    for (a, b) in srd_boot:
        srd_boot[(a, b)][k] = srd_from_abs_shap(sampled[a], sampled[b])
    if (k + 1) % 100 == 0:
        print(f"   bootstrap {k+1}/{N_BOOTSTRAP}  (elapsed {time.time()-t0:.0f}s)", flush=True)


# -------- 5. Report and save --------
print(f"\n[5/5] results", flush=True)
rows = []
for (a, b), boot in srd_boot.items():
    obs = srd_observed[(a, b)]
    ci_lo = float(np.percentile(boot, 2.5))
    ci_hi = float(np.percentile(boot, 97.5))
    boot_mean = float(boot.mean())
    boot_std = float(boot.std(ddof=1))
    p_zero = float(np.mean(boot <= 0))   # share of bootstraps where SRD <= 0
    rows.append({
        "pair": f"{a} ↔ {b}",
        "observed": obs,
        "boot_mean": boot_mean,
        "boot_std": boot_std,
        "ci_low_2_5pct": ci_lo,
        "ci_high_97_5pct": ci_hi,
        "p_srd_le_0": p_zero,
        "B": N_BOOTSTRAP,
    })
    print(f"  {a:7s} ↔ {b:7s}  obs={obs:.4f}  "
          f"95% CI [{ci_lo:.4f}, {ci_hi:.4f}]  P(SRD<=0)={p_zero:.4e}")

df_boot = pd.DataFrame(rows)
df_boot.to_csv(OUT_DIR / "srd_bootstrap_summary.csv", index=False, encoding="utf-8-sig")
# Save raw bootstrap arrays
for (a, b), boot in srd_boot.items():
    np.save(OUT_DIR / f"srd_boot_{a}_{b}.npy", boot)
print(f"\nsaved -> {OUT_DIR/'srd_bootstrap_summary.csv'}")
