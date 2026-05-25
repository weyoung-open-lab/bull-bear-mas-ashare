"""
回归目标快速验证：把目标从「label = (r_future_5 > 1%) 二分类」换成
「直接回归 r_future_5」，看 RankIC / RankICIR 是否提升（论文 §7 讨论）。

只对预报告 §10 表现最好的两个模型做：
  - LogReg (二分类) → Ridge / LinearRegression（回归）
  - LightGBM-shallow（二分类）→ LightGBM-shallow（回归 objective）

输出（results/regression_<ts>/）:
    metrics_compare.csv      二分类 vs 回归各模型的 AUC/IC/RankIC/RankICIR/Top-K
    figures/comparison_bar.png
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge

from config import RANDOM_SEED, RESULTS_DIR, TARGET_RET_COL
from src.data import prepare_data
from src.features import DEFAULT_GROUPS
from src.metrics import evaluate
from src.models import build

mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
mpl.rcParams["axes.unicode_minus"] = False


# 把回归预测套到 evaluate（label 仍然是二分类，AUC 也能算）
def _evaluate(y_label, pred, meta, model_name, loss):
    m = evaluate(y_label, pred, meta)
    row = m.to_row()
    row["model"] = model_name
    row["loss"] = loss
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-tickers", type=int, default=None)
    parser.add_argument("--tag", type=str, default="full")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / f"regression_{ts}_{args.tag}"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=== 回归损失对比 ===")
    print(f"output -> {out_dir}\n")

    # 二分类 LogReg / LGBM-shallow 用 raw / zscore 预处理；这里也做 zscore 与 raw 两份
    print("[1/3] preparing data ...")
    bundle_raw = prepare_data(feature_groups=DEFAULT_GROUPS, preprocess="raw",
                              sample_tickers=args.sample_tickers)
    bundle_zs = prepare_data(feature_groups=DEFAULT_GROUPS, preprocess="zscore",
                             sample_tickers=args.sample_tickers)
    print(f"  X_train={bundle_raw.X_train.shape}  X_test={bundle_raw.X_test.shape}")

    # 回归目标：r_future_5（即 meta_train[TARGET_RET_COL]）
    y_train_reg = bundle_raw.meta_train[TARGET_RET_COL].to_numpy(dtype="float32")
    # 异常值裁剪（防止个别极端值主导 MSE）
    lo = float(np.quantile(y_train_reg, 0.001))
    hi = float(np.quantile(y_train_reg, 0.999))
    y_train_reg = np.clip(y_train_reg, lo, hi)

    rows: list[dict] = []

    # ---------- 1. LogReg (binary) → Ridge (regression) ----------
    print("\n[2/3] LogReg (binary) vs Ridge (regression) ...")
    # baseline binary
    t = time.time()
    lr = build("LogisticRegression")
    lr.fit(bundle_zs.X_train, bundle_zs.y_train)
    pred_lr = lr.predict_proba(bundle_zs.X_test)
    rows.append(_evaluate(bundle_zs.y_test, pred_lr, bundle_zs.meta_test,
                          "LogReg", "binary_BCE"))
    print(f"  LogReg binary fit+predict: {time.time()-t:.1f}s  RankICIR={rows[-1]['rankicir']:.3f}")

    # regression
    t = time.time()
    rd = Ridge(alpha=1.0, random_state=RANDOM_SEED)
    rd.fit(bundle_zs.X_train.to_numpy(dtype="float32"), y_train_reg)
    pred_rd = rd.predict(bundle_zs.X_test.to_numpy(dtype="float32"))
    rows.append(_evaluate(bundle_zs.y_test, pred_rd, bundle_zs.meta_test,
                          "Ridge", "regression_MSE"))
    print(f"  Ridge regression fit+predict: {time.time()-t:.1f}s  RankICIR={rows[-1]['rankicir']:.3f}")

    # ---------- 2. LGBM-shallow (binary) → LGBM-shallow (regression) ----------
    print("\n[3/3] LGBM-shallow (binary) vs LGBM-shallow (regression) ...")
    t = time.time()
    lg = build("LightGBM-shallow")
    lg.fit(bundle_raw.X_train, bundle_raw.y_train)
    pred_lg = lg.predict_proba(bundle_raw.X_test)
    rows.append(_evaluate(bundle_raw.y_test, pred_lg, bundle_raw.meta_test,
                          "LGBM-shallow", "binary_BCE"))
    print(f"  LGBM-shallow binary fit+predict: {time.time()-t:.1f}s  RankICIR={rows[-1]['rankicir']:.3f}")

    t = time.time()
    lg_reg = LGBMRegressor(
        n_estimators=500, learning_rate=0.05, num_leaves=15, max_depth=4,
        min_child_samples=50, n_jobs=-1, random_state=RANDOM_SEED, verbosity=-1,
        objective="regression",
    )
    lg_reg.fit(bundle_raw.X_train, y_train_reg)
    pred_lg_reg = lg_reg.predict(bundle_raw.X_test)
    rows.append(_evaluate(bundle_raw.y_test, pred_lg_reg, bundle_raw.meta_test,
                          "LGBM-shallow", "regression_MSE"))
    print(f"  LGBM-shallow regression fit+predict: {time.time()-t:.1f}s  RankICIR={rows[-1]['rankicir']:.3f}")

    df = pd.DataFrame(rows)
    cols = ["model", "loss", "auc", "ic_mean", "icir", "rankic_mean", "rankicir",
            "top1pct_ret", "top5pct_ret"]
    df_show = df[cols]
    df_show.to_csv(out_dir / "metrics_compare.csv", index=False, encoding="utf-8-sig")
    print("\n=== 结果 ===")
    print(df_show.to_string(index=False))

    # Bar plot：每个模型 binary vs regression 在 RankICIR / RankIC / AUC / Top-5% Ret
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    metrics_to_plot = [
        ("rankicir", "RankICIR (论文主指标)"),
        ("rankic_mean", "RankIC mean"),
        ("auc", "AUC"),
        ("top5pct_ret", "Top-5% 平均 5d 收益"),
    ]
    for ax, (col, title) in zip(axes.flat, metrics_to_plot):
        piv = df.pivot(index="model", columns="loss", values=col)
        piv = piv.reindex(["LogReg", "LGBM-shallow"])
        piv = piv[["binary_BCE", "regression_MSE"]]
        piv.plot.bar(ax=ax, rot=0, edgecolor="black", linewidth=0.5,
                     color=["#7f7f7f", "#d62728"])
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("二分类 BCE vs 回归 MSE 损失对比", y=1.01)
    fig.tight_layout()
    fig.savefig(fig_dir / "comparison_bar.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {fig_dir/'comparison_bar.png'}")

    (out_dir / "config.json").write_text(json.dumps({
        "timestamp": ts,
        "models": ["LogReg", "Ridge", "LGBM-shallow", "LGBM-shallow_reg"],
        "y_train_clip_quantile": [0.001, 0.999],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[done] {out_dir}")


if __name__ == "__main__":
    main()
