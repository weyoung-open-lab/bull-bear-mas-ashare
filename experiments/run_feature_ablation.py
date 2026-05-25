"""
Table 3：特征消融（预报告 §3）。

逐步叠加特征组：G1 → G1+G2 → G1+G2+G3 → … → G1+G2+G3+G4+G5+G6 (Full)
使用 LightGBM-shallow-reg 作为底座（回归损失，主对比表现最佳）。

输出 results/feature_ablation_<ts>/:
    feature_ablation.csv   论文 Table 3
    figures/feature_ablation_curve.png
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

from config import RESULTS_DIR, TARGET_RET_COL
from src.data import prepare_data
from src.metrics import evaluate
from src.models import build, MODEL_REGISTRY

mpl.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False


# 累加路径（与 prereport_final.docx Table 5 一致）
ABLATION_PATH: list[tuple[str, tuple[str, ...]]] = [
    ("G1",                 ("G1",)),
    ("G1+G2",              ("G1", "G2")),
    ("G1+G2+G3",           ("G1", "G2", "G3")),
    ("G1+G2+G3+G4",        ("G1", "G2", "G3", "G4")),
    ("G1+G2+G3+G4+G5",     ("G1", "G2", "G3", "G4", "G5")),
    ("Full(G1-G6)",        ("G1", "G2", "G3", "G4", "G5", "G6")),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="LightGBM-shallow-reg")
    parser.add_argument("--sample-tickers", type=int, default=None)
    parser.add_argument("--tag", default="full")
    args = parser.parse_args()

    if args.base_model not in MODEL_REGISTRY:
        raise SystemExit(f"unknown base-model: {args.base_model}")
    base_cls = MODEL_REGISTRY[args.base_model]
    is_regression = getattr(base_cls, "regression_target", False)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / f"feature_ablation_{ts}_{args.tag}"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Feature Ablation ({args.base_model}) ===")
    print(f"output -> {out_dir}\n")

    rows: list[dict] = []
    for cfg_name, groups in ABLATION_PATH:
        print(f"\n>>> {cfg_name}  (groups={list(groups)})", flush=True)
        bundle = prepare_data(
            feature_groups=groups,
            preprocess=base_cls.preprocess,
            sample_tickers=args.sample_tickers,
        )
        n_feats = bundle.X_train.shape[1]
        print(f"  features={n_feats}  X_train={bundle.X_train.shape}", flush=True)

        if is_regression:
            yr = bundle.meta_train[TARGET_RET_COL].to_numpy(dtype="float32")
            lo, hi = float(np.quantile(yr, 0.001)), float(np.quantile(yr, 0.999))
            y_train = np.clip(yr, lo, hi).astype("float32")
        else:
            y_train = bundle.y_train

        model = build(args.base_model)
        t0 = time.time()
        model.fit(bundle.X_train, y_train)
        pred = model.predict_proba(bundle.X_test)
        runtime = time.time() - t0

        metric = evaluate(bundle.y_test, pred, bundle.meta_test)
        row = {"config": cfg_name, "groups": "+".join(groups), "n_features": n_feats,
               "fit_predict_sec": round(runtime, 2)}
        row.update(metric.to_row())
        rows.append(row)
        print(f"  AUC={row['auc']:.4f}  RankIC={row['rankic_mean']:.4f}  "
              f"RankICIR={row['rankicir']:.3f}  Top-1%={row['top1pct_ret']:.4f}",
              flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "feature_ablation.csv", index=False, encoding="utf-8-sig")
    print(f"\nsaved {out_dir/'feature_ablation.csv'}")

    # ---- 图：消融曲线 ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(len(df))
    axes[0].plot(x, df["rankic_mean"].values, marker="o", color="#1f77b4", lw=1.6,
                 label="RankIC mean")
    axes[0].plot(x, df["rankicir"].values, marker="s", color="#d62728", lw=1.6,
                 label="RankICIR")
    axes[0].set_xticks(x); axes[0].set_xticklabels(df["config"].values, rotation=20, ha="right")
    axes[0].set_title("Ranking metrics vs feature group additions")
    axes[0].axhline(0, color="k", lw=0.6)
    axes[0].grid(alpha=0.3); axes[0].legend()

    axes[1].plot(x, df["top1pct_ret"].values * 100, marker="o", color="#2ca02c", lw=1.6,
                 label="Top-1% 5d ret (%)")
    axes[1].plot(x, df["top5pct_ret"].values * 100, marker="s", color="#9467bd", lw=1.6,
                 label="Top-5% 5d ret (%)")
    axes[1].set_xticks(x); axes[1].set_xticklabels(df["config"].values, rotation=20, ha="right")
    axes[1].set_title("Top-K returns vs feature group additions")
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].grid(alpha=0.3); axes[1].legend()

    fig.suptitle(f"Feature Ablation ({args.base_model})", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "feature_ablation_curve.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {fig_dir/'feature_ablation_curve.png'}")

    (out_dir / "config.json").write_text(json.dumps({
        "timestamp": ts,
        "base_model": args.base_model,
        "ablation_path": [{"config": c, "groups": list(g)} for c, g in ABLATION_PATH],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[done] {out_dir}")


if __name__ == "__main__":
    main()
