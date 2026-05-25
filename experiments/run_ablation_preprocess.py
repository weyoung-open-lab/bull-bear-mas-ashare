"""
Table 4：预处理消融（预报告 §6.2）。

每个代表性模型 × 三种预处理 (raw / zscore / standard)。
代表性模型：
    - Ridge                  （linear）
    - LightGBM-shallow-reg   （GBDT）
    - FT-Transformer-reg     （tabular DL）

输出 results/preprocess_ablation_<ts>/:
    preprocess_ablation.csv
    figures/preprocess_ablation_bar.png
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


MODELS = ["Ridge", "LightGBM-shallow-reg", "FT-Transformer-reg"]
PREPROCESS_MODES = ["raw", "zscore", "standard"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-tickers", type=int, default=None)
    parser.add_argument("--tag", default="full")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / f"preprocess_ablation_{ts}_{args.tag}"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Preprocess Ablation ===\noutput -> {out_dir}\n")

    rows: list[dict] = []
    cache: dict[str, "DataBundle"] = {}  # type: ignore  noqa
    from src.features import DEFAULT_GROUPS
    for mode in PREPROCESS_MODES:
        if mode not in cache:
            print(f"\n=== prepare_data(preprocess={mode}) ===", flush=True)
            cache[mode] = prepare_data(
                feature_groups=DEFAULT_GROUPS,
                preprocess=mode,
                sample_tickers=args.sample_tickers,
            )
            b = cache[mode]
            print(f"   X_train={b.X_train.shape}  X_test={b.X_test.shape}", flush=True)

    for model_name in MODELS:
        if model_name not in MODEL_REGISTRY:
            print(f"  [skip unknown] {model_name}")
            continue
        cls = MODEL_REGISTRY[model_name]
        is_reg = getattr(cls, "regression_target", False)

        for mode in PREPROCESS_MODES:
            print(f"\n>>> {model_name}  preprocess={mode}", flush=True)
            bundle = cache[mode]
            if is_reg:
                yr = bundle.meta_train[TARGET_RET_COL].to_numpy(dtype="float32")
                lo, hi = float(np.quantile(yr, 0.001)), float(np.quantile(yr, 0.999))
                y_train = np.clip(yr, lo, hi).astype("float32")
            else:
                y_train = bundle.y_train

            model = build(model_name)
            t0 = time.time()
            try:
                model.fit(bundle.X_train, y_train)
                pred = model.predict_proba(bundle.X_test)
            except Exception as e:
                print(f"   [FAIL] {type(e).__name__}: {e}")
                continue
            runtime = time.time() - t0
            metric = evaluate(bundle.y_test, pred, bundle.meta_test)
            row = {"model": model_name, "preprocess": mode,
                   "fit_predict_sec": round(runtime, 2)}
            row.update(metric.to_row())
            rows.append(row)
            print(f"  AUC={row['auc']:.4f}  RankIC={row['rankic_mean']:.4f}  "
                  f"RankICIR={row['rankicir']:.3f}  Top-1%={row['top1pct_ret']:.4f}",
                  flush=True)

    df = pd.DataFrame(rows)
    cols = ["model", "preprocess", "auc", "ic_mean", "icir", "rankic_mean", "rankicir",
            "top1pct_ret", "top5pct_ret", "fit_predict_sec"]
    df[cols].to_csv(out_dir / "preprocess_ablation.csv", index=False, encoding="utf-8-sig")
    print(f"\nsaved {out_dir/'preprocess_ablation.csv'}")

    # ---- 图：模型 × 预处理 RankICIR / Sharpe 比较 ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    metrics_to_plot = [("rankicir", "RankICIR"), ("rankic_mean", "RankIC mean")]
    for ax, (col, title) in zip(axes, metrics_to_plot):
        piv = df.pivot(index="model", columns="preprocess", values=col)
        piv = piv.reindex(MODELS)[["raw", "zscore", "standard"]]
        piv.plot.bar(ax=ax, rot=15, edgecolor="black", linewidth=0.5,
                     color=["#7f7f7f", "#1f77b4", "#d62728"])
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(title)
        ax.set_xlabel("Model")
        ax.legend(title="preprocess", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Preprocessing ablation: raw vs zscore vs standard", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "preprocess_ablation_bar.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {fig_dir/'preprocess_ablation_bar.png'}")

    (out_dir / "config.json").write_text(json.dumps({
        "timestamp": ts,
        "models": MODELS,
        "preprocess_modes": PREPROCESS_MODES,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] {out_dir}")


if __name__ == "__main__":
    main()
