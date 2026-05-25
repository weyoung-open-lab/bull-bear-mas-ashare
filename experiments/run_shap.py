"""
论文 §7：四层递进 SHAP 分析（L1–L4）。

L1 全局：测试集抽样 N，TreeSHAP → 全特征重要性 + beeswarm（Figure 2）
L2 年际动态：按年切割，每年独立计算 mean(|SHAP|)，绘制 rank stability 曲线（Figure 3）
L3 条件分析：仅 Top-1% 高分样本，对比与 L1 的差异（Figure 4）
L4 Regime（★）：复用 run_regime_analysis 已经产出的 SRD 矩阵 / 三子模型 SHAP（Figure 5/6）

L1+L3 还会绘制 SHAP 方向性分解图（Figure 6）：每特征拆 SHAP>0 均值与 |SHAP<0| 均值。

用法：
    python -m experiments.run_shap \
        --base-model LightGBM-std \
        --train-on-full \
        --shap-sample 20000 \
        --tag full
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
import shap

from config import DATE_COL, RESULTS_DIR, TARGET_RET_COL, TICKER_COL
from src.data import prepare_data
from src.features import DEFAULT_GROUPS
from src.models import build

mpl.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False


def _shap_values_binary(expl: shap.TreeExplainer, X: pd.DataFrame) -> np.ndarray:
    sv = expl.shap_values(X)
    if isinstance(sv, list):                    # 老 API：[class0, class1]
        sv = sv[1]
    if hasattr(sv, "values"):
        sv = sv.values
    return np.asarray(sv)


def _bar(ax, names, vals, color, title=None):
    order = np.argsort(vals)
    ax.barh(np.array(names)[order], np.array(vals)[order], color=color,
            edgecolor="black", lw=0.4)
    if title:
        ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="LightGBM-std",
                        help="底座模型名（必须是 GBDT，TreeSHAP 兼容）")
    parser.add_argument("--shap-sample", type=int, default=20000)
    parser.add_argument("--top1pct-cap", type=int, default=20000,
                        help="L3 条件分析的 Top-1% 抽样上限")
    parser.add_argument("--sample-tickers", type=int, default=None)
    parser.add_argument("--tag", type=str, default="full")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / f"shap_{ts}_{args.base_model.replace('-','_')}_{args.tag}"
    fig_dir = out_dir / "figures"
    csv_dir = out_dir / "csv"
    fig_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== SHAP L1–L4 (base={args.base_model}) ===")
    print(f"output -> {out_dir}\n")

    # ----------- 1. 数据 + 训练底座 -----------
    print("[1/5] preparing data + training base model ...")
    bundle = prepare_data(
        feature_groups=DEFAULT_GROUPS,
        preprocess="raw",
        sample_tickers=args.sample_tickers,
    )
    feat_cols = bundle.feature_cols
    print(f"  features ({len(feat_cols)})  X_train={bundle.X_train.shape}  X_test={bundle.X_test.shape}")

    base_cls_name = args.base_model
    model = build(base_cls_name)
    t0 = time.time()
    model.fit(bundle.X_train, bundle.y_train)
    print(f"  fit {base_cls_name}: {time.time()-t0:.1f}s")

    raw = model.raw_model
    expl = shap.TreeExplainer(raw)

    # 抽样测试集
    rng = np.random.default_rng(42)
    n_test = len(bundle.X_test)
    test_idx = rng.choice(n_test, size=min(args.shap_sample, n_test), replace=False)
    X_sample = bundle.X_test.iloc[test_idx].copy()
    meta_sample = bundle.meta_test.iloc[test_idx].reset_index(drop=True).copy()

    # 同时用模型预测，用于 L3 选 Top-1%
    pred_all = model.predict_proba(bundle.X_test)
    pred_sample = pred_all[test_idx]

    # ----------- 2. L1 全局 SHAP -----------
    print("\n[2/5] L1 global SHAP ...")
    sv = _shap_values_binary(expl, X_sample)
    importance = np.abs(sv).mean(axis=0)
    pos_mean = np.where(sv > 0, sv, 0).mean(axis=0)
    neg_mean = -np.where(sv < 0, sv, 0).mean(axis=0)
    l1 = pd.DataFrame({
        "feature": feat_cols,
        "mean_abs_shap": importance,
        "pos_mean_shap": pos_mean,
        "neg_mean_shap": neg_mean,
    }).sort_values("mean_abs_shap", ascending=False)
    l1.to_csv(csv_dir / "L1_global.csv", index=False, encoding="utf-8-sig")

    # Figure 2: 横向条形图 + beeswarm
    fig, ax = plt.subplots(figsize=(8, 9))
    _bar(ax, l1["feature"].tolist()[:25], l1["mean_abs_shap"].values[:25],
         color="#4c72b0", title="L1 Global SHAP feature importance (Top 25)")
    ax.set_xlabel("mean(|SHAP|)")
    fig.tight_layout(); fig.savefig(fig_dir / "fig2_L1_global_bar.png", dpi=160); plt.close(fig)

    # beeswarm（用 shap 自带）
    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, X_sample, plot_type="dot", show=False, max_display=20)
    plt.title("L1 Global SHAP beeswarm")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig2_L1_beeswarm.png", dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  saved L1 figures + L1_global.csv")

    # Figure 6: SHAP 方向性分解（论文 §7.1）
    top20 = l1.head(20).copy()
    fig, ax = plt.subplots(figsize=(10, 8))
    y = np.arange(len(top20))[::-1]
    ax.barh(y, top20["pos_mean_shap"].values, color="#2ca02c",
            label="Positive push (mean SHAP > 0)", edgecolor="black", lw=0.4)
    ax.barh(y, -top20["neg_mean_shap"].values, color="#d62728",
            label="Negative pull (|SHAP < 0|)", edgecolor="black", lw=0.4)
    ax.set_yticks(y, top20["feature"])
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("Directional mean SHAP (positive vs negative split)")
    ax.set_title("Directional SHAP decomposition (Top 20)")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / "fig6_directional_top20.png", dpi=160); plt.close(fig)
    print(f"  saved fig6_directional_top20.png")

    # ----------- 3. L2 年际动态 -----------
    print("\n[3/5] L2 yearly stability ...")
    test_meta_full = bundle.meta_test.copy()
    test_meta_full["year"] = pd.to_datetime(test_meta_full[DATE_COL]).dt.year
    years = sorted(test_meta_full["year"].unique().tolist())
    yearly_imp: dict[int, pd.Series] = {}
    cap = max(2000, args.shap_sample // max(len(years), 1))
    for y_ in years:
        idx = test_meta_full.index[test_meta_full["year"] == y_].to_numpy()
        if len(idx) > cap:
            idx = rng.choice(idx, size=cap, replace=False)
        Xy = bundle.X_test.iloc[idx]
        if len(Xy) < 50:
            continue
        sv_y = _shap_values_binary(expl, Xy)
        imp = np.abs(sv_y).mean(axis=0)
        yearly_imp[int(y_)] = pd.Series(imp, index=feat_cols)
        print(f"  year={y_}: n={len(Xy):,}  top3={list(yearly_imp[y_].sort_values(ascending=False).head(3).index)}")

    yearly_df = pd.DataFrame(yearly_imp)
    yearly_df.to_csv(csv_dir / "L2_yearly_importance.csv", encoding="utf-8-sig")

    # 跨年 rank-corr
    rank_df = yearly_df.rank(ascending=False)
    yrs = list(yearly_df.columns)
    rank_corr = np.zeros((len(yrs), len(yrs)))
    for i, a in enumerate(yrs):
        for j, b in enumerate(yrs):
            rank_corr[i, j] = rank_df[a].corr(rank_df[b], method="spearman")
    rank_corr_df = pd.DataFrame(rank_corr, index=yrs, columns=yrs)
    rank_corr_df.to_csv(csv_dir / "L2_rank_correlation.csv", encoding="utf-8-sig")

    # Figure 3：年际 rank-corr 热力图 + Top-N 特征逐年重要性折线
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    import seaborn as sns
    sns.heatmap(rank_corr_df, annot=True, fmt=".2f", cmap="Greens",
                vmin=0.5, vmax=1.0, cbar_kws={"label": "Spearman rank-corr"},
                ax=axes[0], square=True, linewidths=0.5)
    axes[0].set_title("Year-to-year SHAP rank correlation")

    top_feats = l1.head(8)["feature"].tolist()
    yearly_top = yearly_df.loc[top_feats].T
    yearly_top.plot(ax=axes[1], marker="o")
    axes[1].set_title("Top-8 features: yearly mean(|SHAP|)")
    axes[1].set_xlabel("year"); axes[1].set_ylabel("mean(|SHAP|)")
    axes[1].legend(loc="best", fontsize=8, ncol=2)
    axes[1].grid(alpha=0.3)
    fig.suptitle("L2 Year-over-year SHAP stability", y=1.02)
    fig.tight_layout(); fig.savefig(fig_dir / "fig3_L2_yearly_stability.png", dpi=160); plt.close(fig)
    print(f"  saved fig3 + L2 csv")

    # ----------- 4. L3 Top-1% 条件分析 -----------
    print("\n[4/5] L3 Top-1% conditional ...")
    # 每天选 Top-1%
    per_day = test_meta_full.copy()
    per_day["pred"] = pred_all
    per_day["__row"] = np.arange(len(per_day))
    top1_idx_list: list[int] = []
    for d, g in per_day.groupby(DATE_COL, sort=False):
        n = len(g); k = max(1, int(np.ceil(n * 0.01)))
        order = g["pred"].to_numpy().argsort()[::-1][:k]
        top1_idx_list.extend(g.iloc[order]["__row"].tolist())
    top1_idx = np.asarray(top1_idx_list, dtype=int)
    print(f"  total Top-1% rows: {len(top1_idx):,}")
    if len(top1_idx) > args.top1pct_cap:
        top1_idx = rng.choice(top1_idx, size=args.top1pct_cap, replace=False)

    X_top1 = bundle.X_test.iloc[top1_idx]
    sv_top1 = _shap_values_binary(expl, X_top1)
    imp_top1 = np.abs(sv_top1).mean(axis=0)
    l3 = pd.DataFrame({
        "feature": feat_cols,
        "global_importance": importance,
        "top1pct_importance": imp_top1,
    })
    l3["delta"] = l3["top1pct_importance"] - l3["global_importance"]
    l3.sort_values("global_importance", ascending=False).to_csv(
        csv_dir / "L3_top1pct_vs_global.csv", index=False, encoding="utf-8-sig"
    )

    # Figure 4：Full vs Top-1% 双柱状对比 Top-20
    top20_global = l1.head(20)["feature"].tolist()
    sub = l3.set_index("feature").loc[top20_global]
    fig, ax = plt.subplots(figsize=(11, 7))
    y = np.arange(len(sub))[::-1]
    ax.barh(y - 0.2, sub["global_importance"].values, height=0.4, color="#7f7f7f",
            label="Global", edgecolor="black", lw=0.4)
    ax.barh(y + 0.2, sub["top1pct_importance"].values, height=0.4, color="#d62728",
            label="Top-1% selected", edgecolor="black", lw=0.4)
    ax.set_yticks(y, sub.index)
    ax.set_xlabel("mean(|SHAP|)")
    ax.set_title("L3 Top-1% conditional SHAP vs Global (Top 20)")
    ax.legend(); ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / "fig4_L3_top1pct_vs_global.png", dpi=160); plt.close(fig)
    print(f"  saved fig4 + L3 csv")

    # ----------- 5. 配置 + 汇总 -----------
    (out_dir / "config.json").write_text(json.dumps({
        "timestamp": ts,
        "base_model": args.base_model,
        "shap_sample": args.shap_sample,
        "top1pct_cap": args.top1pct_cap,
        "n_features": len(feat_cols),
        "n_test": len(bundle.X_test),
        "years": [int(y) for y in years],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[done] all outputs at: {out_dir}")
    print("  L1 -> figures/fig2_L1_global_bar.png + fig2_L1_beeswarm.png")
    print("  L2 -> figures/fig3_L2_yearly_stability.png")
    print("  L3 -> figures/fig4_L3_top1pct_vs_global.png")
    print("  L1+L3 -> figures/fig6_directional_top20.png")
    print("  L4 -> 见 results/regime_*/figures/regime_srd_heatmap.png（已有）")


if __name__ == "__main__":
    main()
