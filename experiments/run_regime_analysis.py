"""
论文 §5 / §6 Table 5 / Table 6 / Figure 5：Regime-Conditioned SHAP Ensemble。

输出（results/regime_<ts>/）:
    regime_eval.csv          - Table 5：global vs routed 在每个 regime 上的指标
    srd_matrix.csv           - Table 6：3×3 SHAP Regime Divergence 矩阵
    figures/regime_srd_heatmap.png  - Figure 5
    figures/regime_metrics_bar.png  - 各 regime 下 IC/RankIC 增量
    submodel_shap/{regime}_top_features.csv
    config.json

核心步骤：
    1. 加载全量数据，标签化，特征 = G1..G6 - macro_regime_3（27 列）
    2. 训练 3 个 regime 子模型 + 1 个 global 对照模型（同特征）
    3. 路由预测 vs global 预测，per-regime 计算 IC/RankIC/Top-K
    4. 对每个 sub-model 用 TreeSHAP 抽 N 条计算特征重要性排名
    5. 计算 SRD(i,j) = 1 - Spearman(rank_i, rank_j)
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
import seaborn as sns
import shap
from scipy.stats import spearmanr

from config import (
    DATE_COL,
    LABEL_COL,
    REGIME_COL,
    RESULTS_DIR,
    TARGET_RET_COL,
    TICKER_COL,
)
from src.backtest import backtest_topk
from src.data import Preprocessor, build_label, load_dataset, split_train_test
from src.features import DEFAULT_GROUPS, get_feature_columns
from src.metrics import evaluate
from src.models import MODEL_REGISTRY, build
from src.regime_ensemble import REGIMES, RegimeEnsemble

mpl.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------- helpers

def _features_without_regime(groups: tuple[str, ...] = DEFAULT_GROUPS) -> list[str]:
    cols = get_feature_columns(groups)
    return [c for c in cols if c != REGIME_COL]


def _per_regime_metrics(meta: pd.DataFrame, pred: np.ndarray, regime: pd.Series,
                         tag: str) -> list[dict]:
    rows = []
    # 全集
    m = evaluate(np.asarray((meta[TARGET_RET_COL] > 0.01).astype(int)), pred, meta)
    rows.append({"split": "ALL", "model": tag, **m.to_row()})
    # 各 regime 子集
    for r in REGIMES:
        mask = (regime == r).values
        if not mask.any():
            continue
        sub_meta = meta.iloc[mask].reset_index(drop=True)
        sub_pred = pred[mask]
        y = (sub_meta[TARGET_RET_COL] > 0.01).astype(int).to_numpy()
        m = evaluate(y, sub_pred, sub_meta)
        rows.append({"split": r, "model": tag, **m.to_row()})
    return rows


def _compute_shap_importance(model_lgb, X_sample: pd.DataFrame) -> pd.Series:
    expl = shap.TreeExplainer(model_lgb)
    shap_values = expl.shap_values(X_sample)
    # LightGBM binary: shap_values 可能是单个数组或 [class0, class1] 列表
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    importance = np.abs(shap_values).mean(axis=0)
    return pd.Series(importance, index=X_sample.columns).sort_values(ascending=False)


def _srd_matrix(rank_dict: dict[str, pd.Series]) -> pd.DataFrame:
    keys = list(rank_dict.keys())
    n = len(keys)
    out = np.zeros((n, n))
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            if i == j:
                out[i, j] = 0.0
                continue
            common = rank_dict[ki].index.intersection(rank_dict[kj].index)
            r_i = rank_dict[ki].loc[common].rank()
            r_j = rank_dict[kj].loc[common].rank()
            corr, _ = spearmanr(r_i.values, r_j.values)
            out[i, j] = 1.0 - float(corr)
    return pd.DataFrame(out, index=keys, columns=keys)


def _plot_srd_heatmap(srd: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(srd, annot=True, fmt=".3f", cmap="Reds", vmin=0, vmax=0.5,
                cbar_kws={"label": "SRD = 1 - Spearman(rank_i, rank_j)"}, ax=ax,
                square=True, linewidths=0.5)
    ax.set_title("SHAP Regime Divergence (SRD) Matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_regime_metrics_bar(eval_df: pd.DataFrame, path: Path) -> None:
    sub = eval_df[eval_df["split"].isin(["bear", "sideway", "bull", "ALL"])].copy()
    metrics_to_plot = ["rankic_mean", "ic_mean", "auc"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, m in zip(axes, metrics_to_plot):
        piv = sub.pivot(index="split", columns="model", values=m)
        piv = piv.reindex(["bear", "sideway", "bull", "ALL"])
        piv.plot.bar(ax=ax, rot=0, edgecolor="black", linewidth=0.5)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(m)
        ax.legend(loc="best", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Per-regime metrics: Global vs Routed", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="LightGBM-std",
                        help="底座模型名，必须是 GBDT（TreeSHAP 兼容）。"
                             "推荐：LightGBM-std / LightGBM-shallow-reg / CatBoost-reg")
    parser.add_argument("--feature-groups", type=str, default="",
                        help="自定义特征组，逗号分隔，如 'G1,G2,G3,G4'。"
                             "默认为 DEFAULT_GROUPS（G1–G6）。"
                             "macro_regime_3 始终从特征中剔除，仅作路由器。")
    parser.add_argument("--shap-sample", type=int, default=10000,
                        help="每个 sub-model 计算 SHAP 时的抽样大小")
    parser.add_argument("--sample-tickers", type=int, default=None,
                        help="调试用：抽样 N 只股票")
    parser.add_argument("--tag", type=str, default="lgbm_std")
    args = parser.parse_args()

    if args.base_model not in MODEL_REGISTRY:
        raise SystemExit(f"unknown base-model: {args.base_model}. "
                         f"available: {list(MODEL_REGISTRY)}")
    base_cls = MODEL_REGISTRY[args.base_model]
    if base_cls.family != "gbdt":
        raise SystemExit(f"base-model must be GBDT family (TreeSHAP), got {base_cls.family}")
    is_regression = getattr(base_cls, "regression_target", False)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / f"regime_{ts}_{args.tag}"
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "submodel_shap").mkdir(parents=True, exist_ok=True)

    print(f"=== Regime-Conditioned Analysis (base={args.base_model}, tag={args.tag}) ===")
    print(f"output -> {out_dir}\n")

    # ---- 1. 数据 ----
    print("[1/6] loading data ...")
    df = load_dataset()
    df = build_label(df)
    df = df.dropna(subset=[TARGET_RET_COL]).reset_index(drop=True)
    if args.sample_tickers:
        rng = np.random.default_rng(42)
        chosen = rng.choice(df[TICKER_COL].unique(),
                            size=min(args.sample_tickers, df[TICKER_COL].nunique()),
                            replace=False)
        df = df[df[TICKER_COL].isin(chosen)].reset_index(drop=True)

    train, test = split_train_test(df)
    print(f"  train: {len(train):,}  test: {len(test):,}")

    # ---- 2. 特征：自定义组 - macro_regime_3（macro_regime_3 只用作路由器）----
    if args.feature_groups:
        groups = tuple(g.strip() for g in args.feature_groups.split(",") if g.strip())
    else:
        groups = DEFAULT_GROUPS
    feat_cols = _features_without_regime(groups)
    print(f"  feature groups: {list(groups)}")
    print(f"  features ({len(feat_cols)}): {feat_cols[:6]} ... +{max(0, len(feat_cols)-6)}")
    pre = Preprocessor(mode=base_cls.preprocess, feature_cols=feat_cols)
    X_train = pre.fit_transform(train)
    X_test = pre.transform(test)

    # 训练目标：classification 用 LABEL_COL；regression 用 r_future_5（裁剪极端分位）
    if is_regression:
        yr = train[TARGET_RET_COL].to_numpy(dtype="float32")
        lo, hi = float(np.quantile(yr, 0.001)), float(np.quantile(yr, 0.999))
        y_train = np.clip(yr, lo, hi).astype("float32")
        print(f"  regression target: r_future_5 clipped to [{lo:.4f}, {hi:.4f}]")
    else:
        y_train = train[LABEL_COL].to_numpy()
    y_test = test[LABEL_COL].to_numpy()
    regime_train = train[REGIME_COL].astype(str).reset_index(drop=True)
    regime_test = test[REGIME_COL].astype(str).reset_index(drop=True)

    # 训练集 regime 分布
    print("  train regime dist:", regime_train.value_counts().to_dict())
    print("  test  regime dist:", regime_test.value_counts().to_dict())

    # ---- 3. 训练 Routed Ensemble ----
    print(f"\n[2/6] training Regime-Conditioned ensemble ({args.base_model} × 3) ...")
    t0 = time.time()
    ensemble = RegimeEnsemble(base_factory=lambda: build(args.base_model))
    ensemble.fit(X_train, y_train, regime_train)
    print(f"  ensemble train time: {time.time()-t0:.1f}s")

    # ---- 4. 训练 Global 对照 ----
    print(f"\n[3/6] training global comparison model ({args.base_model} on full set) ...")
    t0 = time.time()
    global_model = build(args.base_model).fit(X_train, y_train)
    print(f"  global train time: {time.time()-t0:.1f}s")

    # ---- 5. 预测 + 指标（Table 5）----
    print("\n[4/6] predicting + per-regime metrics ...")
    pred_routed = ensemble.predict_proba(X_test, regime_test)
    pred_global = global_model.predict_proba(X_test)

    # 保存预测
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)
    pred_df = test[[DATE_COL, TICKER_COL, TARGET_RET_COL, REGIME_COL]].reset_index(drop=True)
    pred_df["pred_routed"] = pred_routed
    pred_df["pred_global"] = pred_global
    pred_df.to_parquet(pred_dir / "predictions.parquet", index=False)

    rows = []
    rows += _per_regime_metrics(test, pred_global, regime_test, "global")
    rows += _per_regime_metrics(test, pred_routed, regime_test, "routed")
    eval_df = pd.DataFrame(rows)
    # 增加 Delta 列（routed - global）
    base = eval_df[eval_df["model"] == "global"].set_index("split")
    delta_rows = []
    for split in eval_df["split"].unique():
        if split not in base.index:
            continue
        r = eval_df[(eval_df["model"] == "routed") & (eval_df["split"] == split)].iloc[0]
        g = base.loc[split]
        delta_rows.append({
            "split": split, "model": "delta(routed-global)",
            "ic_mean": r["ic_mean"] - g["ic_mean"],
            "rankic_mean": r["rankic_mean"] - g["rankic_mean"],
            "icir": r["icir"] - g["icir"],
            "rankicir": r["rankicir"] - g["rankicir"],
            "auc": r["auc"] - g["auc"],
            "top1pct_ret": r["top1pct_ret"] - g["top1pct_ret"],
            "top5pct_ret": r["top5pct_ret"] - g["top5pct_ret"],
            "n_days": r["n_days"],
        })
    delta_df = pd.DataFrame(delta_rows)
    eval_df = pd.concat([eval_df, delta_df], ignore_index=True, sort=False)
    eval_df.to_csv(out_dir / "regime_eval.csv", index=False, encoding="utf-8-sig")
    print(f"  saved regime_eval.csv ({len(eval_df)} rows)")

    # 简表打印
    show_cols = ["split", "model", "auc", "ic_mean", "rankic_mean", "rankicir",
                 "top1pct_ret", "top5pct_ret"]
    print(eval_df[show_cols].to_string(index=False))

    # 回测：routed 全集 Top-5%
    bt = backtest_topk(test[[DATE_COL, TICKER_COL, TARGET_RET_COL]], pred_routed, frac=0.05)
    bt_g = backtest_topk(test[[DATE_COL, TICKER_COL, TARGET_RET_COL]], pred_global, frac=0.05)
    print(f"\n  Top-5% Sharpe — global: {bt_g.sharpe:.3f}  routed: {bt.sharpe:.3f}  "
          f"Δ={bt.sharpe-bt_g.sharpe:+.3f}")

    # ---- 6. SHAP Regime Divergence (Table 6 + Figure 5) ----
    print("\n[5/6] computing per-regime SHAP importance (TreeSHAP) ...")
    rng = np.random.default_rng(42)
    importance_dict: dict[str, pd.Series] = {}
    for r, sub in ensemble.models.items():
        mask = (regime_test == r).values
        if mask.sum() < 100:
            print(f"  [skip] regime={r} too few test samples ({mask.sum()})")
            continue
        idx = np.where(mask)[0]
        if len(idx) > args.shap_sample:
            idx = rng.choice(idx, size=args.shap_sample, replace=False)
        Xs = X_test.iloc[idx]
        imp = _compute_shap_importance(sub.raw_model, Xs)
        imp.to_csv(out_dir / "submodel_shap" / f"{r}_top_features.csv",
                   header=["mean_abs_shap"], encoding="utf-8-sig")
        importance_dict[r] = imp
        top5 = imp.head(5).round(4).to_dict()
        print(f"  regime={r:7s} top-5 features: {top5}")

    print("\n[6/6] SRD matrix ...")
    srd = _srd_matrix(importance_dict)
    srd.to_csv(out_dir / "srd_matrix.csv", encoding="utf-8-sig")
    print(srd.round(3).to_string())
    _plot_srd_heatmap(srd, out_dir / "figures" / "regime_srd_heatmap.png")
    _plot_regime_metrics_bar(eval_df, out_dir / "figures" / "regime_metrics_bar.png")

    # Top-K feature 横向对比 figure
    fig, ax = plt.subplots(figsize=(11, 7))
    top_n = 15
    feats_union: list[str] = []
    for r in REGIMES:
        if r in importance_dict:
            for f in importance_dict[r].head(top_n).index:
                if f not in feats_union:
                    feats_union.append(f)
    df_imp = pd.DataFrame({r: importance_dict[r].reindex(feats_union)
                           for r in REGIMES if r in importance_dict})
    df_imp.plot.barh(ax=ax, edgecolor="black", linewidth=0.4)
    ax.set_title("SHAP feature importance per regime sub-model (Top union)")
    ax.invert_yaxis()
    ax.legend(title="regime")
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "regime_top_features_compare.png", dpi=160)
    plt.close(fig)

    # 配置快照
    (out_dir / "config.json").write_text(json.dumps({
        "timestamp": ts,
        "base_model": args.base_model,
        "regression_target": is_regression,
        "preprocess": base_cls.preprocess,
        "feature_groups": list(groups),
        "features": feat_cols,
        "shap_sample_per_regime": args.shap_sample,
        "regime_routing_column": REGIME_COL,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[done] all outputs at: {out_dir}")


if __name__ == "__main__":
    main()
