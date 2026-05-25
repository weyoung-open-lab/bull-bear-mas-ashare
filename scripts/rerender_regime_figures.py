"""Rerender the figures for a regime_*/ run with English labels (paper-ready)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

mpl.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False

REGIMES = ("bear", "sideway", "bull")


def main(run_dir: Path) -> None:
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # --- Figure 5: SRD heatmap ---
    srd = pd.read_csv(run_dir / "srd_matrix.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(srd, annot=True, fmt=".3f", cmap="Reds", vmin=0, vmax=0.5,
                cbar_kws={"label": "SRD = 1 - Spearman(rank_i, rank_j)"},
                square=True, linewidths=0.5, ax=ax)
    ax.set_title("SHAP Regime Divergence (SRD) Matrix")
    fig.tight_layout()
    fig.savefig(fig_dir / "regime_srd_heatmap.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {fig_dir/'regime_srd_heatmap.png'}")

    # --- Per-regime metrics bar (Table 5 view) ---
    eval_df = pd.read_csv(run_dir / "regime_eval.csv")
    sub = eval_df[eval_df["split"].isin(["bear", "sideway", "bull", "ALL"])]
    sub = sub[sub["model"].isin(["global", "routed"])]
    metrics_to_plot = [("rankic_mean", "RankIC mean"),
                       ("rankicir", "RankICIR"),
                       ("auc", "AUC")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (col, title) in zip(axes, metrics_to_plot):
        piv = sub.pivot(index="split", columns="model", values=col)
        piv = piv.reindex(["bear", "sideway", "bull", "ALL"])[["global", "routed"]]
        piv.plot.bar(ax=ax, rot=0, edgecolor="black", linewidth=0.5,
                     color=["#7f7f7f", "#2ca02c"])
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(title)
        ax.set_xlabel("Regime split")
        ax.legend(loc="best", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Per-regime metrics: Global vs Routed (Table 5)", y=1.03)
    fig.tight_layout()
    fig.savefig(fig_dir / "regime_metrics_bar.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {fig_dir/'regime_metrics_bar.png'}")

    # --- Per-regime SHAP top features compare ---
    shap_dir = run_dir / "submodel_shap"
    imp_dict: dict[str, pd.Series] = {}
    for r in REGIMES:
        f = shap_dir / f"{r}_top_features.csv"
        if not f.exists():
            continue
        s = pd.read_csv(f, index_col=0).iloc[:, 0]
        imp_dict[r] = s.sort_values(ascending=False)

    if imp_dict:
        feats_union: list[str] = []
        top_n = 15
        for r in REGIMES:
            if r in imp_dict:
                for f in imp_dict[r].head(top_n).index:
                    if f not in feats_union:
                        feats_union.append(f)
        df_imp = pd.DataFrame({r: imp_dict[r].reindex(feats_union)
                               for r in REGIMES if r in imp_dict})
        fig, ax = plt.subplots(figsize=(11, 7))
        df_imp.plot.barh(ax=ax, edgecolor="black", linewidth=0.4, width=0.78,
                         color=["#d62728", "#7f7f7f", "#2ca02c"])
        ax.invert_yaxis()
        ax.set_title("SHAP feature importance per regime sub-model (Top union 15)")
        ax.set_xlabel("mean(|SHAP|)")
        ax.legend(title="regime")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "regime_top_features_compare.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {fig_dir/'regime_top_features_compare.png'}")

        # 单独画每个 regime top-15
        fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=False)
        colors = {"bear": "#d62728", "sideway": "#7f7f7f", "bull": "#2ca02c"}
        for ax, r in zip(axes, REGIMES):
            if r not in imp_dict:
                ax.axis("off"); continue
            s = imp_dict[r].head(15)[::-1]
            ax.barh(s.index, s.values, color=colors[r], edgecolor="black", lw=0.4)
            ax.set_title(f"regime = {r}")
            ax.set_xlabel("mean(|SHAP|)")
            ax.grid(axis="x", alpha=0.3)
        fig.suptitle("Top-15 SHAP feature importance per regime sub-model", y=1.02)
        fig.tight_layout()
        fig.savefig(fig_dir / "regime_top_features_each.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {fig_dir/'regime_top_features_each.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    main(args.run_dir)
