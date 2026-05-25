"""Rerender SHAP figures with English labels (paper-ready)."""

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


def main(run_dir: Path) -> None:
    csv_dir = run_dir / "csv"
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    l1 = pd.read_csv(csv_dir / "L1_global.csv")
    yearly = pd.read_csv(csv_dir / "L2_yearly_importance.csv", index_col=0)
    yearly.columns = [int(c) for c in yearly.columns]
    rcorr = pd.read_csv(csv_dir / "L2_rank_correlation.csv", index_col=0)
    l3 = pd.read_csv(csv_dir / "L3_top1pct_vs_global.csv")

    # ---- Figure 2 (L1 global bar) ----
    top25 = l1.sort_values("mean_abs_shap", ascending=False).head(25)
    fig, ax = plt.subplots(figsize=(9, 9))
    order = np.argsort(top25["mean_abs_shap"].values)
    ax.barh(top25["feature"].values[order], top25["mean_abs_shap"].values[order],
            color="#4c72b0", edgecolor="black", lw=0.4)
    ax.set_xlabel("mean(|SHAP|)")
    ax.set_title("L1 Global SHAP feature importance (Top 25)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / "fig2_L1_global_bar.png", dpi=160); plt.close(fig)
    print("  saved fig2_L1_global_bar.png")

    # ---- Figure 6 (directional decomposition) ----
    top20 = l1.sort_values("mean_abs_shap", ascending=False).head(20).copy()
    fig, ax = plt.subplots(figsize=(11, 8))
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
    print("  saved fig6_directional_top20.png")

    # ---- Figure 3 (L2 yearly stability) ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    rcorr.index = [int(i) for i in rcorr.index]
    rcorr.columns = [int(c) for c in rcorr.columns]
    sns.heatmap(rcorr, annot=True, fmt=".2f", cmap="Greens",
                vmin=0.5, vmax=1.0, cbar_kws={"label": "Spearman rank-corr"},
                ax=axes[0], square=True, linewidths=0.5)
    axes[0].set_title("Year-to-year SHAP rank correlation")

    top_feats = l1.sort_values("mean_abs_shap", ascending=False).head(8)["feature"].tolist()
    yearly_top = yearly.loc[top_feats].T
    yearly_top.index.name = "year"
    yearly_top.plot(ax=axes[1], marker="o", lw=1.6)
    axes[1].set_title("Top-8 features: yearly mean(|SHAP|)")
    axes[1].set_xlabel("year"); axes[1].set_ylabel("mean(|SHAP|)")
    axes[1].legend(loc="best", fontsize=8, ncol=2)
    axes[1].grid(alpha=0.3)
    fig.suptitle("L2 Year-over-year SHAP stability", y=1.02)
    fig.tight_layout(); fig.savefig(fig_dir / "fig3_L2_yearly_stability.png", dpi=160); plt.close(fig)
    print("  saved fig3_L2_yearly_stability.png")

    # ---- Figure 4 (L3 Top-1% vs Global) ----
    top20_global = l1.sort_values("mean_abs_shap", ascending=False).head(20)["feature"].tolist()
    sub = l3.set_index("feature").loc[top20_global]
    fig, ax = plt.subplots(figsize=(11, 8))
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
    print("  saved fig4_L3_top1pct_vs_global.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    main(args.run_dir)
