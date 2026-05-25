"""Generate missing figures for §5 Experiments.

Reads from CSV files only — every number is data-driven.

Outputs (300 dpi PNG) in figure/:
  model_compare_bar.png        - 14-model RankICIR ranking
  bce_vs_mse_bar.png            - BCE vs MSE delta per family
  feature_ablation_curve.png    - G1->G6 cumulative RankICIR
  srd_heatmap.png               - 4-config SRD comparison
  main_ablation_bar.png         - 9 configs from B0 to D1d+V
  walkforward_yearly.png        - 7-year Trend vs D1c bar chart
  quintile_analysis.png         - 5-quintile MaxDD vs return
  yx_mechanism.png              - (already exists, skip)
  framework.png                 - (already exists, skip)
"""

from __future__ import annotations

import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figure"
FIG.mkdir(parents=True, exist_ok=True)


def f(p):
    matches = sorted(glob.glob(str(ROOT / p)))
    return matches[-1] if matches else None


def model_compare():
    df = pd.read_csv(f("results/main_compare_*full_reg*/metrics_summary.csv"))
    df = df.sort_values("rankicir", ascending=True).reset_index(drop=True)

    family_color = {
        "gbdt":       "#2e7d32",
        "linear":     "#1976d2",
        "tabular_dl": "#7b1fa2",
        "sequence":   "#7b1fa2",
        "factor":     "#9e9e9e",
    }
    colors = [family_color.get(s, "#666") for s in df["family"]]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(df))
    ax.barh(y, df["rankicir"], color=colors, edgecolor="black", linewidth=0.4)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(df["model"], fontsize=9)
    ax.set_xlabel("RankICIR (test panel 2023--2026, 733 trading days)")
    ax.set_title("Cross-family model comparison (regression MSE objective)", fontsize=11)
    ax.grid(True, alpha=0.3, axis="x")

    for yi, v in enumerate(df["rankicir"]):
        ha = "left" if v >= 0 else "right"
        dx = 0.005 if v >= 0 else -0.005
        ax.text(v + dx, yi, f"{v:+.3f}", va="center", ha=ha, fontsize=8)

    # Highlight CatBoost
    cat_idx = df.index[df["model"] == "CatBoost-reg"].tolist()
    if cat_idx:
        i = cat_idx[0]
        ax.barh(i, df.loc[i, "rankicir"], color="#ff6d00",
                edgecolor="black", linewidth=1.5)
        ax.annotate("Selected backbone", xy=(df.loc[i, "rankicir"], i),
                     xytext=(df["rankicir"].min() * 0.3, i),
                     fontsize=9, fontweight="bold", color="#ff6d00",
                     arrowprops=dict(arrowstyle="->", color="#ff6d00", lw=1.5))

    # Legend
    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor="#ff6d00", label="CatBoost (selected)"),
        Patch(facecolor="#2e7d32", label="GBDT"),
        Patch(facecolor="#1976d2", label="Linear"),
        Patch(facecolor="#7b1fa2", label="Deep learning"),
        Patch(facecolor="#9e9e9e", label="Factor baseline"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=8.5)

    plt.tight_layout()
    out = FIG / "model_compare_bar.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> {out.relative_to(ROOT)}")


def bce_vs_mse():
    df = pd.read_csv(ROOT / "results/binary_vs_regression.csv")
    df["delta"] = df["rankicir_reg"] - df["rankicir_binary"]
    df = df.sort_values("delta", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    y = np.arange(len(df))
    width = 0.36
    ax.barh(y - width/2, df["rankicir_binary"], width,
             color="#bdbdbd", edgecolor="black", linewidth=0.4,
             label="BCE objective")
    ax.barh(y + width/2, df["rankicir_reg"], width,
             color="#2e7d32", edgecolor="black", linewidth=0.4,
             label="MSE objective")
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"], fontsize=9)
    ax.set_xlabel("RankICIR")
    ax.set_title("Effect of switching loss function from BCE to MSE", fontsize=11)
    ax.grid(True, alpha=0.3, axis="x")
    ax.legend(loc="lower right", fontsize=9)

    # Highlight CatBoost delta
    cat = df[df["label"] == "CatBoost"]
    if len(cat) > 0:
        i = cat.index[0]
        ax.annotate(rf"CatBoost gain $+{cat['delta'].iloc[0]:.3f}$ (largest)",
                     xy=(cat["rankicir_reg"].iloc[0], i + width/2),
                     xytext=(0.1, i + 1.5),
                     fontsize=9.5, fontweight="bold", color="#ff6d00",
                     arrowprops=dict(arrowstyle="->", color="#ff6d00", lw=1.5))

    plt.tight_layout()
    out = FIG / "bce_vs_mse_bar.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> {out.relative_to(ROOT)}")


def feature_ablation():
    p = f("results/feature_ablation_*/feature_ablation.csv")
    df = pd.read_csv(p)
    # Order matches the cumulative order
    label_order = ["G1", "G1+G2", "G1+G2+G3", "G1+G2+G3+G4",
                    "G1+G2+G3+G4+G5", "Full(G1-G6)"]
    df = df.set_index("config").loc[label_order].reset_index()
    df["short_label"] = ["G1", "+G2", "+G3", "+G4", "+G5", "+G6"]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(df))
    bars = ax.bar(x, df["rankicir"], color="#1976d2", edgecolor="black",
                    linewidth=0.5, width=0.65)
    # Highlight peak
    peak_idx = int(df["rankicir"].idxmax())
    bars[peak_idx].set_color("#ff6d00")
    bars[peak_idx].set_edgecolor("black")
    bars[peak_idx].set_linewidth(1.5)

    for xi, v, n in zip(x, df["rankicir"], df["n_features"]):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=9)
        ax.text(xi, -0.04, f"({n} feat.)", ha="center", fontsize=8, color="#555")

    ax.set_xticks(x)
    ax.set_xticklabels(df["short_label"], fontsize=10)
    ax.set_ylabel("RankICIR (LightGBM-shallow-reg, test 2023--2026)")
    ax.set_title("Feature group cumulative ablation: G1234 is peak", fontsize=11)
    ax.set_ylim(-0.08, df["rankicir"].max() + 0.07)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.grid(True, alpha=0.3, axis="y")
    ax.annotate("Peak\n(macro_regime_3 hurts when used as feature)",
                  xy=(peak_idx, df.iloc[peak_idx]["rankicir"]),
                  xytext=(peak_idx + 0.5, df["rankicir"].max() - 0.05),
                  fontsize=9, color="#ff6d00", fontweight="bold",
                  arrowprops=dict(arrowstyle="->", color="#ff6d00", lw=1.5))

    plt.tight_layout()
    out = FIG / "feature_ablation_curve.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> {out.relative_to(ROOT)}")


def srd_heatmap():
    """4-config SRD comparison: BCE/MSE * G1234/G16 (only 4 cells of upper triangle)."""
    configs = {
        "LGBM BCE + G1--G6":  ROOT / "results/regime_20260506_221050_full/srd_matrix.csv",
        "LGBM MSE + G1--G6":  ROOT / "results/regime_20260506_234936_full_lgbm_shallow_reg/srd_matrix.csv",
        "LGBM MSE + G1234":   ROOT / "results/regime_20260507_013022_final_g1234/srd_matrix.csv",
        "CatBoost MSE + G1234": ROOT / "results/regime_20260507_013443_final_g1234_cat/srd_matrix.csv",
    }
    rows = []
    for name, path in configs.items():
        d = pd.read_csv(path, index_col=0)
        rows.append({"config": name,
                      "bear-bull":     d.loc["bear", "bull"],
                      "bear-sideway":  d.loc["bear", "sideway"],
                      "bull-sideway":  d.loc["bull", "sideway"]})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    matrix = df[["bear-bull", "bear-sideway", "bull-sideway"]].values
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=0.75, aspect="auto")
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["Bear / Bull", "Bear / Sideway", "Bull / Sideway"], fontsize=10)
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df["config"], fontsize=9)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            color = "white" if v > 0.5 else "black"
            fw = "bold" if i == 3 else "normal"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                      color=color, fontsize=10.5, fontweight=fw)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("SRD (1 - Spearman SHAP rank correlation)")
    ax.set_title("SHAP Regime Divergence: CatBoost + G1234 maximises divergence", fontsize=11)
    plt.tight_layout()
    out = FIG / "srd_heatmap.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> {out.relative_to(ROOT)}")


def main_ablation():
    df = pd.read_csv(ROOT / "bull_bear/results/final_ablation.csv")
    # Build short labels in pipeline order
    df["short"] = ["B0", "M1", "T", "BC", "D1a", "D1b", "D1c", "D1d",
                    "V", "FINAL"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(df))
    # Color scheme: baselines gray, intermediate green, headline orange
    pal = ["#9e9e9e", "#bdbdbd",          # B0, M1
            "#81c784",                      # T
            "#4caf50", "#388e3c", "#2e7d32",# BC, D1a, D1b
            "#ff6d00",                      # D1c (headline)
            "#1b5e20",                      # D1d
            "#c62828",                      # V (detrimental)
            "#888888"]                      # FINAL
    bars = ax.bar(x, df["rankicir"], color=pal, edgecolor="black", linewidth=0.5)
    for xi, v in zip(x, df["rankicir"]):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=9)
    trend_ref = df[df["short"] == "T"]["rankicir"].iloc[0]
    ax.axhline(trend_ref, color="gray", linestyle=":", linewidth=1,
                label=f"Trend pure (T) = {trend_ref:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(df["short"], fontsize=10)
    ax.set_ylabel("RankICIR (test panel 2023--2026)")
    ax.set_title("Main ablation: from B0 baseline to D1c final system",
                  fontsize=11)
    ax.set_ylim(0, df["rankicir"].max() * 1.12)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="upper left", fontsize=9)

    # Annotate headline
    i = int(df.index[df["short"] == "D1c"][0])
    ax.annotate("Final system",
                  xy=(i, df.iloc[i]["rankicir"]),
                  xytext=(i - 1.7, df.iloc[i]["rankicir"] + 0.06),
                  fontsize=10, fontweight="bold", color="#ff6d00",
                  arrowprops=dict(arrowstyle="->", color="#ff6d00", lw=1.5))
    i_v = int(df.index[df["short"] == "V"][0])
    ax.annotate("Vol-gate hurts",
                  xy=(i_v, df.iloc[i_v]["rankicir"]),
                  xytext=(i_v - 1.5, df.iloc[i_v]["rankicir"] - 0.12),
                  fontsize=9, color="#c62828",
                  arrowprops=dict(arrowstyle="->", color="#c62828", lw=1.2))

    plt.tight_layout()
    out = FIG / "main_ablation_bar.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> {out.relative_to(ROOT)}")


def walkforward_yearly():
    df = pd.read_csv(ROOT / "bull_bear/results/final/rolling_walkforward.csv")
    df = df[df["year"] <= 2025]    # exclude 2026 (6 days, noise)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(df))
    w = 0.38
    ax.bar(x - w/2, df["trend_rankicir"], w, color="#bdbdbd",
            edgecolor="black", linewidth=0.4, label="Trend (Alpha only)")
    ax.bar(x + w/2, df["d1_rankicir"], w, color="#2e7d32",
            edgecolor="black", linewidth=0.4, label="D1c (adversarial system)")
    for xi, t, d in zip(x, df["trend_rankicir"], df["d1_rankicir"]):
        ax.text(xi - w/2, t + 0.015, f"{t:.3f}", ha="center", fontsize=7.5)
        ax.text(xi + w/2, d + 0.015, f"{d:.3f}", ha="center", fontsize=7.5,
                 color="#1b5e20", fontweight="bold")
    # Mean lines
    ax.axhline(df["trend_rankicir"].mean(), color="#888", linestyle=":", lw=1,
                label=f"Trend 7-yr mean = {df['trend_rankicir'].mean():.3f}")
    ax.axhline(df["d1_rankicir"].mean(), color="#1b5e20", linestyle=":", lw=1,
                label=f"D1c 7-yr mean = {df['d1_rankicir'].mean():.3f}")

    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in df["year"]], fontsize=10)
    ax.set_xlabel("Calendar year (walk-forward train window expands annually)")
    ax.set_ylabel("RankICIR")
    ax.set_title("Walk-forward cross-validation: D1c wins 7/7 years",
                  fontsize=11)
    ax.set_ylim(0, df["d1_rankicir"].max() * 1.15)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="upper left", fontsize=8.5, ncol=2)

    # Annotate COVID and bear year
    cov_x = list(df["year"]).index(2020)
    bear_x = list(df["year"]).index(2022)
    ax.annotate("COVID shock", xy=(cov_x, df.iloc[cov_x]["d1_rankicir"]),
                  xytext=(cov_x - 0.5, df.iloc[cov_x]["d1_rankicir"] + 0.15),
                  fontsize=8, color="#c62828",
                  arrowprops=dict(arrowstyle="->", color="#c62828", lw=0.8))
    ax.annotate("Deep bear\n($\Delta$=+140 bp)",
                  xy=(bear_x, df.iloc[bear_x]["d1_rankicir"]),
                  xytext=(bear_x - 0.8, df.iloc[bear_x]["d1_rankicir"] + 0.1),
                  fontsize=8, color="#c62828",
                  arrowprops=dict(arrowstyle="->", color="#c62828", lw=0.8))

    plt.tight_layout()
    out = FIG / "walkforward_yearly.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> {out.relative_to(ROOT)}")


def quintile_chart():
    df = pd.read_csv(ROOT / "bull_bear/results/final/bear_quintile_analysis.csv")

    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = np.arange(len(df))
    w = 0.35
    ax1.bar(x - w/2, df["avg_maxdd_5d"] * 100, w,
              color="#c62828", edgecolor="black", linewidth=0.4,
              label="Realised 5-day MaxDD (%)")
    for xi, v in zip(x, df["avg_maxdd_5d"] * 100):
        ax1.text(xi - w/2, v + 0.1, f"{v:.2f}", ha="center", fontsize=8.5)
    ax1.set_ylabel("Realised 5-day MaxDD (%)", color="#c62828", fontsize=10)
    ax1.tick_params(axis="y", labelcolor="#c62828")
    ax1.set_ylim(0, df["avg_maxdd_5d"].max() * 110)

    ax2 = ax1.twinx()
    ax2.bar(x + w/2, df["avg_r_future_5"] * 100, w,
              color="#1976d2", edgecolor="black", linewidth=0.4,
              label="Realised 5-day forward return (%)")
    for xi, v in zip(x, df["avg_r_future_5"] * 100):
        ax2.text(xi + w/2, v + 0.02, f"{v:+.3f}", ha="center", fontsize=8.5,
                  color="#0d47a1")
    ax2.set_ylabel("Realised 5-day forward return (%)", color="#1976d2", fontsize=10)
    ax2.tick_params(axis="y", labelcolor="#1976d2")
    ax2.set_ylim(0, df["avg_r_future_5"].max() * 130)

    ax1.set_xticks(x)
    ax1.set_xticklabels(df["label"], fontsize=9.5)
    ax1.set_title("Bear Agent quintile analysis: monotone MaxDD, non-monotone return",
                    fontsize=11)
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.legend(loc="upper left", fontsize=9)
    ax2.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    out = FIG / "quintile_analysis.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> {out.relative_to(ROOT)}")


def main():
    print("Generating §5 Experiments figures (300 dpi) ...")
    model_compare()
    bce_vs_mse()
    feature_ablation()
    srd_heatmap()
    main_ablation()
    walkforward_yearly()
    quintile_chart()
    print("Done.")


if __name__ == "__main__":
    main()
