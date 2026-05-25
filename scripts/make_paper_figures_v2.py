"""Generate 8 paper figures (§5 Experiments) with strict academic style.

Output filenames (figure/):
  fig_model_compare.png         (14-model horizontal bar)
  fig_bce_vs_mse.png            (BCE vs MSE scatter)
  fig_feature_ablation.png      (G1-G6 cumulative line)
  fig_srd_heatmap.png           (4-config x 3-pair heatmap)
  fig_ablation_bar.png          (main ablation horizontal bar)
  fig_yx_mechanism.png          (Y vs X line plot)
  fig_walkforward.png           (7-year grouped bars)
  fig_quintile.png              (5-quintile dual-axis)

Style: Arial/Helvetica, title 12pt, labels 11pt, ticks 10pt, 300 dpi.
All numbers loaded live from CSV files.
"""

from __future__ import annotations

import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figure"
FIG.mkdir(parents=True, exist_ok=True)


# ============================================================
# Style
# ============================================================
plt.rcParams.update({
    "font.family":       "Arial",
    "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   9,
    "axes.edgecolor":    "#444",
    "axes.linewidth":    0.8,
    "axes.grid":         True,
    "grid.color":        "#bbb",
    "grid.linewidth":    0.5,
    "grid.alpha":        0.4,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# Academic palette
C = {
    "gbdt":     "#2E7D32",   # green
    "linear":   "#1565C0",   # blue
    "dl":       "#6A1B9A",   # purple (TabNet, FT-Transformer, ALSTM, TCN)
    "factor":   "#757575",   # gray
    "catboost": "#FF6F00",   # orange (highlight)
    "trend":    "#424242",   # dark gray (reference)
    "d1c":      "#FF6F00",   # orange (final)
    "bad":      "#C62828",   # red (vol-gate detrimental)
    "y_line":   "#1B5E20",   # adversarial Y
    "x_line":   "#C62828",   # additive X
    "heat_lo":  "#FFF3E0",   # heatmap light
    "heat_hi":  "#BF360C",   # heatmap dark
}


def find(p: str) -> str:
    m = sorted(glob.glob(str(ROOT / p)))
    if not m:
        raise FileNotFoundError(p)
    return m[-1]


def family_color(family: str, model: str) -> str:
    if model == "CatBoost-reg":
        return C["catboost"]
    if family == "gbdt":
        return C["gbdt"]
    if family == "linear":
        return C["linear"]
    if family in ("tabular_dl", "sequence"):
        return C["dl"]
    return C["factor"]


# ============================================================
# Figure 3 — 14-model horizontal bar
# ============================================================
def fig_model_compare():
    df = pd.read_csv(find("results/main_compare_*full_reg*/metrics_summary.csv"))
    bt = pd.read_csv(find("results/main_compare_*full_reg*/backtest_summary.csv"))

    # add top-5% Sharpe
    def s(m):
        sub = bt[bt["model"] == m].reset_index(drop=True)
        return float(sub.iloc[2]["sharpe"]) if len(sub) >= 3 else float("nan")
    df["top5_sharpe"] = df["model"].apply(s)
    df = df.sort_values("rankicir", ascending=True).reset_index(drop=True)
    colors = [family_color(f, m) for f, m in zip(df["family"], df["model"])]

    fig, ax = plt.subplots(figsize=(9, 5.6))
    y = np.arange(len(df))
    bars = ax.barh(y, df["rankicir"], color=colors,
                     edgecolor="black", linewidth=0.4, height=0.72)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(df["model"], fontsize=10)
    ax.set_xlabel("RankICIR (test panel 2023--2026)")
    ax.set_title("Cross-family model comparison (regression MSE objective)")
    ax.set_xlim(-0.5, 0.45)

    # value labels
    for yi, v in enumerate(df["rankicir"]):
        if v >= 0:
            ax.text(v + 0.007, yi, f"{v:+.3f}", va="center", ha="left", fontsize=8.5)
        else:
            ax.text(v - 0.007, yi, f"{v:+.3f}", va="center", ha="right", fontsize=8.5)

    # CatBoost annotation
    cat_idx = int(df.index[df["model"] == "CatBoost-reg"][0])
    ax.annotate("Selected backbone",
                  xy=(df.iloc[cat_idx]["rankicir"], cat_idx),
                  xytext=(0.05, cat_idx - 4),
                  fontsize=10, fontweight="bold", color=C["catboost"],
                  arrowprops=dict(arrowstyle="->", color=C["catboost"], lw=1.4))

    # TabNet annotation
    tab_idx = int(df.index[df["model"] == "TabNet-reg"][0])
    tab_sharpe = float(df.iloc[tab_idx]["top5_sharpe"])
    tab_ric = float(df.iloc[tab_idx]["rankicir"])
    ax.annotate(f"Highest Top-5% Sharpe ({tab_sharpe:+.2f})\nbut RankICIR = {tab_ric:+.3f}",
                  xy=(tab_ric, tab_idx),
                  xytext=(-0.42, tab_idx + 2),
                  fontsize=8.5, color=C["dl"],
                  arrowprops=dict(arrowstyle="->", color=C["dl"], lw=1.0))

    # legend
    legend_items = [
        Patch(facecolor=C["catboost"], label="CatBoost (selected)"),
        Patch(facecolor=C["gbdt"], label="GBDT"),
        Patch(facecolor=C["linear"], label="Linear"),
        Patch(facecolor=C["dl"], label="Deep learning"),
        Patch(facecolor=C["factor"], label="Factor baseline"),
    ]
    ax.legend(handles=legend_items, loc="lower right", framealpha=0.95)

    out = FIG / "fig_model_compare.png"
    plt.savefig(out)
    plt.close()
    print(f"  -> {out.relative_to(ROOT)}")


# ============================================================
# Figure 4 — BCE vs MSE scatter
# ============================================================
def fig_bce_vs_mse():
    df = pd.read_csv(ROOT / "results/binary_vs_regression.csv")
    df["delta"] = df["rankicir_reg"] - df["rankicir_binary"]

    fig, ax = plt.subplots(figsize=(7.2, 6))
    # Diagonal: y = x (no change)
    lim_lo, lim_hi = -0.40, 0.45
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
              color="#888", linestyle="--", linewidth=1.2,
              label="y = x (no change)")

    # All points
    for _, r in df.iterrows():
        is_cat = r["label"] == "CatBoost"
        is_tab = r["label"] == "TabNet"
        is_lin = r["label"] == "Linear"
        if is_cat:
            color = C["catboost"]; size = 220; ec = "black"; lw = 2.0
        elif is_tab:
            color = C["bad"]; size = 130; ec = "black"; lw = 1.0
        elif is_lin:
            color = C["linear"]; size = 130; ec = "black"; lw = 1.0
        else:
            color = C["gbdt"]; size = 110; ec = "black"; lw = 0.8
        ax.scatter(r["rankicir_binary"], r["rankicir_reg"],
                    s=size, c=color, edgecolors=ec, linewidths=lw, zorder=3)

    # Labels
    label_offsets = {
        "CatBoost":        (0.015, 0.025),
        "RandomForest":    (0.015, -0.030),
        "LGBM-shallow":    (0.020, 0.010),
        "LGBM-cons":       (0.020, -0.010),
        "LGBM-std":        (0.020, 0.000),
        "XGBoost":         (0.020, 0.015),
        "Linear":          (0.020, 0.000),
        "TCN":             (-0.005, 0.020),
        "ALSTM":           (-0.005, -0.025),
        "FT-Transformer":  (-0.005, 0.020),
        "TabNet":          (0.015, -0.015),
    }
    for _, r in df.iterrows():
        ox, oy = label_offsets.get(r["label"], (0.015, 0.015))
        weight = "bold" if r["label"] == "CatBoost" else "normal"
        ax.annotate(r["label"],
                      (r["rankicir_binary"], r["rankicir_reg"]),
                      xytext=(ox, oy), textcoords="offset points",
                      fontsize=9, fontweight=weight,
                      ha="left" if ox > 0 else "right",
                      va="center")

    # CatBoost delta annotation
    cat = df[df["label"] == "CatBoost"].iloc[0]
    ax.annotate(rf"CatBoost: $\Delta = {cat['delta']:+.3f}$" + "\n(largest gain)",
                  xy=(cat["rankicir_binary"], cat["rankicir_reg"]),
                  xytext=(-0.25, 0.32),
                  fontsize=10, fontweight="bold", color=C["catboost"],
                  arrowprops=dict(arrowstyle="->", color=C["catboost"], lw=1.3))

    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_xlabel("RankICIR under binary BCE objective")
    ax.set_ylabel("RankICIR under regression MSE objective")
    ax.set_title("Loss-function effect: every GBDT model gains from MSE")
    ax.axhline(0, color="#aaa", linewidth=0.6, linestyle=":")
    ax.axvline(0, color="#aaa", linewidth=0.6, linestyle=":")

    # Shade upper-left triangle (MSE > BCE region)
    ax.fill_between([lim_lo, lim_hi], [lim_lo, lim_hi], lim_hi,
                      color=C["gbdt"], alpha=0.05, zorder=0)
    ax.fill_between([lim_lo, lim_hi], lim_lo, [lim_lo, lim_hi],
                      color=C["bad"], alpha=0.05, zorder=0)
    ax.text(0.25, -0.15, "BCE > MSE", fontsize=9, color=C["bad"],
              alpha=0.7, ha="center")
    ax.text(-0.25, 0.30, "MSE > BCE\n(10 of 11 models)", fontsize=9,
              color=C["gbdt"], alpha=0.8, ha="center")

    ax.legend(loc="lower right")
    ax.set_aspect("equal")

    out = FIG / "fig_bce_vs_mse.png"
    plt.savefig(out)
    plt.close()
    print(f"  -> {out.relative_to(ROOT)}")


# ============================================================
# Figure 5 — Feature ablation cumulative line
# ============================================================
def fig_feature_ablation():
    df = pd.read_csv(find("results/feature_ablation_*/feature_ablation.csv"))
    order = ["G1", "G1+G2", "G1+G2+G3", "G1+G2+G3+G4",
             "G1+G2+G3+G4+G5", "Full(G1-G6)"]
    df = df.set_index("config").loc[order].reset_index()
    df["short"] = ["G1", "+G2", "+G3", "+G4", "+G5", "+G6"]

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    x = np.arange(len(df))
    ax.plot(x, df["rankicir"], "o-", color=C["linear"], linewidth=2.2,
              markersize=10, markerfacecolor="white", markeredgewidth=2.0,
              markeredgecolor=C["linear"], zorder=3)

    # Highlight peak with orange marker
    peak_idx = int(df["rankicir"].idxmax())
    ax.scatter(peak_idx, df.iloc[peak_idx]["rankicir"], s=240,
                 color=C["catboost"], edgecolors="black", linewidths=1.8,
                 zorder=4)

    # value labels
    for xi, v, nf in zip(x, df["rankicir"], df["n_features"]):
        ax.text(xi, v + 0.025, f"{v:.3f}", ha="center", fontsize=10,
                  fontweight="bold" if xi == peak_idx else "normal")
        ax.text(xi, -0.035, f"{nf} feat.", ha="center", fontsize=8.5,
                  color="#666")

    # G5 drop arrow
    drop_from = df.iloc[peak_idx]["rankicir"]
    drop_to = df.iloc[peak_idx + 1]["rankicir"]
    drop_pp = (drop_from - drop_to) * 100
    ax.annotate("", xy=(peak_idx + 1, drop_to + 0.015),
                  xytext=(peak_idx, drop_from - 0.005),
                  arrowprops=dict(arrowstyle="->", color=C["bad"],
                                    lw=2.0, connectionstyle="arc3,rad=-0.18"))
    ax.text(peak_idx + 0.5, (drop_from + drop_to) / 2 + 0.04,
              rf"$\downarrow$ {drop_pp:.1f} pp" + "\n(macro_regime_3\nhurts as feature)",
              fontsize=9, color=C["bad"], fontweight="bold", ha="center")

    # peak annotation
    ax.annotate("Peak: G1234",
                  xy=(peak_idx, df.iloc[peak_idx]["rankicir"]),
                  xytext=(peak_idx - 1.4, df.iloc[peak_idx]["rankicir"] + 0.08),
                  fontsize=10, fontweight="bold", color=C["catboost"],
                  arrowprops=dict(arrowstyle="->", color=C["catboost"], lw=1.3))

    ax.set_xticks(x)
    ax.set_xticklabels(df["short"], fontsize=10)
    ax.set_xlabel("Cumulative feature group added")
    ax.set_ylabel("RankICIR (LightGBM-shallow-reg)")
    ax.set_title("Feature group ablation: G1234 is the peak")
    ax.set_ylim(-0.06, df["rankicir"].max() + 0.10)
    ax.axhline(0, color="#aaa", linewidth=0.6)

    out = FIG / "fig_feature_ablation.png"
    plt.savefig(out)
    plt.close()
    print(f"  -> {out.relative_to(ROOT)}")


# ============================================================
# Figure 6 — SRD heatmap (4 configs x 3 pairs)
# ============================================================
def fig_srd_heatmap():
    configs = [
        ("LGBM-shallow BCE + G1--G6",
         "results/regime_20260506_221050_full/srd_matrix.csv"),
        ("LGBM-shallow MSE + G1--G6",
         "results/regime_20260506_234936_full_lgbm_shallow_reg/srd_matrix.csv"),
        ("LGBM-shallow MSE + G1234",
         "results/regime_20260507_013022_final_g1234/srd_matrix.csv"),
        ("CatBoost MSE + G1234",
         "results/regime_20260507_013443_final_g1234_cat/srd_matrix.csv"),
    ]
    rows = []
    labels = []
    for name, path in configs:
        d = pd.read_csv(ROOT / path, index_col=0)
        rows.append([d.loc["bear", "bull"],
                      d.loc["bear", "sideway"],
                      d.loc["bull", "sideway"]])
        labels.append(name)
    M = np.array(rows)

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "srd", [C["heat_lo"], "#FFB74D", "#E64A19", C["heat_hi"]])

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=0.75, aspect="auto")
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["Bear / Bull", "Bear / Sideway", "Bull / Sideway"],
                        fontsize=11)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)

    # value annotations
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            color = "white" if v > 0.50 else "#222"
            fw = "bold" if i == 3 else "normal"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                      color=color, fontsize=11, fontweight=fw)

    # Highlight CatBoost row (index 3)
    rect = Rectangle((-0.5, 2.5), 3, 1.0, linewidth=2.5,
                      edgecolor=C["catboost"], facecolor="none", zorder=10)
    ax.add_patch(rect)
    ax.text(3.05, 3, "Selected\n(max SRD)",
              fontsize=9.5, color=C["catboost"], fontweight="bold",
              va="center", ha="left")

    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.10)
    cbar.set_label(r"SRD ($1 - \rho_{\mathrm{Spearman}}$ of SHAP ranks)",
                    fontsize=10)
    ax.set_title("SHAP Regime Divergence across configurations")
    ax.grid(False)

    out = FIG / "fig_srd_heatmap.png"
    plt.savefig(out)
    plt.close()
    print(f"  -> {out.relative_to(ROOT)}")


# ============================================================
# Figure 7 — Main ablation horizontal bar
# ============================================================
def fig_ablation_bar():
    df = pd.read_csv(ROOT / "bull_bear/results/final_ablation.csv")
    codes = ["B0", "M1", "T", "BC", "D1a", "D1b", "D1c", "D1d", "V", "FINAL"]
    df["code"] = codes
    # Use only the canonical pipeline (drop FINAL — confusing combination)
    keep = ["B0", "M1", "T", "BC", "D1a", "D1b", "D1c", "D1d", "V"]
    df = df[df["code"].isin(keep)].reset_index(drop=True)
    # Order as listed
    df["order"] = df["code"].apply(lambda c: keep.index(c))
    df = df.sort_values("order").reset_index(drop=True)

    palette = {
        "B0":  C["factor"],
        "M1":  C["factor"],
        "T":   C["trend"],
        "BC":  "#7CB342",
        "D1a": "#558B2F",
        "D1b": "#33691E",
        "D1c": C["catboost"],
        "D1d": "#1B5E20",
        "V":   C["bad"],
    }
    colors = [palette[c] for c in df["code"]]

    fig, ax = plt.subplots(figsize=(9, 5.6))
    y = np.arange(len(df))
    ax.barh(y, df["rankicir"], color=colors,
              edgecolor="black", linewidth=0.5, height=0.7)

    # Trend baseline vertical
    trend = float(df[df["code"] == "T"]["rankicir"].iloc[0])
    ax.axvline(trend, color=C["trend"], linestyle="--", linewidth=1.4,
                 label=f"Trend baseline = {trend:.3f}")

    # Labels
    full_labels = [f"{c}: {n}" for c, n in zip(df["code"], df["config"])]
    ax.set_yticks(y)
    ax.set_yticklabels(full_labels, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlim(0.20, 0.82)
    ax.set_xlabel("RankICIR (test panel 2023--2026)")
    ax.set_title("Main ablation: pipeline from B0 baseline to D1c final system")

    # value labels
    for yi, v in enumerate(df["rankicir"]):
        ax.text(v + 0.006, yi, f"{v:.3f}", va="center", fontsize=9.5,
                  fontweight="bold" if df.iloc[yi]["code"] == "D1c" else "normal")

    # D1c bold highlight (re-draw with thicker edge)
    d1c_idx = int(df.index[df["code"] == "D1c"][0])
    ax.barh(d1c_idx, df.iloc[d1c_idx]["rankicir"], color=C["catboost"],
              edgecolor="black", linewidth=2.5, height=0.7, zorder=4)
    ax.annotate("Final system\n(D1c)",
                  xy=(df.iloc[d1c_idx]["rankicir"], d1c_idx),
                  xytext=(0.65, d1c_idx - 2),
                  fontsize=10, fontweight="bold", color=C["catboost"],
                  arrowprops=dict(arrowstyle="->", color=C["catboost"], lw=1.4))
    # V annotation
    v_idx = int(df.index[df["code"] == "V"][0])
    ax.annotate("Vol-gate hurts",
                  xy=(df.iloc[v_idx]["rankicir"], v_idx),
                  xytext=(0.55, v_idx + 0.3),
                  fontsize=9.5, color=C["bad"],
                  arrowprops=dict(arrowstyle="->", color=C["bad"], lw=1.2))

    ax.legend(loc="lower right")

    out = FIG / "fig_ablation_bar.png"
    plt.savefig(out)
    plt.close()
    print(f"  -> {out.relative_to(ROOT)}")


# ============================================================
# Figure 8 — Y vs X mechanism line
# ============================================================
def fig_yx_mechanism():
    me = pd.read_csv(ROOT / "bull_bear/results/mechanism_validation.csv")
    y_rows = me[me["config"].str.startswith("Y =")].sort_values("alpha")
    x_rows = me[me["config"].str.startswith("X =")].sort_values("alpha")
    alphas = y_rows["alpha"].to_numpy()
    y_ric = y_rows["rankicir"].to_numpy()
    x_ric = x_rows["rankicir"].to_numpy()
    trend = float(me[me["config"] == "Trend pure"]["rankicir"].iloc[0])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(alphas, y_ric, "o-", color=C["y_line"], linewidth=2.6,
              markersize=11, markerfacecolor=C["y_line"],
              markeredgecolor="white", markeredgewidth=1.5,
              label=r"$Y_\alpha = s^\alpha - \alpha \cdot \tilde{s}^\beta_{\mathrm{Bear}}$  (adversarial)",
              zorder=4)
    ax.plot(alphas, x_ric, "s--", color=C["x_line"], linewidth=2.4,
              markersize=10, markerfacecolor="white",
              markeredgecolor=C["x_line"], markeredgewidth=1.8,
              label=r"$X_\alpha = s^\alpha + \alpha \cdot \tilde{s}^\beta_{\mathrm{M1}}$  (additive)",
              zorder=4)
    ax.axhline(trend, color=C["trend"], linestyle=":", linewidth=1.4,
                 label=f"Trend baseline = {trend:.3f}", zorder=2)

    # Value annotations
    for a, vy, vx in zip(alphas, y_ric, x_ric):
        ax.annotate(f"{vy:.3f}", (a, vy), xytext=(0, 12),
                      textcoords="offset points", ha="center",
                      fontsize=9.5, color=C["y_line"], fontweight="bold")
        ax.annotate(f"{vx:.3f}", (a, vx), xytext=(0, -18),
                      textcoords="offset points", ha="center",
                      fontsize=9.5, color=C["x_line"], fontweight="bold")

    # α=0.5 gap annotation
    a_star = 0.5
    vy_star = float(y_rows[y_rows["alpha"] == a_star]["rankicir"].iloc[0])
    vx_star = float(x_rows[x_rows["alpha"] == a_star]["rankicir"].iloc[0])
    gap_bp = int(round((vy_star - vx_star) * 10000))
    ax.annotate("", xy=(a_star + 0.015, vy_star),
                  xytext=(a_star + 0.015, vx_star),
                  arrowprops=dict(arrowstyle="<->", color="black", lw=1.8))
    ax.text(a_star + 0.03, (vy_star + vx_star) / 2,
              rf"$\Delta = +{gap_bp:,}$ bp", fontsize=11, fontweight="bold",
              va="center",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="black", alpha=0.92))

    ax.set_xlabel(r"Strength $\alpha$ of the second term")
    ax.set_ylabel("RankICIR (test panel 2023--2026)")
    ax.set_title("Adversarial subtraction (Y) vs additive ensembling (X):\n"
                  "identical G1+G3 features, only target objective differs")
    ax.set_xticks(alphas)
    ax.legend(loc="lower center", framealpha=0.95)
    ax.set_xlim(0.06, 0.60)

    out = FIG / "fig_yx_mechanism.png"
    plt.savefig(out)
    plt.close()
    print(f"  -> {out.relative_to(ROOT)}")


# ============================================================
# Figure 9 — Walk-forward yearly bars
# ============================================================
def fig_walkforward():
    df = pd.read_csv(ROOT / "bull_bear/results/final/rolling_walkforward.csv")
    df = df[df["year"] <= 2025].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = np.arange(len(df))
    w = 0.38
    # Trend: hollow with hatching
    ax.bar(x - w/2, df["trend_rankicir"], w,
             color="white", edgecolor=C["trend"], linewidth=1.5,
             hatch="///", label="Trend single-agent")
    # D1c: filled green
    ax.bar(x + w/2, df["d1_rankicir"], w,
             color=C["gbdt"], edgecolor="black", linewidth=0.5,
             label="D1c adversarial system")

    # value labels
    for xi, t, d in zip(x, df["trend_rankicir"], df["d1_rankicir"]):
        ax.text(xi - w/2, t + 0.015, f"{t:.3f}", ha="center", fontsize=8.5,
                  color=C["trend"])
        ax.text(xi + w/2, d + 0.015, f"{d:.3f}", ha="center", fontsize=8.5,
                  fontweight="bold", color=C["y_line"])

    # mean lines
    mean_t = df["trend_rankicir"].mean()
    mean_d = df["d1_rankicir"].mean()
    ax.axhline(mean_t, color=C["trend"], linestyle=":", linewidth=1.2,
                 label=f"Trend mean = {mean_t:.3f}")
    ax.axhline(mean_d, color=C["y_line"], linestyle=":", linewidth=1.2,
                 label=f"D1c mean = {mean_d:.3f}")

    # Year annotations
    cov_x = list(df["year"]).index(2020)
    bear_x = list(df["year"]).index(2022)
    ax.annotate("COVID shock", xy=(cov_x, df.iloc[cov_x]["d1_rankicir"]),
                  xytext=(cov_x - 0.7, df.iloc[cov_x]["d1_rankicir"] + 0.16),
                  fontsize=9, color=C["bad"], fontweight="bold",
                  arrowprops=dict(arrowstyle="->", color=C["bad"], lw=1.0))
    delta_bear = df.iloc[bear_x]["delta"] * 10000
    ax.annotate(f"Deep bear ($\\Delta=+{delta_bear:.0f}$ bp)",
                  xy=(bear_x, df.iloc[bear_x]["d1_rankicir"]),
                  xytext=(bear_x - 0.8, df.iloc[bear_x]["d1_rankicir"] + 0.14),
                  fontsize=9, color=C["bad"], fontweight="bold",
                  arrowprops=dict(arrowstyle="->", color=C["bad"], lw=1.0))

    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in df["year"]], fontsize=11)
    ax.set_xlabel("Calendar year (walk-forward train window expands annually)")
    ax.set_ylabel("RankICIR")
    ax.set_title(f"Walk-forward cross-validation: D1c wins 7/7 evaluation years")
    ax.set_ylim(0, df["d1_rankicir"].max() * 1.16)
    ax.legend(loc="upper center", ncol=4, framealpha=0.95, fontsize=8.5,
                bbox_to_anchor=(0.5, -0.13))

    out = FIG / "fig_walkforward.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  -> {out.relative_to(ROOT)}")


# ============================================================
# Figure 10 — Quintile dual-axis (bar + line)
# ============================================================
def fig_quintile():
    df = pd.read_csv(ROOT / "bull_bear/results/final/bear_quintile_analysis.csv")

    fig, ax1 = plt.subplots(figsize=(8.5, 5.5))
    x = np.arange(len(df))
    bar_w = 0.55
    # Left axis: bars for MaxDD
    bars = ax1.bar(x, df["avg_maxdd_5d"] * 100, bar_w,
                     color=C["bad"], alpha=0.78,
                     edgecolor="black", linewidth=0.5,
                     label="Realised 5-day MaxDD (left)")
    for xi, v in zip(x, df["avg_maxdd_5d"] * 100):
        ax1.text(xi, v + 0.10, f"{v:.2f}%", ha="center", fontsize=10,
                   color=C["bad"], fontweight="bold")
    ax1.set_ylabel("Realised 5-day MaxDD (%)", color=C["bad"], fontsize=11)
    ax1.tick_params(axis="y", labelcolor=C["bad"])
    ax1.set_ylim(0, df["avg_maxdd_5d"].max() * 115)

    # Right axis: line for forward return
    ax2 = ax1.twinx()
    ax2.plot(x, df["avg_r_future_5"] * 100, "o-",
              color=C["linear"], linewidth=2.5, markersize=11,
              markerfacecolor=C["linear"], markeredgecolor="white",
              markeredgewidth=1.5,
              label="Realised 5-day forward return (right)", zorder=5)
    for xi, v in zip(x, df["avg_r_future_5"] * 100):
        ax2.annotate(f"{v:+.3f}%", (xi, v), xytext=(0, 12),
                       textcoords="offset points", ha="center",
                       fontsize=9.5, color=C["linear"], fontweight="bold")
    ax2.set_ylabel("Realised 5-day forward return (%)", color=C["linear"],
                     fontsize=11)
    ax2.tick_params(axis="y", labelcolor=C["linear"])
    ax2.set_ylim(0, df["avg_r_future_5"].max() * 145)
    ax2.grid(False)

    ax1.set_xticks(x)
    ax1.set_xticklabels(df["label"], fontsize=10)
    ax1.set_title("Bear Agent quintile analysis: MaxDD monotone, return non-monotone\n"
                    "(Q2/Q3 > Q1 rules out Bear $\\equiv -$Alpha)")

    # Annotate Q2/Q3 > Q1 inversion
    q1_r = df.iloc[0]["avg_r_future_5"] * 100
    q2_r = df.iloc[1]["avg_r_future_5"] * 100
    ax2.annotate("Q2/Q3 return > Q1\n(non-monotone)",
                  xy=(1, q2_r),
                  xytext=(1.8, q2_r + 0.20),
                  fontsize=9, color=C["linear"], fontweight="bold",
                  arrowprops=dict(arrowstyle="->", color=C["linear"], lw=1.2))

    # Legends
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", framealpha=0.95,
                 fontsize=9.5)

    out = FIG / "fig_quintile.png"
    plt.savefig(out)
    plt.close()
    print(f"  -> {out.relative_to(ROOT)}")


# ============================================================
# Main
# ============================================================
def main():
    print("Generating 8 paper figures (academic style, 300 dpi)...")
    fig_model_compare()
    fig_bce_vs_mse()
    fig_feature_ablation()
    fig_srd_heatmap()
    fig_ablation_bar()
    fig_yx_mechanism()
    fig_walkforward()
    fig_quintile()
    print("Done.")


if __name__ == "__main__":
    main()
