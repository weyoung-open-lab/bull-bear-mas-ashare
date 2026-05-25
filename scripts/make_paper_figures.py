"""Generate paper figures.

Outputs:
  figure/yx_mechanism.png   - Y vs X line plot
  figure/framework.png      - System architecture
                              ASCII-style matplotlib draft;
                              user should redraw in Inkscape/TikZ
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figure"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def fig_yx_mechanism() -> None:
    """Figure 2: Y vs X mechanism proof, from mechanism_validation.csv."""
    csv = ROOT / "bull_bear" / "results" / "mechanism_validation.csv"
    df = pd.read_csv(csv)

    # Extract per-alpha rows (X = additive, Y = adversarial)
    x_rows = df[df["config"].str.startswith("X = ")].sort_values("alpha")
    y_rows = df[df["config"].str.startswith("Y = ")].sort_values("alpha")
    trend_pure = float(df[df["config"].str.contains("Trend pure", na=False)]["rankicir"].iloc[0])

    alphas = y_rows["alpha"].to_numpy()
    y_ric = y_rows["rankicir"].to_numpy()
    x_ric = x_rows["rankicir"].to_numpy()

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(alphas, y_ric, "o-", color="#1b5e20", linewidth=2.4, markersize=9,
             label=r"$Y_\alpha = s^\alpha - \alpha \cdot \tilde{s}^\beta_{\mathrm{Bear}}$  (adversarial)")
    ax.plot(alphas, x_ric, "s--", color="#c62828", linewidth=2.4, markersize=9,
             label=r"$X_\alpha = s^\alpha + \alpha \cdot \tilde{s}^\beta_{\mathrm{M1}}$  (additive)")
    ax.axhline(trend_pure, color="gray", linestyle=":", linewidth=1.2,
                label=f"Trend pure baseline = {trend_pure:.3f}")

    for a, y, x in zip(alphas, y_ric, x_ric):
        ax.annotate(f"{y:.3f}", (a, y), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=8.5, color="#1b5e20")
        ax.annotate(f"{x:.3f}", (a, x), textcoords="offset points",
                     xytext=(0, -16), ha="center", fontsize=8.5, color="#c62828")

    # Highlight α=0.5 gap
    a_star = 0.5
    y_star = float(y_rows[y_rows["alpha"] == a_star]["rankicir"].iloc[0])
    x_star = float(x_rows[x_rows["alpha"] == a_star]["rankicir"].iloc[0])
    gap_bp = int(round((y_star - x_star) * 10000))
    ax.annotate("", xy=(a_star + 0.012, y_star), xytext=(a_star + 0.012, x_star),
                  arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
    ax.text(a_star + 0.025, (y_star + x_star) / 2,
              rf"$\Delta = +{gap_bp:,}$ bp",
              fontsize=10, fontweight="bold", va="center")

    ax.set_xlabel(r"Subtraction / addition strength  $\alpha$", fontsize=11)
    ax.set_ylabel("RankICIR (test panel 2023--2026)", fontsize=11)
    ax.set_title("Adversarial subtraction ($Y$) vs additive ensembling ($X$)\n"
                  "(identical G1+G3 features; only the target objective differs)",
                  fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9.5)
    ax.set_xlim(0.05, 0.6)

    plt.tight_layout()
    out = FIG_DIR / "yx_mechanism.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> {out.relative_to(ROOT)}")


def fig_framework() -> None:
    """Figure 1: System architecture (matplotlib draft).

    The user is encouraged to redraw this in Inkscape or TikZ for the
    final submission; this provides the layout and labels to copy.
    """
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def box(x, y, w, h, text, color, fontsize=9.5, lw=1.2, edgecolor="black"):
        b = FancyBboxPatch((x, y), w, h,
                            boxstyle="round,pad=0.4",
                            linewidth=lw, edgecolor=edgecolor,
                            facecolor=color)
        ax.add_patch(b)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                 fontsize=fontsize, wrap=True)

    def arrow(x1, y1, x2, y2, color="black", style="->"):
        a = FancyArrowPatch((x1, y1), (x2, y2),
                              arrowstyle=style, mutation_scale=15,
                              color=color, lw=1.3)
        ax.add_patch(a)

    # Layer banners
    ax.text(1, 95, "Layer 1: Features", fontsize=10, fontweight="bold", color="#0d47a1")
    ax.text(1, 73, "Layer 2: Agents",   fontsize=10, fontweight="bold", color="#0d47a1")
    ax.text(1, 47, "Layer 3: Arbitration", fontsize=10, fontweight="bold", color="#0d47a1")
    ax.text(1, 22, "Layer 4: Risk Control", fontsize=10, fontweight="bold", color="#0d47a1")

    # Layer 1
    box(35, 85, 30, 8,
        "Stock-day features\n$\\mathbf{x}_{i,t} \\in \\mathbb{R}^{17}$  (G1--G4)",
        "#e3f2fd")

    # Layer 2: Alpha + Bear
    box(15, 65, 30, 12,
        "Alpha Agent\nCatBoost (G4 trend)\ntarget: $r^{(5d)}_{i,t}$\n$\\rightarrow s^\\alpha_{i,t}$",
        "#c8e6c9")
    box(55, 65, 30, 12,
        "Bear Agent\nCatBoost (G1+G3)\ntarget: max-drawdown $D_{i,t}$\n$\\rightarrow s^\\beta_{i,t}$",
        "#ffcdd2")
    arrow(50, 85, 30, 77)
    arrow(50, 85, 70, 77)

    # Layer 3: Arbitration
    box(25, 40, 50, 11,
        "Adversarial Arbitration\n$c_{i,t} = s^\\alpha_{i,t} - \\alpha(t)\\cdot\\tilde{s}^\\beta_{i,t}$\n"
        "(cross-section $z$-scoring)",
        "#fff9c4", fontsize=10)
    arrow(30, 65, 35, 51)
    arrow(70, 65, 65, 51)

    # Layer 4: Regime + Anomaly
    box(8, 13, 32, 11,
        "Regime Agent\n"
        "$\\alpha(t) = 0.5 + 0.15 P_{\\mathrm{bear}}(t) - 0.15 P_{\\mathrm{bull}}(t)$\n"
        "(range $[0.35, 0.65]$)",
        "#d1c4e9", fontsize=8.5)
    box(60, 13, 32, 11,
        "Anomaly Agent\n"
        "Mahalanobis $D_M(t)$\n"
        "fallback to Alpha if $D_M > \\tau_{99}$",
        "#ffe0b2", fontsize=8.5)
    arrow(25, 24, 35, 40)
    arrow(75, 24, 65, 40)

    # Final output
    box(35, 0, 30, 8,
        "Top-5\\% selection, 5-day hold",
        "#cfd8dc")
    arrow(50, 40, 50, 8)

    plt.tight_layout()
    out = FIG_DIR / "framework.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> {out.relative_to(ROOT)}")


def main() -> None:
    print("Generating paper figures ...")
    fig_yx_mechanism()
    fig_framework()
    print("Done.")


if __name__ == "__main__":
    main()
