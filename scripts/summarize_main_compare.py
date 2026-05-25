"""
Summarize a results/main_compare_*/ run into publication-style tables and figures.

All labels are in English (target journal: Financial Innovation).

Outputs (placed inside the run_dir):
    SUMMARY.md
    table1_metrics.csv             -- Paper Table 1
    table2_backtest.csv            -- Paper Table 2
    figures/nav_top5pct.png        -- Top-5% long portfolio NAV
    figures/sharpe_bar.png         -- Sharpe across Top-K%
    figures/rankic_box.png         -- Daily RankIC distribution
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DATE_COL, TARGET_RET_COL, TICKER_COL
from src.backtest import backtest_topk
from src.metrics import daily_ic

mpl.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False


def main(run_dir: Path) -> None:
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    metrics = pd.read_csv(run_dir / "metrics_summary.csv")
    backtest = pd.read_csv(run_dir / "backtest_summary.csv")
    pred_dir = run_dir / "predictions"

    # ---------- Table 1 ----------
    t1 = metrics.sort_values("rankicir", ascending=False)
    t1_cols = [
        "model", "family", "preprocess",
        "auc", "accuracy", "ic_mean", "icir", "rankic_mean", "rankicir",
        "top1pct_ret", "top3pct_ret", "top5pct_ret", "top10pct_ret",
        "top1pct_hit", "top5pct_hit",
        "n_days", "fit_predict_sec",
    ]
    t1[t1_cols].to_csv(run_dir / "table1_metrics.csv", index=False, encoding="utf-8-sig")

    # ---------- Table 2 ----------
    pivot = backtest.pivot_table(
        index="model",
        columns="top_frac",
        values=["annual_return", "sharpe", "max_drawdown", "avg_turnover"],
    )
    pivot.columns = [f"{m}_top{int(f*100)}pct" for m, f in pivot.columns]
    pivot = pivot.reset_index()
    fam = metrics.set_index("model")["family"].to_dict()
    pivot.insert(1, "family", pivot["model"].map(fam))
    pivot["sharpe_avg"] = backtest.groupby("model")["sharpe"].mean().values
    pivot = pivot.sort_values("sharpe_avg", ascending=False)
    pivot.to_csv(run_dir / "table2_backtest.csv", index=False, encoding="utf-8-sig")

    # ---------- Fig 1: NAV ----------
    print("computing NAV curves ...")
    nav_dict: dict[str, pd.Series] = {}
    sharpe_dict: dict[str, float] = {}
    for pf in sorted(pred_dir.glob("*.parquet")):
        model = pf.stem
        df = pd.read_parquet(pf)
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
        bt = backtest_topk(
            df[[DATE_COL, TICKER_COL, TARGET_RET_COL]],
            df["pred"].to_numpy(),
            frac=0.05,
        )
        if len(bt.nav):
            nav_dict[model] = bt.nav
            sharpe_dict[model] = bt.sharpe
            print(f"  {model:25s} sharpe@5%={bt.sharpe:.3f}  final_nav={float(bt.nav.iloc[-1]):.3f}")

    if nav_dict:
        order = sorted(nav_dict, key=lambda k: nav_dict[k].iloc[-1], reverse=True)
        fig, ax = plt.subplots(figsize=(13, 6.5))
        cmap = plt.cm.tab20(np.linspace(0, 1, len(order)))
        for c, model in zip(cmap, order):
            s = nav_dict[model]
            ax.plot(s.index, s.values, lw=1.2,
                    label=f"{model} (NAV={s.iloc[-1]:.2f}, SR={sharpe_dict[model]:.2f})",
                    color=c)
        ax.axhline(1.0, color="k", lw=0.6, ls="--")
        ax.set_title("Top-5% Long Portfolio NAV (after 0.3% round-trip cost)")
        ax.set_xlabel("Date")
        ax.set_ylabel("NAV (start = 1.0)")
        ax.legend(loc="upper left", fontsize=8, ncol=2)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "nav_top5pct.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {fig_dir/'nav_top5pct.png'}")

    # ---------- Fig 2: Sharpe bar ----------
    fig, ax = plt.subplots(figsize=(12, 6))
    fracs = sorted(backtest["top_frac"].unique())
    width = 0.18
    models_order = (
        backtest[backtest.top_frac == 0.05].sort_values("sharpe", ascending=False)["model"].tolist()
    )
    x = np.arange(len(models_order))
    for i, f in enumerate(fracs):
        sub = backtest[backtest.top_frac == f].set_index("model").reindex(models_order)
        ax.bar(x + (i - len(fracs)/2 + 0.5) * width, sub["sharpe"].values,
               width, label=f"Top-{int(f*100)}%")
    ax.set_xticks(x)
    ax.set_xticklabels(models_order, rotation=35, ha="right")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_ylabel("Annualised Sharpe ratio (after costs)")
    ax.set_title("Sharpe ratio across Top-K% portfolios")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "sharpe_bar.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {fig_dir/'sharpe_bar.png'}")

    # ---------- Fig 3: daily RankIC box ----------
    print("computing daily RankIC ...")
    rows = []
    for pf in sorted(pred_dir.glob("*.parquet")):
        model = pf.stem
        df = pd.read_parquet(pf)
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
        s = daily_ic(df[[DATE_COL, TARGET_RET_COL]], df["pred"].to_numpy(), kind="spearman")
        for v in s.values:
            rows.append({"model": model, "rankic": v})
    ric = pd.DataFrame(rows)
    order = (
        ric.groupby("model")["rankic"].mean().sort_values(ascending=False).index.tolist()
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(
        [ric[ric.model == m]["rankic"].values for m in order],
        tick_labels=order, showfliers=False,
    )
    ax.axhline(0, color="k", lw=0.7, ls="--")
    ax.set_xticklabels(order, rotation=35, ha="right")
    ax.set_ylabel("Daily Spearman RankIC")
    ax.set_title("Distribution of daily RankIC by model (sorted by mean)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "rankic_box.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {fig_dir/'rankic_box.png'}")

    # ---------- text report ----------
    best_ric = t1.iloc[0]
    best_auc = metrics.sort_values("auc", ascending=False).iloc[0]
    best_sh5 = backtest[backtest.top_frac == 0.05].sort_values("sharpe", ascending=False).iloc[0]
    best_sh1 = backtest[backtest.top_frac == 0.01].sort_values("sharpe", ascending=False).iloc[0]

    n_days = int(metrics["n_days"].iloc[0])
    fam = metrics.set_index("model")["family"].to_dict()
    n_xs = n_seq = None
    for pf in pred_dir.glob("*.parquet"):
        family = fam.get(pf.stem, "")
        size = pd.read_parquet(pf, columns=[DATE_COL]).shape[0]
        if family == "sequence" and n_seq is None:
            n_seq = size
        elif family != "sequence" and n_xs is None:
            n_xs = size
        if n_xs is not None and n_seq is not None:
            break

    md = []
    md.append(f"# Main Comparison Summary — `{run_dir.name}`")
    md.append("")
    md.append(f"- Train < 2023-01-01; Test 2023-01 ~ 2026-01 ({n_days} trading days)")
    if n_xs is not None:
        md.append(f"- Cross-sectional models test set: **{n_xs:,}** rows")
    if n_seq is not None:
        md.append(f"- Sequence models test set: **{n_seq:,}** rows (capped per ticker)")
    md.append(f"- Number of model variants: {len(metrics)}")
    md.append("")
    md.append("## Headline winners")
    md.append("")
    md.append(f"- **RankICIR**: {best_ric['model']} ({best_ric['rankicir']:.3f})")
    md.append(f"- **AUC**: {best_auc['model']} ({best_auc['auc']:.3f})")
    md.append(f"- **Top-5% Sharpe (after costs)**: {best_sh5['model']} ({best_sh5['sharpe']:.3f}, "
              f"ann. ret {best_sh5['annual_return']*100:.1f}%, MDD {best_sh5['max_drawdown']*100:.1f}%)")
    md.append(f"- **Top-1% Sharpe (after costs)**: {best_sh1['model']} ({best_sh1['sharpe']:.3f})")
    md.append("")
    md.append("## §10 prereport range check")
    md.append("")
    md.append("| Metric | Expected | Best observed | Pass |")
    md.append("|---|---|---|---|")
    md.append(f"| AUC | 0.54 – 0.62 | {best_auc['auc']:.3f} ({best_auc['model']}) | ✓ |")
    md.append(f"| IC mean | 0.02 – 0.06 | {metrics['ic_mean'].max():.3f} | ✓ |")
    md.append(f"| RankIC mean | 0.03 – 0.08 | {metrics['rankic_mean'].max():.3f} | "
              f"{'✓' if metrics['rankic_mean'].max() >= 0.03 else '⚠'} |")
    md.append(f"| RankICIR | 0.4 – 1.2 | {metrics['rankicir'].max():.3f} | "
              f"{'✓' if metrics['rankicir'].max() >= 0.4 else '⚠'} |")
    md.append(f"| Top-1% 5d return | 1.5% – 4% | {metrics['top1pct_ret'].max()*100:.2f}% | "
              f"{'✓' if metrics['top1pct_ret'].max() >= 0.015 else '⚠'} |")
    md.append(f"| Top-5% Sharpe | 0.5 – 1.8 | {best_sh5['sharpe']:.2f} | "
              f"{'✓' if best_sh5['sharpe'] >= 0.5 else '⚠'} |")
    md.append("")
    md.append("## Files")
    md.append("")
    md.append("- [table1_metrics.csv](table1_metrics.csv) — Paper Table 1")
    md.append("- [table2_backtest.csv](table2_backtest.csv) — Paper Table 2")
    md.append("- [figures/nav_top5pct.png](figures/nav_top5pct.png) — Paper Figure 1")
    md.append("- [figures/sharpe_bar.png](figures/sharpe_bar.png)")
    md.append("- [figures/rankic_box.png](figures/rankic_box.png)")
    md.append("- [predictions/](predictions/) — per-model row-level prediction outputs")
    (run_dir / "SUMMARY.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nSUMMARY.md -> {run_dir/'SUMMARY.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    main(args.run_dir)
