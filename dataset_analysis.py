"""
数据集探索性分析。

读取 config.DATASET_PATH，导出关键统计信息与图表到 config.ANALYSIS_OUTPUT_DIR。
直接运行：python dataset_analysis.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns

from config import (
    ANALYSIS_OUTPUT_DIR,
    DATASET_PATH,
    DATE_COL,
    ID_COLS,
    INDUSTRY_COL,
    REGIME_COL,
    TARGET_COL,
    TICKER_COL,
)

warnings.filterwarnings("ignore")

# Windows 中文字体
mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
mpl.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font="Microsoft YaHei")

PLOT_DIR = ANALYSIS_OUTPUT_DIR / "plots"
TABLE_DIR = ANALYSIS_OUTPUT_DIR / "tables"


def savefig(fig: plt.Figure, name: str) -> None:
    path = PLOT_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {path.relative_to(ANALYSIS_OUTPUT_DIR.parent)}")


def save_table(df: pd.DataFrame, name: str) -> None:
    path = TABLE_DIR / f"{name}.csv"
    df.to_csv(path, encoding="utf-8-sig")
    print(f"  saved -> {path.relative_to(ANALYSIS_OUTPUT_DIR.parent)}")


# ---------------------------------------------------------------- analyses

def overview(df: pd.DataFrame) -> dict:
    info = {
        "rows": len(df),
        "cols": df.shape[1],
        "tickers": df[TICKER_COL].nunique(),
        "dates": df[DATE_COL].nunique(),
        "date_min": df[DATE_COL].min(),
        "date_max": df[DATE_COL].max(),
        "industries": df[INDUSTRY_COL].nunique(),
        "memory_gb": df.memory_usage(deep=True).sum() / 1024**3,
    }
    return info


def missing_analysis(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().sum()
    pct = miss / len(df) * 100
    out = pd.DataFrame({"missing": miss, "missing_pct": pct}).sort_values(
        "missing_pct", ascending=False
    )
    return out


def plot_missing(missing: pd.DataFrame) -> None:
    nz = missing[missing["missing"] > 0]
    if nz.empty:
        return
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(nz))))
    nz["missing_pct"].plot.barh(ax=ax, color="#d9534f")
    ax.invert_yaxis()
    ax.set_xlabel("缺失比例 (%)")
    ax.set_title("各列缺失值占比")
    savefig(fig, "01_missing_values")


def plot_samples_per_day(df: pd.DataFrame) -> None:
    s = df.groupby(DATE_COL).size()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(s.index, s.values, lw=0.8, color="#1f77b4")
    ax.set_title("每个交易日的样本数（覆盖股票数量）")
    ax.set_xlabel("日期")
    ax.set_ylabel("样本数")
    savefig(fig, "02_samples_per_day")


def plot_target_distribution(df: pd.DataFrame) -> None:
    s = df[TARGET_COL].dropna()
    clipped = s.clip(s.quantile(0.001), s.quantile(0.999))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].hist(clipped, bins=120, color="#2ca02c", alpha=0.85)
    axes[0].axvline(0, color="k", lw=0.8, ls="--")
    axes[0].set_title(f"{TARGET_COL} 分布（裁剪到 0.1%~99.9% 分位）")
    axes[0].set_xlabel("未来 5 日收益")

    axes[1].boxplot(clipped, vert=False, showfliers=False)
    axes[1].set_title(f"{TARGET_COL} 箱线图")
    savefig(fig, "03_target_distribution")


def plot_target_by_regime(df: pd.DataFrame) -> None:
    sub = df[[TARGET_COL, REGIME_COL]].dropna()
    order = ["bear", "sideway", "bull"]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=sub, x=REGIME_COL, y=TARGET_COL, order=order,
        showfliers=False, ax=ax, palette="Set2",
    )
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_title("不同宏观状态下 r_future_5 的分布")
    savefig(fig, "04_target_by_regime")


def plot_regime_timeline(df: pd.DataFrame) -> None:
    daily = (
        df.groupby([DATE_COL, REGIME_COL]).size().unstack(fill_value=0)
    )
    daily_pct = daily.div(daily.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(12, 4))
    daily_pct[["bear", "sideway", "bull"]].plot.area(
        ax=ax, color=["#d62728", "#7f7f7f", "#2ca02c"], alpha=0.85
    )
    ax.set_title("宏观状态分布随时间演变")
    ax.set_ylabel("占比")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", ncol=3)
    savefig(fig, "05_regime_timeline")


def plot_industry_top(df: pd.DataFrame, top_n: int = 25) -> None:
    s = df[INDUSTRY_COL].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(9, 7))
    s[::-1].plot.barh(ax=ax, color="#4c72b0")
    ax.set_title(f"行业样本量 Top {top_n}")
    ax.set_xlabel("样本数")
    savefig(fig, "06_industry_top")


def plot_industry_target(df: pd.DataFrame, top_n: int = 25) -> None:
    top_inds = df[INDUSTRY_COL].value_counts().head(top_n).index
    sub = df[df[INDUSTRY_COL].isin(top_inds)][[INDUSTRY_COL, TARGET_COL]].dropna()
    order = (
        sub.groupby(INDUSTRY_COL)[TARGET_COL].median().sort_values().index.tolist()
    )
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.boxplot(
        data=sub, x=TARGET_COL, y=INDUSTRY_COL, order=order,
        showfliers=False, ax=ax, palette="vlag",
    )
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_title(f"行业 Top {top_n} 下 r_future_5 分布（按中位数排序）")
    savefig(fig, "07_industry_target")


def plot_correlation(df: pd.DataFrame, sample: int = 300_000) -> pd.DataFrame:
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != TARGET_COL]
    cols = num_cols + [TARGET_COL]
    n = min(sample, len(df))
    sub = df[cols].sample(n=n, random_state=0).dropna()
    corr = sub.corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        annot=False, square=True, linewidths=0.3, cbar_kws={"shrink": 0.7}, ax=ax,
    )
    ax.set_title(f"特征相关性热力图（样本 {n:,}）")
    savefig(fig, "08_correlation_heatmap")
    return corr


def plot_target_correlation(corr: pd.DataFrame) -> pd.Series:
    s = corr[TARGET_COL].drop(TARGET_COL).sort_values()
    fig, ax = plt.subplots(figsize=(9, max(4, 0.3 * len(s))))
    colors = ["#d62728" if v < 0 else "#2ca02c" for v in s.values]
    ax.barh(s.index, s.values, color=colors)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel(f"与 {TARGET_COL} 的皮尔逊相关")
    ax.set_title(f"特征与 {TARGET_COL} 的相关性")
    savefig(fig, "09_feature_target_correlation")
    return s


def plot_feature_distributions(df: pd.DataFrame, sample: int = 200_000) -> None:
    feats = [
        "ret_1d", "ret_5d", "ret_10d", "roc_20",
        "ema30_slope", "ema180_slope", "bias_60",
        "board_rank_20d_pct", "micro_sentiment_ema5",
        "trend60", "breadth_mom", "vol20",
    ]
    feats = [c for c in feats if c in df.columns]
    n = min(sample, len(df))
    sub = df[feats].sample(n=n, random_state=0).dropna()

    cols = 3
    rows = int(np.ceil(len(feats) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13, 3.2 * rows))
    for ax, c in zip(axes.flat, feats):
        v = sub[c]
        v = v.clip(v.quantile(0.005), v.quantile(0.995))
        ax.hist(v, bins=80, color="#1f77b4", alpha=0.85)
        ax.set_title(c)
    for ax in axes.flat[len(feats):]:
        ax.axis("off")
    fig.suptitle(f"主要特征分布（样本 {n:,}）", y=1.0)
    savefig(fig, "10_feature_distributions")


def plot_target_over_time(df: pd.DataFrame) -> None:
    daily = df.groupby(DATE_COL)[TARGET_COL].mean().dropna()
    rolling = daily.rolling(20, min_periods=5).mean()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(daily.index, daily.values, lw=0.5, alpha=0.4, label="日均", color="#1f77b4")
    ax.plot(rolling.index, rolling.values, lw=1.5, label="20日滚动均值", color="#d62728")
    ax.axhline(0, color="k", lw=0.6, ls="--")
    ax.set_title(f"全市场 {TARGET_COL} 横截面均值随时间变化")
    ax.set_ylabel("平均未来 5 日收益")
    ax.legend()
    savefig(fig, "11_target_over_time")


def plot_target_by_year(df: pd.DataFrame) -> None:
    sub = df[[DATE_COL, TARGET_COL]].dropna().copy()
    sub["year"] = sub[DATE_COL].dt.year
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.boxplot(data=sub, x="year", y=TARGET_COL, showfliers=False, ax=ax, palette="crest")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_title(f"按年份的 {TARGET_COL} 分布")
    savefig(fig, "12_target_by_year")


# ---------------------------------------------------------------- main

def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"读取数据集: {DATASET_PATH}")
    df = pd.read_parquet(DATASET_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    print(f"加载完成: {df.shape}\n")

    # 概览
    info = overview(df)
    print("=== 概览 ===")
    for k, v in info.items():
        print(f"  {k}: {v}")
    print()

    # 表格：概览 / 缺失 / 数值描述
    pd.Series(info).to_csv(TABLE_DIR / "00_overview.csv", encoding="utf-8-sig")

    miss = missing_analysis(df)
    save_table(miss, "01_missing")

    desc = df.select_dtypes(include=[np.number]).describe().T
    save_table(desc, "02_numeric_describe")

    industry_counts = df[INDUSTRY_COL].value_counts().rename_axis(INDUSTRY_COL).to_frame("count")
    save_table(industry_counts, "03_industry_counts")

    regime_counts = df[REGIME_COL].value_counts().rename_axis(REGIME_COL).to_frame("count")
    save_table(regime_counts, "04_regime_counts")

    # 图表
    print("生成图表 ...")
    plot_missing(miss)
    plot_samples_per_day(df)
    plot_target_distribution(df)
    plot_target_by_regime(df)
    plot_regime_timeline(df)
    plot_industry_top(df)
    plot_industry_target(df)
    corr = plot_correlation(df)
    save_table(corr, "05_correlation_matrix")
    feat_corr = plot_target_correlation(corr)
    save_table(feat_corr.to_frame("corr_with_target"), "06_feature_target_corr")
    plot_feature_distributions(df)
    plot_target_over_time(df)
    plot_target_by_year(df)

    # 汇总报告
    report = ANALYSIS_OUTPUT_DIR / "REPORT.md"
    lines = [
        "# 数据集分析报告",
        "",
        f"- 数据集: `{DATASET_PATH}`",
        f"- 行数: {info['rows']:,}",
        f"- 列数: {info['cols']}",
        f"- 股票数: {info['tickers']:,}",
        f"- 日期数: {info['dates']:,}",
        f"- 日期范围: {info['date_min'].date()} ~ {info['date_max'].date()}",
        f"- 行业数: {info['industries']}",
        f"- 内存占用: {info['memory_gb']:.2f} GB",
        "",
        "## 与目标列相关性 Top10（绝对值）",
        "",
        feat_corr.reindex(feat_corr.abs().sort_values(ascending=False).index)
            .head(10).to_frame("corr").to_markdown(),
        "",
        "## 缺失值 Top10",
        "",
        miss[miss["missing"] > 0].head(10).to_markdown(),
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告已生成: {report}")
    print(f"全部输出在: {ANALYSIS_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
