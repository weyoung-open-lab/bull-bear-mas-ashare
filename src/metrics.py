"""
评估指标（预报告 §6.1）。

- AUC / Accuracy（二分类基础）
- IC / RankIC：日截面相关，再对天求均值/标准差
- ICIR / RankICIR：均值 / 标准差，论文 §6 Table 1 主指标
- Top-K Hit Rate / Top-K Future Return
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import accuracy_score, roc_auc_score

from config import DATE_COL, TARGET_RET_COL, TOP_K_FRACS


@dataclass
class MetricResult:
    auc: float
    accuracy: float
    ic_mean: float
    ic_std: float
    icir: float
    rankic_mean: float
    rankic_std: float
    rankicir: float
    topk_returns: dict[float, float]   # {0.01: mean ret, ...}
    topk_hit_rates: dict[float, float]
    n_days: int

    def to_row(self) -> dict:
        row = {
            "auc": self.auc,
            "accuracy": self.accuracy,
            "ic_mean": self.ic_mean,
            "ic_std": self.ic_std,
            "icir": self.icir,
            "rankic_mean": self.rankic_mean,
            "rankic_std": self.rankic_std,
            "rankicir": self.rankicir,
            "n_days": self.n_days,
        }
        for k, v in self.topk_returns.items():
            row[f"top{int(k*100)}pct_ret"] = v
        for k, v in self.topk_hit_rates.items():
            row[f"top{int(k*100)}pct_hit"] = v
        return row


def _safe_corr(a: np.ndarray, b: np.ndarray, kind: str = "pearson") -> float:
    if len(a) < 2:
        return np.nan
    if kind == "spearman":
        a = rankdata(a)
        b = rankdata(b)
    sa = a.std()
    sb = b.std()
    if sa == 0 or sb == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def daily_ic(meta: pd.DataFrame, pred: np.ndarray, kind: str = "pearson") -> pd.Series:
    """每个交易日上 pred 与 r_future_5 的相关系数。"""
    df = meta[[DATE_COL, TARGET_RET_COL]].copy()
    df["pred"] = pred
    df = df.dropna(subset=[TARGET_RET_COL])
    out = (
        df.groupby(DATE_COL, sort=True)
        .apply(lambda g: _safe_corr(g["pred"].to_numpy(), g[TARGET_RET_COL].to_numpy(), kind))
    )
    return out.dropna()


def topk_daily_returns(
    meta: pd.DataFrame, pred: np.ndarray, fracs: tuple[float, ...] = TOP_K_FRACS
) -> tuple[dict[float, float], dict[float, float], dict[float, pd.Series]]:
    """对每天选 Top-K%，返回平均 r_future_5（和命中率）。

    返回 (mean_returns, hit_rates, daily_series)
    daily_series[frac] = 每日 Top-K 平均收益的时间序列（用于 backtest）
    """
    df = meta[[DATE_COL, TARGET_RET_COL]].copy()
    df["pred"] = pred
    df["label_pos"] = (df[TARGET_RET_COL] > 0).astype("int8")

    daily_ret: dict[float, list[tuple[pd.Timestamp, float]]] = {f: [] for f in fracs}
    daily_hit: dict[float, list[float]] = {f: [] for f in fracs}

    for d, g in df.groupby(DATE_COL, sort=True):
        n = len(g)
        if n == 0:
            continue
        order = g["pred"].to_numpy().argsort()[::-1]   # desc
        for f in fracs:
            k = max(1, int(np.ceil(n * f)))
            idx = order[:k]
            sub = g.iloc[idx]
            daily_ret[f].append((d, float(sub[TARGET_RET_COL].mean())))
            daily_hit[f].append(float(sub["label_pos"].mean()))

    mean_ret = {}
    hit = {}
    series = {}
    for f in fracs:
        if daily_ret[f]:
            s = pd.Series({d: v for d, v in daily_ret[f]}).sort_index()
            mean_ret[f] = float(s.mean())
            hit[f] = float(np.mean(daily_hit[f]))
            series[f] = s
        else:
            mean_ret[f] = np.nan
            hit[f] = np.nan
            series[f] = pd.Series(dtype=float)
    return mean_ret, hit, series


def evaluate(
    y_true: np.ndarray,
    pred_score: np.ndarray,
    meta: pd.DataFrame,
    fracs: tuple[float, ...] = TOP_K_FRACS,
) -> MetricResult:
    """主入口：给 (y_true, pred_score, meta) 返回完整指标。"""
    pred = np.asarray(pred_score, dtype=float)

    if len(np.unique(y_true)) >= 2:
        auc = float(roc_auc_score(y_true, pred))
    else:
        auc = np.nan
    acc = float(accuracy_score(y_true, (pred > 0.5).astype(int)))

    ic_series = daily_ic(meta, pred, kind="pearson")
    rankic_series = daily_ic(meta, pred, kind="spearman")

    ic_mean = float(ic_series.mean()) if len(ic_series) else np.nan
    ic_std = float(ic_series.std(ddof=0)) if len(ic_series) else np.nan
    rankic_mean = float(rankic_series.mean()) if len(rankic_series) else np.nan
    rankic_std = float(rankic_series.std(ddof=0)) if len(rankic_series) else np.nan
    icir = ic_mean / ic_std if ic_std and ic_std > 0 else np.nan
    rankicir = rankic_mean / rankic_std if rankic_std and rankic_std > 0 else np.nan

    topk_ret, topk_hit, _ = topk_daily_returns(meta, pred, fracs)

    return MetricResult(
        auc=auc,
        accuracy=acc,
        ic_mean=ic_mean,
        ic_std=ic_std,
        icir=icir,
        rankic_mean=rankic_mean,
        rankic_std=rankic_std,
        rankicir=rankicir,
        topk_returns=topk_ret,
        topk_hit_rates=topk_hit,
        n_days=len(ic_series),
    )
