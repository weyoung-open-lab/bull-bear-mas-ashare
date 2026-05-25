"""
Top-K 选股回测（预报告 §6.1 C 项 / Table 2）。

每天选 Top-K%，等权持有 HOLDING_DAYS=5 天 → 5 个错峰组合；
日组合收益 = 当日活跃组合的等权平均；
扣双边换手成本（cost_one_side × 2），Sharpe 按 252 日年化。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import (
    COST_ONE_SIDE,
    DATE_COL,
    HOLDING_DAYS,
    TARGET_RET_COL,
    TICKER_COL,
    TOP_K_FRACS,
    TRADING_DAYS_PER_YEAR,
)


@dataclass
class BacktestResult:
    annual_return: float
    annual_volatility: float
    sharpe: float
    max_drawdown: float
    avg_turnover: float
    n_days: int
    daily_return: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    nav: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


def _select_topk_per_day(
    meta: pd.DataFrame, pred: np.ndarray, frac: float
) -> dict[pd.Timestamp, set[str]]:
    """每个日期选预测分数 Top frac% 的股票集合。"""
    df = meta[[DATE_COL, TICKER_COL]].copy()
    df["pred"] = pred
    out: dict[pd.Timestamp, set[str]] = {}
    for d, g in df.groupby(DATE_COL, sort=True):
        n = len(g)
        k = max(1, int(np.ceil(n * frac)))
        idx = g["pred"].to_numpy().argsort()[::-1][:k]
        out[d] = set(g.iloc[idx][TICKER_COL].tolist())
    return out


def _daily_returns_table(meta: pd.DataFrame) -> pd.DataFrame:
    """返回 ticker × date 的 1 日收益（基于 ret_1d，从 meta 重新关联）。

    输入 meta 必须含 ret_1d 列。如果只有 r_future_5，就用 r_future_5/HOLDING_DAYS 近似。
    """
    if "ret_1d" in meta.columns:
        col = "ret_1d"
        df = meta[[DATE_COL, TICKER_COL, col]].copy()
        df.rename(columns={col: "ret"}, inplace=True)
    else:
        # fallback：用 r_future_5 的 1/5 作为日收益代理
        df = meta[[DATE_COL, TICKER_COL, TARGET_RET_COL]].copy()
        df["ret"] = df[TARGET_RET_COL] / HOLDING_DAYS
        df = df.drop(columns=[TARGET_RET_COL])
    return df


def backtest_topk(
    meta: pd.DataFrame,
    pred: np.ndarray,
    *,
    frac: float = 0.05,
    holding_days: int = HOLDING_DAYS,
    cost_one_side: float = COST_ONE_SIDE,
) -> BacktestResult:
    """错峰持仓回测：每天选 Top-K%，5 个错峰组合等权平均。"""
    selections = _select_topk_per_day(meta, pred, frac)
    dates = sorted(selections.keys())
    if len(dates) == 0:
        return BacktestResult(np.nan, np.nan, np.nan, np.nan, np.nan, 0)

    rets_table = _daily_returns_table(meta)
    rets_table = rets_table.set_index([DATE_COL, TICKER_COL])["ret"]

    # 把日期映射成连续索引便于错峰
    date_index = pd.Index(sorted(meta[DATE_COL].unique()))
    daily_returns = []
    daily_turnover = []

    # 每天用上一日（同 cohort）持仓 vs 当日选仓计算换手
    prev_holdings: dict[int, set[str]] = {c: set() for c in range(holding_days)}

    for i, d in enumerate(date_index):
        if d not in selections:
            continue
        cohort = i % holding_days
        new_holdings = selections[d]

        # cohort 换手 = 当日新选 vs 上一次该 cohort 持仓
        prev = prev_holdings[cohort]
        if prev:
            turnover = len(new_holdings.symmetric_difference(prev)) / max(
                len(new_holdings) + len(prev), 1
            )
        else:
            turnover = 1.0
        prev_holdings[cohort] = new_holdings

        # 当日 portfolio return = 各 active cohort 的平均收益均值
        # active cohort：所有上一次还在持有期内的 cohort（这里简化为全部 5 个）
        cohort_rets = []
        for c in range(holding_days):
            holdings = prev_holdings[c]
            if not holdings:
                continue
            try:
                day_rets = rets_table.loc[d]
            except KeyError:
                continue
            common = list(holdings & set(day_rets.index))
            if not common:
                continue
            cohort_rets.append(float(day_rets.loc[common].mean()))

        if cohort_rets:
            gross = float(np.mean(cohort_rets))
        else:
            gross = 0.0

        # 该 cohort 当日扣双边换手成本（其它 cohort 不调仓不扣费）
        cost = turnover * cost_one_side * 2 / holding_days
        net = gross - cost
        daily_returns.append((d, net))
        daily_turnover.append(turnover)

    if not daily_returns:
        return BacktestResult(np.nan, np.nan, np.nan, np.nan, np.nan, 0)

    s = pd.Series({d: r for d, r in daily_returns}).sort_index()
    nav = (1.0 + s).cumprod()

    ann_ret = float(s.mean() * TRADING_DAYS_PER_YEAR)
    ann_vol = float(s.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    drawdown = nav / nav.cummax() - 1
    max_dd = float(drawdown.min())
    avg_turnover = float(np.mean(daily_turnover)) if daily_turnover else np.nan

    return BacktestResult(
        annual_return=ann_ret,
        annual_volatility=ann_vol,
        sharpe=sharpe,
        max_drawdown=max_dd,
        avg_turnover=avg_turnover,
        n_days=len(s),
        daily_return=s,
        nav=nav,
    )


def backtest_all_fracs(
    meta: pd.DataFrame, pred: np.ndarray, fracs: tuple[float, ...] = TOP_K_FRACS
) -> dict[float, BacktestResult]:
    return {f: backtest_topk(meta, pred, frac=f) for f in fracs}
