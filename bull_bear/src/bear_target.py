"""构造 Bear Agent 的训练目标 max_drawdown_5d。

数据集只含 r_future_5 (cumulative 5-day forward return) 与 ret_1d (1-day past return)
没有原始 close 价格。我们通过对每只股票按时间排序的 ret_1d 序列做 cumprod，
得到隐含的 "归一化 close" 曲线，再计算 5 日窗口内的最大回撤（相对 t 日）。

对每只股票每天 t（在按 ticker 排序的子序列中位置 k）：
    g[k] = 1 + ret_1d at row k          # ret_1d 在每一行是 close[k]/close[k-1]
    cg[k] = cumprod(g[0..k])             # 隐含归一化价格
    r_dj(k) = cg[k+j]/cg[k] - 1          # 累计 j 日远期收益
    max_drawdown_5d(k) = max(0, -min_{j=1..5} r_dj(k))

最后每只股票的最后 5 行 target 设为 NaN（forward window 不足）。
"""

from __future__ import annotations

import sys
from pathlib import Path
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))

import numpy as np
import pandas as pd

from bull_bear.config_bb import DATE_COL, TICKER_COL

FORWARD_WINDOW = 5


def build_max_drawdown_5d(df: pd.DataFrame, ret_col: str = "ret_1d",
                            window: int = FORWARD_WINDOW) -> pd.DataFrame:
    """对原始 panel 按 ticker 分组添加 max_drawdown_5d 列。

    Parameters
    ----------
    df : pd.DataFrame
        必须含 DATE_COL, TICKER_COL, ret_col 列。
    ret_col : str
        1-day 收益列名，默认 ret_1d。
    window : int
        前看天数（默认 5）。

    Returns
    -------
    pd.DataFrame  原 df + max_drawdown_5d 列。
    """
    df = df.sort_values([TICKER_COL, DATE_COL]).reset_index(drop=True)
    out = np.full(len(df), np.nan, dtype="float64")
    # 按 ticker 分块
    for ticker, sub in df.groupby(TICKER_COL):
        idx = sub.index.to_numpy()
        ret = sub[ret_col].to_numpy(dtype="float64")
        # NaN ret -> 1.0 (no change)；安全处理首日
        g = 1.0 + np.where(np.isnan(ret), 0.0, ret)
        cg = np.cumprod(g)
        n = len(sub)
        if n <= window:
            continue
        # 构造 (window, n) 的 forward cg 矩阵：第 j 行表示 cg[t+j+1] (j=0..window-1)
        shifted = np.full((window, n), np.nan, dtype="float64")
        for j in range(1, window + 1):
            shifted[j - 1, :n - j] = cg[j:]
        # 每个 t 的最小 cg over j ∈ {1..window}
        with np.errstate(invalid="ignore"):
            cg_min = np.nanmin(shifted, axis=0)
            r_min = cg_min / cg - 1.0    # 最小累计收益（最负）
        max_dd = np.maximum(0.0, -r_min)
        # 最后 window 行无效
        max_dd[-window:] = np.nan
        out[idx] = max_dd
    df["max_drawdown_5d"] = out
    return df


def cross_section_zscore(df: pd.DataFrame, col: str,
                            date_col: str = DATE_COL) -> pd.DataFrame:
    """每日截面对指定列做 z-score；输出列名 = col + "_z"。"""
    out = np.zeros(len(df), dtype="float64")
    df = df.reset_index(drop=True)
    for d, g in df.groupby(date_col):
        v = g[col].to_numpy(dtype="float64")
        mu = np.nanmean(v)
        sd = np.nanstd(v, ddof=0)
        if sd > 1e-9 and np.isfinite(mu):
            z = (v - mu) / sd
            z = np.where(np.isfinite(z), z, 0.0)
        else:
            z = np.zeros_like(v)
        out[g.index.to_numpy()] = z
    df[col + "_z"] = out.astype("float32")
    return df


def build_path_vol_5d(df: pd.DataFrame, ret_col: str = "ret_1d",
                       window: int = FORWARD_WINDOW) -> pd.DataFrame:
    """对 panel 添加 path_vol_5d 列。

    path_vol_5d(t) = std(r_d1(t), r_d2(t), ..., r_dwindow(t))
    其中 r_dj(t) = cg[t+j]/cg[t] - 1 （t 日观点下的 j 日累计收益）
    """
    df = df.sort_values([TICKER_COL, DATE_COL]).reset_index(drop=True)
    out = np.full(len(df), np.nan, dtype="float64")
    for ticker, sub in df.groupby(TICKER_COL):
        idx = sub.index.to_numpy()
        ret = sub[ret_col].to_numpy(dtype="float64")
        g = 1.0 + np.where(np.isnan(ret), 0.0, ret)
        cg = np.cumprod(g)
        n = len(sub)
        if n <= window:
            continue
        # 构造 (window, n) 的 r_dj 矩阵：r_dj[t] = cg[t+j]/cg[t] - 1
        r_paths = np.full((window, n), np.nan, dtype="float64")
        for j in range(1, window + 1):
            r_paths[j - 1, :n - j] = cg[j:] / cg[:n - j] - 1.0
        # std over j 轴
        with np.errstate(invalid="ignore"):
            path_std = np.nanstd(r_paths, axis=0, ddof=0)
        path_std[-window:] = np.nan
        out[idx] = path_std
    df["path_vol_5d"] = out
    return df


def build_disappointment(df: pd.DataFrame,
                            alpha_pred_col: str,
                            actual_col: str = "r_future_5",
                            date_col: str = DATE_COL) -> pd.DataFrame:
    """对 panel 计算 disappointment = alpha_rank − actual_rank。

    各百分位排名 ∈ [0, 100]。差值越大表示 Alpha 越高估实际表现。
    NaN 值（无 alpha_pred 或无 r_future_5）跳过。
    """
    df = df.reset_index(drop=True)
    out = np.full(len(df), np.nan, dtype="float64")
    for d, g in df.groupby(date_col):
        ap = g[alpha_pred_col].to_numpy(dtype="float64")
        rf = g[actual_col].to_numpy(dtype="float64")
        # 跨股票截面 percentile rank（均值 method, pct=True * 100）
        ar = pd.Series(ap).rank(method="average", pct=True).to_numpy() * 100.0
        nr = pd.Series(rf).rank(method="average", pct=True).to_numpy() * 100.0
        disap = ar - nr
        # 若 alpha 或 future 是 NaN 则结果 NaN
        invalid = np.isnan(ap) | np.isnan(rf)
        disap[invalid] = np.nan
        out[g.index.to_numpy()] = disap
    df["disappointment"] = out
    return df
