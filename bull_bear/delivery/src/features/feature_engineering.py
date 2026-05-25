"""Bull-Bear Bear Agent V2 派生特征。

加入 abs / square 变换，赋予 Bear Agent 学习"偏离距离"对称信息的能力。
这些是 Alpha Agent 无法通过线性 boosting 学到的非线性轴。
"""

from __future__ import annotations

import pandas as pd


def add_bear_v2_features(df: pd.DataFrame) -> pd.DataFrame:
    """加入 5 个派生特征列：

    abs_bias_60       = |bias_60|
    abs_bias_60_vr    = |bias_60_vr|
    bias_60_sq        = bias_60 ** 2
    abs_ma60_slope    = |ma60_slope|
    abs_ema180_slope  = |ema180_slope|
    """
    df = df.copy()
    df["abs_bias_60"]     = df["bias_60"].abs().astype("float32")
    df["abs_bias_60_vr"]  = df["bias_60_vr"].abs().astype("float32")
    df["bias_60_sq"]      = (df["bias_60"].astype("float64") ** 2).astype("float32")
    df["abs_ma60_slope"]  = df["ma60_slope"].abs().astype("float32")
    df["abs_ema180_slope"] = df["ema180_slope"].abs().astype("float32")
    return df
