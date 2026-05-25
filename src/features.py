"""
特征组注册表（预报告 §3）。

7 个特征组，可任意组合做消融。`get_feature_columns(groups)` 给定组名列表
返回对应特征列；`encode_categorical(df)` 将 macro_regime_3 / 所属行业等
非数值列编码为模型可用的形式。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import INDUSTRY_COL, REGIME_COL


# 特征分组（与 prereport_final.docx Table 5 一致）
FEATURE_GROUPS: dict[str, list[str]] = {
    "G1": ["ret_1d", "ret_3d", "ret_5d", "ret_10d", "momentum_change"],
    "G2": ["ret_1d_minus_5d", "ret_3d_minus_10d", "ret_1d_minus_3d"],
    "G3": [
        "roc_20",
        "ema30_slope_vr", "ema30_slope", "ma30_slope",
        "ema60_slope", "ema90_slope", "ema180_slope",
        "ma60_slope", "ma180_slope",
        "bias_60", "bias_60_vr",
    ],
    "G4": ["board_rank_20d_pct", "board_rs_20d"],
    "G5": [REGIME_COL, "trend60", "breadth_mom"],
    "G6": ["micro_sentiment_ema5", "dispersion", "vol20", "high20_ratio"],
    "G7": [INDUSTRY_COL],
}

# 默认特征 = G1+G2+G3+G4+G5+G6（不含行业；论文主对比基准）
DEFAULT_GROUPS = ("G1", "G2", "G3", "G4", "G5", "G6")

# 所有数值特征（用于 PreprocessPipeline 的列定位）
_REGIME_ORDER = ("bear", "sideway", "bull")
_REGIME_MAP = {"bear": -1.0, "sideway": 0.0, "bull": 1.0}


def get_feature_columns(groups: tuple[str, ...] | list[str] = DEFAULT_GROUPS) -> list[str]:
    cols: list[str] = []
    for g in groups:
        if g not in FEATURE_GROUPS:
            raise KeyError(f"unknown feature group: {g}")
        for c in FEATURE_GROUPS[g]:
            if c not in cols:
                cols.append(c)
    return cols


def encode_categorical(df: pd.DataFrame, industry_codes: dict[str, int] | None = None
                       ) -> tuple[pd.DataFrame, dict[str, int]]:
    """把 regime / 行业编码为数值。industry_codes 由训练集生成，测试集复用。"""
    out = df.copy()
    if REGIME_COL in out.columns and out[REGIME_COL].dtype == object:
        out[REGIME_COL] = out[REGIME_COL].map(_REGIME_MAP).astype("float32")
    if INDUSTRY_COL in out.columns and out[INDUSTRY_COL].dtype == object:
        if industry_codes is None:
            uniques = sorted(out[INDUSTRY_COL].dropna().unique().tolist())
            industry_codes = {v: i for i, v in enumerate(uniques)}
        out[INDUSTRY_COL] = (
            out[INDUSTRY_COL].map(industry_codes).fillna(-1).astype("int32")
        )
    return out, (industry_codes or {})
