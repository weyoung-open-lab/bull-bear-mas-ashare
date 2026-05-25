"""
Regime-Conditioned Ensemble（预报告 §5）。

核心思想：把 macro_regime_3 从「输入特征」改为「样本路由器」，
在 bull/bear/sideway 三个子集上分别训练独立子模型；推理时按当日
macro_regime_3 路由到对应子模型。

类：
    RegimeEnsemble  3 子模型容器 + 路由 fit / predict_proba
工具：
    regime_subset_indices(regime_series) -> dict[regime -> mask]
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from src.models.base import BaseModel


REGIMES: tuple[str, ...] = ("bear", "sideway", "bull")


def regime_subset_indices(regime: pd.Series) -> dict[str, np.ndarray]:
    """返回每个 regime 的布尔掩码（按行索引对齐）。"""
    s = regime.astype(str).to_numpy()
    return {r: (s == r) for r in REGIMES}


class RegimeEnsemble:
    """三个 regime 子模型；按 macro_regime_3 路由（不把它作为输入特征）。"""

    name = "RegimeEnsemble"

    def __init__(self, base_factory: Callable[[], BaseModel]):
        self.base_factory = base_factory
        self.models: dict[str, BaseModel] = {}

    def fit(self, X: pd.DataFrame, y: np.ndarray, regime: pd.Series) -> "RegimeEnsemble":
        masks = regime_subset_indices(regime)
        for r in REGIMES:
            mask = masks[r]
            n = int(mask.sum())
            if n == 0:
                print(f"  [warn] regime={r} has 0 training samples, skip")
                continue
            mdl = self.base_factory()
            mdl.fit(X.iloc[mask], np.asarray(y)[mask])
            self.models[r] = mdl
            print(f"  trained sub-model[{r:7s}] on {n:,} samples")
        return self

    def predict_proba(self, X: pd.DataFrame, regime: pd.Series) -> np.ndarray:
        out = np.full(len(X), np.nan, dtype="float64")
        masks = regime_subset_indices(regime)
        for r, mdl in self.models.items():
            mask = masks[r]
            if not mask.any():
                continue
            out[mask] = mdl.predict_proba(X.iloc[mask])
        # 残余 NaN（regime 缺失）用整体均值兜底
        nan_mask = np.isnan(out)
        if nan_mask.any():
            out[nan_mask] = float(np.nanmean(out))
        return out

    def predict_per_submodel(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """对所有样本，分别用每个子模型预测一遍（用于 cross-regime 分析）。"""
        return {r: m.predict_proba(X) for r, m in self.models.items()}
