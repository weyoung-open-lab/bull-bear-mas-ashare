"""模型抽象基类（预报告 §9 Table 19：fit / predict_proba / raw_model）。"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """所有 benchmark 模型的统一接口。"""

    name: str = "base"
    family: str = "base"          # factor / linear / gbdt / tabular_dl / sequence
    preprocess: str = "raw"        # raw / zscore / standard
    needs_fit: bool = True
    regression_target: bool = False  # True → fit on r_future_5（回归损失）；False → 二分类标签

    def __init__(self, **kwargs):
        self.params = kwargs

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray, **kwargs) -> "BaseModel": ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """返回长度为 len(X) 的一维分数（升序：分高 → 更可能上涨）。"""
        ...

    @property
    def raw_model(self):
        return getattr(self, "_model", None)
