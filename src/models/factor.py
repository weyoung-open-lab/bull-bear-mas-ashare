"""传统因子 baseline（预报告 §4.1 Table 6）。

不需要训练，直接按某个原始因子降序选股。作为「比直接用传统量化因子强多少」
的下界参照。preprocess 必须是 raw（不缩放）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.base import BaseModel


class _FactorBase(BaseModel):
    family = "factor"
    preprocess = "raw"
    needs_fit = False
    factor_col: str = ""          # 子类定义

    def fit(self, X, y, **kw):
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.factor_col not in X.columns:
            raise KeyError(f"{self.factor_col} not in features")
        s = X[self.factor_col].to_numpy(dtype=float)
        # 把缺失填成中位数避免排序异常
        med = np.nanmedian(s)
        s = np.where(np.isnan(s), med, s)
        return s


class Momentum5d(_FactorBase):
    name = "Momentum-5d"
    factor_col = "ret_5d"


class EmaSlope(_FactorBase):
    name = "EMA-slope"
    factor_col = "ema30_slope"


class RelStrength(_FactorBase):
    name = "Rel-Strength"
    factor_col = "board_rs_20d"


FACTOR_MODELS = [Momentum5d, EmaSlope, RelStrength]
