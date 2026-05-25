"""线性 baseline（预报告 §4.2 Table 7）。

LogisticRegression：系数可解释，配合 z-score 预处理。
SHAP 路径：LinearExplainer（精确、极快）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

from config import RANDOM_SEED
from src.models.base import BaseModel


class LogReg(BaseModel):
    name = "LogisticRegression"
    family = "linear"
    preprocess = "zscore"

    def __init__(self, C: float = 1.0, max_iter: int = 200, **kwargs):
        super().__init__(C=C, max_iter=max_iter, **kwargs)
        self._model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver="lbfgs",
            n_jobs=-1,
            random_state=RANDOM_SEED,
        )

    def fit(self, X: pd.DataFrame, y: np.ndarray, **kw):
        self._model.fit(X.to_numpy(dtype="float32"), y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X.to_numpy(dtype="float32"))[:, 1]


LINEAR_MODELS = [LogReg]


# Regression variant
class RidgeReg(BaseModel):
    name = "Ridge"
    family = "linear"
    preprocess = "zscore"
    regression_target = True

    def __init__(self, alpha: float = 1.0):
        super().__init__(alpha=alpha)
        self._model = Ridge(alpha=alpha, random_state=RANDOM_SEED)

    def fit(self, X: pd.DataFrame, y: np.ndarray, **kw):
        self._model.fit(X.to_numpy(dtype="float32"), y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X.to_numpy(dtype="float32"))


LINEAR_REG_MODELS = [RidgeReg]
