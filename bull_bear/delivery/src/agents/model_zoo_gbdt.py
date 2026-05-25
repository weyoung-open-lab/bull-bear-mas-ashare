"""GBDT 集成树模型（预报告 §4.3 Table 8）。

LightGBM 三档（标准 / 浅树 / 保守）+ XGBoost + CatBoost + RandomForest
全部 SHAP TreeExplainer 兼容。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from config import RANDOM_SEED
from src.models.base import BaseModel


# ----------------------------------------------------------------- LightGBM

class _LGBMBase(BaseModel):
    family = "gbdt"
    preprocess = "raw"

    def fit(self, X: pd.DataFrame, y: np.ndarray, **kw):
        self._model.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X)[:, 1]


class LGBMStd(_LGBMBase):
    name = "LightGBM-std"

    def __init__(self):
        super().__init__()
        self._model = LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=63,
            min_child_samples=20, n_jobs=-1, random_state=RANDOM_SEED, verbosity=-1,
        )


class LGBMShallow(_LGBMBase):
    name = "LightGBM-shallow"

    def __init__(self):
        super().__init__()
        self._model = LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=15, max_depth=4,
            min_child_samples=50, n_jobs=-1, random_state=RANDOM_SEED, verbosity=-1,
        )


class LGBMCons(_LGBMBase):
    name = "LightGBM-conservative"

    def __init__(self):
        super().__init__()
        self._model = LGBMClassifier(
            n_estimators=800, learning_rate=0.02, num_leaves=31,
            min_child_samples=100, n_jobs=-1, random_state=RANDOM_SEED, verbosity=-1,
        )


# ----------------------------------------------------------------- XGBoost

class XGBStd(BaseModel):
    name = "XGBoost"
    family = "gbdt"
    preprocess = "raw"

    def __init__(self):
        super().__init__()
        self._model = XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.9, colsample_bytree=0.9,
            tree_method="hist", n_jobs=-1, random_state=RANDOM_SEED,
            eval_metric="auc", verbosity=0,
        )

    def fit(self, X: pd.DataFrame, y: np.ndarray, **kw):
        self._model.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X)[:, 1]


# ----------------------------------------------------------------- CatBoost

class CatBoostStd(BaseModel):
    name = "CatBoost"
    family = "gbdt"
    preprocess = "raw"

    def __init__(self):
        super().__init__()
        self._model = CatBoostClassifier(
            iterations=300, learning_rate=0.05, depth=6,
            random_seed=RANDOM_SEED, verbose=0, allow_writing_files=False,
        )

    def fit(self, X: pd.DataFrame, y: np.ndarray, **kw):
        self._model.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X)[:, 1]


# ----------------------------------------------------------------- RandomForest

class RFStd(BaseModel):
    name = "RandomForest"
    family = "gbdt"
    preprocess = "raw"

    def __init__(self):
        super().__init__()
        # 700 万行用全样本会非常慢：max_samples=0.2 子采样 + max_depth=14 限深
        self._model = RandomForestClassifier(
            n_estimators=300,
            max_depth=14,
            max_features="sqrt",
            max_samples=0.2,
            n_jobs=-1,
            random_state=RANDOM_SEED,
        )

    def fit(self, X: pd.DataFrame, y: np.ndarray, **kw):
        self._model.fit(X.to_numpy(dtype="float32"), y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X.to_numpy(dtype="float32"))[:, 1]


GBDT_MODELS = [LGBMStd, LGBMShallow, LGBMCons, XGBStd, CatBoostStd, RFStd]


# =================================================================
# Regression variants（回归损失，预报告 §7 讨论用）
# =================================================================

from catboost import CatBoostRegressor          # noqa: E402
from lightgbm import LGBMRegressor              # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from xgboost import XGBRegressor                # noqa: E402


class _LGBMRegBase(BaseModel):
    family = "gbdt"
    preprocess = "raw"
    regression_target = True

    def fit(self, X: pd.DataFrame, y: np.ndarray, **kw):
        self._model.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)


class LGBMStdReg(_LGBMRegBase):
    name = "LightGBM-std-reg"
    def __init__(self):
        super().__init__()
        self._model = LGBMRegressor(
            n_estimators=500, learning_rate=0.05, num_leaves=63,
            min_child_samples=20, n_jobs=-1, random_state=RANDOM_SEED,
            verbosity=-1, objective="regression",
        )


class LGBMShallowReg(_LGBMRegBase):
    name = "LightGBM-shallow-reg"
    def __init__(self):
        super().__init__()
        self._model = LGBMRegressor(
            n_estimators=500, learning_rate=0.05, num_leaves=15, max_depth=4,
            min_child_samples=50, n_jobs=-1, random_state=RANDOM_SEED,
            verbosity=-1, objective="regression",
        )


class LGBMConsReg(_LGBMRegBase):
    name = "LightGBM-conservative-reg"
    def __init__(self):
        super().__init__()
        self._model = LGBMRegressor(
            n_estimators=800, learning_rate=0.02, num_leaves=31,
            min_child_samples=100, n_jobs=-1, random_state=RANDOM_SEED,
            verbosity=-1, objective="regression",
        )


class XGBStdReg(BaseModel):
    name = "XGBoost-reg"
    family = "gbdt"
    preprocess = "raw"
    regression_target = True

    def __init__(self):
        super().__init__()
        self._model = XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.9, colsample_bytree=0.9,
            tree_method="hist", n_jobs=-1, random_state=RANDOM_SEED,
            verbosity=0,
        )

    def fit(self, X: pd.DataFrame, y: np.ndarray, **kw):
        self._model.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)


class CatBoostStdReg(BaseModel):
    name = "CatBoost-reg"
    family = "gbdt"
    preprocess = "raw"
    regression_target = True

    def __init__(self):
        super().__init__()
        self._model = CatBoostRegressor(
            iterations=300, learning_rate=0.05, depth=6,
            random_seed=RANDOM_SEED, verbose=0, allow_writing_files=False,
        )

    def fit(self, X: pd.DataFrame, y: np.ndarray, **kw):
        self._model.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)


class RFStdReg(BaseModel):
    name = "RandomForest-reg"
    family = "gbdt"
    preprocess = "raw"
    regression_target = True

    def __init__(self):
        super().__init__()
        self._model = RandomForestRegressor(
            n_estimators=300, max_depth=14, max_features="sqrt",
            max_samples=0.2, n_jobs=-1, random_state=RANDOM_SEED,
        )

    def fit(self, X: pd.DataFrame, y: np.ndarray, **kw):
        self._model.fit(X.to_numpy(dtype="float32"), y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X.to_numpy(dtype="float32"))


GBDT_REG_MODELS = [LGBMStdReg, LGBMShallowReg, LGBMConsReg, XGBStdReg, CatBoostStdReg, RFStdReg]
