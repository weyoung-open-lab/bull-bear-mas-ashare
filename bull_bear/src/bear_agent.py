"""BearAgent — CatBoost-reg 学习未来 5 日最大回撤 z-score。

API 与 strategy_debate 的 StrategyAgent 平行，但目标是 risk 而非 return，
评估指标也用 Spearman(pred, max_drawdown_5d) 而非 IC 上行。
"""

from __future__ import annotations

import sys
from pathlib import Path
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))

from typing import Optional

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from scipy.stats import spearmanr

from bull_bear.config_bb import (
    BEAR_FEATURES,
    BB_MODELS,
    CATBOOST_PARAMS,
    DATE_COL,
    TICKER_COL,
)


class BearAgent:
    """CatBoost-reg 预测 max_drawdown_5d_z（截面 z-scored target）。

    Parameters
    ----------
    features : list[str] | None
        预测特征；默认 BEAR_FEATURES。
    name : str
        模型名（仅用于保存/打印）。
    """

    def __init__(self, features: Optional[list[str]] = None,
                  name: str = "bear"):
        self.name = name
        self.features = list(features) if features is not None else list(BEAR_FEATURES)
        self.model: Optional[CatBoostRegressor] = None
        self._train_medians: Optional[pd.Series] = None
        self.best_iteration_: Optional[int] = None
        self.train_rmse_: Optional[float] = None
        self.val_rmse_: Optional[float] = None

    # ------------------------------------------------------------------
    def train(self, df_train: pd.DataFrame,
                target_col: str = "max_drawdown_5d_z",
                df_val: pd.DataFrame | None = None,
                save: bool = True) -> "BearAgent":
        """训练。固定 300 轮，无早停（与 Alpha Agent 一致）。"""
        # 训练集过滤：drop target NaN（最后 5 行 per ticker + 异常数据）
        tr = df_train.dropna(subset=[target_col]).reset_index(drop=True)
        X = tr[self.features].astype("float32").copy()
        self._train_medians = X.median()
        X = X.fillna(self._train_medians)
        y = tr[target_col].astype("float32").to_numpy()

        m = CatBoostRegressor(**CATBOOST_PARAMS)
        m.fit(Pool(X, y), verbose=False)
        self.model = m
        self.best_iteration_ = int(m.tree_count_)

        train_pred = m.predict(X)
        self.train_rmse_ = float(np.sqrt(np.mean((train_pred - y) ** 2)))

        if df_val is not None:
            vd = df_val.dropna(subset=[target_col]).reset_index(drop=True)
            Xv = vd[self.features].astype("float32").fillna(self._train_medians)
            yv = vd[target_col].astype("float32").to_numpy()
            pv = m.predict(Xv)
            self.val_rmse_ = float(np.sqrt(np.mean((pv - yv) ** 2)))

        print(f"[BearAgent {self.name:5s}]  best_iter={self.best_iteration_:>3d}  "
              f"train_RMSE={self.train_rmse_:.6f}  "
              f"val_RMSE={self.val_rmse_ if self.val_rmse_ is not None else float('nan'):.6f}  "
              f"n_features={len(self.features)}")

        if save:
            self.save()
        return self

    # ------------------------------------------------------------------
    def predict_panel(self, df_panel: pd.DataFrame) -> np.ndarray:
        """对 panel 预测，返回 (N,) float32 数组（与行序对齐）。"""
        if self.model is None:
            raise RuntimeError("BearAgent not trained")
        X = df_panel[self.features].astype("float32")
        if self._train_medians is not None:
            X = X.fillna(self._train_medians)
        else:
            X = X.fillna(X.median())
        return self.model.predict(X).astype("float32")

    def validate_spearman(self, df_val: pd.DataFrame,
                            target_col: str = "max_drawdown_5d") -> float:
        """在验证集上算 Spearman(pred, raw max_drawdown_5d)。

        正值（高分预测高回撤）= bear 信号有效。
        """
        if self.model is None:
            raise RuntimeError("BearAgent not trained")
        vd = df_val.dropna(subset=[target_col]).reset_index(drop=True)
        X = vd[self.features].astype("float32").fillna(self._train_medians)
        pred = self.model.predict(X)
        target = vd[target_col].to_numpy(dtype="float64")
        rho, _ = spearmanr(pred, target)
        return float(rho)

    # ------------------------------------------------------------------
    def save(self, path: Optional[Path] = None) -> Path:
        """保存 .cbm + medians."""
        path = path or (BB_MODELS / f"{self.name}_agent.cbm")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))
        if self._train_medians is not None:
            self._train_medians.to_csv(str(path).replace(".cbm", "_medians.csv"))
        return path

    def load(self, path: Optional[Path] = None) -> "BearAgent":
        """加载已保存的模型。"""
        path = path or (BB_MODELS / f"{self.name}_agent.cbm")
        m = CatBoostRegressor()
        m.load_model(str(path))
        self.model = m
        med_path = Path(str(path).replace(".cbm", "_medians.csv"))
        if med_path.exists():
            self._train_medians = pd.read_csv(med_path, index_col=0).iloc[:, 0]
        return self
