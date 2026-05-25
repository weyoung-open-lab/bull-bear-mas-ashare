"""One-stop evaluator used across bull_bear/experiments.

Combines the parent project's metrics + backtest into a single dict that
captures everything the BBAQ-MAS experiments report: RankIC / RankICIR /
ICIR / AUC / Top-K returns / Top-K Sharpe / annualised return / vol /
MaxDD / turnover. Returns plain Python floats so results can be written
directly to CSV.
"""

from __future__ import annotations

import sys
from pathlib import Path
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))     # add project root

import numpy as np
import pandas as pd

from config import DATE_COL, TARGET_RET_COL, TICKER_COL
from src.backtest import backtest_topk
from src.metrics import evaluate


def evaluate_full(meta: pd.DataFrame, pred: np.ndarray,
                   target_threshold: float = 0.01,
                   frac: float = 0.05) -> dict:
    """Compute the full metric bundle.

    Parameters
    ----------
    meta : DataFrame with columns [DATE_COL, TICKER_COL, TARGET_RET_COL].
    pred : np.ndarray of length len(meta), aligned row-by-row.
    target_threshold : float threshold for the binary label used in AUC.
    frac : Top-K portfolio fraction (default 0.05 = top 5%).
    """
    label = (meta[TARGET_RET_COL] > target_threshold).astype("int8").to_numpy()
    m = evaluate(label, pred, meta[[DATE_COL, TICKER_COL, TARGET_RET_COL]])
    bt = backtest_topk(meta[[DATE_COL, TICKER_COL, TARGET_RET_COL]],
                       pred, frac=frac)
    return {
        "rankic_mean": m.rankic_mean, "rankicir": m.rankicir,
        "ic_mean": m.ic_mean, "icir": m.icir, "auc": m.auc,
        "top1pct_ret": m.topk_returns.get(0.01, np.nan),
        "top5pct_ret": m.topk_returns.get(0.05, np.nan),
        "top5pct_sharpe": bt.sharpe,
        "top5pct_ann_return": bt.annual_return,
        "top5pct_ann_volatility": bt.annual_volatility,
        "top5pct_max_dd": bt.max_drawdown,
        "top5pct_turnover": bt.avg_turnover,
        "n_days": m.n_days,
    }
