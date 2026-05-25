"""Bull-Bear 对抗框架配置。

复用父项目数据集 + 与 Alpha Agent (Trend) 相同的 CatBoost 超参，
保证 Bear Agent 与 Alpha Agent 的差异只来自训练目标，不来自架构差异。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 让父项目 config / src 可见
_PARENT_ROOT = Path(__file__).resolve().parents[1]
if str(_PARENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PARENT_ROOT))

from config import (  # noqa: E402, F401
    DATASET_PATH,
    PROJECT_ROOT,
    DATE_COL,
    TICKER_COL,
    INDUSTRY_COL,
    REGIME_COL,
    TARGET_RET_COL,
    LABEL_COL,
    TOP_K_FRACS,
    HOLDING_DAYS,
    TRADING_DAYS_PER_YEAR,
    COST_ONE_SIDE,
    RANDOM_SEED,
)


# ----------------------------------------------------------------- 特征集

# Alpha Agent (Trend) — canonical G4 CatBoost-reg trained on r_future_5d.
# Stored under bull_bear/results/models/ and tracked via the .gitignore
# allowlist exception (the trained binary is small enough to commit).
ALPHA_AGENT_PATH = Path(__file__).resolve().parent / "results" / "models" / "alpha_agent.cbm"

# Bear Agent V1 用的特征集（与 Alpha 同源 G4，但目标不同）
BEAR_FEATURES = [
    "bias_60",
    "bias_60_vr",
    "ma60_slope",
    "ema180_slope",
    "trend60",
]

# Bear Agent V2：G4 + 显式非线性变换（abs/sq），让 Bear 学到 Alpha 不可达的对称信息
BEAR_FEATURES_V2 = [
    # G4 原始（5）
    "bias_60", "bias_60_vr", "ma60_slope", "ema180_slope", "trend60",
    # 派生非线性（5）
    "abs_bias_60", "abs_bias_60_vr", "bias_60_sq",
    "abs_ma60_slope", "abs_ema180_slope",
]

# Step 3 三个并行方向
# D1: G1+G3 特征（动量+强度，与 Alpha 的 G4 趋势集不重叠）+ max_drawdown 目标
BEAR_FEATURES_D1 = [
    "ret_1d", "ret_3d", "ret_5d",
    "roc_20", "ema30_slope_vr",
    "board_rank_20d_pct", "board_rs_20d",
    "ret_1d_minus_5d",
]

# D2: G4 特征 + path_vol 目标（与 Alpha 同特征，不同目标）
BEAR_FEATURES_D2 = [
    "bias_60", "bias_60_vr", "ma60_slope", "ema180_slope", "trend60",
]

# D3: G4 特征 + disappointment 目标（直接预测 Alpha 失误的位置）
BEAR_FEATURES_D3 = [
    "bias_60", "bias_60_vr", "ma60_slope", "ema180_slope", "trend60",
]

# ----------------------------------------------------------------- 数据切分

TRAIN_START = "2016-01-01"
TRAIN_END   = "2021-12-31"
VAL_START   = "2022-01-01"
VAL_END     = "2022-12-31"
TEST_START  = "2023-01-01"
TEST_END    = "2026-01-31"


# ----------------------------------------------------------------- CatBoost 超参

CATBOOST_PARAMS = dict(
    iterations=300,
    learning_rate=0.05,
    depth=6,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=RANDOM_SEED,
    verbose=0,
    allow_writing_files=False,
)


# ----------------------------------------------------------------- 路径

BB_ROOT = Path(__file__).resolve().parent
BB_RESULTS = BB_ROOT / "results"
BB_MODELS = BB_RESULTS / "models"
BB_RESULTS.mkdir(parents=True, exist_ok=True)
BB_MODELS.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------- 对抗参数

ALPHA_GRID = (0.1, 0.2, 0.3, 0.5)   # bull − α·bear 中 α 的网格
P10_TARGET_RANKICIR = 0.6074         # 复现基线
P10_ALPHA = 0.2
