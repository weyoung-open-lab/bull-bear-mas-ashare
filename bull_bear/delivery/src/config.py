import os
from pathlib import Path

# ----- 数据 -----
# 可用 STOCK_THESIS_DATASET 环境变量覆盖（服务器 Linux 路径）
_default_dataset = r"D:\project1\pythonProject\QMT\单股票每日态势识别\论文\dataset_model_baseline_longer_trend.parquet"
DATASET_PATH = Path(os.environ.get("STOCK_THESIS_DATASET", _default_dataset))

# ----- 项目目录 -----
PROJECT_ROOT = Path(__file__).resolve().parent
ANALYSIS_OUTPUT_DIR = PROJECT_ROOT / "数据集分析输出"
RESULTS_DIR = PROJECT_ROOT / "results"

# ----- 列名 -----
DATE_COL = "date"
TICKER_COL = "ticker"
INDUSTRY_COL = "所属行业"
REGIME_COL = "macro_regime_3"
TARGET_RET_COL = "r_future_5"
LABEL_COL = "label"
ID_COLS = [DATE_COL, TICKER_COL, INDUSTRY_COL, REGIME_COL]

# ----- 数据切分 / 标签 (预报告 §2 / §6) -----
TRAIN_END_DATE = "2023-01-01"   # train: date < ; test: date >=
TARGET_THRESHOLD = 0.01          # r_future_5 > 1% → label=1

# ----- 评估配置 (预报告 §6) -----
TOP_K_FRACS = (0.01, 0.03, 0.05, 0.10)
HOLDING_DAYS = 5
TRADING_DAYS_PER_YEAR = 252
COST_ONE_SIDE = 0.0015           # 单边 0.15%

# ----- regime 阈值 (预报告 §5.2) -----
REGIME_BULL_THRESHOLD = 0.005
REGIME_BEAR_THRESHOLD = -0.005

# ----- 复现 -----
RANDOM_SEED = 42
