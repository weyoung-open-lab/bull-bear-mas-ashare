"""
数据装载、标签构造、训练/测试切分、预处理（预报告 §2 / §3）。

公开 API：
    load_dataset()                   -> 原始 DataFrame（按 date, ticker 排序）
    build_label(df)                  -> 加 binary 标签列
    split_train_test(df)             -> (train_df, test_df)
    Preprocessor                     -> z-score / standard / raw 三档
    prepare_data(...)                -> 一站式打包返回 train/test 矩阵
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import (
    DATASET_PATH,
    DATE_COL,
    LABEL_COL,
    TARGET_RET_COL,
    TARGET_THRESHOLD,
    TICKER_COL,
    TRAIN_END_DATE,
)
from src.features import (
    DEFAULT_GROUPS,
    encode_categorical,
    get_feature_columns,
)


def load_dataset(path=DATASET_PATH) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([DATE_COL, TICKER_COL]).reset_index(drop=True)
    return df


def build_label(df: pd.DataFrame, threshold: float = TARGET_THRESHOLD) -> pd.DataFrame:
    df = df.copy()
    df[LABEL_COL] = (df[TARGET_RET_COL] > threshold).astype("int8")
    return df


def split_train_test(df: pd.DataFrame, train_end: str = TRAIN_END_DATE
                     ) -> tuple[pd.DataFrame, pd.DataFrame]:
    cut = pd.Timestamp(train_end)
    train = df[df[DATE_COL] < cut].reset_index(drop=True)
    test = df[df[DATE_COL] >= cut].reset_index(drop=True)
    return train, test


# ----------------------------------------------------------------- preprocess


@dataclass
class Preprocessor:
    """支持 raw / zscore / standard 三种缩放（预报告 §4.6 / §6.2 Table 4）。

    - raw:      不做缩放（GBDT 默认）
    - zscore:   按特征列做 (x - mean) / std，统计量在训练集 fit
    - standard: zscore + 用 ±3σ 截断（更鲁棒，用于线性 / DL）
    """

    mode: str = "raw"
    feature_cols: list[str] | None = None
    fillna: bool = True

    def __post_init__(self):
        self._mean: pd.Series | None = None
        self._std: pd.Series | None = None
        self._median: pd.Series | None = None
        self._industry_codes: dict[str, int] = {}

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        if self.feature_cols is None:
            raise ValueError("feature_cols must be set before fit()")
        df, codes = encode_categorical(df)
        self._industry_codes = codes
        sub = df[self.feature_cols].astype("float64")
        self._median = sub.median()
        if self.mode in ("zscore", "standard"):
            self._mean = sub.mean()
            std = sub.std(ddof=0).replace(0.0, 1.0)
            self._std = std
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df, _ = encode_categorical(df, industry_codes=self._industry_codes)
        sub = df[self.feature_cols].astype("float32")
        if self.fillna and self._median is not None:
            sub = sub.fillna(self._median.astype("float32"))
        if self.mode == "raw":
            out = sub
        elif self.mode == "zscore":
            out = (sub - self._mean.astype("float32")) / self._std.astype("float32")
        elif self.mode == "standard":
            z = (sub - self._mean.astype("float32")) / self._std.astype("float32")
            out = z.clip(-3.0, 3.0)
        else:
            raise ValueError(f"unknown mode: {self.mode}")
        return out.astype("float32")

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


# ----------------------------------------------------------------- bundle


@dataclass
class DataBundle:
    """训练 / 测试两端的特征矩阵、标签、未来 5 日收益和元信息（date / ticker）。"""

    X_train: pd.DataFrame
    y_train: np.ndarray
    meta_train: pd.DataFrame   # [date, ticker, r_future_5]
    X_test: pd.DataFrame
    y_test: np.ndarray
    meta_test: pd.DataFrame
    feature_cols: list[str]
    preprocessor: Preprocessor


def prepare_data(
    feature_groups: tuple[str, ...] = DEFAULT_GROUPS,
    preprocess: str = "raw",
    sample_tickers: int | None = None,
    seed: int = 42,
) -> DataBundle:
    """一站式：装载 → 标签 → 切分 → 预处理 → 打包。

    sample_tickers 不为 None 时，从全市场随机抽取 N 只股票（用于调试）。
    """
    df = load_dataset()
    df = build_label(df)

    if sample_tickers is not None:
        rng = np.random.default_rng(seed)
        all_tickers = df[TICKER_COL].unique()
        chosen = rng.choice(all_tickers, size=min(sample_tickers, len(all_tickers)),
                            replace=False)
        df = df[df[TICKER_COL].isin(chosen)].reset_index(drop=True)

    # 训练样本要有有效未来收益；测试样本最后 5 天没有 r_future_5 → 同样过滤
    df = df.dropna(subset=[TARGET_RET_COL]).reset_index(drop=True)

    train, test = split_train_test(df)
    feat_cols = get_feature_columns(feature_groups)

    pre = Preprocessor(mode=preprocess, feature_cols=feat_cols)
    X_train = pre.fit_transform(train)
    X_test = pre.transform(test)

    meta_cols = [DATE_COL, TICKER_COL, TARGET_RET_COL]
    return DataBundle(
        X_train=X_train,
        y_train=train[LABEL_COL].to_numpy(dtype="int8"),
        meta_train=train[meta_cols].reset_index(drop=True),
        X_test=X_test,
        y_test=test[LABEL_COL].to_numpy(dtype="int8"),
        meta_test=test[meta_cols].reset_index(drop=True),
        feature_cols=feat_cols,
        preprocessor=pre,
    )
