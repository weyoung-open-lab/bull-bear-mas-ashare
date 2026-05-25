"""模型注册表（预报告 §4 全矩阵）。

ALL_MODELS：5 大类 14 个变体（factor 3 + linear 1 + gbdt 6 + tabular_dl 2 + sequence 2）
"""

from src.models.base import BaseModel
from src.models.factor import FACTOR_MODELS
from src.models.gbdt import GBDT_MODELS, GBDT_REG_MODELS
from src.models.linear import LINEAR_MODELS, LINEAR_REG_MODELS

# Tabular / sequence DL 延迟导入：避免没装 torch 等依赖时失败
try:
    from src.models.tabular_dl import TABULAR_DL_MODELS, TABULAR_DL_REG_MODELS
except Exception:                              # noqa: BLE001
    TABULAR_DL_MODELS: list = []
    TABULAR_DL_REG_MODELS: list = []

try:
    from src.models.sequence import SEQUENCE_MODELS, SEQUENCE_REG_MODELS
except Exception:                              # noqa: BLE001
    SEQUENCE_MODELS: list = []
    SEQUENCE_REG_MODELS: list = []


CROSS_SECTION_MODELS = (
    FACTOR_MODELS + LINEAR_MODELS + GBDT_MODELS + TABULAR_DL_MODELS
    + LINEAR_REG_MODELS + GBDT_REG_MODELS + TABULAR_DL_REG_MODELS
)
ALL_MODELS = CROSS_SECTION_MODELS + SEQUENCE_MODELS + SEQUENCE_REG_MODELS

# 名称索引
MODEL_REGISTRY: dict[str, type[BaseModel]] = {cls.name: cls for cls in ALL_MODELS}


def list_models(family: str | None = None) -> list[type[BaseModel]]:
    if family is None:
        return list(ALL_MODELS)
    return [m for m in ALL_MODELS if m.family == family]


def build(name: str, **kw) -> BaseModel:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model: {name}. available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kw)


__all__ = [
    "BaseModel", "ALL_MODELS", "CROSS_SECTION_MODELS", "MODEL_REGISTRY",
    "FACTOR_MODELS", "LINEAR_MODELS", "GBDT_MODELS", "TABULAR_DL_MODELS", "SEQUENCE_MODELS",
    "list_models", "build",
]
