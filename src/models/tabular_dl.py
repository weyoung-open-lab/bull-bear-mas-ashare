"""Tabular 深度学习（预报告 §4.4 Table 9）。

- TabNet：稀疏注意力 + 内置 attention mask，DeepExplainer 兼容。
- FT-Transformer：把每个特征 tokenize 后用 Transformer 建模特征交互。

依赖（按需安装）：
    pip install pytorch-tabnet
    （FT-Transformer 用本文件内置实现，仅需 torch）
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from config import RANDOM_SEED
from src.models.base import BaseModel


# ----------------------------------------------------------------- TabNet

class TabNetClf(BaseModel):
    name = "TabNet"
    family = "tabular_dl"
    preprocess = "zscore"

    def __init__(self, n_d: int = 32, n_a: int = 32, n_steps: int = 3,
                 max_epochs: int = 30, batch_size: int = 8192, patience: int = 5):
        super().__init__(n_d=n_d, n_a=n_a, n_steps=n_steps,
                         max_epochs=max_epochs, batch_size=batch_size, patience=patience)
        try:
            from pytorch_tabnet.tab_model import TabNetClassifier
        except ImportError as e:
            raise ImportError(
                "TabNet 需要先安装：pip install pytorch-tabnet"
            ) from e
        self._TabNetClassifier = TabNetClassifier
        self._model = TabNetClassifier(
            n_d=n_d, n_a=n_a, n_steps=n_steps,
            seed=RANDOM_SEED, verbose=0,
        )
        self._max_epochs = max_epochs
        self._batch_size = batch_size
        self._patience = patience

    def fit(self, X: pd.DataFrame, y: np.ndarray, X_val=None, y_val=None, **kw):
        Xn = X.to_numpy(dtype="float32")
        yn = np.asarray(y).astype("int64")
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val.to_numpy(dtype="float32"), np.asarray(y_val).astype("int64"))]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(
                Xn, yn,
                eval_set=eval_set,
                max_epochs=self._max_epochs,
                batch_size=self._batch_size,
                virtual_batch_size=min(1024, self._batch_size // 8),
                patience=self._patience,
            )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X.to_numpy(dtype="float32"))[:, 1]


# ----------------------------------------------------------------- FT-Transformer (内置最简实现)

class _FTTransformerNet:
    """延迟导入，避免无 torch 时的 ImportError。"""


def _build_ft_transformer_module(n_features: int, d_token: int = 64, n_blocks: int = 3,
                                  n_heads: int = 8, dropout: float = 0.1):
    import torch
    from torch import nn

    class FeatureTokenizer(nn.Module):
        def __init__(self, n_features: int, d_token: int):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(n_features, d_token) * 0.02)
            self.bias = nn.Parameter(torch.zeros(n_features, d_token))
            self.cls = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: [B, F]  ->  [B, F+1, d_token]
            tokens = x.unsqueeze(-1) * self.weight + self.bias
            cls = self.cls.expand(x.size(0), 1, -1)
            return torch.cat([cls, tokens], dim=1)

    class TransformerBlock(nn.Module):
        def __init__(self, d_token: int, n_heads: int, dropout: float):
            super().__init__()
            self.attn = nn.MultiheadAttention(d_token, n_heads, dropout=dropout,
                                              batch_first=True)
            self.ln1 = nn.LayerNorm(d_token)
            self.ff = nn.Sequential(
                nn.Linear(d_token, d_token * 2), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(d_token * 2, d_token),
            )
            self.ln2 = nn.LayerNorm(d_token)
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            h = self.ln1(x)
            a, _ = self.attn(h, h, h, need_weights=False)
            x = x + self.drop(a)
            x = x + self.drop(self.ff(self.ln2(x)))
            return x

    class FTTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = FeatureTokenizer(n_features, d_token)
            self.blocks = nn.ModuleList(
                [TransformerBlock(d_token, n_heads, dropout) for _ in range(n_blocks)]
            )
            self.head = nn.Sequential(
                nn.LayerNorm(d_token), nn.Linear(d_token, 1),
            )

        def forward(self, x):
            t = self.tok(x)
            for blk in self.blocks:
                t = blk(t)
            cls = t[:, 0]
            return self.head(cls).squeeze(-1)

    return FTTransformer()


class FTTransformerClf(BaseModel):
    name = "FT-Transformer"
    family = "tabular_dl"
    preprocess = "standard"

    def __init__(self, d_token: int = 64, n_blocks: int = 3, n_heads: int = 8,
                 dropout: float = 0.1, lr: float = 1e-3, weight_decay: float = 1e-5,
                 batch_size: int = 4096, max_epochs: int = 12, patience: int = 3):
        super().__init__(d_token=d_token, n_blocks=n_blocks, n_heads=n_heads,
                         dropout=dropout, lr=lr, weight_decay=weight_decay,
                         batch_size=batch_size, max_epochs=max_epochs, patience=patience)
        try:
            import torch  # noqa: F401
        except ImportError as e:
            raise ImportError("FT-Transformer 需要 torch") from e
        self._cfg = self.params
        self._model = None
        self._device = None

    def fit(self, X: pd.DataFrame, y: np.ndarray, X_val=None, y_val=None, **kw):
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device

        cfg = self._cfg
        torch.manual_seed(RANDOM_SEED)
        net = _build_ft_transformer_module(
            n_features=X.shape[1],
            d_token=cfg["d_token"], n_blocks=cfg["n_blocks"],
            n_heads=cfg["n_heads"], dropout=cfg["dropout"],
        ).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"],
                                weight_decay=cfg["weight_decay"])
        loss_fn = nn.BCEWithLogitsLoss()

        Xt = torch.from_numpy(X.to_numpy(dtype="float32"))
        yt = torch.from_numpy(np.asarray(y).astype("float32"))
        ds = TensorDataset(Xt, yt)
        dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True,
                        num_workers=0, drop_last=False)

        if X_val is not None and y_val is not None:
            Xv = torch.from_numpy(X_val.to_numpy(dtype="float32")).to(device)
            yv = torch.from_numpy(np.asarray(y_val).astype("float32")).to(device)
        else:
            Xv = yv = None

        best_loss = float("inf")
        best_state = None
        bad = 0
        for ep in range(cfg["max_epochs"]):
            net.train()
            for xb, yb in dl:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                opt.zero_grad()
                logit = net(xb)
                loss = loss_fn(logit, yb)
                loss.backward()
                opt.step()
            if Xv is not None:
                net.eval()
                with torch.no_grad():
                    val_loss = float(loss_fn(net(Xv), yv).item())
                if val_loss < best_loss - 1e-4:
                    best_loss = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                    if bad >= cfg["patience"]:
                        break
        if best_state is not None:
            net.load_state_dict(best_state)
        self._model = net
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        import torch
        net = self._model
        net.eval()
        device = self._device
        Xt = torch.from_numpy(X.to_numpy(dtype="float32")).to(device)
        outs = []
        bs = self._cfg["batch_size"]
        with torch.no_grad():
            for i in range(0, len(Xt), bs):
                logit = net(Xt[i:i + bs])
                outs.append(torch.sigmoid(logit).cpu().numpy())
        return np.concatenate(outs)


TABULAR_DL_MODELS = [TabNetClf, FTTransformerClf]


# =================================================================
# Regression variants
# =================================================================

class TabNetReg(BaseModel):
    name = "TabNet-reg"
    family = "tabular_dl"
    preprocess = "zscore"
    regression_target = True

    def __init__(self, n_d: int = 32, n_a: int = 32, n_steps: int = 3,
                 max_epochs: int = 30, batch_size: int = 8192, patience: int = 5):
        super().__init__(n_d=n_d, n_a=n_a, n_steps=n_steps,
                         max_epochs=max_epochs, batch_size=batch_size, patience=patience)
        try:
            from pytorch_tabnet.tab_model import TabNetRegressor
        except ImportError as e:
            raise ImportError("TabNet 需要先安装：pip install pytorch-tabnet") from e
        self._model = TabNetRegressor(n_d=n_d, n_a=n_a, n_steps=n_steps,
                                       seed=RANDOM_SEED, verbose=0)
        self._max_epochs = max_epochs
        self._batch_size = batch_size
        self._patience = patience

    def fit(self, X: pd.DataFrame, y: np.ndarray, X_val=None, y_val=None, **kw):
        Xn = X.to_numpy(dtype="float32")
        yn = np.asarray(y).astype("float32").reshape(-1, 1)
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val.to_numpy(dtype="float32"),
                          np.asarray(y_val).astype("float32").reshape(-1, 1))]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(Xn, yn, eval_set=eval_set,
                            max_epochs=self._max_epochs,
                            batch_size=self._batch_size,
                            virtual_batch_size=min(1024, self._batch_size // 8),
                            patience=self._patience)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X.to_numpy(dtype="float32")).flatten()


class FTTransformerReg(BaseModel):
    """FT-Transformer 回归版：MSE 损失，无 sigmoid。"""
    name = "FT-Transformer-reg"
    family = "tabular_dl"
    preprocess = "standard"
    regression_target = True

    def __init__(self, d_token: int = 64, n_blocks: int = 3, n_heads: int = 8,
                 dropout: float = 0.1, lr: float = 1e-3, weight_decay: float = 1e-5,
                 batch_size: int = 4096, max_epochs: int = 12, patience: int = 3):
        super().__init__(d_token=d_token, n_blocks=n_blocks, n_heads=n_heads,
                         dropout=dropout, lr=lr, weight_decay=weight_decay,
                         batch_size=batch_size, max_epochs=max_epochs, patience=patience)
        try:
            import torch  # noqa: F401
        except ImportError as e:
            raise ImportError("FT-Transformer 需要 torch") from e
        self._cfg = self.params
        self._model = None
        self._device = None

    def fit(self, X: pd.DataFrame, y: np.ndarray, X_val=None, y_val=None, **kw):
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        cfg = self._cfg
        torch.manual_seed(RANDOM_SEED)
        net = _build_ft_transformer_module(
            n_features=X.shape[1],
            d_token=cfg["d_token"], n_blocks=cfg["n_blocks"],
            n_heads=cfg["n_heads"], dropout=cfg["dropout"],
        ).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"],
                                weight_decay=cfg["weight_decay"])
        loss_fn = nn.MSELoss()

        Xt = torch.from_numpy(X.to_numpy(dtype="float32"))
        yt = torch.from_numpy(np.asarray(y).astype("float32"))
        ds = TensorDataset(Xt, yt)
        dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)

        if X_val is not None and y_val is not None:
            Xv = torch.from_numpy(X_val.to_numpy(dtype="float32")).to(device)
            yv = torch.from_numpy(np.asarray(y_val).astype("float32")).to(device)
        else:
            Xv = yv = None

        best, bad, best_state = float("inf"), 0, None
        for ep in range(cfg["max_epochs"]):
            net.train()
            for xb, yb in dl:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                opt.zero_grad()
                loss = loss_fn(net(xb), yb)
                loss.backward()
                opt.step()
            if Xv is not None:
                net.eval()
                with torch.no_grad():
                    val_loss = float(loss_fn(net(Xv), yv).item())
                if val_loss < best - 1e-6:
                    best = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                    if bad >= cfg["patience"]:
                        break
        if best_state is not None:
            net.load_state_dict(best_state)
        self._model = net
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        import torch
        net = self._model
        net.eval()
        device = self._device
        Xt = torch.from_numpy(X.to_numpy(dtype="float32")).to(device)
        outs = []
        bs = self._cfg["batch_size"]
        with torch.no_grad():
            for i in range(0, len(Xt), bs):
                outs.append(net(Xt[i:i + bs]).cpu().numpy())
        return np.concatenate(outs)


TABULAR_DL_REG_MODELS = [TabNetReg, FTTransformerReg]
