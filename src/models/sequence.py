"""序列深度学习（预报告 §4.5 Table 10）。

ALSTM (Attention-LSTM) 和 TCN：以 10 日回看窗口构建序列输入。
预测时只输出窗口最后一日（即 meta 中 (date, ticker) 那一天）的概率。

为节省内存，序列以 (ticker, date) 流式构造：
    SequenceWindowBuilder.build(panel_df, dates_to_use)
    -> X_seq: float32 [N, T, F]
       y:     int8    [N]
       meta:  DataFrame[date, ticker, r_future_5]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import (
    DATE_COL,
    LABEL_COL,
    RANDOM_SEED,
    TARGET_RET_COL,
    TICKER_COL,
)
from src.models.base import BaseModel


# ----------------------------------------------------------------- 序列构造器

@dataclass
class SequenceWindowBuilder:
    feature_cols: list[str]
    window: int = 10

    def build(
        self, panel: pd.DataFrame, max_per_ticker: int | None = None, seed: int = RANDOM_SEED
    ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        panel 必须包含 [date, ticker, ...features..., label, r_future_5]，
        已按 (date, ticker) 有序。返回 (X_seq, y, meta)。

        每只 ticker 内按时间滑动 → 长度 window 的输入，最后一天提供 label 和 r_future_5。
        max_per_ticker 限制每只股票抽取的样本数（用于轻量训练）。
        """
        X_list: list[np.ndarray] = []
        y_list: list[int] = []
        date_list: list[pd.Timestamp] = []
        ticker_list: list[str] = []
        ret_list: list[float] = []

        rng = np.random.default_rng(seed)
        feats = self.feature_cols
        T = self.window

        sub = panel[[DATE_COL, TICKER_COL, LABEL_COL, TARGET_RET_COL] + feats].copy()
        # 按 ticker 分组取连续 window
        for tkr, g in sub.groupby(TICKER_COL, sort=False):
            g = g.sort_values(DATE_COL)
            arr = g[feats].to_numpy(dtype="float32")
            n = len(g)
            if n < T:
                continue
            valid_end_idx = np.arange(T - 1, n)
            # label / target / date 取窗口末日
            labels = g[LABEL_COL].to_numpy()[valid_end_idx]
            rets = g[TARGET_RET_COL].to_numpy()[valid_end_idx]
            dates = g[DATE_COL].to_numpy()[valid_end_idx]

            mask = ~np.isnan(rets)
            valid_end_idx = valid_end_idx[mask]
            labels = labels[mask]
            rets = rets[mask]
            dates = dates[mask]

            if max_per_ticker is not None and len(valid_end_idx) > max_per_ticker:
                pick = rng.choice(len(valid_end_idx), size=max_per_ticker, replace=False)
                valid_end_idx = valid_end_idx[pick]
                labels = labels[pick]
                rets = rets[pick]
                dates = dates[pick]

            for j, end in enumerate(valid_end_idx):
                X_list.append(arr[end - T + 1: end + 1])
                y_list.append(int(labels[j]))
                date_list.append(pd.Timestamp(dates[j]))
                ticker_list.append(tkr)
                ret_list.append(float(rets[j]))

        if not X_list:
            X = np.empty((0, T, len(feats)), dtype="float32")
        else:
            X = np.stack(X_list).astype("float32")
        y = np.asarray(y_list, dtype="int8")
        meta = pd.DataFrame({
            DATE_COL: pd.to_datetime(date_list),
            TICKER_COL: ticker_list,
            TARGET_RET_COL: ret_list,
        })
        # 用 median 填 NaN（避免 NaN 进网络）
        if X.size:
            med = np.nanmedian(X.reshape(-1, X.shape[-1]), axis=0)
            inds = np.where(np.isnan(X))
            if len(inds[0]):
                X[inds] = np.take(med, inds[2])
        return X, y, meta


# ----------------------------------------------------------------- 网络

def _build_alstm_module(n_features: int, hidden: int = 64, dropout: float = 0.1):
    import torch
    from torch import nn

    class ALSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(n_features, hidden, num_layers=2, batch_first=True,
                                dropout=dropout)
            self.attn = nn.Sequential(
                nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1),
            )
            self.head = nn.Sequential(
                nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )

        def forward(self, x):                 # x: [B, T, F]
            h, _ = self.lstm(x)               # [B, T, H]
            w = self.attn(h).softmax(dim=1)   # [B, T, 1]
            ctx = (h * w).sum(dim=1)          # [B, H]
            return self.head(ctx).squeeze(-1)

    return ALSTM()


def _build_tcn_module(n_features: int, channels=(64, 64, 64), kernel: int = 3,
                      dropout: float = 0.1):
    import torch
    from torch import nn

    class TemporalBlock(nn.Module):
        def __init__(self, in_c, out_c, dil):
            super().__init__()
            pad = (kernel - 1) * dil
            self.pad = pad
            self.conv1 = nn.Conv1d(in_c, out_c, kernel, padding=pad, dilation=dil)
            self.conv2 = nn.Conv1d(out_c, out_c, kernel, padding=pad, dilation=dil)
            self.drop = nn.Dropout(dropout)
            self.act = nn.ReLU()
            self.proj = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

        def forward(self, x):
            res = self.proj(x)
            o = self.conv1(x)[:, :, :-self.pad] if self.pad else self.conv1(x)
            o = self.act(o); o = self.drop(o)
            o = self.conv2(o)[:, :, :-self.pad] if self.pad else self.conv2(o)
            o = self.act(o); o = self.drop(o)
            return self.act(o + res)

    class TCN(nn.Module):
        def __init__(self):
            super().__init__()
            blocks = []
            in_c = n_features
            for i, out_c in enumerate(channels):
                blocks.append(TemporalBlock(in_c, out_c, dil=2 ** i))
                in_c = out_c
            self.net = nn.Sequential(*blocks)
            self.head = nn.Linear(channels[-1], 1)

        def forward(self, x):                 # x: [B, T, F]
            o = self.net(x.transpose(1, 2))   # [B, C, T]
            return self.head(o[:, :, -1]).squeeze(-1)

    return TCN()


# ----------------------------------------------------------------- 通用训练逻辑

class _SeqClf(BaseModel):
    family = "sequence"
    preprocess = "zscore"

    def __init__(self, lr: float = 1e-3, weight_decay: float = 1e-5,
                 batch_size: int = 1024, max_epochs: int = 8, patience: int = 2):
        super().__init__(lr=lr, weight_decay=weight_decay,
                         batch_size=batch_size, max_epochs=max_epochs, patience=patience)
        try:
            import torch  # noqa: F401
        except ImportError as e:
            raise ImportError("Sequence model requires torch") from e
        self._cfg = self.params
        self._model = None
        self._device = None
        self._n_features = None

    def _build_module(self, n_features: int):  # 子类实现
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray, X_val=None, y_val=None, **kw):
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        torch.manual_seed(RANDOM_SEED)

        self._n_features = X.shape[-1]
        net = self._build_module(self._n_features).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=self._cfg["lr"],
                                weight_decay=self._cfg["weight_decay"])
        loss_fn = nn.BCEWithLogitsLoss()

        Xt = torch.from_numpy(X).float()
        yt = torch.from_numpy(np.asarray(y).astype("float32"))
        ds = TensorDataset(Xt, yt)
        dl = DataLoader(ds, batch_size=self._cfg["batch_size"], shuffle=True,
                        num_workers=0)

        if X_val is not None:
            Xv = torch.from_numpy(X_val).float().to(device)
            yv = torch.from_numpy(np.asarray(y_val).astype("float32")).to(device)
        else:
            Xv = yv = None

        best, bad = float("inf"), 0
        best_state = None
        for ep in range(self._cfg["max_epochs"]):
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
                    vl = float(loss_fn(net(Xv), yv).item())
                if vl < best - 1e-4:
                    best = vl
                    best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                    if bad >= self._cfg["patience"]:
                        break
        if best_state is not None:
            net.load_state_dict(best_state)
        self._model = net
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch
        net = self._model
        net.eval()
        device = self._device
        Xt = torch.from_numpy(X).float().to(device)
        out = []
        bs = self._cfg["batch_size"]
        with torch.no_grad():
            for i in range(0, len(Xt), bs):
                p = torch.sigmoid(net(Xt[i:i + bs])).cpu().numpy()
                out.append(p)
        return np.concatenate(out)


class ALSTMClf(_SeqClf):
    name = "ALSTM"

    def __init__(self, hidden: int = 64, **kw):
        super().__init__(**kw)
        self._hidden = hidden

    def _build_module(self, n_features: int):
        return _build_alstm_module(n_features, hidden=self._hidden)


class TCNClf(_SeqClf):
    name = "TCN"

    def __init__(self, channels=(64, 64, 64), **kw):
        super().__init__(**kw)
        self._channels = tuple(channels)

    def _build_module(self, n_features: int):
        return _build_tcn_module(n_features, channels=self._channels)


SEQUENCE_MODELS = [ALSTMClf, TCNClf]


# =================================================================
# Regression variants（MSE 损失版本）
# =================================================================

class _SeqReg(BaseModel):
    family = "sequence"
    preprocess = "zscore"
    regression_target = True

    def __init__(self, lr: float = 1e-3, weight_decay: float = 1e-5,
                 batch_size: int = 1024, max_epochs: int = 8, patience: int = 2):
        super().__init__(lr=lr, weight_decay=weight_decay,
                         batch_size=batch_size, max_epochs=max_epochs, patience=patience)
        self._cfg = self.params
        self._model = None
        self._device = None
        self._n_features = None

    def _build_module(self, n_features: int):
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray, X_val=None, y_val=None, **kw):
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        torch.manual_seed(RANDOM_SEED)
        self._n_features = X.shape[-1]
        net = self._build_module(self._n_features).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=self._cfg["lr"],
                                weight_decay=self._cfg["weight_decay"])
        loss_fn = nn.MSELoss()

        Xt = torch.from_numpy(X).float()
        yt = torch.from_numpy(np.asarray(y).astype("float32"))
        ds = TensorDataset(Xt, yt)
        dl = DataLoader(ds, batch_size=self._cfg["batch_size"], shuffle=True, num_workers=0)

        if X_val is not None:
            Xv = torch.from_numpy(X_val).float().to(device)
            yv = torch.from_numpy(np.asarray(y_val).astype("float32")).to(device)
        else:
            Xv = yv = None

        best, bad, best_state = float("inf"), 0, None
        for ep in range(self._cfg["max_epochs"]):
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
                    vl = float(loss_fn(net(Xv), yv).item())
                if vl < best - 1e-6:
                    best = vl
                    best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                    if bad >= self._cfg["patience"]:
                        break
        if best_state is not None:
            net.load_state_dict(best_state)
        self._model = net
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch
        net = self._model
        net.eval()
        device = self._device
        Xt = torch.from_numpy(X).float().to(device)
        out = []
        bs = self._cfg["batch_size"]
        with torch.no_grad():
            for i in range(0, len(Xt), bs):
                out.append(net(Xt[i:i + bs]).cpu().numpy())
        return np.concatenate(out)


class ALSTMReg(_SeqReg):
    name = "ALSTM-reg"

    def __init__(self, hidden: int = 64, **kw):
        super().__init__(**kw)
        self._hidden = hidden

    def _build_module(self, n_features: int):
        return _build_alstm_module(n_features, hidden=self._hidden)


class TCNReg(_SeqReg):
    name = "TCN-reg"

    def __init__(self, channels=(64, 64, 64), **kw):
        super().__init__(**kw)
        self._channels = tuple(channels)

    def _build_module(self, n_features: int):
        return _build_tcn_module(n_features, channels=self._channels)


SEQUENCE_REG_MODELS = [ALSTMReg, TCNReg]
