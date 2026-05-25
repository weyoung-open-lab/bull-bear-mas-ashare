"""
run_main_compare.py — 论文 Table 1（主对比）跑批入口（预报告 §8.1）。

按家族（factor / linear / gbdt / tabular_dl / sequence）逐个训练 → 评估 → 回测，
最终汇总到：
    results/main_compare_<timestamp>/metrics_summary.csv
    results/main_compare_<timestamp>/backtest_summary.csv
    results/main_compare_<timestamp>/predictions/<model>.parquet
    results/main_compare_<timestamp>/config_snapshot.yaml

CLI（PowerShell）：
    python -X utf8 -m experiments.run_main_compare \
        --models LightGBM-std,XGBoost,Momentum-5d \
        --sample-tickers 800
    # 或：
    python -X utf8 -m experiments.run_main_compare --families factor,linear,gbdt
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config import RESULTS_DIR, TARGET_RET_COL
from src.backtest import backtest_all_fracs
from src.data import DataBundle, prepare_data
from src.features import DEFAULT_GROUPS
from src.metrics import evaluate
from src.models import (
    ALL_MODELS,
    CROSS_SECTION_MODELS,
    MODEL_REGISTRY,
    SEQUENCE_MODELS,
    build,
)
from src.models.sequence import SequenceWindowBuilder


# ---------------------------------------------------------------- helpers

def _select_models(args) -> list[type]:
    if args.models:
        names = [n.strip() for n in args.models.split(",") if n.strip()]
        unknown = [n for n in names if n not in MODEL_REGISTRY]
        if unknown:
            raise SystemExit(f"unknown models: {unknown}")
        chosen = [MODEL_REGISTRY[n] for n in names]
    elif args.families:
        fams = {f.strip() for f in args.families.split(",") if f.strip()}
        chosen = [m for m in ALL_MODELS if m.family in fams]
    else:
        chosen = list(ALL_MODELS)

    if args.skip_dl:
        chosen = [m for m in chosen if m.family not in ("tabular_dl", "sequence")]
    return chosen


def _bundle_for_preprocess(cache: dict, mode: str, args) -> DataBundle:
    if mode not in cache:
        print(f"\n=== prepare_data(preprocess={mode}) ===", flush=True)
        cache[mode] = prepare_data(
            feature_groups=DEFAULT_GROUPS,
            preprocess=mode,
            sample_tickers=args.sample_tickers,
        )
        b = cache[mode]
        print(f"   X_train={b.X_train.shape}  X_test={b.X_test.shape}  "
              f"y_train_pos={b.y_train.mean():.4f}", flush=True)
    return cache[mode]


def _row_for_metric(model_cls, metric, runtime: float) -> dict:
    row = {
        "model": model_cls.name,
        "family": model_cls.family,
        "preprocess": model_cls.preprocess,
        "fit_predict_sec": round(runtime, 2),
    }
    row.update(metric.to_row())
    return row


def _row_for_backtest(model_cls, backtests: dict) -> list[dict]:
    rows = []
    for frac, bt in backtests.items():
        rows.append({
            "model": model_cls.name,
            "family": model_cls.family,
            "top_frac": frac,
            "annual_return": bt.annual_return,
            "annual_volatility": bt.annual_volatility,
            "sharpe": bt.sharpe,
            "max_drawdown": bt.max_drawdown,
            "avg_turnover": bt.avg_turnover,
            "n_days": bt.n_days,
        })
    return rows


# ---------------------------------------------------------------- 跑批

def _make_regression_target(meta_train: pd.DataFrame) -> np.ndarray:
    y = meta_train[TARGET_RET_COL].to_numpy(dtype="float32")
    lo = float(np.nanquantile(y, 0.001))
    hi = float(np.nanquantile(y, 0.999))
    return np.clip(y, lo, hi).astype("float32")


def _run_cross_section(model_cls, cache, args, out_dir: Path) -> tuple[dict, list[dict]] | None:
    bundle = _bundle_for_preprocess(cache, model_cls.preprocess, args)
    model = build(model_cls.name)
    t0 = time.time()
    if model.needs_fit:
        if getattr(model_cls, "regression_target", False):
            y_train = _make_regression_target(bundle.meta_train)
        else:
            y_train = bundle.y_train
        model.fit(bundle.X_train, y_train)
    pred = model.predict_proba(bundle.X_test)
    runtime = time.time() - t0

    metric = evaluate(bundle.y_test, pred, bundle.meta_test)
    backtests = backtest_all_fracs(bundle.meta_test, pred)

    # 保存预测
    pred_df = bundle.meta_test.copy()
    pred_df["pred"] = pred
    pred_df.to_parquet(out_dir / "predictions" / f"{model_cls.name}.parquet", index=False)

    return _row_for_metric(model_cls, metric, runtime), _row_for_backtest(model_cls, backtests)


def _run_sequence(model_cls, args, out_dir: Path) -> tuple[dict, list[dict]] | None:
    """序列模型独立加载（需要原始 panel 而非 X 矩阵）。"""
    from src.data import build_label, load_dataset, split_train_test, Preprocessor
    from src.features import get_feature_columns

    print(f"\n=== sequence prepare for {model_cls.name} ===", flush=True)
    panel = load_dataset()
    panel = build_label(panel)
    panel = panel.dropna(subset=["r_future_5"])

    if args.sample_tickers:
        rng = np.random.default_rng(42)
        all_t = panel["ticker"].unique()
        chosen = rng.choice(all_t, size=min(args.sample_tickers, len(all_t)), replace=False)
        panel = panel[panel["ticker"].isin(chosen)].reset_index(drop=True)

    feat_cols = get_feature_columns(DEFAULT_GROUPS)
    pre = Preprocessor(mode=model_cls.preprocess, feature_cols=feat_cols)
    train_panel, test_panel = split_train_test(panel)
    pre.fit(train_panel)
    train_panel.loc[:, feat_cols] = pre.transform(train_panel).to_numpy()
    test_panel.loc[:, feat_cols] = pre.transform(test_panel).to_numpy()

    builder = SequenceWindowBuilder(feature_cols=feat_cols, window=10)
    print("  building train sequences ...", flush=True)
    Xtr, ytr, meta_tr = builder.build(train_panel, max_per_ticker=args.seq_max_per_ticker_train)
    print(f"   train seq: X={Xtr.shape}  y_pos={ytr.mean():.4f}", flush=True)
    print("  building test sequences ...", flush=True)
    Xte, yte, meta_te = builder.build(test_panel, max_per_ticker=args.seq_max_per_ticker_test)
    print(f"   test  seq: X={Xte.shape}  y_pos={yte.mean():.4f}", flush=True)

    if Xtr.size == 0 or Xte.size == 0:
        print("  [skip] empty sequences")
        return None

    model = build(model_cls.name)
    t0 = time.time()
    if getattr(model_cls, "regression_target", False):
        y_train = meta_tr[TARGET_RET_COL].to_numpy(dtype="float32")
        lo, hi = float(np.quantile(y_train, 0.001)), float(np.quantile(y_train, 0.999))
        y_train = np.clip(y_train, lo, hi)
        model.fit(Xtr, y_train)
    else:
        model.fit(Xtr, ytr)
    pred = model.predict_proba(Xte)
    runtime = time.time() - t0

    metric = evaluate(yte, pred, meta_te)
    backtests = backtest_all_fracs(meta_te, pred)

    pred_df = meta_te.copy()
    pred_df["pred"] = pred
    pred_df.to_parquet(out_dir / "predictions" / f"{model_cls.name}.parquet", index=False)

    return _row_for_metric(model_cls, metric, runtime), _row_for_backtest(model_cls, backtests)


# ---------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default="",
                        help="逗号分隔的模型名（与 --families 二选一）")
    parser.add_argument("--families", type=str, default="",
                        help="逗号分隔的家族名 factor/linear/gbdt/tabular_dl/sequence")
    parser.add_argument("--skip-dl", action="store_true", help="跳过 DL（tabular + sequence）")
    parser.add_argument("--sample-tickers", type=int, default=None,
                        help="抽样 N 只股票（调试用）")
    parser.add_argument("--seq-max-per-ticker-train", type=int, default=400)
    parser.add_argument("--seq-max-per-ticker-test", type=int, default=200)
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    chosen = _select_models(args)
    print(f"selected models: {[m.name for m in chosen]}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.tag}" if args.tag else ""
    out_dir: Path = RESULTS_DIR / f"main_compare_{ts}{suffix}"
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)

    # 配置快照
    snapshot = {
        "timestamp": ts,
        "models": [m.name for m in chosen],
        "feature_groups": list(DEFAULT_GROUPS),
        "sample_tickers": args.sample_tickers,
    }
    (out_dir / "config_snapshot.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    cache: dict = {}
    metric_rows: list[dict] = []
    backtest_rows: list[dict] = []

    for cls in chosen:
        print(f"\n>>> {cls.name} ({cls.family}, preprocess={cls.preprocess})", flush=True)
        try:
            if cls.family == "sequence":
                result = _run_sequence(cls, args, out_dir)
            else:
                result = _run_cross_section(cls, cache, args, out_dir)
            if result is None:
                continue
            mrow, brows = result
            metric_rows.append(mrow)
            backtest_rows.extend(brows)
            print(f"    AUC={mrow['auc']:.4f}  RankIC={mrow['rankic_mean']:.4f}  "
                  f"RankICIR={mrow['rankicir']:.3f}  "
                  f"top1%_ret={mrow.get('top1pct_ret', np.nan):.4f}",
                  flush=True)
        except Exception as e:
            print(f"   [FAIL] {cls.name}: {type(e).__name__}: {e}")
            traceback.print_exc()

    # 汇总
    if metric_rows:
        metrics_df = pd.DataFrame(metric_rows).sort_values("rankicir", ascending=False)
        metrics_df.to_csv(out_dir / "metrics_summary.csv", index=False, encoding="utf-8-sig")
        print(f"\nmetrics_summary.csv -> {out_dir/'metrics_summary.csv'}")

    if backtest_rows:
        bt_df = pd.DataFrame(backtest_rows).sort_values(["model", "top_frac"])
        bt_df.to_csv(out_dir / "backtest_summary.csv", index=False, encoding="utf-8-sig")
        print(f"backtest_summary.csv -> {out_dir/'backtest_summary.csv'}")

    print(f"\nDone. results: {out_dir}")


if __name__ == "__main__":
    main()
