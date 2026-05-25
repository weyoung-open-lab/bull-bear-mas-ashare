"""检查 QMT 基本面数据的可用性。

数据源调查结论（在写本脚本前已经做过 file listing）：
  /day/  : 9690 个 .csv，仅 OHLCV，无基本面
  Table_all.csv (5484 行) : 单日快照，含 市盈(动)/市净率/净利润/总市值/流通值
  factors*.parquet         : 仅匿名 F1-F5 因子
  没有任何文件含 ROE / ROA / 净利润率 / 营收增长率 / 资产负债率 时间序列

本脚本做：
  1. 筛出 Table_all.csv 中属于我们主数据集 3876 ticker 的行
  2. 对每个可用字段 (PE / PB / 净利润 / 总市值) 报告：
       覆盖率 (非空 / 总数)
       cross-section std (这就是字面意思 — 字段在 5484 只股票上的离散度)
       与最近一天 r_future_5 的 Pearson 一次性 IC（无时间序列）
  3. 显式列出无法验证的字段（ROE/ROA/营收增长率/资产负债率）
"""

from __future__ import annotations

import sys
from pathlib import Path
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

QMT_ROOT = Path(r"D:/project1/pythonProject/QMT")
TABLE_ALL = QMT_ROOT / "data" / "data" / "tdx" / "Table_all.csv"
DATASET   = QMT_ROOT / "单股票每日态势识别" / "论文" / "dataset_model_baseline_longer_trend.parquet"


def normalize_ticker_for_dataset(code: str) -> str:
    """QMT 用大写 'SH688802'，我们的 dataset 用小写 'sh688802'。"""
    return str(code).lower()


def to_numeric(series, na_strings=("亏损", "--", "")) -> pd.Series:
    """把 string-mixed 列转 float；非数字标识转 NaN。"""
    s = series.astype(str).str.strip()
    s = s.replace(list(na_strings), np.nan)
    # 移除百分号
    s = s.str.rstrip("%")
    return pd.to_numeric(s, errors="coerce")


def main() -> None:
    print("=" * 78)
    print("Fundamental data availability check")
    print("=" * 78)

    # ===== 1. Dataset ticker universe =====
    print("\n[1/4] load main dataset + identify ticker universe ...")
    ds = pd.read_parquet(DATASET)
    ds["date"] = pd.to_datetime(ds["date"])
    n_tk = ds["ticker"].nunique()
    print(f"   main dataset: {len(ds):,} rows, {n_tk} unique tickers, "
          f"{ds['date'].min().date()} to {ds['date'].max().date()}")

    main_tickers = set(ds["ticker"].unique())

    # ===== 2. Table_all.csv =====
    print("\n[2/4] load Table_all.csv snapshot ...")
    tab = pd.read_csv(TABLE_ALL, encoding="utf-8")
    # 修剪列名空格
    tab.columns = [c.strip() for c in tab.columns]
    tab["ticker"] = tab["代码"].apply(normalize_ticker_for_dataset)
    n_snap = len(tab)
    n_in_main = tab["ticker"].isin(main_tickers).sum()
    print(f"   Table_all.csv: {n_snap:,} rows total, {n_in_main:,} match main "
          f"dataset (= {n_in_main/n_tk*100:.1f}% of {n_tk} main tickers)")

    # 只保留主数据集股票
    snap = tab[tab["ticker"].isin(main_tickers)].copy()

    # ===== 3. 字段诊断 =====
    print("\n[3/4] per-field diagnostics ...")
    print()

    # 把字符串列转数值
    snap["PE"]            = to_numeric(snap["市盈(动)"])
    snap["PB"]            = to_numeric(snap["市净率"])
    snap["NetProfit"]     = to_numeric(snap["净利润?"])
    snap["MarketCap"]     = to_numeric(snap["总市值"])
    snap["CirculatingVal"] = to_numeric(snap["流通值"])
    snap["Turnover"]       = to_numeric(snap["换手"])     # 当日换手率
    snap["AmplitudePct"]   = to_numeric(snap["振幅%"])    # 振幅
    snap["ChgRatePct"]     = to_numeric(snap["涨幅%"])

    # 用户原列表
    USER_ASKED = {
        "ROE":         None,    # not in snapshot
        "ROA":         None,
        "净利润率":     None,
        "营收增长率":   None,
        "市盈率(PE)":   "PE",
        "市净率(PB)":   "PB",
        "资产负债率":   None,
    }
    # 附加：snapshot 实际可用的非用户列
    EXTRA_AVAILABLE = {
        "净利润(绝对)": "NetProfit",
        "总市值":       "MarketCap",
        "流通值":       "CirculatingVal",
    }

    rows_diag = []

    # 计算与 r_future_5 的 snapshot IC：需要选一个参考日
    # Table_all.csv 是单日快照（具体哪天未知），假设它接近最新日期 → 用 dataset 最新可用日
    ref_date = ds["date"].max()
    print(f"   reference date for IC test (dataset latest): {ref_date.date()}")
    ds_ref = ds.loc[ds["date"] == ref_date, ["ticker", "r_future_5"]]
    n_ref = len(ds_ref)
    has_target = (~ds_ref["r_future_5"].isna()).sum()
    print(f"   reference rows: {n_ref}, with r_future_5: {has_target}")
    if has_target < 100:
        # 退而求其次：用最后一个有 r_future_5 非空的日期
        valid = ds.loc[~ds["r_future_5"].isna(), "date"]
        if len(valid) > 0:
            ref_date = valid.max()
            ds_ref = ds.loc[ds["date"] == ref_date, ["ticker", "r_future_5"]]
            print(f"   -> fallback to last valid r_future_5 date: {ref_date.date()}  "
                  f"rows={len(ds_ref)}")

    # 合并 snapshot + reference-day r_future_5
    joined = snap.merge(ds_ref, on="ticker", how="left")

    print()
    print("  User-requested fundamentals:")
    print(f"  {'Field':18s}  {'Snap col':22s}  {'Coverage':>10s}  "
          f"{'XS std':>14s}  {'Snap IC':>10s}  {'IC (Spearman)':>14s}")
    print("  " + "-" * 100)
    for label, col in USER_ASKED.items():
        if col is None:
            rows_diag.append({
                "field": label, "snap_col": "—",
                "coverage_pct": None,
                "cross_section_std": None,
                "snap_pearson_ic": None,
                "snap_spearman_ic": None,
                "status": "NOT AVAILABLE in QMT",
            })
            print(f"  {label:18s}  {'—':22s}  {'—':>10s}  {'—':>14s}  "
                  f"{'—':>10s}  {'—':>14s}   [NOT AVAILABLE]")
            continue
        v = joined[col].to_numpy(dtype="float64")
        n_valid = int(np.isfinite(v).sum())
        coverage = n_valid / len(joined) * 100
        xs_std = float(np.nanstd(v, ddof=0))
        # IC
        rf = joined["r_future_5"].to_numpy(dtype="float64")
        m = np.isfinite(v) & np.isfinite(rf)
        if m.sum() >= 100:
            p, _ = pearsonr(v[m], rf[m])
            s, _ = spearmanr(v[m], rf[m])
        else:
            p, s = float("nan"), float("nan")
        print(f"  {label:18s}  {col:22s}  {coverage:>9.1f}%  "
              f"{xs_std:>14.4f}  {p:>+10.4f}  {s:>+14.4f}")
        rows_diag.append({
            "field": label, "snap_col": col,
            "coverage_pct": coverage,
            "cross_section_std": xs_std,
            "snap_pearson_ic": p,
            "snap_spearman_ic": s,
            "status": ("AVAILABLE (snapshot only)"
                        if abs(p) > 0.02 or abs(s) > 0.02
                        else "AVAILABLE but weak"),
        })

    print()
    print("  Extra fields available in snapshot (not in user list):")
    print(f"  {'Field':18s}  {'Snap col':22s}  {'Coverage':>10s}  "
          f"{'XS std':>14s}  {'Snap IC':>10s}  {'IC (Spearman)':>14s}")
    print("  " + "-" * 100)
    for label, col in EXTRA_AVAILABLE.items():
        v = joined[col].to_numpy(dtype="float64")
        n_valid = int(np.isfinite(v).sum())
        coverage = n_valid / len(joined) * 100
        xs_std = float(np.nanstd(v, ddof=0))
        rf = joined["r_future_5"].to_numpy(dtype="float64")
        m = np.isfinite(v) & np.isfinite(rf)
        if m.sum() >= 100:
            p, _ = pearsonr(v[m], rf[m])
            s, _ = spearmanr(v[m], rf[m])
        else:
            p, s = float("nan"), float("nan")
        print(f"  {label:18s}  {col:22s}  {coverage:>9.1f}%  "
              f"{xs_std:>14.4f}  {p:>+10.4f}  {s:>+14.4f}")
        rows_diag.append({
            "field": label + " (extra)", "snap_col": col,
            "coverage_pct": coverage,
            "cross_section_std": xs_std,
            "snap_pearson_ic": p,
            "snap_spearman_ic": s,
            "status": "extra snapshot field",
        })

    # ===== 4. IC > 0.02 总结 =====
    print()
    print("  Fields with |IC| > 0.02 (predictive power threshold):")
    strong = [r for r in rows_diag
                if r["snap_pearson_ic"] is not None
                and not np.isnan(r["snap_pearson_ic"])
                and (abs(r["snap_pearson_ic"]) > 0.02 or abs(r["snap_spearman_ic"]) > 0.02)]
    if not strong:
        print("    NONE — no available field passes |IC| > 0.02 on the snapshot.")
    else:
        for r in strong:
            print(f"    {r['field']:18s}  "
                  f"Pearson={r['snap_pearson_ic']:+.4f}  "
                  f"Spearman={r['snap_spearman_ic']:+.4f}")

    # 保存
    out = _HERE.parents[1] / "bull_bear" / "results" / "fundamental_field_check.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_diag).to_csv(out, index=False, encoding="utf-8-sig")

    # ===== 重要说明 =====
    print()
    print("=" * 78)
    print("Findings")
    print("=" * 78)
    print(f"""
- /day/ contains 9690 OHLCV-only files; 3876 of them are the tickers in our
  main dataset, the rest (5814) are indices and unrelated securities — these
  should be filtered out if /day/ is used as a source.

- Table_all.csv is a SINGLE-DAY snapshot (no date column, no time series).
  It contains 4 finance-adjacent fields: 市盈(动), 市净率, 净利润 (absolute),
  总市值. NONE of the requested time-series fundamentals (ROE, ROA, 净利润率,
  营收增长率, 资产负债率) exist anywhere in QMT/.

- The other parquets (factors.parquet / factor_data_with_returns.parquet /
  final_dataset.parquet) all contain only anonymous F1-F5 factors plus the
  same technical features already in our main dataset. No fundamentals there
  either.

- Snapshot-day IC for the 2 available fields (PE, PB) is weak and not
  time-series. We cannot do a proper IC = corr(feature, r_future_5) across
  time without daily fundamental panels.

Recommendation: To incorporate fundamentals seriously, you would need to
acquire a separate time-series fundamental dataset (e.g., Tushare pro
'daily_basic' / 'fina_indicator') and merge by (date, ticker). With only
the current QMT data, fundamental factors cannot be added to the model.
""")
    print(f"Output: {out.relative_to(_HERE.parents[1])}")


if __name__ == "__main__":
    main()
