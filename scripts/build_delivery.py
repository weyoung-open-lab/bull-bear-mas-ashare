"""Assemble the bull_bear/delivery/ package."""

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEL = ROOT / "bull_bear" / "delivery"

found, missing = [], []


def safe_copy(src, dst):
    src = Path(src)
    dst = Path(dst)
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        found.append(str(src.relative_to(ROOT)))
        return True
    missing.append(str(src.relative_to(ROOT)) if src.is_absolute() and ROOT in src.parents else str(src))
    return False


def copy_glob(src_dir, dst_dir, pattern="*"):
    src_dir = Path(src_dir)
    if not src_dir.exists():
        missing.append(str(src_dir.relative_to(ROOT)) + "/")
        return 0
    n = 0
    for p in src_dir.glob(pattern):
        if p.is_file():
            safe_copy(p, Path(dst_dir) / p.name)
            n += 1
    return n


# ------------------------------------------------------------------
# 1. PAPER
# ------------------------------------------------------------------
print("== Copying paper ==")
safe_copy(ROOT / "paper" / "bull_bear_paper.tex", DEL / "paper" / "sections" / "bull_bear_paper.tex")
safe_copy(ROOT / "paper" / "bull_bear_paper.pdf", DEL / "paper" / "bull_bear_paper.pdf")
safe_copy(ROOT / "paper" / "experiments_data_bundle.md", DEL / "paper" / "data_bundle.md")
safe_copy(ROOT / "paper" / "catboost_selection_rationale.md", DEL / "paper" / "catboost_selection_rationale.md")
safe_copy(ROOT / "paper" / "model_selection_report.md", DEL / "paper" / "model_selection_report.md")
safe_copy(ROOT / "paper" / "sections" / "experiments_bundle.md", DEL / "paper" / "sections" / "experiments_bundle.md")

# style files
for sty in ["cas-common.sty", "cas-model2-names.bst", "cas-refs.bib", "cas-sc.cls"]:
    safe_copy(ROOT / "paper" / sty, DEL / "paper" / "sections" / sty)

# els-cas-templates folder
safe_copy(ROOT / "els-cas-templates" / "els-cas-templates", DEL / "paper" / "sections" / "els-cas-templates")

# figures
print("== Copying paper figures ==")
copy_glob(ROOT / "figure", DEL / "paper" / "figures", "*.png")
copy_glob(ROOT / "figure", DEL / "paper" / "figures", "*.pdf")

# ------------------------------------------------------------------
# 2. MODELS
# ------------------------------------------------------------------
print("== Copying models ==")
MODELS_SRC = ROOT / "bull_bear" / "results" / "models"

safe_copy(MODELS_SRC / "bear_D2_l3.0.cbm", DEL / "models" / "bear_D2_lambda3.cbm")
safe_copy(MODELS_SRC / "bear_D2_agent.cbm", DEL / "models" / "bear_D2_lambda3_agent.cbm")
safe_copy(MODELS_SRC / "bear_D2_agent_medians.csv", DEL / "models" / "bear_D2_lambda3_agent_medians.csv")
safe_copy(MODELS_SRC / "bear_D1_agent.cbm", DEL / "models" / "bear_D1_agent.cbm")
safe_copy(MODELS_SRC / "bear_D1_agent_medians.csv", DEL / "models" / "bear_D1_agent_medians.csv")
safe_copy(MODELS_SRC / "bear_agent.cbm", DEL / "models" / "bear_C_agent.cbm")
safe_copy(MODELS_SRC / "bear_agent_medians.csv", DEL / "models" / "bear_C_agent_medians.csv")
safe_copy(MODELS_SRC / "reversal_B_5d.cbm", DEL / "models" / "reversal_agent.cbm")
safe_copy(MODELS_SRC / "reversal_B_1d.cbm", DEL / "models" / "reversal_B_1d.cbm")
safe_copy(MODELS_SRC / "correction_D4.cbm", DEL / "models" / "correction_D4.cbm")
safe_copy(MODELS_SRC / "arb_D3.cbm", DEL / "models" / "arbitrage_D3.cbm")
safe_copy(MODELS_SRC / "README.md", DEL / "models" / "README.md")

# All lambda variants for sensitivity reproduction
for lam in ["0.5", "1.0", "2.0", "3.0", "5.0", "10.0"]:
    safe_copy(MODELS_SRC / f"bear_D2_l{lam}.cbm",
              DEL / "models" / "bear_D2_lambda_sweep" / f"bear_D2_lambda{lam}.cbm")

# Walk-forward bundle
WF_SRC = MODELS_SRC / "walkforward"
WF_DST = DEL / "models" / "walkforward"
if WF_SRC.exists():
    for p in sorted(WF_SRC.iterdir()):
        if p.is_file():
            safe_copy(p, WF_DST / p.name)

# Regime classifier — search common locations
for candidate in [
    ROOT / "bull_bear" / "results" / "models" / "regime_agent.cbm",
    ROOT / "results" / "regime_20260507_013443_final_g1234_cat" / "regime_agent.cbm",
]:
    if candidate.exists():
        safe_copy(candidate, DEL / "models" / "regime_agent" / candidate.name)

# Also copy the regime CatBoost submodel SHAP/eval CSVs as configuration
for src_sub in [
    ROOT / "results" / "regime_20260507_013443_final_g1234_cat" / "regime_eval.csv",
    ROOT / "results" / "regime_20260507_013443_final_g1234_cat" / "srd_matrix.csv",
    ROOT / "results" / "regime_20260507_013443_final_g1234_cat" / "config.json",
]:
    safe_copy(src_sub, DEL / "models" / "regime_agent" / src_sub.name)

# Alpha agent — saved by step12/13 in walkforward sub-CSVs; alpha_W*.cbm in walkforward dir is the canonical alpha
safe_copy(MODELS_SRC / "walkforward" / "alpha_W1.cbm", DEL / "models" / "alpha_agent.cbm")
safe_copy(MODELS_SRC / "walkforward" / "alpha_W1_medians.csv", DEL / "models" / "alpha_agent_medians.csv")

# ------------------------------------------------------------------
# 3. RESULTS
# ------------------------------------------------------------------
print("== Copying results ==")
RES_SRC = ROOT / "bull_bear" / "results"

# 3a. ablation
ABL_DST = DEL / "results" / "ablation"
for f in [
    "final/final_ablation.csv",
    "mechanism_validation.csv",
    "step8_d2_peak.csv",
    "step12_gamma_peak.csv",
    "step13_gamma_adaptive.csv",
    "step16_agent4_practical.csv",
    "final/simple_baseline_comparison.csv",
    "final_ablation.csv",   # also keep root copy
    "step7_agent_interactions.csv",
    "step11_agent4_redesign.csv",
    "step9_d2c_anomaly_position.csv",
    "step14_bear_attribution.csv",
    "step14_risk_manager.csv",
]:
    src = RES_SRC / f
    safe_copy(src, ABL_DST / Path(f).name)

# 3b. validation
VAL_DST = DEL / "results" / "validation"
for f in [
    "final/rolling_walkforward.csv",
    "final/bootstrap_test.csv",
    "final/bear_quintile_analysis.csv",
    "step15_walkforward_d1c_vs_d2f.csv",
    "step12_walkforward_d2f.csv",
    "step12_bootstrap_d2f.csv",
    "step15_bootstrap_d1c_vs_d2f.csv",
    "step11_reversal_standalone.csv",
    "step12_reversal_diagnostics.csv",
]:
    src = RES_SRC / f
    safe_copy(src, VAL_DST / Path(f).name)

# 3c. robustness
ROB_DST = DEL / "results" / "robustness"
for f in [
    "step17_validation_param_search.csv",
    "step17_param_sensitivity.csv",
    "step17_ic_distribution.csv",
    "step17_rolling_rankicir.csv",
    "step18_elliott_wave.csv" if (RES_SRC / "step18_elliott_wave.csv").exists() else "step_wave_integration.csv",
    "step_wave_signal_diag.csv",
    "step19_wave_dev02_methodB.csv",
]:
    src = RES_SRC / f
    safe_copy(src, ROB_DST / Path(f).name)

# Model-selection results live in top-level results/
TOP_RES = ROOT / "results"
SEL_DST = DEL / "results" / "model_selection"
safe_copy(TOP_RES / "main_compare_20260506_204944_full_remote" / "metrics_summary.csv",
          SEL_DST / "main_compare_bce_metrics.csv")
safe_copy(TOP_RES / "main_compare_20260506_204944_full_remote" / "table1_metrics.csv",
          SEL_DST / "main_compare_bce_table1.csv")
safe_copy(TOP_RES / "main_compare_20260506_225947_full_reg" / "metrics_summary.csv",
          SEL_DST / "main_compare_mse_metrics.csv")
safe_copy(TOP_RES / "binary_vs_regression.csv",
          SEL_DST / "binary_vs_regression.csv")
safe_copy(TOP_RES / "feature_ablation_20260506_235253_full" / "feature_ablation.csv",
          SEL_DST / "feature_ablation.csv")
safe_copy(TOP_RES / "regime_20260507_013022_final_g1234" / "srd_matrix.csv",
          SEL_DST / "srd_matrix_lgbm.csv")
safe_copy(TOP_RES / "regime_20260507_013443_final_g1234_cat" / "srd_matrix.csv",
          SEL_DST / "srd_matrix_catboost.csv")
safe_copy(TOP_RES / "significance_t_tests.csv", SEL_DST / "significance_t_tests.csv")
safe_copy(TOP_RES / "significance_srd_permutation.csv", SEL_DST / "significance_srd_permutation.csv")

# ------------------------------------------------------------------
# 4. SOURCE CODE
# ------------------------------------------------------------------
print("== Copying source code ==")
SRC_DST = DEL / "src"

# Feature engineering
safe_copy(ROOT / "bull_bear" / "src" / "feature_engineering.py", SRC_DST / "features" / "feature_engineering.py")
safe_copy(ROOT / "src" / "features.py", SRC_DST / "features" / "features_core.py")
safe_copy(ROOT / "bull_bear" / "src" / "bear_target.py", SRC_DST / "features" / "bear_target.py")
safe_copy(ROOT / "bull_bear" / "config_bb.py", SRC_DST / "features" / "config_bb.py")

# Agent code
safe_copy(ROOT / "bull_bear" / "src" / "bear_agent.py", SRC_DST / "agents" / "bear_agent.py")
safe_copy(ROOT / "bull_bear" / "experiments" / "step1_train_bear.py", SRC_DST / "agents" / "train_bear_D1.py")
safe_copy(ROOT / "bull_bear" / "experiments" / "step8_d2_peak_search.py", SRC_DST / "agents" / "train_bear_D2_lambda_sweep.py")
safe_copy(ROOT / "bull_bear" / "experiments" / "step11_agent4_redesign.py", SRC_DST / "agents" / "train_reversal_agent.py")
safe_copy(ROOT / "bull_bear" / "experiments" / "step12_d2f_final.py", SRC_DST / "agents" / "train_d2f_full.py")
safe_copy(ROOT / "bull_bear" / "experiments" / "step13_gamma_adaptive.py", SRC_DST / "agents" / "gamma_adaptive.py")
safe_copy(ROOT / "src" / "regime_ensemble.py", SRC_DST / "agents" / "regime_ensemble.py")
safe_copy(ROOT / "src" / "models" / "gbdt.py", SRC_DST / "agents" / "model_zoo_gbdt.py")
safe_copy(ROOT / "src" / "models" / "base.py", SRC_DST / "agents" / "model_zoo_base.py")
safe_copy(ROOT / "waves_agent.py", SRC_DST / "agents" / "waves_agent.py")

# Backtest / evaluation
safe_copy(ROOT / "src" / "backtest.py", SRC_DST / "backtest" / "backtest.py")
safe_copy(ROOT / "src" / "metrics.py", SRC_DST / "backtest" / "metrics.py")
safe_copy(ROOT / "src" / "data.py", SRC_DST / "backtest" / "data.py")
safe_copy(ROOT / "bull_bear" / "experiments" / "step16_agent4_practical.py", SRC_DST / "backtest" / "agent4_circuit_breaker.py")
safe_copy(ROOT / "bull_bear" / "experiments" / "step15_d1c_vs_d2f.py", SRC_DST / "backtest" / "walkforward_d1c_vs_d2f.py")
safe_copy(ROOT / "bull_bear" / "experiments" / "step17_leakage_robustness.py", SRC_DST / "backtest" / "step17_robustness.py")

# Main pipeline orchestration
safe_copy(ROOT / "bull_bear" / "experiments" / "step5_final_ablation.py", SRC_DST / "pipeline.py")
safe_copy(ROOT / "bull_bear" / "experiments" / "step6_final_validation.py", SRC_DST / "pipeline_validation.py")

# Top-level config & requirements
safe_copy(ROOT / "config.py", SRC_DST / "config.py")
safe_copy(ROOT / "requirements.txt", DEL / "requirements.txt")

# ------------------------------------------------------------------
# 5. SUMMARY
# ------------------------------------------------------------------
print("\n=== Delivery Summary ===")
total_files = 0
total_bytes = 0
for root, dirs, files in os.walk(DEL):
    rel = Path(root).relative_to(DEL)
    depth = 0 if str(rel) == "." else len(rel.parts)
    indent = "  " * depth
    label = "delivery/" if str(rel) == "." else f"{rel.parts[-1]}/"
    print(f"{indent}{label}")
    for f in sorted(files):
        size = (Path(root) / f).stat().st_size
        total_bytes += size
        total_files += 1
        print(f"{indent}  {f} ({size/1024:.1f} KB)")

print(f"\n总文件数：{total_files}")
print(f"总大小：{total_bytes/1024/1024:.2f} MB")
print(f"成功复制：{len(found)} 项")
if missing:
    print(f"\n未找到（共{len(missing)}项）：")
    for m in missing:
        print(f"  ✗ {m}")
