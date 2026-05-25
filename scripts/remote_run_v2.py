"""
两轮串行远程跑批 + 监控 + 拉回。

阶段一：Table 1 回归版 — 11 个回归模型 + 3 个 factor baseline
阶段二：Regime-Conditioned 回归底座（LightGBM-shallow-reg）

完成后把 results/main_compare_*_full_reg/ 和 results/regime_*_lgbm_shallow_reg/ 拉回本地。
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import paramiko

HOST = "workspace.featurize.cn"
PORT = 33987
USER = "featurize"
PASSWORD = "1062ff56"

REMOTE_ROOT = "/home/featurize/work/stock_thesis"
REMOTE_DATASET = "/home/featurize/data/dataset_model_baseline_longer_trend.parquet"
LOCAL_ROOT = Path(__file__).resolve().parents[1]

REG_MODELS = ",".join([
    # Factor baseline (rank-by-feature, 仍纳入对照)
    "Momentum-5d", "EMA-slope", "Rel-Strength",
    # Linear (binary 已有；这里加 Ridge 回归)
    "Ridge",
    # GBDT regression × 6
    "LightGBM-std-reg", "LightGBM-shallow-reg", "LightGBM-conservative-reg",
    "XGBoost-reg", "CatBoost-reg", "RandomForest-reg",
    # Tabular DL regression × 2
    "TabNet-reg", "FT-Transformer-reg",
    # Sequence DL regression × 2
    "ALSTM-reg", "TCN-reg",
])

CMD_MAIN = (
    f"cd {REMOTE_ROOT} && export STOCK_THESIS_DATASET={REMOTE_DATASET} && "
    f"export PYTHONUNBUFFERED=1 && "
    f"python -m experiments.run_main_compare "
    f"--models {REG_MODELS} "
    f"--seq-max-per-ticker-train 200 --seq-max-per-ticker-test 100 "
    f"--tag full_reg"
)
CMD_REGIME = (
    f"cd {REMOTE_ROOT} && export STOCK_THESIS_DATASET={REMOTE_DATASET} && "
    f"export PYTHONUNBUFFERED=1 && "
    f"python -m experiments.run_regime_analysis "
    f"--base-model LightGBM-shallow-reg "
    f"--shap-sample 20000 "
    f"--tag full_lgbm_shallow_reg"
)

POLL_INTERVAL_SEC = 180


def open_client() -> paramiko.SSHClient:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)
    c.get_transport().set_keepalive(30)
    return c


def run(c, cmd, timeout=60):
    _, o, e = c.exec_command(cmd, timeout=timeout)
    return o.read().decode(errors="replace"), e.read().decode(errors="replace")


def kick_off(cmd: str, log_basename: str) -> tuple[str, str]:
    c = open_client()
    log_dir = f"{REMOTE_ROOT}/logs"
    log_path = f"{log_dir}/{log_basename}_{datetime.now():%Y%m%d_%H%M%S}.log"
    pid_path = f"{log_dir}/{log_basename}.pid"
    run(c, f"mkdir -p {log_dir} && rm -f {pid_path}")
    launch = (
        f"bash -lc \"nohup bash -c '{cmd}' > {log_path} 2>&1 & "
        f"echo \\$! > {pid_path}; disown\""
    )
    run(c, launch, timeout=30)
    time.sleep(2)
    pid_out, _ = run(c, f"cat {pid_path}")
    pid = pid_out.strip()
    c.close()
    return pid, log_path


def remote_alive(pid: str) -> bool:
    c = open_client()
    out, _ = run(c, f"ps -p {pid} -o pid= 2>/dev/null || echo dead")
    c.close()
    return out.strip() not in ("", "dead")


def remote_tail(log_path: str, n: int = 6) -> str:
    c = open_client()
    out, _ = run(c, f"tail -n {n} {log_path}")
    c.close()
    return out.rstrip()


def watch(pid: str, log_path: str, label: str) -> None:
    start = time.time()
    while True:
        time.sleep(POLL_INTERVAL_SEC)
        elapsed = (time.time() - start) / 60
        try:
            alive = remote_alive(pid)
            tail = remote_tail(log_path, n=5)
        except Exception as e:
            print(f"[{datetime.now():%H:%M:%S}] {label} poll error ({e}); retry next tick")
            continue
        print(f"\n[{datetime.now():%H:%M:%S}] {label}  elapsed={elapsed:.1f} min  alive={alive}", flush=True)
        for line in tail.splitlines():
            print("  ", line, flush=True)
        if not alive:
            print(f"[{datetime.now():%H:%M:%S}] {label} done.\n", flush=True)
            return


def list_remote_dirs(pattern: str) -> list[str]:
    c = open_client()
    out, _ = run(c, f"ls -1d {REMOTE_ROOT}/results/{pattern} 2>/dev/null")
    c.close()
    return [d for d in out.strip().splitlines() if d]


def fetch_dir(remote_dir: str) -> Path:
    from stat import S_ISDIR
    name = Path(remote_dir).name
    local_dir = LOCAL_ROOT / "results" / name
    local_dir.mkdir(parents=True, exist_ok=True)
    t = paramiko.Transport((HOST, PORT))
    t.connect(username=USER, password=PASSWORD)
    sftp = paramiko.SFTPClient.from_transport(t)

    def walk(remote, local):
        for entry in sftp.listdir_attr(remote):
            r = f"{remote}/{entry.filename}"
            l = local / entry.filename
            if S_ISDIR(entry.st_mode):
                l.mkdir(exist_ok=True)
                walk(r, l)
            else:
                sftp.get(r, str(l))
                print(f"  fetched {entry.filename} ({entry.st_size} bytes)")

    walk(remote_dir, local_dir)
    sftp.close(); t.close()
    return local_dir


def main():
    print(f"[{datetime.now():%H:%M:%S}] Stage 1: launch main_compare regression run", flush=True)
    pid1, log1 = kick_off(CMD_MAIN, "main_reg")
    print(f"  pid={pid1}  log={log1}")
    watch(pid1, log1, "Stage1")

    print(f"[{datetime.now():%H:%M:%S}] Stage 2: launch regime regression run", flush=True)
    pid2, log2 = kick_off(CMD_REGIME, "regime_reg")
    print(f"  pid={pid2}  log={log2}")
    watch(pid2, log2, "Stage2")

    # 拉回结果
    print(f"\n[{datetime.now():%H:%M:%S}] fetching stage1 results ...")
    dirs = list_remote_dirs("main_compare_*_full_reg")
    if dirs:
        d1 = sorted(dirs)[-1]
        print(f"  remote: {d1}")
        fetch_dir(d1)

    print(f"\n[{datetime.now():%H:%M:%S}] fetching stage2 results ...")
    dirs2 = list_remote_dirs("regime_*_full_lgbm_shallow_reg")
    if dirs2:
        d2 = sorted(dirs2)[-1]
        print(f"  remote: {d2}")
        fetch_dir(d2)

    print(f"\n[{datetime.now():%H:%M:%S}] all done.")


if __name__ == "__main__":
    main()
