"""
v3 chained runner — 等 v2（main_compare_reg + regime_reg）跑完后，
自动接 Table 3 / Table 4 / SHAP 回归底座 重跑，最后 SFTP 拉回所有新结果。

监控点：v2 写在 logs/ 下的 main_reg.pid 与 regime_reg.pid。
当两个 PID 都 dead → 启动 stage 3/4/5。
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

POLL_INTERVAL_SEC = 180

CMD_FEAT_ABLATION = (
    f"cd {REMOTE_ROOT} && export STOCK_THESIS_DATASET={REMOTE_DATASET} && "
    f"export PYTHONUNBUFFERED=1 && "
    f"python -m experiments.run_feature_ablation "
    f"--base-model LightGBM-shallow-reg --tag full"
)
CMD_PREPROC_ABLATION = (
    f"cd {REMOTE_ROOT} && export STOCK_THESIS_DATASET={REMOTE_DATASET} && "
    f"export PYTHONUNBUFFERED=1 && "
    f"python -m experiments.run_ablation_preprocess --tag full"
)
CMD_SHAP_REG = (
    f"cd {REMOTE_ROOT} && export STOCK_THESIS_DATASET={REMOTE_DATASET} && "
    f"export PYTHONUNBUFFERED=1 && "
    f"python -m experiments.run_shap "
    f"--base-model LightGBM-shallow-reg --shap-sample 20000 --top1pct-cap 20000 "
    f"--tag full_reg"
)


def open_client():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)
    c.get_transport().set_keepalive(30)
    return c


def run(c, cmd, timeout=60):
    _, o, e = c.exec_command(cmd, timeout=timeout)
    return o.read().decode(errors="replace"), e.read().decode(errors="replace")


def pid_alive(pid: str) -> bool:
    if not pid or not pid.isdigit():
        return False
    c = open_client()
    out, _ = run(c, f"ps -p {pid} -o pid= 2>/dev/null || echo dead")
    c.close()
    return out.strip() not in ("", "dead")


def read_pid(pid_path: str) -> str:
    c = open_client()
    out, _ = run(c, f"cat {pid_path} 2>/dev/null || echo")
    c.close()
    return out.strip()


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
    pid = read_pid(pid_path)
    c.close()
    return pid, log_path


def remote_tail(log_path: str, n: int = 5) -> str:
    c = open_client()
    out, _ = run(c, f"tail -n {n} {log_path}")
    c.close()
    return out.rstrip()


def watch(pid: str, log_path: str, label: str):
    start = time.time()
    while True:
        time.sleep(POLL_INTERVAL_SEC)
        elapsed = (time.time() - start) / 60
        try:
            alive = pid_alive(pid)
            tail = remote_tail(log_path)
        except Exception as e:
            print(f"[{datetime.now():%H:%M:%S}] {label} poll error: {e}; retry")
            continue
        print(f"\n[{datetime.now():%H:%M:%S}] {label}  elapsed={elapsed:.1f} min  alive={alive}", flush=True)
        for line in tail.splitlines()[-5:]:
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


def wait_v2_done():
    """轮询 v2 的两个 pidfile，直到都已死 → v2 全部完成。"""
    print(f"[{datetime.now():%H:%M:%S}] waiting for v2 to finish ...", flush=True)
    while True:
        time.sleep(POLL_INTERVAL_SEC)
        m_pid = read_pid(f"{REMOTE_ROOT}/logs/main_reg.pid")
        r_pid = read_pid(f"{REMOTE_ROOT}/logs/regime_reg.pid")
        m_alive = pid_alive(m_pid) if m_pid else False
        r_alive = pid_alive(r_pid) if r_pid else False
        print(f"[{datetime.now():%H:%M:%S}] v2 status — main_reg pid={m_pid} alive={m_alive}  "
              f"regime_reg pid={r_pid} alive={r_alive}", flush=True)
        # regime 是 v2 的最后一阶段；只要它出现且死掉就可以
        if r_pid and not r_alive:
            print(f"[{datetime.now():%H:%M:%S}] v2 done (regime_reg exited).", flush=True)
            return
        # 兜底：main 死了且 regime 还没出现，等下一轮
        if m_pid and not m_alive and not r_pid:
            # regime 还没启动，可能 v2 watcher 还在过渡
            continue


def main():
    wait_v2_done()

    print(f"\n[{datetime.now():%H:%M:%S}] Stage 3: Feature Ablation (Table 3)", flush=True)
    pid, log = kick_off(CMD_FEAT_ABLATION, "feat_ablation")
    print(f"  pid={pid}  log={log}")
    watch(pid, log, "Stage3_FeatAblation")

    print(f"\n[{datetime.now():%H:%M:%S}] Stage 4: Preprocess Ablation (Table 4)", flush=True)
    pid, log = kick_off(CMD_PREPROC_ABLATION, "preproc_ablation")
    print(f"  pid={pid}  log={log}")
    watch(pid, log, "Stage4_PreprocAblation")

    print(f"\n[{datetime.now():%H:%M:%S}] Stage 5: SHAP regression base re-run", flush=True)
    pid, log = kick_off(CMD_SHAP_REG, "shap_reg")
    print(f"  pid={pid}  log={log}")
    watch(pid, log, "Stage5_SHAPReg")

    # 拉回三类新结果
    for pattern, label in [
        ("feature_ablation_*_full",          "feature_ablation"),
        ("preprocess_ablation_*_full",       "preprocess_ablation"),
        ("shap_*_full_reg",                   "shap_reg"),
    ]:
        print(f"\n[{datetime.now():%H:%M:%S}] fetching {label} ...", flush=True)
        dirs = list_remote_dirs(pattern)
        if dirs:
            d = sorted(dirs)[-1]
            print(f"  remote: {d}")
            fetch_dir(d)
        else:
            print(f"  [warn] no dir matched {pattern}")

    print(f"\n[{datetime.now():%H:%M:%S}] all stages done.")


if __name__ == "__main__":
    main()
