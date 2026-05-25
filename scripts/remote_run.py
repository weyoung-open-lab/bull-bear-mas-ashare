"""
远程跑批 + 监控 + 拉回结果。

用法（本地）：
    python scripts/remote_run.py

它会：
  1. SSH 到 featurize 服务器
  2. 用 tmux + nohup 启动 full benchmark（ALL families on 全量数据）
  3. 每 3 分钟轮询一次：远端进程状态 + 日志最后 8 行
  4. 远端完成后，rsync/sftp 把 results/ 拉回本地
  5. 退出（本地 Bash run_in_background 会通知 Claude）
"""

from __future__ import annotations

import os
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
TAG = "full_remote"

# 全量跑批：14 个变体 (factor 3 + linear 1 + gbdt 6 + tabular_dl 2 + sequence 2)
REMOTE_CMD = (
    f"cd {REMOTE_ROOT} && "
    f"export STOCK_THESIS_DATASET={REMOTE_DATASET} && "
    f"export PYTHONUNBUFFERED=1 && "
    f"python -m experiments.run_main_compare "
    f"--families factor,linear,gbdt,tabular_dl,sequence "
    f"--seq-max-per-ticker-train 200 "
    f"--seq-max-per-ticker-test 100 "
    f"--tag {TAG}"
)

POLL_INTERVAL_SEC = 180


def open_client() -> paramiko.SSHClient:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)
    c.get_transport().set_keepalive(30)
    return c


def run(c: paramiko.SSHClient, cmd: str, timeout: int = 60) -> tuple[str, str]:
    _, o, e = c.exec_command(cmd, timeout=timeout)
    return o.read().decode(errors="replace"), e.read().decode(errors="replace")


def kick_off() -> tuple[str, str]:
    """启动远端任务，返回 (pid, log_path)。"""
    c = open_client()
    log_dir = f"{REMOTE_ROOT}/logs"
    log_path = f"{log_dir}/full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    pid_path = f"{log_dir}/full.pid"

    setup = f"mkdir -p {log_dir} && rm -f {pid_path}"
    run(c, setup)

    # 启动：bash -c '... > log 2>&1 & echo $! > pid_path'
    launch = (
        f"bash -lc \""
        f"nohup bash -c '{REMOTE_CMD}' > {log_path} 2>&1 & "
        f"echo \\$! > {pid_path}; "
        f"disown"
        f"\""
    )
    out, err = run(c, launch, timeout=30)
    if err.strip():
        print("[launch stderr]", err.strip())

    time.sleep(2)
    pid_out, _ = run(c, f"cat {pid_path}")
    pid = pid_out.strip()
    c.close()
    return pid, log_path


def remote_alive(pid: str) -> bool:
    c = open_client()
    out, _ = run(c, f"ps -p {pid} -o pid= 2>/dev/null || echo dead")
    c.close()
    return out.strip() != "dead" and out.strip() != ""


def remote_tail(log_path: str, n: int = 8) -> str:
    c = open_client()
    out, _ = run(c, f"tail -n {n} {log_path}")
    c.close()
    return out.rstrip()


def list_remote_results(tag: str) -> list[str]:
    c = open_client()
    out, _ = run(c, f"ls -1d {REMOTE_ROOT}/results/main_compare_*_{tag} 2>/dev/null")
    c.close()
    return [d for d in out.strip().splitlines() if d]


def fetch_results(remote_dir: str) -> Path:
    """SFTP 把 remote_dir 拉回本地 results/ 同名子目录。"""
    name = os.path.basename(remote_dir)
    local_dir = LOCAL_ROOT / "results" / name
    local_dir.mkdir(parents=True, exist_ok=True)

    t = paramiko.Transport((HOST, PORT))
    t.connect(username=USER, password=PASSWORD)
    sftp = paramiko.SFTPClient.from_transport(t)

    def walk(remote, local):
        for entry in sftp.listdir_attr(remote):
            from stat import S_ISDIR
            r = f"{remote}/{entry.filename}"
            l = local / entry.filename
            if S_ISDIR(entry.st_mode):
                l.mkdir(exist_ok=True)
                walk(r, l)
            else:
                sftp.get(r, str(l))
                print(f"  fetched {r}  ({entry.st_size} bytes)")

    walk(remote_dir, local_dir)
    sftp.close(); t.close()
    return local_dir


def fetch_log(log_path: str, suffix: str = "") -> Path:
    name = Path(log_path).name + (f".{suffix}" if suffix else "")
    local = LOCAL_ROOT / "results" / name
    t = paramiko.Transport((HOST, PORT))
    t.connect(username=USER, password=PASSWORD)
    sftp = paramiko.SFTPClient.from_transport(t)
    sftp.get(log_path, str(local))
    sftp.close(); t.close()
    return local


def main() -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] launching remote benchmark ...", flush=True)
    pid, log_path = kick_off()
    if not pid or not pid.isdigit():
        raise SystemExit(f"failed to obtain remote PID. log_path={log_path}")
    print(f"[{datetime.now():%H:%M:%S}] remote pid={pid}  log={log_path}", flush=True)

    start = time.time()
    while True:
        time.sleep(POLL_INTERVAL_SEC)
        elapsed = (time.time() - start) / 60
        try:
            alive = remote_alive(pid)
            tail = remote_tail(log_path, n=6)
        except Exception as e:
            print(f"[{datetime.now():%H:%M:%S}] poll error ({type(e).__name__}: {e}); will retry", flush=True)
            continue
        print(f"\n[{datetime.now():%H:%M:%S}] elapsed={elapsed:.1f} min | alive={alive}", flush=True)
        for line in tail.splitlines():
            print("  ", line, flush=True)
        if not alive:
            print(f"\n[{datetime.now():%H:%M:%S}] remote process exited.", flush=True)
            break

    # 拉回最新一个匹配 tag 的结果目录
    print(f"[{datetime.now():%H:%M:%S}] fetching results ...", flush=True)
    dirs = list_remote_results(TAG)
    if not dirs:
        print("no results dir found; fetching log only.")
        fetch_log(log_path)
        return
    remote_dir = sorted(dirs)[-1]
    print(f"  remote_dir = {remote_dir}", flush=True)
    local_dir = fetch_results(remote_dir)
    fetch_log(log_path, suffix="run.log")
    print(f"\n[{datetime.now():%H:%M:%S}] done. local results at: {local_dir}", flush=True)


if __name__ == "__main__":
    main()
