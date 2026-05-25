"""
v4 final-stack runner — combine three winning insights into one ensemble:
  - Regression MSE objective (winning loss)
  - Feature groups G1+G2+G3+G4 (drop macro_regime_3 from features per Table 3)
  - macro_regime_3 used as Regime router (paper §5)

Two configs:
  A. LightGBM-shallow-reg + G1234 + Regime routing
  B. CatBoost-reg + G1234 + Regime routing  (best single model in Table 1)

Each ~10–15 min on RTX 4090. Total ~30 min.
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

POLL_INTERVAL_SEC = 120

CONFIGS = [
    ("lgbm_shallow_reg_g1234",
     f"cd {REMOTE_ROOT} && export STOCK_THESIS_DATASET={REMOTE_DATASET} && "
     f"export PYTHONUNBUFFERED=1 && "
     f"python -m experiments.run_regime_analysis "
     f"--base-model LightGBM-shallow-reg "
     f"--feature-groups G1,G2,G3,G4 "
     f"--shap-sample 20000 "
     f"--tag final_g1234"),
    ("catboost_reg_g1234",
     f"cd {REMOTE_ROOT} && export STOCK_THESIS_DATASET={REMOTE_DATASET} && "
     f"export PYTHONUNBUFFERED=1 && "
     f"python -m experiments.run_regime_analysis "
     f"--base-model CatBoost-reg "
     f"--feature-groups G1,G2,G3,G4 "
     f"--shap-sample 20000 "
     f"--tag final_g1234_cat"),
]


def open_client():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)
    c.get_transport().set_keepalive(30)
    return c


def run(c, cmd, timeout=60):
    _, o, e = c.exec_command(cmd, timeout=timeout)
    return o.read().decode(errors="replace"), e.read().decode(errors="replace")


def kick_off(cmd, log_basename):
    c = open_client()
    log_dir = f"{REMOTE_ROOT}/logs"
    log_path = f"{log_dir}/{log_basename}_{datetime.now():%Y%m%d_%H%M%S}.log"
    pid_path = f"{log_dir}/{log_basename}.pid"
    run(c, f"mkdir -p {log_dir} && rm -f {pid_path}")
    run(c, f"bash -lc \"nohup bash -c '{cmd}' > {log_path} 2>&1 & "
            f"echo \\$! > {pid_path}; disown\"", timeout=30)
    time.sleep(2)
    pid_out, _ = run(c, f"cat {pid_path}")
    pid = pid_out.strip()
    c.close()
    return pid, log_path


def alive(pid):
    if not pid or not pid.isdigit():
        return False
    c = open_client()
    out, _ = run(c, f"ps -p {pid} -o pid= 2>/dev/null || echo dead")
    c.close()
    return out.strip() not in ("", "dead")


def tail(log_path, n=5):
    c = open_client()
    out, _ = run(c, f"tail -n {n} {log_path}")
    c.close()
    return out.rstrip()


def watch(pid, log_path, label):
    start = time.time()
    while True:
        time.sleep(POLL_INTERVAL_SEC)
        elapsed = (time.time() - start) / 60
        try:
            a = alive(pid)
            t = tail(log_path)
        except Exception as e:
            print(f"[{datetime.now():%H:%M:%S}] {label} poll error: {e}")
            continue
        print(f"\n[{datetime.now():%H:%M:%S}] {label} elapsed={elapsed:.1f} min alive={a}", flush=True)
        for ln in t.splitlines()[-5:]:
            print("  ", ln, flush=True)
        if not a:
            print(f"[{datetime.now():%H:%M:%S}] {label} done.\n", flush=True)
            return


def list_remote(pattern):
    c = open_client()
    out, _ = run(c, f"ls -1d {REMOTE_ROOT}/results/{pattern} 2>/dev/null")
    c.close()
    return [d for d in out.strip().splitlines() if d]


def fetch(remote_dir, skip_large=True):
    from stat import S_ISDIR
    name = Path(remote_dir).name
    local = LOCAL_ROOT / "results" / name
    local.mkdir(parents=True, exist_ok=True)
    t = paramiko.Transport((HOST, PORT))
    t.connect(username=USER, password=PASSWORD)
    sftp = paramiko.SFTPClient.from_transport(t)

    def walk(remote, dst):
        for entry in sftp.listdir_attr(remote):
            r = f"{remote}/{entry.filename}"
            l = dst / entry.filename
            if S_ISDIR(entry.st_mode):
                l.mkdir(exist_ok=True)
                walk(r, l)
            else:
                if skip_large and "predictions" in str(dst) and entry.st_size > 5_000_000:
                    print(f"  [skip large] {entry.filename}")
                    continue
                sftp.get(r, str(l))
                print(f"  fetched {entry.filename} ({entry.st_size} B)")

    walk(remote_dir, local)
    sftp.close(); t.close()


def main():
    sync_files = [
        "experiments/run_regime_analysis.py",
    ]
    print(f"[{datetime.now():%H:%M:%S}] syncing modified file ...")
    t = paramiko.Transport((HOST, PORT))
    t.connect(username=USER, password=PASSWORD)
    sftp = paramiko.SFTPClient.from_transport(t)
    for f in sync_files:
        sftp.put(str(LOCAL_ROOT / f), f"{REMOTE_ROOT}/{f}")
        print(f"  uploaded {f}")
    sftp.close(); t.close()

    for label, cmd in CONFIGS:
        print(f"\n[{datetime.now():%H:%M:%S}] launch {label}")
        pid, log = kick_off(cmd, label)
        print(f"  pid={pid} log={log}")
        watch(pid, log, label)

    # 拉回
    for pattern in ("regime_*_final_g1234", "regime_*_final_g1234_cat"):
        for d in list_remote(pattern):
            print(f"\n[{datetime.now():%H:%M:%S}] fetching {Path(d).name}")
            fetch(d)

    print(f"\n[{datetime.now():%H:%M:%S}] all final runs complete.")


if __name__ == "__main__":
    main()
