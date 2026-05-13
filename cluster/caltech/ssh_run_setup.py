"""Run Rasyn-Retro env setup on Caltech HPC over one Duo-authenticated SSH session.

Mirrors ssh.py's auth pattern. Streams stdout/stderr live so you can watch
the pytest output / pip installs / module loads in real time.

Usage (PowerShell or bash):
    set CALTECH_PW=...   # optional; otherwise getpass prompts
    python cluster/caltech/ssh_run_setup.py
    # then approve the Duo push on your phone

What it does on the remote login node:
  1. cd /resnick/scratch/atiwari2/rasyn-retro
  2. git pull origin main
  3. bash cluster/caltech/00_setup_env.sh   (~5-10 min)
  4. bash cluster/caltech/01_smoke_local.sh (~30s)

Setup script is idempotent. Re-running is safe.
"""

import os
import sys
import time
import getpass
import select

import paramiko

# Force unbuffered stdout for live progress when redirected.
sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)

HOST = "login.hpc.caltech.edu"
USER = "atiwari2"
PASSWORD = os.environ.get("CALTECH_PW") or getpass.getpass(f"Password for {USER}@{HOST}: ")
DUO_OPTION = "2"  # Duo push

REPO_DIR = "/resnick/scratch/atiwari2/rasyn-retro"

# Chain commands with `&&` so a failure aborts the rest. `bash -lc` so module
# is available. `2>&1` so stderr interleaves with stdout for clean streaming.
REMOTE_CMD = (
    f"cd {REPO_DIR} && "
    f"echo '=== git pull ===' && git pull --ff-only origin main && "
    f"echo '=== 00_setup_env.sh ===' && bash cluster/caltech/00_setup_env.sh && "
    f"echo '=== 01_smoke_local.sh ===' && bash cluster/caltech/01_smoke_local.sh && "
    f"echo '=== ALL DONE ==='"
)


def kbd(title, instructions, prompt_list):
    out = []
    for prompt, _ in prompt_list:
        p = prompt.lower()
        if "password" in p:
            print("[auth] -> password")
            out.append(PASSWORD)
        elif "passcode" in p or "option" in p:
            print(f"[auth] -> Duo option {DUO_OPTION} (approve push on phone)")
            out.append(DUO_OPTION)
        else:
            print(f"[auth] -> blank for prompt {prompt!r}")
            out.append("")
    return out


def main():
    print("=== Caltech retro setup runner ===")
    print(f"[ssh] connecting to {USER}@{HOST}")
    t0 = time.time()

    tr = paramiko.Transport((HOST, 22))
    tr.banner_timeout = 60
    tr.connect()
    print("[ssh] connected")

    for stage in range(1, 6):
        try:
            still = tr.auth_interactive(USER, kbd)
        except paramiko.AuthenticationException as e:
            print(f"[ssh] stage {stage} auth exception: {e}")
            still = None
        if tr.is_authenticated():
            print(f"[ssh] authenticated (stage {stage})")
            break
        print(f"[ssh] partial auth, still needs: {still}")
        if not still or "keyboard-interactive" not in still:
            print("[ssh] cannot continue")
            return 1
    else:
        print("[ssh] gave up after 5 auth stages")
        return 1

    print(f"[remote] running:\n  {REMOTE_CMD}\n")
    ch = tr.open_session()
    ch.get_pty()  # so output isn't fully buffered
    ch.exec_command(REMOTE_CMD)

    # Stream stdout + stderr live until channel closes.
    while True:
        if ch.recv_ready():
            data = ch.recv(4096)
            if data:
                sys.stdout.write(data.decode("utf-8", errors="replace"))
                sys.stdout.flush()
        if ch.recv_stderr_ready():
            data = ch.recv_stderr(4096)
            if data:
                sys.stderr.write(data.decode("utf-8", errors="replace"))
                sys.stderr.flush()
        if ch.exit_status_ready() and not ch.recv_ready() and not ch.recv_stderr_ready():
            break
        # tiny sleep to avoid busy spin
        time.sleep(0.05)

    # drain remainder
    while ch.recv_ready():
        sys.stdout.write(ch.recv(4096).decode("utf-8", errors="replace"))
    while ch.recv_stderr_ready():
        sys.stderr.write(ch.recv_stderr(4096).decode("utf-8", errors="replace"))

    ec = ch.recv_exit_status()
    tr.close()
    dt = (time.time() - t0) / 60
    print(f"\n[remote] exit={ec}  total={dt:.1f} min")
    return ec


if __name__ == "__main__":
    sys.exit(main())
