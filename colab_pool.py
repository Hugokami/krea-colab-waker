"""
Colab Multi-Account Pool & Quota Fallback Rotator for GitHub Actions and Local Runners.
Manages up to 6 Google Colab accounts and provides automated quota and session conflict fallback.
"""

import sys
import os
import json
import shutil
import subprocess
from pathlib import Path

CONFIG_DIR = Path.home() / ".config" / "colab-cli"
ACCOUNTS_DIR = CONFIG_DIR / "accounts"
ACTIVE_TOKEN = CONFIG_DIR / "token.json"
ACTIVE_SLOT_FILE = CONFIG_DIR / "active_slot.txt"
MAX_SLOTS = 6


def ensure_dirs():
    ACCOUNTS_DIR.mkdir(parents=True, exist_ok=True)
    if not ACTIVE_SLOT_FILE.exists():
        ACTIVE_SLOT_FILE.write_text("1", encoding="utf-8")


def get_active_slot() -> int:
    ensure_dirs()
    try:
        slot = int(ACTIVE_SLOT_FILE.read_text(encoding="utf-8").strip())
        if 1 <= slot <= MAX_SLOTS:
            return slot
    except Exception:
        pass
    return 1


def set_active_slot(slot: int):
    ensure_dirs()
    account_path = ACCOUNTS_DIR / f"account_{slot}.json"
    if not account_path.exists():
        print(f"[colab-pool] Error: Account {slot} is not configured yet ({account_path} not found).")
        return False
    shutil.copyfile(account_path, ACTIVE_TOKEN)
    ACTIVE_SLOT_FILE.write_text(str(slot), encoding="utf-8")
    print(f"[colab-pool] Switched active Colab account to Slot {slot}.")
    return True


def get_configured_slots() -> list[int]:
    ensure_dirs()
    configured = []
    for s in range(1, MAX_SLOTS + 1):
        if (ACCOUNTS_DIR / f"account_{s}.json").exists():
            configured.append(s)
    return configured


def run_with_fallback(colab_args: list[str]) -> int:
    ensure_dirs()
    configured = get_configured_slots()
    if not configured:
        print("[colab-pool] Error: No Colab accounts configured in pool.")
        return 1

    active_slot = get_active_slot()
    if active_slot not in configured:
        active_slot = configured[0]
        set_active_slot(active_slot)

    idx = configured.index(active_slot)
    ordered_slots = configured[idx:] + configured[:idx]

    colab_bin = shutil.which("colab") or "colab"

    for attempt, slot in enumerate(ordered_slots):
        if slot != get_active_slot():
            print(f"[colab-pool] Rotating to Account Slot {slot}...")
            set_active_slot(slot)

        cmd = [colab_bin] + colab_args
        print(f"[colab-pool] Executing: {' '.join(cmd)} (Account Slot {slot})...")
        proc = subprocess.run(cmd, capture_output=True, text=True)

        if proc.stdout:
            sys.stdout.write(proc.stdout)
            sys.stdout.flush()
        if proc.stderr:
            sys.stderr.write(proc.stderr)
            sys.stderr.flush()

        if proc.returncode == 0:
            return 0

        err_text = (proc.stderr + " " + proc.stdout).lower()
        if "already exists" in err_text and "new" in colab_args:
            # Session exists on this account: attempt connect
            sess_name = "krea"
            if "-s" in colab_args:
                idx_s = colab_args.index("-s")
                if idx_s + 1 < len(colab_args):
                    sess_name = colab_args[idx_s + 1]
            print(f"[colab-pool] Session '{sess_name}' already exists on Slot {slot}. Connecting...")
            conn_cmd = [colab_bin, "connect", "-s", sess_name]
            conn_proc = subprocess.run(conn_cmd, capture_output=True, text=True)
            if conn_proc.returncode == 0:
                print(f"[colab-pool] Connected successfully to '{sess_name}'.")
                return 0

        # On any error, rotate to the next pooled account if available
        if attempt < len(ordered_slots) - 1:
            next_slot = ordered_slots[attempt + 1]
            print(f"\n[colab-pool] Notice: Account Slot {slot} encountered an error ({proc.returncode}).")
            print(f"[colab-pool] Automatically rotating to Account Slot {next_slot}...\n")
            continue
        else:
            return proc.returncode

    print("\n[colab-pool] All configured accounts in the pool failed.")
    return 1


def main():
    if len(sys.argv) < 2:
        print("Usage: python colab_pool.py run <colab_args...>")
        sys.exit(0)

    cmd = sys.argv[1].lower()
    if cmd in ("run", "exec"):
        sys.exit(run_with_fallback(sys.argv[2:]))
    else:
        sys.exit(run_with_fallback(sys.argv[1:]))


if __name__ == "__main__":
    main()
