#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
List every TID (thread id) that belongs to the first running
'mysqld' process and print its /proc/.../comm.

No extra packages required.
"""

import os
import sys
import subprocess
from pathlib import Path

def find_mysqld_pid():
    """Return PID (int) of the first 'mysqld' process, or None."""
    try:
        # `pgrep -n` gives newest PID that matches
        pid = subprocess.check_output(["pgrep", "-n", "mysqld"],
                                      text=True).strip()
        return int(pid)
    except subprocess.CalledProcessError:
        return None
    except ValueError:
        return None

def list_tids(pid: int):
    """Yield (tid, comm) pairs for the given PID."""
    task_dir = Path(f"/proc/{pid}/task")
    for tid_dir in task_dir.iterdir():
        if not tid_dir.is_dir():
            continue
        tid = int(tid_dir.name)
        try:
            comm_path = tid_dir / "comm"
            comm = comm_path.read_text().strip()
        except Exception:
            comm = "<unreadable>"
        yield tid, comm

def main():
    pid = find_mysqld_pid()
    if pid is None:
        print("No mysqld process found.", file=sys.stderr)
        sys.exit(1)

    print(f"mysqld PID = {pid}\n")
    print(f"{'TID':>8}  COMM")
    print("-" * 32)
    for tid, comm in sorted(list_tids(pid)):
        print(f"{tid:8d}  {comm}")

if __name__ == "__main__":
    main()
