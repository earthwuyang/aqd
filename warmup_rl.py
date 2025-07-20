#!/usr/bin/env python3
# warmup_rl.py  ——  Bootstrap LinUCB-Δ before benchmark
#
# Usage:
#   python warmup_rl.py --data_dir /path/to/query_costs_trace \
#       --iters 400 --threads 8 --timeout 40000
#
# 依赖: pymysql, tqdm   (与 bench_routing_modes.py 相同)

import argparse, csv, random, time, sys
from pathlib import Path
from threading import Thread, Lock
from statistics import mean
import pymysql
from tqdm import tqdm

HOST, PORT, USER, PASS = "127.0.0.1", 44444, "root", ""
TIMEOUT_TAG = object()           # 特殊标记

def load_all_sql(data_dir: Path, limit: int) -> list[tuple[str,str]]:
    """回收所有 query_costs.csv 里的 SQL（不论 cost/miss-match）。"""
    tasks = []
    for csv_fp in data_dir.rglob("query_costs.csv"):
        db = csv_fp.parent.name
        with csv_fp.open() as f:
            for r in csv.DictReader(f):
                sql = r["query"].rstrip(";").replace('"', "")
                tasks.append((db, sql))
    random.shuffle(tasks)
    return tasks[:limit] if 0 < limit < len(tasks) else tasks

def run_sql(db:str, sql:str, force_row:bool, timeout:int):
    """执行一次 SQL 并返回耗时（秒）; 出错/超时返回 None."""
    try:
        conn = pymysql.connect(host=HOST, port=PORT, user=USER,
                               password=PASS, db=db, autocommit=True)
        cur  = conn.cursor()
        # —— session 开关 —— #
        if force_row:
            cur.execute("SET use_imci_engine = OFF")
        else:
            cur.execute("SET use_imci_engine = ON")
            cur.execute("SET cost_threshold_for_imci = 1")
        cur.execute("SET fann_model_routing_enabled  = ON")
        cur.execute("SET use_mm1_time = ON")
        cur.execute(f"SET max_execution_time = {timeout}")

        # —— run —— #
        t0 = time.perf_counter()
        cur.execute(sql); cur.fetchall()
        return time.perf_counter() - t0
    except pymysql.err.OperationalError as e:
        # 3024 / 1317 = timeout / kill
        if e.args and e.args[0] in (3024, 1317):
            return TIMEOUT_TAG
        return None
    except Exception:
        return None
    finally:
        try: cur.close(); conn.close()
        except: pass

def worker(th_id:int, jobs:list[tuple[str,str]], timeout:int,
           bar:tqdm, lock:Lock, lat:list[float|None]):
    """线程：对列表里的 task 先跑 Row 再跑 Column。"""
    for idx,(db,sql) in jobs:
        row_t = run_sql(db, sql, True,  timeout)
        col_t = run_sql(db, sql, False, timeout)
        with lock:
            lat[idx*2]   = row_t
            lat[idx*2+1] = col_t
            bar.update(2)

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--data_dir", default="/home/wuy/query_costs_trace/")
    pa.add_argument("--iters", type=int, default=600,
                    help="多少条不同 SQL；每条跑 Row+Col 共 2 次")
    pa.add_argument("--threads", type=int, default=16)
    pa.add_argument("--timeout", type=int, default=40000,
                    help="单次 SQL ms")
    pa.add_argument("--seed", type=int, default=2025)
    args = pa.parse_args()
    random.seed(args.seed)

    tasks = load_all_sql(Path(args.data_dir), args.iters)
    if not tasks:
        print("No SQL found", file=sys.stderr); sys.exit(1)

    n_total = len(tasks)*2
    lat = [None]*n_total
    bar = tqdm(total=n_total, ncols=70,
               desc=f"Warm-up {len(tasks)} SQL (Row+Col)",
               leave=False)

    # 划分给线程
    chunk = (len(tasks)+args.threads-1)//args.threads
    lock  = Lock()
    threads=[]
    for i in range(args.threads):
        part = tasks[i*chunk : (i+1)*chunk]
        if not part: break
        idx_jobs=[(idx,task) for idx,task in enumerate(part, start=i*chunk)]
        th = Thread(target=worker,
                    args=(i, idx_jobs, args.timeout, bar, lock, lat),
                    daemon=True)
        th.start(); threads.append(th)
    for th in threads: th.join()
    bar.close()

    # 统计
    ok   = [v for v in lat if v not in (None, TIMEOUT_TAG)]
    to   = lat.count(TIMEOUT_TAG)
    fail = lat.count(None)
    if ok:
        print(f"Done.  avg={mean(ok):.4f}s, ok={len(ok)}, to={to}, fail={fail}")
    else:
        print("All failed!")

if __name__ == "__main__":
    main()
