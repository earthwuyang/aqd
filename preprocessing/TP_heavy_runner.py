#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tp_heavy_runner.py —— 持续制造 InnoDB / 行存 TP-heavy 负载
可用 --threads/-t 指定并发线程数
"""
import argparse, logging, os, random, sys, time, threading, pymysql
from typing import List

def load_queries(path: str) -> List[str]:
    with open(path) as f:
        lines = [l.strip() for l in f.readlines()]
    return [l.replace('"','').strip() for l in lines if l and not l.lstrip().startswith("#")]

def execute_row(cur, q: str, timeout: int):
    # cur.execute(f"SET max_execution_time={timeout}")
    cur.execute("SET use_imci_engine=off")             # 行存
    begin = time.time()
    cur.execute(q); cur.fetchall()
    return time.time() - begin

def worker(tid: int, queries: List[str], args):
    my_q = list(queries)
    conn = pymysql.connect(host=args.host, port=args.port,
                           user=args.user, password=args.password,
                           db=args.db, autocommit=True)
    cur = conn.cursor()
    loop = 0
    try:
        while True:
            if args.shuffle_each_loop:
                random.shuffle(my_q)
            for idx, q in enumerate(my_q, 1):
                try:
                    cost = execute_row(cur, q, args.timeout)
                    if args.debug:
                        logging.debug(f"[T{tid}] loop{loop} #{idx}/{len(my_q)} {cost:.3f}s")
                except Exception as e:
                    logging.warning(f"[T{tid}] loop{loop} #{idx} FAIL: {e}")
            loop += 1
    except KeyboardInterrupt:
        pass
    finally:
        cur.close(); conn.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=44444)
    ap.add_argument("--user", default="root")
    ap.add_argument("--password", default="")
    ap.add_argument("--db", default="tpch_sf100")
    ap.add_argument("--data_dir", default="/home/wuy/query_costs/")
    ap.add_argument("--dataset", default="tpch_sf100")
    ap.add_argument("--timeout", type=int, default=60000)
    ap.add_argument("--shuffle_each_loop", action="store_true", default=True)
    ap.add_argument("--threads", "-t", type=int, default=50)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    tp_path = os.path.join(args.data_dir, "workloads", args.dataset, "TP_queries.sql")
    queries = load_queries(tp_path)
    if not queries:
        logging.error(f"TP workload 为空: {tp_path}"); sys.exit(2)
    logging.info(f"Loaded {len(queries)} TP queries; launching {args.threads} thread(s)…")

    threads = []
    for tid in range(args.threads):
        t = threading.Thread(target=worker, args=(tid, queries, args), daemon=True)
        t.start(); threads.append(t)

    try:
        for t in threads: t.join()
    except KeyboardInterrupt:
        logging.info("Interrupted, exiting …")

if __name__ == "__main__":
    main()
