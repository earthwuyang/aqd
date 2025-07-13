#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bulk collector – iterate over all benchmark datasets, gather up-to 10 k
samples (latency + routing decisions + 24-dim features) per dataset.

Now runs several datasets in parallel with --jobs N (default = #cores).

Author : you
Date   : 2025-07-13
"""

import os, csv, json, time, argparse, logging, random, re, signal, traceback
from typing import Any, List, Tuple

from concurrent.futures import ThreadPoolExecutor, as_completed

import pymysql
from pymysql.err import InternalError, ProgrammingError
from tqdm import tqdm

##############################################################################
# 0. helpers                                                                 #
##############################################################################
def _enable_trace(cur, mem_mb: int = 128) -> None:
    cur.execute(f"SET optimizer_trace_max_mem_size={mem_mb * 1024 * 1024}")
    cur.execute("SET optimizer_trace='enabled=on,one_line=off'")


def _disable_trace(cur) -> None:
    cur.execute("SET optimizer_trace='enabled=off'")


def _fetch_trace(cur) -> str:
    cur.execute("SELECT TRACE FROM INFORMATION_SCHEMA.OPTIMIZER_TRACE")
    rows = cur.fetchall()
    if not rows:
        raise RuntimeError("OPTIMIZER_TRACE is empty")
    return rows[-1][0]


def _extract_24feats(trace_obj: Any) -> List[float]:
    """DFS to find row_column_dispatch_features → 24-dim list."""
    st = [trace_obj]
    while st:
        n = st.pop()
        if isinstance(n, dict):
            if "row_column_dispatch_features" in n:
                fn = n["row_column_dispatch_features"]
                return [fn[str(i)] for i in range(24)]
            st.extend(n.values())
        elif isinstance(n, list):
            st.extend(n)
    raise KeyError("row_column_dispatch_features not found")


def _apply_mode(cur, mode: str, timeout_ms: int) -> None:
    """Switch connection into the requested execution / explain mode."""
    cur.execute(f"SET max_execution_time={timeout_ms}")

    if mode == "row_only":
        cur.execute("SET use_imci_engine=off")

    elif mode == "col_only":
        cur.execute("SET use_imci_engine=forced")

    elif mode in ("cost_threshold", "hybrid", "fann"):
        cur.execute("SET use_imci_engine=on")
        cur.execute("SET cost_threshold_for_imci=1")
        cur.execute("SET imci_optimizer_switch='fast_opt_trivial_query=off'")
        cur.execute("SET imci_auto_update_statistic='SYNC'")
        cur.execute("SET hybrid_opt_compatible_transform_switch=4095")
        cur.execute("SET global hybrid_opt_fetch_imci_stats_thread_enabled=on")

        if mode == "cost_threshold":
            cur.execute("SET cost_threshold_for_imci=50000")
            cur.execute("SET hybrid_opt_dispatch_enabled=off")
            cur.execute("SET fann_model_routing_enabled=off")

        elif mode == "hybrid":
            cur.execute("SET hybrid_opt_dispatch_enabled=on")
            cur.execute("SET fann_model_routing_enabled=off")

        elif mode == "fann":
            cur.execute("SET hybrid_opt_dispatch_enabled=on")
            cur.execute("SET fann_model_routing_enabled=on")
            _enable_trace(cur)
    else:
        raise ValueError(f"unknown mode {mode}")


# Recognise IMCI vs ROW from an EXPLAIN JSON -------------------------------
def _decide_imci(plan_json: str) -> int:
    """json string → 0(row) / 1(column)."""
    return 0 if '"query_block"' in plan_json else 1


def _run_mode(conn,
              sql: str,
              mode: str,
              timeout_ms: int
             ) -> Tuple[int, float | None,
                        List[float] | None, float | None]:
    """
    Returns:
      use_imci_flag,
      latency (s)            – real execution or EXPLAIN wall time,
      24-dim features (fann) or None,
      query_cost (row_only)  or None
    """
    cur = conn.cursor()
    _apply_mode(cur, mode, timeout_ms)

    plan_str = None
    qcost    = None

    # ① row_only:   EXPLAIN (cost) → real run
    if mode == "row_only":
        cur.execute(f"EXPLAIN FORMAT='json' {sql}")
        plan_str = cur.fetchone()[0]
        try:
            j = json.loads(plan_str)
            qcost = j["query_block"]["cost_info"]["query_cost"]
        except Exception:
            qcost = None

        t0 = time.time()
        cur.execute(sql); cur.fetchall()
        latency = time.time() - t0

    # ② col_only:   run directly
    elif mode == "col_only":
        t0 = time.time()
        cur.execute(sql); cur.fetchall()
        latency = time.time() - t0

    # ③ others: EXPLAIN only
    else:
        t0 = time.time()
        cur.execute(f"EXPLAIN FORMAT='json' {sql}")
        plan_str = cur.fetchone()[0]
        latency  = time.time() - t0

    # Decide IMCI
    imci_flag = (int(mode == "col_only") if mode in ("row_only", "col_only")
                 else _decide_imci(plan_str or ""))

    # 24-dim feat (fann only)
    feats = None
    if mode == "fann":
        trace = _fetch_trace(cur)
        feats = _extract_24feats(json.loads(trace))
        _disable_trace(cur)

    cur.close()
    return imci_flag, latency, feats, qcost



##############################################################################
# 1.  per-dataset collector                                                 #
##############################################################################
def collect_dataset(ds: str,
                    args,
                    want: int = 10_000) -> None:
    """
    采集单个数据集：
      • 若 query_costs.csv 已有 ≥ want 条(排除表头)，直接 return；
      • 否则仅补齐 (want − 现有) 条。
    """
    root   = os.path.join(args.data_dir, ds)
    os.makedirs(root, exist_ok=True)
    f_cost = os.path.join(root, "query_costs.csv")
    f_feat = os.path.join(root, "features_24d.csv")

    #------------------------------------------------------------------#
    # 0. 现有条数 → decide skip / remaining                             #
    #------------------------------------------------------------------#
    existing = 0
    if os.path.exists(f_cost):
        with open(f_cost, newline="") as fr:
            existing = max(sum(1 for _ in fr) - 1, 0)   # 减去表头
    if existing >= want:
        logging.info(f"[{ds}] already has {existing} ≥ {want} rows – skip")
        return
    remain = want - existing
    logging.info(f"[{ds}] existing={existing}, need {remain} more samples")

    #------------------------------------------------------------------#
    # 1. workload SQL                                                  #
    #------------------------------------------------------------------#
    queries: List[str] = []
    wk_dir = os.path.join(args.data_dir, "workloads", ds)
    for fn in ("workload_100k_s1_group_order_by_more_complex.sql",
               "TP_queries.sql"):
        fp = os.path.join(wk_dir, fn)
        if os.path.exists(fp):
            with open(fp) as fr:
                queries += [ln.strip().replace('"', "")
                            for ln in fr
                            if ln.strip() and not ln.startswith("#")]
    if not queries:
        logging.warning(f"[{ds}] no workload sql found, skip")
        return
    random.shuffle(queries)

    #------------------------------------------------------------------#
    # 2. 打开 CSV                                                      #
    #------------------------------------------------------------------#
    new_cost_hdr = not os.path.exists(f_cost)
    new_feat_hdr = not os.path.exists(f_feat)

    with open(f_cost, "a", newline="") as fp_cost, \
         open(f_feat, "a", newline="") as fp_feat:

        wr_cost = csv.writer(fp_cost)
        wr_feat = csv.writer(fp_feat)
        if new_cost_hdr:
            wr_cost.writerow(["query_id", "use_imci",
                              "row_time", "column_time", "query_cost",
                              "cost_use_imci", "hybrid_use_imci",
                              "fann_use_imci", "query"])
        if new_feat_hdr:
            wr_feat.writerow(["query_id"] + [f"f{i:02d}" for i in range(24)])

        id_offset = existing          # 继续递增 query_id
        done = 0
        conn = pymysql.connect(host="127.0.0.1", user="root",
                               port=args.port, db=ds)

        for local_id, raw_sql in tqdm(list(enumerate(queries)),
                                      total=min(len(queries), remain),
                                      desc=f"{ds}"):
            if done >= remain:
                break

            sql = raw_sql.rstrip(";") + ";"
            try:
                # row / col real execution
                row_flag, row_t, _ , qcost = _run_mode(conn, sql, "row_only",
                                                       args.timeout)
                col_flag, col_t, _ , _     = _run_mode(conn, sql, "col_only",
                                                       args.timeout)

                # cost-rule / hybrid / fann (EXPLAIN)
                cost_flag,  _ , _ , _  = _run_mode(conn, sql, "cost_threshold",
                                                    args.timeout)
                hybrid_flag,_ , _ , _  = _run_mode(conn, sql, "hybrid",
                                                    args.timeout)
                fann_flag,  _ , feats, _ = _run_mode(conn, sql, "fann",
                                                     args.timeout)

                # 必须要有 24-维特征和两个延迟
                if feats is None or (row_t is None and col_t is None):
                    continue

                truth = (1 if row_t is None else
                         0 if col_t is None else
                         int(col_t < row_t))

                qid = id_offset + local_id
                wr_cost.writerow([qid, truth,
                                  row_t, col_t,
                                  "" if qcost is None else qcost,
                                  cost_flag, hybrid_flag, fann_flag,
                                  sql.replace("\n", " ")])
                wr_feat.writerow([qid] + feats)
                fp_cost.flush(); fp_feat.flush()

                done += 1

            except (InternalError, ProgrammingError) as e:
                logging.error(f"[{ds} #{local_id}] SQL error: {e}")
            except Exception as e:
                logging.error(f"[{ds} #{local_id}] {e}")

        conn.close()
        logging.info(f"[{ds}] collected {done} new queries "
                     f"(total {existing + done})")



##############################################################################
# 2.  main                                                                  #
##############################################################################
DATASETS = [
    "tpch_sf1", "tpch_sf10", "tpch_sf100",
    "tpcds_sf1", "tpcds_sf10", "tpcds_sf100",
    "hybench_sf1", "hybench_sf10",
    "airline", "credit", "carcinogenesis",
    "employee", "financial", "geneea", "hepatitis"
]

def _init_worker():
    """Ignore SIGINT in worker threads – let main thread handle Ctrl-C."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--data_dir", default="/home/wuy/query_costs_trace/")
    pa.add_argument("--port",     type=int, default=44444)
    pa.add_argument("--timeout",  type=int, default=60_000)
    pa.add_argument("--start",    type=int, default=0,
                    help="index in DATASETS to start from")
    pa.add_argument("--end",      type=int, default=len(DATASETS),
                    help="exclusive end index in DATASETS")
    pa.add_argument("--jobs",     type=int, default=os.cpu_count(),
                    help="number of datasets to collect in parallel "
                         "(default = #CPU cores)")
    pa.add_argument("--debug",    action="store_true")
    args = pa.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    sel = DATASETS[args.start : args.end]
    jobs = min(max(1, args.jobs), len(sel))

    logging.info(f"Datasets: {sel}")
    logging.info(f"Running with {jobs} parallel job(s)")

    with ThreadPoolExecutor(max_workers=jobs) as pool:

        fut2ds = { pool.submit(collect_dataset, ds, args): ds for ds in sel }

        try:
            for fut in as_completed(fut2ds):
                ds = fut2ds[fut]
                try:
                    fut.result()
                    logging.info(f"[{ds}] ✓ finished")
                except Exception:
                    logging.error(f"[{ds}] ✗ fatal:\n{traceback.format_exc()}")

        except KeyboardInterrupt:
            logging.warning("Ctrl-C received – cancelling all tasks …")
            for f in fut2ds:
                f.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            raise   # re-raise so shell sees non-zero exit

if __name__ == "__main__":
    main()
