# -*- coding: utf-8 -*-
"""
Python re‑implementation of the core parts of your C++ row/column‑router pipeline
so that you can run modern feature‑analysis toolkits (CAE, HSIC‑Lasso, TabNet,
VIB, etc.) entirely in Python / Jupyter without recompiling the PolarDB kernel.

* Dependencies (PyPI):
  pip install lightgbm==4.3.0 pytorch-tabnet==4.0.0 shap pyhsiclasso concrete-autoencoder-lite
  # plus pandas, numpy, scikit‑learn which you probably already have

* Directory layout expected  —— identical to your C++ world
  DATASET_DIR/
    tpch_sf100/
        query_costs.csv
        row_plans/*.json
        column_plans/*.json          # optional, only used by GNN encoder
    tpcds_sf10/
        ...

* Entry points
  1.  python feature_analysis.py extract  <dataset_dir>  →  features.npy / meta.pkl
  2.  python feature_analysis.py train    <features.npy> →  LightGBM model + shap.png
  3.  python feature_analysis.py vib      <features.npy> →  VIB‑MLP feature mask
  4.  python feature_analysis.py select   <features.npy> --method hsic|shap|cae

You can call the individual utility functions from your own notebook as well.

The code purposefully follows the same naming and algorithmic structure as your
original C++ (plan2feat, Agg walk, Sample, Split, etc.) so that you can diff
one‑to‑one when in doubt.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
#  1.  Helpers (mostly 1‑to‑1 ports of your C++ free functions)
# ---------------------------------------------------------------------------

NUMBER_RE = re.compile(r"([\d.]+)([KMG]?)", re.I)

SCALE_MAP = {"": 1.0, "K": 1e3, "M": 1e6, "G": 1e9}

def str_size_to_num(s: str) -> float:
    """Convert MySQL DATA_LENGTH strings like "3.5M" to float(bytes)."""
    if not s:
        return 0.0
    m = NUMBER_RE.match(s.strip())
    if not m:
        return 0.0
    val, suf = m.groups()
    return float(val) * SCALE_MAP.get(suf.upper(), 1.0)

def log_tanh(v: float, c: float = 20.0) -> float:
    return math.tanh(math.log1p(max(0.0, v)) / c)

def safe_f(o: Any, k: str | None = None) -> float:  # overload like C++
    if k is None:
        v = o
    else:
        if not isinstance(o, dict) or k not in o:
            return 0.0
        v = o[k]
    try:
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            return float(v)
    except Exception:
        return 0.0
    return 0.0

# ---------------------------------------------------------------------------
#  2.  Aggregator used inside plan2feat  (faithful to your C++ Agg struct)
# ---------------------------------------------------------------------------
# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from typing import Any, Dict, List
import math
import numpy as np

NUM_FEATS = 124          # === 与 C++ ORIG_FEATS 完全一致 ===

# ---------------------------------------------------------------------------
#  Human-readable names for the 124-dim plan2feat vector
#  (index == position in the list)
# ---------------------------------------------------------------------------

FEAT_NAMES: list[str] = [
    # ---- 0-6  avg per-table costs / rows ---------------------------------
    "avg_rows_examined_logtanh",         # 0
    "avg_rows_produced_logtanh",         # 1
    "avg_filtered_pct_logtanh",          # 2
    "avg_read_cost_logtanh",             # 3
    "avg_eval_cost_logtanh",             # 4
    "avg_prefix_cost_logtanh",           # 5
    "avg_data_read_logtanh",             # 6
    # ---- 7-12 access-type shares ----------------------------------------
    "ratio_access_range",                # 7
    "ratio_access_ref",                  # 8
    "ratio_access_eq_ref",               # 9
    "ratio_access_index",                # 10
    "ratio_access_full",                 # 11
    "ratio_using_index_hint",            # 12
    # ---- 13-22 selectivity / shape / rc-ec ratios ------------------------
    "selectivity_mean",                  # 13
    "selectivity_min",                   # 14
    "selectivity_max",                   # 15
    "plan_max_depth",                    # 16
    "fanout_max",                        # 17
    "has_group_by",                      # 18
    "has_order_by",                      # 19
    "uses_temp_table",                   # 20
    "avg_read_eval_ratio",               # 21
    "max_read_eval_ratio",               # 22
    # ---- 23-27 root level & derived ratios -------------------------------
    "log_query_cost",                    # 23
    "root_rows_logtanh",                 # 24
    "avg_prefix_read_ratio",             # 25
    "avg_read_rows_ratio",               # 26
    "avg_eval_rows_ratio",               # 27
    # ---- 28-41 misc counters & flags -------------------------------------
    "is_single_table",                   # 28
    "is_multi_table",                    # 29
    "depth_index_use_signal",            # 30
    "index_vs_fullscan_signal",          # 31
    "table_count",                       # 32
    "avg_pk_len",                        # 33
    "max_prefix_cost_logtanh",           # 34
    "min_read_cost_logtanh",             # 35
    "join_fraction",                     # 36
    "rows_examined_root_ratio",          # 37
    "selectivity_range",                 # 38
    "index_use_ratio",                   # 39
    "log_query_cost_dup",                # 40
    "is_big_cost_query",                 # 41
    "avg_cover_ratio",                   # 42
    "all_covering_flag",                 # 43
    # ---- 44-56 repeats & histograms --------------------------------------
    "filtered_vs_selectivity",           # 44
    "table_count_dup",                   # 45
    "log_table_count",                   # 46
    "sum_pk_len",                        # 47
    "avg_pk_len_dup",                    # 48
    "cover_count",                       # 49
    "cover_ratio_dup",                   # 50
    "ratio_using_index_rep",             # 51
    "ratio_range_rep",                   # 52
    "ratio_ref_rep",                     # 53
    "ratio_eq_ref_rep",                 # 54
    "ratio_index_rep",                   # 55
    "ratio_full_rep",                    # 56
    "avg_prefix_cost_norm",              # 57
    "min_read_cost_dup",                 # 58
    "selectivity_range_dup",             # 59
    # ---- 60-65 extremes & row-win signals --------------------------------
    "max_read_eval_ratio_dup",           # 60
    "fanout_max_dup",                    # 61
    "sel_max_over_sel_min",              # 62
    "outer_rows_logtanh",                # 63
    "eq_chain_depth",                    # 64
    "late_fanout_signal",                # 65
    # ---- 66-76 table-level metadata --------------------------------------
    "rel_rows_avg",                      # 66
    "rel_rows_max",                      # 67
    "rel_data_mb",                       # 68
    "rel_index_mb",                      # 69
    "fragmentation_max",                 # 70
    "partition_avg",                     # 71
    "update_pct_max",                    # 72
    "idx_count_norm16",                  # 73
    "unique_index_ratio",                # 74
    "cover_ratio_avg",                   # 75
    "pk_len_log_avg",                    # 76
    "reserved1",                         # 77
    "reserved2",                         # 78
    "reserved3",                         # 79
    "any_compressed_tbl",                # 80
    "cpu_cores_norm64",                  # 81
    "bp_hit_placeholder",                # 82
    "imci_hit_placeholder",              # 83
    # ---- 84-95 column histogram buckets ----------------------------------
    "col_width_le4_ratio",               # 84
    "col_width_le16_ratio",              # 85
    "col_width_le64_ratio",              # 86
    "col_width_gt64_ratio",              # 87
    "ndv_le1k_ratio",                    # 88
    "ndv_le100k_ratio",                  # 89
    "ndv_gt100k_ratio",                  # 90
    "dtype_int_ratio",                   # 91
    "dtype_float_ratio",                 # 92
    "dtype_string_ratio",                # 93
    "dtype_datetime_ratio",              # 94
    "dtype_bool_ratio",                  # 95
    # ---- 96-123 fan-out & heuristic flags -------------------------------
    "depth3_prefix_ratio",               # 96
    "cum_fanout_log",                    # 97
    "fanout_amp_per_depth",              # 98
    "rows_max_logscale",                 # 99
    "data_mb_total_logscale",            # 100
    "qcost_per_row",                     # 101
    "log_qcost_per_krows",               # 102
    "log_qcost_per_mbread",              # 103
    "point_access_flag",                 # 104
    "narrow_row_flag",                   # 105
    "hash_prefetch_flag",                # 106
    "table_cnt_norm64",                  # 107
    "minmax_no_group_flag",              # 108
    "probe_vs_outer_log",                # 109
    "biggest_ratio_log",                 # 110
    "root_rows_norm1e8",                 # 111
    "point_and_narrow_flag",             # 112
    "mid_fan_signal_flag",               # 113
    "agg_flag_dup",                      # 114
    # predicate-selectivity buckets
    "sel_avg_le_1pct",                   # 115
    "sel_avg_1_10pct",                   # 116
    "sel_avg_10_50pct",                  # 117
    "sel_avg_gt_50pct",                  # 118
    # predicate-pushdown buckets
    "pushdown_none",                     # 119
    "pushdown_0_30",                     # 120
    "pushdown_30_70",                    # 121
    "pushdown_gt_70",                    # 122
    "join_ratio_max_log",                # 123
]
assert len(FEAT_NAMES) == NUM_FEATS


# ---------------------------------------------------------------------------
#  0.  一些微型工具（同名函数在你工程里已有时可删）
# ---------------------------------------------------------------------------
def log_tanh(v: float, c: float = 20.0) -> float:
    return math.tanh(math.log1p(max(0.0, v)) / c)

def log_scale(v: float, k: float = 1e6) -> float:
    return math.log1p(v) / math.log1p(k)

def safe_f(node: Dict[str, Any], k: str) -> float:
    v = node.get(k, 0)
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0

def str_size_to_num(s: str) -> float:
    if not s:
        return 0.0
    s = s.strip()
    unit = s[-1].upper()
    mul = 1.0
    if unit == "G":
        mul = 1e9
        s = s[:-1]
    elif unit == "M":
        mul = 1e6
        s = s[:-1]
    elif unit == "K":
        mul = 1e3
        s = s[:-1]
    try:
        return float(s) * mul
    except ValueError:
        return 0.0

from dataclasses import dataclass
from typing import Dict, Set
import pymysql, math, json, logging

logging.basicConfig(level=logging.INFO)

# ---------- 结构体 ----------
@dataclass
class ColStats:
    avg_width: float = 8.0        # 字节
    ndv:       float = 1_000.0    # 近似基数
    dtype:     str   = "string"   # int/float/string/datetime/bool

@dataclass
class TblStats:
    rows:       float = 0.0
    data_mb:    float = 0.0
    index_mb:   float = 0.0
    frag_ratio: float = 0.0
    partitions: int   = 1
    upd_pct:    float = 0.0
    idx_cnt:    int   = 0
    uniq_cnt:   int   = 0
    cover_cols: int   = 0
    total_cols: int   = 0
    pk_len:     int   = 0
    compressed: bool  = False

# ---------- 全局缓存 ----------
_COL_STATS: Dict[str, ColStats]      = {}   # key = db.tbl.col
_TBL_STATS: Dict[str, TblStats]      = {}   # key = db.tbl
_INDEX_COLS: Dict[str, Set[str]]     = {}   # key = index-name


def _connect(host: str, port: int, user: str, passwd: str, db: str | None = None):
    return pymysql.connect(host=host, user=user, password=passwd,
                           port=port, database=db,
                           charset="utf8mb4", autocommit=True,
                           cursorclass=pymysql.cursors.DictCursor)


def _map_dtype(t: str) -> str:
    t = t.lower()
    if "int" in t:          return "int"
    if t in ("float", "double") or "decimal" in t:  return "float"
    if t in ("date", "datetime", "timestamp", "time"): return "datetime"
    if t in ("bool", "boolean"): return "bool"
    return "string"

def _ndv_from_hist(hist_json: str) -> float:
    try:
        j = json.loads(hist_json)
        return sum(b.get("distinct-range", 0.0) for b in j["buckets"]) or -1.0
    except Exception:
        return -1.0

def populate_col_stats(host: str, port: int,
                       user: str, passwd: str,
                       schemas: list[str]) -> None:
    """一次性扫完需要的 schema，写入 _COL_STATS"""
    global _COL_STATS
    _COL_STATS.clear()

    conn = _connect(host, port, user, passwd, "information_schema")
    with conn.cursor() as cur:
        in_list = ",".join(f"'{s}'" for s in schemas)
        sql = f"""
        SELECT  C.TABLE_SCHEMA, C.TABLE_NAME, C.COLUMN_NAME,
                C.DATA_TYPE,
                COALESCE(C.CHARACTER_OCTET_LENGTH,
                         CEILING(C.NUMERIC_PRECISION/8), 8) AS AVG_LEN,
                CS.HISTOGRAM
        FROM    COLUMNS C
        LEFT JOIN COLUMN_STATISTICS CS
               ON CS.SCHEMA_NAME=C.TABLE_SCHEMA
              AND CS.TABLE_NAME =C.TABLE_NAME
              AND CS.COLUMN_NAME=C.COLUMN_NAME
        WHERE   C.TABLE_SCHEMA IN ({in_list})
        """
        cur.execute(sql)
        for row in cur:
            key = f"{row['TABLE_SCHEMA']}.{row['TABLE_NAME']}.{row['COLUMN_NAME']}"
            cs  = ColStats()
            cs.avg_width = max(1.0, float(row['AVG_LEN']))
            cs.dtype     = _map_dtype(row['DATA_TYPE'])
            ndv = _ndv_from_hist(row['HISTOGRAM']) if row['HISTOGRAM'] else -1
            cs.ndv       = ndv if ndv > 0 else 1_000.0   # 平稳默认
            _COL_STATS[key] = cs
    conn.close()
    logging.info("populate_col_stats: %d columns", len(_COL_STATS))

def populate_table_stats(host: str, port: int,
                         user: str, passwd: str,
                         schemas: list[str]) -> None:
    global _TBL_STATS
    _TBL_STATS.clear()

    conn = _connect(host, port, user, passwd, "information_schema")
    with conn.cursor() as cur:
        in_list = ",".join(f"'{s}'" for s in schemas)

        # --- TABLES 基础 ---
        cur.execute(f"""
            SELECT TABLE_SCHEMA, TABLE_NAME,
                   TABLE_ROWS, DATA_LENGTH, INDEX_LENGTH,
                   DATA_FREE, ROW_FORMAT
            FROM   TABLES
            WHERE  TABLE_SCHEMA IN ({in_list})
        """)
        for r in cur:
            key = f"{r['TABLE_SCHEMA']}.{r['TABLE_NAME']}"
            s = _TBL_STATS.setdefault(key, TblStats())
            s.rows     = float(r['TABLE_ROWS'] or 0)
            s.data_mb  = float(r['DATA_LENGTH'] or 0) / 1e6
            s.index_mb = float(r['INDEX_LENGTH'] or 0) / 1e6
            s.frag_ratio = (float(r['DATA_FREE'] or 0) /
                            max(1.0, r['DATA_LENGTH']+r['INDEX_LENGTH']))
            s.compressed = ("compress" in (r['ROW_FORMAT'] or "").lower())

        # --- 分区数 ---
        cur.execute(f"""
            SELECT TABLE_SCHEMA, TABLE_NAME, COUNT(*) AS P
            FROM   PARTITIONS
            WHERE  TABLE_SCHEMA IN ({in_list})
            GROUP  BY 1,2
        """)
        for r in cur:
            _TBL_STATS[f"{r['TABLE_SCHEMA']}.{r['TABLE_NAME']}"].partitions = int(r['P'])

        # --- INDEX 信息 ---
        cur.execute(f"""
            SELECT TABLE_SCHEMA, TABLE_NAME, NON_UNIQUE,
                   COLUMN_NAME, SEQ_IN_INDEX, INDEX_NAME, SUB_PART
            FROM   STATISTICS
            WHERE  TABLE_SCHEMA IN ({in_list})
            ORDER  BY TABLE_SCHEMA, TABLE_NAME, INDEX_NAME, SEQ_IN_INDEX
        """)
        for r in cur:
            key = f"{r['TABLE_SCHEMA']}.{r['TABLE_NAME']}"
            ts = _TBL_STATS.setdefault(key, TblStats())
            ts.idx_cnt += 1
            if r['NON_UNIQUE'] == 0:
                ts.uniq_cnt += 1
            ts.total_cols += 1
            if r['SUB_PART'] in (None, 0):
                ts.cover_cols += 1
            # 粗估 PK 长度（第一列全长）
            if r['INDEX_NAME'] == "PRIMARY" and r['SEQ_IN_INDEX'] == 1:
                ts.pk_len += 8

        # --- perf_schema 行更新读比例 ---
        # cur.execute(f"""
        #     SELECT OBJECT_SCHEMA, OBJECT_NAME, ROWS_UPDATED, ROWS_READ
        #     FROM   performance_schema.table_io_waits_summary_by_table
        #     WHERE  OBJECT_SCHEMA IN ({in_list})
        # """)
        # for r in cur:
        #     key = f"{r['OBJECT_SCHEMA']}.{r['OBJECT_NAME']}"
        #     ts  = _TBL_STATS.setdefault(key, TblStats())
        #     upd = float(r['ROWS_UPDATED'] or 0)
        #     rd  = float(r['ROWS_READ'] or 0)
        #     ts.upd_pct = upd / max(1.0, rd)
        try:
            cur.execute(f"""
                SELECT OBJECT_SCHEMA, OBJECT_NAME,
                    /* 8.0+ */  ROWS_UPDATED  AS U,
                    /* 5.7  */  COUNT_WRITE  AS CW,
                    /* 8.0+ */  ROWS_READ     AS R,
                    /* 5.7  */  COUNT_READ   AS CR
                FROM   performance_schema.table_io_waits_summary_by_table
                WHERE  OBJECT_SCHEMA IN ({in_list})
            """)
            for r in cur:
                key = f"{r['OBJECT_SCHEMA']}.{r['OBJECT_NAME']}"
                ts  = _TBL_STATS.setdefault(key, TblStats())
                upd = float(r.get("U")  or r.get("CW") or 0)
                rd  = float(r.get("R")  or r.get("CR") or 0)
                ts.upd_pct = upd / max(1.0, rd)
        except pymysql.err.OperationalError as e:
            logging.warning("skip update/read ratio - %s", e)

    conn.close()
    logging.info("populate_table_stats: %d tables", len(_TBL_STATS))


def get_index_defs(host: str, port: int,
                   user: str, passwd: str,
                   schemas: list[str]) -> None:
    """填充 _INDEX_COLS：index-name -> {col1,col2,…}"""
    global _INDEX_COLS
    _INDEX_COLS.clear()

    conn = _connect(host, port, user, passwd, "information_schema")
    with conn.cursor() as cur:
        in_list = ",".join(f"'{s}'" for s in schemas)
        cur.execute(f"""
            SELECT TABLE_SCHEMA, TABLE_NAME,
                   INDEX_NAME, COLUMN_NAME
            FROM   STATISTICS
            WHERE  TABLE_SCHEMA IN ({in_list})
            ORDER  BY TABLE_SCHEMA, TABLE_NAME, INDEX_NAME, SEQ_IN_INDEX
        """)
        for r in cur:
            idx = r['INDEX_NAME']
            _INDEX_COLS.setdefault(idx, set()).add(r['COLUMN_NAME'])
    conn.close()
    logging.info("get_index_defs: %d indexes", len(_INDEX_COLS))


def lookup_col_stats(db: str, tbl: str, col: str) -> ColStats:
    return _COL_STATS.get(f"{db}.{tbl}.{col}", ColStats())

def lookup_tbl(db: str, tbl: str) -> TblStats:
    return _TBL_STATS.get(f"{db}.{tbl}", TblStats())





# ---------------------------------------------------------------------------
#  1.  完整聚合结构 Agg  (字段保持与 C++ 同名)
# ---------------------------------------------------------------------------
@dataclass
class Agg:
    # numeric accumulators
    re: float = 0.0
    rp: float = 0.0
    f:  float = 0.0
    rc: float = 0.0
    ec: float = 0.0
    pc: float = 0.0
    dr: float = 0.0
    cnt: int   = 0

    # extrema & sums
    maxPrefix: float = 0.0
    minRead:   float = 1e30
    selSum:    float = 0.0
    selMin:    float = 1.0
    selMax:    float = 0.0
    fanoutMax: float = 0.0
    lateFanMax: float = 0.0
    pcDepth3:  float = 0.0
    ratioSum:  float = 0.0
    ratioMax:  float = 0.0

    # plan flags
    grp: bool = False
    ord: bool = False
    tmp: bool = False
    hashJoin: bool = False

    # counters
    cRange: int = 0
    cRef:   int = 0
    cEq:    int = 0
    cIdx:   int = 0
    cFull:  int = 0
    idxUse: int = 0

    preds_total:  int = 0
    preds_pushed: int = 0

    sumPK:       int = 0
    coverCount:  int = 0

    # row-win signals
    outerRows:    float = 0.0
    eqChainDepth: int   = 0
    _curEqChain:  int   = 0

    join_ratio_max: float = 0.0
    maxDepth: int = 0

# ---------------------------------------------------------------------------
#  2.  递归遍历，与 C++ walk() 行为保持一致
# ---------------------------------------------------------------------------
def walk(node: Any, a: Agg, depth: int = 1) -> None:
    if isinstance(node, dict):
        # ---- TABLE node ----
        if "table" in node and isinstance(node["table"], dict):
            t = node["table"]
            ci = t.get("cost_info", {})

            # numeric fields
            re_val = safe_f(t, "rows_examined_per_scan")
            rp_val = safe_f(t, "rows_produced_per_join")
            fl_val = safe_f(t, "filtered")
            rc_val = safe_f(ci, "read_cost")
            ec_val = safe_f(ci, "eval_cost")
            pc_val = safe_f(ci, "prefix_cost")
            dr_raw = ci.get("data_read_per_join", 0)
            dr_val = str_size_to_num(dr_raw) if isinstance(dr_raw, str) else float(dr_raw)

            # aggregate
            a.re += re_val; a.rp += rp_val; a.f += fl_val
            a.rc += rc_val; a.ec += ec_val; a.pc += pc_val; a.dr += dr_val; a.cnt += 1
            if depth == 3:
                a.pcDepth3 += pc_val

            a.maxPrefix = max(a.maxPrefix, pc_val)
            a.minRead   = min(a.minRead,  rc_val)

            if re_val > 0:
                sel = rp_val / re_val
                a.selSum += sel
                a.selMin  = min(a.selMin, sel)
                a.selMax  = max(a.selMax, sel)
                a.fanoutMax = max(a.fanoutMax, sel)
                if depth >= 4:
                    a.lateFanMax = max(a.lateFanMax, sel)

            ratio = rc_val / ec_val if ec_val > 0 else rc_val
            a.ratioSum += ratio
            a.ratioMax  = max(a.ratioMax, ratio)

            # predicates
            if any(k in t for k in ("attached_condition",
                                    "pushed_index_condition",
                                    "pushed_join_condition")):
                a.preds_total += 1
                if any(k in t for k in ("pushed_index_condition",
                                        "pushed_join_condition")):
                    a.preds_pushed += 1

            # access-type
            at = t.get("access_type", "ALL")
            if   at == "range":  a.cRange += 1
            elif at == "ref":    a.cRef   += 1
            elif at == "eq_ref": a.cEq    += 1
            elif at == "index":  a.cIdx   += 1
            else:                a.cFull  += 1

            if t.get("using_index"):
                a.idxUse += 1
            if t.get("using_join_buffer") == "hash join":
                a.hashJoin = True

            # row-wins signals
            if a.outerRows == 0 and at != "ALL":
                a.outerRows = re_val
            if at == "eq_ref":
                a._curEqChain += 1
                a.eqChainDepth = max(a.eqChainDepth, a._curEqChain)
            else:
                a._curEqChain = 0

        # plan-level flags
        if node.get("grouping_operation"):                 a.grp = True
        if node.get("ordering_operation") or node.get("using_filesort"): a.ord = True
        if node.get("using_temporary_table"):              a.tmp = True

        # join ratio (first two branches of nested_loop)
        if "nested_loop" in node and isinstance(node["nested_loop"], list):
            nl = node["nested_loop"]
            if len(nl) >= 2:
                def rows_of(x):
                    return safe_f(x.get("table", x), "rows_produced_per_join")
                l, r = rows_of(nl[0]), rows_of(nl[1])
                if l > 0 and r > 0:
                    a.join_ratio_max = max(a.join_ratio_max,
                                           max(l, r) / max(1.0, min(l, r)))

        # recurse
        for k, v in node.items():
            if k != "table":
                walk(v, a, depth + 1)

    elif isinstance(node, list):
        for item in node:
            walk(item, a, depth)

    a.maxDepth = max(a.maxDepth, depth)

# ---------------------------------------------------------------------------
#  3.  plan2feat() – 124 维与 C++ 完全同步
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
#  plan2feat – faithful Python port of the 124-dim C++ extractor
# ---------------------------------------------------------------------------
import math, os
from typing import Any, Dict, List
import numpy as np

def _normalise(hist: List[float]) -> List[float]:
    s = sum(hist) or 1.0
    return [v / s for v in hist]

def plan2feat(plan_json: Dict[str, Any]) -> np.ndarray:
    if "query_block" not in plan_json:
        raise ValueError("invalid plan: no query_block")

    # ── 0) pick the root query-block (skip UNION branches) ─────────────────
    qb = plan_json["query_block"]
    if "union_result" in qb:
        specs = qb["union_result"].get("query_specifications", [])
        if specs:
            qb = specs[0]["query_block"]

    # ── 1) aggregate per-table metrics ─────────────────────────────────────
    a = Agg()
    walk(qb, a)
    if a.cnt == 0:
        raise ValueError("plan has no table nodes")

    inv = 1.0 / a.cnt
    feats: List[float] = []
    push = feats.append          # tiny macro

    qCost   = safe_f(qb.get("cost_info", {}), "query_cost")
    rootRow = safe_f(qb, "rows_produced_per_join")

    # ───────────────────────── 0‒64 : *identical* to C++ ───────────────────
    # basic costs / rows
    push(log_tanh(a.re * inv)); push(log_tanh(a.rp * inv)); push(log_tanh(a.f  * inv))
    push(log_tanh(a.rc * inv)); push(log_tanh(a.ec * inv)); push(log_tanh(a.pc * inv))
    push(log_tanh(a.dr * inv))

    # access-type counters
    push(a.cRange*inv); push(a.cRef*inv); push(a.cEq*inv)
    push(a.cIdx*inv);   push(a.cFull*inv); push(a.idxUse*inv)

    # selectivity & shape
    push(a.selSum*inv); push(a.selMin); push(a.selMax)
    push(a.maxDepth);   push(a.fanoutMax)

    # flags
    push(float(a.grp)); push(float(a.ord)); push(float(a.tmp))

    # rc/ec ratios
    push(a.ratioSum*inv); push(a.ratioMax)

    # root-level info
    push(math.log1p(qCost)/15.0); push(log_tanh(rootRow))

    # derived ratios
    push(log_tanh((a.pc*inv)/max(1e-6, a.rc*inv)))
    push(log_tanh((a.rc*inv)/max(1e-6, a.re*inv)))
    push(log_tanh((a.ec*inv)/max(1e-6, a.re*inv)))

    # misc counters
    push(float(a.cnt == 1)); push(float(a.cnt > 1))
    push(log_tanh(a.maxDepth * (a.idxUse*inv)))
    push(log_tanh((a.idxUse*inv)/max(a.cFull*inv, 1e-3)))

    # other stats
    push(a.cnt)
    push(a.cnt and a.sumPK/a.cnt or 0.0)
    push(log_tanh(a.maxPrefix))
    push(log_tanh(a.minRead if a.minRead < 1e30 else 0.0))
    push(a.cnt>1 and (a.cnt-1)/a.cnt or 0.0)
    push(rootRow>0 and a.re/rootRow or 0.0)
    push(a.selMax - a.selMin)
    push(a.idxUse / max(1, a.cRange+a.cRef+a.cEq+a.cIdx))

    # covering-index & big-cost flags
    push(math.log1p(qCost)/15.0)
    push(float(math.log1p(qCost) > 11.5))
    push(a.cnt and a.coverCount/a.cnt or 0.0)
    push(float(a.coverCount == a.cnt))

    # log-diffs & counts
    push(log_tanh(a.re*inv) - log_tanh(a.selSum*inv))
    push(a.cnt)
    push(log_tanh(a.cnt))

    # PK / cover counters
    push(a.sumPK); push(a.cnt and a.sumPK/a.cnt or 0.0)
    push(a.coverCount); push(a.cnt and a.coverCount/a.cnt or 0.0)

    # repeated access shares
    push(a.idxUse*inv); push(a.cRange*inv); push(a.cRef*inv)
    push(a.cEq*inv);    push(a.cIdx*inv);   push(a.cFull*inv)

    # prefix / read extremes
    push(log_tanh(a.maxPrefix*inv))
    push(log_tanh(a.minRead if a.minRead < 1e30 else 0.0))
    push(a.selMax - a.selMin)

    # extreme ratios
    push(a.ratioMax)
    push(a.fanoutMax)
    push(a.selMin>0 and a.selMax/a.selMin or 0.0)

    # row-win signals
    push(log_tanh(a.outerRows))
    push(a.eqChainDepth)

    # ───────────────────────── 65 : late fan-out ───────────────────────────
    late_f = 0.0
    if a.lateFanMax > a.selMin + 1e-9:
        late_f = min(1.0, math.log(a.lateFanMax/max(1e-6, a.selMin))/4.0)
    push(late_f)                                            # 65

    # ── 2) table-level metadata (66-82) ────────────────────────────────────
    touched_tbls = set()
    def _collect_tbl(n: Any, db_hint: str=""):
        if isinstance(n, dict):
            if "table" in n and isinstance(n["table"], dict):
                t = n["table"]
                tbl = t.get("table_name", "")
                db  = t.get("table_schema", db_hint)
                touched_tbls.add( (db, tbl) )
            for v in n.values():
                _collect_tbl(v, db_hint)
        elif isinstance(n, list):
            for v in n: _collect_tbl(v, db_hint)
    _collect_tbl(qb)

    rows_avg=rows_max=data_mb_avg=idx_mb_avg=frag_max=part_avg=upd_max=0.0
    idx_cnt_avg=uniq_ratio_avg=cover_ratio_avg=pk_len_log_avg=0.0
    compressed_any=False
    tbl_n = len(touched_tbls)

    for db,tbl in touched_tbls:
        ts = lookup_tbl(db, tbl)
        rows_avg      += ts.rows
        rows_max       = max(rows_max, ts.rows)
        data_mb_avg   += ts.data_mb
        idx_mb_avg    += ts.index_mb
        frag_max       = max(frag_max, ts.frag_ratio)
        part_avg      += ts.partitions
        upd_max        = max(upd_max, ts.upd_pct)
        idx_cnt_avg   += ts.idx_cnt
        uniq_ratio_avg += ts.idx_cnt and ts.uniq_cnt/ts.idx_cnt or 0.0
        cover_ratio_avg+= ts.total_cols and ts.cover_cols/ts.total_cols or 0.0
        pk_len_log_avg += math.log1p(ts.pk_len)
        compressed_any = compressed_any or ts.compressed

    if tbl_n:
        rows_avg      /= tbl_n
        data_mb_avg   /= tbl_n
        idx_mb_avg    /= tbl_n
        part_avg      /= tbl_n
        idx_cnt_avg   /= tbl_n
        uniq_ratio_avg/= tbl_n
        cover_ratio_avg/=tbl_n
        pk_len_log_avg/= tbl_n

    rel_rows_avg = rows_avg / max(1.0, rootRow)
    rel_rows_max = rows_max / max(1.0, rows_avg or 1.0)
    rel_data_mb  = data_mb_avg / max(1.0, data_mb_avg+idx_mb_avg)
    rel_idx_mb   = idx_mb_avg  / max(1.0, data_mb_avg)

    push(log_scale(rel_rows_avg, 1e2))      # 66
    push(log_scale(rel_rows_max, 1e2))      # 67
    push(log_scale(rel_data_mb , 1e0))      # 68
    push(log_scale(rel_idx_mb  , 1e0))      # 69
    push(frag_max)                          # 70
    push(part_avg)                          # 71
    push(upd_max)                           # 72
    push(idx_cnt_avg/16.0)                  # 73
    push(uniq_ratio_avg)                    # 74
    push(cover_ratio_avg)                   # 75
    push(pk_len_log_avg)                    # 76
    push(0.0); push(0.0); push(0.0)         # 77-79 reserved
    push(float(compressed_any))             # 80
    push(os.cpu_count()/64.0)               # 81
    push(0.0); push(0.0)                    # 82-83 (bp-hit / IMCI-hit)

    # ── 3) column-histogram features (84-95) ───────────────────────────────
    touched_cols = set()
    def _collect_col(n: Any, db_hint: str="", tbl_hint: str=""):
        if isinstance(n, dict):
            if "table" in n and isinstance(n["table"], dict):
                t = n["table"]
                tbl_hint = t.get("table_name", tbl_hint)
                db_hint  = t.get("table_schema", db_hint)
                if "used_columns" in t and isinstance(t["used_columns"], list):
                    for c in t["used_columns"]:
                        if isinstance(c, str):
                            touched_cols.add( (db_hint, tbl_hint, c) )
            for v in n.values():
                _collect_col(v, db_hint, tbl_hint)
        elif isinstance(n, list):
            for v in n: _collect_col(v, db_hint, tbl_hint)
    _collect_col(qb)

    width_hist  = [0.0, 0.0, 0.0, 0.0]   # ≤4 / ≤16 / ≤64 / >64
    ndv_hist    = [0.0, 0.0, 0.0]        # ≤1e3 / ≤1e5 / >1e5
    type_hist   = [0.0, 0.0, 0.0, 0.0, 0.0]  # int/float/str/datetime/bool

    dtype2idx = {"int":0,"float":1,"string":2,"datetime":3,"bool":4}

    for db,tbl,col in touched_cols:
        cs = lookup_col_stats(db,tbl,col)
        w = cs.avg_width
        if   w<=4:  width_hist[0]+=1
        elif w<=16: width_hist[1]+=1
        elif w<=64: width_hist[2]+=1
        else:       width_hist[3]+=1

        n = cs.ndv
        if   n<=1e3: ndv_hist[0]+=1
        elif n<=1e5: ndv_hist[1]+=1
        else:        ndv_hist[2]+=1

        type_hist[ dtype2idx.get(cs.dtype,"string") ] += 1

    width_hist = _normalise(width_hist)
    ndv_hist   = _normalise(ndv_hist)
    type_hist  = _normalise(type_hist)

    for v in width_hist: push(v)          # 84-87
    for v in ndv_hist:   push(v)          # 88-90
    for v in type_hist:  push(v)          # 91-95

    # depth-3 prefix-cost ratio
    pc_d3 = max(1e-6, a.pcDepth3)
    push(log_tanh( (a.pc if a.pc>0 else 1e-6) / pc_d3 ))  # 96

    # ── 4) fan-out & heuristics (97-123) ───────────────────────────────────
    cum_fan = amp_step = 0.0
    if a.outerRows>0 and rootRow > a.outerRows*1.0001:
        cum_fan  = math.log(rootRow/a.outerRows)
        amp_step = cum_fan / max(1, a.maxDepth-1)

    rows_max_raw   = rows_max
    data_mb_total  = data_mb_avg * tbl_n
    qcost_per_row  = qCost / max(1.0, rows_max_raw)
    kRows          = max(1.0, rows_max_raw / 1e3)
    mbRead         = max(1.0, data_mb_total)

    push(cum_fan)                                        # 97
    push(amp_step)                                       # 98
    push(log_scale(rows_max_raw ,1e10))                  # 99
    push(log_scale(data_mb_total,1e10))                  #100
    push(qcost_per_row)                                  #101
    push(math.log1p(qCost/kRows)/15.0)                   #102
    push(math.log1p(qCost/mbRead)/15.0)                  #103

    point_ratio  = (a.cRef+a.cEq)*inv
    narrow_ratio = width_hist[0]+width_hist[1]
    hash_prefetch= a.hashJoin and (a.idxUse*inv > 0.5)

    push(float(point_ratio>0.70))                        #104
    push(float(narrow_ratio>0.80))                       #105
    push(float(hash_prefetch))                           #106

    tbl_cnt       = tbl_n
    agg_flag      = False   # MIN/MAX-w/o-GROUP: implement if you need
    probe_vs_outer= (rootRow/max(1.0,a.outerRows)) if a.outerRows else 0.0
    biggest_ratio = (a.selMax / a.outerRows) if a.outerRows else 1.0
    late_rows_norm= log_scale(rootRow,1e8)

    push(min(1.0, tbl_cnt/64.0))                         #107
    push(float(agg_flag))                                #108
    push(log_tanh(probe_vs_outer))                       #109
    push(log_tanh(biggest_ratio))                        #110
    push(late_rows_norm)                                 #111

    outer_rows_norm = a.outerRows and log_scale(a.outerRows,1e6) or 0.0
    fanout = a.fanoutMax
    point_and_narrow = (point_ratio>0.70 and narrow_ratio>0.80)
    mid_fan_signal   = (outer_rows_norm<0.05) and (5<fanout<20) and (cum_fan>0.4)
    push(float(point_and_narrow))                        #112
    push(float(mid_fan_signal))                          #113
    push(float(agg_flag))                                #114

    # predicate-selectivity one-hots (4)
    sel_avg = a.selSum*inv
    push(float(sel_avg<=0.01)); push(float(0.01<sel_avg<=0.10))
    push(float(0.10<sel_avg<=0.50)); push(float(sel_avg>0.50))  #115-118

    # predicate pushdown rate one-hots (4)
    pdr = a.preds_total and a.preds_pushed/a.preds_total or 0.0
    push(float(pdr==0.0)); push(float(0.0<pdr<=0.30))
    push(float(0.30<pdr<=0.70)); push(float(pdr>0.70))          #119-122

    push(math.log1p(a.join_ratio_max))                          #123

    if len(feats) != NUM_FEATS:
        raise RuntimeError(f"feature length {len(feats)} != {NUM_FEATS}")

    return np.asarray(feats, dtype=np.float32)



# ---------------------------------------------------------------------------
#  5.  Dataset loader (reads row_plans/*json + query_costs.csv)
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    features: np.ndarray
    label: int            # 1 = column faster
    dir_tag: str
    row_t: float
    col_t: float
    qcost: float = 0.0         # ↑ query_cost from JSON plan
    fann_pred: int = -1   # kernel model label, -1 = not provided
    hybrid_pred: int = -1 # MySQL hybrid-optimizer label

# ---------------------------------------------------------------------------
#  Simple wrapper so the loader can call extractor.extract(plan_json)
# ---------------------------------------------------------------------------
class PlanFeatureExtractor:
    """Thin façade over plan2feat()."""
    def __init__(self):
        self.dim = NUM_FEATS          # handy to know downstream

    def extract(self, plan_json: Dict[str, Any]) -> np.ndarray:
        return plan2feat(plan_json)
    
ROW_PLAN_DIRNAME = "row_plans"
COSTS_CSV_NAME   = "query_costs.csv"

ID_COL    = "query_id"      # <-- matches the 1st column in your CSV
ROW_T_COL = "row_time"      # <-- already OK
COL_T_COL = "column_time"   # <-- change from "col_time" to "column_time"


from pathlib import Path
from typing import List



# ---------------------------------------------------------------------------
#  load_dataset – 现在同时解析 qcost / fann_pred / hybrid_pred
# ---------------------------------------------------------------------------
def load_dataset(dir_path: Path) -> List[Sample]:
    row_plan_dir = dir_path / ROW_PLAN_DIRNAME
    csv_path     = dir_path / COSTS_CSV_NAME

    meta = pd.read_csv(csv_path, dtype={ID_COL: str}).set_index(ID_COL)

    # 可选列：FANN / hybrid，若不存在就充 -1
    has_fann   = "fann_model_label"    in meta.columns
    has_hybrid = "hybrid_optimizer_label" in meta.columns

    samples: List[Sample] = []
    for json_path in row_plan_dir.glob("*.json"):
        qid = json_path.stem
        if qid not in meta.index:
            continue

        stats   = meta.loc[qid]
        row_t = float(stats[ROW_T_COL])
        col_t = float(stats[COL_T_COL])

        # ⇩ 新增：任何一边缺失就跳过
        if math.isnan(row_t) or math.isnan(col_t):
            continue
        label   = int(col_t < row_t)
        fann_lb = int(stats["fann_model_label"])       if has_fann   else -1
        hyb_lb  = int(stats["hybrid_optimizer_label"]) if has_hybrid else -1

        with json_path.open() as f:
            plan = json.load(f)

        try:
            feat = plan2feat(plan)
        except ValueError:
            continue

        qcost = safe_f(
            plan.get("query_block", {}).get("cost_info", {}), "query_cost"
        )

        samples.append(Sample(
            feat, label, dir_path.name,
            row_t, col_t, qcost,
            fann_lb, hyb_lb
        ))

    return samples

# ---------------------------------------------------------------------------
#  train_lightgbm – 训练后在验证集输出全套指标
# ---------------------------------------------------------------------------
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

def _method_stats(pred, gt, row_t, col_t):
    """返回 TP,FP,TN,FN  和平均运行时"""
    cm = confusion_matrix(gt, pred)   # shape = (2,2)
    tn, fp, fn, tp = cm.ravel()
    avg_rt = np.mean(np.where(pred == 1, col_t, row_t))
    return dict(TP=tp, FP=fp, TN=tn, FN=fn,
                Acc=accuracy_score(gt, pred),
                Prec=precision_score(gt, pred, zero_division=0),
                Rec=recall_score(gt, pred, zero_division=0),
                F1=f1_score(gt, pred, zero_division=0),
                AvgRT=avg_rt)

def train_lightgbm(samples: List[Sample],
                   out_model: Path,
                   seed: int = 7,
                   n_rounds: int = 800,
                   valid_frac: float = 0.2):

    # ------------------ 拆分 ------------------
    X   = np.stack([s.features for s in samples])
    y   = np.array([s.label    for s in samples])
    row = np.array([s.row_t    for s in samples])
    col = np.array([s.col_t    for s in samples])
    qcs = np.array([s.qcost    for s in samples])
    fan = np.array([s.fann_pred   for s in samples])
    hyb = np.array([s.hybrid_pred for s in samples])

    X_tr, X_val, y_tr, y_val, row_tr, row_val, col_tr, col_val, \
    qcs_tr, qcs_val, fan_tr, fan_val, hyb_tr, hyb_val = train_test_split(
        X, y, row, col, qcs, fan, hyb,
        test_size=valid_frac, random_state=seed, stratify=y)

    dtr  = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtr)

    params = dict(
        objective="binary",
        learning_rate=0.05,
        num_leaves=256,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        metric=["binary_logloss"],
        seed=seed,
    )
    callbacks = [
        lgb.log_evaluation(period=50),
        lgb.early_stopping(stopping_rounds=30)
    ]
    booster = lgb.train(params, dtr,
                        num_boost_round=n_rounds,
                        valid_sets=[dval],
                        callbacks=callbacks)

    booster.save_model(str(out_model))

    # ------------------ 预测 ------------------
    lgb_pred   = (booster.predict(X_val) > 0.5).astype(int)
    row_pred   = np.zeros_like(y_val)          # 总选行
    col_pred   = np.ones_like(y_val)           # 总选列

    # cost-rule：简单阈值，可按需要改
    cost_thr   = np.median(qcs_tr)             # 用训练集中位数作阈
    cost_pred  = (qcs_val > cost_thr).astype(int)

    # hybrid / fann 若缺失就回落到 cost_pred
    hyb_pred   = np.where(hyb_val >= 0, hyb_val, cost_pred)
    fann_pred  = np.where(fan_val >= 0, fan_val, cost_pred)

    # Oracle：谁快选谁
    oracle_pred = (col_val < row_val).astype(int)

    # ------------------ 统计 ------------------
    names   = ["Row" , "Column", "CostRule",
               "Hybrid", "Kernel", "LightGBM", "Oracle"]
    preds   = [row_pred, col_pred, cost_pred,
               hyb_pred, fann_pred, lgb_pred , oracle_pred]

    print("\n| Method   | TP | FP | TN | FN |  Acc | Prec | Rec  |  F1  | Avg-RT |")
    print(  "|----------|----|----|----|----|------|------|------|------|--------|")
    for n, p in zip(names, preds):
        S = _method_stats(p, y_val, row_val, col_val)
        print(f"| {n:<8} | {S['TP']:>2} | {S['FP']:>2} | {S['TN']:>2} | {S['FN']:>2} |"
              f" {S['Acc']:.3f} | {S['Prec']:.3f} | {S['Rec']:.3f} | {S['F1']:.3f} |"
              f" {S['AvgRT']:.4f} |")

    print(f"\n[train] LightGBM model saved to {out_model}")
    return booster



# ---------------------------------------------------------------------------
#  7.  VIB feature mask (PyTorch implementation)
# ---------------------------------------------------------------------------
def vib_feature_mask(X: np.ndarray,
                     y: np.ndarray,
                     latent_dim: int = NUM_FEATS,
                     epochs: int = 400,
                     seed: int = 7,
                     beta: float = 5e-5):

    # ── 0  Sanitise & standardise ──────────────────────────────────────────
    X = X.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    mu, std = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-9
    X = (X - mu) / std

    import torch, torch.nn as nn, torch.optim as optim
    torch.manual_seed(seed)

    class VIBNet(nn.Module):
        def __init__(self, in_dim, latent):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Linear(in_dim, 128), nn.ReLU(),
                nn.Linear(128, 2 * latent)
            )
            self.cls = nn.Linear(latent, 1)

        def forward(self, x):
            h          = self.enc(x)
            mu, logvar = h.chunk(2, dim=-1)

            # ---- NEW: numerically-safe std --------------------------------
            logvar = torch.clamp(logvar, -10.0, 10.0)   # ↔ σ in [e⁻¹⁰, e¹⁰]
            std    = torch.exp(0.5 * logvar)
            # ----------------------------------------------------------------

            eps = torch.randn_like(std)
            z   = mu + eps * std
            out = self.cls(z).squeeze(1)

            kl  = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
            return out, kl

    model = VIBNet(X.shape[1], latent_dim)
    opt   = optim.Adam(model.parameters(), lr=1e-4)      # a bit smaller LR
    bce   = nn.BCEWithLogitsLoss(reduction="none")

    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y.astype(np.float32))

    for epoch in range(epochs):
        opt.zero_grad()
        logits, kl = model(X_t)
        loss = (bce(logits, y_t) + beta * kl).mean()
        loss.backward()

        # ---- NEW: gradient-clipping ---------------------------------------
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        # -------------------------------------------------------------------

        opt.step()
        if (epoch + 1) % 20 == 0:
            print(f"[VIB] epoch {epoch+1}/{epochs}  loss={loss.item():.5f}")

    # feature importance: same as before
    W_enc = model.enc[0].weight.detach().cpu().numpy()      # 128×F
    W_map = model.enc[2].weight.detach().cpu().numpy()      # 2L×128
    W_cls = model.cls.weight.detach().cpu().numpy()         # 1×L
    W_lat = np.mean(np.abs(W_map), axis=0)                  # 128
    scores = np.abs(W_enc).T @ W_lat

    scores = scores * std.squeeze()     #  std has shape (1, F); squeeze to (F,)
    scores = scores / scores.max()      #  keep 0-1 normalisation
    return scores / scores.max()


# ---------------------------------------------------------------------------
#  8.  CLI glue  -------------------------------------------------------------
# ---------------------------------------------------------------------------
from pathlib import Path, PurePosixPath   # already imported later, but we need Path here
import os

DATA_ROOT = Path(                       # <- **change this to your master dir once**
    os.environ.get("ROWCOL_DATA_ROOT",
                   "/home/wuy/query_costs")   # <-- your base directory
).expanduser().resolve()

# ---------------------------------------------------------------------------
#  main – 现在所有子命令都只接收 dataset                                   |
# ---------------------------------------------------------------------------
def main(argv: List[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Row/Column Router Python Toolkit")
    sub = p.add_subparsers(dest="cmd", required=True)

    # extract
    ps = sub.add_parser("extract")
    ps.add_argument("dataset", type=str)

    # train / vib / select 统统只要 dataset
    for cmd in ("train", "vib", "select"):
        ps = sub.add_parser(cmd)
        ps.add_argument("dataset", type=str)
        ps.add_argument("--use_idx",
                        type=str,
                        default=None,
                        help="npy file with feature indices to keep")
        if cmd == "select":
            ps.add_argument("--method",
                            choices=["hsic", "shap", "cae"],
                            default="hsic")

    args = p.parse_args(argv)

    # ── 连接信息-schema（可按需删掉）──────────────────────────────────────
    schemas = [
        "tpch_sf1", "tpch_sf10", "tpch_sf100",
        "tpcds_sf1", "tpcds_sf10", "tpcds_sf100",
        "hybench_sf1", "hybench_sf10",
        "airline", "credit", "carcinogenesis",
        "employee", "financial", "geneea", "hepatitis",
    ]
    host, port, user, passwd = "127.0.0.1", 44444, "root", ""
    populate_col_stats(host, port, user, passwd, schemas)
    populate_table_stats(host, port, user, passwd, schemas)
    get_index_defs(host, port, user, passwd, schemas)

    # ── 路径推断辅助 ────────────────────────────────────────────────────
    def _paths(ds: str):
        ds_dir = (DATA_ROOT / ds).resolve()
        feat_p = ds_dir / "features.npy"
        lbl_p  = ds_dir / "labels.npy"
        return ds_dir, feat_p, lbl_p

    # ── extract ───────────────────────────────────────────────────────
    if args.cmd == "extract":
        ds_dir, feat_p, lbl_p = _paths(args.dataset)
        samples = load_dataset(ds_dir)
        X = np.stack([s.features for s in samples])
        y = np.array([s.label for s in samples])
        np.save(feat_p, X); np.save(lbl_p, y)
        print(f"[extract] saved {feat_p} {lbl_p}  shape={X.shape}")
        return

    # 其余子命令都共用同一份 features / labels
    ds_dir, feat_p, lbl_p = _paths(args.dataset)
    if not feat_p.exists() or not lbl_p.exists():
        sys.exit(f"[error] {feat_p} 或 {lbl_p} 不存在，请先运行 extract")

    X = np.load(feat_p); y = np.load(lbl_p)

    if args.use_idx is not None:
        idx_path = Path(args.use_idx)
        if not idx_path.is_absolute():          # allow relative to dataset dir
            idx_path = ds_dir / idx_path
        if not idx_path.exists():
            sys.exit(f"[error] cannot find --use_idx file: {idx_path}")
        keep_idx = np.load(idx_path).astype(int)
        X = X[:, keep_idx]                      # keep only those columns

        # also narrow the human-readable names so plots look right
        global FEAT_NAMES                       # we’ll reuse it later
        FEAT_NAMES = [FEAT_NAMES[i] for i in keep_idx]

        print(f"[info] using feature subset from {idx_path}  "
              f"→ X shape now {X.shape}")

    # ── train ─────────────────────────────────────────────────────────
    if args.cmd == "train":
        samples = load_dataset(ds_dir)
        train_lightgbm(samples, ds_dir / "lgb_router.txt")
        return

    # ── vib ───────────────────────────────────────────────────────────
    if args.cmd == "vib":
        scores = vib_feature_mask(X, y)

        # -------- save raw scores --------
        out = ds_dir / "vib_scores.npy"
        np.save(out, scores)
        print(f"[vib] saved {out}")

        # -------- save a bar-plot (head-less) --------
        def save_vib_plot(scores: np.ndarray,
                        out_dir: Path,
                        k: int = 32,
                        feat_names: List[str] = FEAT_NAMES):   # ← default argument
            """
            Save a horizontal bar-chart of the k most important VIB features.
            """
            import matplotlib
            matplotlib.use("Agg")           # head-less backend
            import matplotlib.pyplot as plt
            import numpy as np

            top   = scores.argsort()[::-1][:k]        # indices of top-k
            vals  = scores[top][::-1]                 # smallest at bottom
            labels = [feat_names[i] for i in top][::-1]   # ← human names

            plt.figure(figsize=(6, 0.3 * k + 1.5))
            plt.barh(range(k), vals, align="center")
            plt.yticks(range(k), labels, fontsize=6)
            plt.xlabel("VIB importance (normalised)")
            plt.title(f"Top-{k} VIB features")
            plt.tight_layout()

            fig_path = out_dir / f"vib_top{k}.png"
            plt.savefig(fig_path, dpi=150)
            plt.close()
            print(f"[vib] plot saved {fig_path}")

        # optional: if you have a list with 124 human-readable names
        # FEAT_NAMES = [...]
        save_vib_plot(scores, ds_dir, k=NUM_FEATS)  # or k=20, k=len(scores), …

        return

    # ── select ────────────────────────────────────────────────────────
    if args.cmd == "select":
        if args.method == "hsic":
            from pyhsiclasso import HSICLasso
            hsic = HSICLasso()
            hsic.input(X.astype(np.float64), y.reshape(-1, 1).astype(np.float64))
            hsic.regression(num_feat=32)        # <- your existing call

            idx       = np.array(hsic.get_index())          # top-k indices
            alpha_raw = hsic.get_alpha()                    # (F,) – raw weights
            np.save(ds_dir / "hsic_idx.npy",   idx)
            np.save(ds_dir / "hsic_scores.npy", alpha_raw)
            print(f"[HSIC] top-features saved to {ds_dir/'hsic_idx.npy'}")

            # --------  pretty plot  ----------------------------------------
            import matplotlib
            matplotlib.use("Agg")               # head-less backend
            import matplotlib.pyplot as plt

            k = 32                               # same k as above
            top   = alpha_raw.argsort()[::-1][:k]
            vals  = alpha_raw[top][::-1]          # small→large, bottom-up
            labels = [FEAT_NAMES[i] for i in top][::-1]

            plt.figure(figsize=(6, 0.28*k + 1.5))
            plt.barh(range(k), vals, align="center")
            plt.yticks(range(k), labels, fontsize=6)
            plt.xlabel("HSIC-Lasso α weight")
            plt.title(f"Top-{k} HSIC features")
            plt.tight_layout()

            fig_path = ds_dir / f"hsic_top{k}.png"
            plt.savefig(fig_path, dpi=150)
            plt.close()
            print(f"[HSIC] plot saved → {fig_path}")
        elif args.method == "shap":
            booster = lgb.Booster(model_file=str(ds_dir / "lgb_router.txt"))
            import shap, matplotlib
            explainer = shap.TreeExplainer(booster)

            # ---- 1) full SHAP matrix --------------------------------------------
            shap_values = explainer.shap_values(X)
            # For LightGBM binary models shap_values is a list: [class-0, class-1]
            if isinstance(shap_values, list):
                shap_values = shap_values[1]          # use the positive class

            # ---- 2) aggregate → per-feature score --------------------------------
            scores = np.mean(np.abs(shap_values), axis=0)   # |impact|
            np.save(ds_dir / "shap_scores.npy", scores)

            # top-k indices (k = 32 by default)
            k = NUM_FEATS
            imp_idx = scores.argsort()[::-1][:k]
            np.save(ds_dir / "shap_idx.npy", imp_idx)
            print(f"[SHAP] top-{k} indices saved → {ds_dir/'shap_idx.npy'}")

            # ---- 3) optional pretty plot -----------------------------------------
            def save_shap_plot(scores, out_dir: Path, k: int = 32,
                            feat_names: list[str] | None = None):
                matplotlib.use("Agg")                 # head-less backend
                import matplotlib.pyplot as plt

                top = scores.argsort()[::-1][:k]
                vals   = scores[top][::-1]             # small → large, bottom-up
                labels = ([feat_names[i] for i in top] if feat_names is not None
                        else [f"f{idx}" for idx in top])[::-1]

                plt.figure(figsize=(6, 0.28*k + 1.5))
                plt.barh(range(k), vals, align="center")
                plt.yticks(range(k), labels, fontsize=6)
                plt.xlabel("mean(|SHAP|)")
                plt.title(f"Top-{k} SHAP features")
                plt.tight_layout()

                fig_path = out_dir / f"shap_top{k}.png"
                plt.savefig(fig_path, dpi=150)
                plt.close()
                print(f"[SHAP] plot saved → {fig_path}")

            # —— list of 124 human-readable feature names (same as for VIB plot) ——
            # FEAT_NAMES = np.array([...])   # define once globally
            save_shap_plot(scores, ds_dir, k=k, feat_names=FEAT_NAMES)
        # 若要启用 cae，自行解注释即可
        # elif args.method == "cae":
        #     from concrete_autoencoder import layers as cae_layers
        #     import tensorflow as tf
        #     tf.random.set_seed(7)
        #     from tensorflow import keras
        #     in_dim = X.shape[1]
        #     k = 32
        #     inputs = keras.Input(shape=(in_dim,))
        #     encoded = cae_layers.ConcreteSelect(k)(inputs)
        #     decoded = keras.layers.Dense(in_dim, activation="linear")(encoded)
        #     cae = keras.Model(inputs, decoded)
        #     cae.compile(optimizer="adam", loss="mse")
        #     cae.fit(X, X, epochs=300, batch_size=256, verbose=2)
        #     sel_mask = cae.layers[1].get_support()
        #     idx = np.where(sel_mask)[0]
        #     np.save("cae_idx.npy", idx)
        #     print("CAE selected idx saved cae_idx.npy")



if __name__ == "__main__":
    main()
