#!/usr/bin/env python3
"""
Build GNN labels (query_hash,target) from collected execution_data JSONs.
- Reads data/execution_data/*_execution_data.json
- Extracts query_hash from postgres_plan JSON (requires plan logging enabled)
- Uses log_time_difference if present; else computes from postgres_time/duckdb_time
- Writes data/execution_data/gnn_labels.csv
"""

import json
import csv
import glob
from pathlib import Path

BASE = Path(__file__).resolve().parent
EXEC_DIR = BASE / 'data' / 'execution_data'
OUT_CSV = EXEC_DIR / 'gnn_labels.csv'

def normalize_sql(s: str) -> str:
    # Collapse whitespace to single spaces and lowercase all characters
    out = []
    in_space = False
    for ch in s:
        if ch.isspace():
            if not in_space:
                out.append(' ')
                in_space = True
        else:
            out.append(ch.lower())
            in_space = False
    return ''.join(out)

def aqd_query_hash(sql: str) -> str:
    s = normalize_sql(sql)
    h = 0
    for ch in s:
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
    tail = (len(s) ^ 0xDEADBEEF) & 0xFFFFFFFF
    return f"{h:08x}{tail:08x}"

def main():
    EXEC_DIR.mkdir(parents=True, exist_ok=True)
    label_map = {}
    total = 0
    with_plan = 0
    for fp in sorted(EXEC_DIR.glob('*_execution_data.json')):
        try:
            arr = json.loads(fp.read_text())
        except Exception:
            continue
        for rec in arr:
            total += 1
            qh = None
            # Prefer query_hash from logged plan, if present
            plan_str = rec.get('postgres_plan')
            if plan_str:
                try:
                    plan_obj = json.loads(plan_str)
                    qh = plan_obj.get('query_hash')
                    if qh:
                        with_plan += 1
                except Exception:
                    qh = None
            # Fallback: compute hash from query text
            if not qh:
                sql = rec.get('query_text') or ''
                if not sql:
                    continue
                qh = aqd_query_hash(sql)
            # derive target
            tgt = rec.get('log_time_difference')
            if tgt is None:
                pt = rec.get('postgres_time') or 0.0
                dt = rec.get('duckdb_time') or 0.0
                if pt > 0 and dt > 0:
                    try:
                        import math
                        tgt = math.log(pt / dt)
                    except Exception:
                        tgt = None
            if tgt is None:
                continue
            label_map[qh] = tgt

    # write csv
    with OUT_CSV.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['query_hash','target'])
        for k, v in label_map.items():
            w.writerow([k, v])

    print(f"Wrote {len(label_map)} labels to {OUT_CSV} (from {with_plan}/{total} records with plans)")

if __name__ == '__main__':
    main()
