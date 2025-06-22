#!/usr/bin/env python3
import argparse, csv, json, math, sys
from pathlib import Path

def safe_float(x):
    try: return float(x)
    except: return float("nan")

def parse_row_plan(path: Path) -> float:
    """
    Extracts the original 'query_cost' from the JSON file at path.
    """
    try:
        j = json.loads(path.read_text())
        return float(j.get("query_block", {})
                       .get("cost_info", {})
                       .get("query_cost", 0.0))
    except:
        return 0.0

def process_dir(base: Path, d: str, cost_threshold: float, gap_threshold: float):
    csvp   = base / d / "query_costs.csv"
    rowdir = base / d / "row_plans"
    if not csvp.exists():
        print(f"[WARN] missing {csvp}", file=sys.stderr)
        return

    outp = Path(f"{d}.csv")
    with open(csvp, newline='') as inf, open(outp, 'w', newline='') as outf:
        reader = csv.reader(inf)
        writer = csv.writer(outf)
        writer.writerow([
            "query_id",
            "row_time",
            "col_time",
            "gap",
            "use_imci",
            "cost_threshold_label",
            "hybrid_optimizer_label",
            "fann_model_label",
            "sqlQuery"
        ])
        next(reader, None)
        for parts in reader:
            if len(parts) < 7:
                continue
            qid         = parts[0]
            use_imci    = int(parts[1])
            row_t       = safe_float(parts[2])
            col_t       = safe_float(parts[3])
            hybrid_flag = int(parts[4])
            fann_flag   = int(parts[5])
            sql         = parts[6]

            if math.isnan(row_t) or math.isnan(col_t):
                continue
            gap = abs(row_t - col_t)
            # *** only keep gaps > gap_threshold ***
            if gap <= gap_threshold:
                continue

            plan_path = rowdir / f"{qid}.json"
            orig_cost = parse_row_plan(plan_path) if plan_path.exists() else 0.0
            cost_label = 1 if orig_cost > cost_threshold else 0

            writer.writerow([
                qid,
                f"{row_t:.6f}",
                f"{col_t:.6f}",
                f"{gap:.6f}",
                use_imci,
                cost_label,
                hybrid_flag,
                fann_flag,
                sql
            ])
    print(f"[INFO] wrote {outp} (gaps > {gap_threshold}s) for dataset '{d}'")

def main():
    p = argparse.ArgumentParser(
        description="Export only queries with gap>threshold"
    )
    p.add_argument("--base", default="/home/wuy/query_costs",
                   help="base dir containing data_dirs")
    p.add_argument("--data_dirs", required=True, nargs="+",
                   help="one or more dataset subdirs under base")
    p.add_argument("--cost_threshold", type=float, default=50000,
                   help="threshold on original query_cost to predict columnar")
    p.add_argument("--gap_threshold", type=float, default=10.0,
                   help="only report rows where |row_time−col_time| > this")
    args = p.parse_args()

    base = Path(args.base)
    for d in args.data_dirs:
        process_dir(base, d, args.cost_threshold, args.gap_threshold)

if __name__ == "__main__":
    main()
