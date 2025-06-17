#!/usr/bin/env python3
import argparse, json, math, sys
from pathlib import Path

def safe_float(x):
    try: return float(x)
    except: return float("nan")

def safe_log1p(x):
    return math.log1p(max(0.0, x))

def parse_row_plan(path):
    """
    Returns orig_query_cost (float) extracted from
    the JSON at path (row plan).
    """
    try:
        j = json.loads(Path(path).read_text())
        qc = j.get("query_block", {}).get("cost_info", {}).get("query_cost", 0.0)
        return float(qc)
    except:
        return 0.0

def load_samples(base, dirs):
    samples = []
    for d in dirs:
        csvp = Path(base) / d / "query_costs.csv"
        rowdir = Path(base) / d / "row_plans"
        if not csvp.exists():
            print(f"[WARN] missing {csvp}", file=sys.stderr)
            continue
        for ln in csvp.read_text().splitlines()[1:]:
            parts = ln.split(",")
            if len(parts) < 6: 
                continue
            qid, lab, rt, ct, hyb, fann = parts[:6]
            if not lab: 
                continue
            label = int(lab)
            row_t = safe_float(rt)
            col_t = safe_float(ct)
            # skip exact ties or NaNs
            if abs(row_t - col_t) < 1e-12 or math.isnan(row_t) or math.isnan(col_t):
                continue
            hybrid_flag = int(hyb)
            fann_flag   = int(fann)
            # parse orig cost
            plan_path = rowdir / f"{qid}.json"
            orig_cost = parse_row_plan(plan_path) if plan_path.exists() else 0.0
            samples.append({
                "label": label,
                "row_t": row_t,
                "col_t": col_t,
                "hybrid": hybrid_flag,
                "fann": fann_flag,
                "cost": orig_cost
            })
    return samples

def compute_accuracies(subset):
    n = len(subset)
    if n == 0:
        return {"cost":None, "hybrid":None, "fann":None, "n":0}
    # cost rule: choose col if cost>50000
    correct_cost   = sum((s["cost"] > 50000) == bool(s["label"]) for s in subset)
    correct_hybrid = sum((s["hybrid"]    == s["label"]) for s in subset)
    correct_fann   = sum((s["fann"]      == s["label"]) for s in subset)
    return {
        "n": n,
        "cost":   correct_cost   / n,
        "hybrid": correct_hybrid / n,
        "fann":   correct_fann   / n
    }

def main():
    p = argparse.ArgumentParser(
        description="Accuracy of cost-rule, hybrid, and FANN vs. gap thresholds"
    )
    p.add_argument("--base", default='/home/wuy/query_costs',
                   help="base dir containing data_dirs")
    p.add_argument("--data_dirs", required=True, action="append",
                   help="one or more dataset subdirs under base")
    p.add_argument("--thresholds", nargs="+", type=float,
                   default=[1e-3,1e-2,1e-1,1,10,20,30,40,50],
                   help="gap thresholds to evaluate")
    args = p.parse_args()

    samples = load_samples(args.base, args.data_dirs)
    if not samples:
        print("No samples loaded.", file=sys.stderr)
        sys.exit(1)

    print(f"{'gap':>8}  {'count':>6}  {'cost-acc':>8}  {'hyb-acc':>8}  {'fann-acc':>9}")
    print("-"*50)
    for thr in sorted(args.thresholds):
        subset = [s for s in samples
                  if abs(s["row_t"] - s["col_t"]) > thr]
        m = compute_accuracies(subset)
        if m["n"] == 0:
            print(f"{thr:8g}  {0:6d}  {'-':>8}  {'-':>8}  {'-':>9}")
        else:
            print(f"{thr:8g}  {m['n']:6d}  "
                  f"{m['cost']:.4f}  "
                  f"{m['hybrid']:.4f}  "
                  f"{m['fann']:.4f}")

if __name__ == "__main__":
    main()
