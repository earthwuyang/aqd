#!/usr/bin/env python3
# compare_modes_auto.py  (latency & makespan speed-up)

import argparse, subprocess, re, sys, shlex, statistics, textwrap, time

PAT = re.compile(
    r"===\s+(lgbm_kernel(?:_mm1)?)\s+===.*?makespan\s+([0-9.]+).*?avg\s+([0-9.]+)",
    re.S)

def run_once(bench, args):
    """调用 benchmark_dispatch_modes.py 并返回 {'static':(mk,avg), 'mm1':(mk,avg)}"""
    print("➜", " ".join(shlex.quote(c) for c in [sys.executable, bench, *args]))
    out = subprocess.check_output([sys.executable, bench, *args], text=True)
    m = PAT.findall(out)
    if len(m) < 2:
        raise RuntimeError("没有匹配到两种模式的结果，请检查输出/正则")
    return {name: (float(mk), float(avg)) for name, mk, avg in m}

def mean_std(vals):
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0)

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""
        不向 benchmark_dispatch_modes.py 注入任何它不认识的参数。
        只做：
          • 预热  (--warm_n)            默认 50
          • 正式 (--rounds) 轮对比      默认 3
        其余所有参数原封不动透传。
        """))
    ap.add_argument("--bench", default='/home/wuy/simple_row_column_routing/benchmark_dispatch_modes.py')
    ap.add_argument("--warm_n", type=int, default=50)
    ap.add_argument("--rounds", type=int, default=3)
    args, passthru = ap.parse_known_intermixed_args()

    # ---------- 0. 预热：仅缩小 -n ----------
    warm = passthru.copy()
    if "-n" in warm:
        idx = warm.index("-n") + 1
        warm[idx] = str(args.warm_n)
    else:
        warm += ["-n", str(args.warm_n)]
    run_once(args.bench, warm)
    time.sleep(1)

    # ---------- 1. 多轮正式 ----------
    sta_lat, mm1_lat = [], []
    sta_mk,  mm1_mk  = [], []
    for r in range(1, args.rounds + 1):
        print(f"\n=== Round {r} ===")
        res = run_once(args.bench, passthru)
        mk_sta, lat_sta = res["lgbm_kernel"]
        mk_mm1, lat_mm1 = res["lgbm_kernel_mm1"]

        sta_lat.append(lat_sta);   mm1_lat.append(lat_mm1)
        sta_mk.append(mk_sta);     mm1_mk.append(mk_mm1)

    # ---------- 2. 统计 ----------
    m_sta_lat, s_sta_lat = mean_std(sta_lat)
    m_mm1_lat, s_mm1_lat = mean_std(mm1_lat)
    m_sta_mk,  s_sta_mk  = mean_std(sta_mk)
    m_mm1_mk,  s_mm1_mk  = mean_std(mm1_mk)

    speed_lat = m_sta_lat / m_mm1_lat if m_mm1_lat else float('inf')
    speed_mk  = m_sta_mk  / m_mm1_mk  if m_mm1_mk  else float('inf')

    # ---------- 3. 输出 ----------
    print("\n----- 汇总 -----")
    print(f"STATIC   avg-lat {m_sta_lat:.4f} ±{s_sta_lat:.4f}   makespan {m_sta_mk:.2f}s ±{s_sta_mk:.2f}")
    print(f"ADAPTIVE avg-lat {m_mm1_lat:.4f} ±{s_mm1_lat:.4f}   makespan {m_mm1_mk:.2f}s ±{s_mm1_mk:.2f}")
    print(f"⚡ Latency Speed-up  ≈ {speed_lat:.2f}×")
    print(f"⚡ Makespan Speed-up ≈ {speed_mk:.2f}×")

if __name__ == "__main__":
    main()
