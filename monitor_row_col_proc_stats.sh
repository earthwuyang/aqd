#!/usr/bin/env bash
# monitor_row_col_proc_stats.sh  —— 0.5 s 刷新行/列 CPU%

PID=107910            # ← 改成你的 mysqld PID
INT=0.5               # 刷新周期秒
HZ=$(getconf CLK_TCK) # 每秒 jiffies
CPUS=$(nproc)         # 逻辑 CPU 数

# 读取某线程 utime+stime
jiffies_of() { awk '{print $14+$15}' "/proc/$PID/task/$1/stat" 2>/dev/null; }

while :; do
  # 用 -F 匹配固定字符串，避免 [ 触发正则
  COL_TIDS=( $(grep -Fl "imci[" /proc/$PID/task/*/comm | awk -F/ '{print $(NF-1)}') )
  ROW_TIDS=( $(grep  -FL "imci[" /proc/$PID/task/*/comm | awk -F/ '{print $(NF-1)}') )

  sum() { local t=0; for id in "$@"; do (( t += $(jiffies_of "$id") )); done; echo $t; }

  c0=$(sum "${COL_TIDS[@]}"); r0=$(sum "${ROW_TIDS[@]}")
  sleep "$INT"
  c1=$(sum "${COL_TIDS[@]}"); r1=$(sum "${ROW_TIDS[@]}")

  col_pct=$(awk -v d=$((c1-c0)) -v hz=$HZ -v iv=$INT -v n=$CPUS \
              'BEGIN{printf "%.2f", d*100/(hz*iv*n)}')
  row_pct=$(awk -v d=$((r1-r0)) -v hz=$HZ -v iv=$INT -v n=$CPUS \
              'BEGIN{printf "%.2f", d*100/(hz*iv*n)}')

  printf "%s  Row:%s%%  Col:%s%%  (T:%d|%d)\n" \
         "$(date '+%H:%M:%S.%3N')" "$row_pct" "$col_pct" \
         "${#ROW_TIDS[@]}" "${#COL_TIDS[@]}"
done
