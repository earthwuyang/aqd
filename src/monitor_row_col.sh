#!/usr/bin/env bash
# monitor_row_col_0.5s.sh —— 每 0.5 s 打印 Row / Col CPU%

PID=107910
INTERVAL=0.5
HZ=$(getconf CLK_TCK)
CPUS=$(nproc)

get_tids() {          # $1 = "imci" or "!imci"
  local pattern=$1
  if [[ $pattern == "!imci" ]]; then
    grep -L "imci" /proc/$PID/task/*/comm
  else
    grep -l "imci" /proc/$PID/task/*/comm
  fi | awk -F'/' '{print $(NF-1)}'
}

sample_group() {      # 参数 = 一串 TID
  local total=0
  for tid in "$@"; do
    j=$(awk '{print $14+$15}' /proc/$PID/task/$tid/stat 2>/dev/null)
    (( total += j ))
  done
  echo "$total"
}

while true; do
  IMCI_TIDS=( $(get_tids "imci") )
  ROW_TIDS=(  $(get_tids "!imci") )

  r0=$(sample_group "${ROW_TIDS[@]}")
  c0=$(sample_group "${IMCI_TIDS[@]}")
  sleep "$INTERVAL"
  r1=$(sample_group "${ROW_TIDS[@]}")
  c1=$(sample_group "${IMCI_TIDS[@]}")

  row_pct=$(awk -v d=$((r1-r0)) -v hz=$HZ -v iv=$INTERVAL -v n=$CPUS \
            'BEGIN{printf "%.2f", d*100/(hz*iv*n)}')
  col_pct=$(awk -v d=$((c1-c0)) -v hz=$HZ -v iv=$INTERVAL -v n=$CPUS \
            'BEGIN{printf "%.2f", d*100/(hz*iv*n)}')

  printf "%s  Row:%s%%(%dT)  Col:%s%%(%dT)\n" \
         "$(date '+%H:%M:%S.%3N')" "$row_pct" "${#ROW_TIDS[@]}" \
         "$col_pct" "${#IMCI_TIDS[@]}"
done
