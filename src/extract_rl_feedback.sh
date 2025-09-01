#!/bin/bash
# diagnose_rl_logs.sh - Diagnose RL logging issues

echo "=== RL Logging Diagnostic ==="
echo ""

# Check log file
LOG_PATH="/home/wuy/mypolardb/db/log/master-error.log"

echo "1. Checking log file..."
if [ -f "$LOG_PATH" ]; then
    echo "   ✓ Log file exists at $LOG_PATH"
    LOG_SIZE=$(sudo ls -lh "$LOG_PATH" 2>/dev/null | awk '{print $5}')
    echo "   Size: $LOG_SIZE"
else
    echo "   ✗ Log file NOT found at $LOG_PATH"
    echo "   Checking other possible locations..."
    
    # Try to find mysql error logs
    POSSIBLE_LOGS=$(sudo find /home/wuy/mypolardb -name "*error*.log" -type f 2>/dev/null | head -10)
    if [ -n "$POSSIBLE_LOGS" ]; then
        echo "   Found these error logs:"
        echo "$POSSIBLE_LOGS" | sed 's/^/     /'
    fi
fi

echo ""
echo "2. Checking for RL-related log entries..."

# Try different patterns
PATTERNS=(
    "\[RL\]"
    "RL feedback"
    "bandit_feedback"
    "LinUCB"
    "choose_column"
    "route=COL"
    "route=ROW"
    "[Hybrid]"
    "lightgbm_dynamic"
    "use_mm1_time"
)

for pattern in "${PATTERNS[@]}"; do
    COUNT=$(sudo grep -c "$pattern" "$LOG_PATH" 2>/dev/null || echo 0)
    if [ "$COUNT" -gt 0 ]; then
        echo "   ✓ Found $COUNT entries matching '$pattern'"
        echo "     Sample:"
        sudo grep "$pattern" "$LOG_PATH" 2>/dev/null | tail -2 | sed 's/^/       /'
    fi
done

echo ""
echo "3. Checking recent log entries..."
echo "   Last 10 lines of error log:"
sudo tail -10 "$LOG_PATH" 2>/dev/null | sed 's/^/     /'

echo ""
echo "4. Checking MySQL process..."
MYSQL_PID=$(pgrep -f "mysqld.*polardb" | head -1)
if [ -n "$MYSQL_PID" ]; then
    echo "   ✓ MySQL process found (PID: $MYSQL_PID)"
    
    # Check if it's the expected PID
    if [ "$MYSQL_PID" = "17748" ]; then
        echo "   ✓ PID matches expected value (17748)"
    else
        echo "   ⚠ PID ($MYSQL_PID) doesn't match expected (17748)"
    fi
else
    echo "   ✗ MySQL process not found"
fi

echo ""
echo "5. Checking MySQL variables..."
# Try to connect and check relevant variables
mysql -h127.0.0.1 -P44444 -uroot -e "
SELECT 
    VARIABLE_NAME, 
    VARIABLE_VALUE 
FROM 
    performance_schema.global_variables 
WHERE 
    VARIABLE_NAME IN (
        'use_mm1_time',
        'fann_model_routing_enabled',
        'hybrid_opt_dispatch_enabled',
        'cost_threshold_for_imci'
    )
ORDER BY 
    VARIABLE_NAME;
" 2>/dev/null | sed 's/^/     /'

echo ""
echo "6. Alternative log locations..."

# Check for logs in /tmp
TMP_LOGS=$(ls -la /tmp/*polardb*.log /tmp/*rl*.log 2>/dev/null)
if [ -n "$TMP_LOGS" ]; then
    echo "   Found logs in /tmp:"
    echo "$TMP_LOGS" | sed 's/^/     /'
fi

echo ""
echo "=== Diagnostic Summary ==="
echo ""
echo "If you don't see any RL feedback entries, possible causes:"
echo "1. The new C++ code with improved bandit_feedback() hasn't been compiled/deployed"
echo "2. The lightgbm_dynamic mode isn't being tested (check your benchmark script)"
echo "3. The logs are being written elsewhere"
echo "4. The MySQL server needs to be restarted after code changes"
echo ""
echo "Next steps:"
echo "1. Verify the C++ changes are compiled into your MySQL binary"
echo "2. Restart MySQL: sudo systemctl restart mysql (or your restart command)"
echo "3. Run a simple test query with lightgbm_dynamic mode enabled"
echo "4. Check the log again"