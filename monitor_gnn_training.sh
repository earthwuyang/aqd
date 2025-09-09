#!/bin/bash

echo "Monitoring GNN Training Progress..."
echo "=================================="

while true; do
    # Check if the process is still running
    if ! screen -ls | grep -q "gnn_training"; then
        echo "Training completed or stopped!"
        break
    fi
    
    # Get the last epoch info
    LAST_EPOCH=$(grep "^Epoch" logs/gnn_training.log 2>/dev/null | tail -1)
    if [ ! -z "$LAST_EPOCH" ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') - $LAST_EPOCH"
    fi
    
    # Check if training completed
    if grep -q "=== Model Saved ===" logs/gnn_training.log 2>/dev/null; then
        echo "Training completed successfully!"
        grep -A 5 "=== Model Saved ===" logs/gnn_training.log
        break
    fi
    
    # Check for early stopping
    if grep -q "Early stopping triggered" logs/gnn_training.log 2>/dev/null; then
        echo "Early stopping triggered!"
        grep "Early stopping triggered" logs/gnn_training.log | tail -1
    fi
    
    sleep 30
done

echo ""
echo "Final status:"
tail -20 logs/gnn_training.log