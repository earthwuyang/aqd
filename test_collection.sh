#!/bin/bash

# Test script for dual execution data collection

echo "Testing dual execution data collection..."
echo "========================================="

# Test with small sample from available datasets
echo ""
echo "Testing with financial dataset (10 queries)..."
python3 collect_dual_execution_data.py \
    --datasets financial \
    --sample-size 10 \
    --timeout 10

echo ""
echo "Testing with employee dataset (10 queries)..."
python3 collect_dual_execution_data.py \
    --datasets employee \
    --sample-size 10 \
    --timeout 10

echo ""
echo "Checking output files..."
ls -la dual_execution_data/

echo ""
echo "Sample output from financial.json:"
if [ -f dual_execution_data/financial.json ]; then
    python3 -c "import json; data = json.load(open('dual_execution_data/financial.json')); print(f'Total queries: {len(data)}'); print(f'First query type: {data[0][\"query_type\"] if data else \"N/A\"}')"
fi

echo ""
echo "Test complete!"