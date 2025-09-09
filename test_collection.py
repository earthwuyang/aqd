#!/usr/bin/env python3
"""
Test script to verify data collection is working properly
"""

import json
import sys
from pathlib import Path

def test_unified_json(file_path):
    """Test that the unified JSON file has proper structure"""
    
    print(f"\nTesting file: {file_path}")
    
    if not Path(file_path).exists():
        print(f"ERROR: File does not exist: {file_path}")
        return False
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse JSON: {e}")
        return False
    
    if not isinstance(data, list):
        print(f"ERROR: JSON should be a list, got {type(data)}")
        return False
    
    print(f"Found {len(data)} records")
    
    if len(data) == 0:
        print("WARNING: No records found")
        return True
    
    # Check first record structure
    record = data[0]
    required_fields = [
        'dataset', 'query_type', 'query_text',
        'postgres_time', 'duckdb_time', 'log_time_difference',
        'features', 'postgres_plan_json'
    ]
    
    missing_fields = []
    for field in required_fields:
        if field not in record:
            missing_fields.append(field)
    
    if missing_fields:
        print(f"ERROR: Missing required fields: {missing_fields}")
        return False
    
    # Check data quality
    records_with_plans = 0
    records_with_valid_plans = 0
    sample_plan = None
    
    for i, record in enumerate(data):
        if record.get('postgres_plan_json'):
            records_with_plans += 1
            
            plan = record['postgres_plan_json']
            
            # Check if plan is valid (not just a string fragment)
            if isinstance(plan, (list, dict)):
                # Check for proper plan structure
                if isinstance(plan, list) and len(plan) > 0:
                    if 'Plan' in plan[0]:
                        records_with_valid_plans += 1
                        if not sample_plan and i < 5:  # Get sample from first 5
                            sample_plan = plan
                elif isinstance(plan, dict) and 'Plan' in plan:
                    records_with_valid_plans += 1
                    if not sample_plan and i < 5:
                        sample_plan = plan
            elif isinstance(plan, str) and plan not in [']}', 'null', '']:
                # Might be a stringified plan
                try:
                    parsed = json.loads(plan)
                    if isinstance(parsed, (list, dict)):
                        records_with_valid_plans += 1
                except:
                    pass
    
    print(f"\nData Quality Report:")
    print(f"  Records with plans: {records_with_plans}/{len(data)} ({100*records_with_plans/len(data):.1f}%)")
    print(f"  Valid plans: {records_with_valid_plans}/{len(data)} ({100*records_with_valid_plans/len(data):.1f}%)")
    
    # Check features
    if data[0].get('features'):
        num_features = len(data[0]['features'])
        print(f"  Number of features: {num_features}")
        
        # Show sample features
        sample_features = list(data[0]['features'].keys())[:5]
        print(f"  Sample features: {sample_features}")
    
    # Show sample plan structure
    if sample_plan:
        print(f"\n  Sample plan structure:")
        if isinstance(sample_plan, list) and len(sample_plan) > 0:
            if 'Plan' in sample_plan[0]:
                plan_node = sample_plan[0]['Plan']
                print(f"    Node Type: {plan_node.get('Node Type', 'Unknown')}")
                print(f"    Total Cost: {plan_node.get('Total Cost', 'N/A')}")
                print(f"    Plan Rows: {plan_node.get('Plan Rows', 'N/A')}")
        print("\n  ✓ Plan structure looks good")
    else:
        print("\n  ⚠ No valid plan samples found")
    
    # Check log_time_difference distribution
    time_diffs = [r['log_time_difference'] for r in data if r.get('log_time_difference') is not None]
    if time_diffs:
        import statistics
        print(f"\n  Log time difference stats:")
        print(f"    Count: {len(time_diffs)}")
        print(f"    Mean: {statistics.mean(time_diffs):.3f}")
        print(f"    Median: {statistics.median(time_diffs):.3f}")
        print(f"    Std Dev: {statistics.stdev(time_diffs):.3f}" if len(time_diffs) > 1 else "    Std Dev: N/A")
    
    return records_with_valid_plans > 0

if __name__ == "__main__":
    # Test the unified training data files
    data_dir = Path("data/execution_data")
    
    if len(sys.argv) > 1:
        # Test specific file
        file_path = sys.argv[1]
        success = test_unified_json(file_path)
    else:
        # Test all unified files
        success = True
        for json_file in data_dir.glob("*_unified_training_data.json"):
            if not test_unified_json(json_file):
                success = False
        
        # Also test the global file
        global_file = data_dir / "all_datasets_unified_training_data.json"
        if global_file.exists():
            if not test_unified_json(global_file):
                success = False
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed. Check the data collection process.")
        sys.exit(1)