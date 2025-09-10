#!/usr/bin/env python3
"""
Verify that all scripts use consistent routing method numbers
According to aqd_query_router.c:
  0 = AQD_ROUTE_DEFAULT
  1 = AQD_ROUTE_COST_THRESHOLD  
  2 = AQD_ROUTE_LIGHTGBM
  3 = AQD_ROUTE_GNN
"""

import glob
import re
from pathlib import Path

def check_file(filepath):
    """Check a file for routing method settings"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Find all SET aqd.routing_method statements
    pattern = r'SET aqd\.routing_method\s*=\s*(\d+)'
    matches = re.findall(pattern, content)
    
    # Also find method mappings in dictionaries
    dict_pattern = r"'(default|cost_threshold|lightgbm|gnn)':\s*(\d+)"
    dict_matches = re.findall(dict_pattern, content)
    
    return matches, dict_matches

def main():
    print("Verifying routing method consistency across all Python scripts...")
    print("=" * 60)
    print("Expected values (from aqd_query_router.c):")
    print("  0 = AQD_ROUTE_DEFAULT")
    print("  1 = AQD_ROUTE_COST_THRESHOLD")
    print("  2 = AQD_ROUTE_LIGHTGBM")
    print("  3 = AQD_ROUTE_GNN")
    print("=" * 60)
    
    # Expected mappings
    expected = {
        'default': '0',
        'cost_threshold': '1',
        'lightgbm': '2',
        'gnn': '3'
    }
    
    # Check all Python files
    python_files = glob.glob('/home/wuy/DB/pg_duckdb_postgres_2/*.py')
    
    issues = []
    
    for filepath in sorted(python_files):
        filename = Path(filepath).name
        
        # Skip this verification script
        if filename == 'verify_routing_consistency.py':
            continue
            
        set_matches, dict_matches = check_file(filepath)
        
        if set_matches or dict_matches:
            print(f"\n{filename}:")
            
            # Check SET statements
            if set_matches:
                print(f"  SET statements found: {set_matches}")
            
            # Check dictionary mappings
            if dict_matches:
                for method, value in dict_matches:
                    if value != expected[method]:
                        issues.append(f"{filename}: {method} = {value} (should be {expected[method]})")
                        print(f"  ❌ {method}: {value} (should be {expected[method]})")
                    else:
                        print(f"  ✓ {method}: {value}")
    
    print("\n" + "=" * 60)
    if issues:
        print("ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ All scripts use consistent routing method numbers!")
    
    # Also check the C header
    print("\n" + "=" * 60)
    print("Checking C header definition...")
    header_path = '/home/wuy/DB/pg_duckdb_postgres_2/postgres_src/src/include/aqd_query_router.h'
    if Path(header_path).exists():
        with open(header_path, 'r') as f:
            content = f.read()
        
        # Find enum definition
        enum_pattern = r'AQD_ROUTE_(\w+)\s*=\s*(\d+)'
        enum_matches = re.findall(enum_pattern, content)
        
        if enum_matches:
            print("C header enum values:")
            for name, value in enum_matches:
                print(f"  AQD_ROUTE_{name} = {value}")
    
    return len(issues) == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)