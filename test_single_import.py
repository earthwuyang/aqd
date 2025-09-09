#!/usr/bin/env python3
"""
Test the modified import_benchmark_datasets.py with a single dataset
"""

from import_benchmark_datasets import BenchmarkDatasetImporter

def test_single_dataset():
    """Test importing a single dataset as a database"""
    importer = BenchmarkDatasetImporter()
    
    try:
        # Connect to databases
        importer.connect()
        
        # Test with a small dataset
        test_database = 'world'  # Usually has good, small data
        
        print(f"🧪 Testing database-per-dataset import with: {test_database}")
        
        success = importer.import_database(test_database)
        
        if success:
            print(f"✅ Successfully imported {test_database} as a database!")
            return True
        else:
            print(f"✗ Failed to import {test_database}")
            return False
            
    except Exception as e:
        print(f"✗ Error during test: {e}")
        return False
    
    finally:
        # Clean up connections
        if importer.mysql_conn:
            importer.mysql_conn.close()
        if importer.pg_conn:
            importer.pg_conn.close()

if __name__ == "__main__":
    success = test_single_dataset()
    exit(0 if success else 1)