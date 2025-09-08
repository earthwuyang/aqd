#!/usr/bin/env python3
"""
AQD Training Data Collection Pipeline
Collects SQL query features and execution times from PostgreSQL and DuckDB
for ML model training to predict query routing decisions.
"""

import psycopg2
import subprocess
import json
import time
import random
import os
from datetime import datetime
from pathlib import Path
import tempfile

# Database configurations  
POSTGRESQL_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'user': 'postgres',
    'database': 'benchmark'
}

DUCKDB_CONFIG = {
    'binary': '/usr/bin/duckdb',
    'database': '/tmp/benchmark_datasets.db'
}

class AQDTrainingDataCollector:
    """
    Collects training data for AQD ML model by running queries on both 
    PostgreSQL and DuckDB and extracting execution features.
    """
    
    def __init__(self):
        self.pg_conn = None
        self.training_data = []
        self.stats = {
            'queries_executed': 0,
            'successful_pairs': 0,
            'failed_queries': 0
        }
    
    def connect_postgresql(self):
        """Connect to PostgreSQL"""
        try:
            self.pg_conn = psycopg2.connect(**POSTGRESQL_CONFIG)
            return True
        except Exception as e:
            print(f"PostgreSQL connection failed: {e}")
            return False
    
    def execute_query_postgresql(self, query):
        """Execute query on PostgreSQL and measure execution time"""
        cursor = self.pg_conn.cursor()
        start_time = time.perf_counter()
        
        try:
            cursor.execute(query)
            result = cursor.fetchall()
            end_time = time.perf_counter()
            
            execution_time = end_time - start_time
            row_count = len(result) if result else 0
            
            return {
                'success': True,
                'execution_time': execution_time,
                'row_count': row_count,
                'error': None
            }
        except Exception as e:
            end_time = time.perf_counter()
            return {
                'success': False,
                'execution_time': end_time - start_time,
                'row_count': 0,
                'error': str(e)
            }
        finally:
            cursor.close()
    
    def execute_query_duckdb(self, query):
        """Execute query on DuckDB and measure execution time"""
        # Write query to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sql', delete=False) as f:
            f.write(query)
            query_file = f.name
        
        try:
            start_time = time.perf_counter()
            
            # Execute query via DuckDB CLI
            result = subprocess.run([
                DUCKDB_CONFIG['binary'], 
                DUCKDB_CONFIG['database'], 
                '-c', query
            ], capture_output=True, text=True, timeout=30)
            
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            
            if result.returncode == 0:
                # Count output lines as rough row estimate
                row_count = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
                return {
                    'success': True,
                    'execution_time': execution_time,
                    'row_count': row_count,
                    'error': None
                }
            else:
                return {
                    'success': False,
                    'execution_time': execution_time,
                    'row_count': 0,
                    'error': result.stderr.strip()
                }
        
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'execution_time': 30.0,  # timeout
                'row_count': 0,
                'error': 'Query timeout'
            }
        except Exception as e:
            return {
                'success': False,
                'execution_time': 0,
                'row_count': 0,
                'error': str(e)
            }
        finally:
            os.unlink(query_file)
    
    def extract_basic_query_features(self, query):
        """Extract basic features from SQL query text"""
        query_lower = query.lower().strip()
        
        features = {
            # Query type classification
            'is_select': 'select' in query_lower,
            'is_aggregate': any(agg in query_lower for agg in ['sum(', 'count(', 'avg(', 'max(', 'min(']),
            'is_join': any(join in query_lower for join in ['join', 'inner join', 'left join', 'right join']),
            'is_group_by': 'group by' in query_lower,
            'is_order_by': 'order by' in query_lower,
            'has_where': 'where' in query_lower,
            'has_having': 'having' in query_lower,
            'has_subquery': '(' in query_lower and 'select' in query_lower.split('(', 1)[1],
            
            # Complexity metrics
            'query_length': len(query),
            'word_count': len(query.split()),
            'table_count': query_lower.count('from ') + query_lower.count('join '),
            'condition_count': query_lower.count('where ') + query_lower.count('and ') + query_lower.count('or '),
            'select_items': query_lower.count(',') + 1 if 'select' in query_lower else 0,
            
            # String-based heuristics
            'likely_oltp': any(op in query_lower for op in ['insert', 'update', 'delete']) or 
                          ('limit' in query_lower and not any(agg in query_lower for agg in ['sum', 'count', 'avg'])),
            'likely_olap': any(agg in query_lower for agg in ['sum(', 'count(*)', 'avg(']) or 'group by' in query_lower,
            
            # Query pattern classification
            'complexity_score': self.calculate_complexity_score(query_lower)
        }
        
        return features
    
    def calculate_complexity_score(self, query_lower):
        """Calculate a simple complexity score based on query patterns"""
        score = 0
        
        # Base complexity
        score += len(query_lower.split()) * 0.1
        
        # Join complexity
        score += query_lower.count('join') * 2
        score += query_lower.count('left join') * 1
        score += query_lower.count('inner join') * 1.5
        
        # Aggregation complexity  
        score += query_lower.count('group by') * 2
        score += query_lower.count('having') * 1.5
        score += query_lower.count('order by') * 1
        
        # Subquery complexity
        score += query_lower.count('select') * 1.5  # Multiple selects = subqueries
        
        # Condition complexity
        score += query_lower.count('where') * 0.5
        score += query_lower.count('and') * 0.3
        score += query_lower.count('or') * 0.5
        
        return score
    
    def collect_query_pair(self, query):
        """Execute query on both systems and collect training data"""
        print(f"Executing query: {query[:100]}...")
        
        # Extract features
        features = self.extract_basic_query_features(query)
        
        # Execute on PostgreSQL
        pg_result = self.execute_query_postgresql(query)
        
        # Execute on DuckDB  
        duck_result = self.execute_query_duckdb(query)
        
        self.stats['queries_executed'] += 1
        
        # Only keep successful pairs for training
        if pg_result['success'] and duck_result['success']:
            # Calculate time difference (log transformed as in AQD paper)
            postgres_time = max(pg_result['execution_time'], 0.001)  # Avoid log(0)
            duckdb_time = max(duck_result['execution_time'], 0.001)
            
            time_diff = postgres_time - duckdb_time
            log_time_diff = time_diff  # We'll apply log transform during ML training
            
            training_sample = {
                'query': query,
                'features': features,
                'postgres_time': postgres_time,
                'duckdb_time': duckdb_time, 
                'time_difference': time_diff,
                'log_time_difference': log_time_diff,
                'postgres_rows': pg_result['row_count'],
                'duckdb_rows': duck_result['row_count'],
                'timestamp': datetime.now().isoformat()
            }
            
            self.training_data.append(training_sample)
            self.stats['successful_pairs'] += 1
            print(f"  ✓ Success - PG: {postgres_time:.3f}s, Duck: {duckdb_time:.3f}s, Diff: {time_diff:.3f}s")
            return True
        else:
            self.stats['failed_queries'] += 1
            print(f"  ✗ Failed - PG: {pg_result.get('error', 'OK')}, Duck: {duck_result.get('error', 'OK')}")
            return False
    
    def generate_sample_queries(self):
        """Generate sample queries for testing (replace with real benchmark queries)"""
        sample_queries = [
            "SELECT COUNT(*) FROM users",
            "SELECT * FROM orders WHERE order_date > '2023-01-01' LIMIT 100",
            "SELECT customer_id, SUM(amount) FROM orders GROUP BY customer_id",
            "SELECT u.name, COUNT(o.id) FROM users u LEFT JOIN orders o ON u.id = o.customer_id GROUP BY u.id, u.name",
            "SELECT * FROM products WHERE price > 100 ORDER BY price DESC LIMIT 50",
            "SELECT AVG(price) FROM products WHERE category = 'Electronics'",
            "SELECT EXTRACT(month FROM order_date), SUM(amount) FROM orders GROUP BY 1 ORDER BY 1",
            "SELECT * FROM users WHERE created_at BETWEEN '2023-01-01' AND '2023-12-31'",
            "SELECT category, AVG(price), COUNT(*) FROM products GROUP BY category HAVING COUNT(*) > 10",
            "SELECT o.id, u.name, o.amount FROM orders o JOIN users u ON o.customer_id = u.id WHERE o.amount > 500"
        ]
        return sample_queries
    
    def collect_training_data(self, num_samples=1000):
        """Main training data collection process"""
        print("🚀 Starting AQD Training Data Collection")
        print("="*60)
        
        if not self.connect_postgresql():
            print("Failed to connect to PostgreSQL")
            return False
        
        # Check DuckDB
        if not os.path.exists(DUCKDB_CONFIG['binary']):
            print(f"DuckDB binary not found at {DUCKDB_CONFIG['binary']}")
            return False
        
        print(f"Target samples: {num_samples}")
        print(f"PostgreSQL: {POSTGRESQL_CONFIG['host']}:{POSTGRESQL_CONFIG['port']}")
        print(f"DuckDB: {DUCKDB_CONFIG['database']}")
        print("-"*60)
        
        # Generate or load queries
        queries = self.generate_sample_queries()
        
        # Collect training data
        collected = 0
        while collected < num_samples and collected < len(queries) * 10:  # Safety limit
            # Pick a random query (with variations)
            base_query = random.choice(queries)
            
            # Add some variations
            query = self.add_query_variations(base_query)
            
            if self.collect_query_pair(query):
                collected += 1
                
                # Periodic save
                if collected % 50 == 0:
                    self.save_training_data()
                    print(f"Saved progress: {collected}/{num_samples} samples collected")
        
        # Final save
        self.save_training_data()
        
        # Print summary
        print("\n" + "="*60)
        print("📊 Training Data Collection Summary")
        print("="*60)
        print(f"Queries executed: {self.stats['queries_executed']}")
        print(f"Successful pairs: {self.stats['successful_pairs']}")
        print(f"Failed queries: {self.stats['failed_queries']}")
        print(f"Success rate: {self.stats['successful_pairs']/self.stats['queries_executed']*100:.1f}%")
        print(f"Training samples: {len(self.training_data)}")
        
        return len(self.training_data) >= min(num_samples * 0.8, 100)  # 80% success target
    
    def add_query_variations(self, query):
        """Add random variations to queries to increase diversity"""
        variations = [query]  # Start with original
        
        # Add LIMIT variations
        if 'limit' not in query.lower():
            variations.append(f"{query} LIMIT {random.choice([10, 50, 100, 500])}")
        
        # Add ORDER BY variations for non-aggregate queries
        if 'order by' not in query.lower() and 'group by' not in query.lower():
            if 'users' in query.lower():
                variations.append(f"{query} ORDER BY id")
            elif 'products' in query.lower():
                variations.append(f"{query} ORDER BY price DESC")
        
        return random.choice(variations)
    
    def save_training_data(self):
        """Save collected training data to files"""
        output_dir = Path('/tmp/aqd_training_data')
        output_dir.mkdir(exist_ok=True)
        
        # Save full dataset
        with open(output_dir / 'training_data.json', 'w') as f:
            json.dump({
                'training_samples': self.training_data,
                'stats': self.stats,
                'timestamp': datetime.now().isoformat(),
                'sample_count': len(self.training_data)
            }, f, indent=2)
        
        # Save features-only CSV for ML training
        if self.training_data:
            import csv
            with open(output_dir / 'features.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Header
                sample_features = list(self.training_data[0]['features'].keys())
                header = sample_features + ['postgres_time', 'duckdb_time', 'time_difference', 'log_time_difference']
                writer.writerow(header)
                
                # Data rows
                for sample in self.training_data:
                    row = []
                    for feature in sample_features:
                        row.append(sample['features'][feature])
                    row.extend([
                        sample['postgres_time'],
                        sample['duckdb_time'], 
                        sample['time_difference'],
                        sample['log_time_difference']
                    ])
                    writer.writerow(row)
        
        print(f"💾 Training data saved to {output_dir}")

def main():
    """Main function"""
    collector = AQDTrainingDataCollector()
    success = collector.collect_training_data(num_samples=500)  # Start with 500 samples
    
    if success:
        print("\n🎉 Training data collection completed successfully!")
        print("Ready for LightGBM model training.")
    else:
        print("\n❌ Training data collection failed or incomplete.")
    
    return success

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)