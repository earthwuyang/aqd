#!/usr/bin/env python3
"""
Comprehensive ZSCE Training Data Collection for PostgreSQL + DuckDB AQD
Executes ZSCE-generated queries on both engines and collects real performance data
"""

import os
import json
import time
import psycopg2
import duckdb
import logging
import glob
from typing import Dict, List, Any, Tuple
from datetime import datetime
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import subprocess

# Database connection configuration
POSTGRESQL_CONFIG = {
    'host': 'localhost', 
    'port': 5433,
    'database': 'postgres',
    'user': 'wuy',
    'password': ''
}

DUCKDB_PATH = '/home/wuy/DB/pg_duckdb_postgres/benchmark_datasets.db'

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/zsce_training_collection.log'),
        logging.StreamHandler()
    ]
)

class ZSCETrainingDataCollector:
    """Collects training data by executing ZSCE queries on both PostgreSQL and DuckDB"""
    
    def __init__(self):
        self.pg_conn = None
        self.duck_conn = None
        self.collected_data = []
        self.lock = threading.Lock()
        self.stats = {
            'total_queries': 0,
            'successful_dual_executions': 0,
            'postgresql_only': 0,
            'duckdb_only': 0,
            'failed_executions': 0,
            'start_time': datetime.now()
        }
        
    def connect_engines(self):
        """Connect to both engines"""
        try:
            self.pg_conn = psycopg2.connect(**POSTGRESQL_CONFIG)
            self.pg_conn.autocommit = True
            logging.info("✓ Connected to PostgreSQL")
        except Exception as e:
            logging.error(f"PostgreSQL connection failed: {e}")
            return False
        
        try:
            self.duck_conn = duckdb.connect(DUCKDB_PATH)
            logging.info("✓ Connected to DuckDB")
        except Exception as e:
            logging.error(f"DuckDB connection failed: {e}")
            return False
            
        return True
    
    def adapt_query_for_schema(self, query: str, dataset: str) -> Tuple[str, str]:
        """Adapt synthetic query to work with real dataset schemas"""
        # For now, we'll work with synthetic queries but target the right schema
        # This is a limitation we'll note but proceed with to collect real performance data
        pg_query = query.replace('FROM users', f'FROM "{dataset}".users') \
                        .replace('FROM orders', f'FROM "{dataset}".orders') \
                        .replace('FROM products', f'FROM "{dataset}".products') \
                        .replace('JOIN users', f'JOIN "{dataset}".users') \
                        .replace('JOIN orders', f'JOIN "{dataset}".orders') \
                        .replace('JOIN products', f'FROM "{dataset}".products')
        
        duck_query = query.replace('FROM users', f'FROM "{dataset}".users') \
                          .replace('FROM orders', f'FROM "{dataset}".orders') \
                          .replace('FROM products', f'FROM "{dataset}".products') \
                          .replace('JOIN users', f'JOIN "{dataset}".users') \
                          .replace('JOIN orders', f'JOIN "{dataset}".orders') \
                          .replace('JOIN products', f'FROM "{dataset}".products')
        
        return pg_query, duck_query
    
    def execute_on_postgresql(self, query: str, timeout: float = 30.0) -> Tuple[bool, float, str]:
        """Execute query on PostgreSQL and measure time"""
        try:
            cursor = self.pg_conn.cursor()
            start_time = time.time()
            cursor.execute(query)
            results = cursor.fetchall()
            end_time = time.time()
            cursor.close()
            
            execution_time = end_time - start_time
            return True, execution_time, None
        except Exception as e:
            return False, 0.0, str(e)
    
    def execute_on_duckdb(self, query: str, timeout: float = 30.0) -> Tuple[bool, float, str]:
        """Execute query on DuckDB and measure time"""
        try:
            start_time = time.time()
            results = self.duck_conn.execute(query).fetchall()
            end_time = time.time()
            
            execution_time = end_time - start_time
            return True, execution_time, None
        except Exception as e:
            return False, 0.0, str(e)
    
    def extract_query_features(self, query: str) -> Dict[str, Any]:
        """Extract features from query for ML model"""
        query_lower = query.lower().strip()
        
        features = {
            # Basic query characteristics
            'query_length': len(query),
            'word_count': len(query.split()),
            
            # SQL keywords
            'has_select': 1 if 'select' in query_lower else 0,
            'has_from': 1 if 'from' in query_lower else 0,
            'has_where': 1 if 'where' in query_lower else 0,
            'has_join': 1 if 'join' in query_lower else 0,
            'has_group_by': 1 if 'group by' in query_lower else 0,
            'has_order_by': 1 if 'order by' in query_lower else 0,
            'has_having': 1 if 'having' in query_lower else 0,
            'has_limit': 1 if 'limit' in query_lower else 0,
            
            # Aggregation functions
            'has_count': 1 if 'count(' in query_lower else 0,
            'has_sum': 1 if 'sum(' in query_lower else 0,
            'has_avg': 1 if 'avg(' in query_lower else 0,
            'has_min': 1 if 'min(' in query_lower else 0,
            'has_max': 1 if 'max(' in query_lower else 0,
            
            # Query complexity
            'num_joins': query_lower.count('join'),
            'num_tables': len([word for word in query.split() if word.lower() in ['from', 'join']]),
            'num_conditions': query_lower.count('and') + query_lower.count('or'),
            'num_aggregates': (query_lower.count('count(') + query_lower.count('sum(') + 
                             query_lower.count('avg(') + query_lower.count('min(') + 
                             query_lower.count('max(')),
            
            # Operators
            'num_equals': query.count('='),
            'num_comparisons': (query.count('>') + query.count('<') + 
                              query.count('>=') + query.count('<=')),
            'num_likes': query_lower.count('like'),
            'num_ins': query_lower.count(' in '),
            
            # String operations
            'num_strings': query.count("'"),
            'num_numbers': len([word for word in query.split() if word.replace('.', '').isdigit()]),
            
            # Advanced features
            'has_subquery': 1 if '(' in query and 'select' in query_lower else 0,
            'has_union': 1 if 'union' in query_lower else 0,
            'has_distinct': 1 if 'distinct' in query_lower else 0,
            'has_case': 1 if 'case' in query_lower else 0,
        }
        
        return features
    
    def collect_query_execution_data(self, query: str, dataset: str, query_type: str) -> Dict[str, Any]:
        """Execute query on both engines and collect data"""
        with self.lock:
            self.stats['total_queries'] += 1
        
        # Adapt query for schemas (limitation: using synthetic tables for now)
        pg_query, duck_query = self.adapt_query_for_schema(query, dataset)
        
        # Execute on both engines
        pg_success, pg_time, pg_error = self.execute_on_postgresql(pg_query)
        duck_success, duck_time, duck_error = self.execute_on_duckdb(duck_query)
        
        # Extract query features
        features = self.extract_query_features(query)
        
        # Determine optimal engine based on performance
        optimal_engine = 'unknown'
        if pg_success and duck_success:
            optimal_engine = 'duckdb' if duck_time < pg_time * 0.95 else 'postgresql'
            with self.lock:
                self.stats['successful_dual_executions'] += 1
        elif pg_success:
            optimal_engine = 'postgresql'
            with self.lock:
                self.stats['postgresql_only'] += 1
        elif duck_success:
            optimal_engine = 'duckdb'
            with self.lock:
                self.stats['duckdb_only'] += 1
        else:
            with self.lock:
                self.stats['failed_executions'] += 1
        
        # Collect training record
        record = {
            'query': query,
            'dataset': dataset,
            'query_type': query_type,
            'postgresql_time': pg_time if pg_success else None,
            'duckdb_time': duck_time if duck_success else None,
            'postgresql_success': pg_success,
            'duckdb_success': duck_success,
            'postgresql_error': pg_error,
            'duckdb_error': duck_error,
            'optimal_engine': optimal_engine,
            'performance_ratio': duck_time / pg_time if pg_success and duck_success and pg_time > 0 else None,
            'timestamp': datetime.now().isoformat(),
            **features
        }
        
        return record
    
    def load_queries_from_file(self, file_path: str) -> List[str]:
        """Load queries from SQL file"""
        try:
            with open(file_path, 'r') as f:
                queries = [line.strip() for line in f if line.strip() and not line.startswith('--')]
            return queries
        except Exception as e:
            logging.error(f"Failed to load queries from {file_path}: {e}")
            return []
    
    def collect_training_data(self, max_queries_per_file: int = 100):
        """Main collection process"""
        logging.info("🚀 Starting ZSCE Training Data Collection")
        logging.info("=" * 60)
        
        if not self.connect_engines():
            logging.error("Failed to connect to engines")
            return False
        
        # Find all generated query files
        query_files = glob.glob('data/zsce_queries_production/*/*.sql')
        logging.info(f"Found {len(query_files)} ZSCE query files")
        
        if not query_files:
            logging.error("No ZSCE query files found! Run ZSCE generation first.")
            return False
        
        # Process each query file
        for query_file in query_files:
            dataset = os.path.basename(os.path.dirname(query_file))
            query_type = 'AP' if 'ap_queries' in query_file else 'TP'
            
            logging.info(f"Processing {dataset} {query_type} queries from {query_file}")
            
            queries = self.load_queries_from_file(query_file)
            if not queries:
                continue
            
            # Limit queries for testing
            queries = queries[:max_queries_per_file]
            
            logging.info(f"Executing {len(queries)} {query_type} queries on {dataset}")
            
            # Execute queries and collect data
            for i, query in enumerate(queries):
                try:
                    record = self.collect_query_execution_data(query, dataset, query_type)
                    self.collected_data.append(record)
                    
                    if (i + 1) % 10 == 0:
                        logging.info(f"  Processed {i + 1}/{len(queries)} queries")
                        
                except Exception as e:
                    logging.error(f"Error processing query {i}: {e}")
                    continue
        
        # Save collected data
        self.save_training_data()
        self.print_collection_summary()
        
        return True
    
    def save_training_data(self):
        """Save collected training data to JSON file"""
        output_file = f'data/zsce_training_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        os.makedirs('data', exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(self.collected_data, f, indent=2, default=str)
        
        logging.info(f"💾 Saved {len(self.collected_data)} training records to {output_file}")
        return output_file
    
    def print_collection_summary(self):
        """Print collection statistics"""
        duration = datetime.now() - self.stats['start_time']
        
        logging.info("\n" + "="*60)
        logging.info("📈 ZSCE TRAINING DATA COLLECTION SUMMARY")
        logging.info("="*60)
        logging.info(f"⏱️  Duration: {duration}")
        logging.info(f"📊 Total queries processed: {self.stats['total_queries']:,}")
        logging.info(f"✅ Successful dual executions: {self.stats['successful_dual_executions']:,}")
        logging.info(f"🔸 PostgreSQL only: {self.stats['postgresql_only']:,}")
        logging.info(f"🔹 DuckDB only: {self.stats['duckdb_only']:,}")
        logging.info(f"❌ Failed executions: {self.stats['failed_executions']:,}")
        logging.info(f"💾 Training records collected: {len(self.collected_data):,}")
        
        if self.stats['successful_dual_executions'] > 0:
            dual_rate = (self.stats['successful_dual_executions'] / self.stats['total_queries']) * 100
            logging.info(f"📈 Dual execution success rate: {dual_rate:.1f}%")

def run_in_tmux():
    """Run the collection process in a tmux session"""
    session_name = "zsce_training_collection"
    
    # Kill existing session if any
    subprocess.run(["tmux", "kill-session", "-t", session_name], 
                  capture_output=True)
    
    # Create new tmux session and run collection
    python_cmd = f"cd /home/wuy/DB/pg_duckdb_postgres && export PATH=/home/wuy/DB/pg_duckdb_postgres/postgresql_local/bin:$PATH && python3 -c 'from collect_zsce_training_data import ZSCETrainingDataCollector; collector = ZSCETrainingDataCollector(); result = collector.collect_training_data(max_queries_per_file=200); print(\"Collection completed with result:\", result)'"
    
    tmux_cmd = [
        "tmux", "new-session", "-d", "-s", session_name,
        "bash", "-c", python_cmd
    ]
    
    print(f"🚀 Starting ZSCE training data collection in tmux session: {session_name}")
    print(f"📋 Monitor progress with: tmux attach -t {session_name}")
    print(f"⏹️  Stop with: tmux kill-session -t {session_name}")
    
    result = subprocess.run(tmux_cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ Tmux session started successfully")
        print(f"🔍 Check status: tmux list-sessions")
        return True
    else:
        print(f"✗ Failed to start tmux session: {result.stderr}")
        return False

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--tmux":
        success = run_in_tmux()
        sys.exit(0 if success else 1)
    else:
        collector = ZSCETrainingDataCollector()
        success = collector.collect_training_data(max_queries_per_file=200)
        sys.exit(0 if success else 1)