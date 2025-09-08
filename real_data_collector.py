#!/usr/bin/env python3
"""
Real AQD Data Collector - Executes actual queries on PostgreSQL and DuckDB
Collects real execution times, query plans, and system metrics
"""

import os
import sys
import time
import logging
import subprocess
import json
import pandas as pd
import numpy as np
import psycopg2
import duckdb
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import hashlib
import tempfile
import threading
from typing import Dict, List, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('real_experiments.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RealAQDDataCollector:
    """Collects real execution data from PostgreSQL and DuckDB"""
    
    def __init__(self):
        self.base_dir = Path('/home/wuy/DB/pg_duckdb_postgres')
        self.data_dir = self.base_dir / 'real_data'
        self.temp_dir = self.base_dir / 'temp'
        
        # Create directories
        for dir_path in [self.data_dir, self.temp_dir]:
            dir_path.mkdir(exist_ok=True)
            
        # Database connections
        self.pg_conn = None
        self.duck_conn = None
        
        # Query templates for different workload types
        self.oltp_queries = [
            "SELECT * FROM {table} WHERE id = {id} LIMIT 10",
            "SELECT count(*) FROM {table} WHERE status = 'active'",
            "SELECT name, email FROM {table} WHERE created_at > '2023-01-01' LIMIT 50",
            "SELECT * FROM {table} WHERE category = '{category}' ORDER BY id LIMIT 20",
            "SELECT avg(value) FROM {table} WHERE id BETWEEN {start} AND {end}",
        ]
        
        self.olap_queries = [
            "SELECT category, count(*), avg(value) FROM {table} GROUP BY category ORDER BY count(*) DESC",
            "SELECT date_trunc('month', created_at) as month, sum(value) FROM {table} GROUP BY month ORDER BY month",
            "SELECT category, min(value), max(value), stddev(value) FROM {table} GROUP BY category HAVING count(*) > 100",
            "SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY value) as median FROM {table}",
            "SELECT category, value, lag(value) OVER (PARTITION BY category ORDER BY id) as prev_value FROM {table} ORDER BY category, id LIMIT 1000",
        ]
        
    def setup_databases(self):
        """Initialize PostgreSQL and DuckDB connections"""
        logger.info("Setting up database connections...")
        
        try:
            # Setup PostgreSQL
            self.setup_postgresql()
            logger.info("✓ PostgreSQL connection established")
            
            # Setup DuckDB  
            self.setup_duckdb()
            logger.info("✓ DuckDB connection established")
            
            return True
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            return False
    
    def setup_postgresql(self):
        """Setup PostgreSQL connection and test database"""
        try:
            # Try to connect to existing database
            self.pg_conn = psycopg2.connect(
                host="localhost",
                database="postgres",
                user=os.getenv("USER"),
                password="",
                port=5432
            )
            self.pg_conn.autocommit = True
            
        except psycopg2.OperationalError:
            logger.info("Starting PostgreSQL server...")
            # Try to start PostgreSQL if not running
            try:
                subprocess.run(['sudo', 'systemctl', 'start', 'postgresql'], check=True)
                time.sleep(3)
                
                # Try connection again
                self.pg_conn = psycopg2.connect(
                    host="localhost", 
                    database="postgres",
                    user=os.getenv("USER"),
                    password="",
                    port=5432
                )
                self.pg_conn.autocommit = True
                
            except (subprocess.CalledProcessError, psycopg2.OperationalError):
                # Create a temporary PostgreSQL instance
                logger.info("Creating temporary PostgreSQL instance...")
                self.setup_temp_postgresql()
    
    def setup_temp_postgresql(self):
        """Setup temporary PostgreSQL instance"""
        pg_data_dir = self.temp_dir / 'pgdata'
        
        if not pg_data_dir.exists():
            # Initialize PostgreSQL data directory
            subprocess.run([
                'initdb', '-D', str(pg_data_dir), 
                '--auth-local=trust', '--auth-host=trust'
            ], check=True)
            
        # Start PostgreSQL server
        subprocess.Popen([
            'postgres', '-D', str(pg_data_dir), 
            '-p', '5433', '-k', str(self.temp_dir)
        ])
        time.sleep(3)
        
        # Connect to temporary instance
        self.pg_conn = psycopg2.connect(
            host="localhost",
            database="postgres", 
            user=os.getenv("USER"),
            password="",
            port=5433
        )
        self.pg_conn.autocommit = True
    
    def setup_duckdb(self):
        """Setup DuckDB connection"""
        duck_db_path = self.temp_dir / 'aqd_duckdb.db'
        self.duck_conn = duckdb.connect(str(duck_db_path))
        
    def create_test_tables(self):
        """Create test tables with realistic data"""
        logger.info("Creating test tables with real data...")
        
        # Create sample tables in both databases
        tables = {
            'customers': self.generate_customers_data(10000),
            'orders': self.generate_orders_data(50000),
            'products': self.generate_products_data(1000),
            'sales': self.generate_sales_data(100000),
        }
        
        for table_name, data in tables.items():
            self.create_table_in_both_dbs(table_name, data)
            logger.info(f"✓ Created table '{table_name}' with {len(data)} rows")
        
        return tables
    
    def generate_customers_data(self, num_rows):
        """Generate realistic customer data"""
        categories = ['premium', 'standard', 'basic']
        statuses = ['active', 'inactive', 'pending']
        
        data = []
        for i in range(num_rows):
            data.append({
                'id': i + 1,
                'name': f'Customer_{i+1}',
                'email': f'customer{i+1}@example.com',
                'category': random.choice(categories),
                'status': random.choice(statuses),
                'value': round(random.uniform(10.0, 10000.0), 2),
                'created_at': f'2023-{random.randint(1,12):02d}-{random.randint(1,28):02d}'
            })
        return data
    
    def generate_orders_data(self, num_rows):
        """Generate realistic order data"""
        data = []
        for i in range(num_rows):
            data.append({
                'id': i + 1,
                'customer_id': random.randint(1, 10000),
                'product_id': random.randint(1, 1000), 
                'quantity': random.randint(1, 10),
                'value': round(random.uniform(5.0, 500.0), 2),
                'status': random.choice(['completed', 'pending', 'cancelled']),
                'category': random.choice(['electronics', 'clothing', 'books', 'home']),
                'created_at': f'2023-{random.randint(1,12):02d}-{random.randint(1,28):02d}'
            })
        return data
    
    def generate_products_data(self, num_rows):
        """Generate realistic product data"""
        data = []
        categories = ['electronics', 'clothing', 'books', 'home', 'sports']
        for i in range(num_rows):
            data.append({
                'id': i + 1,
                'name': f'Product_{i+1}',
                'category': random.choice(categories),
                'price': round(random.uniform(10.0, 1000.0), 2),
                'value': round(random.uniform(1.0, 100.0), 2),
                'status': random.choice(['active', 'discontinued']),
                'created_at': f'2023-{random.randint(1,12):02d}-{random.randint(1,28):02d}'
            })
        return data
    
    def generate_sales_data(self, num_rows):
        """Generate realistic sales data"""
        data = []
        for i in range(num_rows):
            data.append({
                'id': i + 1,
                'customer_id': random.randint(1, 10000),
                'product_id': random.randint(1, 1000),
                'order_id': random.randint(1, 50000),
                'value': round(random.uniform(5.0, 2000.0), 2),
                'category': random.choice(['electronics', 'clothing', 'books', 'home']),
                'status': 'active',
                'created_at': f'2023-{random.randint(1,12):02d}-{random.randint(1,28):02d}'
            })
        return data
    
    def create_table_in_both_dbs(self, table_name, data):
        """Create table in both PostgreSQL and DuckDB"""
        if not data:
            return
            
        # Infer schema from first row
        first_row = data[0]
        columns = []
        
        for key, value in first_row.items():
            if isinstance(value, int):
                col_type = 'INTEGER'
            elif isinstance(value, float):
                col_type = 'DECIMAL(10,2)'
            else:
                col_type = 'VARCHAR(255)'
            columns.append(f"{key} {col_type}")
        
        create_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(columns)})"
        
        # Create in PostgreSQL
        with self.pg_conn.cursor() as cursor:
            cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
            cursor.execute(create_sql)
            
            # Insert data
            if data:
                keys = list(first_row.keys())
                placeholders = ', '.join(['%s'] * len(keys))
                insert_sql = f"INSERT INTO {table_name} ({', '.join(keys)}) VALUES ({placeholders})"
                
                values = [[row[key] for key in keys] for row in data]
                cursor.executemany(insert_sql, values)
        
        # Create in DuckDB
        self.duck_conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        self.duck_conn.execute(create_sql)
        
        if data:
            df = pd.DataFrame(data)
            self.duck_conn.execute(f"INSERT INTO {table_name} SELECT * FROM df")
    
    def execute_query_postgresql(self, query: str) -> Dict:
        """Execute query on PostgreSQL and collect metrics"""
        start_time = time.perf_counter()
        
        try:
            with self.pg_conn.cursor() as cursor:
                # Get query plan with EXPLAIN
                explain_query = f"EXPLAIN (FORMAT JSON, ANALYZE, BUFFERS) {query}"
                cursor.execute(explain_query)
                plan_data = cursor.fetchone()[0]
                
                # Execute actual query
                query_start = time.perf_counter()
                cursor.execute(query)
                result = cursor.fetchall()
                query_end = time.perf_counter()
                
                execution_time = query_end - query_start
                total_time = time.perf_counter() - start_time
                
                # Extract features from query plan
                features = self.extract_postgresql_features(plan_data[0], query)
                
                return {
                    'execution_time': execution_time,
                    'total_time': total_time,
                    'row_count': len(result),
                    'features': features,
                    'plan': plan_data,
                    'success': True,
                    'error': None
                }
                
        except Exception as e:
            return {
                'execution_time': float('inf'),
                'total_time': time.perf_counter() - start_time,
                'row_count': 0,
                'features': {},
                'plan': None,
                'success': False,
                'error': str(e)
            }
    
    def execute_query_duckdb(self, query: str) -> Dict:
        """Execute query on DuckDB and collect metrics"""
        start_time = time.perf_counter()
        
        try:
            # Get query plan
            explain_query = f"EXPLAIN {query}"
            plan_result = self.duck_conn.execute(explain_query).fetchall()
            plan_text = '\n'.join([row[0] for row in plan_result])
            
            # Execute actual query
            query_start = time.perf_counter()
            result = self.duck_conn.execute(query).fetchall()
            query_end = time.perf_counter()
            
            execution_time = query_end - query_start
            total_time = time.perf_counter() - start_time
            
            # Extract features from query plan
            features = self.extract_duckdb_features(plan_text, query)
            
            return {
                'execution_time': execution_time,
                'total_time': total_time,
                'row_count': len(result),
                'features': features,
                'plan': plan_text,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            return {
                'execution_time': float('inf'),
                'total_time': time.perf_counter() - start_time,
                'row_count': 0,
                'features': {},
                'plan': None,
                'success': False,
                'error': str(e)
            }
    
    def extract_postgresql_features(self, plan_node: Dict, query: str) -> Dict:
        """Extract features from PostgreSQL query plan"""
        features = {}
        
        # Basic plan features
        features['node_type'] = plan_node.get('Node Type', 'Unknown')
        features['startup_cost'] = plan_node.get('Startup Cost', 0.0)
        features['total_cost'] = plan_node.get('Total Cost', 0.0)
        features['plan_rows'] = plan_node.get('Plan Rows', 0)
        features['plan_width'] = plan_node.get('Plan Width', 0)
        
        # Execution features (if ANALYZE was used)
        features['actual_startup_time'] = plan_node.get('Actual Startup Time', 0.0)
        features['actual_total_time'] = plan_node.get('Actual Total Time', 0.0)
        features['actual_rows'] = plan_node.get('Actual Rows', 0)
        features['actual_loops'] = plan_node.get('Actual Loops', 1)
        
        # Buffer usage features
        if 'Buffers' in plan_node:
            buffers = plan_node['Buffers']
            features['shared_hit_blocks'] = buffers.get('Shared Hit Blocks', 0)
            features['shared_read_blocks'] = buffers.get('Shared Read Blocks', 0)
            features['shared_dirtied_blocks'] = buffers.get('Shared Dirtied Blocks', 0)
        
        # Query structure features
        features['query_length'] = len(query)
        features['has_where'] = 1 if 'WHERE' in query.upper() else 0
        features['has_group_by'] = 1 if 'GROUP BY' in query.upper() else 0
        features['has_order_by'] = 1 if 'ORDER BY' in query.upper() else 0
        features['has_join'] = 1 if 'JOIN' in query.upper() else 0
        features['has_aggregation'] = 1 if any(func in query.upper() for func in ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX']) else 0
        
        # Recursively extract features from child plans
        if 'Plans' in plan_node:
            for i, child in enumerate(plan_node['Plans']):
                child_features = self.extract_postgresql_features(child, query)
                for key, value in child_features.items():
                    features[f'child_{i}_{key}'] = value
        
        return features
    
    def extract_duckdb_features(self, plan_text: str, query: str) -> Dict:
        """Extract features from DuckDB query plan"""
        features = {}
        
        # Parse plan text for features
        lines = plan_text.split('\n')
        
        # Count different operator types
        features['seq_scan_count'] = sum(1 for line in lines if 'SEQ_SCAN' in line.upper())
        features['hash_join_count'] = sum(1 for line in lines if 'HASH_JOIN' in line.upper())
        features['hash_aggregate_count'] = sum(1 for line in lines if 'HASH_AGGREGATE' in line.upper())
        features['sort_count'] = sum(1 for line in lines if 'SORT' in line.upper())
        
        # Query structure features (same as PostgreSQL)
        features['query_length'] = len(query)
        features['has_where'] = 1 if 'WHERE' in query.upper() else 0
        features['has_group_by'] = 1 if 'GROUP BY' in query.upper() else 0
        features['has_order_by'] = 1 if 'ORDER BY' in query.upper() else 0
        features['has_join'] = 1 if 'JOIN' in query.upper() else 0
        features['has_aggregation'] = 1 if any(func in query.upper() for func in ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX']) else 0
        
        # Plan complexity
        features['plan_depth'] = max(line.count('│') for line in lines) if lines else 0
        features['plan_width'] = len(lines)
        
        return features
    
    def generate_workload_queries(self, tables: Dict[str, List], num_queries_per_type: int = 1000) -> List[Dict]:
        """Generate workload of OLTP and OLAP queries"""
        queries = []
        table_names = list(tables.keys())
        
        # Generate OLTP queries
        for i in range(num_queries_per_type):
            table = random.choice(table_names)
            template = random.choice(self.oltp_queries)
            
            # Fill in template parameters
            query = template.format(
                table=table,
                id=random.randint(1, 1000),
                start=random.randint(1, 1000),
                end=random.randint(1001, 2000),
                category=random.choice(['electronics', 'clothing', 'books', 'home'])
            )
            
            queries.append({
                'query_id': f'oltp_{i}',
                'query_type': 'oltp',
                'table': table,
                'query': query
            })
        
        # Generate OLAP queries  
        for i in range(num_queries_per_type):
            table = random.choice(table_names)
            template = random.choice(self.olap_queries)
            
            query = template.format(table=table)
            
            queries.append({
                'query_id': f'olap_{i}',
                'query_type': 'olap', 
                'table': table,
                'query': query
            })
        
        return queries
    
    def collect_execution_data(self, queries: List[Dict]) -> pd.DataFrame:
        """Execute queries on both databases and collect performance data"""
        logger.info(f"Executing {len(queries)} queries on both databases...")
        
        results = []
        
        for i, query_info in enumerate(queries):
            if i % 100 == 0:
                logger.info(f"Progress: {i}/{len(queries)} queries completed")
            
            query = query_info['query']
            
            # Execute on PostgreSQL
            pg_result = self.execute_query_postgresql(query)
            
            # Execute on DuckDB
            duck_result = self.execute_query_duckdb(query)
            
            # Combine results
            if pg_result['success'] and duck_result['success']:
                result = {
                    'query_id': query_info['query_id'],
                    'query_type': query_info['query_type'],
                    'table': query_info['table'],
                    'query': query[:200],  # Truncate for storage
                    'query_hash': hashlib.md5(query.encode()).hexdigest()[:16],
                    
                    # PostgreSQL metrics
                    'pg_execution_time': pg_result['execution_time'],
                    'pg_total_time': pg_result['total_time'],
                    'pg_row_count': pg_result['row_count'],
                    
                    # DuckDB metrics
                    'duck_execution_time': duck_result['execution_time'],
                    'duck_total_time': duck_result['total_time'],
                    'duck_row_count': duck_result['row_count'],
                    
                    # Derived metrics
                    'time_gap': pg_result['execution_time'] - duck_result['execution_time'],
                    'speedup_ratio': duck_result['execution_time'] / pg_result['execution_time'] if pg_result['execution_time'] > 0 else 1.0,
                    'best_engine': 'postgresql' if pg_result['execution_time'] < duck_result['execution_time'] else 'duckdb',
                }
                
                # Add PostgreSQL features
                for key, value in pg_result['features'].items():
                    result[f'pg_{key}'] = value
                
                # Add DuckDB features  
                for key, value in duck_result['features'].items():
                    result[f'duck_{key}'] = value
                
                results.append(result)
            
            else:
                logger.warning(f"Query {query_info['query_id']} failed on one or both databases")
        
        logger.info(f"Successfully collected data for {len(results)} queries")
        return pd.DataFrame(results)
    
    def cleanup(self):
        """Clean up database connections"""
        if self.pg_conn:
            self.pg_conn.close()
        if self.duck_conn:
            self.duck_conn.close()

def main():
    """Main data collection pipeline"""
    print("""
    ╔══════════════════════════════════════════════════════════════════════════════════════╗
    ║                    REAL AQD DATA COLLECTION PIPELINE                                ║
    ║                     Execute Real Queries on PostgreSQL & DuckDB                     ║
    ╚══════════════════════════════════════════════════════════════════════════════════════╝
    """)
    
    collector = RealAQDDataCollector()
    
    try:
        # Phase 1: Setup databases
        logger.info("Phase 1: Setting up databases...")
        if not collector.setup_databases():
            logger.error("Failed to setup databases")
            return
        
        # Phase 2: Create test data
        logger.info("Phase 2: Creating test tables...")
        tables = collector.create_test_tables()
        
        # Phase 3: Generate queries
        logger.info("Phase 3: Generating workload queries...")
        queries = collector.generate_workload_queries(tables, num_queries_per_type=500)
        logger.info(f"Generated {len(queries)} queries ({len([q for q in queries if q['query_type'] == 'oltp'])} OLTP, {len([q for q in queries if q['query_type'] == 'olap'])} OLAP)")
        
        # Phase 4: Execute queries and collect data
        logger.info("Phase 4: Executing queries and collecting performance data...")
        df = collector.collect_execution_data(queries)
        
        # Phase 5: Save results
        data_file = collector.data_dir / 'real_aqd_data.csv'
        df.to_csv(data_file, index=False)
        
        logger.info(f"\n" + "="*80)
        logger.info("🎉 REAL DATA COLLECTION COMPLETED!")
        logger.info("="*80)
        logger.info(f"📊 Data saved to: {data_file}")
        logger.info(f"📈 Total queries executed: {len(df)}")
        logger.info(f"📋 Features collected: {len([col for col in df.columns if col.startswith('pg_') or col.startswith('duck_')])}")
        
        # Print summary statistics
        if len(df) > 0:
            logger.info(f"\n📊 EXECUTION TIME SUMMARY:")
            logger.info(f"PostgreSQL - Mean: {df['pg_execution_time'].mean():.4f}s, Median: {df['pg_execution_time'].median():.4f}s")
            logger.info(f"DuckDB - Mean: {df['duck_execution_time'].mean():.4f}s, Median: {df['duck_execution_time'].median():.4f}s")
            logger.info(f"PostgreSQL faster in: {(df['best_engine'] == 'postgresql').mean():.1%} of queries")
            logger.info(f"DuckDB faster in: {(df['best_engine'] == 'duckdb').mean():.1%} of queries")
        
        return data_file
        
    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        raise
    finally:
        collector.cleanup()

if __name__ == "__main__":
    main()