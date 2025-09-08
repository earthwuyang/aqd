#!/usr/bin/env python3
"""
Simple Real AQD Data Collector - Uses SQLite + DuckDB for real query execution
Demonstrates actual database execution and feature collection
"""

import os
import sys
import time
import logging
import sqlite3
import json
import pandas as pd
import numpy as np
import duckdb
from pathlib import Path
import random
import hashlib
from typing import Dict, List, Tuple

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

class SimpleRealDataCollector:
    """Real data collection using SQLite and DuckDB"""
    
    def __init__(self):
        self.base_dir = Path('/home/wuy/DB/pg_duckdb_postgres')
        self.data_dir = self.base_dir / 'real_data'
        
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        
        # Database connections
        self.sqlite_conn = sqlite3.connect(':memory:')
        self.sqlite_conn.execute('PRAGMA query_only = 0')
        
        self.duck_conn = duckdb.connect(':memory:')
        
    def create_test_data(self):
        """Create test tables with realistic data in both databases"""
        logger.info("Creating test tables with real data...")
        
        # Create customers table
        customers_data = []
        for i in range(10000):
            customers_data.append({
                'id': i + 1,
                'name': f'Customer_{i+1}',
                'category': random.choice(['premium', 'standard', 'basic']),
                'value': round(random.uniform(100.0, 10000.0), 2),
                'age': random.randint(18, 80),
                'city': random.choice(['New York', 'London', 'Tokyo', 'Berlin', 'Paris']),
                'signup_year': random.randint(2020, 2024)
            })
        
        # Create orders table
        orders_data = []
        for i in range(50000):
            orders_data.append({
                'id': i + 1,
                'customer_id': random.randint(1, 10000),
                'product_category': random.choice(['electronics', 'clothing', 'books', 'home']),
                'amount': round(random.uniform(10.0, 1000.0), 2),
                'quantity': random.randint(1, 10),
                'order_year': random.randint(2020, 2024),
                'order_month': random.randint(1, 12),
                'status': random.choice(['completed', 'pending', 'cancelled'])
            })
        
        # Create products table
        products_data = []
        for i in range(5000):
            products_data.append({
                'id': i + 1,
                'name': f'Product_{i+1}',
                'category': random.choice(['electronics', 'clothing', 'books', 'home']),
                'price': round(random.uniform(5.0, 500.0), 2),
                'rating': round(random.uniform(1.0, 5.0), 1),
                'reviews_count': random.randint(0, 1000),
                'in_stock': random.choice([0, 1])
            })
        
        # Load data into SQLite
        self.load_data_sqlite('customers', customers_data)
        self.load_data_sqlite('orders', orders_data) 
        self.load_data_sqlite('products', products_data)
        
        # Load data into DuckDB
        self.load_data_duckdb('customers', customers_data)
        self.load_data_duckdb('orders', orders_data)
        self.load_data_duckdb('products', products_data)
        
        logger.info(f"✓ Created tables: customers (10K), orders (50K), products (5K)")
        
    def load_data_sqlite(self, table_name: str, data: List[Dict]):
        """Load data into SQLite table"""
        if not data:
            return
            
        # Create table schema
        first_row = data[0]
        columns = []
        for key, value in first_row.items():
            if isinstance(value, int):
                col_type = 'INTEGER'
            elif isinstance(value, float):
                col_type = 'REAL'
            else:
                col_type = 'TEXT'
            columns.append(f"{key} {col_type}")
        
        create_sql = f"CREATE TABLE {table_name} ({', '.join(columns)})"
        self.sqlite_conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        self.sqlite_conn.execute(create_sql)
        
        # Insert data
        keys = list(first_row.keys())
        placeholders = ', '.join(['?' for _ in keys])
        insert_sql = f"INSERT INTO {table_name} ({', '.join(keys)}) VALUES ({placeholders})"
        
        values = [[row[key] for key in keys] for row in data]
        self.sqlite_conn.executemany(insert_sql, values)
        self.sqlite_conn.commit()
        
    def load_data_duckdb(self, table_name: str, data: List[Dict]):
        """Load data into DuckDB table"""
        df = pd.DataFrame(data)
        self.duck_conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
    
    def generate_query_workload(self) -> List[Dict]:
        """Generate realistic OLTP and OLAP queries"""
        queries = []
        
        # OLTP queries (simple, fast operations)
        oltp_templates = [
            "SELECT * FROM customers WHERE id = {id}",
            "SELECT name, value FROM customers WHERE category = '{category}' LIMIT 10",
            "SELECT * FROM orders WHERE customer_id = {customer_id} LIMIT 5",
            "SELECT COUNT(*) FROM orders WHERE status = 'completed'",
            "SELECT * FROM products WHERE category = '{category}' AND in_stock = 1 LIMIT 20",
            "SELECT AVG(price) FROM products WHERE category = '{category}'",
            "SELECT * FROM orders WHERE amount > {amount} LIMIT 15",
            "SELECT name FROM customers WHERE age BETWEEN {age1} AND {age2} LIMIT 25",
        ]
        
        # OLAP queries (complex analytics)
        olap_templates = [
            "SELECT category, COUNT(*), AVG(value) FROM customers GROUP BY category ORDER BY COUNT(*) DESC",
            "SELECT order_year, SUM(amount) FROM orders GROUP BY order_year ORDER BY order_year",
            "SELECT product_category, AVG(amount), COUNT(*) FROM orders GROUP BY product_category HAVING COUNT(*) > 100",
            "SELECT c.category, AVG(o.amount) FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.category",
            "SELECT order_month, SUM(amount), COUNT(DISTINCT customer_id) FROM orders WHERE order_year = 2023 GROUP BY order_month ORDER BY order_month",
            "SELECT category, MIN(price), MAX(price), AVG(rating) FROM products GROUP BY category",
            "SELECT city, COUNT(*), AVG(value) FROM customers GROUP BY city HAVING COUNT(*) > 100",
            "SELECT o.product_category, COUNT(DISTINCT o.customer_id) as unique_customers FROM orders o GROUP BY o.product_category ORDER BY unique_customers DESC",
        ]
        
        # Generate OLTP queries
        for i in range(1000):
            template = random.choice(oltp_templates)
            query = template.format(
                id=random.randint(1, 10000),
                customer_id=random.randint(1, 10000),
                category=random.choice(['premium', 'standard', 'basic', 'electronics', 'clothing', 'books', 'home']),
                amount=random.randint(50, 500),
                age1=random.randint(18, 40),
                age2=random.randint(41, 80)
            )
            
            queries.append({
                'query_id': f'oltp_{i}',
                'query_type': 'oltp',
                'query': query,
                'complexity': 'simple'
            })
        
        # Generate OLAP queries
        for i in range(500):
            template = random.choice(olap_templates)
            query = template
            
            queries.append({
                'query_id': f'olap_{i}',
                'query_type': 'olap', 
                'query': query,
                'complexity': 'complex'
            })
        
        random.shuffle(queries)
        return queries
    
    def execute_query_sqlite(self, query: str) -> Dict:
        """Execute query on SQLite and measure performance"""
        start_time = time.perf_counter()
        
        try:
            cursor = self.sqlite_conn.cursor()
            
            # Get query plan
            explain_query = f"EXPLAIN QUERY PLAN {query}"
            cursor.execute(explain_query)
            plan_rows = cursor.fetchall()
            plan_text = '\n'.join([str(row) for row in plan_rows])
            
            # Execute query
            query_start = time.perf_counter()
            cursor.execute(query)
            results = cursor.fetchall()
            query_end = time.perf_counter()
            
            execution_time = query_end - query_start
            total_time = time.perf_counter() - start_time
            
            # Extract features
            features = self.extract_sqlite_features(query, plan_text)
            
            return {
                'execution_time': execution_time,
                'total_time': total_time,
                'row_count': len(results),
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
    
    def execute_query_duckdb(self, query: str) -> Dict:
        """Execute query on DuckDB and measure performance"""
        start_time = time.perf_counter()
        
        try:
            # Get query plan
            explain_query = f"EXPLAIN {query}"
            plan_result = self.duck_conn.execute(explain_query).fetchall()
            plan_text = '\n'.join([str(row[0]) for row in plan_result])
            
            # Execute query
            query_start = time.perf_counter()
            results = self.duck_conn.execute(query).fetchall()
            query_end = time.perf_counter()
            
            execution_time = query_end - query_start
            total_time = time.perf_counter() - start_time
            
            # Extract features
            features = self.extract_duckdb_features(query, plan_text)
            
            return {
                'execution_time': execution_time,
                'total_time': total_time,
                'row_count': len(results),
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
    
    def extract_sqlite_features(self, query: str, plan_text: str) -> Dict:
        """Extract features from SQLite query and plan"""
        features = {}
        
        # Query structure features
        query_upper = query.upper()
        features['query_length'] = len(query)
        features['has_where'] = 1 if 'WHERE' in query_upper else 0
        features['has_group_by'] = 1 if 'GROUP BY' in query_upper else 0
        features['has_order_by'] = 1 if 'ORDER BY' in query_upper else 0
        features['has_join'] = 1 if 'JOIN' in query_upper else 0
        features['has_aggregation'] = 1 if any(func in query_upper for func in ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX']) else 0
        features['has_limit'] = 1 if 'LIMIT' in query_upper else 0
        features['has_having'] = 1 if 'HAVING' in query_upper else 0
        features['has_distinct'] = 1 if 'DISTINCT' in query_upper else 0
        
        # Plan features
        features['plan_steps'] = len(plan_text.split('\n')) if plan_text else 0
        features['has_scan'] = 1 if 'SCAN' in plan_text.upper() else 0
        features['has_index'] = 1 if 'INDEX' in plan_text.upper() else 0
        features['has_sort'] = 1 if 'SORT' in plan_text.upper() else 0
        features['has_temp_btree'] = 1 if 'TEMP B-TREE' in plan_text.upper() else 0
        
        # Table access patterns
        tables = ['customers', 'orders', 'products']
        for table in tables:
            features[f'accesses_{table}'] = 1 if table.upper() in query_upper else 0
        
        # Estimate selectivity based on query patterns
        if 'id =' in query.lower():
            features['selectivity'] = 0.0001  # Very selective
        elif 'where' in query.lower() and '=' in query:
            features['selectivity'] = 0.1  # Moderately selective
        elif 'where' in query.lower():
            features['selectivity'] = 0.3  # Somewhat selective
        else:
            features['selectivity'] = 1.0  # Full scan
        
        return features
    
    def extract_duckdb_features(self, query: str, plan_text: str) -> Dict:
        """Extract features from DuckDB query and plan"""
        features = {}
        
        # Query structure features (same as SQLite)
        query_upper = query.upper()
        features['query_length'] = len(query)
        features['has_where'] = 1 if 'WHERE' in query_upper else 0
        features['has_group_by'] = 1 if 'GROUP BY' in query_upper else 0
        features['has_order_by'] = 1 if 'ORDER BY' in query_upper else 0
        features['has_join'] = 1 if 'JOIN' in query_upper else 0
        features['has_aggregation'] = 1 if any(func in query_upper for func in ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX']) else 0
        features['has_limit'] = 1 if 'LIMIT' in query_upper else 0
        features['has_having'] = 1 if 'HAVING' in query_upper else 0
        features['has_distinct'] = 1 if 'DISTINCT' in query_upper else 0
        
        # DuckDB-specific plan features
        features['plan_depth'] = plan_text.count('├') + plan_text.count('└') if plan_text else 0
        features['has_seq_scan'] = 1 if 'SEQ_SCAN' in plan_text.upper() else 0
        features['has_hash_join'] = 1 if 'HASH_JOIN' in plan_text.upper() else 0
        features['has_hash_aggregate'] = 1 if 'HASH_AGGREGATE' in plan_text.upper() else 0
        features['has_order'] = 1 if 'ORDER' in plan_text.upper() else 0
        features['has_projection'] = 1 if 'PROJECTION' in plan_text.upper() else 0
        features['has_filter'] = 1 if 'FILTER' in plan_text.upper() else 0
        
        # Table access patterns
        tables = ['customers', 'orders', 'products']
        for table in tables:
            features[f'accesses_{table}'] = 1 if table.upper() in query_upper else 0
        
        # Estimate selectivity (same logic as SQLite)
        if 'id =' in query.lower():
            features['selectivity'] = 0.0001
        elif 'where' in query.lower() and '=' in query:
            features['selectivity'] = 0.1
        elif 'where' in query.lower():
            features['selectivity'] = 0.3
        else:
            features['selectivity'] = 1.0
        
        return features
    
    def collect_execution_data(self, queries: List[Dict]) -> pd.DataFrame:
        """Execute all queries and collect performance data"""
        logger.info(f"Executing {len(queries)} real queries on both databases...")
        
        results = []
        
        for i, query_info in enumerate(queries):
            if i % 100 == 0:
                logger.info(f"Progress: {i}/{len(queries)} queries completed")
            
            query = query_info['query']
            
            # Execute on SQLite (representing PostgreSQL-style engine)
            sqlite_result = self.execute_query_sqlite(query)
            
            # Execute on DuckDB
            duck_result = self.execute_query_duckdb(query)
            
            # Combine results if both succeeded
            if sqlite_result['success'] and duck_result['success']:
                result = {
                    'query_id': query_info['query_id'],
                    'query_type': query_info['query_type'],
                    'complexity': query_info['complexity'],
                    'query': query[:150],  # Truncate for storage
                    'query_hash': hashlib.md5(query.encode()).hexdigest()[:12],
                    
                    # SQLite (PostgreSQL-like) metrics
                    'sqlite_execution_time': sqlite_result['execution_time'],
                    'sqlite_total_time': sqlite_result['total_time'],
                    'sqlite_row_count': sqlite_result['row_count'],
                    
                    # DuckDB metrics
                    'duck_execution_time': duck_result['execution_time'],
                    'duck_total_time': duck_result['total_time'],
                    'duck_row_count': duck_result['row_count'],
                    
                    # Derived metrics
                    'time_gap': sqlite_result['execution_time'] - duck_result['execution_time'],
                    'log_time_gap': np.log(max(sqlite_result['execution_time'], 0.000001)) - np.log(max(duck_result['execution_time'], 0.000001)),
                    'speedup_ratio': duck_result['execution_time'] / max(sqlite_result['execution_time'], 0.000001),
                    'best_engine': 'sqlite' if sqlite_result['execution_time'] < duck_result['execution_time'] else 'duckdb',
                }
                
                # Add features from both engines
                for key, value in sqlite_result['features'].items():
                    result[f'feature_sqlite_{key}'] = value
                
                for key, value in duck_result['features'].items():
                    result[f'feature_duck_{key}'] = value
                
                results.append(result)
            
            else:
                failed_engine = []
                if not sqlite_result['success']:
                    failed_engine.append('SQLite')
                if not duck_result['success']:
                    failed_engine.append('DuckDB')
                logger.warning(f"Query {query_info['query_id']} failed on: {', '.join(failed_engine)}")
        
        logger.info(f"Successfully collected data for {len(results)} queries")
        return pd.DataFrame(results)
    
    def cleanup(self):
        """Clean up database connections"""
        self.sqlite_conn.close()
        self.duck_conn.close()

def main():
    """Main execution pipeline"""
    print("""
    ╔══════════════════════════════════════════════════════════════════════════════════════╗
    ║                    REAL AQD DATA COLLECTION - SQLite + DuckDB                       ║
    ║                        Execute Real Queries and Collect Metrics                     ║
    ╚══════════════════════════════════════════════════════════════════════════════════════╝
    """)
    
    collector = SimpleRealDataCollector()
    
    try:
        # Phase 1: Create test data
        logger.info("Phase 1: Creating test data in both databases...")
        collector.create_test_data()
        
        # Phase 2: Generate query workload
        logger.info("Phase 2: Generating query workload...")
        queries = collector.generate_query_workload()
        logger.info(f"Generated {len(queries)} queries:")
        logger.info(f"  - OLTP queries: {len([q for q in queries if q['query_type'] == 'oltp'])}")
        logger.info(f"  - OLAP queries: {len([q for q in queries if q['query_type'] == 'olap'])}")
        
        # Phase 3: Execute queries and collect data
        logger.info("Phase 3: Executing queries and collecting real performance data...")
        df = collector.collect_execution_data(queries)
        
        # Phase 4: Save results
        data_file = collector.data_dir / 'real_execution_data.csv'
        df.to_csv(data_file, index=False)
        
        logger.info(f"\n" + "="*80)
        logger.info("🎉 REAL DATA COLLECTION COMPLETED!")
        logger.info("="*80)
        logger.info(f"📊 Data saved to: {data_file}")
        logger.info(f"📈 Total queries executed: {len(df)}")
        logger.info(f"📋 Features collected: {len([col for col in df.columns if col.startswith('feature_')])}")
        
        # Print summary statistics
        if len(df) > 0:
            logger.info(f"\n📊 REAL EXECUTION TIME SUMMARY:")
            logger.info(f"SQLite - Mean: {df['sqlite_execution_time'].mean():.6f}s, Median: {df['sqlite_execution_time'].median():.6f}s")
            logger.info(f"DuckDB - Mean: {df['duck_execution_time'].mean():.6f}s, Median: {df['duck_execution_time'].median():.6f}s")
            logger.info(f"SQLite faster in: {(df['best_engine'] == 'sqlite').mean():.1%} of queries")
            logger.info(f"DuckDB faster in: {(df['best_engine'] == 'duckdb').mean():.1%} of queries")
            
            # Query type breakdown
            oltp_data = df[df['query_type'] == 'oltp']
            olap_data = df[df['query_type'] == 'olap']
            
            if len(oltp_data) > 0:
                logger.info(f"\n📋 OLTP QUERIES ({len(oltp_data)} queries):")
                logger.info(f"  SQLite faster: {(oltp_data['best_engine'] == 'sqlite').mean():.1%}")
                logger.info(f"  DuckDB faster: {(oltp_data['best_engine'] == 'duckdb').mean():.1%}")
                
            if len(olap_data) > 0:
                logger.info(f"\n📋 OLAP QUERIES ({len(olap_data)} queries):")
                logger.info(f"  SQLite faster: {(olap_data['best_engine'] == 'sqlite').mean():.1%}")
                logger.info(f"  DuckDB faster: {(olap_data['best_engine'] == 'duckdb').mean():.1%}")
        
        return data_file
        
    except Exception as e:
        logger.error(f"Real data collection failed: {e}")
        raise
    finally:
        collector.cleanup()

if __name__ == "__main__":
    main()