#!/usr/bin/env python3
"""
Real PostgreSQL + DuckDB Data Collection System
Executes actual queries on PostgreSQL and DuckDB and collects performance metrics
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
import random
import hashlib
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('postgres_duckdb_experiments.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PostgresDuckDBCollector:
    """Real data collection using PostgreSQL and DuckDB"""
    
    def __init__(self):
        self.base_dir = Path('/home/wuy/DB/pg_duckdb_postgres')
        self.data_dir = self.base_dir / 'postgres_duckdb_data'
        
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        
        # Database connections
        self.pg_conn = None
        self.duck_conn = None
        
    def setup_postgresql(self):
        """Setup PostgreSQL connection"""
        logger.info("Setting up PostgreSQL connection...")
        
        try:
            # Try connecting to default PostgreSQL installation
            self.pg_conn = psycopg2.connect(
                host="localhost",
                database="postgres",
                user=os.getenv("USER", "postgres"),
                password="",
                port=5432
            )
            self.pg_conn.autocommit = True
            logger.info("✓ Connected to PostgreSQL")
            return True
            
        except psycopg2.OperationalError as e:
            logger.warning(f"Failed to connect to PostgreSQL on port 5432: {e}")
            
            # Try alternative connection
            try:
                self.pg_conn = psycopg2.connect(
                    host="localhost",
                    database="postgres",
                    user="postgres",
                    password="",
                    port=5432
                )
                self.pg_conn.autocommit = True
                logger.info("✓ Connected to PostgreSQL with postgres user")
                return True
            except Exception as e2:
                logger.error(f"Could not connect to PostgreSQL: {e2}")
                return False
    
    def setup_duckdb(self):
        """Setup DuckDB connection"""
        logger.info("Setting up DuckDB connection...")
        try:
            duck_db_path = self.data_dir / 'postgres_duckdb_comparison.db'
            self.duck_conn = duckdb.connect(str(duck_db_path))
            logger.info("✓ Connected to DuckDB")
            return True
        except Exception as e:
            logger.error(f"Could not connect to DuckDB: {e}")
            return False
    
    def create_test_data(self):
        """Create test tables with realistic data in both databases"""
        logger.info("Creating test data in both PostgreSQL and DuckDB...")
        
        # Create test database schema
        try:
            # Drop and recreate test database for clean state
            with self.pg_conn.cursor() as cursor:
                cursor.execute("DROP DATABASE IF EXISTS aqd_test")
                cursor.execute("CREATE DATABASE aqd_test")
            
            # Reconnect to test database
            self.pg_conn.close()
            self.pg_conn = psycopg2.connect(
                host="localhost",
                database="aqd_test",
                user=os.getenv("USER", "postgres"),
                password="",
                port=5432
            )
            self.pg_conn.autocommit = True
            
        except Exception as e:
            logger.warning(f"Could not create test database, using default: {e}")
        
        # Generate larger test datasets
        customers_data = []
        for i in range(50000):  # 50K customers
            customers_data.append({
                'id': i + 1,
                'name': f'Customer_{i+1}',
                'email': f'customer{i+1}@example.com',
                'category': random.choice(['premium', 'standard', 'basic', 'enterprise']),
                'region': random.choice(['North', 'South', 'East', 'West', 'Central']),
                'value': round(random.uniform(100.0, 50000.0), 2),
                'age': random.randint(18, 85),
                'registration_year': random.randint(2018, 2024),
                'credit_score': random.randint(300, 850),
                'is_active': random.choice([True, False])
            })
        
        orders_data = []
        for i in range(200000):  # 200K orders
            orders_data.append({
                'id': i + 1,
                'customer_id': random.randint(1, 50000),
                'product_category': random.choice(['electronics', 'clothing', 'books', 'home', 'sports', 'automotive']),
                'product_id': random.randint(1, 10000),
                'amount': round(random.uniform(10.0, 5000.0), 2),
                'quantity': random.randint(1, 20),
                'order_date': f'2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}',
                'status': random.choice(['completed', 'pending', 'cancelled', 'shipped']),
                'discount': round(random.uniform(0.0, 0.3), 2),
                'shipping_cost': round(random.uniform(0.0, 50.0), 2)
            })
        
        products_data = []
        for i in range(10000):  # 10K products
            products_data.append({
                'id': i + 1,
                'name': f'Product_{i+1}',
                'category': random.choice(['electronics', 'clothing', 'books', 'home', 'sports', 'automotive']),
                'price': round(random.uniform(5.0, 2000.0), 2),
                'cost': round(random.uniform(2.0, 800.0), 2),
                'rating': round(random.uniform(1.0, 5.0), 1),
                'reviews_count': random.randint(0, 5000),
                'in_stock': random.randint(0, 1000),
                'supplier_id': random.randint(1, 100),
                'weight': round(random.uniform(0.1, 50.0), 2)
            })
        
        # Load data into PostgreSQL
        self.load_data_postgresql('customers', customers_data)
        self.load_data_postgresql('orders', orders_data)
        self.load_data_postgresql('products', products_data)
        
        # Load data into DuckDB
        self.load_data_duckdb('customers', customers_data)
        self.load_data_duckdb('orders', orders_data)
        self.load_data_duckdb('products', products_data)
        
        logger.info(f"✓ Created tables: customers (50K), orders (200K), products (10K)")
        
    def load_data_postgresql(self, table_name: str, data: List[Dict]):
        """Load data into PostgreSQL table"""
        if not data:
            return
            
        first_row = data[0]
        columns = []
        
        for key, value in first_row.items():
            if isinstance(value, int):
                col_type = 'INTEGER'
            elif isinstance(value, float):
                col_type = 'DECIMAL(10,2)'
            elif isinstance(value, bool):
                col_type = 'BOOLEAN'
            else:
                col_type = 'VARCHAR(255)'
            columns.append(f"{key} {col_type}")
        
        create_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(columns)})"
        
        with self.pg_conn.cursor() as cursor:
            cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
            cursor.execute(create_sql)
            
            # Create indexes for better performance
            cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_id ON {table_name}(id)")
            
            if table_name == 'orders':
                cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_orders_customer_id ON {table_name}(customer_id)")
                cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_orders_product_id ON {table_name}(product_id)")
                cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_orders_category ON {table_name}(product_category)")
            elif table_name == 'customers':
                cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_customers_category ON {table_name}(category)")
                cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_customers_region ON {table_name}(region)")
            
            # Bulk insert data
            keys = list(first_row.keys())
            placeholders = ', '.join(['%s'] * len(keys))
            insert_sql = f"INSERT INTO {table_name} ({', '.join(keys)}) VALUES ({placeholders})"
            
            # Insert in batches
            batch_size = 1000
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                values = [[row[key] for key in keys] for row in batch]
                cursor.executemany(insert_sql, values)
            
            # Update table statistics
            cursor.execute(f"ANALYZE {table_name}")
    
    def load_data_duckdb(self, table_name: str, data: List[Dict]):
        """Load data into DuckDB table"""
        df = pd.DataFrame(data)
        self.duck_conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        self.duck_conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
        
        # Create indexes in DuckDB
        if table_name == 'orders':
            try:
                self.duck_conn.execute(f"CREATE INDEX idx_orders_customer_id ON {table_name}(customer_id)")
                self.duck_conn.execute(f"CREATE INDEX idx_orders_product_id ON {table_name}(product_id)")
            except:
                pass  # DuckDB might not support all index types
    
    def generate_realistic_queries(self) -> List[Dict]:
        """Generate realistic OLTP and OLAP queries"""
        queries = []
        
        # OLTP queries (simple, fast operations)
        oltp_templates = [
            "SELECT * FROM customers WHERE id = {id}",
            "SELECT name, email, category FROM customers WHERE category = '{category}' LIMIT 20",
            "SELECT * FROM orders WHERE customer_id = {customer_id} ORDER BY order_date DESC LIMIT 10",
            "SELECT COUNT(*) FROM orders WHERE status = 'completed' AND order_date >= '2024-01-01'",
            "SELECT * FROM products WHERE category = '{category}' AND in_stock > 0 ORDER BY rating DESC LIMIT 25",
            "SELECT AVG(price) FROM products WHERE category = '{category}'",
            "SELECT * FROM orders WHERE amount > {amount} AND status = 'completed' LIMIT 30",
            "SELECT customer_id, COUNT(*) FROM orders WHERE order_date >= '2024-06-01' GROUP BY customer_id LIMIT 50",
            "UPDATE customers SET is_active = true WHERE id = {id}",
            "INSERT INTO orders (customer_id, product_id, amount, quantity, order_date, status) VALUES ({customer_id}, {product_id}, {amount}, {quantity}, '2024-09-01', 'pending')",
        ]
        
        # OLAP queries (complex analytics)
        olap_templates = [
            "SELECT category, COUNT(*) as customer_count, AVG(value) as avg_value FROM customers GROUP BY category ORDER BY customer_count DESC",
            "SELECT product_category, DATE_TRUNC('month', order_date::date) as month, SUM(amount) as revenue FROM orders GROUP BY product_category, month ORDER BY month, revenue DESC",
            "SELECT c.region, COUNT(DISTINCT o.customer_id) as unique_customers, SUM(o.amount) as total_revenue FROM customers c JOIN orders o ON c.id = o.customer_id WHERE o.status = 'completed' GROUP BY c.region",
            "SELECT product_category, AVG(amount) as avg_order, PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amount) as median_order FROM orders WHERE status = 'completed' GROUP BY product_category ORDER BY avg_order DESC",
            "SELECT c.category, c.region, COUNT(o.id) as order_count, AVG(o.amount) as avg_amount FROM customers c LEFT JOIN orders o ON c.id = o.customer_id GROUP BY c.category, c.region HAVING COUNT(o.id) > 10 ORDER BY order_count DESC",
            "SELECT product_id, COUNT(*) as order_frequency, SUM(amount) as total_revenue FROM orders WHERE order_date >= '2024-01-01' GROUP BY product_id ORDER BY order_frequency DESC LIMIT 100",
            "WITH monthly_sales AS (SELECT DATE_TRUNC('month', order_date::date) as month, SUM(amount) as sales FROM orders GROUP BY month) SELECT month, sales, LAG(sales) OVER (ORDER BY month) as prev_month_sales FROM monthly_sales ORDER BY month",
            "SELECT c.age / 10 * 10 as age_group, AVG(o.amount) as avg_spend, COUNT(o.id) as order_count FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY age_group ORDER BY age_group",
            "SELECT p.category, p.price / 100 * 100 as price_range, COUNT(o.id) as sales_count FROM products p JOIN orders o ON p.id = o.product_id GROUP BY p.category, price_range ORDER BY p.category, price_range",
        ]
        
        # Generate OLTP queries
        for i in range(1500):  # More OLTP queries
            template = random.choice(oltp_templates)
            
            # Skip INSERT/UPDATE for read-only comparison
            if 'INSERT' in template or 'UPDATE' in template:
                template = "SELECT COUNT(*) FROM customers WHERE category = '{category}'"
            
            query = template.format(
                id=random.randint(1, 50000),
                customer_id=random.randint(1, 50000),
                product_id=random.randint(1, 10000),
                category=random.choice(['premium', 'standard', 'basic', 'electronics', 'clothing', 'books']),
                amount=random.randint(100, 1000),
                quantity=random.randint(1, 5)
            )
            
            queries.append({
                'query_id': f'oltp_{i}',
                'query_type': 'oltp',
                'query': query,
                'complexity': 'simple'
            })
        
        # Generate OLAP queries
        for i in range(500):  # Fewer but more complex OLAP queries
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
    
    def execute_query_postgresql(self, query: str) -> Dict:
        """Execute query on PostgreSQL and measure performance"""
        start_time = time.perf_counter()
        
        try:
            with self.pg_conn.cursor() as cursor:
                # Execute query with timing
                query_start = time.perf_counter()
                cursor.execute(query)
                
                # Fetch results to ensure complete execution
                if cursor.description:  # SELECT query
                    results = cursor.fetchall()
                    row_count = len(results)
                else:  # Non-SELECT query
                    row_count = cursor.rowcount
                    
                query_end = time.perf_counter()
                
                execution_time = query_end - query_start
                total_time = time.perf_counter() - start_time
                
                # Extract simple features
                features = self.extract_query_features(query, 'postgresql')
                
                return {
                    'execution_time': execution_time,
                    'total_time': total_time,
                    'row_count': row_count,
                    'features': features,
                    'success': True,
                    'error': None
                }
                
        except Exception as e:
            return {
                'execution_time': float('inf'),
                'total_time': time.perf_counter() - start_time,
                'row_count': 0,
                'features': {},
                'success': False,
                'error': str(e)
            }
    
    def execute_query_duckdb(self, query: str) -> Dict:
        """Execute query on DuckDB and measure performance"""
        start_time = time.perf_counter()
        
        try:
            # Execute query
            query_start = time.perf_counter()
            results = self.duck_conn.execute(query).fetchall()
            query_end = time.perf_counter()
            
            execution_time = query_end - query_start
            total_time = time.perf_counter() - start_time
            
            # Extract features
            features = self.extract_query_features(query, 'duckdb')
            
            return {
                'execution_time': execution_time,
                'total_time': total_time,
                'row_count': len(results),
                'features': features,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            return {
                'execution_time': float('inf'),
                'total_time': time.perf_counter() - start_time,
                'row_count': 0,
                'features': {},
                'success': False,
                'error': str(e)
            }
    
    def extract_query_features(self, query: str, engine: str) -> Dict:
        """Extract features from query"""
        features = {}
        query_upper = query.upper()
        
        # Basic query features
        features['query_length'] = len(query)
        features['has_where'] = 1 if 'WHERE' in query_upper else 0
        features['has_group_by'] = 1 if 'GROUP BY' in query_upper else 0
        features['has_order_by'] = 1 if 'ORDER BY' in query_upper else 0
        features['has_join'] = 1 if 'JOIN' in query_upper else 0
        features['has_aggregation'] = 1 if any(func in query_upper for func in ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX']) else 0
        features['has_limit'] = 1 if 'LIMIT' in query_upper else 0
        features['has_having'] = 1 if 'HAVING' in query_upper else 0
        features['has_distinct'] = 1 if 'DISTINCT' in query_upper else 0
        features['has_subquery'] = 1 if '(' in query and 'SELECT' in query_upper else 0
        
        # Table access patterns
        tables = ['customers', 'orders', 'products']
        for table in tables:
            features[f'accesses_{table}'] = 1 if table.upper() in query_upper else 0
        
        # Query complexity estimation
        complexity_score = (
            features['has_join'] * 3 +
            features['has_group_by'] * 2 +
            features['has_aggregation'] * 2 +
            features['has_subquery'] * 3 +
            features['has_having'] * 2 +
            features['has_order_by'] * 1
        )
        features['complexity_score'] = complexity_score
        
        # Selectivity estimation
        if '=' in query:
            features['selectivity'] = 0.001  # Very selective
        elif 'WHERE' in query_upper:
            features['selectivity'] = 0.1   # Moderately selective
        else:
            features['selectivity'] = 1.0   # Full scan
        
        # Engine-specific features
        features[f'engine_{engine}'] = 1
        
        return features
    
    def collect_execution_data(self, queries: List[Dict]) -> pd.DataFrame:
        """Execute all queries and collect performance data"""
        logger.info(f"Executing {len(queries)} real queries on PostgreSQL and DuckDB...")
        
        results = []
        failed_queries = 0
        
        for i, query_info in enumerate(queries):
            if i % 100 == 0:
                logger.info(f"Progress: {i}/{len(queries)} queries completed")
            
            query = query_info['query']
            
            # Execute on PostgreSQL
            pg_result = self.execute_query_postgresql(query)
            
            # Execute on DuckDB
            duck_result = self.execute_query_duckdb(query)
            
            # Combine results if both succeeded
            if pg_result['success'] and duck_result['success']:
                result = {
                    'query_id': query_info['query_id'],
                    'query_type': query_info['query_type'],
                    'complexity': query_info['complexity'],
                    'query': query[:200],  # Truncate for storage
                    'query_hash': hashlib.md5(query.encode()).hexdigest()[:12],
                    
                    # PostgreSQL metrics
                    'postgres_execution_time': pg_result['execution_time'],
                    'postgres_total_time': pg_result['total_time'],
                    'postgres_row_count': pg_result['row_count'],
                    
                    # DuckDB metrics
                    'duckdb_execution_time': duck_result['execution_time'],
                    'duckdb_total_time': duck_result['total_time'],
                    'duckdb_row_count': duck_result['row_count'],
                    
                    # Derived metrics
                    'time_gap': pg_result['execution_time'] - duck_result['execution_time'],
                    'log_time_gap': np.log(max(pg_result['execution_time'], 0.000001)) - np.log(max(duck_result['execution_time'], 0.000001)),
                    'speedup_ratio': duck_result['execution_time'] / max(pg_result['execution_time'], 0.000001),
                    'best_engine': 'postgres' if pg_result['execution_time'] < duck_result['execution_time'] else 'duckdb',
                }
                
                # Add PostgreSQL features
                for key, value in pg_result['features'].items():
                    result[f'pg_{key}'] = value
                
                # Add DuckDB features  
                for key, value in duck_result['features'].items():
                    result[f'duck_{key}'] = value
                
                results.append(result)
            
            else:
                failed_queries += 1
                if failed_queries % 50 == 0:
                    logger.warning(f"Failed queries so far: {failed_queries}")
        
        logger.info(f"Successfully collected data for {len(results)} queries")
        logger.info(f"Failed queries: {failed_queries}")
        
        return pd.DataFrame(results)
    
    def cleanup(self):
        """Clean up database connections"""
        if self.pg_conn:
            self.pg_conn.close()
        if self.duck_conn:
            self.duck_conn.close()

def main():
    """Main PostgreSQL + DuckDB data collection pipeline"""
    print("""
    ╔══════════════════════════════════════════════════════════════════════════════════════╗
    ║                 REAL POSTGRESQL + DUCKDB DATA COLLECTION                            ║
    ║                     Execute Real Queries on Actual Database Systems                 ║
    ╚══════════════════════════════════════════════════════════════════════════════════════╝
    """)
    
    collector = PostgresDuckDBCollector()
    
    try:
        # Phase 1: Setup databases
        logger.info("Phase 1: Setting up PostgreSQL and DuckDB...")
        if not collector.setup_postgresql():
            logger.error("Failed to setup PostgreSQL")
            return
        if not collector.setup_duckdb():
            logger.error("Failed to setup DuckDB")
            return
        
        # Phase 2: Create test data
        logger.info("Phase 2: Creating comprehensive test data...")
        collector.create_test_data()
        
        # Phase 3: Generate realistic queries
        logger.info("Phase 3: Generating realistic query workload...")
        queries = collector.generate_realistic_queries()
        logger.info(f"Generated {len(queries)} queries:")
        logger.info(f"  - OLTP queries: {len([q for q in queries if q['query_type'] == 'oltp'])}")
        logger.info(f"  - OLAP queries: {len([q for q in queries if q['query_type'] == 'olap'])}")
        
        # Phase 4: Execute queries and collect data
        logger.info("Phase 4: Executing queries on PostgreSQL and DuckDB...")
        df = collector.collect_execution_data(queries)
        
        # Phase 5: Save results
        data_file = collector.data_dir / 'postgres_duckdb_execution_data.csv'
        df.to_csv(data_file, index=False)
        
        logger.info(f"\n" + "="*80)
        logger.info("🎉 REAL POSTGRESQL + DUCKDB DATA COLLECTION COMPLETED!")
        logger.info("="*80)
        logger.info(f"📊 Data saved to: {data_file}")
        logger.info(f"📈 Total queries executed: {len(df)}")
        logger.info(f"📋 Features collected: {len([col for col in df.columns if col.startswith('pg_') or col.startswith('duck_')])}")
        
        # Print summary statistics
        if len(df) > 0:
            logger.info(f"\n📊 REAL EXECUTION TIME SUMMARY:")
            logger.info(f"PostgreSQL - Mean: {df['postgres_execution_time'].mean():.6f}s, Median: {df['postgres_execution_time'].median():.6f}s")
            logger.info(f"DuckDB - Mean: {df['duckdb_execution_time'].mean():.6f}s, Median: {df['duckdb_execution_time'].median():.6f}s")
            logger.info(f"PostgreSQL faster in: {(df['best_engine'] == 'postgres').mean():.1%} of queries")
            logger.info(f"DuckDB faster in: {(df['best_engine'] == 'duckdb').mean():.1%} of queries")
            
            # Query type breakdown
            oltp_data = df[df['query_type'] == 'oltp']
            olap_data = df[df['query_type'] == 'olap']
            
            if len(oltp_data) > 0:
                logger.info(f"\n📋 OLTP QUERIES ({len(oltp_data)} queries):")
                logger.info(f"  PostgreSQL faster: {(oltp_data['best_engine'] == 'postgres').mean():.1%}")
                logger.info(f"  DuckDB faster: {(oltp_data['best_engine'] == 'duckdb').mean():.1%}")
                
            if len(olap_data) > 0:
                logger.info(f"\n📋 OLAP QUERIES ({len(olap_data)} queries):")
                logger.info(f"  PostgreSQL faster: {(olap_data['best_engine'] == 'postgres').mean():.1%}")
                logger.info(f"  DuckDB faster: {(olap_data['best_engine'] == 'duckdb').mean():.1%}")
        
        return data_file
        
    except Exception as e:
        logger.error(f"Real data collection failed: {e}")
        raise
    finally:
        collector.cleanup()

if __name__ == "__main__":
    main()