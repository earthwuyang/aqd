#!/usr/bin/env python3
"""
Dual-Engine Training Data Collection for PostgreSQL + DuckDB AQD
Executes ZSCE-generated queries on both engines and collects performance data
"""

import os
import json
import time
import psycopg2
import duckdb
import logging
from typing import Dict, List, Any, Tuple
from datetime import datetime
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Database connection configuration
POSTGRESQL_CONFIG = {
    'host': 'localhost', 
    'port': 5433,  # Using local PostgreSQL installation
    'database': 'postgres',  # Use main database with imported schemas
    'user': 'wuy',
    'password': ''
}

DUCKDB_PATH = '/home/wuy/DB/pg_duckdb_postgres/benchmark_datasets.db'

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/dual_engine_collection.log'),
        logging.StreamHandler()
    ]
)

class DualEngineCollector:
    """Collects training data by executing queries on both PostgreSQL and DuckDB"""
    
    def __init__(self):
        self.pg_conn = None
        self.duck_conn = None
        self.collected_data = []
        self.lock = threading.Lock()
        
    def connect_postgresql(self):
        """Connect to PostgreSQL"""
        try:
            self.pg_conn = psycopg2.connect(**POSTGRESQL_CONFIG)
            self.pg_conn.autocommit = True
            logging.info("Connected to PostgreSQL")
            return True
        except Exception as e:
            logging.error(f"PostgreSQL connection failed: {e}")
            return False
    
    def connect_duckdb(self):
        """Connect to DuckDB"""
        try:
            os.makedirs(os.path.dirname(DUCKDB_PATH), exist_ok=True)
            self.duck_conn = duckdb.connect(DUCKDB_PATH)
            logging.info("Connected to DuckDB")
            return True
        except Exception as e:
            logging.error(f"DuckDB connection failed: {e}")
            return False
    
    def setup_test_data(self):
        """Setup test data in both databases"""
        logging.info("Setting up test data in both databases...")
        
        # Setup PostgreSQL data
        if self.pg_conn:
            try:
                cursor = self.pg_conn.cursor()
                
                # Create tables
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        user_id INTEGER PRIMARY KEY,
                        name VARCHAR(100),
                        email VARCHAR(100),
                        created_at TIMESTAMP,
                        status VARCHAR(20),
                        age INTEGER,
                        city VARCHAR(50)
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS products (
                        product_id INTEGER PRIMARY KEY,
                        name VARCHAR(100),
                        category VARCHAR(50),
                        price DECIMAL(10,2),
                        stock INTEGER,
                        created_at TIMESTAMP,
                        description TEXT
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS orders (
                        order_id INTEGER PRIMARY KEY,
                        user_id INTEGER,
                        product_id INTEGER,
                        quantity INTEGER,
                        price DECIMAL(10,2),
                        order_date TIMESTAMP,
                        status VARCHAR(20),
                        total_amount DECIMAL(10,2),
                        FOREIGN KEY (user_id) REFERENCES users(user_id),
                        FOREIGN KEY (product_id) REFERENCES products(product_id)
                    )
                """)
                
                # Check if data exists
                cursor.execute("SELECT COUNT(*) FROM users")
                user_count = cursor.fetchone()[0]
                
                if user_count == 0:
                    # Insert test data
                    logging.info("Inserting test data into PostgreSQL...")
                    
                    # Insert users
                    users_data = []
                    for i in range(1, 10001):
                        users_data.append((i, f'User_{i}', f'user_{i}@example.com', 
                                         datetime.now(), np.random.choice(['active', 'inactive', 'premium']),
                                         np.random.randint(18, 80), 
                                         np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'])))
                    
                    cursor.executemany("""
                        INSERT INTO users (user_id, name, email, created_at, status, age, city) 
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, users_data)
                    
                    # Insert products
                    products_data = []
                    for i in range(1, 1001):
                        products_data.append((i, f'Product_{i}', 
                                           np.random.choice(['Electronics', 'Clothing', 'Books', 'Home', 'Sports']),
                                           np.random.uniform(10, 1000), np.random.randint(0, 100),
                                           datetime.now(), f'Description for product {i}'))
                    
                    cursor.executemany("""
                        INSERT INTO products (product_id, name, category, price, stock, created_at, description)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, products_data)
                    
                    # Insert orders
                    orders_data = []
                    for i in range(1, 50001):
                        user_id = np.random.randint(1, 10001)
                        product_id = np.random.randint(1, 1001)
                        quantity = np.random.randint(1, 5)
                        price = np.random.uniform(10, 1000)
                        orders_data.append((i, user_id, product_id, quantity, price,
                                          datetime.now(), np.random.choice(['pending', 'completed', 'cancelled']),
                                          price * quantity))
                    
                    cursor.executemany("""
                        INSERT INTO orders (order_id, user_id, product_id, quantity, price, order_date, status, total_amount)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, orders_data)
                    
                    logging.info("PostgreSQL test data inserted successfully")
                else:
                    logging.info(f"PostgreSQL already has {user_count} users, skipping data insertion")
                    
            except Exception as e:
                logging.error(f"Error setting up PostgreSQL data: {e}")
        
        # Setup DuckDB data (copy from PostgreSQL)
        if self.duck_conn and self.pg_conn:
            try:
                # Create tables in DuckDB
                self.duck_conn.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        user_id INTEGER PRIMARY KEY,
                        name VARCHAR,
                        email VARCHAR,
                        created_at TIMESTAMP,
                        status VARCHAR,
                        age INTEGER,
                        city VARCHAR
                    )
                """)
                
                self.duck_conn.execute("""
                    CREATE TABLE IF NOT EXISTS products (
                        product_id INTEGER PRIMARY KEY,
                        name VARCHAR,
                        category VARCHAR,
                        price DECIMAL(10,2),
                        stock INTEGER,
                        created_at TIMESTAMP,
                        description TEXT
                    )
                """)
                
                self.duck_conn.execute("""
                    CREATE TABLE IF NOT EXISTS orders (
                        order_id INTEGER PRIMARY KEY,
                        user_id INTEGER,
                        product_id INTEGER,
                        quantity INTEGER,
                        price DECIMAL(10,2),
                        order_date TIMESTAMP,
                        status VARCHAR,
                        total_amount DECIMAL(10,2)
                    )
                """)
                
                # Check if data exists
                user_count = self.duck_conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
                
                if user_count == 0:
                    logging.info("Copying data from PostgreSQL to DuckDB...")
                    
                    # Copy users
                    pg_cursor = self.pg_conn.cursor()
                    pg_cursor.execute("SELECT * FROM users")
                    users = pg_cursor.fetchall()
                    
                    for user in users:
                        self.duck_conn.execute("""
                            INSERT INTO users VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, user)
                    
                    # Copy products
                    pg_cursor.execute("SELECT * FROM products") 
                    products = pg_cursor.fetchall()
                    
                    for product in products:
                        self.duck_conn.execute("""
                            INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, product)
                    
                    # Copy orders
                    pg_cursor.execute("SELECT * FROM orders")
                    orders = pg_cursor.fetchall()
                    
                    for order in orders:
                        self.duck_conn.execute("""
                            INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, order)
                    
                    logging.info("DuckDB data copy completed successfully")
                else:
                    logging.info(f"DuckDB already has {user_count} users, skipping data copy")
                    
            except Exception as e:
                logging.error(f"Error setting up DuckDB data: {e}")
    
    def execute_query_postgresql(self, query: str, timeout: int = 30) -> Tuple[float, bool, str]:
        """Execute query on PostgreSQL and measure performance"""
        try:
            cursor = self.pg_conn.cursor()
            start_time = time.perf_counter()
            cursor.execute(query)
            result = cursor.fetchall()  # Ensure full execution
            end_time = time.perf_counter()
            
            execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
            return execution_time, True, ""
            
        except Exception as e:
            return 0.0, False, str(e)
    
    def execute_query_duckdb(self, query: str, timeout: int = 30) -> Tuple[float, bool, str]:
        """Execute query on DuckDB and measure performance"""
        try:
            start_time = time.perf_counter()
            result = self.duck_conn.execute(query).fetchall()
            end_time = time.perf_counter()
            
            execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
            return execution_time, True, ""
            
        except Exception as e:
            return 0.0, False, str(e)
    
    def extract_query_features(self, query: str) -> Dict[str, Any]:
        """Extract features from SQL query for ML training"""
        query_upper = query.upper()
        
        features = {
            'query_length': len(query),
            'num_select': query_upper.count('SELECT'),
            'num_join': query_upper.count('JOIN'),
            'num_where': query_upper.count('WHERE'),
            'num_group_by': query_upper.count('GROUP BY'),
            'num_order_by': query_upper.count('ORDER BY'),
            'num_having': query_upper.count('HAVING'),
            'num_distinct': query_upper.count('DISTINCT'),
            'num_union': query_upper.count('UNION'),
            'num_subquery': query.count('(SELECT'),
            'num_with': query_upper.count('WITH'),
            'num_case': query_upper.count('CASE'),
            'num_like': query_upper.count('LIKE'),
            'num_in': query_upper.count(' IN '),
            'num_between': query_upper.count('BETWEEN'),
            'num_aggregates': (query_upper.count('SUM(') + query_upper.count('COUNT(') + 
                              query_upper.count('AVG(') + query_upper.count('MIN(') + query_upper.count('MAX(')),
            'has_limit': 1 if 'LIMIT' in query_upper else 0,
            'has_offset': 1 if 'OFFSET' in query_upper else 0,
            'num_tables': max(1, query_upper.count('FROM') + query_upper.count('JOIN')),
            'estimated_complexity': len(query.split()) / 10.0  # Simple complexity estimate
        }
        
        return features
    
    def determine_optimal_engine(self, pg_time: float, duck_time: float, pg_success: bool, duck_success: bool) -> str:
        """Determine which engine performed better"""
        if not pg_success and not duck_success:
            return 'postgresql'  # Default fallback
        elif not pg_success:
            return 'duckdb'
        elif not duck_success:
            return 'postgresql'
        else:
            # Choose faster engine with 5% tolerance to avoid noise
            if duck_time < pg_time * 0.95:
                return 'duckdb'
            else:
                return 'postgresql'
    
    def collect_single_query(self, query_id: str, query: str, query_type: str) -> Dict[str, Any]:
        """Collect training data for a single query"""
        # Execute on PostgreSQL
        pg_time, pg_success, pg_error = self.execute_query_postgresql(query)
        
        # Execute on DuckDB  
        duck_time, duck_success, duck_error = self.execute_query_duckdb(query)
        
        # Extract features
        features = self.extract_query_features(query)
        
        # Determine optimal engine
        optimal_engine = self.determine_optimal_engine(pg_time, duck_time, pg_success, duck_success)
        
        record = {
            'query_id': query_id,
            'query_text': query,
            'query_type': query_type,
            'pg_execution_time': pg_time if pg_success else None,
            'duck_execution_time': duck_time if duck_success else None,
            'pg_success': pg_success,
            'duck_success': duck_success,
            'pg_error': pg_error if pg_error else None,
            'duck_error': duck_error if duck_error else None,
            'optimal_engine': optimal_engine,
            'features': features,
            'collected_at': datetime.now().isoformat()
        }
        
        return record
    
    def collect_from_query_files(self, ap_file: str, tp_file: str, max_queries_per_file: int = 5000) -> List[Dict[str, Any]]:
        """Collect training data from ZSCE-generated query files"""
        logging.info(f"Collecting data from AP file: {ap_file}")
        logging.info(f"Collecting data from TP file: {tp_file}")
        
        collected_data = []
        
        # Process AP queries
        if os.path.exists(ap_file):
            with open(ap_file, 'r') as f:
                ap_queries = [line.strip() for line in f if line.strip()]
            
            logging.info(f"Processing {min(len(ap_queries), max_queries_per_file)} AP queries...")
            
            for i, query in enumerate(ap_queries[:max_queries_per_file]):
                if i % 100 == 0:
                    logging.info(f"Processed {i} AP queries")
                
                query_id = f"ap_{i+1}"
                record = self.collect_single_query(query_id, query, 'olap')
                collected_data.append(record)
        
        # Process TP queries
        if os.path.exists(tp_file):
            with open(tp_file, 'r') as f:
                tp_queries = [line.strip() for line in f if line.strip()]
            
            logging.info(f"Processing {min(len(tp_queries), max_queries_per_file)} TP queries...")
            
            for i, query in enumerate(tp_queries[:max_queries_per_file]):
                if i % 100 == 0:
                    logging.info(f"Processed {i} TP queries")
                
                query_id = f"tp_{i+1}"
                record = self.collect_single_query(query_id, query, 'oltp')
                collected_data.append(record)
        
        return collected_data
    
    def save_training_data(self, data: List[Dict[str, Any]], output_file: str):
        """Save collected training data to JSON file"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        training_data = {
            'collection_metadata': {
                'collected_at': datetime.now().isoformat(),
                'total_queries': len(data),
                'postgresql_config': POSTGRESQL_CONFIG,
                'duckdb_path': DUCKDB_PATH,
                'ap_queries': len([d for d in data if d['query_type'] == 'olap']),
                'tp_queries': len([d for d in data if d['query_type'] == 'oltp']),
                'successful_pg': len([d for d in data if d['pg_success']]),
                'successful_duck': len([d for d in data if d['duck_success']]),
                'optimal_postgresql': len([d for d in data if d['optimal_engine'] == 'postgresql']),
                'optimal_duckdb': len([d for d in data if d['optimal_engine'] == 'duckdb'])
            },
            'data': data
        }
        
        with open(output_file, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        logging.info(f"Training data saved to: {output_file}")
        logging.info(f"Total queries: {len(data)}")
        logging.info(f"PostgreSQL optimal: {training_data['collection_metadata']['optimal_postgresql']}")
        logging.info(f"DuckDB optimal: {training_data['collection_metadata']['optimal_duckdb']}")
    
    def close_connections(self):
        """Close database connections"""
        if self.pg_conn:
            self.pg_conn.close()
        if self.duck_conn:
            self.duck_conn.close()

def main():
    """Main execution function"""
    logging.info("=== Dual-Engine Training Data Collection ===")
    
    # Create output directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data/training_data', exist_ok=True)
    
    collector = DualEngineCollector()
    
    # Connect to databases
    if not collector.connect_postgresql():
        logging.error("Failed to connect to PostgreSQL")
        return 1
    
    if not collector.connect_duckdb():
        logging.error("Failed to connect to DuckDB")
        return 1
    
    try:
        # Setup test data
        collector.setup_test_data()
        
        # Define query file paths
        query_dir = 'data/zsce_queries/aqd_benchmark'
        ap_file = f'{query_dir}/workload_10k_ap_queries.sql'
        tp_file = f'{query_dir}/workload_10k_tp_queries.sql'
        
        # Check if query files exist
        if not os.path.exists(ap_file) or not os.path.exists(tp_file):
            logging.error(f"Query files not found. Please run ZSCE query generation first.")
            logging.error(f"Expected files: {ap_file}, {tp_file}")
            return 1
        
        # Collect training data
        training_data = collector.collect_from_query_files(ap_file, tp_file, max_queries_per_file=1000)
        
        # Save training data
        output_file = 'data/training_data/dual_engine_training_data.json'
        collector.save_training_data(training_data, output_file)
        
        print(f"\n{'='*80}")
        print("🚀 DUAL-ENGINE DATA COLLECTION COMPLETE")
        print(f"{'='*80}")
        print(f"📊 Total Queries Processed: {len(training_data):,}")
        print(f"🐘 PostgreSQL Optimal: {len([d for d in training_data if d['optimal_engine'] == 'postgresql']):,}")
        print(f"🦆 DuckDB Optimal: {len([d for d in training_data if d['optimal_engine'] == 'duckdb']):,}")
        print(f"💾 Training Data: {output_file}")
        print(f"{'='*80}")
        
    except Exception as e:
        logging.error(f"Data collection failed: {e}")
        return 1
    
    finally:
        collector.close_connections()
    
    return 0

if __name__ == "__main__":
    exit(main())