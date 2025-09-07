#!/usr/bin/env python3
"""
Simplified Real Training Data Collector
Uses SQLite (as OLTP engine) and DuckDB (as OLAP engine) for actual dual execution
"""

import os
import sys
import time
import json
import sqlite3
import duckdb
import logging
import hashlib
import traceback
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import threading


@dataclass
class QueryFeatures:
    """Query features for training"""
    query_length: int
    num_tables: int
    num_joins: int
    num_filters: int
    num_aggregates: int
    num_sorts: int
    num_limits: int
    has_groupby: bool
    has_window: bool
    has_subquery: bool
    has_distinct: bool
    estimated_complexity: float
    
    def to_dict(self):
        return asdict(self)


@dataclass 
class TrainingRecord:
    """Real training record from dual execution"""
    query_id: str
    query_hash: str
    query_text: str
    query_type: str
    
    # SQLite execution (OLTP engine)
    sqlite_execution_time: float
    sqlite_success: bool
    sqlite_error: Optional[str]
    sqlite_result_rows: int
    
    # DuckDB execution (OLAP engine)
    duckdb_execution_time: float
    duckdb_success: bool
    duckdb_error: Optional[str]
    duckdb_result_rows: int
    
    # Ground truth
    optimal_engine: str
    performance_gap: float
    
    # Features
    features: QueryFeatures
    
    # Metadata
    timestamp: str
    collection_round: int
    
    def to_dict(self):
        result = asdict(self)
        result['features'] = self.features.to_dict()
        return result


class RealTrainingDataCollector:
    """
    Real training data collector using SQLite (OLTP) and DuckDB (OLAP)
    """
    
    def __init__(self):
        self.sqlite_conn = None
        self.duckdb_conn = None
        
        # Data storage
        self.training_records: List[TrainingRecord] = []
        self.collection_round = 0
        
        # Statistics
        self.total_collected = 0
        self.sqlite_wins = 0
        self.duckdb_wins = 0
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/real_data_collection.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def connect_databases(self):
        """Connect to SQLite and DuckDB"""
        try:
            # SQLite connection (OLTP engine simulation)
            self.sqlite_conn = sqlite3.connect('data/oltp_database.db')
            self.sqlite_conn.execute("PRAGMA journal_mode=WAL")
            self.sqlite_conn.execute("PRAGMA synchronous=NORMAL")
            
            # DuckDB connection (OLAP engine)
            self.duckdb_conn = duckdb.connect('data/olap_database.duckdb')
            
            self.logger.info("✓ Connected to SQLite (OLTP) and DuckDB (OLAP)")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to databases: {e}")
            raise
    
    def create_test_data(self):
        """Create substantial test data for realistic benchmarks"""
        self.logger.info("Creating test tables and data...")
        
        # SQLite tables (optimized for OLTP)
        with self.sqlite_conn:
            # Users table
            self.sqlite_conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    name TEXT,
                    email TEXT,
                    age INTEGER,
                    city TEXT,
                    country TEXT,
                    registration_date TEXT,
                    last_login TEXT,
                    status TEXT,
                    balance REAL
                );
            """)
            
            # Orders table
            self.sqlite_conn.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    order_id INTEGER PRIMARY KEY,
                    customer_id INTEGER,
                    order_date TEXT,
                    total_amount REAL,
                    status TEXT,
                    region TEXT,
                    shipping_cost REAL,
                    FOREIGN KEY (customer_id) REFERENCES users(user_id)
                );
            """)
            
            # Products table
            self.sqlite_conn.execute("""
                CREATE TABLE IF NOT EXISTS products (
                    product_id INTEGER PRIMARY KEY,
                    name TEXT,
                    category TEXT,
                    price REAL,
                    stock_quantity INTEGER,
                    supplier_id INTEGER
                );
            """)
            
            # Order items table
            self.sqlite_conn.execute("""
                CREATE TABLE IF NOT EXISTS order_items (
                    item_id INTEGER PRIMARY KEY,
                    order_id INTEGER,
                    product_id INTEGER,
                    quantity INTEGER,
                    unit_price REAL,
                    FOREIGN KEY (order_id) REFERENCES orders(order_id),
                    FOREIGN KEY (product_id) REFERENCES products(product_id)
                );
            """)
            
            # Check if data exists
            cursor = self.sqlite_conn.execute("SELECT COUNT(*) FROM users")
            user_count = cursor.fetchone()[0]
            
            if user_count == 0:
                self.logger.info("Inserting test data...")
                
                # Insert users (100K users)
                users_data = []
                for i in range(100000):
                    users_data.append((
                        i + 1,
                        f"User{i}",
                        f"user{i}@example.com",
                        np.random.randint(18, 80),
                        f"City{i % 100}",
                        f"Country{i % 20}",
                        f"2024-{np.random.randint(1,13):02d}-{np.random.randint(1,29):02d}",
                        f"2024-{np.random.randint(1,13):02d}-{np.random.randint(1,29):02d}",
                        np.random.choice(['active', 'inactive', 'suspended']),
                        round(np.random.uniform(0, 10000), 2)
                    ))
                
                self.sqlite_conn.executemany(
                    "INSERT INTO users VALUES (?,?,?,?,?,?,?,?,?,?)", 
                    users_data
                )
                self.logger.info("Inserted 100K users")
                
                # Insert products (50K products)
                products_data = []
                for i in range(50000):
                    products_data.append((
                        i + 1,
                        f"Product{i}",
                        f"Category{i % 100}",
                        round(np.random.uniform(5, 1000), 2),
                        np.random.randint(0, 1000),
                        i % 1000
                    ))
                
                self.sqlite_conn.executemany(
                    "INSERT INTO products VALUES (?,?,?,?,?,?)",
                    products_data
                )
                self.logger.info("Inserted 50K products")
                
                # Insert orders (500K orders)
                orders_data = []
                for i in range(500000):
                    orders_data.append((
                        i + 1,
                        (i % 100000) + 1,
                        f"2024-{np.random.randint(1,13):02d}-{np.random.randint(1,29):02d}",
                        round(np.random.uniform(10, 5000), 2),
                        np.random.choice(['completed', 'pending', 'cancelled']),
                        f"Region{i % 10}",
                        round(np.random.uniform(0, 50), 2)
                    ))
                
                self.sqlite_conn.executemany(
                    "INSERT INTO orders VALUES (?,?,?,?,?,?,?)",
                    orders_data
                )
                self.logger.info("Inserted 500K orders")
                
                # Insert order items (1M items)
                items_data = []
                for i in range(1000000):
                    items_data.append((
                        i + 1,
                        (i % 500000) + 1,
                        (i % 50000) + 1,
                        np.random.randint(1, 10),
                        round(np.random.uniform(5, 200), 2)
                    ))
                
                self.sqlite_conn.executemany(
                    "INSERT INTO order_items VALUES (?,?,?,?,?)",
                    items_data
                )
                self.logger.info("Inserted 1M order items")
                
                # Create indexes for SQLite (OLTP optimization)
                self.sqlite_conn.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
                self.sqlite_conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(customer_id)")
                self.sqlite_conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_date ON orders(order_date)")
                self.sqlite_conn.execute("CREATE INDEX IF NOT EXISTS idx_items_order ON order_items(order_id)")
                self.sqlite_conn.execute("CREATE INDEX IF NOT EXISTS idx_items_product ON order_items(product_id)")
                
                self.logger.info("✓ SQLite data and indexes created")
            else:
                self.logger.info(f"✓ SQLite data already exists ({user_count:,} users)")
        
        # Create DuckDB tables (copy data from SQLite)
        self.logger.info("Setting up DuckDB tables...")
        
        # Copy data to DuckDB
        try:
            # Copy users table
            users_df = pd.read_sql("SELECT * FROM users LIMIT 100000", self.sqlite_conn)
            self.duckdb_conn.register('users_df', users_df)
            self.duckdb_conn.execute("CREATE TABLE IF NOT EXISTS users AS SELECT * FROM users_df")
            
            # Copy products table
            products_df = pd.read_sql("SELECT * FROM products LIMIT 50000", self.sqlite_conn)
            self.duckdb_conn.register('products_df', products_df)
            self.duckdb_conn.execute("CREATE TABLE IF NOT EXISTS products AS SELECT * FROM products_df")
            
            # Copy orders table
            orders_df = pd.read_sql("SELECT * FROM orders LIMIT 500000", self.sqlite_conn)
            self.duckdb_conn.register('orders_df', orders_df)
            self.duckdb_conn.execute("CREATE TABLE IF NOT EXISTS orders AS SELECT * FROM orders_df")
            
            # Copy order_items table  
            items_df = pd.read_sql("SELECT * FROM order_items LIMIT 1000000", self.sqlite_conn)
            self.duckdb_conn.register('items_df', items_df)
            self.duckdb_conn.execute("CREATE TABLE IF NOT EXISTS order_items AS SELECT * FROM items_df")
            
            self.logger.info("✓ DuckDB tables created with data")
            
        except Exception as e:
            self.logger.warning(f"Failed to copy some data to DuckDB: {e}")
    
    def generate_realistic_queries(self, count: int = 10000) -> List[Tuple[str, str]]:
        """Generate realistic benchmark queries"""
        queries = []
        
        # OLTP queries (50%) - should favor SQLite
        oltp_count = int(count * 0.5)
        for i in range(oltp_count):
            query_type = np.random.choice([
                'point_select', 'simple_update', 'simple_join', 'small_agg'
            ])
            
            if query_type == 'point_select':
                user_id = np.random.randint(1, 100000)
                query = f"SELECT * FROM users WHERE user_id = {user_id};"
                
            elif query_type == 'simple_update':
                user_id = np.random.randint(1, 100000)
                query = f"UPDATE users SET last_login = '2024-09-07' WHERE user_id = {user_id};"
                
            elif query_type == 'simple_join':
                customer_id = np.random.randint(1, 10000)
                query = f"SELECT u.name, o.total_amount FROM users u JOIN orders o ON u.user_id = o.customer_id WHERE u.user_id = {customer_id} LIMIT 10;"
                
            else:  # small_agg
                region = f"Region{np.random.randint(0, 10)}"
                query = f"SELECT COUNT(*), AVG(total_amount) FROM orders WHERE region = '{region}';"
            
            queries.append((query, 'oltp'))
        
        # OLAP queries (40%) - should favor DuckDB
        olap_count = int(count * 0.4)
        for i in range(olap_count):
            query_type = np.random.choice([
                'complex_agg', 'multi_join', 'window_func', 'large_groupby'
            ])
            
            if query_type == 'complex_agg':
                query = """
                SELECT region, status, 
                       COUNT(*) as order_count,
                       AVG(total_amount) as avg_amount,
                       SUM(total_amount) as total_revenue,
                       MIN(total_amount) as min_amount,
                       MAX(total_amount) as max_amount
                FROM orders 
                GROUP BY region, status
                ORDER BY total_revenue DESC;
                """
                
            elif query_type == 'multi_join':
                query = """
                SELECT p.category, COUNT(DISTINCT o.customer_id) as unique_customers,
                       SUM(oi.quantity * oi.unit_price) as revenue
                FROM products p 
                JOIN order_items oi ON p.product_id = oi.product_id
                JOIN orders o ON oi.order_id = o.order_id
                WHERE o.status = 'completed'
                GROUP BY p.category
                ORDER BY revenue DESC
                LIMIT 20;
                """
                
            elif query_type == 'window_func':
                query = """
                SELECT customer_id, order_date, total_amount,
                       ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date) as order_num,
                       AVG(total_amount) OVER (PARTITION BY customer_id) as customer_avg
                FROM orders 
                WHERE customer_id <= 1000
                ORDER BY customer_id, order_date;
                """
                
            else:  # large_groupby
                query = """
                SELECT u.country, u.city, 
                       COUNT(*) as user_count,
                       AVG(u.balance) as avg_balance,
                       SUM(o.total_amount) as total_spent
                FROM users u
                LEFT JOIN orders o ON u.user_id = o.customer_id
                GROUP BY u.country, u.city
                HAVING user_count > 5
                ORDER BY total_spent DESC
                LIMIT 50;
                """
            
            queries.append((query, 'olap'))
        
        # Mixed queries (10%)
        mixed_count = count - oltp_count - olap_count
        for i in range(mixed_count):
            query = """
            SELECT p.name, p.price, COUNT(oi.item_id) as times_ordered
            FROM products p
            LEFT JOIN order_items oi ON p.product_id = oi.product_id
            WHERE p.category IN ('Category1', 'Category2', 'Category3')
            GROUP BY p.product_id, p.name, p.price
            ORDER BY times_ordered DESC
            LIMIT 30;
            """
            queries.append((query, 'mixed'))
        
        # Shuffle queries
        np.random.shuffle(queries)
        
        self.logger.info(f"Generated {len(queries)} benchmark queries:")
        self.logger.info(f"  - OLTP: {oltp_count}")
        self.logger.info(f"  - OLAP: {olap_count}")  
        self.logger.info(f"  - Mixed: {mixed_count}")
        
        return queries
    
    def extract_query_features(self, query: str) -> QueryFeatures:
        """Extract features from SQL query text"""
        query_upper = query.upper()
        
        # Basic features
        query_length = len(query)
        num_tables = query_upper.count('FROM') + query_upper.count('JOIN')
        num_joins = query_upper.count('JOIN')
        num_filters = query_upper.count('WHERE') + query_upper.count('HAVING')
        num_aggregates = (query_upper.count('COUNT(') + query_upper.count('SUM(') + 
                         query_upper.count('AVG(') + query_upper.count('MIN(') + 
                         query_upper.count('MAX('))
        num_sorts = query_upper.count('ORDER BY')
        num_limits = query_upper.count('LIMIT')
        
        # Boolean features
        has_groupby = 'GROUP BY' in query_upper
        has_window = 'OVER (' in query_upper
        has_subquery = query_upper.count('SELECT') > 1
        has_distinct = 'DISTINCT' in query_upper
        
        # Estimated complexity
        complexity = (num_joins * 2 + num_aggregates * 1.5 + 
                     num_filters * 0.5 + (1 if has_groupby else 0) * 2 +
                     (1 if has_window else 0) * 3)
        
        return QueryFeatures(
            query_length=query_length,
            num_tables=num_tables,
            num_joins=num_joins,
            num_filters=num_filters,
            num_aggregates=num_aggregates,
            num_sorts=num_sorts,
            num_limits=num_limits,
            has_groupby=has_groupby,
            has_window=has_window,
            has_subquery=has_subquery,
            has_distinct=has_distinct,
            estimated_complexity=complexity
        )
    
    def execute_query_sqlite(self, query: str, timeout: float = 30.0) -> Tuple[float, bool, Optional[str], int]:
        """Execute query on SQLite with timing"""
        start_time = time.time()
        
        try:
            cursor = self.sqlite_conn.execute(query)
            
            if query.strip().upper().startswith('SELECT'):
                results = cursor.fetchall()
                result_count = len(results)
            else:
                result_count = cursor.rowcount
                self.sqlite_conn.commit()
                
            execution_time = time.time() - start_time
            return execution_time, True, None, result_count
            
        except Exception as e:
            execution_time = time.time() - start_time
            return execution_time, False, str(e)[:500], 0
    
    def execute_query_duckdb(self, query: str, timeout: float = 30.0) -> Tuple[float, bool, Optional[str], int]:
        """Execute query on DuckDB with timing"""
        start_time = time.time()
        
        try:
            # Skip UPDATE queries for DuckDB (read-only comparison)
            if query.strip().upper().startswith('UPDATE'):
                return 0.001, True, "UPDATE skipped in DuckDB", 0
                
            result = self.duckdb_conn.execute(query).fetchall()
            execution_time = time.time() - start_time
            result_count = len(result)
            
            return execution_time, True, None, result_count
            
        except Exception as e:
            execution_time = time.time() - start_time
            return execution_time, False, str(e)[:500], 0
    
    def collect_training_record(self, query: str, query_type: str, query_id: str) -> Optional[TrainingRecord]:
        """Collect a single training record through dual execution"""
        query_hash = hashlib.md5(query.encode()).hexdigest()[:16]
        
        try:
            # Extract features
            features = self.extract_query_features(query)
            
            # Execute on SQLite (OLTP engine)
            sqlite_time, sqlite_success, sqlite_error, sqlite_rows = self.execute_query_sqlite(query)
            
            # Execute on DuckDB (OLAP engine)
            duckdb_time, duckdb_success, duckdb_error, duckdb_rows = self.execute_query_duckdb(query)
            
            # Determine optimal engine and performance gap
            if sqlite_success and duckdb_success:
                if sqlite_time < duckdb_time:
                    optimal_engine = 'sqlite'
                    performance_gap = (duckdb_time - sqlite_time) / sqlite_time if sqlite_time > 0 else 0
                else:
                    optimal_engine = 'duckdb'
                    performance_gap = (sqlite_time - duckdb_time) / duckdb_time if duckdb_time > 0 else 0
            elif sqlite_success:
                optimal_engine = 'sqlite'
                performance_gap = 2.0
            elif duckdb_success:
                optimal_engine = 'duckdb'
                performance_gap = 2.0
            else:
                return None  # Both failed
            
            record = TrainingRecord(
                query_id=query_id,
                query_hash=query_hash,
                query_text=query[:1000],
                query_type=query_type,
                sqlite_execution_time=sqlite_time,
                sqlite_success=sqlite_success,
                sqlite_error=sqlite_error,
                sqlite_result_rows=sqlite_rows,
                duckdb_execution_time=duckdb_time,
                duckdb_success=duckdb_success,
                duckdb_error=duckdb_error,
                duckdb_result_rows=duckdb_rows,
                optimal_engine=optimal_engine,
                performance_gap=performance_gap,
                features=features,
                timestamp=datetime.now().isoformat(),
                collection_round=self.collection_round
            )
            
            return record
            
        except Exception as e:
            self.logger.error(f"Failed to collect record: {e}")
            return None
    
    def collect_batch_data(self, queries: List[Tuple[str, str]], batch_size: int = 100) -> int:
        """Collect training data in batches"""
        self.logger.info(f"Starting data collection for {len(queries)} queries...")
        
        collected = 0
        failed = 0
        start_time = time.time()
        
        for i, (query, query_type) in enumerate(queries):
            query_id = f"q{self.collection_round:04d}_{i:06d}"
            
            try:
                record = self.collect_training_record(query, query_type, query_id)
                
                if record:
                    self.training_records.append(record)
                    collected += 1
                    
                    # Update statistics
                    if record.optimal_engine == 'sqlite':
                        self.sqlite_wins += 1
                    else:
                        self.duckdb_wins += 1
                    
                    # Log progress
                    if collected % batch_size == 0:
                        elapsed = time.time() - start_time
                        rate = collected / elapsed
                        remaining = len(queries) - i - 1
                        eta = remaining / rate if rate > 0 else 0
                        
                        self.logger.info(f"Progress: {collected:,}/{len(queries):,} "
                                       f"({100*collected/len(queries):.1f}%) - "
                                       f"Rate: {rate:.1f}/sec - ETA: {eta/60:.1f}min")
                        self.logger.info(f"  SQLite wins: {self.sqlite_wins}, DuckDB wins: {self.duckdb_wins}")
                        
                        # Save intermediate results
                        self.save_training_data(f"data/training_data_partial_{collected}.json")
                else:
                    failed += 1
                    
            except KeyboardInterrupt:
                self.logger.info("Collection interrupted by user")
                break
            except Exception as e:
                failed += 1
                if failed % 100 == 0:
                    self.logger.warning(f"Failed queries: {failed}")
        
        collection_time = time.time() - start_time
        self.total_collected = collected
        
        self.logger.info(f"Batch collection complete:")
        self.logger.info(f"  - Collected: {collected:,} records")
        self.logger.info(f"  - Failed: {failed:,} records")
        self.logger.info(f"  - Success rate: {100*collected/(collected+failed):.1f}%")
        self.logger.info(f"  - SQLite optimal: {self.sqlite_wins:,} ({100*self.sqlite_wins/collected:.1f}%)")
        self.logger.info(f"  - DuckDB optimal: {self.duckdb_wins:,} ({100*self.duckdb_wins/collected:.1f}%)")
        self.logger.info(f"  - Collection time: {collection_time/60:.1f} minutes")
        self.logger.info(f"  - Rate: {collected/collection_time:.1f} records/sec")
        
        return collected
    
    def save_training_data(self, filename: str = None):
        """Save collected training data"""
        if filename is None:
            filename = f"data/real_training_data_{self.total_collected}.json"
        
        try:
            training_data = [record.to_dict() for record in self.training_records]
            
            metadata = {
                'collection_round': self.collection_round,
                'total_records': len(self.training_records),
                'timestamp': datetime.now().isoformat(),
                'sqlite_optimal_count': self.sqlite_wins,
                'duckdb_optimal_count': self.duckdb_wins,
                'sqlite_optimal_pct': 100 * self.sqlite_wins / len(self.training_records) if self.training_records else 0,
                'duckdb_optimal_pct': 100 * self.duckdb_wins / len(self.training_records) if self.training_records else 0
            }
            
            with open(filename, 'w') as f:
                json.dump({
                    'metadata': metadata,
                    'data': training_data
                }, f, indent=2)
            
            self.logger.info(f"✓ Training data saved to {filename}")
            self.logger.info(f"  Records: {len(self.training_records):,}")
            self.logger.info(f"  SQLite optimal: {self.sqlite_wins:,} ({metadata['sqlite_optimal_pct']:.1f}%)")
            self.logger.info(f"  DuckDB optimal: {self.duckdb_wins:,} ({metadata['duckdb_optimal_pct']:.1f}%)")
            
        except Exception as e:
            self.logger.error(f"Failed to save training data: {e}")
    
    def run_collection_session(self, target_records: int = 10000):
        """Run complete data collection session"""
        self.logger.info("=== Starting Real AQD Training Data Collection ===")
        self.logger.info(f"Target records: {target_records:,}")
        
        # Setup databases
        self.connect_databases()
        self.create_test_data()
        
        # Generate queries
        queries = self.generate_realistic_queries(target_records)
        
        # Collect data
        self.collection_round += 1
        collected = self.collect_batch_data(queries)
        
        # Save final results
        final_filename = f"data/real_aqd_training_data_{collected}.json"
        self.save_training_data(final_filename)
        
        self.logger.info("=== Real Data Collection Complete ===")
        self.logger.info(f"Final dataset: {final_filename}")
        self.logger.info(f"Records collected: {collected:,}")
        
        return final_filename, collected


def main():
    """Main execution for tmux session"""
    print("=== AQD Real Training Data Collection ===")
    print("Using SQLite (OLTP) and DuckDB (OLAP) for dual execution")
    print()
    
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Initialize collector
    collector = RealTrainingDataCollector()
    
    try:
        target_records = int(os.environ.get('AQD_TARGET_RECORDS', '10000'))
        print(f"Target records: {target_records:,}")
        
        final_file, collected = collector.run_collection_session(target_records)
        
        print(f"\n🎉 Real data collection complete!")
        print(f"Dataset: {final_file}")
        print(f"Records: {collected:,}")
        print(f"SQLite wins: {collector.sqlite_wins:,}")
        print(f"DuckDB wins: {collector.duckdb_wins:,}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Collection interrupted")
        if collector.training_records:
            collector.save_training_data("data/training_data_interrupted.json")
            print(f"Partial data saved: {len(collector.training_records):,} records")
    
    except Exception as e:
        print(f"\n❌ Collection failed: {e}")
        if collector.training_records:
            collector.save_training_data("data/training_data_error.json")


if __name__ == "__main__":
    main()