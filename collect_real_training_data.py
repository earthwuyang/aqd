#!/usr/bin/env python3
"""
Real AQD Training Data Collection Script
Collects actual training data through dual execution on PostgreSQL and DuckDB
"""

import os
import sys
import time
import json
import psycopg2
import duckdb
import logging
import hashlib
import traceback
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from datetime import datetime


@dataclass
class QueryFeatures:
    """Real query features extracted from PostgreSQL EXPLAIN"""
    # Cost model features
    startup_cost: float
    total_cost: float
    plan_rows: float
    plan_width: int
    
    # Plan structure features  
    num_tables: int
    num_joins: int
    num_filters: int
    num_aggregates: int
    num_sorts: int
    num_limits: int
    plan_depth: int
    
    # Operator counts
    num_seqscan: int
    num_indexscan: int
    num_hashjoin: int
    num_nestloop: int
    num_mergejoin: int
    num_material: int
    
    # Boolean features
    has_groupby: bool
    has_window: bool
    has_subquery: bool
    has_cte: bool
    has_distinct: bool
    
    # Derived features
    estimated_selectivity: float
    cost_per_row: float
    join_complexity: float
    
    def to_dict(self):
        return asdict(self)


@dataclass 
class TrainingRecord:
    """Real training record from dual execution"""
    query_id: str
    query_hash: str
    query_text: str
    query_type: str  # Inferred from analysis
    
    # PostgreSQL execution
    pg_execution_time: float
    pg_success: bool
    pg_error: Optional[str]
    pg_result_rows: int
    
    # DuckDB execution  
    duck_execution_time: float
    duck_success: bool
    duck_error: Optional[str]
    duck_result_rows: int
    
    # Ground truth
    optimal_engine: str  # 'postgresql' or 'duckdb'
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


class RealDataCollector:
    """
    Real training data collector using actual PostgreSQL and DuckDB execution
    """
    
    def __init__(self, 
                 pg_host: str = "localhost",
                 pg_port: int = 5432,
                 pg_user: str = "aqd_user",
                 pg_password: str = "aqd_password",
                 pg_database: str = "aqd_test"):
        
        self.pg_host = pg_host
        self.pg_port = pg_port 
        self.pg_user = pg_user
        self.pg_password = pg_password
        self.pg_database = pg_database
        
        # Database connections
        self.pg_conn = None
        self.duck_conn = None
        
        # Data storage
        self.training_records: List[TrainingRecord] = []
        self.collection_round = 0
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/data_collection.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def connect_databases(self):
        """Connect to both PostgreSQL and DuckDB"""
        try:
            # PostgreSQL connection
            self.pg_conn = psycopg2.connect(
                host=self.pg_host,
                port=self.pg_port,
                user=self.pg_user,
                password=self.pg_password,
                database=self.pg_database
            )
            self.pg_conn.autocommit = True
            
            # DuckDB connection (in-memory for now)
            self.duck_conn = duckdb.connect(':memory:')
            
            self.logger.info("✓ Connected to both PostgreSQL and DuckDB")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to databases: {e}")
            raise
    
    def create_test_tables(self):
        """Create test tables with substantial data for meaningful benchmarks"""
        self.logger.info("Creating test tables and inserting data...")
        
        # PostgreSQL tables
        with self.pg_conn.cursor() as cursor:
            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id SERIAL PRIMARY KEY,
                    name VARCHAR(100),
                    email VARCHAR(100),
                    age INTEGER,
                    city VARCHAR(50),
                    country VARCHAR(50),
                    registration_date DATE,
                    last_login TIMESTAMP,
                    status VARCHAR(20),
                    balance DECIMAL(10,2)
                );
            """)
            
            # Orders table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    order_id SERIAL PRIMARY KEY,
                    customer_id INTEGER REFERENCES users(user_id),
                    order_date TIMESTAMP,
                    total_amount DECIMAL(10,2),
                    status VARCHAR(20),
                    region VARCHAR(50),
                    shipping_cost DECIMAL(8,2),
                    discount DECIMAL(8,2),
                    payment_method VARCHAR(20)
                );
            """)
            
            # Products table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS products (
                    product_id SERIAL PRIMARY KEY,
                    name VARCHAR(100),
                    category VARCHAR(50),
                    price DECIMAL(10,2),
                    cost DECIMAL(10,2),
                    stock_quantity INTEGER,
                    supplier_id INTEGER,
                    created_date DATE
                );
            """)
            
            # Order_items table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS order_items (
                    item_id SERIAL PRIMARY KEY,
                    order_id INTEGER REFERENCES orders(order_id),
                    product_id INTEGER REFERENCES products(product_id),
                    quantity INTEGER,
                    unit_price DECIMAL(10,2),
                    discount_pct DECIMAL(5,2)
                );
            """)
            
            # Check if data already exists
            cursor.execute("SELECT COUNT(*) FROM users;")
            user_count = cursor.fetchone()[0]
            
            if user_count == 0:
                self.logger.info("Inserting test data (this may take a while)...")
                
                # Insert users (50K users)
                self.logger.info("Inserting 50K users...")
                for i in range(50000):
                    cursor.execute("""
                        INSERT INTO users (name, email, age, city, country, registration_date, last_login, status, balance)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
                    """, (
                        f"User{i}",
                        f"user{i}@example.com", 
                        np.random.randint(18, 80),
                        f"City{i % 100}",
                        f"Country{i % 20}",
                        '2024-01-01',
                        f'2024-{np.random.randint(1,13):02d}-{np.random.randint(1,29):02d}',
                        np.random.choice(['active', 'inactive', 'suspended']),
                        round(np.random.uniform(0, 10000), 2)
                    ))
                    
                    if (i + 1) % 10000 == 0:
                        self.logger.info(f"Inserted {i + 1} users...")
                
                # Insert products (10K products)
                self.logger.info("Inserting 10K products...")
                for i in range(10000):
                    cursor.execute("""
                        INSERT INTO products (name, category, price, cost, stock_quantity, supplier_id, created_date)
                        VALUES (%s, %s, %s, %s, %s, %s, %s);
                    """, (
                        f"Product{i}",
                        f"Category{i % 50}",
                        round(np.random.uniform(5, 1000), 2),
                        round(np.random.uniform(2, 500), 2),
                        np.random.randint(0, 1000),
                        i % 100,
                        '2024-01-01'
                    ))
                
                # Insert orders (200K orders)
                self.logger.info("Inserting 200K orders...")
                for i in range(200000):
                    cursor.execute("""
                        INSERT INTO orders (customer_id, order_date, total_amount, status, region, shipping_cost, discount, payment_method)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
                    """, (
                        (i % 50000) + 1,
                        f'2024-{np.random.randint(1,13):02d}-{np.random.randint(1,29):02d}',
                        round(np.random.uniform(10, 5000), 2),
                        np.random.choice(['completed', 'pending', 'cancelled']),
                        f"Region{i % 10}",
                        round(np.random.uniform(0, 50), 2),
                        round(np.random.uniform(0, 100), 2),
                        np.random.choice(['credit_card', 'paypal', 'bank_transfer'])
                    ))
                    
                    if (i + 1) % 50000 == 0:
                        self.logger.info(f"Inserted {i + 1} orders...")
                
                # Insert order items (500K items)
                self.logger.info("Inserting 500K order items...")
                for i in range(500000):
                    cursor.execute("""
                        INSERT INTO order_items (order_id, product_id, quantity, unit_price, discount_pct)
                        VALUES (%s, %s, %s, %s, %s);
                    """, (
                        (i % 200000) + 1,
                        (i % 10000) + 1,
                        np.random.randint(1, 10),
                        round(np.random.uniform(5, 200), 2),
                        round(np.random.uniform(0, 20), 2)
                    ))
                    
                    if (i + 1) % 100000 == 0:
                        self.logger.info(f"Inserted {i + 1} order items...")
                
                # Create indexes for performance
                self.logger.info("Creating indexes...")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_customer_id ON orders(customer_id);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_date ON orders(order_date);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_order_items_order_id ON order_items(order_id);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_order_items_product_id ON order_items(product_id);")
                
                self.logger.info("✓ Test data insertion complete!")
            else:
                self.logger.info(f"✓ Test data already exists ({user_count:,} users)")
        
        # Create corresponding DuckDB tables by copying from PostgreSQL
        self.logger.info("Creating DuckDB tables...")
        
        # Install postgres extension for DuckDB
        try:
            self.duck_conn.install_extension("postgres_scanner")
            self.duck_conn.load_extension("postgres_scanner")
            
            # Attach PostgreSQL database
            self.duck_conn.execute(f"""
                CALL postgres_attach('host={self.pg_host} port={self.pg_port} user={self.pg_user} password={self.pg_password} dbname={self.pg_database}');
            """)
            
            self.logger.info("✓ DuckDB connected to PostgreSQL via postgres_scanner")
            
        except Exception as e:
            self.logger.warning(f"Failed to setup postgres_scanner: {e}")
            # Fall back to copying data directly
            self._copy_data_to_duckdb()
    
    def _copy_data_to_duckdb(self):
        """Copy data from PostgreSQL to DuckDB for benchmarking"""
        self.logger.info("Copying data to DuckDB...")
        
        # Get PostgreSQL data
        with self.pg_conn.cursor() as cursor:
            # Copy users
            cursor.execute("SELECT * FROM users LIMIT 10000;")  # Subset for performance
            users_data = cursor.fetchall()
            
            cursor.execute("SELECT * FROM products LIMIT 5000;")
            products_data = cursor.fetchall()
            
            cursor.execute("SELECT * FROM orders LIMIT 20000;")
            orders_data = cursor.fetchall()
            
            cursor.execute("SELECT * FROM order_items LIMIT 50000;")
            items_data = cursor.fetchall()
        
        # Create DuckDB tables and insert data
        self.duck_conn.execute("""
            CREATE TABLE users (
                user_id INTEGER, name VARCHAR, email VARCHAR, age INTEGER,
                city VARCHAR, country VARCHAR, registration_date DATE,
                last_login TIMESTAMP, status VARCHAR, balance DECIMAL
            );
        """)
        
        self.duck_conn.execute("""
            CREATE TABLE products (
                product_id INTEGER, name VARCHAR, category VARCHAR, price DECIMAL,
                cost DECIMAL, stock_quantity INTEGER, supplier_id INTEGER, created_date DATE
            );
        """)
        
        self.duck_conn.execute("""
            CREATE TABLE orders (
                order_id INTEGER, customer_id INTEGER, order_date TIMESTAMP,
                total_amount DECIMAL, status VARCHAR, region VARCHAR,
                shipping_cost DECIMAL, discount DECIMAL, payment_method VARCHAR
            );
        """)
        
        self.duck_conn.execute("""
            CREATE TABLE order_items (
                item_id INTEGER, order_id INTEGER, product_id INTEGER,
                quantity INTEGER, unit_price DECIMAL, discount_pct DECIMAL
            );
        """)
        
        # Insert data
        self.duck_conn.executemany("INSERT INTO users VALUES (?,?,?,?,?,?,?,?,?,?)", users_data)
        self.duck_conn.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?,?)", products_data)  
        self.duck_conn.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?)", orders_data)
        self.duck_conn.executemany("INSERT INTO order_items VALUES (?,?,?,?,?,?)", items_data)
        
        self.logger.info("✓ Data copied to DuckDB")
    
    def generate_benchmark_queries(self, count: int = 5000) -> List[Tuple[str, str]]:
        """Generate realistic benchmark queries for data collection"""
        queries = []
        
        # OLTP queries (40%)
        oltp_count = int(count * 0.4)
        for i in range(oltp_count):
            query_type = np.random.choice([
                'point_select', 'simple_join', 'update', 'insert'
            ])
            
            if query_type == 'point_select':
                user_id = np.random.randint(1, 10000)
                query = f"SELECT * FROM users WHERE user_id = {user_id};"
                
            elif query_type == 'simple_join':
                customer_id = np.random.randint(1, 10000)
                query = f"SELECT u.name, COUNT(o.order_id) FROM users u LEFT JOIN orders o ON u.user_id = o.customer_id WHERE u.user_id = {customer_id} GROUP BY u.name;"
                
            elif query_type == 'update':
                user_id = np.random.randint(1, 10000)
                query = f"UPDATE users SET last_login = NOW() WHERE user_id = {user_id};"
                
            else:  # insert
                query = f"INSERT INTO users (name, email, age, city, status, balance) VALUES ('TestUser{i}', 'test{i}@example.com', 25, 'TestCity', 'active', 100.00);"
            
            queries.append((query, 'oltp'))
        
        # OLAP queries (40%)
        olap_count = int(count * 0.4)
        for i in range(olap_count):
            query_type = np.random.choice([
                'aggregation', 'complex_join', 'window_function', 'groupby_having'
            ])
            
            if query_type == 'aggregation':
                query = "SELECT region, COUNT(*) as order_count, AVG(total_amount) as avg_amount, SUM(total_amount) as total_revenue FROM orders WHERE order_date >= '2024-01-01' GROUP BY region ORDER BY total_revenue DESC;"
                
            elif query_type == 'complex_join':
                query = """
                SELECT p.category, COUNT(DISTINCT o.customer_id) as unique_customers, 
                       SUM(oi.quantity * oi.unit_price) as revenue,
                       AVG(oi.unit_price) as avg_price
                FROM products p 
                JOIN order_items oi ON p.product_id = oi.product_id
                JOIN orders o ON oi.order_id = o.order_id
                WHERE o.status = 'completed'
                GROUP BY p.category
                HAVING revenue > 1000
                ORDER BY revenue DESC
                LIMIT 20;
                """
                
            elif query_type == 'window_function':
                query = """
                SELECT customer_id, order_date, total_amount,
                       ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date) as order_num,
                       SUM(total_amount) OVER (PARTITION BY customer_id) as customer_total
                FROM orders 
                WHERE order_date >= '2024-01-01'
                ORDER BY customer_id, order_date;
                """
                
            else:  # groupby_having
                query = """
                SELECT u.country, u.city, COUNT(*) as user_count, AVG(u.balance) as avg_balance
                FROM users u
                WHERE u.status = 'active'
                GROUP BY u.country, u.city
                HAVING COUNT(*) > 10 AND AVG(u.balance) > 1000
                ORDER BY user_count DESC, avg_balance DESC
                LIMIT 50;
                """
            
            queries.append((query, 'olap'))
        
        # Mixed queries (20%)
        mixed_count = count - oltp_count - olap_count
        for i in range(mixed_count):
            query_type = np.random.choice(['medium_join', 'filtered_agg', 'subquery'])
            
            if query_type == 'medium_join':
                query = """
                SELECT u.name, u.city, o.total_amount, o.order_date
                FROM users u
                JOIN orders o ON u.user_id = o.customer_id
                WHERE u.country = 'Country1' AND o.total_amount > 500
                ORDER BY o.order_date DESC
                LIMIT 100;
                """
                
            elif query_type == 'filtered_agg':
                region = f"Region{np.random.randint(0, 10)}"
                query = f"""
                SELECT DATE_TRUNC('month', order_date) as month,
                       COUNT(*) as order_count,
                       SUM(total_amount) as monthly_revenue
                FROM orders 
                WHERE region = '{region}' AND status = 'completed'
                GROUP BY DATE_TRUNC('month', order_date)
                ORDER BY month;
                """
                
            else:  # subquery
                query = """
                SELECT p.name, p.category, p.price,
                       (SELECT COUNT(*) FROM order_items oi WHERE oi.product_id = p.product_id) as times_ordered
                FROM products p
                WHERE p.price > (SELECT AVG(price) FROM products WHERE category = p.category)
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
        """Extract detailed features from PostgreSQL EXPLAIN output"""
        try:
            with self.pg_conn.cursor() as cursor:
                # Get detailed execution plan
                cursor.execute(f"EXPLAIN (FORMAT JSON, ANALYZE FALSE, BUFFERS FALSE) {query}")
                plan_json = cursor.fetchone()[0][0]
                
                # Extract basic cost features
                startup_cost = plan_json.get('Startup Cost', 0)
                total_cost = plan_json.get('Total Cost', 0)
                plan_rows = plan_json.get('Plan Rows', 0)
                plan_width = plan_json.get('Plan Width', 0)
                
                # Analyze plan tree
                features = self._analyze_plan_tree(plan_json)
                
                # Calculate derived features
                cost_per_row = total_cost / max(1, plan_rows)
                estimated_selectivity = min(1.0, plan_rows / max(1, plan_rows * 10))
                join_complexity = features['num_joins'] * (1 + features['plan_depth'] / 10)
                
                return QueryFeatures(
                    startup_cost=startup_cost,
                    total_cost=total_cost,
                    plan_rows=plan_rows,
                    plan_width=plan_width,
                    num_tables=features['num_tables'],
                    num_joins=features['num_joins'],
                    num_filters=features['num_filters'],
                    num_aggregates=features['num_aggregates'],
                    num_sorts=features['num_sorts'],
                    num_limits=features['num_limits'],
                    plan_depth=features['plan_depth'],
                    num_seqscan=features['num_seqscan'],
                    num_indexscan=features['num_indexscan'],
                    num_hashjoin=features['num_hashjoin'],
                    num_nestloop=features['num_nestloop'],
                    num_mergejoin=features['num_mergejoin'],
                    num_material=features['num_material'],
                    has_groupby=features['has_groupby'],
                    has_window=features['has_window'],
                    has_subquery=features['has_subquery'],
                    has_cte=features['has_cte'],
                    has_distinct=features['has_distinct'],
                    estimated_selectivity=estimated_selectivity,
                    cost_per_row=cost_per_row,
                    join_complexity=join_complexity
                )
                
        except Exception as e:
            self.logger.warning(f"Failed to extract features for query: {e}")
            # Return default features
            return QueryFeatures(
                startup_cost=0, total_cost=100, plan_rows=1, plan_width=10,
                num_tables=1, num_joins=0, num_filters=0, num_aggregates=0,
                num_sorts=0, num_limits=0, plan_depth=1, num_seqscan=1,
                num_indexscan=0, num_hashjoin=0, num_nestloop=0, num_mergejoin=0,
                num_material=0, has_groupby=False, has_window=False, has_subquery=False,
                has_cte=False, has_distinct=False, estimated_selectivity=1.0,
                cost_per_row=100, join_complexity=0
            )
    
    def _analyze_plan_tree(self, node: dict, depth: int = 0) -> dict:
        """Recursively analyze PostgreSQL plan tree"""
        features = {
            'num_tables': 0, 'num_joins': 0, 'num_filters': 0, 'num_aggregates': 0,
            'num_sorts': 0, 'num_limits': 0, 'plan_depth': depth, 'num_seqscan': 0,
            'num_indexscan': 0, 'num_hashjoin': 0, 'num_nestloop': 0, 'num_mergejoin': 0,
            'num_material': 0, 'has_groupby': False, 'has_window': False,
            'has_subquery': False, 'has_cte': False, 'has_distinct': False
        }
        
        node_type = node.get('Node Type', '')
        
        # Count node types
        if 'Scan' in node_type:
            features['num_tables'] += 1
            if 'Seq Scan' in node_type:
                features['num_seqscan'] += 1
            elif 'Index' in node_type:
                features['num_indexscan'] += 1
        
        if 'Join' in node_type:
            features['num_joins'] += 1
            if 'Hash Join' in node_type:
                features['num_hashjoin'] += 1
            elif 'Nested Loop' in node_type:
                features['num_nestloop'] += 1
            elif 'Merge Join' in node_type:
                features['num_mergejoin'] += 1
        
        if 'Sort' in node_type:
            features['num_sorts'] += 1
        if 'Limit' in node_type:
            features['num_limits'] += 1
        if 'Aggregate' in node_type or 'Group' in node_type:
            features['num_aggregates'] += 1
            features['has_groupby'] = True
        if 'WindowAgg' in node_type:
            features['has_window'] = True
        if 'Material' in node_type:
            features['num_material'] += 1
        if 'Unique' in node_type:
            features['has_distinct'] = True
        if 'CTE' in node_type:
            features['has_cte'] = True
        if 'SubPlan' in node_type:
            features['has_subquery'] = True
        
        # Count filters
        if 'Filter' in node:
            features['num_filters'] += 1
        if 'Index Cond' in node:
            features['num_filters'] += 1
        if 'Hash Cond' in node:
            features['num_filters'] += 1
        if 'Join Filter' in node:
            features['num_filters'] += 1
        
        # Recursively process child plans
        if 'Plans' in node:
            for child in node['Plans']:
                child_features = self._analyze_plan_tree(child, depth + 1)
                # Merge features
                for key in features:
                    if key == 'plan_depth':
                        features[key] = max(features[key], child_features[key])
                    elif isinstance(features[key], bool):
                        features[key] = features[key] or child_features[key]
                    else:
                        features[key] += child_features[key]
        
        return features
    
    def execute_query_postgresql(self, query: str, timeout: float = 30.0) -> Tuple[float, bool, Optional[str], int]:
        """Execute query on PostgreSQL with timing"""
        start_time = time.time()
        
        try:
            with self.pg_conn.cursor() as cursor:
                cursor.execute(query)
                
                if query.strip().upper().startswith('SELECT'):
                    results = cursor.fetchall()
                    result_count = len(results)
                else:
                    result_count = cursor.rowcount
                    
                execution_time = time.time() - start_time
                return execution_time, True, None, result_count
                
        except Exception as e:
            execution_time = time.time() - start_time
            return execution_time, False, str(e)[:500], 0
    
    def execute_query_duckdb(self, query: str, timeout: float = 30.0) -> Tuple[float, bool, Optional[str], int]:
        """Execute query on DuckDB with timing"""
        start_time = time.time()
        
        try:
            result = self.duck_conn.execute(query).fetchall()
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
            
            # Execute on PostgreSQL
            pg_time, pg_success, pg_error, pg_rows = self.execute_query_postgresql(query)
            
            # Execute on DuckDB
            duck_time, duck_success, duck_error, duck_rows = self.execute_query_duckdb(query)
            
            # Determine optimal engine and performance gap
            if pg_success and duck_success:
                if pg_time < duck_time:
                    optimal_engine = 'postgresql'
                    performance_gap = (duck_time - pg_time) / pg_time
                else:
                    optimal_engine = 'duckdb'
                    performance_gap = (pg_time - duck_time) / duck_time
            elif pg_success:
                optimal_engine = 'postgresql'
                performance_gap = 2.0  # Penalty for DuckDB failure
            elif duck_success:
                optimal_engine = 'duckdb'
                performance_gap = 2.0  # Penalty for PostgreSQL failure
            else:
                # Both failed - skip this record
                return None
            
            record = TrainingRecord(
                query_id=query_id,
                query_hash=query_hash,
                query_text=query[:1000],  # Truncate for storage
                query_type=query_type,
                pg_execution_time=pg_time,
                pg_success=pg_success,
                pg_error=pg_error,
                pg_result_rows=pg_rows,
                duck_execution_time=duck_time,
                duck_success=duck_success,
                duck_error=duck_error,
                duck_result_rows=duck_rows,
                optimal_engine=optimal_engine,
                performance_gap=performance_gap,
                features=features,
                timestamp=datetime.now().isoformat(),
                collection_round=self.collection_round
            )
            
            return record
            
        except Exception as e:
            self.logger.error(f"Failed to collect training record: {e}")
            self.logger.error(f"Query: {query[:200]}...")
            return None
    
    def collect_batch_data(self, queries: List[Tuple[str, str]], batch_size: int = 100):
        """Collect training data in batches with progress tracking"""
        self.logger.info(f"Starting batch data collection for {len(queries)} queries...")
        
        collected_count = 0
        failed_count = 0
        start_time = time.time()
        
        for i, (query, query_type) in enumerate(queries):
            query_id = f"q{self.collection_round:04d}_{i:06d}"
            
            try:
                record = self.collect_training_record(query, query_type, query_id)
                
                if record:
                    self.training_records.append(record)
                    collected_count += 1
                    
                    # Log progress
                    if collected_count % batch_size == 0:
                        elapsed_time = time.time() - start_time
                        rate = collected_count / elapsed_time
                        remaining = len(queries) - i - 1
                        eta = remaining / rate if rate > 0 else 0
                        
                        self.logger.info(f"Progress: {collected_count:,}/{len(queries):,} "
                                       f"({100*collected_count/len(queries):.1f}%) - "
                                       f"Rate: {rate:.1f} records/sec - "
                                       f"ETA: {eta/60:.1f} min")
                        
                        # Save intermediate results
                        self.save_training_data(f"data/training_data_round{self.collection_round}_partial.json")
                else:
                    failed_count += 1
                    
            except KeyboardInterrupt:
                self.logger.info("Collection interrupted by user")
                break
            except Exception as e:
                failed_count += 1
                self.logger.warning(f"Failed to process query {i}: {e}")
        
        collection_time = time.time() - start_time
        
        self.logger.info(f"Batch collection complete:")
        self.logger.info(f"  - Collected: {collected_count:,} records")
        self.logger.info(f"  - Failed: {failed_count:,} records")
        self.logger.info(f"  - Success rate: {100*collected_count/(collected_count+failed_count):.1f}%")
        self.logger.info(f"  - Collection time: {collection_time/60:.1f} minutes")
        self.logger.info(f"  - Rate: {collected_count/collection_time:.1f} records/sec")
        
        return collected_count
    
    def save_training_data(self, filename: str = None):
        """Save collected training data to JSON file"""
        if filename is None:
            filename = f"data/training_data_round{self.collection_round}_{len(self.training_records)}.json"
        
        try:
            training_data = [record.to_dict() for record in self.training_records]
            
            with open(filename, 'w') as f:
                json.dump({
                    'metadata': {
                        'collection_round': self.collection_round,
                        'total_records': len(self.training_records),
                        'collection_timestamp': datetime.now().isoformat(),
                        'postgresql_records': len([r for r in self.training_records if r.optimal_engine == 'postgresql']),
                        'duckdb_records': len([r for r in self.training_records if r.optimal_engine == 'duckdb']),
                    },
                    'data': training_data
                }, f, indent=2)
            
            self.logger.info(f"✓ Training data saved to {filename}")
            self.logger.info(f"  - Records: {len(self.training_records):,}")
            self.logger.info(f"  - PostgreSQL optimal: {len([r for r in self.training_records if r.optimal_engine == 'postgresql']):,}")
            self.logger.info(f"  - DuckDB optimal: {len([r for r in self.training_records if r.optimal_engine == 'duckdb']):,}")
            
        except Exception as e:
            self.logger.error(f"Failed to save training data: {e}")
    
    def run_data_collection_session(self, target_records: int = 10000):
        """Run a complete data collection session"""
        self.logger.info(f"=== Starting AQD Data Collection Session ===")
        self.logger.info(f"Target records: {target_records:,}")
        
        # Connect to databases
        self.connect_databases()
        
        # Setup test tables
        self.create_test_tables()
        
        # Generate queries
        queries = self.generate_benchmark_queries(target_records)
        
        # Collect data
        self.collection_round += 1
        collected = self.collect_batch_data(queries)
        
        # Save final data
        final_filename = f"data/aqd_training_data_final_round{self.collection_round}.json"
        self.save_training_data(final_filename)
        
        self.logger.info(f"=== Data Collection Session Complete ===")
        self.logger.info(f"Final dataset: {final_filename}")
        
        return final_filename


def main():
    """Main data collection script for tmux execution"""
    print("=== AQD Real Training Data Collection ===")
    print("This script will collect real training data through dual execution")
    print()
    
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Initialize collector
    collector = RealDataCollector()
    
    # Run collection session
    try:
        target_records = int(os.environ.get('AQD_TARGET_RECORDS', '5000'))
        print(f"Target records: {target_records:,}")
        print("Starting collection...")
        
        final_file = collector.run_data_collection_session(target_records)
        
        print(f"\n🎉 Data collection complete!")
        print(f"Dataset saved to: {final_file}")
        print(f"Records collected: {len(collector.training_records):,}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Collection interrupted by user")
        if collector.training_records:
            collector.save_training_data("data/training_data_interrupted.json")
            print(f"Partial data saved: {len(collector.training_records):,} records")
    
    except Exception as e:
        print(f"\n❌ Collection failed: {e}")
        if collector.training_records:
            collector.save_training_data("data/training_data_error.json")
            print(f"Partial data saved: {len(collector.training_records):,} records")


if __name__ == "__main__":
    main()