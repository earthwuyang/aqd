#!/usr/bin/env python3
"""
Comprehensive training data collection for AQD system
Collects 10K OLAP + 10K OLTP queries for each of 10+ datasets
"""
import json
import numpy as np
import pandas as pd
import psycopg2
import duckdb
import time
import logging
import hashlib
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/comprehensive_data_collection.log'),
        logging.StreamHandler()
    ]
)

class ComprehensiveDataCollector:
    def __init__(self):
        self.pg_conn = self.connect_postgresql()
        self.duck_conn = self.connect_duckdb()
        self.datasets = self.define_datasets()
        self.collected_data = []
        
    def connect_postgresql(self):
        return psycopg2.connect(
            host='localhost', port=5432, database='aqd_test',
            user='aqd_user', password='aqd_password'
        )
    
    def connect_duckdb(self):
        conn = duckdb.connect(':memory:')
        # Install and load postgres_scanner
        conn.execute("INSTALL postgres_scanner")
        conn.execute("LOAD postgres_scanner") 
        # Connect to PostgreSQL
        conn.execute("CALL postgres_attach('host=localhost port=5432 dbname=aqd_test user=aqd_user password=aqd_password')")
        return conn
    
    def define_datasets(self):
        """Define 12 different dataset characteristics for comprehensive testing"""
        return {
            'dataset_1_small_oltp': {
                'name': 'Small OLTP Workload',
                'table_filters': {'users': 'user_id <= 10000', 'orders': 'order_id <= 20000'},
                'complexity': 'low'
            },
            'dataset_2_medium_oltp': {
                'name': 'Medium OLTP Workload', 
                'table_filters': {'users': 'user_id <= 25000', 'orders': 'order_id <= 50000'},
                'complexity': 'medium'
            },
            'dataset_3_large_oltp': {
                'name': 'Large OLTP Workload',
                'table_filters': {'users': 'user_id <= 50000', 'orders': 'order_id <= 200000'},
                'complexity': 'high'
            },
            'dataset_4_small_olap': {
                'name': 'Small OLAP Analytics',
                'table_filters': {'users': 'user_id <= 10000', 'orders': 'total_amount > 50'},
                'complexity': 'analytical_light'
            },
            'dataset_5_medium_olap': {
                'name': 'Medium OLAP Analytics',
                'table_filters': {'orders': 'total_amount > 25', 'products': 'price > 10'},
                'complexity': 'analytical_medium'
            },
            'dataset_6_large_olap': {
                'name': 'Large OLAP Analytics',
                'table_filters': {},
                'complexity': 'analytical_heavy'
            },
            'dataset_7_time_series': {
                'name': 'Time Series Analysis',
                'table_filters': {'orders': "order_date >= '2023-01-01'"},
                'complexity': 'temporal'
            },
            'dataset_8_user_behavior': {
                'name': 'User Behavior Analysis',
                'table_filters': {'users': 'status = \'active\'', 'orders': 'status = \'completed\''},
                'complexity': 'behavioral'
            },
            'dataset_9_product_analytics': {
                'name': 'Product Performance Analytics',
                'table_filters': {'products': 'stock_quantity > 0'},
                'complexity': 'product_focused'
            },
            'dataset_10_financial': {
                'name': 'Financial Reporting',
                'table_filters': {'orders': 'total_amount > 100'},
                'complexity': 'financial'
            },
            'dataset_11_geographic': {
                'name': 'Geographic Analysis',
                'table_filters': {'users': "country IS NOT NULL", 'orders': "region IS NOT NULL"},
                'complexity': 'geographic'
            },
            'dataset_12_mixed_workload': {
                'name': 'Mixed OLTP/OLAP Workload',
                'table_filters': {},
                'complexity': 'hybrid'
            }
        }
    
    def generate_oltp_queries(self, dataset_config: Dict, count: int = 10000) -> List[str]:
        """Generate OLTP queries for transactional processing"""
        queries = []
        
        # OLTP Query templates - fast, simple, single-record operations
        oltp_templates = [
            # Point lookups
            "SELECT * FROM users WHERE user_id = {user_id}",
            "SELECT * FROM products WHERE product_id = {product_id}",
            "SELECT * FROM orders WHERE order_id = {order_id}",
            
            # Simple filters
            "SELECT name, email FROM users WHERE status = '{status}' LIMIT 100",
            "SELECT product_id, name, price FROM products WHERE category = '{category}' LIMIT 50",
            "SELECT order_id, total_amount FROM orders WHERE customer_id = {customer_id}",
            
            # Small aggregations
            "SELECT COUNT(*) FROM orders WHERE customer_id = {customer_id}",
            "SELECT SUM(total_amount) FROM orders WHERE customer_id = {customer_id} AND status = 'completed'",
            
            # Recent data queries
            "SELECT * FROM orders WHERE customer_id = {customer_id} AND order_date > NOW() - INTERVAL '30 days'",
            "SELECT * FROM users WHERE last_login > NOW() - INTERVAL '7 days' LIMIT 100",
            
            # Simple joins
            "SELECT u.name, o.total_amount FROM users u JOIN orders o ON u.user_id = o.customer_id WHERE u.user_id = {user_id}",
            "SELECT p.name, p.price FROM products p WHERE p.product_id IN (SELECT product_id FROM order_items WHERE order_id = {order_id})",
            
            # Insert/Update simulations (as SELECT for testing)
            "SELECT user_id FROM users WHERE email = '{email}' LIMIT 1",
            "SELECT order_id FROM orders WHERE customer_id = {customer_id} ORDER BY order_date DESC LIMIT 10",
        ]
        
        # Generate OLTP queries with dataset filters
        for _ in range(count):
            template = random.choice(oltp_templates)
            
            # Apply dataset-specific parameters
            query = template.format(
                user_id=random.randint(1, 50000),
                product_id=random.randint(1, 10000),
                order_id=random.randint(1, 200000),
                customer_id=random.randint(1, 50000),
                status=random.choice(['active', 'inactive', 'pending']),
                category=random.choice(['Electronics', 'Clothing', 'Books', 'Home']),
                email=f"user{random.randint(1, 50000)}@example.com"
            )
            
            # Apply dataset filters
            for table, condition in dataset_config.get('table_filters', {}).items():
                if table.upper() in query.upper():
                    query = f"SELECT * FROM ({query}) subq WHERE EXISTS (SELECT 1 FROM {table} WHERE {condition})"
                    break
            
            queries.append(query)
        
        return queries
    
    def generate_olap_queries(self, dataset_config: Dict, count: int = 10000) -> List[str]:
        """Generate OLAP queries for analytical processing"""
        queries = []
        
        # OLAP Query templates - complex, analytical, multi-table operations
        olap_templates = [
            # Time-based aggregations
            """SELECT DATE_TRUNC('month', order_date) as month,
                      COUNT(*) as order_count,
                      SUM(total_amount) as revenue,
                      AVG(total_amount) as avg_order_value
               FROM orders
               WHERE order_date >= '2023-01-01'
               GROUP BY DATE_TRUNC('month', order_date)
               ORDER BY month""",
            
            # Customer segmentation
            """SELECT u.country, u.city,
                      COUNT(DISTINCT u.user_id) as customers,
                      SUM(o.total_amount) as total_revenue,
                      AVG(o.total_amount) as avg_revenue_per_customer
               FROM users u
               JOIN orders o ON u.user_id = o.customer_id
               WHERE o.status = 'completed'
               GROUP BY u.country, u.city
               HAVING COUNT(DISTINCT u.user_id) > 10
               ORDER BY total_revenue DESC""",
            
            # Product performance analysis
            """SELECT p.category, p.name,
                      COUNT(oi.order_id) as times_ordered,
                      SUM(oi.quantity) as total_quantity_sold,
                      SUM(oi.quantity * p.price) as total_revenue
               FROM products p
               JOIN order_items oi ON p.product_id = oi.product_id
               JOIN orders o ON oi.order_id = o.order_id
               WHERE o.status = 'completed'
               GROUP BY p.category, p.name
               ORDER BY total_revenue DESC
               LIMIT 100""",
            
            # Cohort analysis
            """WITH first_purchase AS (
                   SELECT customer_id, MIN(order_date) as first_order_date
                   FROM orders
                   GROUP BY customer_id
               ),
               monthly_orders AS (
                   SELECT o.customer_id,
                          fp.first_order_date,
                          DATE_TRUNC('month', o.order_date) as order_month,
                          COUNT(*) as orders_in_month
                   FROM orders o
                   JOIN first_purchase fp ON o.customer_id = fp.customer_id
                   GROUP BY o.customer_id, fp.first_order_date, DATE_TRUNC('month', o.order_date)
               )
               SELECT DATE_TRUNC('month', first_order_date) as cohort_month,
                      order_month,
                      COUNT(DISTINCT customer_id) as active_customers
               FROM monthly_orders
               GROUP BY DATE_TRUNC('month', first_order_date), order_month
               ORDER BY cohort_month, order_month""",
            
            # Regional sales analysis
            """SELECT o.region,
                      DATE_TRUNC('quarter', o.order_date) as quarter,
                      COUNT(*) as orders,
                      SUM(o.total_amount) as revenue,
                      COUNT(DISTINCT o.customer_id) as unique_customers,
                      AVG(o.total_amount) as avg_order_value
               FROM orders o
               GROUP BY o.region, DATE_TRUNC('quarter', o.order_date)
               ORDER BY quarter, revenue DESC""",
            
            # Customer lifetime value
            """WITH customer_metrics AS (
                   SELECT u.user_id, u.registration_date,
                          COUNT(o.order_id) as total_orders,
                          SUM(o.total_amount) as lifetime_value,
                          MIN(o.order_date) as first_order,
                          MAX(o.order_date) as last_order
                   FROM users u
                   LEFT JOIN orders o ON u.user_id = o.customer_id AND o.status = 'completed'
                   GROUP BY u.user_id, u.registration_date
               )
               SELECT 
                   CASE 
                       WHEN lifetime_value = 0 THEN 'No Purchase'
                       WHEN lifetime_value < 100 THEN 'Low Value'
                       WHEN lifetime_value < 500 THEN 'Medium Value'
                       ELSE 'High Value'
                   END as customer_segment,
                   COUNT(*) as customer_count,
                   AVG(lifetime_value) as avg_lifetime_value,
                   AVG(total_orders) as avg_orders_per_customer
               FROM customer_metrics
               GROUP BY customer_segment
               ORDER BY avg_lifetime_value DESC""",
            
            # Inventory and demand analysis
            """SELECT p.category,
                      p.supplier_id,
                      COUNT(*) as products_in_category,
                      SUM(p.stock_quantity) as total_stock,
                      AVG(p.price) as avg_price,
                      COALESCE(SUM(demand.total_demand), 0) as total_demand
               FROM products p
               LEFT JOIN (
                   SELECT oi.product_id, SUM(oi.quantity) as total_demand
                   FROM order_items oi
                   JOIN orders o ON oi.order_id = o.order_id
                   WHERE o.order_date >= NOW() - INTERVAL '90 days'
                   GROUP BY oi.product_id
               ) demand ON p.product_id = demand.product_id
               GROUP BY p.category, p.supplier_id
               ORDER BY total_demand DESC NULLS LAST""",
            
            # Window functions analysis
            """SELECT u.user_id, u.name, u.registration_date,
                      o.order_date, o.total_amount,
                      ROW_NUMBER() OVER (PARTITION BY u.user_id ORDER BY o.order_date) as order_sequence,
                      LAG(o.total_amount) OVER (PARTITION BY u.user_id ORDER BY o.order_date) as prev_order_amount,
                      SUM(o.total_amount) OVER (PARTITION BY u.user_id ORDER BY o.order_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as running_total
               FROM users u
               JOIN orders o ON u.user_id = o.customer_id
               WHERE o.status = 'completed'
               ORDER BY u.user_id, o.order_date"""
        ]
        
        # Generate OLAP queries
        for _ in range(count):
            base_query = random.choice(olap_templates)
            
            # Add dataset-specific filters
            filters = dataset_config.get('table_filters', {})
            if filters:
                # Apply filters to the base query
                for table, condition in filters.items():
                    if table.upper() in base_query.upper():
                        # Add WHERE clause or extend existing WHERE
                        if 'WHERE' in base_query.upper():
                            base_query = base_query.replace('WHERE', f'WHERE ({condition}) AND (', 1) + ')'
                        else:
                            # Add WHERE before GROUP BY, ORDER BY, or at the end
                            for clause in ['GROUP BY', 'ORDER BY', 'LIMIT']:
                                if clause in base_query.upper():
                                    pos = base_query.upper().find(clause)
                                    base_query = base_query[:pos] + f'WHERE {condition} ' + base_query[pos:]
                                    break
                            else:
                                base_query += f' WHERE {condition}'
                        break
            
            queries.append(base_query)
        
        return queries
    
    def execute_dual_query(self, query: str, query_id: str, query_type: str) -> Dict:
        """Execute query on both PostgreSQL and DuckDB"""
        result = {
            'query_id': query_id,
            'query_text': query.strip(),
            'query_type': query_type,
            'query_hash': hashlib.md5(query.encode()).hexdigest()[:16]
        }
        
        # Execute on PostgreSQL
        try:
            start_time = time.time()
            with self.pg_conn.cursor() as cur:
                self.pg_conn.rollback()  # Clean state
                cur.execute(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query}")
                plan_result = cur.fetchone()[0][0]
                pg_time = plan_result.get('Execution Time', 0) / 1000.0  # Convert to seconds
                
                # Get actual row count
                cur.execute(query)
                rows = cur.fetchall()
                result.update({
                    'pg_execution_time': pg_time,
                    'pg_success': True,
                    'pg_error': None,
                    'pg_result_rows': len(rows),
                    'pg_plan': plan_result
                })
                self.pg_conn.rollback()
        except Exception as e:
            result.update({
                'pg_execution_time': float('inf'),
                'pg_success': False,
                'pg_error': str(e),
                'pg_result_rows': 0,
                'pg_plan': None
            })
            try:
                self.pg_conn.rollback()
            except:
                pass
        
        # Execute on DuckDB
        try:
            start_time = time.time()
            duck_result = self.duck_conn.execute(query).fetchall()
            duck_time = time.time() - start_time
            result.update({
                'duck_execution_time': duck_time,
                'duck_success': True,
                'duck_error': None,
                'duck_result_rows': len(duck_result)
            })
        except Exception as e:
            result.update({
                'duck_execution_time': float('inf'),
                'duck_success': False,
                'duck_error': str(e),
                'duck_result_rows': 0
            })
        
        # Determine optimal engine
        if result['pg_success'] and result['duck_success']:
            if result['pg_execution_time'] <= result['duck_execution_time']:
                result['optimal_engine'] = 'postgresql'
                result['performance_gap'] = result['duck_execution_time'] / result['pg_execution_time']
            else:
                result['optimal_engine'] = 'duckdb'
                result['performance_gap'] = result['pg_execution_time'] / result['duck_execution_time']
        elif result['pg_success']:
            result['optimal_engine'] = 'postgresql'
            result['performance_gap'] = float('inf')
        elif result['duck_success']:
            result['optimal_engine'] = 'duckdb'
            result['performance_gap'] = float('inf')
        else:
            result['optimal_engine'] = 'none'
            result['performance_gap'] = 1.0
        
        # Extract features
        result['features'] = self.extract_features(query, result.get('pg_plan'))
        
        return result
    
    def extract_features(self, query: str, plan: Dict) -> Dict:
        """Extract comprehensive features from query and execution plan"""
        features = {
            'query_length': len(query),
            'num_select': query.upper().count('SELECT'),
            'num_join': query.upper().count('JOIN'),
            'num_where': query.upper().count('WHERE'),
            'num_group_by': query.upper().count('GROUP BY'),
            'num_order_by': query.upper().count('ORDER BY'),
            'num_having': query.upper().count('HAVING'),
            'num_distinct': query.upper().count('DISTINCT'),
            'num_union': query.upper().count('UNION'),
            'num_subquery': query.count('(SELECT'),
            'num_with': query.upper().count('WITH'),
            'num_case': query.upper().count('CASE'),
        }
        
        # Plan-based features
        if plan:
            try:
                features.update({
                    'startup_cost': plan.get('Startup Cost', 0),
                    'total_cost': plan.get('Total Cost', 0),
                    'plan_rows': plan.get('Plan Rows', 0),
                    'plan_width': plan.get('Plan Width', 0)
                })
            except:
                features.update({
                    'startup_cost': 0,
                    'total_cost': 0,
                    'plan_rows': 0,
                    'plan_width': 0
                })
        
        return features
    
    def collect_dataset(self, dataset_name: str, dataset_config: Dict):
        """Collect training data for a specific dataset"""
        logging.info(f"Starting collection for {dataset_name}: {dataset_config['name']}")
        
        # Generate queries
        logging.info(f"Generating 10K OLTP queries for {dataset_name}")
        oltp_queries = self.generate_oltp_queries(dataset_config, 10000)
        
        logging.info(f"Generating 10K OLAP queries for {dataset_name}")
        olap_queries = self.generate_olap_queries(dataset_config, 10000)
        
        # Collect data
        dataset_results = []
        total_queries = len(oltp_queries) + len(olap_queries)
        
        logging.info(f"Executing {total_queries} queries for {dataset_name}")
        
        # Process OLTP queries
        for i, query in enumerate(oltp_queries):
            if i % 500 == 0:
                logging.info(f"{dataset_name} OLTP progress: {i}/10000")
            
            query_id = f"{dataset_name}_oltp_{i:06d}"
            result = self.execute_dual_query(query, query_id, 'oltp')
            result['dataset'] = dataset_name
            dataset_results.append(result)
            
            # Save intermediate results every 1000 queries
            if (i + 1) % 1000 == 0:
                self.save_intermediate_results(dataset_name, dataset_results, 'oltp', i + 1)
        
        # Process OLAP queries
        for i, query in enumerate(olap_queries):
            if i % 500 == 0:
                logging.info(f"{dataset_name} OLAP progress: {i}/10000")
            
            query_id = f"{dataset_name}_olap_{i:06d}"
            result = self.execute_dual_query(query, query_id, 'olap')
            result['dataset'] = dataset_name
            dataset_results.append(result)
            
            # Save intermediate results every 1000 queries
            if (i + 1) % 1000 == 0:
                self.save_intermediate_results(dataset_name, dataset_results, 'olap', i + 1 + 10000)
        
        logging.info(f"Completed {dataset_name}: {len(dataset_results)} queries executed")
        return dataset_results
    
    def save_intermediate_results(self, dataset_name: str, results: List[Dict], query_type: str, count: int):
        """Save intermediate results"""
        os.makedirs('data/comprehensive', exist_ok=True)
        filename = f"data/comprehensive/{dataset_name}_{query_type}_partial_{count}.json"
        
        pg_optimal = len([r for r in results if r.get('optimal_engine') == 'postgresql'])
        duck_optimal = len([r for r in results if r.get('optimal_engine') == 'duckdb'])
        
        output_data = {
            'metadata': {
                'dataset': dataset_name,
                'query_type': query_type,
                'total_records': len(results),
                'postgresql_records': pg_optimal,
                'duckdb_records': duck_optimal,
                'collection_timestamp': datetime.now().isoformat()
            },
            'data': results
        }
        
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logging.info(f"Saved {len(results)} records to {filename}")
    
    def collect_all_datasets(self):
        """Collect training data for all datasets"""
        logging.info("Starting comprehensive training data collection")
        logging.info(f"Target: {len(self.datasets)} datasets × 20K queries each = {len(self.datasets) * 20000} total queries")
        
        all_results = []
        
        for dataset_name, dataset_config in self.datasets.items():
            try:
                dataset_results = self.collect_dataset(dataset_name, dataset_config)
                all_results.extend(dataset_results)
                
                # Save dataset-specific results
                self.save_dataset_results(dataset_name, dataset_results)
                
            except Exception as e:
                logging.error(f"Failed to collect data for {dataset_name}: {e}")
                continue
        
        # Save final comprehensive results
        self.save_final_results(all_results)
        
        logging.info(f"Comprehensive collection complete: {len(all_results)} total queries")
        return all_results
    
    def save_dataset_results(self, dataset_name: str, results: List[Dict]):
        """Save complete dataset results"""
        os.makedirs('data/comprehensive', exist_ok=True)
        filename = f"data/comprehensive/{dataset_name}_complete.json"
        
        pg_optimal = len([r for r in results if r.get('optimal_engine') == 'postgresql'])
        duck_optimal = len([r for r in results if r.get('optimal_engine') == 'duckdb'])
        
        output_data = {
            'metadata': {
                'dataset': dataset_name,
                'total_records': len(results),
                'oltp_records': len([r for r in results if r.get('query_type') == 'oltp']),
                'olap_records': len([r for r in results if r.get('query_type') == 'olap']),
                'postgresql_records': pg_optimal,
                'duckdb_records': duck_optimal,
                'collection_timestamp': datetime.now().isoformat()
            },
            'data': results
        }
        
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logging.info(f"Dataset {dataset_name} complete: {len(results)} queries saved to {filename}")
    
    def save_final_results(self, all_results: List[Dict]):
        """Save final comprehensive results"""
        os.makedirs('data/comprehensive', exist_ok=True)
        filename = f"data/comprehensive/final_comprehensive_training_data.json"
        
        # Generate statistics
        pg_optimal = len([r for r in all_results if r.get('optimal_engine') == 'postgresql'])
        duck_optimal = len([r for r in all_results if r.get('optimal_engine') == 'duckdb'])
        oltp_count = len([r for r in all_results if r.get('query_type') == 'oltp'])
        olap_count = len([r for r in all_results if r.get('query_type') == 'olap'])
        
        # Dataset distribution
        dataset_stats = {}
        for result in all_results:
            dataset = result.get('dataset', 'unknown')
            if dataset not in dataset_stats:
                dataset_stats[dataset] = {'total': 0, 'oltp': 0, 'olap': 0, 'pg_optimal': 0, 'duck_optimal': 0}
            dataset_stats[dataset]['total'] += 1
            dataset_stats[dataset][result.get('query_type', 'unknown')] += 1
            if result.get('optimal_engine') == 'postgresql':
                dataset_stats[dataset]['pg_optimal'] += 1
            elif result.get('optimal_engine') == 'duckdb':
                dataset_stats[dataset]['duck_optimal'] += 1
        
        output_data = {
            'metadata': {
                'total_records': len(all_results),
                'oltp_records': oltp_count,
                'olap_records': olap_count,
                'postgresql_records': pg_optimal,
                'duckdb_records': duck_optimal,
                'datasets_collected': len(dataset_stats),
                'dataset_statistics': dataset_stats,
                'collection_timestamp': datetime.now().isoformat()
            },
            'data': all_results
        }
        
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logging.info(f"Final comprehensive training data saved: {len(all_results)} queries to {filename}")

def main():
    """Main execution function"""
    logging.info("=== Starting Comprehensive Training Data Collection ===")
    
    collector = ComprehensiveDataCollector()
    
    try:
        all_results = collector.collect_all_datasets()
        
        print("\n" + "="*80)
        print("🎯 COMPREHENSIVE TRAINING DATA COLLECTION COMPLETE")
        print("="*80)
        print(f"📊 Total Queries Collected: {len(all_results):,}")
        
        oltp_count = len([r for r in all_results if r.get('query_type') == 'oltp'])
        olap_count = len([r for r in all_results if r.get('query_type') == 'olap'])
        print(f"📈 OLTP Queries: {oltp_count:,}")
        print(f"📉 OLAP Queries: {olap_count:,}")
        
        pg_optimal = len([r for r in all_results if r.get('optimal_engine') == 'postgresql'])
        duck_optimal = len([r for r in all_results if r.get('optimal_engine') == 'duckdb'])
        print(f"🐘 PostgreSQL Optimal: {pg_optimal:,} ({pg_optimal/len(all_results)*100:.1f}%)")
        print(f"🦆 DuckDB Optimal: {duck_optimal:,} ({duck_optimal/len(all_results)*100:.1f}%)")
        
        print(f"📁 Data Location: data/comprehensive/")
        print("="*80)
        
    except Exception as e:
        logging.error(f"Collection failed: {e}")
        raise
    finally:
        if hasattr(collector, 'pg_conn'):
            collector.pg_conn.close()
        if hasattr(collector, 'duck_conn'):
            collector.duck_conn.close()

if __name__ == "__main__":
    main()