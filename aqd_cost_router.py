"""
AQD Cost-Threshold Routing Implementation for PostgreSQL + pg_duckdb
Based on the AQD paper: "AQD: Online Adaptive Query Dispatcher for HTAP Databases"
"""

import psycopg2
import psycopg2.extras
import logging
import time
import json
import hashlib
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np


class RoutingDecision(Enum):
    """Query routing decisions"""
    POSTGRESQL = "postgresql"  # Route to PostgreSQL (row-store)
    DUCKDB = "duckdb"         # Route to DuckDB (column-store) via pg_duckdb
    

@dataclass
class QueryFeatures:
    """Query features extracted from PostgreSQL planner"""
    # Cost model features
    startup_cost: float
    total_cost: float
    plan_rows: float
    plan_width: int
    
    # Plan structure features
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
    
    # Query characteristics
    has_groupby: bool
    has_window: bool
    has_subquery: bool
    estimated_selectivity: float
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'startup_cost': self.startup_cost,
            'total_cost': self.total_cost,
            'plan_rows': self.plan_rows,
            'plan_width': self.plan_width,
            'num_joins': self.num_joins,
            'num_filters': self.num_filters,
            'num_aggregates': self.num_aggregates,
            'num_sorts': self.num_sorts,
            'num_limits': self.num_limits,
            'plan_depth': self.plan_depth,
            'num_seqscan': self.num_seqscan,
            'num_indexscan': self.num_indexscan,
            'num_hashjoin': self.num_hashjoin,
            'num_nestloop': self.num_nestloop,
            'num_mergejoin': self.num_mergejoin,
            'has_groupby': self.has_groupby,
            'has_window': self.has_window,
            'has_subquery': self.has_subquery,
            'estimated_selectivity': self.estimated_selectivity
        }


@dataclass
class RoutingResult:
    """Result of query routing and execution"""
    query_hash: str
    decision: RoutingDecision
    execution_time: float
    success: bool
    error_message: Optional[str]
    features: QueryFeatures
    result_rows: int


class AQDCostRouter:
    """
    AQD Cost-Threshold Router for PostgreSQL + pg_duckdb
    
    Implements cost-threshold routing as described in the AQD paper.
    Routes queries between PostgreSQL (OLTP) and DuckDB (OLAP) based on cost estimates.
    """
    
    def __init__(self, 
                 pg_host: str = "localhost",
                 pg_port: int = 5432,
                 pg_user: str = "postgres",
                 pg_password: str = "postgres",
                 pg_database: str = "aqd_test",
                 cost_threshold: float = 1000.0):
        """
        Initialize AQD Cost Router
        
        Args:
            pg_host: PostgreSQL host
            pg_port: PostgreSQL port
            pg_user: PostgreSQL username
            pg_password: PostgreSQL password
            pg_database: PostgreSQL database name
            cost_threshold: Cost threshold for routing decisions
        """
        self.pg_host = pg_host
        self.pg_port = pg_port
        self.pg_user = pg_user
        self.pg_password = pg_password
        self.pg_database = pg_database
        self.cost_threshold = cost_threshold
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Connection pool
        self.connection = None
        
    def connect(self):
        """Establish connection to PostgreSQL with pg_duckdb extension"""
        try:
            self.connection = psycopg2.connect(
                host=self.pg_host,
                port=self.pg_port,
                user=self.pg_user,
                password=self.pg_password,
                database=self.pg_database
            )
            self.connection.autocommit = False
            
            # Enable pg_duckdb extension
            with self.connection.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS pg_duckdb;")
                self.connection.commit()
                
            self.logger.info("Connected to PostgreSQL with pg_duckdb extension")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise
            
    def disconnect(self):
        """Close PostgreSQL connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
            
    def extract_query_features(self, query: str) -> QueryFeatures:
        """
        Extract query features from PostgreSQL query plan
        
        Args:
            query: SQL query string
            
        Returns:
            QueryFeatures object with extracted features
        """
        if not self.connection:
            self.connect()
            
        with self.connection.cursor() as cursor:
            # Get detailed execution plan
            explain_query = f"EXPLAIN (FORMAT JSON, ANALYZE false, BUFFERS false) {query}"
            cursor.execute(explain_query)
            plan_result = cursor.fetchone()[0]
            
            plan = plan_result[0]  # First element of JSON array
            
            # Extract basic cost features
            startup_cost = plan.get('Startup Cost', 0)
            total_cost = plan.get('Total Cost', 0)
            plan_rows = plan.get('Plan Rows', 0)
            plan_width = plan.get('Plan Width', 0)
            
            # Recursively analyze plan nodes
            features_dict = self._analyze_plan_node(plan)
            
            # Calculate estimated selectivity
            estimated_selectivity = min(1.0, plan_rows / max(1, plan_rows * 10))  # Rough estimate
            
            return QueryFeatures(
                startup_cost=startup_cost,
                total_cost=total_cost,
                plan_rows=plan_rows,
                plan_width=plan_width,
                num_joins=features_dict.get('joins', 0),
                num_filters=features_dict.get('filters', 0),
                num_aggregates=features_dict.get('aggregates', 0),
                num_sorts=features_dict.get('sorts', 0),
                num_limits=features_dict.get('limits', 0),
                plan_depth=features_dict.get('depth', 0),
                num_seqscan=features_dict.get('seqscan', 0),
                num_indexscan=features_dict.get('indexscan', 0),
                num_hashjoin=features_dict.get('hashjoin', 0),
                num_nestloop=features_dict.get('nestloop', 0),
                num_mergejoin=features_dict.get('mergejoin', 0),
                has_groupby=features_dict.get('has_groupby', False),
                has_window=features_dict.get('has_window', False),
                has_subquery=features_dict.get('has_subquery', False),
                estimated_selectivity=estimated_selectivity
            )
    
    def _analyze_plan_node(self, node: dict, depth: int = 0) -> dict:
        """
        Recursively analyze a plan node to extract features
        
        Args:
            node: Plan node dictionary
            depth: Current depth in plan tree
            
        Returns:
            Dictionary of extracted features
        """
        features = {
            'joins': 0, 'filters': 0, 'aggregates': 0, 'sorts': 0, 'limits': 0,
            'seqscan': 0, 'indexscan': 0, 'hashjoin': 0, 'nestloop': 0, 'mergejoin': 0,
            'has_groupby': False, 'has_window': False, 'has_subquery': False,
            'depth': depth
        }
        
        node_type = node.get('Node Type', '')
        
        # Count operator types
        if 'Join' in node_type:
            features['joins'] += 1
            if 'Hash' in node_type:
                features['hashjoin'] += 1
            elif 'Nested Loop' in node_type:
                features['nestloop'] += 1
            elif 'Merge' in node_type:
                features['mergejoin'] += 1
                
        if 'Scan' in node_type:
            if 'Seq' in node_type:
                features['seqscan'] += 1
            elif 'Index' in node_type:
                features['indexscan'] += 1
                
        if 'Sort' in node_type:
            features['sorts'] += 1
        if 'Limit' in node_type:
            features['limits'] += 1
        if 'Aggregate' in node_type or 'Group' in node_type:
            features['aggregates'] += 1
            features['has_groupby'] = True
        if 'Window' in node_type:
            features['has_window'] = True
        if 'SubPlan' in node_type or 'CTE' in node_type:
            features['has_subquery'] = True
            
        # Count filters (approximate from conditions)
        if 'Filter' in node:
            features['filters'] += 1
        if 'Index Cond' in node:
            features['filters'] += 1
        if 'Hash Cond' in node:
            features['filters'] += 1
        if 'Join Filter' in node:
            features['filters'] += 1
            
        # Recursively process child plans
        if 'Plans' in node:
            for child_plan in node['Plans']:
                child_features = self._analyze_plan_node(child_plan, depth + 1)
                # Merge features
                for key in features:
                    if key == 'depth':
                        features[key] = max(features[key], child_features[key])
                    elif isinstance(features[key], bool):
                        features[key] = features[key] or child_features[key]
                    else:
                        features[key] += child_features[key]
        
        return features
    
    def make_routing_decision(self, features: QueryFeatures) -> RoutingDecision:
        """
        Make routing decision based on cost threshold
        
        Args:
            features: Query features
            
        Returns:
            Routing decision
        """
        # AQD cost-threshold routing logic
        # Route to DuckDB (column-store) if query exceeds cost threshold
        # This typically indicates OLAP workloads
        if features.total_cost > self.cost_threshold:
            return RoutingDecision.DUCKDB
            
        # Route to PostgreSQL (row-store) for low-cost queries
        # This typically indicates OLTP workloads
        return RoutingDecision.POSTGRESQL
    
    def execute_query_postgresql(self, query: str) -> Tuple[float, bool, Optional[str], int]:
        """
        Execute query on PostgreSQL (row-store)
        
        Args:
            query: SQL query string
            
        Returns:
            Tuple of (execution_time, success, error_message, result_rows)
        """
        start_time = time.time()
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query)
                
                # Handle SELECT queries
                if query.strip().upper().startswith('SELECT'):
                    results = cursor.fetchall()
                    result_rows = len(results)
                else:
                    result_rows = cursor.rowcount
                    self.connection.commit()
                    
                execution_time = time.time() - start_time
                return execution_time, True, None, result_rows
                
        except Exception as e:
            execution_time = time.time() - start_time
            return execution_time, False, str(e), 0
    
    def execute_query_duckdb(self, query: str) -> Tuple[float, bool, Optional[str], int]:
        """
        Execute query on DuckDB (column-store) via pg_duckdb extension
        
        Args:
            query: SQL query string
            
        Returns:
            Tuple of (execution_time, success, error_message, result_rows)
        """
        start_time = time.time()
        
        try:
            with self.connection.cursor() as cursor:
                # Use duckdb.execute() for DuckDB execution
                duckdb_query = f"SELECT duckdb.execute({psycopg2.extensions.adapt(query).getquoted().decode()})"
                cursor.execute(duckdb_query)
                
                results = cursor.fetchall()
                result_rows = len(results) if results else 0
                    
                execution_time = time.time() - start_time
                return execution_time, True, None, result_rows
                
        except Exception as e:
            execution_time = time.time() - start_time
            return execution_time, False, str(e), 0
    
    def route_and_execute_query(self, query: str) -> RoutingResult:
        """
        Route and execute query based on AQD cost-threshold routing
        
        Args:
            query: SQL query string
            
        Returns:
            RoutingResult with execution details
        """
        # Generate query hash for tracking
        query_hash = hashlib.md5(query.encode()).hexdigest()[:16]
        
        # Extract query features
        features = self.extract_query_features(query)
        
        # Make routing decision
        decision = self.make_routing_decision(features)
        
        # Execute query based on routing decision
        if decision == RoutingDecision.POSTGRESQL:
            execution_time, success, error_msg, result_rows = self.execute_query_postgresql(query)
        else:  # RoutingDecision.DUCKDB
            execution_time, success, error_msg, result_rows = self.execute_query_duckdb(query)
        
        # Log routing decision and result
        self.logger.info(f"Query {query_hash}: {decision.value}, "
                        f"cost={features.total_cost:.2f}, "
                        f"time={execution_time:.4f}s, success={success}")
        
        return RoutingResult(
            query_hash=query_hash,
            decision=decision,
            execution_time=execution_time,
            success=success,
            error_message=error_msg,
            features=features,
            result_rows=result_rows
        )
    
    def set_cost_threshold(self, threshold: float):
        """Update cost threshold for routing decisions"""
        self.cost_threshold = threshold
        self.logger.info(f"Updated cost threshold to {threshold}")


def main():
    """Example usage of AQD Cost Router"""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize router
    router = AQDCostRouter(cost_threshold=1000.0)
    router.connect()
    
    # Example queries
    test_queries = [
        # OLTP query (low cost, should route to PostgreSQL)
        "SELECT * FROM users WHERE user_id = 12345 LIMIT 1;",
        
        # OLAP query (high cost, should route to DuckDB)
        "SELECT region, COUNT(*), AVG(sales), SUM(quantity) FROM transactions GROUP BY region ORDER BY region;",
        
        # Mixed query
        "SELECT customer_id, SUM(amount) FROM orders WHERE order_date > '2024-01-01' GROUP BY customer_id HAVING SUM(amount) > 1000;"
    ]
    
    results = []
    for query in test_queries:
        print(f"\nExecuting query: {query[:50]}...")
        result = router.route_and_execute_query(query)
        results.append(result)
        
        print(f"Decision: {result.decision.value}")
        print(f"Cost: {result.features.total_cost:.2f}")
        print(f"Time: {result.execution_time:.4f}s")
        print(f"Success: {result.success}")
    
    router.disconnect()
    
    # Print summary
    print(f"\nSummary:")
    print(f"Total queries: {len(results)}")
    print(f"PostgreSQL: {sum(1 for r in results if r.decision == RoutingDecision.POSTGRESQL)}")
    print(f"DuckDB: {sum(1 for r in results if r.decision == RoutingDecision.DUCKDB)}")
    print(f"Success rate: {sum(1 for r in results if r.success) / len(results) * 100:.1f}%")


if __name__ == "__main__":
    main()