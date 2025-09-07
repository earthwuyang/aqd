"""
AQD Machine Learning-Based Routing Implementation
Based on the AQD paper's LightGBM classifier with Taylor-weighted boosting
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import psycopg2
import psycopg2.extras
import logging
import time
import json
import pickle
import hashlib
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import warnings

from aqd_cost_router import AQDCostRouter, QueryFeatures, RoutingDecision, RoutingResult

warnings.filterwarnings('ignore')


@dataclass
class TrainingRecord:
    """Training record for ML model"""
    query_hash: str
    query: str
    pg_execution_time: float
    duckdb_execution_time: float
    pg_success: bool
    duckdb_success: bool
    optimal_engine: str  # 'postgresql' or 'duckdb'
    performance_gap: float  # |pg_time - duck_time| / min(pg_time, duck_time)
    features: QueryFeatures
    
    def to_dict(self):
        """Convert to dictionary for serialization"""
        result = asdict(self)
        result['features'] = self.features.to_dict()
        return result


class AQDMLRouter(AQDCostRouter):
    """
    AQD Machine Learning Router using LightGBM
    
    Extends the cost-threshold router with ML-based routing decisions.
    Implements the LightGBM classifier from the AQD paper.
    """
    
    def __init__(self, 
                 pg_host: str = "localhost",
                 pg_port: int = 5432,
                 pg_user: str = "postgres", 
                 pg_password: str = "postgres",
                 pg_database: str = "aqd_test",
                 model_path: Optional[str] = None):
        """
        Initialize AQD ML Router
        
        Args:
            pg_host: PostgreSQL host
            pg_port: PostgreSQL port
            pg_user: PostgreSQL username
            pg_password: PostgreSQL password
            pg_database: PostgreSQL database name
            model_path: Path to trained LightGBM model
        """
        super().__init__(pg_host, pg_port, pg_user, pg_password, pg_database)
        
        # ML model components
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.model_path = model_path
        
        # Training data storage
        self.training_data: List[TrainingRecord] = []
        
        # Load model if path provided
        if model_path:
            self.load_model(model_path)
    
    def extract_feature_vector(self, features: QueryFeatures) -> np.ndarray:
        """
        Extract feature vector for ML model
        
        Args:
            features: Query features
            
        Returns:
            Numpy array of normalized features
        """
        # Define feature vector based on AQD paper (142 raw features, reduced to 32)
        feature_vector = np.array([
            # Cost model features (4 features)
            features.startup_cost,
            features.total_cost, 
            features.plan_rows,
            features.plan_width,
            
            # Plan structure features (6 features)
            features.num_joins,
            features.num_filters,
            features.num_aggregates,
            features.num_sorts,
            features.num_limits,
            features.plan_depth,
            
            # Operator counts (5 features)
            features.num_seqscan,
            features.num_indexscan,
            features.num_hashjoin,
            features.num_nestloop,
            features.num_mergejoin,
            
            # Boolean features as integers (3 features)
            int(features.has_groupby),
            int(features.has_window),
            int(features.has_subquery),
            
            # Derived features (14 features)
            features.estimated_selectivity,
            features.total_cost / max(1, features.plan_rows),  # cost per row
            features.num_joins / max(1, features.num_filters) if features.num_filters > 0 else 0,  # join to filter ratio
            np.log1p(features.total_cost),  # log cost
            np.log1p(features.plan_rows),   # log rows
            features.num_aggregates + features.num_sorts,  # total aggregate ops
            features.num_seqscan + features.num_indexscan,  # total scans
            features.num_hashjoin + features.num_nestloop + features.num_mergejoin,  # total joins
            int(features.num_joins > 0),  # has joins
            int(features.num_aggregates > 0),  # has aggregates  
            int(features.total_cost > 1000),  # high cost query
            int(features.plan_rows > 10000),  # high cardinality
            int(features.num_joins > 2),  # complex join query
            int(features.plan_depth > 5),  # deep plan
        ])
        
        # Apply scaling if scaler is available
        if self.scaler is not None:
            feature_vector = self.scaler.transform(feature_vector.reshape(1, -1))[0]
        
        return feature_vector
    
    def collect_training_data(self, queries: List[str], num_samples: int = None) -> List[TrainingRecord]:
        """
        Collect training data by dual execution on both engines
        
        Args:
            queries: List of SQL queries to execute
            num_samples: Maximum number of samples to collect
            
        Returns:
            List of training records
        """
        training_records = []
        
        if num_samples:
            queries = queries[:num_samples]
            
        self.logger.info(f"Collecting training data for {len(queries)} queries...")
        
        for i, query in enumerate(queries):
            try:
                query_hash = hashlib.md5(query.encode()).hexdigest()[:16]
                
                # Extract features
                features = self.extract_query_features(query)
                
                # Execute on PostgreSQL
                pg_time, pg_success, pg_error, pg_rows = self.execute_query_postgresql(query)
                
                # Simulate DuckDB execution (since we don't have actual pg_duckdb)
                # In real implementation, this would be: self.execute_query_duckdb(query)
                duck_time, duck_success, duck_error, duck_rows = self._simulate_duckdb_execution(query, features)
                
                # Determine optimal engine
                if pg_success and duck_success:
                    optimal_engine = "postgresql" if pg_time < duck_time else "duckdb"
                    performance_gap = abs(pg_time - duck_time) / min(pg_time, duck_time)
                elif pg_success:
                    optimal_engine = "postgresql" 
                    performance_gap = 1.0  # Penalty for DuckDB failure
                elif duck_success:
                    optimal_engine = "duckdb"
                    performance_gap = 1.0  # Penalty for PostgreSQL failure
                else:
                    continue  # Skip if both failed
                
                # Create training record
                record = TrainingRecord(
                    query_hash=query_hash,
                    query=query,
                    pg_execution_time=pg_time,
                    duckdb_execution_time=duck_time,
                    pg_success=pg_success,
                    duckdb_success=duck_success,
                    optimal_engine=optimal_engine,
                    performance_gap=performance_gap,
                    features=features
                )
                
                training_records.append(record)
                
                if (i + 1) % 100 == 0:
                    self.logger.info(f"Collected {i + 1}/{len(queries)} training samples")
                    
            except Exception as e:
                self.logger.warning(f"Failed to collect training data for query {i}: {e}")
                continue
        
        self.training_data.extend(training_records)
        self.logger.info(f"Collected {len(training_records)} training records")
        
        return training_records
    
    def _simulate_duckdb_execution(self, query: str, features: QueryFeatures) -> Tuple[float, bool, Optional[str], int]:
        """
        Simulate DuckDB execution for training data collection
        
        In a real implementation, this would execute the query on DuckDB via pg_duckdb.
        For simulation, we model DuckDB performance based on query characteristics.
        
        Args:
            query: SQL query
            features: Query features
            
        Returns:
            Tuple of (execution_time, success, error_message, result_rows)
        """
        try:
            # DuckDB simulation model based on query characteristics
            base_time = 0.001  # Base overhead
            
            # DuckDB excels at analytical queries
            if features.num_aggregates > 0 or features.has_groupby or features.has_window:
                # Analytical advantage: 2-5x faster than PostgreSQL for complex aggregations
                analytical_factor = 0.2 + (features.num_aggregates * 0.05)
                execution_time = base_time + (features.total_cost * 0.0001 * analytical_factor)
            elif features.num_joins > 2:
                # Complex joins: DuckDB's columnar storage helps
                join_factor = 0.3 + (features.num_joins * 0.1)  
                execution_time = base_time + (features.total_cost * 0.0001 * join_factor)
            else:
                # Simple queries: PostgreSQL usually faster
                simple_factor = 1.2 + (features.num_filters * 0.1)
                execution_time = base_time + (features.total_cost * 0.0001 * simple_factor)
            
            # Add some realistic noise
            noise = np.random.normal(1.0, 0.1)
            execution_time *= max(0.1, noise)
            
            # Simulate occasional failures (5% failure rate)
            success = np.random.random() > 0.05
            error_msg = None if success else "Simulated DuckDB execution error"
            
            # Estimate result rows (same as plan rows with some variance)
            result_rows = int(features.plan_rows * np.random.uniform(0.8, 1.2))
            
            return execution_time, success, error_msg, result_rows
            
        except Exception as e:
            return 0.1, False, str(e), 0
    
    def train_model(self, training_records: Optional[List[TrainingRecord]] = None, 
                   test_size: float = 0.2, random_state: int = 42) -> Dict:
        """
        Train LightGBM model using collected training data
        
        Args:
            training_records: Training data (uses self.training_data if None)
            test_size: Proportion of data for testing
            random_state: Random state for reproducibility
            
        Returns:
            Dictionary with training metrics
        """
        if training_records is None:
            training_records = self.training_data
            
        if not training_records:
            raise ValueError("No training data available")
        
        self.logger.info(f"Training LightGBM model on {len(training_records)} samples...")
        
        # Prepare feature matrix and labels
        X = []
        y = []
        weights = []
        
        for record in training_records:
            feature_vector = self.extract_feature_vector(record.features)
            X.append(feature_vector)
            
            # Binary classification: 0 = PostgreSQL, 1 = DuckDB
            label = 1 if record.optimal_engine == "duckdb" else 0
            y.append(label)
            
            # Taylor-weighted boosting: weight by performance gap
            # Emphasize costly mispredictions as described in AQD paper
            weight = 1.0 + record.performance_gap
            weights.append(weight)
        
        X = np.array(X)
        y = np.array(y)
        weights = np.array(weights)
        
        # Feature scaling
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train/test split
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X_scaled, y, weights, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # LightGBM parameters (based on AQD paper)
        lgb_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': random_state
        }
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train, weight=w_train)
        valid_data = lgb.Dataset(X_test, label=y_test, weight=w_test, reference=train_data)
        
        # Train model
        self.model = lgb.train(
            lgb_params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=100,
            callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(0)]
        )
        
        # Evaluate model
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        train_pred_binary = (train_pred > 0.5).astype(int)
        test_pred_binary = (test_pred > 0.5).astype(int)
        
        train_accuracy = accuracy_score(y_train, train_pred_binary)
        test_accuracy = accuracy_score(y_test, test_pred_binary)
        
        # Feature importance
        feature_importance = self.model.feature_importance(importance_type='gain')
        
        metrics = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_importance': feature_importance.tolist(),
            'num_features': len(feature_importance)
        }
        
        self.logger.info(f"Model training complete:")
        self.logger.info(f"  Train accuracy: {train_accuracy:.4f}")
        self.logger.info(f"  Test accuracy: {test_accuracy:.4f}")
        
        return metrics
    
    def make_ml_routing_decision(self, features: QueryFeatures) -> Tuple[RoutingDecision, float]:
        """
        Make routing decision using trained ML model
        
        Args:
            features: Query features
            
        Returns:
            Tuple of (routing_decision, confidence_score)
        """
        if self.model is None:
            raise ValueError("ML model not trained. Call train_model() first.")
        
        # Extract and scale features
        feature_vector = self.extract_feature_vector(features)
        
        # Predict using LightGBM
        prediction = self.model.predict(feature_vector.reshape(1, -1))[0]
        
        # Convert to routing decision
        if prediction > 0.5:
            decision = RoutingDecision.DUCKDB
        else:
            decision = RoutingDecision.POSTGRESQL
            
        # Confidence is distance from decision boundary
        confidence = abs(prediction - 0.5) * 2
        
        return decision, confidence
    
    def route_and_execute_query_ml(self, query: str) -> RoutingResult:
        """
        Route and execute query using ML-based routing
        
        Args:
            query: SQL query string
            
        Returns:
            RoutingResult with execution details
        """
        # Generate query hash
        query_hash = hashlib.md5(query.encode()).hexdigest()[:16]
        
        # Extract features
        features = self.extract_query_features(query)
        
        # Make ML-based routing decision
        decision, confidence = self.make_ml_routing_decision(features)
        
        # Execute query
        if decision == RoutingDecision.POSTGRESQL:
            execution_time, success, error_msg, result_rows = self.execute_query_postgresql(query)
        else:  # RoutingDecision.DUCKDB
            # In real implementation: execute_query_duckdb(query)
            execution_time, success, error_msg, result_rows = self._simulate_duckdb_execution(query, features)
        
        # Log result
        self.logger.info(f"Query {query_hash}: ML-{decision.value} "
                        f"(conf={confidence:.3f}), time={execution_time:.4f}s")
        
        return RoutingResult(
            query_hash=query_hash,
            decision=decision,
            execution_time=execution_time,
            success=success,
            error_message=error_msg,
            features=features,
            result_rows=result_rows
        )
    
    def save_model(self, path: str):
        """Save trained model and scaler"""
        if self.model is None:
            raise ValueError("No trained model to save")
            
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
            
        self.logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model and scaler"""
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data.get('feature_names')
            
            self.logger.info(f"Model loaded from {path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model from {path}: {e}")
            raise
    
    def save_training_data(self, path: str):
        """Save training data to JSON file"""
        training_data_dict = [record.to_dict() for record in self.training_data]
        
        with open(path, 'w') as f:
            json.dump(training_data_dict, f, indent=2)
            
        self.logger.info(f"Training data saved to {path} ({len(self.training_data)} records)")
    
    def load_training_data(self, path: str):
        """Load training data from JSON file"""
        try:
            with open(path, 'r') as f:
                training_data_dict = json.load(f)
            
            self.training_data = []
            for record_dict in training_data_dict:
                # Reconstruct QueryFeatures
                features_dict = record_dict['features']
                features = QueryFeatures(**features_dict)
                
                # Reconstruct TrainingRecord
                record_dict['features'] = features
                record = TrainingRecord(**record_dict)
                self.training_data.append(record)
            
            self.logger.info(f"Training data loaded from {path} ({len(self.training_data)} records)")
            
        except Exception as e:
            self.logger.error(f"Failed to load training data from {path}: {e}")
            raise


def main():
    """Example usage of AQD ML Router"""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize ML router
    router = AQDMLRouter()
    router.connect()
    
    # Example queries for training
    training_queries = [
        # OLTP queries (should favor PostgreSQL)
        "SELECT * FROM users WHERE user_id = 12345 LIMIT 1;",
        "SELECT name, email FROM customers WHERE customer_id = 67890;",
        "UPDATE orders SET status = 'shipped' WHERE order_id = 11111;",
        "INSERT INTO log_events (timestamp, event_type, user_id) VALUES (NOW(), 'login', 12345);",
        
        # OLAP queries (should favor DuckDB)  
        "SELECT region, COUNT(*), AVG(sales), SUM(quantity) FROM transactions GROUP BY region ORDER BY region;",
        "SELECT product_category, EXTRACT(YEAR FROM order_date) as year, SUM(revenue) FROM sales GROUP BY product_category, year;",
        "SELECT customer_id, SUM(amount) FROM orders WHERE order_date > '2024-01-01' GROUP BY customer_id HAVING SUM(amount) > 1000;",
        "SELECT DATE_TRUNC('month', created_at) as month, COUNT(*) FROM events WHERE created_at > '2024-01-01' GROUP BY month ORDER BY month;",
        
        # Mixed queries
        "SELECT o.customer_id, c.name, SUM(o.total_amount) FROM orders o JOIN customers c ON o.customer_id = c.id GROUP BY o.customer_id, c.name;",
        "SELECT p.category, AVG(oi.quantity * oi.unit_price) FROM products p JOIN order_items oi ON p.id = oi.product_id GROUP BY p.category;"
    ]
    
    print("=== AQD ML Router Demo ===\n")
    
    # Step 1: Collect training data
    print("1. Collecting training data...")
    training_records = router.collect_training_data(training_queries)
    print(f"   Collected {len(training_records)} training samples")
    
    # Step 2: Train ML model
    print("\n2. Training LightGBM model...")
    metrics = router.train_model()
    print(f"   Training accuracy: {metrics['train_accuracy']:.4f}")
    print(f"   Test accuracy: {metrics['test_accuracy']:.4f}")
    
    # Step 3: Test ML routing
    print("\n3. Testing ML-based routing...")
    test_queries = [
        "SELECT COUNT(*) FROM users WHERE last_login > '2024-01-01';",
        "SELECT product_id, SUM(sales) FROM transactions GROUP BY product_id ORDER BY SUM(sales) DESC LIMIT 10;",
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query[:50]}...")
        result = router.route_and_execute_query_ml(query)
        print(f"  ML Decision: {result.decision.value}")
        print(f"  Execution time: {result.execution_time:.4f}s")
        print(f"  Success: {result.success}")
    
    # Step 4: Save model and data
    router.save_model("models/aqd_lightgbm_model.pkl")
    router.save_training_data("data/aqd_training_data.json")
    
    router.disconnect()
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()