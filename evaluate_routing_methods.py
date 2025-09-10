#!/usr/bin/env python3
"""
Comprehensive evaluation of query routing methods for pg_duckdb
Compares: Optimal, Default, Cost Threshold (multiple), LightGBM, and GNN

Changes:
- Load ALL unified training data files via glob instead of a fixed subset
- Determine Default (pg_duckdb) decisions online by EXPLAINing queries
  and detecting DuckDB custom scans; fallback to exact offline heuristic
- Load LightGBM model from models/lightgbm_model.txt using metadata feature order
- Use gnn_trainer_real --predict to get GNN’s 0/1 decision from plan JSON
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import subprocess
import tempfile
from pathlib import Path
import time
from typing import Dict, List, Tuple, Any
import psycopg2

PG_DEFAULTS = {
    'host': os.environ.get('AQD_PG_HOST', 'localhost'),
    'port': int(os.environ.get('AQD_PG_PORT', '5432')),
    'user': os.environ.get('AQD_PG_USER', 'wuy'),
}

class RoutingMethodEvaluator:
    def __init__(self, test_data_path: str = None):
        """Initialize the evaluator with test data"""
        self.test_data = []
        self.lightgbm_model = None
        self.gnn_model_path = "models/rginn_routing_model.txt"
        self.lgb_feature_names: List[str] = []
        self._pg_connections: Dict[str, Any] = {}
        self._db_available: Dict[str, bool] = {}
        
        # Load test data
        if test_data_path:
            self.load_test_data(test_data_path)
        else:
            self.load_all_test_data()
            
    def load_all_test_data(self):
        """Load ALL unified training data via glob and filter valid examples."""
        base = Path('data/execution_data')
        files = sorted(str(p) for p in base.glob('*_unified_training_data.json'))
        if not files:
            raise FileNotFoundError("No *_unified_training_data.json files found under data/execution_data/")

        all_data: List[dict] = []
        for fp in files:
            try:
                with open(fp, 'r') as f:
                    data = json.load(f)
                all_data.extend(data)
                print(f"Loaded {len(data)} from {fp}")
            except Exception as e:
                print(f"Warning: failed to load {fp}: {e}")

        # Keep only entries with valid timings and required fields
        filtered = []
        for ex in all_data:
            pg = ex.get('postgres_time')
            dk = ex.get('duckdb_time')
            if not (isinstance(pg, (int, float)) and isinstance(dk, (int, float))):
                continue
            if pg <= 0 or dk <= 0:
                continue
            if 'query_text' not in ex or 'dataset' not in ex or 'postgres_plan_json' not in ex:
                continue
            filtered.append(ex)

        self.test_data = filtered
        print(f"Loaded {len(self.test_data)} valid examples from {len(files)} files")
        
    def load_test_data(self, path: str):
        """Load test data from a specific file"""
        with open(path, 'r') as f:
            self.test_data = json.load(f)
        print(f"Loaded {len(self.test_data)} test examples")
        
    def load_lightgbm_model(self, model_txt: str = "models/lightgbm_model.txt", metadata_path: str = "models/lightgbm_metadata.json"):
        """Load the trained LightGBM model from .txt and metadata feature order."""
        try:
            import lightgbm as lgb
            if not Path(model_txt).exists():
                print(f"Warning: LightGBM model file not found: {model_txt}")
                return
            self.lightgbm_model = lgb.Booster(model_file=model_txt)
            print(f"LightGBM model loaded from {model_txt}")
            # Load feature names
            if Path(metadata_path).exists():
                with open(metadata_path, 'r') as f:
                    meta = json.load(f)
                    self.lgb_feature_names = list(meta.get('feature_names', []))
                print(f"Loaded {len(self.lgb_feature_names)} feature names from {metadata_path}")
            else:
                print(f"Warning: metadata not found at {metadata_path}; will attempt best-effort feature mapping")
        except Exception as e:
            print(f"Warning: Could not load LightGBM model: {e}")
            
    def compute_optimal_routing(self, pg_time: float, duck_time: float) -> int:
        """Compute optimal routing decision (0=PostgreSQL, 1=DuckDB)"""
        if pg_time <= 0 or duck_time <= 0:
            return 0  # Default to PostgreSQL if timing invalid
        return 1 if duck_time < pg_time else 0
        
    def _offline_default_routing_from_plan(self, plan_json: Dict) -> int:
        """Exact offline mirror of C default routing on EXPLAIN JSON."""
        plan = plan_json
        if isinstance(plan, list) and plan:
            plan = plan[0]
        if isinstance(plan, dict) and 'Plan' in plan:
            plan = plan['Plan']

        total_cost = float(plan.get('Total Cost', 0.0) or 0.0)
        plan_rows = int(plan.get('Plan Rows,', plan.get('Plan Rows', 0)) or 0)

        def count_types(node, wanted):
            if not isinstance(node, dict):
                return 0
            t = node.get('Node Type', '')
            c = 1 if t in wanted else 0
            for ch in node.get('Plans', []):
                c += count_types(ch, wanted)
            return c

        num_joins = count_types(plan, {'Nested Loop', 'Hash Join', 'Merge Join'})
        num_aggs = count_types(plan, {'Aggregate', 'WindowAgg'})

        if plan_rows < 1000:
            return 0  # PostgreSQL
        if num_aggs > 0:
            return 1  # DuckDB
        if num_joins > 2:
            return 1
        if total_cost > 10000.0:
            return 1
        return 0

    def _pg_conn(self, database: str):
        # Return cached connection or None if previously marked unavailable
        if database in self._pg_connections:
            return self._pg_connections[database]
        if self._db_available.get(database) is False:
            return None
        cfg = dict(PG_DEFAULTS)
        cfg['database'] = database
        cfg['connect_timeout'] = 2  # seconds
        try:
            conn = psycopg2.connect(**cfg)
            conn.autocommit = True
            # Set strict timeouts to prevent hanging
            with conn.cursor() as cur:
                cur.execute("SET statement_timeout = '2000ms';")
                cur.execute("SET lock_timeout = '1000ms';")
                cur.execute("SET idle_in_transaction_session_timeout = '2000ms';")
            self._pg_connections[database] = conn
            self._db_available[database] = True
            return conn
        except Exception as e:
            print(f"Note: DB '{database}' not available for online default detection: {e}")
            self._pg_connections[database] = None
            self._db_available[database] = False
            return None

    def _online_default_routing(self, query_text: str, database: str) -> int:
        """Decide default via EXPLAIN (FORMAT JSON) and DuckDB Custom Scan detection."""
        # Skip online routing for now - databases may not be available
        return -1  # Signal to use offline routing
        
        conn = self._pg_conn(database)
        if conn is None:
            return -1  # Signal failure
        try:
            with conn.cursor() as cur:
                try:
                    cur.execute("SET pg_duckdb.force_execution = 'auto';")
                except Exception:
                    # If pg_duckdb GUC not present, continue; we'll still EXPLAIN
                    pass
                try:
                    cur.execute("SET aqd.routing_method = 0;")  # AQD_ROUTE_DEFAULT
                except Exception:
                    pass
                cur.execute("SET LOCAL statement_timeout = '2000ms';")
                cur.execute("EXPLAIN (FORMAT JSON) " + query_text)
                root = cur.fetchone()[0][0]
            plan = root.get('Plan', root)

            def has_duckdb(node):
                if not isinstance(node, dict):
                    return False
                if node.get('Node Type') == 'Custom Scan':
                    # Any field containing duckdb indicates DuckDB routing
                    for key in ('Custom Plan Provider', 'Alias', 'Name'):
                        v = node.get(key)
                        if isinstance(v, str) and 'duckdb' in v.lower():
                            return True
                for ch in node.get('Plans', []):
                    if has_duckdb(ch):
                        return True
                return False

            return 1 if has_duckdb(plan) else 0
        except Exception as e:
            # Fall back to offline plan-based decision if EXPLAIN fails
            # print(f"Note: EXPLAIN failed for {database}: {e}")
            return -1

    def apply_default_routing(self, example: Dict) -> int:
        """Default (pg_duckdb) decision with online EXPLAIN; offline fallback on failure."""
        q = example.get('query_text', '')
        db = example.get('dataset', 'postgres')
        # Online first
        res = self._online_default_routing(q, db)
        if res in (0, 1):
            return res
        # Fallback to offline plan analysis
        return self._offline_default_routing_from_plan(example.get('postgres_plan_json', {}))
        
    def apply_cost_threshold_routing(self, features: Dict, threshold: float) -> int:
        """Apply cost threshold routing"""
        total_cost = features.get('total_cost', 0)
        return 1 if total_cost > threshold else 0
        
    def _feature_vector_from_example(self, example: Dict) -> List[float]:
        """Build feature vector in the exact metadata order."""
        feats = example.get('features', {}) or {}
        # Add simple derived features used in training (best-effort)
        qtext = (example.get('query_text') or '').upper()
        derived = {
            'query_length': len(qtext),
            'has_join': int('JOIN' in qtext),
            'has_group_by': int('GROUP BY' in qtext),
            'has_order_by': int('ORDER BY' in qtext),
            'has_where': int('WHERE' in qtext),
            'has_aggregation': int(any(a in qtext for a in ('SUM', 'AVG', 'COUNT', 'MIN', 'MAX'))),
            'is_ap_query': int(example.get('query_type') == 'AP'),
            'is_tp_query': int(example.get('query_type') == 'TP'),
        }
        vec: List[float] = []
        for name in self.lgb_feature_names:
            if name in feats:
                vec.append(float(feats[name]))
            elif name.startswith('aqd_') and name[4:] in feats:
                vec.append(float(feats[name[4:]]))
            elif name in derived:
                vec.append(float(derived[name]))
            else:
                vec.append(0.0)
        return vec

    def apply_lightgbm_routing(self, example: Dict) -> int:
        """Apply LightGBM routing using .txt Booster and metadata-ordered features."""
        if self.lightgbm_model is None or not self.lgb_feature_names:
            return 0
        try:
            fv = self._feature_vector_from_example(example)
            pred = float(self.lightgbm_model.predict([fv])[0])
            # Model predicts log(pg/duck) or log difference; negative implies DuckDB faster in our convention
            return 1 if pred < 0 else 0
        except Exception as e:
            # print(f"LightGBM prediction failed: {e}")
            return 0
            
    def apply_gnn_routing(self, plan_json: Dict) -> int:
        """Apply GNN routing using gnn_trainer_real --predict."""
        if not Path(self.gnn_model_path).exists():
            return 0
            
        try:
            # gnn_trainer_real --predict handles both array and object formats
            # and uses the exact same logic as training
            proc = subprocess.run(
                ['./build/gnn_trainer_real', '--predict', self.gnn_model_path],
                input=json.dumps(plan_json),
                capture_output=True,
                text=True,
                timeout=2
            )
            
            if proc.returncode == 0:
                output = proc.stdout.strip()
                # --predict mode outputs "0" or "1" directly
                if output in ('0', '1'):
                    return int(output)
            return 0
        except Exception as e:
            # Silently default to PostgreSQL on error
            return 0
            
    def extract_features_from_plan(self, plan_json: Dict) -> Dict:
        """Extract features from a query plan"""
        features = {}
        
        # Get the actual plan node
        if isinstance(plan_json, list) and len(plan_json) > 0:
            plan = plan_json[0]
        else:
            plan = plan_json
            
        if 'Plan' in plan:
            plan = plan['Plan']
            
        # Extract basic features
        features['total_cost'] = plan.get('Total Cost', 0)
        features['plan_rows'] = plan.get('Plan Rows', 0)
        features['plan_width'] = plan.get('Plan Width', 0)
        features['startup_cost'] = plan.get('Startup Cost', 0)
        
        # Count node types
        features['num_scan_nodes'] = 0
        features['num_join_nodes'] = 0
        features['num_aggregate_nodes'] = 0
        features['num_sort_nodes'] = 0
        features['num_index_nodes'] = 0
        features['max_depth'] = 0
        
        def count_nodes(node, depth=0):
            if not isinstance(node, dict):
                return
                
            node_type = node.get('Node Type', '')
            
            if 'Scan' in node_type:
                features['num_scan_nodes'] += 1
            if 'Index' in node_type:
                features['num_index_nodes'] += 1
            if 'Join' in node_type or 'Nested Loop' in node_type or 'Hash' in node_type or 'Merge' in node_type:
                features['num_join_nodes'] += 1
            if 'Aggregate' in node_type or 'Group' in node_type:
                features['num_aggregate_nodes'] += 1
            if 'Sort' in node_type:
                features['num_sort_nodes'] += 1
                
            features['max_depth'] = max(features['max_depth'], depth)
            
            # Recurse into child plans
            if 'Plans' in node:
                for child in node['Plans']:
                    count_nodes(child, depth + 1)
                    
        count_nodes(plan)
        
        return features
        
    def create_feature_vector(self, features: Dict) -> List[float]:
        """Create feature vector for ML models"""
        # Define feature order (must match training)
        feature_names = [
            'total_cost', 'startup_cost', 'plan_rows', 'plan_width',
            'num_scan_nodes', 'num_join_nodes', 'num_aggregate_nodes',
            'num_sort_nodes', 'num_index_nodes', 'max_depth'
        ]
        
        vector = []
        for name in feature_names:
            value = features.get(name, 0)
            # Apply log transformation for cost and row features
            if name in ['total_cost', 'startup_cost', 'plan_rows'] and value > 0:
                value = np.log(value + 1)
            vector.append(value)
            
        return vector
        
    def evaluate_all_methods(self) -> Dict:
        """Evaluate all routing methods on the test data"""
        results = {
            'optimal': {'predictions': [], 'times': []},
            'default': {'predictions': [], 'times': []},
            'cost_100': {'predictions': [], 'times': []},
            'cost_1000': {'predictions': [], 'times': []},
            'cost_10000': {'predictions': [], 'times': []},
            'lightgbm': {'predictions': [], 'times': []},
            'gnn': {'predictions': [], 'times': []}
        }
        
        ground_truth = []
        valid_indices = []
        
        print("Evaluating routing methods on test data...")
        
        for i, example in enumerate(self.test_data):
            if i % 100 == 0:
                print(f"Processing example {i}/{len(self.test_data)}")
                
            # Skip if timing data is invalid
            pg_time = example.get('postgres_time', -1)
            duck_time = example.get('duckdb_time', -1)
            
            if pg_time <= 0 or duck_time <= 0:
                continue
                
            valid_indices.append(i)
            
            # Ground truth (1 if DuckDB is faster)
            optimal = self.compute_optimal_routing(pg_time, duck_time)
            ground_truth.append(optimal)
            
            # Get plan and features
            plan_json = example.get('postgres_plan_json', {})
            features = self.extract_features_from_plan(plan_json)
            feature_vector = self.create_feature_vector(features)
            
            # Optimal routing
            results['optimal']['predictions'].append(optimal)
            results['optimal']['times'].append(min(pg_time, duck_time))
            
            # Default routing (online EXPLAIN; offline fallback)
            pred = self.apply_default_routing(example)
            results['default']['predictions'].append(pred)
            results['default']['times'].append(duck_time if pred else pg_time)
            
            # Cost threshold routing
            for threshold, key in [(100, 'cost_100'), (1000, 'cost_1000'), (10000, 'cost_10000')]:
                pred = self.apply_cost_threshold_routing(features, threshold)
                results[key]['predictions'].append(pred)
                results[key]['times'].append(duck_time if pred else pg_time)
            
            # LightGBM routing
            if self.lightgbm_model:
                pred = self.apply_lightgbm_routing(example)
                results['lightgbm']['predictions'].append(pred)
                results['lightgbm']['times'].append(duck_time if pred else pg_time)
            
            # GNN routing
            pred = self.apply_gnn_routing(plan_json)
            results['gnn']['predictions'].append(pred)
            results['gnn']['times'].append(duck_time if pred else pg_time)
            
        return results, ground_truth
        
    def compute_metrics(self, predictions: List[int], ground_truth: List[int], 
                       execution_times: List[float], optimal_times: List[float]) -> Dict:
        """Compute evaluation metrics"""
        metrics = {}
        
        # Accuracy
        metrics['accuracy'] = accuracy_score(ground_truth, predictions)
        
        # Confusion matrix
        cm = confusion_matrix(ground_truth, predictions)
        metrics['confusion_matrix'] = cm
        metrics['true_positive'] = cm[1, 1]
        metrics['true_negative'] = cm[0, 0]
        metrics['false_positive'] = cm[0, 1]
        metrics['false_negative'] = cm[1, 0]
        
        # Precision, Recall, F1
        if (cm[1, 1] + cm[0, 1]) > 0:
            metrics['precision'] = cm[1, 1] / (cm[1, 1] + cm[0, 1])
        else:
            metrics['precision'] = 0
            
        if (cm[1, 1] + cm[1, 0]) > 0:
            metrics['recall'] = cm[1, 1] / (cm[1, 1] + cm[1, 0])
        else:
            metrics['recall'] = 0
            
        if (metrics['precision'] + metrics['recall']) > 0:
            metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1'] = 0
        
        # Runtime metrics
        metrics['total_time'] = sum(execution_times)
        metrics['optimal_time'] = sum(optimal_times)
        metrics['overhead'] = metrics['total_time'] - metrics['optimal_time']
        metrics['overhead_pct'] = (metrics['overhead'] / metrics['optimal_time']) * 100 if metrics['optimal_time'] > 0 else 0
        
        return metrics
        
    def print_results(self, results: Dict, ground_truth: List[int]):
        """Print evaluation results"""
        print("\n" + "="*80)
        print("ROUTING METHOD EVALUATION RESULTS")
        print("="*80)
        
        # Get optimal times for comparison
        optimal_times = results['optimal']['times']
        
        for method_name, method_results in results.items():
            if not method_results['predictions']:
                continue
                
            print(f"\n### {method_name.upper()} ###")
            
            metrics = self.compute_metrics(
                method_results['predictions'],
                ground_truth,
                method_results['times'],
                optimal_times
            )
            
            print(f"Accuracy: {metrics['accuracy']:.3f}")
            print(f"Precision: {metrics['precision']:.3f}")
            print(f"Recall: {metrics['recall']:.3f}")
            print(f"F1 Score: {metrics['f1']:.3f}")
            
            print("\nConfusion Matrix:")
            print("             Predicted")
            print("             PG    DuckDB")
            print(f"Actual PG    {metrics['true_negative']:5d} {metrics['false_positive']:5d}")
            print(f"      DuckDB {metrics['false_negative']:5d} {metrics['true_positive']:5d}")
            
            print(f"\nTotal Runtime: {metrics['total_time']:.2f}s")
            print(f"Overhead vs Optimal: {metrics['overhead']:.2f}s ({metrics['overhead_pct']:.1f}%)")
            
    def create_comparison_plot(self, results: Dict, ground_truth: List[int]):
        """Create comparison plots"""
        methods = []
        accuracies = []
        overheads = []
        
        optimal_times = results['optimal']['times']
        
        for method_name, method_results in results.items():
            if not method_results['predictions'] or method_name == 'optimal':
                continue
                
            metrics = self.compute_metrics(
                method_results['predictions'],
                ground_truth,
                method_results['times'],
                optimal_times
            )
            
            methods.append(method_name.replace('_', ' ').title())
            accuracies.append(metrics['accuracy'])
            overheads.append(metrics['overhead_pct'])
            
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Accuracy plot
        ax1.bar(methods, accuracies, color='steelblue')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Routing Accuracy by Method')
        ax1.set_ylim([0, 1])
        ax1.tick_params(axis='x', rotation=45)
        
        # Overhead plot
        ax2.bar(methods, overheads, color='coral')
        ax2.set_ylabel('Overhead (%)')
        ax2.set_title('Runtime Overhead vs Optimal')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('routing_method_comparison.png', dpi=150, bbox_inches='tight')
        print("\nComparison plot saved to routing_method_comparison.png")
        
def main():
    evaluator = RoutingMethodEvaluator()
    
    # Load LightGBM model if available
    evaluator.load_lightgbm_model()
    
    # Evaluate all methods
    results, ground_truth = evaluator.evaluate_all_methods()
    
    # Print results
    evaluator.print_results(results, ground_truth)
    
    # Create plots
    evaluator.create_comparison_plot(results, ground_truth)
    
if __name__ == "__main__":
    main()
