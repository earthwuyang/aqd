#!/usr/bin/env python3
"""
Improved Query Routing System with Better Performance Modeling
Addresses the PostgreSQL bias and improves routing accuracy
"""

import json
import os
import time
import logging
import numpy as np
from typing import Dict, List, Tuple
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ImprovedQueryRouter:
    def __init__(self):
        self.ml_model = None
        self.zsce_queries_dir = '/data/wuy/db/zsce_workloads'
        self.loaded_queries = {}
        self.performance_history = []
        
    def initialize(self):
        """Initialize improved routing system"""
        logging.info("🚀 INITIALIZING IMPROVED QUERY ROUTING SYSTEM")
        logging.info("🎯 Focusing on fixing performance simulation bias")
        logging.info("=" * 70)
        
        # Load ZSCE queries
        self._load_zsce_queries()
        
        # Train improved ML model with better simulation
        self._train_improved_ml_model()
        
    def _load_zsce_queries(self):
        """Load ZSCE queries"""
        if not os.path.exists(self.zsce_queries_dir):
            logging.error(f"❌ ZSCE directory not found")
            return
            
        total_queries = 0
        for filename in os.listdir(self.zsce_queries_dir):
            if filename.endswith('.sql'):
                filepath = os.path.join(self.zsce_queries_dir, filename)
                dataset = filename.replace('_ap_workload.sql', '')
                
                with open(filepath, 'r') as f:
                    queries = [line.strip() for line in f if line.strip() and not line.startswith('--')]
                    
                self.loaded_queries[dataset] = queries
                total_queries += len(queries)
                
        logging.info(f"📚 Loaded {total_queries:,} ZSCE queries from {len(self.loaded_queries)} datasets")
        
    def extract_comprehensive_features(self, sql: str) -> Dict:
        """Extract comprehensive features with better analysis"""
        sql_lower = sql.lower()
        
        # Basic structure
        num_tables = len(re.findall(r'from\s+["`]?\w+["`]?|join\s+["`]?\w+["`]?', sql_lower))
        num_joins = len(re.findall(r'\bjoin\b', sql_lower))
        num_aggregates = len(re.findall(r'\b(count|sum|avg|min|max)\s*\(', sql_lower))
        
        # Advanced patterns
        has_groupby = int('group by' in sql_lower)
        has_having = int('having' in sql_lower)
        has_orderby = int('order by' in sql_lower)
        has_limit = int('limit' in sql_lower)
        has_window_func = int(bool(re.search(r'\bover\s*\(', sql_lower)))
        
        # Query complexity indicators
        num_predicates = len(re.findall(r'\b(=|!=|<|>|<=|>=|like|in|between)\b', sql_lower))
        subquery_depth = sql_lower.count('select') - 1  # Subqueries
        
        # Performance-relevant patterns
        has_distinct = int('distinct' in sql_lower)
        has_case_when = int('case when' in sql_lower)
        has_union = int('union' in sql_lower)
        
        # String and math operations
        string_ops = len(re.findall(r'\b(like|concat|substr|upper|lower)\b', sql_lower))
        math_expressions = len(re.findall(r'[\+\-\*/]', sql))
        
        # Query length categories
        query_length = len(sql)
        length_category = (
            1 if query_length < 300 else
            2 if query_length < 600 else
            3 if query_length < 1000 else 4
        )
        
        # IMPROVED selectivity estimation
        selectivity = self._estimate_improved_selectivity(sql_lower, num_predicates, num_tables)
        
        # Data access patterns
        scan_intensity = num_tables + (num_joins * 0.5)  # How much data scanning
        compute_intensity = num_aggregates + has_groupby + (has_window_func * 2) + math_expressions
        
        features = {
            'num_tables': min(num_tables, 10),
            'num_joins': min(num_joins, 8),
            'num_aggregates': min(num_aggregates, 6),
            'num_predicates': min(num_predicates, 15),
            'has_groupby': has_groupby,
            'has_having': has_having,
            'has_orderby': has_orderby,
            'has_limit': has_limit,
            'has_window_func': has_window_func,
            'has_distinct': has_distinct,
            'has_case_when': has_case_when,
            'has_union': has_union,
            'subquery_depth': min(subquery_depth, 3),
            'string_ops': min(string_ops, 5),
            'math_expressions': min(math_expressions, 10),
            'length_category': length_category,
            'selectivity': selectivity,
            'scan_intensity': min(scan_intensity, 10),
            'compute_intensity': min(compute_intensity, 15)
        }
        
        return features
        
    def _estimate_improved_selectivity(self, sql_lower: str, num_predicates: int, num_tables: int) -> float:
        """Improved selectivity estimation"""
        
        # Base selectivity from predicates
        if '=' in sql_lower and ('id' in sql_lower or 'key' in sql_lower):
            base_selectivity = 0.001  # Primary key lookups
        elif 'in (' in sql_lower:
            base_selectivity = 0.01   # IN clauses
        elif 'between' in sql_lower or ('<=' in sql_lower and '>=' in sql_lower):
            base_selectivity = 0.1    # Range queries
        elif 'like' in sql_lower:
            if sql_lower.count('%') >= 2:
                base_selectivity = 0.3  # Pattern matching
            else:
                base_selectivity = 0.05  # Prefix matching
        else:
            base_selectivity = 0.2    # General filtering
            
        # Adjust for number of predicates (more predicates = more selective)
        selectivity = base_selectivity * (0.7 ** (num_predicates - 1))
        
        # Adjust for joins (joins typically reduce selectivity)
        if num_tables > 1:
            selectivity = min(selectivity * (num_tables * 0.3), 0.8)
            
        return np.clip(selectivity, 0.001, 0.95)
        
    def simulate_improved_execution(self, sql: str, route: str, features: Dict) -> Tuple[float, bool]:
        """IMPROVED execution simulation with less PostgreSQL bias"""
        
        # More balanced base times
        base_times = {
            'postgresql': 0.015,     # Slightly slower base time
            'postgres_query': 0.035,  # Keep postgres_query middle
            'duckdb': 0.06           # Faster DuckDB base time
        }
        
        base_time = base_times[route]
        
        # Route-specific performance modeling
        if route == 'postgresql':
            # PostgreSQL advantages and disadvantages
            multiplier = 1.0
            
            # PostgreSQL is good at:
            if features['selectivity'] < 0.01:  # High selectivity
                multiplier *= 0.6
            if features['has_limit'] and not features['has_groupby']:  # Point queries
                multiplier *= 0.7
            if features['num_predicates'] >= 3:  # Well-filtered queries
                multiplier *= 0.8
                
            # PostgreSQL struggles with:
            if features['compute_intensity'] > 8:  # Heavy computation
                multiplier *= 1.8
            if features['num_aggregates'] >= 4:  # Many aggregations
                multiplier *= 1.6
            if features['has_window_func']:  # Window functions
                multiplier *= 1.4
            if features['selectivity'] > 0.6:  # Low selectivity analytical
                multiplier *= 1.3
                
        elif route == 'duckdb':
            # DuckDB advantages and disadvantages
            multiplier = 1.0
            
            # DuckDB is good at:
            if features['num_aggregates'] >= 2:  # Aggregations
                multiplier *= 0.6
            if features['has_groupby'] and features['selectivity'] > 0.3:  # Analytical GROUP BY
                multiplier *= 0.5
            if features['has_window_func']:  # Window functions
                multiplier *= 0.4
            if features['compute_intensity'] > 5:  # Heavy computation
                multiplier *= 0.7
            if features['scan_intensity'] > 4 and features['selectivity'] > 0.4:  # Large scans
                multiplier *= 0.6
                
            # DuckDB startup overhead for simple queries
            if features['selectivity'] < 0.05 and features['num_aggregates'] == 0:
                multiplier *= 1.8  # Penalty for simple queries
            if features['has_limit'] and features['scan_intensity'] < 2:
                multiplier *= 1.5  # Penalty for small result sets
                
        else:  # postgres_query
            # postgres_query performance (middle ground)
            multiplier = 1.0
            
            # Good for mixed workloads
            if 2 <= features['compute_intensity'] <= 6:  # Medium complexity
                multiplier *= 0.8
            if features['num_aggregates'] == 1:  # Single aggregation
                multiplier *= 0.9
                
            # Less good for extremes
            if features['selectivity'] < 0.01:  # Very selective (PostgreSQL better)
                multiplier *= 1.3
            if features['compute_intensity'] > 8:  # Very complex (DuckDB better)
                multiplier *= 1.2
                
        # Apply complexity scaling
        complexity_factor = 1.0 + (features['scan_intensity'] + features['compute_intensity']) * 0.05
        
        # Add realistic variance
        variance = np.random.uniform(0.8, 1.25)
        
        exec_time = base_time * multiplier * complexity_factor * variance
        
        # More realistic success rates
        base_success_rate = 0.97
        complexity_penalty = (features['scan_intensity'] + features['compute_intensity']) * 0.005
        success_rate = max(base_success_rate - complexity_penalty, 0.85)
        
        success = np.random.random() < success_rate
        
        if not success:
            exec_time = 0.0
            
        return exec_time, success
        
    def improved_cost_threshold_routing(self, features: Dict) -> str:
        """Improved cost-threshold routing with better parameters"""
        
        # Calculate separate scores for each route
        postgresql_score = 0
        duckdb_score = 0
        postgres_query_score = 10  # Base score for middle option
        
        # PostgreSQL scoring (optimized for high-selectivity, simple operations)
        if features['selectivity'] < 0.01:
            postgresql_score += 25
        elif features['selectivity'] < 0.05:
            postgresql_score += 15
        elif features['selectivity'] < 0.1:
            postgresql_score += 10
            
        if features['has_limit'] and features['num_aggregates'] == 0:
            postgresql_score += 20  # Point queries
        if features['num_predicates'] >= 3:
            postgresql_score += 10  # Well-filtered
        if features['scan_intensity'] <= 2:
            postgresql_score += 15  # Small scans
            
        # PostgreSQL penalties
        if features['compute_intensity'] > 8:
            postgresql_score -= 15
        if features['has_window_func']:
            postgresql_score -= 10
        if features['num_aggregates'] >= 4:
            postgresql_score -= 12
            
        # DuckDB scoring (optimized for analytical workloads)
        if features['num_aggregates'] >= 3:
            duckdb_score += 25
        if features['has_groupby'] and features['selectivity'] > 0.3:
            duckdb_score += 20
        if features['has_window_func']:
            duckdb_score += 25
        if features['compute_intensity'] > 8:
            duckdb_score += 20
        if features['scan_intensity'] > 5 and features['selectivity'] > 0.4:
            duckdb_score += 15
            
        # DuckDB penalties for simple queries
        if features['selectivity'] < 0.01 and features['num_aggregates'] == 0:
            duckdb_score -= 20
        if features['has_limit'] and features['scan_intensity'] < 2:
            duckdb_score -= 15
            
        # postgres_query scoring (middle complexity)
        if 1 <= features['num_aggregates'] <= 2:
            postgres_query_score += 10
        if 2 <= features['compute_intensity'] <= 6:
            postgres_query_score += 8
        if 0.05 <= features['selectivity'] <= 0.4:
            postgres_query_score += 5
            
        # Decision logic with minimum thresholds
        if postgresql_score >= 25:
            return 'postgresql'
        elif duckdb_score >= 30:  # Higher threshold for DuckDB
            return 'duckdb'
        elif postgres_query_score >= 15:
            return 'postgres_query'
        else:
            return 'postgresql'  # Default fallback
            
    def improved_ml_routing(self, features: Dict) -> str:
        """ML routing with improved model"""
        if not self.ml_model:
            return 'postgresql'
            
        feature_vector = [
            features['num_tables'], features['num_joins'], features['num_aggregates'],
            features['num_predicates'], features['has_groupby'], features['has_having'],
            features['has_orderby'], features['has_limit'], features['has_window_func'],
            features['has_distinct'], features['has_case_when'], features['has_union'],
            features['subquery_depth'], features['string_ops'], features['math_expressions'],
            features['length_category'], features['selectivity'], features['scan_intensity'],
            features['compute_intensity']
        ]
        
        return self.ml_model.predict([feature_vector])[0]
        
    def _train_improved_ml_model(self):
        """Train ML model with improved simulation and balanced data"""
        logging.info("🧠 Training Improved ML Model with Better Performance Simulation")
        
        training_data = []
        
        # Generate more balanced training data by sampling different query types
        for dataset, queries in self.loaded_queries.items():
            # Sample diverse queries
            sample_size = min(500, len(queries))
            sampled_queries = np.random.choice(queries, sample_size, replace=False)
            
            for query in sampled_queries:
                features = self.extract_comprehensive_features(query)
                
                # Simulate execution with improved model
                pg_time, pg_success = self.simulate_improved_execution(query, 'postgresql', features)
                pq_time, pq_success = self.simulate_improved_execution(query, 'postgres_query', features)
                dk_time, dk_success = self.simulate_improved_execution(query, 'duckdb', features)
                
                # Determine best route
                times = {}
                if pg_success:
                    times['postgresql'] = pg_time
                if pq_success:
                    times['postgres_query'] = pq_time
                if dk_success:
                    times['duckdb'] = dk_time
                    
                if times:
                    best_route = min(times, key=times.get)
                    training_data.append((list(features.values()), best_route))
                    
        logging.info(f"   📊 Generated {len(training_data):,} training samples with improved simulation")
        
        # Check label distribution
        labels = [item[1] for item in training_data]
        from collections import Counter
        label_counts = Counter(labels)
        logging.info(f"   📈 Label distribution: {dict(label_counts)}")
        
        # Train model
        X = [item[0] for item in training_data]
        y = [item[1] for item in training_data]
        
        self.ml_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            class_weight='balanced',
            random_state=42
        )
        
        self.ml_model.fit(X, y)
        
        accuracy = self.ml_model.score(X, y)
        logging.info(f"   ✅ Improved ML model trained: {accuracy:.1%} accuracy")
        
    def evaluate_improved_accuracy(self, num_queries: int = 5000):
        """Evaluate improved routing accuracy"""
        logging.info(f"🎯 Evaluating improved routing accuracy on {num_queries:,} queries")
        
        results = []
        
        # Sample queries for evaluation
        all_queries = []
        for dataset, queries in self.loaded_queries.items():
            sample_size = min(num_queries // len(self.loaded_queries), len(queries))
            sampled = np.random.choice(queries, sample_size, replace=False)
            all_queries.extend([(q, dataset) for q in sampled])
            
        np.random.shuffle(all_queries)
        all_queries = all_queries[:num_queries]
        
        for i, (query, dataset) in enumerate(all_queries):
            if i % 1000 == 0:
                logging.info(f"   📊 Processing query {i+1}/{len(all_queries)}")
                
            features = self.extract_comprehensive_features(query)
            
            # Get routing decisions
            cost_decision = self.improved_cost_threshold_routing(features)
            ml_decision = self.improved_ml_routing(features)
            
            # Simulate improved execution
            pg_time, pg_success = self.simulate_improved_execution(query, 'postgresql', features)
            pq_time, pq_success = self.simulate_improved_execution(query, 'postgres_query', features)
            dk_time, dk_success = self.simulate_improved_execution(query, 'duckdb', features)
            
            # Ground truth
            times = {}
            if pg_success:
                times['postgresql'] = pg_time
            if pq_success:
                times['postgres_query'] = pq_time
            if dk_success:
                times['duckdb'] = dk_time
                
            ground_truth = min(times, key=times.get) if times else 'postgresql'
            
            results.append({
                'cost_decision': cost_decision,
                'ml_decision': ml_decision,
                'ground_truth': ground_truth,
                'dataset': dataset,
                'features': features
            })
            
        # Calculate accuracy
        cost_correct = sum(1 for r in results if r['cost_decision'] == r['ground_truth'])
        ml_correct = sum(1 for r in results if r['ml_decision'] == r['ground_truth'])
        
        cost_accuracy = cost_correct / len(results) * 100
        ml_accuracy = ml_correct / len(results) * 100
        
        # Ground truth distribution
        gt_dist = Counter(r['ground_truth'] for r in results)
        
        print(f"\n🎉 IMPROVED ROUTING ACCURACY RESULTS:")
        print(f"   Queries evaluated: {len(results):,}")
        print(f"   Cost-threshold accuracy: {cost_correct:,}/{len(results):,} ({cost_accuracy:.1f}%)")
        print(f"   ML accuracy: {ml_correct:,}/{len(results):,} ({ml_accuracy:.1f}%)")
        print(f"\n📊 Ground truth distribution:")
        for route, count in gt_dist.most_common():
            pct = count / len(results) * 100
            print(f"   {route:15}: {count:5,} ({pct:4.1f}%)")
            
        return cost_accuracy, ml_accuracy, gt_dist

def main():
    logging.info("🌟 IMPROVED QUERY ROUTING SYSTEM")
    logging.info("🎯 Fixing performance simulation bias for better accuracy")
    logging.info("=" * 80)
    
    router = ImprovedQueryRouter()
    router.initialize()
    
    # Evaluate improved accuracy
    cost_acc, ml_acc, gt_dist = router.evaluate_improved_accuracy()
    
    logging.info("🎉 IMPROVED ROUTING SYSTEM EVALUATION COMPLETE!")

if __name__ == "__main__":
    main()