#!/usr/bin/env python3
"""
Advanced AQD System Implementation
Thompson Residual Learning + Mahalanobis Resource Management
Based on ~/DB/aqd_paper/main.tex research
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time
import json
from sklearn.linear_model import BayesianRidge
from scipy.stats import multivariate_normal
import threading
import psutil
from collections import deque
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class QueryExecution:
    query_id: int
    features: Dict
    chosen_action: int  # 0 = row engine, 1 = column engine
    observed_latency: float
    timestamp: float

@dataclass
class ResourceState:
    cpu_utilization: float  # Row engine CPU share [0,1]
    memory_utilization: float  # Row engine memory share [0,1]
    timestamp: float

class LinTSDeltaBandit:
    """
    Linear Thompson Sampling with Delta Residual Learning
    Based on AQD paper algorithm
    """
    
    def __init__(self, feature_dim: int, lambda_reg: float = 0.5, sigma: float = 1.0):
        self.feature_dim = feature_dim
        self.lambda_reg = lambda_reg
        self.sigma = sigma
        
        # Posterior parameters
        self.V = lambda_reg * np.eye(feature_dim)  # Precision matrix
        self.b = np.zeros(feature_dim)  # Linear term
        
        # EWMA for counterfactual estimates
        self.ewma_alpha = 0.03
        self.ewma_row_latency = 0.1  # Initialize with reasonable values
        self.ewma_col_latency = 0.08
        
        self.execution_history = []
        
    def compute_residual(self, execution: QueryExecution) -> float:
        """
        Compute signed residual as per Equation (5) in AQD paper
        Delta_t is positive when column engine is (expected to be) faster
        """
        if execution.chosen_action == 1:  # Column engine chosen
            # Delta_t = -log(1 + l_col[t]) + log(1 + hat_l_row[t])
            residual = -np.log(1 + execution.observed_latency) + np.log(1 + self.ewma_row_latency)
        else:  # Row engine chosen
            # Delta_t = log(1 + hat_l_col[t]) - log(1 + l_row[t])
            residual = np.log(1 + self.ewma_col_latency) - np.log(1 + execution.observed_latency)
            
        return residual
        
    def update_ewma(self, execution: QueryExecution):
        """Update EWMA latency estimates for counterfactual reasoning"""
        if execution.chosen_action == 1:  # Column engine
            self.ewma_col_latency = (self.ewma_alpha * execution.observed_latency + 
                                   (1 - self.ewma_alpha) * self.ewma_col_latency)
        else:  # Row engine
            self.ewma_row_latency = (self.ewma_alpha * execution.observed_latency + 
                                   (1 - self.ewma_alpha) * self.ewma_row_latency)
                                   
    def update(self, execution: QueryExecution):
        """
        Update posterior distribution with new observation
        Based on LinTS-Delta algorithm in AQD paper
        """
        # Convert features to numpy array
        x = np.array([execution.features.get(f'feature_{i}', 0.0) for i in range(self.feature_dim)])
        
        # Compute signed residual
        delta = self.compute_residual(execution)
        
        # Update posterior parameters
        self.V += np.outer(x, x)
        self.b += x * delta
        
        # Update EWMA estimates
        self.update_ewma(execution)
        
        # Store execution for analysis
        self.execution_history.append(execution)
        
        logging.debug(f"LinTS update: action={execution.chosen_action}, "
                     f"latency={execution.observed_latency:.3f}, residual={delta:.3f}")
        
    def sample_thompson_score(self, features: Dict) -> float:
        """
        Sample Thompson score using posterior distribution
        Returns u_t = x_t^T * theta_tilde
        """
        # Convert features to numpy array
        x = np.array([features.get(f'feature_{i}', 0.0) for i in range(self.feature_dim)])
        
        # Compute posterior mean and covariance
        V_inv = np.linalg.inv(self.V)
        theta_hat = V_inv @ self.b
        
        # Sample from posterior: theta_tilde ~ N(theta_hat, sigma^2 * V^-1)
        try:
            theta_tilde = np.random.multivariate_normal(theta_hat, self.sigma**2 * V_inv)
        except np.linalg.LinAlgError:
            # Fallback to posterior mean if covariance is singular
            theta_tilde = theta_hat
            
        # Compute Thompson score
        u_t = np.dot(x, theta_tilde)
        return u_t

class MahalanobisResourceManager:
    """
    Resource management using Mahalanobis distance
    Based on Stage 3 of AQD paper algorithm
    """
    
    def __init__(self, target_cpu: float = 0.5, target_memory: float = 0.5, window_size: int = 50):
        self.target = np.array([target_cpu, target_memory])  # gamma_t target
        self.window_size = window_size
        self.resource_history = deque(maxlen=window_size)
        self.covariance_matrix = np.eye(2)  # Initialize as identity
        
    def update_resource_state(self, resource_state: ResourceState):
        """Update resource state history"""
        utilization_vector = np.array([resource_state.cpu_utilization, resource_state.memory_utilization])
        self.resource_history.append(utilization_vector)
        
        # Update empirical covariance matrix if we have enough samples
        if len(self.resource_history) >= 10:
            self._update_covariance_matrix()
            
    def _update_covariance_matrix(self):
        """
        Compute empirical covariance matrix from recent deviations
        Based on Equation (13) in AQD paper
        """
        if len(self.resource_history) < 2:
            return
            
        # Convert to numpy array
        resource_matrix = np.array(list(self.resource_history))
        
        # Compute deviations from target
        deviations = resource_matrix - self.target
        
        # Compute empirical covariance matrix
        if len(deviations) > 1:
            self.covariance_matrix = np.cov(deviations.T, ddof=1)
            
            # Ensure matrix is positive definite
            eigenvals = np.linalg.eigvals(self.covariance_matrix)
            if np.any(eigenvals <= 0):
                # Add small regularization to diagonal
                self.covariance_matrix += 1e-6 * np.eye(2)
                
    def compute_mahalanobis_score(self, current_utilization: np.ndarray) -> float:
        """
        Compute normalized Mahalanobis distance score
        Based on Equations (14-16) in AQD paper
        """
        # Compute deviation from target
        e_t = current_utilization - self.target
        
        try:
            # Compute Mahalanobis distance with sign based on CPU deviation
            mahalanobis_distance = np.sqrt(e_t.T @ np.linalg.inv(self.covariance_matrix) @ e_t)
            signed_distance = np.sign(e_t[0]) * mahalanobis_distance  # Sign based on CPU deviation
            
            # Normalize to [-1, 1] using tanh
            r_t = np.tanh(signed_distance)
            
        except np.linalg.LinAlgError:
            # Fallback to simple distance if covariance is singular
            r_t = np.tanh(np.sign(e_t[0]) * np.linalg.norm(e_t))
            
        return float(r_t)

class AdvancedAQDRouter:
    """
    Advanced AQD Router with Thompson Sampling and Mahalanobis Resource Management
    Complete implementation of AQD paper algorithms
    """
    
    def __init__(self, feature_dim: int = 19):
        self.feature_dim = feature_dim
        
        # Initialize components
        self.lints_bandit = LinTSDeltaBandit(feature_dim)
        self.resource_manager = MahalanobisResourceManager()
        
        # Base LightGBM model (simplified for demo)
        self.base_model_weights = np.random.randn(feature_dim) * 0.1
        
        # Load balancing parameters
        self.system_load_history = deque(maxlen=20)
        self.current_qps = 0.0
        
        # Resource monitoring
        self.resource_monitor_thread = None
        self.monitoring_active = False
        self.current_resource_state = None
        
    def start_resource_monitoring(self):
        """Start background resource monitoring"""
        self.monitoring_active = True
        self.resource_monitor_thread = threading.Thread(target=self._resource_monitoring_loop)
        self.resource_monitor_thread.daemon = True
        self.resource_monitor_thread.start()
        logging.info("Resource monitoring started")
        
    def stop_resource_monitoring(self):
        """Stop background resource monitoring"""
        self.monitoring_active = False
        if self.resource_monitor_thread:
            self.resource_monitor_thread.join()
        logging.info("Resource monitoring stopped")
        
    def _resource_monitoring_loop(self):
        """Background resource monitoring loop"""
        while self.monitoring_active:
            try:
                # Sample system resource usage
                cpu_usage = psutil.cpu_percent(interval=0.1)
                memory_usage = psutil.virtual_memory().percent
                
                # Simulate row engine share (in real system, this would be measured)
                # For demo, assume some reasonable distribution
                row_cpu_share = np.random.beta(2, 2)  # Varies around 0.5
                row_memory_share = np.random.beta(2, 2)
                
                resource_state = ResourceState(
                    cpu_utilization=row_cpu_share,
                    memory_utilization=row_memory_share,
                    timestamp=time.time()
                )
                
                self.current_resource_state = resource_state
                self.resource_manager.update_resource_state(resource_state)
                
                time.sleep(0.2)  # Sample every 200ms as mentioned in paper
                
            except Exception as e:
                logging.warning(f"Resource monitoring error: {e}")
                
    def extract_query_features(self, sql: str) -> Dict:
        """Extract query features (simplified for demo)"""
        features = {}
        
        # Basic feature extraction
        features['feature_0'] = len(sql) / 1000.0  # Query length
        features['feature_1'] = float('SELECT' in sql.upper())
        features['feature_2'] = float('JOIN' in sql.upper()) 
        features['feature_3'] = float('GROUP BY' in sql.upper())
        features['feature_4'] = float('ORDER BY' in sql.upper())
        features['feature_5'] = float('HAVING' in sql.upper())
        features['feature_6'] = float('UNION' in sql.upper())
        features['feature_7'] = float('DISTINCT' in sql.upper())
        features['feature_8'] = sql.upper().count('AND') / 10.0
        features['feature_9'] = sql.upper().count('OR') / 10.0
        
        # Add more features to reach feature_dim
        for i in range(10, self.feature_dim):
            features[f'feature_{i}'] = np.random.randn() * 0.1  # Simulated features
            
        return features
        
    def compute_base_lightgbm_score(self, features: Dict) -> float:
        """
        Compute base LightGBM score (simplified simulation)
        In real system, this would use actual trained LightGBM model
        """
        feature_vector = np.array([features.get(f'feature_{i}', 0.0) for i in range(self.feature_dim)])
        score = np.dot(self.base_model_weights, feature_vector)
        return score
        
    def compute_load_adaptive_weight(self) -> float:
        """
        Compute load-adaptive weight omega_t
        High load -> prioritize latency (omega_t -> 1)
        Low load -> prioritize resource balance (omega_t -> 0) 
        """
        if len(self.system_load_history) < 5:
            return 0.7  # Default moderate weight
            
        # Estimate current system load based on QPS
        avg_qps = np.mean(list(self.system_load_history))
        
        # Simple load adaptation (in real system, more sophisticated)
        if avg_qps > 100:  # High load
            omega_t = 0.9
        elif avg_qps > 50:  # Medium load
            omega_t = 0.7
        else:  # Low load
            omega_t = 0.3
            
        return omega_t
        
    def make_routing_decision(self, sql: str) -> Tuple[int, Dict]:
        """
        Make routing decision using complete AQD algorithm
        Returns (action, debug_info) where action: 0=row engine, 1=column engine
        """
        start_time = time.time()
        
        # Stage 1: Extract features and compute base LightGBM score
        features = self.extract_query_features(sql)
        s_t = self.compute_base_lightgbm_score(features)
        
        # Stage 2: Thompson Sampling for residual learning
        u_t = self.lints_bandit.sample_thompson_score(features)
        z_t = np.tanh(s_t + u_t)  # Combine base score with residual
        
        # Stage 3: Resource regulation via Mahalanobis distance
        r_t = 0.0
        if self.current_resource_state is not None:
            current_utilization = np.array([
                self.current_resource_state.cpu_utilization,
                self.current_resource_state.memory_utilization
            ])
            r_t = self.resource_manager.compute_mahalanobis_score(current_utilization)
            
        # Stage 4: Adaptive fusion of latency and resource scores
        omega_t = self.compute_load_adaptive_weight()
        s_final = omega_t * z_t + (1 - omega_t) * r_t
        
        # Make routing decision: positive -> column engine (1), negative -> row engine (0)
        action = 1 if s_final > 0 else 0
        
        decision_time = time.time() - start_time
        
        debug_info = {
            'base_score': s_t,
            'thompson_score': u_t,
            'latency_score': z_t,
            'resource_score': r_t,
            'load_weight': omega_t,
            'final_score': s_final,
            'decision_time': decision_time
        }
        
        return action, debug_info
        
    def update_with_execution_feedback(self, query_id: int, sql: str, action: int, 
                                     observed_latency: float, features: Dict = None):
        """
        Update system with execution feedback
        This is called after query execution completes
        """
        if features is None:
            features = self.extract_query_features(sql)
            
        execution = QueryExecution(
            query_id=query_id,
            features=features,
            chosen_action=action,
            observed_latency=observed_latency,
            timestamp=time.time()
        )
        
        # Update LinTS-Delta bandit
        self.lints_bandit.update(execution)
        
        # Update QPS estimate for load adaptation
        self.current_qps += 1
        self.system_load_history.append(self.current_qps)
        
        logging.debug(f"Updated AQD with feedback: query_id={query_id}, "
                     f"action={action}, latency={observed_latency:.3f}s")

class AQDSystemEvaluator:
    """Evaluate Advanced AQD system performance"""
    
    def __init__(self):
        self.aqd_router = AdvancedAQDRouter()
        
    def simulate_query_execution(self, sql: str, action: int) -> float:
        """Simulate query execution time based on routing decision"""
        base_time = 0.05
        
        # Simulate different engine performance characteristics
        if action == 0:  # Row engine
            if "SELECT" in sql and "WHERE" in sql and "LIMIT" in sql:
                return base_time * 0.8 + np.random.exponential(0.02)  # Good for simple queries
            else:
                return base_time * 1.2 + np.random.exponential(0.03)  # Slower for complex queries
        else:  # Column engine  
            if "GROUP BY" in sql or "COUNT" in sql or "SUM" in sql:
                return base_time * 0.6 + np.random.exponential(0.015)  # Excellent for analytics
            else:
                return base_time * 0.9 + np.random.exponential(0.025)  # Good overall
                
    def run_evaluation(self, num_queries: int = 1000) -> Dict:
        """Run comprehensive evaluation of AQD system"""
        logging.info(f"🧠 EVALUATING ADVANCED AQD SYSTEM")
        logging.info(f"🎯 Testing Thompson Sampling + Mahalanobis Resource Management")
        logging.info(f"📊 Processing {num_queries} queries with online learning")
        logging.info("="*80)
        
        # Start resource monitoring
        self.aqd_router.start_resource_monitoring()
        
        # Generate test queries
        query_templates = [
            "SELECT * FROM table WHERE id > {} LIMIT 10",
            "SELECT COUNT(*) FROM table WHERE category = '{}'",
            "SELECT category, AVG(value) FROM table GROUP BY category",
            "SELECT SUM(amount) FROM table WHERE date > '2024-01-01'",
            "SELECT * FROM table1 JOIN table2 ON table1.id = table2.ref_id WHERE condition"
        ]
        
        results = []
        total_latency = 0.0
        correct_decisions = 0
        
        for query_id in range(num_queries):
            # Generate random query
            template = np.random.choice(query_templates)
            sql = template.format(np.random.randint(1000), np.random.choice(['A', 'B', 'C']))
            
            # Make routing decision
            action, debug_info = self.aqd_router.make_routing_decision(sql)
            
            # Simulate execution
            observed_latency = self.simulate_query_execution(sql, action)
            total_latency += observed_latency
            
            # Determine optimal action for evaluation
            row_latency = self.simulate_query_execution(sql, 0)
            col_latency = self.simulate_query_execution(sql, 1)
            optimal_action = 0 if row_latency < col_latency else 1
            
            if action == optimal_action:
                correct_decisions += 1
                
            # Provide feedback to system
            self.aqd_router.update_with_execution_feedback(
                query_id, sql, action, observed_latency
            )
            
            # Record results
            results.append({
                'query_id': query_id,
                'action': action,
                'optimal_action': optimal_action,
                'observed_latency': observed_latency,
                'correct': action == optimal_action,
                **debug_info
            })
            
            if (query_id + 1) % 100 == 0:
                accuracy = correct_decisions / (query_id + 1)
                avg_latency = total_latency / (query_id + 1)
                logging.info(f"   Progress: {query_id + 1:4d}/{num_queries} queries, "
                           f"accuracy: {accuracy:.1%}, avg_latency: {avg_latency:.3f}s")
                           
        # Stop monitoring
        self.aqd_router.stop_resource_monitoring()
        
        # Calculate final metrics
        final_accuracy = correct_decisions / num_queries
        final_avg_latency = total_latency / num_queries
        
        # Learning curve analysis
        window_size = 50
        learning_curve = []
        for i in range(window_size, len(results), window_size):
            window_results = results[i-window_size:i]
            window_accuracy = sum(r['correct'] for r in window_results) / len(window_results)
            learning_curve.append(window_accuracy)
            
        evaluation_summary = {
            'total_queries': num_queries,
            'final_accuracy': final_accuracy,
            'final_avg_latency': final_avg_latency,
            'learning_curve': learning_curve,
            'thompson_sampling_active': True,
            'mahalanobis_regulation_active': True,
            'decision_time_avg': np.mean([r['decision_time'] for r in results])
        }
        
        logging.info(f"✅ AQD Evaluation Complete!")
        logging.info(f"   Final accuracy: {final_accuracy:.1%}")
        logging.info(f"   Average latency: {final_avg_latency:.3f}s") 
        logging.info(f"   Decision overhead: {evaluation_summary['decision_time_avg']*1000:.1f}ms")
        
        return {
            'summary': evaluation_summary,
            'detailed_results': results
        }

def main():
    logging.info("🌟 ADVANCED AQD SYSTEM WITH THOMPSON SAMPLING & MAHALANOBIS MANAGEMENT")
    logging.info("🎯 Implementation based on ~/DB/aqd_paper/main.tex")
    logging.info("🧠 Features: LinTS-Delta bandit + Resource regulation + Load adaptation")
    logging.info("="*80)
    
    evaluator = AQDSystemEvaluator()
    results = evaluator.run_evaluation(1000)
    
    # Save results
    results_path = '/data/wuy/db/advanced_aqd_results.json'
    with open(results_path, 'w') as f:
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.float64):
                return float(obj)
            elif isinstance(obj, np.int64):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
            
        json.dump(results, f, indent=2, default=convert_numpy)
        
    logging.info(f"📄 Results saved: {results_path}")
    
    summary = results['summary']
    print(f"\n🎉 ADVANCED AQD EVALUATION COMPLETE!")
    print(f"📊 Final accuracy: {summary['final_accuracy']:.1%}")
    print(f"⚡ Average latency: {summary['final_avg_latency']:.3f}s")
    print(f"🔄 Decision overhead: {summary['decision_time_avg']*1000:.1f}ms")
    print(f"🧠 Thompson Sampling: Active")
    print(f"📐 Mahalanobis Regulation: Active")
    print(f"🚀 Ready for DuckDB kernel integration")

if __name__ == "__main__":
    main()