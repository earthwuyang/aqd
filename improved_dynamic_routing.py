#!/usr/bin/env python3
"""
Improved Dynamic Routing with Warm-up Phase
Solve the cold start problem for LightGBM Dynamic performance
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time
import json
from datetime import datetime
from advanced_aqd_system import AdvancedAQDRouter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class WarmUpConfig:
    num_warm_up_queries: int
    exploration_rate: float
    learning_rate: float
    use_guided_exploration: bool

@dataclass
class DynamicRoutingResult:
    method_name: str
    warm_up_queries: int
    accuracy_during_warmup: float
    accuracy_after_warmup: float
    final_accuracy: float
    avg_latency: float
    convergence_point: int
    learning_curve: List[float]

class ImprovedDynamicRouter:
    """
    Improved Dynamic Router with proper warm-up phase and learning optimization
    """
    
    def __init__(self, warm_up_config: WarmUpConfig):
        self.config = warm_up_config
        self.aqd_router = AdvancedAQDRouter()
        
        # Query templates for warm-up and testing
        self.query_templates = [
            # Simple OLTP queries (should favor PostgreSQL)
            "SELECT * FROM users WHERE user_id = {} LIMIT 1",
            "SELECT name, email FROM customers WHERE customer_id = {}",
            "UPDATE products SET price = {} WHERE product_id = {}",
            "INSERT INTO orders (customer_id, total) VALUES ({}, {})",
            
            # Complex OLAP queries (should favor DuckDB)
            "SELECT region, COUNT(*), AVG(sales) FROM transactions GROUP BY region ORDER BY AVG(sales) DESC",
            "SELECT YEAR(order_date), SUM(total) FROM orders WHERE total > {} GROUP BY YEAR(order_date)",
            "SELECT category, COUNT(DISTINCT customer_id) FROM purchases GROUP BY category HAVING COUNT(*) > 100",
            "SELECT t1.region, t2.category, AVG(t1.amount) FROM sales t1 JOIN products t2 ON t1.product_id = t2.id GROUP BY t1.region, t2.category",
            
            # Mixed queries (postgres_query candidates)
            "SELECT customer_id, COUNT(*) FROM orders WHERE order_date > '2024-01-01' GROUP BY customer_id",
            "SELECT AVG(price) FROM products WHERE category IN ('Electronics', 'Books', 'Clothing')",
            "SELECT COUNT(*) FROM transactions WHERE amount BETWEEN {} AND {} GROUP BY DATE(created_at)",
            "SELECT status, COUNT(*) FROM orders WHERE total > (SELECT AVG(total) FROM orders) GROUP BY status"
        ]
        
    def generate_warm_up_queries(self, num_queries: int) -> List[str]:
        """Generate diverse queries for warm-up phase"""
        queries = []
        
        # Ensure we have a good mix of query types for warm-up
        oltp_templates = self.query_templates[:4]  # Simple queries
        olap_templates = self.query_templates[4:8]  # Complex analytics
        mixed_templates = self.query_templates[8:]  # Mixed complexity
        
        for i in range(num_queries):
            # Balanced distribution during warm-up
            if i % 3 == 0:
                template = np.random.choice(oltp_templates)
            elif i % 3 == 1:
                template = np.random.choice(olap_templates)
            else:
                template = np.random.choice(mixed_templates)
                
            # Fill parameters
            if '{}' in template:
                params = [np.random.randint(1, 10000) for _ in range(template.count('{}'))]
                query = template.format(*params)
            else:
                query = template
                
            queries.append(query)
            
        return queries
        
    def simulate_execution_with_optimal(self, query: str, route: str) -> Tuple[float, str]:
        """
        Simulate execution and return actual time + optimal route
        """
        base_time = 0.05
        query_upper = query.upper()
        
        # Simulate execution times for all routes to determine optimal
        route_times = {}
        
        # PostgreSQL performance characteristics
        if any(keyword in query_upper for keyword in ["WHERE", "LIMIT 1", "=", "INSERT", "UPDATE"]):
            pg_multiplier = 0.6  # Excellent for point queries
        elif any(keyword in query_upper for keyword in ["GROUP BY", "COUNT", "SUM", "AVG"]):
            pg_multiplier = 1.4  # Slower for analytics
        else:
            pg_multiplier = 0.9
            
        # DuckDB performance characteristics  
        if any(keyword in query_upper for keyword in ["GROUP BY", "COUNT", "SUM", "AVG", "HAVING"]):
            duck_multiplier = 0.5  # Excellent for analytics
        elif "LIMIT 1" in query_upper:
            duck_multiplier = 1.1  # Slower for point queries
        else:
            duck_multiplier = 0.8
            
        # postgres_query performance (middle ground)
        pq_multiplier = 0.85
        
        # Calculate execution times with variance
        route_times = {
            'postgresql': base_time * pg_multiplier * (1 + np.random.exponential(0.2)),
            'duckdb': base_time * duck_multiplier * (1 + np.random.exponential(0.2)),
            'postgres_query': base_time * pq_multiplier * (1 + np.random.exponential(0.2))
        }
        
        # Determine optimal route
        optimal_route = min(route_times, key=route_times.get)
        
        # Return execution time for chosen route
        execution_time = route_times[route]
        
        return max(0.001, execution_time), optimal_route
        
    def run_guided_warm_up(self, warm_up_queries: List[str]) -> List[Dict]:
        """
        Run guided warm-up phase with known optimal decisions
        This helps the algorithm learn faster by providing good initial examples
        """
        logging.info(f"🔥 Running guided warm-up with {len(warm_up_queries)} queries")
        
        self.aqd_router.start_resource_monitoring()
        warm_up_results = []
        
        for query_id, query in enumerate(warm_up_queries):
            # For guided warm-up, we sometimes use the optimal decision
            # to help the algorithm learn faster
            
            if self.config.use_guided_exploration and np.random.random() < 0.3:
                # 30% of warm-up queries use optimal decision for faster learning
                _, optimal_route = self.simulate_execution_with_optimal(query, 'duckdb')
                action = 1 if optimal_route == 'duckdb' else 0
                execution_time, _ = self.simulate_execution_with_optimal(query, optimal_route)
            else:
                # Regular AQD decision
                action, debug_info = self.aqd_router.make_routing_decision(query)
                chosen_route = 'duckdb' if action == 1 else 'postgresql'
                execution_time, optimal_route = self.simulate_execution_with_optimal(query, chosen_route)
                
            # Provide feedback to AQD system
            self.aqd_router.update_with_execution_feedback(query_id, query, action, execution_time)
            
            is_correct = (('duckdb' if action == 1 else 'postgresql') == optimal_route)
            warm_up_results.append({
                'query_id': query_id,
                'correct': is_correct,
                'execution_time': execution_time
            })
            
        self.aqd_router.stop_resource_monitoring()
        
        warm_up_accuracy = sum(r['correct'] for r in warm_up_results) / len(warm_up_results)
        logging.info(f"   ✅ Warm-up complete: {warm_up_accuracy:.1%} accuracy")
        
        return warm_up_results
        
    def evaluate_warmed_up_dynamic_routing(self, test_queries: List[str], 
                                         warm_up_results: List[Dict]) -> DynamicRoutingResult:
        """
        Evaluate dynamic routing performance after warm-up
        """
        logging.info(f"🧠 Evaluating warmed-up dynamic routing on {len(test_queries)} queries")
        
        # Continue monitoring (it was stopped after warm-up)
        self.aqd_router.start_resource_monitoring()
        
        test_results = []
        learning_curve = []
        window_size = 50  # Calculate accuracy every 50 queries
        
        for query_id, query in enumerate(test_queries):
            # Make routing decision using warmed-up AQD system
            action, debug_info = self.aqd_router.make_routing_decision(query)
            chosen_route = 'duckdb' if action == 1 else 'postgresql'
            
            # Simulate execution
            execution_time, optimal_route = self.simulate_execution_with_optimal(query, chosen_route)
            
            # Continue learning with feedback
            actual_query_id = len(warm_up_results) + query_id  # Continue from warm-up
            self.aqd_router.update_with_execution_feedback(actual_query_id, query, action, execution_time)
            
            is_correct = (chosen_route == optimal_route)
            test_results.append({
                'query_id': query_id,
                'correct': is_correct,
                'execution_time': execution_time,
                'chosen_route': chosen_route,
                'optimal_route': optimal_route
            })
            
            # Update learning curve
            if (query_id + 1) % window_size == 0:
                window_accuracy = sum(r['correct'] for r in test_results[-window_size:]) / window_size
                learning_curve.append(window_accuracy)
                
        self.aqd_router.stop_resource_monitoring()
        
        # Calculate final metrics
        final_accuracy = sum(r['correct'] for r in test_results) / len(test_results)
        avg_latency = sum(r['execution_time'] for r in test_results) / len(test_results)
        
        # Find convergence point (where accuracy stabilizes)
        convergence_point = len(test_queries)  # Default to end
        if len(learning_curve) > 5:
            # Look for point where accuracy doesn't change much
            for i in range(2, len(learning_curve)):
                recent_variance = np.std(learning_curve[i-2:i+1])
                if recent_variance < 0.02:  # Accuracy stable within 2%
                    convergence_point = (i + 1) * window_size
                    break
                    
        # Calculate accuracy during different phases
        warm_up_accuracy = sum(r['correct'] for r in warm_up_results) / len(warm_up_results)
        
        # Accuracy in first half vs second half of test
        mid_point = len(test_results) // 2
        first_half_accuracy = sum(r['correct'] for r in test_results[:mid_point]) / mid_point
        second_half_accuracy = sum(r['correct'] for r in test_results[mid_point:]) / (len(test_results) - mid_point)
        
        logging.info(f"   ✅ Final accuracy: {final_accuracy:.1%}")
        logging.info(f"   📈 First half: {first_half_accuracy:.1%}, Second half: {second_half_accuracy:.1%}")
        logging.info(f"   🎯 Convergence at query: {convergence_point}")
        
        return DynamicRoutingResult(
            method_name=f"lightgbm_dynamic_warmup_{self.config.num_warm_up_queries}",
            warm_up_queries=self.config.num_warm_up_queries,
            accuracy_during_warmup=warm_up_accuracy,
            accuracy_after_warmup=first_half_accuracy,
            final_accuracy=final_accuracy,
            avg_latency=avg_latency,
            convergence_point=convergence_point,
            learning_curve=learning_curve
        )

class WarmUpExperimentRunner:
    """
    Run comprehensive warm-up experiments with different configurations
    """
    
    def __init__(self):
        self.base_query_count = 1000
        
    def run_warm_up_experiments(self) -> Dict:
        """
        Run experiments with different warm-up configurations
        """
        logging.info("🌟 COMPREHENSIVE WARM-UP EXPERIMENTS")
        logging.info("🎯 Testing LightGBM Dynamic with different warm-up phases")
        logging.info("="*80)
        
        # Test different warm-up configurations
        warm_up_configs = [
            # No warm-up (original)
            WarmUpConfig(0, 0.1, 0.1, False),
            # Light warm-up
            WarmUpConfig(50, 0.1, 0.1, False),
            # Moderate warm-up
            WarmUpConfig(100, 0.1, 0.1, False),
            # Heavy warm-up
            WarmUpConfig(200, 0.1, 0.1, False),
            # Guided warm-up (best practices)
            WarmUpConfig(100, 0.15, 0.1, True),
            # Extensive guided warm-up
            WarmUpConfig(300, 0.2, 0.15, True)
        ]
        
        results = []
        test_queries = self._generate_test_queries(self.base_query_count)
        
        for config in warm_up_configs:
            logging.info(f"🔥 Testing warm-up: {config.num_warm_up_queries} queries, "
                        f"guided={config.use_guided_exploration}")
                        
            router = ImprovedDynamicRouter(config)
            
            if config.num_warm_up_queries > 0:
                # Run warm-up phase
                warm_up_queries = router.generate_warm_up_queries(config.num_warm_up_queries)
                warm_up_results = router.run_guided_warm_up(warm_up_queries)
            else:
                warm_up_results = []
                
            # Evaluate on test queries
            result = router.evaluate_warmed_up_dynamic_routing(test_queries, warm_up_results)
            results.append(result)
            
            time.sleep(1)  # Brief pause between experiments
            
        return {
            'results': results,
            'test_configuration': {
                'base_query_count': self.base_query_count,
                'warm_up_configs_tested': len(warm_up_configs)
            },
            'timestamp': datetime.now().isoformat()
        }
        
    def _generate_test_queries(self, num_queries: int) -> List[str]:
        """Generate consistent test queries for fair comparison"""
        np.random.seed(42)  # Consistent seed for reproducible results
        
        query_templates = [
            "SELECT * FROM users WHERE user_id = {} LIMIT 1",
            "SELECT region, COUNT(*), AVG(sales) FROM transactions GROUP BY region",
            "SELECT customer_id, COUNT(*) FROM orders GROUP BY customer_id HAVING COUNT(*) > 5",
            "UPDATE products SET price = {} WHERE product_id = {}",
            "SELECT category, SUM(amount) FROM purchases WHERE date > '2024-01-01' GROUP BY category",
            "SELECT * FROM customers WHERE name LIKE '{}%' ORDER BY created_at DESC LIMIT 10",
            "SELECT YEAR(order_date), AVG(total) FROM orders GROUP BY YEAR(order_date)",
            "INSERT INTO logs (user_id, action, timestamp) VALUES ({}, '{}', NOW())"
        ]
        
        queries = []
        for _ in range(num_queries):
            template = np.random.choice(query_templates)
            if '{}' in template:
                params = []
                for _ in range(template.count('{}')):
                    if 'LIKE' in template:
                        params.append(np.random.choice(['John', 'Jane', 'Mike', 'Sarah', 'David']))
                    elif 'action' in template:
                        params.append(np.random.choice(['login', 'purchase', 'view', 'logout']))
                    else:
                        params.append(np.random.randint(1, 10000))
                query = template.format(*params)
            else:
                query = template
            queries.append(query)
            
        return queries
        
    def generate_warm_up_analysis_report(self, experiment_results: Dict):
        """Generate comprehensive warm-up analysis report"""
        results = experiment_results['results']
        
        print(f"\n🔥 WARM-UP EXPERIMENT RESULTS")
        print(f"="*70)
        print(f"Test Scale: {experiment_results['test_configuration']['base_query_count']} queries per config")
        
        # Performance comparison table
        print(f"\n📊 WARM-UP IMPACT ANALYSIS:")
        print(f"{'Warm-up Queries':<15} {'Guided':<8} {'Final Accuracy':<15} {'Convergence':<12} {'Improvement':<12}")
        print(f"{'-'*15} {'-'*8} {'-'*15} {'-'*12} {'-'*12}")
        
        baseline_accuracy = None
        for result in results:
            if result.warm_up_queries == 0:
                baseline_accuracy = result.final_accuracy
                break
                
        for result in results:
            guided_str = "Yes" if "guided" in result.method_name else "No"
            convergence_str = f"{result.convergence_point}" if result.convergence_point < 1000 else "No"
            
            if baseline_accuracy and result.warm_up_queries > 0:
                improvement = ((result.final_accuracy - baseline_accuracy) / baseline_accuracy) * 100
                improvement_str = f"+{improvement:.1f}%"
            else:
                improvement_str = "baseline"
                
            print(f"{result.warm_up_queries:<15} "
                  f"{guided_str:<8} "
                  f"{result.final_accuracy:<14.1%} "
                  f"{convergence_str:<12} "
                  f"{improvement_str:<12}")
                  
        # Best configuration analysis
        best_result = max(results, key=lambda r: r.final_accuracy)
        
        print(f"\n🏆 BEST WARM-UP CONFIGURATION:")
        print(f"   Warm-up queries: {best_result.warm_up_queries}")
        print(f"   Final accuracy: {best_result.final_accuracy:.1%}")
        print(f"   Convergence point: {best_result.convergence_point} queries")
        print(f"   Improvement: {((best_result.final_accuracy - baseline_accuracy) / baseline_accuracy) * 100:+.1f}%")
        
        # Learning curve analysis
        print(f"\n📈 LEARNING PROGRESSION ANALYSIS:")
        for result in results:
            if result.warm_up_queries > 0 and len(result.learning_curve) > 0:
                initial_acc = result.learning_curve[0] if result.learning_curve else result.accuracy_after_warmup
                final_acc = result.learning_curve[-1] if result.learning_curve else result.final_accuracy
                learning_improvement = ((final_acc - initial_acc) / initial_acc) * 100 if initial_acc > 0 else 0
                
                print(f"   {result.warm_up_queries} warm-up: "
                      f"{initial_acc:.1%} → {final_acc:.1%} "
                      f"({learning_improvement:+.1f}% during test)")
                      
        # Save detailed results
        results_path = '/data/wuy/db/warm_up_experiment_results.json'
        
        json_results = []
        for result in results:
            json_results.append({
                'method_name': result.method_name,
                'warm_up_queries': result.warm_up_queries,
                'accuracy_during_warmup': result.accuracy_during_warmup,
                'accuracy_after_warmup': result.accuracy_after_warmup,
                'final_accuracy': result.final_accuracy,
                'avg_latency': result.avg_latency,
                'convergence_point': result.convergence_point,
                'learning_curve': result.learning_curve
            })
            
        with open(results_path, 'w') as f:
            json.dump({
                'results': json_results,
                'test_configuration': experiment_results['test_configuration'],
                'timestamp': experiment_results['timestamp'],
                'best_configuration': {
                    'warm_up_queries': best_result.warm_up_queries,
                    'final_accuracy': best_result.final_accuracy,
                    'improvement_vs_baseline': ((best_result.final_accuracy - baseline_accuracy) / baseline_accuracy) * 100
                }
            }, f, indent=2)
            
        logging.info(f"📊 Detailed warm-up results saved: {results_path}")
        
        return results_path

def main():
    logging.info("🌟 IMPROVED DYNAMIC ROUTING WITH WARM-UP PHASE")
    logging.info("🔥 Solving the cold start problem for LightGBM Dynamic")
    logging.info("🎯 Testing different warm-up strategies and configurations")
    logging.info("="*80)
    
    runner = WarmUpExperimentRunner()
    experiment_results = runner.run_warm_up_experiments()
    results_path = runner.generate_warm_up_analysis_report(experiment_results)
    
    print(f"\n🎉 WARM-UP EXPERIMENTS COMPLETE!")
    print(f"✅ Tested {len(experiment_results['results'])} warm-up configurations")
    print(f"📄 Detailed analysis: {results_path}")
    print(f"🚀 Identified optimal warm-up strategy for production deployment")

if __name__ == "__main__":
    main()