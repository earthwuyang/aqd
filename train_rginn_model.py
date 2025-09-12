#!/usr/bin/env python3
"""
Train R-GIN (Relational Graph Isomorphism Network) model for query routing.
This script trains the GNN model using collected dual execution data.
"""

import json
import numpy as np
import os
import argparse
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Operation type mapping
OPERATION_TYPES = {
    'Seq Scan': 1,
    'Index Scan': 2,
    'Index Only Scan': 3,
    'Bitmap Heap Scan': 4,
    'Bitmap Index Scan': 5,
    'Nested Loop': 6,
    'Merge Join': 7,
    'Hash Join': 8,
    'Hash': 9,
    'Aggregate': 10,
    'WindowAgg': 11,
    'Sort': 12,
    'Limit': 13,
    'Materialize': 14,
    'Gather': 15,
    'Gather Merge': 16,
    'Append': 17,
    'Merge Append': 18,
    'Subquery Scan': 19
}

NUM_OP_TYPES = 20
FEATURE_DIM = 32  # 20 for one-hot encoding + 12 numeric features

class PlanGraph:
    """Represents a query plan as a graph."""
    
    def __init__(self):
        self.nodes = []
        self.edges = {0: [], 1: [], 2: []}  # Three relation types
        self.features = []
    
    def add_node(self, features):
        """Add a node with features."""
        node_idx = len(self.nodes)
        self.nodes.append(node_idx)
        self.features.append(features)
        return node_idx
    
    def add_edge(self, src, dst, rel_type):
        """Add an edge with relation type."""
        self.edges[rel_type].append((src, dst))

def extract_node_features(plan_node: Dict) -> np.ndarray:
    """
    Extract features from a plan node.
    
    Args:
        plan_node: Plan node dictionary
        
    Returns:
        Feature vector of dimension FEATURE_DIM
    """
    features = np.zeros(FEATURE_DIM)
    
    # One-hot encoding of operation type
    node_type = plan_node.get('Node Type', 'Unknown')
    op_id = OPERATION_TYPES.get(node_type, 0)
    if 0 <= op_id < NUM_OP_TYPES:
        features[op_id] = 1.0
    
    # Numeric features (normalized)
    idx = NUM_OP_TYPES
    
    # Log-normalized rows
    plan_rows = plan_node.get('Plan Rows', 0)
    features[idx] = np.log(plan_rows + 1) / 10.0
    idx += 1
    
    # Normalized width
    plan_width = plan_node.get('Plan Width', 0)
    features[idx] = plan_width / 100.0
    idx += 1
    
    # Log-normalized costs
    total_cost = plan_node.get('Total Cost', 0)
    features[idx] = np.log(total_cost + 1) / 10.0
    idx += 1
    
    startup_cost = plan_node.get('Startup Cost', 0)
    features[idx] = np.log(startup_cost + 1) / 10.0
    idx += 1
    
    # Cost per row
    if plan_rows > 0:
        features[idx] = (total_cost / plan_rows) / 1000.0
    else:
        features[idx] = total_cost / 1000.0
    idx += 1
    
    # Parallel features
    features[idx] = 1.0 if plan_node.get('Parallel Aware', False) else 0.0
    idx += 1
    
    workers = plan_node.get('Workers Planned', 0)
    features[idx] = workers / 10.0
    idx += 1
    
    # Additional boolean features
    features[idx] = 1.0 if plan_node.get('Filter') else 0.0
    idx += 1
    
    features[idx] = 1.0 if plan_node.get('Join Filter') else 0.0
    idx += 1
    
    features[idx] = 1.0 if plan_node.get('Index Cond') else 0.0
    idx += 1
    
    features[idx] = 1.0 if plan_node.get('One-Time Filter') else 0.0
    idx += 1
    
    features[idx] = 1.0 if plan_node.get('Recheck Cond') else 0.0
    idx += 1
    
    return features

def get_relation_type(node_type: str) -> int:
    """
    Get relation type for an edge based on child node type.
    
    Args:
        node_type: Node type string
        
    Returns:
        Relation type (0: scan, 1: join, 2: other)
    """
    op_id = OPERATION_TYPES.get(node_type, 0)
    
    # Scan operations
    if 1 <= op_id <= 5:
        return 0
    
    # Join operations
    if 6 <= op_id <= 8:
        return 1
    
    # Other operations
    return 2

def plan_to_graph(plan_json: Dict) -> PlanGraph:
    """
    Convert a PostgreSQL plan JSON to a graph.
    
    Args:
        plan_json: Plan dictionary from EXPLAIN JSON
        
    Returns:
        PlanGraph object
    """
    graph = PlanGraph()
    
    def process_node(node: Dict, parent_idx: Optional[int] = None) -> int:
        """Recursively process plan nodes."""
        # Extract features and add node
        features = extract_node_features(node)
        node_idx = graph.add_node(features)
        
        # Add edge from parent if exists
        if parent_idx is not None:
            rel_type = get_relation_type(node.get('Node Type', 'Unknown'))
            graph.add_edge(parent_idx, node_idx, rel_type)
        
        # Process children
        if 'Plans' in node:
            for child in node['Plans']:
                process_node(child, node_idx)
        
        return node_idx
    
    # Start from root
    if 'Plan' in plan_json:
        process_node(plan_json['Plan'])
    
    return graph

class RGINNModel:
    """R-GIN model implementation in Python for training."""
    
    def __init__(self, input_dim=FEATURE_DIM, hidden_dim=32, num_relations=3, eps=0.0):
        """
        Initialize R-GIN model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_relations: Number of relation types
            eps: Epsilon for self-loops
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self.eps = eps
        
        # Initialize parameters with small random values
        np.random.seed(42)
        scale = 0.1
        
        # Input projection
        self.W_input = np.random.randn(hidden_dim, input_dim) * scale
        self.b_input = np.zeros(hidden_dim)
        
        # Relation transformations
        self.W_rel = np.random.randn(num_relations, hidden_dim, hidden_dim) * scale
        
        # MLP
        self.W_mlp = np.random.randn(hidden_dim, hidden_dim) * scale
        self.b_mlp = np.zeros(hidden_dim)
        
        # Output
        self.W_output = np.random.randn(hidden_dim) * scale
        self.b_output = 0.0
    
    def forward(self, graph: PlanGraph) -> float:
        """
        Forward pass through the model.
        
        Args:
            graph: PlanGraph object
            
        Returns:
            Prediction score
        """
        if not graph.nodes:
            return 0.0
        
        num_nodes = len(graph.nodes)
        features = np.array(graph.features)
        
        # Step 1: Input projection
        h0 = np.maximum(0, features @ self.W_input.T + self.b_input)
        
        # Step 2: Message passing
        messages = (1 + self.eps) * h0
        
        # Aggregate messages per relation type
        for rel_type in range(self.num_relations):
            for src, dst in graph.edges[rel_type]:
                # Apply relation-specific transform
                transformed = h0[src] @ self.W_rel[rel_type].T
                messages[dst] += transformed
        
        # Step 3: MLP transformation
        h1 = np.maximum(0, messages @ self.W_mlp.T + self.b_mlp)
        
        # Step 4: Graph-level readout (mean pooling)
        graph_repr = np.mean(h1, axis=0)
        
        # Step 5: Final prediction
        prediction = graph_repr @ self.W_output + self.b_output
        
        return prediction
    
    def train_step(self, graphs: List[PlanGraph], targets: np.ndarray, 
                   weights: np.ndarray = None, learning_rate: float = 0.001,
                   lambda_l1: float = 0.0, lambda_l2: float = 0.01) -> float:
        """
        Single training step with full backpropagation through GNN layers.
        
        Args:
            graphs: List of PlanGraph objects
            targets: Target values (log time differences)
            weights: Sample weights for self-paced learning
            learning_rate: Learning rate
            lambda_l1: L1 regularization coefficient
            lambda_l2: L2 regularization coefficient
            
        Returns:
            Weighted average loss
        """
        if weights is None:
            weights = np.ones(len(graphs))
        
        total_loss = 0.0
        total_weight = np.sum(weights)
        gradients = self._zero_gradients()
        
        for graph, target, weight in zip(graphs, targets, weights):
            if not graph.nodes:
                continue
                
            num_nodes = len(graph.nodes)
            features = np.array(graph.features)
            
            # Forward pass with intermediate activations stored
            # Step 1: Input projection
            z0 = features @ self.W_input.T + self.b_input
            h0 = np.maximum(0, z0)  # ReLU
            
            # Step 2: Message passing
            messages = (1 + self.eps) * h0
            
            # Aggregate messages per relation type
            for rel_type in range(self.num_relations):
                for src, dst in graph.edges[rel_type]:
                    transformed = h0[src] @ self.W_rel[rel_type].T
                    messages[dst] += transformed
            
            # Step 3: MLP transformation
            z1 = messages @ self.W_mlp.T + self.b_mlp
            h1 = np.maximum(0, z1)  # ReLU
            
            # Step 4: Graph-level readout (mean pooling)
            graph_repr = np.mean(h1, axis=0)
            
            # Step 5: Final prediction
            prediction = graph_repr @ self.W_output + self.b_output
            
            # Compute weighted Huber loss
            error = prediction - target
            if abs(error) <= 1.0:
                loss = 0.5 * error ** 2
                loss_grad = error
            else:
                loss = abs(error) - 0.5
                loss_grad = np.sign(error)
            
            weighted_loss = weight * loss
            total_loss += weighted_loss
            
            # Backpropagation
            weighted_grad = weight * loss_grad
            
            # Gradient w.r.t output layer
            gradients['W_output'] += weighted_grad * graph_repr
            gradients['b_output'] += weighted_grad
            
            # Gradient w.r.t graph representation (before pooling)
            d_graph_repr = weighted_grad * self.W_output
            d_h1 = np.ones((num_nodes, self.hidden_dim)) * d_graph_repr / num_nodes
            
            # Gradient through MLP ReLU
            d_z1 = d_h1 * (z1 > 0)
            
            # Gradient w.r.t MLP weights
            gradients['W_mlp'] += np.outer(d_z1.sum(axis=0), messages.mean(axis=0))
            gradients['b_mlp'] += d_z1.sum(axis=0)
            
            # Gradient w.r.t messages
            d_messages = d_z1 @ self.W_mlp
            
            # Gradient through message passing
            d_h0 = d_messages * (1 + self.eps)
            
            # Gradient through relation transformations
            for rel_type in range(self.num_relations):
                for src, dst in graph.edges[rel_type]:
                    # Gradient w.r.t relation weights
                    gradients['W_rel'][rel_type] += np.outer(d_messages[dst], h0[src])
                    # Accumulate gradient to source nodes
                    d_h0[src] += d_messages[dst] @ self.W_rel[rel_type]
            
            # Gradient through input ReLU
            d_z0 = d_h0 * (z0 > 0)
            
            # Gradient w.r.t input projection
            gradients['W_input'] += d_z0.T @ features / num_nodes
            gradients['b_input'] += d_z0.mean(axis=0)
        
        # Add L2 regularization
        gradients['W_input'] += lambda_l2 * self.W_input
        gradients['b_input'] += lambda_l2 * self.b_input
        gradients['W_rel'] += lambda_l2 * self.W_rel
        gradients['W_mlp'] += lambda_l2 * self.W_mlp
        gradients['b_mlp'] += lambda_l2 * self.b_mlp
        gradients['W_output'] += lambda_l2 * self.W_output
        
        # Add L1 regularization (soft thresholding)
        if lambda_l1 > 0:
            gradients['W_input'] += lambda_l1 * np.sign(self.W_input)
            gradients['W_rel'] += lambda_l1 * np.sign(self.W_rel)
            gradients['W_mlp'] += lambda_l1 * np.sign(self.W_mlp)
            gradients['W_output'] += lambda_l1 * np.sign(self.W_output)
        
        # Update all parameters with gradient descent
        if total_weight > 0:
            scale = learning_rate / total_weight
            self.W_input -= scale * gradients['W_input']
            self.b_input -= scale * gradients['b_input']
            self.W_rel -= scale * gradients['W_rel']
            self.W_mlp -= scale * gradients['W_mlp']
            self.b_mlp -= scale * gradients['b_mlp']
            self.W_output -= scale * gradients['W_output']
            self.b_output -= scale * gradients['b_output']
        
        return total_loss / total_weight if total_weight > 0 else 0.0
    
    def _zero_gradients(self) -> Dict:
        """Initialize zero gradients."""
        return {
            'W_input': np.zeros_like(self.W_input),
            'b_input': np.zeros_like(self.b_input),
            'W_rel': np.zeros_like(self.W_rel),
            'W_mlp': np.zeros_like(self.W_mlp),
            'b_mlp': np.zeros_like(self.b_mlp),
            'W_output': np.zeros_like(self.W_output),
            'b_output': 0.0
        }
    
    def save(self, filepath: str):
        """Save model parameters to file."""
        with open(filepath, 'w') as f:
            # Write dimensions
            f.write(f"{self.input_dim} {self.hidden_dim}\n")
            
            # Write input projection
            for row in self.W_input:
                f.write(' '.join(map(str, row)) + '\n')
            f.write(' '.join(map(str, self.b_input)) + '\n')
            
            # Write relation transforms
            for rel in range(self.num_relations):
                for row in self.W_rel[rel]:
                    f.write(' '.join(map(str, row)) + '\n')
            
            # Write MLP
            for row in self.W_mlp:
                f.write(' '.join(map(str, row)) + '\n')
            f.write(' '.join(map(str, self.b_input)) + '\n')
            
            # Write output
            f.write(' '.join(map(str, self.W_output)) + '\n')
            f.write(str(self.b_output) + '\n')
        
        logger.info(f"Model saved to {filepath}")

def load_training_data(data_file: str) -> Tuple[List[PlanGraph], np.ndarray]:
    """
    Load training data from JSON file.
    
    Args:
        data_file: Path to JSON file with dual execution data
        
    Returns:
        Tuple of (graphs, targets)
    """
    graphs = []
    targets = []
    
    with open(data_file, 'r') as f:
        data = json.load(f)
        
        # Handle both list and dict formats (summary files are dicts)
        if isinstance(data, dict):
            logger.info(f"Skipping summary file: {data_file}")
            return graphs, np.array(targets)
        
        data_list = data if isinstance(data, list) else [data]
        
        for item in data_list:
            # Handle different field names
            postgres_time = item.get('postgres_time_ms') or item.get('postgres_time')
            duckdb_time = item.get('duckdb_time_ms') or item.get('duckdb_time')
            
            # Skip if no plan available or times are missing
            if not item.get('postgres_plan') or postgres_time is None or duckdb_time is None:
                continue
            
            # Skip failed queries (negative times indicate failure)
            if postgres_time < 0 or duckdb_time < 0:
                continue
            
            # Convert plan to graph
            try:
                # Parse the plan JSON if it's a string
                plan = item['postgres_plan']
                if isinstance(plan, str):
                    plan = json.loads(plan)
                
                # Handle both formats: direct plan or wrapped in array
                if isinstance(plan, list) and len(plan) > 0:
                    plan = plan[0]
                
                graph = plan_to_graph(plan)
                graphs.append(graph)
                
                # Calculate target: log time ratio
                # Positive = PostgreSQL slower (prefer DuckDB)
                # Negative = DuckDB slower (prefer PostgreSQL)
                # Convert ms to seconds if needed
                pg_time_sec = postgres_time / 1000.0 if postgres_time > 10 else postgres_time
                duck_time_sec = duckdb_time / 1000.0 if duckdb_time > 10 else duckdb_time
                
                pg_time_sec = max(pg_time_sec, 0.001)  # Avoid log(0)
                duck_time_sec = max(duck_time_sec, 0.001)
                log_ratio = np.log(pg_time_sec / duck_time_sec)
                targets.append(log_ratio)
                
            except Exception as e:
                logger.warning(f"Failed to process plan: {e}")
                continue
    
    logger.info(f"Loaded {len(graphs)} training examples from {data_file}")
    return graphs, np.array(targets)

def evaluate_model(model: RGINNModel, graphs: List[PlanGraph], 
                   targets: np.ndarray, threshold: float = 0.0) -> Dict:
    """
    Evaluate model performance.
    
    Args:
        model: Trained model
        graphs: Test graphs
        targets: True targets (log time differences)
        threshold: Decision threshold
        
    Returns:
        Dictionary with evaluation metrics
    """
    predictions = []
    for graph in graphs:
        pred = model.forward(graph)
        predictions.append(pred)
    
    predictions = np.array(predictions)
    
    # Binary classification: >threshold means route to DuckDB (1), else PostgreSQL (0)
    pred_binary = (predictions > threshold).astype(int)
    true_binary = (targets > threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(true_binary, pred_binary)
    
    # Handle the case where precision/recall might be undefined
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_binary, pred_binary, average='binary', zero_division=0
    )
    
    # Regression metrics
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    
    # Confusion matrix with explicit labels
    # Format: cm[true_label][predicted_label]
    # Labels: 0 = PostgreSQL, 1 = DuckDB
    cm = confusion_matrix(true_binary, pred_binary, labels=[0, 1])
    
    # Ensure confusion matrix is 2x2 even if some classes are missing
    if cm.shape != (2, 2):
        cm_full = np.zeros((2, 2), dtype=int)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                cm_full[i, j] = cm[i, j]
        cm = cm_full
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mse': mse,
        'mae': mae,
        'confusion_matrix': cm.tolist(),
        'threshold': threshold,
        'pred_counts': {
            'postgres': np.sum(pred_binary == 0),
            'duckdb': np.sum(pred_binary == 1)
        },
        'true_counts': {
            'postgres': np.sum(true_binary == 0),
            'duckdb': np.sum(true_binary == 1)
        }
    }

def compute_sample_weights(targets: np.ndarray, epoch: int, max_epochs: int,
                          balance_factor: float = 2.0) -> np.ndarray:
    """
    Compute sample weights for self-paced learning.
    
    Args:
        targets: Log time differences
        epoch: Current epoch
        max_epochs: Maximum number of epochs
        balance_factor: Factor to balance positive/negative samples
        
    Returns:
        Sample weights
    """
    weights = np.ones(len(targets))
    
    # 1. Emphasize samples with large performance gaps
    # Use exponential scaling for large differences
    abs_targets = np.abs(targets)
    gap_weights = np.exp(np.minimum(abs_targets, 3.0))  # Cap at exp(3) to avoid overflow
    
    # 2. Balance positive and negative samples
    positive_mask = targets > 0
    negative_mask = targets <= 0
    num_positive = np.sum(positive_mask)
    num_negative = np.sum(negative_mask)
    
    if num_positive > 0 and num_negative > 0:
        # Inverse frequency weighting
        positive_weight = len(targets) / (2.0 * num_positive)
        negative_weight = len(targets) / (2.0 * num_negative)
        
        weights[positive_mask] *= positive_weight
        weights[negative_mask] *= negative_weight
    
    # 3. Self-paced learning: gradually include harder samples
    # Early epochs focus on easier samples (smaller gaps)
    pace_factor = (epoch + 1) / max_epochs  # 0 to 1 over training
    
    # Sort by difficulty (absolute target value)
    difficulty_order = np.argsort(abs_targets)
    num_active = int(len(targets) * (0.3 + 0.7 * pace_factor))  # Start with 30%, end with 100%
    
    # Reduce weights for harder samples in early epochs
    hard_samples = difficulty_order[num_active:]
    if len(hard_samples) > 0:
        weights[hard_samples] *= 0.1 * pace_factor
    
    # 4. Combine with gap emphasis
    weights *= gap_weights
    
    # Normalize weights
    weights = weights / np.mean(weights)
    
    return weights


def main():
    parser = argparse.ArgumentParser(description='Train R-GIN model with self-paced learning')
    parser.add_argument('--data-dir', default='dual_execution_data',
                       help='Directory with training data')
    parser.add_argument('--output-dir', default='models',
                       help='Output directory for trained models')
    parser.add_argument('--hidden-dim', type=int, default=32,
                       help='Hidden dimension')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--test-split', type=float, default=0.2,
                       help='Test set split ratio')
    parser.add_argument('--threshold', type=float, default=0.0,
                       help='Decision threshold')
    parser.add_argument('--lambda-l1', type=float, default=0.0,
                       help='L1 regularization coefficient')
    parser.add_argument('--lambda-l2', type=float, default=0.01,
                       help='L2 regularization coefficient')
    parser.add_argument('--self-paced', action='store_true',
                       help='Use self-paced learning with sample weights')
    parser.add_argument('--lr-decay', type=float, default=0.95,
                       help='Learning rate decay per epoch')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load all training data
    all_graphs = []
    all_targets = []
    
    # Load data from all JSON files in the directory
    for filename in os.listdir(args.data_dir):
        if filename.endswith('.json') and not filename.startswith('.'):
            filepath = os.path.join(args.data_dir, filename)
            logger.info(f"Loading data from {filepath}")
            graphs, targets = load_training_data(filepath)
            all_graphs.extend(graphs)
            all_targets.extend(targets)
    
    if not all_graphs:
        logger.error("No training data found")
        return
    
    all_targets = np.array(all_targets)
    logger.info(f"Total training examples: {len(all_graphs)}")
    
    # Analyze data distribution
    positive_samples = np.sum(all_targets > args.threshold)
    negative_samples = len(all_targets) - positive_samples
    logger.info(f"Data distribution: {positive_samples} positive (prefer DuckDB), {negative_samples} negative (prefer PostgreSQL)")
    logger.info(f"Class balance: {positive_samples/len(all_targets)*100:.1f}% positive, {negative_samples/len(all_targets)*100:.1f}% negative")
    
    # Split data with stratification if possible
    indices = np.arange(len(all_graphs))
    binary_targets = (all_targets > args.threshold).astype(int)
    
    try:
        # Try stratified split to maintain class balance
        train_idx, test_idx = train_test_split(
            indices, test_size=args.test_split, random_state=42,
            stratify=binary_targets if len(np.unique(binary_targets)) > 1 else None
        )
    except:
        # Fall back to random split if stratification fails
        train_idx, test_idx = train_test_split(
            indices, test_size=args.test_split, random_state=42
        )
    
    train_graphs = [all_graphs[i] for i in train_idx]
    train_targets = all_targets[train_idx]
    test_graphs = [all_graphs[i] for i in test_idx]
    test_targets = all_targets[test_idx]
    
    logger.info(f"Training set: {len(train_graphs)}, Test set: {len(test_graphs)}")
    
    # Initialize model
    model = RGINNModel(
        input_dim=FEATURE_DIM,
        hidden_dim=args.hidden_dim
    )
    
    # Training loop with self-paced learning
    logger.info("Starting training...")
    if args.self_paced:
        logger.info("Using self-paced learning with sample weights")
    
    best_f1 = 0.0
    best_loss = float('inf')
    current_lr = args.learning_rate
    
    for epoch in range(args.epochs):
        # Compute sample weights for self-paced learning
        if args.self_paced:
            sample_weights = compute_sample_weights(
                train_targets, epoch, args.epochs
            )
        else:
            sample_weights = np.ones(len(train_targets))
        
        # Shuffle training data while preserving weights
        perm = np.random.permutation(len(train_graphs))
        train_graphs_shuffled = [train_graphs[i] for i in perm]
        train_targets_shuffled = train_targets[perm]
        train_weights_shuffled = sample_weights[perm]
        
        # Train in batches
        total_loss = 0.0
        total_samples = 0
        
        for i in range(0, len(train_graphs), args.batch_size):
            batch_graphs = train_graphs_shuffled[i:i+args.batch_size]
            batch_targets = train_targets_shuffled[i:i+args.batch_size]
            batch_weights = train_weights_shuffled[i:i+args.batch_size]
            
            loss = model.train_step(
                batch_graphs, batch_targets, batch_weights,
                learning_rate=current_lr,
                lambda_l1=args.lambda_l1,
                lambda_l2=args.lambda_l2
            )
            total_loss += loss * len(batch_graphs)
            total_samples += len(batch_graphs)
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        
        # Learning rate decay
        current_lr *= args.lr_decay
        
        # Evaluate on test set
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == args.epochs - 1:
            test_metrics = evaluate_model(model, test_graphs, test_targets, args.threshold)
            
            # Log with more details
            logger.info(f"Epoch {epoch+1}/{args.epochs} - "
                       f"LR: {current_lr:.6f}, "
                       f"Train Loss: {avg_loss:.4f}, "
                       f"Test Acc: {test_metrics['accuracy']:.3f}, "
                       f"Test F1: {test_metrics['f1']:.3f}, "
                       f"Test Prec: {test_metrics['precision']:.3f}, "
                       f"Test Rec: {test_metrics['recall']:.3f}")
            
            # Save best model based on F1 score
            if test_metrics['f1'] > best_f1:
                best_f1 = test_metrics['f1']
                best_loss = avg_loss
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                model_path = os.path.join(args.output_dir, f'rginn_model_best_{timestamp}.txt')
                model.save(model_path)
                logger.info(f"  -> New best F1: {best_f1:.3f}")
    
    # Final evaluation
    logger.info("\n" + "="*60)
    logger.info("FINAL EVALUATION")
    logger.info("="*60)
    
    final_metrics = evaluate_model(model, test_graphs, test_targets, args.threshold)
    
    # Additional diagnostics
    test_binary = (test_targets > args.threshold).astype(int)
    num_positive = np.sum(test_binary)
    num_negative = len(test_binary) - num_positive
    
    logger.info(f"Test set distribution: {num_negative} PostgreSQL, {num_positive} DuckDB")
    logger.info(f"Model predictions: {final_metrics['pred_counts']['postgres']} PostgreSQL, {final_metrics['pred_counts']['duckdb']} DuckDB")
    logger.info(f"Threshold used: {args.threshold}")
    logger.info("")
    logger.info(f"Accuracy: {final_metrics['accuracy']:.3f}")
    logger.info(f"Precision: {final_metrics['precision']:.3f} (for DuckDB class)")
    logger.info(f"Recall: {final_metrics['recall']:.3f} (for DuckDB class)")
    logger.info(f"F1 Score: {final_metrics['f1']:.3f}")
    logger.info(f"MSE: {final_metrics['mse']:.4f}")
    logger.info(f"MAE: {final_metrics['mae']:.4f}")
    logger.info("")
    logger.info("Confusion Matrix:")
    logger.info("                     Predicted")
    logger.info("                 PostgreSQL  DuckDB")
    logger.info(f"True PostgreSQL:     {final_metrics['confusion_matrix'][0][0]:3d}       {final_metrics['confusion_matrix'][0][1]:3d}")
    logger.info(f"True DuckDB:         {final_metrics['confusion_matrix'][1][0]:3d}       {final_metrics['confusion_matrix'][1][1]:3d}")
    logger.info("")
    logger.info("Matrix interpretation:")
    logger.info(f"  TN (True Negatives): {final_metrics['confusion_matrix'][0][0]} - Correctly predicted PostgreSQL")
    logger.info(f"  FP (False Positives): {final_metrics['confusion_matrix'][0][1]} - PostgreSQL incorrectly predicted as DuckDB")
    logger.info(f"  FN (False Negatives): {final_metrics['confusion_matrix'][1][0]} - DuckDB incorrectly predicted as PostgreSQL")
    logger.info(f"  TP (True Positives): {final_metrics['confusion_matrix'][1][1]} - Correctly predicted DuckDB")
    
    # Save final model and metrics
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    final_model_path = os.path.join(args.output_dir, f'rginn_model_final_{timestamp}.txt')
    model.save(final_model_path)
    
    metrics_path = os.path.join(args.output_dir, f'rginn_metrics_{timestamp}.json')
    # Convert numpy types to Python types for JSON serialization
    json_metrics = {
        'accuracy': float(final_metrics['accuracy']),
        'precision': float(final_metrics['precision']),
        'recall': float(final_metrics['recall']),
        'f1': float(final_metrics['f1']),
        'mse': float(final_metrics['mse']),
        'mae': float(final_metrics['mae']),
        'confusion_matrix': [[int(x) for x in row] for row in final_metrics['confusion_matrix']],
        'threshold': float(args.threshold)
    }
    # Add optional metrics if they exist
    for key in ['num_predictions_postgres', 'num_predictions_duckdb', 
                'num_true_postgres', 'num_true_duckdb']:
        if key in final_metrics:
            json_metrics[key] = int(final_metrics[key])
    
    with open(metrics_path, 'w') as f:
        json.dump(json_metrics, f, indent=2)
    
    logger.info(f"\nModel saved to: {final_model_path}")
    logger.info(f"Metrics saved to: {metrics_path}")


if __name__ == '__main__':
    main()