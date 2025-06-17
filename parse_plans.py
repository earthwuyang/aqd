import torch
from torch_geometric.data import HeteroData, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import HeteroConv, GCNConv, global_mean_pool
from torch.nn import Linear, ReLU, Dropout
import torch.nn.functional as F
from typing import Dict, Any, Tuple, List
import itertools
import json
import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import torch.nn as nn
import numpy as np


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NodeIndexer:
    """
    Helper class to assign unique IDs to nodes based on their type and identifier.
    """
    def __init__(self):
        self.node_id = 0
        self.node_map = {}  # Maps (node_type, identifier) to unique ID
        self.type_to_ids = {}  # Maps node_type to list of node IDs

    def add_node(self, node_type: str, identifier: str) -> int:
        key = (node_type, identifier)
        if key not in self.node_map:
            self.node_map[key] = self.node_id
            if node_type not in self.type_to_ids:
                self.type_to_ids[node_type] = []
            self.type_to_ids[node_type].append(self.node_id)
            logger.debug(f"Added node: {key} with ID {self.node_id}")
            self.node_id += 1
        return self.node_map[key]

def encode_categorical(categories: List[str]) -> Tuple[torch.Tensor, Dict[str, int]]:
    """
    Encodes categorical variables into integer indices.

    Args:
        categories (List[str]): List of category strings.

    Returns:
        Tuple[torch.Tensor, Dict[str, int]]: Tensor of encoded categories and the mapping dictionary.
    """
    unique = sorted(list(set(categories)))
    mapping = {cat: idx for idx, cat in enumerate(unique)}
    encoded = torch.tensor([mapping[cat] for cat in categories], dtype=torch.long)
    return encoded, mapping
def convert_data_size_to_numeric(size_str: str) -> Any:
    """
    Converts a data size string with units to a numeric value in bytes.
    
    Args:
        size_str (str): Data size string, e.g., "16G", "5M", "544".
    
    Returns:
        float: Numeric value in bytes.
    """
    size_str = size_str.strip()
    if size_str.endswith('E'):  # Exabytes
        return float(size_str[:-1]) * 1e18
    elif size_str.endswith('P'):  # Petabytes
        return float(size_str[:-1]) * 1e15
    elif size_str.endswith('T'):  # Terabytes
        return float(size_str[:-1]) * 1e12
    elif size_str.endswith('G'):
        return float(size_str[:-1]) * 1e9
    elif size_str.endswith('M'):
        return float(size_str[:-1]) * 1e6
    elif size_str.endswith('K'):
        return float(size_str[:-1]) * 1e3
    else:
        try:
            return float(size_str)
        except ValueError:
            print(f"Invalid data size string as numeric value: {size_str}")
            return size_str

def get_nested_loop(query_block: Dict[str, Any], tables: List[Dict[str, Any]]):
    if "nested_loop" in query_block:
        nested_loops = query_block["nested_loop"]
        for loop in nested_loops:
            table = loop.get("table", {})
            tables.append(table)
def get_grouping(query_block: Dict[str, Any], tables: List[Dict[str, Any]]):
    if "grouping_operation" in query_block:
        grouping_op = query_block["grouping_operation"]
        get_nested_loop(grouping_op, tables)
def get_ordering(query_block: Dict[str, Any], tables: List[Dict[str, Any]]):
    if "ordering_operation" in query_block:
        ordering_op = query_block["ordering_operation"]
        get_grouping(ordering_op, tables)

def parse_row_plan(json_str: str, row_stats: Dict[str, Any]) -> List[List[float]]:
    """
    Parses a JSON query plan and converts it into a torch_geometric HeteroData object.
    
    Args:
        json_str (str): A JSON string representing the query plan.
    
    Returns:
        HeteroData: The resulting heterogeneous graph data.
    """

    data = HeteroData()
    
    # Collect all tables
    query_block = json_str.get("query_block", {})
    query_cost = query_block.get("cost_info", {})
    query_cost = np.log1p(float(query_cost.get("query_cost", 0.0)))
    query_cost = (query_cost - row_stats['query_cost']['center']) / row_stats['query_cost']['scale']

    sort_cost = query_block.get("ordering_operation", {}).get("cost_info", {}).get("sort_cost", 0.0)
    sort_cost = np.log1p(float(sort_cost))
    sort_cost = (sort_cost - row_stats['sort_cost']['center']) / row_stats['sort_cost']['scale']

    # TODO: sort cost, file cost ...
    tables = []
    
    get_ordering(query_block, tables)
    get_grouping(query_block, tables)
    get_nested_loop(query_block, tables)

    table_features = []
    for table in tables:
        inner_feature  = []

        # table_name
        
        access_type = table.get("access_type", "UNKNOWN")
        access_type_encoded = row_stats['access_type']['value_dict'].get(access_type, row_stats['access_type']['no_vals'])
        access_type_onehot = np.eye(row_stats['access_type']['no_vals'] + 1)[access_type_encoded]
        inner_feature.extend(access_type_onehot)
        
        # possible_keys
        
        rows_examined_per_scan = np.log1p(float(table.get("rows_examined_per_scan", 0)))    
        rows_examined_per_scan = (rows_examined_per_scan - row_stats['rows_examined_per_scan']['center']) / row_stats['rows_examined_per_scan']['scale']
        
        rows_produced_per_join = np.log1p(float(table.get("rows_produced_per_join", 0)))    
        rows_produced_per_join = (rows_produced_per_join - row_stats['rows_produced_per_join']['center']) / row_stats['rows_produced_per_join']['scale']
        
        filtered = np.log1p(float(table.get("filtered", 0)))    
        filtered = (filtered - row_stats['filtered']['center']) / row_stats['filtered']['scale']
        
        cost_info = table.get("cost_info", {})
        read_cost = np.log1p(float(cost_info.get("read_cost", 0.0)))    
        read_cost = (read_cost - row_stats['read_cost']['center']) / row_stats['read_cost']['scale']
        
        eval_cost = np.log1p(float(cost_info.get("eval_cost", 0.0)))
        eval_cost = (eval_cost - row_stats['eval_cost']['center']) / row_stats['eval_cost']['scale']
        
        prefix_cost = np.log1p(float(cost_info.get("prefix_cost", 0.0)))
        prefix_cost = (prefix_cost - row_stats['prefix_cost']['center']) / row_stats['prefix_cost']['scale']

        data_read = cost_info.get("data_read_per_join", "0")
        # Convert data_read to numeric value, assuming units like G, M, etc.
        data_read_numeric = convert_data_size_to_numeric(data_read)
        data_read_numeric = (data_read_numeric - row_stats['data_read_per_join']['center']) / row_stats['data_read_per_join']['scale']

        key_length = np.log1p(float(table.get("key_length", 0)))
        key_length = (key_length - row_stats['key_length']['center']) / row_stats['key_length']['scale']

        used_key_parts = table.get("used_key_parts", [])
        used_columns = table.get("used_columns", [])
        # Add more features as needed
        inner_feature.extend([rows_examined_per_scan, rows_produced_per_join, filtered, read_cost, eval_cost, prefix_cost, data_read_numeric, key_length, len(used_key_parts), len(used_columns)])
        # print(f"len(inner_feature): {len(inner_feature)}")
        table_features.append(inner_feature)

    if len(table_features) == 0:
        table_features = np.zeros((1, 16))
    data['table_features'] = torch.tensor(table_features, dtype=torch.float)
    data['query_cost'] = torch.tensor([query_cost], dtype=torch.float)
    data['sort_cost'] = torch.tensor([sort_cost], dtype=torch.float)

    return data

def parse_column_plan(plan: Dict[str, Any]) -> HeteroData:
    """
    Parses the column execution plan into a PyTorch Geometric HeteroData object.

    Args:
        plan (Dict[str, Any]): The column plan JSON as a dictionary.

    Returns:
        HeteroData: The constructed heterogeneous graph.
    """
    data = HeteroData()
    indexer = NodeIndexer()

    # Temporary storage for features
    operation_types = []
    table_names = []
    condition_preds = []

    # Edges storage
    edges = {
        ('operation', 'op_to_table', 'table'): [],
        ('operation', 'op_to_condition', 'condition'): [],
        ('operation', 'op_to_operation', 'operation'): []
    }

    def traverse(node_name: str, content: Any, parent_op_id: int = None):
        """
        Recursively traverses the column plan to extract nodes and edges.

        Args:
            node_name (str): The name of the current node.
            content (Any): The content of the current node.
            parent_op_id (int, optional): The parent operation node ID. Defaults to None.
        """
        current_op_id = None
        if isinstance(content, dict):
            # Identify if the current node is an operation by checking for '(' in the node name
            if '(' in node_name and ')' in node_name:
                op_type = node_name.split('(')[0].strip()
                operation_identifier = f"operation_{op_type}_{node_name}"
                current_op_id = indexer.add_node("operation", operation_identifier)
                operation_types.append(op_type)
                logger.debug(f"Added operation node: {operation_identifier} with ID {current_op_id}")

                # Add edge from parent operation to current operation
                if parent_op_id is not None:
                    edges[('operation', 'op_to_operation', 'operation')].append((parent_op_id, current_op_id))
                    logger.debug(f"Connected operation {parent_op_id} to {current_op_id} via op_to_operation")

            # Handle specific keys for tables and conditions
            if "InputTable" in content:
                table_info = content["InputTable"]
                table_name = table_info.get("TableName", "unknown_table")
                table_identifier = f"table_{table_name}"
                table_id = indexer.add_node("table", table_identifier)
                table_names.append(table_name)
                logger.debug(f"Added table node: {table_identifier} with ID {table_id}")

                # Add edge from current operation to table
                if current_op_id is not None:
                    edges[('operation', 'op_to_table', 'table')].append((current_op_id, table_id))
                    logger.debug(f"Connected operation {current_op_id} to table {table_id} via op_to_table")

            if "Pred" in content:
                pred = content["Pred"]
                condition_identifier = f"condition_pred_{pred}"
                condition_id = indexer.add_node("condition", condition_identifier)
                condition_preds.append(pred)
                logger.debug(f"Added condition node: {condition_identifier} with ID {condition_id}")

                # Add edge from current operation to condition
                if current_op_id is not None:
                    edges[('operation', 'op_to_condition', 'condition')].append((current_op_id, condition_id))
                    logger.debug(f"Connected operation {current_op_id} to condition {condition_id} via op_to_condition")

            # Traverse child nodes
            for key, value in content.items():
                if isinstance(value, dict) or isinstance(value, list):
                    traverse(key, value, current_op_id)
                else:
                    # Optionally handle leaf nodes or additional features
                    pass

        elif isinstance(content, list):
            for item in content:
                traverse(node_name, item, parent_op_id)
        else:
            # Leaf node (e.g., string, number), do nothing or handle if necessary
            pass

    # Start traversing from the root
    for root_op, root_content in plan.items():
        traverse(root_op, root_content)
        logger.debug(f"Finished traversing root operation: {root_op}")

    # Encode categorical features
    # Operation Types
    if operation_types:
        op_encoded, op_mapping = encode_categorical(operation_types)
        data['operation'].x = torch.tensor(op_encoded, dtype=torch.float)
        logger.debug(f"Encoded operation types: {op_mapping}")
    else:
        data['operation'].x = torch.empty((0, 0))
        logger.warning("No operation types found in column plan.")

    # Table Names
    if table_names:
        table_encoded, table_mapping = encode_categorical(table_names)
        data['table'].x = torch.tensor(table_encoded, dtype=torch.float)
        logger.debug(f"Encoded table names: {table_mapping}")
    else:
        data['table'].x = torch.empty((0, 0))
        logger.warning("No table names found in column plan.")

    # Condition Predicates
    if condition_preds:
        cond_encoded, cond_mapping = encode_categorical(condition_preds)
        data['condition'].x = torch.tensor(cond_encoded, dtype=torch.float)
        logger.debug(f"Encoded condition predicates: {cond_mapping}")
    else:
        data['condition'].x = torch.empty((0, 0))
        logger.warning("No condition predicates found in column plan.")

    data.edge_index_dict = {}
    # Assign edges
    for edge_type, edge_list in edges.items():
        if edge_list:
            src_nodes, dst_nodes = zip(*edge_list)
            edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
            data[edge_type[0], edge_type[1], edge_type[2]].edge_index = edge_index
            logger.debug(f"Added edge type {edge_type} with {len(edge_list)} edges.")

    return data



class QueryPlanDataset(Dataset):
    """
    Custom Dataset for loading query plan pairs and labels.
    """
    def __init__(self, data_dir: str, row_plan_dir: str, column_plan_dir: str, labels_file: str, transform=None):
        super().__init__()
        self.row_plan_dir = row_plan_dir
        self.column_plan_dir = column_plan_dir
        self.labels_df = pd.read_csv(labels_file)
        self.query_ids = self.labels_df['query_id'].tolist()
        self.labels = self.labels_df['use_imci'].tolist()  # Ensure 'use_imci' is binary (0 or 1)
        self.transform = transform
        row_stats_file = os.path.join(data_dir, 'row_plan_statistics.json')
        with open(row_stats_file, 'r') as f:
            self.row_stats = json.load(f)

    def len(self):
        return len(self.query_ids)

    def get(self, idx):
        # print(f"idx: {idx}")
        query_id = self.query_ids[idx]
        label = self.labels[idx]

        # Load row plan
        row_plan_path = os.path.join(self.row_plan_dir, f'{query_id}.json')
        if not os.path.exists(row_plan_path):
            logger.error(f"Row plan file not found for query ID {query_id}")
            raise FileNotFoundError(f"Row plan file not found for query ID {query_id}")
        with open(row_plan_path, 'r') as f:
            row_plan = json.load(f)
        row_graph = parse_row_plan(row_plan, self.row_stats)

        # Load column plan
        column_plan_path = os.path.join(self.column_plan_dir, f'{query_id}.json')
        if not os.path.exists(column_plan_path):
            logger.error(f"Column plan file not found for query ID {query_id}")
            raise FileNotFoundError(f"Column plan file not found for query ID {query_id}")
        with open(column_plan_path, 'r') as f:
            column_plan = json.load(f)
        column_graph = parse_column_plan(column_plan)

        # Apply transforms if any
        if self.transform:
            # row_graph = self.transform(row_graph)
            column_graph = self.transform(column_graph)

        return row_graph, column_graph, torch.tensor(label, dtype=torch.long)
    
class RowPlanLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, device):
        super(RowPlanLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.device = device

    def forward(self, x):
        x = x.unsqueeze(0)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class HeteroGNNEncoder(torch.nn.Module):
    """
    Heterogeneous GNN Encoder for encoding HeteroData graphs into fixed-size embeddings.
    """
    def __init__(self, hidden_channels, output_dim):
        super().__init__()
        self.conv1 = HeteroConv({
            ('operation', 'op_to_table_edge', 'table'): GCNConv(-1, hidden_channels, add_self_loops=False),
            ('operation', 'op_to_condition_edge', 'condition'): GCNConv(-1, hidden_channels, add_self_loops=False),
            ('operation', 'op_to_operation_edge', 'operation'): GCNConv(-1, hidden_channels, add_self_loops=False)
        }, aggr='sum')

        self.conv2 = HeteroConv({
            ('operation', 'op_to_table_edge', 'table'): GCNConv(hidden_channels, hidden_channels, add_self_loops=False),
            ('operation', 'op_to_condition_edge', 'condition'): GCNConv(hidden_channels, hidden_channels, add_self_loops=False),
            ('operation', 'op_to_operation_edge', 'operation'): GCNConv(hidden_channels, hidden_channels, add_self_loops=False)
        }, aggr='sum')

        self.relu = ReLU()
        self.dropout = Dropout(p=0.5)
        self.fc = Linear(hidden_channels, output_dim)  # Final output dimension after pooling
        self.output_dim = output_dim

    def forward(self, data: HeteroData) -> torch.Tensor:
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict

        # Initialize a flag to check if any edge is present
        edge_present = False

        # Check if edges are available for each edge type
        for edge_type, edge_index in edge_index_dict.items():
            if edge_index.size(0) > 0:  # Check if there are edges
                edge_present = True
                break

        if not edge_present:
            # If no edges are present, skip the convolutions and directly pass through the node features
            
            graph_embedding = torch.zeros([self.output_dim])
            return graph_embedding

        # Perform the two convolution layers
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: self.relu(x) for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: self.relu(x) for key, x in x_dict.items()}
        x_dict = {key: self.dropout(x) for key, x in x_dict.items()}

        # Apply global pooling to aggregate node features into graph-level embeddings
        pooled_embeddings = []
        for node_type, x in x_dict.items():
            if 'batch' in data[node_type]:
                batch = data[node_type].batch
            else:
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            pooled_embeddings.append(global_mean_pool(x, batch))
        if len(pooled_embeddings) > 0:
            # Combine embeddings from all node types
            graph_embedding = torch.cat(pooled_embeddings, dim=1)

            # Pass through the final linear layer
            graph_embedding = self.fc(graph_embedding)
        else:
            graph_embedding = torch.zeros([self.output_dim])

        return graph_embedding


class PlanClassifier(torch.nn.Module):
    """
    Model that encodes row and column plans and classifies which is faster.
    """
    def __init__(self, hidden_channels, embedding_dim, column_embedding_dim=64, output_dim=2, device='cpu'):
        super().__init__()
        self.column_embedding_dim = column_embedding_dim
        self.row_encoder = RowPlanLSTM(16, hidden_channels, 2, hidden_channels, device)
        self.encoder = HeteroGNNEncoder(hidden_channels, column_embedding_dim)
        self.linear1 = Linear(embedding_dim + column_embedding_dim + 2, hidden_channels)
        self.relu = ReLU()
        self.dropout = Dropout(p=0.5)
        self.linear2 = Linear(hidden_channels, output_dim)
        self.device = device

    def forward(self,row_data: HeteroData, column_data: HeteroData) -> torch.Tensor:
        table_features = row_data.table_features
        query_cost = row_data.query_cost
        sort_cost = row_data.sort_cost

        lstm_out = self.row_encoder(table_features)
        lstm_out = torch.sum(lstm_out, dim=0).squeeze(0)
        row_embedding = torch.cat([lstm_out, query_cost, sort_cost])
        # row_embedding = torch.zeros_like(row_embedding)
        column_embedding = self.encoder(column_data)
        # column_embedding = torch.zeros_like(column_embedding)
        # print(f"row_embedding: {row_embedding.shape}, column_embedding: {column_embedding.shape}")
        # Concatenate row and column embeddings
        combined = torch.cat([row_embedding, column_embedding], dim=0)

        # Classification layers
        x = self.linear1(combined)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x

# Training and Evaluation Functions

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for row_graph, column_graph, labels in loader:
        row_graph = row_graph.to(device)
        column_graph = column_graph.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        out = model(row_graph, column_graph).unsqueeze(0)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    preds = []
    true = []
    with torch.no_grad():
        for row_graph, column_graph, labels in loader:
            row_graph = row_graph.to(device)
            column_graph = column_graph.to(device)
            labels = labels.to(device)

            out = model(row_graph, column_graph)
            pred = out.argmax(dim=0).unsqueeze(0)
            preds.extend(pred.cpu().numpy())
            true.extend(labels.cpu().numpy())
    acc = accuracy_score(true, preds)
    f1 = f1_score(true, preds)
    roc_auc = roc_auc_score(true, preds)
    return acc, f1, roc_auc


def main():
    # Define directories and files
    data_dir = '/home/wuy/query_costs'
    row_plan_dir = os.path.join(data_dir, 'row_plans')
    column_plan_dir = os.path.join(data_dir, 'column_plans')
    labels_file = os.path.join(data_dir, 'query_costs.csv')

    # Check if directories and labels file exist
    if not os.path.exists(row_plan_dir):
        logger.error(f"Row plan directory not found: {row_plan_dir}")
        return
    if not os.path.exists(column_plan_dir):
        logger.error(f"Column plan directory not found: {column_plan_dir}")
        return
    if not os.path.exists(labels_file):
        logger.error(f"Labels file not found: {labels_file}")
        return

    # Create dataset
    dataset = QueryPlanDataset(data_dir, row_plan_dir, column_plan_dir, labels_file)
    logger.info(f"Total samples: {len(dataset)}")

    # Split into train and validation
    train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
    logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # Create DataLoaders
    batch_size = 1
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Initialize model, optimizer, and loss
    hidden_channels = 64
    # Determine embedding_dim based on node feature dimensions
    # For example, if 'operation' has 10 features, 'table' has 15, etc., set accordingly
    # Here, we'll set embedding_dim to hidden_channels for simplicity
    embedding_dim = hidden_channels
    model = PlanClassifier(hidden_channels, embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    num_epochs = 100
    best_val_auc = 0
    for epoch in range(1, num_epochs + 1):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_acc, val_f1, val_auc = evaluate(model, val_loader, device)
        logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}, Val ROC-AUC={val_auc:.4f}")

        # Save the best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), os.path.join(data_dir, 'best_model.pth'))
            logger.info(f"Saved best model with ROC-AUC={best_val_auc:.4f}")

    logger.info("Training complete.")

    

if __name__ == "__main__":
    main()
