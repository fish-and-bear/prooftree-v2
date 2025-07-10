"""Simplified step predictor model for algebraic operations.

This module implements a simpler model that predicts the operation type
and relevant parameters for the next step, which can then be applied
using rule-based transformations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Batch
from typing import Dict, List, Tuple, Optional
from enum import IntEnum


class OperationType(IntEnum):
    """Enumeration of algebraic operation types."""
    ADD_TO_BOTH_SIDES = 0
    SUBTRACT_FROM_BOTH_SIDES = 1
    MULTIPLY_BOTH_SIDES = 2
    DIVIDE_BOTH_SIDES = 3
    EXPAND = 4
    FACTOR = 5
    COMBINE_LIKE_TERMS = 6
    DISTRIBUTE = 7
    ISOLATE_VARIABLE = 8
    SIMPLIFY = 9
    APPLY_QUADRATIC_FORMULA = 10
    COMPLETE_THE_SQUARE = 11
    NO_OP = 12  # For final state
    
    @classmethod
    def from_string(cls, op_str: str) -> 'OperationType':
        """Convert string operation name to enum."""
        try:
            return cls[op_str.upper()]
        except KeyError:
            return cls.NO_OP


class SimpleGraphEncoder(nn.Module):
    """Simplified graph encoder for algebraic expressions."""
    
    def __init__(self,
                 input_dim: int = 20,
                 hidden_dim: int = 64,
                 output_dim: int = 128,
                 num_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        self.convs.append(GCNConv(hidden_dim, output_dim))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder."""
        for i, (conv, bn) in enumerate(zip(self.convs[:-1], self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Final layer
        x = self.convs[-1](x, edge_index)
        
        # Global pooling
        graph_repr = global_mean_pool(x, batch)
        
        return graph_repr


class StepPredictor(nn.Module):
    """Predicts the next algebraic operation to perform."""
    
    def __init__(self,
                 node_feature_dim: int = 20,
                 hidden_dim: int = 64,
                 graph_output_dim: int = 128,
                 num_operations: int = 13,
                 num_graph_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        
        # Graph encoder
        self.encoder = SimpleGraphEncoder(
            input_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=graph_output_dim,
            num_layers=num_graph_layers,
            dropout=dropout
        )
        
        # Operation classifier
        self.operation_head = nn.Sequential(
            nn.Linear(graph_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_operations)
        )
        
        # Parameter predictor (for operations that need parameters)
        self.param_head = nn.Sequential(
            nn.Linear(graph_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Predicts a scalar parameter
        )
        
        # Binary classifier for whether we're done
        self.done_head = nn.Sequential(
            nn.Linear(graph_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)  # Binary: done or not done
        )
    
    def forward(self, graph: Batch) -> Dict[str, torch.Tensor]:
        """Forward pass to predict next operation.
        
        Args:
            graph: Batched graph data
            
        Returns:
            Dictionary with:
                - operation_logits: Logits for operation type
                - param_value: Predicted parameter value
                - done_logits: Logits for whether solving is complete
                - graph_encoding: Graph representations
        """
        # Encode graph
        graph_encoding = self.encoder(graph.x, graph.edge_index, graph.batch)
        
        # Predict operation
        operation_logits = self.operation_head(graph_encoding)
        
        # Predict parameter
        param_value = self.param_head(graph_encoding)
        
        # Predict if done
        done_logits = self.done_head(graph_encoding)
        
        return {
            'operation_logits': operation_logits,
            'param_value': param_value,
            'done_logits': done_logits,
            'graph_encoding': graph_encoding
        }
    
    def predict_step(self, graph: Batch) -> Tuple[List[int], List[float], List[bool]]:
        """Predict the next step for a batch of graphs.
        
        Args:
            graph: Batched graph data
            
        Returns:
            Tuple of (operations, parameters, done_flags)
        """
        with torch.no_grad():
            output = self.forward(graph)
            
            # Get predictions
            operations = output['operation_logits'].argmax(dim=-1).tolist()
            parameters = output['param_value'].squeeze(-1).tolist()
            done_probs = F.softmax(output['done_logits'], dim=-1)
            done_flags = (done_probs[:, 1] > 0.5).tolist()
        
        return operations, parameters, done_flags


class StepPredictorConfig:
    """Configuration for the step predictor model."""
    
    def __init__(self):
        self.node_feature_dim = 20
        self.hidden_dim = 64
        self.graph_output_dim = 128
        self.num_operations = 13
        self.num_graph_layers = 3
        self.dropout = 0.1
        
        # Training settings
        self.learning_rate = 1e-3
        self.weight_decay = 1e-5
        self.batch_size = 32
        self.num_epochs = 50
        self.gradient_clip = 1.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return vars(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'StepPredictorConfig':
        """Create from dictionary."""
        config = cls()
        for key, value in config_dict.items():
            setattr(config, key, value)
        return config


def create_step_predictor(config: Optional[StepPredictorConfig] = None) -> StepPredictor:
    """Create a step predictor model with the given configuration.
    
    Args:
        config: Model configuration (uses default if None)
        
    Returns:
        StepPredictor model
    """
    if config is None:
        config = StepPredictorConfig()
    
    model = StepPredictor(
        node_feature_dim=config.node_feature_dim,
        hidden_dim=config.hidden_dim,
        graph_output_dim=config.graph_output_dim,
        num_operations=config.num_operations,
        num_graph_layers=config.num_graph_layers,
        dropout=config.dropout
    )
    
    return model


if __name__ == "__main__":
    # Test the model
    from torch_geometric.data import Data
    
    config = StepPredictorConfig()
    model = create_step_predictor(config)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create dummy batch
    graphs = []
    for _ in range(4):
        x = torch.randn(5, config.node_feature_dim)
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4],
                                  [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long)
        graphs.append(Data(x=x, edge_index=edge_index))
    
    batch = Batch.from_data_list(graphs)
    
    # Test forward pass
    output = model(batch)
    print(f"\nOutput shapes:")
    print(f"Operation logits: {output['operation_logits'].shape}")
    print(f"Parameter values: {output['param_value'].shape}")
    print(f"Done logits: {output['done_logits'].shape}")
    
    # Test prediction
    operations, parameters, done_flags = model.predict_step(batch)
    print(f"\nPredictions:")
    print(f"Operations: {operations}")
    print(f"Parameters: {parameters}")
    print(f"Done flags: {done_flags}") 