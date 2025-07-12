"""Graph Neural Network model for algebraic step prediction.

This module implements the GNN architecture that processes algebraic expressions
as graphs and predicts the next step in solving.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.data import Data, Batch
from typing import Dict, Tuple, Optional, List
import math


class GraphEncoder(nn.Module):
    """Enhanced graph encoder using Graph Convolutional or Attention layers with residual connections."""
    
    def __init__(self,
                 input_dim: int = 20,  # Node feature dimension
                 hidden_dim: int = 128,
                 output_dim: int = 256,
                 num_layers: int = 3,
                 dropout: float = 0.1,
                 conv_type: str = "gcn",  # "gcn", "gat", or "gat_multi"
                 aggregation: str = "mean",  # "mean", "sum", "max", or "attention"
                 use_residual: bool = True,
                 use_layer_norm: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.conv_type = conv_type
        self.aggregation = aggregation
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        
        # Create convolutional layers
        self.convs = nn.ModuleList()
        
        # First layer
        if conv_type == "gcn":
            self.convs.append(GCNConv(input_dim, hidden_dim))
        elif conv_type == "gat":
            self.convs.append(GATConv(input_dim, hidden_dim, heads=4, concat=False))
        elif conv_type == "gat_multi":
            self.convs.append(GATConv(input_dim, hidden_dim, heads=8, concat=True))
            # Adjust hidden_dim for multi-head concatenation
            hidden_dim = hidden_dim * 8
        else:
            raise ValueError(f"Unknown conv_type: {conv_type}")
        
        # Hidden layers
        for _ in range(num_layers - 2):
            if conv_type == "gcn":
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif conv_type == "gat":
                self.convs.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))
            elif conv_type == "gat_multi":
                self.convs.append(GATConv(hidden_dim, hidden_dim, heads=8, concat=True))
                hidden_dim = hidden_dim * 8
        
        # Output layer
        if num_layers > 1:
            if conv_type == "gcn":
                self.convs.append(GCNConv(hidden_dim, output_dim))
            elif conv_type == "gat":
                self.convs.append(GATConv(hidden_dim, output_dim, heads=1, concat=False))
            elif conv_type == "gat_multi":
                self.convs.append(GATConv(hidden_dim, output_dim, heads=1, concat=False))
        
        # Layer normalization layers
        if use_layer_norm:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(hidden_dim) for _ in range(num_layers - 1)
            ])
            if num_layers > 1:
                self.layer_norms.append(nn.LayerNorm(output_dim))
        else:
            self.layer_norms = None
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Aggregation function
        if aggregation == "mean":
            self.aggregate = global_mean_pool
        elif aggregation == "sum":
            self.aggregate = global_add_pool
        elif aggregation == "max":
            self.aggregate = global_max_pool
        elif aggregation == "attention":
            self.attention_weights = nn.Linear(output_dim, 1)
            self.aggregate = self._attention_pool
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
        
        # Output projection for attention aggregation
        if aggregation == "attention":
            self.output_projection = nn.Linear(output_dim, output_dim)
        else:
            self.output_projection = None
    
    def _attention_pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Attention-based graph pooling."""
        # Compute attention weights
        attention_weights = self.attention_weights(x)  # [num_nodes, 1]
        
        # Apply softmax within each batch
        batch_size = batch.max().item() + 1
        pooled_features = []
        
        for i in range(batch_size):
            mask = (batch == i)
            if mask.sum() > 0:
                batch_weights = attention_weights[mask]
                batch_features = x[mask]
                
                # Softmax attention weights
                attention_scores = F.softmax(batch_weights, dim=0)
                
                # Weighted sum
                pooled = torch.sum(batch_features * attention_scores, dim=0)
                pooled_features.append(pooled)
            else:
                # Empty batch - use zero vector
                pooled_features.append(torch.zeros_like(x[0]))
        
        pooled = torch.stack(pooled_features)
        
        # Apply output projection
        if self.output_projection is not None:
            pooled = self.output_projection(pooled)
        
        return pooled
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Enhanced forward pass through the graph encoder with residual connections.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment for nodes [num_nodes]
            
        Returns:
            Graph-level representations [batch_size, output_dim]
        """
        # Store initial input for residual connection
        if self.use_residual and self.input_dim == self.hidden_dim:
            residual = x
        
        # Apply convolutional layers with residual connections
        for i, conv in enumerate(self.convs):
            x_prev = x
            
            # Apply convolution
            x = conv(x, edge_index)
            
            # Apply layer normalization if enabled
            if self.layer_norms is not None and i < len(self.layer_norms):
                x = self.layer_norms[i](x)
            
            # Apply activation and dropout (except last layer)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout(x)
                
                # Add residual connection if dimensions match
                if self.use_residual and x.size(-1) == x_prev.size(-1):
                    x = x + x_prev
        
        # Aggregate node features to graph level
        graph_repr = self.aggregate(x, batch)
        
        return graph_repr


class AlgebraicStepDecoder(nn.Module):
    """Decoder for predicting the next algebraic expression."""
    
    def __init__(self,
                 input_dim: int = 256,
                 hidden_dim: int = 512,
                 vocab_size: int = 1000,
                 max_length: int = 100,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.num_layers = num_layers
        
        # Embedding for output tokens
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, dropout, max_length)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Project graph encoding to decoder dimension
        self.graph_projection = nn.Linear(input_dim, hidden_dim)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.graph_projection.weight)
        nn.init.xavier_uniform_(self.output_projection.weight)
    
    def forward(self,
                graph_encoding: torch.Tensor,
                target: Optional[torch.Tensor] = None,
                target_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the decoder.
        
        Args:
            graph_encoding: Graph representations [batch_size, input_dim]
            target: Target token sequences [batch_size, seq_len] (training only)
            target_mask: Attention mask for target [seq_len, seq_len]
            
        Returns:
            Logits for next tokens [batch_size, seq_len, vocab_size]
        """
        batch_size = graph_encoding.size(0)
        
        # Project graph encoding
        memory = self.graph_projection(graph_encoding).unsqueeze(1)  # [batch, 1, hidden]
        
        if target is not None:
            # Training mode - teacher forcing
            tgt_embed = self.embedding(target)  # [batch, seq_len, hidden]
            tgt_embed = self.pos_encoding(tgt_embed)
            
            # Decode
            output = self.transformer_decoder(
                tgt_embed,
                memory,
                tgt_mask=target_mask
            )
            
            # Project to vocabulary
            logits = self.output_projection(output)
            
            return logits
        else:
            # Inference mode - autoregressive generation
            # This would be implemented for actual inference
            raise NotImplementedError("Inference mode not yet implemented")


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class GraphNeuralAlgebraTutor(nn.Module):
    """Complete Graph Neural Algebra Tutor model."""
    
    def __init__(self,
                 node_feature_dim: int = 20,
                 graph_hidden_dim: int = 128,
                 graph_output_dim: int = 256,
                 decoder_hidden_dim: int = 512,
                 vocab_size: int = 1000,
                 max_expr_length: int = 100,
                 num_graph_layers: int = 3,
                 num_decoder_layers: int = 2,
                 dropout: float = 0.1,
                 conv_type: str = "gcn",
                 aggregation: str = "mean"):
        super().__init__()
        
        # Graph encoder
        self.encoder = GraphEncoder(
            input_dim=node_feature_dim,
            hidden_dim=graph_hidden_dim,
            output_dim=graph_output_dim,
            num_layers=num_graph_layers,
            dropout=dropout,
            conv_type=conv_type,
            aggregation=aggregation
        )
        
        # Expression decoder
        self.decoder = AlgebraicStepDecoder(
            input_dim=graph_output_dim,
            hidden_dim=decoder_hidden_dim,
            vocab_size=vocab_size,
            max_length=max_expr_length,
            num_layers=num_decoder_layers,
            dropout=dropout
        )
        
        # Operation classifier (optional)
        self.operation_classifier = nn.Sequential(
            nn.Linear(graph_output_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 15)  # Number of operation types
        )
    
    def forward(self,
                current_graph: Batch,
                target_tokens: Optional[torch.Tensor] = None,
                target_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through the model.
        
        Args:
            current_graph: Batched graph data
            target_tokens: Target expression tokens (training only)
            target_mask: Attention mask for target
            
        Returns:
            Dictionary with:
                - expression_logits: Logits for next expression tokens
                - operation_logits: Logits for operation classification
                - graph_encoding: Graph-level representations
        """
        # Encode current graph
        graph_encoding = self.encoder(
            current_graph.x,
            current_graph.edge_index,
            current_graph.batch
        )
        
        # Predict operation type
        operation_logits = self.operation_classifier(graph_encoding)
        
        # Decode next expression
        expression_logits = None
        if target_tokens is not None or not self.training:
            expression_logits = self.decoder(
                graph_encoding,
                target_tokens,
                target_mask
            )
        
        return {
            'expression_logits': expression_logits,
            'operation_logits': operation_logits,
            'graph_encoding': graph_encoding
        }
    
    def predict_operation(self, current_graph: Batch) -> torch.Tensor:
        """Predict just the operation type.
        
        Args:
            current_graph: Batched graph data
            
        Returns:
            Operation predictions [batch_size]
        """
        with torch.no_grad():
            output = self.forward(current_graph)
            operation_probs = F.softmax(output['operation_logits'], dim=-1)
            predictions = operation_probs.argmax(dim=-1)
        
        return predictions
    
    def generate_next_expression(self,
                               current_graph: Batch,
                               tokenizer,
                               max_length: int = 50,
                               temperature: float = 1.0) -> List[str]:
        """Generate the next expression autoregressively.
        
        Args:
            current_graph: Batched graph data
            tokenizer: Tokenizer for converting between tokens and text
            max_length: Maximum generation length
            temperature: Sampling temperature
            
        Returns:
            List of generated expressions (one per graph in batch)
        """
        # This would implement autoregressive generation
        # For now, returning placeholder
        batch_size = current_graph.num_graphs
        return ["x = placeholder" for _ in range(batch_size)]


class AlgebraModelConfig:
    """Configuration for the algebra tutor model."""
    
    def __init__(self):
        # Graph encoder settings
        self.node_feature_dim = 20
        self.graph_hidden_dim = 128
        self.graph_output_dim = 256
        self.num_graph_layers = 3
        self.conv_type = "gcn"  # or "gat"
        self.aggregation = "mean"
        
        # Decoder settings
        self.decoder_hidden_dim = 512
        self.vocab_size = 1000
        self.max_expr_length = 100
        self.num_decoder_layers = 2
        
        # Training settings
        self.dropout = 0.1
        self.learning_rate = 1e-3
        self.weight_decay = 1e-5
        self.batch_size = 32
        self.num_epochs = 100
        self.gradient_clip = 1.0
        
        # Loss weights
        self.expression_loss_weight = 1.0
        self.operation_loss_weight = 0.5
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return vars(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'AlgebraModelConfig':
        """Create from dictionary."""
        config = cls()
        for key, value in config_dict.items():
            setattr(config, key, value)
        return config


if __name__ == "__main__":
    # Test the model
    config = AlgebraModelConfig()
    model = GraphNeuralAlgebraTutor(
        node_feature_dim=config.node_feature_dim,
        graph_hidden_dim=config.graph_hidden_dim,
        graph_output_dim=config.graph_output_dim,
        decoder_hidden_dim=config.decoder_hidden_dim,
        vocab_size=config.vocab_size,
        max_expr_length=config.max_expr_length,
        num_graph_layers=config.num_graph_layers,
        num_decoder_layers=config.num_decoder_layers,
        dropout=config.dropout,
        conv_type=config.conv_type,
        aggregation=config.aggregation
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    from torch_geometric.data import Data, Batch
    
    # Create dummy batch
    graphs = []
    for _ in range(4):
        x = torch.randn(5, config.node_feature_dim)
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4],
                                  [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long)
        graphs.append(Data(x=x, edge_index=edge_index))
    
    batch = Batch.from_data_list(graphs)
    target_tokens = torch.randint(0, config.vocab_size, (4, 20))
    
    # Forward pass
    output = model(batch, target_tokens)
    print(f"Operation logits shape: {output['operation_logits'].shape}")
    print(f"Expression logits shape: {output['expression_logits'].shape}")
    print(f"Graph encoding shape: {output['graph_encoding'].shape}") 