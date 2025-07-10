"""Expression Graph module for converting algebraic expressions to graph representations.

This module provides functionality to convert SymPy expressions into graph structures
that can be processed by Graph Neural Networks.
"""

import torch
import torch_geometric
from torch_geometric.data import Data
import sympy as sp
from sympy import Symbol, Integer, Float, Rational, Add, Mul, Pow, Eq, sin, cos, tan, exp, log
import networkx as nx
from typing import Union, Optional, List, Tuple
import re

class ExpressionNode:
    """Represents a node in the expression graph."""
    
    def __init__(self, node_id: int, node_type: str, value: Optional[Union[str, float]] = None):
        self.id = node_id
        self.type = node_type
        self.value = value
        self.features = None
    
    def __repr__(self):
        return f"ExpressionNode(id={self.id}, type='{self.type}', value={self.value})"

class ExpressionGraph:
    """Converts algebraic expressions to graph representations."""
    
    # Node type mappings
    OPERATOR_TYPES = {
        Add: 'add',
        Mul: 'mul',
        Pow: 'pow',
        Eq: 'eq',
        sp.core.add.Add: 'add',
        sp.core.mul.Mul: 'mul',
        sp.core.power.Pow: 'pow',
    }
    
    FUNCTION_TYPES = {
        sin: 'sin',
        cos: 'cos',
        tan: 'tan',
        exp: 'exp',
        log: 'log',
    }
    
    # Feature dimensions
    NODE_TYPE_DIM = 4  # operator, variable, constant, function
    OPERATOR_TYPE_DIM = 10  # add, mul, pow, eq, etc.
    VALUE_DIM = 1  # for constants
    VAR_DIM = 5  # for variable names
    
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.node_map = {}
        self.next_id = 0
    
    def expr_to_graph(self, expr: Union[sp.Basic, str]) -> Data:
        """Convert expression to graph representation.
        
        Args:
            expr: SymPy expression or string
            
        Returns:
            PyTorch Geometric Data object
        """
        # Reset for new expression
        self.nodes = []
        self.edges = []
        self.node_map = {}
        self.next_id = 0
        
        # Parse string to SymPy if needed
        if isinstance(expr, str):
            expr = self._parse_expression_string(expr)
        
        # Build graph recursively
        root_id = self._build_graph(expr)
        
        # Convert to PyTorch Geometric format
        return self._to_pyg_data()
    
    def _parse_expression_string(self, expr_str: str) -> sp.Basic:
        """Parse string expression to SymPy, handling equations.
        
        Args:
            expr_str: String expression or equation
            
        Returns:
            SymPy expression
        """
        # Clean the string
        expr_str = expr_str.strip()
        
        # Handle equations (containing =)
        if '=' in expr_str:
            # Split on = and parse each side
            left_str, right_str = expr_str.split('=', 1)
            left_expr = sp.parse_expr(left_str.strip())
            right_expr = sp.parse_expr(right_str.strip())
            return Eq(left_expr, right_expr)
        else:
            # Regular expression
            return sp.parse_expr(expr_str)
    
    def _build_graph(self, expr: sp.Basic) -> int:
        """Recursively build graph from SymPy expression.
        
        Args:
            expr: SymPy expression
            
        Returns:
            Node ID of the root of this subexpression
        """
        # Check if we've already processed this expression
        if expr in self.node_map:
            return self.node_map[expr]
        
        node_id = self.next_id
        self.next_id += 1
        
        # Handle different expression types
        if isinstance(expr, Symbol):
            # Variable node
            node = ExpressionNode(node_id, 'variable', str(expr))
            self.nodes.append(node)
            
        elif isinstance(expr, (Integer, Float, Rational)):
            # Constant node
            node = ExpressionNode(node_id, 'constant', float(expr))
            self.nodes.append(node)
            
        elif type(expr) in self.OPERATOR_TYPES:
            # Operator node
            op_type = self.OPERATOR_TYPES[type(expr)]
            node = ExpressionNode(node_id, 'operator', op_type)
            self.nodes.append(node)
            
            # Process arguments
            for arg in expr.args:
                child_id = self._build_graph(arg)
                self.edges.append((node_id, child_id))
                
        elif expr.func in self.FUNCTION_TYPES:
            # Function node
            func_type = self.FUNCTION_TYPES[expr.func]
            node = ExpressionNode(node_id, 'function', func_type)
            self.nodes.append(node)
            
            # Process arguments
            for arg in expr.args:
                child_id = self._build_graph(arg)
                self.edges.append((node_id, child_id))
                
        else:
            # Generic expression - treat as operator
            node = ExpressionNode(node_id, 'operator', str(expr.func))
            self.nodes.append(node)
            
            # Process arguments if any
            if hasattr(expr, 'args'):
                for arg in expr.args:
                    child_id = self._build_graph(arg)
                    self.edges.append((node_id, child_id))
        
        self.node_map[expr] = node_id
        return node_id
    
    def _to_pyg_data(self) -> Data:
        """Convert internal graph representation to PyTorch Geometric Data.
        
        Returns:
            PyTorch Geometric Data object
        """
        # Create node features
        node_features = []
        for node in self.nodes:
            features = self._create_node_features(node)
            node.features = features
            node_features.append(features)
        
        # Stack features
        x = torch.stack(node_features)
        
        # Create edge index
        if self.edges:
            edge_index = torch.tensor(self.edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # Create data object
        data = Data(x=x, edge_index=edge_index)
        data.num_nodes = len(self.nodes)
        
        return data
    
    def _create_node_features(self, node: ExpressionNode) -> torch.Tensor:
        """Create feature vector for a node.
        
        Args:
            node: ExpressionNode
            
        Returns:
            Feature tensor
        """
        features = []
        
        # Node type features (one-hot)
        node_type_vec = torch.zeros(4)
        if node.type == 'operator':
            node_type_vec[0] = 1
        elif node.type == 'variable':
            node_type_vec[1] = 1
        elif node.type == 'constant':
            node_type_vec[2] = 1
        elif node.type == 'function':
            node_type_vec[3] = 1
        features.append(node_type_vec)
        
        # Operator/function type features (one-hot)
        op_type_vec = torch.zeros(self.OPERATOR_TYPE_DIM)
        if node.type in ['operator', 'function'] and isinstance(node.value, str):
            op_map = {
                'add': 0, 'mul': 1, 'pow': 2, 'eq': 3,
                'sin': 4, 'cos': 5, 'tan': 6, 'exp': 7, 'log': 8
            }
            if node.value in op_map:
                op_type_vec[op_map[node.value]] = 1
        features.append(op_type_vec)
        
        # Value features (for constants)
        value_vec = torch.zeros(self.VALUE_DIM)
        if node.type == 'constant' and node.value is not None:
            value_vec[0] = float(node.value)
        features.append(value_vec)
        
        # Variable name embedding (simple hash)
        var_vec = torch.zeros(5)
        if node.type == 'variable' and node.value is not None:
            # Simple hash-based embedding
            hash_val = hash(node.value) % 32
            for i in range(5):
                if hash_val & (1 << i):
                    var_vec[i] = 1
        features.append(var_vec)
        
        return torch.cat(features)
    
    def graph_to_networkx(self, data: Data) -> nx.DiGraph:
        """Convert PyTorch Geometric Data to NetworkX graph for visualization.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            NetworkX directed graph
        """
        G = nx.DiGraph()
        
        # Add nodes with features
        for i, node in enumerate(self.nodes):
            G.add_node(i, 
                      type=node.type,
                      value=node.value,
                      label=self._get_node_label(node))
        
        # Add edges
        edge_index = data.edge_index.numpy()
        for i in range(edge_index.shape[1]):
            G.add_edge(edge_index[0, i], edge_index[1, i])
        
        return G
    
    def _get_node_label(self, node: ExpressionNode) -> str:
        """Get display label for a node.
        
        Args:
            node: ExpressionNode
            
        Returns:
            Display label string
        """
        if node.type == 'variable':
            return str(node.value)
        elif node.type == 'constant':
            return str(node.value)
        elif node.type in ['operator', 'function']:
            op_symbols = {
                'add': '+', 'mul': '×', 'pow': '^', 'eq': '=',
                'sin': 'sin', 'cos': 'cos', 'tan': 'tan',
                'exp': 'exp', 'log': 'log'
            }
            return op_symbols.get(node.value, str(node.value))
        else:
            return str(node.value)

# Global converter instance
_converter = ExpressionGraph()

def expression_to_graph(expr: Union[sp.Basic, str]) -> Data:
    """Convert expression to graph representation.
    
    Args:
        expr: SymPy expression or string
        
    Returns:
        PyTorch Geometric Data object
    """
    return _converter.expr_to_graph(expr)

def visualize_expression_graph(expr: Union[sp.Basic, str], 
                              save_path: Optional[str] = None) -> None:
    """Visualize expression as a graph.
    
    Args:
        expr: Expression to visualize
        save_path: Optional path to save the visualization
    """
    import matplotlib.pyplot as plt
    
    # Convert to graph
    data = expression_to_graph(expr)
    G = _converter.graph_to_networkx(data)
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=1000)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='gray', 
                          arrows=True, arrowsize=20)
    
    # Add labels
    labels = {node: G.nodes[node]['label'] for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=12, font_weight='bold')
    
    plt.title(f"Expression Graph: {expr}")
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close() 