import torch
from torch_geometric.data import Data, Batch
import pytest
from src.models.gnn_model import GraphNeuralAlgebraTutor

def make_dummy_graph(num_nodes=5, num_edges=6, node_feature_dim=20):
    x = torch.randn((num_nodes, node_feature_dim))
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    return Data(x=x, edge_index=edge_index)

def test_gnn_model_instantiation():
    model = GraphNeuralAlgebraTutor()
    assert model is not None

def test_gnn_model_forward_pass():
    model = GraphNeuralAlgebraTutor()
    # Create a batch of 2 dummy graphs
    data_list = [make_dummy_graph(), make_dummy_graph()]
    batch = Batch.from_data_list(data_list)
    # Dummy target tokens (batch_size, seq_len)
    target_tokens = torch.randint(0, 1000, (2, 10))
    out = model(batch, target_tokens)
    assert isinstance(out, dict)
    assert 'expression_logits' in out
    assert out['expression_logits'].shape[0] == 2  # batch size
    assert out['expression_logits'].shape[1] == 10  # seq_len

def test_gnn_model_predict_operation():
    model = GraphNeuralAlgebraTutor()
    data_list = [make_dummy_graph(), make_dummy_graph()]
    batch = Batch.from_data_list(data_list)
    op_logits = model.predict_operation(batch)
    assert op_logits.shape[0] == 2 