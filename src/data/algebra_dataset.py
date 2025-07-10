"""PyTorch dataset for algebra problems.

This module provides PyTorch Dataset classes for loading and processing
algebra problems for training Graph Neural Networks.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
import sympy as sp

from .dataset_generator import AlgebraProblem, AlgebraDatasetGenerator
from ..graph import expression_to_graph


class AlgebraStepDataset(Dataset):
    """PyTorch dataset for algebra problem steps.
    
    Each item is a pair of (current_state_graph, next_state_graph) representing
    one step in solving an algebra problem.
    """
    
    def __init__(self, 
                 problems: Optional[List[AlgebraProblem]] = None,
                 filepath: Optional[Union[str, Path]] = None,
                 max_steps: Optional[int] = None):
        """Initialize dataset.
        
        Args:
            problems: List of algebra problems (if None, must provide filepath)
            filepath: Path to JSON file containing problems
            max_steps: Maximum number of steps to include per problem
        """
        if problems is None and filepath is None:
            raise ValueError("Must provide either problems or filepath")
        
        if problems is None:
            problems = AlgebraDatasetGenerator.load_dataset(filepath)
        
        self.problems = problems
        self.max_steps = max_steps
        
        # Build step pairs from all problems
        self.step_pairs = []
        self._build_step_pairs()
    
    def _build_step_pairs(self):
        """Build all step pairs from problems."""
        for problem in self.problems:
            steps = problem.steps
            
            # Limit steps if needed
            if self.max_steps and len(steps) > self.max_steps:
                steps = steps[:self.max_steps]
            
            # Create pairs of consecutive steps
            for i in range(len(steps) - 1):
                current_expr = steps[i]['expression']
                next_expr = steps[i + 1]['expression']
                operation = steps[i + 1]['operation']
                description = steps[i + 1]['description']
                
                self.step_pairs.append({
                    'problem_id': problem.problem_id,
                    'problem_type': problem.problem_type,
                    'step_index': i,
                    'current_expression': current_expr,
                    'next_expression': next_expr,
                    'operation': operation,
                    'description': description,
                    'difficulty': problem.difficulty
                })
    
    def __len__(self) -> int:
        return len(self.step_pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[Data, str, int]]:
        """Get a single step pair.
        
        Returns:
            Dictionary containing:
                - current_graph: PyG Data object for current state
                - next_graph: PyG Data object for next state
                - operation: Operation performed
                - metadata: Additional information
        """
        step_pair = self.step_pairs[idx]
        
        # Convert expressions to graphs
        try:
            current_graph = expression_to_graph(step_pair['current_expression'])
            next_graph = expression_to_graph(step_pair['next_expression'])
        except Exception as e:
            print(f"Error converting expressions to graphs: {e}")
            print(f"Current: {step_pair['current_expression']}")
            print(f"Next: {step_pair['next_expression']}")
            # Return dummy data
            current_graph = Data(x=torch.zeros(1, 20), edge_index=torch.empty(2, 0, dtype=torch.long))
            next_graph = Data(x=torch.zeros(1, 20), edge_index=torch.empty(2, 0, dtype=torch.long))
        
        return {
            'current_graph': current_graph,
            'next_graph': next_graph,
            'operation': step_pair['operation'],
            'description': step_pair['description'],
            'metadata': {
                'problem_id': step_pair['problem_id'],
                'problem_type': step_pair['problem_type'],
                'step_index': step_pair['step_index'],
                'difficulty': step_pair['difficulty']
            }
        }


class AlgebraProblemDataset(Dataset):
    """PyTorch dataset for full algebra problems.
    
    Each item is a complete problem with all its solution steps.
    """
    
    def __init__(self,
                 problems: Optional[List[AlgebraProblem]] = None,
                 filepath: Optional[Union[str, Path]] = None):
        """Initialize dataset.
        
        Args:
            problems: List of algebra problems (if None, must provide filepath)
            filepath: Path to JSON file containing problems
        """
        if problems is None and filepath is None:
            raise ValueError("Must provide either problems or filepath")
        
        if problems is None:
            problems = AlgebraDatasetGenerator.load_dataset(filepath)
        
        self.problems = problems
    
    def __len__(self) -> int:
        return len(self.problems)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[List[Data], AlgebraProblem]]:
        """Get a complete problem.
        
        Returns:
            Dictionary containing:
                - problem: The AlgebraProblem object
                - graphs: List of PyG Data objects for each step
        """
        problem = self.problems[idx]
        
        # Convert all steps to graphs
        graphs = []
        for step in problem.steps:
            try:
                graph = expression_to_graph(step['expression'])
                graphs.append(graph)
            except Exception as e:
                print(f"Error converting expression to graph: {e}")
                print(f"Expression: {step['expression']}")
                # Add dummy graph
                graphs.append(Data(x=torch.zeros(1, 20), 
                                 edge_index=torch.empty(2, 0, dtype=torch.long)))
        
        return {
            'problem': problem,
            'graphs': graphs
        }


def collate_step_batch(batch: List[Dict]) -> Dict:
    """Custom collate function for step dataset.
    
    Args:
        batch: List of dictionaries from AlgebraStepDataset
        
    Returns:
        Batched data dictionary
    """
    # Batch the graphs
    current_graphs = [item['current_graph'] for item in batch]
    next_graphs = [item['next_graph'] for item in batch]
    
    current_batch = Batch.from_data_list(current_graphs)
    next_batch = Batch.from_data_list(next_graphs)
    
    # Collect other data
    operations = [item['operation'] for item in batch]
    descriptions = [item['description'] for item in batch]
    metadata = [item['metadata'] for item in batch]
    
    return {
        'current_batch': current_batch,
        'next_batch': next_batch,
        'operations': operations,
        'descriptions': descriptions,
        'metadata': metadata
    }


def create_data_loaders(train_problems: List[AlgebraProblem],
                       val_problems: List[AlgebraProblem],
                       test_problems: List[AlgebraProblem],
                       batch_size: int = 32,
                       num_workers: int = 0,
                       max_steps: Optional[int] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for train, validation, and test sets.
    
    Args:
        train_problems: Training problems
        val_problems: Validation problems
        test_problems: Test problems
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        max_steps: Maximum steps per problem
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = AlgebraStepDataset(train_problems, max_steps=max_steps)
    val_dataset = AlgebraStepDataset(val_problems, max_steps=max_steps)
    test_dataset = AlgebraStepDataset(test_problems, max_steps=max_steps)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_step_batch,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_step_batch,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_step_batch,
        num_workers=num_workers
    )
    
    print(f"Created data loaders:")
    print(f"  Train: {len(train_dataset)} steps, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset)} steps, {len(val_loader)} batches")
    print(f"  Test: {len(test_dataset)} steps, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the dataset
    from .dataset_generator import AlgebraDatasetGenerator, create_train_val_test_split
    
    # Generate small test dataset
    generator = AlgebraDatasetGenerator(seed=42)
    problems = generator.generate_mixed_dataset(n_linear=20, n_quadratic=10, n_simplify=10)
    
    # Split dataset
    train_probs, val_probs, test_probs = create_train_val_test_split(problems, seed=42)
    
    # Create datasets
    train_dataset = AlgebraStepDataset(train_probs)
    print(f"Train dataset size: {len(train_dataset)}")
    
    # Test single item
    item = train_dataset[0]
    print(f"\nExample item:")
    print(f"Current graph nodes: {item['current_graph'].num_nodes}")
    print(f"Next graph nodes: {item['next_graph'].num_nodes}")
    print(f"Operation: {item['operation']}")
    print(f"Description: {item['description']}")
    
    # Test data loader
    train_loader, val_loader, test_loader = create_data_loaders(
        train_probs, val_probs, test_probs, batch_size=4
    )
    
    # Get one batch
    batch = next(iter(train_loader))
    print(f"\nBatch info:")
    print(f"Current batch nodes: {batch['current_batch'].num_nodes}")
    print(f"Next batch nodes: {batch['next_batch'].num_nodes}")
    print(f"Batch size: {batch['current_batch'].num_graphs}") 