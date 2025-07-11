#!/usr/bin/env python3
"""
Training script for the Graph Neural Algebra Tutor.

This script:
1. Generates synthetic algebra problems
2. Trains the GNN model
3. Evaluates performance
4. Saves the trained model
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
import numpy as np
from tqdm import tqdm

from src.data.dataset_generator import AlgebraDatasetGenerator, create_train_val_test_split
from src.models.gnn_model import GraphNeuralAlgebraTutor
from src.graph import expression_to_graph
from src.verification import AlgebraicVerifier

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlgebraDataset(torch.utils.data.Dataset):
    """PyTorch dataset for algebra problems."""
    
    def __init__(self, problems: List, max_length: int = 50):
        self.problems = problems
        self.max_length = max_length
        self.verifier = AlgebraicVerifier()
    
    def __len__(self):
        return len(self.problems)
    
    def __getitem__(self, idx):
        problem = self.problems[idx]
        
        # Convert initial expression to graph
        current_graph = expression_to_graph(problem.initial_expression)
        
        # For now, we'll use a simple approach: predict the next step
        # In a full implementation, we'd need tokenization
        if len(problem.steps) > 1:
            next_step = problem.steps[1]['expression']
        else:
            next_step = problem.final_answer
        
        # Create dummy target tokens (in real implementation, use proper tokenizer)
        target_tokens = torch.randint(0, 1000, (min(len(next_step), self.max_length),))
        
        return {
            'graph': current_graph,
            'target': target_tokens,
            'problem_id': problem.problem_id
        }


def collate_fn(batch):
    """Collate function for DataLoader."""
    graphs = [item['graph'] for item in batch]
    targets = [item['target'] for item in batch]
    problem_ids = [item['problem_id'] for item in batch]
    
    # Batch graphs
    batched_graphs = Batch.from_data_list(graphs)
    
    # Pad targets
    max_len = max(len(t) for t in targets)
    padded_targets = torch.zeros(len(targets), max_len, dtype=torch.long)
    for i, target in enumerate(targets):
        padded_targets[i, :len(target)] = target
    
    return {
        'graphs': batched_graphs,
        'targets': padded_targets,
        'problem_ids': problem_ids
    }


def train_epoch(model: nn.Module, 
                dataloader: DataLoader, 
                optimizer: optim.Optimizer,
                criterion: nn.Module,
                device: torch.device) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        graphs = batch['graphs'].to(device)
        targets = batch['targets'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(graphs, targets)
        loss = criterion(outputs['expression_logits'].view(-1, outputs['expression_logits'].size(-1)), 
                        targets.view(-1))
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return {'train_loss': total_loss / num_batches}


def evaluate(model: nn.Module, 
             dataloader: DataLoader, 
             criterion: nn.Module,
             device: torch.device) -> Dict[str, float]:
    """Evaluate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            graphs = batch['graphs'].to(device)
            targets = batch['targets'].to(device)
            
            outputs = model(graphs, targets)
            loss = criterion(outputs['expression_logits'].view(-1, outputs['expression_logits'].size(-1)), 
                            targets.view(-1))
            
            total_loss += loss.item()
            num_batches += 1
    
    return {'val_loss': total_loss / num_batches}


def main():
    parser = argparse.ArgumentParser(description="Train Graph Neural Algebra Tutor")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    parser.add_argument("--generate-data", action="store_true", help="Generate new data")
    parser.add_argument("--n-train", type=int, default=1000, help="Number of training problems")
    parser.add_argument("--n-val", type=int, default=200, help="Number of validation problems")
    parser.add_argument("--n-test", type=int, default=200, help="Number of test problems")
    
    args = parser.parse_args()
    
    # Create directories
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    data_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    
    # Set device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Generate or load data
    if args.generate_data or not (data_dir / "train.json").exists():
        logger.info("Generating training data...")
        generator = AlgebraDatasetGenerator(seed=42)
        
        # Generate mixed dataset
        all_problems = generator.generate_mixed_dataset(
            n_linear=args.n_train // 2,
            n_quadratic=args.n_train // 4,
            n_simplify=args.n_train // 4
        )
        
        # Split into train/val/test
        train_problems, val_problems, test_problems = create_train_val_test_split(
            all_problems, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42
        )
        
        # Save datasets
        generator.save_dataset(train_problems, data_dir / "train.json")
        generator.save_dataset(val_problems, data_dir / "val.json")
        generator.save_dataset(test_problems, data_dir / "test.json")
        
        logger.info(f"Generated {len(train_problems)} train, {len(val_problems)} val, {len(test_problems)} test problems")
    else:
        logger.info("Loading existing data...")
        train_problems = AlgebraDatasetGenerator.load_dataset(data_dir / "train.json")
        val_problems = AlgebraDatasetGenerator.load_dataset(data_dir / "val.json")
        test_problems = AlgebraDatasetGenerator.load_dataset(data_dir / "test.json")
    
    # Create datasets
    train_dataset = AlgebraDataset(train_problems)
    val_dataset = AlgebraDataset(val_problems)
    test_dataset = AlgebraDataset(test_problems)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Create model
    model = GraphNeuralAlgebraTutor().to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(args.num_epochs):
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_metrics['train_loss'])
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_metrics['val_loss'])
        
        logger.info(f"Train loss: {train_metrics['train_loss']:.4f}, Val loss: {val_metrics['val_loss']:.4f}")
        
        # Save best model
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, output_dir / "best_model.pth")
            logger.info("Saved best model")
    
    # Test on test set
    logger.info("Evaluating on test set...")
    test_metrics = evaluate(model, test_loader, criterion, device)
    logger.info(f"Test loss: {test_metrics['val_loss']:.4f}")
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_loss': test_metrics['val_loss'],
        'best_val_loss': best_val_loss,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
    }
    
    with open(output_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main() 