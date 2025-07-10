"""Training script for the Graph Neural Algebra Tutor model."""

import argparse
import json
import logging
from pathlib import Path
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml
import wandb

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data import (
    AlgebraDatasetGenerator,
    create_train_val_test_split,
    create_data_loaders
)
from src.models import (
    StepPredictor,
    StepPredictorConfig,
    OperationType,
    create_step_predictor
)
from src.utils import AlgebraicOperation


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for the Graph Neural Algebra Tutor model."""
    
    def __init__(self,
                 model: StepPredictor,
                 config: StepPredictorConfig,
                 train_loader,
                 val_loader,
                 device: str = 'cpu',
                 checkpoint_dir: Optional[Path] = None,
                 use_wandb: bool = False):
        """Initialize trainer.
        
        Args:
            model: Model to train
            config: Training configuration
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            use_wandb: Whether to use Weights & Biases logging
        """
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.use_wandb = use_wandb
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Loss functions
        self.operation_criterion = nn.CrossEntropyLoss()
        self.param_criterion = nn.MSELoss()
        self.done_criterion = nn.CrossEntropyLoss()
        
        # Initialize tensorboard
        if checkpoint_dir:
            self.writer = SummaryWriter(checkpoint_dir / 'runs')
        else:
            self.writer = None
        
        # Best validation loss
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.early_stopping_patience = 10
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        total_loss = 0
        operation_loss_sum = 0
        param_loss_sum = 0
        done_loss_sum = 0
        correct_operations = 0
        correct_done = 0
        total_samples = 0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch} - Training')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            current_batch = batch['current_batch'].to(self.device)
            
            # Get labels
            operation_labels = self._get_operation_labels(batch['operations'])
            param_labels = self._get_param_labels(batch)
            done_labels = self._get_done_labels(batch)
            
            # Forward pass
            output = self.model(current_batch)
            
            # Calculate losses
            operation_loss = self.operation_criterion(
                output['operation_logits'],
                operation_labels
            )
            
            param_loss = self.param_criterion(
                output['param_value'].squeeze(),
                param_labels
            )
            
            done_loss = self.done_criterion(
                output['done_logits'],
                done_labels
            )
            
            # Combined loss
            loss = operation_loss + 0.1 * param_loss + 0.5 * done_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            
            # Update weights
            self.optimizer.step()
            
            # Update metrics
            batch_size = current_batch.num_graphs
            total_loss += loss.item() * batch_size
            operation_loss_sum += operation_loss.item() * batch_size
            param_loss_sum += param_loss.item() * batch_size
            done_loss_sum += done_loss.item() * batch_size
            
            # Calculate accuracy
            pred_operations = output['operation_logits'].argmax(dim=1)
            correct_operations += (pred_operations == operation_labels).sum().item()
            
            pred_done = output['done_logits'].argmax(dim=1)
            correct_done += (pred_done == done_labels).sum().item()
            
            total_samples += batch_size
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'op_acc': f"{correct_operations / total_samples:.3f}"
            })
        
        # Calculate average metrics
        metrics = {
            'train_loss': total_loss / total_samples,
            'train_operation_loss': operation_loss_sum / total_samples,
            'train_param_loss': param_loss_sum / total_samples,
            'train_done_loss': done_loss_sum / total_samples,
            'train_operation_accuracy': correct_operations / total_samples,
            'train_done_accuracy': correct_done / total_samples,
        }
        
        return metrics
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        total_loss = 0
        operation_loss_sum = 0
        param_loss_sum = 0
        done_loss_sum = 0
        correct_operations = 0
        correct_done = 0
        total_samples = 0
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc=f'Epoch {epoch} - Validation')
            
            for batch in progress_bar:
                # Move batch to device
                current_batch = batch['current_batch'].to(self.device)
                
                # Get labels
                operation_labels = self._get_operation_labels(batch['operations'])
                param_labels = self._get_param_labels(batch)
                done_labels = self._get_done_labels(batch)
                
                # Forward pass
                output = self.model(current_batch)
                
                # Calculate losses
                operation_loss = self.operation_criterion(
                    output['operation_logits'],
                    operation_labels
                )
                
                param_loss = self.param_criterion(
                    output['param_value'].squeeze(),
                    param_labels
                )
                
                done_loss = self.done_criterion(
                    output['done_logits'],
                    done_labels
                )
                
                # Combined loss
                loss = operation_loss + 0.1 * param_loss + 0.5 * done_loss
                
                # Update metrics
                batch_size = current_batch.num_graphs
                total_loss += loss.item() * batch_size
                operation_loss_sum += operation_loss.item() * batch_size
                param_loss_sum += param_loss.item() * batch_size
                done_loss_sum += done_loss.item() * batch_size
                
                # Calculate accuracy
                pred_operations = output['operation_logits'].argmax(dim=1)
                correct_operations += (pred_operations == operation_labels).sum().item()
                
                pred_done = output['done_logits'].argmax(dim=1)
                correct_done += (pred_done == done_labels).sum().item()
                
                total_samples += batch_size
        
        # Calculate average metrics
        metrics = {
            'val_loss': total_loss / total_samples,
            'val_operation_loss': operation_loss_sum / total_samples,
            'val_param_loss': param_loss_sum / total_samples,
            'val_done_loss': done_loss_sum / total_samples,
            'val_operation_accuracy': correct_operations / total_samples,
            'val_done_accuracy': correct_done / total_samples,
        }
        
        return metrics
    
    def train(self, num_epochs: int):
        """Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(1, num_epochs + 1):
            # Train for one epoch
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Update learning rate
            self.scheduler.step(val_metrics['val_loss'])
            
            # Log metrics
            self._log_metrics(epoch, {**train_metrics, **val_metrics})
            
            # Save checkpoint if improved
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.epochs_without_improvement = 0
                
                if self.checkpoint_dir:
                    self._save_checkpoint(epoch, val_metrics)
                    logger.info(f"New best model saved with val_loss: {self.best_val_loss:.4f}")
            else:
                self.epochs_without_improvement += 1
                
                if self.epochs_without_improvement >= self.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch} epochs")
                    break
            
            # Print epoch summary
            logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"Train Op Acc: {train_metrics['train_operation_accuracy']:.3f}, "
                f"Val Op Acc: {val_metrics['val_operation_accuracy']:.3f}"
            )
    
    def _get_operation_labels(self, operations: List[str]) -> torch.Tensor:
        """Convert operation strings to label tensor."""
        labels = []
        for op in operations:
            if op:
                try:
                    op_type = OperationType.from_string(op)
                    labels.append(op_type.value)
                except:
                    labels.append(OperationType.NO_OP.value)
            else:
                labels.append(OperationType.NO_OP.value)
        
        return torch.tensor(labels, dtype=torch.long).to(self.device)
    
    def _get_param_labels(self, batch: Dict) -> torch.Tensor:
        """Extract parameter labels from batch."""
        # For now, return zeros as placeholder
        batch_size = batch['current_batch'].num_graphs
        return torch.zeros(batch_size).to(self.device)
    
    def _get_done_labels(self, batch: Dict) -> torch.Tensor:
        """Extract done labels from batch."""
        # Check if this is the last step
        done_labels = []
        for desc in batch['descriptions']:
            is_done = 'solution' in desc.lower() or 'final' in desc.lower()
            done_labels.append(1 if is_done else 0)
        
        return torch.tensor(done_labels, dtype=torch.long).to(self.device)
    
    def _log_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Log metrics to tensorboard and wandb."""
        # Log to tensorboard
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, epoch)
        
        # Log to wandb
        if self.use_wandb:
            wandb.log(metrics, step=epoch)
    
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config.to_dict(),
        }
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Also save as best model
        best_path = self.checkpoint_dir / 'best_model.pt'
        torch.save(checkpoint, best_path)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Graph Neural Algebra Tutor')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--output-dir', type=str, default='checkpoints', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num-epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--use-wandb', action='store_true', help='Use Weights & Biases')
    parser.add_argument('--wandb-project', type=str, default='graph-algebra-tutor', help='W&B project name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--generate-data', action='store_true', help='Generate new training data')
    parser.add_argument('--n-train-problems', type=int, default=10000, help='Number of training problems')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_dir = output_dir / f'run_{timestamp}'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Load or create config
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = StepPredictorConfig.from_dict(config_dict)
    else:
        config = StepPredictorConfig()
        config.batch_size = args.batch_size
        config.learning_rate = args.learning_rate
        config.num_epochs = args.num_epochs
    
    # Save config
    with open(checkpoint_dir / 'config.yaml', 'w') as f:
        yaml.dump(config.to_dict(), f)
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            config=config.to_dict(),
            name=f'run_{timestamp}'
        )
    
    # Generate or load data
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    if args.generate_data:
        logger.info("Generating training data...")
        generator = AlgebraDatasetGenerator(seed=args.seed)
        
        # Generate problems
        problems = generator.generate_mixed_dataset(
            n_linear=int(args.n_train_problems * 0.5),
            n_quadratic=int(args.n_train_problems * 0.3),
            n_simplify=int(args.n_train_problems * 0.2)
        )
        
        # Save dataset
        generator.save_dataset(problems, data_dir / 'train_dataset.json')
        
        # Split dataset
        train_problems, val_problems, test_problems = create_train_val_test_split(
            problems,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            seed=args.seed
        )
    else:
        # Load existing dataset
        logger.info("Loading existing dataset...")
        from src.data import AlgebraDatasetGenerator
        
        dataset_path = data_dir / 'train_dataset.json'
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {dataset_path}. Use --generate-data to create it.")
        
        problems = AlgebraDatasetGenerator.load_dataset(dataset_path)
        train_problems, val_problems, test_problems = create_train_val_test_split(
            problems,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            seed=args.seed
        )
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        train_problems,
        val_problems,
        test_problems,
        batch_size=config.batch_size,
        num_workers=0
    )
    
    # Create model
    logger.info("Creating model...")
    model = create_step_predictor(config)
    logger.info(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        checkpoint_dir=checkpoint_dir,
        use_wandb=args.use_wandb
    )
    
    # Train model
    logger.info("Starting training...")
    trainer.train(args.num_epochs)
    
    # Final evaluation on test set
    logger.info("Evaluating on test set...")
    # Would implement test evaluation here
    
    logger.info("Training complete!")
    
    if args.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main() 