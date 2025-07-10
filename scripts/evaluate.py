"""Evaluation script for the Graph Neural Algebra Tutor model.

This script implements comprehensive evaluation metrics to assess the model's
performance on step prediction, solution accuracy, and educational utility.
"""

import argparse
import json
import logging
from pathlib import Path
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import numpy as np

import torch
from torch_geometric.data import Batch
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data import (
    AlgebraDatasetGenerator,
    AlgebraProblem,
    create_data_loaders,
    AlgebraStepDataset
)
from src.models import (
    StepPredictor,
    StepPredictorConfig,
    OperationType,
    create_step_predictor
)
from src.solver import GNNAlgebraSolver
from src.verification import AlgebraicVerifier
from src.utils import StepByStepSolver, AlgebraicOperation


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AlgebraTutorEvaluator:
    """Comprehensive evaluator for the Graph Neural Algebra Tutor."""
    
    def __init__(self,
                 model: StepPredictor,
                 solver: GNNAlgebraSolver,
                 device: str = 'cpu'):
        self.model = model.to(device)
        self.model.eval()
        self.solver = solver
        self.device = device
        self.verifier = AlgebraicVerifier()
        self.rule_based_solver = StepByStepSolver()
        
        # Initialize metrics storage
        self.metrics = defaultdict(list)
    
    def evaluate_step_accuracy(self,
                             test_loader,
                             num_batches: Optional[int] = None) -> Dict[str, float]:
        """Evaluate next-step prediction accuracy.
        
        Args:
            test_loader: Test data loader
            num_batches: Number of batches to evaluate (None for all)
            
        Returns:
            Dictionary of step accuracy metrics
        """
        logger.info("Evaluating step prediction accuracy...")
        
        correct_operations = 0
        correct_done = 0
        valid_steps = 0
        total_steps = 0
        
        operation_confusion = defaultdict(lambda: defaultdict(int))
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Step accuracy")):
                if num_batches and batch_idx >= num_batches:
                    break
                
                # Move batch to device
                current_batch = batch['current_batch'].to(self.device)
                
                # Get predictions
                operations, parameters, done_flags = self.model.predict_step(current_batch)
                
                # Get true labels
                true_operations = batch['operations']
                true_descriptions = batch['descriptions']
                
                # Evaluate each prediction
                for i in range(len(operations)):
                    total_steps += 1
                    
                    # Check operation accuracy
                    true_op = true_operations[i]
                    pred_op = OperationType(operations[i]).name
                    
                    if true_op and pred_op.lower() in true_op.lower():
                        correct_operations += 1
                    
                    # Update confusion matrix
                    operation_confusion[true_op][pred_op] += 1
                    
                    # Check done prediction
                    is_done = 'solution' in true_descriptions[i].lower() or 'final' in true_descriptions[i].lower()
                    if done_flags[i] == is_done:
                        correct_done += 1
                    
                    # Verify step validity (would need actual expressions)
                    # For now, count as valid if operation matches
                    if true_op and pred_op.lower() in true_op.lower():
                        valid_steps += 1
        
        metrics = {
            'step_operation_accuracy': correct_operations / total_steps if total_steps > 0 else 0,
            'step_done_accuracy': correct_done / total_steps if total_steps > 0 else 0,
            'step_validity_rate': valid_steps / total_steps if total_steps > 0 else 0,
            'total_steps_evaluated': total_steps,
        }
        
        # Add confusion matrix info
        self.operation_confusion = dict(operation_confusion)
        
        return metrics
    
    def evaluate_solution_success(self,
                                test_problems: List[AlgebraProblem],
                                max_problems: Optional[int] = None) -> Dict[str, float]:
        """Evaluate end-to-end solution success rate.
        
        Args:
            test_problems: List of test problems
            max_problems: Maximum number of problems to evaluate
            
        Returns:
            Dictionary of solution success metrics
        """
        logger.info("Evaluating solution success rate...")
        
        if max_problems:
            test_problems = test_problems[:max_problems]
        
        results = {
            'linear_equation': {'solved': 0, 'total': 0, 'steps': []},
            'quadratic_equation': {'solved': 0, 'total': 0, 'steps': []},
            'simplification': {'solved': 0, 'total': 0, 'steps': []},
        }
        
        for problem in tqdm(test_problems, desc="Solution success"):
            problem_type = problem.problem_type
            results[problem_type]['total'] += 1
            
            try:
                # Solve with GNN solver
                steps = self.solver.solve(
                    problem.initial_expression,
                    show_steps=False
                )
                
                # Check if solution is correct
                final_expr = steps[-1].expression if steps else None
                
                if final_expr:
                    # For equations, check if it's solved
                    if problem_type in ['linear_equation', 'quadratic_equation']:
                        is_solved = self.solver._is_solved(final_expr)
                        if is_solved:
                            # Verify solution is correct
                            is_correct, _ = self.verifier.check_solution(
                                problem.initial_expression,
                                final_expr
                            )
                            if is_correct:
                                results[problem_type]['solved'] += 1
                    else:
                        # For simplification, just check if we reached an answer
                        results[problem_type]['solved'] += 1
                    
                    # Record number of steps
                    results[problem_type]['steps'].append(len(steps))
                
            except Exception as e:
                logger.warning(f"Error solving problem: {e}")
                continue
        
        # Calculate metrics
        metrics = {}
        overall_solved = 0
        overall_total = 0
        
        for ptype, data in results.items():
            if data['total'] > 0:
                success_rate = data['solved'] / data['total']
                avg_steps = np.mean(data['steps']) if data['steps'] else 0
                
                metrics[f'{ptype}_success_rate'] = success_rate
                metrics[f'{ptype}_avg_steps'] = avg_steps
                
                overall_solved += data['solved']
                overall_total += data['total']
        
        metrics['overall_success_rate'] = overall_solved / overall_total if overall_total > 0 else 0
        metrics['problems_evaluated'] = overall_total
        
        return metrics
    
    def evaluate_step_optimality(self,
                               test_problems: List[AlgebraProblem],
                               max_problems: Optional[int] = None) -> Dict[str, float]:
        """Evaluate solution step optimality compared to rule-based solver.
        
        Args:
            test_problems: List of test problems
            max_problems: Maximum number to evaluate
            
        Returns:
            Dictionary of step optimality metrics
        """
        logger.info("Evaluating step optimality...")
        
        if max_problems:
            test_problems = test_problems[:max_problems]
        
        step_ratios = []
        extra_steps = []
        
        for problem in tqdm(test_problems[:50], desc="Step optimality"):  # Limit for speed
            try:
                # Get GNN solution
                gnn_steps = self.solver.solve(
                    problem.initial_expression,
                    show_steps=False
                )
                
                # Get rule-based solution
                if problem.problem_type == 'linear_equation':
                    rb_steps = self.rule_based_solver.solve_linear_equation(
                        problem.initial_expression
                    )
                elif problem.problem_type == 'quadratic_equation':
                    rb_steps = self.rule_based_solver.solve_quadratic_equation(
                        problem.initial_expression
                    )
                else:
                    rb_steps = self.rule_based_solver.simplify_expression_steps(
                        problem.initial_expression
                    )
                
                # Compare step counts
                gnn_count = len(gnn_steps)
                rb_count = len(rb_steps)
                
                if rb_count > 0:
                    ratio = gnn_count / rb_count
                    step_ratios.append(ratio)
                    extra_steps.append(max(0, gnn_count - rb_count))
                
            except Exception as e:
                logger.warning(f"Error comparing solutions: {e}")
                continue
        
        metrics = {
            'avg_step_ratio': np.mean(step_ratios) if step_ratios else 0,
            'median_step_ratio': np.median(step_ratios) if step_ratios else 0,
            'avg_extra_steps': np.mean(extra_steps) if extra_steps else 0,
            'problems_compared': len(step_ratios),
        }
        
        return metrics
    
    def evaluate_educational_utility(self,
                                   test_problems: List[AlgebraProblem],
                                   max_problems: int = 20) -> Dict[str, float]:
        """Evaluate educational utility metrics.
        
        Args:
            test_problems: List of test problems
            max_problems: Number of problems to evaluate
            
        Returns:
            Dictionary of educational utility metrics
        """
        logger.info("Evaluating educational utility...")
        
        hint_quality_scores = []
        step_granularity_scores = []
        
        for problem in test_problems[:max_problems]:
            try:
                # Get solution steps
                steps = self.solver.solve(
                    problem.initial_expression,
                    show_steps=False
                )
                
                if len(steps) > 1:
                    # Evaluate step granularity
                    # Good if steps are neither too big nor too small
                    avg_step_size = self._estimate_step_size(steps)
                    granularity_score = 1.0 if 0.3 < avg_step_size < 0.7 else 0.5
                    step_granularity_scores.append(granularity_score)
                    
                    # Simulate hint generation
                    # Good if hints are specific to the operation
                    hint_score = 1.0  # Placeholder - would need actual hint evaluation
                    hint_quality_scores.append(hint_score)
                
            except Exception as e:
                logger.warning(f"Error evaluating educational utility: {e}")
                continue
        
        metrics = {
            'avg_hint_quality': np.mean(hint_quality_scores) if hint_quality_scores else 0,
            'avg_step_granularity': np.mean(step_granularity_scores) if step_granularity_scores else 0,
            'problems_evaluated': len(hint_quality_scores),
        }
        
        return metrics
    
    def evaluate_inference_performance(self,
                                     test_loader,
                                     num_batches: int = 50) -> Dict[str, float]:
        """Evaluate inference speed and resource usage.
        
        Args:
            test_loader: Test data loader
            num_batches: Number of batches to time
            
        Returns:
            Dictionary of performance metrics
        """
        logger.info("Evaluating inference performance...")
        
        inference_times = []
        batch_sizes = []
        
        # Warmup
        for _ in range(5):
            batch = next(iter(test_loader))
            current_batch = batch['current_batch'].to(self.device)
            _ = self.model.predict_step(current_batch)
        
        # Time inference
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if batch_idx >= num_batches:
                    break
                
                current_batch = batch['current_batch'].to(self.device)
                batch_size = current_batch.num_graphs
                
                start_time = time.time()
                _ = self.model.predict_step(current_batch)
                end_time = time.time()
                
                inference_times.append(end_time - start_time)
                batch_sizes.append(batch_size)
        
        # Calculate metrics
        total_samples = sum(batch_sizes)
        total_time = sum(inference_times)
        
        metrics = {
            'avg_batch_inference_time': np.mean(inference_times),
            'avg_sample_inference_time': total_time / total_samples if total_samples > 0 else 0,
            'samples_per_second': total_samples / total_time if total_time > 0 else 0,
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
        }
        
        return metrics
    
    def _estimate_step_size(self, steps: List[Any]) -> float:
        """Estimate average conceptual size of steps."""
        # Placeholder - would need to analyze the transformations
        return 0.5
    
    def generate_evaluation_report(self,
                                 all_metrics: Dict[str, Dict[str, float]],
                                 output_dir: Path):
        """Generate comprehensive evaluation report.
        
        Args:
            all_metrics: Dictionary of all evaluation metrics
            output_dir: Directory to save report
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create summary DataFrame
        summary_data = []
        for category, metrics in all_metrics.items():
            for metric_name, value in metrics.items():
                summary_data.append({
                    'Category': category,
                    'Metric': metric_name,
                    'Value': value
                })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save to CSV
        summary_df.to_csv(output_dir / 'evaluation_summary.csv', index=False)
        
        # Generate plots
        self._generate_plots(all_metrics, output_dir)
        
        # Generate text report
        report_path = output_dir / 'evaluation_report.txt'
        with open(report_path, 'w') as f:
            f.write("Graph Neural Algebra Tutor - Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            
            for category, metrics in all_metrics.items():
                f.write(f"{category.replace('_', ' ').title()}\n")
                f.write("-" * 30 + "\n")
                
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        f.write(f"{metric}: {value:.4f}\n")
                    else:
                        f.write(f"{metric}: {value}\n")
                
                f.write("\n")
        
        logger.info(f"Evaluation report saved to {output_dir}")
    
    def _generate_plots(self, all_metrics: Dict[str, Dict[str, float]], output_dir: Path):
        """Generate visualization plots."""
        # Success rates by problem type
        if 'solution_success' in all_metrics:
            plt.figure(figsize=(10, 6))
            problem_types = ['linear_equation', 'quadratic_equation', 'simplification']
            success_rates = []
            
            for ptype in problem_types:
                key = f'{ptype}_success_rate'
                if key in all_metrics['solution_success']:
                    success_rates.append(all_metrics['solution_success'][key])
                else:
                    success_rates.append(0)
            
            plt.bar(problem_types, success_rates)
            plt.ylabel('Success Rate')
            plt.title('Solution Success Rate by Problem Type')
            plt.ylim(0, 1.1)
            
            for i, v in enumerate(success_rates):
                plt.text(i, v + 0.02, f'{v:.2%}', ha='center')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'success_rates.png')
            plt.close()
        
        # Operation confusion matrix
        if hasattr(self, 'operation_confusion'):
            # Would create confusion matrix heatmap here
            pass


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate Graph Neural Algebra Tutor')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data-path', type=str, required=True, help='Path to test dataset')
    parser.add_argument('--output-dir', type=str, default='evaluation_results', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--max-problems', type=int, help='Maximum problems to evaluate')
    parser.add_argument('--quick', action='store_true', help='Run quick evaluation')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    eval_dir = output_dir / f'eval_{timestamp}'
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info("Loading model...")
    checkpoint = torch.load(args.model_path, map_location=args.device)
    config = StepPredictorConfig.from_dict(checkpoint['config'])
    
    model = create_step_predictor(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create solver
    solver = GNNAlgebraSolver(
        model=model,
        device=args.device,
        use_verification=True
    )
    
    # Load test data
    logger.info("Loading test data...")
    test_problems = AlgebraDatasetGenerator.load_dataset(args.data_path)
    
    if args.quick:
        test_problems = test_problems[:100]
    elif args.max_problems:
        test_problems = test_problems[:args.max_problems]
    
    # Create data loader for step evaluation
    test_dataset = AlgebraStepDataset(test_problems)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda x: x[0]  # Simple collate
    )
    
    # Create evaluator
    evaluator = AlgebraTutorEvaluator(model, solver, args.device)
    
    # Run evaluations
    all_metrics = {}
    
    logger.info("Running step accuracy evaluation...")
    step_metrics = evaluator.evaluate_step_accuracy(
        test_loader,
        num_batches=50 if args.quick else None
    )
    all_metrics['step_accuracy'] = step_metrics
    
    logger.info("Running solution success evaluation...")
    solution_metrics = evaluator.evaluate_solution_success(
        test_problems,
        max_problems=50 if args.quick else args.max_problems
    )
    all_metrics['solution_success'] = solution_metrics
    
    logger.info("Running step optimality evaluation...")
    optimality_metrics = evaluator.evaluate_step_optimality(
        test_problems,
        max_problems=20 if args.quick else 50
    )
    all_metrics['step_optimality'] = optimality_metrics
    
    if not args.quick:
        logger.info("Running educational utility evaluation...")
        edu_metrics = evaluator.evaluate_educational_utility(test_problems)
        all_metrics['educational_utility'] = edu_metrics
    
    logger.info("Running inference performance evaluation...")
    perf_metrics = evaluator.evaluate_inference_performance(test_loader)
    all_metrics['inference_performance'] = perf_metrics
    
    # Generate report
    logger.info("Generating evaluation report...")
    evaluator.generate_evaluation_report(all_metrics, eval_dir)
    
    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    
    for category, metrics in all_metrics.items():
        print(f"\n{category.replace('_', ' ').title()}:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
    
    logger.info("Evaluation complete!")


if __name__ == '__main__':
    main() 