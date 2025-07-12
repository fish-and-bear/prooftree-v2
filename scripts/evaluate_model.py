#!/usr/bin/env python3
"""
Comprehensive evaluation script for the Graph Neural Algebra Tutor.

This script evaluates the model on various metrics:
1. Step Accuracy - How often the model predicts correct next steps
2. Solution Success Rate - Percentage of problems solved correctly
3. Step Validity - Mathematical correctness of all steps
4. Solution Optimality - Efficiency of solution paths
5. Generalization - Performance on unseen problem types
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import torch
import sympy as sp

from src.data.dataset_generator import AlgebraDatasetGenerator, create_train_val_test_split
from src.solver import GNNAlgebraSolver
from src.verification import AlgebraicVerifier
from src.utils import StepByStepSolver


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    step_accuracy: float
    solution_success_rate: float
    step_validity: float
    avg_solution_length: float
    inference_time_ms: float
    problem_type_breakdown: Dict[str, Dict[str, float]]
    difficulty_breakdown: Dict[int, Dict[str, float]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'step_accuracy': self.step_accuracy,
            'solution_success_rate': self.solution_success_rate,
            'step_validity': self.step_validity,
            'avg_solution_length': self.avg_solution_length,
            'inference_time_ms': self.inference_time_ms,
            'problem_type_breakdown': self.problem_type_breakdown,
            'difficulty_breakdown': self.difficulty_breakdown
        }


class ComprehensiveEvaluator:
    """Comprehensive evaluator for the Graph Neural Algebra Tutor."""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 device: str = 'cpu',
                 use_verification: bool = True):
        """Initialize evaluator.
        
        Args:
            model_path: Path to trained model (None for default)
            device: Device to run evaluation on
            use_verification: Whether to use symbolic verification
        """
        self.device = device
        self.use_verification = use_verification
        
        # Initialize solver
        self.solver = GNNAlgebraSolver(
            model_path=model_path,
            device=device,
            use_verification=use_verification
        )
        
        # Initialize verifier for independent verification
        self.verifier = AlgebraicVerifier()
        
        # Initialize baseline solver for comparison
        self.baseline_solver = StepByStepSolver()
        
        logger.info(f"Initialized evaluator with device: {device}")
    
    def evaluate_step_accuracy(self, problems: List) -> Tuple[float, Dict[str, Any]]:
        """Evaluate step prediction accuracy.
        
        Args:
            problems: List of problems to evaluate
            
        Returns:
            Tuple of (accuracy, detailed_metrics)
        """
        correct_steps = 0
        total_steps = 0
        step_details = []
        
        for problem in tqdm(problems, desc="Evaluating step accuracy"):
            try:
                # Get ground truth steps
                if hasattr(problem, 'steps') and len(problem.steps) > 1:
                    for i in range(len(problem.steps) - 1):
                        current_expr = problem.steps[i]['expression']
                        expected_next = problem.steps[i + 1]['expression']
                        
                        # Get model prediction
                        predicted_steps = self.solver.solve(
                            current_expr, 
                            show_steps=False,
                            return_all_steps=True
                        )
                        
                        if len(predicted_steps) > 1:
                            predicted_next = predicted_steps[1].expression
                            
                            # Check if prediction matches expected
                            is_correct = self._expressions_equivalent(
                                predicted_next, expected_next
                            )
                            
                            if is_correct:
                                correct_steps += 1
                            
                            step_details.append({
                                'problem_id': problem.problem_id,
                                'current': str(current_expr),
                                'expected': str(expected_next),
                                'predicted': str(predicted_next),
                                'correct': is_correct
                            })
                        
                        total_steps += 1
                        
            except Exception as e:
                logger.warning(f"Error evaluating step accuracy for problem {problem.problem_id}: {e}")
                continue
        
        accuracy = correct_steps / total_steps if total_steps > 0 else 0.0
        
        return accuracy, {
            'correct_steps': correct_steps,
            'total_steps': total_steps,
            'step_details': step_details
        }
    
    def evaluate_solution_success_rate(self, problems: List) -> Tuple[float, Dict[str, Any]]:
        """Evaluate solution success rate by problem type.
        
        Args:
            problems: List of problems to evaluate
            
        Returns:
            Tuple of (overall_success_rate, breakdown_by_type)
        """
        results = {}
        overall_solved = 0
        overall_total = 0
        
        # Group problems by type
        problems_by_type = {}
        for problem in problems:
            ptype = problem.problem_type
            if ptype not in problems_by_type:
                problems_by_type[ptype] = []
            problems_by_type[ptype].append(problem)
        
        for problem_type, type_problems in problems_by_type.items():
            solved = 0
            total = len(type_problems)
            solution_lengths = []
            
            for problem in tqdm(type_problems, desc=f"Evaluating {problem_type}"):
                try:
                    # Solve with model
                    steps = self.solver.solve(
                        problem.initial_expression,
                        show_steps=False,
                        return_all_steps=True
                    )
                    
                    if steps:
                        final_expr = steps[-1].expression
                        solution_lengths.append(len(steps))
                        
                        # Check if solution is correct
                        is_correct = self._verify_solution(
                            problem.initial_expression,
                            final_expr,
                            problem.final_answer
                        )
                        
                        if is_correct:
                            solved += 1
                            overall_solved += 1
                    
                except Exception as e:
                    logger.warning(f"Error solving problem {problem.problem_id}: {e}")
                    continue
            
            overall_total += total
            success_rate = solved / total if total > 0 else 0.0
            
            results[problem_type] = {
                'solved': solved,
                'total': total,
                'success_rate': success_rate,
                'avg_solution_length': np.mean(solution_lengths) if solution_lengths else 0,
                'solution_lengths': solution_lengths
            }
        
        overall_success_rate = overall_solved / overall_total if overall_total > 0 else 0.0
        
        return overall_success_rate, results
    
    def evaluate_step_validity(self, problems: List) -> Tuple[float, Dict[str, Any]]:
        """Evaluate mathematical validity of all generated steps.
        
        Args:
            problems: List of problems to evaluate
            
        Returns:
            Tuple of (validity_rate, detailed_results)
        """
        valid_steps = 0
        total_steps = 0
        invalid_examples = []
        
        for problem in tqdm(problems, desc="Evaluating step validity"):
            try:
                steps = self.solver.solve(
                    problem.initial_expression,
                    show_steps=False,
                    return_all_steps=True
                )
                
                # Check each step transition
                for i in range(len(steps) - 1):
                    current = steps[i].expression
                    next_step = steps[i + 1].expression
                    
                    is_valid, explanation = self.verifier.verify_step_validity(
                        current, next_step
                    )
                    
                    if is_valid:
                        valid_steps += 1
                    else:
                        invalid_examples.append({
                            'problem_id': problem.problem_id,
                            'current': str(current),
                            'next': str(next_step),
                            'explanation': explanation
                        })
                    
                    total_steps += 1
                    
            except Exception as e:
                logger.warning(f"Error evaluating step validity for problem {problem.problem_id}: {e}")
                continue
        
        validity_rate = valid_steps / total_steps if total_steps > 0 else 0.0
        
        return validity_rate, {
            'valid_steps': valid_steps,
            'total_steps': total_steps,
            'invalid_examples': invalid_examples
        }
    
    def evaluate_solution_optimality(self, problems: List) -> Tuple[float, Dict[str, Any]]:
        """Evaluate solution optimality (step count efficiency).
        
        Args:
            problems: List of problems to evaluate
            
        Returns:
            Tuple of (optimality_score, detailed_results)
        """
        optimality_scores = []
        comparison_data = []
        
        for problem in tqdm(problems, desc="Evaluating solution optimality"):
            try:
                # Get model solution
                model_steps = self.solver.solve(
                    problem.initial_expression,
                    show_steps=False,
                    return_all_steps=True
                )
                
                # Get baseline solution for comparison
                baseline_steps = self._get_baseline_solution(problem)
                
                if model_steps and baseline_steps:
                    model_length = len(model_steps)
                    baseline_length = len(baseline_steps)
                    
                    # Optimality score (lower is better, 1.0 is optimal)
                    optimality = baseline_length / model_length if model_length > 0 else 0.0
                    optimality_scores.append(optimality)
                    
                    comparison_data.append({
                        'problem_id': problem.problem_id,
                        'model_steps': model_length,
                        'baseline_steps': baseline_length,
                        'optimality': optimality
                    })
                    
            except Exception as e:
                logger.warning(f"Error evaluating optimality for problem {problem.problem_id}: {e}")
                continue
        
        avg_optimality = np.mean(optimality_scores) if optimality_scores else 0.0
        
        return avg_optimality, {
            'optimality_scores': optimality_scores,
            'comparison_data': comparison_data,
            'avg_optimality': avg_optimality
        }
    
    def evaluate_generalization(self, problems: List) -> Dict[str, Any]:
        """Evaluate generalization across different problem characteristics.
        
        Args:
            problems: List of problems to evaluate
            
        Returns:
            Dictionary of generalization metrics
        """
        results = {}
        
        # Group by difficulty
        by_difficulty = {}
        for problem in problems:
            diff = problem.difficulty
            if diff not in by_difficulty:
                by_difficulty[diff] = []
            by_difficulty[diff].append(problem)
        
        # Evaluate each difficulty level
        for difficulty, diff_problems in by_difficulty.items():
            success_rate, _ = self.evaluate_solution_success_rate(diff_problems)
            results[f'difficulty_{difficulty}'] = {
                'success_rate': success_rate,
                'num_problems': len(diff_problems)
            }
        
        # Evaluate by problem complexity (number of operations)
        by_complexity = {'simple': [], 'medium': [], 'complex': []}
        for problem in problems:
            expr_str = str(problem.initial_expression)
            op_count = sum(1 for c in expr_str if c in '+-*/^=')
            
            if op_count <= 2:
                by_complexity['simple'].append(problem)
            elif op_count <= 4:
                by_complexity['medium'].append(problem)
            else:
                by_complexity['complex'].append(problem)
        
        for complexity, comp_problems in by_complexity.items():
            if comp_problems:
                success_rate, _ = self.evaluate_solution_success_rate(comp_problems)
                results[f'complexity_{complexity}'] = {
                    'success_rate': success_rate,
                    'num_problems': len(comp_problems)
                }
        
        return results
    
    def benchmark_performance(self, problems: List) -> Dict[str, float]:
        """Benchmark inference performance.
        
        Args:
            problems: List of problems to benchmark
            
        Returns:
            Dictionary of performance metrics
        """
        inference_times = []
        
        for problem in tqdm(problems[:50], desc="Benchmarking performance"):  # Sample for speed
            try:
                start_time = time.time()
                
                steps = self.solver.solve(
                    problem.initial_expression,
                    show_steps=False,
                    return_all_steps=True
                )
                
                end_time = time.time()
                inference_times.append((end_time - start_time) * 1000)  # Convert to ms
                
            except Exception as e:
                logger.warning(f"Error benchmarking problem {problem.problem_id}: {e}")
                continue
        
        return {
            'avg_inference_time_ms': np.mean(inference_times) if inference_times else 0.0,
            'median_inference_time_ms': np.median(inference_times) if inference_times else 0.0,
            'std_inference_time_ms': np.std(inference_times) if inference_times else 0.0,
            'total_problems_benchmarked': len(inference_times)
        }
    
    def run_comprehensive_evaluation(self, problems: List) -> EvaluationMetrics:
        """Run comprehensive evaluation on all metrics.
        
        Args:
            problems: List of problems to evaluate
            
        Returns:
            EvaluationMetrics object with all results
        """
        logger.info("Starting comprehensive evaluation...")
        
        # Step accuracy
        logger.info("Evaluating step accuracy...")
        step_accuracy, step_details = self.evaluate_step_accuracy(problems)
        
        # Solution success rate
        logger.info("Evaluating solution success rate...")
        success_rate, type_breakdown = self.evaluate_solution_success_rate(problems)
        
        # Step validity
        logger.info("Evaluating step validity...")
        validity_rate, validity_details = self.evaluate_step_validity(problems)
        
        # Solution optimality
        logger.info("Evaluating solution optimality...")
        optimality, optimality_details = self.evaluate_solution_optimality(problems)
        
        # Generalization
        logger.info("Evaluating generalization...")
        generalization = self.evaluate_generalization(problems)
        
        # Performance benchmarking
        logger.info("Benchmarking performance...")
        performance = self.benchmark_performance(problems)
        
        # Calculate average solution length
        all_lengths = []
        for ptype_data in type_breakdown.values():
            all_lengths.extend(ptype_data['solution_lengths'])
        avg_solution_length = np.mean(all_lengths) if all_lengths else 0.0
        
        # Create metrics object
        metrics = EvaluationMetrics(
            step_accuracy=step_accuracy,
            solution_success_rate=success_rate,
            step_validity=validity_rate,
            avg_solution_length=avg_solution_length,
            inference_time_ms=performance['avg_inference_time_ms'],
            problem_type_breakdown=type_breakdown,
            difficulty_breakdown=generalization
        )
        
        logger.info("Comprehensive evaluation completed!")
        return metrics
    
    def _expressions_equivalent(self, expr1, expr2) -> bool:
        """Check if two expressions are mathematically equivalent."""
        try:
            is_valid, _ = self.verifier.verify_step_validity(expr1, expr2)
            return is_valid
        except:
            return False
    
    def _verify_solution(self, original_problem, solution, expected_answer) -> bool:
        """Verify if a solution is correct."""
        try:
            # Check if solution satisfies original problem
            is_correct, _ = self.verifier.check_solution(original_problem, solution)
            return is_correct
        except:
            return False
    
    def _get_baseline_solution(self, problem) -> List:
        """Get baseline solution using rule-based solver."""
        try:
            if problem.problem_type == 'linear_equation':
                return self.baseline_solver.solve_linear_equation(problem.initial_expression)
            elif problem.problem_type == 'quadratic_equation':
                return self.baseline_solver.solve_quadratic_equation(problem.initial_expression)
            else:
                return self.baseline_solver.simplify_expression_steps(problem.initial_expression)
        except:
            return []


def create_evaluation_report(metrics: EvaluationMetrics, output_dir: Path):
    """Create comprehensive evaluation report with visualizations."""
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics as JSON
    with open(output_dir / 'evaluation_metrics.json', 'w') as f:
        json.dump(metrics.to_dict(), f, indent=2)
    
    # Create visualizations
    plt.style.use('seaborn-v0_8')
    
    # Overall metrics bar chart
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Main metrics
    main_metrics = {
        'Step Accuracy': metrics.step_accuracy,
        'Solution Success Rate': metrics.solution_success_rate,
        'Step Validity': metrics.step_validity,
        'Avg Solution Length': metrics.avg_solution_length / 10  # Normalize for display
    }
    
    axes[0, 0].bar(main_metrics.keys(), main_metrics.values())
    axes[0, 0].set_title('Overall Performance Metrics')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Success rate by problem type
    if metrics.problem_type_breakdown:
        types = list(metrics.problem_type_breakdown.keys())
        rates = [metrics.problem_type_breakdown[t]['success_rate'] for t in types]
        
        axes[0, 1].bar(types, rates)
        axes[0, 1].set_title('Success Rate by Problem Type')
        axes[0, 1].set_ylabel('Success Rate')
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Difficulty breakdown
    if metrics.difficulty_breakdown:
        difficulties = list(metrics.difficulty_breakdown.keys())
        diff_rates = [metrics.difficulty_breakdown[d]['success_rate'] for d in difficulties]
        
        axes[1, 0].bar(difficulties, diff_rates)
        axes[1, 0].set_title('Success Rate by Difficulty')
        axes[1, 0].set_ylabel('Success Rate')
    
    # Performance metrics
    perf_data = {
        'Inference Time (ms)': metrics.inference_time_ms,
        'Avg Steps': metrics.avg_solution_length
    }
    
    axes[1, 1].bar(perf_data.keys(), perf_data.values())
    axes[1, 1].set_title('Performance Metrics')
    axes[1, 1].set_ylabel('Value')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'evaluation_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create detailed report
    report_lines = [
        "# Graph Neural Algebra Tutor - Evaluation Report",
        "",
        f"## Overall Performance",
        f"- **Step Accuracy**: {metrics.step_accuracy:.3f}",
        f"- **Solution Success Rate**: {metrics.solution_success_rate:.3f}",
        f"- **Step Validity**: {metrics.step_validity:.3f}",
        f"- **Average Solution Length**: {metrics.avg_solution_length:.1f} steps",
        f"- **Average Inference Time**: {metrics.inference_time_ms:.1f} ms",
        "",
        "## Problem Type Breakdown",
    ]
    
    for ptype, data in metrics.problem_type_breakdown.items():
        report_lines.extend([
            f"### {ptype.title()}",
            f"- Success Rate: {data['success_rate']:.3f}",
            f"- Problems Solved: {data['solved']}/{data['total']}",
            f"- Average Steps: {data['avg_solution_length']:.1f}",
            ""
        ])
    
    with open(output_dir / 'evaluation_report.md', 'w') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"Evaluation report saved to {output_dir}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate Graph Neural Algebra Tutor')
    parser.add_argument('--model-path', type=str, help='Path to trained model')
    parser.add_argument('--data-path', type=str, help='Path to test dataset')
    parser.add_argument('--output-dir', type=str, default='evaluation_results', help='Output directory')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    parser.add_argument('--num-problems', type=int, default=1000, help='Number of problems to evaluate')
    parser.add_argument('--generate-data', action='store_true', help='Generate test data')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load or generate test data
    if args.generate_data or not args.data_path:
        logger.info("Generating test data...")
        generator = AlgebraDatasetGenerator(seed=42)
        
        # Generate diverse test set
        problems = generator.generate_mixed_dataset(
            n_linear=args.num_problems // 3,
            n_quadratic=args.num_problems // 3,
            n_simplify=args.num_problems // 3
        )
        
        # Save generated data
        generator.save_dataset(problems, output_dir / 'test_data.json')
        
    else:
        logger.info(f"Loading test data from {args.data_path}")
        problems = AlgebraDatasetGenerator.load_dataset(args.data_path)
    
    # Initialize evaluator
    evaluator = ComprehensiveEvaluator(
        model_path=args.model_path,
        device=args.device,
        use_verification=True
    )
    
    # Run evaluation
    metrics = evaluator.run_comprehensive_evaluation(problems[:args.num_problems])
    
    # Create report
    create_evaluation_report(metrics, output_dir)
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Step Accuracy: {metrics.step_accuracy:.3f}")
    print(f"Solution Success Rate: {metrics.solution_success_rate:.3f}")
    print(f"Step Validity: {metrics.step_validity:.3f}")
    print(f"Average Solution Length: {metrics.avg_solution_length:.1f} steps")
    print(f"Average Inference Time: {metrics.inference_time_ms:.1f} ms")
    print(f"\nDetailed report saved to: {output_dir}")


if __name__ == "__main__":
    main() 