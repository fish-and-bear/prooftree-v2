#!/usr/bin/env python3
"""
Benchmarking script for the Graph Neural Algebra Tutor.

This script measures:
1. Inference latency
2. Memory usage
3. Throughput
4. Scalability with different input sizes
"""

import argparse
import json
import logging
import time
import psutil
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from src.models.gnn_model import GraphNeuralAlgebraTutor
from src.graph import expression_to_graph
from src.solver import GNNAlgebraSolver
from scripts.train_gnn import AlgebraDataset, collate_fn

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def measure_memory_usage():
    """Measure current memory usage."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB


def benchmark_inference_latency(model: nn.Module,
                               dataloader: DataLoader,
                               device: torch.device,
                               num_runs: int = 100) -> Dict[str, float]:
    """Benchmark inference latency."""
    model.eval()
    latencies = []
    
    # Warm up
    logger.info("Warming up...")
    for _ in range(10):
        batch = next(iter(dataloader))
        graphs = batch['graphs'].to(device)
        _ = model(graphs)
    
    # Benchmark
    logger.info(f"Benchmarking inference latency ({num_runs} runs)...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            if i >= num_runs:
                break
                
            graphs = batch['graphs'].to(device)
            
            start_time = time.time()
            _ = model(graphs)
            end_time = time.time()
            
            latency = (end_time - start_time) * 1000  # Convert to ms
            latencies.append(latency)
    
    return {
        'mean_latency_ms': np.mean(latencies),
        'std_latency_ms': np.std(latencies),
        'min_latency_ms': np.min(latencies),
        'max_latency_ms': np.max(latencies),
        'p50_latency_ms': np.percentile(latencies, 50),
        'p95_latency_ms': np.percentile(latencies, 95),
        'p99_latency_ms': np.percentile(latencies, 99),
        'throughput_problems_per_sec': len(latencies) / (np.sum(latencies) / 1000)
    }


def benchmark_memory_usage(model: nn.Module,
                          dataloader: DataLoader,
                          device: torch.device) -> Dict[str, float]:
    """Benchmark memory usage."""
    model.eval()
    memory_usage = []
    
    # Measure baseline memory
    baseline_memory = measure_memory_usage()
    
    logger.info("Benchmarking memory usage...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Memory benchmark"):
            graphs = batch['graphs'].to(device)
            
            # Measure memory before inference
            memory_before = measure_memory_usage()
            
            # Run inference
            _ = model(graphs)
            
            # Measure memory after inference
            memory_after = measure_memory_usage()
            
            memory_usage.append(memory_after - memory_before)
    
    return {
        'baseline_memory_mb': baseline_memory,
        'mean_memory_increase_mb': np.mean(memory_usage),
        'max_memory_increase_mb': np.max(memory_usage),
        'std_memory_increase_mb': np.std(memory_usage)
    }


def benchmark_scalability(model: nn.Module,
                         device: torch.device) -> Dict[str, Dict[str, float]]:
    """Benchmark scalability with different input sizes."""
    model.eval()
    results = {}
    
    # Test different graph sizes
    graph_sizes = [5, 10, 20, 50, 100]
    
    logger.info("Benchmarking scalability...")
    for size in graph_sizes:
        logger.info(f"Testing with {size} nodes...")
        
        # Create dummy graph with specified size
        x = torch.randn((size, 20))  # 20 features
        edge_index = torch.randint(0, size, (2, size * 2))  # Some edges
        graph = torch_geometric.data.Data(x=x, edge_index=edge_index)
        
        # Create batch
        batch = torch_geometric.data.Batch.from_data_list([graph])
        batch = batch.to(device)
        
        # Measure latency
        latencies = []
        for _ in range(50):  # Multiple runs for averaging
            start_time = time.time()
            _ = model(batch)
            end_time = time.time()
            latencies.append((end_time - start_time) * 1000)
        
        results[f'{size}_nodes'] = {
            'mean_latency_ms': np.mean(latencies),
            'std_latency_ms': np.std(latencies)
        }
    
    return results


def benchmark_solver_performance(solver: GNNAlgebraSolver,
                                test_problems: List[str]) -> Dict[str, float]:
    """Benchmark the full solver performance."""
    latencies = []
    step_counts = []
    
    logger.info("Benchmarking solver performance...")
    for problem in tqdm(test_problems, desc="Solver benchmark"):
        start_time = time.time()
        steps = solver.solve(problem, show_steps=False)
        end_time = time.time()
        
        latency = (end_time - start_time) * 1000
        latencies.append(latency)
        step_counts.append(len(steps))
    
    return {
        'mean_solve_time_ms': np.mean(latencies),
        'std_solve_time_ms': np.std(latencies),
        'mean_steps_per_problem': np.mean(step_counts),
        'std_steps_per_problem': np.std(step_counts),
        'throughput_problems_per_sec': len(latencies) / (np.sum(latencies) / 1000)
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark Graph Neural Algebra Tutor")
    parser.add_argument("--model-path", type=str, help="Path to trained model (optional)")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--output-dir", type=str, default="benchmark_results", help="Output directory")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-runs", type=int, default=100, help="Number of benchmark runs")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Set device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load model if provided
    if args.model_path:
        model = GraphNeuralAlgebraTutor().to(device)
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model from {args.model_path}")
    else:
        model = GraphNeuralAlgebraTutor().to(device)
        logger.info("Using untrained model for benchmarking")
    
    # Load test data if available
    data_dir = Path(args.data_dir)
    if (data_dir / "test.json").exists():
        from src.data.dataset_generator import AlgebraDatasetGenerator
        test_problems = AlgebraDatasetGenerator.load_dataset(data_dir / "test.json")
        test_dataset = AlgebraDataset(test_problems[:100])  # Use subset for benchmarking
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        logger.info(f"Loaded {len(test_dataset)} test problems")
    else:
        logger.warning("No test data found, skipping data-dependent benchmarks")
        test_loader = None
    
    # Run benchmarks
    results = {}
    
    # Model inference benchmarks
    if test_loader:
        logger.info("Running inference latency benchmark...")
        latency_results = benchmark_inference_latency(model, test_loader, device, args.num_runs)
        results['inference_latency'] = latency_results
        
        logger.info("Running memory usage benchmark...")
        memory_results = benchmark_memory_usage(model, test_loader, device)
        results['memory_usage'] = memory_results
    
    # Scalability benchmark
    logger.info("Running scalability benchmark...")
    scalability_results = benchmark_scalability(model, device)
    results['scalability'] = scalability_results
    
    # Solver performance benchmark
    if test_loader:
        logger.info("Running solver performance benchmark...")
        solver = GNNAlgebraSolver(use_verification=True)
        test_expressions = [p.initial_expression for p in test_problems[:50]]
        solver_results = benchmark_solver_performance(solver, test_expressions)
        results['solver_performance'] = solver_results
    
    # Print results
    logger.info("Benchmark Results:")
    if 'inference_latency' in results:
        logger.info(f"Mean Inference Latency: {results['inference_latency']['mean_latency_ms']:.2f} ms")
        logger.info(f"Inference Throughput: {results['inference_latency']['throughput_problems_per_sec']:.2f} problems/sec")
    
    if 'memory_usage' in results:
        logger.info(f"Mean Memory Increase: {results['memory_usage']['mean_memory_increase_mb']:.2f} MB")
    
    if 'solver_performance' in results:
        logger.info(f"Mean Solve Time: {results['solver_performance']['mean_solve_time_ms']:.2f} ms")
        logger.info(f"Solver Throughput: {results['solver_performance']['throughput_problems_per_sec']:.2f} problems/sec")
    
    # Save results
    with open(output_dir / "benchmark_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_dir / 'benchmark_results.json'}")


if __name__ == "__main__":
    main() 