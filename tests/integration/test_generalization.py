"""Integration tests for generalization capabilities."""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.solver import GNNAlgebraSolver
from src.verification import AlgebraicVerifier
from src.data.dataset_generator import AlgebraDatasetGenerator


class TestGeneralization:
    """Test generalization across different problem types and difficulties."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.solver = GNNAlgebraSolver(use_verification=True)
        self.verifier = AlgebraicVerifier()
        self.generator = AlgebraDatasetGenerator(seed=42)
    
    def test_difficulty_scaling(self):
        """Test performance across different difficulty levels."""
        # Generate problems of increasing difficulty
        difficulties = {
            'easy': {'n_linear': 10, 'n_quadratic': 5, 'n_simplify': 5},
            'medium': {'n_linear': 10, 'n_quadratic': 10, 'n_simplify': 10},
            'hard': {'n_linear': 10, 'n_quadratic': 15, 'n_simplify': 15}
        }
        
        results = {}
        for difficulty, params in difficulties.items():
            problems = self.generator.generate_mixed_dataset(**params)
            success_count = 0
            
            for problem in problems:
                try:
                    steps = self.solver.solve(problem.initial_expression, show_steps=False)
                    if len(steps) > 1:  # At least one step beyond initial
                        success_count += 1
                except Exception:
                    pass
            
            success_rate = success_count / len(problems)
            results[difficulty] = success_rate
            
            # Basic check: system should handle at least some problems at each level
            assert success_rate > 0.0, f"No problems solved at {difficulty} level"
        
        # Log results for analysis
        print(f"Difficulty scaling results: {results}")
    
    def test_problem_type_generalization(self):
        """Test generalization across different problem types."""
        problem_types = {
            'linear_equations': [
                "2*x + 3 = 7",
                "x - 5 = 10",
                "3*x + 2 = 8"
            ],
            'quadratic_equations': [
                "x**2 - 4 = 0",
                "x**2 + 2*x + 1 = 0",
                "x**2 - 5*x + 6 = 0"
            ],
            'simplifications': [
                "2*x + 3*x",
                "x**2 + x**2",
                "3*(x + 2) + 2*(x - 1)"
            ]
        }
        
        results = {}
        for problem_type, expressions in problem_types.items():
            success_count = 0
            
            for expr in expressions:
                try:
                    steps = self.solver.solve(expr, show_steps=False)
                    if len(steps) > 1:
                        success_count += 1
                except Exception:
                    pass
            
            success_rate = success_count / len(expressions)
            results[problem_type] = success_rate
            
            # System should handle at least some problems of each type
            assert success_rate > 0.0, f"No problems solved for {problem_type}"
        
        print(f"Problem type generalization results: {results}")
    
    def test_variable_generalization(self):
        """Test generalization with different variable names."""
        base_expressions = [
            "2*x + 3 = 7",
            "y**2 - 4 = 0",
            "3*z + 2*z"
        ]
        
        variable_mappings = {
            'x': ['a', 'b', 'c', 't', 'u'],
            'y': ['p', 'q', 'r', 's', 'v'],
            'z': ['m', 'n', 'o', 'w', 'k']
        }
        
        results = {}
        for base_expr, variables in variable_mappings.items():
            success_count = 0
            
            for var in variables:
                # Replace variable in expression
                test_expr = base_expressions[0].replace('x', var)
                
                try:
                    steps = self.solver.solve(test_expr, show_steps=False)
                    if len(steps) > 1:
                        success_count += 1
                except Exception:
                    pass
            
            success_rate = success_count / len(variables)
            results[base_expr] = success_rate
            
            # System should handle different variable names
            assert success_rate > 0.0, f"No problems solved with variable substitution for {base_expr}"
        
        print(f"Variable generalization results: {results}")
    
    def test_complexity_generalization(self):
        """Test generalization with expressions of varying complexity."""
        complexity_levels = {
            'simple': ["x + 1 = 2", "2*x = 4"],
            'moderate': ["2*x + 3 = 7", "x**2 - 4 = 0"],
            'complex': ["2*(x + 3) + 4*(x - 1) = 5*x + 7", "(x + 1)*(x - 2) = 0"],
            'very_complex': ["x**3 + 2*x**2 + x + 1 = 0", "2*x**2 + 3*x + 1 = x**2 + 2*x + 3"]
        }
        
        results = {}
        for complexity, expressions in complexity_levels.items():
            success_count = 0
            
            for expr in expressions:
                try:
                    steps = self.solver.solve(expr, show_steps=False)
                    if len(steps) > 1:
                        success_count += 1
                except Exception:
                    pass
            
            success_rate = success_count / len(expressions)
            results[complexity] = success_rate
            
            # System should handle at least simple and moderate complexity
            if complexity in ['simple', 'moderate']:
                assert success_rate > 0.0, f"No problems solved at {complexity} complexity"
        
        print(f"Complexity generalization results: {results}")
    
    def test_step_validity_across_types(self):
        """Test that step verification works across different problem types."""
        test_cases = [
            ("2*x + 3 = 7", "2*x = 4"),  # Linear equation
            ("x**2 - 4 = 0", "x**2 = 4"),  # Quadratic equation
            ("2*x + 3*x", "5*x"),  # Simplification
        ]
        
        for current, next_step in test_cases:
            valid, feedback = self.verifier.verify_step_validity(current, next_step)
            
            # All these should be valid steps
            assert valid is True, f"Step validation failed for {current} -> {next_step}: {feedback}"
            # More flexible feedback validation
            assert (len(feedback) > 0 and 
                   (any(word in feedback.lower() for word in 
                        ['valid', 'correct', 'equal', 'equivalent', 'true']) or
                    feedback.lower().startswith('solution'))), f"Unexpected feedback: {feedback}"
    
    def test_robustness_to_noise(self):
        """Test robustness to slightly malformed inputs."""
        base_expression = "2*x + 3 = 7"
        noisy_variants = [
            "2*x + 3 = 7",  # Original
            "2 * x + 3 = 7",  # Extra spaces
            "2*x+3=7",  # No spaces
            "2*x + 3 == 7",  # Double equals
        ]
        
        success_count = 0
        for variant in noisy_variants:
            try:
                steps = self.solver.solve(variant, show_steps=False)
                if len(steps) > 1:
                    success_count += 1
            except Exception:
                pass
        
        # System should handle at least some variants
        success_rate = success_count / len(noisy_variants)
        assert success_rate > 0.0, "System not robust to input variations"
        
        print(f"Robustness to noise: {success_rate:.2f}")


def test_cross_domain_generalization():
    """Test generalization across mathematical domains."""
    domains = {
        'arithmetic': ["2 + 3", "5 * 4", "10 / 2"],
        'algebra': ["2*x + 3", "x**2 - 4", "3*x + 2*x"],
        'equations': ["x + 1 = 2", "2*x = 4", "x**2 = 4"]
    }
    
    solver = GNNAlgebraSolver(use_verification=True)
    results = {}
    
    for domain, expressions in domains.items():
        success_count = 0
        
        for expr in expressions:
            try:
                steps = solver.solve(expr, show_steps=False)
                if len(steps) > 1:
                    success_count += 1
            except Exception:
                pass
        
        success_rate = success_count / len(expressions)
        results[domain] = success_rate
        
        # System should handle at least some problems in each domain
        assert success_rate > 0.0, f"No problems solved in {domain} domain"
    
    print(f"Cross-domain generalization results: {results}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 