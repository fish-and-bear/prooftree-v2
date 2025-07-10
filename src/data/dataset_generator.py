"""Dataset generator for algebra problems with step-by-step solutions.

This module creates training datasets by generating various types of algebra
problems and their step-by-step solutions.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, asdict
import numpy as np
from tqdm import tqdm

import sympy as sp
from sympy import Symbol, Eq

from ..utils import (
    StepByStepSolver,
    generate_random_linear_equation,
    generate_random_quadratic_equation,
    AlgebraicStep,
)
from ..graph import expression_to_graph


@dataclass
class AlgebraProblem:
    """Represents an algebra problem with its solution steps."""
    problem_id: str
    problem_type: str  # 'linear_equation', 'quadratic_equation', 'simplification'
    problem_text: str
    initial_expression: str
    steps: List[Dict[str, str]]  # List of {expression, operation, description}
    final_answer: str
    difficulty: int  # 1-5 scale
    metadata: Dict[str, any] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AlgebraProblem':
        """Create from dictionary."""
        return cls(**data)


class AlgebraDatasetGenerator:
    """Generates datasets of algebra problems."""
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.solver = StepByStepSolver()
        self.problem_counter = 0
    
    def generate_linear_equation_problems(self, 
                                        n_problems: int,
                                        difficulty_range: Tuple[int, int] = (1, 3)) -> List[AlgebraProblem]:
        """Generate linear equation problems.
        
        Args:
            n_problems: Number of problems to generate
            difficulty_range: Range of difficulty levels
            
        Returns:
            List of algebra problems
        """
        problems = []
        
        for _ in tqdm(range(n_problems), desc="Generating linear equations"):
            difficulty = random.randint(*difficulty_range)
            
            # Adjust ranges based on difficulty
            if difficulty == 1:
                coeff_range = (-5, 5)
                const_range = (-10, 10)
            elif difficulty == 2:
                coeff_range = (-10, 10)
                const_range = (-20, 20)
            else:  # difficulty >= 3
                coeff_range = (-20, 20)
                const_range = (-50, 50)
            
            # Generate equation
            equation = generate_random_linear_equation(
                coeff_range=coeff_range,
                const_range=const_range
            )
            
            # Solve step by step
            try:
                steps = self.solver.solve_linear_equation(equation)
                
                # Convert steps to dictionary format
                step_dicts = []
                for step in steps:
                    step_dict = {
                        "expression": str(step.expression),
                        "operation": step.operation.name if step.operation else "",
                        "description": step.description
                    }
                    step_dicts.append(step_dict)
                
                # Create problem object
                problem = AlgebraProblem(
                    problem_id=f"linear_{self.problem_counter:05d}",
                    problem_type="linear_equation",
                    problem_text=f"Solve for x: {equation}",
                    initial_expression=str(equation),
                    steps=step_dicts,
                    final_answer=str(steps[-1].expression),
                    difficulty=difficulty,
                    metadata={
                        "variable": "x",
                        "equation_form": "ax + b = cx + d"
                    }
                )
                
                problems.append(problem)
                self.problem_counter += 1
                
            except Exception as e:
                print(f"Error generating problem: {e}")
                continue
        
        return problems
    
    def generate_quadratic_equation_problems(self,
                                           n_problems: int,
                                           difficulty_range: Tuple[int, int] = (2, 4)) -> List[AlgebraProblem]:
        """Generate quadratic equation problems.
        
        Args:
            n_problems: Number of problems to generate
            difficulty_range: Range of difficulty levels
            
        Returns:
            List of algebra problems
        """
        problems = []
        
        for _ in tqdm(range(n_problems), desc="Generating quadratic equations"):
            difficulty = random.randint(*difficulty_range)
            
            # Adjust ranges based on difficulty
            if difficulty <= 2:
                coeff_range = (-3, 3)
                root_range = (-5, 5)
            elif difficulty == 3:
                coeff_range = (-5, 5)
                root_range = (-10, 10)
            else:  # difficulty >= 4
                coeff_range = (-10, 10)
                root_range = (-15, 15)
            
            # Generate equation
            equation = generate_random_quadratic_equation(
                coeff_range=coeff_range,
                root_range=root_range
            )
            
            # Solve step by step
            try:
                steps = self.solver.solve_quadratic_equation(equation)
                
                # Convert steps to dictionary format
                step_dicts = []
                for step in steps:
                    step_dict = {
                        "expression": str(step.expression),
                        "operation": step.operation.name if step.operation else "",
                        "description": step.description
                    }
                    step_dicts.append(step_dict)
                
                # Create problem object
                problem = AlgebraProblem(
                    problem_id=f"quadratic_{self.problem_counter:05d}",
                    problem_type="quadratic_equation",
                    problem_text=f"Solve for x: {equation}",
                    initial_expression=str(equation),
                    steps=step_dicts,
                    final_answer=str(steps[-1].expression),
                    difficulty=difficulty,
                    metadata={
                        "variable": "x",
                        "equation_form": "ax^2 + bx + c = 0"
                    }
                )
                
                problems.append(problem)
                self.problem_counter += 1
                
            except Exception as e:
                print(f"Error generating problem: {e}")
                continue
        
        return problems
    
    def generate_simplification_problems(self,
                                       n_problems: int,
                                       difficulty_range: Tuple[int, int] = (1, 3)) -> List[AlgebraProblem]:
        """Generate expression simplification problems.
        
        Args:
            n_problems: Number of problems to generate
            difficulty_range: Range of difficulty levels
            
        Returns:
            List of algebra problems
        """
        problems = []
        
        for _ in tqdm(range(n_problems), desc="Generating simplification problems"):
            difficulty = random.randint(*difficulty_range)
            x = Symbol('x')
            
            # Generate expression based on difficulty
            if difficulty == 1:
                # Simple like terms
                n_terms = random.randint(2, 4)
                coeffs = [random.randint(-5, 5) for _ in range(n_terms)]
                consts = [random.randint(-10, 10) for _ in range(n_terms)]
                expr = sum(c*x for c in coeffs) + sum(consts)
                
            elif difficulty == 2:
                # More complex with multiplication
                expr1 = random.randint(1, 5) * x + random.randint(-10, 10)
                expr2 = random.randint(1, 5) * x + random.randint(-10, 10)
                if random.choice([True, False]):
                    expr = expr1 + expr2
                else:
                    expr = random.randint(2, 4) * (expr1)
                    
            else:  # difficulty >= 3
                # Complex with powers or products
                if random.choice([True, False]):
                    # Expand a product
                    expr1 = random.randint(1, 3) * x + random.randint(-5, 5)
                    expr2 = random.randint(1, 3) * x + random.randint(-5, 5)
                    expr = expr1 * expr2
                else:
                    # Mix of terms
                    expr = (random.randint(1, 3) * x**2 + 
                           random.randint(-5, 5) * x + 
                           random.randint(-10, 10))
            
            # Simplify step by step
            try:
                steps = self.solver.simplify_expression_steps(expr)
                
                # Convert steps to dictionary format
                step_dicts = []
                for step in steps:
                    step_dict = {
                        "expression": str(step.expression),
                        "operation": step.operation.name if step.operation else "",
                        "description": step.description
                    }
                    step_dicts.append(step_dict)
                
                # Create problem object
                problem = AlgebraProblem(
                    problem_id=f"simplify_{self.problem_counter:05d}",
                    problem_type="simplification",
                    problem_text=f"Simplify: {expr}",
                    initial_expression=str(expr),
                    steps=step_dicts,
                    final_answer=str(steps[-1].expression),
                    difficulty=difficulty,
                    metadata={
                        "variable": "x",
                        "expression_type": "polynomial"
                    }
                )
                
                problems.append(problem)
                self.problem_counter += 1
                
            except Exception as e:
                print(f"Error generating problem: {e}")
                continue
        
        return problems
    
    def generate_mixed_dataset(self,
                             n_linear: int = 1000,
                             n_quadratic: int = 500,
                             n_simplify: int = 500) -> List[AlgebraProblem]:
        """Generate a mixed dataset with various problem types.
        
        Args:
            n_linear: Number of linear equation problems
            n_quadratic: Number of quadratic equation problems
            n_simplify: Number of simplification problems
            
        Returns:
            List of all problems
        """
        all_problems = []
        
        # Generate each type
        all_problems.extend(self.generate_linear_equation_problems(n_linear))
        all_problems.extend(self.generate_quadratic_equation_problems(n_quadratic))
        all_problems.extend(self.generate_simplification_problems(n_simplify))
        
        # Shuffle the dataset
        random.shuffle(all_problems)
        
        return all_problems
    
    def save_dataset(self, problems: List[AlgebraProblem], filepath: Union[str, Path]):
        """Save dataset to JSON file.
        
        Args:
            problems: List of problems to save
            filepath: Path to save the JSON file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dictionaries
        data = {
            "metadata": {
                "total_problems": len(problems),
                "problem_types": {
                    "linear_equation": sum(1 for p in problems if p.problem_type == "linear_equation"),
                    "quadratic_equation": sum(1 for p in problems if p.problem_type == "quadratic_equation"),
                    "simplification": sum(1 for p in problems if p.problem_type == "simplification"),
                },
                "difficulty_distribution": {
                    str(i): sum(1 for p in problems if p.difficulty == i) for i in range(1, 6)
                }
            },
            "problems": [p.to_dict() for p in problems]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Dataset saved to {filepath}")
        print(f"Total problems: {len(problems)}")
        print(f"Problem types: {data['metadata']['problem_types']}")
    
    @staticmethod
    def load_dataset(filepath: Union[str, Path]) -> List[AlgebraProblem]:
        """Load dataset from JSON file.
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            List of algebra problems
        """
        filepath = Path(filepath)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        problems = [AlgebraProblem.from_dict(p) for p in data['problems']]
        
        print(f"Loaded {len(problems)} problems from {filepath}")
        if 'metadata' in data:
            print(f"Problem types: {data['metadata']['problem_types']}")
        
        return problems


def create_train_val_test_split(problems: List[AlgebraProblem],
                               train_ratio: float = 0.8,
                               val_ratio: float = 0.1,
                               test_ratio: float = 0.1,
                               seed: Optional[int] = None) -> Tuple[List[AlgebraProblem], 
                                                                   List[AlgebraProblem], 
                                                                   List[AlgebraProblem]]:
    """Split dataset into train, validation, and test sets.
    
    Args:
        problems: List of all problems
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_problems, val_problems, test_problems)
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")
    
    if seed is not None:
        random.seed(seed)
    
    # Shuffle problems
    problems = problems.copy()
    random.shuffle(problems)
    
    # Calculate split points
    n_total = len(problems)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # Split
    train_problems = problems[:n_train]
    val_problems = problems[n_train:n_train + n_val]
    test_problems = problems[n_train + n_val:]
    
    print(f"Dataset split: Train={len(train_problems)}, Val={len(val_problems)}, Test={len(test_problems)}")
    
    return train_problems, val_problems, test_problems


if __name__ == "__main__":
    # Example usage
    generator = AlgebraDatasetGenerator(seed=42)
    
    # Generate a small test dataset
    print("Generating test dataset...")
    problems = generator.generate_mixed_dataset(
        n_linear=10,
        n_quadratic=5,
        n_simplify=5
    )
    
    # Save dataset
    generator.save_dataset(problems, "data/test_dataset.json")
    
    # Load and verify
    loaded_problems = AlgebraDatasetGenerator.load_dataset("data/test_dataset.json")
    
    # Show example problem
    if loaded_problems:
        example = loaded_problems[0]
        print(f"\nExample Problem:")
        print(f"Type: {example.problem_type}")
        print(f"Problem: {example.problem_text}")
        print(f"Steps:")
        for i, step in enumerate(example.steps):
            print(f"  {i}: {step['expression']} ({step['description']})") 