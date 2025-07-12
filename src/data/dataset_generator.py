"""Enhanced dataset generator for algebra problems with step-by-step solutions.

This module creates comprehensive training datasets by generating various types of algebra
problems including advanced patterns, complex expressions, and multi-step derivations.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, asdict
import numpy as np
from tqdm import tqdm

import sympy as sp
from sympy import Symbol, Eq, solve, expand, factor, simplify, collect, cancel, apart

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
    problem_type: str  # 'linear_equation', 'quadratic_equation', 'simplification', 'advanced'
    problem_text: str
    initial_expression: str
    steps: List[Dict[str, str]]  # List of {expression, operation, description}
    final_answer: str
    difficulty: int  # 1-10 scale (expanded from 1-5)
    complexity_score: float  # 0.0-1.0 based on expression complexity
    metadata: Dict[str, any] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AlgebraProblem':
        """Create from dictionary."""
        return cls(**data)


class EnhancedAlgebraDatasetGenerator:
    """Enhanced generator for comprehensive algebra datasets."""
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.solver = StepByStepSolver()
        self.problem_counter = 0
        
        # Define advanced problem types
        self.advanced_patterns = [
            'polynomial_division', 'rational_expressions', 'factoring_complex',
            'systems_of_equations', 'inequalities', 'absolute_value',
            'radical_equations', 'exponential_equations', 'logarithmic_equations',
            'trigonometric_equations', 'complex_numbers', 'parametric_equations'
        ]
    
    def calculate_complexity_score(self, expression: str) -> float:
        """Calculate complexity score (0.0-1.0) based on expression features."""
        score = 0.0
        
        # Length factor
        score += min(len(expression) / 100.0, 0.2)
        
        # Variable count
        variables = set(c for c in expression if c.isalpha())
        score += min(len(variables) / 5.0, 0.2)
        
        # Operator complexity
        operators = expression.count('+') + expression.count('-') + expression.count('*') + expression.count('/')
        score += min(operators / 10.0, 0.2)
        
        # Parentheses depth
        depth = 0
        max_depth = 0
        for char in expression:
            if char == '(':
                depth += 1
                max_depth = max(max_depth, depth)
            elif char == ')':
                depth -= 1
        score += min(max_depth / 5.0, 0.2)
        
        # Special functions
        special_funcs = ['sin', 'cos', 'tan', 'log', 'ln', 'exp', 'sqrt', 'abs']
        for func in special_funcs:
            if func in expression.lower():
                score += 0.1
        
        return min(score, 1.0)
    
    def generate_advanced_polynomial_problems(self, n_problems: int) -> List[AlgebraProblem]:
        """Generate advanced polynomial problems."""
        problems = []
        
        for _ in tqdm(range(n_problems), desc="Generating advanced polynomials"):
            try:
                # Generate complex polynomial expressions
                x = Symbol('x')
                y = Symbol('y')
                
                # Random polynomial degree (2-6)
                degree = random.randint(2, 6)
                coeffs = [random.randint(-10, 10) for _ in range(degree + 1)]
                
                # Create polynomial
                poly = sum(coeffs[i] * x**i for i in range(len(coeffs)))
                
                # Add some complexity
                if random.random() < 0.5:
                    poly = poly * (x + random.randint(-5, 5))
                
                if random.random() < 0.3:
                    poly = expand(poly * (x**2 + random.randint(-3, 3) * x + random.randint(-3, 3)))
                
                # Create equation
                equation = Eq(poly, 0)
                
                # Solve step by step
                steps = self._solve_polynomial_step_by_step(equation)
                
                if steps:
                    step_dicts = [{"expression": str(s.expression), "operation": s.operation.name if s.operation else "", "description": s.description} for s in steps]
                    
                    problem = AlgebraProblem(
                        problem_id=f"poly_{self.problem_counter:05d}",
                        problem_type="polynomial_equation",
                        problem_text=f"Solve: {equation}",
                        initial_expression=str(equation),
                        steps=step_dicts,
                        final_answer=str(steps[-1].expression),
                        difficulty=random.randint(6, 9),
                        complexity_score=self.calculate_complexity_score(str(equation)),
                        metadata={"degree": degree, "equation_form": "polynomial"}
                    )
                    
                    problems.append(problem)
                    self.problem_counter += 1
                    
            except Exception as e:
                continue
        
        return problems
    
    def generate_rational_expression_problems(self, n_problems: int) -> List[AlgebraProblem]:
        """Generate rational expression problems."""
        problems = []
        
        for _ in tqdm(range(n_problems), desc="Generating rational expressions"):
            try:
                x = Symbol('x')
                
                # Generate numerator and denominator
                num_degree = random.randint(1, 3)
                den_degree = random.randint(1, 3)
                
                num_coeffs = [random.randint(-8, 8) for _ in range(num_degree + 1)]
                den_coeffs = [random.randint(-8, 8) for _ in range(den_degree + 1)]
                
                numerator = sum(num_coeffs[i] * x**i for i in range(len(num_coeffs)))
                denominator = sum(den_coeffs[i] * x**i for i in range(len(den_coeffs)))
                
                # Create rational expression
                rational_expr = numerator / denominator
                
                # Simplify step by step
                steps = self._simplify_rational_step_by_step(rational_expr)
                
                if steps:
                    step_dicts = [{"expression": str(s.expression), "operation": s.operation.name if s.operation else "", "description": s.description} for s in steps]
                    
                    problem = AlgebraProblem(
                        problem_id=f"rational_{self.problem_counter:05d}",
                        problem_type="rational_expression",
                        problem_text=f"Simplify: {rational_expr}",
                        initial_expression=str(rational_expr),
                        steps=step_dicts,
                        final_answer=str(steps[-1].expression),
                        difficulty=random.randint(5, 8),
                        complexity_score=self.calculate_complexity_score(str(rational_expr)),
                        metadata={"numerator_degree": num_degree, "denominator_degree": den_degree}
                    )
                    
                    problems.append(problem)
                    self.problem_counter += 1
                    
            except Exception as e:
                continue
        
        return problems
    
    def generate_system_of_equations_problems(self, n_problems: int) -> List[AlgebraProblem]:
        """Generate systems of equations problems."""
        problems = []
        
        for _ in tqdm(range(n_problems), desc="Generating systems of equations"):
            try:
                x, y = Symbol('x'), Symbol('y')
                
                # Generate 2x2 system
                a1, b1, c1 = random.randint(-5, 5), random.randint(-5, 5), random.randint(-10, 10)
                a2, b2, c2 = random.randint(-5, 5), random.randint(-5, 5), random.randint(-10, 10)
                
                eq1 = Eq(a1 * x + b1 * y, c1)
                eq2 = Eq(a2 * x + b2 * y, c2)
                
                system = [eq1, eq2]
                
                # Solve step by step
                steps = self._solve_system_step_by_step(system)
                
                if steps:
                    step_dicts = [{"expression": str(s.expression), "operation": s.operation.name if s.operation else "", "description": s.description} for s in steps]
                    
                    problem = AlgebraProblem(
                        problem_id=f"system_{self.problem_counter:05d}",
                        problem_type="system_of_equations",
                        problem_text=f"Solve the system: {eq1}, {eq2}",
                        initial_expression=f"[{eq1}, {eq2}]",
                        steps=step_dicts,
                        final_answer=str(steps[-1].expression),
                        difficulty=random.randint(7, 9),
                        complexity_score=self.calculate_complexity_score(f"{eq1} {eq2}"),
                        metadata={"variables": ["x", "y"], "method": "substitution"}
                    )
                    
                    problems.append(problem)
                    self.problem_counter += 1
                    
            except Exception as e:
                continue
        
        return problems
    
    def generate_inequality_problems(self, n_problems: int) -> List[AlgebraProblem]:
        """Generate inequality problems."""
        problems = []
        
        for _ in tqdm(range(n_problems), desc="Generating inequalities"):
            try:
                x = Symbol('x')
                
                # Generate inequality
                left_side = random.randint(-10, 10) * x + random.randint(-20, 20)
                right_side = random.randint(-10, 10) * x + random.randint(-20, 20)
                
                # Random inequality operator
                operators = ['<', '<=', '>', '>=']
                op = random.choice(operators)
                
                if op == '<':
                    inequality = left_side < right_side
                elif op == '<=':
                    inequality = left_side <= right_side
                elif op == '>':
                    inequality = left_side > right_side
                else:
                    inequality = left_side >= right_side
                
                # Solve step by step
                steps = self._solve_inequality_step_by_step(inequality)
                
                if steps:
                    step_dicts = [{"expression": str(s.expression), "operation": s.operation.name if s.operation else "", "description": s.description} for s in steps]
                    
                    problem = AlgebraProblem(
                        problem_id=f"inequality_{self.problem_counter:05d}",
                        problem_type="inequality",
                        problem_text=f"Solve: {inequality}",
                        initial_expression=str(inequality),
                        steps=step_dicts,
                        final_answer=str(steps[-1].expression),
                        difficulty=random.randint(4, 7),
                        complexity_score=self.calculate_complexity_score(str(inequality)),
                        metadata={"operator": op, "variable": "x"}
                    )
                    
                    problems.append(problem)
                    self.problem_counter += 1
                    
            except Exception as e:
                continue
        
        return problems
    
    def _solve_polynomial_step_by_step(self, equation: Eq) -> List[AlgebraicStep]:
        """Solve polynomial equation step by step."""
        steps = [AlgebraicStep(equation, None, "Original polynomial equation")]
        
        try:
            # Step 1: Move everything to one side
            standard_form = equation.lhs - equation.rhs
            if standard_form != equation.lhs:
                steps.append(AlgebraicStep(Eq(standard_form, 0), None, "Move all terms to left side"))
            
            # Step 2: Factor if possible
            try:
                factored = factor(standard_form)
                if factored != standard_form:
                    steps.append(AlgebraicStep(Eq(factored, 0), None, "Factor the polynomial"))
                    standard_form = factored
            except:
                pass
            
            # Step 3: Solve
            solutions = solve(standard_form)
            if solutions:
                if len(solutions) == 1:
                    result = Eq(Symbol('x'), solutions[0])
                    steps.append(AlgebraicStep(result, None, f"Solution: x = {solutions[0]}"))
                else:
                    result = Eq(Symbol('x'), sp.FiniteSet(*solutions))
                    steps.append(AlgebraicStep(result, None, f"Solutions: x = {solutions}"))
            
        except Exception as e:
            steps.append(AlgebraicStep(equation, None, f"Error solving: {e}"))
        
        return steps
    
    def _simplify_rational_step_by_step(self, expr) -> List[AlgebraicStep]:
        """Simplify rational expression step by step."""
        steps = [AlgebraicStep(expr, None, "Original rational expression")]
        
        try:
            # Step 1: Factor numerator and denominator
            num, den = expr.as_numer_denom()
            
            try:
                factored_num = factor(num)
                factored_den = factor(den)
                if factored_num != num or factored_den != den:
                    factored_expr = factored_num / factored_den
                    steps.append(AlgebraicStep(factored_expr, None, "Factor numerator and denominator"))
                    expr = factored_expr
            except:
                pass
            
            # Step 2: Cancel common factors
            try:
                simplified = cancel(expr)
                if simplified != expr:
                    steps.append(AlgebraicStep(simplified, None, "Cancel common factors"))
                    expr = simplified
            except:
                pass
            
            # Step 3: Expand if beneficial
            try:
                expanded = expand(expr)
                if expanded != expr and len(str(expanded)) < len(str(expr)):
                    steps.append(AlgebraicStep(expanded, None, "Expand expression"))
                    expr = expanded
            except:
                pass
            
        except Exception as e:
            steps.append(AlgebraicStep(expr, None, f"Error simplifying: {e}"))
        
        return steps
    
    def _solve_system_step_by_step(self, system: List[Eq]) -> List[AlgebraicStep]:
        """Solve system of equations step by step."""
        steps = [AlgebraicStep(system, None, "Original system of equations")]
        
        try:
            x, y = Symbol('x'), Symbol('y')
            
            # Step 1: Solve one equation for one variable
            eq1, eq2 = system[0], system[1]
            
            # Solve eq1 for x
            x_expr = solve(eq1, x)[0]
            steps.append(AlgebraicStep(x_expr, None, f"Solve first equation for x: x = {x_expr}"))
            
            # Step 2: Substitute into second equation
            substituted = eq2.subs(x, x_expr)
            steps.append(AlgebraicStep(substituted, None, f"Substitute x = {x_expr} into second equation"))
            
            # Step 3: Solve for y
            y_solution = solve(substituted, y)[0]
            steps.append(AlgebraicStep(Eq(y, y_solution), None, f"Solve for y: y = {y_solution}"))
            
            # Step 4: Substitute back to find x
            x_solution = x_expr.subs(y, y_solution)
            steps.append(AlgebraicStep(Eq(x, x_solution), None, f"Substitute y = {y_solution} to find x: x = {x_solution}"))
            
            # Final solution
            final_solution = f"x = {x_solution}, y = {y_solution}"
            steps.append(AlgebraicStep(final_solution, None, f"Solution: {final_solution}"))
            
        except Exception as e:
            steps.append(AlgebraicStep(system, None, f"Error solving system: {e}"))
        
        return steps
    
    def _solve_inequality_step_by_step(self, inequality) -> List[AlgebraicStep]:
        """Solve inequality step by step."""
        steps = [AlgebraicStep(inequality, None, "Original inequality")]
        
        try:
            # Extract left and right sides
            if hasattr(inequality, 'lhs') and hasattr(inequality, 'rhs'):
                lhs, rhs = inequality.lhs, inequality.rhs
            else:
                # Handle different inequality types
                lhs, rhs = inequality.args[0], inequality.args[1]
            
            # Step 1: Move all terms to left side
            standard_form = lhs - rhs
            if standard_form != lhs:
                steps.append(AlgebraicStep(standard_form < 0, None, "Move all terms to left side"))
            
            # Step 2: Simplify
            simplified = simplify(standard_form)
            if simplified != standard_form:
                steps.append(AlgebraicStep(simplified < 0, None, "Simplify left side"))
                standard_form = simplified
            
            # Step 3: Solve for variable
            x = Symbol('x')
            solution = solve(standard_form < 0, x)
            if solution:
                steps.append(AlgebraicStep(solution, None, f"Solution: {solution}"))
            
        except Exception as e:
            steps.append(AlgebraicStep(inequality, None, f"Error solving inequality: {e}"))
        
        return steps 