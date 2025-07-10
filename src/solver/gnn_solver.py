"""GNN-based solver for step-by-step algebra problem solving.

This module integrates the Graph Neural Network model with symbolic
verification to solve algebra problems step by step.
"""

import torch
from torch_geometric.data import Batch
import sympy as sp
from sympy import Symbol, Eq, solve
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
from pathlib import Path

from ..graph import expression_to_graph, ExpressionGraph
from ..models import StepPredictor, OperationType, create_step_predictor
from ..verification import AlgebraicVerifier
from ..utils import AlgebraicTransformer, AlgebraicStep, AlgebraicOperation


logger = logging.getLogger(__name__)


class GNNAlgebraSolver:
    """Solves algebra problems using GNN predictions with symbolic verification."""
    
    def __init__(self,
                 model: Optional[StepPredictor] = None,
                 model_path: Optional[Union[str, Path]] = None,
                 device: str = 'cpu',
                 max_steps: int = 20,
                 use_verification: bool = True):
        """Initialize the solver.
        
        Args:
            model: Pre-loaded model (if None, will create default or load from path)
            model_path: Path to saved model weights
            device: Device to run model on ('cpu' or 'cuda')
            max_steps: Maximum number of solution steps
            use_verification: Whether to verify each step with SymPy
        """
        self.device = device
        self.max_steps = max_steps
        self.use_verification = use_verification
        
        # Initialize model
        if model is None:
            model = create_step_predictor()
            if model_path:
                model.load_state_dict(torch.load(model_path, map_location=device))
        
        self.model = model.to(device)
        self.model.eval()
        
        # Initialize components
        self.transformer = AlgebraicTransformer()
        self.verifier = AlgebraicVerifier()
        self.graph_converter = ExpressionGraph()
    
    def solve(self,
              problem: Union[str, sp.Basic, sp.Eq],
              show_steps: bool = True,
              return_all_steps: bool = True) -> Union[sp.Basic, List[AlgebraicStep]]:
        """Solve an algebra problem step by step.
        
        Args:
            problem: The problem to solve (equation or expression)
            show_steps: Whether to print steps as they're generated
            return_all_steps: Whether to return all steps or just the final answer
            
        Returns:
            Final answer or list of all steps
        """
        # Parse problem
        if isinstance(problem, str):
            problem = sp.parse_expr(problem, transformations='all')
            if '=' in str(problem):
                # Parse as equation
                parts = str(problem).split('=')
                if len(parts) == 2:
                    lhs = sp.parse_expr(parts[0])
                    rhs = sp.parse_expr(parts[1])
                    problem = Eq(lhs, rhs)
        
        # Initialize solution steps
        steps = [AlgebraicStep(problem, None, "Original problem")]
        current_expr = problem
        
        if show_steps:
            print(f"Solving: {problem}\n")
        
        # Solve step by step
        for step_num in range(self.max_steps):
            # Check if already solved
            if self._is_solved(current_expr):
                steps.append(AlgebraicStep(
                    current_expr,
                    AlgebraicOperation.SIMPLIFY,
                    "Solution found"
                ))
                break
            
            # Predict next step
            try:
                next_expr, operation, description = self._predict_next_step(current_expr)
                
                # Verify step if enabled
                if self.use_verification:
                    is_valid, verification_msg = self.verifier.verify_step_validity(
                        current_expr, next_expr, operation
                    )
                    
                    if not is_valid:
                        logger.warning(f"Invalid step predicted: {verification_msg}")
                        # Try alternative approach
                        next_expr, operation, description = self._fallback_step(current_expr)
                        if next_expr is None:
                            break
                
                # Add step
                steps.append(AlgebraicStep(next_expr, operation, description))
                current_expr = next_expr
                
                if show_steps:
                    print(f"Step {step_num + 1}: {next_expr} ({description})")
                
            except Exception as e:
                logger.error(f"Error in step {step_num + 1}: {e}")
                # Try fallback
                next_expr, operation, description = self._fallback_step(current_expr)
                if next_expr is None:
                    break
                steps.append(AlgebraicStep(next_expr, operation, description))
                current_expr = next_expr
        
        if show_steps:
            print(f"\nFinal answer: {current_expr}")
        
        if return_all_steps:
            return steps
        else:
            return current_expr
    
    def _predict_next_step(self,
                          current_expr: Union[sp.Basic, sp.Eq]) -> Tuple[sp.Basic, str, str]:
        """Predict the next step using the GNN model.
        
        Args:
            current_expr: Current expression or equation
            
        Returns:
            Tuple of (next_expression, operation_name, description)
        """
        # Convert to graph
        graph_data = expression_to_graph(current_expr)
        batch = Batch.from_data_list([graph_data]).to(self.device)
        
        # Get model prediction
        with torch.no_grad():
            operations, parameters, done_flags = self.model.predict_step(batch)
        
        operation_idx = operations[0]
        parameter = parameters[0]
        is_done = done_flags[0]
        
        if is_done:
            return current_expr, "SIMPLIFY", "Solution complete"
        
        # Map operation index to operation type
        operation_type = OperationType(operation_idx)
        
        # Apply the predicted operation
        next_expr, description = self._apply_operation(
            current_expr, operation_type, parameter
        )
        
        return next_expr, operation_type.name, description
    
    def _apply_operation(self,
                        expr: Union[sp.Basic, sp.Eq],
                        operation: OperationType,
                        parameter: float) -> Tuple[sp.Basic, str]:
        """Apply a predicted operation to an expression.
        
        Args:
            expr: Current expression or equation
            operation: Operation type to apply
            parameter: Parameter for the operation
            
        Returns:
            Tuple of (new_expression, description)
        """
        if not isinstance(expr, Eq):
            # Handle expression simplification
            if operation == OperationType.EXPAND:
                result = self.transformer.expand_expression(expr)
                return result, "Expanded expression"
            elif operation == OperationType.FACTOR:
                result = self.transformer.factor_expression(expr)
                return result, "Factored expression"
            elif operation == OperationType.SIMPLIFY:
                result = self.transformer.simplify_expression(expr)
                return result, "Simplified expression"
            elif operation == OperationType.COMBINE_LIKE_TERMS:
                result = self.transformer.combine_like_terms(expr)
                return result, "Combined like terms"
            else:
                # Default to simplification
                result = self.transformer.simplify_expression(expr)
                return result, "Simplified expression"
        
        # Handle equations
        if operation == OperationType.ADD_TO_BOTH_SIDES:
            # Use parameter as the term to add
            term = self._parameter_to_term(expr, parameter)
            result = self.transformer.add_to_both_sides(expr, term)
            return result, f"Add {term} to both sides"
        
        elif operation == OperationType.SUBTRACT_FROM_BOTH_SIDES:
            # Find appropriate term to subtract
            term = self._find_term_to_move(expr)
            if term:
                result = self.transformer.subtract_from_both_sides(expr, term)
                return result, f"Subtract {term} from both sides"
        
        elif operation == OperationType.MULTIPLY_BOTH_SIDES:
            factor = self._parameter_to_factor(expr, parameter)
            if factor != 0:
                result = self.transformer.multiply_both_sides(expr, factor)
                return result, f"Multiply both sides by {factor}"
        
        elif operation == OperationType.DIVIDE_BOTH_SIDES:
            divisor = self._find_coefficient(expr)
            if divisor and divisor != 1:
                result = self.transformer.divide_both_sides(expr, divisor)
                return result, f"Divide both sides by {divisor}"
        
        elif operation == OperationType.EXPAND:
            result = Eq(
                self.transformer.expand_expression(expr.lhs),
                self.transformer.expand_expression(expr.rhs)
            )
            return result, "Expanded both sides"
        
        elif operation == OperationType.SIMPLIFY:
            result = Eq(
                self.transformer.simplify_expression(expr.lhs),
                self.transformer.simplify_expression(expr.rhs)
            )
            return result, "Simplified both sides"
        
        # Default fallback
        return expr, "No operation applied"
    
    def _fallback_step(self,
                      expr: Union[sp.Basic, sp.Eq]) -> Tuple[Optional[sp.Basic], Optional[str], str]:
        """Generate a fallback step using rule-based approach.
        
        Args:
            expr: Current expression or equation
            
        Returns:
            Tuple of (next_expression, operation_name, description)
        """
        if isinstance(expr, Eq):
            # Try to isolate variable
            variables = list(expr.free_symbols)
            if variables:
                var = variables[0]
                
                # Check if we can solve directly
                try:
                    solutions = solve(expr, var)
                    if solutions:
                        if isinstance(solutions, list):
                            solution = solutions[0]
                        else:
                            solution = solutions
                        result = Eq(var, solution)
                        return result, "ISOLATE_VARIABLE", f"Solved for {var}"
                except:
                    pass
                
                # Try moving terms
                if isinstance(expr.rhs, sp.Add):
                    for term in expr.rhs.args:
                        if var not in term.free_symbols:
                            result = self.transformer.subtract_from_both_sides(expr, term)
                            return result, "SUBTRACT_FROM_BOTH_SIDES", f"Subtract {term} from both sides"
                
                if isinstance(expr.lhs, sp.Add):
                    for term in expr.lhs.args:
                        if var not in term.free_symbols:
                            result = self.transformer.subtract_from_both_sides(expr, term)
                            return result, "SUBTRACT_FROM_BOTH_SIDES", f"Subtract {term} from both sides"
        
        return None, None, "No fallback step available"
    
    def _is_solved(self, expr: Union[sp.Basic, sp.Eq, str]) -> bool:
        """Check if expression is solved.
        
        Args:
            expr: Expression or equation to check
            
        Returns:
            True if solved
        """
        # Convert string to SymPy if needed
        if isinstance(expr, str):
            try:
                if '=' in expr:
                    # Handle equation
                    left_str, right_str = expr.split('=', 1)
                    left_expr = sp.parse_expr(left_str.strip())
                    right_expr = sp.parse_expr(right_str.strip())
                    expr = sp.Eq(left_expr, right_expr)
                else:
                    expr = sp.parse_expr(expr)
            except:
                return False
        
        if isinstance(expr, sp.Eq):
            variables = list(expr.free_symbols)
            if len(variables) == 1:
                var = variables[0]
                # Check if it's in the form var = value
                if expr.lhs == var and var not in expr.rhs.free_symbols:
                    return True
                if expr.rhs == var and var not in expr.lhs.free_symbols:
                    return True
            elif len(variables) == 0:
                # No variables, check if it's a true statement
                return expr.lhs == expr.rhs
        else:
            # For expressions, check if fully simplified
            try:
                simplified = sp.simplify(expr)
                return simplified == expr and not any(
                    isinstance(expr, (sp.Add, sp.Mul)) for expr in sp.preorder_traversal(expr)
                )
            except:
                return False
        
        return False
    
    def _parameter_to_term(self, expr: sp.Eq, parameter: float) -> sp.Basic:
        """Convert parameter to appropriate term for the equation."""
        # Simple heuristic: use parameter as coefficient
        variables = list(expr.free_symbols)
        if variables:
            return parameter * variables[0]
        else:
            return sp.sympify(parameter)
    
    def _parameter_to_factor(self, expr: sp.Eq, parameter: float) -> sp.Basic:
        """Convert parameter to appropriate factor."""
        if abs(parameter) < 0.1:
            return 1
        return int(round(parameter))
    
    def _find_term_to_move(self, expr: sp.Eq) -> Optional[sp.Basic]:
        """Find a term that should be moved to the other side."""
        variables = list(expr.free_symbols)
        if not variables:
            return None
        
        var = variables[0]
        
        # Check RHS for constant terms
        if isinstance(expr.rhs, sp.Add):
            for term in expr.rhs.args:
                if var not in term.free_symbols:
                    return term
        elif var not in expr.rhs.free_symbols and expr.rhs != 0:
            return expr.rhs
        
        # Check LHS for constant terms
        if isinstance(expr.lhs, sp.Add):
            for term in expr.lhs.args:
                if var not in term.free_symbols:
                    return term
        
        return None
    
    def _find_coefficient(self, expr: sp.Eq) -> Optional[sp.Basic]:
        """Find the coefficient of the variable to divide by."""
        variables = list(expr.free_symbols)
        if not variables:
            return None
        
        var = variables[0]
        
        # Check if LHS is just coefficient * variable
        if expr.lhs.is_Mul:
            coeffs = [arg for arg in expr.lhs.args if var not in arg.free_symbols]
            if coeffs:
                return sp.Mul(*coeffs)
        
        # Check if LHS has a common factor
        if isinstance(expr.lhs, sp.Add):
            # Try to factor out common coefficient
            factored = sp.factor(expr.lhs)
            if factored.is_Mul:
                for arg in factored.args:
                    if var not in arg.free_symbols:
                        return arg
        
        return None


class InteractiveSolver:
    """Interactive solver that can provide hints and check student work."""
    
    def __init__(self, solver: GNNAlgebraSolver):
        self.solver = solver
        self.verifier = AlgebraicVerifier()
    
    def get_hint(self,
                 current_expr: Union[str, sp.Basic, sp.Eq]) -> str:
        """Get a hint for the next step.
        
        Args:
            current_expr: Current expression or equation
            
        Returns:
            Hint text
        """
        try:
            next_expr, operation, description = self.solver._predict_next_step(current_expr)
            
            # Generate hint based on operation
            if "add" in operation.lower():
                return f"Try adding the same value to both sides of the equation."
            elif "subtract" in operation.lower():
                return f"Try subtracting a term from both sides to isolate the variable."
            elif "multiply" in operation.lower():
                return f"Try multiplying both sides by a value."
            elif "divide" in operation.lower():
                return f"Try dividing both sides by the coefficient of the variable."
            elif "expand" in operation.lower():
                return f"Try expanding the expression."
            elif "factor" in operation.lower():
                return f"Try factoring the expression."
            elif "simplify" in operation.lower():
                return f"Try simplifying the expression."
            else:
                return f"Consider: {description}"
                
        except Exception as e:
            logger.error(f"Error generating hint: {e}")
            return "Try isolating the variable by moving terms to the other side."
    
    def check_step(self,
                  current_expr: Union[str, sp.Basic, sp.Eq],
                  student_next: Union[str, sp.Basic, sp.Eq]) -> Tuple[bool, str]:
        """Check if a student's next step is valid.
        
        Args:
            current_expr: Current expression or equation
            student_next: Student's proposed next step
            
        Returns:
            Tuple of (is_valid, feedback)
        """
        # Verify the step
        is_valid, explanation = self.verifier.verify_step_validity(
            current_expr, student_next
        )
        
        if is_valid:
            # Check if it's a good step
            if self.solver._is_solved(student_next):
                return True, "Excellent! You've found the solution."
            else:
                return True, "Good! That's a valid step."
        else:
            # Provide helpful feedback
            hint = self.get_hint(current_expr)
            return False, f"That step doesn't look quite right. {hint}"


if __name__ == "__main__":
    # Test the solver
    solver = GNNAlgebraSolver(use_verification=True)
    
    # Test linear equation
    print("Test 1: Linear Equation")
    problem1 = "2*x + 5 = 9"
    steps1 = solver.solve(problem1, show_steps=True)
    
    print("\n" + "="*50 + "\n")
    
    # Test quadratic equation
    print("Test 2: Quadratic Equation")
    problem2 = "x**2 - 5*x + 6 = 0"
    steps2 = solver.solve(problem2, show_steps=True)
    
    print("\n" + "="*50 + "\n")
    
    # Test expression simplification
    print("Test 3: Expression Simplification")
    problem3 = "2*x + 3*x - 5 + 7"
    steps3 = solver.solve(problem3, show_steps=True)
    
    # Test interactive features
    print("\n" + "="*50 + "\n")
    print("Test 4: Interactive Solver")
    interactive = InteractiveSolver(solver)
    
    current = "3*x - 4 = 5"
    print(f"Problem: {current}")
    hint = interactive.get_hint(current)
    print(f"Hint: {hint}")
    
    # Check student step
    student_step = "3*x = 9"
    valid, feedback = interactive.check_step(current, student_step)
    print(f"Student step: {student_step}")
    print(f"Feedback: {feedback}") 