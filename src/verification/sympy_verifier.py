"""SymPy-based verification for algebraic transformations.

This module provides functionality to verify that algebraic transformations
are mathematically valid using SymPy's symbolic capabilities.
"""

import sympy as sp
from sympy import Symbol, Eq, simplify, expand, solve
from typing import Union, Tuple, List, Optional, Dict, Any
import logging


logger = logging.getLogger(__name__)


class AlgebraicVerifier:
    """Verifies the correctness of algebraic transformations."""
    
    def __init__(self):
        self.tolerance = 1e-10  # For numerical comparisons
    
    def verify_equation_transformation(self,
                                     eq1: Union[str, sp.Eq],
                                     eq2: Union[str, sp.Eq]) -> Tuple[bool, str]:
        """Verify that two equations are equivalent transformations.
        
        Args:
            eq1: First equation (original)
            eq2: Second equation (transformed)
            
        Returns:
            Tuple of (is_valid, explanation)
        """
        try:
            # Parse strings to SymPy equations
            if isinstance(eq1, str):
                eq1 = sp.parse_expr(eq1, transformations='all')
                if not isinstance(eq1, Eq):
                    eq1 = Eq(eq1, 0)
            
            if isinstance(eq2, str):
                eq2 = sp.parse_expr(eq2, transformations='all')
                if not isinstance(eq2, Eq):
                    eq2 = Eq(eq2, 0)
            
            # Get variables
            vars1 = list(eq1.free_symbols)
            vars2 = list(eq2.free_symbols)
            
            # Check if variables match
            if set(vars1) != set(vars2):
                return False, f"Variable mismatch: {vars1} vs {vars2}"
            
            # If no variables, check direct equality
            if not vars1:
                is_equal = simplify(eq1.lhs - eq1.rhs) == simplify(eq2.lhs - eq2.rhs)
                return is_equal, "Direct numerical equality check"
            
            # Solve both equations
            try:
                solutions1 = solve(eq1, vars1)
                solutions2 = solve(eq2, vars2)
                
                # Convert to sets for comparison
                if isinstance(solutions1, dict):
                    solutions1 = [solutions1]
                if isinstance(solutions2, dict):
                    solutions2 = [solutions2]
                
                # Check if solution sets are equal
                if self._solution_sets_equal(solutions1, solutions2, vars1[0]):
                    return True, "Solution sets are equal"
                else:
                    return False, f"Different solution sets: {solutions1} vs {solutions2}"
                    
            except Exception as e:
                # Fallback: check if eq1 - eq2 simplifies to 0
                diff = simplify(
                    (eq1.lhs - eq1.rhs) - (eq2.lhs - eq2.rhs)
                )
                if diff == 0:
                    return True, "Equations are algebraically equivalent"
                else:
                    return False, f"Equations differ by: {diff}"
        
        except Exception as e:
            logger.error(f"Error verifying transformation: {e}")
            return False, f"Verification error: {str(e)}"
    
    def verify_expression_equality(self,
                                 expr1: Union[str, sp.Basic],
                                 expr2: Union[str, sp.Basic]) -> Tuple[bool, str]:
        """Verify that two expressions are equivalent.
        
        Args:
            expr1: First expression
            expr2: Second expression
            
        Returns:
            Tuple of (is_equal, explanation)
        """
        try:
            # Parse strings to SymPy expressions
            if isinstance(expr1, str):
                expr1 = sp.parse_expr(expr1, transformations='all')
            if isinstance(expr2, str):
                expr2 = sp.parse_expr(expr2, transformations='all')
            
            # Simplify difference
            diff = simplify(expr1 - expr2)
            
            if diff == 0:
                return True, "Expressions are equal"
            else:
                # Try expanding both and comparing
                expanded1 = expand(expr1)
                expanded2 = expand(expr2)
                if simplify(expanded1 - expanded2) == 0:
                    return True, "Expressions are equal after expansion"
                else:
                    return False, f"Expressions differ by: {diff}"
        
        except Exception as e:
            logger.error(f"Error verifying expression equality: {e}")
            return False, f"Verification error: {str(e)}"
    
    def verify_step_validity(self,
                           current: Union[str, sp.Basic, sp.Eq],
                           next_step: Union[str, sp.Basic, sp.Eq],
                           operation: Optional[str] = None) -> Tuple[bool, str]:
        """Verify that a step transformation is valid.
        
        Args:
            current: Current expression/equation
            next_step: Next expression/equation
            operation: Optional operation type for context
            
        Returns:
            Tuple of (is_valid, explanation)
        """
        try:
            # Parse current expression
            if isinstance(current, str):
                if '=' in current:
                    # Handle equation
                    left_str, right_str = current.split('=', 1)
                    left_expr = sp.parse_expr(left_str.strip())
                    right_expr = sp.parse_expr(right_str.strip())
                    current_parsed = sp.Eq(left_expr, right_expr)
                else:
                    current_parsed = sp.parse_expr(current)
            else:
                current_parsed = current
            
            # Parse next step
            if isinstance(next_step, str):
                if '=' in next_step:
                    # Handle equation
                    left_str, right_str = next_step.split('=', 1)
                    left_expr = sp.parse_expr(left_str.strip())
                    right_expr = sp.parse_expr(right_str.strip())
                    next_parsed = sp.Eq(left_expr, right_expr)
                else:
                    next_parsed = sp.parse_expr(next_step)
            else:
                next_parsed = next_step
            
            # Check if both are equations
            if isinstance(current_parsed, sp.Eq) and isinstance(next_parsed, sp.Eq):
                return self.verify_equation_transformation(current_parsed, next_parsed)
            
            # Check if both are expressions
            elif not isinstance(current_parsed, sp.Eq) and not isinstance(next_parsed, sp.Eq):
                return self.verify_expression_equality(current_parsed, next_parsed)
            
            else:
                return False, "Type mismatch: cannot compare equation with expression"
                
        except Exception as e:
            logger.error(f"Error in step validity verification: {e}")
            return False, f"Parsing error: {str(e)}"
    
    def check_solution(self,
                      equation: Union[str, sp.Eq],
                      solution: Union[str, sp.Basic, Dict[sp.Symbol, sp.Basic]]) -> Tuple[bool, str]:
        """Check if a solution satisfies an equation.
        
        Args:
            equation: The equation to check
            solution: The proposed solution (can be value or dict of var: value)
            
        Returns:
            Tuple of (is_correct, explanation)
        """
        try:
            print(f"DEBUG: Input equation: {equation}, type: {type(equation)}")
            print(f"DEBUG: Input solution: {solution}, type: {type(solution)}")
            
            # Parse equation
            if isinstance(equation, str):
                if '=' in equation:
                    # Handle equation
                    left_str, right_str = equation.split('=', 1)
                    left_expr = sp.parse_expr(left_str.strip())
                    right_expr = sp.parse_expr(right_str.strip())
                    equation = sp.Eq(left_expr, right_expr)
                else:
                    equation = sp.parse_expr(equation)
                    if not isinstance(equation, sp.Eq):
                        equation = sp.Eq(equation, 0)
            
            print(f"DEBUG: Parsed equation: {equation}, type: {type(equation)}")
            
            # Parse solution
            if isinstance(solution, str):
                # Try to parse as equation (e.g., "x = 2")
                if '=' in solution:
                    left_str, right_str = solution.split('=', 1)
                    left_expr = sp.parse_expr(left_str.strip())
                    right_expr = sp.parse_expr(right_str.strip())
                    solution = {left_expr: right_expr}
                else:
                    # Assume it's just a value for the first variable
                    vars_list = list(equation.free_symbols)
                    if vars_list:
                        solution = {vars_list[0]: sp.parse_expr(solution)}
                    else:
                        return False, "No variables in equation"
            
            elif not isinstance(solution, dict):
                # Convert single value to dict
                vars_list = list(equation.free_symbols)
                if vars_list:
                    solution = {vars_list[0]: solution}
                else:
                    return False, "No variables in equation"
            
            print(f"DEBUG: Parsed solution: {solution}, type: {type(solution)}")
            
            # Substitute and check
            substituted = equation.subs(solution)
            print(f"DEBUG: Substituted result: {substituted}, type: {type(substituted)}")
            
            # Handle boolean result (when substitution yields True/False)
            if isinstance(substituted, bool) or str(substituted) in ['True', 'False']:
                if substituted == True or str(substituted) == 'True':
                    return True, f"Solution {solution} satisfies the equation"
                else:
                    return False, f"Solution {solution} does not satisfy the equation"
            
            # Simplify both sides
            elif isinstance(substituted, sp.Eq):
                lhs_val = simplify(substituted.lhs)
                rhs_val = simplify(substituted.rhs)
                
                # Check equality
                if lhs_val == rhs_val:
                    return True, f"Solution {solution} satisfies the equation"
                else:
                    return False, f"LHS = {lhs_val}, RHS = {rhs_val}"
            else:
                # Should not happen
                return False, "Substitution did not yield an equation"
        
        except Exception as e:
            logger.error(f"Error checking solution: {e}")
            return False, f"Verification error: {str(e)}"
    
    def _solution_sets_equal(self,
                           sols1: List[Any],
                           sols2: List[Any],
                           var: sp.Symbol) -> bool:
        """Check if two solution sets are equal.
        
        Args:
            sols1: First solution set
            sols2: Second solution set
            var: Variable being solved for
            
        Returns:
            True if solution sets are equal
        """
        # Handle empty solutions
        if not sols1 and not sols2:
            return True
        if len(sols1) != len(sols2):
            return False
        
        # Convert to comparable format
        def normalize_solution(sol):
            if isinstance(sol, dict):
                return sol.get(var, sol)
            return sol
        
        sols1_normalized = [simplify(normalize_solution(s)) for s in sols1]
        sols2_normalized = [simplify(normalize_solution(s)) for s in sols2]
        
        # Sort and compare
        try:
            sols1_normalized.sort(key=str)
            sols2_normalized.sort(key=str)
            
            for s1, s2 in zip(sols1_normalized, sols2_normalized):
                if simplify(s1 - s2) != 0:
                    return False
            return True
        except:
            # If sorting fails, try set comparison
            return set(str(s) for s in sols1_normalized) == set(str(s) for s in sols2_normalized)


class StepValidator:
    """Validates individual algebraic steps based on operation type."""
    
    def __init__(self):
        self.verifier = AlgebraicVerifier()
    
    def validate_add_to_both_sides(self,
                                 eq1: sp.Eq,
                                 eq2: sp.Eq,
                                 term: Optional[sp.Basic] = None) -> Tuple[bool, str]:
        """Validate that eq2 is eq1 with term added to both sides."""
        if term is None:
            # Try to infer the term
            diff_lhs = simplify(eq2.lhs - eq1.lhs)
            diff_rhs = simplify(eq2.rhs - eq1.rhs)
            if diff_lhs == diff_rhs:
                term = diff_lhs
            else:
                return False, f"Inconsistent additions: {diff_lhs} vs {diff_rhs}"
        
        expected = Eq(eq1.lhs + term, eq1.rhs + term)
        is_valid, _ = self.verifier.verify_equation_transformation(expected, eq2)
        
        if is_valid:
            return True, f"Added {term} to both sides"
        else:
            return False, "Invalid addition to both sides"
    
    def validate_subtract_from_both_sides(self,
                                        eq1: sp.Eq,
                                        eq2: sp.Eq,
                                        term: Optional[sp.Basic] = None) -> Tuple[bool, str]:
        """Validate that eq2 is eq1 with term subtracted from both sides."""
        if term is None:
            # Try to infer the term
            diff_lhs = simplify(eq1.lhs - eq2.lhs)
            diff_rhs = simplify(eq1.rhs - eq2.rhs)
            if diff_lhs == diff_rhs:
                term = diff_lhs
            else:
                return False, f"Inconsistent subtractions: {diff_lhs} vs {diff_rhs}"
        
        expected = Eq(eq1.lhs - term, eq1.rhs - term)
        is_valid, _ = self.verifier.verify_equation_transformation(expected, eq2)
        
        if is_valid:
            return True, f"Subtracted {term} from both sides"
        else:
            return False, "Invalid subtraction from both sides"
    
    def validate_multiply_both_sides(self,
                                   eq1: sp.Eq,
                                   eq2: sp.Eq,
                                   factor: Optional[sp.Basic] = None) -> Tuple[bool, str]:
        """Validate that eq2 is eq1 with both sides multiplied by factor."""
        if factor is None:
            # Try to infer the factor
            if eq1.lhs != 0:
                factor = simplify(eq2.lhs / eq1.lhs)
                if simplify(eq2.rhs / eq1.rhs) != factor:
                    return False, "Inconsistent multiplication factors"
            else:
                return False, "Cannot infer factor when LHS is zero"
        
        if factor == 0:
            return False, "Cannot multiply by zero"
        
        expected = Eq(eq1.lhs * factor, eq1.rhs * factor)
        is_valid, _ = self.verifier.verify_equation_transformation(expected, eq2)
        
        if is_valid:
            return True, f"Multiplied both sides by {factor}"
        else:
            return False, "Invalid multiplication of both sides"
    
    def validate_divide_both_sides(self,
                                 eq1: sp.Eq,
                                 eq2: sp.Eq,
                                 divisor: Optional[sp.Basic] = None) -> Tuple[bool, str]:
        """Validate that eq2 is eq1 with both sides divided by divisor."""
        if divisor is None:
            # Try to infer the divisor
            if eq2.lhs != 0:
                divisor = simplify(eq1.lhs / eq2.lhs)
                if simplify(eq1.rhs / eq2.rhs) != divisor:
                    return False, "Inconsistent division factors"
            else:
                return False, "Cannot infer divisor when result LHS is zero"
        
        if divisor == 0:
            return False, "Cannot divide by zero"
        
        expected = Eq(eq1.lhs / divisor, eq1.rhs / divisor)
        is_valid, _ = self.verifier.verify_equation_transformation(expected, eq2)
        
        if is_valid:
            return True, f"Divided both sides by {divisor}"
        else:
            return False, "Invalid division of both sides"


if __name__ == "__main__":
    # Test the verifier
    verifier = AlgebraicVerifier()
    
    # Test equation transformations
    print("Testing equation transformations:")
    eq1 = "2*x + 5 = 9"
    eq2 = "2*x = 4"
    valid, explanation = verifier.verify_equation_transformation(eq1, eq2)
    print(f"{eq1} -> {eq2}: {valid} ({explanation})")
    
    # Test expression equality
    print("\nTesting expression equality:")
    expr1 = "x**2 + 2*x + 1"
    expr2 = "(x + 1)**2"
    equal, explanation = verifier.verify_expression_equality(expr1, expr2)
    print(f"{expr1} == {expr2}: {equal} ({explanation})")
    
    # Test solution checking
    print("\nTesting solution checking:")
    equation = "x**2 - 5*x + 6 = 0"
    solution1 = "x = 2"
    solution2 = "x = 3"
    solution3 = "x = 4"
    
    for sol in [solution1, solution2, solution3]:
        correct, explanation = verifier.check_solution(equation, sol)
        print(f"{equation}, {sol}: {correct} ({explanation})") 