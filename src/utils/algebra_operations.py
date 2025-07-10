"""Algebraic operations and transformations for step-by-step solving.

This module provides functions to perform valid algebraic transformations
that mimic human problem-solving steps.
"""

from typing import List, Tuple, Optional, Union, Dict, Any
import sympy as sp
from sympy import Symbol, Eq, Add, Mul, Pow, simplify, expand, factor, solve
from enum import Enum, auto
import random


class AlgebraicOperation(Enum):
    """Types of algebraic operations."""
    ADD_TO_BOTH_SIDES = auto()
    SUBTRACT_FROM_BOTH_SIDES = auto()
    MULTIPLY_BOTH_SIDES = auto()
    DIVIDE_BOTH_SIDES = auto()
    EXPAND = auto()
    FACTOR = auto()
    COMBINE_LIKE_TERMS = auto()
    DISTRIBUTE = auto()
    ISOLATE_VARIABLE = auto()
    SIMPLIFY = auto()
    APPLY_QUADRATIC_FORMULA = auto()
    COMPLETE_THE_SQUARE = auto()
    
    def __str__(self):
        return self.name.replace('_', ' ').title()


class AlgebraicStep:
    """Represents a single step in solving an algebraic problem."""
    
    def __init__(self, 
                 expression: Union[sp.Basic, sp.Eq],
                 operation: Optional[AlgebraicOperation] = None,
                 description: Optional[str] = None):
        self.expression = expression
        self.operation = operation
        self.description = description or (str(operation) if operation else "")
    
    def __repr__(self):
        return f"Step({self.expression}, {self.operation})"
    
    def __str__(self):
        if self.description:
            return f"{self.expression} ({self.description})"
        return str(self.expression)


class AlgebraicTransformer:
    """Performs step-by-step algebraic transformations."""
    
    @staticmethod
    def add_to_both_sides(equation: sp.Eq, term: sp.Basic) -> sp.Eq:
        """Add a term to both sides of an equation."""
        return Eq(equation.lhs + term, equation.rhs + term)
    
    @staticmethod
    def subtract_from_both_sides(equation: sp.Eq, term: sp.Basic) -> sp.Eq:
        """Subtract a term from both sides of an equation."""
        return Eq(equation.lhs - term, equation.rhs - term)
    
    @staticmethod
    def multiply_both_sides(equation: sp.Eq, factor: sp.Basic) -> sp.Eq:
        """Multiply both sides of an equation by a factor."""
        if factor == 0:
            raise ValueError("Cannot multiply by zero")
        return Eq(equation.lhs * factor, equation.rhs * factor)
    
    @staticmethod
    def divide_both_sides(equation: sp.Eq, divisor: sp.Basic) -> sp.Eq:
        """Divide both sides of an equation by a divisor."""
        if divisor == 0:
            raise ValueError("Cannot divide by zero")
        return Eq(equation.lhs / divisor, equation.rhs / divisor)
    
    @staticmethod
    def expand_expression(expr: sp.Basic) -> sp.Basic:
        """Expand an expression."""
        return expand(expr)
    
    @staticmethod
    def factor_expression(expr: sp.Basic) -> sp.Basic:
        """Factor an expression."""
        return factor(expr)
    
    @staticmethod
    def simplify_expression(expr: sp.Basic) -> sp.Basic:
        """Simplify an expression."""
        return simplify(expr)
    
    @staticmethod
    def combine_like_terms(expr: sp.Basic) -> sp.Basic:
        """Combine like terms in an expression."""
        # Expand first to separate terms, then collect
        expanded = expand(expr)
        if isinstance(expanded, Add):
            # Group by variable powers
            return sp.collect(expanded, list(expr.free_symbols))
        return expanded


class StepByStepSolver:
    """Generates step-by-step solutions for algebraic problems."""
    
    def __init__(self):
        self.transformer = AlgebraicTransformer()
    
    def solve_linear_equation(self, equation: Union[str, sp.Eq], 
                            variable: Optional[Symbol] = None) -> List[AlgebraicStep]:
        """Solve a linear equation step by step.
        
        Args:
            equation: The equation to solve (string or SymPy equation)
            variable: The variable to solve for (auto-detected if None)
            
        Returns:
            List of algebraic steps
        """
        # Parse equation if string
        if isinstance(equation, str):
            equation = sp.parse_expr(equation, transformations='all')
            if not isinstance(equation, Eq):
                # Assume equation equals zero
                equation = Eq(equation, 0)
        
        # Get the variable
        if variable is None:
            variables = list(equation.free_symbols)
            if len(variables) != 1:
                raise ValueError(f"Expected exactly one variable, found {variables}")
            variable = variables[0]
        
        steps = [AlgebraicStep(equation, None, "Original equation")]
        current_eq = equation
        
        # Expand if needed
        if any(isinstance(arg, (Mul, Pow)) for arg in current_eq.lhs.args if isinstance(current_eq.lhs, Add)):
            expanded_lhs = self.transformer.expand_expression(current_eq.lhs)
            expanded_rhs = self.transformer.expand_expression(current_eq.rhs)
            if expanded_lhs != current_eq.lhs or expanded_rhs != current_eq.rhs:
                current_eq = Eq(expanded_lhs, expanded_rhs)
                steps.append(AlgebraicStep(current_eq, AlgebraicOperation.EXPAND))
        
        # Move all terms with variable to left side
        var_terms_rhs = []
        const_terms_rhs = []
        
        if isinstance(current_eq.rhs, Add):
            for term in current_eq.rhs.args:
                if variable in term.free_symbols:
                    var_terms_rhs.append(term)
                else:
                    const_terms_rhs.append(term)
        else:
            if variable in current_eq.rhs.free_symbols:
                var_terms_rhs.append(current_eq.rhs)
            else:
                const_terms_rhs.append(current_eq.rhs)
        
        # Subtract variable terms from right side
        for term in var_terms_rhs:
            current_eq = self.transformer.subtract_from_both_sides(current_eq, term)
            steps.append(AlgebraicStep(
                current_eq, 
                AlgebraicOperation.SUBTRACT_FROM_BOTH_SIDES,
                f"Subtract {term} from both sides"
            ))
        
        # Move constant terms to right side
        const_terms_lhs = []
        var_terms_lhs = []
        
        if isinstance(current_eq.lhs, Add):
            for term in current_eq.lhs.args:
                if variable in term.free_symbols:
                    var_terms_lhs.append(term)
                else:
                    const_terms_lhs.append(term)
        else:
            if variable in current_eq.lhs.free_symbols:
                var_terms_lhs.append(current_eq.lhs)
            else:
                const_terms_lhs.append(current_eq.lhs)
        
        # Subtract constant terms from left side
        for term in const_terms_lhs:
            current_eq = self.transformer.subtract_from_both_sides(current_eq, term)
            steps.append(AlgebraicStep(
                current_eq,
                AlgebraicOperation.SUBTRACT_FROM_BOTH_SIDES,
                f"Subtract {term} from both sides"
            ))
        
        # Simplify both sides
        current_eq = Eq(
            self.transformer.simplify_expression(current_eq.lhs),
            self.transformer.simplify_expression(current_eq.rhs)
        )
        steps.append(AlgebraicStep(current_eq, AlgebraicOperation.SIMPLIFY))
        
        # Isolate variable
        coefficient = current_eq.lhs.as_coefficient(variable)
        if coefficient and coefficient != 1:
            current_eq = self.transformer.divide_both_sides(current_eq, coefficient)
            steps.append(AlgebraicStep(
                current_eq,
                AlgebraicOperation.DIVIDE_BOTH_SIDES,
                f"Divide both sides by {coefficient}"
            ))
        
        # Final simplification
        current_eq = Eq(
            self.transformer.simplify_expression(current_eq.lhs),
            self.transformer.simplify_expression(current_eq.rhs)
        )
        if current_eq != steps[-1].expression:
            steps.append(AlgebraicStep(
                current_eq,
                AlgebraicOperation.SIMPLIFY,
                "Solution"
            ))
        
        return steps
    
    def solve_quadratic_equation(self, equation: Union[str, sp.Eq],
                               variable: Optional[Symbol] = None,
                               method: str = "auto") -> List[AlgebraicStep]:
        """Solve a quadratic equation step by step.
        
        Args:
            equation: The equation to solve
            variable: The variable to solve for
            method: Solving method ('factor', 'quadratic_formula', 'complete_square', 'auto')
            
        Returns:
            List of algebraic steps
        """
        # Parse equation if string
        if isinstance(equation, str):
            equation = sp.parse_expr(equation, transformations='all')
            if not isinstance(equation, Eq):
                equation = Eq(equation, 0)
        
        # Get the variable
        if variable is None:
            variables = list(equation.free_symbols)
            if len(variables) != 1:
                raise ValueError(f"Expected exactly one variable, found {variables}")
            variable = variables[0]
        
        steps = [AlgebraicStep(equation, None, "Original equation")]
        current_eq = equation
        
        # Move all terms to left side
        if current_eq.rhs != 0:
            current_eq = Eq(current_eq.lhs - current_eq.rhs, 0)
            steps.append(AlgebraicStep(
                current_eq,
                AlgebraicOperation.SUBTRACT_FROM_BOTH_SIDES,
                "Move all terms to left side"
            ))
        
        # Expand and simplify
        current_eq = Eq(
            self.transformer.expand_expression(current_eq.lhs),
            current_eq.rhs
        )
        if current_eq != steps[-1].expression:
            steps.append(AlgebraicStep(current_eq, AlgebraicOperation.EXPAND))
        
        # Collect terms
        current_eq = Eq(
            sp.collect(current_eq.lhs, variable),
            current_eq.rhs
        )
        if current_eq != steps[-1].expression:
            steps.append(AlgebraicStep(
                current_eq,
                AlgebraicOperation.COMBINE_LIKE_TERMS
            ))
        
        # Extract coefficients
        poly = sp.Poly(current_eq.lhs, variable)
        coeffs = poly.all_coeffs()
        
        if len(coeffs) == 3:  # ax^2 + bx + c
            a, b, c = coeffs
        elif len(coeffs) == 2:  # ax^2 + bx or ax^2 + c
            if poly.degree() == 2:
                a, b = coeffs
                c = 0
            else:
                a = 0
                b, c = coeffs
        else:
            # Not a quadratic
            return self.solve_linear_equation(equation, variable)
        
        # Choose solving method
        if method == "auto":
            # Try factoring first
            factored = factor(current_eq.lhs)
            if isinstance(factored, Mul):
                method = "factor"
            else:
                method = "quadratic_formula"
        
        if method == "factor":
            factored = factor(current_eq.lhs)
            if isinstance(factored, Mul):
                current_eq = Eq(factored, 0)
                steps.append(AlgebraicStep(
                    current_eq,
                    AlgebraicOperation.FACTOR
                ))
                
                # Solve each factor
                solutions = []
                for factor_expr in factored.args:
                    if variable in factor_expr.free_symbols:
                        factor_eq = Eq(factor_expr, 0)
                        steps.append(AlgebraicStep(
                            factor_eq,
                            None,
                            f"Set factor {factor_expr} = 0"
                        ))
                        # Solve the factor equation
                        sol = solve(factor_eq, variable)
                        if sol:
                            solutions.extend(sol)
                            steps.append(AlgebraicStep(
                                Eq(variable, sol[0]),
                                AlgebraicOperation.ISOLATE_VARIABLE,
                                f"Solution from {factor_expr} = 0"
                            ))
            else:
                method = "quadratic_formula"  # Fallback
        
        if method == "quadratic_formula":
            # Apply quadratic formula
            discriminant = b**2 - 4*a*c
            steps.append(AlgebraicStep(
                sp.parse_expr(f"discriminant = {discriminant}"),
                None,
                f"Calculate discriminant: b² - 4ac = {b}² - 4({a})({c})"
            ))
            
            if discriminant < 0:
                steps.append(AlgebraicStep(
                    sp.parse_expr("No real solutions"),
                    None,
                    "Discriminant < 0, no real solutions"
                ))
            else:
                sqrt_disc = sp.sqrt(discriminant)
                sol1 = (-b + sqrt_disc) / (2*a)
                sol2 = (-b - sqrt_disc) / (2*a)
                
                steps.append(AlgebraicStep(
                    Eq(variable, (-b + sp.sqrt(discriminant))/(2*a)),
                    AlgebraicOperation.APPLY_QUADRATIC_FORMULA,
                    "Apply quadratic formula: x = (-b ± √discriminant) / 2a"
                ))
                
                steps.append(AlgebraicStep(
                    Eq(variable, simplify(sol1)),
                    AlgebraicOperation.SIMPLIFY,
                    f"First solution: x = {simplify(sol1)}"
                ))
                
                if sol1 != sol2:
                    steps.append(AlgebraicStep(
                        Eq(variable, simplify(sol2)),
                        AlgebraicOperation.SIMPLIFY,
                        f"Second solution: x = {simplify(sol2)}"
                    ))
        
        return steps
    
    def simplify_expression_steps(self, expression: Union[str, sp.Basic]) -> List[AlgebraicStep]:
        """Simplify an expression step by step.
        
        Args:
            expression: The expression to simplify
            
        Returns:
            List of algebraic steps
        """
        # Parse expression if string
        if isinstance(expression, str):
            expression = sp.parse_expr(expression, transformations='all')
        
        steps = [AlgebraicStep(expression, None, "Original expression")]
        current = expression
        
        # Expand if needed
        expanded = self.transformer.expand_expression(current)
        if expanded != current:
            current = expanded
            steps.append(AlgebraicStep(current, AlgebraicOperation.EXPAND))
        
        # Combine like terms
        if isinstance(current, Add):
            combined = self.transformer.combine_like_terms(current)
            if combined != current:
                current = combined
                steps.append(AlgebraicStep(
                    current,
                    AlgebraicOperation.COMBINE_LIKE_TERMS
                ))
        
        # Try factoring
        factored = self.transformer.factor_expression(current)
        if factored != current and len(str(factored)) < len(str(current)):
            current = factored
            steps.append(AlgebraicStep(current, AlgebraicOperation.FACTOR))
        
        # Final simplification
        simplified = self.transformer.simplify_expression(current)
        if simplified != current:
            current = simplified
            steps.append(AlgebraicStep(
                current,
                AlgebraicOperation.SIMPLIFY,
                "Final simplified form"
            ))
        
        return steps


def generate_random_linear_equation(variable: str = 'x',
                                  coeff_range: Tuple[int, int] = (-10, 10),
                                  const_range: Tuple[int, int] = (-20, 20)) -> sp.Eq:
    """Generate a random linear equation.
    
    Args:
        variable: Variable name
        coeff_range: Range for coefficients
        const_range: Range for constants
        
    Returns:
        SymPy equation
    """
    x = Symbol(variable)
    
    # Generate ax + b = cx + d
    a = random.randint(coeff_range[0], coeff_range[1])
    if a == 0:
        a = 1
    
    b = random.randint(const_range[0], const_range[1])
    c = random.randint(coeff_range[0], coeff_range[1])
    d = random.randint(const_range[0], const_range[1])
    
    # Ensure it's not trivial
    if a == c and b == d:
        d = b + random.randint(1, 5)
    
    return Eq(a*x + b, c*x + d)


def generate_random_quadratic_equation(variable: str = 'x',
                                     coeff_range: Tuple[int, int] = (-5, 5),
                                     root_range: Tuple[int, int] = (-10, 10)) -> sp.Eq:
    """Generate a random quadratic equation.
    
    Args:
        variable: Variable name
        coeff_range: Range for leading coefficient
        root_range: Range for roots
        
    Returns:
        SymPy equation
    """
    x = Symbol(variable)
    
    # Generate from roots: a(x - r1)(x - r2) = 0
    a = random.randint(coeff_range[0], coeff_range[1])
    if a == 0:
        a = 1
    
    r1 = random.randint(root_range[0], root_range[1])
    r2 = random.randint(root_range[0], root_range[1])
    
    # Expand to standard form
    expr = a * (x - r1) * (x - r2)
    expanded = expand(expr)
    
    return Eq(expanded, 0)


if __name__ == "__main__":
    # Test the solver
    solver = StepByStepSolver()
    
    # Test linear equation
    print("Linear Equation Example:")
    eq1 = "2*x + 5 = 9"
    steps1 = solver.solve_linear_equation(eq1)
    for i, step in enumerate(steps1):
        print(f"Step {i}: {step}")
    
    print("\nQuadratic Equation Example:")
    eq2 = "x**2 - 5*x + 6 = 0"
    steps2 = solver.solve_quadratic_equation(eq2)
    for i, step in enumerate(steps2):
        print(f"Step {i}: {step}")
    
    print("\nExpression Simplification Example:")
    expr3 = "2*x + 3*x - 5 + 7"
    steps3 = solver.simplify_expression_steps(expr3)
    for i, step in enumerate(steps3):
        print(f"Step {i}: {step}") 