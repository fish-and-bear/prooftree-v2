import pytest
from src.data.dataset_generator import AlgebraDatasetGenerator, AlgebraProblem

def test_generate_linear_equation_problems():
    gen = AlgebraDatasetGenerator(seed=42)
    problems = gen.generate_linear_equation_problems(5)
    assert len(problems) == 5
    for p in problems:
        assert isinstance(p, AlgebraProblem)
        assert p.problem_type == 'linear_equation'
        assert len(p.steps) > 0
        assert p.final_answer is not None

def test_generate_quadratic_equation_problems():
    gen = AlgebraDatasetGenerator(seed=42)
    problems = gen.generate_quadratic_equation_problems(3)
    assert len(problems) == 3
    for p in problems:
        assert isinstance(p, AlgebraProblem)
        assert p.problem_type == 'quadratic_equation'
        assert len(p.steps) > 0
        assert p.final_answer is not None

def test_generate_simplification_problems():
    gen = AlgebraDatasetGenerator(seed=42)
    problems = gen.generate_simplification_problems(2)
    assert len(problems) == 2
    for p in problems:
        assert isinstance(p, AlgebraProblem)
        assert p.problem_type == 'simplification'
        assert len(p.steps) > 0
        assert p.final_answer is not None 