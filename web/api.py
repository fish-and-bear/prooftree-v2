"""
Flask API Backend for Graph Neural Algebra Tutor

This API provides endpoints for step-by-step algebraic solving using the GNN solver.
It integrates with the existing GNNAlgebraSolver and provides robust error handling.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sympy as sp
from sympy import Symbol, Eq, solve
import logging
import traceback
from typing import List, Dict, Any, Optional, Union
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.solver import GNNAlgebraSolver, InteractiveSolver
from src.verification import AlgebraicVerifier
from src.utils import AlgebraicStep, AlgebraicOperation
from src.graph import expression_to_graph, ExpressionGraph

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Global solver instance (loaded once)
_solver = None
_interactive_solver = None
_verifier = None

def get_solver():
    """Get or create the GNN solver instance."""
    global _solver, _interactive_solver, _verifier
    
    if _solver is None:
        try:
            logger.info("Initializing GNN solver...")
            _solver = GNNAlgebraSolver(use_verification=True, max_steps=50)
            _interactive_solver = InteractiveSolver(_solver)
            _verifier = AlgebraicVerifier()
            logger.info("GNN solver initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize solver: {e}")
            raise
    
    return _solver, _interactive_solver, _verifier

@app.route('/api/solve', methods=['POST'])
def solve_step_by_step():
    """
    Solve an algebraic expression or equation step by step.
    
    Expected JSON payload:
    {
        "expression": "2*x + 5 = 9",
        "show_verification": true,
        "max_steps": 20
    }
    
    Returns:
    {
        "success": true,
        "steps": [
            {
                "expression": "2*x + 5 = 9",
                "operation": "ORIGINAL",
                "explanation": "Original problem",
                "confidence": 1.0,
                "timestamp": 1234567890,
                "strategy": "gnn",
                "fallbackUsed": false
            },
            ...
        ],
        "finalStrategy": "gnn",
        "limitations": []
    }
    """
    try:
        # Parse request
        data = request.get_json()
        if not data or 'expression' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing expression in request body'
            }), 400
        
        expression = data['expression'].strip()
        show_verification = data.get('show_verification', True)
        max_steps = data.get('max_steps', 20)
        
        if not expression:
            return jsonify({
                'success': False,
                'error': 'Expression cannot be empty'
            }), 400
        
        logger.info(f"Solving expression: {expression}")
        
        # Get solver
        solver, interactive_solver, verifier = get_solver()
        
        # Parse and validate expression
        try:
            # Handle equations with '='
            if '=' in expression:
                parts = expression.split('=')
                if len(parts) != 2:
                    return jsonify({
                        'success': False,
                        'error': 'Invalid equation format. Use exactly one equals sign.'
                    }), 400
                
                # Clean up the expression parts
                lhs_str = parts[0].strip()
                rhs_str = parts[1].strip()
                
                # Handle different power notations
                lhs_str = lhs_str.replace('^', '**')
                rhs_str = rhs_str.replace('^', '**')
                
                try:
                    lhs = sp.parse_expr(lhs_str)
                    rhs = sp.parse_expr(rhs_str)
                    problem = Eq(lhs, rhs)
                except Exception as parse_error:
                    # Try alternative parsing
                    try:
                        # Handle implicit multiplication (e.g., 2x -> 2*x)
                        lhs_str = lhs_str.replace('x', '*x').replace('**x', '^x')
                        rhs_str = rhs_str.replace('x', '*x').replace('**x', '^x')
                        lhs = sp.parse_expr(lhs_str)
                        rhs = sp.parse_expr(rhs_str)
                        problem = Eq(lhs, rhs)
                    except:
                        return jsonify({
                            'success': False,
                            'error': f'Failed to parse equation: {str(parse_error)}. Try using explicit operators (e.g., 2*x instead of 2x).'
                        }), 400
            else:
                # Parse as expression
                expr_str = expression.replace('^', '**')
                try:
                    problem = sp.parse_expr(expr_str)
                except Exception as parse_error:
                    # Try alternative parsing
                    try:
                        expr_str = expr_str.replace('x', '*x').replace('**x', '^x')
                        problem = sp.parse_expr(expr_str)
                    except:
                        return jsonify({
                            'success': False,
                            'error': f'Failed to parse expression: {str(parse_error)}. Try using explicit operators (e.g., 2*x instead of 2x).'
                        }), 400
                
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Failed to parse expression: {str(e)}'
            }), 400
        
        # Solve step by step
        try:
            steps = solver.solve(
                problem=problem,
                show_steps=False,
                return_all_steps=True,
                use_multiple_strategies=True
            )
            
            if not steps:
                return jsonify({
                    'success': False,
                    'error': 'No solution steps generated'
                }), 500
            
            # Convert steps to API format
            api_steps = []
            limitations = []
            final_strategy = "gnn"
            
            for i, step in enumerate(steps):
                # Determine confidence based on step type
                confidence = 0.9  # Default high confidence for GNN steps
                strategy = "gnn"
                fallback_used = False
                
                # Adjust confidence based on operation type
                if step.operation:
                    if step.operation in ['SIMPLIFY', 'EXPAND', 'COMBINE_LIKE_TERMS']:
                        confidence = 0.95
                    elif step.operation in ['SOLVE', 'SOLVE_QUADRATIC', 'SOLVE_LINEAR']:
                        confidence = 0.85
                    elif step.operation in ['MOVE_TERMS', 'ISOLATE_VARIABLE']:
                        confidence = 0.9
                    elif step.operation == 'SOLUTION_FOUND':
                        confidence = 1.0
                        strategy = "solution_detection"
                
                # Check if this was a fallback step
                if hasattr(step, 'fallback_used') and step.fallback_used:
                    fallback_used = True
                    strategy = "fallback"
                    confidence *= 0.8  # Reduce confidence for fallback steps
                
                api_step = {
                    'expression': str(step.expression),
                    'operation': step.operation or 'UNKNOWN',
                    'explanation': step.description or step.operation or 'Step applied',
                    'confidence': confidence,
                    'timestamp': int(i * 1000),  # Simulate timestamps
                    'strategy': strategy,
                    'fallbackUsed': fallback_used
                }
                
                api_steps.append(api_step)
                
                # Check for limitations
                if fallback_used:
                    limitations.append(f"Step {i+1} used fallback strategy")
            
            # Add solution found step if not already present
            if api_steps and api_steps[-1]['operation'] != 'SOLUTION_FOUND':
                final_expr = api_steps[-1]['expression']
                if '=' in final_expr and ('x' in final_expr or 'y' in final_expr or 'z' in final_expr):
                    # Check if it looks like a solution
                    try:
                        if 'x =' in final_expr or 'y =' in final_expr or 'z =' in final_expr:
                            api_steps.append({
                                'expression': final_expr,
                                'operation': 'SOLUTION_FOUND',
                                'explanation': f'Solution found: {final_expr}',
                                'confidence': 1.0,
                                'timestamp': len(api_steps) * 1000,
                                'strategy': 'solution_detection',
                                'fallbackUsed': False
                            })
                    except:
                        pass
            
            return jsonify({
                'success': True,
                'steps': api_steps,
                'finalStrategy': final_strategy,
                'limitations': limitations
            })
            
        except Exception as e:
            logger.error(f"Error during solving: {e}")
            logger.error(traceback.format_exc())
            return jsonify({
                'success': False,
                'error': f'Solving error: {str(e)}'
            }), 500
            
    except Exception as e:
        logger.error(f"Unexpected error in solve endpoint: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        solver, _, _ = get_solver()
        return jsonify({
            'status': 'healthy',
            'solver_loaded': solver is not None,
            'message': 'Graph Neural Algebra Tutor API is running'
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/api/verify', methods=['POST'])
def verify_step():
    """
    Verify if a step is mathematically valid.
    
    Expected JSON payload:
    {
        "current_expression": "2*x + 5 = 9",
        "next_expression": "2*x = 4",
        "operation": "MOVE_TERMS"
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'Missing request body'}), 400
        
        current_expr = data.get('current_expression')
        next_expr = data.get('next_expression')
        operation = data.get('operation')
        
        if not all([current_expr, next_expr]):
            return jsonify({'success': False, 'error': 'Missing expressions'}), 400
        
        # Parse expressions
        try:
            current = sp.parse_expr(current_expr)
            next_expression = sp.parse_expr(next_expr)
        except Exception as e:
            return jsonify({'success': False, 'error': f'Parse error: {str(e)}'}), 400
        
        # Verify step
        _, verifier = get_solver()
        is_valid, message = verifier.verify_step_validity(current, next_expression, operation)
        
        return jsonify({
            'success': True,
            'isValid': is_valid,
            'message': message
        })
        
    except Exception as e:
        logger.error(f"Error in verify endpoint: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/hint', methods=['POST'])
def get_hint():
    """
    Get a hint for the current expression.
    
    Expected JSON payload:
    {
        "expression": "2*x + 5 = 9"
    }
    """
    try:
        data = request.get_json()
        if not data or 'expression' not in data:
            return jsonify({'success': False, 'error': 'Missing expression'}), 400
        
        expression = data['expression'].strip()
        
        # Parse expression
        try:
            if '=' in expression:
                parts = expression.split('=')
                lhs = sp.parse_expr(parts[0].strip())
                rhs = sp.parse_expr(parts[1].strip())
                problem = Eq(lhs, rhs)
            else:
                problem = sp.parse_expr(expression)
        except Exception as e:
            return jsonify({'success': False, 'error': f'Parse error: {str(e)}'}), 400
        
        # Get hint
        _, interactive_solver, _ = get_solver()
        hint = interactive_solver.get_hint(problem)
        
        return jsonify({
            'success': True,
            'hint': hint
        })
        
    except Exception as e:
        logger.error(f"Error in hint endpoint: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Initialize solver on startup
    try:
        get_solver()
        logger.info("API server starting...")
        app.run(host='0.0.0.0', port=5000, debug=False)
    except Exception as e:
        logger.error(f"Failed to start API server: {e}")
        sys.exit(1) 