"""Command-line interface for the Graph Neural Algebra Tutor solver."""

import click
import sys
from pathlib import Path
from typing import Optional

from .gnn_solver import GNNAlgebraSolver, InteractiveSolver


@click.command()
@click.argument('problem', required=False)
@click.option('--interactive', '-i', is_flag=True, help='Run in interactive mode')
@click.option('--model-path', '-m', type=click.Path(exists=True), help='Path to trained model')
@click.option('--no-verify', is_flag=True, help='Disable symbolic verification')
@click.option('--max-steps', type=int, default=20, help='Maximum solution steps')
@click.option('--device', type=click.Choice(['cpu', 'cuda']), default='cpu', help='Device to use')
def main(problem: Optional[str] = None,
         interactive: bool = False,
         model_path: Optional[str] = None,
         no_verify: bool = False,
         max_steps: int = 20,
         device: str = 'cpu'):
    """Graph Neural Algebra Tutor - Solve algebra problems step by step.
    
    Examples:
        gnat-solve "2*x + 5 = 9"
        gnat-solve "x**2 - 5*x + 6 = 0"
        gnat-solve -i  # Interactive mode
    """
    # Initialize solver
    solver = GNNAlgebraSolver(
        model_path=model_path,
        device=device,
        max_steps=max_steps,
        use_verification=not no_verify
    )
    
    if interactive or problem is None:
        # Interactive mode
        run_interactive_mode(solver)
    else:
        # Single problem mode
        print(f"\nGraph Neural Algebra Tutor")
        print("=" * 50)
        solver.solve(problem, show_steps=True)


def run_interactive_mode(solver: GNNAlgebraSolver):
    """Run the solver in interactive mode."""
    interactive_solver = InteractiveSolver(solver)
    
    print("\nGraph Neural Algebra Tutor - Interactive Mode")
    print("=" * 50)
    print("Commands:")
    print("  solve <problem>  - Solve a problem step by step")
    print("  hint             - Get a hint for the current problem")
    print("  check <step>     - Check if your step is correct")
    print("  help             - Show this help message")
    print("  quit             - Exit the program")
    print()
    
    current_problem = None
    
    while True:
        try:
            # Get user input
            user_input = input("> ").strip()
            
            if not user_input:
                continue
            
            # Parse command
            parts = user_input.split(maxsplit=1)
            command = parts[0].lower()
            
            if command == 'quit' or command == 'exit':
                print("Goodbye!")
                break
            
            elif command == 'help':
                print("\nAvailable commands:")
                print("  solve <problem>  - Solve a problem step by step")
                print("  hint             - Get a hint for the current problem")
                print("  check <step>     - Check if your step is correct")
                print("  help             - Show this help message")
                print("  quit             - Exit the program")
            
            elif command == 'solve':
                if len(parts) < 2:
                    print("Please provide a problem to solve.")
                    print("Example: solve 2*x + 5 = 9")
                else:
                    problem = parts[1]
                    current_problem = problem
                    print(f"\nSolving: {problem}")
                    print("-" * 40)
                    solver.solve(problem, show_steps=True)
                    print()
            
            elif command == 'hint':
                if current_problem is None:
                    print("No current problem. Use 'solve <problem>' first.")
                else:
                    hint = interactive_solver.get_hint(current_problem)
                    print(f"Hint: {hint}")
            
            elif command == 'check':
                if current_problem is None:
                    print("No current problem. Use 'solve <problem>' first.")
                elif len(parts) < 2:
                    print("Please provide your step to check.")
                    print("Example: check 2*x = 4")
                else:
                    student_step = parts[1]
                    valid, feedback = interactive_solver.check_step(current_problem, student_step)
                    print(f"Your step: {student_step}")
                    print(f"Feedback: {feedback}")
                    
                    if valid:
                        # Update current problem to the student's step
                        current_problem = student_step
            
            else:
                # Try to parse as a problem directly
                if '=' in user_input or any(op in user_input for op in ['+', '-', '*', '/', '**']):
                    current_problem = user_input
                    print(f"\nSolving: {user_input}")
                    print("-" * 40)
                    solver.solve(user_input, show_steps=True)
                    print()
                else:
                    print(f"Unknown command: {command}")
                    print("Type 'help' for available commands.")
        
        except KeyboardInterrupt:
            print("\nUse 'quit' to exit.")
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.")


if __name__ == '__main__':
    main() 