"""
Streamlit Web Interface for Graph Neural Algebra Tutor

This application provides an interactive web interface for the Graph Neural Algebra Tutor,
allowing users to:
1. Solve algebra problems step-by-step
2. Get hints and explanations
3. Verify their own solutions
4. Explore different problem types
5. Visualize the solution process
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyBboxPatch
import plotly.graph_objects as go
import plotly.express as px
import sympy as sp
from sympy import latex
import json
import time
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.solver import GNNAlgebraSolver, InteractiveSolver
from src.verification import AlgebraicVerifier
from src.data.dataset_generator import AlgebraDatasetGenerator
from src.graph import expression_to_graph, ExpressionGraph
from src.utils import AlgebraicStep, AlgebraicOperation

# Import D3 visualizer
try:
    from d3_visualizer import render_d3_graph
    D3_AVAILABLE = True
except ImportError:
    try:
        from d3_visualizer_simple import render_d3_graph
        D3_AVAILABLE = True
    except ImportError:
        D3_AVAILABLE = False


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Page configuration
st.set_page_config(
    page_title="Graph Neural Algebra Tutor",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .step-container {
        background-color: #f8f9fa;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    
    .hint-container {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    
    .success-container {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    
    .error-container {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    
    .math-expression {
        font-size: 1.2rem;
        font-family: 'Times New Roman', serif;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_solver():
    """Load the GNN solver (cached for performance)."""
    try:
        solver = GNNAlgebraSolver(use_verification=True)
        interactive_solver = InteractiveSolver(solver)
        return solver, interactive_solver
    except Exception as e:
        st.error(f"Error loading solver: {e}")
        return None, None


@st.cache_resource
def load_verifier():
    """Load the algebraic verifier (cached for performance)."""
    return AlgebraicVerifier()


@st.cache_resource
def load_generator():
    """Load the dataset generator (cached for performance)."""
    return AlgebraDatasetGenerator()


def render_latex(expression):
    """Render mathematical expression as LaTeX."""
    try:
        if isinstance(expression, str):
            # Try to parse as SymPy expression
            try:
                expr = sp.parse_expr(expression)
                return f"$${latex(expr)}$$"
            except:
                # If parsing fails, return as-is
                return f"$${expression}$$"
        else:
            # Already a SymPy expression
            return f"$${latex(expression)}$$"
    except Exception as e:
        return str(expression)


def visualize_expression_graph(expression):
    """Visualize the graph representation of an expression using D3.js."""
    if D3_AVAILABLE:
        try:
            # Render D3.js visualization
            render_d3_graph(expression, width=800, height=600)
            return True
        except Exception as e:
            st.error(f"Error with D3.js visualization: {e}")
            return None
    else:
        # Fallback to Plotly visualization
        try:
            # Convert expression to graph
            graph_converter = ExpressionGraph()
            graph_data = graph_converter.expr_to_graph(expression)
            
            # Convert to NetworkX for visualization
            nx_graph = graph_converter.graph_to_networkx(graph_data)
            
            # Create layout
            pos = nx.spring_layout(nx_graph, k=2, iterations=50)
            
            # Create edge traces
            edge_x = []
            edge_y = []
            for edge in nx_graph.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            # Create node traces
            node_x = []
            node_y = []
            node_text = []
            node_colors = []
            
            for node in nx_graph.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                
                # Get node information
                node_info = nx_graph.nodes[node]
                node_text.append(node_info.get('label', str(node)))
                
                # Color by node type
                node_type = node_info.get('type', 'unknown')
                if node_type == 'operator':
                    node_colors.append('#ff7f0e')  # Orange
                elif node_type == 'variable':
                    node_colors.append('#2ca02c')  # Green
                elif node_type == 'constant':
                    node_colors.append('#1f77b4')  # Blue
                else:
                    node_colors.append('#d62728')  # Red
            
            # Create Plotly figure
            fig = go.Figure()
            
            # Add edges
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=2, color='#888'),
                hoverinfo='none',
                mode='lines'
            ))
            
            # Add nodes
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                marker=dict(
                    size=20,
                    color=node_colors,
                    line=dict(width=2, color='black')
                ),
                text=node_text,
                textposition="middle center",
                hoverinfo='text',
                hovertext=node_text
            ))
            
            fig.update_layout(
                title="Expression Graph Visualization (Plotly Fallback)",
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Operators (orange), Variables (green), Constants (blue)",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(size=12)
                )],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=400
            )
            
            return fig
        
        except Exception as e:
            st.error(f"Error visualizing graph: {e}")
            return None


def main():
    """Main application."""
    
    # Header
    st.markdown('<div class="main-header">🧮 Graph Neural Algebra Tutor</div>', unsafe_allow_html=True)
    
    # Load components
    solver, interactive_solver = load_solver()
    verifier = load_verifier()
    generator = load_generator()
    
    if solver is None:
        st.error("Failed to load solver. Please check your installation.")
        return
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    mode = st.sidebar.selectbox(
        "Choose Mode",
        ["🎯 Solve Problems", "🤔 Get Hints", "✅ Check Solutions", "🎲 Random Problems", "📊 Visualize Graphs", "📚 Learn About GNNs"]
    )
    
    # Main content based on mode
    if mode == "🎯 Solve Problems":
        solve_problems_interface(solver, verifier)
    elif mode == "🤔 Get Hints":
        hints_interface(interactive_solver)
    elif mode == "✅ Check Solutions":
        check_solutions_interface(verifier)
    elif mode == "🎲 Random Problems":
        random_problems_interface(generator, solver)
    elif mode == "📊 Visualize Graphs":
        visualize_graphs_interface()
    elif mode == "📚 Learn About GNNs":
        learn_about_gnns_interface()


def solve_problems_interface(solver, verifier):
    """Interface for solving algebra problems step-by-step."""
    
    st.header("🎯 Solve Algebra Problems")
    
    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enter Your Problem")
        problem_input = st.text_input(
            "Enter an algebra problem (e.g., '2*x + 5 = 9' or 'x**2 - 4*x + 3 = 0'):",
            placeholder="2*x + 5 = 9"
        )
        
        # Problem type selection
        problem_type = st.selectbox(
            "Problem Type (optional - for better solving)",
            ["Auto-detect", "Linear Equation", "Quadratic Equation", "Expression Simplification"]
        )
        
        # Options
        show_steps = st.checkbox("Show step-by-step solution", value=True)
        show_verification = st.checkbox("Show verification for each step", value=False)
        show_graph = st.checkbox("Show graph visualization", value=False)
    
    with col2:
        st.subheader("Quick Examples")
        examples = [
            "2*x + 5 = 9",
            "x**2 - 4*x + 3 = 0",
            "3*(x + 2) = 15",
            "2*x + 3*x - 7",
            "x/2 + 3 = 7"
        ]
        
        for example in examples:
            if st.button(f"📝 {example}", key=f"example_{example}"):
                st.session_state.problem_input = example
                st.rerun()
    
    # Use session state for problem input
    if 'problem_input' in st.session_state:
        problem_input = st.session_state.problem_input
    
    # Solve button
    if st.button("🚀 Solve Problem", type="primary"):
        if problem_input.strip():
            solve_problem(problem_input, solver, verifier, show_steps, show_verification, show_graph)
        else:
            st.warning("Please enter a problem to solve.")


def solve_problem(problem_input, solver, verifier, show_steps, show_verification, show_graph):
    """Solve a problem and display results."""
    
    try:
        # Display the problem
        st.subheader("Problem:")
        st.markdown(f'<div class="math-expression">{render_latex(problem_input)}</div>', unsafe_allow_html=True)
        
        # Show graph visualization if requested
        if show_graph:
            with st.expander("🔍 Graph Visualization", expanded=True):
                fig = visualize_expression_graph(problem_input)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        
        # Solve the problem
        with st.spinner("Solving problem..."):
            start_time = time.time()
            steps = solver.solve(problem_input, show_steps=False, return_all_steps=True)
            solve_time = time.time() - start_time
        
        if steps:
            st.success(f"✅ Problem solved in {solve_time:.2f} seconds!")
            
            # Display solution steps
            if show_steps and len(steps) > 1:
                st.subheader("Solution Steps:")
                
                for i, step in enumerate(steps):
                    with st.container():
                        step_html = f"""
                        <div class="step-container">
                            <strong>Step {i + 1}:</strong> {step.description if step.description else ""}
                            <div class="math-expression">{render_latex(step.expression)}</div>
                        </div>
                        """
                        st.markdown(step_html, unsafe_allow_html=True)
                        
                        # Show verification if requested
                        if show_verification and i > 0:
                            prev_step = steps[i - 1]
                            is_valid, explanation = verifier.verify_step_validity(
                                prev_step.expression, step.expression
                            )
                            
                            if is_valid:
                                st.markdown(f'<div class="success-container">✅ Verification: {explanation}</div>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div class="error-container">❌ Verification: {explanation}</div>', unsafe_allow_html=True)
            
            # Final answer
            final_answer = steps[-1].expression
            st.subheader("Final Answer:")
            st.markdown(f'<div class="math-expression" style="font-size: 1.5rem; color: #28a745;">{render_latex(final_answer)}</div>', unsafe_allow_html=True)
            
            # Additional information
            with st.expander("📊 Solution Statistics"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Number of Steps", len(steps))
                with col2:
                    st.metric("Solve Time", f"{solve_time:.2f}s")
                with col3:
                    st.metric("Steps per Second", f"{len(steps) / solve_time:.1f}")
        
        else:
            st.error("❌ Could not solve the problem. Please check your input.")
    
    except Exception as e:
        st.error(f"❌ Error solving problem: {str(e)}")


def hints_interface(interactive_solver):
    """Interface for getting hints."""
    
    st.header("🤔 Get Hints")
    
    st.write("Stuck on a problem? Get a hint for the next step!")
    
    # Input
    current_expression = st.text_input(
        "Enter your current expression or equation:",
        placeholder="2*x + 5 = 9"
    )
    
    if st.button("💡 Get Hint"):
        if current_expression.strip():
            try:
                hint = interactive_solver.get_hint(current_expression)
                
                hint_html = f"""
                <div class="hint-container">
                    <strong>💡 Hint:</strong><br>
                    {hint}
                </div>
                """
                st.markdown(hint_html, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error getting hint: {str(e)}")
        else:
            st.warning("Please enter an expression to get a hint.")


def check_solutions_interface(verifier):
    """Interface for checking student solutions."""
    
    st.header("✅ Check Your Solutions")
    
    st.write("Enter your step and we'll check if it's mathematically correct!")
    
    # Input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Current Step")
        current_step = st.text_input(
            "Enter the current expression:",
            placeholder="2*x + 5 = 9"
        )
    
    with col2:
        st.subheader("Your Next Step")
        next_step = st.text_input(
            "Enter your next step:",
            placeholder="2*x = 4"
        )
    
    if st.button("🔍 Check Step"):
        if current_step.strip() and next_step.strip():
            try:
                is_valid, explanation = verifier.verify_step_validity(current_step, next_step)
                
                if is_valid:
                    success_html = f"""
                    <div class="success-container">
                        <strong>✅ Correct!</strong><br>
                        {explanation}
                    </div>
                    """
                    st.markdown(success_html, unsafe_allow_html=True)
                else:
                    error_html = f"""
                    <div class="error-container">
                        <strong>❌ Incorrect</strong><br>
                        {explanation}
                    </div>
                    """
                    st.markdown(error_html, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error checking step: {str(e)}")
        else:
            st.warning("Please enter both current and next steps.")


def random_problems_interface(generator, solver):
    """Interface for generating and solving random problems."""
    
    st.header("🎲 Random Problems")
    
    st.write("Generate random algebra problems to practice!")
    
    # Problem generation options
    col1, col2 = st.columns(2)
    
    with col1:
        problem_type = st.selectbox(
            "Problem Type",
            ["Linear Equations", "Quadratic Equations", "Expression Simplification", "Mixed"]
        )
        
        difficulty = st.slider("Difficulty Level", 1, 5, 3)
    
    with col2:
        num_problems = st.slider("Number of Problems", 1, 10, 3)
        
        auto_solve = st.checkbox("Auto-solve generated problems", value=False)
    
    if st.button("🎲 Generate Problems"):
        try:
            # Generate problems
            problems = []
            
            if problem_type == "Linear Equations":
                problems = generator.generate_linear_equation_problems(num_problems, difficulty)
            elif problem_type == "Quadratic Equations":
                problems = generator.generate_quadratic_equation_problems(num_problems, difficulty)
            elif problem_type == "Expression Simplification":
                problems = generator.generate_simplification_problems(num_problems, difficulty)
            else:  # Mixed
                problems = generator.generate_mixed_dataset(
                    n_linear=num_problems // 3,
                    n_quadratic=num_problems // 3,
                    n_simplify=num_problems // 3
                )
            
            # Display problems
            for i, problem in enumerate(problems[:num_problems]):
                st.subheader(f"Problem {i + 1}")
                st.markdown(f'<div class="math-expression">{render_latex(problem.initial_expression)}</div>', unsafe_allow_html=True)
                
                if auto_solve:
                    with st.expander(f"Solution for Problem {i + 1}"):
                        try:
                            steps = solver.solve(problem.initial_expression, show_steps=False, return_all_steps=True)
                            for j, step in enumerate(steps):
                                st.markdown(f"**Step {j + 1}:** {render_latex(step.expression)}")
                        except Exception as e:
                            st.error(f"Error solving problem: {e}")
                
                st.markdown("---")
        
        except Exception as e:
            st.error(f"Error generating problems: {str(e)}")


def visualize_graphs_interface():
    """Interface for visualizing expression graphs."""
    
    st.header("📊 Visualize Expression Graphs")
    
    st.write("See how algebraic expressions are represented as graphs!")
    
    # Input
    expression_input = st.text_input(
        "Enter an expression to visualize:",
        placeholder="2*x + 5"
    )
    
    # Example expressions
    st.subheader("Examples:")
    examples = [
        "x + 2",
        "2*x + 5",
        "x**2 + 3*x + 2",
        "2*x + 5 = 9",
        "sin(x) + cos(x)"
    ]
    
    example_cols = st.columns(len(examples))
    for i, example in enumerate(examples):
        with example_cols[i]:
            if st.button(example, key=f"viz_example_{i}"):
                expression_input = example
                st.rerun()
    
    if expression_input.strip():
        try:
            # Visualize the graph
            result = visualize_expression_graph(expression_input)
            
            if result is True:
                # D3.js visualization was rendered
                st.success("✅ D3.js graph visualization rendered successfully!")
                
                # Show graph statistics
                with st.expander("📈 Graph Statistics"):
                    try:
                        graph_data = expression_to_graph(expression_input)
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Number of Nodes", graph_data.num_nodes)
                        with col2:
                            st.metric("Number of Edges", graph_data.edge_index.size(1))
                        with col3:
                            st.metric("Node Features", graph_data.x.size(1))
                        
                        # Show node types
                        st.subheader("Node Information")
                        graph_converter = ExpressionGraph()
                        nx_graph = graph_converter.graph_to_networkx(graph_data)
                        
                        node_data = []
                        for node_id, node_data_dict in nx_graph.nodes(data=True):
                            node_data.append({
                                "Node ID": node_id,
                                "Type": node_data_dict.get('type', 'unknown'),
                                "Label": node_data_dict.get('label', 'N/A')
                            })
                        
                        if node_data:
                            df = pd.DataFrame(node_data)
                            st.dataframe(df, use_container_width=True)
                    
                    except Exception as e:
                        st.error(f"Error getting graph statistics: {e}")
            
            elif result is not None:
                # Plotly fallback visualization
                st.plotly_chart(result, use_container_width=True)
                
                # Show graph statistics
                with st.expander("📈 Graph Statistics"):
                    try:
                        graph_data = expression_to_graph(expression_input)
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Number of Nodes", graph_data.num_nodes)
                        with col2:
                            st.metric("Number of Edges", graph_data.edge_index.size(1))
                        with col3:
                            st.metric("Node Features", graph_data.x.size(1))
                        
                        # Show node types
                        st.subheader("Node Information")
                        graph_converter = ExpressionGraph()
                        nx_graph = graph_converter.graph_to_networkx(graph_data)
                        
                        node_data = []
                        for node_id, node_data_dict in nx_graph.nodes(data=True):
                            node_data.append({
                                "Node ID": node_id,
                                "Type": node_data_dict.get('type', 'unknown'),
                                "Label": node_data_dict.get('label', 'N/A')
                            })
                        
                        if node_data:
                            df = pd.DataFrame(node_data)
                            st.dataframe(df, use_container_width=True)
                    
                    except Exception as e:
                        st.error(f"Error getting graph statistics: {e}")
            
        except Exception as e:
            st.error(f"Error visualizing expression: {str(e)}")


def learn_about_gnns_interface():
    """Educational interface about Graph Neural Networks."""
    
    st.header("📚 Learn About Graph Neural Networks")
    
    st.write("Understand how Graph Neural Networks work for algebra!")
    
    # Introduction
    st.subheader("What are Graph Neural Networks?")
    st.write("""
    Graph Neural Networks (GNNs) are a type of neural network designed to work with graph-structured data.
    In our algebra tutor, we represent mathematical expressions as graphs where:
    
    - **Nodes** represent mathematical elements (numbers, variables, operators)
    - **Edges** represent relationships between these elements
    - **Features** encode the type and properties of each node
    """)
    
    # Why GNNs for Algebra
    st.subheader("Why Use GNNs for Algebra?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Traditional Approach:**")
        st.write("- Treat expressions as sequences of symbols")
        st.write("- Miss structural relationships")
        st.write("- Harder to learn mathematical rules")
        
    with col2:
        st.write("**GNN Approach:**")
        st.write("- Capture mathematical structure explicitly")
        st.write("- Learn relationships between components")
        st.write("- Better generalization to new problems")
    
    # Architecture
    st.subheader("Our GNN Architecture")
    
    architecture_diagram = """
    ```
    Input Expression → Graph Representation → GNN Encoder → Step Decoder → Next Step
                                                      ↓
                                              Operation Classifier
    ```
    """
    st.markdown(architecture_diagram)
    
    st.write("""
    1. **Graph Representation**: Convert expressions to graphs
    2. **GNN Encoder**: Process graph structure to create embeddings
    3. **Step Decoder**: Generate the next step in the solution
    4. **Operation Classifier**: Predict the type of operation to perform
    """)
    
    # Interactive Example
    st.subheader("Interactive Example")
    
    example_expr = st.selectbox(
        "Choose an expression to see its graph representation:",
        ["2*x + 5", "x**2 + 3*x + 2", "2*x + 5 = 9"]
    )
    
    if example_expr:
        # Show the graph
        fig = visualize_expression_graph(example_expr)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Explain the graph
        st.write("**Graph Explanation:**")
        if example_expr == "2*x + 5":
            st.write("""
            - **Orange nodes**: Operators (+, *)
            - **Green nodes**: Variables (x)
            - **Blue nodes**: Constants (2, 5)
            - **Edges**: Show how operations combine elements
            """)
        elif example_expr == "x**2 + 3*x + 2":
            st.write("""
            - More complex structure with multiple operations
            - Power operation (x**2) creates additional nodes
            - Addition combines all terms
            """)
        elif example_expr == "2*x + 5 = 9":
            st.write("""
            - Equation node (=) at the root
            - Left and right sides as separate subgraphs
            - Maintains the equation structure
            """)
    
    # Benefits
    st.subheader("Benefits of Our Approach")
    
    benefits = [
        "**Step-by-step reasoning**: Generates human-like solution steps",
        "**Mathematical correctness**: Symbolic verification ensures valid steps",
        "**Interpretability**: Can explain why each step is taken",
        "**Generalization**: Works on problems not seen during training",
        "**Educational value**: Shows the reasoning process"
    ]
    
    for benefit in benefits:
        st.write(f"✅ {benefit}")
    
    # Research Context
    st.subheader("Research Context")
    
    st.write("""
    This work contributes to the field of **AI for Education** by:
    
    - Combining symbolic reasoning with neural networks
    - Focusing on step-by-step problem solving rather than just answers
    - Providing educational explanations and hints
    - Ensuring mathematical rigor through verification
    
    The approach bridges the gap between powerful neural models and the precision
    required for mathematical education.
    """)


if __name__ == "__main__":
    main() 