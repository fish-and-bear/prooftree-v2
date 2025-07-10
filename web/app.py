"""Streamlit web application for the Graph Neural Algebra Tutor."""

import streamlit as st
import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import networkx as nx
from io import BytesIO
import base64

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.solver import GNNAlgebraSolver, InteractiveSolver
from src.graph import expression_to_graph, ExpressionGraph, visualize_expression_graph
from src.models import create_step_predictor


# Page configuration
st.set_page_config(
    page_title="Graph Neural Algebra Tutor",
    page_icon="🧮",
    layout="wide"
)


# Custom CSS
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        margin: 5px;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .step-container {
        background-color: #f0f0f0;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .hint-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #2196F3;
        margin: 10px 0;
    }
    .success-box {
        background-color: #e8f5e9;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
        margin: 10px 0;
    }
    .error-box {
        background-color: #ffebee;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #f44336;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_solver(model_path: str = None):
    """Load the algebra solver (cached for performance)."""
    solver = GNNAlgebraSolver(
        model_path=model_path,
        device='cpu',
        use_verification=True
    )
    return solver


def render_expression_graph(expression: str):
    """Render the graph visualization of an expression."""
    try:
        # Create graph
        converter = ExpressionGraph()
        graph_data = converter.expr_to_graph(expression)
        G = converter.graph_to_networkx(graph_data)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw nodes with colors
        node_colors = []
        for node in G.nodes():
            node_type = G.nodes[node]['type']
            if node_type == 'operator':
                node_colors.append('#87CEEB')  # Light blue
            elif node_type == 'variable':
                node_colors.append('#90EE90')  # Light green
            elif node_type == 'constant':
                node_colors.append('#FFFFE0')  # Light yellow
            elif node_type == 'function':
                node_colors.append('#FFB6C1')  # Light coral
            else:
                node_colors.append('#D3D3D3')  # Light gray
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1000, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color='gray', 
                              arrows=True, arrowsize=20, arrowstyle='->', ax=ax)
        
        # Draw labels
        labels = nx.get_node_attributes(G, 'label')
        nx.draw_networkx_labels(G, pos, labels, font_size=12, ax=ax)
        
        ax.set_title(f"Expression Graph: {expression}")
        ax.axis('off')
        plt.tight_layout()
        
        return fig
    except Exception as e:
        st.error(f"Error visualizing graph: {e}")
        return None


def main():
    """Main Streamlit application."""
    # Header
    st.title("🧮 Graph Neural Algebra Tutor")
    st.markdown("### Step-by-step algebra solving powered by Graph Neural Networks")
    
    # Initialize session state
    if 'current_problem' not in st.session_state:
        st.session_state.current_problem = None
    if 'solution_steps' not in st.session_state:
        st.session_state.solution_steps = []
    if 'current_step_index' not in st.session_state:
        st.session_state.current_step_index = 0
    if 'show_graph' not in st.session_state:
        st.session_state.show_graph = False
    if 'practice_mode' not in st.session_state:
        st.session_state.practice_mode = False
    
    # Load solver
    solver = load_solver()
    interactive_solver = InteractiveSolver(solver)
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        # Mode selection
        mode = st.radio("Mode", ["Step-by-Step Solver", "Practice Mode", "Explore Graphs"])
        st.session_state.practice_mode = (mode == "Practice Mode")
        
        # Example problems
        st.subheader("Example Problems")
        examples = [
            "2*x + 5 = 9",
            "3*x - 4 = 5",
            "x**2 - 5*x + 6 = 0",
            "2*(x + 3) = 10",
            "x/2 + 3 = 7",
            "2*x + 3*x - 5",
            "x**2 + 2*x + 1",
        ]
        
        for example in examples:
            if st.button(example, key=f"ex_{example}"):
                st.session_state.current_problem = example
                st.session_state.solution_steps = []
                st.session_state.current_step_index = 0
        
        # Options
        st.subheader("Options")
        st.session_state.show_graph = st.checkbox("Show Expression Graphs", value=st.session_state.show_graph)
        
        # About
        st.subheader("About")
        st.markdown("""
        This tutor uses Graph Neural Networks to understand algebraic expressions 
        as graphs and predict solution steps. Each step is verified using symbolic 
        mathematics to ensure correctness.
        """)
    
    # Main content area
    if mode == "Step-by-Step Solver":
        render_solver_mode(solver, interactive_solver)
    elif mode == "Practice Mode":
        render_practice_mode(solver, interactive_solver)
    else:  # Explore Graphs
        render_graph_explorer()


def render_solver_mode(solver, interactive_solver):
    """Render the step-by-step solver mode."""
    st.header("Step-by-Step Solver")
    
    # Problem input
    col1, col2 = st.columns([4, 1])
    with col1:
        problem_input = st.text_input(
            "Enter an algebra problem:",
            value=st.session_state.current_problem or "",
            placeholder="e.g., 2*x + 5 = 9"
        )
    
    with col2:
        solve_button = st.button("Solve", type="primary")
    
    # Solve problem
    if solve_button and problem_input:
        with st.spinner("Solving..."):
            try:
                steps = solver.solve(problem_input, show_steps=False)
                st.session_state.current_problem = problem_input
                st.session_state.solution_steps = steps
                st.session_state.current_step_index = 0
                st.success("Problem solved!")
            except Exception as e:
                st.error(f"Error solving problem: {e}")
    
    # Display solution steps
    if st.session_state.solution_steps:
        st.subheader("Solution Steps")
        
        # Step navigation
        col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
        with col1:
            if st.button("◀ Previous") and st.session_state.current_step_index > 0:
                st.session_state.current_step_index -= 1
        
        with col2:
            if st.button("Next ▶") and st.session_state.current_step_index < len(st.session_state.solution_steps) - 1:
                st.session_state.current_step_index += 1
        
        with col3:
            if st.button("Show All"):
                st.session_state.current_step_index = -1
        
        # Display steps
        if st.session_state.current_step_index == -1:
            # Show all steps
            for i, step in enumerate(st.session_state.solution_steps):
                with st.container():
                    st.markdown(f"""
                    <div class="step-container">
                        <b>Step {i}:</b> {step.expression}<br>
                        <i>{step.description}</i>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.session_state.show_graph and i < len(st.session_state.solution_steps) - 1:
                        fig = render_expression_graph(str(step.expression))
                        if fig:
                            st.pyplot(fig)
                            plt.close()
        else:
            # Show current step
            current_step = st.session_state.solution_steps[st.session_state.current_step_index]
            st.markdown(f"""
            <div class="step-container">
                <b>Step {st.session_state.current_step_index}:</b> {current_step.expression}<br>
                <i>{current_step.description}</i>
            </div>
            """, unsafe_allow_html=True)
            
            # Show graph if enabled
            if st.session_state.show_graph:
                fig = render_expression_graph(str(current_step.expression))
                if fig:
                    st.pyplot(fig)
                    plt.close()
            
            # Progress bar
            progress = (st.session_state.current_step_index + 1) / len(st.session_state.solution_steps)
            st.progress(progress)


def render_practice_mode(solver, interactive_solver):
    """Render the practice mode."""
    st.header("Practice Mode")
    st.markdown("Work through problems step by step with hints and feedback!")
    
    # Problem selection
    if not st.session_state.current_problem:
        st.info("Select a problem from the sidebar or enter your own below.")
        
        problem_input = st.text_input(
            "Enter a problem to practice:",
            placeholder="e.g., 2*x + 5 = 9"
        )
        
        if st.button("Start Practice") and problem_input:
            st.session_state.current_problem = problem_input
            st.session_state.current_step_index = 0
            st.rerun()
    else:
        # Display current problem
        st.markdown(f"### Current Problem: `{st.session_state.current_problem}`")
        
        # Student input
        col1, col2 = st.columns([3, 1])
        with col1:
            student_step = st.text_input(
                "Enter your next step:",
                placeholder="e.g., 2*x = 4",
                key="student_input"
            )
        
        with col2:
            check_button = st.button("Check Step", type="primary")
        
        # Check student's step
        if check_button and student_step:
            valid, feedback = interactive_solver.check_step(
                st.session_state.current_problem,
                student_step
            )
            
            if valid:
                st.markdown(f"""
                <div class="success-box">
                    ✅ {feedback}
                </div>
                """, unsafe_allow_html=True)
                
                # Update current problem to student's step
                st.session_state.current_problem = student_step
                
                # Check if solved
                if solver._is_solved(student_step):
                    st.balloons()
                    st.success("🎉 Congratulations! You've solved the problem!")
                    
                    if st.button("Try Another Problem"):
                        st.session_state.current_problem = None
                        st.rerun()
            else:
                st.markdown(f"""
                <div class="error-box">
                    ❌ {feedback}
                </div>
                """, unsafe_allow_html=True)
        
        # Hint button
        if st.button("Get Hint 💡"):
            hint = interactive_solver.get_hint(st.session_state.current_problem)
            st.markdown(f"""
            <div class="hint-box">
                💡 <b>Hint:</b> {hint}
            </div>
            """, unsafe_allow_html=True)
        
        # Show solution button
        if st.button("Show Solution"):
            with st.expander("Complete Solution"):
                steps = solver.solve(st.session_state.current_problem, show_steps=False)
                for i, step in enumerate(steps):
                    st.write(f"**Step {i}:** {step.expression}")
                    st.write(f"*{step.description}*")
        
        # Reset button
        if st.button("Start New Problem"):
            st.session_state.current_problem = None
            st.rerun()


def render_graph_explorer():
    """Render the graph exploration mode."""
    st.header("Explore Expression Graphs")
    st.markdown("See how algebraic expressions are represented as graphs!")
    
    # Expression input
    expression = st.text_input(
        "Enter an expression or equation:",
        placeholder="e.g., 2*x + 5 = 9 or x**2 + 2*x + 1"
    )
    
    if st.button("Visualize Graph") and expression:
        try:
            # Create two columns
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Graph Visualization")
                fig = render_expression_graph(expression)
                if fig:
                    st.pyplot(fig)
                    plt.close()
            
            with col2:
                st.subheader("Graph Properties")
                
                # Get graph data
                converter = ExpressionGraph()
                graph_data = converter.expr_to_graph(expression)
                
                st.write(f"**Number of nodes:** {graph_data.num_nodes}")
                st.write(f"**Number of edges:** {graph_data.edge_index.shape[1]}")
                st.write(f"**Feature dimension:** {graph_data.x.shape[1]}")
                
                # Show node details
                st.write("**Node Details:**")
                for i, node in enumerate(converter.nodes):
                    st.write(f"- Node {i}: {node.type} ({node.value})")
            
            # Show feature matrix
            with st.expander("View Feature Matrix"):
                st.write("Node feature matrix shape:", graph_data.x.shape)
                st.dataframe(graph_data.x.numpy())
            
        except Exception as e:
            st.error(f"Error: {e}")
    
    # Examples gallery
    st.subheader("Example Expressions")
    
    examples = {
        "Linear Equation": "2*x + 5 = 9",
        "Quadratic Expression": "x**2 + 2*x + 1",
        "Rational Expression": "(x + 1) / (x - 1)",
        "Complex Expression": "2*x**2 + 3*x - 5",
        "Nested Expression": "((x + 1) * (x - 2)) / (x + 3)",
    }
    
    cols = st.columns(3)
    for i, (name, expr) in enumerate(examples.items()):
        with cols[i % 3]:
            if st.button(name):
                st.session_state.explorer_expr = expr
                st.rerun()


def main_cli():
    """Entry point for command line interface."""
    main()


if __name__ == "__main__":
    main() 