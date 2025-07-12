#!/usr/bin/env python3
"""
Test script for D3.js integration with the Graph Neural Algebra Tutor.

This script tests the D3.js visualization functionality and ensures
it integrates properly with the Streamlit web interface.
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_d3_visualizer_import():
    """Test that the D3 visualizer can be imported."""
    print("Testing D3 visualizer import...")
    
    try:
        # Change to web directory
        web_dir = project_root / "web"
        os.chdir(web_dir)
        
        # Try to import the D3 visualizer
        from d3_visualizer import D3GraphVisualizer, render_d3_graph, create_d3_visualization_file
        
        print("✅ D3 visualizer imported successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import D3 visualizer: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error importing D3 visualizer: {e}")
        return False

def test_d3_visualizer_creation():
    """Test creating a D3 visualizer instance."""
    print("\nTesting D3 visualizer creation...")
    
    try:
        from d3_visualizer import D3GraphVisualizer
        
        # Create visualizer instance
        visualizer = D3GraphVisualizer()
        print("✅ D3 visualizer instance created successfully!")
        
        # Test basic methods
        assert hasattr(visualizer, 'visualize_expression')
        assert hasattr(visualizer, 'render_in_streamlit')
        assert hasattr(visualizer, 'create_standalone_visualization')
        
        print("✅ D3 visualizer methods available!")
        return True
        
    except Exception as e:
        print(f"❌ Error creating D3 visualizer: {e}")
        return False

def test_graph_data_conversion():
    """Test converting graph data to D3 format."""
    print("\nTesting graph data conversion...")
    
    try:
        from d3_visualizer import D3GraphVisualizer
        from src.graph import expression_to_graph
        
        visualizer = D3GraphVisualizer()
        
        # Test with a simple expression
        expression = "2*x + 5"
        graph_data = expression_to_graph(expression)
        
        # Convert to D3 format
        d3_data = visualizer.convert_graph_to_d3_format(graph_data)
        
        # Check structure
        assert 'nodes' in d3_data
        assert 'links' in d3_data
        assert isinstance(d3_data['nodes'], list)
        assert isinstance(d3_data['links'], list)
        
        print(f"✅ Graph data converted successfully!")
        print(f"   - Nodes: {len(d3_data['nodes'])}")
        print(f"   - Links: {len(d3_data['links'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error converting graph data: {e}")
        return False

def test_expression_visualization():
    """Test creating visualization for an expression."""
    print("\nTesting expression visualization...")
    
    try:
        from d3_visualizer import D3GraphVisualizer
        
        visualizer = D3GraphVisualizer()
        
        # Test with different expressions
        expressions = [
            "x + 2",
            "2*x + 5",
            "x**2 + 3*x + 2",
            "2*x + 5 = 9"
        ]
        
        for expression in expressions:
            print(f"  Testing expression: {expression}")
            
            # Create visualization HTML
            html_content = visualizer.visualize_expression(expression)
            
            # Check that HTML was generated
            assert html_content is not None
            assert len(html_content) > 0
            assert 'd3.js' in html_content.lower()
            assert 'graph-container' in html_content
            
            print(f"    ✅ HTML generated ({len(html_content)} characters)")
        
        print("✅ All expression visualizations created successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error creating expression visualization: {e}")
        return False

def test_standalone_file_creation():
    """Test creating standalone HTML files."""
    print("\nTesting standalone file creation...")
    
    try:
        from d3_visualizer import create_d3_visualization_file
        
        # Create a standalone file
        expression = "2*x + 5 = 9"
        output_path = create_d3_visualization_file(expression, "test_visualization.html")
        
        # Check that file was created
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0
        
        print(f"✅ Standalone file created: {output_path}")
        
        # Clean up
        os.remove(output_path)
        print("✅ Test file cleaned up")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating standalone file: {e}")
        return False

def test_streamlit_integration():
    """Test that the Streamlit app can import the D3 visualizer."""
    print("\nTesting Streamlit integration...")
    
    try:
        # Change to web directory
        web_dir = project_root / "web"
        os.chdir(web_dir)
        
        # Try to import the app (this will test the D3_AVAILABLE flag)
        import app
        
        # Check if D3_AVAILABLE is defined
        assert hasattr(app, 'D3_AVAILABLE')
        print(f"✅ D3_AVAILABLE flag: {app.D3_AVAILABLE}")
        
        # Test the visualize_expression_graph function
        result = app.visualize_expression_graph("2*x + 5")
        print(f"✅ visualize_expression_graph result: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Streamlit integration: {e}")
        return False

def main():
    """Run all tests."""
    print("🧮 Testing D3.js Integration for Graph Neural Algebra Tutor")
    print("=" * 60)
    
    tests = [
        test_d3_visualizer_import,
        test_d3_visualizer_creation,
        test_graph_data_conversion,
        test_expression_visualization,
        test_standalone_file_creation,
        test_streamlit_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! D3.js integration is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 