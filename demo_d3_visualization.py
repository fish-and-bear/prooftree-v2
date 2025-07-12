#!/usr/bin/env python3
"""
Demo script for D3.js Graph Visualization

This script demonstrates the D3.js visualization capabilities of the
Graph Neural Algebra Tutor by creating interactive visualizations for
various mathematical expressions.
"""

import sys
import os
from pathlib import Path
import webbrowser
import time

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def demo_basic_expressions():
    """Demo basic mathematical expressions."""
    print("🧮 D3.js Graph Visualization Demo")
    print("=" * 50)
    
    try:
        from web.d3_visualizer_simple import create_d3_visualization_file
        
        # Test expressions
        expressions = [
            ("Simple Addition", "x + 2"),
            ("Multiplication", "2 * x"),
            ("Linear Equation", "2*x + 5 = 9"),
            ("Quadratic Expression", "x**2 + 3*x + 2"),
            ("Complex Expression", "3*(x + 2) - 5*x"),
            ("Function", "sin(x) + cos(x)"),
            ("Fraction", "x/2 + 3"),
            ("Power", "x**3 - 2*x**2 + x")
        ]
        
        created_files = []
        
        for title, expression in expressions:
            print(f"\n📊 Creating visualization for: {title}")
            print(f"   Expression: {expression}")
            
            # Create filename
            filename = f"demo_{title.lower().replace(' ', '_').replace('*', 'x').replace('/', 'div').replace('+', 'plus').replace('-', 'minus').replace('=', 'eq').replace('**', 'pow')}.html"
            
            # Create visualization
            output_path = create_d3_visualization_file(expression, filename)
            created_files.append((title, expression, output_path))
            
            print(f"   ✅ Created: {output_path}")
        
        return created_files
        
    except Exception as e:
        print(f"❌ Error in demo: {e}")
        return []

def demo_interactive_features():
    """Demo interactive features of the D3.js visualization."""
    print("\n🎯 Interactive Features Demo")
    print("=" * 50)
    
    try:
        from web.d3_visualizer_simple import D3GraphVisualizer
        
        # Create a complex expression for demonstration
        expression = "2*x**2 + 3*x + 1 = 0"
        
        print(f"Creating interactive visualization for: {expression}")
        
        visualizer = D3GraphVisualizer()
        html_content = visualizer.visualize_expression(expression, width=1000, height=700)
        
        # Save to file
        output_path = "interactive_demo.html"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ Interactive demo created: {output_path}")
        print("\n🎮 Interactive Features:")
        print("   • Zoom in/out with mouse wheel")
        print("   • Drag nodes to reposition them")
        print("   • Hover over nodes to highlight connections")
        print("   • Click nodes to see details")
        print("   • Pan by dragging the background")
        
        return output_path
        
    except Exception as e:
        print(f"❌ Error in interactive demo: {e}")
        return None

def demo_graph_statistics():
    """Demo graph statistics and analysis."""
    print("\n📈 Graph Statistics Demo")
    print("=" * 50)
    
    try:
        from web.d3_visualizer_simple import D3GraphVisualizer
        
        expressions = [
            "x + 2",
            "2*x + 5",
            "x**2 + 3*x + 2",
            "2*x + 5 = 9",
            "sin(x) + cos(x)"
        ]
        
        visualizer = D3GraphVisualizer()
        
        for expression in expressions:
            print(f"\n📊 Analyzing: {expression}")
            
            # Try to get real graph data
            try:
                from src.graph import expression_to_graph
                graph_data = expression_to_graph(expression)
                d3_data = visualizer.convert_graph_to_d3_format(graph_data)
                
                print(f"   Nodes: {len(d3_data['nodes'])}")
                print(f"   Edges: {len(d3_data['links'])}")
                
                # Count node types
                node_types = {}
                for node in d3_data['nodes']:
                    node_type = node['type']
                    node_types[node_type] = node_types.get(node_type, 0) + 1
                
                print("   Node Types:")
                for node_type, count in node_types.items():
                    print(f"     • {node_type}: {count}")
                    
            except ImportError:
                print("   Using mock data (src.graph not available)")
                d3_data = visualizer._generate_mock_graph_data()
                print(f"   Mock Nodes: {len(d3_data['nodes'])}")
                print(f"   Mock Edges: {len(d3_data['links'])}")
        
    except Exception as e:
        print(f"❌ Error in statistics demo: {e}")

def open_visualizations_in_browser(files):
    """Open visualization files in the default web browser."""
    print("\n🌐 Opening visualizations in browser...")
    
    for title, expression, filepath in files:
        try:
            # Convert to absolute path
            abs_path = os.path.abspath(filepath)
            file_url = f"file:///{abs_path.replace(os.sep, '/')}"
            
            print(f"   Opening: {title}")
            webbrowser.open(file_url)
            
            # Small delay to prevent overwhelming the browser
            time.sleep(1)
            
        except Exception as e:
            print(f"   ❌ Error opening {title}: {e}")

def create_demo_index():
    """Create an index page that links to all demo visualizations."""
    print("\n📋 Creating demo index page...")
    
    try:
        # Find all demo HTML files
        demo_files = []
        for file in os.listdir('.'):
            if file.startswith('demo_') and file.endswith('.html'):
                demo_files.append(file)
        
        if not demo_files:
            print("   No demo files found")
            return
        
        # Create index HTML
        index_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>D3.js Graph Visualization Demo Index</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            margin: 0;
            font-size: 2.5rem;
            font-weight: 300;
        }
        
        .header p {
            margin: 15px 0 0 0;
            opacity: 0.9;
            font-size: 1.1rem;
        }
        
        .content {
            padding: 30px;
        }
        
        .demo-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .demo-card {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 10px;
            padding: 20px;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .demo-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }
        
        .demo-card h3 {
            margin: 0 0 10px 0;
            color: #495057;
            font-size: 1.2rem;
        }
        
        .demo-card p {
            margin: 0 0 15px 0;
            color: #6c757d;
            font-size: 0.9rem;
        }
        
        .demo-link {
            display: inline-block;
            background: #007bff;
            color: white;
            text-decoration: none;
            padding: 8px 16px;
            border-radius: 5px;
            font-size: 0.9rem;
            transition: background-color 0.2s;
        }
        
        .demo-link:hover {
            background: #0056b3;
        }
        
        .features {
            background: #e3f2fd;
            border-radius: 10px;
            padding: 20px;
            margin-top: 30px;
        }
        
        .features h3 {
            margin: 0 0 15px 0;
            color: #1976d2;
        }
        
        .features ul {
            margin: 0;
            padding-left: 20px;
        }
        
        .features li {
            margin: 5px 0;
            color: #424242;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧮 Graph Neural Algebra Tutor</h1>
            <p>D3.js Graph Visualization Demo Collection</p>
        </div>
        
        <div class="content">
            <h2>📊 Available Visualizations</h2>
            <p>Click on any visualization below to see the interactive D3.js graph representation of mathematical expressions.</p>
            
            <div class="demo-grid">
"""
        
        # Add demo cards
        for file in demo_files:
            # Extract title from filename
            title = file.replace('demo_', '').replace('.html', '').replace('_', ' ').title()
            
            index_html += f"""
                <div class="demo-card">
                    <h3>{title}</h3>
                    <p>Interactive D3.js visualization of mathematical expression</p>
                    <a href="{file}" class="demo-link" target="_blank">View Visualization</a>
                </div>
"""
        
        index_html += """
            </div>
            
            <div class="features">
                <h3>🎮 Interactive Features</h3>
                <ul>
                    <li><strong>Zoom:</strong> Use mouse wheel to zoom in/out</li>
                    <li><strong>Pan:</strong> Drag the background to pan around</li>
                    <li><strong>Node Interaction:</strong> Drag nodes to reposition them</li>
                    <li><strong>Hover Effects:</strong> Hover over nodes to highlight connections</li>
                    <li><strong>Node Details:</strong> Click nodes to see detailed information</li>
                    <li><strong>Responsive:</strong> Works on different screen sizes</li>
                </ul>
            </div>
            
            <div class="features">
                <h3>🎨 Visual Design</h3>
                <ul>
                    <li><strong>Color Coding:</strong> Different node types have distinct colors</li>
                    <li><strong>Node Shapes:</strong> Different shapes for different mathematical elements</li>
                    <li><strong>Force Layout:</strong> Automatic positioning using D3.js force simulation</li>
                    <li><strong>Smooth Animations:</strong> Elegant transitions and animations</li>
                    <li><strong>Legend:</strong> Clear legend explaining node types</li>
                    <li><strong>Statistics:</strong> Real-time graph statistics and information</li>
                </ul>
            </div>
        </div>
    </div>
</body>
</html>
"""
        
        # Write index file
        with open('demo_index.html', 'w', encoding='utf-8') as f:
            f.write(index_html)
        
        print("   ✅ Created: demo_index.html")
        
        # Open index in browser
        abs_path = os.path.abspath('demo_index.html')
        file_url = f"file:///{abs_path.replace(os.sep, '/')}"
        webbrowser.open(file_url)
        
    except Exception as e:
        print(f"   ❌ Error creating index: {e}")

def main():
    """Run the complete demo."""
    print("🚀 Starting D3.js Graph Visualization Demo")
    print("=" * 60)
    
    # Change to web directory for file creation
    web_dir = project_root / "web"
    os.chdir(web_dir)
    
    # Run demos
    files = demo_basic_expressions()
    interactive_file = demo_interactive_features()
    demo_graph_statistics()
    
    if files:
        print(f"\n✅ Created {len(files)} visualization files")
        
        # Ask user if they want to open in browser
        try:
            response = input("\n🌐 Open visualizations in browser? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                open_visualizations_in_browser(files)
                if interactive_file:
                    webbrowser.open(f"file:///{os.path.abspath(interactive_file).replace(os.sep, '/')}")
        except KeyboardInterrupt:
            print("\n⏹️  Demo interrupted by user")
    
    # Create index page
    create_demo_index()
    
    print("\n🎉 Demo completed!")
    print("\n📁 Generated files:")
    for file in os.listdir('.'):
        if file.endswith('.html'):
            print(f"   • {file}")
    
    print("\n💡 Tips:")
    print("   • Open demo_index.html to see all visualizations")
    print("   • Try interacting with the graphs (zoom, drag, hover)")
    print("   • Check the browser console for any JavaScript errors")
    print("   • The visualizations work offline once loaded")

if __name__ == "__main__":
    main() 