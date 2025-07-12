"""
D3.js Graph Visualizer for Streamlit Integration

This module provides D3.js-based graph visualization capabilities for the
Graph Neural Algebra Tutor web interface.
"""

import streamlit as st
import streamlit.components.v1 as components
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import tempfile
import webbrowser
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
import socket

from src.graph import expression_to_graph, ExpressionGraph
from src.utils import AlgebraicStep


class D3GraphVisualizer:
    """D3.js-based graph visualizer for mathematical expressions."""
    
    def __init__(self, static_dir: str = None):
        """Initialize the D3.js visualizer.
        
        Args:
            static_dir: Directory containing static files (D3.js, CSS, etc.)
        """
        self.static_dir = static_dir or Path(__file__).parent / "static"
        self.temp_dir = None
        self.server = None
        self.server_thread = None
        
    def _find_free_port(self) -> int:
        """Find a free port for the local server."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port
    
    def _start_local_server(self, port: int):
        """Start a local HTTP server to serve static files."""
        os.chdir(self.temp_dir)
        
        class CustomHandler(SimpleHTTPRequestHandler):
            def end_headers(self):
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
                self.send_header('Access-Control-Allow-Headers', 'Content-Type')
                super().end_headers()
        
        self.server = HTTPServer(('localhost', port), CustomHandler)
        self.server.serve_forever()
    
    def _create_visualization_html(self, graph_data: Dict[str, Any], 
                                 expression: str, 
                                 width: int = 800, 
                                 height: int = 600) -> str:
        """Create HTML content for the D3.js visualization."""
        
        # Convert graph data to JSON
        graph_json = json.dumps(graph_data, default=str)
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Graph Visualization - {expression}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }}
        
        .header h1 {{
            margin: 0;
            font-size: 1.8rem;
            font-weight: 300;
        }}
        
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
            font-size: 1rem;
        }}
        
        .visualization {{
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        
        #graph-container {{
            border: 2px solid #e9ecef;
            border-radius: 8px;
            background: white;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }}
        
        .info-panel {{
            margin-top: 20px;
            padding: 15px;
            background: #e3f2fd;
            border-radius: 8px;
            border-left: 4px solid #2196f3;
            width: 100%;
            max-width: 800px;
        }}
        
        .info-panel h3 {{
            margin: 0 0 10px 0;
            color: #1976d2;
        }}
        
        .info-panel p {{
            margin: 5px 0;
            color: #424242;
        }}
        
        .legend {{
            margin-top: 20px;
            padding: 15px;
            background: white;
            border-radius: 8px;
            border: 1px solid #dee2e6;
            width: 100%;
            max-width: 800px;
        }}
        
        .legend h4 {{
            margin: 0 0 15px 0;
            color: #495057;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 8px 0;
        }}
        
        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 50%;
            margin-right: 10px;
            border: 1px solid #333;
        }}
        
        .tooltip {{
            position: absolute;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 8px;
            border-radius: 4px;
            font-size: 12px;
            pointer-events: none;
            z-index: 1000;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧮 Graph Neural Algebra Tutor</h1>
            <p>D3.js Visualization: {expression}</p>
        </div>
        
        <div class="visualization">
            <div id="graph-container"></div>
            
            <div class="info-panel" id="info-panel">
                <h3>Graph Information</h3>
                <div id="graph-stats"></div>
            </div>
            
            <div class="legend" id="legend">
                <h4>Node Types</h4>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #ff7f0e;"></div>
                    <div class="legend-text">Operators (+, -, *, /, =)</div>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #2ca02c;"></div>
                    <div class="legend-text">Variables (x, y, z)</div>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #1f77b4;"></div>
                    <div class="legend-text">Constants (numbers)</div>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #9467bd;"></div>
                    <div class="legend-text">Functions (sin, cos, etc.)</div>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #d62728;"></div>
                    <div class="legend-text">Equations (=)</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // D3.js Graph Visualization Class
        class D3GraphVisualizer {{
            constructor(containerId, width = 800, height = 600) {{
                this.containerId = containerId;
                this.width = width;
                this.height = height;
                this.svg = null;
                this.simulation = null;
                this.nodes = [];
                this.links = [];
                this.nodeTypes = {{
                    'operator': {{ color: '#ff7f0e', size: 25, symbol: 'circle' }},
                    'variable': {{ color: '#2ca02c', size: 20, symbol: 'diamond' }},
                    'constant': {{ color: '#1f77b4', size: 18, symbol: 'circle' }},
                    'function': {{ color: '#9467bd', size: 22, symbol: 'square' }},
                    'equation': {{ color: '#d62728', size: 30, symbol: 'rect' }},
                    'unknown': {{ color: '#7f7f7f', size: 15, symbol: 'circle' }}
                }};
                this.init();
            }}

            init() {{
                const container = document.getElementById(this.containerId);
                if (container) {{
                    container.innerHTML = '';
                }}

                this.svg = d3.select(`#${{this.containerId}}`)
                    .append('svg')
                    .attr('width', this.width)
                    .attr('height', this.height)
                    .attr('viewBox', `0 0 ${{this.width}} ${{this.height}}`)
                    .style('background-color', '#f8f9fa')
                    .style('border', '1px solid #dee2e6')
                    .style('border-radius', '8px');

                const zoom = d3.zoom()
                    .scaleExtent([0.1, 4])
                    .on('zoom', (event) => {{
                        this.svg.select('g').attr('transform', event.transform);
                    }});

                this.svg.call(zoom);
                this.svg.append('g').attr('class', 'graph-container');
                this.createLegend();
            }}

            createLegend() {{
                const legend = this.svg.append('g')
                    .attr('class', 'legend')
                    .attr('transform', `translate(10, 10)`);

                let yOffset = 0;
                Object.entries(this.nodeTypes).forEach(([type, config]) => {{
                    const legendItem = legend.append('g')
                        .attr('transform', `translate(0, ${{yOffset}})`);

                    if (config.symbol === 'circle') {{
                        legendItem.append('circle')
                            .attr('r', config.size / 2)
                            .attr('fill', config.color)
                            .attr('stroke', '#333')
                            .attr('stroke-width', 1);
                    }} else if (config.symbol === 'diamond') {{
                        const size = config.size / 2;
                        legendItem.append('polygon')
                            .attr('points', `0,-${{size}} ${{size}},0 0,${{size}} -${{size}},0`)
                            .attr('fill', config.color)
                            .attr('stroke', '#333')
                            .attr('stroke-width', 1);
                    }} else if (config.symbol === 'square') {{
                        const size = config.size / 2;
                        legendItem.append('rect')
                            .attr('width', config.size)
                            .attr('height', config.size)
                            .attr('x', -size)
                            .attr('y', -size)
                            .attr('fill', config.color)
                            .attr('stroke', '#333')
                            .attr('stroke-width', 1);
                    }} else if (config.symbol === 'rect') {{
                        const size = config.size / 2;
                        legendItem.append('rect')
                            .attr('width', config.size * 1.5)
                            .attr('height', config.size)
                            .attr('x', -size * 0.75)
                            .attr('y', -size)
                            .attr('fill', config.color)
                            .attr('stroke', '#333')
                            .attr('stroke-width', 1);
                    }}

                    legendItem.append('text')
                        .attr('x', config.size + 5)
                        .attr('y', 4)
                        .attr('font-size', '12px')
                        .attr('font-weight', 'bold')
                        .text(type.charAt(0).toUpperCase() + type.slice(1));

                    yOffset += 25;
                }});
            }}

            updateGraph(graphData) {{
                this.nodes = graphData.nodes || [];
                this.links = graphData.links || [];

                this.svg.select('.graph-container').selectAll('*').remove();

                if (this.simulation) {{
                    this.simulation.stop();
                }}

                this.simulation = d3.forceSimulation(this.nodes)
                    .force('link', d3.forceLink(this.links).id(d => d.id).distance(80))
                    .force('charge', d3.forceManyBody().strength(-300))
                    .force('center', d3.forceCenter(this.width / 2, this.height / 2))
                    .force('collision', d3.forceCollide().radius(d => this.nodeTypes[d.type]?.size || 20));

                const link = this.svg.select('.graph-container')
                    .append('g')
                    .attr('class', 'links')
                    .selectAll('line')
                    .data(this.links)
                    .enter()
                    .append('line')
                    .attr('stroke', '#999')
                    .attr('stroke-width', 2)
                    .attr('stroke-opacity', 0.6)
                    .attr('marker-end', 'url(#arrowhead)');

                this.svg.select('.graph-container')
                    .append('defs')
                    .append('marker')
                    .attr('id', 'arrowhead')
                    .attr('viewBox', '0 -5 10 10')
                    .attr('refX', 15)
                    .attr('refY', 0)
                    .attr('markerWidth', 6)
                    .attr('markerHeight', 6)
                    .attr('orient', 'auto')
                    .append('path')
                    .attr('d', 'M0,-5L10,0L0,5')
                    .attr('fill', '#999');

                const node = this.svg.select('.graph-container')
                    .append('g')
                    .attr('class', 'nodes')
                    .selectAll('g')
                    .data(this.nodes)
                    .enter()
                    .append('g')
                    .attr('class', 'node')
                    .call(d3.drag()
                        .on('start', this.dragstarted.bind(this))
                        .on('drag', this.dragged.bind(this))
                        .on('end', this.dragended.bind(this)));

                node.each((d, i, nodes) => {{
                    const nodeGroup = d3.select(nodes[i]);
                    const config = this.nodeTypes[d.type] || this.nodeTypes.unknown;

                    if (config.symbol === 'circle') {{
                        nodeGroup.append('circle')
                            .attr('r', config.size)
                            .attr('fill', config.color)
                            .attr('stroke', '#333')
                            .attr('stroke-width', 2)
                            .style('cursor', 'pointer');
                    }} else if (config.symbol === 'diamond') {{
                        nodeGroup.append('polygon')
                            .attr('points', `0,-${{config.size}} ${{config.size}},0 0,${{config.size}} -${{config.size}},0`)
                            .attr('fill', config.color)
                            .attr('stroke', '#333')
                            .attr('stroke-width', 2)
                            .style('cursor', 'pointer');
                    }} else if (config.symbol === 'square') {{
                        nodeGroup.append('rect')
                            .attr('width', config.size * 2)
                            .attr('height', config.size * 2)
                            .attr('x', -config.size)
                            .attr('y', -config.size)
                            .attr('fill', config.color)
                            .attr('stroke', '#333')
                            .attr('stroke-width', 2)
                            .style('cursor', 'pointer');
                    }} else if (config.symbol === 'rect') {{
                        nodeGroup.append('rect')
                            .attr('width', config.size * 3)
                            .attr('height', config.size * 2)
                            .attr('x', -config.size * 1.5)
                            .attr('y', -config.size)
                            .attr('fill', config.color)
                            .attr('stroke', '#333')
                            .attr('stroke-width', 2)
                            .style('cursor', 'pointer');
                    }}
                }});

                node.append('text')
                    .attr('text-anchor', 'middle')
                    .attr('dy', '.35em')
                    .attr('font-size', '12px')
                    .attr('font-weight', 'bold')
                    .attr('fill', 'white')
                    .text(d => d.label)
                    .style('pointer-events', 'none');

                node.on('mouseover', this.handleMouseOver.bind(this))
                    .on('mouseout', this.handleMouseOut.bind(this))
                    .on('click', this.handleNodeClick.bind(this));

                this.simulation.on('tick', () => {{
                    link
                        .attr('x1', d => d.source.x)
                        .attr('y1', d => d.source.y)
                        .attr('x2', d => d.target.x)
                        .attr('y2', d => d.target.y);

                    node.attr('transform', d => `translate(${{d.x}},${{d.y}})`);
                }});

                this.animateNodes();
            }}

            animateNodes() {{
                this.svg.selectAll('.node')
                    .style('opacity', 0)
                    .transition()
                    .duration(800)
                    .delay((d, i) => i * 100)
                    .style('opacity', 1);

                this.svg.selectAll('.links line')
                    .style('opacity', 0)
                    .transition()
                    .duration(600)
                    .delay(400)
                    .style('opacity', 0.6);
            }}

            handleMouseOver(event, d) {{
                const connectedLinks = this.links.filter(link => 
                    link.source.id === d.id || link.target.id === d.id
                );
                const connectedNodes = this.nodes.filter(node => 
                    connectedLinks.some(link => 
                        link.source.id === node.id || link.target.id === node.id
                    )
                );

                this.svg.selectAll('.node').style('opacity', 0.3);
                this.svg.selectAll('.links line').style('opacity', 0.1);

                this.svg.selectAll('.node').filter(node => 
                    connectedNodes.some(n => n.id === node.id)
                ).style('opacity', 1);

                this.svg.selectAll('.links line').filter(link => 
                    connectedLinks.some(l => 
                        (l.source.id === link.source.id && l.target.id === link.target.id) ||
                        (l.source.id === link.target.id && l.target.id === link.source.id)
                    )
                ).style('opacity', 0.8);

                this.showTooltip(event, d);
            }}

            handleMouseOut(event, d) {{
                this.svg.selectAll('.node').style('opacity', 1);
                this.svg.selectAll('.links line').style('opacity', 0.6);
                this.hideTooltip();
            }}

            handleNodeClick(event, d) {{
                const node = d3.select(event.currentTarget);
                const isSelected = node.classed('selected');

                this.svg.selectAll('.node').classed('selected', false);
                this.svg.selectAll('.node circle, .node polygon, .node rect')
                    .attr('stroke-width', 2);

                if (!isSelected) {{
                    node.classed('selected', true);
                    node.select('circle, polygon, rect').attr('stroke-width', 4);
                    this.showNodeDetails(d);
                }}
            }}

            showTooltip(event, d) {{
                const tooltip = d3.select('body').select('.tooltip');
                
                if (tooltip.empty()) {{
                    d3.select('body').append('div')
                        .attr('class', 'tooltip')
                        .style('position', 'absolute')
                        .style('background', 'rgba(0, 0, 0, 0.8)')
                        .style('color', 'white')
                        .style('padding', '8px')
                        .style('border-radius', '4px')
                        .style('font-size', '12px')
                        .style('pointer-events', 'none')
                        .style('z-index', 1000);
                }}

                d3.select('.tooltip')
                    .style('left', (event.pageX + 10) + 'px')
                    .style('top', (event.pageY - 10) + 'px')
                    .html(`
                        <strong>${{d.label}}</strong><br>
                        Type: ${{d.type}}<br>
                        ID: ${{d.id}}
                    `);
            }}

            hideTooltip() {{
                d3.select('.tooltip').remove();
            }}

            showNodeDetails(d) {{
                let detailsPanel = d3.select('#node-details');
                
                if (detailsPanel.empty()) {{
                    detailsPanel = d3.select('body').append('div')
                        .attr('id', 'node-details')
                        .style('position', 'fixed')
                        .style('top', '20px')
                        .style('right', '20px')
                        .style('background', 'white')
                        .style('border', '1px solid #ccc')
                        .style('border-radius', '8px')
                        .style('padding', '15px')
                        .style('box-shadow', '0 4px 8px rgba(0,0,0,0.1)')
                        .style('z-index', 1000)
                        .style('max-width', '300px');
                }}

                detailsPanel.html(`
                    <h4 style="margin: 0 0 10px 0; color: #333;">Node Details</h4>
                    <p><strong>Label:</strong> ${{d.label}}</p>
                    <p><strong>Type:</strong> ${{d.type}}</p>
                    <p><strong>ID:</strong> ${{d.id}}</p>
                    <button onclick="document.getElementById('node-details').remove()" 
                            style="background: #dc3545; color: white; border: none; padding: 5px 10px; border-radius: 4px; cursor: pointer;">
                        Close
                    </button>
                `);
            }}

            dragstarted(event, d) {{
                if (!event.active) this.simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            }}

            dragged(event, d) {{
                d.fx = event.x;
                d.fy = event.y;
            }}

            dragended(event, d) {{
                if (!event.active) this.simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            }}

            convertGraphData(graphData) {{
                const nodes = [];
                const links = [];
                const nodeMap = new Map();

                if (graphData.x && graphData.edge_index) {{
                    const numNodes = graphData.x.length;
                    const edgeIndex = graphData.edge_index;

                    for (let i = 0; i < numNodes; i++) {{
                        const features = graphData.x[i];
                        const nodeType = this.determineNodeType(features);
                        const label = this.getNodeLabel(features, i);
                        
                        nodes.push({{
                            id: i,
                            label: label,
                            type: nodeType,
                            features: features,
                            x: Math.random() * this.width,
                            y: Math.random() * this.height
                        }});
                        nodeMap.set(i, nodes[i]);
                    }}

                    for (let i = 0; i < edgeIndex[0].length; i++) {{
                        const source = edgeIndex[0][i];
                        const target = edgeIndex[1][i];
                        
                        links.push({{
                            source: source,
                            target: target,
                            id: `${{source}}-${{target}}`
                        }});
                    }}
                }} else if (graphData.nodes && graphData.edges) {{
                    graphData.nodes.forEach((node, i) => {{
                        nodes.push({{
                            id: node.id || i,
                            label: node.label || node.id || i,
                            type: node.type || 'unknown',
                            features: node.features || [],
                            x: Math.random() * this.width,
                            y: Math.random() * this.height
                        }});
                        nodeMap.set(node.id || i, nodes[i]);
                    }});

                    graphData.edges.forEach((edge, i) => {{
                        links.push({{
                            source: edge.source,
                            target: edge.target,
                            id: `${{edge.source}}-${{edge.target}}`
                        }});
                    }});
                }}

                return {{ nodes, links }};
            }}

            determineNodeType(features) {{
                if (!Array.isArray(features) || features.length === 0) return 'unknown';
                
                const firstFeature = features[0];
                if (firstFeature > 0.8) return 'operator';
                if (firstFeature > 0.5) return 'variable';
                if (firstFeature > 0.2) return 'constant';
                return 'function';
            }}

            getNodeLabel(features, index) {{
                if (!Array.isArray(features) || features.length === 0) return `N${{index}}`;
                
                const firstFeature = features[0];
                if (firstFeature > 0.8) return ['+', '-', '*', '/', '='][index % 5];
                if (firstFeature > 0.5) return `x${{index}}`;
                if (firstFeature > 0.2) return index.toString();
                return `f${{index}}`;
            }}

            render(graphData) {{
                const d3Data = this.convertGraphData(graphData);
                this.updateGraph(d3Data);
                this.updateStats(d3Data);
            }}

            updateStats(graphData) {{
                const statsDiv = document.getElementById('graph-stats');
                const nodeCount = graphData.nodes.length;
                const edgeCount = graphData.links.length;
                
                const nodeTypes = {{}};
                graphData.nodes.forEach(node => {{
                    nodeTypes[node.type] = (nodeTypes[node.type] || 0) + 1;
                }});
                
                statsDiv.innerHTML = `
                    <p><strong>Total Nodes:</strong> ${{nodeCount}}</p>
                    <p><strong>Total Edges:</strong> ${{edgeCount}}</p>
                    <p><strong>Node Types:</strong></p>
                    <ul>
                        ${{Object.entries(nodeTypes).map(([type, count]) => 
                            `<li>${{type.charAt(0).toUpperCase() + type.slice(1)}}: ${{count}}</li>`
                        ).join('')}}
                    </ul>
                `;
            }}
        }}

        // Initialize visualization
        const visualizer = new D3GraphVisualizer('graph-container', {width}, {height});
        
        // Load and render graph data
        const graphData = {graph_json};
        visualizer.render(graphData);
    </script>
</body>
</html>
"""
        
        return html_content
    
    def convert_graph_to_d3_format(self, graph_data) -> Dict[str, Any]:
        """Convert our graph format to D3.js format."""
        try:
            # Convert PyTorch Geometric format to D3 format
            if hasattr(graph_data, 'x') and hasattr(graph_data, 'edge_index'):
                nodes = []
                links = []
                
                num_nodes = graph_data.x.size(0)
                edge_index = graph_data.edge_index
                
                # Convert nodes
                for i in range(num_nodes):
                    features = graph_data.x[i].tolist()
                    node_type = self._determine_node_type(features)
                    label = self._get_node_label(features, i)
                    
                    nodes.append({
                        'id': i,
                        'label': label,
                        'type': node_type,
                        'features': features
                    })
                
                # Convert edges
                for i in range(edge_index.size(1)):
                    source = edge_index[0][i].item()
                    target = edge_index[1][i].item()
                    
                    links.append({
                        'source': source,
                        'target': target,
                        'id': f"{source}-{target}"
                    })
                
                return {'nodes': nodes, 'links': links}
            
            # Handle NetworkX format
            elif hasattr(graph_data, 'nodes') and hasattr(graph_data, 'edges'):
                nodes = []
                links = []
                
                for node_id, node_data in graph_data.nodes(data=True):
                    nodes.append({
                        'id': node_id,
                        'label': node_data.get('label', str(node_id)),
                        'type': node_data.get('type', 'unknown'),
                        'features': node_data.get('features', [])
                    })
                
                for edge in graph_data.edges():
                    links.append({
                        'source': edge[0],
                        'target': edge[1],
                        'id': f"{edge[0]}-{edge[1]}"
                    })
                
                return {'nodes': nodes, 'links': links}
            
            else:
                raise ValueError("Unsupported graph data format")
                
        except Exception as e:
            # Return mock data if conversion fails
            return self._generate_mock_graph_data()
    
    def _determine_node_type(self, features: List[float]) -> str:
        """Determine node type from features."""
        if not features:
            return 'unknown'
        
        first_feature = features[0]
        if first_feature > 0.8:
            return 'operator'
        elif first_feature > 0.5:
            return 'variable'
        elif first_feature > 0.2:
            return 'constant'
        else:
            return 'function'
    
    def _get_node_label(self, features: List[float], index: int) -> str:
        """Generate node label from features."""
        if not features:
            return f"N{index}"
        
        first_feature = features[0]
        if first_feature > 0.8:
            return ['+', '-', '*', '/', '='][index % 5]
        elif first_feature > 0.5:
            return f"x{index}"
        elif first_feature > 0.2:
            return str(index)
        else:
            return f"f{index}"
    
    def _generate_mock_graph_data(self) -> Dict[str, Any]:
        """Generate mock graph data for demonstration."""
        nodes = [
            {'id': 0, 'label': '2', 'type': 'constant', 'features': [0.3, 0.1, 0.2]},
            {'id': 1, 'label': '*', 'type': 'operator', 'features': [0.9, 0.8, 0.7]},
            {'id': 2, 'label': 'x', 'type': 'variable', 'features': [0.6, 0.5, 0.4]},
            {'id': 3, 'label': '+', 'type': 'operator', 'features': [0.85, 0.9, 0.8]},
            {'id': 4, 'label': '5', 'type': 'constant', 'features': [0.25, 0.15, 0.3]},
            {'id': 5, 'label': '=', 'type': 'equation', 'features': [0.95, 0.9, 0.85]},
            {'id': 6, 'label': '9', 'type': 'constant', 'features': [0.2, 0.1, 0.25]}
        ]
        
        links = [
            {'source': 0, 'target': 1, 'id': '0-1'},
            {'source': 1, 'target': 2, 'id': '1-2'},
            {'source': 1, 'target': 3, 'id': '1-3'},
            {'source': 3, 'target': 4, 'id': '3-4'},
            {'source': 3, 'target': 5, 'id': '3-5'},
            {'source': 5, 'target': 6, 'id': '5-6'}
        ]
        
        return {'nodes': nodes, 'links': links}
    
    def visualize_expression(self, expression: str, width: int = 800, height: int = 600) -> str:
        """Create D3.js visualization for an expression."""
        try:
            # Convert expression to graph
            graph_data = expression_to_graph(expression)
            
            # Convert to D3 format
            d3_data = self.convert_graph_to_d3_format(graph_data)
            
            # Create HTML
            html_content = self._create_visualization_html(d3_data, expression, width, height)
            
            return html_content
            
        except Exception as e:
            # Fallback to mock data
            d3_data = self._generate_mock_graph_data()
            html_content = self._create_visualization_html(d3_data, expression, width, height)
            return html_content
    
    def render_in_streamlit(self, expression: str, width: int = 800, height: int = 600):
        """Render D3.js visualization in Streamlit."""
        html_content = self.visualize_expression(expression, width, height)
        components.html(html_content, height=height + 100, scrolling=False)
    
    def create_standalone_visualization(self, expression: str, output_path: str = None):
        """Create a standalone HTML file with D3.js visualization."""
        html_content = self.visualize_expression(expression)
        
        if output_path is None:
            output_path = f"graph_visualization_{expression.replace('*', 'x').replace('/', 'div').replace('+', 'plus').replace('-', 'minus').replace('=', 'eq')}.html"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_path


def render_d3_graph(expression: str, width: int = 800, height: int = 600):
    """Convenience function to render D3.js graph in Streamlit."""
    visualizer = D3GraphVisualizer()
    visualizer.render_in_streamlit(expression, width, height)


def create_d3_visualization_file(expression: str, output_path: str = None) -> str:
    """Convenience function to create standalone D3.js visualization file."""
    visualizer = D3GraphVisualizer()
    return visualizer.create_standalone_visualization(expression, output_path) 