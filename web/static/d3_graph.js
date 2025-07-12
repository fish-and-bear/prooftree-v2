// D3.js Graph Visualization for Mathematical Expressions
// This module provides interactive graph visualization for algebraic expressions

class D3GraphVisualizer {
    constructor(containerId, width = 800, height = 600) {
        this.containerId = containerId;
        this.width = width;
        this.height = height;
        this.svg = null;
        this.simulation = null;
        this.nodes = [];
        this.links = [];
        this.nodeTypes = {
            'operator': { color: '#ff7f0e', size: 25, symbol: 'circle' },
            'variable': { color: '#2ca02c', size: 20, symbol: 'diamond' },
            'constant': { color: '#1f77b4', size: 18, symbol: 'circle' },
            'function': { color: '#9467bd', size: 22, symbol: 'square' },
            'equation': { color: '#d62728', size: 30, symbol: 'rect' },
            'unknown': { color: '#7f7f7f', size: 15, symbol: 'circle' }
        };
        this.init();
    }

    init() {
        // Clear existing content
        const container = document.getElementById(this.containerId);
        if (container) {
            container.innerHTML = '';
        }

        // Create SVG
        this.svg = d3.select(`#${this.containerId}`)
            .append('svg')
            .attr('width', this.width)
            .attr('height', this.height)
            .attr('viewBox', `0 0 ${this.width} ${this.height}`)
            .style('background-color', '#f8f9fa')
            .style('border', '1px solid #dee2e6')
            .style('border-radius', '8px');

        // Add zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on('zoom', (event) => {
                this.svg.select('g').attr('transform', event.transform);
            });

        this.svg.call(zoom);

        // Create main group for graph elements
        this.svg.append('g').attr('class', 'graph-container');

        // Add legend
        this.createLegend();
    }

    createLegend() {
        const legend = this.svg.append('g')
            .attr('class', 'legend')
            .attr('transform', `translate(10, 10)`);

        let yOffset = 0;
        Object.entries(this.nodeTypes).forEach(([type, config]) => {
            const legendItem = legend.append('g')
                .attr('transform', `translate(0, ${yOffset})`);

            // Node symbol
            if (config.symbol === 'circle') {
                legendItem.append('circle')
                    .attr('r', config.size / 2)
                    .attr('fill', config.color)
                    .attr('stroke', '#333')
                    .attr('stroke-width', 1);
            } else if (config.symbol === 'diamond') {
                const size = config.size / 2;
                legendItem.append('polygon')
                    .attr('points', `0,-${size} ${size},0 0,${size} -${size},0`)
                    .attr('fill', config.color)
                    .attr('stroke', '#333')
                    .attr('stroke-width', 1);
            } else if (config.symbol === 'square') {
                const size = config.size / 2;
                legendItem.append('rect')
                    .attr('width', config.size)
                    .attr('height', config.size)
                    .attr('x', -size)
                    .attr('y', -size)
                    .attr('fill', config.color)
                    .attr('stroke', '#333')
                    .attr('stroke-width', 1);
            } else if (config.symbol === 'rect') {
                const size = config.size / 2;
                legendItem.append('rect')
                    .attr('width', config.size * 1.5)
                    .attr('height', config.size)
                    .attr('x', -size * 0.75)
                    .attr('y', -size)
                    .attr('fill', config.color)
                    .attr('stroke', '#333')
                    .attr('stroke-width', 1);
            }

            // Label
            legendItem.append('text')
                .attr('x', config.size + 5)
                .attr('y', 4)
                .attr('font-size', '12px')
                .attr('font-weight', 'bold')
                .text(type.charAt(0).toUpperCase() + type.slice(1));

            yOffset += 25;
        });
    }

    updateGraph(graphData) {
        this.nodes = graphData.nodes || [];
        this.links = graphData.links || [];

        // Clear existing graph
        this.svg.select('.graph-container').selectAll('*').remove();

        if (this.simulation) {
            this.simulation.stop();
        }

        // Create force simulation
        this.simulation = d3.forceSimulation(this.nodes)
            .force('link', d3.forceLink(this.links).id(d => d.id).distance(80))
            .force('charge', d3.forceManyBody().strength(-300))
            .force('center', d3.forceCenter(this.width / 2, this.height / 2))
            .force('collision', d3.forceCollide().radius(d => this.nodeTypes[d.type]?.size || 20));

        // Create links
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

        // Create arrow marker
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

        // Create nodes
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

        // Add node shapes
        node.each((d, i, nodes) => {
            const nodeGroup = d3.select(nodes[i]);
            const config = this.nodeTypes[d.type] || this.nodeTypes.unknown;

            if (config.symbol === 'circle') {
                nodeGroup.append('circle')
                    .attr('r', config.size)
                    .attr('fill', config.color)
                    .attr('stroke', '#333')
                    .attr('stroke-width', 2)
                    .style('cursor', 'pointer');
            } else if (config.symbol === 'diamond') {
                nodeGroup.append('polygon')
                    .attr('points', `0,-${config.size} ${config.size},0 0,${config.size} -${config.size},0`)
                    .attr('fill', config.color)
                    .attr('stroke', '#333')
                    .attr('stroke-width', 2)
                    .style('cursor', 'pointer');
            } else if (config.symbol === 'square') {
                nodeGroup.append('rect')
                    .attr('width', config.size * 2)
                    .attr('height', config.size * 2)
                    .attr('x', -config.size)
                    .attr('y', -config.size)
                    .attr('fill', config.color)
                    .attr('stroke', '#333')
                    .attr('stroke-width', 2)
                    .style('cursor', 'pointer');
            } else if (config.symbol === 'rect') {
                nodeGroup.append('rect')
                    .attr('width', config.size * 3)
                    .attr('height', config.size * 2)
                    .attr('x', -config.size * 1.5)
                    .attr('y', -config.size)
                    .attr('fill', config.color)
                    .attr('stroke', '#333')
                    .attr('stroke-width', 2)
                    .style('cursor', 'pointer');
            }
        });

        // Add node labels
        node.append('text')
            .attr('text-anchor', 'middle')
            .attr('dy', '.35em')
            .attr('font-size', '12px')
            .attr('font-weight', 'bold')
            .attr('fill', 'white')
            .text(d => d.label)
            .style('pointer-events', 'none');

        // Add hover effects
        node.on('mouseover', this.handleMouseOver.bind(this))
            .on('mouseout', this.handleMouseOut.bind(this))
            .on('click', this.handleNodeClick.bind(this));

        // Update positions on simulation tick
        this.simulation.on('tick', () => {
            link
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);

            node.attr('transform', d => `translate(${d.x},${d.y})`);
        });

        // Add animation
        this.animateNodes();
    }

    animateNodes() {
        // Add entrance animation
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
    }

    handleMouseOver(event, d) {
        // Highlight connected nodes and links
        const connectedLinks = this.links.filter(link => 
            link.source.id === d.id || link.target.id === d.id
        );
        const connectedNodes = this.nodes.filter(node => 
            connectedLinks.some(link => 
                link.source.id === node.id || link.target.id === node.id
            )
        );

        // Dim other elements
        this.svg.selectAll('.node').style('opacity', 0.3);
        this.svg.selectAll('.links line').style('opacity', 0.1);

        // Highlight connected elements
        this.svg.selectAll('.node').filter(node => 
            connectedNodes.some(n => n.id === node.id)
        ).style('opacity', 1);

        this.svg.selectAll('.links line').filter(link => 
            connectedLinks.some(l => 
                (l.source.id === link.source.id && l.target.id === link.target.id) ||
                (l.source.id === link.target.id && l.target.id === link.source.id)
            )
        ).style('opacity', 0.8);

        // Show tooltip
        this.showTooltip(event, d);
    }

    handleMouseOut(event, d) {
        // Restore opacity
        this.svg.selectAll('.node').style('opacity', 1);
        this.svg.selectAll('.links line').style('opacity', 0.6);

        // Hide tooltip
        this.hideTooltip();
    }

    handleNodeClick(event, d) {
        // Toggle node selection
        const node = d3.select(event.currentTarget);
        const isSelected = node.classed('selected');

        // Clear previous selections
        this.svg.selectAll('.node').classed('selected', false);
        this.svg.selectAll('.node circle, .node polygon, .node rect')
            .attr('stroke-width', 2);

        if (!isSelected) {
            // Select this node
            node.classed('selected', true);
            node.select('circle, polygon, rect').attr('stroke-width', 4);
            
            // Show node details
            this.showNodeDetails(d);
        }
    }

    showTooltip(event, d) {
        const tooltip = d3.select('body').select('.tooltip');
        
        if (tooltip.empty()) {
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
        }

        d3.select('.tooltip')
            .style('left', (event.pageX + 10) + 'px')
            .style('top', (event.pageY - 10) + 'px')
            .html(`
                <strong>${d.label}</strong><br>
                Type: ${d.type}<br>
                ID: ${d.id}
            `);
    }

    hideTooltip() {
        d3.select('.tooltip').remove();
    }

    showNodeDetails(d) {
        // Create or update details panel
        let detailsPanel = d3.select('#node-details');
        
        if (detailsPanel.empty()) {
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
        }

        detailsPanel.html(`
            <h4 style="margin: 0 0 10px 0; color: #333;">Node Details</h4>
            <p><strong>Label:</strong> ${d.label}</p>
            <p><strong>Type:</strong> ${d.type}</p>
            <p><strong>ID:</strong> ${d.id}</p>
            ${d.features ? `<p><strong>Features:</strong> ${JSON.stringify(d.features)}</p>` : ''}
            <button onclick="document.getElementById('node-details').remove()" 
                    style="background: #dc3545; color: white; border: none; padding: 5px 10px; border-radius: 4px; cursor: pointer;">
                Close
            </button>
        `);
    }

    dragstarted(event, d) {
        if (!event.active) this.simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }

    dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }

    dragended(event, d) {
        if (!event.active) this.simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }

    // Method to convert graph data from our format to D3 format
    convertGraphData(graphData) {
        const nodes = [];
        const links = [];
        const nodeMap = new Map();

        // Convert nodes
        if (graphData.x && graphData.edge_index) {
            // PyTorch Geometric format
            const numNodes = graphData.x.size(0);
            const edgeIndex = graphData.edge_index;

            for (let i = 0; i < numNodes; i++) {
                const features = graphData.x[i];
                const nodeType = this.determineNodeType(features);
                const label = this.getNodeLabel(features, i);
                
                nodes.push({
                    id: i,
                    label: label,
                    type: nodeType,
                    features: features.tolist(),
                    x: Math.random() * this.width,
                    y: Math.random() * this.height
                });
                nodeMap.set(i, nodes[i]);
            }

            // Convert edges
            for (let i = 0; i < edgeIndex.size(1); i++) {
                const source = edgeIndex[0][i].item();
                const target = edgeIndex[1][i].item();
                
                links.push({
                    source: source,
                    target: target,
                    id: `${source}-${target}`
                });
            }
        } else if (graphData.nodes && graphData.edges) {
            // NetworkX format
            graphData.nodes.forEach((node, i) => {
                nodes.push({
                    id: node.id || i,
                    label: node.label || node.id || i,
                    type: node.type || 'unknown',
                    features: node.features || [],
                    x: Math.random() * this.width,
                    y: Math.random() * this.height
                });
                nodeMap.set(node.id || i, nodes[i]);
            });

            graphData.edges.forEach((edge, i) => {
                links.push({
                    source: edge.source,
                    target: edge.target,
                    id: `${edge.source}-${edge.target}`
                });
            });
        }

        return { nodes, links };
    }

    determineNodeType(features) {
        // Simple heuristic based on feature values
        if (features.dim() === 0) return 'unknown';
        
        const featureArray = features.tolist();
        
        // This is a simplified heuristic - in practice, you'd want more sophisticated logic
        if (featureArray.length > 0) {
            const firstFeature = featureArray[0];
            if (firstFeature > 0.8) return 'operator';
            if (firstFeature > 0.5) return 'variable';
            if (firstFeature > 0.2) return 'constant';
            return 'function';
        }
        
        return 'unknown';
    }

    getNodeLabel(features, index) {
        // Simplified label generation
        const featureArray = features.tolist();
        if (featureArray.length > 0) {
            const firstFeature = featureArray[0];
            if (firstFeature > 0.8) return ['+', '-', '*', '/', '='][index % 5];
            if (firstFeature > 0.5) return `x${index}`;
            if (firstFeature > 0.2) return index.toString();
            return `f${index}`;
        }
        return `N${index}`;
    }

    // Public method to update the graph
    render(graphData) {
        const d3Data = this.convertGraphData(graphData);
        this.updateGraph(d3Data);
    }

    // Method to resize the visualization
    resize(width, height) {
        this.width = width;
        this.height = height;
        this.svg
            .attr('width', width)
            .attr('height', height);
        
        if (this.simulation) {
            this.simulation
                .force('center', d3.forceCenter(width / 2, height / 2))
                .restart();
        }
    }

    // Method to clear the visualization
    clear() {
        if (this.simulation) {
            this.simulation.stop();
        }
        this.svg.select('.graph-container').selectAll('*').remove();
        this.nodes = [];
        this.links = [];
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = D3GraphVisualizer;
} 