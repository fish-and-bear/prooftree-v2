"use client";

import React, { useRef, useEffect } from 'react';
import * as d3 from 'd3';

export type NodeType = {
  id: string | number;
  label: string;
  type: 'operator' | 'variable' | 'constant' | 'function' | 'equation' | 'unknown';
};

export type LinkType = {
  source: string | number;
  target: string | number;
};

export interface GraphVisualizerProps {
  nodes: NodeType[];
  links: LinkType[];
  width?: number;
  height?: number;
  onNodeClick?: (node: NodeType) => void;
  highlightedNodeId?: string | number;
}

const nodeColors: Record<NodeType['type'], string> = {
  operator: '#ff7f0e',
  variable: '#2ca02c',
  constant: '#1f77b4',
  function: '#9467bd',
  equation: '#d62728',
  unknown: '#7f7f7f',
};

const nodeShapes: Record<NodeType['type'], 'circle' | 'rect' | 'diamond' | 'square'> = {
  operator: 'circle',
  variable: 'diamond',
  constant: 'circle',
  function: 'square',
  equation: 'rect',
  unknown: 'circle',
};

export const GraphVisualizer: React.FC<GraphVisualizerProps> = ({ nodes, links, width = 800, height = 500, onNodeClick, highlightedNodeId }) => {
  const svgRef = useRef<SVGSVGElement | null>(null);

  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    // Set up simulation
    const simulation = d3.forceSimulation(nodes as any)
      .force('link', d3.forceLink(links as any).id((d: any) => d.id).distance(100))
      .force('charge', d3.forceManyBody().strength(-350))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(40));

    // Draw links
    svg.append('g')
      .attr('stroke', '#999')
      .attr('stroke-opacity', 0.6)
      .selectAll('line')
      .data(links)
      .join('line')
      .attr('stroke-width', 2);

    // Draw nodes
    const node = svg.append('g')
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)
      .selectAll('g')
      .data(nodes)
      .join('g')
      .call(d3.drag<any, NodeType>()
        .on('start', (event, d) => {
          if (!event.active) simulation.alphaTarget(0.3).restart();
          (d as any).fx = (d as any).x;
          (d as any).fy = (d as any).y;
        })
        .on('drag', (event, d) => {
          (d as any).fx = event.x;
          (d as any).fy = event.y;
        })
        .on('end', (event, d) => {
          if (!event.active) simulation.alphaTarget(0);
          (d as any).fx = null;
          (d as any).fy = null;
        })
      )
      .on('click', function(event, d) {
        event.stopPropagation(); // Prevent event bubbling
        if (onNodeClick) {
          onNodeClick(d);
        }
        // Enhanced visual feedback: highlight node briefly with better animation
        const nodeElement = d3.select(this);
        const shapeElements = nodeElement.selectAll('circle, rect, polygon');
        
        // Store original stroke values
        const originalStroke = shapeElements.attr('stroke');
        const originalStrokeWidth = shapeElements.attr('stroke-width');
        
        // Animate to highlight
        shapeElements
          .transition().duration(150)
          .attr('stroke', '#6366f1')
          .attr('stroke-width', 6)
          .transition().duration(600)
          .attr('stroke', originalStroke)
          .attr('stroke-width', originalStrokeWidth);
      });

    // Node shapes
    node.each(function (d) {
      const g = d3.select(this);
      const color = nodeColors[d.type];
      const shape = nodeShapes[d.type];
      // Highlight if this is the highlighted node
      const isHighlighted = highlightedNodeId !== undefined && d.id === highlightedNodeId;
      if (shape === 'circle') {
        g.append('circle')
          .attr('r', 28)
          .attr('fill', color)
          .attr('filter', isHighlighted ? 'url(#glow)' : null)
          .attr('stroke', isHighlighted ? '#6366f1' : '#fff')
          .attr('stroke-width', isHighlighted ? 8 : 2);
      } else if (shape === 'diamond') {
        g.append('polygon')
          .attr('points', '0,-28 28,0 0,28 -28,0')
          .attr('fill', color)
          .attr('filter', isHighlighted ? 'url(#glow)' : null)
          .attr('stroke', isHighlighted ? '#6366f1' : '#fff')
          .attr('stroke-width', isHighlighted ? 8 : 2);
      } else if (shape === 'square') {
        g.append('rect')
          .attr('x', -22)
          .attr('y', -22)
          .attr('width', 44)
          .attr('height', 44)
          .attr('fill', color)
          .attr('filter', isHighlighted ? 'url(#glow)' : null)
          .attr('stroke', isHighlighted ? '#6366f1' : '#fff')
          .attr('stroke-width', isHighlighted ? 8 : 2);
      } else if (shape === 'rect') {
        g.append('rect')
          .attr('x', -32)
          .attr('y', -18)
          .attr('width', 64)
          .attr('height', 36)
          .attr('fill', color)
          .attr('filter', isHighlighted ? 'url(#glow)' : null)
          .attr('stroke', isHighlighted ? '#6366f1' : '#fff')
          .attr('stroke-width', isHighlighted ? 8 : 2);
      }
    });

    // Add SVG filter for glow effect
    if (highlightedNodeId !== undefined) {
      svg.append('defs').append('filter')
        .attr('id', 'glow')
        .html(`
          <feGaussianBlur stdDeviation="4.5" result="coloredBlur"/>
          <feMerge>
            <feMergeNode in="coloredBlur"/>
            <feMergeNode in="SourceGraphic"/>
          </feMerge>
        `);
    }

    // Node labels
    node.append('text')
      .text(d => d.label)
      .attr('text-anchor', 'middle')
      .attr('dy', '.35em')
      .attr('font-size', 18)
      .attr('fill', '#fff')
      .attr('pointer-events', 'none');

    // Tooltip
    const tooltip = d3.select('body').append('div')
      .attr('class', 'd3-tooltip')
      .style('position', 'absolute')
      .style('z-index', '10')
      .style('visibility', 'hidden')
      .style('background', 'rgba(0,0,0,0.8)')
      .style('color', '#fff')
      .style('padding', '8px 12px')
      .style('border-radius', '6px')
      .style('font-size', '14px');

    node.on('mouseover', (event, d) => {
      tooltip.html(`<b>${d.label}</b><br/>Type: ${d.type}`)
        .style('visibility', 'visible');
    })
      .on('mousemove', (event) => {
        tooltip.style('top', (event.pageY - 10) + 'px')
          .style('left', (event.pageX + 10) + 'px');
      })
      .on('mouseout', () => {
        tooltip.style('visibility', 'hidden');
      });

    simulation.on('tick', () => {
      svg.selectAll('line')
        .attr('x1', d => (d as any).source.x)
        .attr('y1', d => (d as any).source.y)
        .attr('x2', d => (d as any).target.x)
        .attr('y2', d => (d as any).target.y);
      node.attr('transform', d => `translate(${(d as any).x},${(d as any).y})`);
    });

    return () => {
      tooltip.remove();
      simulation.stop();
    };
  }, [nodes, links, width, height, onNodeClick, highlightedNodeId]);

  return (
    <div className="w-full flex justify-center items-center">
      <svg ref={svgRef} width={width} height={height} className="bg-gray-50 rounded-xl shadow-lg border border-gray-200" />
    </div>
  );
};

export default GraphVisualizer; 