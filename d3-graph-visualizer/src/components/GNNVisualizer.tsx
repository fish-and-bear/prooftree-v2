"use client";

import React, { useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';
import { GNNSolver, GNNPrediction, AlgebraicStep } from '../utils/gnnSolver';

interface GNNVisualizerProps {
  expression: string;
  onStepComplete?: (step: AlgebraicStep) => void;
  onPrediction?: (prediction: GNNPrediction) => void;
  showConfidence?: boolean;
  showStepDetails?: boolean;
  educationalMode?: boolean;
}

interface GraphNode {
  id: string;
  label: string;
  type: 'expression' | 'operation' | 'result' | 'step';
  value?: string;
  confidence?: number;
  stepNumber?: number;
  explanation?: string;
  color?: string;
}

interface GraphEdge {
  source: string;
  target: string;
  type: 'transformation' | 'step' | 'operation';
  label?: string;
  weight?: number;
}

interface StepVisualization {
  stepNumber: number;
  beforeExpression: string;
  afterExpression: string;
  operation: string;
  explanation: string;
  confidence: number;
  graphNodes: GraphNode[];
  graphEdges: GraphEdge[];
}

export const GNNVisualizer: React.FC<GNNVisualizerProps> = ({ 
  expression, 
  onStepComplete, 
  onPrediction,
  showConfidence = true,
  showStepDetails = true,
  educationalMode = true
}) => {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [currentStep, setCurrentStep] = useState<number>(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [gnnSteps, setGnnSteps] = useState<AlgebraicStep[]>([]);
  const [predictions, setPredictions] = useState<GNNPrediction[]>([]);
  const [stepVisualizations, setStepVisualizations] = useState<StepVisualization[]>([]);
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [processingProgress, setProcessingProgress] = useState<number>(0);
  const [currentExpression, setCurrentExpression] = useState<string>(expression);
  const [solvingHistory, setSolvingHistory] = useState<StepVisualization[]>([]);
  
  const solver = new GNNSolver();

  // Enhanced feature extraction with educational focus
  const extractExpressionFeatures = (expr: string) => {
    const features = {
      hasVariables: /[a-zA-Z]/.test(expr),
      hasConstants: /[0-9]+/.test(expr),
      hasExponents: /\^/.test(expr),
      hasParentheses: /\(.*\)/.test(expr),
      hasFractions: /\/[^/]/.test(expr),
      hasEquality: /=/.test(expr),
      hasInequality: /[<>≤≥]/.test(expr),
      hasSquareRoot: /√/.test(expr),
      hasLogarithms: /log/.test(expr),
      variableCount: (expr.match(/[a-zA-Z]/g) || []).length,
      operatorCount: (expr.match(/[+\-*/^]/g) || []).length,
      complexity: expr.length,
      depth: (expr.match(/\(/g) || []).length,
      isLinear: !/\^/.test(expr) && !/√/.test(expr),
      isQuadratic: /\^2/.test(expr) || /x\*x/.test(expr),
      hasMultipleVariables: (expr.match(/[a-zA-Z]/g) || []).filter((v, i, arr) => arr.indexOf(v) === i).length > 1
    };

    return features;
  };

  // Generate step-by-step visualization graph
  const generateStepVisualization = (stepNumber: number, beforeExpr: string, afterExpr: string, operation: string, explanation: string, confidence: number): StepVisualization => {
    const nodes: GraphNode[] = [];
    const edges: GraphEdge[] = [];
    
    // Add step number node
    const stepNodeId = `step_${stepNumber}`;
    nodes.push({
      id: stepNodeId,
      label: `Step ${stepNumber}`,
      type: 'step',
      stepNumber: stepNumber,
      color: '#4299e1'
    });
    
    // Add before expression node
    const beforeNodeId = `before_${stepNumber}`;
    nodes.push({
      id: beforeNodeId,
      label: beforeExpr,
      type: 'expression',
      value: beforeExpr,
      color: '#f56565'
    });
    
    // Add operation node
    const operationNodeId = `operation_${stepNumber}`;
    nodes.push({
      id: operationNodeId,
      label: operation,
      type: 'operation',
      confidence: confidence,
      color: '#9f7aea'
    });
    
    // Add after expression node
    const afterNodeId = `after_${stepNumber}`;
    nodes.push({
      id: afterNodeId,
      label: afterExpr,
      type: 'result',
      value: afterExpr,
      color: '#48bb78'
    });
    
    // Add explanation node if in educational mode
    if (educationalMode) {
      const explanationNodeId = `explanation_${stepNumber}`;
      nodes.push({
        id: explanationNodeId,
        label: explanation,
        type: 'step',
        explanation: explanation,
        color: '#ed8936'
      });
      
      // Connect explanation to operation
      edges.push({
        source: operationNodeId,
        target: explanationNodeId,
        type: 'step',
        label: 'explains'
      });
    }
    
    // Connect nodes
    edges.push({
      source: stepNodeId,
      target: beforeNodeId,
      type: 'step',
      label: 'starts with'
    });
    
    edges.push({
      source: beforeNodeId,
      target: operationNodeId,
      type: 'transformation',
      label: 'apply'
    });
    
    edges.push({
      source: operationNodeId,
      target: afterNodeId,
      type: 'transformation',
      label: 'results in'
    });
    
    // Connect to previous step if exists
    if (stepNumber > 1) {
      edges.push({
        source: `after_${stepNumber - 1}`,
        target: beforeNodeId,
        type: 'step',
        label: 'continues from'
      });
    }
    
    return {
      stepNumber,
      beforeExpression: beforeExpr,
      afterExpression: afterExpr,
      operation,
      explanation,
      confidence,
      graphNodes: nodes,
      graphEdges: edges
    };
  };

  // Enhanced step-by-step solving with educational visualization
  const solveStepByStep = async (expr: string) => {
    setIsProcessing(true);
    setProcessingProgress(0);
    setSolvingHistory([]);
    
    let currentExpr = expr;
    let stepNumber = 1;
    
    while (stepNumber <= 10) { // Limit to prevent infinite loops
      setProcessingProgress((stepNumber - 1) * 10);
      
      // Check if solved
      if (solver.isSolved(currentExpr)) {
        break;
      }
      
      // Get prediction
      const prediction = solver.predictNextStep(currentExpr);
      setPredictions(prev => [...prev, prediction]);
      
      if (onPrediction) {
        onPrediction(prediction);
      }
      
      // Apply operation
      const step = solver.applyOperation(currentExpr, prediction);
      setGnnSteps(prev => [...prev, step]);
      
      if (onStepComplete) {
        onStepComplete(step);
      }
      
      // Generate visualization for this step
      const visualization = generateStepVisualization(
        stepNumber,
        currentExpr,
        step.expression,
        step.operation || 'UNKNOWN',
        step.explanation,
        prediction.confidence
      );
      
      setSolvingHistory(prev => [...prev, visualization]);
      setStepVisualizations(prev => [...prev, visualization]);
      
      // Update current expression
      currentExpr = step.expression;
      setCurrentExpression(currentExpr);
      
      // Check if operation was unsupported
      if (step.operation === 'UNSUPPORTED') {
        break;
      }
      
      stepNumber++;
      
      // Add delay for visualization
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
    
    setProcessingProgress(100);
    setIsProcessing(false);
  };

  // Render the step-by-step solving visualization
  useEffect(() => {
    if (!svgRef.current || solvingHistory.length === 0) return;
    
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    // Combine all nodes and edges from all steps
    const allNodes: GraphNode[] = [];
    const allEdges: GraphEdge[] = [];
    
    solvingHistory.forEach(step => {
      allNodes.push(...step.graphNodes);
      allEdges.push(...step.graphEdges);
    });

    const width = 1200;
    const height = 800;

    // Set up simulation with better layout for educational visualization
    const simulation = d3.forceSimulation(allNodes as any)
      .force('link', d3.forceLink(allEdges as any).id((d: any) => d.id).distance(120))
      .force('charge', d3.forceManyBody().strength(-500))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('x', d3.forceX().x((d: any) => {
        // Position nodes by step number
        if (d.stepNumber) {
          return d.stepNumber * 200 + 100;
        }
        return width / 2;
      }))
      .force('y', d3.forceY().y((d: any) => {
        // Position nodes by type
        switch (d.type) {
          case 'step': return 50;
          case 'expression': return 200;
          case 'operation': return 350;
          case 'result': return 500;
          default: return 650;
        }
      }))
      .force('collision', d3.forceCollide().radius(40));

    // Color scheme for educational visualization
    const nodeColors = {
      step: '#4299e1',
      expression: '#f56565',
      operation: '#9f7aea',
      result: '#48bb78'
    };

    // Draw edges with labels
    const link = svg.append('g')
      .attr('stroke', '#999')
      .attr('stroke-opacity', 0.6)
      .selectAll('line')
      .data(allEdges)
      .join('line')
      .attr('stroke-width', d => (d.weight || 1) * 2)
      .attr('marker-end', 'url(#arrow)');

    // Add arrow marker
    svg.append('defs').append('marker')
      .attr('id', 'arrow')
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 15)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-5L10,0L0,5')
      .attr('fill', '#999');

    // Draw nodes with enhanced styling
    const node = svg.append('g')
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)
      .selectAll('g')
      .data(allNodes)
      .join('g')
      .call(d3.drag<any, GraphNode>()
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
        }));

    // Add circles for nodes
    node.append('circle')
      .attr('r', d => {
        switch (d.type) {
          case 'step': return 25;
          case 'expression': return 35;
          case 'operation': return 30;
          case 'result': return 35;
          default: return 20;
        }
      })
      .attr('fill', d => d.color || nodeColors[d.type as keyof typeof nodeColors])
      .on('click', (event, d) => setSelectedNode(d))
      .on('mouseover', function(event, d) {
        d3.select(this).attr('stroke-width', 4);
        // Show tooltip
        const tooltip = d3.select('body').append('div')
          .attr('class', 'tooltip')
          .style('position', 'absolute')
          .style('background', 'rgba(0,0,0,0.8)')
          .style('color', 'white')
          .style('padding', '8px')
          .style('border-radius', '4px')
          .style('pointer-events', 'none');
        
        if (d.explanation) {
          tooltip.html(`<strong>${d.label}</strong><br/>${d.explanation}`);
        } else {
          tooltip.html(d.label);
        }
        
        tooltip.style('left', (event.pageX + 10) + 'px')
               .style('top', (event.pageY - 10) + 'px');
      })
      .on('mouseout', function() {
        d3.select(this).attr('stroke-width', 2);
        d3.selectAll('.tooltip').remove();
      });

    // Add labels
    node.append('text')
      .text(d => d.label.length > 20 ? d.label.substring(0, 17) + '...' : d.label)
      .attr('text-anchor', 'middle')
      .attr('dy', '0.35em')
      .attr('font-size', '12px')
      .attr('fill', 'white')
      .attr('font-weight', 'bold');

    // Add confidence indicators
    if (showConfidence) {
      node.filter(d => d.confidence !== undefined).append('circle')
        .attr('r', 8)
        .attr('cx', 25)
        .attr('cy', -25)
        .attr('fill', d => d3.interpolateRdYlGn(d.confidence || 0))
        .attr('stroke', 'white')
        .attr('stroke-width', 1);
    }

    // Update positions on simulation tick
    simulation.on('tick', () => {
      link
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y);

      node
        .attr('transform', (d: any) => `translate(${d.x},${d.y})`);
    });

  }, [solvingHistory, showConfidence]);

  // Start solving when expression changes
  useEffect(() => {
    if (expression && expression.trim()) {
      solveStepByStep(expression);
    }
  }, [expression]);

  return (
    <div className="gnn-visualizer">
      <div className="controls">
        <button 
          onClick={() => solveStepByStep(expression)}
          disabled={isProcessing}
          className="solve-button"
        >
          {isProcessing ? 'Solving...' : 'Solve Step by Step'}
        </button>
        
        {isProcessing && (
          <div className="progress-bar">
            <div 
              className="progress-fill" 
              style={{ width: `${processingProgress}%` }}
            />
            <span className="progress-text">{processingProgress}%</span>
          </div>
        )}
      </div>

      <div className="current-expression">
        <h3>Current Expression:</h3>
        <div className="expression-display">{currentExpression}</div>
      </div>

      <div className="visualization-container">
        <svg ref={svgRef} width="1200" height="800" />
      </div>

      {selectedNode && (
        <div className="node-details">
          <h4>Node Details:</h4>
          <p><strong>Type:</strong> {selectedNode.type}</p>
          <p><strong>Label:</strong> {selectedNode.label}</p>
          {selectedNode.confidence !== undefined && (
            <p><strong>Confidence:</strong> {(selectedNode.confidence * 100).toFixed(1)}%</p>
          )}
          {selectedNode.explanation && (
            <p><strong>Explanation:</strong> {selectedNode.explanation}</p>
          )}
        </div>
      )}

      {showStepDetails && solvingHistory.length > 0 && (
        <div className="step-details">
          <h3>Solving Steps:</h3>
          <div className="steps-list">
            {solvingHistory.map((step, index) => (
              <div key={index} className="step-item">
                <div className="step-header">
                  <span className="step-number">Step {step.stepNumber}</span>
                  <span className="operation">{step.operation}</span>
                  {showConfidence && (
                    <span className="confidence">{(step.confidence * 100).toFixed(1)}%</span>
                  )}
                </div>
                <div className="step-content">
                  <div className="expression-before">{step.beforeExpression}</div>
                  <div className="arrow">→</div>
                  <div className="expression-after">{step.afterExpression}</div>
                </div>
                <div className="explanation">{step.explanation}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      <style jsx>{`
        .gnn-visualizer {
          font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        .controls {
          margin-bottom: 20px;
          display: flex;
          align-items: center;
          gap: 20px;
        }
        
        .solve-button {
          padding: 10px 20px;
          background: #4299e1;
          color: white;
          border: none;
          border-radius: 5px;
          cursor: pointer;
          font-size: 16px;
        }
        
        .solve-button:disabled {
          background: #a0aec0;
          cursor: not-allowed;
        }
        
        .progress-bar {
          width: 200px;
          height: 20px;
          background: #e2e8f0;
          border-radius: 10px;
          overflow: hidden;
          position: relative;
        }
        
        .progress-fill {
          height: 100%;
          background: #48bb78;
          transition: width 0.3s ease;
        }
        
        .progress-text {
          position: absolute;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          font-size: 12px;
          color: #2d3748;
        }
        
        .current-expression {
          margin-bottom: 20px;
          padding: 15px;
          background: #f7fafc;
          border-radius: 8px;
        }
        
        .expression-display {
          font-family: 'Courier New', monospace;
          font-size: 18px;
          font-weight: bold;
          color: #2d3748;
        }
        
        .visualization-container {
          margin-bottom: 20px;
          border: 1px solid #e2e8f0;
          border-radius: 8px;
          overflow: hidden;
        }
        
        .node-details {
          margin-bottom: 20px;
          padding: 15px;
          background: #edf2f7;
          border-radius: 8px;
        }
        
        .step-details {
          margin-top: 20px;
        }
        
        .steps-list {
          display: flex;
          flex-direction: column;
          gap: 15px;
        }
        
        .step-item {
          padding: 15px;
          background: #f7fafc;
          border-radius: 8px;
          border-left: 4px solid #4299e1;
        }
        
        .step-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 10px;
        }
        
        .step-number {
          font-weight: bold;
          color: #2d3748;
        }
        
        .operation {
          background: #9f7aea;
          color: white;
          padding: 4px 8px;
          border-radius: 4px;
          font-size: 12px;
        }
        
        .confidence {
          background: #48bb78;
          color: white;
          padding: 4px 8px;
          border-radius: 4px;
          font-size: 12px;
        }
        
        .step-content {
          display: flex;
          align-items: center;
          gap: 15px;
          margin-bottom: 10px;
        }
        
        .expression-before, .expression-after {
          font-family: 'Courier New', monospace;
          padding: 8px;
          background: white;
          border-radius: 4px;
          border: 1px solid #e2e8f0;
          flex: 1;
        }
        
        .arrow {
          font-size: 20px;
          color: #4299e1;
          font-weight: bold;
        }
        
        .explanation {
          color: #4a5568;
          font-style: italic;
        }
      `}</style>
    </div>
  );
};

export default GNNVisualizer; 