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
}

interface GNNNode {
  id: string;
  type: 'input' | 'hidden' | 'output' | 'feature';
  label: string;
  value?: number;
  confidence?: number;
  layer: number;
  explanation?: string;
}

interface GNNEdge {
  source: string;
  target: string;
  weight: number;
  type: 'forward' | 'attention' | 'skip';
  attention?: number;
}

interface StepDetail {
  stepNumber: number;
  operation: string;
  beforeExpression: string;
  afterExpression: string;
  confidence: number;
  explanation: string;
  timeMs: number;
}

export const GNNVisualizer: React.FC<GNNVisualizerProps> = ({ 
  expression, 
  onStepComplete, 
  onPrediction,
  showConfidence = true,
  showStepDetails = true
}) => {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [currentStep, setCurrentStep] = useState<number>(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [gnnSteps, setGnnSteps] = useState<AlgebraicStep[]>([]);
  const [predictions, setPredictions] = useState<GNNPrediction[]>([]);
  const [stepDetails, setStepDetails] = useState<StepDetail[]>([]);
  const [selectedNode, setSelectedNode] = useState<GNNNode | null>(null);
  const [processingProgress, setProcessingProgress] = useState<number>(0);
  
  const solver = new GNNSolver();

  // Enhanced feature extraction with more detailed analysis
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

  // Generate enhanced GNN nodes and edges with attention mechanisms
  const generateGNNGraph = (expr: string) => {
    const nodes: GNNNode[] = [];
    const edges: GNNEdge[] = [];
    
    // Extract features
    const features = extractExpressionFeatures(expr);

    // Input layer (features) with explanations
    const featureExplanations = {
      hasVariables: "Contains variables (x, y, etc.)",
      hasConstants: "Contains numerical constants",
      hasExponents: "Contains exponential terms",
      hasParentheses: "Contains grouped expressions",
      hasFractions: "Contains fractional expressions",
      hasEquality: "Is an equation (contains =)",
      hasInequality: "Contains inequality operators",
      hasSquareRoot: "Contains square root operations",
      hasLogarithms: "Contains logarithmic functions",
      variableCount: "Number of variable occurrences",
      operatorCount: "Number of mathematical operators",
      complexity: "Expression length/complexity",
      depth: "Nesting depth of parentheses",
      isLinear: "Is a linear expression",
      isQuadratic: "Is a quadratic expression",
      hasMultipleVariables: "Contains multiple different variables"
    };

    Object.entries(features).forEach(([key, value], index) => {
      const nodeId = `feature_${index}`;
      nodes.push({
        id: nodeId,
        type: 'feature',
        label: key,
        value: typeof value === 'number' ? value : (value ? 1 : 0),
        confidence: typeof value === 'number' ? Math.min(value / 10, 1) : (value ? 0.8 : 0.2),
        layer: 0,
        explanation: featureExplanations[key as keyof typeof featureExplanations]
      });
    });

    // Hidden layers with attention mechanisms
    const hiddenLayerSize = 12;
    for (let layer = 1; layer <= 3; layer++) {
      for (let i = 0; i < hiddenLayerSize; i++) {
        const nodeId = `hidden_${layer}_${i}`;
        const activation = Math.random() * 0.6 + 0.2;
        nodes.push({
          id: nodeId,
          type: 'hidden',
          label: `H${layer}.${i}`,
          value: activation,
          confidence: activation,
          layer,
          explanation: `Hidden layer ${layer} neuron ${i} (activation: ${activation.toFixed(2)})`
        });
      }
    }

    // Output layer (possible operations) with confidence scores
    const operations = [
      { name: 'COMBINE_LIKE_TERMS', confidence: 0.85, explanation: 'Combine similar terms' },
      { name: 'DISTRIBUTE', confidence: 0.72, explanation: 'Apply distributive property' },
      { name: 'SIMPLIFY', confidence: 0.91, explanation: 'Simplify the expression' },
      { name: 'ADD_TO_BOTH_SIDES', confidence: 0.78, explanation: 'Add term to both sides' },
      { name: 'SUBTRACT_FROM_BOTH_SIDES', confidence: 0.76, explanation: 'Subtract term from both sides' },
      { name: 'MULTIPLY_BOTH_SIDES', confidence: 0.68, explanation: 'Multiply both sides by factor' },
      { name: 'DIVIDE_BOTH_SIDES', confidence: 0.73, explanation: 'Divide both sides by factor' },
      { name: 'MOVE_TERMS', confidence: 0.82, explanation: 'Move terms across equals sign' },
      { name: 'APPLY_QUADRATIC_FORMULA', confidence: 0.45, explanation: 'Apply quadratic formula' },
      { name: 'EXPAND', confidence: 0.79, explanation: 'Expand parentheses' },
      { name: 'FACTOR', confidence: 0.67, explanation: 'Factor the expression' },
      { name: 'SOLVE', confidence: 0.88, explanation: 'Solve for variable' }
    ];
    
    operations.forEach((op, index) => {
      const nodeId = `output_${index}`;
      nodes.push({
        id: nodeId,
        type: 'output',
        label: op.name,
        confidence: op.confidence,
        layer: 4,
        explanation: op.explanation
      });
    });

    // Generate edges with attention weights
    nodes.forEach(node => {
      if (node.layer < 4) {
        // Connect to next layer
        const nextLayerNodes = nodes.filter(n => n.layer === node.layer + 1);
        nextLayerNodes.forEach(target => {
          const attention = Math.random() * 0.8 + 0.2;
          edges.push({
            source: node.id,
            target: target.id,
            weight: attention,
            type: node.layer === 0 ? 'attention' : 'forward',
            attention
          });
        });
      }
    });

    return { nodes, edges };
  };

  // Enhanced GNN processing with step-by-step visualization
  const animateGNNProcessing = async (expr: string) => {
    setIsProcessing(true);
    setProcessingProgress(0);
    
    // Step 1: Feature extraction (20%)
    await new Promise(resolve => setTimeout(resolve, 300));
    setProcessingProgress(20);
    
    // Step 2: Graph encoding (40%)
    await new Promise(resolve => setTimeout(resolve, 400));
    setProcessingProgress(40);
    
    // Step 3: GNN prediction (60%)
    const prediction = solver.predictNextStep(expr);
    setPredictions([prediction]);
    
    if (onPrediction) {
      onPrediction(prediction);
    }
    
    await new Promise(resolve => setTimeout(resolve, 300));
    setProcessingProgress(60);
    
    // Step 4: Operation application (80%)
    const step = solver.applyOperation(expr, prediction);
    setGnnSteps([step]);
    
    if (onStepComplete) {
      onStepComplete(step);
    }
    
    await new Promise(resolve => setTimeout(resolve, 200));
    setProcessingProgress(80);
    
    // Step 5: Verification (100%)
    const stepDetail: StepDetail = {
      stepNumber: currentStep + 1,
      operation: prediction.operation,
      beforeExpression: expr,
      afterExpression: step.expression,
      confidence: prediction.confidence,
      explanation: step.explanation,
      timeMs: Date.now()
    };
    
    setStepDetails(prev => [...prev, stepDetail]);
    setProcessingProgress(100);
    
    await new Promise(resolve => setTimeout(resolve, 100));
    setIsProcessing(false);
    setProcessingProgress(0);
  };

  // Render GNN graph
  useEffect(() => {
    if (!svgRef.current || !expression) return;
    
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const { nodes, edges } = generateGNNGraph(expression);
    const width = 800;
    const height = 500;

    // Set up simulation
    const simulation = d3.forceSimulation(nodes as any)
      .force('link', d3.forceLink(edges as any).id((d: any) => d.id).distance(80))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('x', d3.forceX().x((d: any) => d.layer * 150 + 100))
      .force('collision', d3.forceCollide().radius(25));

    // Color scheme
    const nodeColors = {
      feature: '#4299e1',
      hidden: '#9f7aea',
      output: '#f56565',
      input: '#48bb78'
    };

    // Draw edges
    svg.append('g')
      .attr('stroke', '#999')
      .attr('stroke-opacity', 0.6)
      .selectAll('line')
      .data(edges)
      .join('line')
      .attr('stroke-width', d => d.weight * 3);

    // Draw nodes
    const node = svg.append('g')
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)
      .selectAll('g')
      .data(nodes)
      .join('g')
      .call(d3.drag<any, GNNNode>()
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
      );

    // Node shapes
    node.each(function(d) {
      const g = d3.select(this);
      const color = nodeColors[d.type as keyof typeof nodeColors];
      const radius = d.type === 'output' ? 20 : 15;
      
      g.append('circle')
        .attr('r', radius)
        .attr('fill', color)
        .attr('opacity', d.value ? 0.7 + d.value * 0.3 : 0.8);

      // Add confidence indicator for output nodes
      if (d.confidence && d.type === 'output') {
        g.append('circle')
          .attr('r', radius * (d.confidence || 0))
          .attr('fill', 'none')
          .attr('stroke', '#fbbf24')
          .attr('stroke-width', 2)
          .attr('opacity', 0.6);
      }
    });

    // Node labels
    node.append('text')
      .text(d => d.label)
      .attr('text-anchor', 'middle')
      .attr('dy', '.35em')
      .attr('font-size', 10)
      .attr('fill', '#fff')
      .attr('pointer-events', 'none');

    // Layer labels
    const layers = ['Features', 'Hidden 1', 'Hidden 2', 'Operations'];
    layers.forEach((label, i) => {
      svg.append('text')
        .text(label)
        .attr('x', i * 150 + 100)
        .attr('y', 30)
        .attr('text-anchor', 'middle')
        .attr('font-size', 14)
        .attr('font-weight', 'bold')
        .attr('fill', '#374151');
    });

    // Animation
    simulation.on('tick', () => {
      svg.selectAll('line')
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y);
      
      node.attr('transform', (d: any) => `translate(${d.x},${d.y})`);
    });

    return () => {
      simulation.stop();
    };
  }, [expression]);

  return (
    <div className="w-full">
      <div className="mb-4 p-4 bg-gray-50 rounded-lg">
        <h3 className="text-lg font-semibold text-gray-800 mb-2">🧠 GNN Processing</h3>
        <div className="flex items-center gap-4">
          <button
            onClick={() => animateGNNProcessing(expression)}
            disabled={isProcessing || !expression}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg font-semibold shadow hover:bg-blue-700 transition disabled:opacity-50"
          >
            {isProcessing ? 'Processing...' : 'Run GNN Analysis'}
          </button>
          {predictions.length > 0 && (
            <div className="text-sm text-gray-600">
              <strong>Predicted:</strong> {predictions[0].operation} 
              (confidence: {(predictions[0].confidence * 100).toFixed(1)}%)
            </div>
          )}
        </div>
        {predictions.length > 0 && (
          <div className="mt-2 text-sm text-gray-700">
            <strong>Reasoning:</strong> {predictions[0].reasoning}
          </div>
        )}
      </div>
      
      <div className="w-full flex justify-center">
        <svg 
          ref={svgRef} 
          width={800} 
          height={500} 
          className="bg-white rounded-xl shadow-lg border border-gray-200" 
        />
      </div>
      
      {gnnSteps.length > 0 && (
        <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-lg">
          <h4 className="font-semibold text-green-800 mb-2">GNN Step Result:</h4>
          <div className="text-sm text-green-700">
            <div><strong>Operation:</strong> {gnnSteps[0].operation}</div>
            <div><strong>Before:</strong> {gnnSteps[0].beforeExpression}</div>
            <div><strong>After:</strong> {gnnSteps[0].afterExpression}</div>
            <div><strong>Confidence:</strong> {(gnnSteps[0].confidence * 100).toFixed(1)}%</div>
          </div>
        </div>
      )}
    </div>
  );
};

export default GNNVisualizer; 