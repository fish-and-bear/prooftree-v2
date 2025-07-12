"use client";

import React, { useState, useRef } from 'react';
import { parse, MathNode, ConstantNode, evaluate } from 'mathjs';
import { astToGraph } from '../utils/astToGraph';
import { GNNSolver, AlgebraicStep, GNNPrediction } from '../utils/gnnSolver';
import type { NodeType } from '../components/GraphVisualizer';
import '../app/globals.css';
import GraphVisualizer from '../components/GraphVisualizer';
import GNNVisualizer from '../components/GNNVisualizer';
import cloneDeep from 'lodash.clonedeep';

const EXAMPLES = [
  '2*x + 5 = 9',
  'x^2 + 2*x + 1 = 0',
  '3*(x + 2) = 15',
  'x/2 + 3 = 7',
  '2*x + 3*x = 10',
  'x^2 - 4 = 0',
  'x^2 + 5*x + 6 = 0',
  '3*x - 7 = 2*x + 3',
];

// Helper function to create user-friendly error messages
function createUserFriendlyError(error: any, expression: string): { message: string; details?: string; position?: number; suggestion?: string } {
  const errorMessage = error.message || 'Unknown error';

  // Smart suggestion for exponentiation
  let suggestion: string | undefined = undefined;
  if (expression.includes('**')) {
    suggestion = 'It looks like you used "**" for exponentiation. Please use "^" instead (e.g., x^2).';
  }
  // Smart suggestion for equations
  if (errorMessage.includes('Invalid left hand side of assignment operator') && expression.includes('=')) {
    suggestion = 'To write equations, use equal(..., ...) (e.g., equal(2*x + 5, 9)).';
  }

  // Check if it's a parse error from mathjs
  if (errorMessage.includes('Value expected') || errorMessage.includes('Unexpected end') || errorMessage.includes('Unexpected character')) {
    // Extract position if available
    const positionMatch = errorMessage.match(/char (\d+)/);
    const position = positionMatch ? parseInt(positionMatch[1]) : undefined;

    let message = 'Please enter a valid algebraic expression.';
    let details = errorMessage;

    if (position !== undefined && position < expression.length) {
      const char = expression[position];
      message = `Invalid character "${char}" at position ${position + 1}. Please check your expression.`;

      // Show the expression with error highlighting
      const before = expression.substring(0, position);
      const after = expression.substring(position + 1);
      details = `${before}[${char}]${after}`;
    } else if (expression.trim() === '') {
      message = 'Please enter an algebraic expression.';
      details = 'Try one of the example expressions below.';
    }

    return { message, details, position, suggestion };
  }

  return { message: 'Invalid expression. Please check your syntax.', details: errorMessage, suggestion };
}

// Helper to pause for React state updates
function wait(ms: number) {
  return new Promise(res => setTimeout(res, ms));
}

// Helper to check if a node can be evaluated (has no variables)
function canEvaluate(node: MathNode): boolean {
  try {
    // Check if the node contains any variables
    const symbols = new Set<string>();
    const constants = new Set(['pi', 'e', 'i', 'Infinity', 'NaN']);
    
    function collectSymbols(n: any) {
      if ((n as any).isSymbolNode) {
        const symbolName = (n as any).name;
        // Only treat non-constants as variables
        if (!constants.has(symbolName)) {
          symbols.add(symbolName);
        }
      }
      if ((n as any).args) {
        for (const child of (n as any).args) {
          collectSymbols(child);
        }
      }
    }
    collectSymbols(node);
    
    // Can evaluate if no variables found
    return symbols.size === 0;
  } catch {
    return false;
  }
}

// Helper to safely evaluate a MathJS node
function safeEvaluate(node: MathNode): number | boolean | null {
  try {
    if (!canEvaluate(node)) {
      return null; // Cannot evaluate expressions with variables
    }
    
    // Create a scope with common mathematical constants
    const scope = {
      pi: Math.PI,
      e: Math.E,
      i: 'i', // Complex number
      Infinity: Infinity,
      NaN: NaN
    };
    
    // Use mathjs evaluate function with scope - convert node to string first
    const result = evaluate(node.toString(), scope);
    
    // Check if result is a valid number or boolean
    if (typeof result === 'number' && !isNaN(result) && isFinite(result)) {
      return result;
    } else if (typeof result === 'boolean') {
      return result;
    }
    
    return null;
  } catch (error) {
    console.warn('Evaluation failed:', error);
    return null;
  }
}

// Enhanced evaluation that works with any expression
function smartEvaluate(node: MathNode): { result: number | boolean | null; canEvaluate: boolean; reason?: string } {
  try {
    // First try to evaluate the entire node
    if (canEvaluate(node)) {
      const scope = {
        pi: Math.PI,
        e: Math.E,
        i: 'i',
        Infinity: Infinity,
        NaN: NaN
      };
      
      const result = evaluate(node.toString(), scope);
      
      if (typeof result === 'number' && !isNaN(result) && isFinite(result)) {
        return { result, canEvaluate: true };
      } else if (typeof result === 'boolean') {
        return { result, canEvaluate: true };
      }
    }
    
    // If we can't evaluate the whole node, check if it's an operator with constant arguments
    if ((node as any).isOperatorNode && (node as any).args) {
      const args = (node as any).args;
      const op = (node as any).op;
      
      // For binary operators, check if both arguments are constants
      if (args.length === 2 && op && ['+', '-', '*', '/', '^'].includes(op)) {
        const leftCanEval = canEvaluate(args[0]);
        const rightCanEval = canEvaluate(args[1]);
        
        if (leftCanEval && rightCanEval) {
          // Both arguments are constants, we can evaluate
          const scope = {
            pi: Math.PI,
            e: Math.E,
            i: 'i',
            Infinity: Infinity,
            NaN: NaN
          };
          
          const result = evaluate(node.toString(), scope);
          
          if (typeof result === 'number' && !isNaN(result) && isFinite(result)) {
            return { result, canEvaluate: true };
          } else if (typeof result === 'boolean') {
            return { result, canEvaluate: true };
          }
        } else if (leftCanEval || rightCanEval) {
          // One argument is constant, provide partial evaluation info
          const constantSide = leftCanEval ? 'left' : 'right';
          const constantValue = leftCanEval ? args[0].toString() : args[1].toString();
          return { 
            result: null, 
            canEvaluate: false, 
            reason: `Can partially evaluate: ${constantSide} side is constant (${constantValue})` 
          };
        }
      }
      
      // For unary operators, check if the argument is constant
      if (args.length === 1 && op && ['sin', 'cos', 'tan', 'log', 'exp', 'sqrt'].includes(op)) {
        if (canEvaluate(args[0])) {
          const scope = {
            pi: Math.PI,
            e: Math.E,
            i: 'i',
            Infinity: Infinity,
            NaN: NaN
          };
          
          const result = evaluate(node.toString(), scope);
          
          if (typeof result === 'number' && !isNaN(result) && isFinite(result)) {
            return { result, canEvaluate: true };
          }
        }
      }
    }
    
    // Check if this is a constant node
    if ((node as any).isConstantNode) {
      return { result: (node as any).value, canEvaluate: true };
    }
    
    // Check if this is a symbol node (variable)
    if ((node as any).isSymbolNode) {
      const symbolName = (node as any).name;
      const constants = new Set(['pi', 'e', 'i', 'Infinity', 'NaN']);
      
      if (constants.has(symbolName)) {
        // It's a mathematical constant
        const scope = {
          pi: Math.PI,
          e: Math.E,
          i: 'i',
          Infinity: Infinity,
          NaN: NaN
        };
        
        const result = evaluate(symbolName, scope);
        return { result, canEvaluate: true };
      } else {
        return { result: null, canEvaluate: false, reason: `Variable: ${symbolName}` };
      }
    }
    
    return { result: null, canEvaluate: false, reason: 'Cannot evaluate this expression' };
    
  } catch (error) {
    console.warn('Smart evaluation failed:', error);
    return { result: null, canEvaluate: false, reason: `Evaluation error: ${(error as Error).message}` };
  }
}

export default function Home() {
  const [expression, setExpression] = useState('2*x + 5 = 9');
  const [error, setError] = useState<{ message: string; details?: string; position?: number; suggestion?: string } | null>(null);
  const [showErrorDetails, setShowErrorDetails] = useState(false);
  const [ast, setAst] = useState<MathNode | null>(null);
  const [highlightedNodeId, setHighlightedNodeId] = useState<string | number | undefined>(undefined);
  const [isAnimating, setIsAnimating] = useState(false);
  const [evaluationMessage, setEvaluationMessage] = useState<string>('');
  const [gnnSteps, setGnnSteps] = useState<AlgebraicStep[]>([]);
  const [currentGnnStep, setCurrentGnnStep] = useState<number>(0);
  const [isGnnSolving, setIsGnnSolving] = useState(false);
  const [showGnnVisualizer, setShowGnnVisualizer] = useState(false);
  const animationLock = useRef(false);
  
  const gnnSolver = new GNNSolver();

  // Parse expression to AST on change
  React.useEffect(() => {
    if (!expression.trim()) {
      setError(null);
      setAst(null);
      setEvaluationMessage('');
      return;
    }
    try {
      setError(null);
      let expr = expression.trim();
      // Auto-convert 'a = b' to 'equal(a, b)' for MathJS compatibility
      if (/^[^=]+=[^=]+$/.test(expr)) {
        const [left, right] = expr.split('=');
        expr = `equal(${left.trim()}, ${right.trim()})`;
      }
      const parsedAst: MathNode = parse(expr);
      setAst(parsedAst);
      setEvaluationMessage('');
    } catch (e: any) {
      const friendlyError = createUserFriendlyError(e, expression);
      setError(friendlyError);
      setAst(null);
      setEvaluationMessage('');
    }
  }, [expression]);

  // Always derive graph and idToAst from AST
  const graph = React.useMemo(() => {
    if (!ast) return { nodes: [], links: [], idToAst: {} };
    return astToGraph(ast);
  }, [ast]);

  const handleExpressionChange = (newExpression: string) => {
    setExpression(newExpression);
    setShowErrorDetails(false);
    setGnnSteps([]);
    setCurrentGnnStep(0);
  };

  // GNN-powered step-by-step solving
  const handleGNNSolve = async () => {
    if (isGnnSolving) return;
    
    setIsGnnSolving(true);
    setEvaluationMessage('🧠 GNN analyzing expression...');
    
    try {
      // Get all steps from GNN solver
      const steps = gnnSolver.solveStepByStep(expression);
      setGnnSteps(steps);
      setCurrentGnnStep(0);
      
      if (steps.length === 0) {
        setEvaluationMessage('No solution steps found for this expression.');
        setIsGnnSolving(false);
        return;
      }
      
      setEvaluationMessage(`🧠 GNN found ${steps.length} solution steps. Starting step-by-step solving...`);
      setShowGnnVisualizer(true);
      
      // Animate through each step with better feedback
      for (let i = 0; i < steps.length; i++) {
        setCurrentGnnStep(i);
        const step = steps[i];
        setEvaluationMessage(`Step ${i + 1}/${steps.length}: ${step.operation} - ${step.description} (${(step.confidence * 100).toFixed(0)}% confidence)`);
        
        // Update the expression display to show the current step
        if (step.afterExpression !== step.beforeExpression) {
          setExpression(step.afterExpression);
        }
        
        await wait(2000); // Longer pause to see the transformation
      }
      
      // Show final result
      if (steps.length > 0) {
        const lastStep = steps[steps.length - 1];
        if (lastStep.operation === 'SOLUTION_FOUND') {
          setEvaluationMessage(`✅ Solution found: ${lastStep.afterExpression}`);
        } else {
          setEvaluationMessage(`✅ GNN solution complete! Final expression: ${lastStep.afterExpression}`);
        }
      }
      
    } catch (error) {
      console.error('GNN solving error:', error);
      setEvaluationMessage('An error occurred during GNN solving.');
    }
    
    setIsGnnSolving(false);
  };

  // Handle GNN step completion
  const handleGNNStepComplete = (step: AlgebraicStep) => {
    setEvaluationMessage(`GNN applied: ${step.operation} - ${step.description}`);
  };

  // Handle GNN prediction
  const handleGNNPrediction = (prediction: GNNPrediction) => {
    setEvaluationMessage(`GNN predicts: ${prediction.operation} (${(prediction.confidence * 100).toFixed(1)}% confidence)`);
  };

  // Stepwise, animated evaluation with GNN intelligence
  const handleNodeClick = async (node: NodeType) => {
    if (isAnimating || animationLock.current) return;
    if (!ast || !graph.idToAst) return;
    if (node.type !== 'operator') return;
    
    const astNode = graph.idToAst[node.id as number];
    if (!astNode) return;
    
    setIsAnimating(true);
    animationLock.current = true;
    setEvaluationMessage('🧠 Analyzing operator...');

    try {
      // Get the expression represented by this operator node
      const nodeExpression = astNode.toString();
      
      setEvaluationMessage(`Analyzing: ${nodeExpression}`);
      await wait(600);
      
      // Try to evaluate the operator node
      const value = smartEvaluate(astNode);
      
      if (value.canEvaluate && value.result !== null) {
        // The node can be evaluated - replace it with the result
        setEvaluationMessage(`✅ Evaluating: ${nodeExpression} = ${value.result}`);
        await wait(800);
        
        // Replace with constant node
        let currentAst = cloneDeep(ast);
        let replaced = false;
        
        function walk(parent: any, key: string | null) {
          if (parent && typeof parent === 'object') {
            for (const k in parent) {
              if (parent[k] === astNode) {
                parent[k] = new ConstantNode(value.result);
                replaced = true;
                return;
              } else if (parent[k] && typeof parent[k] === 'object') {
                walk(parent[k], k);
                if (replaced) return;
              }
            }
          }
        }
        
        walk(currentAst, null);
        
        if (replaced) {
          setAst(currentAst);
          setEvaluationMessage(`✅ Successfully evaluated: ${nodeExpression} → ${value.result}`);
        } else {
          setEvaluationMessage(`⚠️ Could not update AST, but evaluation was: ${nodeExpression} = ${value.result}`);
        }
        
        await wait(600);
      } else {
        // The node cannot be evaluated - provide helpful feedback
        setEvaluationMessage(`🧠 Cannot evaluate: ${nodeExpression}`);
        await wait(400);
        
        if (value.reason) {
          setEvaluationMessage(`💡 ${value.reason}`);
        } else {
          setEvaluationMessage(`💡 This expression contains variables and cannot be evaluated further`);
        }
        
        await wait(800);
      }
      
    } catch (error) {
      console.error('Evaluation error:', error);
      setEvaluationMessage('An error occurred during evaluation.');
    }
    
    setHighlightedNodeId(undefined);
    setIsAnimating(false);
    animationLock.current = false;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-200 to-purple-100 flex flex-col items-center py-8 px-2">
      <div className="w-full max-w-4xl bg-white rounded-2xl shadow-xl p-8 mt-8">
        <h1 className="text-3xl font-bold text-center text-indigo-700 mb-2">🧮 Graph Neural Algebra Tutor</h1>
        <p className="text-center text-lg text-gray-500 mb-6">World-class D3.js Visualization of Algebraic Expressions</p>
        <div className="flex flex-col sm:flex-row gap-4 items-center justify-center mb-6">
          <div className="w-full sm:w-2/3 relative">
            <input
              className={`w-full px-4 py-2 border-2 rounded-lg focus:outline-none focus:ring-2 text-lg shadow transition-colors ${
                error ? 'border-red-400 focus:ring-red-400' : 'border-indigo-300 focus:ring-indigo-400'
              }`}
              type="text"
              value={expression}
              onChange={e => handleExpressionChange(e.target.value)}
              placeholder="Enter an algebraic expression (e.g., x^2 + 3*x + 2)"
              aria-label="Algebraic expression input"
            />
            {error?.position !== undefined && (
              <div className="absolute top-full left-0 right-0 mt-1 text-xs text-red-600 font-mono">
                {' '.repeat(error.position)}↑ Error here
              </div>
            )}
          </div>
          <div className="flex gap-2">
            <button
              className="px-4 py-2 bg-indigo-600 text-white rounded-lg font-semibold shadow hover:bg-indigo-700 transition"
              onClick={() => handleExpressionChange('')}
              title="Clear input"
            >
              Clear
            </button>
            <button
              className={`px-4 py-2 rounded-lg font-semibold shadow transition ${
                isGnnSolving 
                  ? 'bg-gray-400 text-white cursor-not-allowed' 
                  : 'bg-green-600 text-white hover:bg-green-700'
              }`}
              onClick={handleGNNSolve}
              disabled={isGnnSolving || !expression.trim()}
              title="Solve with GNN"
            >
              {isGnnSolving ? '🧠 Solving...' : '🧠 GNN Solve'}
            </button>
          </div>
        </div>
        <div className="flex flex-wrap gap-2 justify-center mb-4">
          {EXAMPLES.map((ex, i) => (
            <button
              key={i}
              className="px-3 py-1 bg-indigo-100 text-indigo-700 rounded-full border border-indigo-200 hover:bg-indigo-200 transition text-sm font-mono"
              onClick={() => handleExpressionChange(ex)}
            >
              {ex}
            </button>
          ))}
        </div>
        {error && (
          <div className="bg-red-50 border border-red-200 text-red-800 px-4 py-3 rounded-lg mb-4">
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <div className="font-semibold text-red-900 mb-1">⚠️ {error.message}</div>
                {error.suggestion && (
                  <div className="text-xs text-indigo-700 bg-indigo-50 rounded px-2 py-1 mb-2 border border-indigo-200 inline-block">
                    💡 {error.suggestion}
                  </div>
                )}
                {error.details && (
                  <div className="text-sm">
                    <button
                      onClick={() => setShowErrorDetails(!showErrorDetails)}
                      className="text-red-600 hover:text-red-800 underline text-xs"
                    >
                      {showErrorDetails ? 'Hide' : 'Show'} technical details
                    </button>
                    {showErrorDetails && (
                      <div className="mt-2 p-2 bg-red-100 rounded font-mono text-xs">
                        {error.details}
                      </div>
                    )}
                  </div>
                )}
              </div>
              <button
                onClick={() => setError(null)}
                className="text-red-400 hover:text-red-600 ml-2"
                title="Dismiss error"
              >
                ✕
              </button>
            </div>
          </div>
        )}
        {evaluationMessage && (
          <div className="bg-blue-50 border border-blue-200 text-blue-800 px-4 py-3 rounded-lg mb-4">
            <div className="flex items-center">
              <div className="flex-1">
                <div className="font-semibold text-blue-900">🔄 {evaluationMessage}</div>
              </div>
              {isAnimating && (
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
              )}
            </div>
          </div>
        )}
        <div className="w-full flex flex-col items-center">
          <div className="w-full max-w-3xl">
            <GraphVisualizer nodes={graph.nodes} links={graph.links} width={800} height={420} onNodeClick={handleNodeClick} highlightedNodeId={highlightedNodeId} />
          </div>
        </div>
        
        {/* GNN Visualizer */}
        {showGnnVisualizer && (
          <div className="w-full mt-8">
            <GNNVisualizer 
              expression={expression}
              onStepComplete={handleGNNStepComplete}
              onPrediction={handleGNNPrediction}
            />
          </div>
        )}
        
        {/* GNN Solution Steps */}
        {gnnSteps.length > 0 && (
          <div className="w-full mt-6">
            <h3 className="text-lg font-semibold text-indigo-700 mb-4">🧠 GNN Solution Steps</h3>
            <div className="space-y-3">
              {gnnSteps.map((step, index) => (
                <div 
                  key={index}
                  className={`p-4 rounded-lg border-2 transition-all ${
                    index === currentGnnStep 
                      ? 'border-green-500 bg-green-50' 
                      : index < currentGnnStep 
                        ? 'border-gray-300 bg-gray-50' 
                        : 'border-gray-200 bg-white'
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-3">
                      <div className={`w-8 h-8 rounded-full flex items-center justify-center text-white font-bold text-sm ${
                        index === currentGnnStep 
                          ? 'bg-green-500' 
                          : index < currentGnnStep 
                            ? 'bg-gray-400' 
                            : 'bg-gray-300'
                      }`}>
                        {index + 1}
                      </div>
                      <div className="font-semibold text-gray-800">{step.operation}</div>
                      <div className="text-sm text-gray-500">({(step.confidence * 100).toFixed(1)}% confidence)</div>
                    </div>
                    {index === currentGnnStep && (
                      <div className="animate-pulse text-green-600">🔄 Processing...</div>
                    )}
                  </div>
                  <div className="ml-11">
                    <div className="text-sm text-gray-600 mb-2">{step.reasoning}</div>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <span className="font-medium text-gray-700">Before:</span>
                        <div className="font-mono bg-gray-100 p-2 rounded mt-1">{step.beforeExpression}</div>
                      </div>
                      <div>
                        <span className="font-medium text-gray-700">After:</span>
                        <div className="font-mono bg-green-100 p-2 rounded mt-1">{step.afterExpression}</div>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
        <div className="flex flex-col md:flex-row gap-8 mt-8 justify-between">
          <div className="flex-1">
            <h2 className="text-lg font-semibold text-indigo-700 mb-2">📊 Graph Statistics</h2>
            <ul className="text-gray-700 text-base">
              <li>Nodes: <b>{graph.nodes.length}</b></li>
              <li>Edges: <b>{graph.links.length}</b></li>
              <li>Node Types:
                <ul className="ml-4 mt-1">
                  {['operator','variable','constant','function','equation','unknown'].map(type => (
                    <li key={type} className="flex items-center gap-2">
                      <span className={`inline-block w-4 h-4 rounded-full ${type === 'operator' ? 'bg-orange-400' : type === 'variable' ? 'bg-green-500' : type === 'constant' ? 'bg-blue-500' : type === 'function' ? 'bg-purple-400' : type === 'equation' ? 'bg-red-500' : 'bg-gray-400'}`}></span>
                      <span className="capitalize">{type}</span>: <b>{graph.nodes.filter(n => n.type === type).length}</b>
                    </li>
                  ))}
                </ul>
              </li>
            </ul>
          </div>
          <div className="flex-1">
            <h2 className="text-lg font-semibold text-indigo-700 mb-2">🗺️ Legend</h2>
            <ul className="text-base text-gray-700 space-y-1">
              <li><span className="inline-block w-4 h-4 rounded-full bg-orange-400 mr-2"></span>Operator (+, -, *, /, =)</li>
              <li><span className="inline-block w-4 h-4 rounded-full bg-green-500 mr-2"></span>Variable (x, y, z)</li>
              <li><span className="inline-block w-4 h-4 rounded-full bg-blue-500 mr-2"></span>Constant (numbers)</li>
              <li><span className="inline-block w-4 h-4 rounded-full bg-purple-400 mr-2"></span>Function (sin, cos, etc.)</li>
              <li><span className="inline-block w-4 h-4 rounded-full bg-red-500 mr-2"></span>Equation (=)</li>
              <li><span className="inline-block w-4 h-4 rounded-full bg-gray-400 mr-2"></span>Unknown</li>
            </ul>
          </div>
        </div>
        <div className="mt-6 p-4 bg-indigo-50 rounded-lg border border-indigo-200">
          <h3 className="text-lg font-semibold text-indigo-700 mb-2">🎯 How to Use</h3>
          <ul className="text-gray-700 text-sm space-y-1">
            <li>• <strong>🧠 GNN Solve</strong> - Click to let AI solve equations step-by-step with confidence scores</li>
            <li>• <strong>Click on operator nodes</strong> (orange circles) to evaluate constant expressions</li>
            <li>• <strong>Watch the GNN visualizer</strong> to see how AI analyzes and predicts next steps</li>
            <li>• <strong>Follow solution steps</strong> with detailed reasoning and confidence levels</li>
            <li>• <strong>Try the example equations</strong> to see different types of algebraic problems</li>
          </ul>
        </div>
      </div>
      <footer className="mt-12 text-center text-gray-400 text-sm">
        &copy; {new Date().getFullYear()} Graph Neural Algebra Tutor &mdash; D3.js + Next.js + TypeScript
      </footer>
    </div>
  );
}
