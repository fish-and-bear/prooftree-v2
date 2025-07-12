// TypeScript interfaces and utilities for GNN solver integration

export interface GNNPrediction {
  operation: string;
  confidence: number;
  parameters?: number[];
  explanation?: string;
}

export interface AlgebraicStep {
  expression: string;
  operation?: string;
  explanation: string;
  confidence?: number;
  timestamp?: number;
}

export interface SolverConfig {
  maxSteps: number;
  useVerification: boolean;
  showIntermediateSteps: boolean;
  confidenceThreshold: number;
}

export class GNNSolver {
  private config: SolverConfig;

  constructor(config: Partial<SolverConfig> = {}) {
    this.config = {
      maxSteps: 20,
      useVerification: true,
      showIntermediateSteps: true,
      confidenceThreshold: 0.5,
      ...config
    };
  }

  /**
   * Predict the next step for an algebraic expression
   */
  predictNextStep(expression: string): GNNPrediction {
    // This would typically call the Python backend
    // For now, we'll simulate the prediction
    
    const operations = [
      'COMBINE_LIKE_TERMS',
      'DISTRIBUTE', 
      'SIMPLIFY',
      'ADD_TO_BOTH_SIDES',
      'SUBTRACT_FROM_BOTH_SIDES',
      'MULTIPLY_BOTH_SIDES',
      'DIVIDE_BOTH_SIDES',
      'MOVE_TERMS',
      'EXPAND',
      'FACTOR'
    ];

    // Simple heuristic-based prediction
    const prediction = this.heuristicPrediction(expression);
    
    return {
      operation: prediction.operation,
      confidence: prediction.confidence,
      parameters: prediction.parameters,
      explanation: prediction.explanation
    };
  }

  /**
   * Apply an operation to an expression
   */
  applyOperation(expression: string, prediction: GNNPrediction): AlgebraicStep {
    // This would typically call the Python backend
    // For now, we'll simulate the operation
    
    const result = this.simulateOperation(expression, prediction.operation);
    
    return {
      expression: result.expression,
      operation: prediction.operation,
      explanation: result.explanation,
      confidence: prediction.confidence,
      timestamp: Date.now()
    };
  }

  /**
   * Solve an equation step by step
   */
  async solveStepByStep(expression: string): Promise<AlgebraicStep[]> {
    const steps: AlgebraicStep[] = [];
    let currentExpression = expression;
    
    for (let step = 0; step < this.config.maxSteps; step++) {
      const prediction = this.predictNextStep(currentExpression);
      
      if (prediction.confidence < this.config.confidenceThreshold) {
        break;
      }
      
      const stepResult = this.applyOperation(currentExpression, prediction);
      steps.push(stepResult);
      
      currentExpression = stepResult.expression;
      
      // Check if solved
      if (this.isSolved(currentExpression)) {
        break;
      }
    }
    
    return steps;
  }

  /**
   * Simple heuristic-based prediction
   */
  private heuristicPrediction(expression: string): {
    operation: string;
    confidence: number;
    parameters?: number[];
    explanation: string;
  } {
    // Check for equations
    if (expression.includes('=')) {
      if (expression.includes('(')) {
        return {
          operation: 'EXPAND',
          confidence: 0.85,
          explanation: 'Expand parentheses in equation'
        };
      }
      
      if (expression.includes('+') || expression.includes('-')) {
        return {
          operation: 'MOVE_TERMS',
          confidence: 0.78,
          explanation: 'Move terms to isolate variable'
        };
      }
      
      return {
        operation: 'SIMPLIFY',
        confidence: 0.72,
        explanation: 'Simplify both sides'
      };
    }
    
    // Check for expressions
    if (expression.includes('(')) {
      return {
        operation: 'EXPAND',
        confidence: 0.82,
        explanation: 'Expand parentheses'
      };
    }
    
    if (expression.includes('+') || expression.includes('-')) {
      return {
        operation: 'COMBINE_LIKE_TERMS',
        confidence: 0.88,
        explanation: 'Combine like terms'
      };
    }
    
    return {
      operation: 'SIMPLIFY',
      confidence: 0.65,
      explanation: 'Simplify expression'
    };
  }

  /**
   * Simulate operation application
   */
  private simulateOperation(expression: string, operation: string): {
    expression: string;
    explanation: string;
  } {
    // This is a simplified simulation
    // In a real implementation, this would call the Python backend
    
    switch (operation) {
      case 'SIMPLIFY':
        return {
          expression: expression.replace(/\s+/g, ''),
          explanation: `Simplified: ${expression}`
        };
      
      case 'EXPAND':
        return {
          expression: expression.replace(/\(/g, '').replace(/\)/g, ''),
          explanation: `Expanded: ${expression}`
        };
      
      case 'COMBINE_LIKE_TERMS':
        return {
          expression: expression,
          explanation: `Combined like terms in: ${expression}`
        };
      
      case 'MOVE_TERMS':
        return {
          expression: expression,
          explanation: `Moved terms in: ${expression}`
        };
      
      default:
        return {
          expression: expression,
          explanation: `Applied ${operation} to: ${expression}`
        };
    }
  }

  /**
   * Check if expression is solved
   */
  private isSolved(expression: string): boolean {
    // Simple heuristic: check if it's in the form x = value
    const match = expression.match(/^([a-zA-Z])\s*=\s*([^=]+)$/);
    return match !== null;
  }

  /**
   * Get confidence score for a prediction
   */
  getConfidenceScore(expression: string, operation: string): number {
    // Simple confidence scoring based on expression features
    let confidence = 0.5;
    
    if (expression.includes('=')) confidence += 0.1;
    if (expression.includes('(')) confidence += 0.1;
    if (expression.includes('+') || expression.includes('-')) confidence += 0.1;
    if (expression.includes('*') || expression.includes('/')) confidence += 0.1;
    
    return Math.min(confidence, 0.95);
  }

  /**
   * Validate an operation
   */
  validateOperation(expression: string, operation: string): boolean {
    const validOperations = [
      'SIMPLIFY', 'EXPAND', 'COMBINE_LIKE_TERMS', 'MOVE_TERMS',
      'ADD_TO_BOTH_SIDES', 'SUBTRACT_FROM_BOTH_SIDES',
      'MULTIPLY_BOTH_SIDES', 'DIVIDE_BOTH_SIDES'
    ];
    
    return validOperations.includes(operation);
  }
} 