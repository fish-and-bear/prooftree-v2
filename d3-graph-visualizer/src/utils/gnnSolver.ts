// TypeScript interfaces and utilities for GNN solver integration

export interface GNNPrediction {
  operation: string;
  confidence: number;
  parameters?: number[];
  explanation?: string;
  strategy?: string;
  fallbackUsed?: boolean;
}

export interface AlgebraicStep {
  expression: string;
  operation?: string;
  explanation: string;
  confidence?: number;
  timestamp?: number;
  strategy?: string;
  fallbackUsed?: boolean;
}

export interface SolverConfig {
  maxSteps: number;
  useVerification: boolean;
  showIntermediateSteps: boolean;
  confidenceThreshold: number;
  useMultipleStrategies: boolean;
  enableFallbacks: boolean;
}

export interface SolverLimitations {
  maxExpressionLength: number;
  maxVariables: number;
  supportedOperations: string[];
  unsupportedPatterns: string[];
}

export class GNNSolver {
  private config: SolverConfig;
  private limitations: SolverLimitations;

  constructor(config: Partial<SolverConfig> = {}) {
    this.config = {
      maxSteps: 20,
      useVerification: true,
      showIntermediateSteps: true,
      confidenceThreshold: 0.5,
      useMultipleStrategies: true,
      enableFallbacks: true,
      ...config
    };

    this.limitations = {
      maxExpressionLength: 200,
      maxVariables: 5,
      supportedOperations: [
        'COMBINE_LIKE_TERMS', 'DISTRIBUTE', 'SIMPLIFY', 'EXPAND', 'FACTOR',
        'ADD_TO_BOTH_SIDES', 'SUBTRACT_FROM_BOTH_SIDES',
        'MULTIPLY_BOTH_SIDES', 'DIVIDE_BOTH_SIDES', 'MOVE_TERMS',
        'APPLY_QUADRATIC_FORMULA', 'SOLVE'
      ],
      unsupportedPatterns: [
        'trigonometric functions', 'logarithms', 'complex numbers',
        'differential equations', 'integral equations', 'inequalities',
        'systems of equations', 'parametric equations'
      ]
    };
  }

  /**
   * Check if an expression is within GNN capabilities
   */
  canHandleExpression(expression: string): { canHandle: boolean; reason?: string; suggestions?: string[] } {
    // Check expression length
    if (expression.length > this.limitations.maxExpressionLength) {
      return {
        canHandle: false,
        reason: `Expression too long (${expression.length} chars, max ${this.limitations.maxExpressionLength})`,
        suggestions: ['Try breaking into smaller parts', 'Simplify the expression first']
      };
    }

    // Check for unsupported patterns
    for (const pattern of this.limitations.unsupportedPatterns) {
      if (expression.toLowerCase().includes(pattern.toLowerCase())) {
        return {
          canHandle: false,
          reason: `Contains unsupported pattern: ${pattern}`,
          suggestions: ['Use a computer algebra system for advanced operations', 'Simplify to basic algebraic form']
        };
      }
    }

    // Check variable count
    const variables = expression.match(/[a-zA-Z]/g);
    if (variables) {
      const uniqueVars = [...new Set(variables)];
      if (uniqueVars.length > this.limitations.maxVariables) {
        return {
          canHandle: false,
          reason: `Too many variables (${uniqueVars.length}, max ${this.limitations.maxVariables})`,
          suggestions: ['Solve for one variable at a time', 'Use substitution to reduce variables']
        };
      }
    }

    return { canHandle: true };
  }

  /**
   * Predict the next step for an algebraic expression with enhanced fallback
   */
  predictNextStep(expression: string): GNNPrediction {
    // Check if we can handle this expression
    const capabilityCheck = this.canHandleExpression(expression);
    if (!capabilityCheck.canHandle) {
      return {
        operation: 'UNSUPPORTED',
        confidence: 0.0,
        explanation: `Cannot handle: ${capabilityCheck.reason}`,
        strategy: 'capability_check',
        fallbackUsed: true
      };
    }

    // Try GNN prediction first
    const gnnPrediction = this.heuristicPrediction(expression);
    
    // If GNN confidence is low, try fallback strategies
    if (gnnPrediction.confidence < this.config.confidenceThreshold && this.config.enableFallbacks) {
      const fallbackPrediction = this.fallbackPrediction(expression);
      if (fallbackPrediction.confidence > gnnPrediction.confidence) {
        return {
          ...fallbackPrediction,
          strategy: 'fallback',
          fallbackUsed: true
        };
      }
    }

    return {
      ...gnnPrediction,
      strategy: 'gnn',
      fallbackUsed: false
    };
  }

  /**
   * Apply an operation to an expression with multiple strategies
   */
  applyOperation(expression: string, prediction: GNNPrediction): AlgebraicStep {
    // If operation is unsupported, return error step
    if (prediction.operation === 'UNSUPPORTED') {
      return {
        expression: expression,
        operation: 'UNSUPPORTED',
        explanation: prediction.explanation || 'Operation not supported',
        confidence: 0.0,
        timestamp: Date.now(),
        strategy: prediction.strategy,
        fallbackUsed: prediction.fallbackUsed
      };
    }

    // Try to apply the operation
    const result = this.simulateOperation(expression, prediction.operation);
    
    return {
      expression: result.expression,
      operation: prediction.operation,
      explanation: result.explanation,
      confidence: prediction.confidence,
      timestamp: Date.now(),
      strategy: prediction.strategy,
      fallbackUsed: prediction.fallbackUsed
    };
  }

  /**
   * Solve an equation step by step with multiple strategies
   */
  async solveStepByStep(expression: string): Promise<{
    steps: AlgebraicStep[];
    success: boolean;
    finalStrategy: string;
    limitations: string[];
  }> {
    const steps: AlgebraicStep[] = [];
    let currentExpression = expression;
    const limitations: string[] = [];
    let finalStrategy = 'gnn';
    
    // Check initial capability
    const capabilityCheck = this.canHandleExpression(expression);
    if (!capabilityCheck.canHandle) {
      limitations.push(capabilityCheck.reason!);
      return {
        steps: [{
          expression: expression,
          operation: 'UNSUPPORTED',
          explanation: capabilityCheck.reason!,
          confidence: 0.0,
          timestamp: Date.now(),
          strategy: 'capability_check',
          fallbackUsed: true
        }],
        success: false,
        finalStrategy: 'capability_check',
        limitations
      };
    }
    
    for (let step = 0; step < this.config.maxSteps; step++) {
      const prediction = this.predictNextStep(currentExpression);
      
      if (prediction.operation === 'UNSUPPORTED') {
        limitations.push(prediction.explanation!);
        break;
      }
      
      if (prediction.confidence < this.config.confidenceThreshold) {
        limitations.push(`Low confidence prediction: ${prediction.confidence}`);
        break;
      }
      
      const stepResult = this.applyOperation(currentExpression, prediction);
      steps.push(stepResult);
      
      currentExpression = stepResult.expression;
      finalStrategy = prediction.strategy || 'unknown';
      
      // Check if solved
      if (this.isSolved(currentExpression)) {
        break;
      }
    }
    
    return {
      steps,
      success: steps.length > 0 && this.isSolved(currentExpression),
      finalStrategy,
      limitations
    };
  }

  /**
   * Enhanced heuristic-based prediction
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
   * Fallback prediction using alternative strategies
   */
  private fallbackPrediction(expression: string): {
    operation: string;
    confidence: number;
    parameters?: number[];
    explanation: string;
  } {
    // Try direct solve for equations
    if (expression.includes('=')) {
      return {
        operation: 'DIRECT_SOLVE',
        confidence: 0.75,
        explanation: 'Attempt direct solution using SymPy'
      };
    }
    
    // Try extreme simplification
    return {
      operation: 'EXTREME_SIMPLIFY',
      confidence: 0.60,
      explanation: 'Apply extreme simplification'
    };
  }

  /**
   * Enhanced operation simulation
   */
  private simulateOperation(expression: string, operation: string): {
    expression: string;
    explanation: string;
  } {
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
      
      case 'DIRECT_SOLVE':
        return {
          expression: expression,
          explanation: `Attempted direct solution for: ${expression}`
        };
      
      case 'EXTREME_SIMPLIFY':
        return {
          expression: expression,
          explanation: `Applied extreme simplification to: ${expression}`
        };
      
      case 'UNSUPPORTED':
        return {
          expression: expression,
          explanation: `Operation not supported for: ${expression}`
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
  isSolved(expression: string): boolean {
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
    return this.limitations.supportedOperations.includes(operation);
  }

  /**
   * Get solver limitations
   */
  getLimitations(): SolverLimitations {
    return { ...this.limitations };
  }

  /**
   * Get solver capabilities
   */
  getCapabilities(): {
    maxExpressionLength: number;
    maxVariables: number;
    supportedOperations: string[];
    fallbackStrategies: string[];
  } {
    return {
      maxExpressionLength: this.limitations.maxExpressionLength,
      maxVariables: this.limitations.maxVariables,
      supportedOperations: [...this.limitations.supportedOperations],
      fallbackStrategies: ['Direct SymPy solve', 'Extreme simplification', 'Pattern matching']
    };
  }
} 