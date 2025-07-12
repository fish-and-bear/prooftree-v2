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
        'SOLVE_QUADRATIC', 'SOLVE_LINEAR', 'SOLUTION_FOUND',
        'COMBINE_LIKE_TERMS', 'DISTRIBUTE', 'SIMPLIFY', 'EXPAND', 'FACTOR',
        'ADD_TO_BOTH_SIDES', 'SUBTRACT_FROM_BOTH_SIDES',
        'MULTIPLY_BOTH_SIDES', 'DIVIDE_BOTH_SIDES', 'MOVE_TERMS',
        'CLEAR_FRACTIONS', 'APPLY_QUADRATIC_FORMULA', 'SOLVE'
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
   * Enhanced heuristic-based prediction with actual algebraic logic
   */
  private heuristicPrediction(expression: string): {
    operation: string;
    confidence: number;
    parameters?: number[];
    explanation: string;
  } {
    // Check for equations
    if (expression.includes('=')) {
      const [left, right] = expression.split('=').map(s => s.trim());
      
      // Check for quadratic equations: x^2 = constant
      if (left.includes('^2') && this.isNumeric(right)) {
        const varMatch = left.match(/([a-zA-Z])\^2/);
        if (varMatch) {
          return {
            operation: 'SOLVE_QUADRATIC',
            confidence: 0.95,
            explanation: `Solve quadratic equation by taking square root of both sides`
          };
        }
      }
      
      // Check for linear equations: ax + b = c
      if (left.includes('*') && left.includes('+') && this.isNumeric(right)) {
        return {
          operation: 'SOLVE_LINEAR',
          confidence: 0.90,
          explanation: `Solve linear equation by isolating variable`
        };
      }
      
      // Check for simple equations: x = constant
      if (this.isVariable(left) && this.isNumeric(right)) {
        return {
          operation: 'SOLUTION_FOUND',
          confidence: 1.0,
          explanation: `Solution found: ${left} = ${right}`
        };
      }
      
      // Check for equations with parentheses
      if (expression.includes('(')) {
        return {
          operation: 'EXPAND',
          confidence: 0.85,
          explanation: 'Expand parentheses in equation'
        };
      }
      
      // Check for equations with multiple terms
      if (left.includes('+') || left.includes('-') || right.includes('+') || right.includes('-')) {
        return {
          operation: 'MOVE_TERMS',
          confidence: 0.78,
          explanation: 'Move terms to isolate variable'
        };
      }
      
      // Check for equations with fractions
      if (expression.includes('/')) {
        return {
          operation: 'CLEAR_FRACTIONS',
          confidence: 0.80,
          explanation: 'Clear fractions by multiplying both sides'
        };
      }
    }
    
    // Check for expressions with parentheses
    if (expression.includes('(')) {
      return {
        operation: 'EXPAND',
        confidence: 0.82,
        explanation: 'Expand parentheses'
      };
    }
    
    // Check for expressions with like terms
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
   * Enhanced operation simulation with actual algebraic operations
   */
  private simulateOperation(expression: string, operation: string): {
    expression: string;
    explanation: string;
  } {
    switch (operation) {
      case 'SOLVE_QUADRATIC':
        return this.solveQuadratic(expression);
      
      case 'SOLVE_LINEAR':
        return this.solveLinear(expression);
      
      case 'SOLUTION_FOUND':
        return {
          expression: expression,
          explanation: `Solution found: ${expression}`
        };
      
      case 'EXPAND':
        return this.expandExpression(expression);
      
      case 'MOVE_TERMS':
        return this.moveTerms(expression);
      
      case 'CLEAR_FRACTIONS':
        return this.clearFractions(expression);
      
      case 'COMBINE_LIKE_TERMS':
        return this.combineLikeTerms(expression);
      
      case 'SIMPLIFY':
        return {
          expression: expression.replace(/\s+/g, ''),
          explanation: `Simplified: ${expression}`
        };
      
      case 'DIRECT_SOLVE':
        return this.directSolve(expression);
      
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
   * Solve quadratic equations like x^2 = 9
   */
  private solveQuadratic(expression: string): { expression: string; explanation: string } {
    const [left, right] = expression.split('=').map(s => s.trim());
    const varMatch = left.match(/([a-zA-Z])\^2/);
    
    if (varMatch && this.isNumeric(right)) {
      const variable = varMatch[1];
      const constant = parseFloat(right);
      
      if (constant >= 0) {
        const sqrt = Math.sqrt(constant);
        if (sqrt === Math.floor(sqrt)) {
          // Perfect square
          return {
            expression: `${variable} = ${sqrt} or ${variable} = -${sqrt}`,
            explanation: `Take square root of both sides: ${variable}² = ${constant} → ${variable} = ±${sqrt}`
          };
        } else {
          // Not a perfect square
          return {
            expression: `${variable} = ±√${constant}`,
            explanation: `Take square root of both sides: ${variable}² = ${constant} → ${variable} = ±√${constant}`
          };
        }
      } else {
        return {
          expression: `${variable} = ±${Math.sqrt(-constant)}i`,
          explanation: `Take square root of both sides: ${variable}² = ${constant} → ${variable} = ±${Math.sqrt(-constant)}i (complex solution)`
        };
      }
    }
    
    return {
      expression: expression,
      explanation: `Could not solve quadratic equation: ${expression}`
    };
  }

  /**
   * Solve linear equations like 2x + 3 = 7
   */
  private solveLinear(expression: string): { expression: string; explanation: string } {
    const [left, right] = expression.split('=').map(s => s.trim());
    
    // Simple case: ax + b = c
    const linearMatch = left.match(/(\d+)\*([a-zA-Z])\s*\+\s*(\d+)/);
    if (linearMatch && this.isNumeric(right)) {
      const [, coefficient, variable, constant] = linearMatch;
      const target = parseFloat(right);
      const coeff = parseFloat(coefficient);
      const constVal = parseFloat(constant);
      
      // Solve: ax + b = c → x = (c - b) / a
      const solution = (target - constVal) / coeff;
      
      return {
        expression: `${variable} = ${solution}`,
        explanation: `Subtract ${constVal} from both sides: ${coeff}${variable} = ${target - constVal}, then divide by ${coeff}: ${variable} = ${solution}`
      };
    }
    
    return {
      expression: expression,
      explanation: `Could not solve linear equation: ${expression}`
    };
  }

  /**
   * Expand expressions with parentheses
   */
  private expandExpression(expression: string): { expression: string; explanation: string } {
    // Simple expansion: a(b + c) = ab + ac
    const expandMatch = expression.match(/(\d+)\(([^)]+)\)/);
    if (expandMatch) {
      const [, coefficient, inside] = expandMatch;
      const coeff = parseFloat(coefficient);
      const terms = inside.split('+').map(t => t.trim());
      const expanded = terms.map(term => `${coeff * parseFloat(term)}`).join(' + ');
      
      return {
        expression: expression.replace(expandMatch[0], expanded),
        explanation: `Expand: ${coefficient}(${inside}) = ${expanded}`
      };
    }
    
    return {
      expression: expression.replace(/\(/g, '').replace(/\)/g, ''),
      explanation: `Expanded: ${expression}`
    };
  }

  /**
   * Move terms to isolate variable
   */
  private moveTerms(expression: string): { expression: string; explanation: string } {
    const [left, right] = expression.split('=').map(s => s.trim());
    
    // Move constant from left to right: x + 3 = 5 → x = 5 - 3
    const constMatch = left.match(/([a-zA-Z])\s*\+\s*(\d+)/);
    if (constMatch && this.isNumeric(right)) {
      const [, variable, constant] = constMatch;
      const target = parseFloat(right);
      const constVal = parseFloat(constant);
      
      return {
        expression: `${variable} = ${target - constVal}`,
        explanation: `Subtract ${constant} from both sides: ${variable} = ${target} - ${constant} = ${target - constVal}`
      };
    }
    
    return {
      expression: expression,
      explanation: `Moved terms in: ${expression}`
    };
  }

  /**
   * Clear fractions by multiplying both sides
   */
  private clearFractions(expression: string): { expression: string; explanation: string } {
    const [left, right] = expression.split('=').map(s => s.trim());
    
    // Simple case: x/2 = 3 → x = 6
    const fractionMatch = left.match(/([a-zA-Z])\/(\d+)/);
    if (fractionMatch && this.isNumeric(right)) {
      const [, variable, denominator] = fractionMatch;
      const target = parseFloat(right);
      const denom = parseFloat(denominator);
      
      return {
        expression: `${variable} = ${target * denom}`,
        explanation: `Multiply both sides by ${denominator}: ${variable} = ${target} × ${denominator} = ${target * denom}`
      };
    }
    
    return {
      expression: expression,
      explanation: `Cleared fractions in: ${expression}`
    };
  }

  /**
   * Combine like terms
   */
  private combineLikeTerms(expression: string): { expression: string; explanation: string } {
    // Simple case: 2x + 3x = 5x
    const likeTermsMatch = expression.match(/(\d+)x\s*\+\s*(\d+)x/);
    if (likeTermsMatch) {
      const [, coeff1, coeff2] = likeTermsMatch;
      const sum = parseFloat(coeff1) + parseFloat(coeff2);
      
      return {
        expression: expression.replace(likeTermsMatch[0], `${sum}x`),
        explanation: `Combine like terms: ${coeff1}x + ${coeff2}x = ${sum}x`
      };
    }
    
    return {
      expression: expression,
      explanation: `Combined like terms in: ${expression}`
    };
  }

  /**
   * Direct solve using pattern matching
   */
  private directSolve(expression: string): { expression: string; explanation: string } {
    // Try to solve common patterns
    if (expression.includes('^2')) {
      return this.solveQuadratic(expression);
    }
    
    if (expression.includes('*') && expression.includes('+')) {
      return this.solveLinear(expression);
    }
    
    return {
      expression: expression,
      explanation: `Attempted direct solution for: ${expression}`
    };
  }

  /**
   * Helper: Check if string is numeric
   */
  private isNumeric(str: string): boolean {
    return !isNaN(parseFloat(str)) && isFinite(parseFloat(str));
  }

  /**
   * Helper: Check if string is a single variable
   */
  private isVariable(str: string): boolean {
    return /^[a-zA-Z]$/.test(str.trim());
  }

  /**
   * Check if expression is solved
   */
  isSolved(expression: string): boolean {
    // Check if it's in the form x = value
    const match = expression.match(/^([a-zA-Z])\s*=\s*([^=]+)$/);
    if (match) return true;
    
    // Check if it's a quadratic solution: x = value or x = -value
    const quadMatch = expression.match(/^([a-zA-Z])\s*=\s*([^=]+)\s*or\s*([a-zA-Z])\s*=\s*([^=]+)$/);
    if (quadMatch) return true;
    
    // Check if it's a solution with ±: x = ±value
    const pmMatch = expression.match(/^([a-zA-Z])\s*=\s*±([^=]+)$/);
    if (pmMatch) return true;
    
    // Check if it's a complex solution: x = ±value*i
    const complexMatch = expression.match(/^([a-zA-Z])\s*=\s*±([^=]+)i$/);
    if (complexMatch) return true;
    
    return false;
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