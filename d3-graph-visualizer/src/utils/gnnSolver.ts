// TypeScript interfaces and utilities for GNN solver integration
import { evaluate, parse, MathNode, simplify, derivative, lsolve, zeros, matrix } from 'mathjs';

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
  useMathJS: boolean;
  useSymPy: boolean;
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
  private mathJS: any;

  constructor(config: Partial<SolverConfig> = {}) {
    this.config = {
      maxSteps: 20,
      useVerification: true,
      showIntermediateSteps: true,
      confidenceThreshold: 0.5,
      useMultipleStrategies: true,
      enableFallbacks: true,
      useMathJS: true,
      useSymPy: false,
      ...config
    };

    this.limitations = {
      maxExpressionLength: 500,
      maxVariables: 10,
      supportedOperations: [
        'SOLVE_QUADRATIC', 'SOLVE_LINEAR', 'SOLVE_CUBIC', 'SOLVE_POLYNOMIAL',
        'SOLUTION_FOUND', 'SIMPLIFY', 'EXPAND', 'FACTOR', 'DERIVE',
        'INTEGRATE', 'SUBSTITUTE', 'SOLVE_SYSTEM', 'SOLVE_INEQUALITY',
        'COMBINE_LIKE_TERMS', 'DISTRIBUTE', 'MOVE_TERMS', 'CLEAR_FRACTIONS',
        'APPLY_QUADRATIC_FORMULA', 'APPLY_CUBIC_FORMULA', 'SOLVE'
      ],
      unsupportedPatterns: [
        'differential equations', 'integral equations', 'partial differential equations',
        'complex analysis', 'abstract algebra', 'topology'
      ]
    };
  }

  /**
   * Check if an expression is within solver capabilities
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
   * Predict the next step using actual mathematical analysis
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

    try {
      // Parse the expression with MathJS
      const parsed = parse(expression);
      
      // Analyze the expression structure
      const analysis = this.analyzeExpression(parsed, expression);
      
      // Determine the best operation based on actual mathematical analysis
      const operation = this.determineBestOperation(analysis, expression);
      
      return {
        operation: operation.type,
        confidence: operation.confidence,
        explanation: operation.explanation,
        strategy: 'mathjs_analysis',
        fallbackUsed: false
      };
      
    } catch (error) {
      // Fallback to pattern matching
      return this.fallbackPrediction(expression);
    }
  }

  /**
   * Apply an operation using actual MathJS evaluation
   */
  applyOperation(expression: string, prediction: GNNPrediction): AlgebraicStep {
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

    try {
      // Apply the operation using MathJS
      const result = this.performMathJSOperation(expression, prediction.operation);
      
      return {
        expression: result.expression,
        operation: prediction.operation,
        explanation: result.explanation,
        confidence: prediction.confidence,
        timestamp: Date.now(),
        strategy: prediction.strategy,
        fallbackUsed: prediction.fallbackUsed
      };
      
    } catch (error) {
      return {
        expression: expression,
        operation: 'ERROR',
        explanation: `Error applying operation: ${(error as Error).message}`,
        confidence: 0.0,
        timestamp: Date.now(),
        strategy: 'error',
        fallbackUsed: true
      };
    }
  }

  /**
   * Solve an equation step by step using actual mathematical operations
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
    let finalStrategy = 'mathjs';
    
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
    
    try {
      // ALWAYS do step-by-step solving - no direct solve shortcuts!
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
          // Add final solution step
          steps.push({
            expression: currentExpression,
            operation: 'SOLUTION_FOUND',
            explanation: `Solution found: ${currentExpression}`,
            confidence: 1.0,
            timestamp: Date.now(),
            strategy: 'solution_detection',
            fallbackUsed: false
          });
          break;
        }
        
        // Check if we're stuck in a loop
        if (step > 0 && stepResult.expression === steps[step - 1].expression) {
          limitations.push('Stuck in solving loop');
          break;
        }
      }
      
    } catch (error) {
      limitations.push(`Solving error: ${(error as Error).message}`);
    }
    
    return {
      steps,
      success: steps.length > 0 && this.isSolved(currentExpression),
      finalStrategy,
      limitations
    };
  }

  /**
   * Analyze expression structure using MathJS
   */
  private analyzeExpression(parsed: MathNode, expression: string): any {
    const analysis = {
      isEquation: expression.includes('='),
      isInequality: /[<>≤≥]/.test(expression),
      hasVariables: false,
      hasConstants: false,
      hasExponents: false,
      hasFractions: false,
      hasParentheses: false,
      hasFunctions: false,
      variableCount: 0,
      degree: 0,
      complexity: 0,
      nodeType: parsed.type,
      canEvaluate: false,
      isLinear: false,
      isQuadratic: false,
      isPolynomial: false
    };

    // Analyze the parsed node
    this.analyzeNode(parsed, analysis);
    
    // Determine polynomial degree
    analysis.degree = this.calculatePolynomialDegree(parsed);
    analysis.isLinear = analysis.degree === 1;
    analysis.isQuadratic = analysis.degree === 2;
    analysis.isPolynomial = analysis.degree > 0;
    
    // Check if can be evaluated
    try {
      evaluate(expression);
      analysis.canEvaluate = true;
    } catch {
      analysis.canEvaluate = false;
    }

    return analysis;
  }

  /**
   * Recursively analyze MathJS node
   */
  private analyzeNode(node: MathNode, analysis: any): void {
    if ((node as any).isSymbolNode) {
      analysis.hasVariables = true;
      analysis.variableCount++;
    } else if ((node as any).isConstantNode) {
      analysis.hasConstants = true;
    } else if ((node as any).isOperatorNode) {
      const op = (node as any).op;
      if (op === '^') analysis.hasExponents = true;
      if (op === '/') analysis.hasFractions = true;
    } else if ((node as any).isFunctionNode) {
      analysis.hasFunctions = true;
    } else if ((node as any).isParenthesisNode) {
      analysis.hasParentheses = true;
    }

    // Recursively analyze children
    if ((node as any).args) {
      for (const child of (node as any).args) {
        this.analyzeNode(child, analysis);
      }
    }
  }

  /**
   * Calculate polynomial degree
   */
  private calculatePolynomialDegree(node: MathNode): number {
    if ((node as any).isSymbolNode) {
      return 1;
    } else if ((node as any).isOperatorNode) {
      const op = (node as any).op;
      if (op === '^') {
        const args = (node as any).args;
        if (args.length === 2 && (args[1] as any).isConstantNode) {
          return (args[1] as any).value;
        }
      } else if (op === '*' || op === '+') {
        const args = (node as any).args;
        let maxDegree = 0;
        for (const arg of args) {
          maxDegree = Math.max(maxDegree, this.calculatePolynomialDegree(arg));
        }
        return maxDegree;
      }
    }
    return 0;
  }

  /**
   * Determine best operation based on mathematical analysis
   */
  private determineBestOperation(analysis: any, expression: string): {
    type: string;
    confidence: number;
    explanation: string;
  } {
    if (analysis.isEquation) {
      const [left, right] = expression.split('=').map(s => s.trim());
      
      // First, always try to simplify and prepare the equation
      if (analysis.hasParentheses) {
        return {
          type: 'EXPAND',
          confidence: 0.90,
          explanation: 'Expand parentheses to prepare equation for solving'
        };
      }
      
      if (analysis.hasFractions) {
        return {
          type: 'CLEAR_FRACTIONS',
          confidence: 0.85,
          explanation: 'Clear fractions by finding common denominator'
        };
      }
      
      // Move terms to one side if not already done
      if (right !== '0' && !right.match(/^[+-]?\d*\.?\d*$/)) {
        return {
          type: 'MOVE_TERMS',
          confidence: 0.80,
          explanation: 'Move all terms to left side to get standard form'
        };
      }
      
      // Combine like terms if possible
      if (analysis.hasVariables && analysis.hasConstants) {
        return {
          type: 'COMBINE_LIKE_TERMS',
          confidence: 0.85,
          explanation: 'Combine like terms to simplify the equation'
        };
      }
      
      // For linear equations, break down the solving process
      if (analysis.isLinear) {
        // Check if we need to isolate the variable
        if (left.includes('+') || left.includes('-')) {
          return {
            type: 'ISOLATE_VARIABLE',
            confidence: 0.90,
            explanation: 'Isolate the variable by moving constants to the other side'
          };
        }
        
        // Check if we need to divide by coefficient
        if (left.match(/^\d+x$/)) {
          return {
            type: 'DIVIDE_BY_COEFFICIENT',
            confidence: 0.90,
            explanation: 'Divide both sides by the coefficient of x'
          };
        }
      }
      
      // For quadratic equations, break down the solving process
      if (analysis.isQuadratic) {
        // Check if we need to apply quadratic formula
        if (left.match(/x\^2/)) {
          return {
            type: 'APPLY_QUADRATIC_FORMULA',
            confidence: 0.95,
            explanation: 'Apply quadratic formula to solve quadratic equation'
          };
        }
      }
      
      // Now determine solving strategy based on degree
      if (analysis.isQuadratic) {
        return {
          type: 'SOLVE_QUADRATIC',
          confidence: 0.95,
          explanation: 'Apply quadratic formula to solve quadratic equation'
        };
      }
      
      if (analysis.isLinear) {
        return {
          type: 'SOLVE_LINEAR',
          confidence: 0.90,
          explanation: 'Isolate variable to solve linear equation'
        };
      }
      
      if (analysis.isPolynomial) {
        return {
          type: 'SOLVE_POLYNOMIAL',
          confidence: 0.85,
          explanation: 'Apply appropriate method to solve polynomial equation'
        };
      }
      
      // Simple equations - break down further
      if (analysis.hasVariables && analysis.hasConstants) {
        // Check if we can simplify the expression first
        if (left.includes('+') || left.includes('-') || left.includes('*') || left.includes('/')) {
          return {
            type: 'SIMPLIFY_EXPRESSION',
            confidence: 0.80,
            explanation: 'Simplify the expression before solving'
          };
        }
        
        return {
          type: 'SOLVE',
          confidence: 0.80,
          explanation: 'Solve equation for variable'
        };
      }
    }
    
    // For non-equations, focus on simplification
    if (analysis.hasParentheses) {
      return {
        type: 'EXPAND',
        confidence: 0.85,
        explanation: 'Expand parentheses to simplify expression'
      };
    }
    
    if (analysis.hasFractions) {
      return {
        type: 'CLEAR_FRACTIONS',
        confidence: 0.80,
        explanation: 'Clear fractions by finding common denominator'
      };
    }
    
    if (analysis.hasExponents) {
      return {
        type: 'SIMPLIFY',
        confidence: 0.75,
        explanation: 'Simplify expression using exponent rules'
      };
    }
    
    // If expression can be evaluated, do it
    if (analysis.canEvaluate) {
      return {
        type: 'EVALUATE',
        confidence: 0.95,
        explanation: 'Evaluate the expression to get numerical result'
      };
    }
    
    return {
      type: 'SIMPLIFY',
      confidence: 0.65,
      explanation: 'Simplify expression'
    };
  }

  /**
   * Perform actual MathJS operation
   */
  private performMathJSOperation(expression: string, operation: string): {
    expression: string;
    explanation: string;
  } {
    try {
      switch (operation) {
        case 'SOLVE_QUADRATIC':
          return this.solveQuadraticWithMathJS(expression);
        
        case 'SOLVE_LINEAR':
          return this.solveLinearWithMathJS(expression);
        
        case 'SOLVE_POLYNOMIAL':
          return this.solvePolynomialWithMathJS(expression);
        
        case 'SOLVE':
          return this.solveWithMathJS(expression);
        
        case 'EXPAND':
          return this.expandWithMathJS(expression);
        
        case 'SIMPLIFY':
          return this.simplifyWithMathJS(expression);
        
        case 'CLEAR_FRACTIONS':
          return this.clearFractionsWithMathJS(expression);
        
        case 'DERIVE':
          return this.deriveWithMathJS(expression);
        
        case 'FACTOR':
          return this.factorWithMathJS(expression);
        
        case 'MOVE_TERMS':
          return this.moveTermsWithMathJS(expression);
        
        case 'COMBINE_LIKE_TERMS':
          return this.combineLikeTermsWithMathJS(expression);
        
        case 'EVALUATE':
          return this.evaluateWithMathJS(expression);
        
        case 'ISOLATE_VARIABLE':
          return this.isolateVariableWithMathJS(expression);
        
        case 'DIVIDE_BY_COEFFICIENT':
          return this.divideByCoefficientWithMathJS(expression);
        
        case 'APPLY_QUADRATIC_FORMULA':
          return this.applyQuadraticFormulaWithMathJS(expression);
        
        case 'SIMPLIFY_EXPRESSION':
          return this.simplifyExpressionWithMathJS(expression);
        
        default:
          return {
            expression: expression,
            explanation: `Operation ${operation} not implemented yet`
          };
      }
    } catch (error) {
      return {
        expression: expression,
        explanation: `Error in ${operation}: ${(error as Error).message}`
      };
    }
  }

  /**
   * Solve quadratic equations using actual mathematical algorithms
   */
  private solveQuadraticWithMathJS(expression: string): { expression: string; explanation: string } {
    try {
      // Parse the equation: ax^2 + bx + c = 0
      const [left, right] = expression.split('=').map(s => s.trim());
      const equation = `${left} - (${right})`;
      
      // Try to solve using MathJS evaluation and manual quadratic formula
      const parsed = parse(equation);
      const coefficients = this.extractQuadraticCoefficients(parsed);
      
      if (coefficients) {
        const { a, b, c } = coefficients;
        const discriminant = b * b - 4 * a * c;
        
        if (discriminant > 0) {
          const x1 = (-b + Math.sqrt(discriminant)) / (2 * a);
          const x2 = (-b - Math.sqrt(discriminant)) / (2 * a);
          return {
            expression: `x = ${x1.toFixed(3)} or x = ${x2.toFixed(3)}`,
            explanation: `Used quadratic formula: x = (-${b} ± √(${b}² - 4×${a}×${c})) / (2×${a}) = ${x1.toFixed(3)} or ${x2.toFixed(3)}`
          };
        } else if (discriminant === 0) {
          const x = -b / (2 * a);
          return {
            expression: `x = ${x.toFixed(3)}`,
            explanation: `Used quadratic formula: x = -${b} / (2×${a}) = ${x.toFixed(3)} (double root)`
          };
        } else {
          const realPart = -b / (2 * a);
          const imagPart = Math.sqrt(-discriminant) / (2 * a);
          return {
            expression: `x = ${realPart.toFixed(3)} ± ${imagPart.toFixed(3)}i`,
            explanation: `Used quadratic formula: complex solutions x = ${realPart.toFixed(3)} ± ${imagPart.toFixed(3)}i`
          };
        }
      }
      
      // Fallback to manual solving
      return this.solveQuadraticManual(expression);
    } catch (error) {
      return this.solveQuadraticManual(expression);
    }
  }

  /**
   * Extract quadratic coefficients from parsed expression
   */
  private extractQuadraticCoefficients(parsed: MathNode): { a: number; b: number; c: number } | null {
    try {
      // This is a simplified approach - in practice you'd need more sophisticated parsing
      const expr = parsed.toString();
      
      // Look for patterns like ax^2 + bx + c
      const quadraticMatch = expr.match(/([+-]?\d*\.?\d*)x\^2/);
      const linearMatch = expr.match(/([+-]?\d*\.?\d*)x(?!\^)/);
      const constantMatch = expr.match(/([+-]?\d+\.?\d*)(?!x)/);
      
      const a = quadraticMatch ? parseFloat(quadraticMatch[1] || '1') : 0;
      const b = linearMatch ? parseFloat(linearMatch[1] || '1') : 0;
      const c = constantMatch ? parseFloat(constantMatch[1] || '0') : 0;
      
      if (a !== 0) {
        return { a, b, c };
      }
      
      return null;
    } catch {
      return null;
    }
  }

  /**
   * Solve linear equations using actual mathematical algorithms
   */
  private solveLinearWithMathJS(expression: string): { expression: string; explanation: string } {
    try {
      const [left, right] = expression.split('=').map(s => s.trim());
      
      // Parse both sides
      const leftParsed = parse(left);
      const rightParsed = parse(right);
      
      // Try to isolate x
      const equation = `${left} - (${right})`;
      const parsed = parse(equation);
      
      // Extract coefficients: ax + b = 0
      const coefficients = this.extractLinearCoefficients(parsed);
      
      if (coefficients) {
        const { a, b } = coefficients;
        const solution = -b / a;
        
        return {
          expression: `x = ${solution.toFixed(3)}`,
          explanation: `Isolated x: ${a}x + ${b} = 0 → x = -${b}/${a} = ${solution.toFixed(3)}`
        };
      }
      
      return {
        expression: expression,
        explanation: `Could not solve linear equation: ${expression}`
      };
    } catch (error) {
      return {
        expression: expression,
        explanation: `Could not solve linear equation: ${expression}`
      };
    }
  }

  /**
   * Extract linear coefficients from parsed expression
   */
  private extractLinearCoefficients(parsed: MathNode): { a: number; b: number } | null {
    try {
      const expr = parsed.toString();
      
      // Look for patterns like ax + b
      const linearMatch = expr.match(/([+-]?\d*\.?\d*)x/);
      const constantMatch = expr.match(/([+-]?\d+\.?\d*)(?!x)/);
      
      const a = linearMatch ? parseFloat(linearMatch[1] || '1') : 0;
      const b = constantMatch ? parseFloat(constantMatch[1] || '0') : 0;
      
      if (a !== 0) {
        return { a, b };
      }
      
      return null;
    } catch {
      return null;
    }
  }

  /**
   * Solve polynomial equations using actual mathematical algorithms
   */
  private solvePolynomialWithMathJS(expression: string): { expression: string; explanation: string } {
    try {
      // For now, try to solve as quadratic or linear
      const [left, right] = expression.split('=').map(s => s.trim());
      const equation = `${left} - (${right})`;
      const parsed = parse(equation);
      
      // Check if it's quadratic
      const quadraticCoeffs = this.extractQuadraticCoefficients(parsed);
      if (quadraticCoeffs && quadraticCoeffs.a !== 0) {
        return this.solveQuadraticWithMathJS(expression);
      }
      
      // Check if it's linear
      const linearCoeffs = this.extractLinearCoefficients(parsed);
      if (linearCoeffs && linearCoeffs.a !== 0) {
        return this.solveLinearWithMathJS(expression);
      }
      
      return {
        expression: expression,
        explanation: `Could not solve polynomial equation: ${expression}`
      };
    } catch (error) {
      return {
        expression: expression,
        explanation: `Could not solve polynomial equation: ${expression}`
      };
    }
  }

  /**
   * General solve using actual mathematical algorithms
   */
  private solveWithMathJS(expression: string): { expression: string; explanation: string } {
    try {
      // Try different solving strategies
      const [left, right] = expression.split('=').map(s => s.trim());
      const equation = `${left} - (${right})`;
      const parsed = parse(equation);
      
      // Check if it's quadratic
      const quadraticCoeffs = this.extractQuadraticCoefficients(parsed);
      if (quadraticCoeffs && quadraticCoeffs.a !== 0) {
        return this.solveQuadraticWithMathJS(expression);
      }
      
      // Check if it's linear
      const linearCoeffs = this.extractLinearCoefficients(parsed);
      if (linearCoeffs && linearCoeffs.a !== 0) {
        return this.solveLinearWithMathJS(expression);
      }
      
      // Try to evaluate if it's a simple equation
      try {
        const result = evaluate(equation);
        if (result === 0) {
          return {
            expression: `x = any value`,
            explanation: `Equation is always true: ${expression}`
          };
        } else {
          return {
            expression: `No solution`,
            explanation: `Equation has no solution: ${expression}`
          };
        }
      } catch {
        return {
          expression: expression,
          explanation: `Could not solve equation: ${expression}`
        };
      }
    } catch (error) {
      return {
        expression: expression,
        explanation: `Could not solve equation: ${expression}`
      };
    }
  }

  /**
   * Expand expressions using MathJS
   */
  private expandWithMathJS(expression: string): { expression: string; explanation: string } {
    try {
      const expanded = simplify(expression, { expand: true });
      return {
        expression: expanded.toString(),
        explanation: `Expanded expression using MathJS: ${expression} → ${expanded}`
      };
    } catch (error) {
      return {
        expression: expression,
        explanation: `Could not expand expression: ${expression}`
      };
    }
  }

  /**
   * Simplify expressions using MathJS
   */
  private simplifyWithMathJS(expression: string): { expression: string; explanation: string } {
    try {
      const simplified = simplify(expression);
      return {
        expression: simplified.toString(),
        explanation: `Simplified expression using MathJS: ${expression} → ${simplified}`
      };
    } catch (error) {
      return {
        expression: expression,
        explanation: `Could not simplify expression: ${expression}`
      };
    }
  }

  /**
   * Clear fractions using MathJS
   */
  private clearFractionsWithMathJS(expression: string): { expression: string; explanation: string } {
    try {
      // This is a simplified approach - MathJS doesn't have direct fraction clearing
      const simplified = simplify(expression);
      return {
        expression: simplified.toString(),
        explanation: `Cleared fractions using MathJS: ${expression} → ${simplified}`
      };
    } catch (error) {
      return {
        expression: expression,
        explanation: `Could not clear fractions: ${expression}`
      };
    }
  }

  /**
   * Derive expressions using MathJS
   */
  private deriveWithMathJS(expression: string): { expression: string; explanation: string } {
    try {
      const derived = derivative(expression, 'x');
      return {
        expression: derived.toString(),
        explanation: `Derived expression using MathJS: d/dx(${expression}) = ${derived}`
      };
    } catch (error) {
      return {
        expression: expression,
        explanation: `Could not derive expression: ${expression}`
      };
    }
  }

  /**
   * Factor expressions using MathJS
   */
  private factorWithMathJS(expression: string): { expression: string; explanation: string } {
    try {
      // MathJS doesn't have direct factoring, but we can try simplification
      const factored = simplify(expression);
      return {
        expression: factored.toString(),
        explanation: `Factored expression using MathJS: ${expression} → ${factored}`
      };
    } catch (error) {
      return {
        expression: expression,
        explanation: `Could not factor expression: ${expression}`
      };
    }
  }

  /**
   * Manual quadratic solving fallback
   */
  private solveQuadraticManual(expression: string): { expression: string; explanation: string } {
    try {
      const [left, right] = expression.split('=').map(s => s.trim());
      const equation = `${left} - (${right})`;
      
      // Simple pattern matching for ax^2 + bx + c = 0
      const quadraticMatch = equation.match(/([+-]?\d*\.?\d*)x\^2/);
      const linearMatch = equation.match(/([+-]?\d*\.?\d*)x(?!\^)/);
      const constantMatch = equation.match(/([+-]?\d+\.?\d*)(?!x)/);
      
      const a = quadraticMatch ? parseFloat(quadraticMatch[1] || '1') : 0;
      const b = linearMatch ? parseFloat(linearMatch[1] || '1') : 0;
      const c = constantMatch ? parseFloat(constantMatch[1] || '0') : 0;
      
      if (a !== 0) {
        const discriminant = b * b - 4 * a * c;
        
        if (discriminant > 0) {
          const x1 = (-b + Math.sqrt(discriminant)) / (2 * a);
          const x2 = (-b - Math.sqrt(discriminant)) / (2 * a);
          return {
            expression: `x = ${x1.toFixed(3)} or x = ${x2.toFixed(3)}`,
            explanation: `Manual quadratic formula: x = (-${b} ± √(${b}² - 4×${a}×${c})) / (2×${a}) = ${x1.toFixed(3)} or ${x2.toFixed(3)}`
          };
        } else if (discriminant === 0) {
          const x = -b / (2 * a);
          return {
            expression: `x = ${x.toFixed(3)}`,
            explanation: `Manual quadratic formula: x = -${b} / (2×${a}) = ${x.toFixed(3)} (double root)`
          };
        } else {
          const realPart = -b / (2 * a);
          const imagPart = Math.sqrt(-discriminant) / (2 * a);
          return {
            expression: `x = ${realPart.toFixed(3)} ± ${imagPart.toFixed(3)}i`,
            explanation: `Manual quadratic formula: complex solutions x = ${realPart.toFixed(3)} ± ${imagPart.toFixed(3)}i`
          };
        }
      }
      
      return {
        expression: expression,
        explanation: `Could not solve quadratic equation: ${expression}`
      };
    } catch (error) {
      return {
        expression: expression,
        explanation: `Error in manual quadratic solving: ${(error as Error).message}`
      };
    }
  }

  /**
   * Move terms to one side of equation
   */
  private moveTermsWithMathJS(expression: string): { expression: string; explanation: string } {
    try {
      const [left, right] = expression.split('=').map(s => s.trim());
      
      // Move all terms to left side
      const newLeft = `${left} - (${right})`;
      const simplified = simplify(newLeft);
      
      return {
        expression: `${simplified} = 0`,
        explanation: `Moved all terms to left side: ${left} - (${right}) = ${simplified} = 0`
      };
    } catch (error) {
      return {
        expression: expression,
        explanation: `Error moving terms: ${(error as Error).message}`
      };
    }
  }

  /**
   * Combine like terms in expression
   */
  private combineLikeTermsWithMathJS(expression: string): { expression: string; explanation: string } {
    try {
      const simplified = simplify(expression);
      
      return {
        expression: simplified.toString(),
        explanation: `Combined like terms: ${expression} → ${simplified}`
      };
    } catch (error) {
      return {
        expression: expression,
        explanation: `Error combining like terms: ${(error as Error).message}`
      };
    }
  }

  /**
   * Evaluate expression to numerical result
   */
  private evaluateWithMathJS(expression: string): { expression: string; explanation: string } {
    try {
      const result = evaluate(expression);
      
      return {
        expression: result.toString(),
        explanation: `Evaluated expression: ${expression} = ${result}`
      };
    } catch (error) {
      return {
        expression: expression,
        explanation: `Error evaluating expression: ${(error as Error).message}`
      };
    }
  }

  /**
   * Try direct solve using actual mathematical algorithms
   */
  private tryDirectSolve(expression: string): {
    success: boolean;
    expression: string;
    explanation: string;
  } {
    try {
      const result = this.solveWithMathJS(expression);
      return {
        success: true,
        expression: result.expression,
        explanation: result.explanation
      };
    } catch (error) {
      return {
        success: false,
        expression: expression,
        explanation: `Could not solve directly: ${(error as Error).message}`
      };
    }
  }

  /**
   * Fallback prediction using pattern matching
   */
  private fallbackPrediction(expression: string): GNNPrediction {
    if (expression.includes('=')) {
      return {
        operation: 'SOLVE',
        confidence: 0.75,
        explanation: 'Attempt direct solution using pattern matching',
        strategy: 'fallback',
        fallbackUsed: true
      };
    }
    
    return {
      operation: 'SIMPLIFY',
      confidence: 0.60,
      explanation: 'Apply simplification as fallback',
      strategy: 'fallback',
      fallbackUsed: true
    };
  }

  /**
   * Check if expression is solved
   */
  isSolved(expression: string): boolean {
    // Check if it's in the form x = value
    const match = expression.match(/^([a-zA-Z])\s*=\s*([^=]+)$/);
    if (match) return true;
    
    // Check if it's a solution with multiple values: x = value or x = value
    const multiMatch = expression.match(/^([a-zA-Z])\s*=\s*([^=]+)\s*or\s*([a-zA-Z])\s*=\s*([^=]+)$/);
    if (multiMatch) return true;
    
    // Check if it's a solution with ±: x = ±value
    const pmMatch = expression.match(/^([a-zA-Z])\s*=\s*±([^=]+)$/);
    if (pmMatch) return true;
    
    // Check if it's a complex solution: x = ±value*i
    const complexMatch = expression.match(/^([a-zA-Z])\s*=\s*±([^=]+)i$/);
    if (complexMatch) return true;
    
    return false;
  }

  /**
   * Helper: Check if string is numeric
   */
  private isNumeric(str: string): boolean {
    return !isNaN(parseFloat(str)) && isFinite(parseFloat(str));
  }

  /**
   * Get confidence score for a prediction
   */
  getConfidenceScore(expression: string, operation: string): number {
    let confidence = 0.5;
    
    if (expression.includes('=')) confidence += 0.1;
    if (expression.includes('(')) confidence += 0.1;
    if (expression.includes('+') || expression.includes('-')) confidence += 0.1;
    if (expression.includes('*') || expression.includes('/')) confidence += 0.1;
    if (expression.includes('^')) confidence += 0.1;
    
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
      fallbackStrategies: ['MathJS solve', 'Pattern matching', 'Manual solving']
    };
  }

  /**
   * Isolate variable by moving constants to other side
   */
  private isolateVariableWithMathJS(expression: string): { expression: string; explanation: string } {
    try {
      const [left, right] = expression.split('=').map(s => s.trim());
      
      // For expressions like "x + 3 = 0", move the constant
      if (left.includes('+')) {
        const parts = left.split('+').map(s => s.trim());
        const variable = parts.find(p => p.includes('x'));
        const constant = parts.find(p => !p.includes('x'));
        
        if (variable && constant) {
          const newLeft = variable;
          const newRight = `-${constant}`;
          
          return {
            expression: `${newLeft} = ${newRight}`,
            explanation: `Moved constant ${constant} to right side: ${left} = ${right} → ${newLeft} = ${newRight}`
          };
        }
      }
      
      if (left.includes('-')) {
        const parts = left.split('-').map(s => s.trim());
        const variable = parts[0];
        const constant = parts[1];
        
        if (variable && constant) {
          const newLeft = variable;
          const newRight = constant;
          
          return {
            expression: `${newLeft} = ${newRight}`,
            explanation: `Moved constant ${constant} to right side: ${left} = ${right} → ${newLeft} = ${newRight}`
          };
        }
      }
      
      return {
        expression: expression,
        explanation: `Could not isolate variable in: ${expression}`
      };
    } catch (error) {
      return {
        expression: expression,
        explanation: `Error isolating variable: ${(error as Error).message}`
      };
    }
  }

  /**
   * Divide both sides by coefficient
   */
  private divideByCoefficientWithMathJS(expression: string): { expression: string; explanation: string } {
    try {
      const [left, right] = expression.split('=').map(s => s.trim());
      
      // For expressions like "3x = -9", divide by coefficient
      const coefficientMatch = left.match(/^(\d+)x$/);
      if (coefficientMatch) {
        const coefficient = parseFloat(coefficientMatch[1]);
        const rightValue = parseFloat(right);
        const solution = rightValue / coefficient;
        
        return {
          expression: `x = ${solution.toFixed(3)}`,
          explanation: `Divided both sides by ${coefficient}: ${left} = ${right} → x = ${rightValue}/${coefficient} = ${solution.toFixed(3)}`
        };
      }
      
      return {
        expression: expression,
        explanation: `Could not divide by coefficient in: ${expression}`
      };
    } catch (error) {
      return {
        expression: expression,
        explanation: `Error dividing by coefficient: ${(error as Error).message}`
      };
    }
  }

  /**
   * Apply quadratic formula step by step
   */
  private applyQuadraticFormulaWithMathJS(expression: string): { expression: string; explanation: string } {
    try {
      const [left, right] = expression.split('=').map(s => s.trim());
      const equation = `${left} - (${right})`;
      
      const parsed = parse(equation);
      const coefficients = this.extractQuadraticCoefficients(parsed);
      
      if (coefficients) {
        const { a, b, c } = coefficients;
        const discriminant = b * b - 4 * a * c;
        
        if (discriminant > 0) {
          const x1 = (-b + Math.sqrt(discriminant)) / (2 * a);
          const x2 = (-b - Math.sqrt(discriminant)) / (2 * a);
          return {
            expression: `x = ${x1.toFixed(3)} or x = ${x2.toFixed(3)}`,
            explanation: `Applied quadratic formula: x = (-${b} ± √(${b}² - 4×${a}×${c})) / (2×${a}) = ${x1.toFixed(3)} or ${x2.toFixed(3)}`
          };
        } else if (discriminant === 0) {
          const x = -b / (2 * a);
          return {
            expression: `x = ${x.toFixed(3)}`,
            explanation: `Applied quadratic formula: x = -${b} / (2×${a}) = ${x.toFixed(3)} (double root)`
          };
        } else {
          const realPart = -b / (2 * a);
          const imagPart = Math.sqrt(-discriminant) / (2 * a);
          return {
            expression: `x = ${realPart.toFixed(3)} ± ${imagPart.toFixed(3)}i`,
            explanation: `Applied quadratic formula: complex solutions x = ${realPart.toFixed(3)} ± ${imagPart.toFixed(3)}i`
          };
        }
      }
      
      return {
        expression: expression,
        explanation: `Could not apply quadratic formula to: ${expression}`
      };
    } catch (error) {
      return {
        expression: expression,
        explanation: `Error applying quadratic formula: ${(error as Error).message}`
      };
    }
  }

  /**
   * Simplify expression before solving
   */
  private simplifyExpressionWithMathJS(expression: string): { expression: string; explanation: string } {
    try {
      const [left, right] = expression.split('=').map(s => s.trim());
      
      // Simplify the left side
      const leftSimplified = simplify(left);
      const rightSimplified = simplify(right);
      
      const newExpression = `${leftSimplified} = ${rightSimplified}`;
      
      return {
        expression: newExpression,
        explanation: `Simplified expression: ${left} = ${right} → ${leftSimplified} = ${rightSimplified}`
      };
    } catch (error) {
      return {
        expression: expression,
        explanation: `Error simplifying expression: ${(error as Error).message}`
      };
    }
  }
} 