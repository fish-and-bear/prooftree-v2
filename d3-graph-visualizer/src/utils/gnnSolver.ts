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

export class GNNSolver {
  // Only API call remains
  async solveStepByStep(expression: string): Promise<{
    steps: AlgebraicStep[];
    success: boolean;
    finalStrategy: string;
    limitations: string[];
  }> {
    try {
      const response = await fetch('/api/solve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ expression })
      });
      if (!response.ok) {
        throw new Error(`Backend error: ${response.statusText}`);
      }
      const data = await response.json();
      return data;
    } catch (error: any) {
      return {
        steps: [],
        success: false,
        finalStrategy: 'api_error',
        limitations: [error.message || 'Unknown error contacting backend']
      };
    }
  }
} 