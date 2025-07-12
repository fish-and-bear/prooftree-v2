import { MathNode } from 'mathjs';
import { NodeType, LinkType } from '../components/GraphVisualizer';

// Helper to determine node type from mathjs node
function getNodeType(node: MathNode): NodeType['type'] {
  if ((node as any).isOperatorNode) return 'operator';
  if ((node as any).isSymbolNode) return 'variable';
  if ((node as any).isConstantNode) return 'constant';
  if ((node as any).isFunctionNode) return 'function';
  if (node.type === 'AssignmentNode' || node.type === 'RelationalNode') return 'equation';
  return 'unknown';
}

// Recursively traverse AST and build nodes/links, and id->AST mapping
export function astToGraph(ast: MathNode) {
  const nodes: NodeType[] = [];
  const links: LinkType[] = [];
  const idToAst: Record<number, MathNode> = {};
  let nextId = 1;

  function traverse(node: MathNode, parentId?: number): number {
    // Always assign a new _graphId
    (node as any)._graphId = nextId++;
    const thisId = (node as any)._graphId;
    const type = getNodeType(node);
    let label = '';
    if ((node as any).isOperatorNode) label = (node as any).op;
    else if ((node as any).isSymbolNode) label = (node as any).name;
    else if ((node as any).isConstantNode) label = String((node as any).value);
    else if ((node as any).isFunctionNode) label = (node as any).fn.name;
    else if (node.type === 'AssignmentNode' || node.type === 'RelationalNode') label = node.type === 'AssignmentNode' ? '=' : (node as any).op;
    else label = node.type;
    nodes.push({ id: thisId, label, type });
    idToAst[thisId] = node;
    if (parentId !== undefined) {
      links.push({ source: parentId, target: thisId });
    }
    // Recursively process children
    if ((node as any).args) {
      for (const child of (node as any).args) {
        traverse(child, thisId);
      }
    } else if ((node as any).value && typeof (node as any).value === 'object' && (node as any).value.isNode) {
      traverse((node as any).value, thisId);
    } else if ((node as any).content && (node as any).content.isNode) {
      traverse((node as any).content, thisId);
    } else if ((node as any).object && (node as any).object.isNode) {
      traverse((node as any).object, thisId);
    }
    return thisId;
  }

  traverse(ast);
  return { nodes, links, idToAst };
} 