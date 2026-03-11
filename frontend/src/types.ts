export type ScalarValue = number | string | { $param: string };

export type SceneNodeType =
  | "primitive"
  | "boolean"
  | "transform"
  | "field_expr"
  | "domain_op"
  | "lattice"
  | "turbomachine";

export type QualityProfile = "interactive" | "medium" | "high" | "ultra";
export type MeshLatticeType = "gyroid" | "schwarz_p" | "diamond";

export interface ParameterSpec {
  name: string;
  type: "float";
  default: number;
  min: number;
  max: number;
  step: number;
}

export interface SceneNode {
  id: string;
  type: SceneNodeType;
  primitive?: string | null;
  op?: string | null;
  inputs: string[];
  params: Record<string, ScalarValue>;
  transform?: Record<string, ScalarValue[]> | null;
  expr?: Record<string, unknown> | null;
  bounds_hint?: [ScalarValue, ScalarValue][] | null;
}

export interface SceneIR {
  nodes: SceneNode[];
  root_node_id: string;
  parameter_schema: ParameterSpec[];
  source_hash?: string | null;
}

export interface CompileDiagnostics {
  warnings: string[];
  inferred_bounds?: [[number, number], [number, number], [number, number]] | null;
}

export interface GridConfig {
  bounds: [[number, number], [number, number], [number, number]];
  resolution: number;
}

export interface MeshPayload {
  vertices: [number, number, number][];
  indices: [number, number, number][];
  normals: [number, number, number][];
}

export interface PreviewStats {
  eval_ms: number;
  mesh_ms: number;
  tri_count: number;
  cache_hit?: boolean;
}

export interface PreviewMeshResponse {
  mesh: MeshPayload;
  stats: PreviewStats;
}

export interface MeshWorkflowParams {
  shellThickness: number;
  latticeType: MeshLatticeType;
  latticePitch: number;
  latticeThickness: number;
  latticePhase: number;
}
