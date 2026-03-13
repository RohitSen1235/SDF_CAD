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
export type ComputePrecision = "float32" | "float16";
export type ComputeBackend = "auto" | "cpu" | "cuda";
export type MeshBackend = "auto" | "cpu" | "cuda";
export type MeshingMode = "uniform" | "adaptive";
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
  encoding: "mesh-f32-u32-base64-v1";
  vertex_count: number;
  face_count: number;
  vertices_b64: string;
  indices_b64: string;
  normals_b64: string;
}

export interface FieldPayload {
  encoding: "f32-base64";
  resolution: number;
  bounds: [[number, number], [number, number], [number, number]];
  data: string;
}

export interface PreviewStats {
  eval_ms: number;
  mesh_ms?: number | null;
  tri_count: number;
  voxel_count?: number;
  cache_hit?: boolean;
  compute_precision?: ComputePrecision;
  compute_backend?: "cpu" | "cuda";
  mesh_backend?: "cpu" | "cuda";
  preview_mode?: "mesh" | "field" | "analytic_raymarch";
  compile_ms?: number | null;
  program_bytes?: number | null;
  gpu_eval_mode?: string | null;
  fallback_reason?: string | null;
}

export interface PreviewMeshResponse {
  mesh: MeshPayload;
  stats: PreviewStats;
  field?: FieldPayload | null;
}

export interface PreviewFieldResponse {
  field: FieldPayload;
  stats: PreviewStats;
}

export interface SceneProgramPayload {
  mode: "dsl";
  bounds: [[number, number], [number, number], [number, number]];
  glsl_sdf: string;
  quality_profile: QualityProfile;
  max_steps: number;
  hit_epsilon: number;
  normal_epsilon: number;
}

export interface MeshProgramPayload {
  mode: "mesh_lattice";
  bounds: [[number, number], [number, number], [number, number]];
  quality_profile: QualityProfile;
  triangles_encoding: "f32-base64";
  triangles_data: string;
  triangle_count: number;
  bvh_encoding: "f32-base64";
  bvh_data: string;
  bvh_node_count: number;
  shell_thickness: number;
  lattice_type: "gyroid" | "schwarz_p" | "diamond";
  lattice_pitch: number;
  lattice_thickness: number;
  lattice_phase: number;
  max_steps: number;
  hit_epsilon: number;
  normal_epsilon: number;
}

export interface PreviewProgramResponse {
  program: SceneProgramPayload | MeshProgramPayload | null;
  capabilities: {
    analytic_supported: boolean;
    fallback_reason?: string | null;
  };
  stats: PreviewStats;
}

export interface MeshWorkflowParams {
  shellThickness: number;
  latticeType: MeshLatticeType;
  latticePitch: number;
  latticeThickness: number;
  latticePhase: number;
}
