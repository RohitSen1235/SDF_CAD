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
export type SelectionKind = "fixed";
export type LoadKind = "point" | "surface";

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
  resolution_xyz: [number, number, number];
}

export interface MeshPayloadBase64 {
  encoding: "mesh-f32-u32-base64-v1";
  vertex_count: number;
  face_count: number;
  vertices_b64: string;
  indices_b64: string;
  normals_b64: string;
}

export interface MeshPayloadBinary {
  encoding: "mesh-f32-u32-binary-v1";
  vertex_count: number;
  face_count: number;
  vertices: Float32Array;
  indices: Uint32Array;
  normals: Float32Array;
}

export type MeshPayload = MeshPayloadBase64 | MeshPayloadBinary;

export interface FieldPayloadBase64 {
  encoding: "f32-base64";
  resolution_xyz: [number, number, number];
  bounds: [[number, number], [number, number], [number, number]];
  data: string;
}

export interface FieldPayloadBinary {
  encoding: "f32-binary-v1";
  resolution_xyz: [number, number, number];
  bounds: [[number, number], [number, number], [number, number]];
  data: Float32Array;
}

export type FieldPayload = FieldPayloadBase64 | FieldPayloadBinary;

export interface PreviewStats {
  eval_ms: number;
  mesh_ms?: number | null;
  tri_count: number;
  voxel_count?: number;
  cache_hit?: boolean;
  field_cache_hit?: boolean;
  mesh_cache_hit?: boolean;
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

export interface UploadedFieldPreviewClientTelemetry {
  trace_id: string;
  client_response_wait_ms: number;
  client_download_ms: number;
  client_decode_ms: number;
  client_texture_upload_and_first_frame_ms: number;
  client_total_visible_ms: number;
}

export interface UploadedFieldPreviewTrace {
  traceId: string;
  clientResponseWaitMs: number;
  clientDownloadMs: number;
  clientDecodeMs: number;
  fieldAssignedAtMs: number;
}

export interface UploadedPreviewFieldResponse extends PreviewFieldResponse {
  hostField?: FieldPayload | null;
  trace?: UploadedFieldPreviewTrace | null;
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

export interface PreviewProgramResponse {
  program: SceneProgramPayload | null;
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

export interface UploadedMeshMemoryContext {
  meshSpan: number;
  meshExtents: [number, number, number];
  resolutionXYZ?: [number, number, number] | null;
  availableCpuBytes: number | null;
  availableGpuFreeBytes: number | null;
  availableGpuTotalBytes: number | null;
  cpuBytesPerVoxel: number;
  gpuBytesPerVoxel: number;
  meshGpuBytesPerVoxel: number;
  safetyFactor: number;
}

export interface UploadedMeshPreprocessResponse {
  mesh: MeshPayloadBinary;
  memoryContext: UploadedMeshMemoryContext | null;
}

export interface SelectionPoint {
  point_xyz: [number, number, number];
}

export interface ConstraintRegion {
  kind: SelectionKind;
  points: SelectionPoint[];
  radius: number;
}

export interface LoadRegion {
  kind: LoadKind;
  points: SelectionPoint[];
  direction_xyz: [number, number, number];
  magnitude: number;
  radius: number;
}

export interface StructuralMaterial {
  youngs_modulus: number;
  poissons_ratio: number;
  density_floor: number;
  stiffness_floor_ratio: number;
  simp_penalty: number;
}

export interface StructuralOptimizationConfig {
  resolution: number;
  target_volume_fraction: number;
  max_iterations: number;
  cg_max_iterations: number;
  cg_tolerance: number;
  optimization_tolerance: number;
  filter_radius_voxels: number;
  min_density: number;
  oc_move_limit: number;
  density_iso_threshold: number;
}

export interface StructuralOptimizationRequest {
  design_space_file_name: string;
  design_space_file_data_base64: string;
  non_design_space_file_name: string;
  non_design_space_file_data_base64: string;
  compute_backend: ComputeBackend;
  mesh_backend: MeshBackend;
  execution_mode: "auto" | "inline" | "queued";
  constraints: ConstraintRegion[];
  loads: LoadRegion[];
  material: StructuralMaterial;
  config: StructuralOptimizationConfig;
}

export interface StructuralOptimizationPreprocessResponse {
  design_mesh: MeshPayload;
  non_design_mesh: MeshPayload;
  combined_mesh: MeshPayload;
  bounds: [[number, number], [number, number], [number, number]];
  resolution_xyz: [number, number, number];
  diagnostics: string[];
}

export interface OptimizationHistoryEntry {
  iteration: number;
  objective_value: number;
  active_volume_fraction: number;
  removed_voxels: number;
  max_displacement: number;
}

export interface StructuralOptimizationIterationResult {
  iteration: number;
  objective_value: number;
  active_volume_fraction: number;
  removed_voxels: number;
  mesh?: MeshPayload | null;
  density_field?: FieldPayload | null;
  displacement_field?: FieldPayload | null;
  stress_field?: FieldPayload | null;
  strain_field?: FieldPayload | null;
}

export interface StructuralOptimizationResultResponse {
  history: OptimizationHistoryEntry[];
  final_iteration: StructuralOptimizationIterationResult;
  bounds: [[number, number], [number, number], [number, number]];
  resolution_xyz: [number, number, number];
  compute_backend_used: "cpu" | "cuda";
  mesh_backend_used: "cpu" | "cuda";
  stop_reason: "target_volume_reached" | "objective_converged" | "density_converged" | "max_iterations";
}

export interface StructuralOptimizationJobAcceptedResponse {
  job_id: string;
  status: "queued" | "running" | "succeeded" | "failed";
  status_url: string;
  result_url: string;
  progress_url: string;
}

export interface StructuralOptimizationProgressResponse {
  job_id: string;
  status: "queued" | "running" | "succeeded" | "failed";
  current_iteration: number;
  max_iterations: number;
  iterations: StructuralOptimizationIterationResult[];
  history: OptimizationHistoryEntry[];
  stop_reason?: StructuralOptimizationResultResponse["stop_reason"] | null;
  detail?: string | null;
  final_result?: StructuralOptimizationResultResponse | null;
}
