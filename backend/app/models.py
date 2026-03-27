from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


NodeType = Literal[
    "primitive",
    "boolean",
    "transform",
    "field_expr",
    "domain_op",
    "lattice",
    "turbomachine",
]

ScalarRef = dict[str, str]
ScalarValue = float | int | str | ScalarRef

QualityProfile = Literal["interactive", "medium", "high", "ultra"]
ComputePrecision = Literal["float32", "float16"]
ComputeBackend = Literal["auto", "cpu", "cuda"]
MeshBackend = Literal["auto", "cpu", "cuda"]
MeshingMode = Literal["uniform", "adaptive"]
ExecutionMode = Literal["auto", "inline", "queued"]
UploadedFieldStorageMode = Literal["auto", "dense", "octree_sparse"]


class ParameterSpec(BaseModel):
    name: str
    type: Literal["float"] = "float"
    default: float
    min: float
    max: float
    step: float

    @model_validator(mode="after")
    def validate_range(self) -> "ParameterSpec":
        if self.min >= self.max:
            raise ValueError(f"Parameter {self.name}: min must be lower than max")
        if not (self.min <= self.default <= self.max):
            raise ValueError(f"Parameter {self.name}: default must be within min/max")
        if self.step <= 0:
            raise ValueError(f"Parameter {self.name}: step must be > 0")
        return self


class SceneNode(BaseModel):
    id: str
    type: NodeType
    primitive: str | None = None
    op: str | None = None
    inputs: list[str] = Field(default_factory=list)
    params: dict[str, ScalarValue] = Field(default_factory=dict)
    transform: dict[str, list[ScalarValue]] | None = None
    expr: dict[str, Any] | None = None
    bounds_hint: list[list[ScalarValue]] | None = None


class SceneIR(BaseModel):
    nodes: list[SceneNode]
    root_node_id: str
    parameter_schema: list[ParameterSpec] = Field(default_factory=list)
    source_hash: str | None = None


class CompileSceneRequest(BaseModel):
    source: str


class CompileDiagnostics(BaseModel):
    warnings: list[str] = Field(default_factory=list)
    inferred_bounds: list[list[float]] | None = None


class CompileSceneResponse(BaseModel):
    scene_ir: SceneIR
    diagnostics: CompileDiagnostics = Field(default_factory=CompileDiagnostics)


class GridConfig(BaseModel):
    bounds: list[list[float]] = Field(
        default_factory=lambda: [[-1.5, 1.5], [-1.5, 1.5], [-1.5, 1.5]]
    )
    resolution_xyz: list[int] = Field(default_factory=lambda: [64, 64, 64])

    @field_validator("resolution_xyz")
    @classmethod
    def validate_resolution_xyz(cls, value: list[int]) -> list[int]:
        if len(value) != 3:
            raise ValueError("resolution_xyz must contain 3 axis values")
        for axis_res in value:
            if axis_res < 16:
                raise ValueError("resolution_xyz values must be >= 16")
            if axis_res > 256:
                raise ValueError("resolution_xyz values must be <= 256")
        return value

    @field_validator("bounds")
    @classmethod
    def validate_bounds(cls, value: list[list[float]]) -> list[list[float]]:
        if len(value) != 3:
            raise ValueError("bounds must contain 3 axis ranges")
        for axis in value:
            if len(axis) != 2:
                raise ValueError("each axis range must contain [min, max]")
            low, high = axis
            if low >= high:
                raise ValueError("each axis range must have min < max")
            if abs(low) > 200 or abs(high) > 200:
                raise ValueError("bounds must stay within [-200, 200]")
        return value


class PreviewMeshRequest(BaseModel):
    scene_ir: SceneIR
    parameter_values: dict[str, float] = Field(default_factory=dict)
    grid: GridConfig | None = None
    quality_profile: QualityProfile = "interactive"
    compute_precision: ComputePrecision = "float32"
    compute_backend: ComputeBackend = "auto"
    mesh_backend: MeshBackend = "auto"
    meshing_mode: MeshingMode = "uniform"
    execution_mode: ExecutionMode = "auto"


class MeshPayload(BaseModel):
    encoding: Literal["mesh-f32-u32-base64-v1"] = "mesh-f32-u32-base64-v1"
    vertex_count: int
    face_count: int
    vertices_b64: str
    indices_b64: str
    normals_b64: str


class FieldPayload(BaseModel):
    encoding: Literal["f32-base64"] = "f32-base64"
    resolution_xyz: list[int]
    bounds: list[list[float]]
    data: str


class PreviewStats(BaseModel):
    eval_ms: float
    mesh_ms: float | None = None
    tri_count: int
    voxel_count: int | None = None
    cache_hit: bool = False
    field_cache_hit: bool = False
    mesh_cache_hit: bool = False
    compute_precision: ComputePrecision = "float32"
    compute_backend: Literal["cpu", "cuda"] = "cpu"
    mesh_backend: Literal["cpu", "cuda"] = "cpu"
    preview_mode: Literal["mesh", "field", "analytic_raymarch"] = "mesh"
    compile_ms: float | None = None
    program_bytes: int | None = None
    gpu_eval_mode: str | None = None
    fallback_reason: str | None = None


class PreviewMeshResponse(BaseModel):
    mesh: MeshPayload
    stats: PreviewStats
    field: FieldPayload | None = None


class PreviewFieldRequest(BaseModel):
    scene_ir: SceneIR
    parameter_values: dict[str, float] = Field(default_factory=dict)
    grid: GridConfig | None = None
    quality_profile: QualityProfile = "interactive"
    compute_precision: ComputePrecision = "float32"
    compute_backend: ComputeBackend = "auto"


class PreviewFieldResponse(BaseModel):
    field: FieldPayload
    stats: PreviewStats


class UploadedPreviewFieldResponse(PreviewFieldResponse):
    host_field: FieldPayload | None = None


class UploadedFieldPreviewClientTelemetry(BaseModel):
    trace_id: str
    client_response_wait_ms: float = Field(ge=0.0)
    client_download_ms: float = Field(ge=0.0)
    client_decode_ms: float = Field(ge=0.0)
    client_texture_upload_and_first_frame_ms: float = Field(ge=0.0)
    client_total_visible_ms: float = Field(ge=0.0)


class ExportMeshRequest(BaseModel):
    scene_ir: SceneIR
    parameter_values: dict[str, float] = Field(default_factory=dict)
    format: Literal["stl", "obj"] = "stl"
    quality_profile: QualityProfile = "high"
    grid: GridConfig | None = None
    compute_precision: ComputePrecision = "float32"
    compute_backend: ComputeBackend = "auto"
    mesh_backend: MeshBackend = "auto"
    meshing_mode: MeshingMode = "uniform"
    execution_mode: ExecutionMode = "auto"


class JobAcceptedResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "succeeded", "failed"] = "queued"
    status_url: str
    result_url: str


class StructuralOptimizationJobAcceptedResponse(JobAcceptedResponse):
    progress_url: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    task_name: str
    detail: str | None = None


class PreviewWsRequest(BaseModel):
    scene_ir: SceneIR
    parameter_values: dict[str, float] = Field(default_factory=dict)
    base_grid: GridConfig | None = None
    quality_profile: QualityProfile = "high"
    compute_precision: ComputePrecision = "float32"
    compute_backend: ComputeBackend = "auto"
    mesh_backend: MeshBackend = "auto"
    meshing_mode: MeshingMode = "uniform"


class PreviewWsResponse(BaseModel):
    phase: Literal["coarse", "fine", "error"]
    mesh: MeshPayload | None = None
    stats: PreviewStats | None = None
    error: str | None = None


class UploadedMeshPreviewWsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_name: str
    file_data_base64: str
    shell_thickness: float
    lattice_type: Literal["gyroid", "schwarz_p", "diamond"]
    lattice_pitch: float
    lattice_thickness: float
    lattice_phase: float = 0.0
    voxels_per_lattice_period: int = 6
    compute_backend: ComputeBackend = "auto"
    mesh_backend: MeshBackend = "auto"
    meshing_mode: MeshingMode = "uniform"
    field_storage_mode: UploadedFieldStorageMode = "auto"


class UploadedMeshPreviewWsResponse(BaseModel):
    phase: Literal["field", "mesh", "error"]
    mesh: MeshPayload | None = None
    field: FieldPayload | None = None
    stats: PreviewStats | None = None
    error: str | None = None


class ApiError(BaseModel):
    detail: str
    extra: dict[str, Any] | None = None


class PreviewProgramRequest(BaseModel):
    scene_ir: SceneIR
    parameter_values: dict[str, float] = Field(default_factory=dict)
    quality_profile: QualityProfile = "high"
    grid: GridConfig | None = None


class SceneProgramPayload(BaseModel):
    mode: Literal["dsl"]
    bounds: list[list[float]]
    glsl_sdf: str
    quality_profile: QualityProfile
    max_steps: int
    hit_epsilon: float
    normal_epsilon: float


class ProgramCapabilities(BaseModel):
    analytic_supported: bool
    fallback_reason: str | None = None


class PreviewProgramResponse(BaseModel):
    program: SceneProgramPayload | None = None
    capabilities: ProgramCapabilities
    stats: PreviewStats


class SelectionPoint(BaseModel):
    point_xyz: list[float] = Field(min_length=3, max_length=3)

    @field_validator("point_xyz")
    @classmethod
    def validate_point_xyz(cls, value: list[float]) -> list[float]:
        if len(value) != 3 or not all(isinstance(v, (int, float)) and abs(float(v)) < 1e6 for v in value):
            raise ValueError("point_xyz must contain 3 finite coordinates")
        return [float(v) for v in value]


class ConstraintRegion(BaseModel):
    kind: Literal["fixed"] = "fixed"
    points: list[SelectionPoint] = Field(default_factory=list)
    radius: float = Field(default=0.0, ge=0.0)


class LoadRegion(BaseModel):
    kind: Literal["point", "surface"] = "point"
    points: list[SelectionPoint] = Field(default_factory=list)
    direction_xyz: list[float] = Field(default_factory=lambda: [0.0, -1.0, 0.0], min_length=3, max_length=3)
    magnitude: float = Field(default=1.0)
    radius: float = Field(default=0.0, ge=0.0)

    @field_validator("direction_xyz")
    @classmethod
    def validate_direction_xyz(cls, value: list[float]) -> list[float]:
        if len(value) != 3:
            raise ValueError("direction_xyz must contain 3 values")
        parsed = [float(v) for v in value]
        if not any(abs(v) > 0.0 for v in parsed):
            raise ValueError("direction_xyz must not be the zero vector")
        return parsed


class StructuralMaterial(BaseModel):
    youngs_modulus: float = Field(default=1.0, gt=0.0)
    poissons_ratio: float = Field(default=0.30, ge=0.0, lt=0.5)
    density_floor: float = Field(default=1e-3, gt=0.0, le=1.0)
    stiffness_floor_ratio: float = Field(default=1e-3, gt=0.0, le=1.0)
    simp_penalty: float = Field(default=3.0, ge=1.0, le=6.0)


class StructuralOptimizationConfig(BaseModel):
    resolution: int = Field(default=96, ge=32, le=192)
    target_volume_fraction: float = Field(default=0.35, gt=0.01, lt=1.0)
    max_iterations: int = Field(default=40, ge=1, le=64)
    cg_max_iterations: int = Field(default=200, ge=8, le=2000)
    cg_tolerance: float = Field(default=1e-6, gt=0.0, le=1e-2)
    optimization_tolerance: float = Field(default=1e-3, gt=0.0, le=1e-1)
    filter_radius_voxels: float = Field(default=1.5, ge=0.0, le=8.0)
    min_density: float = Field(default=1e-3, gt=0.0, le=1.0)
    oc_move_limit: float = Field(default=0.2, gt=0.0, le=1.0)
    density_iso_threshold: float = Field(default=0.30, gt=0.0, lt=1.0)


class StructuralOptimizationRequest(BaseModel):
    design_space_file_name: str
    design_space_file_data_base64: str
    non_design_space_file_name: str
    non_design_space_file_data_base64: str
    compute_backend: ComputeBackend = "auto"
    mesh_backend: MeshBackend = "auto"
    execution_mode: ExecutionMode = "auto"
    constraints: list[ConstraintRegion] = Field(default_factory=list)
    loads: list[LoadRegion] = Field(default_factory=list)
    material: StructuralMaterial = Field(default_factory=StructuralMaterial)
    config: StructuralOptimizationConfig = Field(default_factory=StructuralOptimizationConfig)


class StructuralOptimizationPreprocessResponse(BaseModel):
    design_mesh: MeshPayload
    non_design_mesh: MeshPayload
    combined_mesh: MeshPayload
    bounds: list[list[float]]
    resolution_xyz: list[int]
    diagnostics: list[str] = Field(default_factory=list)


class OptimizationHistoryEntry(BaseModel):
    iteration: int
    objective_value: float
    active_volume_fraction: float
    removed_voxels: int
    max_displacement: float


class StructuralOptimizationIterationResult(BaseModel):
    iteration: int
    objective_value: float
    active_volume_fraction: float
    removed_voxels: int
    mesh: MeshPayload | None = None
    density_field: FieldPayload | None = None
    displacement_field: FieldPayload | None = None
    stress_field: FieldPayload | None = None
    strain_field: FieldPayload | None = None


StructuralOptimizationStopReason = Literal[
    "target_volume_reached",
    "objective_converged",
    "density_converged",
    "max_iterations",
]


class StructuralOptimizationResultResponse(BaseModel):
    history: list[OptimizationHistoryEntry] = Field(default_factory=list)
    final_iteration: StructuralOptimizationIterationResult
    bounds: list[list[float]]
    resolution_xyz: list[int]
    compute_backend_used: Literal["cpu", "cuda"] = "cpu"
    mesh_backend_used: Literal["cpu", "cuda"] = "cpu"
    stop_reason: StructuralOptimizationStopReason


class StructuralOptimizationIterationWebhookRequest(BaseModel):
    iteration_result: StructuralOptimizationIterationResult | None = None
    history_entry: OptimizationHistoryEntry | None = None
    bounds: list[list[float]] | None = None
    resolution_xyz: list[int] | None = None
    compute_backend_used: Literal["cpu", "cuda"] | None = None
    mesh_backend_used: Literal["cpu", "cuda"] | None = None
    is_final: bool = False
    stop_reason: StructuralOptimizationStopReason | None = None
    failure_detail: str | None = None

    @model_validator(mode="after")
    def validate_payload(self) -> "StructuralOptimizationIterationWebhookRequest":
        if self.failure_detail:
            return self
        if self.iteration_result is None or self.history_entry is None:
            raise ValueError("iteration_result and history_entry are required unless failure_detail is provided")
        if self.bounds is None or self.resolution_xyz is None:
            raise ValueError("bounds and resolution_xyz are required for iteration callbacks")
        if self.compute_backend_used is None or self.mesh_backend_used is None:
            raise ValueError("compute_backend_used and mesh_backend_used are required for iteration callbacks")
        return self


class StructuralOptimizationProgressResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    current_iteration: int = 0
    max_iterations: int
    iterations: list[StructuralOptimizationIterationResult] = Field(default_factory=list)
    history: list[OptimizationHistoryEntry] = Field(default_factory=list)
    stop_reason: StructuralOptimizationStopReason | None = None
    detail: str | None = None
    final_result: StructuralOptimizationResultResponse | None = None
