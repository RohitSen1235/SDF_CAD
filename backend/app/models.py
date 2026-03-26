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
