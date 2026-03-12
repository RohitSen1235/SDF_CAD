from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


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
    resolution: int = 64

    @field_validator("resolution")
    @classmethod
    def validate_resolution(cls, value: int) -> int:
        if value < 16:
            raise ValueError("resolution must be >= 16")
        if value > 256:
            raise ValueError("resolution must be <= 256")
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


class MeshPayload(BaseModel):
    vertices: list[list[float]]
    indices: list[list[int]]
    normals: list[list[float]]


class FieldPayload(BaseModel):
    encoding: Literal["f32-base64"] = "f32-base64"
    resolution: int
    bounds: list[list[float]]
    data: str


class PreviewStats(BaseModel):
    eval_ms: float
    mesh_ms: float | None = None
    tri_count: int
    voxel_count: int | None = None
    cache_hit: bool = False
    compute_precision: ComputePrecision = "float32"
    compute_backend: Literal["cpu", "cuda"] = "cpu"
    mesh_backend: Literal["cpu", "cuda"] = "cpu"
    preview_mode: Literal["mesh", "field"] = "mesh"


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


class ApiError(BaseModel):
    detail: str
    extra: dict[str, Any] | None = None
