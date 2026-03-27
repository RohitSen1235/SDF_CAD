import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { Viewer } from "./components/Viewer";
import {
  buildStructuralOptimizationRequest,
  compileScene,
  commitUploadedMesh,
  exportMesh,
  exportUploadedMesh,
  previewField,
  previewMesh,
  preprocessStructuralOptimization,
  preprocessUploadedMesh,
  runStructuralOptimizationPhased,
  previewUploadedMeshField,
  submitUploadedFieldPreviewTelemetry
} from "./lib/api";
import {
  CompileDiagnostics,
  ConstraintRegion,
  ComputeBackend,
  ComputePrecision,
  FieldPayload,
  LoadRegion,
  MeshBackend,
  MeshLatticeType,
  MeshingMode,
  MeshPayload,
  MeshWorkflowParams,
  StructuralMaterial,
  StructuralOptimizationConfig,
  StructuralOptimizationIterationResult,
  StructuralOptimizationPreprocessResponse,
  StructuralOptimizationResultResponse,
  PreviewStats,
  QualityProfile,
  SceneIR,
  UploadedFieldPreviewTrace,
  UploadedMeshMemoryContext
} from "./types";

interface SavedFieldExpression {
  name: string;
  source: string;
  updatedAt: number;
}

type WorkflowMode = "dsl" | "mesh" | "generative";
type TransformMode = "translate" | "rotate" | "scale";
type GenerativePickMode = "fixed" | "load" | null;

interface ViewerMarker {
  position: [number, number, number];
  color: string;
  size?: number;
}

interface WorkspaceDraft {
  version: 1;
  workflow: WorkflowMode;
  source: string;
  quality: QualityProfile;
  computePrecision: ComputePrecision;
  computeBackend: ComputeBackend;
  meshBackend: MeshBackend;
  meshingMode: MeshingMode;
  meshShellThickness: number;
  meshLatticeType: MeshLatticeType;
  meshLatticePitch: number;
  meshLatticeThickness: number;
  meshLatticePhase: number;
  voxelsPerLatticePeriod: number;
  wireframe: boolean;
  showGrid: boolean;
  showAxes: boolean;
  transformMode: TransformMode;
  sectionEnabled: boolean;
  sectionLevel: number;
}

const GYROID_FILL_SOURCE = `# Gyroid conformal fill
param shell default=0.08 min=0.03 max=0.2 step=0.01
host = sphere(r=1.0)
lat = gyroid(pitch=0.45, thickness=0.1)
root = conformal_fill(host, lat, wall=$shell, mode="hybrid")
`;

const COMPRESSOR_STAGE_SOURCE = `# Centrifugal compressor with volute
param twist default=0.9 min=0.1 max=1.4 step=0.05
imp = impeller_centrifugal(r_in=0.22, r_out=0.95, hub_h=0.45, blade_count=9, blade_twist=$twist)
vol = volute_casing(throat_radius=0.34, outlet_radius=1.25, width=0.48, wall=0.08)
root = union(imp, vol)
`;

const FIELD_EXPRESSION_SOURCE = `# Pure implicit expression
a = sin(x * 3.0) + cos(y * 3.0) + sin(z * 3.0)
root = abs(a) - 0.45
`;

const EXAMPLES: Record<string, string> = {
  "Field Expression": FIELD_EXPRESSION_SOURCE
};

const FUNCTION_SIGNATURES = [
  'spline(points="x y z; ...", radius=0.08, samples=20, closed=0|1)',
  'freeform_surface(heights="16 numbers", x=1.0, z=1.0, thickness=0.06)',
  'freeform(heights="16 numbers", x=1.0, z=1.0, thickness=0.06)',
  "gyroid(pitch=1.0, phase=0.0, thickness=0.08)",
  "schwarz_p(pitch=1.0, phase=0.0, thickness=0.08)",
  "diamond(pitch=1.0, phase=0.0, thickness=0.08)",
  'strut_lattice(type="bcc|fcc|octet", pitch=1.0, radius=0.08)',
  'conformal_fill(host, lattice, wall=0.1, offset=0.0, mode="shell|clip|hybrid")',
  "repeat(child, x=1.0, y=1.0, z=1.0)",
  'twist(child, k=1.0, axis="x|y|z")',
  'bend(child, k=0.5, axis="x|y|z")',
  "shell(child, t=0.1)",
  "offset(child, d=0.0)",
  'circular_array(child, count=12, axis="x|y|z", phase=0.0)',
  "impeller_centrifugal(r_in, r_out, hub_h, blade_count, blade_thickness, blade_twist, shroud_gap)",
  "radial_turbine(r_in, r_out, hub_h, blade_count, blade_thickness, blade_twist)",
  "volute_casing(throat_radius, outlet_radius, area_growth, width, wall, tongue_clearance)"
];

const QUALITY_ORDER: QualityProfile[] = ["interactive", "medium", "high", "ultra"];
const MODULE_TABS: Array<{ mode: WorkflowMode; label: string }> = [
  { mode: "dsl", label: "DSL" },
  { mode: "mesh", label: "Lattice Infill" },
  { mode: "generative", label: "Topology Opt" }
];
const SAVED_EXPRESSIONS_KEY = "sdfcad.savedFieldExpressions.v1";
const WORKSPACE_DRAFT_KEY = "sdfcad.workspaceDraft.v1";
const BUILTIN_SAVED_EXPRESSIONS: SavedFieldExpression[] = [
  {
    name: "Gyroid Fill",
    source: GYROID_FILL_SOURCE,
    updatedAt: 0
  },
  {
    name: "Compressor Stage",
    source: COMPRESSOR_STAGE_SOURCE,
    updatedAt: 0
  }
];

function loadSavedExpressions(): SavedFieldExpression[] {
  if (typeof window === "undefined") {
    return BUILTIN_SAVED_EXPRESSIONS;
  }
  try {
    const raw = window.localStorage.getItem(SAVED_EXPRESSIONS_KEY);
    if (!raw) {
      return BUILTIN_SAVED_EXPRESSIONS;
    }
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) {
      return BUILTIN_SAVED_EXPRESSIONS;
    }
    const loaded = parsed
      .map((item) => ({
        name: typeof item?.name === "string" ? item.name : "",
        source: typeof item?.source === "string" ? item.source : "",
        updatedAt: typeof item?.updatedAt === "number" ? item.updatedAt : 0
      }))
      .filter((item) => item.name.length > 0 && item.source.length > 0)
      .sort((a, b) => b.updatedAt - a.updatedAt);

    const mergedByName = new Map<string, SavedFieldExpression>();
    for (const entry of BUILTIN_SAVED_EXPRESSIONS) {
      mergedByName.set(entry.name, entry);
    }
    for (const entry of loaded) {
      mergedByName.set(entry.name, entry);
    }
    return [...mergedByName.values()].sort((a, b) => b.updatedAt - a.updatedAt);
  } catch {
    return BUILTIN_SAVED_EXPRESSIONS;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function asFiniteNumber(value: unknown): number | null {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return null;
  }
  return value;
}

function loadWorkspaceDraft(): Partial<WorkspaceDraft> {
  if (typeof window === "undefined") {
    return {};
  }
  try {
    const raw = window.localStorage.getItem(WORKSPACE_DRAFT_KEY);
    if (!raw) {
      return {};
    }
    const parsed = JSON.parse(raw);
    if (!isRecord(parsed) || parsed.version !== 1) {
      return {};
    }

    const draft: Partial<WorkspaceDraft> = {};

    if (parsed.workflow === "dsl" || parsed.workflow === "mesh" || parsed.workflow === "generative") {
      draft.workflow = parsed.workflow;
    }
    if (typeof parsed.source === "string") {
      draft.source = parsed.source;
    }
    if (
      parsed.quality === "interactive" ||
      parsed.quality === "medium" ||
      parsed.quality === "high" ||
      parsed.quality === "ultra"
    ) {
      draft.quality = parsed.quality;
    }
    if (parsed.computePrecision === "float32" || parsed.computePrecision === "float16") {
      draft.computePrecision = parsed.computePrecision;
    }
    if (parsed.computeBackend === "auto" || parsed.computeBackend === "cpu" || parsed.computeBackend === "cuda") {
      draft.computeBackend = parsed.computeBackend;
    }
    if (parsed.meshBackend === "auto" || parsed.meshBackend === "cpu" || parsed.meshBackend === "cuda") {
      draft.meshBackend = parsed.meshBackend;
    }
    if (parsed.meshingMode === "uniform" || parsed.meshingMode === "adaptive") {
      draft.meshingMode = parsed.meshingMode;
    }
    if (parsed.meshLatticeType === "gyroid" || parsed.meshLatticeType === "schwarz_p" || parsed.meshLatticeType === "diamond") {
      draft.meshLatticeType = parsed.meshLatticeType;
    }
    if (parsed.transformMode === "translate" || parsed.transformMode === "rotate" || parsed.transformMode === "scale") {
      draft.transformMode = parsed.transformMode;
    }
    if (typeof parsed.wireframe === "boolean") {
      draft.wireframe = parsed.wireframe;
    }
    if (typeof parsed.showGrid === "boolean") {
      draft.showGrid = parsed.showGrid;
    }
    if (typeof parsed.showAxes === "boolean") {
      draft.showAxes = parsed.showAxes;
    }
    if (typeof parsed.sectionEnabled === "boolean") {
      draft.sectionEnabled = parsed.sectionEnabled;
    }

    const meshShellThickness = asFiniteNumber(parsed.meshShellThickness);
    if (meshShellThickness !== null) {
      draft.meshShellThickness = meshShellThickness;
    }
    const meshLatticePitch = asFiniteNumber(parsed.meshLatticePitch);
    if (meshLatticePitch !== null) {
      draft.meshLatticePitch = meshLatticePitch;
    }
    const meshLatticeThickness = asFiniteNumber(parsed.meshLatticeThickness);
    if (meshLatticeThickness !== null) {
      draft.meshLatticeThickness = meshLatticeThickness;
    }
    const meshLatticePhase = asFiniteNumber(parsed.meshLatticePhase);
    if (meshLatticePhase !== null) {
      draft.meshLatticePhase = meshLatticePhase;
    }
    const sectionLevel = asFiniteNumber(parsed.sectionLevel);
    if (sectionLevel !== null) {
      draft.sectionLevel = sectionLevel;
    }
    if (parsed.voxelsPerLatticePeriod === 4 || parsed.voxelsPerLatticePeriod === 6 || parsed.voxelsPerLatticePeriod === 8) {
      draft.voxelsPerLatticePeriod = parsed.voxelsPerLatticePeriod;
    }

    return draft;
  } catch {
    return {};
  }
}

function inferPreviewBounds(
  diagnostics: CompileDiagnostics | null
): [[number, number], [number, number], [number, number]] | undefined {
  if (!diagnostics?.inferred_bounds) {
    return undefined;
  }
  const src = diagnostics.inferred_bounds;
  const expanded = src.map((axis) => {
    const center = (axis[0] + axis[1]) * 0.5;
    const half = Math.max((axis[1] - axis[0]) * 0.6, 0.5);
    return [center - half, center + half] as [number, number];
  });
  return expanded as [[number, number], [number, number], [number, number]];
}

function resolutionForQuality(profile: QualityProfile): number {
  if (profile === "interactive") {
    return 64;
  }
  if (profile === "medium") {
    return 128;
  }
  if (profile === "high") {
    return 192;
  }
  return 256;
}

function capResolutionXYZ(
  resolutionXYZ: [number, number, number],
  maxTotalVoxels: number
): [number, number, number] {
  const current = resolutionXYZ[0] * resolutionXYZ[1] * resolutionXYZ[2];
  if (current <= maxTotalVoxels) {
    return resolutionXYZ;
  }
  const scale = Math.cbrt(maxTotalVoxels / current);
  const dims = [
    Math.max(24, Math.floor(resolutionXYZ[0] * scale)),
    Math.max(24, Math.floor(resolutionXYZ[1] * scale)),
    Math.max(24, Math.floor(resolutionXYZ[2] * scale))
  ];
  while (dims[0] * dims[1] * dims[2] > maxTotalVoxels) {
    const maxIndex = dims[0] >= dims[1] && dims[0] >= dims[2] ? 0 : dims[1] >= dims[2] ? 1 : 2;
    if (dims[maxIndex] <= 24) {
      break;
    }
    dims[maxIndex] -= 1;
  }
  return [dims[0], dims[1], dims[2]];
}

function computeRequiredResolutionXYZ(
  meshExtents: [number, number, number],
  latticePitch: number,
  voxelsPerPeriod: number
): [number, number, number] {
  const spacing = latticePitch / Math.max(voxelsPerPeriod, 1);
  const padded = [
    meshExtents[0] + 4.0 * latticePitch,
    meshExtents[1] + 4.0 * latticePitch,
    meshExtents[2] + 4.0 * latticePitch
  ];
  const raw: [number, number, number] = [
    Math.max(24, Math.ceil(padded[0] / spacing) + 1),
    Math.max(24, Math.ceil(padded[1] / spacing) + 1),
    Math.max(24, Math.ceil(padded[2] / spacing) + 1)
  ];
  return capResolutionXYZ(raw, 1024 ** 3);
}

function computeMinShellThickness(latticePitch: number, voxelsPerPeriod: number): number {
  if (voxelsPerPeriod <= 0) {
    return 0;
  }
  return 2.0 * (latticePitch / voxelsPerPeriod);
}

function computeMinLatticeThickness(latticePitch: number, voxelsPerPeriod: number): number {
  if (voxelsPerPeriod <= 0) {
    return 0;
  }
  return latticePitch / voxelsPerPeriod;
}

function estimateRequiredBytes(
  resolutionXYZ: [number, number, number],
  bytesPerVoxel: number,
  safetyFactor: number
): number {
  const voxelCount = resolutionXYZ[0] * resolutionXYZ[1] * resolutionXYZ[2];
  return Math.ceil(voxelCount * bytesPerVoxel * safetyFactor);
}

function formatResolutionXYZ(resolutionXYZ: [number, number, number]): string {
  return `${resolutionXYZ[0]} x ${resolutionXYZ[1]} x ${resolutionXYZ[2]}`;
}

function bytesToGiB(value: number): string {
  return `${(value / (1024 ** 3)).toFixed(2)} GiB`;
}

function bytesToMiB(value: number): string {
  return `${Math.round(value / (1024 * 1024))} MiB`;
}

function describeOptimizationStopReason(reason: StructuralOptimizationResultResponse["stop_reason"] | null): string | null {
  if (reason === "target_volume_reached") {
    return "Stopped: target volume fraction reached.";
  }
  if (reason === "objective_converged") {
    return "Stopped: objective converged.";
  }
  if (reason === "density_converged") {
    return "Stopped: density update converged.";
  }
  if (reason === "max_iterations") {
    return "Stopped: maximum iteration count reached.";
  }
  return null;
}

export default function App() {
  const [workspaceDraft] = useState<Partial<WorkspaceDraft>>(() => loadWorkspaceDraft());
  const [workflow, setWorkflow] = useState<WorkflowMode>(workspaceDraft.workflow ?? "dsl");

  const [source, setSource] = useState(workspaceDraft.source ?? EXAMPLES["Field Expression"]);
  const [sourceDirty, setSourceDirty] = useState(true);
  const [sceneIr, setSceneIr] = useState<SceneIR | null>(null);
  const [diagnostics, setDiagnostics] = useState<CompileDiagnostics | null>(null);
  const [params, setParams] = useState<Record<string, number>>({});

  const [meshFile, setMeshFile] = useState<File | null>(null);
  const [generativeDesignFile, setGenerativeDesignFile] = useState<File | null>(null);
  const [generativeNonDesignFile, setGenerativeNonDesignFile] = useState<File | null>(null);
  const [meshShellThickness, setMeshShellThickness] = useState(workspaceDraft.meshShellThickness ?? 2.0);
  const [meshLatticeType, setMeshLatticeType] = useState<MeshLatticeType>(workspaceDraft.meshLatticeType ?? "gyroid");
  const [meshLatticePitch, setMeshLatticePitch] = useState(workspaceDraft.meshLatticePitch ?? 5.0);
  const [meshLatticeThickness, setMeshLatticeThickness] = useState(workspaceDraft.meshLatticeThickness ?? 0.5);
  const [meshLatticePhase, setMeshLatticePhase] = useState(workspaceDraft.meshLatticePhase ?? 0.0);
  const [voxelsPerLatticePeriod, setVoxelsPerLatticePeriod] = useState(workspaceDraft.voxelsPerLatticePeriod ?? 6);
  const [meshCommitted, setMeshCommitted] = useState(false);
  const [meshMemoryContext, setMeshMemoryContext] = useState<UploadedMeshMemoryContext | null>(null);

  const [field, setField] = useState<FieldPayload | null>(null);
  const [mesh, setMesh] = useState<MeshPayload | null>(null);
  const [stats, setStats] = useState<PreviewStats | null>(null);
  const [uploadedFieldPreviewTrace, setUploadedFieldPreviewTrace] = useState<UploadedFieldPreviewTrace | null>(null);
  const [generativePreprocess, setGenerativePreprocess] = useState<StructuralOptimizationPreprocessResponse | null>(null);
  const [generativeHistory, setGenerativeHistory] = useState<StructuralOptimizationResultResponse["history"]>([]);
  const [generativeStopReason, setGenerativeStopReason] = useState<StructuralOptimizationResultResponse["stop_reason"] | null>(null);
  const [generativePickMode, setGenerativePickMode] = useState<GenerativePickMode>(null);
  const [generativeConstraints, setGenerativeConstraints] = useState<ConstraintRegion[]>([]);
  const [generativeLoads, setGenerativeLoads] = useState<LoadRegion[]>([]);
  const [generativeLoadDirection, setGenerativeLoadDirection] = useState<[number, number, number]>([0, -1, 0]);
  const [generativeLoadMagnitude, setGenerativeLoadMagnitude] = useState(1.0);
  const [generativeMaterial, setGenerativeMaterial] = useState<StructuralMaterial>({
    youngs_modulus: 1.0,
    poissons_ratio: 0.30,
    density_floor: 1e-3,
    stiffness_floor_ratio: 1e-3,
    simp_penalty: 3.0
  });
  const [generativeConfig, setGenerativeConfig] = useState<StructuralOptimizationConfig>({
    resolution: 96,
    target_volume_fraction: 0.35,
    max_iterations: 40,
    cg_max_iterations: 200,
    cg_tolerance: 1e-6,
    optimization_tolerance: 1e-3,
    filter_radius_voxels: 1.5,
    min_density: 1e-3,
    oc_move_limit: 0.2,
    density_iso_threshold: 0.3
  });
  const [error, setError] = useState<string | null>(null);
  const [isCompiling, setIsCompiling] = useState(false);
  const [isPreviewing, setIsPreviewing] = useState(false);
  const [wireframe, setWireframe] = useState(workspaceDraft.wireframe ?? false);
  const [showGrid, setShowGrid] = useState(workspaceDraft.showGrid ?? true);
  const [showAxes, setShowAxes] = useState(workspaceDraft.showAxes ?? true);
  const [transformMode, setTransformMode] = useState<TransformMode>(workspaceDraft.transformMode ?? "translate");
  const [fitSignal, setFitSignal] = useState(0);
  const [quality, setQuality] = useState<QualityProfile>(workspaceDraft.quality ?? "high");
  const [computePrecision, setComputePrecision] = useState<ComputePrecision>(workspaceDraft.computePrecision ?? "float32");
  const [computeBackend, setComputeBackend] = useState<ComputeBackend>(workspaceDraft.computeBackend ?? "auto");
  const [meshBackend, setMeshBackend] = useState<MeshBackend>(workspaceDraft.meshBackend ?? "auto");
  const [meshingMode, setMeshingMode] = useState<MeshingMode>(workspaceDraft.meshingMode ?? "uniform");
  const [uploadedMeshFieldSignature, setUploadedMeshFieldSignature] = useState<string | null>(null);
  const [sectionEnabled, setSectionEnabled] = useState(workspaceDraft.sectionEnabled ?? false);
  const [sectionLevel, setSectionLevel] = useState(workspaceDraft.sectionLevel ?? 0);
  const [sectionYBounds, setSectionYBounds] = useState<[number, number]>([-2, 2]);
  const [isSignatureHelpOpen, setIsSignatureHelpOpen] = useState(false);
  const [savedExpressions, setSavedExpressions] = useState<SavedFieldExpression[]>(() => loadSavedExpressions());
  const [expressionName, setExpressionName] = useState("");
  const [selectedExpressionName, setSelectedExpressionName] = useState("");
  const [expressionStatus, setExpressionStatus] = useState<string | null>(null);
  const previewRunIdRef = useRef(0);
  const previewControllersRef = useRef<{ field: AbortController | null; mesh: AbortController | null }>({
    field: null,
    mesh: null
  });
  const meshFieldPreviewRunIdRef = useRef(0);
  const meshFieldPreviewControllerRef = useRef<AbortController | null>(null);
  const preprocessControllerRef = useRef<AbortController | null>(null);
  const generativePreprocessControllerRef = useRef<AbortController | null>(null);
  const uploadedFieldTelemetrySentRef = useRef<Set<string>>(new Set());

  const meshWorkflowParams = useMemo<MeshWorkflowParams>(
    () => ({
      shellThickness: meshShellThickness,
      latticeType: meshLatticeType,
      latticePitch: meshLatticePitch,
      latticeThickness: meshLatticeThickness,
      latticePhase: meshLatticePhase
    }),
    [meshShellThickness, meshLatticeType, meshLatticePitch, meshLatticeThickness, meshLatticePhase]
  );

  const currentUploadedMeshFieldSignature = useMemo(() => {
    if (!meshFile) {
      return null;
    }
    return [
      meshFile.name,
      meshFile.size,
      meshFile.lastModified,
      meshShellThickness,
      meshLatticeType,
      meshLatticePitch,
      meshLatticeThickness,
      meshLatticePhase,
      computeBackend,
      voxelsPerLatticePeriod
    ].join("|");
  }, [
    meshFile,
    meshShellThickness,
    meshLatticeType,
    meshLatticePitch,
    meshLatticeThickness,
    meshLatticePhase,
    computeBackend,
    voxelsPerLatticePeriod
  ]);

  const meshMemoryRisk = (() => {
    if (!meshMemoryContext) {
      return null;
    }

    // Always derive the live estimate from the current controls.
    // Backend resolution hints are useful for diagnostics, but they must not
    // freeze the gate after the user changes pitch / sampling settings.
    const resolutionXYZ = computeRequiredResolutionXYZ(
      meshMemoryContext.meshExtents,
      meshLatticePitch,
      voxelsPerLatticePeriod
    );
    const resolutionLabel = formatResolutionXYZ(resolutionXYZ);
    const requiredCpuBytes = estimateRequiredBytes(
      resolutionXYZ,
      meshMemoryContext.cpuBytesPerVoxel,
      meshMemoryContext.safetyFactor
    );
    const requiredFieldGpuBytes = estimateRequiredBytes(
      resolutionXYZ,
      meshMemoryContext.gpuBytesPerVoxel,
      meshMemoryContext.safetyFactor
    );
    const requiredMeshGpuBytes =
      meshBackend !== "cpu" && meshingMode !== "adaptive"
        ? estimateRequiredBytes(resolutionXYZ, meshMemoryContext.meshGpuBytesPerVoxel, meshMemoryContext.safetyFactor)
        : 0;
    const requiredGpuBytes = requiredFieldGpuBytes + requiredMeshGpuBytes;
    const cpuFatal =
      meshMemoryContext.availableCpuBytes != null && requiredCpuBytes > meshMemoryContext.availableCpuBytes;
    const fieldGpuCheckEnabled =
      computeBackend === "cuda" || (computeBackend === "auto" && meshMemoryContext.availableGpuFreeBytes != null);
    const meshGpuCheckEnabled =
      meshBackend !== "cpu" &&
      meshingMode !== "adaptive" &&
      meshMemoryContext.availableGpuFreeBytes != null;
    const gpuCheckEnabled = fieldGpuCheckEnabled || meshGpuCheckEnabled;
    const gpuFatal =
      gpuCheckEnabled &&
      meshMemoryContext.availableGpuFreeBytes != null &&
      requiredGpuBytes > meshMemoryContext.availableGpuFreeBytes;
    const fatal = cpuFatal || gpuFatal;

    let fatalMessage: string | null = null;
    if (fatal) {
      const parts = [
        `Estimated memory exceeds available capacity for resolution ${resolutionLabel}.`,
        `Required CPU: ${bytesToGiB(requiredCpuBytes)} (available: ${
          meshMemoryContext.availableCpuBytes != null
            ? bytesToGiB(meshMemoryContext.availableCpuBytes)
            : "unknown"
        }).`
      ];
      if (gpuCheckEnabled) {
        parts.push(
          `Required GPU field: ${bytesToGiB(requiredFieldGpuBytes)}.`,
          `Required GPU meshing: ${bytesToGiB(requiredMeshGpuBytes)}.`,
          `Required GPU total: ${bytesToGiB(requiredGpuBytes)} (available: ${
            meshMemoryContext.availableGpuFreeBytes != null
              ? bytesToGiB(meshMemoryContext.availableGpuFreeBytes)
              : "unknown"
          }).`
        );
      }
      parts.push("Increase unit cell size, reduce voxels per period, or switch to CPU backend.");
      fatalMessage = parts.join(" ");
    }

    return {
      resolutionXYZ,
      requiredCpuBytes,
      requiredFieldGpuBytes,
      requiredMeshGpuBytes,
      requiredGpuBytes,
      cpuFatal,
      gpuFatal,
      fatal,
      fatalMessage,
      gpuCheckEnabled
    };
  })();

  const minShellThickness = useMemo(
    () => computeMinShellThickness(meshLatticePitch, voxelsPerLatticePeriod),
    [meshLatticePitch, voxelsPerLatticePeriod]
  );
  const shellTooThin = meshShellThickness < minShellThickness;
  const minLatticeThickness = useMemo(
    () => computeMinLatticeThickness(meshLatticePitch, voxelsPerLatticePeriod),
    [meshLatticePitch, voxelsPerLatticePeriod]
  );
  const latticeTooThin = meshLatticeThickness < minLatticeThickness;

  const meshMemoryFatal = Boolean(meshMemoryRisk?.fatal);
  const meshWarnings = useMemo(() => {
    const warnings: string[] = [];
    if (shellTooThin) {
      warnings.push(
        `Shell thickness ${meshShellThickness.toFixed(2)} mm is below the minimum ${minShellThickness.toFixed(
          2
        )} mm. The lattice may protrude outside the shell surface. Increase shell thickness or reduce unit cell size.`
      );
    }
    if (latticeTooThin) {
      warnings.push(
        `Lattice half-thickness ${meshLatticeThickness.toFixed(2)} mm is below the minimum ${minLatticeThickness.toFixed(
          2
        )} mm. Strut walls may not render correctly. Increase half-thickness or reduce unit cell size.`
      );
    }
    if (meshMemoryRisk?.fatalMessage) {
      warnings.push(meshMemoryRisk.fatalMessage);
    }
    if (workflow === "mesh" && stats?.fallback_reason) {
      warnings.push(stats.fallback_reason);
    }
    return warnings;
  }, [
    shellTooThin,
    latticeTooThin,
    meshShellThickness,
    minShellThickness,
    meshLatticeThickness,
    minLatticeThickness,
    meshMemoryRisk?.fatalMessage,
    workflow,
    stats?.fallback_reason
  ]);

  const generativeStopReasonMessage = describeOptimizationStopReason(generativeStopReason);

  const abortActivePreview = useCallback(() => {
    previewControllersRef.current.field?.abort();
    previewControllersRef.current.mesh?.abort();
    previewControllersRef.current = { field: null, mesh: null };
  }, []);

  const abortActiveMeshFieldPreview = useCallback(() => {
    meshFieldPreviewControllerRef.current?.abort();
    meshFieldPreviewControllerRef.current = null;
  }, []);

  const cancelPendingMeshFieldPreview = useCallback(() => {
    abortActiveMeshFieldPreview();
  }, [abortActiveMeshFieldPreview]);

  const runMeshFieldPreview = useCallback(async () => {
    if (!meshFile) {
      setError("Upload an STL or OBJ file first.");
      return;
    }

    abortActiveMeshFieldPreview();
    setField(null);
    setStats(null);
    setUploadedFieldPreviewTrace(null);

    const runId = meshFieldPreviewRunIdRef.current + 1;
    meshFieldPreviewRunIdRef.current = runId;
    const controller = new AbortController();
    meshFieldPreviewControllerRef.current = controller;

    setIsPreviewing(true);
    setError(null);

    try {
      const response = await previewUploadedMeshField(
        meshFile,
        meshWorkflowParams,
        computeBackend,
        voxelsPerLatticePeriod,
        controller.signal
      );
      if (meshFieldPreviewRunIdRef.current !== runId) {
        return;
      }
      const fieldAssignedAtMs = performance.now();
      setField(response.field);
      setStats(response.stats);
      setUploadedMeshFieldSignature(currentUploadedMeshFieldSignature);
      setUploadedFieldPreviewTrace(
        response.trace
          ? {
              ...response.trace,
              fieldAssignedAtMs
            }
          : null
      );
      setMeshCommitted(false);
      setError(null);
    } catch (previewError) {
      if ((previewError as Error).name === "AbortError") {
        return;
      }
      if (meshFieldPreviewRunIdRef.current !== runId) {
        return;
      }
      setError((previewError as Error).message);
    } finally {
      if (meshFieldPreviewControllerRef.current === controller) {
        meshFieldPreviewControllerRef.current = null;
      }
      if (meshFieldPreviewRunIdRef.current === runId) {
        setIsPreviewing(false);
      }
    }
  }, [
    abortActiveMeshFieldPreview,
    computeBackend,
    meshFile,
    meshWorkflowParams,
    currentUploadedMeshFieldSignature,
    voxelsPerLatticePeriod
  ]);

  const compileAndSync = useCallback(
    async (
      nextSource: string
    ): Promise<{ sceneIr: SceneIR; diagnostics: CompileDiagnostics; params: Record<string, number> } | null> => {
      setIsCompiling(true);
      try {
        const compiled = await compileScene(nextSource);
        const nextParams: Record<string, number> = {};
        for (const spec of compiled.sceneIr.parameter_schema) {
          nextParams[spec.name] = params[spec.name] ?? spec.default;
        }
        setSceneIr(compiled.sceneIr);
        setDiagnostics(compiled.diagnostics);
        setParams(nextParams);
        setSourceDirty(false);
        setError(null);
        return { sceneIr: compiled.sceneIr, diagnostics: compiled.diagnostics, params: nextParams };
      } catch (compileError) {
        setError((compileError as Error).message);
        return null;
      } finally {
        setIsCompiling(false);
      }
    },
    [params]
  );

  const runDslPreview = useCallback(
    async (
      nextSceneIr: SceneIR,
      nextParams: Record<string, number>,
      nextDiagnostics: CompileDiagnostics | null
    ): Promise<void> => {
      abortActivePreview();
      const runId = previewRunIdRef.current + 1;
      previewRunIdRef.current = runId;
      setIsPreviewing(true);
      setError(null);
      setUploadedMeshFieldSignature(null);
      // Clear stale visuals before starting a new Generate Shape run.
      setField(null);
      setMesh(null);
      setStats(null);
      setUploadedFieldPreviewTrace(null);

      try {
        const gridBounds = inferPreviewBounds(nextDiagnostics);
        const resolution = resolutionForQuality(quality);
        const grid = gridBounds
          ? { bounds: gridBounds, resolution_xyz: [resolution, resolution, resolution] as [number, number, number] }
          : undefined;

        // Step 1: Evaluate the SDF field grid and display it as a ray-marched
        // volume while the mesh is being generated.
        let fieldSucceeded = false;
        let lastError: Error | null = null;

        try {
          const fieldController = new AbortController();
          previewControllersRef.current.field = fieldController;
          const fieldResponse = await previewField(
            nextSceneIr,
            nextParams,
            quality,
            computePrecision,
            computeBackend,
            fieldController.signal,
            grid
          );
          if (previewRunIdRef.current !== runId) {
            return;
          }
          fieldSucceeded = true;
          setField(fieldResponse.field);
          setStats(fieldResponse.stats);
          setUploadedFieldPreviewTrace(null);
        } catch (previewError) {
          if ((previewError as Error).name === "AbortError") {
            return;
          }
          lastError = previewError as Error;
        } finally {
          if (previewControllersRef.current.field?.signal.aborted !== true) {
            previewControllersRef.current.field = null;
          }
        }

        // Step 2: Generate the triangle mesh. Once ready it replaces the
        // field volume as the final display.
        try {
          const meshController = new AbortController();
          previewControllersRef.current.mesh = meshController;
          const meshResponse = await previewMesh(
            nextSceneIr,
            nextParams,
            quality,
            computePrecision,
            computeBackend,
            meshBackend,
            meshingMode,
            meshController.signal,
            grid
          );
          if (previewRunIdRef.current !== runId) {
            return;
          }
          // Mesh is the final display — clear field volume.
          setField(null);
          setMesh(meshResponse.mesh);
          setStats(meshResponse.stats);
          setUploadedFieldPreviewTrace(null);
          setError(null);
          return;
        } catch (previewError) {
          if ((previewError as Error).name === "AbortError") {
            return;
          }
          lastError = previewError as Error;
        } finally {
          if (previewControllersRef.current.mesh?.signal.aborted !== true) {
            previewControllersRef.current.mesh = null;
          }
        }

        if (previewRunIdRef.current !== runId) {
          return;
        }
        if (fieldSucceeded && lastError) {
          setError(lastError.message);
          return;
        }
        if (!fieldSucceeded && lastError) {
          setError(lastError.message);
        }
      } finally {
        if (previewRunIdRef.current === runId) {
          setIsPreviewing(false);
        }
      }
    },
    [abortActivePreview, quality, computePrecision, computeBackend, meshBackend, meshingMode]
  );

  useEffect(() => {
    return () => {
      abortActivePreview();
    };
  }, [abortActivePreview]);

  // Update section Y bounds whenever the active field or DSL diagnostics change.
  // For the lattice infill workflow the field payload carries the actual part bounds;
  // for the DSL workflow we fall back to the compiler-inferred bounds.
  useEffect(() => {
    let yMin: number | null = null;
    let yMax: number | null = null;

    if (field?.bounds) {
      // field.bounds = [[xmin,xmax],[ymin,ymax],[zmin,zmax]]
      yMin = field.bounds[1][0];
      yMax = field.bounds[1][1];
    } else if (diagnostics?.inferred_bounds) {
      yMin = diagnostics.inferred_bounds[1][0];
      yMax = diagnostics.inferred_bounds[1][1];
    }

    if (yMin !== null && yMax !== null && yMax > yMin) {
      // Add a small margin so the slider can reach just past the surface.
      const margin = (yMax - yMin) * 0.05;
      const newMin = yMin - margin;
      const newMax = yMax + margin;
      setSectionYBounds([newMin, newMax]);
      // Clamp the current section level to the new bounds.
      setSectionLevel((prev) => Math.min(Math.max(prev, newMin), newMax));
    }
  }, [field, diagnostics]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const draft: WorkspaceDraft = {
      version: 1,
      workflow,
      source,
      quality,
      computePrecision,
      computeBackend,
      meshBackend,
      meshingMode,
      meshShellThickness,
      meshLatticeType,
      meshLatticePitch,
      meshLatticeThickness,
      meshLatticePhase,
      voxelsPerLatticePeriod,
      wireframe,
      showGrid,
      showAxes,
      transformMode,
      sectionEnabled,
      sectionLevel
    };
    try {
      window.localStorage.setItem(WORKSPACE_DRAFT_KEY, JSON.stringify(draft));
    } catch {
      // Ignore storage failures (private mode/quota), keep UI responsive.
    }
    try {
      window.localStorage.setItem(SAVED_EXPRESSIONS_KEY, JSON.stringify(savedExpressions));
    } catch {
      // Ignore storage failures (private mode/quota), keep UI responsive.
    }
  }, [
    savedExpressions,
    workflow,
    source,
    quality,
    computePrecision,
    computeBackend,
    meshBackend,
    meshingMode,
    meshShellThickness,
    meshLatticeType,
    meshLatticePitch,
    meshLatticeThickness,
    meshLatticePhase,
    voxelsPerLatticePeriod,
    wireframe,
    showGrid,
    showAxes,
    transformMode,
    sectionEnabled,
    sectionLevel
  ]);

  useEffect(() => {
    if (!selectedExpressionName) {
      return;
    }
    const stillExists = savedExpressions.some((entry) => entry.name === selectedExpressionName);
    if (!stillExists) {
      setSelectedExpressionName("");
    }
  }, [savedExpressions, selectedExpressionName]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setIsSignatureHelpOpen(false);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  const onExportDsl = async (format: "stl" | "obj") => {
    if (!sceneIr) {
      return;
    }
    try {
      await exportMesh(
        sceneIr,
        params,
        format,
        quality === "interactive" ? "high" : quality,
        computePrecision,
        computeBackend,
        meshBackend,
        meshingMode
      );
    } catch (exportError) {
      setError((exportError as Error).message);
    }
  };

  const onCompileDsl = async () => {
    await compileAndSync(source);
  };

  const onGenerateDsl = async () => {
    let activeScene = sceneIr;
    let activeDiagnostics = diagnostics;
    let activeParams = params;

    if (!activeScene || sourceDirty) {
      const compiled = await compileAndSync(source);
      if (!compiled) {
        return;
      }
      activeScene = compiled.sceneIr;
      activeDiagnostics = compiled.diagnostics;
      activeParams = compiled.params;
    }

    if (!activeScene) {
      return;
    }

    await runDslPreview(activeScene, activeParams, activeDiagnostics);
  };

  const onGenerateMesh = async () => {
    if (!meshFile) {
      setError("Upload an STL or OBJ file first.");
      return;
    }
    if (meshMemoryFatal) {
      setError(meshMemoryRisk?.fatalMessage ?? "Estimated memory exceeds available system capacity.");
      return;
    }
    setMeshCommitted(false);
    await runMeshFieldPreview();
  };

  const onCommitMesh = async () => {
    if (!meshFile) {
      setError("Upload an STL or OBJ file first.");
      return;
    }
    if (uploadedMeshFieldSignature !== currentUploadedMeshFieldSignature) {
      setError("Generate Field first before generating the final mesh.");
      return;
    }

    cancelPendingMeshFieldPreview();
    setIsPreviewing(true);
    setError(null);
    try {
      const response = await commitUploadedMesh(
        meshFile,
        meshWorkflowParams,
        computeBackend,
        meshBackend,
        meshingMode,
        voxelsPerLatticePeriod
      );
      setMesh(response.mesh);
      setStats(response.stats);
      setUploadedFieldPreviewTrace(null);
      setMeshCommitted(true);
      setError(null);
    } catch (previewError) {
      setMeshCommitted(false);
      setError((previewError as Error).message);
    } finally {
      setIsPreviewing(false);
    }
  };

  const onExportUploaded = async (format: "stl" | "obj") => {
    if (!meshFile) {
      setError("Upload an STL or OBJ file before exporting.");
      return;
    }
    if (!meshCommitted) {
      setError("Commit the design first to compute the export-ready mesh.");
      return;
    }

    try {
      await exportUploadedMesh(
        meshFile,
        meshWorkflowParams,
        format,
        computeBackend,
        meshBackend,
        meshingMode,
        voxelsPerLatticePeriod
      );
    } catch (exportError) {
      setError((exportError as Error).message);
    }
  };

  const generativeMarkers = useMemo<ViewerMarker[]>(() => {
    const markers: ViewerMarker[] = [];
    for (const constraint of generativeConstraints) {
      for (const point of constraint.points) {
        markers.push({ position: point.point_xyz, color: "#5eead4", size: 0.05 });
      }
    }
    for (const load of generativeLoads) {
      for (const point of load.points) {
        markers.push({ position: point.point_xyz, color: "#fb7185", size: 0.05 });
      }
    }
    return markers;
  }, [generativeConstraints, generativeLoads]);

  const onPreprocessGenerative = useCallback(async () => {
    if (!generativeDesignFile || !generativeNonDesignFile) {
      setError("Upload both design-space and non-design-space meshes first.");
      return;
    }
    generativePreprocessControllerRef.current?.abort();
    const controller = new AbortController();
    generativePreprocessControllerRef.current = controller;
    setIsPreviewing(true);
    setError(null);
    try {
      const response = await preprocessStructuralOptimization(
        generativeDesignFile,
        generativeNonDesignFile,
        generativeConfig.resolution,
        computeBackend,
        meshBackend,
        controller.signal
      );
      if (generativePreprocessControllerRef.current !== controller) {
        return;
      }
      setGenerativePreprocess(response);
      setGenerativeStopReason(null);
      setMesh(response.combined_mesh);
      setField(null);
      setStats({
        eval_ms: 0,
        mesh_ms: null,
        tri_count: response.combined_mesh.face_count,
        voxel_count: response.resolution_xyz[0] * response.resolution_xyz[1] * response.resolution_xyz[2],
        compute_backend: "cpu",
        mesh_backend: "cpu",
        preview_mode: "mesh"
      });
    } catch (preprocessError) {
      if ((preprocessError as Error).name !== "AbortError") {
        setError((preprocessError as Error).message);
      }
    } finally {
      if (generativePreprocessControllerRef.current === controller) {
        generativePreprocessControllerRef.current = null;
      }
      setIsPreviewing(false);
    }
  }, [computeBackend, generativeConfig.resolution, generativeDesignFile, generativeNonDesignFile, meshBackend]);

  const onGenerativeMeshPick = useCallback(
    (point: [number, number, number]) => {
      if (generativePickMode === "fixed") {
        setGenerativeConstraints((previous) => {
          const next = [...previous];
          if (!next.length) {
            next.push({ kind: "fixed", points: [], radius: 0 });
          }
          next[0] = {
            ...next[0],
            points: [...next[0].points, { point_xyz: point }]
          };
          return next;
        });
      } else if (generativePickMode === "load") {
        setGenerativeLoads((previous) => [
          ...previous,
          {
            kind: "point",
            points: [{ point_xyz: point }],
            direction_xyz: generativeLoadDirection,
            magnitude: generativeLoadMagnitude,
            radius: 0
          }
        ]);
      }
    },
    [generativeLoadDirection, generativeLoadMagnitude, generativePickMode]
  );

  const onRunGenerative = useCallback(async () => {
    if (!generativeDesignFile || !generativeNonDesignFile) {
      setError("Upload both design-space and non-design-space meshes first.");
      return;
    }
    setIsPreviewing(true);
    setError(null);
    setGenerativeHistory([]);
    setGenerativeStopReason(null);
    setField(null);
    try {
      const request = await buildStructuralOptimizationRequest(
        {
          compute_backend: computeBackend,
          mesh_backend: meshBackend,
          execution_mode: "queued",
          constraints: generativeConstraints,
          loads: generativeLoads,
          material: generativeMaterial,
          config: generativeConfig
        },
        generativeDesignFile,
        generativeNonDesignFile
      );
      const result = await runStructuralOptimizationPhased(request, (iteration: StructuralOptimizationIterationResult) => {
        setMesh(iteration.mesh ?? null);
        setField(iteration.stress_field ?? null);
        setGenerativeHistory((previous) => [
          ...previous,
          {
            iteration: iteration.iteration,
            objective_value: iteration.objective_value,
            active_volume_fraction: iteration.active_volume_fraction,
            removed_voxels: iteration.removed_voxels,
            max_displacement: 0
          }
        ]);
      });
      setMesh(result.final_iteration.mesh ?? null);
      setField(result.final_iteration.stress_field ?? null);
      setGenerativeStopReason(result.stop_reason);
    } catch (runError) {
      setGenerativeStopReason(null);
      setError((runError as Error).message);
    } finally {
      setIsPreviewing(false);
    }
  }, [
    computeBackend,
    generativeConfig,
    generativeConstraints,
    generativeDesignFile,
    generativeLoads,
    generativeMaterial,
    generativeNonDesignFile,
    meshBackend
  ]);

  useEffect(() => {
    if (workflow !== "mesh") {
      return;
    }
    if (!meshFile || uploadedMeshFieldSignature !== currentUploadedMeshFieldSignature) {
      setMeshCommitted(false);
    }
  }, [workflow, meshFile, uploadedMeshFieldSignature, currentUploadedMeshFieldSignature]);

  useEffect(() => {
    setMesh(null);
    setStats(null);
    setMeshCommitted(false);
  }, [meshBackend]);

  useEffect(() => {
    abortActiveMeshFieldPreview();
    setMeshCommitted(false);
  }, [
    meshShellThickness,
    meshLatticeType,
    meshLatticePitch,
    meshLatticeThickness,
    meshLatticePhase,
    computeBackend,
    voxelsPerLatticePeriod,
    abortActiveMeshFieldPreview
  ]);

  const onUploadedFieldPreviewVisible = useCallback(
    async ({
      traceId,
      textureReadyAtMs: _textureReadyAtMs,
      firstVisibleFrameAtMs
    }: {
      traceId: string;
      textureReadyAtMs: number;
      firstVisibleFrameAtMs: number;
    }) => {
      if (uploadedFieldTelemetrySentRef.current.has(traceId)) {
        return;
      }
      if (!uploadedFieldPreviewTrace || uploadedFieldPreviewTrace.traceId !== traceId) {
        return;
      }
      uploadedFieldTelemetrySentRef.current.add(traceId);
      try {
        const textureAndFirstFrameMs = Math.max(0, firstVisibleFrameAtMs - uploadedFieldPreviewTrace.fieldAssignedAtMs);
        await submitUploadedFieldPreviewTelemetry({
          trace_id: traceId,
          client_response_wait_ms: uploadedFieldPreviewTrace.clientResponseWaitMs,
          client_download_ms: uploadedFieldPreviewTrace.clientDownloadMs,
          client_decode_ms: uploadedFieldPreviewTrace.clientDecodeMs,
          client_texture_upload_and_first_frame_ms: textureAndFirstFrameMs,
          client_total_visible_ms:
            uploadedFieldPreviewTrace.clientResponseWaitMs +
            uploadedFieldPreviewTrace.clientDownloadMs +
            uploadedFieldPreviewTrace.clientDecodeMs +
            textureAndFirstFrameMs
        });
      } catch (telemetryError) {
        console.warn("Uploaded field preview telemetry failed", telemetryError);
      }
    },
    [uploadedFieldPreviewTrace]
  );

  useEffect(() => {
    return () => {
      preprocessControllerRef.current?.abort();
      preprocessControllerRef.current = null;
      generativePreprocessControllerRef.current?.abort();
      generativePreprocessControllerRef.current = null;
    };
  }, []);

  useEffect(() => cancelPendingMeshFieldPreview, [cancelPendingMeshFieldPreview]);

  const onSaveExpression = () => {
    const name = expressionName.trim();
    if (!name) {
      setExpressionStatus("Enter a name before saving.");
      return;
    }

    const entry: SavedFieldExpression = {
      name,
      source,
      updatedAt: Date.now()
    };

    setSavedExpressions((previous) => {
      const withoutExisting = previous.filter((item) => item.name !== name);
      return [entry, ...withoutExisting].sort((a, b) => b.updatedAt - a.updatedAt);
    });
    setSelectedExpressionName(name);
    setExpressionStatus(`Saved "${name}".`);
    setError(null);
  };

  const onLoadExpression = () => {
    if (!selectedExpressionName) {
      setExpressionStatus("Select a saved expression to load.");
      return;
    }
    const entry = savedExpressions.find((item) => item.name === selectedExpressionName);
    if (!entry) {
      setExpressionStatus("Selected expression no longer exists.");
      return;
    }
    setSource(entry.source);
    setSourceDirty(true);
    setExpressionName(entry.name);
    setExpressionStatus(`Loaded "${entry.name}".`);
    setError(null);
  };

  const isBusy = isCompiling || isPreviewing;
  const busyLabel = isCompiling ? "Compiling..." : isPreviewing ? "Computing..." : "";

  return (
    <div className="shell">
      <header className="topbar">
        <div className="topbar-brand">
          <h1>SDF CAD Studio</h1>
        </div>
        <div className="topbar-center">
          <div className="workflow-toggle workflow-toggle-prominent" role="tablist" aria-label="Module tabs">
            {MODULE_TABS.map((tab) => (
              <button
                key={tab.mode}
                type="button"
                className={workflow === tab.mode ? "active" : ""}
                onClick={() => setWorkflow(tab.mode)}
                role="tab"
                aria-selected={workflow === tab.mode}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>
        <div className="topbar-side">
          {workflow === "dsl" ? <span className="pill">Q: {quality}</span> : null}
          {isBusy ? (
            <div className="busy-indicator" role="status" aria-live="polite" aria-label={busyLabel}>
              <span className="busy-label">{busyLabel}</span>
              <div className="busy-bar" aria-hidden="true">
                <span />
              </div>
            </div>
          ) : null}
        </div>
      </header>

      <main className="layout">
        <section className="panel editor-panel">
          {workflow === "dsl" ? (
            <>
              <div className="panel-title-row">
                <h2>DSL Editor</h2>
                <div className="inline-actions">
                  <select
                    onChange={(event) => {
                      setSource(EXAMPLES[event.target.value]);
                      setSourceDirty(true);
                    }}
                    defaultValue="Field Expression"
                  >
                    {Object.keys(EXAMPLES).map((name) => (
                      <option key={name} value={name}>
                        {name}
                      </option>
                    ))}
                  </select>
                  <button type="button" onClick={() => setIsSignatureHelpOpen(true)}>
                    Signature Help
                  </button>
                  <button onClick={() => void onCompileDsl()}>Compile now</button>
                  <button type="button" onClick={() => void onGenerateDsl()}>
                    Generate Shape
                  </button>
                  <button type="button" onClick={() => void onExportDsl("stl")}>
                    Export STL
                  </button>
                  <button type="button" onClick={() => void onExportDsl("obj")}>
                    Export OBJ
                  </button>
                </div>
              </div>

              <textarea
                value={source}
                onChange={(event) => {
                  setSource(event.target.value);
                  setSourceDirty(true);
                }}
                spellCheck={false}
                className="editor"
                aria-label="DSL source editor"
              />

              <h3>Field Expressions</h3>
              <div className="saved-expr-row">
                <input
                  type="text"
                  value={expressionName}
                  onChange={(event) => setExpressionName(event.target.value)}
                  placeholder="Expression name"
                  aria-label="Field expression name"
                />
                <button type="button" onClick={onSaveExpression}>
                  Save
                </button>
                <select
                  value={selectedExpressionName}
                  onChange={(event) => setSelectedExpressionName(event.target.value)}
                  aria-label="Saved field expressions"
                >
                  <option value="">Load saved...</option>
                  {savedExpressions.map((item) => (
                    <option key={item.name} value={item.name}>
                      {item.name}
                    </option>
                  ))}
                </select>
                <button type="button" onClick={onLoadExpression}>
                  Load
                </button>
              </div>
              <p className="muted">{expressionStatus ?? "Save and reuse field expressions from this browser."}</p>

              <h3>Parameters ({sceneIr?.parameter_schema.length ?? 0})</h3>
              <div className="params-list">
                {sceneIr?.parameter_schema.length ? (
                  sceneIr.parameter_schema.map((spec) => (
                    <label key={spec.name} className="slider-row">
                      <span>
                        {spec.name}: {params[spec.name]?.toFixed(3)}
                      </span>
                      <input
                        type="range"
                        min={spec.min}
                        max={spec.max}
                        step={spec.step}
                        value={params[spec.name] ?? spec.default}
                        onChange={(event) =>
                          setParams((previous) => ({
                            ...previous,
                            [spec.name]: Number(event.target.value)
                          }))
                        }
                      />
                      <input
                        type="number"
                        min={spec.min}
                        max={spec.max}
                        step={spec.step}
                        value={params[spec.name] ?? spec.default}
                        onChange={(event) =>
                          setParams((previous) => ({
                            ...previous,
                            [spec.name]: Number(event.target.value)
                          }))
                        }
                      />
                    </label>
                  ))
                ) : (
                  <p className="muted">Declare parameters using: param name default=.. min=.. max=.. step=..</p>
                )}
              </div>
            </>
          ) : workflow === "mesh" ? (
            <>
              <div className="mesh-workflow-grid">
                <div className="mesh-control-stack">
                  <section className="mesh-card">
                    <h3>Input & Geometry</h3>
                    <div className="mesh-detail-grid">
                      <label className="slider-row mesh-file-row">
                        <span>Input mesh (.stl/.obj)</span>
                        <input
                          type="file"
                          accept=".stl,.obj"
                          aria-label="Lattice infill file upload"
                          onChange={(event) => {
                            const selected = event.target.files?.[0] ?? null;

                            // Abort any in-flight preprocess for the previous file
                            preprocessControllerRef.current?.abort();
                            preprocessControllerRef.current = null;
                            abortActiveMeshFieldPreview();

                            setMeshFile(selected);
                            setField(null);
                            setMesh(null);
                            setStats(null);
                            setUploadedFieldPreviewTrace(null);
                            setUploadedMeshFieldSignature(null);
                            setMeshCommitted(false);
                            setMeshMemoryContext(null);
                            setError(null);

                            if (!selected) {
                              return;
                            }

                            // Upload to backend immediately:
                            //   1. Warms metadata cache for the subsequent "Generate Field"
                            //   2. Returns the raw outer mesh for immediate display in the viewer
                            const controller = new AbortController();
                            preprocessControllerRef.current = controller;
                            setIsPreviewing(true);

                            preprocessUploadedMesh(
                              selected,
                              meshWorkflowParams,
                              computeBackend,
                              voxelsPerLatticePeriod,
                              controller.signal
                            )
                              .then((preprocessResult) => {
                                if (preprocessControllerRef.current !== controller) {
                                  return; // Superseded by a newer file selection
                                }
                                setMesh(preprocessResult.mesh);
                                setMeshMemoryContext(preprocessResult.memoryContext);
                              })
                              .catch((preprocessError: Error) => {
                                if (preprocessError.name === "AbortError") {
                                  return;
                                }
                                if (preprocessControllerRef.current !== controller) {
                                  return;
                                }
                                // Preprocess failure is non-fatal — user can still click
                                // "Generate Field" which will run the full pipeline inline
                                setMesh(null);
                                setMeshMemoryContext(null);
                                setError(preprocessError.message);
                              })
                              .finally(() => {
                                if (preprocessControllerRef.current === controller) {
                                  preprocessControllerRef.current = null;
                                  setIsPreviewing(false);
                                }
                              });
                          }}
                        />
                      </label>

                      <label className="slider-row">
                        <span>Shell thickness (mm)</span>
                        <input
                          type="number"
                          step={0.5}
                          min={0.1}
                          value={meshShellThickness}
                          aria-label="Shell thickness (mm)"
                          onChange={(event) => setMeshShellThickness(Number(event.target.value))}
                        />
                      </label>

                      <label className="slider-row">
                        <span>Lattice type</span>
                        <select
                          aria-label="Lattice type"
                          value={meshLatticeType}
                          onChange={(event) => setMeshLatticeType(event.target.value as MeshLatticeType)}
                        >
                          <option value="gyroid">gyroid</option>
                          <option value="schwarz_p">schwarz_p</option>
                          <option value="diamond">diamond</option>
                        </select>
                      </label>

                      <label className="slider-row">
                        <span>Unit cell size (mm)</span>
                        <input
                          type="number"
                          step={0.5}
                          min={0.5}
                          value={meshLatticePitch}
                          aria-label="Unit cell size (mm)"
                          onChange={(event) => setMeshLatticePitch(Number(event.target.value))}
                        />
                      </label>

                      <label className="slider-row">
                        <span>Lattice half-thickness (mm)</span>
                        <input
                          type="number"
                          step={0.1}
                          min={0.05}
                          value={meshLatticeThickness}
                          aria-label="Lattice half-thickness (mm)"
                          onChange={(event) => setMeshLatticeThickness(Number(event.target.value))}
                        />
                      </label>

                      <label className="slider-row">
                        <span>Lattice phase</span>
                        <input
                          type="number"
                          step={0.05}
                          value={meshLatticePhase}
                          aria-label="Lattice phase"
                          onChange={(event) => setMeshLatticePhase(Number(event.target.value))}
                        />
                      </label>
                    </div>

                    <p className="muted">{meshFile ? `Selected: ${meshFile.name}` : "No file selected."}</p>
                    <p className="muted">
                      Min recommended shell thickness at current settings:{" "}
                      <strong>{minShellThickness.toFixed(2)} mm</strong> (2 x unit cell size / sampling quality)
                    </p>
                    <p className="muted">
                      Strut total width = 2 x half-thickness = <strong>{(meshLatticeThickness * 2).toFixed(2)} mm</strong>
                    </p>
                    <p className="muted">
                      Min resolvable half-thickness at current settings:{" "}
                      <strong>{minLatticeThickness.toFixed(2)} mm</strong> (1 voxel = unit cell size / sampling quality)
                    </p>
                  </section>

                  <section className="mesh-card">
                    <h3>Sampling & Compute</h3>
                    <div className="mesh-detail-grid">
                      <label className="slider-row">
                        <span>Sampling quality (voxels/period)</span>
                        <select
                          aria-label="Voxels per lattice period"
                          value={voxelsPerLatticePeriod}
                          onChange={(event) => setVoxelsPerLatticePeriod(Number(event.target.value))}
                        >
                          <option value={4}>4 — Draft (fast)</option>
                          <option value={6}>6 — Standard (default)</option>
                          <option value={8}>8 — Fine (slow)</option>
                        </select>
                      </label>

                      <label className="slider-row">
                        <span>Field backend</span>
                        <select
                          aria-label="Lattice infill field backend"
                          value={computeBackend}
                          onChange={(event) => setComputeBackend(event.target.value as ComputeBackend)}
                        >
                          <option value="auto">auto</option>
                          <option value="cpu">cpu</option>
                          <option value="cuda">cuda</option>
                        </select>
                      </label>

                      <label className="slider-row">
                        <span>Mesher backend</span>
                        <select
                          aria-label="Lattice infill mesher backend"
                          value={meshBackend}
                          onChange={(event) => setMeshBackend(event.target.value as MeshBackend)}
                        >
                          <option value="auto">auto</option>
                          <option value="cpu">cpu</option>
                          <option value="cuda">cuda</option>
                        </select>
                      </label>

                      <label className="slider-row">
                        <span>Meshing mode</span>
                        <select
                          aria-label="Lattice infill meshing mode"
                          value={meshingMode}
                          onChange={(event) => setMeshingMode(event.target.value as MeshingMode)}
                        >
                          <option value="uniform">uniform</option>
                          <option value="adaptive">adaptive</option>
                        </select>
                      </label>
                    </div>

                    <div className="resolution-info">
                      {meshMemoryRisk ? (
                        <>
                          <p className="muted">
                            Est. resolution for uploaded mesh:{" "}
                            <strong>{formatResolutionXYZ(meshMemoryRisk.resolutionXYZ)}</strong>
                          </p>
                          <p className="muted">
                            Est. required CPU memory: <strong>{bytesToMiB(meshMemoryRisk.requiredCpuBytes)}</strong>
                            {meshMemoryContext?.availableCpuBytes != null
                              ? ` (available: ${bytesToMiB(meshMemoryContext.availableCpuBytes)})`
                              : " (available: unknown)"}
                          </p>
                          <p className="muted">
                            Est. required GPU memory (field):{" "}
                            <strong>{bytesToMiB(meshMemoryRisk.requiredFieldGpuBytes)}</strong>
                          </p>
                          <p className="muted">
                            Est. required GPU memory (meshing):{" "}
                            <strong>{bytesToMiB(meshMemoryRisk.requiredMeshGpuBytes)}</strong>
                          </p>
                          <p className="muted">
                            Est. required GPU memory (total):{" "}
                            <strong>{bytesToMiB(meshMemoryRisk.requiredGpuBytes)}</strong>
                            {meshMemoryContext?.availableGpuFreeBytes != null
                              ? ` (available: ${bytesToMiB(meshMemoryContext.availableGpuFreeBytes)})`
                              : " (available: unknown)"}
                          </p>
                        </>
                      ) : (
                        <p className="muted">Memory estimate will appear after mesh preprocess completes.</p>
                      )}
                    </div>

                    <p className="muted">Tip: `adaptive` is currently slower on CPU; use `uniform` for fastest previews.</p>

                    <div className="mesh-actions">
                      <button
                        type="button"
                        onClick={() => void onGenerateMesh()}
                        disabled={!meshFile || meshMemoryFatal}
                      >
                        Generate Field
                      </button>
                      <button
                        type="button"
                        onClick={() => void onCommitMesh()}
                        disabled={
                          !meshFile ||
                          uploadedMeshFieldSignature !== currentUploadedMeshFieldSignature ||
                          isPreviewing
                        }
                      >
                        Generate Final Mesh
                      </button>
                      <button
                        type="button"
                        onClick={() => void onExportUploaded("stl")}
                        disabled={!meshCommitted}
                      >
                        Export STL
                      </button>
                      <button
                        type="button"
                        onClick={() => void onExportUploaded("obj")}
                        disabled={!meshCommitted}
                      >
                        Export OBJ
                      </button>
                    </div>
                  </section>
                </div>

                <aside className="warning-panel">
                  <div className="warning-panel-head">
                    <h3>Warning Display Area</h3>
                    <span className="pill">{meshWarnings.length ? `${meshWarnings.length} active` : "Clear"}</span>
                  </div>
                  {meshWarnings.length ? (
                    <div className="warning-stack">
                      {meshWarnings.map((warning, index) => (
                        <p key={`${index}-${warning}`} className="warning">
                          {warning}
                        </p>
                      ))}
                    </div>
                  ) : (
                    <p className="muted">No active warnings.</p>
                  )}
                </aside>
              </div>
            </>
          ) : (
            <>
              <div className="mesh-workflow-grid">
                <div className="mesh-control-stack">
                  <section className="mesh-card">
                    <h3>Design Domains</h3>
                    <div className="mesh-detail-grid">
                      <label className="slider-row mesh-file-row">
                        <span>Design space mesh</span>
                        <input
                          type="file"
                          accept=".stl,.obj"
                          onChange={(event) => {
                            setGenerativeDesignFile(event.target.files?.[0] ?? null);
                            setGenerativePreprocess(null);
                          }}
                        />
                      </label>
                      <label className="slider-row mesh-file-row">
                        <span>Non-design mesh</span>
                        <input
                          type="file"
                          accept=".stl,.obj"
                          onChange={(event) => {
                            setGenerativeNonDesignFile(event.target.files?.[0] ?? null);
                            setGenerativePreprocess(null);
                          }}
                        />
                      </label>
                      <label className="slider-row">
                        <span>Resolution</span>
                        <select
                          value={generativeConfig.resolution}
                          onChange={(event) =>
                            setGenerativeConfig((previous) => ({ ...previous, resolution: Number(event.target.value) }))
                          }
                        >
                          <option value={64}>64</option>
                          <option value={96}>96</option>
                          <option value={128}>128</option>
                          <option value={160}>160</option>
                        </select>
                      </label>
                    </div>
                    <div className="mesh-actions">
                      <button type="button" onClick={() => void onPreprocessGenerative()}>
                        Preprocess Domain
                      </button>
                    </div>
                    <p className="muted">{generativeDesignFile ? `Design: ${generativeDesignFile.name}` : "No design-space mesh selected."}</p>
                    <p className="muted">
                      {generativeNonDesignFile ? `Non-design: ${generativeNonDesignFile.name}` : "No non-design mesh selected."}
                    </p>
                    {generativePreprocess?.diagnostics.length ? (
                      <div className="warning-stack">
                        {generativePreprocess.diagnostics.map((warning) => (
                          <p key={warning} className="warning">
                            {warning}
                          </p>
                        ))}
                      </div>
                    ) : null}
                  </section>

                  <section className="mesh-card">
                    <h3>Supports & Loads</h3>
                    <div className="mesh-actions">
                      <button type="button" onClick={() => setGenerativePickMode("fixed")}>
                        Pick Fixed Support
                      </button>
                      <button type="button" onClick={() => setGenerativePickMode("load")}>
                        Pick Load Point
                      </button>
                      <button type="button" onClick={() => setGenerativePickMode(null)}>
                        Stop Picking
                      </button>
                      <button type="button" onClick={() => setGenerativeConstraints([])}>
                        Clear Supports
                      </button>
                      <button type="button" onClick={() => setGenerativeLoads([])}>
                        Clear Loads
                      </button>
                    </div>
                    <div className="mesh-detail-grid">
                      <label className="slider-row">
                        <span>Load X</span>
                        <input
                          type="number"
                          value={generativeLoadDirection[0]}
                          onChange={(event) =>
                            setGenerativeLoadDirection([
                              Number(event.target.value),
                              generativeLoadDirection[1],
                              generativeLoadDirection[2]
                            ])
                          }
                        />
                      </label>
                      <label className="slider-row">
                        <span>Load Y</span>
                        <input
                          type="number"
                          value={generativeLoadDirection[1]}
                          onChange={(event) =>
                            setGenerativeLoadDirection([
                              generativeLoadDirection[0],
                              Number(event.target.value),
                              generativeLoadDirection[2]
                            ])
                          }
                        />
                      </label>
                      <label className="slider-row">
                        <span>Load Z</span>
                        <input
                          type="number"
                          value={generativeLoadDirection[2]}
                          onChange={(event) =>
                            setGenerativeLoadDirection([
                              generativeLoadDirection[0],
                              generativeLoadDirection[1],
                              Number(event.target.value)
                            ])
                          }
                        />
                      </label>
                      <label className="slider-row">
                        <span>Load magnitude</span>
                        <input
                          type="number"
                          value={generativeLoadMagnitude}
                          onChange={(event) => setGenerativeLoadMagnitude(Number(event.target.value))}
                        />
                      </label>
                    </div>
                    <p className="muted">Pick mode: {generativePickMode ?? "off"}</p>
                    <p className="muted">
                      In pick mode, click directly on the mesh surface in the viewer. Cyan dots mark supports and pink dots mark load points.
                    </p>
                    <p className="muted">Supports: {generativeConstraints.reduce((sum, item) => sum + item.points.length, 0)}</p>
                    <p className="muted">Loads: {generativeLoads.length}</p>
                    {generativeConstraints.length > 0 ? (
                      <div className="warning-stack">
                        {generativeConstraints.flatMap((constraint) =>
                          constraint.points.map((point, index) => (
                            <p key={`fixed-${index}`} className="muted">
                              Support {index + 1}: {point.point_xyz.map((value) => value.toFixed(2)).join(", ")}
                            </p>
                          ))
                        )}
                        {generativeLoads.map((load, index) => (
                          <p key={`load-${index}`} className="muted">
                            Load {index + 1}: {load.points[0]?.point_xyz.map((value) => value.toFixed(2)).join(", ")} | dir{" "}
                            {load.direction_xyz.map((value) => value.toFixed(2)).join(", ")} | mag {load.magnitude.toFixed(2)}
                          </p>
                        ))}
                      </div>
                    ) : null}
                  </section>

                  <section className="mesh-card">
                    <h3>Optimization Setup</h3>
                    <div className="mesh-detail-grid">
                      <label className="slider-row">
                        <span>Target volume fraction</span>
                        <input
                          type="number"
                          step={0.05}
                          min={0.05}
                          max={0.95}
                          value={generativeConfig.target_volume_fraction}
                          onChange={(event) =>
                            setGenerativeConfig((previous) => ({
                              ...previous,
                              target_volume_fraction: Number(event.target.value)
                            }))
                          }
                        />
                      </label>
                      <label className="slider-row">
                        <span>Max iterations</span>
                        <input
                          type="number"
                          min={1}
                          max={64}
                          value={generativeConfig.max_iterations}
                          onChange={(event) =>
                            setGenerativeConfig((previous) => ({ ...previous, max_iterations: Number(event.target.value) }))
                          }
                        />
                      </label>
                      <label className="slider-row">
                        <span>CG tolerance</span>
                        <input
                          type="number"
                          step={1e-6}
                          min={1e-8}
                          max={1e-2}
                          value={generativeConfig.cg_tolerance}
                          onChange={(event) =>
                            setGenerativeConfig((previous) => ({ ...previous, cg_tolerance: Number(event.target.value) }))
                          }
                        />
                      </label>
                      <label className="slider-row">
                        <span>Optimization tolerance</span>
                        <input
                          type="number"
                          step={1e-4}
                          min={1e-5}
                          max={0.1}
                          value={generativeConfig.optimization_tolerance}
                          onChange={(event) =>
                            setGenerativeConfig((previous) => ({
                              ...previous,
                              optimization_tolerance: Number(event.target.value)
                            }))
                          }
                        />
                      </label>
                      <label className="slider-row">
                        <span>Min density</span>
                        <input
                          type="number"
                          step={0.001}
                          min={0.001}
                          max={0.5}
                          value={generativeConfig.min_density}
                          onChange={(event) =>
                            setGenerativeConfig((previous) => ({ ...previous, min_density: Number(event.target.value) }))
                          }
                        />
                      </label>
                      <label className="slider-row">
                        <span>OC move limit</span>
                        <input
                          type="number"
                          step={0.01}
                          min={0.01}
                          max={1}
                          value={generativeConfig.oc_move_limit}
                          onChange={(event) =>
                            setGenerativeConfig((previous) => ({ ...previous, oc_move_limit: Number(event.target.value) }))
                          }
                        />
                      </label>
                      <label className="slider-row">
                        <span>Density iso threshold</span>
                        <input
                          type="number"
                          step={0.01}
                          min={0.05}
                          max={0.95}
                          value={generativeConfig.density_iso_threshold}
                          onChange={(event) =>
                            setGenerativeConfig((previous) => ({
                              ...previous,
                              density_iso_threshold: Number(event.target.value)
                            }))
                          }
                        />
                      </label>
                      <label className="slider-row">
                        <span>Young's modulus</span>
                        <input
                          type="number"
                          step={0.1}
                          min={0.1}
                          value={generativeMaterial.youngs_modulus}
                          onChange={(event) =>
                            setGenerativeMaterial((previous) => ({ ...previous, youngs_modulus: Number(event.target.value) }))
                          }
                        />
                      </label>
                      <label className="slider-row">
                        <span>Poisson ratio</span>
                        <input
                          type="number"
                          step={0.01}
                          min={0}
                          max={0.49}
                          value={generativeMaterial.poissons_ratio}
                          onChange={(event) =>
                            setGenerativeMaterial((previous) => ({ ...previous, poissons_ratio: Number(event.target.value) }))
                          }
                        />
                      </label>
                    </div>
                    <div className="mesh-actions">
                      <button type="button" onClick={() => void onRunGenerative()}>
                        Run Optimization
                      </button>
                    </div>
                    <p className="muted">The first pass supports point-based supports and loads picked directly in the viewer.</p>
                  </section>
                </div>

                <aside className="warning-panel">
                  <div className="warning-panel-head">
                    <h3>Optimization History</h3>
                    <span className="pill">{generativeHistory.length} iters</span>
                  </div>
                  {generativeHistory.length ? (
                    <div className="warning-stack">
                      {generativeStopReasonMessage ? (
                        <p className="muted">{generativeStopReasonMessage}</p>
                      ) : null}
                      {generativeHistory.map((entry) => (
                        <p key={entry.iteration} className="muted">
                          Iter {entry.iteration}: obj {entry.objective_value.toFixed(3)}, volume{" "}
                          {(entry.active_volume_fraction * 100).toFixed(1)}%, removed {entry.removed_voxels}
                        </p>
                      ))}
                    </div>
                  ) : (
                    <p className="muted">Run preprocessing and then start optimization to load queued iteration progress here.</p>
                  )}
                </aside>
              </div>
            </>
          )}
        </section>

        <section className="panel viewer-panel">
          <div className="toolbar">
            <button onClick={() => setTransformMode("translate")}>Move</button>
            <button onClick={() => setTransformMode("rotate")}>Rotate</button>
            <button onClick={() => setTransformMode("scale")}>Scale</button>
            <button onClick={() => setWireframe((value) => !value)}>{wireframe ? "Solid" : "Wire"}</button>
            <button onClick={() => setShowGrid((value) => !value)}>{showGrid ? "Hide Grid" : "Grid"}</button>
            <button onClick={() => setShowAxes((value) => !value)}>{showAxes ? "Hide Axes" : "Axes"}</button>
            <button onClick={() => setFitSignal((value) => value + 1)}>Fit View</button>
            <button onClick={() => setSectionEnabled((value) => !value)}>
              {sectionEnabled ? "Section Off" : "Section On"}
            </button>
            {workflow === "dsl" ? (
              <label className="inline-label">
                Quality
                <select value={quality} onChange={(event) => setQuality(event.target.value as QualityProfile)}>
                  {QUALITY_ORDER.map((item) => (
                    <option key={item} value={item}>
                      {item}
                    </option>
                  ))}
                </select>
              </label>
            ) : null}
            {workflow === "dsl" ? (
              <label className="inline-label">
                SDF Precision
                <select
                  value={computePrecision}
                  onChange={(event) => setComputePrecision(event.target.value as ComputePrecision)}
                >
                  <option value="float32">Float32 (default)</option>
                  <option value="float16">Float16 (explicit)</option>
                </select>
              </label>
            ) : null}
            {workflow === "dsl" ? (
              <label className="inline-label">
                Eval Backend
                <select value={computeBackend} onChange={(event) => setComputeBackend(event.target.value as ComputeBackend)}>
                  <option value="auto">Auto</option>
                  <option value="cpu">CPU</option>
                  <option value="cuda">CUDA</option>
                </select>
              </label>
            ) : null}
            {workflow === "dsl" ? (
              <label className="inline-label">
                Mesh Backend
                <select value={meshBackend} onChange={(event) => setMeshBackend(event.target.value as MeshBackend)}>
                  <option value="auto">Auto</option>
                  <option value="cpu">CPU</option>
                  <option value="cuda">CUDA</option>
                </select>
              </label>
            ) : null}
            {workflow === "dsl" ? (
              <label className="inline-label">
                Meshing Mode
                <select value={meshingMode} onChange={(event) => setMeshingMode(event.target.value as MeshingMode)}>
                  <option value="uniform">Uniform</option>
                  <option value="adaptive">Adaptive</option>
                </select>
              </label>
            ) : null}
          </div>

          {sectionEnabled ? (
            <div className="section-row">
              <span>Section Y: {sectionLevel.toFixed(2)}</span>
              <input
                type="range"
                min={sectionYBounds[0]}
                max={sectionYBounds[1]}
                step={(sectionYBounds[1] - sectionYBounds[0]) / 200}
                value={sectionLevel}
                onChange={(event) => setSectionLevel(Number(event.target.value))}
              />
            </div>
          ) : null}

          <div className="viewer-wrap">
            <Viewer
              mesh={mesh}
              field={field}
              pickMarkers={workflow === "generative" ? generativeMarkers : []}
              onMeshPick={workflow === "generative" ? onGenerativeMeshPick : null}
              pickModeActive={workflow === "generative" && generativePickMode !== null}
              uploadedMeshPreviewActive={workflow === "mesh" && !meshCommitted && field != null}
              uploadedFieldPreviewTrace={workflow === "mesh" ? uploadedFieldPreviewTrace : null}
              onUploadedFieldPreviewVisible={onUploadedFieldPreviewVisible}
              wireframe={wireframe}
              transformMode={transformMode}
              fitSignal={fitSignal}
              showAxes={showAxes}
              showGrid={showGrid}
              sectionEnabled={sectionEnabled}
              sectionLevel={sectionLevel}
            />
          </div>

          <div className="stats-row">
            <span>Mode: {stats?.preview_mode ?? "mesh"}</span>
            <span>Triangles: {stats?.tri_count ?? 0}</span>
            <span>Voxels: {stats?.voxel_count ?? 0}</span>
            <span>Eval: {stats ? stats.eval_ms.toFixed(1) : "0.0"} ms</span>
            <span>Mesh: {stats?.mesh_ms != null ? stats.mesh_ms.toFixed(1) : "n/a"} ms</span>
            <span>Eval backend: {stats?.compute_backend ?? "cpu"}</span>
            <span>Mesh backend: {stats?.mesh_backend ?? "cpu"}</span>
            <span>Field cache: {stats?.field_cache_hit ? "hit" : "miss"}</span>
            <span>Mesh cache: {stats?.mesh_cache_hit ? "hit" : "miss"}</span>
          </div>
          {workflow === "dsl" && stats?.fallback_reason ? <p className="warning">{stats.fallback_reason}</p> : null}
          {error ? <p className="error">{error}</p> : null}
        </section>
      </main>

      {isSignatureHelpOpen && workflow === "dsl" ? (
        <div className="modal-backdrop" onClick={() => setIsSignatureHelpOpen(false)}>
          <section
            className="modal-card"
            role="dialog"
            aria-modal="true"
            aria-labelledby="signature-help-title"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="modal-head">
              <h3 id="signature-help-title">Signature Help</h3>
              <button type="button" onClick={() => setIsSignatureHelpOpen(false)}>
                Close
              </button>
            </div>
            <pre className="composition modal-signatures">{FUNCTION_SIGNATURES.join("\n")}</pre>
          </section>
        </div>
      ) : null}
    </div>
  );
}
