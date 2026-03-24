import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { Viewer } from "./components/Viewer";
import {
  compileScene,
  exportMesh,
  exportUploadedMesh,
  previewField,
  previewMesh,
  preprocessUploadedMesh,
  previewUploadedMesh,
  previewUploadedMeshField,
  submitUploadedFieldPreviewTelemetry
} from "./lib/api";
import {
  CompileDiagnostics,
  ComputeBackend,
  ComputePrecision,
  FieldPayload,
  MeshBackend,
  MeshLatticeType,
  MeshingMode,
  MeshPayload,
  MeshWorkflowParams,
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
const SAVED_EXPRESSIONS_KEY = "sdfcad.savedFieldExpressions.v1";
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

type WorkflowMode = "dsl" | "mesh";

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

function computeRequiredResolution(meshSpan: number, latticePitch: number, voxelsPerPeriod: number): number {
  return Math.min(1024, Math.max(24, Math.ceil((meshSpan * 1.24 / latticePitch) * voxelsPerPeriod)));
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
  resolution: number,
  bytesPerVoxel: number,
  safetyFactor: number
): number {
  return Math.ceil((resolution ** 3) * bytesPerVoxel * safetyFactor);
}

function bytesToGiB(value: number): string {
  return `${(value / (1024 ** 3)).toFixed(2)} GiB`;
}

function bytesToMiB(value: number): string {
  return `${Math.round(value / (1024 * 1024))} MiB`;
}

export default function App() {
  const [workflow, setWorkflow] = useState<WorkflowMode>("dsl");

  const [source, setSource] = useState(EXAMPLES["Field Expression"]);
  const [sourceDirty, setSourceDirty] = useState(true);
  const [sceneIr, setSceneIr] = useState<SceneIR | null>(null);
  const [diagnostics, setDiagnostics] = useState<CompileDiagnostics | null>(null);
  const [params, setParams] = useState<Record<string, number>>({});

  const [meshFile, setMeshFile] = useState<File | null>(null);
  const [meshShellThickness, setMeshShellThickness] = useState(2.0);
  const [meshLatticeType, setMeshLatticeType] = useState<MeshLatticeType>("gyroid");
  const [meshLatticePitch, setMeshLatticePitch] = useState(5.0);
  const [meshLatticeThickness, setMeshLatticeThickness] = useState(0.5);
  const [meshLatticePhase, setMeshLatticePhase] = useState(0.0);
  const [voxelsPerLatticePeriod, setVoxelsPerLatticePeriod] = useState(6);
  const [meshCommitted, setMeshCommitted] = useState(false);
  const [meshMemoryContext, setMeshMemoryContext] = useState<UploadedMeshMemoryContext | null>(null);

  const [field, setField] = useState<FieldPayload | null>(null);
  const [mesh, setMesh] = useState<MeshPayload | null>(null);
  const [stats, setStats] = useState<PreviewStats | null>(null);
  const [uploadedFieldPreviewTrace, setUploadedFieldPreviewTrace] = useState<UploadedFieldPreviewTrace | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isCompiling, setIsCompiling] = useState(false);
  const [isPreviewing, setIsPreviewing] = useState(false);
  const [wireframe, setWireframe] = useState(false);
  const [showGrid, setShowGrid] = useState(true);
  const [showAxes, setShowAxes] = useState(true);
  const [transformMode, setTransformMode] = useState<"translate" | "rotate" | "scale">("translate");
  const [fitSignal, setFitSignal] = useState(0);
  const [quality, setQuality] = useState<QualityProfile>("high");
  const [computePrecision, setComputePrecision] = useState<ComputePrecision>("float32");
  const [computeBackend, setComputeBackend] = useState<ComputeBackend>("auto");
  const [meshBackend, setMeshBackend] = useState<MeshBackend>("auto");
  const [meshingMode, setMeshingMode] = useState<MeshingMode>("uniform");
  const [sectionEnabled, setSectionEnabled] = useState(false);
  const [sectionLevel, setSectionLevel] = useState(0);
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

  const meshMemoryRisk = useMemo(() => {
    if (!meshMemoryContext) {
      return null;
    }

    const resolution = computeRequiredResolution(
      meshMemoryContext.meshSpan,
      meshLatticePitch,
      voxelsPerLatticePeriod
    );
    const requiredCpuBytes = estimateRequiredBytes(
      resolution,
      meshMemoryContext.cpuBytesPerVoxel,
      meshMemoryContext.safetyFactor
    );
    const requiredGpuBytes = estimateRequiredBytes(
      resolution,
      meshMemoryContext.gpuBytesPerVoxel,
      meshMemoryContext.safetyFactor
    );
    const cpuFatal =
      meshMemoryContext.availableCpuBytes != null && requiredCpuBytes > meshMemoryContext.availableCpuBytes;
    const gpuCheckEnabled =
      computeBackend === "cuda" || (computeBackend === "auto" && meshMemoryContext.availableGpuFreeBytes != null);
    const gpuFatal =
      gpuCheckEnabled &&
      meshMemoryContext.availableGpuFreeBytes != null &&
      requiredGpuBytes > meshMemoryContext.availableGpuFreeBytes;
    const fatal = cpuFatal || gpuFatal;

    let fatalMessage: string | null = null;
    if (fatal) {
      const parts = [
        `Estimated memory exceeds available capacity for resolution ${resolution}.`,
        `Required CPU: ${bytesToGiB(requiredCpuBytes)} (available: ${
          meshMemoryContext.availableCpuBytes != null
            ? bytesToGiB(meshMemoryContext.availableCpuBytes)
            : "unknown"
        }).`
      ];
      if (gpuCheckEnabled) {
        parts.push(
          `Required GPU: ${bytesToGiB(requiredGpuBytes)} (available: ${
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
      resolution,
      requiredCpuBytes,
      requiredGpuBytes,
      cpuFatal,
      gpuFatal,
      fatal,
      fatalMessage,
      gpuCheckEnabled
    };
  }, [meshMemoryContext, meshLatticePitch, voxelsPerLatticePeriod, computeBackend]);

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
      setMesh(null);
      setStats(response.stats);
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
      // Clear stale visuals before starting a new Generate Shape run.
      setField(null);
      setMesh(null);
      setStats(null);
      setUploadedFieldPreviewTrace(null);

      try {
        const gridBounds = inferPreviewBounds(nextDiagnostics);
        const resolution = resolutionForQuality(quality);
        const grid = gridBounds ? { bounds: gridBounds, resolution } : undefined;

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
  // For the Mesh Workflow the field payload carries the actual part bounds;
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
    try {
      window.localStorage.setItem(SAVED_EXPRESSIONS_KEY, JSON.stringify(savedExpressions));
    } catch {
      // Ignore storage failures (private mode/quota), keep UI responsive.
    }
  }, [savedExpressions]);

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
    setField(null);
    setMesh(null);
    setStats(null);
    setUploadedFieldPreviewTrace(null);
    setMeshCommitted(false);
    await runMeshFieldPreview();
  };

  const onCommitMesh = async () => {
    if (!meshFile) {
      setError("Upload an STL or OBJ file first.");
      return;
    }
    if (meshMemoryFatal) {
      setError(meshMemoryRisk?.fatalMessage ?? "Estimated memory exceeds available system capacity.");
      return;
    }

    cancelPendingMeshFieldPreview();
    setIsPreviewing(true);
    setError(null);
    try {
      const response = await previewUploadedMesh(
        meshFile,
        meshWorkflowParams,
        computeBackend,
        meshBackend,
        meshingMode,
        voxelsPerLatticePeriod
      );
      setField(response.field ?? null);
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

  useEffect(() => {
    if (workflow !== "mesh") {
      return;
    }
    setMeshCommitted(false);
    setUploadedFieldPreviewTrace(null);
  }, [
    workflow,
    meshFile,
    meshShellThickness,
    meshLatticeType,
    meshLatticePitch,
    meshLatticeThickness,
    meshLatticePhase,
    computeBackend,
    meshBackend,
    meshingMode,
    voxelsPerLatticePeriod
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

  return (
    <div className="shell">
      <header className="topbar">
        <h1>SDF CAD Studio</h1>
        <div className="status-pills">
          <span className={workflow === "dsl" && isCompiling ? "pill active" : "pill"}>
            {workflow === "dsl" ? (isCompiling ? "Compiling" : "Compiled") : "Mesh Workflow"}
          </span>
          <span className={isPreviewing ? "pill active" : "pill"}>{isPreviewing ? "Previewing" : "Preview Ready"}</span>
          {workflow === "dsl" ? <span className="pill">Q: {quality}</span> : null}
        </div>
      </header>

      <main className="layout">
        <section className="panel editor-panel">
          <div className="workflow-toggle" role="tablist" aria-label="Workflow mode">
            <button
              type="button"
              className={workflow === "dsl" ? "active" : ""}
              onClick={() => setWorkflow("dsl")}
              role="tab"
              aria-selected={workflow === "dsl"}
            >
              DSL
            </button>
            <button
              type="button"
              className={workflow === "mesh" ? "active" : ""}
              onClick={() => setWorkflow("mesh")}
              role="tab"
              aria-selected={workflow === "mesh"}
            >
              Mesh
            </button>
          </div>

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
          ) : (
            <>
              <div className="panel-title-row">
                <h2>Mesh Workflow</h2>
              </div>

              <div className="mesh-controls">
                <label className="slider-row">
                  <span>Input mesh (.stl/.obj)</span>
                  <input
                    type="file"
                    accept=".stl,.obj"
                    aria-label="Mesh file upload"
                    onChange={(event) => {
                      const selected = event.target.files?.[0] ?? null;

                      // Abort any in-flight preprocess for the previous file
                      preprocessControllerRef.current?.abort();
                      preprocessControllerRef.current = null;

                      setMeshFile(selected);
                      setField(null);
                      setMesh(null);
                      setStats(null);
                      setUploadedFieldPreviewTrace(null);
                      setMeshCommitted(false);
                      setMeshMemoryContext(null);
                      setError(null);

                      if (!selected) {
                        return;
                      }

                      // Upload to backend immediately:
                      //   1. Warms metadata cache for the subsequent "Preview Field"
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
                          // "Preview Field" which will run the full pipeline inline
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
                <p className="muted">{meshFile ? `Selected: ${meshFile.name}` : "No file selected."}</p>

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
                <p className="muted">
                  Min recommended shell thickness at current settings:{" "}
                  <strong>{minShellThickness.toFixed(2)} mm</strong> (2 x unit cell size / sampling quality)
                </p>
                {shellTooThin ? (
                  <p className="warning">
                    Shell thickness {meshShellThickness.toFixed(2)} mm is below the minimum{" "}
                    {minShellThickness.toFixed(2)} mm. The lattice may protrude outside the shell surface.
                    Increase shell thickness or reduce unit cell size.
                  </p>
                ) : null}

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
                <p className="muted">
                  Strut total width = 2 x half-thickness = <strong>{(meshLatticeThickness * 2).toFixed(2)} mm</strong>
                </p>
                <p className="muted">
                  Min resolvable half-thickness at current settings:{" "}
                  <strong>{minLatticeThickness.toFixed(2)} mm</strong> (1 voxel = unit cell size / sampling quality)
                </p>
                {latticeTooThin ? (
                  <p className="warning">
                    Lattice half-thickness {meshLatticeThickness.toFixed(2)} mm is below the minimum{" "}
                    {minLatticeThickness.toFixed(2)} mm. Strut walls may not render correctly.
                    Increase half-thickness or reduce unit cell size.
                  </p>
                ) : null}

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

                <div className="resolution-info">
                  {meshMemoryRisk ? (
                    <>
                      <p className="muted">
                        Est. resolution for uploaded mesh: <strong>{meshMemoryRisk.resolution}³</strong>
                        {meshMemoryRisk.resolution >= 1024 ? " (clamped to 1024³ max)" : ""}
                      </p>
                      <p className="muted">
                        Est. required CPU memory: <strong>{bytesToMiB(meshMemoryRisk.requiredCpuBytes)}</strong>
                        {meshMemoryContext?.availableCpuBytes != null
                          ? ` (available: ${bytesToMiB(meshMemoryContext.availableCpuBytes)})`
                          : " (available: unknown)"}
                      </p>
                      <p className="muted">
                        Est. required GPU memory: <strong>{bytesToMiB(meshMemoryRisk.requiredGpuBytes)}</strong>
                        {meshMemoryContext?.availableGpuFreeBytes != null
                          ? ` (available: ${bytesToMiB(meshMemoryContext.availableGpuFreeBytes)})`
                          : " (available: unknown)"}
                      </p>
                    </>
                  ) : (
                    <p className="muted">Memory estimate will appear after mesh preprocess completes.</p>
                  )}
                </div>

                <label className="slider-row">
                  <span>Field backend</span>
                  <select
                    aria-label="Mesh field backend"
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
                    aria-label="Mesh mesher backend"
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
                    aria-label="Mesh meshing mode"
                    value={meshingMode}
                    onChange={(event) => setMeshingMode(event.target.value as MeshingMode)}
                  >
                    <option value="uniform">uniform</option>
                    <option value="adaptive">adaptive</option>
                  </select>
                </label>
                <p className="muted">Tip: `adaptive` is currently slower on CPU; use `uniform` for fastest previews.</p>

                {meshMemoryRisk?.fatalMessage ? <p className="warning">{meshMemoryRisk.fatalMessage}</p> : null}

                <button
                  type="button"
                  onClick={() => void onGenerateMesh()}
                  disabled={!meshFile || meshMemoryFatal}
                >
                  Preview Field
                </button>
                <button
                  type="button"
                  onClick={() => void onCommitMesh()}
                  disabled={!meshFile || meshMemoryFatal}
                >
                  Commit Design & Compute Mesh
                </button>
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
            <button
              onClick={() => void (workflow === "dsl" ? onExportDsl("stl") : onExportUploaded("stl"))}
              disabled={workflow === "mesh" && !meshCommitted}
            >
              Export STL
            </button>
            <button
              onClick={() => void (workflow === "dsl" ? onExportDsl("obj") : onExportUploaded("obj"))}
              disabled={workflow === "mesh" && !meshCommitted}
            >
              Export OBJ
            </button>
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
              uploadedFieldPreviewTrace={field && !mesh ? uploadedFieldPreviewTrace : null}
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
          {stats?.fallback_reason ? <p className="warning">{stats.fallback_reason}</p> : null}
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
