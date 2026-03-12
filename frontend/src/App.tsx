import { useCallback, useEffect, useMemo, useState } from "react";

import { Viewer } from "./components/Viewer";
import {
  compileScene,
  exportMesh,
  exportUploadedMesh,
  previewField,
  previewMesh,
  previewUploadedMesh
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
  SceneIR
} from "./types";

const EXAMPLES: Record<string, string> = {
  "Gyroid Fill": `# Gyroid conformal fill
param shell default=0.08 min=0.03 max=0.2 step=0.01
host = sphere(r=1.0)
lat = gyroid(pitch=0.45, thickness=0.1)
root = conformal_fill(host, lat, wall=$shell, mode="hybrid")
`,
  "Compressor Stage": `# Centrifugal compressor with volute
param twist default=0.9 min=0.1 max=1.4 step=0.05
imp = impeller_centrifugal(r_in=0.22, r_out=0.95, hub_h=0.45, blade_count=9, blade_twist=$twist)
vol = volute_casing(throat_radius=0.34, outlet_radius=1.25, width=0.48, wall=0.08)
root = union(imp, vol)
`,
  "Field Expression": `# Pure implicit expression
a = sin(x * 3.0) + cos(y * 3.0) + sin(z * 3.0)
root = abs(a) - 0.45
`
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

type WorkflowMode = "dsl" | "mesh";

interface SavedFieldExpression {
  name: string;
  source: string;
  updatedAt: number;
}

function loadSavedExpressions(): SavedFieldExpression[] {
  if (typeof window === "undefined") {
    return [];
  }
  try {
    const raw = window.localStorage.getItem(SAVED_EXPRESSIONS_KEY);
    if (!raw) {
      return [];
    }
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) {
      return [];
    }
    return parsed
      .map((item) => ({
        name: typeof item?.name === "string" ? item.name : "",
        source: typeof item?.source === "string" ? item.source : "",
        updatedAt: typeof item?.updatedAt === "number" ? item.updatedAt : 0
      }))
      .filter((item) => item.name.length > 0 && item.source.length > 0)
      .sort((a, b) => b.updatedAt - a.updatedAt);
  } catch {
    return [];
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

export default function App() {
  const [workflow, setWorkflow] = useState<WorkflowMode>("dsl");

  const [source, setSource] = useState(EXAMPLES["Gyroid Fill"]);
  const [sceneIr, setSceneIr] = useState<SceneIR | null>(null);
  const [diagnostics, setDiagnostics] = useState<CompileDiagnostics | null>(null);
  const [params, setParams] = useState<Record<string, number>>({});
  const [previewParams, setPreviewParams] = useState<Record<string, number>>({});
  const [previewRevision, setPreviewRevision] = useState(0);

  const [meshFile, setMeshFile] = useState<File | null>(null);
  const [meshShellThickness, setMeshShellThickness] = useState(0.08);
  const [meshLatticeType, setMeshLatticeType] = useState<MeshLatticeType>("gyroid");
  const [meshLatticePitch, setMeshLatticePitch] = useState(0.45);
  const [meshLatticeThickness, setMeshLatticeThickness] = useState(0.09);
  const [meshLatticePhase, setMeshLatticePhase] = useState(0.0);
  const [meshPreviewQuality, setMeshPreviewQuality] = useState<QualityProfile>("medium");
  const [meshExportQuality, setMeshExportQuality] = useState<QualityProfile>("high");

  const [field, setField] = useState<FieldPayload | null>(null);
  const [mesh, setMesh] = useState<MeshPayload | null>(null);
  const [stats, setStats] = useState<PreviewStats | null>(null);
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
  const [isSignatureHelpOpen, setIsSignatureHelpOpen] = useState(false);
  const [savedExpressions, setSavedExpressions] = useState<SavedFieldExpression[]>(() => loadSavedExpressions());
  const [expressionName, setExpressionName] = useState("");
  const [selectedExpressionName, setSelectedExpressionName] = useState("");
  const [expressionStatus, setExpressionStatus] = useState<string | null>(null);

  const previewBounds = useMemo(() => inferPreviewBounds(diagnostics), [diagnostics]);

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

  const compileAndSync = useCallback(
    async (nextSource: string) => {
      setIsCompiling(true);
      try {
        const compiled = await compileScene(nextSource);
        setSceneIr(compiled.sceneIr);
        setDiagnostics(compiled.diagnostics);
        setParams((previous) => {
          const nextParams: Record<string, number> = {};
          for (const spec of compiled.sceneIr.parameter_schema) {
            nextParams[spec.name] = previous[spec.name] ?? spec.default;
          }
          setPreviewParams(nextParams);
          setPreviewRevision((value) => value + 1);
          return nextParams;
        });
        setError(null);
      } catch (compileError) {
        setError((compileError as Error).message);
      } finally {
        setIsCompiling(false);
      }
    },
    [setSceneIr]
  );

  useEffect(() => {
    if (workflow !== "dsl") {
      return;
    }
    const timer = window.setTimeout(() => {
      void compileAndSync(source);
    }, 420);
    return () => window.clearTimeout(timer);
  }, [source, compileAndSync, workflow]);

  useEffect(() => {
    if (workflow !== "dsl" || !sceneIr || previewRevision === 0) {
      return;
    }

    const controllers: AbortController[] = [];
    let active = true;
    let fineTimer: number | undefined;

    const doPreview = async () => {
      setIsPreviewing(true);
      setError(null);

      const runStep = async (profile: QualityProfile) => {
        const controller = new AbortController();
        controllers.push(controller);
        const resolution =
          profile === "interactive" ? 64 : profile === "medium" ? 128 : profile === "high" ? 192 : 256;
        const grid = previewBounds ? { bounds: previewBounds, resolution } : undefined;
        try {
          const fieldResponse = await previewField(
            sceneIr,
            previewParams,
            profile,
            computePrecision,
            computeBackend,
            controller.signal,
            grid
          );
          const meshController = new AbortController();
          controllers.push(meshController);
          void previewMesh(
            sceneIr,
            previewParams,
            profile,
            computePrecision,
            computeBackend,
            meshBackend,
            meshingMode,
            meshController.signal,
            grid
          )
            .then((meshResponse) => {
              if (!active) {
                return;
              }
              setMesh(meshResponse.mesh);
            })
            .catch(() => {
              // Keep field preview active even if backup mesh fetch fails.
            });
          return { kind: "field" as const, response: fieldResponse };
        } catch (error) {
          if ((error as Error).name === "AbortError") {
            throw error;
          }
          const meshResponse = await previewMesh(
            sceneIr,
            previewParams,
            profile,
            computePrecision,
            computeBackend,
            meshBackend,
            meshingMode,
            controller.signal,
            grid
          );
          return { kind: "mesh" as const, response: meshResponse };
        }
      };

      try {
        const coarse = await runStep("interactive");
        if (!active) {
          return;
        }
        if (coarse.kind === "field") {
          setField(coarse.response.field);
          setStats(coarse.response.stats);
        } else {
          setField(null);
          setMesh(coarse.response.mesh);
          setStats(coarse.response.stats);
        }

        if (quality !== "interactive") {
          fineTimer = window.setTimeout(async () => {
            try {
              const fine = await runStep(quality);
              if (!active) {
                return;
              }
              if (fine.kind === "field") {
                setField(fine.response.field);
                setStats(fine.response.stats);
              } else {
                setField(null);
                setMesh(fine.response.mesh);
                setStats(fine.response.stats);
              }
              setIsPreviewing(false);
            } catch (err) {
              if (!active) {
                return;
              }
              if ((err as Error).name !== "AbortError") {
                setError((err as Error).message);
              }
              setIsPreviewing(false);
            }
          }, 550);
        } else {
          setIsPreviewing(false);
        }
      } catch (err) {
        if (!active) {
          return;
        }
        if ((err as Error).name !== "AbortError") {
          setError((err as Error).message);
        }
        setIsPreviewing(false);
      }
    };

    void doPreview();

    return () => {
      active = false;
      if (fineTimer !== undefined) {
        window.clearTimeout(fineTimer);
      }
      for (const controller of controllers) {
        controller.abort();
      }
    };
  }, [
    workflow,
    sceneIr,
    previewRevision,
    quality,
    computePrecision,
    computeBackend,
    meshBackend,
    meshingMode,
    previewBounds,
    previewParams
  ]);

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
      const blob = await exportMesh(
        sceneIr,
        params,
        format,
        quality === "interactive" ? "high" : quality,
        computePrecision,
        computeBackend,
        meshBackend,
        meshingMode
      );
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = `sdf-model.${format}`;
      link.click();
      URL.revokeObjectURL(link.href);
    } catch (exportError) {
      setError((exportError as Error).message);
    }
  };

  const onGenerateDsl = () => {
    if (!sceneIr) {
      return;
    }
    setPreviewParams(params);
    setPreviewRevision((value) => value + 1);
  };

  const onGenerateMesh = async () => {
    if (!meshFile) {
      setError("Upload an STL or OBJ file first.");
      return;
    }

    setIsPreviewing(true);
    setError(null);
    try {
      const response = await previewUploadedMesh(
        meshFile,
        meshWorkflowParams,
        meshPreviewQuality,
        computeBackend,
        meshBackend,
        meshingMode
      );
      setField(null);
      setMesh(response.mesh);
      setStats(response.stats);
    } catch (previewError) {
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

    try {
      const blob = await exportUploadedMesh(
        meshFile,
        meshWorkflowParams,
        format,
        meshExportQuality,
        computeBackend,
        meshBackend,
        meshingMode
      );
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = `mesh-lattice.${format}`;
      link.click();
      URL.revokeObjectURL(link.href);
    } catch (exportError) {
      setError((exportError as Error).message);
    }
  };

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
    setExpressionName(entry.name);
    setExpressionStatus(`Loaded "${entry.name}".`);
    setError(null);
  };

  const statusQuality = workflow === "dsl" ? quality : meshPreviewQuality;

  return (
    <div className="shell">
      <header className="topbar">
        <h1>SDF CAD Studio</h1>
        <div className="status-pills">
          <span className={workflow === "dsl" && isCompiling ? "pill active" : "pill"}>
            {workflow === "dsl" ? (isCompiling ? "Compiling" : "Compiled") : "Mesh Workflow"}
          </span>
          <span className={isPreviewing ? "pill active" : "pill"}>{isPreviewing ? "Previewing" : "Preview Ready"}</span>
          <span className="pill">Q: {statusQuality}</span>
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
                  <select onChange={(event) => setSource(EXAMPLES[event.target.value])} defaultValue="Gyroid Fill">
                    {Object.keys(EXAMPLES).map((name) => (
                      <option key={name} value={name}>
                        {name}
                      </option>
                    ))}
                  </select>
                  <button type="button" onClick={() => setIsSignatureHelpOpen(true)}>
                    Signature Help
                  </button>
                  <button onClick={() => void compileAndSync(source)}>Compile now</button>
                  <button type="button" onClick={onGenerateDsl}>
                    Generate Shape
                  </button>
                </div>
              </div>

              <textarea
                value={source}
                onChange={(event) => setSource(event.target.value)}
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
                    onChange={(event) => setMeshFile(event.target.files?.[0] ?? null)}
                  />
                </label>
                <p className="muted">{meshFile ? `Selected: ${meshFile.name}` : "No file selected."}</p>

                <label className="slider-row">
                  <span>Shell thickness</span>
                  <input
                    type="number"
                    step={0.01}
                    min={0.001}
                    value={meshShellThickness}
                    aria-label="Shell thickness"
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
                  <span>Lattice pitch</span>
                  <input
                    type="number"
                    step={0.01}
                    min={0.001}
                    value={meshLatticePitch}
                    aria-label="Lattice pitch"
                    onChange={(event) => setMeshLatticePitch(Number(event.target.value))}
                  />
                </label>

                <label className="slider-row">
                  <span>Lattice thickness</span>
                  <input
                    type="number"
                    step={0.01}
                    min={0.001}
                    value={meshLatticeThickness}
                    aria-label="Lattice thickness"
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

                <label className="slider-row">
                  <span>Preview quality</span>
                  <select
                    aria-label="Mesh preview quality"
                    value={meshPreviewQuality}
                    onChange={(event) => setMeshPreviewQuality(event.target.value as QualityProfile)}
                  >
                    {QUALITY_ORDER.map((item) => (
                      <option key={item} value={item}>
                        {item}
                      </option>
                    ))}
                  </select>
                </label>

                <label className="slider-row">
                  <span>Export quality</span>
                  <select
                    aria-label="Mesh export quality"
                    value={meshExportQuality}
                    onChange={(event) => setMeshExportQuality(event.target.value as QualityProfile)}
                  >
                    {QUALITY_ORDER.map((item) => (
                      <option key={item} value={item}>
                        {item}
                      </option>
                    ))}
                  </select>
                </label>

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

                <button type="button" onClick={() => void onGenerateMesh()}>
                  Generate Mesh Preview
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
            <button onClick={() => void (workflow === "dsl" ? onExportDsl("stl") : onExportUploaded("stl"))}>
              Export STL
            </button>
            <button onClick={() => void (workflow === "dsl" ? onExportDsl("obj") : onExportUploaded("obj"))}>
              Export OBJ
            </button>
          </div>

          {sectionEnabled ? (
            <div className="section-row">
              <span>Section Y: {sectionLevel.toFixed(2)}</span>
              <input
                type="range"
                min={-2}
                max={2}
                step={0.01}
                value={sectionLevel}
                onChange={(event) => setSectionLevel(Number(event.target.value))}
              />
            </div>
          ) : null}

          <div className="viewer-wrap">
            <Viewer
              mesh={mesh}
              field={field}
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
            <span>Cache: {stats?.cache_hit ? "hit" : "miss"}</span>
          </div>
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
