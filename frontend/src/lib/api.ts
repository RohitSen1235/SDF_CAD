import {
  ComputeBackend,
  CompileDiagnostics,
  ComputePrecision,
  GridConfig,
  MeshBackend,
  MeshingMode,
  MeshWorkflowParams,
  PreviewFieldResponse,
  PreviewMeshResponse,
  QualityProfile,
  SceneIR
} from "../types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

export interface CompileSceneResult {
  sceneIr: SceneIR;
  diagnostics: CompileDiagnostics;
}

async function parseJsonOrThrow(response: Response): Promise<any> {
  if (response.ok) {
    return response.json();
  }
  const payload = await response.json().catch(() => ({ detail: "Request failed" }));
  throw new Error(payload.detail ?? "Request failed");
}

function asNetworkError(error: unknown): Error {
  const err = error as Error;
  if (err?.name === "AbortError") {
    return err;
  }
  return new Error(
    `Cannot reach backend API at ${API_BASE}. Start backend with: cd backend && source .venv/bin/activate && uvicorn app.main:app --reload`
  );
}

export async function compileScene(source: string): Promise<CompileSceneResult> {
  try {
    const response = await fetch(`${API_BASE}/api/v1/scene/compile`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source })
    });
    const payload = await parseJsonOrThrow(response);
    return {
      sceneIr: payload.scene_ir as SceneIR,
      diagnostics: (payload.diagnostics ?? { warnings: [] }) as CompileDiagnostics
    };
  } catch (error) {
    if ((error as Error)?.name === "AbortError") {
      throw error;
    }
    if (error instanceof TypeError) {
      throw asNetworkError(error);
    }
    throw error;
  }
}

export async function previewMesh(
  sceneIr: SceneIR,
  parameterValues: Record<string, number>,
  qualityProfile: QualityProfile,
  computePrecision: ComputePrecision = "float32",
  computeBackend: ComputeBackend = "auto",
  meshBackend: MeshBackend = "auto",
  meshingMode: MeshingMode = "uniform",
  signal?: AbortSignal,
  grid?: GridConfig
): Promise<PreviewMeshResponse> {
  try {
    const response = await fetch(`${API_BASE}/api/v1/preview/mesh`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal,
      body: JSON.stringify({
        scene_ir: sceneIr,
        parameter_values: parameterValues,
        quality_profile: qualityProfile,
        compute_precision: computePrecision,
        compute_backend: computeBackend,
        mesh_backend: meshBackend,
        meshing_mode: meshingMode,
        grid
      })
    });
    return parseJsonOrThrow(response);
  } catch (error) {
    if ((error as Error)?.name === "AbortError") {
      throw error;
    }
    if (error instanceof TypeError) {
      throw asNetworkError(error);
    }
    throw error;
  }
}

export async function previewField(
  sceneIr: SceneIR,
  parameterValues: Record<string, number>,
  qualityProfile: QualityProfile,
  computePrecision: ComputePrecision = "float32",
  computeBackend: ComputeBackend = "auto",
  signal?: AbortSignal,
  grid?: GridConfig
): Promise<PreviewFieldResponse> {
  try {
    const response = await fetch(`${API_BASE}/api/v1/preview/field`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal,
      body: JSON.stringify({
        scene_ir: sceneIr,
        parameter_values: parameterValues,
        quality_profile: qualityProfile,
        compute_precision: computePrecision,
        compute_backend: computeBackend,
        grid
      })
    });
    return parseJsonOrThrow(response);
  } catch (error) {
    if ((error as Error)?.name === "AbortError") {
      throw error;
    }
    if (error instanceof TypeError) {
      throw asNetworkError(error);
    }
    throw error;
  }
}

export async function exportMesh(
  sceneIr: SceneIR,
  parameterValues: Record<string, number>,
  format: "stl" | "obj",
  qualityProfile: QualityProfile,
  computePrecision: ComputePrecision = "float32",
  computeBackend: ComputeBackend = "auto",
  meshBackend: MeshBackend = "auto",
  meshingMode: MeshingMode = "uniform"
): Promise<Blob> {
  try {
    const response = await fetch(`${API_BASE}/api/v1/export`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        scene_ir: sceneIr,
        parameter_values: parameterValues,
        format,
        quality_profile: qualityProfile,
        compute_precision: computePrecision,
        compute_backend: computeBackend,
        mesh_backend: meshBackend,
        meshing_mode: meshingMode
      })
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({ detail: "Export failed" }));
      throw new Error(payload.detail ?? "Export failed");
    }
    return response.blob();
  } catch (error) {
    if ((error as Error)?.name === "AbortError") {
      throw error;
    }
    if (error instanceof TypeError) {
      throw asNetworkError(error);
    }
    throw error;
  }
}

function appendMeshWorkflowFormData(
  body: FormData,
  file: File,
  params: MeshWorkflowParams,
  qualityProfile: QualityProfile,
  computeBackend: ComputeBackend = "auto",
  meshBackend: MeshBackend = "auto",
  meshingMode: MeshingMode = "uniform"
): void {
  body.append("file", file, file.name);
  body.append("shell_thickness", String(params.shellThickness));
  body.append("lattice_type", params.latticeType);
  body.append("lattice_pitch", String(params.latticePitch));
  body.append("lattice_thickness", String(params.latticeThickness));
  body.append("lattice_phase", String(params.latticePhase));
  body.append("quality_profile", qualityProfile);
  body.append("compute_backend", computeBackend);
  body.append("mesh_backend", meshBackend);
  body.append("meshing_mode", meshingMode);
}

export async function previewUploadedMesh(
  file: File,
  params: MeshWorkflowParams,
  qualityProfile: QualityProfile,
  computeBackend: ComputeBackend = "auto",
  meshBackend: MeshBackend = "auto",
  meshingMode: MeshingMode = "uniform"
): Promise<PreviewMeshResponse> {
  try {
    const body = new FormData();
    appendMeshWorkflowFormData(body, file, params, qualityProfile, computeBackend, meshBackend, meshingMode);
    const response = await fetch(`${API_BASE}/api/v1/mesh/preview`, {
      method: "POST",
      body
    });
    return parseJsonOrThrow(response);
  } catch (error) {
    if ((error as Error)?.name === "AbortError") {
      throw error;
    }
    if (error instanceof TypeError) {
      throw asNetworkError(error);
    }
    throw error;
  }
}

export async function exportUploadedMesh(
  file: File,
  params: MeshWorkflowParams,
  format: "stl" | "obj",
  qualityProfile: QualityProfile,
  computeBackend: ComputeBackend = "auto",
  meshBackend: MeshBackend = "auto",
  meshingMode: MeshingMode = "uniform"
): Promise<Blob> {
  try {
    const body = new FormData();
    appendMeshWorkflowFormData(body, file, params, qualityProfile, computeBackend, meshBackend, meshingMode);
    body.append("format", format);
    const response = await fetch(`${API_BASE}/api/v1/mesh/export`, {
      method: "POST",
      body
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({ detail: "Export failed" }));
      throw new Error(payload.detail ?? "Export failed");
    }
    return response.blob();
  } catch (error) {
    if ((error as Error)?.name === "AbortError") {
      throw error;
    }
    if (error instanceof TypeError) {
      throw asNetworkError(error);
    }
    throw error;
  }
}
