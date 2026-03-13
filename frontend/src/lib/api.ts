import {
  ComputeBackend,
  CompileDiagnostics,
  ComputePrecision,
  FieldPayload,
  GridConfig,
  MeshBackend,
  MeshingMode,
  MeshWorkflowParams,
  PreviewStats,
  PreviewFieldResponse,
  PreviewProgramResponse,
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

function toWebSocketBase(httpBase: string): string {
  if (httpBase.startsWith("https://")) {
    return `wss://${httpBase.slice("https://".length)}`;
  }
  if (httpBase.startsWith("http://")) {
    return `ws://${httpBase.slice("http://".length)}`;
  }
  return httpBase;
}

function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result;
      if (typeof result !== "string") {
        reject(new Error("Failed to read uploaded mesh file"));
        return;
      }
      const marker = "base64,";
      const idx = result.indexOf(marker);
      resolve(idx >= 0 ? result.slice(idx + marker.length) : result);
    };
    reader.onerror = () => reject(new Error("Failed to read uploaded mesh file"));
    reader.readAsDataURL(file);
  });
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

export async function previewProgram(
  sceneIr: SceneIR,
  parameterValues: Record<string, number>,
  qualityProfile: QualityProfile,
  signal?: AbortSignal,
  grid?: GridConfig
): Promise<PreviewProgramResponse> {
  try {
    const response = await fetch(`${API_BASE}/api/v1/preview/program`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal,
      body: JSON.stringify({
        scene_ir: sceneIr,
        parameter_values: parameterValues,
        quality_profile: qualityProfile,
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

export async function previewUploadedMeshProgram(
  file: File,
  params: MeshWorkflowParams,
  qualityProfile: QualityProfile
): Promise<PreviewProgramResponse> {
  try {
    const body = new FormData();
    body.append("file", file, file.name);
    body.append("shell_thickness", String(params.shellThickness));
    body.append("lattice_type", params.latticeType);
    body.append("lattice_pitch", String(params.latticePitch));
    body.append("lattice_thickness", String(params.latticeThickness));
    body.append("lattice_phase", String(params.latticePhase));
    body.append("quality_profile", qualityProfile);
    const response = await fetch(`${API_BASE}/api/v1/mesh/program`, {
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

export async function previewUploadedMeshPhased(
  file: File,
  params: MeshWorkflowParams,
  qualityProfile: QualityProfile,
  computeBackend: ComputeBackend = "auto",
  meshBackend: MeshBackend = "auto",
  meshingMode: MeshingMode = "uniform",
  onField?: (field: FieldPayload, stats?: PreviewStats) => void
): Promise<PreviewMeshResponse> {
  const fileBase64 = await fileToBase64(file);
  const wsBase = toWebSocketBase(API_BASE);
  const url = `${wsBase}/api/v1/mesh/preview/ws`;
  const fallbackToHttp = async (): Promise<PreviewMeshResponse> =>
    previewUploadedMesh(file, params, qualityProfile, computeBackend, meshBackend, meshingMode);

  return new Promise<PreviewMeshResponse>((resolve, reject) => {
    let settled = false;
    let latestField: FieldPayload | null = null;
    const ws = new WebSocket(url);

    ws.onopen = () => {
      ws.send(
        JSON.stringify({
          file_name: file.name,
          file_data_base64: fileBase64,
          shell_thickness: params.shellThickness,
          lattice_type: params.latticeType,
          lattice_pitch: params.latticePitch,
          lattice_thickness: params.latticeThickness,
          lattice_phase: params.latticePhase,
          quality_profile: qualityProfile,
          compute_backend: computeBackend,
          mesh_backend: meshBackend,
          meshing_mode: meshingMode
        })
      );
    };

    ws.onmessage = (event: MessageEvent<string>) => {
      let payload: any;
      try {
        payload = JSON.parse(event.data);
      } catch {
        return;
      }

      if (payload.phase === "field" && payload.field) {
        latestField = payload.field as FieldPayload;
        if (onField) {
          onField(latestField, payload.stats as PreviewStats | undefined);
        }
        return;
      }

      if (payload.phase === "mesh" && payload.mesh && payload.stats) {
        settled = true;
        resolve({
          mesh: payload.mesh as PreviewMeshResponse["mesh"],
          stats: payload.stats as PreviewStats,
          field: (payload.field as FieldPayload | null | undefined) ?? latestField
        });
        return;
      }

      if (payload.phase === "error") {
        settled = true;
        reject(new Error(payload.error ?? "Mesh preview websocket failed"));
      }
    };

    ws.onerror = () => {
      if (settled) {
        return;
      }
      settled = true;
      fallbackToHttp().then(resolve).catch(reject);
    };

    ws.onclose = () => {
      if (settled) {
        return;
      }
      settled = true;
      fallbackToHttp().then(resolve).catch(reject);
    };
  });
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
