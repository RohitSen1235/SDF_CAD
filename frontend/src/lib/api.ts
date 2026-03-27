import {
  ComputeBackend,
  CompileDiagnostics,
  ComputePrecision,
  FieldPayload,
  FieldPayloadBinary,
  GridConfig,
  MeshBackend,
  MeshingMode,
  MeshPayloadBinary,
  UploadedMeshMemoryContext,
  UploadedMeshPreprocessResponse,
  MeshWorkflowParams,
  PreviewStats,
  PreviewFieldResponse,
  PreviewMeshResponse,
  QualityProfile,
  SceneIR,
  UploadedFieldPreviewClientTelemetry,
  UploadedPreviewFieldResponse,
  StructuralOptimizationIterationResult,
  StructuralOptimizationJobAcceptedResponse,
  StructuralOptimizationPreprocessResponse as StructuralOptimizationPreprocessResponseJson,
  StructuralOptimizationProgressResponse,
  StructuralOptimizationRequest,
  StructuralOptimizationResultResponse
} from "../types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

export interface CompileSceneResult {
  sceneIr: SceneIR;
  diagnostics: CompileDiagnostics;
}

interface JobAcceptedResponse {
  job_id: string;
  status: "queued" | "running" | "succeeded" | "failed";
  status_url: string;
  result_url: string;
}

interface JobStatusResponse {
  job_id: string;
  status: "queued" | "running" | "succeeded" | "failed";
  task_name: string;
  detail?: string | null;
}

async function parseJsonOrThrow(response: Response): Promise<any> {
  if (response.ok) {
    return response.json();
  }
  const payload = await response.json().catch(() => ({ detail: "Request failed" }));
  throw new Error(payload.detail ?? "Request failed");
}

async function parseErrorResponse(response: Response, fallback: string): Promise<never> {
  const payload = await response.json().catch(() => ({ detail: fallback }));
  throw new Error(payload.detail ?? fallback);
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

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

function toWebSocketBase(base: string): string {
  if (base.startsWith("https://")) {
    return `wss://${base.slice("https://".length)}`;
  }
  if (base.startsWith("http://")) {
    return `ws://${base.slice("http://".length)}`;
  }
  return base;
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

function parseStatsHeader(response: Response): PreviewStats {
  const raw = response.headers.get("X-SDF-Stats");
  if (!raw) {
    throw new Error("Missing X-SDF-Stats response header");
  }
  try {
    return JSON.parse(raw) as PreviewStats;
  } catch {
    throw new Error("Invalid X-SDF-Stats response header");
  }
}

function parseBoundsHeader(response: Response): [[number, number], [number, number], [number, number]] {
  const raw = response.headers.get("X-SDF-Bounds");
  if (!raw) {
    throw new Error("Missing X-SDF-Bounds response header");
  }
  try {
    const parsed = JSON.parse(raw) as [[number, number], [number, number], [number, number]];
    if (!Array.isArray(parsed) || parsed.length !== 3) {
      throw new Error();
    }
    return parsed;
  } catch {
    throw new Error("Invalid X-SDF-Bounds response header");
  }
}

function parseResolutionXYZHeader(response: Response): [number, number, number] {
  const raw = response.headers.get("X-SDF-Resolution-XYZ");
  if (!raw) {
    throw new Error("Missing X-SDF-Resolution-XYZ response header");
  }
  const parts = raw.split(",").map((part) => Number(part.trim()));
  if (
    parts.length !== 3 ||
    !Number.isFinite(parts[0]) ||
    !Number.isFinite(parts[1]) ||
    !Number.isFinite(parts[2]) ||
    parts[0] <= 0 ||
    parts[1] <= 0 ||
    parts[2] <= 0
  ) {
    throw new Error("Invalid X-SDF-Resolution-XYZ response header");
  }
  return [parts[0], parts[1], parts[2]];
}

function parseOptionalResolutionXYZHeader(response: Response): [number, number, number] | null {
  const raw = response.headers.get("X-SDF-Resolution-XYZ");
  if (!raw || raw.trim() === "") {
    return null;
  }
  const parts = raw.split(",").map((part) => Number(part.trim()));
  if (
    parts.length !== 3 ||
    !Number.isFinite(parts[0]) ||
    !Number.isFinite(parts[1]) ||
    !Number.isFinite(parts[2]) ||
    parts[0] <= 0 ||
    parts[1] <= 0 ||
    parts[2] <= 0
  ) {
    return null;
  }
  return [parts[0], parts[1], parts[2]];
}

function parseMeshExtentsHeader(response: Response): [number, number, number] | null {
  const raw = response.headers.get("X-SDF-Mesh-Extents");
  if (!raw || raw.trim() === "") {
    return null;
  }
  const parts = raw.split(",").map((part) => Number(part.trim()));
  if (
    parts.length !== 3 ||
    !Number.isFinite(parts[0]) ||
    !Number.isFinite(parts[1]) ||
    !Number.isFinite(parts[2]) ||
    parts[0] <= 0 ||
    parts[1] <= 0 ||
    parts[2] <= 0
  ) {
    return null;
  }
  return [parts[0], parts[1], parts[2]];
}

function parseTraceIdHeader(response: Response): string | null {
  const raw = response.headers.get("X-SDF-Trace-Id");
  if (!raw) {
    return null;
  }
  const value = raw.trim();
  return value.length > 0 ? value : null;
}

function parseOptionalNumericHeader(response: Response, name: string): number | null {
  const raw = response.headers.get(name);
  if (raw == null || raw.trim() === "") {
    return null;
  }
  const parsed = Number(raw);
  if (!Number.isFinite(parsed)) {
    return null;
  }
  return parsed;
}

function parseUploadedMeshMemoryContext(response: Response): UploadedMeshMemoryContext | null {
  const meshSpan = parseOptionalNumericHeader(response, "X-SDF-Mesh-Span");
  const meshExtents = parseMeshExtentsHeader(response);
  const resolutionXYZ = parseOptionalResolutionXYZHeader(response);
  const cpuBytesPerVoxel = parseOptionalNumericHeader(response, "X-SDF-CPU-Bytes-Per-Voxel");
  const gpuBytesPerVoxel = parseOptionalNumericHeader(response, "X-SDF-GPU-Bytes-Per-Voxel");
  const meshGpuBytesPerVoxel = parseOptionalNumericHeader(response, "X-SDF-MESH-GPU-Bytes-Per-Voxel");
  const safetyFactor = parseOptionalNumericHeader(response, "X-SDF-Memory-Safety-Factor");
  if (
    meshSpan == null ||
    meshExtents == null ||
    cpuBytesPerVoxel == null ||
    gpuBytesPerVoxel == null ||
    meshGpuBytesPerVoxel == null ||
    safetyFactor == null
  ) {
    return null;
  }
  return {
    meshSpan,
    meshExtents,
    resolutionXYZ,
    availableCpuBytes: parseOptionalNumericHeader(response, "X-SDF-Available-CPU-Bytes"),
    availableGpuFreeBytes: parseOptionalNumericHeader(response, "X-SDF-Available-GPU-Free-Bytes"),
    availableGpuTotalBytes: parseOptionalNumericHeader(response, "X-SDF-Available-GPU-Total-Bytes"),
    cpuBytesPerVoxel,
    gpuBytesPerVoxel,
    meshGpuBytesPerVoxel,
    safetyFactor
  };
}

export function decodeBinaryMeshPacket(buffer: ArrayBuffer): MeshPayloadBinary {
  const view = new DataView(buffer);
  if (buffer.byteLength < 16) {
    throw new Error("Invalid binary mesh payload");
  }
  const magic = new TextDecoder().decode(new Uint8Array(buffer, 0, 8));
  if (magic !== "SDFMESH1") {
    throw new Error("Unsupported binary mesh payload format");
  }

  const vertexCount = view.getUint32(8, true);
  const faceCount = view.getUint32(12, true);
  const verticesBytes = vertexCount * 3 * Float32Array.BYTES_PER_ELEMENT;
  const indicesBytes = faceCount * 3 * Uint32Array.BYTES_PER_ELEMENT;
  const normalsBytes = vertexCount * 3 * Float32Array.BYTES_PER_ELEMENT;
  const expected = 16 + verticesBytes + indicesBytes + normalsBytes;
  if (buffer.byteLength !== expected) {
    throw new Error("Binary mesh payload size mismatch");
  }

  let offset = 16;
  const vertices = new Float32Array(buffer.slice(offset, offset + verticesBytes));
  offset += verticesBytes;
  const indices = new Uint32Array(buffer.slice(offset, offset + indicesBytes));
  offset += indicesBytes;
  const normals = new Float32Array(buffer.slice(offset, offset + normalsBytes));

  return {
    encoding: "mesh-f32-u32-binary-v1",
    vertex_count: vertexCount,
    face_count: faceCount,
    vertices,
    indices,
    normals
  };
}

function decodeBinaryFieldPayload(
  buffer: ArrayBuffer,
  resolutionXYZ: [number, number, number],
  bounds: [[number, number], [number, number], [number, number]]
): FieldPayloadBinary {
  const values = new Float32Array(buffer.slice(0));
  const expected = resolutionXYZ[0] * resolutionXYZ[1] * resolutionXYZ[2];
  if (values.length !== expected) {
    throw new Error("Binary field payload size mismatch");
  }
  return {
    encoding: "f32-binary-v1",
    resolution_xyz: resolutionXYZ,
    bounds,
    data: values
  };
}

function decodeBinaryUploadedFieldPayload(
  buffer: ArrayBuffer,
  resolutionXYZ: [number, number, number],
  bounds: [[number, number], [number, number], [number, number]]
): { field: FieldPayloadBinary; hostField: FieldPayloadBinary | null } {
  const expectedFloats = resolutionXYZ[0] * resolutionXYZ[1] * resolutionXYZ[2];
  const expectedBytes = expectedFloats * Float32Array.BYTES_PER_ELEMENT;
  if (buffer.byteLength !== expectedBytes && buffer.byteLength !== expectedBytes * 2) {
    throw new Error("Binary uploaded field payload size mismatch");
  }

  const hasHostField = buffer.byteLength === expectedBytes * 2;
  const hostField = hasHostField ? decodeBinaryFieldPayload(buffer.slice(0, expectedBytes), resolutionXYZ, bounds) : null;
  const field = decodeBinaryFieldPayload(buffer.slice(hasHostField ? expectedBytes : 0), resolutionXYZ, bounds);
  return { field, hostField };
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
    const requestBody = {
      scene_ir: sceneIr,
      parameter_values: parameterValues,
      quality_profile: qualityProfile,
      compute_precision: computePrecision,
      compute_backend: computeBackend,
      mesh_backend: meshBackend,
      meshing_mode: meshingMode,
      grid
    };
    const response = await fetch(`${API_BASE}/api/v1/preview/mesh.binary`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal,
      body: JSON.stringify(requestBody)
    });
    if (response.ok) {
      const packet = await response.arrayBuffer();
      return {
        mesh: decodeBinaryMeshPacket(packet),
        stats: parseStatsHeader(response)
      };
    }
    if (response.status !== 404 && response.status !== 405) {
      await parseErrorResponse(response, "Mesh preview failed");
    }

    const fallback = await fetch(`${API_BASE}/api/v1/preview/mesh`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal,
      body: JSON.stringify(requestBody)
    });
    return parseJsonOrThrow(fallback);
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
    const requestBody = {
      scene_ir: sceneIr,
      parameter_values: parameterValues,
      quality_profile: qualityProfile,
      compute_precision: computePrecision,
      compute_backend: computeBackend,
      grid
    };
    const response = await fetch(`${API_BASE}/api/v1/preview/field.binary`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal,
      body: JSON.stringify(requestBody)
    });
    if (response.ok) {
      const stats = parseStatsHeader(response);
      const resolutionXYZ = parseResolutionXYZHeader(response);
      const bounds = parseBoundsHeader(response);
      const packet = await response.arrayBuffer();
      return {
        field: decodeBinaryFieldPayload(packet, resolutionXYZ, bounds),
        stats
      };
    }
    if (response.status !== 404 && response.status !== 405) {
      await parseErrorResponse(response, "Field preview failed");
    }

    const fallback = await fetch(`${API_BASE}/api/v1/preview/field`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal,
      body: JSON.stringify(requestBody)
    });
    return parseJsonOrThrow(fallback);
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
): Promise<void> {
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
        meshing_mode: meshingMode,
        execution_mode: "auto"
      })
    });

    if (!response.ok) {
      await parseErrorResponse(response, "Export failed");
    }

    const contentType = response.headers.get("content-type") ?? "";
    if (contentType.includes("application/json")) {
      const accepted = (await response.json()) as JobAcceptedResponse;
      await waitForJobAndTriggerBrowserDownload(accepted);
      return;
    }

    const blob = await response.blob();
    triggerBlobDownload(blob, `sdf-model.${format}`);
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

async function waitForJobAndTriggerBrowserDownload(job: JobAcceptedResponse): Promise<void> {
  const maxAttempts = 600;
  for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
    const statusResponse = await fetch(`${API_BASE}${job.status_url}`, { method: "GET" });
    const statusPayload = (await parseJsonOrThrow(statusResponse)) as JobStatusResponse;
    if (statusPayload.status === "succeeded") {
      window.location.assign(`${API_BASE}${job.result_url}`);
      return;
    }
    if (statusPayload.status === "failed") {
      throw new Error(statusPayload.detail ?? "Export job failed");
    }
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  throw new Error("Export timed out while waiting for queued job result");
}

function triggerBlobDownload(blob: Blob, filename: string): void {
  const link = document.createElement("a");
  const href = URL.createObjectURL(blob);
  link.href = href;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(href);
}

function appendMeshWorkflowFormData(
  body: FormData,
  file: File,
  params: MeshWorkflowParams,
  computeBackend: ComputeBackend = "auto",
  meshBackend: MeshBackend = "auto",
  meshingMode: MeshingMode = "uniform",
  voxelsPerLatticePeriod: number = 6
): void {
  body.append("file", file, file.name);
  body.append("shell_thickness", String(params.shellThickness));
  body.append("lattice_type", params.latticeType);
  body.append("lattice_pitch", String(params.latticePitch));
  body.append("lattice_thickness", String(params.latticeThickness));
  body.append("lattice_phase", String(params.latticePhase));
  body.append("voxels_per_lattice_period", String(voxelsPerLatticePeriod));
  body.append("compute_backend", computeBackend);
  body.append("mesh_backend", meshBackend);
  body.append("meshing_mode", meshingMode);
}

export async function previewUploadedMesh(
  file: File,
  params: MeshWorkflowParams,
  computeBackend: ComputeBackend = "auto",
  meshBackend: MeshBackend = "auto",
  meshingMode: MeshingMode = "uniform",
  voxelsPerLatticePeriod: number = 6
): Promise<PreviewMeshResponse> {
  try {
    const body = new FormData();
    appendMeshWorkflowFormData(body, file, params, computeBackend, meshBackend, meshingMode, voxelsPerLatticePeriod);
    const response = await fetch(`${API_BASE}/api/v1/mesh/preview.binary`, {
      method: "POST",
      body
    });
    if (response.ok) {
      const packet = await response.arrayBuffer();
      return {
        mesh: decodeBinaryMeshPacket(packet),
        stats: parseStatsHeader(response)
      };
    }
    if (response.status !== 404 && response.status !== 405) {
      await parseErrorResponse(response, "Uploaded mesh preview failed");
    }

    const fallbackBody = new FormData();
    appendMeshWorkflowFormData(fallbackBody, file, params, computeBackend, meshBackend, meshingMode, voxelsPerLatticePeriod);
    const fallback = await fetch(`${API_BASE}/api/v1/mesh/preview`, {
      method: "POST",
      body: fallbackBody
    });
    return parseJsonOrThrow(fallback);
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

export async function commitUploadedMesh(
  file: File,
  params: MeshWorkflowParams,
  computeBackend: ComputeBackend = "auto",
  meshBackend: MeshBackend = "auto",
  meshingMode: MeshingMode = "uniform",
  voxelsPerLatticePeriod: number = 6
): Promise<PreviewMeshResponse> {
  try {
    const body = new FormData();
    appendMeshWorkflowFormData(body, file, params, computeBackend, meshBackend, meshingMode, voxelsPerLatticePeriod);
    const response = await fetch(`${API_BASE}/api/v1/mesh/commit.binary`, {
      method: "POST",
      body
    });
    if (response.ok) {
      const packet = await response.arrayBuffer();
      return {
        mesh: decodeBinaryMeshPacket(packet),
        stats: parseStatsHeader(response)
      };
    }
    if (response.status !== 404 && response.status !== 405) {
      await parseErrorResponse(response, "Uploaded mesh commit failed");
    }

    const fallbackBody = new FormData();
    appendMeshWorkflowFormData(
      fallbackBody,
      file,
      params,
      computeBackend,
      meshBackend,
      meshingMode,
      voxelsPerLatticePeriod
    );
    const fallback = await fetch(`${API_BASE}/api/v1/mesh/commit`, {
      method: "POST",
      body: fallbackBody
    });
    return parseJsonOrThrow(fallback);
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

export async function previewUploadedMeshField(
  file: File,
  params: MeshWorkflowParams,
  computeBackend: ComputeBackend = "auto",
  voxelsPerLatticePeriod: number = 6,
  signal?: AbortSignal
): Promise<UploadedPreviewFieldResponse> {
  try {
    const body = new FormData();
    appendMeshWorkflowFormData(
      body,
      file,
      params,
      computeBackend,
      "auto",
      "uniform",
      voxelsPerLatticePeriod
    );
    const fetchStartedAt = performance.now();
    const response = await fetch(`${API_BASE}/api/v1/mesh/field.binary`, {
      method: "POST",
      body,
      signal
    });
    if (response.ok) {
      const headersReceivedAt = performance.now();
      const stats = parseStatsHeader(response);
      const resolutionXYZ = parseResolutionXYZHeader(response);
      const bounds = parseBoundsHeader(response);
      const traceId = parseTraceIdHeader(response);
      const packet = await response.arrayBuffer();
      const arrayBufferDoneAt = performance.now();
      const { field, hostField } = decodeBinaryUploadedFieldPayload(packet, resolutionXYZ, bounds);
      const decodeDoneAt = performance.now();
      return {
        field,
        hostField,
        stats,
        trace: traceId
          ? {
              traceId,
              clientResponseWaitMs: headersReceivedAt - fetchStartedAt,
              clientDownloadMs: arrayBufferDoneAt - headersReceivedAt,
              clientDecodeMs: decodeDoneAt - arrayBufferDoneAt,
              fieldAssignedAtMs: decodeDoneAt
            }
          : null
      };
    }
    if (response.status !== 404 && response.status !== 405) {
      await parseErrorResponse(response, "Uploaded field preview failed");
    }

    const fallbackBody = new FormData();
    appendMeshWorkflowFormData(
      fallbackBody,
      file,
      params,
      computeBackend,
      "auto",
      "uniform",
      voxelsPerLatticePeriod
    );
    const fallback = await fetch(`${API_BASE}/api/v1/mesh/field`, {
      method: "POST",
      body: fallbackBody,
      signal
    });
    const fallbackResponseReceivedAt = performance.now();
    const payload = (await parseJsonOrThrow(fallback)) as {
      field: UploadedPreviewFieldResponse["field"];
      host_field?: UploadedPreviewFieldResponse["hostField"];
      stats: PreviewStats;
    };
    const fallbackDecodeDoneAt = performance.now();
    const traceId = parseTraceIdHeader(fallback);
    return {
      field: payload.field,
      hostField: payload.host_field ?? null,
      stats: payload.stats,
      trace: traceId
        ? {
            traceId,
            clientResponseWaitMs: fallbackResponseReceivedAt - fetchStartedAt,
            clientDownloadMs: 0,
            clientDecodeMs: fallbackDecodeDoneAt - fallbackResponseReceivedAt,
            fieldAssignedAtMs: fallbackDecodeDoneAt
          }
        : null
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

export async function preprocessUploadedMesh(
  file: File,
  params: MeshWorkflowParams,
  computeBackend: ComputeBackend = "auto",
  voxelsPerLatticePeriod: number = 6,
  signal?: AbortSignal
): Promise<UploadedMeshPreprocessResponse> {
  // Preprocess only needs the upload and a few compatibility fields; the host
  // SDF is built later when the user clicks "Preview Field".
  const body = new FormData();
  body.append("file", file, file.name);
  body.append("lattice_pitch", String(params.latticePitch));
  body.append("voxels_per_lattice_period", String(voxelsPerLatticePeriod));
  body.append("compute_backend", computeBackend);
  try {
    const response = await fetch(`${API_BASE}/api/v1/mesh/preprocess`, {
      method: "POST",
      body,
      signal
    });
    if (!response.ok) {
      await parseErrorResponse(response, "Mesh preprocess failed");
    }
    const packet = await response.arrayBuffer();
    return {
      mesh: decodeBinaryMeshPacket(packet),
      memoryContext: parseUploadedMeshMemoryContext(response)
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

export async function preprocessStructuralOptimization(
  designFile: File,
  nonDesignFile: File,
  resolution: number,
  computeBackend: ComputeBackend = "auto",
  meshBackend: MeshBackend = "auto",
  signal?: AbortSignal
): Promise<StructuralOptimizationPreprocessResponseJson> {
  const body = new FormData();
  body.append("design_space_file", designFile, designFile.name);
  body.append("non_design_space_file", nonDesignFile, nonDesignFile.name);
  body.append("resolution", String(resolution));
  body.append("compute_backend", computeBackend);
  body.append("mesh_backend", meshBackend);
  try {
    const response = await fetch(`${API_BASE}/api/v1/optimization/structural/preprocess`, {
      method: "POST",
      body,
      signal
    });
    if (!response.ok) {
      await parseErrorResponse(response, "Structural optimization preprocess failed");
    }
    return (await response.json()) as StructuralOptimizationPreprocessResponseJson;
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

export async function runStructuralOptimizationPhased(
  request: StructuralOptimizationRequest,
  onIteration?: (iteration: StructuralOptimizationIterationResult) => void
): Promise<StructuralOptimizationResultResponse> {
  const submitResponse = await fetch(`${API_BASE}/api/v1/optimization/structural/jobs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request)
  }).catch((error: unknown) => {
    throw asNetworkError(error);
  });

  if (!submitResponse.ok) {
    await parseErrorResponse(submitResponse, "Structural optimization job submission failed");
  }

  const accepted = (await submitResponse.json()) as StructuralOptimizationJobAcceptedResponse;
  let afterIteration = 0;

  while (true) {
    let progressResponse: Response;
    try {
      const progressUrl = new URL(accepted.progress_url, API_BASE);
      progressUrl.searchParams.set("after_iteration", String(afterIteration));
      progressResponse = await fetch(progressUrl.toString());
    } catch (error) {
      throw asNetworkError(error);
    }

    if (!progressResponse.ok) {
      await parseErrorResponse(progressResponse, "Structural optimization progress polling failed");
    }

    const progress = (await progressResponse.json()) as StructuralOptimizationProgressResponse;
    for (const iteration of progress.iterations) {
      afterIteration = Math.max(afterIteration, iteration.iteration);
      onIteration?.(iteration);
    }

    if (progress.status === "failed") {
      throw new Error(progress.detail ?? "Structural optimization failed");
    }
    if (progress.status === "succeeded") {
      if (progress.final_result) {
        return progress.final_result;
      }
      const resultResponse = await fetch(new URL(accepted.result_url, API_BASE).toString()).catch((error: unknown) => {
        throw asNetworkError(error);
      });
      if (!resultResponse.ok) {
        await parseErrorResponse(resultResponse, "Structural optimization result retrieval failed");
      }
      return (await resultResponse.json()) as StructuralOptimizationResultResponse;
    }

    await sleep(1000);
  }
}

export async function buildStructuralOptimizationRequest(
  baseRequest: Omit<StructuralOptimizationRequest, "design_space_file_name" | "design_space_file_data_base64" | "non_design_space_file_name" | "non_design_space_file_data_base64">,
  designFile: File,
  nonDesignFile: File
): Promise<StructuralOptimizationRequest> {
  const [designB64, nonDesignB64] = await Promise.all([fileToBase64(designFile), fileToBase64(nonDesignFile)]);
  return {
    ...baseRequest,
    design_space_file_name: designFile.name,
    design_space_file_data_base64: designB64,
    non_design_space_file_name: nonDesignFile.name,
    non_design_space_file_data_base64: nonDesignB64
  };
}

export async function submitUploadedFieldPreviewTelemetry(
  payload: UploadedFieldPreviewClientTelemetry
): Promise<void> {
  try {
    const response = await fetch(`${API_BASE}/api/v1/internal/mesh/field-preview-telemetry`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (!response.ok) {
      await parseErrorResponse(response, "Uploaded field preview telemetry failed");
    }
  } catch (error) {
    if (error instanceof TypeError) {
      throw asNetworkError(error);
    }
    throw error;
  }
}

export async function previewUploadedMeshPhased(
  file: File,
  params: MeshWorkflowParams,
  computeBackend: ComputeBackend = "auto",
  meshBackend: MeshBackend = "auto",
  meshingMode: MeshingMode = "uniform",
  onField?: (field: FieldPayload, stats?: PreviewStats) => void,
  voxelsPerLatticePeriod: number = 6
): Promise<PreviewMeshResponse> {
  const fileBase64 = await fileToBase64(file);
  const wsBase = toWebSocketBase(API_BASE);
  const url = `${wsBase}/api/v1/mesh/preview/ws`;
  const fallbackToHttp = async (): Promise<PreviewMeshResponse> =>
    previewUploadedMesh(file, params, computeBackend, meshBackend, meshingMode, voxelsPerLatticePeriod);

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
          voxels_per_lattice_period: voxelsPerLatticePeriod,
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
  computeBackend: ComputeBackend = "auto",
  meshBackend: MeshBackend = "auto",
  meshingMode: MeshingMode = "uniform",
  voxelsPerLatticePeriod: number = 6
): Promise<void> {
  try {
    const body = new FormData();
    appendMeshWorkflowFormData(body, file, params, computeBackend, meshBackend, meshingMode, voxelsPerLatticePeriod);
    body.append("format", format);
    body.append("execution_mode", "auto");
    const response = await fetch(`${API_BASE}/api/v1/mesh/export`, {
      method: "POST",
      body
    });

    if (!response.ok) {
      await parseErrorResponse(response, "Export failed");
    }

    const contentType = response.headers.get("content-type") ?? "";
    if (contentType.includes("application/json")) {
      const accepted = (await response.json()) as JobAcceptedResponse;
      await waitForJobAndTriggerBrowserDownload(accepted);
      return;
    }

    const blob = await response.blob();
    triggerBlobDownload(blob, `mesh-lattice.${format}`);
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
