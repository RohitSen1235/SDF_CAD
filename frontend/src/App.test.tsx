import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";

const viewerMock = vi.fn();

vi.mock("./components/Viewer", () => ({
  Viewer: (props: unknown) => {
    viewerMock(props);
    return <div data-testid="viewer" />;
  }
}));

const compileScene = vi.fn();
const previewField = vi.fn();
const previewMesh = vi.fn();
const exportMesh = vi.fn();
const preprocessUploadedMesh = vi.fn();
const commitUploadedMesh = vi.fn();
const previewUploadedMeshField = vi.fn();
const exportUploadedMesh = vi.fn();
const submitUploadedFieldPreviewTelemetry = vi.fn();

vi.mock("./lib/api", () => ({
  compileScene: (...args: unknown[]) => compileScene(...args),
  previewField: (...args: unknown[]) => previewField(...args),
  previewMesh: (...args: unknown[]) => previewMesh(...args),
  exportMesh: (...args: unknown[]) => exportMesh(...args),
  preprocessUploadedMesh: (...args: unknown[]) => preprocessUploadedMesh(...args),
  commitUploadedMesh: (...args: unknown[]) => commitUploadedMesh(...args),
  previewUploadedMeshField: (...args: unknown[]) => previewUploadedMeshField(...args),
  exportUploadedMesh: (...args: unknown[]) => exportUploadedMesh(...args),
  submitUploadedFieldPreviewTelemetry: (...args: unknown[]) => submitUploadedFieldPreviewTelemetry(...args)
}));

const compiledScene = {
  nodes: [
    { id: "n1", type: "primitive", primitive: "sphere", inputs: [], params: { r: { $param: "radius" } } },
    { id: "n2", type: "primitive", primitive: "cylinder", inputs: [], params: { r: 0.2, h: 2 } },
    { id: "n3", type: "boolean", op: "difference", inputs: ["n1", "n2"], params: {} }
  ],
  root_node_id: "n3",
  parameter_schema: [{ name: "radius", type: "float", default: 0.8, min: 0.2, max: 1.5, step: 0.1 }],
  source_hash: "abc"
};

function encodeBuffer(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  for (let i = 0; i < bytes.length; i += 1) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

const fieldPreviewPayload = {
  field: {
    encoding: "f32-base64",
    resolution: 2,
    bounds: [
      [-1, 1],
      [-1, 1],
      [-1, 1]
    ],
    data: "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="
  },
  stats: {
    eval_ms: 3,
    mesh_ms: null,
    tri_count: 0,
    voxel_count: 8,
    preview_mode: "field"
  }
};

const meshPreviewPayload = {
  mesh: {
    encoding: "mesh-f32-u32-base64-v1",
    vertex_count: 3,
    face_count: 1,
    vertices_b64: encodeBuffer(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]).buffer),
    indices_b64: encodeBuffer(new Uint32Array([0, 1, 2]).buffer),
    normals_b64: encodeBuffer(new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]).buffer)
  },
  stats: {
    eval_ms: 3,
    mesh_ms: 4,
    tri_count: 1
  }
};

describe("App", () => {
  const outerMeshPayload = {
    encoding: "mesh-f32-u32-binary-v1" as const,
    vertex_count: 4,
    face_count: 4,
    vertices: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]),
    indices: new Uint32Array([0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3]),
    normals: new Float32Array([0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1])
  };

  beforeEach(() => {
    vi.clearAllMocks();
    viewerMock.mockClear();
    window.localStorage.clear();
    compileScene.mockResolvedValue({
      sceneIr: compiledScene,
      diagnostics: { warnings: [], inferred_bounds: [[-1, 1], [-1, 1], [-1, 1]] }
    });
    previewField.mockResolvedValue(fieldPreviewPayload);
    previewMesh.mockResolvedValue(meshPreviewPayload);
    commitUploadedMesh.mockResolvedValue({
      ...meshPreviewPayload,
      stats: {
        ...meshPreviewPayload.stats,
        eval_ms: 0,
        mesh_ms: 4,
        field_cache_hit: true,
        mesh_cache_hit: false
      }
    });
    exportMesh.mockResolvedValue(new Blob(["ok"]));
    preprocessUploadedMesh.mockResolvedValue({
      mesh: outerMeshPayload,
      memoryContext: {
        meshSpan: 1,
        availableCpuBytes: 16 * 1024 * 1024 * 1024,
        availableGpuFreeBytes: 8 * 1024 * 1024 * 1024,
        availableGpuTotalBytes: 12 * 1024 * 1024 * 1024,
        cpuBytesPerVoxel: 32,
        gpuBytesPerVoxel: 40,
        safetyFactor: 1.25
      }
    });
    previewUploadedMeshField.mockResolvedValue(fieldPreviewPayload);
    exportUploadedMesh.mockResolvedValue(new Blob(["ok"]));
    submitUploadedFieldPreviewTelemetry.mockResolvedValue(undefined);
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("renders DSL workflow with header module tabs", () => {
    render(<App />);
    expect(screen.getByRole("tab", { name: "DSL" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("tab", { name: "Lattice Infill" })).toHaveAttribute("aria-selected", "false");
    expect(screen.getByRole("button", { name: "Compile now" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Generate Shape" })).toBeInTheDocument();
  });

  it("does not auto-compile or auto-preview when source changes", () => {
    render(<App />);
    fireEvent.change(screen.getByLabelText("DSL source editor"), { target: { value: "root = sphere(r=1.2)" } });
    expect(compileScene).not.toHaveBeenCalled();
    expect(previewField).not.toHaveBeenCalled();
    expect(previewMesh).not.toHaveBeenCalled();
  });

  it("compiles only when Compile now is clicked", async () => {
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "Compile now" }));

    await waitFor(() => {
      expect(compileScene).toHaveBeenCalledTimes(1);
    });
    expect(previewField).not.toHaveBeenCalled();
    expect(previewMesh).not.toHaveBeenCalled();
  });

  it("runs exactly one field and one mesh preview on Generate Shape", async () => {
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "Generate Shape" }));

    await waitFor(() => {
      expect(previewMesh).toHaveBeenCalledTimes(1);
    });

    expect(compileScene).toHaveBeenCalledTimes(1);
    expect(previewField).toHaveBeenCalledTimes(1);
    expect(previewMesh).toHaveBeenCalledTimes(1);
    const fieldCall = previewField.mock.calls[0];
    const meshCall = previewMesh.mock.calls[0];
    expect(fieldCall?.[2]).toBe("high");
    expect(meshCall?.[2]).toBe("high");
    expect(fieldCall?.[6]).toMatchObject({ resolution: 192 });
    expect(meshCall?.[8]).toMatchObject({ resolution: 192 });
  });

  it("sends selected quality, precision, and backends only on Generate Shape", async () => {
    render(<App />);

    fireEvent.change(screen.getByLabelText("Quality"), { target: { value: "ultra" } });
    fireEvent.change(screen.getByLabelText("SDF Precision"), { target: { value: "float16" } });
    fireEvent.change(screen.getByLabelText("Eval Backend"), { target: { value: "cuda" } });
    fireEvent.change(screen.getByLabelText("Mesh Backend"), { target: { value: "cuda" } });
    fireEvent.change(screen.getByLabelText("Meshing Mode"), { target: { value: "adaptive" } });

    expect(previewField).not.toHaveBeenCalled();
    expect(previewMesh).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: "Generate Shape" }));
    await waitFor(() => {
      expect(previewMesh).toHaveBeenCalledTimes(1);
    });

    const fieldCall = previewField.mock.calls[0];
    const meshCall = previewMesh.mock.calls[0];
    expect(fieldCall?.[2]).toBe("ultra");
    expect(fieldCall?.[3]).toBe("float16");
    expect(fieldCall?.[4]).toBe("cuda");
    expect(fieldCall?.[6]).toMatchObject({ resolution: 256 });
    expect(meshCall?.[2]).toBe("ultra");
    expect(meshCall?.[3]).toBe("float16");
    expect(meshCall?.[4]).toBe("cuda");
    expect(meshCall?.[5]).toBe("cuda");
    expect(meshCall?.[6]).toBe("adaptive");
    expect(meshCall?.[8]).toMatchObject({ resolution: 256 });
  });

  it("does not auto-rerun preview when quality/backend settings change", async () => {
    render(<App />);

    fireEvent.click(screen.getByRole("button", { name: "Generate Shape" }));
    await waitFor(() => {
      expect(previewMesh).toHaveBeenCalledTimes(1);
    });

    fireEvent.change(screen.getByLabelText("Quality"), { target: { value: "medium" } });
    fireEvent.change(screen.getByLabelText("Eval Backend"), { target: { value: "cuda" } });
    fireEvent.change(screen.getByLabelText("Mesh Backend"), { target: { value: "cuda" } });
    fireEvent.change(screen.getByLabelText("Meshing Mode"), { target: { value: "adaptive" } });

    expect(previewField).toHaveBeenCalledTimes(1);
    expect(previewMesh).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByRole("button", { name: "Generate Shape" }));
    await waitFor(() => {
      expect(previewMesh).toHaveBeenCalledTimes(2);
    });

    expect(previewField).toHaveBeenCalledTimes(2);
    expect(compileScene).toHaveBeenCalledTimes(1);
  });

  it("does not preview when Generate Shape compile fails", async () => {
    compileScene.mockRejectedValueOnce(new Error("Compile busted"));

    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "Generate Shape" }));

    await waitFor(() => {
      expect(compileScene).toHaveBeenCalledTimes(1);
    });

    expect(previewField).not.toHaveBeenCalled();
    expect(previewMesh).not.toHaveBeenCalled();
    expect(await screen.findByText("Compile busted")).toBeInTheDocument();
  });

  it("saves and loads field expressions", () => {
    render(<App />);

    const editor = screen.getByLabelText("DSL source editor");
    const nameInput = screen.getByLabelText("Field expression name");
    const savedSelect = screen.getByLabelText("Saved field expressions");

    const savedSource = "root = sphere(r=1.2)";
    fireEvent.change(editor, { target: { value: savedSource } });
    fireEvent.change(nameInput, { target: { value: "SphereExpr" } });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));

    fireEvent.change(editor, { target: { value: "root = box(size=1.0)" } });
    fireEvent.change(savedSelect, { target: { value: "SphereExpr" } });
    fireEvent.click(screen.getByRole("button", { name: "Load" }));

    expect(editor).toHaveValue(savedSource);
  });

  it("does not auto-preview uploaded field when lattice infill parameters change", async () => {
    render(<App />);
    fireEvent.click(screen.getByRole("tab", { name: "Lattice Infill" }));

    commitUploadedMesh.mockClear();
    previewUploadedMeshField.mockClear();

    const file = new File(["v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"], "test.obj", { type: "text/plain" });
    const fileInput = screen.getByLabelText("Lattice infill file upload") as HTMLInputElement;
    fireEvent.change(fileInput, { target: { files: [file] } });

    // Wait for preprocess to complete before checking param-change behavior
    await waitFor(() => expect(preprocessUploadedMesh).toHaveBeenCalledTimes(1));

    expect(previewUploadedMeshField).not.toHaveBeenCalled();
    expect(commitUploadedMesh).not.toHaveBeenCalled();

    fireEvent.change(screen.getByLabelText("Unit cell size (mm)"), { target: { value: "0.55" } });
    fireEvent.change(screen.getByLabelText("Lattice half-thickness (mm)"), { target: { value: "0.12" } });
    fireEvent.change(screen.getByLabelText("Lattice phase"), { target: { value: "0.35" } });
    fireEvent.change(screen.getByLabelText("Shell thickness (mm)"), { target: { value: "0.1" } });
    fireEvent.change(screen.getByLabelText("Lattice infill field backend"), { target: { value: "cuda" } });
    fireEvent.change(screen.getByLabelText("Voxels per lattice period"), { target: { value: "8" } });

    expect(previewUploadedMeshField).not.toHaveBeenCalled();
  });

  it("enables commit only after Generate Field succeeds and disables it again on field changes", async () => {
    render(<App />);
    fireEvent.click(screen.getByRole("tab", { name: "Lattice Infill" }));

    const file = new File(["v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"], "test.obj", { type: "text/plain" });
    fireEvent.change(screen.getByLabelText("Lattice infill file upload"), { target: { files: [file] } });

    await waitFor(() => expect(preprocessUploadedMesh).toHaveBeenCalledTimes(1));

    const commitButton = screen.getByRole("button", { name: "Generate Final Mesh" });
    expect(commitButton).toBeDisabled();

    fireEvent.click(screen.getByRole("button", { name: "Generate Field" }));
    await waitFor(() => expect(previewUploadedMeshField).toHaveBeenCalledTimes(1));
    expect(commitButton).not.toBeDisabled();

    fireEvent.change(screen.getByLabelText("Lattice infill field backend"), { target: { value: "cuda" } });
    expect(commitButton).toBeDisabled();

    fireEvent.click(screen.getByRole("button", { name: "Generate Field" }));
    await waitFor(() => expect(previewUploadedMeshField).toHaveBeenCalledTimes(2));
    expect(commitButton).not.toBeDisabled();
  });

  it("removes uploaded mesh quality controls while keeping DSL quality", () => {
    render(<App />);

    expect(screen.getByLabelText("Quality")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("tab", { name: "Lattice Infill" }));

    expect(screen.queryByLabelText("Lattice infill preview quality")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Lattice infill export quality")).not.toBeInTheDocument();
    expect(screen.getByLabelText("Voxels per lattice period")).toBeInTheDocument();
  });

  it("hides mesh memory estimate until preprocess returns backend memory context", () => {
    render(<App />);
    fireEvent.click(screen.getByRole("tab", { name: "Lattice Infill" }));
    expect(screen.getByText("Memory estimate will appear after mesh preprocess completes.")).toBeInTheDocument();
  });

  it("shows mesh memory estimate after preprocess", async () => {
    render(<App />);
    fireEvent.click(screen.getByRole("tab", { name: "Lattice Infill" }));

    const file = new File(["v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"], "test.obj", { type: "text/plain" });
    fireEvent.change(screen.getByLabelText("Lattice infill file upload"), { target: { files: [file] } });

    expect(await screen.findByText(/Est\. resolution for uploaded mesh:/i)).toBeInTheDocument();
    expect(screen.getByText(/Est\. required CPU memory:/i)).toBeInTheDocument();
  });

  it("shows mm-scale thickness guidance for the lattice infill workflow", () => {
    render(<App />);
    fireEvent.click(screen.getByRole("tab", { name: "Lattice Infill" }));

    expect(screen.getByText(/Min recommended shell thickness at current settings:/i)).toBeInTheDocument();
    expect(
      screen.queryAllByText((_, element) =>
        element?.textContent?.includes("Strut total width = 2 x half-thickness = 1.00 mm") ?? false
      ).length
    ).toBeGreaterThan(0);
    expect(screen.getByText(/Lattice half-thickness 0\.50 mm is below the minimum/i)).toBeInTheDocument();
  });

  it("warns and blocks lattice infill preview actions when memory estimate is fatal", async () => {
    preprocessUploadedMesh.mockResolvedValueOnce({
      mesh: outerMeshPayload,
      memoryContext: {
        meshSpan: 50,
        availableCpuBytes: 256 * 1024 * 1024,
        availableGpuFreeBytes: 256 * 1024 * 1024,
        availableGpuTotalBytes: 1024 * 1024 * 1024,
        cpuBytesPerVoxel: 32,
        gpuBytesPerVoxel: 40,
        safetyFactor: 1.25
      }
    });

    render(<App />);
    fireEvent.click(screen.getByRole("tab", { name: "Lattice Infill" }));
    const file = new File(["v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"], "test.obj", { type: "text/plain" });
    fireEvent.change(screen.getByLabelText("Lattice infill file upload"), { target: { files: [file] } });

    fireEvent.change(screen.getByLabelText("Unit cell size (mm)"), { target: { value: "0.5" } });

    expect(await screen.findByText(/Estimated memory exceeds available capacity/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Generate Field" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Generate Final Mesh" })).toBeDisabled();
  });

  it("re-enables lattice infill preview actions after reducing memory demand", async () => {
    preprocessUploadedMesh.mockResolvedValueOnce({
      mesh: outerMeshPayload,
      memoryContext: {
        meshSpan: 50,
        availableCpuBytes: 256 * 1024 * 1024,
        availableGpuFreeBytes: 256 * 1024 * 1024,
        availableGpuTotalBytes: 1024 * 1024 * 1024,
        cpuBytesPerVoxel: 32,
        gpuBytesPerVoxel: 40,
        safetyFactor: 1.25
      }
    });

    render(<App />);
    fireEvent.click(screen.getByRole("tab", { name: "Lattice Infill" }));
    const file = new File(["v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"], "test.obj", { type: "text/plain" });
    fireEvent.change(screen.getByLabelText("Lattice infill file upload"), { target: { files: [file] } });
    fireEvent.change(screen.getByLabelText("Unit cell size (mm)"), { target: { value: "0.5" } });
    await screen.findByText(/Estimated memory exceeds available capacity/i);

    fireEvent.change(screen.getByLabelText("Unit cell size (mm)"), { target: { value: "20" } });

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Generate Field" })).not.toBeDisabled();
    });
    const commitButton = screen.getByRole("button", { name: "Generate Final Mesh" });
    expect(commitButton).toBeDisabled();

    fireEvent.click(screen.getByRole("button", { name: "Generate Field" }));
    await waitFor(() => {
      expect(previewUploadedMeshField).toHaveBeenCalledTimes(1);
    });
    expect(commitButton).not.toBeDisabled();
  });

  it("runs uploaded field preview immediately when Generate Field is clicked", async () => {
    render(<App />);
    fireEvent.click(screen.getByRole("tab", { name: "Lattice Infill" }));

    previewUploadedMeshField.mockClear();

    const file = new File(["v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"], "test.obj", { type: "text/plain" });
    const fileInput = screen.getByLabelText("Lattice infill file upload") as HTMLInputElement;
    fireEvent.change(fileInput, { target: { files: [file] } });

    fireEvent.click(screen.getByRole("button", { name: "Generate Field" }));

    await waitFor(() => {
      expect(previewUploadedMeshField).toHaveBeenCalledTimes(1);
    });

    expect(previewUploadedMeshField).toHaveBeenCalledTimes(1);
    expect(previewUploadedMeshField.mock.calls[0]?.[1]).toMatchObject({
      latticePitch: 5.0,
      latticeThickness: 0.5,
      latticePhase: 0
    });
  });

  it("shows a warning when uploaded field preview falls back from GPU memory pressure", async () => {
    previewUploadedMeshField.mockResolvedValueOnce({
      ...fieldPreviewPayload,
      stats: {
        ...fieldPreviewPayload.stats,
        fallback_reason:
          "GPU voxel fill ran out of memory at resolution 192. Try lowering the preview resolution, or reduce voxels_per_lattice_period / increase lattice_pitch. The preview was retried on CPU."
      }
    });

    render(<App />);
    fireEvent.click(screen.getByRole("tab", { name: "Lattice Infill" }));

    const file = new File(["v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"], "test.obj", { type: "text/plain" });
    fireEvent.change(screen.getByLabelText("Lattice infill file upload"), { target: { files: [file] } });
    fireEvent.click(screen.getByRole("button", { name: "Generate Field" }));

    expect(
      await screen.findByText(/GPU voxel fill ran out of memory at resolution 192/i)
    ).toBeInTheDocument();
  });

  it("shows separate field and mesh cache status for cached field previews", async () => {
    previewUploadedMeshField.mockResolvedValueOnce({
      ...fieldPreviewPayload,
      stats: {
        ...fieldPreviewPayload.stats,
        eval_ms: 0,
        field_cache_hit: true,
        mesh_cache_hit: false
      }
    });

    render(<App />);
    fireEvent.click(screen.getByRole("tab", { name: "Lattice Infill" }));

    const file = new File(["v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"], "test.obj", { type: "text/plain" });
    fireEvent.change(screen.getByLabelText("Lattice infill file upload"), { target: { files: [file] } });
    fireEvent.click(screen.getByRole("button", { name: "Generate Field" }));

    await waitFor(() => {
      expect(screen.getByText("Eval: 0.0 ms")).toBeInTheDocument();
    });
    expect(screen.getByText("Field cache: hit")).toBeInTheDocument();
    expect(screen.getByText("Mesh cache: miss")).toBeInTheDocument();
  });

  it("posts uploaded field preview telemetry after the first visible field frame", async () => {
    vi.spyOn(performance, "now").mockReturnValue(100);
    previewUploadedMeshField.mockResolvedValueOnce({
      ...fieldPreviewPayload,
      trace: {
        traceId: "trace-123",
        clientResponseWaitMs: 11,
        clientDownloadMs: 22,
        clientDecodeMs: 33,
        fieldAssignedAtMs: 100
      }
    });

    render(<App />);
    fireEvent.click(screen.getByRole("tab", { name: "Lattice Infill" }));

    const file = new File(["v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"], "test.obj", { type: "text/plain" });
    fireEvent.change(screen.getByLabelText("Lattice infill file upload"), { target: { files: [file] } });
    fireEvent.click(screen.getByRole("button", { name: "Generate Field" }));

    await waitFor(() => {
      expect(previewUploadedMeshField).toHaveBeenCalledTimes(1);
    });

    const latestViewerProps = viewerMock.mock.calls[viewerMock.mock.calls.length - 1]?.[0] as Record<string, unknown>;
    const onUploadedFieldPreviewVisible = latestViewerProps.onUploadedFieldPreviewVisible as
      | ((payload: { traceId: string; textureReadyAtMs: number; firstVisibleFrameAtMs: number }) => void)
      | undefined;
    expect(onUploadedFieldPreviewVisible).toBeDefined();

    onUploadedFieldPreviewVisible?.({
      traceId: "trace-123",
      textureReadyAtMs: 120,
      firstVisibleFrameAtMs: 140
    });

    await waitFor(() => {
      expect(submitUploadedFieldPreviewTelemetry).toHaveBeenCalledWith({
        trace_id: "trace-123",
        client_response_wait_ms: 11,
        client_download_ms: 22,
        client_decode_ms: 33,
        client_texture_upload_and_first_frame_ms: 40,
        client_total_visible_ms: 106
      });
    });
  });

  it("clears the previous preview immediately when Generate Field is clicked", async () => {
    let resolvePreview: ((value: typeof fieldPreviewPayload) => void) | null = null;
    previewUploadedMeshField.mockImplementationOnce(
      () =>
        new Promise<typeof fieldPreviewPayload>((resolve) => {
          resolvePreview = resolve;
        })
    );

    render(<App />);
    fireEvent.click(screen.getByRole("tab", { name: "Lattice Infill" }));

    const file = new File(["v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"], "test.obj", { type: "text/plain" });
    const fileInput = screen.getByLabelText("Lattice infill file upload") as HTMLInputElement;
    fireEvent.change(fileInput, { target: { files: [file] } });

    fireEvent.click(screen.getByRole("button", { name: "Generate Field" }));
    await waitFor(() => {
      expect(previewUploadedMeshField).toHaveBeenCalledTimes(1);
    });

    const clearedViewerCall = await waitFor(() => {
      const found = viewerMock.mock.calls.find(([props]) => {
        const viewerProps = props as Record<string, unknown>;
        return viewerProps.field == null && viewerProps.mesh == null;
      });
      expect(found).toBeDefined();
      return found;
    });
    const clearedViewerProps = clearedViewerCall?.[0] as Record<string, unknown>;
    expect(clearedViewerProps.field).toBeNull();
    expect(clearedViewerProps.mesh).toBeNull();

    expect(resolvePreview).not.toBeNull();
    resolvePreview!(fieldPreviewPayload);

    await waitFor(() => {
      const latestViewerProps = viewerMock.mock.calls[viewerMock.mock.calls.length - 1]?.[0] as Record<string, unknown>;
      expect(latestViewerProps.field).toEqual(fieldPreviewPayload.field);
      expect(latestViewerProps.mesh).toBeNull();
    });
  });

  it("clears the mesh preview after parameter changes and requires re-preview", async () => {
    render(<App />);
    fireEvent.click(screen.getByRole("tab", { name: "Lattice Infill" }));

    const file = new File(["v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"], "test.obj", { type: "text/plain" });
    const fileInput = screen.getByLabelText("Lattice infill file upload") as HTMLInputElement;
    fireEvent.change(fileInput, { target: { files: [file] } });

    fireEvent.click(screen.getByRole("button", { name: "Generate Field" }));
    await waitFor(() => {
      expect(previewUploadedMeshField).toHaveBeenCalledTimes(1);
    });

    fireEvent.change(screen.getByLabelText("Lattice phase"), { target: { value: "0.35" } });
    expect(previewUploadedMeshField).toHaveBeenCalledTimes(1);
    const latestViewerProps = viewerMock.mock.calls[viewerMock.mock.calls.length - 1]?.[0] as Record<string, unknown>;
    expect(latestViewerProps.field).toBeNull();
    expect(latestViewerProps.mesh).toBeNull();
  });

  it("does not pass transparentShell to Viewer for field previews", async () => {
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "Generate Shape" }));

    await waitFor(() => {
      expect(previewField).toHaveBeenCalledTimes(1);
    });

    const fieldOnlyCall = viewerMock.mock.calls.find(([props]) => {
      const viewerProps = props as Record<string, unknown>;
      return viewerProps.field != null && viewerProps.mesh == null;
    });

    expect(fieldOnlyCall).toBeDefined();
    const fieldOnlyProps = fieldOnlyCall?.[0] as Record<string, unknown>;
    expect("transparentShell" in fieldOnlyProps).toBe(false);
  });

  it("calls preprocessUploadedMesh and shows outer mesh after file selection", async () => {
    render(<App />);
    fireEvent.click(screen.getByRole("tab", { name: "Lattice Infill" }));

    const file = new File(["v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"], "test.obj", { type: "text/plain" });
    const fileInput = screen.getByLabelText("Lattice infill file upload") as HTMLInputElement;
    fireEvent.change(fileInput, { target: { files: [file] } });

    // preprocessUploadedMesh should be called immediately on file selection
    await waitFor(() => {
      expect(preprocessUploadedMesh).toHaveBeenCalledTimes(1);
    });

    // The outer mesh returned by preprocess should be displayed in the viewer
    await waitFor(() => {
      const latestViewerProps = viewerMock.mock.calls[viewerMock.mock.calls.length - 1]?.[0] as Record<string, unknown>;
      expect(latestViewerProps.mesh).not.toBeNull();
    });
  });

  it("shows error but allows Generate Field after preprocess failure", async () => {
    preprocessUploadedMesh.mockRejectedValueOnce(new Error("Preprocess failed: invalid mesh"));

    render(<App />);
    fireEvent.click(screen.getByRole("tab", { name: "Lattice Infill" }));

    const file = new File(["v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"], "test.obj", { type: "text/plain" });
    fireEvent.change(screen.getByLabelText("Lattice infill file upload"), { target: { files: [file] } });

    await waitFor(() => {
      expect(screen.getByText("Preprocess failed: invalid mesh")).toBeInTheDocument();
    });

    // User can still click Generate Field — it will run the full pipeline inline
    previewUploadedMeshField.mockClear();
    fireEvent.click(screen.getByRole("button", { name: "Generate Field" }));
    await waitFor(() => {
      expect(previewUploadedMeshField).toHaveBeenCalledTimes(1);
    });
  });

  it("commits mesh only when commit button is clicked", async () => {
    render(<App />);
    fireEvent.click(screen.getByRole("tab", { name: "Lattice Infill" }));
    commitUploadedMesh.mockClear();

    const file = new File(["v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"], "test.obj", { type: "text/plain" });
    const fileInput = screen.getByLabelText("Lattice infill file upload") as HTMLInputElement;
    fireEvent.change(fileInput, { target: { files: [file] } });
    fireEvent.click(screen.getByRole("button", { name: "Generate Field" }));
    await waitFor(() => {
      expect(previewUploadedMeshField).toHaveBeenCalledTimes(1);
    });

    fireEvent.click(screen.getByRole("button", { name: "Generate Final Mesh" }));
    await waitFor(() => {
      expect(commitUploadedMesh).toHaveBeenCalled();
    });
  });

  it("sends selected lattice infill backends on commit", async () => {
    render(<App />);
    fireEvent.click(screen.getByRole("tab", { name: "Lattice Infill" }));
    commitUploadedMesh.mockClear();

    const file = new File(["v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"], "test.obj", { type: "text/plain" });
    const fileInput = screen.getByLabelText("Lattice infill file upload") as HTMLInputElement;
    fireEvent.change(fileInput, { target: { files: [file] } });
    fireEvent.click(screen.getByRole("button", { name: "Generate Field" }));
    await waitFor(() => {
      expect(previewUploadedMeshField).toHaveBeenCalledTimes(1);
    });
    fireEvent.change(screen.getByLabelText("Lattice infill field backend"), { target: { value: "cuda" } });
    fireEvent.change(screen.getByLabelText("Lattice infill mesher backend"), { target: { value: "cuda" } });

    expect(screen.getByRole("button", { name: "Generate Final Mesh" })).toBeDisabled();

    fireEvent.click(screen.getByRole("button", { name: "Generate Field" }));
    await waitFor(() => {
      expect(previewUploadedMeshField).toHaveBeenCalledTimes(2);
    });

    fireEvent.click(screen.getByRole("button", { name: "Generate Final Mesh" }));
    await waitFor(() => {
      expect(commitUploadedMesh).toHaveBeenCalled();
    });

    const calls = commitUploadedMesh.mock.calls;
    const lastCall = calls[calls.length - 1];
    expect(lastCall?.[2]).toBe("cuda");
    expect(lastCall?.[3]).toBe("cuda");
  });

  it("sends selected meshing mode for lattice infill commit", async () => {
    render(<App />);
    fireEvent.click(screen.getByRole("tab", { name: "Lattice Infill" }));
    commitUploadedMesh.mockClear();

    const file = new File(["v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"], "test.obj", { type: "text/plain" });
    const fileInput = screen.getByLabelText("Lattice infill file upload") as HTMLInputElement;
    fireEvent.change(fileInput, { target: { files: [file] } });
    fireEvent.click(screen.getByRole("button", { name: "Generate Field" }));
    await waitFor(() => {
      expect(previewUploadedMeshField).toHaveBeenCalledTimes(1);
    });
    fireEvent.change(screen.getByLabelText("Lattice infill meshing mode"), { target: { value: "adaptive" } });

    fireEvent.click(screen.getByRole("button", { name: "Generate Final Mesh" }));
    await waitFor(() => {
      expect(commitUploadedMesh).toHaveBeenCalled();
    });

    const calls = commitUploadedMesh.mock.calls;
    const lastCall = calls[calls.length - 1];
    expect(lastCall?.[4]).toBe("adaptive");
  });

  it("does not pass a quality argument in lattice infill preview or commit calls", async () => {
    render(<App />);
    fireEvent.click(screen.getByRole("tab", { name: "Lattice Infill" }));
    previewUploadedMeshField.mockClear();
    commitUploadedMesh.mockClear();

    const file = new File(["v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"], "test.obj", { type: "text/plain" });
    const fileInput = screen.getByLabelText("Lattice infill file upload") as HTMLInputElement;
    fireEvent.change(fileInput, { target: { files: [file] } });

    fireEvent.click(screen.getByRole("button", { name: "Generate Field" }));
    await waitFor(() => {
      expect(previewUploadedMeshField).toHaveBeenCalled();
    });

    fireEvent.click(screen.getByRole("button", { name: "Generate Final Mesh" }));
    await waitFor(() => {
      expect(commitUploadedMesh).toHaveBeenCalled();
    });

    expect(previewUploadedMeshField.mock.calls[0]).toHaveLength(5);
    expect(commitUploadedMesh.mock.calls[0]).toHaveLength(6);
  });

  it("disables mesh export until commit completes and resets on input changes", async () => {
    render(<App />);
    fireEvent.click(screen.getByRole("tab", { name: "Lattice Infill" }));

    const file = new File(["v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"], "test.obj", { type: "text/plain" });
    const fileInput = screen.getByLabelText("Lattice infill file upload") as HTMLInputElement;
    fireEvent.change(fileInput, { target: { files: [file] } });
    fireEvent.click(screen.getByRole("button", { name: "Generate Field" }));
    await waitFor(() => {
      expect(previewUploadedMeshField).toHaveBeenCalledTimes(1);
    });

    const exportStl = screen.getByRole("button", { name: "Export STL" });
    expect(exportStl).toBeDisabled();

    fireEvent.click(screen.getByRole("button", { name: "Generate Final Mesh" }));
    await waitFor(() => {
      expect(commitUploadedMesh).toHaveBeenCalled();
    });

    expect(exportStl).not.toBeDisabled();
    fireEvent.click(exportStl);

    await waitFor(() => {
      expect(exportUploadedMesh).toHaveBeenCalled();
    });

    const calls = exportUploadedMesh.mock.calls;
    const lastCall = calls[calls.length - 1];
    expect(lastCall?.[2]).toBe("stl");
    expect(lastCall?.[3]).toBe("auto");

    fireEvent.change(screen.getByLabelText("Unit cell size (mm)"), { target: { value: "0.55" } });
    expect(exportStl).toBeDisabled();
  });

  it("shows separate field and mesh cache status for cached uploaded mesh commits", async () => {
    commitUploadedMesh.mockResolvedValueOnce({
      ...meshPreviewPayload,
      stats: {
        ...meshPreviewPayload.stats,
        eval_ms: 0,
        mesh_ms: 0,
        field_cache_hit: true,
        mesh_cache_hit: true
      }
    });

    render(<App />);
    fireEvent.click(screen.getByRole("tab", { name: "Lattice Infill" }));

    const file = new File(["v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"], "test.obj", { type: "text/plain" });
    fireEvent.change(screen.getByLabelText("Lattice infill file upload"), { target: { files: [file] } });
    fireEvent.click(screen.getByRole("button", { name: "Generate Field" }));
    await waitFor(() => {
      expect(previewUploadedMeshField).toHaveBeenCalledTimes(1);
    });
    fireEvent.click(screen.getByRole("button", { name: "Generate Final Mesh" }));

    await waitFor(() => {
      expect(screen.getByText("Mesh: 0.0 ms")).toBeInTheDocument();
    });
    expect(screen.getByText("Eval: 0.0 ms")).toBeInTheDocument();
    expect(screen.getByText("Field cache: hit")).toBeInTheDocument();
    expect(screen.getByText("Mesh cache: hit")).toBeInTheDocument();
  });
});
