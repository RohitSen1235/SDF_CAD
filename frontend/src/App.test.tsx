import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";

vi.mock("./components/Viewer", () => ({
  Viewer: () => <div data-testid="viewer" />
}));

const compileScene = vi.fn();
const previewField = vi.fn();
const previewMesh = vi.fn();
const exportMesh = vi.fn();
const previewUploadedMesh = vi.fn();
const exportUploadedMesh = vi.fn();

vi.mock("./lib/api", () => ({
  compileScene: (...args: unknown[]) => compileScene(...args),
  previewField: (...args: unknown[]) => previewField(...args),
  previewMesh: (...args: unknown[]) => previewMesh(...args),
  exportMesh: (...args: unknown[]) => exportMesh(...args),
  previewUploadedMesh: (...args: unknown[]) => previewUploadedMesh(...args),
  exportUploadedMesh: (...args: unknown[]) => exportUploadedMesh(...args)
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
    vertices: [
      [0, 0, 0],
      [1, 0, 0],
      [0, 1, 0]
    ],
    indices: [[0, 1, 2]],
    normals: [
      [0, 0, 1],
      [0, 0, 1],
      [0, 0, 1]
    ]
  },
  stats: {
    eval_ms: 3,
    mesh_ms: 4,
    tri_count: 1
  }
};

describe("App", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    window.localStorage.clear();
    compileScene.mockResolvedValue({
      sceneIr: compiledScene,
      diagnostics: { warnings: [], inferred_bounds: [[-1, 1], [-1, 1], [-1, 1]] }
    });
    previewField.mockResolvedValue(fieldPreviewPayload);
    previewMesh.mockResolvedValue(meshPreviewPayload);
    exportMesh.mockResolvedValue(new Blob(["ok"]));
    previewUploadedMesh.mockResolvedValue(meshPreviewPayload);
    exportUploadedMesh.mockResolvedValue(new Blob(["ok"]));
  });

  it("renders DSL workflow with manual compile controls", () => {
    render(<App />);
    expect(screen.getByRole("tab", { name: "DSL" })).toHaveAttribute("aria-selected", "true");
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

  it("calls mesh preview API when Generate is clicked", async () => {
    render(<App />);
    fireEvent.click(screen.getByRole("tab", { name: "Mesh" }));

    previewUploadedMesh.mockClear();

    const file = new File(["v 0 0 0\n"], "test.obj", { type: "text/plain" });
    const fileInput = screen.getByLabelText("Mesh file upload") as HTMLInputElement;
    fireEvent.change(fileInput, { target: { files: [file] } });

    fireEvent.click(screen.getByRole("button", { name: "Generate Mesh Preview" }));

    await waitFor(() => {
      expect(previewUploadedMesh).toHaveBeenCalled();
    });
  });

  it("sends selected mesh workflow backends", async () => {
    render(<App />);
    fireEvent.click(screen.getByRole("tab", { name: "Mesh" }));
    previewUploadedMesh.mockClear();

    const file = new File(["v 0 0 0\n"], "test.obj", { type: "text/plain" });
    const fileInput = screen.getByLabelText("Mesh file upload") as HTMLInputElement;
    fireEvent.change(fileInput, { target: { files: [file] } });
    fireEvent.change(screen.getByLabelText("Mesh field backend"), { target: { value: "cuda" } });
    fireEvent.change(screen.getByLabelText("Mesh mesher backend"), { target: { value: "cuda" } });

    fireEvent.click(screen.getByRole("button", { name: "Generate Mesh Preview" }));
    await waitFor(() => {
      expect(previewUploadedMesh).toHaveBeenCalled();
    });

    const calls = previewUploadedMesh.mock.calls;
    const lastCall = calls[calls.length - 1];
    expect(lastCall?.[3]).toBe("cuda");
    expect(lastCall?.[4]).toBe("cuda");
  });

  it("sends selected meshing mode for mesh workflow preview", async () => {
    render(<App />);
    fireEvent.click(screen.getByRole("tab", { name: "Mesh" }));
    previewUploadedMesh.mockClear();

    const file = new File(["v 0 0 0\n"], "test.obj", { type: "text/plain" });
    const fileInput = screen.getByLabelText("Mesh file upload") as HTMLInputElement;
    fireEvent.change(fileInput, { target: { files: [file] } });
    fireEvent.change(screen.getByLabelText("Mesh meshing mode"), { target: { value: "adaptive" } });

    fireEvent.click(screen.getByRole("button", { name: "Generate Mesh Preview" }));
    await waitFor(() => {
      expect(previewUploadedMesh).toHaveBeenCalled();
    });

    const calls = previewUploadedMesh.mock.calls;
    const lastCall = calls[calls.length - 1];
    expect(lastCall?.[5]).toBe("adaptive");
  });

  it("uses mesh export quality for uploaded export", async () => {
    render(<App />);
    fireEvent.click(screen.getByRole("tab", { name: "Mesh" }));

    const file = new File(["v 0 0 0\n"], "test.obj", { type: "text/plain" });
    const fileInput = screen.getByLabelText("Mesh file upload") as HTMLInputElement;
    fireEvent.change(fileInput, { target: { files: [file] } });

    fireEvent.change(screen.getByLabelText("Mesh export quality"), { target: { value: "ultra" } });
    fireEvent.click(screen.getByRole("button", { name: "Export STL" }));

    await waitFor(() => {
      expect(exportUploadedMesh).toHaveBeenCalled();
    });

    const calls = exportUploadedMesh.mock.calls;
    const lastCall = calls[calls.length - 1];
    expect(lastCall?.[2]).toBe("stl");
    expect(lastCall?.[3]).toBe("ultra");
  });
});
