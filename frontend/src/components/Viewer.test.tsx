import { render } from "@testing-library/react";
import { forwardRef, useImperativeHandle } from "react";
import type { ComponentProps, ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";
import * as THREE from "three";

import { HARD_EDGE_MESH_MATERIAL_PROPS, Viewer } from "./Viewer";

vi.mock("@react-three/fiber", () => ({
  Canvas: ({ children, onCreated }: { children: ReactNode; onCreated?: (state: any) => void }) => {
    onCreated?.({ gl: { localClippingEnabled: false } });
    return <div data-testid="canvas">{children}</div>;
  },
  useThree: () => ({ camera: new THREE.PerspectiveCamera(45, 1, 0.01, 1000) })
}));

vi.mock("@react-three/drei", () => ({
  TransformControls: ({ children }: { children: ReactNode }) => <>{children}</>,
  OrbitControls: forwardRef((_props, ref) => {
    useImperativeHandle(ref, () => ({
      target: new THREE.Vector3(),
      update: vi.fn()
    }));
    return <div data-testid="orbit-controls" />;
  })
}));

function encodeBuffer(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  for (let i = 0; i < bytes.length; i += 1) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

const sampleMesh = {
  encoding: "mesh-f32-u32-base64-v1" as const,
  vertex_count: 3,
  face_count: 1,
  vertices_b64: encodeBuffer(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]).buffer),
  indices_b64: encodeBuffer(new Uint32Array([0, 1, 2]).buffer),
  normals_b64: encodeBuffer(new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]).buffer)
};

function encodeFieldData(values: Float32Array): string {
  const bytes = new Uint8Array(values.buffer.slice(0));
  let binary = "";
  for (let i = 0; i < bytes.length; i += 1) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

const sampleField = {
  encoding: "f32-base64" as const,
  resolution_xyz: [2, 2, 2] as [number, number, number],
  bounds: [
    [-1, 1],
    [-1, 1],
    [-1, 1]
  ] as [[number, number], [number, number], [number, number]],
  data: encodeFieldData(new Float32Array([-1, -1, -1, -1, -1, -1, -1, -1]))
};

function renderViewer(overrides?: Partial<ComponentProps<typeof Viewer>>) {
  return render(
    <Viewer
      mesh={sampleMesh}
      field={sampleField}
      uploadedMeshPreviewActive={true}
      wireframe={false}
      transformMode="translate"
      fitSignal={0}
      showAxes={false}
      showGrid={false}
      sectionEnabled
      sectionLevel={0.0}
      {...overrides}
    />
  );
}

describe("Viewer section capping", () => {
  it("enables flat shading for lit mesh materials by default", () => {
    expect(HARD_EDGE_MESH_MATERIAL_PROPS.flatShading).toBe(true);
  });

  it("uses layered translucent rendering when uploaded mesh preview is active", () => {
    const { container } = renderViewer();

    expect(container.querySelectorAll("meshbasicmaterial").length).toBe(2);
    expect(container.querySelectorAll("meshstandardmaterial").length).toBe(2);
    expect(container.querySelectorAll("shadermaterial").length).toBe(1);
    expect(container.querySelectorAll("boxgeometry").length).toBe(1);
  });

  it("uses a translucent shader path for field-only rendering", () => {
    const { container } = renderViewer({ mesh: null });
    const fieldMesh = container.querySelector("mesh > boxgeometry");

    expect(fieldMesh).not.toBeNull();
    expect(container.querySelectorAll("shadermaterial").length).toBeGreaterThanOrEqual(2);
    expect(container.querySelectorAll("meshbasicmaterial").length).toBe(0);
    expect(container.querySelectorAll("meshstandardmaterial").length).toBe(0);
  });
});
