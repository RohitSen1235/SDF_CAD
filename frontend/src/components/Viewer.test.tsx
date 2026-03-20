import { render } from "@testing-library/react";
import { forwardRef, useImperativeHandle } from "react";
import type { ComponentProps, ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";
import * as THREE from "three";

import { Viewer } from "./Viewer";

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
  resolution: 2,
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
  it("uses mesh stencil cap when mesh and field are both present", () => {
    const { container } = renderViewer();

    expect(container.querySelectorAll("meshbasicmaterial").length).toBe(2);
    expect(container.querySelectorAll("meshstandardmaterial").length).toBe(2);
    expect(container.querySelectorAll("shadermaterial").length).toBe(0);
  });

  it("keeps shader section cap for field-only rendering", () => {
    const { container } = renderViewer({ mesh: null });

    expect(container.querySelectorAll("shadermaterial").length).toBeGreaterThanOrEqual(2);
    expect(container.querySelectorAll("meshbasicmaterial").length).toBe(0);
    expect(container.querySelectorAll("meshstandardmaterial").length).toBe(0);
  });
});
