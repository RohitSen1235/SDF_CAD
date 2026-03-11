import { OrbitControls, TransformControls } from "@react-three/drei";
import { Canvas, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef, useState } from "react";
import type { RefObject } from "react";
import * as THREE from "three";

import { MeshPayload } from "../types";

export interface ViewerProps {
  mesh: MeshPayload | null;
  wireframe: boolean;
  transformMode: "translate" | "rotate" | "scale";
  fitSignal: number;
  showAxes: boolean;
  showGrid: boolean;
  sectionEnabled: boolean;
  sectionLevel: number;
}

function packVec3(values: [number, number, number][]): Float32Array {
  const out = new Float32Array(values.length * 3);
  for (let i = 0; i < values.length; i += 1) {
    const offset = i * 3;
    const value = values[i];
    out[offset] = value[0];
    out[offset + 1] = value[1];
    out[offset + 2] = value[2];
  }
  return out;
}

function packIndices(
  faces: [number, number, number][],
  vertexCount: number
): Uint16Array | Uint32Array {
  const out =
    vertexCount > 65535 ? new Uint32Array(faces.length * 3) : new Uint16Array(faces.length * 3);
  for (let i = 0; i < faces.length; i += 1) {
    const offset = i * 3;
    const tri = faces[i];
    out[offset] = tri[0];
    out[offset + 1] = tri[1];
    out[offset + 2] = tri[2];
  }
  return out;
}

function toGeometry(mesh: MeshPayload | null): THREE.BufferGeometry | null {
  if (!mesh) {
    return null;
  }

  const geometry = new THREE.BufferGeometry();
  const vertices = packVec3(mesh.vertices);
  const indices = packIndices(mesh.indices, mesh.vertices.length);
  geometry.setAttribute("position", new THREE.BufferAttribute(vertices, 3));
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));

  if (mesh.normals.length === mesh.vertices.length) {
    const normals = packVec3(mesh.normals);
    geometry.setAttribute("normal", new THREE.BufferAttribute(normals, 3));
  } else {
    geometry.computeVertexNormals();
  }

  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  return geometry;
}

function FitCamera({
  meshRef,
  orbitRef,
  fitSignal,
  geometry
}: {
  meshRef: RefObject<THREE.Mesh>;
  orbitRef: RefObject<any>;
  fitSignal: number;
  geometry: THREE.BufferGeometry | null;
}) {
  const { camera } = useThree();

  useEffect(() => {
    if (!geometry || !meshRef.current) {
      return;
    }

    const box = new THREE.Box3().setFromObject(meshRef.current);
    if (box.isEmpty()) {
      return;
    }

    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);
    const perspective = camera as THREE.PerspectiveCamera;

    const fov = (perspective.fov * Math.PI) / 180;
    const distance = (maxDim * 0.5) / Math.tan(fov * 0.5);

    perspective.position.set(
      center.x + distance * 1.4,
      center.y + distance * 0.9,
      center.z + distance * 1.4
    );
    perspective.near = Math.max(distance / 100, 0.01);
    perspective.far = distance * 100;
    perspective.updateProjectionMatrix();

    const orbit = orbitRef.current;
    if (orbit) {
      orbit.target.copy(center);
      orbit.update();
    }
  }, [camera, fitSignal, geometry, meshRef, orbitRef]);

  return null;
}

export function Viewer({
  mesh,
  wireframe,
  transformMode,
  fitSignal,
  showAxes,
  showGrid,
  sectionEnabled,
  sectionLevel
}: ViewerProps) {
  const [orbitEnabled, setOrbitEnabled] = useState(true);
  const orbitRef = useRef<any>(null);
  const meshRef = useRef<THREE.Mesh>(null);

  const geometry = useMemo(() => toGeometry(mesh), [mesh]);
  const sectionPlane = useMemo(() => {
    if (!sectionEnabled) {
      return null;
    }
    return new THREE.Plane(new THREE.Vector3(0, -1, 0), sectionLevel);
  }, [sectionEnabled, sectionLevel]);
  const clippingPlanes = useMemo(() => (sectionPlane ? [sectionPlane] : []), [sectionPlane]);
  const capSize = useMemo(() => {
    if (!geometry?.boundingSphere) {
      return 120;
    }
    return Math.max(120, geometry.boundingSphere.radius * 8);
  }, [geometry]);

  useEffect(() => {
    return () => {
      if (geometry) {
        geometry.dispose();
      }
    };
  }, [geometry]);

  return (
    <Canvas
      camera={{ position: [3.5, 2.5, 3.5], fov: 45 }}
      dpr={[1, 1.75]}
      gl={{ stencil: true }}
      onCreated={({ gl }) => {
        gl.localClippingEnabled = true;
      }}
    >
      <color attach="background" args={["#0d1721"]} />
      <ambientLight intensity={0.35} />
      <hemisphereLight args={["#dbeeff", "#4b5f73", 0.45]} />
      <directionalLight position={[5, 6, 4]} intensity={0.8} />
      <directionalLight position={[-5, -6, -4]} intensity={0.7} />

      {showGrid ? <gridHelper args={[12, 24, "#2f536d", "#1d2f3d"]} /> : null}
      {showAxes ? <axesHelper args={[2.5]} /> : null}

      {geometry ? (
        <TransformControls
          mode={transformMode}
          onMouseDown={() => setOrbitEnabled(false)}
          onMouseUp={() => setOrbitEnabled(true)}
          size={0.75}
        >
          <group>
            {sectionEnabled ? (
              <>
                <mesh geometry={geometry} renderOrder={1}>
                  <meshBasicMaterial
                    colorWrite={false}
                    depthWrite={false}
                    depthTest={false}
                    clippingPlanes={clippingPlanes}
                    side={THREE.BackSide}
                    stencilWrite
                    stencilFunc={THREE.AlwaysStencilFunc}
                    stencilFail={THREE.IncrementWrapStencilOp}
                    stencilZFail={THREE.IncrementWrapStencilOp}
                    stencilZPass={THREE.IncrementWrapStencilOp}
                  />
                </mesh>
                <mesh geometry={geometry} renderOrder={1}>
                  <meshBasicMaterial
                    colorWrite={false}
                    depthWrite={false}
                    depthTest={false}
                    clippingPlanes={clippingPlanes}
                    side={THREE.FrontSide}
                    stencilWrite
                    stencilFunc={THREE.AlwaysStencilFunc}
                    stencilFail={THREE.DecrementWrapStencilOp}
                    stencilZFail={THREE.DecrementWrapStencilOp}
                    stencilZPass={THREE.DecrementWrapStencilOp}
                  />
                </mesh>
              </>
            ) : null}
            <mesh ref={meshRef} geometry={geometry} castShadow receiveShadow renderOrder={3}>
              <meshStandardMaterial
                color="#8be9fd"
                roughness={0.2}
                metalness={0.05}
                wireframe={wireframe}
                clippingPlanes={clippingPlanes}
                clipShadows
              />
            </mesh>
          </group>
        </TransformControls>
      ) : null}

      {geometry && sectionEnabled && sectionPlane ? (
        <mesh position={[0, sectionLevel, 0]} rotation={[-Math.PI * 0.5, 0, 0]} renderOrder={2}>
          <planeGeometry args={[capSize, capSize]} />
          <meshStandardMaterial
            color="#ff3b30"
            roughness={0.5}
            metalness={0.05}
            side={THREE.DoubleSide}
            stencilWrite
            stencilRef={0}
            stencilFunc={THREE.NotEqualStencilFunc}
            stencilFail={THREE.ReplaceStencilOp}
            stencilZFail={THREE.ReplaceStencilOp}
            stencilZPass={THREE.ReplaceStencilOp}
          />
        </mesh>
      ) : null}

      <OrbitControls ref={orbitRef} makeDefault enabled={orbitEnabled} />
      <FitCamera meshRef={meshRef} orbitRef={orbitRef} fitSignal={fitSignal} geometry={geometry} />
    </Canvas>
  );
}
