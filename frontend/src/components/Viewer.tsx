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

function toGeometry(mesh: MeshPayload | null): THREE.BufferGeometry | null {
  if (!mesh) {
    return null;
  }

  const geometry = new THREE.BufferGeometry();
  const vertices = new Float32Array(mesh.vertices.flat());
  const indices = mesh.indices.flat();
  geometry.setAttribute("position", new THREE.BufferAttribute(vertices, 3));
  geometry.setIndex(indices);

  if (mesh.normals.length === mesh.vertices.length) {
    const normals = new Float32Array(mesh.normals.flat());
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
  const clippingPlanes = useMemo(() => {
    if (!sectionEnabled) {
      return [];
    }
    return [new THREE.Plane(new THREE.Vector3(0, -1, 0), sectionLevel)];
  }, [sectionEnabled, sectionLevel]);

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
          <mesh ref={meshRef} geometry={geometry} castShadow receiveShadow>
            <meshStandardMaterial
              color="#8be9fd"
              roughness={0.2}
              metalness={0.05}
              wireframe={wireframe}
              clippingPlanes={clippingPlanes}
              clipShadows
            />
          </mesh>
        </TransformControls>
      ) : null}

      <OrbitControls ref={orbitRef} makeDefault enabled={orbitEnabled} />
      <FitCamera meshRef={meshRef} orbitRef={orbitRef} fitSignal={fitSignal} geometry={geometry} />
    </Canvas>
  );
}
