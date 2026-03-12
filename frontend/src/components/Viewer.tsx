import { OrbitControls, TransformControls } from "@react-three/drei";
import { Canvas, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef, useState } from "react";
import type { RefObject } from "react";
import * as THREE from "three";

import { FieldPayload, MeshPayload } from "../types";

export interface ViewerProps {
  mesh: MeshPayload | null;
  field: FieldPayload | null;
  wireframe: boolean;
  transformMode: "translate" | "rotate" | "scale";
  fitSignal: number;
  showAxes: boolean;
  showGrid: boolean;
  sectionEnabled: boolean;
  sectionLevel: number;
}

const FIELD_VERTEX_SHADER = `
  out vec3 vWorldPos;
  void main() {
    vec4 worldPos = modelMatrix * vec4(position, 1.0);
    vWorldPos = worldPos.xyz;
    gl_Position = projectionMatrix * viewMatrix * worldPos;
  }
`;

const FIELD_FRAGMENT_SHADER = `
  precision highp float;
  precision highp sampler3D;

  uniform sampler3D uField;
  uniform vec3 uBoundsMin;
  uniform vec3 uBoundsMax;
  uniform vec3 uColor;
  uniform float uSectionEnabled;
  uniform float uSectionLevel;
  uniform float uResolution;
  uniform float uStepScale;

  in vec3 vWorldPos;
  out vec4 outColor;

  vec2 intersectAABB(vec3 ro, vec3 rd, vec3 bmin, vec3 bmax) {
    vec3 t0 = (bmin - ro) / rd;
    vec3 t1 = (bmax - ro) / rd;
    vec3 tsmaller = min(t0, t1);
    vec3 tbigger = max(t0, t1);
    float tNear = max(max(tsmaller.x, tsmaller.y), tsmaller.z);
    float tFar = min(min(tbigger.x, tbigger.y), tbigger.z);
    return vec2(tNear, tFar);
  }

  float sampleField(vec3 worldPos) {
    vec3 uvw = (worldPos - uBoundsMin) / (uBoundsMax - uBoundsMin);
    return texture(uField, clamp(uvw, 0.0, 1.0)).r;
  }

  vec3 estimateNormal(vec3 worldPos, float eps) {
    vec3 ex = vec3(eps, 0.0, 0.0);
    vec3 ey = vec3(0.0, eps, 0.0);
    vec3 ez = vec3(0.0, 0.0, eps);
    float dx = sampleField(worldPos + ex) - sampleField(worldPos - ex);
    float dy = sampleField(worldPos + ey) - sampleField(worldPos - ey);
    float dz = sampleField(worldPos + ez) - sampleField(worldPos - ez);
    return normalize(vec3(dx, dy, dz));
  }

  void main() {
    vec3 ro = cameraPosition;
    vec3 rd = normalize(vWorldPos - cameraPosition);

    vec2 hit = intersectAABB(ro, rd, uBoundsMin, uBoundsMax);
    if (hit.x > hit.y) {
      discard;
    }

    float tStart = max(hit.x, 0.0);
    float tEnd = hit.y;
    if (tEnd <= 0.0) {
      discard;
    }

    const int MAX_STEPS = 1536;
    vec3 extent = (uBoundsMax - uBoundsMin);
    float voxel = min(min(extent.x, extent.y), extent.z) / max(uResolution, 2.0);
    float minStep = max(voxel * 0.25, 1e-4);
    float maxStep = max(voxel * 1.25, minStep);
    float hitEps = max(voxel * 0.8, 1e-4);

    float t = tStart;
    float prev = sampleField(ro + rd * tStart);
    bool hitSurface = false;
    vec3 pHit = vec3(0.0);

    for (int i = 0; i < MAX_STEPS; i++) {
      if (t > tEnd) {
        break;
      }
      vec3 p = ro + rd * t;
      float s = sampleField(p);

      if (abs(s) <= hitEps) {
        pHit = p;
        hitSurface = true;
        break;
      }

      bool crossed = (prev > 0.0 && s <= 0.0) || (prev < 0.0 && s >= 0.0);
      if (crossed) {
        float a = prev / (prev - s + 1e-6);
        float prevT = max(t - minStep, tStart);
        pHit = mix(ro + rd * prevT, p, clamp(a, 0.0, 1.0));
        hitSurface = true;
        break;
      }

      float stepSize = clamp(abs(s) * 0.55, minStep, maxStep) * uStepScale;
      prev = s;
      t += stepSize;
    }

    if (!hitSurface) {
      discard;
    }

    if (uSectionEnabled > 0.5 && pHit.y > uSectionLevel) {
      discard;
    }

    float eps = max(voxel * 0.9, 1e-4);
    vec3 normal = estimateNormal(pHit, eps);
    vec3 lightDir = normalize(vec3(0.55, 0.75, 0.4));
    float diff = max(dot(normal, lightDir), 0.0);
    float hemi = 0.4 + 0.6 * (normal.y * 0.5 + 0.5);
    vec3 lit = uColor * (0.28 + 0.72 * diff) * hemi;

    outColor = vec4(lit, 1.0);
  }
`;

const SECTION_CAP_VERTEX_SHADER = `
  out vec3 vWorldPos;
  void main() {
    vec4 worldPos = modelMatrix * vec4(position, 1.0);
    vWorldPos = worldPos.xyz;
    gl_Position = projectionMatrix * viewMatrix * worldPos;
  }
`;

const SECTION_CAP_FRAGMENT_SHADER = `
  precision highp float;
  precision highp sampler3D;

  uniform sampler3D uField;
  uniform vec3 uBoundsMin;
  uniform vec3 uBoundsMax;
  uniform vec3 uColor;

  in vec3 vWorldPos;
  out vec4 outColor;

  void main() {
    vec3 uvw = (vWorldPos - uBoundsMin) / (uBoundsMax - uBoundsMin);
    if (
      any(lessThan(uvw, vec3(0.0))) ||
      any(greaterThan(uvw, vec3(1.0)))
    ) {
      discard;
    }

    float sdf = texture(uField, clamp(uvw, 0.0, 1.0)).r;
    if (sdf > 0.0) {
      discard;
    }

    outColor = vec4(uColor, 1.0);
  }
`;

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

function decodeFloat32Base64(base64: string): Float32Array {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  const outBuffer = new ArrayBuffer(bytes.byteLength);
  new Uint8Array(outBuffer).set(bytes);
  return new Float32Array(outBuffer);
}

function toFieldTexture(field: FieldPayload | null): THREE.Data3DTexture | null {
  if (!field || field.encoding !== "f32-base64") {
    return null;
  }

  const values = decodeFloat32Base64(field.data);
  const expected = field.resolution * field.resolution * field.resolution;
  if (values.length !== expected) {
    return null;
  }

  const texture = new THREE.Data3DTexture(
    values as unknown as BufferSource,
    field.resolution,
    field.resolution,
    field.resolution
  );
  texture.internalFormat = "R32F";
  texture.format = THREE.RedFormat;
  texture.type = THREE.FloatType;
  // Use nearest filtering for broad WebGL2 compatibility with float 3D textures.
  texture.minFilter = THREE.NearestFilter;
  texture.magFilter = THREE.NearestFilter;
  texture.wrapS = THREE.ClampToEdgeWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.wrapR = THREE.ClampToEdgeWrapping;
  texture.generateMipmaps = false;
  texture.unpackAlignment = 1;
  texture.needsUpdate = true;
  return texture;
}

function FitCamera({
  orbitRef,
  fitSignal,
  targetBox
}: {
  orbitRef: RefObject<any>;
  fitSignal: number;
  targetBox: THREE.Box3 | null;
}) {
  const { camera } = useThree();

  useEffect(() => {
    if (!targetBox || targetBox.isEmpty()) {
      return;
    }

    const center = targetBox.getCenter(new THREE.Vector3());
    const size = targetBox.getSize(new THREE.Vector3());
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
  }, [camera, fitSignal, orbitRef, targetBox]);

  return null;
}

export function Viewer({
  mesh,
  field,
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

  const fieldTexture = useMemo(() => toFieldTexture(field), [field]);
  const fieldBounds = useMemo(() => {
    if (!field) {
      return null;
    }
    const min = new THREE.Vector3(field.bounds[0][0], field.bounds[1][0], field.bounds[2][0]);
    const max = new THREE.Vector3(field.bounds[0][1], field.bounds[1][1], field.bounds[2][1]);
    const center = min.clone().add(max).multiplyScalar(0.5);
    const size = max.clone().sub(min);
    return { min, max, center, size };
  }, [field]);

  const hasField = fieldTexture !== null && fieldBounds !== null;

  const targetBox = useMemo(() => {
    if (fieldBounds) {
      return new THREE.Box3(fieldBounds.min.clone(), fieldBounds.max.clone());
    }
    if (geometry?.boundingBox) {
      return geometry.boundingBox.clone();
    }
    return null;
  }, [fieldBounds, geometry]);

  useEffect(() => {
    return () => {
      if (geometry) {
        geometry.dispose();
      }
    };
  }, [geometry]);

  useEffect(() => {
    return () => {
      if (fieldTexture) {
        fieldTexture.dispose();
      }
    };
  }, [fieldTexture]);

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

      {hasField && fieldBounds && !geometry ? (
        <mesh ref={meshRef} position={fieldBounds.center.toArray() as [number, number, number]} renderOrder={3}>
          <boxGeometry args={fieldBounds.size.toArray() as [number, number, number]} />
          <shaderMaterial
            side={THREE.FrontSide}
            transparent={false}
            glslVersion={THREE.GLSL3}
            vertexShader={FIELD_VERTEX_SHADER}
            fragmentShader={FIELD_FRAGMENT_SHADER}
            toneMapped={false}
            uniforms={{
              uField: { value: fieldTexture },
              uBoundsMin: { value: fieldBounds.min },
              uBoundsMax: { value: fieldBounds.max },
              uColor: { value: new THREE.Color("#8be9fd") },
              uSectionEnabled: { value: sectionEnabled ? 1.0 : 0.0 },
              uSectionLevel: { value: sectionLevel },
              uResolution: { value: field?.resolution ?? 64 },
              uStepScale: { value: 1.0 }
            }}
          />
        </mesh>
      ) : null}

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

      {hasField && !geometry && sectionEnabled && fieldBounds ? (
        <mesh position={[0, sectionLevel, 0]} rotation={[-Math.PI * 0.5, 0, 0]} renderOrder={2}>
          <planeGeometry args={[capSize, capSize]} />
          <shaderMaterial
            glslVersion={THREE.GLSL3}
            vertexShader={SECTION_CAP_VERTEX_SHADER}
            fragmentShader={SECTION_CAP_FRAGMENT_SHADER}
            uniforms={{
              uField: { value: fieldTexture },
              uBoundsMin: { value: fieldBounds.min },
              uBoundsMax: { value: fieldBounds.max },
              uColor: { value: new THREE.Color("#ff3b30") }
            }}
            side={THREE.DoubleSide}
            transparent
            depthWrite={false}
            toneMapped={false}
          />
        </mesh>
      ) : null}

      <OrbitControls ref={orbitRef} makeDefault enabled={orbitEnabled} />
      <FitCamera orbitRef={orbitRef} fitSignal={fitSignal} targetBox={targetBox} />
    </Canvas>
  );
}
