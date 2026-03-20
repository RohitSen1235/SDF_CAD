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
  /** When true the field renderer uses a ghost outer shell so the interior
   *  TPMS lattice is visible through it. Used by the Mesh workflow. */
  transparentShell?: boolean;
}

const FIELD_VERTEX_SHADER = `
  out vec3 vWorldPos;
  void main() {
    vec4 worldPos = modelMatrix * vec4(position, 1.0);
    vWorldPos = worldPos.xyz;
    gl_Position = projectionMatrix * viewMatrix * worldPos;
  }
`;

// Smooth ray-marched field renderer.
// Uses hardware LinearFilter on the 3D texture for smooth trilinear sampling,
// and a wider normal-estimation epsilon to avoid voxel-grid bumpiness.
//
// When uTransparentShell > 0.5 (mesh workflow), the shader performs multi-hit
// compositing: the outer shell is rendered as a ghost (low alpha) so the
// interior TPMS lattice struts are visible through it.
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
  // When > 0.5: render outer shell as semi-transparent ghost so interior
  // lattice is visible. Used by the Mesh workflow field preview.
  uniform float uTransparentShell;

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

  // Hardware-interpolated field sample — relies on LinearFilter set on the texture.
  // The GPU sampler performs trilinear interpolation natively, giving smooth results.
  float sampleField(vec3 worldPos) {
    vec3 uvw = clamp((worldPos - uBoundsMin) / (uBoundsMax - uBoundsMin), 0.0, 1.0);
    return texture(uField, uvw).r;
  }

  // Normal estimation with a wide epsilon (2x voxel size) to smooth out
  // quantization noise from the discrete SDF grid.
  vec3 estimateNormal(vec3 worldPos, float eps) {
    vec3 ex = vec3(eps, 0.0, 0.0);
    vec3 ey = vec3(0.0, eps, 0.0);
    vec3 ez = vec3(0.0, 0.0, eps);
    float dx = sampleField(worldPos + ex) - sampleField(worldPos - ex);
    float dy = sampleField(worldPos + ey) - sampleField(worldPos - ey);
    float dz = sampleField(worldPos + ez) - sampleField(worldPos - ez);
    return normalize(vec3(dx, dy, dz));
  }

  vec3 shade(vec3 p, float voxel) {
    float normalEps = max(voxel * 2.0, 1e-3);
    vec3 normal = estimateNormal(p, normalEps);
    vec3 lightDir = normalize(vec3(0.55, 0.75, 0.4));
    float diff = max(dot(normal, lightDir), 0.0);
    float hemi = 0.4 + 0.6 * (normal.y * 0.5 + 0.5);
    return uColor * (0.28 + 0.72 * diff) * hemi;
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

    vec3 extent = (uBoundsMax - uBoundsMin);
    float voxel = min(min(extent.x, extent.y), extent.z) / max(uResolution, 2.0);
    float minStep = max(voxel * 0.25, 1e-4);
    float maxStep = max(voxel * 1.25, minStep);
    float hitEps = max(voxel * 0.8, 1e-4);

    // ── Opaque mode (DSL workflow) ────────────────────────────────────────────
    if (uTransparentShell < 0.5) {
      const int MAX_STEPS = 1536;
      float t = tStart;
      float prev = sampleField(ro + rd * tStart);
      bool hitSurface = false;
      vec3 pHit = vec3(0.0);

      for (int i = 0; i < MAX_STEPS; i++) {
        if (t > tEnd) break;
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

      if (!hitSurface) { discard; }
      if (uSectionEnabled > 0.5 && pHit.y > uSectionLevel) { discard; }

      // Use 2x voxel size for normal epsilon — wider neighbourhood averages out
      // grid quantization and produces smooth, non-bumpy shading.
      outColor = vec4(shade(pHit, voxel), 1.0);
      return;
    }

    // ── Transparent-shell mode (Mesh workflow) ────────────────────────────────
    // Multi-hit compositing: collect up to 2 surface crossings.
    // Hit 0 = outer shell → ghost alpha (0.18).
    // Hit 1 = interior lattice strut → full alpha (1.0).
    // This lets the user see the TPMS lattice through the translucent skin.
    const int MAX_STEPS_T = 1536;
    const float SHELL_ALPHA = 0.18;
    // After the first hit we step past the surface by a small skip (relative to
    // voxel size) to avoid re-detecting the same crossing immediately.
    float SKIP_DIST = voxel * 3.0;

    vec4 accum = vec4(0.0);   // accumulated RGBA (pre-multiplied alpha)
    float t = tStart;
    float prev = sampleField(ro + rd * tStart);
    int hitCount = 0;

    for (int i = 0; i < MAX_STEPS_T; i++) {
      if (t > tEnd) break;
      if (accum.a >= 0.99) break;   // fully opaque — early exit

      vec3 p = ro + rd * t;
      float s = sampleField(p);

      bool onSurface = abs(s) <= hitEps;
      bool crossed   = (prev > 0.0 && s <= 0.0) || (prev < 0.0 && s >= 0.0);

      if (onSurface || crossed) {
        vec3 pHit;
        if (onSurface) {
          pHit = p;
        } else {
          float a = prev / (prev - s + 1e-6);
          float prevT = max(t - minStep, tStart);
          pHit = mix(ro + rd * prevT, p, clamp(a, 0.0, 1.0));
        }

        // Section plane clipping — discard the whole ray if the first visible
        // hit is above the section level.
        if (uSectionEnabled > 0.5 && pHit.y > uSectionLevel) {
          // Skip past this surface and keep looking below the cut plane.
          t += SKIP_DIST + minStep;
          prev = sampleField(ro + rd * t);
          hitCount++;
          if (hitCount >= 2) break;
          continue;
        }

        vec3 litColor = shade(pHit, voxel);

        float hitAlpha;
        if (hitCount == 0) {
          // Outer shell — ghost/transparent
          hitAlpha = SHELL_ALPHA;
        } else {
          // Interior lattice — fully opaque
          hitAlpha = 1.0;
        }

        // Alpha-over compositing (front-to-back)
        float remaining = 1.0 - accum.a;
        accum.rgb += litColor * hitAlpha * remaining;
        accum.a   += hitAlpha * remaining;

        hitCount++;
        if (hitCount >= 2) break;

        // Skip past this surface to find the next one.
        t += SKIP_DIST;
        prev = sampleField(ro + rd * t);
        continue;
      }

      float stepSize = clamp(abs(s) * 0.55, minStep, maxStep) * uStepScale;
      prev = s;
      t += stepSize;
    }

    if (accum.a < 0.01) { discard; }
    outColor = accum;
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

function decodeBase64Bytes(base64: string): Uint8Array {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}

function toGeometry(mesh: MeshPayload | null): THREE.BufferGeometry | null {
  if (!mesh || mesh.encoding !== "mesh-f32-u32-base64-v1") {
    return null;
  }

  const expectedVerticesBytes = mesh.vertex_count * 3 * Float32Array.BYTES_PER_ELEMENT;
  const expectedIndicesBytes = mesh.face_count * 3 * Uint32Array.BYTES_PER_ELEMENT;
  const expectedNormalsBytes = mesh.vertex_count * 3 * Float32Array.BYTES_PER_ELEMENT;

  const verticesBytes = decodeBase64Bytes(mesh.vertices_b64);
  const indicesBytes = decodeBase64Bytes(mesh.indices_b64);
  const normalsBytes = decodeBase64Bytes(mesh.normals_b64);

  if (
    verticesBytes.byteLength !== expectedVerticesBytes ||
    indicesBytes.byteLength !== expectedIndicesBytes ||
    normalsBytes.byteLength !== expectedNormalsBytes
  ) {
    return null;
  }

  const vertices = new Float32Array(verticesBytes.buffer.slice(0));
  const indices = new Uint32Array(indicesBytes.buffer.slice(0));
  const normals = new Float32Array(normalsBytes.buffer.slice(0));

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(vertices, 3));
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));
  if (normals.length === vertices.length) {
    geometry.setAttribute("normal", new THREE.BufferAttribute(normals, 3));
  } else {
    geometry.computeVertexNormals();
  }

  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  return geometry;
}

function decodeFloat32Base64(base64: string): Float32Array {
  const bytes = decodeBase64Bytes(base64);
  return new Float32Array(bytes.buffer.slice(0));
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
  // Use LinearFilter for smooth hardware-interpolated trilinear sampling.
  // This eliminates voxel-grid artifacts and produces smooth surface rendering.
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
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
  sectionLevel,
  transparentShell = false
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
            transparent={transparentShell}
            depthWrite={!transparentShell}
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
              uStepScale: { value: 1.0 },
              uTransparentShell: { value: transparentShell ? 1.0 : 0.0 }
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

      {fieldTexture && !geometry && sectionEnabled && fieldBounds ? (
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
