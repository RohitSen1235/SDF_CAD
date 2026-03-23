import { OrbitControls, TransformControls } from "@react-three/drei";
import { Canvas, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef, useState } from "react";
import type { RefObject } from "react";
import * as THREE from "three";

import { FieldPayload, MeshPayload, UploadedFieldPreviewTrace } from "../types";

export interface ViewerProps {
  mesh: MeshPayload | null;
  field: FieldPayload | null;
  uploadedFieldPreviewTrace?: UploadedFieldPreviewTrace | null;
  onUploadedFieldPreviewVisible?: ((timing: {
    traceId: string;
    textureReadyAtMs: number;
    firstVisibleFrameAtMs: number;
  }) => void) | null;
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

// Smooth ray-marched field renderer.
// Uses hardware LinearFilter on the 3D texture for smooth trilinear sampling,
// and a wider normal-estimation epsilon to avoid voxel-grid bumpiness.
//
// Field-only rendering always uses multi-hit compositing so the outer shell
// reads as a soft ghost while interior field morphology remains visible.
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

  vec3 shade(vec3 p, float voxel, vec3 baseColor, float surfaceMix) {
    float normalEps = max(voxel * mix(2.0, 3.1, surfaceMix), 1e-3);
    vec3 normal = estimateNormal(p, normalEps);
    vec3 keyLight = normalize(vec3(0.55, 0.78, 0.38));
    vec3 fillLight = normalize(vec3(-0.45, 0.3, -0.85));
    vec3 viewDir = normalize(cameraPosition - p);
    vec3 halfVec = normalize(keyLight + viewDir);
    float diffKey = max(dot(normal, keyLight), 0.0);
    float diffFill = max(dot(normal, fillLight), 0.0);
    float hemi = mix(0.84, 0.7 + 0.3 * (normal.y * 0.5 + 0.5), surfaceMix);
    float rim = pow(1.0 - abs(dot(normal, viewDir)), mix(3.1, 1.95, surfaceMix));
    float diffuse = mix(0.92 + 0.16 * diffKey + 0.05 * diffFill, 0.76 + 0.58 * diffKey + 0.22 * diffFill, surfaceMix);
    vec3 lit = baseColor * diffuse * hemi;
    float rimStrength = mix(0.08, 0.34, surfaceMix);
    float specular = pow(max(dot(normal, halfVec), 0.0), mix(24.0, 52.0, surfaceMix));
    float glossStrength = smoothstep(0.22, 0.95, surfaceMix) * 0.28;
    vec3 glossColor = mix(baseColor, vec3(0.97, 0.995, 1.0), 0.72);
    return lit + baseColor * rim * rimStrength + glossColor * specular * glossStrength;
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

    const int MAX_STEPS = 2048;
    const int MAX_SURFACE_HITS = 4;
    const float SHELL_ALPHA = 0.12;
    float skipDist = voxel * 2.2;
    float raySpan = max(tEnd - tStart, minStep);

    vec3 shellColor = mix(vec3(0.86, 0.95, 1.0), vec3(0.94, 0.985, 1.0), 0.62);
    vec3 interiorColor = mix(uColor, vec3(0.42, 0.98, 0.96), 0.58);

    vec3 accumColor = vec3(0.0);
    float accumAlpha = 0.0;
    float t = tStart;
    float prev = sampleField(ro + rd * tStart);
    int hitCount = 0;

    for (int i = 0; i < MAX_STEPS; i++) {
      if (t > tEnd) break;
      if (accumAlpha >= 0.94) break;
      if (hitCount >= MAX_SURFACE_HITS) break;

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

        if (uSectionEnabled > 0.5 && pHit.y > uSectionLevel) {
          t += skipDist + minStep;
          if (t > tEnd) {
            break;
          }
          prev = sampleField(ro + rd * min(t, tEnd));
          continue;
        }

        float travel = clamp((length(pHit - ro) - tStart) / raySpan, 0.0, 1.0);
        float layerMix = clamp(float(hitCount) / float(MAX_SURFACE_HITS - 1), 0.0, 1.0);
        float surfaceMix = smoothstep(0.12, 0.8, layerMix);
        float interiorMix = smoothstep(0.22, 1.0, surfaceMix);
        vec3 baseColor = mix(shellColor, interiorColor, surfaceMix);
        vec3 litColor = shade(pHit, voxel, baseColor, surfaceMix);
        litColor *= mix(1.18, mix(1.2, 1.1, travel), interiorMix);

        float hitAlpha;
        if (hitCount == 0) {
          hitAlpha = SHELL_ALPHA;
        } else {
          float layerAlpha = mix(0.5, 0.32, clamp(float(hitCount - 1) / 3.0, 0.0, 1.0));
          hitAlpha = layerAlpha * mix(1.04, 0.94, travel);
        }

        float remaining = 1.0 - accumAlpha;
        accumColor += litColor * hitAlpha * remaining;
        accumAlpha += hitAlpha * remaining;

        hitCount++;
        t += skipDist;
        if (t > tEnd) {
          break;
        }
        prev = sampleField(ro + rd * min(t, tEnd));
        continue;
      }

      float stepSize = clamp(abs(s) * 0.55, minStep, maxStep) * uStepScale;
      prev = s;
      t += stepSize;
    }

    if (accumAlpha < 0.02) { discard; }
    outColor = vec4(accumColor / accumAlpha, min(accumAlpha, 0.88));
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
  if (!mesh) {
    return null;
  }
  let vertices: Float32Array;
  let indices: Uint32Array;
  let normals: Float32Array;

  if (mesh.encoding === "mesh-f32-u32-binary-v1") {
    vertices = mesh.vertices;
    indices = mesh.indices;
    normals = mesh.normals;
  } else if (mesh.encoding === "mesh-f32-u32-base64-v1") {
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

    vertices = new Float32Array(verticesBytes.buffer, verticesBytes.byteOffset, expectedVerticesBytes / 4);
    indices = new Uint32Array(indicesBytes.buffer, indicesBytes.byteOffset, expectedIndicesBytes / 4);
    normals = new Float32Array(normalsBytes.buffer, normalsBytes.byteOffset, expectedNormalsBytes / 4);
  } else {
    return null;
  }

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
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / Float32Array.BYTES_PER_ELEMENT);
}

function toFieldTexture(field: FieldPayload | null): THREE.Data3DTexture | null {
  if (!field) {
    return null;
  }
  const values = field.encoding === "f32-binary-v1" ? field.data : field.encoding === "f32-base64" ? decodeFloat32Base64(field.data) : null;
  if (!values) {
    return null;
  }
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
  uploadedFieldPreviewTrace = null,
  onUploadedFieldPreviewVisible = null,
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
  const reportedFieldTraceIdRef = useRef<string | null>(null);

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
  const fieldOnly = hasField && !geometry;

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

  useEffect(() => {
    if (!fieldTexture || !fieldBounds || !uploadedFieldPreviewTrace || !onUploadedFieldPreviewVisible) {
      return;
    }
    if (reportedFieldTraceIdRef.current === uploadedFieldPreviewTrace.traceId) {
      return;
    }
    reportedFieldTraceIdRef.current = uploadedFieldPreviewTrace.traceId;
    const textureReadyAtMs = performance.now();
    let cancelled = false;
    const frameId = requestAnimationFrame(() => {
      if (cancelled) {
        return;
      }
      onUploadedFieldPreviewVisible({
        traceId: uploadedFieldPreviewTrace.traceId,
        textureReadyAtMs,
        firstVisibleFrameAtMs: performance.now()
      });
    });
    return () => {
      cancelled = true;
      cancelAnimationFrame(frameId);
    };
  }, [fieldBounds, fieldTexture, onUploadedFieldPreviewVisible, uploadedFieldPreviewTrace]);

  return (
    <Canvas
      camera={{ position: [3.5, 2.5, 3.5], fov: 45 }}
      dpr={[1, 1.75]}
      gl={{ stencil: true }}
      onCreated={({ gl }) => {
        gl.localClippingEnabled = true;
      }}
    >
      <color attach="background" args={[fieldOnly ? "#30556f" : "#1b3347"]} />
      <ambientLight intensity={fieldOnly ? 0.72 : 0.56} />
      <hemisphereLight args={["#f4fbff", fieldOnly ? "#89abc4" : "#7190ab", fieldOnly ? 0.92 : 0.72]} />
      <directionalLight position={[5, 6, 4]} intensity={fieldOnly ? 1.18 : 1.08} />
      <directionalLight position={[-5, -6, -4]} intensity={fieldOnly ? 0.78 : 0.9} />

      {showGrid ? <gridHelper args={[12, 24, "#2f536d", "#1d2f3d"]} /> : null}
      {showAxes ? <axesHelper args={[2.5]} /> : null}

      {fieldOnly && fieldBounds ? (
        <mesh ref={meshRef} position={fieldBounds.center.toArray() as [number, number, number]} renderOrder={3}>
          <boxGeometry args={fieldBounds.size.toArray() as [number, number, number]} />
          <shaderMaterial
            side={THREE.FrontSide}
            transparent
            depthWrite={false}
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
