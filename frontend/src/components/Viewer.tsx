import { OrbitControls, TransformControls } from "@react-three/drei";
import { Canvas, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef, useState } from "react";
import type { RefObject } from "react";
import * as THREE from "three";

import { FieldPayload, MeshPayload, MeshProgramPayload, SceneProgramPayload } from "../types";

export interface ViewerProps {
  mesh: MeshPayload | null;
  field: FieldPayload | null;
  analyticSceneProgram?: SceneProgramPayload | null;
  analyticMeshProgram?: MeshProgramPayload | null;
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

  // Manual trilinear interpolation — avoids NearestFilter voxel-grid artifacts
  // on hardware that does not support linear filtering for R32F 3D textures.
  float sampleField(vec3 worldPos) {
    vec3 uvw = (worldPos - uBoundsMin) / (uBoundsMax - uBoundsMin);
    uvw = clamp(uvw, 0.0, 1.0);
    float n = uResolution;
    // Convert to texel-space [0, n-1] and split into integer + fractional parts.
    vec3 tc = uvw * (n - 1.0);
    vec3 ti = floor(tc);
    vec3 tf = tc - ti;
    // Normalise back to [0,1] UV for texelFetch-equivalent via texture().
    float inv = 1.0 / n;
    vec3 uv0 = (ti + 0.5) * inv;
    vec3 uv1 = (ti + 1.5) * inv;
    float c000 = texture(uField, vec3(uv0.x, uv0.y, uv0.z)).r;
    float c100 = texture(uField, vec3(uv1.x, uv0.y, uv0.z)).r;
    float c010 = texture(uField, vec3(uv0.x, uv1.y, uv0.z)).r;
    float c110 = texture(uField, vec3(uv1.x, uv1.y, uv0.z)).r;
    float c001 = texture(uField, vec3(uv0.x, uv0.y, uv1.z)).r;
    float c101 = texture(uField, vec3(uv1.x, uv0.y, uv1.z)).r;
    float c011 = texture(uField, vec3(uv0.x, uv1.y, uv1.z)).r;
    float c111 = texture(uField, vec3(uv1.x, uv1.y, uv1.z)).r;
    float c00 = mix(c000, c100, tf.x);
    float c10 = mix(c010, c110, tf.x);
    float c01 = mix(c001, c101, tf.x);
    float c11 = mix(c011, c111, tf.x);
    float c0  = mix(c00, c10, tf.y);
    float c1  = mix(c01, c11, tf.y);
    return mix(c0, c1, tf.z);
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

const ANALYTIC_DSL_FRAGMENT_TEMPLATE = `
  precision highp float;

  uniform vec3 uBoundsMin;
  uniform vec3 uBoundsMax;
  uniform vec3 uColor;
  uniform float uSectionEnabled;
  uniform float uSectionLevel;
  uniform float uMaxSteps;
  uniform float uHitEps;
  uniform float uNormalEps;

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

  __SDF_SCENE__

  vec3 estimateNormal(vec3 p) {
    vec2 e = vec2(1.0, -1.0) * uNormalEps;
    return normalize(
      e.xyy * sdfScene(p + e.xyy) +
      e.yyx * sdfScene(p + e.yyx) +
      e.yxy * sdfScene(p + e.yxy) +
      e.xxx * sdfScene(p + e.xxx)
    );
  }

  void main() {
    vec3 ro = cameraPosition;
    vec3 rd = normalize(vWorldPos - cameraPosition);
    vec2 hit = intersectAABB(ro, rd, uBoundsMin, uBoundsMax);
    if (hit.x > hit.y) discard;

    float t = max(hit.x, 0.0);
    float tEnd = hit.y;
    bool hitSurface = false;
    vec3 pHit = vec3(0.0);
    float maxSteps = max(uMaxSteps, 8.0);

    for (int i = 0; i < 640; i++) {
      if (float(i) >= maxSteps || t > tEnd) break;
      vec3 p = ro + rd * t;
      float d = sdfScene(p);
      if (abs(d) <= uHitEps) {
        hitSurface = true;
        pHit = p;
        break;
      }
      float stepSize = clamp(abs(d) * 0.9, uHitEps * 0.6, 0.2);
      t += stepSize;
    }

    if (!hitSurface) discard;
    if (uSectionEnabled > 0.5 && pHit.y > uSectionLevel) discard;

    vec3 n = estimateNormal(pHit);
    vec3 lightDir = normalize(vec3(0.55, 0.75, 0.4));
    float diff = max(dot(n, lightDir), 0.0);
    float hemi = 0.4 + 0.6 * (n.y * 0.5 + 0.5);
    vec3 lit = uColor * (0.28 + 0.72 * diff) * hemi;
    outColor = vec4(lit, 1.0);
  }
`;

const ANALYTIC_MESH_FRAGMENT_SHADER = `
  precision highp float;
  precision highp sampler2D;

  uniform vec3 uBoundsMin;
  uniform vec3 uBoundsMax;
  uniform vec3 uColor;
  uniform float uSectionEnabled;
  uniform float uSectionLevel;
  uniform float uMaxSteps;
  uniform float uHitEps;
  uniform float uNormalEps;
  uniform sampler2D uTriTex;
  uniform vec2 uTriTexSize;
  uniform sampler2D uBvhTex;
  uniform vec2 uBvhTexSize;
  uniform float uTriangleCount;
  uniform float uShellThickness;
  uniform float uLatticePitch;
  uniform float uLatticeThickness;
  uniform float uLatticePhase;
  uniform float uLatticeType;

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

  float readScalar(sampler2D tex, vec2 texSize, int scalarIndex) {
    int texelIndex = scalarIndex / 4;
    int chan = scalarIndex - texelIndex * 4;
    int width = int(texSize.x);
    int x = texelIndex - (texelIndex / width) * width;
    int y = texelIndex / width;
    vec4 t = texelFetch(tex, ivec2(x, y), 0);
    if (chan == 0) return t.x;
    if (chan == 1) return t.y;
    if (chan == 2) return t.z;
    return t.w;
  }

  vec3 readVec3(sampler2D tex, vec2 texSize, int startIndex) {
    return vec3(
      readScalar(tex, texSize, startIndex),
      readScalar(tex, texSize, startIndex + 1),
      readScalar(tex, texSize, startIndex + 2)
    );
  }

  float sdBoxDistance(vec3 p, vec3 bmin, vec3 bmax) {
    vec3 c = 0.5 * (bmin + bmax);
    vec3 h = 0.5 * (bmax - bmin);
    vec3 q = abs(p - c) - h;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
  }

  vec3 closestPointTriangle(vec3 p, vec3 a, vec3 b, vec3 c) {
    vec3 ab = b - a;
    vec3 ac = c - a;
    vec3 ap = p - a;
    float d1 = dot(ab, ap);
    float d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) return a;
    vec3 bp = p - b;
    float d3 = dot(ab, bp);
    float d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) return b;
    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
      float v = d1 / (d1 - d3);
      return a + v * ab;
    }
    vec3 cp = p - c;
    float d5 = dot(ab, cp);
    float d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) return c;
    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
      float w = d2 / (d2 - d6);
      return a + w * ac;
    }
    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
      float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
      return b + w * (c - b);
    }
    float denom = 1.0 / (va + vb + vc);
    float v = vb * denom;
    float w = vc * denom;
    return a + ab * v + ac * w;
  }

  float sdfHostMesh(vec3 p) {
    float best = 1e9;
    float signOut = 1.0;
    int stack[64];
    int sp = 0;
    stack[sp++] = 0;

    for (int iter = 0; iter < 1024; iter++) {
      if (sp <= 0) break;
      int ni = stack[--sp];
      int base = ni * 12;
      vec3 bmin = readVec3(uBvhTex, uBvhTexSize, base + 0);
      vec3 bmax = readVec3(uBvhTex, uBvhTexSize, base + 4);
      float boxDist = sdBoxDistance(p, bmin, bmax);
      if (boxDist > best) continue;
      int left = int(readScalar(uBvhTex, uBvhTexSize, base + 8));
      int right = int(readScalar(uBvhTex, uBvhTexSize, base + 9));
      int start = int(readScalar(uBvhTex, uBvhTexSize, base + 10));
      int count = int(readScalar(uBvhTex, uBvhTexSize, base + 11));
      if (count > 0) {
        for (int i = 0; i < 64; i++) {
          if (i >= count) break;
          int triIndex = start + i;
          if (float(triIndex) >= uTriangleCount) break;
          int triBase = triIndex * 18;
          vec3 a = readVec3(uTriTex, uTriTexSize, triBase + 0);
          vec3 b = readVec3(uTriTex, uTriTexSize, triBase + 3);
          vec3 c = readVec3(uTriTex, uTriTexSize, triBase + 6);
          vec3 na = readVec3(uTriTex, uTriTexSize, triBase + 9);
          vec3 nb = readVec3(uTriTex, uTriTexSize, triBase + 12);
          vec3 nc = readVec3(uTriTex, uTriTexSize, triBase + 15);
          vec3 cp = closestPointTriangle(p, a, b, c);
          float d = length(p - cp);
          if (d < best) {
            best = d;
            vec3 pn = normalize(na + nb + nc);
            signOut = dot(p - cp, pn) < 0.0 ? -1.0 : 1.0;
          }
        }
      } else {
        if (left >= 0 && sp < 63) stack[sp++] = left;
        if (right >= 0 && sp < 63) stack[sp++] = right;
      }
    }
    return best * signOut;
  }

  float latticeField(vec3 p) {
    float scale = 6.28318530718 / max(abs(uLatticePitch), 1e-6);
    float u = p.x * scale + uLatticePhase;
    float v = p.y * scale + uLatticePhase;
    float w = p.z * scale + uLatticePhase;
    float base;
    if (uLatticeType < 0.5) {
      base = sin(u) * cos(v) + sin(v) * cos(w) + sin(w) * cos(u);
    } else if (uLatticeType < 1.5) {
      base = cos(u) + cos(v) + cos(w);
    } else {
      base =
        sin(u) * sin(v) * sin(w)
        + sin(u) * cos(v) * cos(w)
        + cos(u) * sin(v) * cos(w)
        + cos(u) * cos(v) * sin(w);
    }
    return abs(base) - abs(uLatticeThickness);
  }

  float sdfScene(vec3 p) {
    float host = sdfHostMesh(p);
    float shell = max(host, -host - abs(uShellThickness));
    float cavity = host + abs(uShellThickness);
    float lattice = latticeField(p);
    float clipped = max(lattice, cavity);
    return min(shell, clipped);
  }

  vec3 estimateNormal(vec3 p) {
    vec2 e = vec2(1.0, -1.0) * uNormalEps;
    return normalize(
      e.xyy * sdfScene(p + e.xyy) +
      e.yyx * sdfScene(p + e.yyx) +
      e.yxy * sdfScene(p + e.yxy) +
      e.xxx * sdfScene(p + e.xxx)
    );
  }

  void main() {
    vec3 ro = cameraPosition;
    vec3 rd = normalize(vWorldPos - cameraPosition);
    vec2 hit = intersectAABB(ro, rd, uBoundsMin, uBoundsMax);
    if (hit.x > hit.y) discard;

    float t = max(hit.x, 0.0);
    float tEnd = hit.y;
    bool hitSurface = false;
    vec3 pHit = vec3(0.0);
    float maxSteps = max(uMaxSteps, 8.0);

    for (int i = 0; i < 640; i++) {
      if (float(i) >= maxSteps || t > tEnd) break;
      vec3 p = ro + rd * t;
      float d = sdfScene(p);
      if (abs(d) <= uHitEps) {
        hitSurface = true;
        pHit = p;
        break;
      }
      float stepSize = clamp(abs(d) * 0.9, uHitEps * 0.6, 0.2);
      t += stepSize;
    }

    if (!hitSurface) discard;
    if (uSectionEnabled > 0.5 && pHit.y > uSectionLevel) discard;

    vec3 n = estimateNormal(pHit);
    vec3 lightDir = normalize(vec3(0.55, 0.75, 0.4));
    float diff = max(dot(n, lightDir), 0.0);
    float hemi = 0.4 + 0.6 * (n.y * 0.5 + 0.5);
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

function toPackedFloatTexture(values: Float32Array): { texture: THREE.DataTexture; width: number; height: number } {
  const texelCount = Math.max(1, Math.ceil(values.length / 4));
  const width = Math.min(1024, texelCount);
  const height = Math.max(1, Math.ceil(texelCount / width));
  const out = new Float32Array(width * height * 4);
  out.set(values);
  const texture = new THREE.DataTexture(out, width, height, THREE.RGBAFormat, THREE.FloatType);
  texture.internalFormat = "RGBA32F";
  texture.minFilter = THREE.NearestFilter;
  texture.magFilter = THREE.NearestFilter;
  texture.wrapS = THREE.ClampToEdgeWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.generateMipmaps = false;
  texture.unpackAlignment = 1;
  texture.needsUpdate = true;
  return { texture, width, height };
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
  analyticSceneProgram,
  analyticMeshProgram,
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
  const analyticDslFragment = useMemo(() => {
    if (!analyticSceneProgram?.glsl_sdf) {
      return null;
    }
    return ANALYTIC_DSL_FRAGMENT_TEMPLATE.replace("__SDF_SCENE__", analyticSceneProgram.glsl_sdf);
  }, [analyticSceneProgram]);
  const analyticTri = useMemo(() => {
    if (!analyticMeshProgram) {
      return null;
    }
    const values = decodeFloat32Base64(analyticMeshProgram.triangles_data);
    return toPackedFloatTexture(values);
  }, [analyticMeshProgram]);
  const analyticBvh = useMemo(() => {
    if (!analyticMeshProgram) {
      return null;
    }
    const values = decodeFloat32Base64(analyticMeshProgram.bvh_data);
    return toPackedFloatTexture(values);
  }, [analyticMeshProgram]);
  const fieldBounds = useMemo(() => {
    if (analyticSceneProgram) {
      const min = new THREE.Vector3(
        analyticSceneProgram.bounds[0][0],
        analyticSceneProgram.bounds[1][0],
        analyticSceneProgram.bounds[2][0]
      );
      const max = new THREE.Vector3(
        analyticSceneProgram.bounds[0][1],
        analyticSceneProgram.bounds[1][1],
        analyticSceneProgram.bounds[2][1]
      );
      const center = min.clone().add(max).multiplyScalar(0.5);
      const size = max.clone().sub(min);
      return { min, max, center, size };
    }
    if (analyticMeshProgram) {
      const min = new THREE.Vector3(
        analyticMeshProgram.bounds[0][0],
        analyticMeshProgram.bounds[1][0],
        analyticMeshProgram.bounds[2][0]
      );
      const max = new THREE.Vector3(
        analyticMeshProgram.bounds[0][1],
        analyticMeshProgram.bounds[1][1],
        analyticMeshProgram.bounds[2][1]
      );
      const center = min.clone().add(max).multiplyScalar(0.5);
      const size = max.clone().sub(min);
      return { min, max, center, size };
    }
    if (!field) {
      return null;
    }
    const min = new THREE.Vector3(field.bounds[0][0], field.bounds[1][0], field.bounds[2][0]);
    const max = new THREE.Vector3(field.bounds[0][1], field.bounds[1][1], field.bounds[2][1]);
    const center = min.clone().add(max).multiplyScalar(0.5);
    const size = max.clone().sub(min);
    return { min, max, center, size };
  }, [field, analyticSceneProgram, analyticMeshProgram]);

  const hasField = (fieldTexture !== null || analyticDslFragment !== null || analyticMeshProgram !== null) && fieldBounds !== null;

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
    return () => {
      if (analyticTri?.texture) {
        analyticTri.texture.dispose();
      }
    };
  }, [analyticTri]);

  useEffect(() => {
    return () => {
      if (analyticBvh?.texture) {
        analyticBvh.texture.dispose();
      }
    };
  }, [analyticBvh]);

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
          {analyticMeshProgram && analyticTri && analyticBvh ? (
            <shaderMaterial
              side={THREE.FrontSide}
              transparent={false}
              glslVersion={THREE.GLSL3}
              vertexShader={FIELD_VERTEX_SHADER}
              fragmentShader={ANALYTIC_MESH_FRAGMENT_SHADER}
              toneMapped={false}
              uniforms={{
                uBoundsMin: { value: fieldBounds.min },
                uBoundsMax: { value: fieldBounds.max },
                uColor: { value: new THREE.Color("#8be9fd") },
                uSectionEnabled: { value: sectionEnabled ? 1.0 : 0.0 },
                uSectionLevel: { value: sectionLevel },
                uMaxSteps: { value: analyticMeshProgram.max_steps },
                uHitEps: { value: analyticMeshProgram.hit_epsilon },
                uNormalEps: { value: analyticMeshProgram.normal_epsilon },
                uTriTex: { value: analyticTri.texture },
                uTriTexSize: { value: new THREE.Vector2(analyticTri.width, analyticTri.height) },
                uBvhTex: { value: analyticBvh.texture },
                uBvhTexSize: { value: new THREE.Vector2(analyticBvh.width, analyticBvh.height) },
                uTriangleCount: { value: analyticMeshProgram.triangle_count },
                uShellThickness: { value: analyticMeshProgram.shell_thickness },
                uLatticePitch: { value: analyticMeshProgram.lattice_pitch },
                uLatticeThickness: { value: analyticMeshProgram.lattice_thickness },
                uLatticePhase: { value: analyticMeshProgram.lattice_phase },
                uLatticeType: {
                  value:
                    analyticMeshProgram.lattice_type === "gyroid"
                      ? 0.0
                      : analyticMeshProgram.lattice_type === "schwarz_p"
                        ? 1.0
                        : 2.0
                }
              }}
            />
          ) : analyticDslFragment && analyticSceneProgram ? (
            <shaderMaterial
              side={THREE.FrontSide}
              transparent={false}
              glslVersion={THREE.GLSL3}
              vertexShader={FIELD_VERTEX_SHADER}
              fragmentShader={analyticDslFragment}
              toneMapped={false}
              uniforms={{
                uBoundsMin: { value: fieldBounds.min },
                uBoundsMax: { value: fieldBounds.max },
                uColor: { value: new THREE.Color("#8be9fd") },
                uSectionEnabled: { value: sectionEnabled ? 1.0 : 0.0 },
                uSectionLevel: { value: sectionLevel },
                uMaxSteps: { value: analyticSceneProgram.max_steps },
                uHitEps: { value: analyticSceneProgram.hit_epsilon },
                uNormalEps: { value: analyticSceneProgram.normal_epsilon }
              }}
            />
          ) : (
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
          )}
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

      {fieldTexture && !analyticSceneProgram && !analyticMeshProgram && !geometry && sectionEnabled && fieldBounds ? (
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
