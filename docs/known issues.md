# SDF CAD Known Issues

Last updated: 2026-03-13

## 1) Analytic preview only supports a subset of DSL nodes
- Symptoms: `Generate Shape` may fall back from analytic raymarch preview to field/mesh preview with an "Unsupported analytic nodes..." message.
- Why: The analytic GLSL compiler rejects unsupported node/expr patterns and returns `analytic_supported=false`.
- Workaround: Use standard field/mesh preview for unsupported graphs; simplify node types if analytic preview is required.

## 2) CUDA requests can still execute on CPU (or fail if explicitly forced)
- Symptoms: Selecting `cuda` may still report CPU backend, or explicit CUDA meshing can fail with `CUDA meshing failed: ...`.
- Why: Auto mode falls back when CUDA/CuPy/runtime is unavailable or errors at runtime.
- Workaround: Use `auto` for resiliency; use explicit `cuda` only when hard-fail behavior is desired.

## 3) Adaptive meshing is CPU-only and can be slower than uniform meshing
- Symptoms: `adaptive` mode may be slower than expected, especially on CPU.
- Why: Adaptive path currently routes to CPU implementation.
- Workaround: Prefer `uniform` for faster previews unless adaptive output is specifically needed.

## 4) Strict mesh-upload requirements reject many real-world files
- Symptoms: Upload failures for non-watertight, non-manifold, degenerate, or non-triangle meshes.
- Why: Upload validation enforces a watertight edge-manifold triangle mesh and finite geometry.
- Workaround: Repair/retopologize meshes before upload (remove non-manifold edges, triangulate, fix degenerates, close holes).

## 5) Upload format and size limits
- Symptoms: Upload is rejected for unsupported file types or large payloads.
- Why: Only `.stl` and `.obj` are accepted; max upload size is 200 MB.
- Workaround: Convert to STL/OBJ and reduce mesh size before upload.

## 6) Voxelization can fail on thin/leaky/self-intersecting geometry
- Symptoms: Errors such as:
`Voxelization failed: no solid volume detected`,
`Voxelization leaked to grid boundary`,
or `Voxelization could not recover interior volume`.
- Why: Rasterization + hole-filling cannot robustly recover volume for certain problematic inputs.
- Workaround: Clean the source mesh, ensure closed solid geometry, and/or increase model scale.

## 7) Bounds and resolution hard limits
- Symptoms: Requests fail for very large bounds/resolution; large models can clip if bounds are too tight.
- Why: Grid resolution is capped at 256 and bounds are constrained to `[-200, 200]` per axis.
- Workaround: Re-center/scale models and tune bounds manually for large scenes.

## 8) Bounds inference is not always possible
- Symptoms: Compile diagnostics warning:
`Could not infer finite bounds for the root node...`
- Why: Certain node graphs cannot provide finite inferred bounds.
- Workaround: Provide explicit grid bounds for predictable previews/exports.

## 9) Long evaluations can time out
- Symptoms: `408 Field evaluation timeout exceeded`.
- Why: Server-side evaluation timeout is fixed at 20 seconds.
- Workaround: Lower quality/resolution, simplify scene complexity, or use queued jobs for heavier workloads.

## 10) Queue behavior depends on Redis/Celery availability
- Symptoms: `execution_mode=auto` may run inline; `queued` mode can return queue-unavailable errors.
- Why: Queue-backed execution requires Redis + Celery; without it, auto falls back and queued mode cannot proceed.
- Workaround: Run Redis/Celery in deployed environments where queueing is expected.

## 11) Job artifacts are temporary
- Symptoms: Downloading completed job results can return `404 Export artifact no longer available`.
- Why: Export artifacts are deleted after streaming and old files are purged periodically.
- Workaround: Download artifacts immediately after completion.

## 12) Preview caches are process-local
- Symptoms: Cache hit rates drop in multi-worker deployments; repeated requests across workers may recompute.
- Why: In-memory caches are local to each API process.
- Workaround: Expect per-worker warmup or add shared/distributed cache if cross-worker reuse is required.

## 13) WebSocket phased mesh preview may downgrade to HTTP
- Symptoms: Progressive `field -> mesh` updates disappear when WS errors/closes; request still succeeds later.
- Why: Frontend falls back to HTTP preview when websocket path fails.
- Workaround: Ensure WS connectivity for phased updates; otherwise expect single-response behavior.
