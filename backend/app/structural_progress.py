from __future__ import annotations

import json
import os
import time
from typing import Any

from .models import (
    OptimizationHistoryEntry,
    StructuralOptimizationIterationWebhookRequest,
    StructuralOptimizationProgressResponse,
    StructuralOptimizationResultResponse,
)
from .worker import REDIS_URL

try:
    import redis
except Exception:  # pragma: no cover - handled by caller
    redis = None  # type: ignore[assignment]


STRUCTURAL_PROGRESS_TTL_SECONDS = int(os.getenv("SDF_CAD_STRUCTURAL_PROGRESS_TTL_SECONDS", str(24 * 3600)))


class StructuralOptimizationProgressStore:
    def __init__(self, client: Any, ttl_seconds: int = STRUCTURAL_PROGRESS_TTL_SECONDS) -> None:
        self.client = client
        self.ttl_seconds = ttl_seconds

    @classmethod
    def from_env(cls) -> "StructuralOptimizationProgressStore":
        if redis is None:
            raise RuntimeError("Redis client is unavailable for structural optimization progress storage")
        client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        return cls(client=client)

    def _meta_key(self, job_id: str) -> str:
        return f"sdfcad:structural-opt:{job_id}:meta"

    def _iterations_key(self, job_id: str) -> str:
        return f"sdfcad:structural-opt:{job_id}:iterations"

    def _history_key(self, job_id: str) -> str:
        return f"sdfcad:structural-opt:{job_id}:history"

    def initialize_job(self, *, job_id: str, max_iterations: int, callback_token: str) -> None:
        now = f"{time.time():.6f}"
        meta_key = self._meta_key(job_id)
        iterations_key = self._iterations_key(job_id)
        history_key = self._history_key(job_id)
        pipe = self.client.pipeline()
        pipe.delete(meta_key, iterations_key, history_key)
        pipe.hset(
            meta_key,
            mapping={
                "status": "queued",
                "current_iteration": "0",
                "max_iterations": str(int(max_iterations)),
                "callback_token": callback_token,
                "stop_reason": "",
                "detail": "",
                "bounds_json": "",
                "resolution_xyz_json": "",
                "compute_backend_used": "",
                "mesh_backend_used": "",
                "final_result_json": "",
                "created_at": now,
                "updated_at": now,
            },
        )
        pipe.expire(meta_key, self.ttl_seconds)
        pipe.expire(iterations_key, self.ttl_seconds)
        pipe.expire(history_key, self.ttl_seconds)
        pipe.execute()

    def clear_job(self, job_id: str) -> None:
        self.client.delete(self._meta_key(job_id), self._iterations_key(job_id), self._history_key(job_id))

    def mark_running(self, job_id: str) -> None:
        meta_key = self._meta_key(job_id)
        if not self.client.exists(meta_key):
            raise KeyError(job_id)
        self.client.hset(meta_key, mapping={"status": "running", "updated_at": f"{time.time():.6f}"})
        self.client.expire(meta_key, self.ttl_seconds)

    def mark_failed(self, job_id: str, detail: str) -> None:
        meta_key = self._meta_key(job_id)
        if not self.client.exists(meta_key):
            raise KeyError(job_id)
        self.client.hset(
            meta_key,
            mapping={
                "status": "failed",
                "detail": detail,
                "updated_at": f"{time.time():.6f}",
            },
        )
        self.client.expire(meta_key, self.ttl_seconds)

    def validate_callback_token(self, job_id: str, token: str) -> bool:
        stored = self.client.hget(self._meta_key(job_id), "callback_token")
        return isinstance(stored, str) and stored == token

    def _load_history(self, job_id: str) -> list[OptimizationHistoryEntry]:
        entries = self.client.lrange(self._history_key(job_id), 0, -1)
        return [OptimizationHistoryEntry.model_validate_json(raw) for raw in entries]

    def persist_callback(self, job_id: str, payload: StructuralOptimizationIterationWebhookRequest) -> StructuralOptimizationProgressResponse:
        meta_key = self._meta_key(job_id)
        if not self.client.exists(meta_key):
            raise KeyError(job_id)

        now = f"{time.time():.6f}"
        pipe = self.client.pipeline()
        if payload.iteration_result is not None:
            pipe.rpush(self._iterations_key(job_id), payload.iteration_result.model_dump_json())
            pipe.expire(self._iterations_key(job_id), self.ttl_seconds)
        if payload.history_entry is not None:
            pipe.rpush(self._history_key(job_id), payload.history_entry.model_dump_json())
            pipe.expire(self._history_key(job_id), self.ttl_seconds)

        meta_update: dict[str, str] = {"updated_at": now}
        if payload.iteration_result is not None:
            meta_update["current_iteration"] = str(int(payload.iteration_result.iteration))
            meta_update["status"] = "running"
        if payload.bounds is not None:
            meta_update["bounds_json"] = json.dumps(payload.bounds, separators=(",", ":"))
        if payload.resolution_xyz is not None:
            meta_update["resolution_xyz_json"] = json.dumps(payload.resolution_xyz, separators=(",", ":"))
        if payload.compute_backend_used is not None:
            meta_update["compute_backend_used"] = payload.compute_backend_used
        if payload.mesh_backend_used is not None:
            meta_update["mesh_backend_used"] = payload.mesh_backend_used
        if payload.failure_detail:
            meta_update["status"] = "failed"
            meta_update["detail"] = payload.failure_detail
        if payload.is_final and not payload.failure_detail:
            meta_update["status"] = "succeeded"
            meta_update["stop_reason"] = payload.stop_reason or ""
        pipe.hset(meta_key, mapping=meta_update)
        pipe.expire(meta_key, self.ttl_seconds)
        pipe.execute()

        if payload.is_final and payload.iteration_result is not None and payload.history_entry is not None and not payload.failure_detail:
            history = self._load_history(job_id)
            final_result = StructuralOptimizationResultResponse(
                history=history,
                final_iteration=payload.iteration_result,
                bounds=payload.bounds or [],
                resolution_xyz=payload.resolution_xyz or [],
                compute_backend_used=payload.compute_backend_used or "cpu",
                mesh_backend_used=payload.mesh_backend_used or "cpu",
                stop_reason=payload.stop_reason or "max_iterations",
            )
            self.client.hset(meta_key, mapping={"final_result_json": final_result.model_dump_json(), "updated_at": now})
            self.client.expire(meta_key, self.ttl_seconds)

        return self.get_progress(job_id=job_id, after_iteration=max(0, (payload.iteration_result.iteration - 1) if payload.iteration_result else 0))

    def get_progress(self, *, job_id: str, after_iteration: int = 0) -> StructuralOptimizationProgressResponse:
        meta_key = self._meta_key(job_id)
        meta = self.client.hgetall(meta_key)
        if not meta:
            raise KeyError(job_id)

        start_index = max(0, int(after_iteration))
        raw_iterations = self.client.lrange(self._iterations_key(job_id), start_index, -1)
        raw_history = self.client.lrange(self._history_key(job_id), 0, -1)
        final_result_raw = meta.get("final_result_json", "")
        final_result = StructuralOptimizationResultResponse.model_validate_json(final_result_raw) if final_result_raw else None

        return StructuralOptimizationProgressResponse(
            job_id=job_id,
            status=str(meta.get("status") or "queued"),  # type: ignore[arg-type]
            current_iteration=int(meta.get("current_iteration") or 0),
            max_iterations=int(meta.get("max_iterations") or 0),
            iterations=[item for item in (self._parse_iteration(raw) for raw in raw_iterations) if item is not None],
            history=[OptimizationHistoryEntry.model_validate_json(raw) for raw in raw_history],
            stop_reason=(str(meta.get("stop_reason")) if meta.get("stop_reason") else None),  # type: ignore[arg-type]
            detail=str(meta.get("detail")) if meta.get("detail") else None,
            final_result=final_result,
        )

    def _parse_iteration(self, raw: str) -> Any:
        from .models import StructuralOptimizationIterationResult

        if not raw:
            return None
        return StructuralOptimizationIterationResult.model_validate_json(raw)


_store: StructuralOptimizationProgressStore | None = None


def get_structural_progress_store() -> StructuralOptimizationProgressStore:
    global _store
    if _store is None:
        _store = StructuralOptimizationProgressStore.from_env()
    return _store
