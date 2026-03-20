from __future__ import annotations

import logging
from typing import Any

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except Exception:
    cp = None  # type: ignore[assignment]
    CUPY_AVAILABLE = False


def cleanup_gpu_memory(reason: str = "unknown") -> dict[str, int] | None:
    if not CUPY_AVAILABLE or cp is None:
        return None
    try:
        pool = cp.get_default_memory_pool()
        pinned = cp.get_default_pinned_memory_pool()
        before = {
            "used_bytes": int(pool.used_bytes()),
            "total_bytes": int(pool.total_bytes()),
            "pinned_free_blocks": int(pinned.n_free_blocks()),
        }
        cp.cuda.Stream.null.synchronize()
        pool.free_all_blocks()
        pinned.free_all_blocks()
        after = {
            "used_bytes": int(pool.used_bytes()),
            "total_bytes": int(pool.total_bytes()),
            "pinned_free_blocks": int(pinned.n_free_blocks()),
        }
        logger = logging.getLogger(__name__)
        logger.debug("gpu_cleanup reason=%s before=%s after=%s", reason, before, after)
        return {"before_used_bytes": before["used_bytes"], "after_used_bytes": after["used_bytes"]}
    except Exception:
        return None


def with_gpu_cleanup(fn: Any, reason: str):
    try:
        return fn()
    finally:
        cleanup_gpu_memory(reason=reason)
