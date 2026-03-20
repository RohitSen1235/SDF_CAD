from __future__ import annotations

import os

from celery import Celery


REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CELERY_TASK_ALWAYS_EAGER = os.getenv("CELERY_TASK_ALWAYS_EAGER", "0") in {"1", "true", "True"}

celery_app = Celery(
    "sdf_cad",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["app.worker_tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    task_track_started=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_ignore_result=False,
    task_always_eager=CELERY_TASK_ALWAYS_EAGER,
    task_store_eager_result=True,
)
