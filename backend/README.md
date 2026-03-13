# SDF CAD Backend

## Run

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
uvicorn app.main:app --reload
```

The API is served on `http://127.0.0.1:8000`.

## Worker Queue (Redis + Celery)

Set Redis URL (defaults to `redis://localhost:6379/0`):

```bash
export REDIS_URL=redis://localhost:6379/0
```

Run API and worker in separate terminals:

```bash
uvicorn app.main:app --reload
celery -A app.worker:celery_app worker --loglevel=info --concurrency=2
```

### Deployment profile

- API workers: start with `2-4` uvicorn workers for mixed traffic.
- Celery concurrency: start with `2` and increase based on CPU/GPU capacity.
- Redis: required for durable queue broker + result backend.
- Timeouts/retries: keep task retries disabled by default for deterministic modeling requests; add explicit retries only for infra/transient failures.

### Scalability notes

- In-memory preview caches are process-local by design.
- Queue-backed job APIs (`/api/v1/jobs/*`) are the recommended path for high-resolution or long-running requests.
