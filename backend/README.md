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
