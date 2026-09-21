#!/usr/bin/env bash
# Start the API locally on http://127.0.0.1:8000  (docs at /docs)
cd "$(dirname "$0")"
exec ../.venv/bin/uvicorn app.api:app --reload --port "${PORT:-8000}"
