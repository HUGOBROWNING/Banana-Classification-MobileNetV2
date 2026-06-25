#!/usr/bin/env bash
# Start Streamlit, freeing the port if a previous instance is still running.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_NAME="${CONDA_ENV:-banana-clean}"
PORT="${PORT:-8501}"

if lsof -ti ":$PORT" >/dev/null 2>&1; then
  echo "Port $PORT in use — stopping previous Streamlit instance..."
  lsof -ti ":$PORT" | xargs kill -9 2>/dev/null || true
  sleep 1
fi

cd "$ROOT/streamlit_app"
exec conda run --no-capture-output -n "$ENV_NAME" streamlit run app.py \
  --server.port "$PORT" \
  --server.address localhost
