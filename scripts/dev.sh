#!/usr/bin/env bash
# Start the Banana Ripeness Classifier (Streamlit app).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_NAME="${CONDA_ENV:-banana-clean}"
PORT="${PORT:-8501}"

echo "Using conda env: $ENV_NAME"
echo "Starting Streamlit on http://localhost:$PORT ..."

if ! conda env list | grep -q "^${ENV_NAME} "; then
  echo "Conda env '$ENV_NAME' not found. Create it with:"
  echo "  conda create -n $ENV_NAME python=3.11 -y"
  echo "  conda run -n $ENV_NAME pip install -r $ROOT/streamlit_app/requirements.txt"
  exit 1
fi

cd "$ROOT/streamlit_app"
exec conda run -n "$ENV_NAME" streamlit run app.py \
  --server.port "$PORT" \
  --server.address localhost
