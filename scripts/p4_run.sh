#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PYTHON=${BRAIN_PYTHON:-/Volumes/Samsung/conda/envs/brain/bin/python}

if [ ! -x "$PYTHON" ]; then
  echo "brain Python not found: $PYTHON" >&2
  exit 2
fi

cd "$ROOT"
exec "$PYTHON" scripts/p4_control.py "$@"
