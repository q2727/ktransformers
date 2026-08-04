#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"
BASE_VENV="${BASE_VENV:-${REPO_ROOT}/.venv}"
DSV4_VENV="${DSV4_VENV:-/data/qinchong/venvs/ktransformers-dsv4}"

if [[ ! -x "${BASE_VENV}/bin/python" ]]; then
  echo "Base KT environment is missing: ${BASE_VENV}" >&2
  exit 1
fi

if [[ ! -x "${DSV4_VENV}/bin/python" ]]; then
  mkdir -p "${DSV4_VENV}"
  cp -a --reflink=auto "${BASE_VENV}/." "${DSV4_VENV}/"
fi

"${DSV4_VENV}/bin/python" -m pip install --upgrade \
  "flashinfer-python==${FLASHINFER_VERSION:-0.6.9}" \
  "flashinfer-cubin==${FLASHINFER_VERSION:-0.6.9}" \
  "transformers==4.57.1" \
  "tilelang==${TILELANG_VERSION:-0.1.10}" \
  "apache-tvm-ffi==0.1.11"

"${DSV4_VENV}/bin/python" - <<'PY'
import sys
import flashinfer
import tilelang
import torch
import transformers
import tvm_ffi

print("python_prefix", sys.prefix)
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("flashinfer", flashinfer.__version__)
print("tilelang", tilelang.__version__)
print("tvm_ffi", tvm_ffi.__version__)
PY
