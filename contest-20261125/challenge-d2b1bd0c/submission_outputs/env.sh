#!/usr/bin/env bash
PLASMA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -d "$PLASMA_ROOT/workspace/physics" ]]; then PLASMA_ROOT="$(dirname "$PLASMA_ROOT")"; fi
mkdir -p "$PLASMA_ROOT/native/tmp"
export TMPDIR="$PLASMA_ROOT/native/tmp"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export LD_LIBRARY_PATH="$PLASMA_ROOT/native/deps/usr/lib/aarch64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$PLASMA_ROOT/.venv/bin:$PATH"
