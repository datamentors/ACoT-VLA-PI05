#!/bin/bash
# serve_stream.sh [port] [server flags...]   (default port 8001)
#   OPTIONAL HELPER, not required: it only saves typing. It starts the server-owned RTC
#   stream server (scripts/serve_rtc_stream.py --rtc.clock wall) on the AgiBot G2 with this
#   branch's checkpoint. Running serve_rtc_stream.py by hand with the same env vars and
#   arguments does exactly the same thing.
#
#   Client: g2_rtc_stream_bridge (roboticsG2 deploy/g2-rtc-stream-bridge).
#   Code  = this checkout (put first on PYTHONPATH), so `git switch <branch>` picks the run.
#   Env   = the robot's existing venv (jax 0.10.2 / CUDA 13 for Thor sm_110); not managed by uv.lock.
#   GEMMs run on cuBLAS (--xla_gpu_enable_triton_gemm=false): on the Thor cuBLAS compiled in
#   9.3 s with a 396 ms first inference; Triton took ~7 min and 430-780 ms per inference.
#
#   Defaults (this branch): LoopPick step 49000, config pi05_g2_loop_pick_ft.
#   Overrides: CFG=<config> DIR=<checkpoint dir> VENV=<venv dir> ROOT=<server data dir>
#   Flags tested on the robot on 2026-10-20 (Ft3):
#     serve_stream.sh 8001 --fresh-noise --rtc.queue-threshold 15 \
#       --rtc.execution-horizon 25 --rtc.prefix-attention-schedule exp \
#       --rtc.max-guidance-weight 10 --rtc.no-readiness-gate
#   Refuses to start if the port is taken or another serve_rtc_stream.py is running:
#   a second model compiling at the same time overloads the CPU the robot stack shares.
#   Stop that PID yourself.
set -euo pipefail
CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ROOT=${ROOT:-/data/datasets/acot_pi05_server}
PORT=${1:-8001}
shift || true
CFG=${CFG:-pi05_g2_loop_pick_ft}
DIR=${DIR:-$ROOT/checkpoints/g2_loop_pick_step49000}
VENV=${VENV:-$ROOT/ACoT-VLA/.venv}
LOG=$ROOT/logs/rtc_stream_$(basename "$DIR").log

BUSY=$(ss -ltnpH "sport = :$PORT" 2>/dev/null | grep -oE 'pid=[0-9]+' | head -1 || true)
if [ -n "$BUSY" ]; then
  echo "port $PORT is in use ($BUSY); stop it first: kill ${BUSY#pid=}" >&2
  exit 1
fi
RUNNING=$(pgrep -f 'scripts/serve_rtc_stream.py' || true)
if [ -n "$RUNNING" ]; then
  echo "serve_rtc_stream.py already running (PID $(echo $RUNNING)); stop it first: kill $(echo $RUNNING)" >&2
  exit 1
fi
[ -d "$DIR" ] || { echo "no checkpoint dir: $DIR" >&2; exit 1; }

cd "$CODE"
mkdir -p "$ROOT/logs" "$ROOT/tmp"
env -u JAX_COMPILATION_CACHE_DIR \
    XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" \
    PYTHONPATH="$CODE/src:$CODE/packages/openpi-client/src" \
    TMPDIR="$ROOT/tmp" \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.48 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_ALLOCATOR=platform \
nohup "$VENV/bin/python" scripts/serve_rtc_stream.py \
  --config "$CFG" \
  --dir "$DIR" \
  --port "$PORT" \
  --rtc.clock wall \
  "$@" \
  > "$LOG" 2>&1 &
echo "PID=$!  config=$CFG  dir=$DIR  port=$PORT  code=$CODE@$(git -C "$CODE" rev-parse --short HEAD)"
echo "log: $LOG"
