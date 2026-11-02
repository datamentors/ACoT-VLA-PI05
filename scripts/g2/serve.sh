#!/bin/bash
# serve.sh [port]   (default 8000)
#   OPTIONAL HELPER, not required: it only saves typing. It starts the pause-per-inference
#   policy server (scripts/serve_policy.py, method "infer") on the AgiBot G2 with this
#   branch's checkpoint. Running serve_policy.py by hand with the same env vars and
#   arguments does exactly the same thing.
#
#   Client: g2_simple_bridge (roboticsG2 deploy/g2-simple-bridge), --host 127.0.0.1.
#   Code  = this checkout (put first on PYTHONPATH), so `git switch <branch>` picks the run.
#   Env   = the robot's existing venv (jax 0.10.2 / CUDA 13 for Thor sm_110); not managed by uv.lock.
#
#   Defaults (this branch): LoopPick step 49000, config pi05_g2_loop_pick_ft.
#   Overrides: CFG=<config> DIR=<checkpoint dir> VENV=<venv dir> ROOT=<server data dir>
#   FRESH_NOISE=1   new diffusion noise every inference (default: one fixed key).
#   G2SIM_FULL_WAIST_TASKS: task "fold_towels" keeps all 21 dims (waist live); other task
#   names get 16 dims and the bridge holds the waist.
#   Refuses to start if the port is already taken: stop that PID yourself.
set -euo pipefail
CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ROOT=${ROOT:-/data/datasets/acot_pi05_server}
PORT=${1:-8000}
CFG=${CFG:-pi05_g2_loop_pick_ft}
DIR=${DIR:-$ROOT/checkpoints/g2_loop_pick_step49000}
VENV=${VENV:-$ROOT/ACoT-VLA/.venv}
LOG=$ROOT/logs/serve_$(basename "$DIR")_$PORT.log
NOISE_ARG=""
[ "${FRESH_NOISE:-0}" = "1" ] && NOISE_ARG="--fresh-noise"

BUSY=$(ss -ltnpH "sport = :$PORT" 2>/dev/null | grep -oE 'pid=[0-9]+' | head -1 || true)
if [ -n "$BUSY" ]; then
  echo "port $PORT is in use ($BUSY); stop it first: kill ${BUSY#pid=}" >&2
  exit 1
fi
[ -d "$DIR" ] || { echo "no checkpoint dir: $DIR" >&2; exit 1; }

cd "$CODE"
mkdir -p "$ROOT/logs" "$ROOT/tmp"
# TMPDIR on /data: the robot root overlay (/tmp) fills up; torch needs a temp dir at import.
env -u XLA_FLAGS -u JAX_COMPILATION_CACHE_DIR \
    PYTHONPATH="$CODE/src:$CODE/packages/openpi-client/src" \
    G2SIM_FULL_WAIST_TASKS=fold_towels \
    TMPDIR="$ROOT/tmp" \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.48 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_ALLOCATOR=platform \
nohup "$VENV/bin/python" scripts/serve_policy.py \
  --port "$PORT" $NOISE_ARG \
  policy:checkpoint \
  --policy.config "$CFG" \
  --policy.dir "$DIR" \
  > "$LOG" 2>&1 &
echo "PID=$!  config=$CFG  dir=$DIR  port=$PORT  fresh_noise=${FRESH_NOISE:-0}  code=$CODE@$(git -C "$CODE" rev-parse --short HEAD)"
echo "log: $LOG"
