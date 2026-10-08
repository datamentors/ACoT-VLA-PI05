#!/bin/bash
# run_sorting_matrix.sh [variant ...]
#   Sequential Genie Sim comparison on sorting_packages with manipulation_pi05:
#   pause-per-inference (stock client), the client-side RTC (rtc_policy.py +
#   rtc_infer) and the server-side RTC stream (serve_rtc_stream.py), EP episodes each.
#   Without arguments every variant below runs; otherwise only the named ones.
#
#   Per variant: start its server, wait for the port, run the sim, stop the server
#   (by the PID listening on its port), stop any leftover sim process (by PID inside
#   geniesim3), and copy the new evaluate_ret_*.json to $OUT/<variant>_eval.json.
#   Progress: $OUT/progress.log.
#
#   Old paths run from ~/g2sim/ACoT-VLA (serve_policy.sh, unchanged); the stream
#   server runs from the ai-738 branch copy ~/g2sim/ACoT-VLA-ai738.
set -u
# Genie Sim's num_episode is per task instance: sorting_packages has 10 instances
# (llm_task/sorting_packages/0..9), so EP=1 already means 10 episodes, one of each.
EP=${EP:-1}
OUT=${OUT:-$HOME/g2sim/runs/matrix_$(date +%Y%m%d)}
SORT=/workspace/source/geniesim_benchmark/src/geniesim_benchmark/config/g2op_manip_sorting_packages.yaml
EVAL_DIR=$HOME/g2sim/genie_sim/output/benchmark/warehouse_g2_op/sorting_packages
CKPT=$HOME/g2sim/ckpt/checkpoints/manipulation_pi05
RUN_TIMEOUT=${RUN_TIMEOUT:-5400}
mkdir -p "$OUT"

log() { echo "$(date +%F_%T) $*" | tee -a "$OUT/progress.log"; }

wait_port() {  # port -> 0 once listening, 1 after ~10 min
  for _ in $(seq 300); do ss -ltn | grep -q ":$1 " && return 0; sleep 2; done
  return 1
}

stop_port() {  # kill whatever listens on the port, by PID
  local pid
  pid=$(ss -ltnp | grep ":$1 " | grep -o "pid=[0-9]*" | head -1 | cut -d= -f2)
  [ -n "$pid" ] && kill "$pid"
  for _ in $(seq 30); do ss -ltn | grep -q ":$1 " || return 0; sleep 1; done
  [ -n "$pid" ] && kill -9 "$pid"
}

stop_sim() {  # leftover benchmark processes inside geniesim3, by PID
  local pids
  pids=$(docker exec geniesim3 pgrep -f "app/app.py|run_rtc.py|run_rtc_stream.py" | tr "\n" " ")
  [ -n "${pids// /}" ] && docker exec geniesim3 kill $pids
  sleep 5
}

start_old_server() {  # name
  PORT=8000 setsid nohup bash "$HOME/g2sim/bin/serve_policy.sh" > "$OUT/$1_server.log" 2>&1 < /dev/null &
  wait_port 8000
}

start_stream_server() {  # name, extra serve_rtc_stream flags...
  local name=$1; shift
  (
    cd "$HOME/g2sim/ACoT-VLA-ai738" || exit 1
    PYTHONPATH=$PWD/src:$PWD/packages/openpi-client/src XLA_PYTHON_CLIENT_MEM_FRACTION=0.48 \
      setsid nohup "$HOME/g2sim/ACoT-VLA/.venv/bin/python" scripts/serve_rtc_stream.py \
      --config pi05_genie_sim_manip_20260613 --dir "$CKPT" --port 8001 --rtc.clock external \
      --rtc.trace "$OUT/${name}_server.jsonl" "$@" > "$OUT/${name}_server.log" 2>&1 < /dev/null &
  )
  wait_port 8001
}

run() {  # name kind [flags...]
  local name=$1 kind=$2; shift 2
  local before after rc port
  before=$(ls -t "$EVAL_DIR"/evaluate_ret_*.json 2>/dev/null | head -1)
  log "START $name ($kind $*)"
  case $kind in
    pause)  port=8000; start_old_server "$name" || { log "FAIL $name: server never listened"; stop_port 8000; return; }
            timeout "$RUN_TIMEOUT" bash "$HOME/g2sim/bin/sim_ft.sh" "$name" "$EP" sorting_packages > /dev/null 2>&1 ;;
    oldrtc) port=8000; start_old_server "$name" || { log "FAIL $name: server never listened"; stop_port 8000; return; }
            CFG=$SORT timeout "$RUN_TIMEOUT" bash "$HOME/g2sim/bin/sim_rtc.sh" "$name" "$EP" "$@" > /dev/null 2>&1 ;;
    stream) port=8001; start_stream_server "$name" "$@" || { log "FAIL $name: server never listened"; stop_port 8001; return; }
            CFG=$SORT timeout "$RUN_TIMEOUT" bash "$HOME/g2sim/bin/sim_rtc_stream.sh" "$name" "$EP" > /dev/null 2>&1 ;;
    *) log "unknown kind $kind"; return ;;
  esac
  rc=$?
  stop_sim
  stop_port "$port"
  after=$(ls -t "$EVAL_DIR"/evaluate_ret_*.json 2>/dev/null | head -1)
  if [ -n "$after" ] && [ "$after" != "$before" ]; then
    cp "$after" "$OUT/${name}_eval.json"
    log "END $name rc=$rc eval=$(basename "$after")"
  else
    log "END $name rc=$rc eval=none"
  fi
}

declare -A VARIANTS=(
  [pause]="pause"
  [oldrtc_h20]="oldrtc --execution-horizon 20"
  [oldrtc_h25]="oldrtc --execution-horizon 25"
  [oldrtc_noguide_h20]="oldrtc --execution-horizon 20 --no-guidance"
  [new_t15_exp_fresh]="stream --rtc.queue-threshold 15"
  [new_t15_linear_fresh]="stream --rtc.queue-threshold 15 --rtc.prefix-attention-schedule linear"
  [new_t15_exp_fixed]="stream --rtc.queue-threshold 15 --no-fresh-noise"
  [new_t10_exp_fresh]="stream --rtc.queue-threshold 10"
  [new_t20_exp_fresh]="stream --rtc.queue-threshold 20"
)
ORDER=(new_t15_exp_fresh pause oldrtc_h20 oldrtc_h25 oldrtc_noguide_h20
       new_t15_linear_fresh new_t15_exp_fixed new_t10_exp_fresh new_t20_exp_fresh)
[ $# -gt 0 ] && ORDER=("$@")

log "matrix: EP=$EP OUT=$OUT variants=${ORDER[*]}"
stop_sim
for v in "${ORDER[@]}"; do
  [ -n "${VARIANTS[$v]:-}" ] || { log "unknown variant $v"; continue; }
  # shellcheck disable=SC2086
  run "$v" ${VARIANTS[$v]}
done
log "matrix done"
