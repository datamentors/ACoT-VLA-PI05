#!/bin/bash
# run_matrix.sh [run-id ...]
#   Genie Sim A/B of pause-per-inference, the client-side RTC (rtc_client_policy.py +
#   rtc_infer) and the server-side RTC stream (serve_rtc_stream.py) over tasks and
#   checkpoints. Both RTCs get the same parameter variants:
#
#     base     re-infer at 15 rows left, H=20, EXP schedule, fresh noise, guided
#     linear   base with the linear schedule
#     fixed    base with one fixed noise key
#     t30      base re-inferring at 30 rows left (continuous refills)
#     noguide  base without guidance (old: no leftover sent; new: guidance weight 0)
#     h25      base with H=25
#   plus old_asran (the client-side RTC as originally run: T30, H20, linear, fixed key)
#   and pause (stock CoRobotPolicy).
#
#   Run ids are task__model__variant. Every run: start its server, wait for the port,
#   run num_episode=1 (one episode per task instance: 10 for sorting_packages, 20 for
#   open_door), stop the sim and the server by PID, copy the new evaluate_ret_*.json to
#   $OUT/<run-id>_eval.json and the sim log to $OUT/<run-id>.log. Runs already present in
#   $OUT (an _eval.json exists) are skipped, so the script can be restarted.
#   All servers run from the ai-738 branch copy ~/g2sim/ACoT-VLA-ai738.
set -u
OUT=${OUT:-$HOME/g2sim/runs/matrix_20261008}
EP=${EP:-1}
RUN_TIMEOUT=${RUN_TIMEOUT:-5400}
CODE=$HOME/g2sim/ACoT-VLA-ai738
PY=$HOME/g2sim/ACoT-VLA/.venv/bin/python
HOST_IP=$(hostname -I | awk '{print $1}')
mkdir -p "$OUT"

log() { echo "$(date +%F_%T) $*" | tee -a "$OUT/progress.log"; }

task_cfg() {  # task -> benchmark yaml (container path)
  echo "/workspace/source/geniesim_benchmark/src/geniesim_benchmark/config/g2op_manip_$1.yaml"
}
task_eval_dir() {
  case $1 in
    sorting_packages) echo "$HOME/g2sim/genie_sim/output/benchmark/warehouse_g2_op/sorting_packages" ;;
    open_door) echo "$HOME/g2sim/genie_sim/output/benchmark/home_g2_op/open_door" ;;
  esac
}
model_config() {
  case $1 in
    manip) echo "pi05_genie_sim_manip_20260613" ;;
    ipw) echo "pi05_g2_ipw_vision_action" ;;
  esac
}
model_dir() {
  case $1 in
    manip) echo "$HOME/g2sim/ckpt/checkpoints/manipulation_pi05" ;;
    ipw) echo "$HOME/g2sim/ACoT-VLA/checkpoints/pi05_g2_ipw_vision_action/InfPoseWaistVisionGcmd/56000" ;;
  esac
}
# Waist like the existing paths: manipulation_pi05 holds dims 16:19 on sorting_packages
# (Policy.post_process) and sends no waist on other tasks; the waist fine-tune takes all
# five waist dims from the model.
stream_waist() {  # model task
  if [ "$1" = ipw ]; then echo model; elif [ "$2" = sorting_packages ]; then echo hold; else echo off; fi
}
old_waist_env() {  # model -> G2SIM_FULL_WAIST_TASKS for Policy.post_process
  [ "$1" = ipw ] && echo "sorting_packages,open_door" || echo ""
}

wait_port() { for _ in $(seq 300); do ss -ltn | grep -q ":$1 " && return 0; sleep 2; done; return 1; }
stop_port() {
  local pid
  pid=$(ss -ltnp | grep ":$1 " | grep -o "pid=[0-9]*" | head -1 | cut -d= -f2)
  [ -n "$pid" ] && kill "$pid"
  for _ in $(seq 30); do ss -ltn | grep -q ":$1 " || return 0; sleep 1; done
  [ -n "$pid" ] && kill -9 "$pid"
}
stop_sim() {
  local pids
  pids=$(docker exec geniesim3 pgrep -f "app/app.py|run_rtc.py|run_rtc_client.py|run_rtc_stream.py" | tr "\n" " ")
  [ -n "${pids// /}" ] && docker exec geniesim3 kill $pids
  sleep 5
}

start_old_server() {  # id model extra-server-flags...
  local id=$1 model=$2; shift 2
  (
    cd "$CODE" || exit 1
    export PYTHONPATH=$PWD/src:$PWD/packages/openpi-client/src
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.48 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
    export JAX_COMPILATION_CACHE_DIR=$HOME/g2sim/xla-cache
    export G2SIM_FULL_WAIST_TASKS
    G2SIM_FULL_WAIST_TASKS=$(old_waist_env "$model")
    setsid nohup "$PY" scripts/serve_policy.py --port 8000 "$@" policy:checkpoint \
      --policy.config "$(model_config "$model")" --policy.dir "$(model_dir "$model")" \
      > "$OUT/${id}_server.log" 2>&1 < /dev/null &
  )
  wait_port 8000
}

start_stream_server() {  # id model task extra-server-flags...
  local id=$1 model=$2 task=$3; shift 3
  (
    cd "$CODE" || exit 1
    PYTHONPATH=$PWD/src:$PWD/packages/openpi-client/src XLA_PYTHON_CLIENT_MEM_FRACTION=0.48 \
      setsid nohup "$PY" scripts/serve_rtc_stream.py \
      --config "$(model_config "$model")" --dir "$(model_dir "$model")" --port 8001 --rtc.clock external \
      --rtc.waist "$(stream_waist "$model" "$task")" --rtc.trace "$OUT/${id}_server.jsonl" "$@" \
      > "$OUT/${id}_server.log" 2>&1 < /dev/null &
  )
  wait_port 8001
}

sim_in_container() {  # id launcher-command-with-args
  docker exec -e DISPLAY=:1 geniesim3 bash -lc "cd /workspace/harness && $2" > "$OUT/$1.log" 2>&1
}

# variant -> "kind|server flags|client flags"
variant_spec() {
  case $1 in
    pause)       echo "pause||" ;;
    old_asran)   echo "old||--queue-threshold 30 --execution-horizon 20" ;;
    old_base)    echo "old|--fresh-noise|--queue-threshold 15 --execution-horizon 20 --prefix-attention-schedule exp" ;;
    old_linear)  echo "old|--fresh-noise|--queue-threshold 15 --execution-horizon 20 --prefix-attention-schedule linear" ;;
    old_fixed)   echo "old||--queue-threshold 15 --execution-horizon 20 --prefix-attention-schedule exp" ;;
    old_t30)     echo "old|--fresh-noise|--queue-threshold 30 --execution-horizon 20 --prefix-attention-schedule exp" ;;
    old_noguide) echo "old|--fresh-noise|--queue-threshold 15 --execution-horizon 20 --no-guidance" ;;
    old_h25)     echo "old|--fresh-noise|--queue-threshold 15 --execution-horizon 25 --prefix-attention-schedule exp" ;;
    new_base)    echo "new|--rtc.queue-threshold 15|" ;;
    new_linear)  echo "new|--rtc.queue-threshold 15 --rtc.prefix-attention-schedule linear|" ;;
    new_fixed)   echo "new|--rtc.queue-threshold 15 --no-fresh-noise|" ;;
    new_t30)     echo "new|--rtc.queue-threshold 30|" ;;
    new_noguide) echo "new|--rtc.queue-threshold 15 --rtc.max-guidance-weight 0|" ;;
    new_h25)     echo "new|--rtc.queue-threshold 15 --rtc.execution-horizon 25|" ;;
  esac
}

run() {  # task__model__variant
  local id=$1 task model rest variant spec kind sflags cflags edir before after port cfg rc
  task=${id%%__*}; rest=${id#*__}; model=${rest%%__*}; variant=${rest#*__}
  [ -f "$OUT/${id}_eval.json" ] && { log "SKIP $id (already scored)"; return; }
  spec=$(variant_spec "$variant")
  [ -n "$spec" ] || { log "unknown variant $variant"; return; }
  IFS="|" read -r kind sflags cflags <<< "$spec"
  edir=$(task_eval_dir "$task"); cfg=$(task_cfg "$task")
  before=$(ls -t "$edir"/evaluate_ret_*.json 2>/dev/null | head -1)
  log "START $id ($kind server[$sflags] client[$cflags])"
  local app="--config $cfg --benchmark.num_episode=$EP --benchmark.record=true"
  case $kind in
    pause)
      port=8000; start_old_server "$id" "$model" || { log "FAIL $id: server never listened"; stop_port 8000; return; }
      docker exec -e DISPLAY=:1 geniesim3 bash -lc \
        "cd /workspace && G2SIM_RENDER_TIMEOUT=600 G2SIM_ACTION_TRACE=/workspace/runs/${id}_act.jsonl timeout $RUN_TIMEOUT python3 source/geniesim_benchmark/src/geniesim_benchmark/app/app.py $app --benchmark.infer_host=$HOST_IP:8000" \
        > "$OUT/$id.log" 2>&1 ;;
    old)
      port=8000
      # shellcheck disable=SC2086
      start_old_server "$id" "$model" $sflags || { log "FAIL $id: server never listened"; stop_port 8000; return; }
      sim_in_container "$id" "timeout $RUN_TIMEOUT python3 run_rtc_client.py $cflags --trace /workspace/runs/$id.jsonl -- $app --benchmark.infer_host=$HOST_IP:8000" ;;
    new)
      port=8001
      # shellcheck disable=SC2086
      start_stream_server "$id" "$model" "$task" $sflags || { log "FAIL $id: server never listened"; stop_port 8001; return; }
      sim_in_container "$id" "timeout $RUN_TIMEOUT python3 run_rtc_stream.py --trace /workspace/runs/$id.jsonl -- $app --benchmark.infer_host=$HOST_IP:8001" ;;
  esac
  rc=$?
  stop_sim
  stop_port "$port"
  after=$(ls -t "$edir"/evaluate_ret_*.json 2>/dev/null | head -1)
  if [ -n "$after" ] && [ "$after" != "$before" ]; then
    cp "$after" "$OUT/${id}_eval.json"
    log "END $id rc=$rc eval=$(basename "$after")"
  else
    log "END $id rc=$rc eval=none"
  fi
}

RTC=(base linear fixed t30 noguide h25)
ALL=()
for tm in sorting_packages__manip open_door__manip sorting_packages__ipw open_door__ipw; do
  ALL+=("${tm}__pause" "${tm}__old_asran")
  for v in "${RTC[@]}"; do ALL+=("${tm}__old_$v" "${tm}__new_$v"); done
done
[ $# -gt 0 ] && ALL=("$@")

log "matrix: OUT=$OUT EP=$EP runs=${#ALL[@]}"
stop_sim
for id in "${ALL[@]}"; do run "$id"; done
log "matrix done"
