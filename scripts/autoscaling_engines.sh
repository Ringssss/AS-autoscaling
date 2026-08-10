#!/usr/bin/env bash
set -euo pipefail

ACTION=${1:-}
PROFILE=${2:-}
ROOT=/home/zhujianian/agentshift
SGLANG_ROOT=/home/zhujianian/agentshift-sglang
PYTHON=/home/zhujianian/miniconda3/envs/sglang-bench/bin/python
STATE_DIR="/tmp/agentshift-autoscaling-${PROFILE}"
LOG_ROOT="$ROOT/results/autoscaling/servers/${PROFILE}"

usage() {
  echo "usage: $0 {start|stop|status} {qwen8b|qwen32b}" >&2
  exit 2
}

[[ -n "$ACTION" && -n "$PROFILE" ]] || usage

case "$PROFILE" in
  qwen8b)
    MODEL=/mnt/models/Qwen3-8B-sglang-tp2
    PORT_BASE=32000
    TP_SIZE=2
    ENGINE_COUNT=4
    MEM_FRACTION=0.82
    MAX_TOKENS=270000
    LOAD_ARGS=(
      --load-format sharded_state
      --model-loader-extra-config '{"pattern":"model-rank-{rank}-part-{part}.safetensors"}'
    )
    ;;
  qwen32b)
    MODEL=/mnt/models/Qwen3-32B
    PORT_BASE=32100
    TP_SIZE=4
    ENGINE_COUNT=2
    MEM_FRACTION=0.55
    MAX_TOKENS=45000
    LOAD_ARGS=()
    ;;
  *) usage ;;
esac

status() {
  local alive=0
  for index in $(seq 0 $((ENGINE_COUNT - 1))); do
    local pid_file="$STATE_DIR/engine-${index}.pid"
    if [[ -f "$pid_file" ]] && kill -0 "$(<"$pid_file")" 2>/dev/null; then
      echo "engine-${index} pid=$(<"$pid_file") port=$((PORT_BASE + index)) alive"
      alive=$((alive + 1))
    else
      echo "engine-${index} port=$((PORT_BASE + index)) stopped"
    fi
  done
  [[ "$alive" -eq "$ENGINE_COUNT" ]]
}

stop() {
  if [[ ! -d "$STATE_DIR" ]]; then
    return
  fi
  for pid_file in "$STATE_DIR"/*.pid; do
    [[ -f "$pid_file" ]] || continue
    pid=$(<"$pid_file")
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  for _ in $(seq 1 60); do
    remaining=0
    for pid_file in "$STATE_DIR"/*.pid; do
      [[ -f "$pid_file" ]] || continue
      kill -0 "$(<"$pid_file")" 2>/dev/null && remaining=$((remaining + 1))
    done
    [[ "$remaining" -eq 0 ]] && return
    sleep 1
  done
  for pid_file in "$STATE_DIR"/*.pid; do
    [[ -f "$pid_file" ]] || continue
    pid=$(<"$pid_file")
    kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
  done
}

start() {
  if [[ -d "$STATE_DIR" ]] && status >/dev/null 2>&1; then
    echo "$PROFILE engines are already running" >&2
    exit 1
  fi
  mkdir -p "$STATE_DIR" "$LOG_ROOT"
  run_stamp=$(date +%Y%m%d-%H%M%S)
  for index in $(seq 0 $((ENGINE_COUNT - 1))); do
    port=$((PORT_BASE + index))
    first=$((index * TP_SIZE))
    devices=$(seq -s, "$first" $((first + TP_SIZE - 1)))
    log="$LOG_ROOT/${run_stamp}-engine-${index}.log"
    setsid env \
      CUDA_VISIBLE_DEVICES="$devices" \
      PYTHONPATH="$SGLANG_ROOT/python" \
      TOKENIZERS_PARALLELISM=false \
      "$PYTHON" -m sglang.launch_server \
      --model-path "$MODEL" \
      --host 127.0.0.1 \
      --port "$port" \
      --tp-size "$TP_SIZE" \
      --mem-fraction-static "$MEM_FRACTION" \
      --max-total-tokens "$MAX_TOKENS" \
      --page-size 1 \
      --chunked-prefill-size 4096 \
      --disable-cuda-graph \
      --disable-piecewise-cuda-graph \
      "${LOAD_ARGS[@]}" \
      --log-level warning \
      >"$log" 2>&1 < /dev/null &
    echo "$!" >"$STATE_DIR/engine-${index}.pid"
    echo "$log" >"$STATE_DIR/engine-${index}.log"
  done

  deadline=$((SECONDS + 900))
  for index in $(seq 0 $((ENGINE_COUNT - 1))); do
    port=$((PORT_BASE + index))
    while ! curl -fsS --max-time 2 "http://127.0.0.1:${port}/health" >/dev/null; do
      pid=$(<"$STATE_DIR/engine-${index}.pid")
      if ! kill -0 "$pid" 2>/dev/null; then
        echo "engine-${index} exited; log: $(<"$STATE_DIR/engine-${index}.log")" >&2
        exit 1
      fi
      if (( SECONDS >= deadline )); then
        echo "timed out waiting for engine-${index}; log: $(<"$STATE_DIR/engine-${index}.log")" >&2
        exit 1
      fi
      sleep 2
    done
    echo "engine-${index} ready on port ${port}"
  done
}

case "$ACTION" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  *) usage ;;
esac
