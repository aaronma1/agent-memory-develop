#!/usr/bin/env bash
# M3-Bench local model services: start the three model types run.py needs with vLLM on an A100.
#
#   chat  : Qwen/Qwen3-VL-8B-Instruct     text + video understanding (memory build, answers)  :8000
#   asr   : openai/whisper-large-v3-turbo /v1/audio/transcriptions                            :8001
#   embed : BAAI/bge-m3                   /v1/embeddings (1024-dim)                           :8002
#
# Judging still uses gpt-4o-2024-11-20 on OpenRouter (the official M3-Agent setting); it is not
# deployed here.
#
# Usage:
#   serve.sh setup     create venv, install vLLM, download models (no GPU needed)
#   serve.sh           setup + start + self-test, print the .env snippet, then wait in the
#                      foreground; Ctrl-C stops all services
#   serve.sh test      rerun the self-test against running services
#
# GPU allocation:
#   - if CUDA_VISIBLE_DEVICES is set (e.g. by a job scheduler) only those GPUs are used;
#   - otherwise GPUS=0,3 picks them explicitly; with neither, it picks GPUs with no processes at
#     all, up to MAX_GPUS.
#   1 GPU (A100-40GB): all three services share it, chat 0.68, asr/embed 0.10 each.
#   N>=2 GPUs: the last one hosts asr+embed, the other N-1 run chat data-parallel.
#
# Common env vars (all have defaults):
#   M3_VLLM_HOME     venv, logs and pid files           ~/m3bench-vllm
#   HF_HOME          model cache                        ~/.cache/huggingface
#   HF_ENDPOINT      e.g. https://hf-mirror.com inside mainland China
#   VLLM_VERSION     pin vLLM (e.g. 0.11.0); empty installs the latest and logs the version
#   CHAT_MODEL / ASR_MODEL / EMBED_MODEL                swap models
#   CHAT_MAX_LEN     chat context length                32768 (16384 on a single GPU)
#   HOST             listen address                     127.0.0.1 (use an SSH tunnel, or 0.0.0.0)
#   PUBLIC_HOST      host name written to the .env snippet   same as HOST
#   VLLM_API_KEY     when set, all three services require this key
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
M3_VLLM_HOME="${M3_VLLM_HOME:-$HOME/m3bench-vllm}"
VENV="$M3_VLLM_HOME/.venv"
LOG_DIR="$M3_VLLM_HOME/logs"
PY="$VENV/bin/python"

CHAT_MODEL="${CHAT_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
ASR_MODEL="${ASR_MODEL:-openai/whisper-large-v3-turbo}"
EMBED_MODEL="${EMBED_MODEL:-BAAI/bge-m3}"
CHAT_NAME="${CHAT_NAME:-qwen3-vl-8b}"
ASR_NAME="${ASR_NAME:-whisper-large-v3-turbo}"
EMBED_NAME="${EMBED_NAME:-bge-m3}"
CHAT_PORT="${CHAT_PORT:-8000}"
ASR_PORT="${ASR_PORT:-8001}"
EMBED_PORT="${EMBED_PORT:-8002}"
HOST="${HOST:-127.0.0.1}"
PUBLIC_HOST="${PUBLIC_HOST:-$HOST}"
MAX_GPUS="${MAX_GPUS:-3}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1200}"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

# --------------------------------------------------------------------------- #
# setup: venv + vLLM + models
# --------------------------------------------------------------------------- #
setup() {
  command -v uv >/dev/null || die "uv is required (https://docs.astral.sh/uv/)"
  mkdir -p "$M3_VLLM_HOME" "$LOG_DIR"
  if [[ ! -x "$PY" ]]; then
    log "creating venv: $VENV"
    uv venv --python 3.12 "$VENV"
  fi
  local spec="vllm[audio]"
  [[ -n "${VLLM_VERSION:-}" ]] && spec="vllm[audio]==${VLLM_VERSION}"
  log "installing $spec"
  uv pip install --python "$PY" "$spec" "huggingface_hub[cli]" opencv-python-headless
  log "vLLM version: $("$PY" -c 'import vllm; print(vllm.__version__)')"
  local model
  for model in "$CHAT_MODEL" "$ASR_MODEL" "$EMBED_MODEL"; do
    log "downloading $model"
    "$PY" -c 'import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1]))' "$model"
  done
}

# --------------------------------------------------------------------------- #
# GPU selection
# --------------------------------------------------------------------------- #
busy_gpu_indices() {
  # Any GPU with a compute process counts as busy (on a shared box, low usage is still someone).
  local busy_uuids
  busy_uuids="$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | sort -u)"
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader | while IFS=', ' read -r idx uuid; do
    if grep -qxF "$uuid" <<<"$busy_uuids"; then echo "$idx"; fi
  done
}

pick_gpus() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    log "using assigned CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo "$CUDA_VISIBLE_DEVICES"
    return
  fi
  command -v nvidia-smi >/dev/null || die "nvidia-smi not found"
  local busy
  busy="$(busy_gpu_indices)"
  if [[ -n "${GPUS:-}" ]]; then
    local g
    for g in ${GPUS//,/ }; do
      if grep -qxF "$g" <<<"$busy"; then die "GPU $g has processes on it; it is not free"; fi
    done
    echo "$GPUS"
    return
  fi
  local free=()
  local idx
  while read -r idx; do
    grep -qxF "$idx" <<<"$busy" || free+=("$idx")
  done < <(nvidia-smi --query-gpu=index --format=csv,noheader)
  ((${#free[@]} > 0)) || die "no free GPU (every GPU has processes)"
  free=("${free[@]:0:$MAX_GPUS}")
  local joined
  joined="$(IFS=,; echo "${free[*]}")"
  log "auto-selected free GPUs: $joined"
  echo "$joined"
}

# --------------------------------------------------------------------------- #
# Start
# --------------------------------------------------------------------------- #
PIDS=()

serve_help() {
  # serve flags differ across vLLM versions (--task embed -> --runner pooling); probe the help.
  "$VENV/bin/vllm" serve --help=all 2>/dev/null || "$VENV/bin/vllm" serve --help 2>/dev/null
}

launch() {  # launch <name> <gpus> <log> -- vllm serve args...
  local name="$1" gpus="$2"
  shift 3
  local logfile="$LOG_DIR/$name.log"
  log "starting $name (GPU $gpus) -> $logfile"
  CUDA_VISIBLE_DEVICES="$gpus" "$VENV/bin/vllm" serve "$@" >"$logfile" 2>&1 &
  PIDS+=("$!")
  echo "$!" >"$LOG_DIR/$name.pid"
}

wait_ready() {  # wait_ready <name> <port>
  local name="$1" port="$2" deadline=$((SECONDS + STARTUP_TIMEOUT))
  until curl -sf "http://127.0.0.1:$port/health" >/dev/null; do
    local pid
    pid="$(cat "$LOG_DIR/$name.pid")"
    kill -0 "$pid" 2>/dev/null || { tail -n 40 "$LOG_DIR/$name.log" >&2; die "$name failed to start"; }
    ((SECONDS < deadline)) || die "$name not ready within ${STARTUP_TIMEOUT}s"
    sleep 5
  done
  log "$name ready (:$port)"
}

stop_all() {
  trap - EXIT INT TERM
  ((${#PIDS[@]} == 0)) && return 0
  log "stopping vLLM services"
  kill "${PIDS[@]}" 2>/dev/null || true
  wait "${PIDS[@]}" 2>/dev/null || true
}

start() {
  [[ -x "$VENV/bin/vllm" ]] || die "run $0 setup first"
  local gpus count chat_gpus aux_gpus chat_util aux_util chat_len
  gpus="$(pick_gpus)"
  IFS=',' read -r -a gpu_list <<<"$gpus"
  count="${#gpu_list[@]}"
  if ((count == 1)); then
    chat_gpus="$gpus"; aux_gpus="$gpus"
    chat_util=0.68; aux_util=0.10; chat_len="${CHAT_MAX_LEN:-16384}"
  else
    aux_gpus="${gpu_list[count-1]}"
    chat_gpus="$(IFS=,; echo "${gpu_list[*]:0:count-1}")"
    chat_util=0.90; aux_util=0.40; chat_len="${CHAT_MAX_LEN:-32768}"
  fi

  local auth=()
  [[ -n "${VLLM_API_KEY:-}" ]] && auth=(--api-key "$VLLM_API_KEY")
  local help embed_flags=()
  help="$(serve_help)"
  if grep -q -- '--runner' <<<"$help"; then embed_flags=(--runner pooling); else embed_flags=(--task embed); fi
  local mm_limit='{"image":0,"video":1}'

  trap stop_all EXIT
  trap 'stop_all; exit 130' INT TERM
  # chat starts first: on a shared GPU later services size against the remaining memory, so
  # start order decides who gets the budget first.
  local chat_dp=$(( $(tr -cd ',' <<<"$chat_gpus" | wc -c) + 1 ))
  launch chat "$chat_gpus" -- "$CHAT_MODEL" \
    --served-model-name "$CHAT_NAME" --host "$HOST" --port "$CHAT_PORT" \
    --dtype bfloat16 --max-model-len "$chat_len" --gpu-memory-utilization "$chat_util" \
    --data-parallel-size "$chat_dp" --limit-mm-per-prompt "$mm_limit" "${auth[@]}"
  wait_ready chat "$CHAT_PORT"
  launch asr "$aux_gpus" -- "$ASR_MODEL" \
    --served-model-name "$ASR_NAME" --host "$HOST" --port "$ASR_PORT" \
    --gpu-memory-utilization "$aux_util" "${auth[@]}"
  wait_ready asr "$ASR_PORT"
  launch embed "$aux_gpus" -- "$EMBED_MODEL" \
    --served-model-name "$EMBED_NAME" --host "$HOST" --port "$EMBED_PORT" \
    --gpu-memory-utilization "$aux_util" "${embed_flags[@]}" "${auth[@]}"
  wait_ready embed "$EMBED_PORT"
}

smoke_test() {
  "$PY" "$SCRIPT_DIR/smoke_test.py" \
    --chat "http://127.0.0.1:$CHAT_PORT/v1" --chat-model "$CHAT_NAME" \
    --asr "http://127.0.0.1:$ASR_PORT/v1" --asr-model "$ASR_NAME" \
    --embed "http://127.0.0.1:$EMBED_PORT/v1" --embed-model "$EMBED_NAME" \
    --api-key "${VLLM_API_KEY:-EMPTY}"
}

print_env() {
  local key="${VLLM_API_KEY:-EMPTY}"
  cat <<EOF

# ---- add to evaluation/environment/.env (judging stays on OpenRouter; keep OPENROUTER_API_KEY) ----
M3BENCH_LLM_BASE_URL=http://$PUBLIC_HOST:$CHAT_PORT/v1
M3BENCH_LLM_API_KEY=$key
M3BENCH_TEXT_MODEL=$CHAT_NAME
M3BENCH_VLM_MODEL=$CHAT_NAME
M3BENCH_ANSWER_MODEL=$CHAT_NAME
M3BENCH_ASR_BASE_URL=http://$PUBLIC_HOST:$ASR_PORT/v1
M3BENCH_ASR_MODEL=$ASR_NAME
M3BENCH_ASR_API_KEY=$key
EMBED_BASE_URL=http://$PUBLIC_HOST:$EMBED_PORT/v1
EMBED_MODEL_NAME=$EMBED_NAME
EMBED_API_KEY=$key
EMBEDDING_MODEL_DIMENTION=1024
EOF
}

case "${1:-up}" in
  setup) setup ;;
  test) smoke_test ;;
  up)
    setup
    start
    smoke_test
    print_env
    log "services running (logs: $LOG_DIR). Ctrl-C to stop."
    wait -n "${PIDS[@]}" || true
    die "a service exited unexpectedly; see $LOG_DIR"
    ;;
  *) die "unknown subcommand: $1 (available: setup | up | test)" ;;
esac
