#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/noslab-gpu/tkdgjs/qlm}"

# --- common paths ---
BASE_EXP_DIR="${BASE_EXP_DIR:-${PROJECT_ROOT}}"
PY="${PY:-/home/noslab-gpu/tkdgjs/tkdgjs/bin/python}"

if [[ ! -d "$PROJECT_ROOT" ]]; then
  echo "ERROR: PROJECT_ROOT does not exist: $PROJECT_ROOT" >&2
  exit 1
fi

cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT"
export QLMPROJDIR="$PROJECT_ROOT"

# Install editable package with the intended Python
"$PY" -m pip install -e .

# --- common env (same for all 6 runs) ---
export VLLM_GPU_MEMORY_UTILIZATION=0.9
export VLLM_MAX_NUM_SEQS=16
export VLLM_MAX_NUM_BATCHED_TOKENS=8192
export VLLM_MAX_MODEL_LEN=1280
export VLLM_EXTRA_ARGS="--enforce-eager"
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
export VLLM_ENABLE_CHUNKED_PREFILL=1
export VLLM_DTYPE=half

export RUN_TIME_S=600
export SLEEPS=0.01
export PUSH_INTERVAL_S=0.01
export PROMPT_POOL_LIMIT=10000
export PROMPT_MIN_TOKENS=512
export MAX_INPUT_TOKENS=1024
export EXP_SLEEP=30

VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_STABILIZE_SLEEP="${VLLM_STABILIZE_SLEEP:-300}"
VLLM_START_SCRIPT="${VLLM_START_SCRIPT:-${PROJECT_ROOT}/qlm/endpoints/start_vllm.sh}"
VLLM_MODEL="${VLLM_MODEL:-}"
CURRENT_VLLM_PID=""

# --- per-run configs (EDIT THESE 6 ITEMS AS YOU WANT) ---
# RESULT_DIR suffix label (used in folder name)
RUN_LABELS=(
  "native_dram"
  "native_disk"
  "native_dram_disk"
  "cachegen_dram"
  "cachegen_disk"
  "cachegen_dram_disk"
)

# LMCACHE_CONFIG_FILE for each run
LMCACHE_CONFIG_FILES=(
  "${PROJECT_ROOT}/lmcache_native_dram.yaml"
  "${PROJECT_ROOT}/lmcache_native_disk.yaml"
  "${PROJECT_ROOT}/lmcache_native_dram_disk.yaml"
  "${PROJECT_ROOT}/lmcache_cachegen_dram.yaml"
  "${PROJECT_ROOT}/lmcache_cachegen_disk.yaml"
  "${PROJECT_ROOT}/lmcache_cachegen_dram_disk.yaml"
)

# benchmark script path for each run
# (same benchmark repeated 6 times; replace per-run if needed)
BENCH_SCRIPTS=(
  "benchmarks/hol_cachegen_patched.py"
  "benchmarks/hol_cachegen_patched.py"
  "benchmarks/hol_cachegen_patched.py"
  "benchmarks/hol_cachegen_patched.py"
  "benchmarks/hol_cachegen_patched.py"
  "benchmarks/hol_cachegen_patched.py"
)

# --- sanity checks ---
if [[ ! -x "$PY" ]]; then
  echo "ERROR: PY is not executable: $PY" >&2
  exit 1
fi

if [[ ${#RUN_LABELS[@]} -ne 6 || ${#LMCACHE_CONFIG_FILES[@]} -ne 6 || ${#BENCH_SCRIPTS[@]} -ne 6 ]]; then
  echo "ERROR: You must provide exactly 6 entries for RUN_LABELS / LMCACHE_CONFIG_FILES / BENCH_SCRIPTS" >&2
  echo "  RUN_LABELS=${#RUN_LABELS[@]}, LMCACHE_CONFIG_FILES=${#LMCACHE_CONFIG_FILES[@]}, BENCH_SCRIPTS=${#BENCH_SCRIPTS[@]}" >&2
  exit 1
fi

for f in "${LMCACHE_CONFIG_FILES[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: Missing config file: $f" >&2
    exit 1
  fi
done

for b in "${BENCH_SCRIPTS[@]}"; do
  if [[ ! -f "$b" ]]; then
    echo "ERROR: Missing benchmark script: $b" >&2
    exit 1
  fi
done

extract_local_disk_path() {
  local config_file="$1"
  local line=""
  local value=""

  while IFS= read -r line; do
    line="${line%%#*}"
    line="${line#${line%%[![:space:]]*}}"
    [[ -z "$line" ]] && continue
    [[ "$line" != local_disk:* ]] && continue

    value="${line#local_disk:}"
    value="${value#${value%%[![:space:]]*}}"
    value="${value%${value##*[![:space:]]}}"
    value="${value%\"}"
    value="${value#\"}"
    value="${value%\'}"
    value="${value#\'}"

    if [[ -z "$value" || "$value" == "null" || "$value" == "None" ]]; then
      return 1
    fi

    if [[ "$value" == file://* ]]; then
      value="${value#file://}"
    fi

    value="${value%/}"
    [[ -z "$value" ]] && return 1

    printf '%s\n' "$value"
    return 0
  done < "$config_file"

  return 1
}

kill_vllm_on_port() {
  local pids=()
  if command -v lsof >/dev/null 2>&1; then
    mapfile -t pids < <(lsof -tiTCP:"${VLLM_PORT}" -sTCP:LISTEN 2>/dev/null || true)
  elif command -v fuser >/dev/null 2>&1; then
    mapfile -t pids < <(fuser -n tcp "${VLLM_PORT}" 2>/dev/null | tr ' ' '\n' | grep -E '^[0-9]+$' || true)
  fi

  if [[ ${#pids[@]} -gt 0 ]]; then
    kill -TERM "${pids[@]}" 2>/dev/null || true
    sleep 2
    kill -KILL "${pids[@]}" 2>/dev/null || true
  fi

  pkill -u "$(id -u)" -TERM -f 'vllm|api_server|uvicorn' 2>/dev/null || true
  sleep 2
  pkill -u "$(id -u)" -KILL -f 'vllm|api_server|uvicorn' 2>/dev/null || true
}

stop_vllm_instance() {
  if [[ -n "${CURRENT_VLLM_PID}" ]] && kill -0 "${CURRENT_VLLM_PID}" 2>/dev/null; then
    kill -TERM "${CURRENT_VLLM_PID}" 2>/dev/null || true
    sleep 2
    kill -KILL "${CURRENT_VLLM_PID}" 2>/dev/null || true
  fi
  CURRENT_VLLM_PID=""
  kill_vllm_on_port
}

start_vllm_instance() {
  local vllm_log_file="$1"

  if [[ ! -f "${VLLM_START_SCRIPT}" ]]; then
    echo "ERROR: Missing VLLM_START_SCRIPT: ${VLLM_START_SCRIPT}" >&2
    exit 1
  fi

  stop_vllm_instance

  local cmd=("bash" "${VLLM_START_SCRIPT}" --port "${VLLM_PORT}")
  if [[ -n "${VLLM_MODEL}" ]]; then
    cmd+=(--model "${VLLM_MODEL}")
  fi

  "${cmd[@]}" >"${vllm_log_file}" 2>&1 &
  CURRENT_VLLM_PID=$!

  if ! kill -0 "${CURRENT_VLLM_PID}" 2>/dev/null; then
    echo "ERROR: Failed to start vLLM process" >&2
    exit 1
  fi
}

trap stop_vllm_instance EXIT

for f in "${LMCACHE_CONFIG_FILES[@]}"; do
  disk_path="$(extract_local_disk_path "$f" || true)"
  [[ -z "$disk_path" ]] && continue

  mkdir -p "$disk_path"
  probe_file="${disk_path}/.lmcache_write_probe.$$"
  if ! : > "$probe_file"; then
    echo "ERROR: local_disk is not writable (from $f): $disk_path" >&2
    exit 1
  fi
  rm -f "$probe_file"

  echo "[preflight] local_disk ready: $disk_path"
done

# --- run loop ---
for i in "${!RUN_LABELS[@]}"; do
  ts="$(date +%Y%m%d_%H%M%S)"
  RESULT_DIR="${BASE_EXP_DIR}/result_${ts}_${RUN_LABELS[$i]}"
  mkdir -p "$RESULT_DIR"

  export LMCACHE_CONFIG_FILE="${LMCACHE_CONFIG_FILES[$i]}"
  export LMCACHE_RESULT_DIR="$RESULT_DIR"

  LOG_FILE="$RESULT_DIR/run_${RUN_LABELS[$i]}.log"

  {
    echo "============================================================"
    echo "RUN $((i + 1))/6  |  $(date)"
    echo "RESULT_DIR=$RESULT_DIR"
    echo "LMCACHE_CONFIG_FILE=$LMCACHE_CONFIG_FILE"
    echo "BENCH_SCRIPT=${BENCH_SCRIPTS[$i]}"
    echo "PY=$PY"
    echo "============================================================"
  } | tee -a "$LOG_FILE"

  VLLM_LOG_FILE="$RESULT_DIR/vllm_${RUN_LABELS[$i]}.log"
  echo "Starting vLLM for RUN $((i + 1))/6 on port ${VLLM_PORT}" | tee -a "$LOG_FILE"
  start_vllm_instance "$VLLM_LOG_FILE"
  echo "Stabilizing vLLM for ${VLLM_STABILIZE_SLEEP}s before benchmark" | tee -a "$LOG_FILE"
  sleep "$VLLM_STABILIZE_SLEEP"

  # Run (pipefail makes python nonzero exit stop the script)
  "$PY" -u "${BENCH_SCRIPTS[$i]}" 2>&1 | tee -a "$LOG_FILE"

  echo "DONE RUN $((i + 1))/6  |  $(date)" | tee -a "$LOG_FILE"
  stop_vllm_instance
  sleep "$EXP_SLEEP"
done

echo "ALL RUNS COMPLETED  |  $(date)"
