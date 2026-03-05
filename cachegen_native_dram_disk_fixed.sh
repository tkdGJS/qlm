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
export VLLM_STARTUP_WAIT_S=300

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

  echo "Benchmark will start vLLM via Endpoint (VLLM_STARTUP_WAIT_S=${VLLM_STARTUP_WAIT_S}s)" | tee -a "$LOG_FILE"

  # Run (pipefail makes python nonzero exit stop the script)
  "$PY" -u "${BENCH_SCRIPTS[$i]}" 2>&1 | tee -a "$LOG_FILE"

  echo "DONE RUN $((i + 1))/6  |  $(date)" | tee -a "$LOG_FILE"
  sleep "$EXP_SLEEP"
done

echo "ALL RUNS COMPLETED  |  $(date)"
