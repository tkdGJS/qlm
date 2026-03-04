#!/usr/bin/env bash
set -euo pipefail

# --- common paths ---
BASE_EXP_DIR="/home/noslab-gpu/tkdgjs/qlm"
PY="/home/noslab-gpu/tkdgjs/tkdgjs/bin/python"

export PYTHONPATH="$(pwd)"
export QLMPROJDIR="$(pwd)"

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
  "/home/noslab-gpu/tkdgjs/qlm/lmcache_native_dram.yaml"
  "/home/noslab-gpu/tkdgjs/qlm/lmcache_native_disk.yaml"
  "/home/noslab-gpu/tkdgjs/qlm/lmcache_native_dram_disk.yaml"
  "/home/noslab-gpu/tkdgjs/qlm/lmcache_cachegen_dram.yaml"
  "/home/noslab-gpu/tkdgjs/qlm/lmcache_cachegen_disk.yaml"
  "/home/noslab-gpu/tkdgjs/qlm/lmcache_cachegen_dram_disk.yaml"
)

# benchmark script path for each run
# (same benchmark repeated 6 times; replace per-run if needed)
BENCH_SCRIPTS=(
  "benchmarks/hol_cachegen.py"
  "benchmarks/hol_cachegen.py"
  "benchmarks/hol_cachegen.py"
  "benchmarks/hol_cachegen.py"
  "benchmarks/hol_cachegen.py"
  "benchmarks/hol_cachegen.py"
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

# --- run loop ---
for i in "${!RUN_LABELS[@]}"; do
  ts="$(date +%Y%m%d_%H%M%S)"
  RESULT_DIR="${BASE_EXP_DIR}/result_${ts}_${RUN_LABELS[$i]}"
  mkdir -p "$RESULT_DIR"

  export LMCACHE_CONFIG_FILE="${LMCACHE_CONFIG_FILES[$i]}"
  export LMCACHE_RESULT_DIR="$RESULT_DIR"

  LOG_FILE="$RESULT_DIR/run.log"

  {
    echo "============================================================"
    echo "RUN $((i + 1))/6  |  $(date)"
    echo "RESULT_DIR=$RESULT_DIR"
    echo "LMCACHE_CONFIG_FILE=$LMCACHE_CONFIG_FILE"
    echo "BENCH_SCRIPT=${BENCH_SCRIPTS[$i]}"
    echo "PY=$PY"
    echo "============================================================"
  } | tee -a "$LOG_FILE"

  # Run (pipefail makes python nonzero exit stop the script)
  "$PY" -u "${BENCH_SCRIPTS[$i]}" 2>&1 | tee -a "$LOG_FILE"

  echo "DONE RUN $((i + 1))/6  |  $(date)" | tee -a "$LOG_FILE"
  sleep "$EXP_SLEEP"
done

echo "ALL RUNS COMPLETED  |  $(date)"
