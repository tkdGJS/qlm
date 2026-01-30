#!/usr/bin/env bash
set -euo pipefail

OPTIONS=$(getopt -o m:p:h --long model:,port:,help -- "$@") || exit 1
eval set -- "$OPTIONS"

MODEL="meta-llama/Llama-3.2-1B-Instruct"
PORT="8000"

while true; do
  case "$1" in
  -m | --model)
    MODEL="$2"
    shift 2
    ;;
  -p | --port)
    PORT="$2"
    shift 2
    ;;
  -h | --help)
    echo "Usage: $0 --model <model_name> --port <port_number>"
    echo
    echo "Env overrides:"
    echo "  VLLM_DTYPE=half|bfloat16|float16..."
    echo "  VLLM_MAX_MODEL_LEN=32768"
    echo "  VLLM_MAX_NUM_SEQS=256"
    echo "  VLLM_MAX_NUM_BATCHED_TOKENS=131072"
    echo "  VLLM_GPU_MEMORY_UTILIZATION=0.9"
    echo "  VLLM_PREEMPTION_MODE=swap|recompute"
    echo "  VLLM_SWAP_SPACE=16"
    echo "  VLLM_SCHEDULING_POLICY=priority|fcfs"
    echo "  VLLM_ENABLE_CHUNKED_PREFILL=1|0   (default=1)"
    echo "  VLLM_EXTRA_ARGS='...'"
    exit 0
    ;;
  --)
    shift
    break
    ;;
  *)
    echo "Unexpected option: $1"
    exit 1
    ;;
  esac
done

# ---- env override 적용 ----
DTYPE="${VLLM_DTYPE:-half}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-128}"
MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-131072}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
#PREEMPTION_MODE="${VLLM_PREEMPTION_MODE:-swap}"
PREEMPTION_MODE="${VLLM_PREEMPTION_MODE:-None}"
SWAP_SPACE="${VLLM_SWAP_SPACE:-16}"
SCHEDULING_POLICY="${VLLM_SCHEDULING_POLICY:-priority}"
#SCHEDULING_POLICY="${VLLM_SCHEDULING_POLICY:-fcfs}"
#ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL:-1}"
ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL:-${CHUNKED_PREFILL:-1}}"
EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

CHUNK_FLAG=""
if [[ "${ENABLE_CHUNKED_PREFILL}" == "1" ]]; then
  CHUNK_FLAG="--enable-chunked-prefill"
fi

echo "[start_vllm] model=${MODEL} port=${PORT}"
echo "[start_vllm] dtype=${DTYPE}"
echo "[start_vllm] max_model_len=${MAX_MODEL_LEN}"
echo "[start_vllm] max_num_seqs=${MAX_NUM_SEQS}"
echo "[start_vllm] max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}"
echo "[start_vllm] gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
echo "[start_vllm] preemption_mode=${PREEMPTION_MODE} swap_space=${SWAP_SPACE}"
echo "[start_vllm] scheduling_policy=${SCHEDULING_POLICY}"
echo "[start_vllm] chunked_prefill=${ENABLE_CHUNKED_PREFILL}"
echo "[start_vllm] extra_args=${EXTRA_ARGS}"

exec vllm serve "${MODEL}" \
  --port "${PORT}" \
  --dtype="${DTYPE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --preemption-mode "${PREEMPTION_MODE}" \
  --swap-space "${SWAP_SPACE}" \
  --scheduling-policy "${SCHEDULING_POLICY}" \
  ${CHUNK_FLAG} \
  ${EXTRA_ARGS}
