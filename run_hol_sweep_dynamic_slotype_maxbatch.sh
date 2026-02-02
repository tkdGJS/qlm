#!/usr/bin/env bash
set -euo pipefail

# ====== 설정 ======
# sweep 할 값들 (원하는대로 수정)
#SLEEPS=(0.001 0.005 0.01 0.05 0.1)
#SLEEPS=(0.0001 0.001 0.01)
SLEEPS=(0.0001)

# VQ push 속도 sweep:
#  - MODE=rps : PUSH_RATE_RPS로 sweep (권장)
#  - MODE=interval : PUSH_INTERVAL_S로 sweep
PUSH_MODE="interval"
PUSH_RPS_LIST=(1 2 5 10 20 50) # PUSH_MODE=rps 일 때 사용
#PUSH_INTERVAL_LIST=(1.0 0.5 0.1 0.05 0.01 0.005 0.001) # PUSH_MODE=interval 일 때 사용
PUSH_INTERVAL_LIST=(0.001 0.01 0.1) # PUSH_MODE=interval 일 때 사용
#PUSH_INTERVAL_LIST=(0.01) # PUSH_MODE=interval 일 때 사용

# (추가) max_batch_size sweep: 20~120, 20씩
#MAX_BATCH_SIZES=(100 80 60 40 20)
MAX_BATCH_SIZES=(100 80 60)

#CHUNKED_PREFILL_VALUES=(0 1)
CHUNKED_PREFILL_VALUES=(1 0)
# push 종료 후 큐 드레인 대기(초) - 요청 처리 완료까지 기다리게 하려면 0보다 크게!
#DRAIN_TIMEOUT_S_DEFAULT=600
DRAIN_TIMEOUT_S_DEFAULT=30

SORT_ALGO="timsort"
SORT_PROFILE="1"

BENCH_CMD=(python benchmarks/hol_slotype.py)

# 로그/CSV 저장 폴더 (원하면 변경)
OUT_DIR="results_hol_sweep_slotype"
mkdir -p "$OUT_DIR"

# 실험 전/후 휴식(초)
REST_SEC=30

# DRAM cache drop 시도 여부(권한 필요할 수 있음)
DROP_CACHES=1

# GPU reset 강제 시도(권한/환경에 따라 위험/실패 가능) - 기본 OFF 추천
GPU_RESET=0
# ==================

timestamp() { date +"%Y%m%d_%H%M%S"; }

log() { echo "[$(date +"%F %T")] $*"; }

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "ERROR: required command not found: $1" >&2
    exit 1
  }
}

VLLM_PORT=8000

kill_vllm_on_port() {
  local port="${VLLM_PORT}"
  log "Killing vLLM server/listeners on TCP:${port} (reboot-like reset)"

  local pids=()
  if command -v lsof >/dev/null 2>&1; then
    mapfile -t pids < <(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null | sort -u || true)
  elif command -v fuser >/dev/null 2>&1; then
    mapfile -t pids < <(fuser -n tcp "$port" 2>/dev/null | tr ' ' '\n' | grep -E '^[0-9]+$' | sort -u || true)
  else
    log "WARNING: neither lsof nor fuser found; fallback to pattern kill only"
  fi

  if ((${#pids[@]} > 0)); then
    log "PIDs listening on ${port}: ${pids[*]}"
    kill -TERM "${pids[@]}" 2>/dev/null || true
    sleep 2
    kill -KILL "${pids[@]}" 2>/dev/null || true
  fi

  pkill -u "$(id -u)" -TERM -f 'vllm|api_server|uvicorn' || true
  sleep 2
  pkill -u "$(id -u)" -KILL -f 'vllm|api_server|uvicorn' || true
}

kill_hol_related_user() {
  log "Killing HOL-related python processes (user=$(id -un))"
  pkill -u "$(id -u)" -TERM -f 'benchmarks/hol_slotype.py' || true
  pkill -u "$(id -u)" -TERM -f 'qlm\.|Queue\(|VirtualQueue' || true
  sleep 2
  pkill -u "$(id -u)" -KILL -f 'benchmarks/hol_slotype.py' || true
  pkill -u "$(id -u)" -KILL -f 'qlm\.|Queue\(|VirtualQueue' || true
}

kill_all_python_user() {
  kill_vllm_on_port
  kill_hol_related_user
}

cleanup_gpu() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "nvidia-smi not found; skip GPU cleanup"
    return
  fi
  if ! command -v timeout >/dev/null 2>&1; then
    log "timeout not found; consider: sudo apt-get install coreutils"
  fi

  log "GPU processes (before cleanup):"
  timeout 10 nvidia-smi || {
    log "WARNING: nvidia-smi timed out (10s)"
    return
  }

  mapfile -t pids < <(timeout 10 nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | awk '{print $1}' | sort -u || true)
  if ((${#pids[@]} > 0)); then
    log "Killing GPU compute PIDs: ${pids[*]}"
    kill -TERM "${pids[@]}" 2>/dev/null || true
    sleep 2
    kill -KILL "${pids[@]}" 2>/dev/null || true
  else
    log "No GPU compute PIDs found"
  fi

  log "GPU processes (after cleanup):"
  timeout 10 nvidia-smi || log "WARNING: nvidia-smi timed out (10s)"
}

cleanup_dram() {
  log "DRAM/cache cleanup: sync + drop_caches(3)"
  sync || true
  if [[ "$DROP_CACHES" == "1" ]]; then
    echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null ||
      log "WARNING: drop_caches failed (sudo password required or not permitted)"
  else
    log "DROP_CACHES=0 so skipping /proc/sys/vm/drop_caches"
  fi
}

reboot_like_cleanup() {
  log "=== Reboot-like cleanup start ==="
  kill_all_python_user
  cleanup_gpu
  #  cleanup_dram
  log "=== Reboot-like cleanup end ==="
}

# (추가) config.yaml의 max_batch_size만 안전하게 치환 (주석/나머지 라인 보존)
set_max_batch_size() {
  local cfg_file="$1"
  local new_val="$2"

  local tmp
  tmp="$(mktemp)"
  awk -v v="$new_val" '
    BEGIN { done=0 }
    {
      if (!done && $1=="max_batch_size:") {
        print "max_batch_size: " v
        done=1
        next
      }
      print
    }
    END {
      if (!done) print "max_batch_size: " v
    }
  ' "$cfg_file" >"$tmp"
  mv "$tmp" "$cfg_file"
}

run_one() {
  local sleep_s="$1"
  local push_mode="$2"
  local push_val="$3"
  local mb="$4"
  local chunked_prefill="$5"

  local ts
  ts="$(timestamp)"

  local push_tag
  if [[ "$push_mode" == "rps" ]]; then
    push_tag="pushrps_${push_val}"
  else
    push_tag="pushint_${push_val}"
  fi

  local base="timsort_hol_sleep_${sleep_s}_${push_tag}_mb${mb}_chunked_${chunked_prefill}_${ts}"
  local log_file="${OUT_DIR}/${base}.txt"
  local csv_file="${OUT_DIR}/${base}.csv"

  log "---- Experiment start: max_batch_size=${mb}, QLM_QUEUE_LOOP_SLEEP=${sleep_s}, PUSH_MODE=${push_mode}, PUSH_VAL=${push_val}, CHUNKED_PREFILL=${chunked_prefill}, DRAIN_TIMEOUT_S=${DRAIN_TIMEOUT_S_DEFAULT} ----"

  log "Pre-rest ${REST_SEC}s..."
  sleep "$REST_SEC"

  reboot_like_cleanup

  log "Running benchmark -> ${log_file}"

  # 환경변수 세팅 후 실행, 로그는 파일로 저장
  # - DRAIN_TIMEOUT_S는 항상 적용
  # - RPS 모드에서는 PUSH_INTERVAL_S=0으로 강제(hol.py에서 RPS 적용 조건 보장)
  if [[ "$push_mode" == "rps" ]]; then
    QLM_QUEUE_LOOP_SLEEP="${sleep_s}" \
      QLM_SORT_ALGO="${SORT_ALGO}" \
      QLM_SORT_PROFILE="${SORT_PROFILE}" \
      DRAIN_TIMEOUT_S="${DRAIN_TIMEOUT_S_DEFAULT}" \
      PUSH_INTERVAL_S="0" \
      PUSH_RATE_RPS="${push_val}" \
      CHUNKED_PREFILL="${chunked_prefill}" \
      VLLM_ENABLE_CHUNKED_PREFILL="${chunked_prefill}" \
      "${BENCH_CMD[@]}" >>"${log_file}"
  else
    QLM_QUEUE_LOOP_SLEEP="${sleep_s}" \
      QLM_SORT_ALGO="${SORT_ALGO}" \
      QLM_SORT_PROFILE="${SORT_PROFILE}" \
      DRAIN_TIMEOUT_S="${DRAIN_TIMEOUT_S_DEFAULT}" \
      PUSH_INTERVAL_S="${push_val}" \
      PUSH_RATE_RPS="0" \
      CHUNKED_PREFILL="${chunked_prefill}" \
      VLLM_ENABLE_CHUNKED_PREFILL="${chunked_prefill}" \
      "${BENCH_CMD[@]}" >>"${log_file}"
  fi

  log "Benchmark done."

  reboot_like_cleanup

  require_cmd python3
  if [[ -f "log2csv_monitoring_slotype.py" ]]; then
    log "Converting log -> csv: ${csv_file}"
    python3 log2csv_monitoring_slotype.py "${log_file}" -o "${csv_file}"
  else
    log "WARNING: log2csv_monitoring_slotype.py not found in current directory. Skipping csv conversion."
  fi

  log "Post-rest ${REST_SEC}s..."
  sleep "$REST_SEC"

  log "---- Experiment end: max_batch_size=${mb}, QLM_QUEUE_LOOP_SLEEP=${sleep_s}, PUSH_MODE=${push_mode}, PUSH_VAL=${push_val}, CHUNKED_PREFILL=${chunked_prefill} ----"
  echo
}

main() {
  if [[ "$DROP_CACHES" == "1" || "$GPU_RESET" == "1" ]]; then
    log "Requesting sudo credential (for drop_caches/gpu-reset if enabled)..."
    sudo -v || true
  fi

  require_cmd awk
  require_cmd mktemp
  require_cmd python3

  if [[ -z "${QLMPROJDIR:-}" ]]; then
    echo "ERROR: QLMPROJDIR is not set. config.py expects \$QLMPROJDIR/qlm/config.yaml" >&2
    exit 1
  fi

  local CONFIG_FILE="${QLMPROJDIR}/qlm/config.yaml"
  if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: config.yaml not found at: $CONFIG_FILE" >&2
    exit 1
  fi

  # 원본 config 백업 + 어떤 종료/에러에도 원복
  local BACKUP_FILE="${CONFIG_FILE}.bak_hol_sweep_$$"
  cp "$CONFIG_FILE" "$BACKUP_FILE"
  trap 'cp "$BACKUP_FILE" "$CONFIG_FILE" >/dev/null 2>&1 || true; rm -f "$BACKUP_FILE" >/dev/null 2>&1 || true' EXIT

  log "Output dir: ${OUT_DIR}"
  log "CONFIG_FILE: ${CONFIG_FILE} (backup: ${BACKUP_FILE})"
  log "MAX_BATCH_SIZES: ${MAX_BATCH_SIZES[*]}"
  log "CHUNKED_PREFILL_VALUES: ${CHUNKED_PREFILL_VALUES[*]}"
  log "SLEEPS: ${SLEEPS[*]}"
  log "PUSH_MODE: ${PUSH_MODE}"
  log "PUSH_RPS_LIST: ${PUSH_RPS_LIST[*]}"
  log "PUSH_INTERVAL_LIST: ${PUSH_INTERVAL_LIST[*]}"
  log "DRAIN_TIMEOUT_S_DEFAULT: ${DRAIN_TIMEOUT_S_DEFAULT}"
  log "Command: ${BENCH_CMD[*]}"
  log "SORT: QLM_SORT_ALGO=${SORT_ALGO}, QLM_SORT_PROFILE=${SORT_PROFILE}"
  log "REST_SEC: ${REST_SEC}"
  log "DROP_CACHES: ${DROP_CACHES}, GPU_RESET: ${GPU_RESET}"
  echo

  # (추가) max_batch_size sweep 바깥 루프
  for mb in "${MAX_BATCH_SIZES[@]}"; do
    log "=== Setting max_batch_size=${mb} in ${CONFIG_FILE} ==="
    set_max_batch_size "$CONFIG_FILE" "$mb"

    for chunked in "${CHUNKED_PREFILL_VALUES[@]}"; do
      for s in "${SLEEPS[@]}"; do
        if [[ "$PUSH_MODE" == "rps" ]]; then
          for r in "${PUSH_RPS_LIST[@]}"; do
            run_one "$s" "rps" "$r" "$mb" "$chunked"
          done
        else
          for itv in "${PUSH_INTERVAL_LIST[@]}"; do
            run_one "$s" "interval" "$itv" "$mb" "$chunked"
          done
        fi
      done
    done
  done

  log "All experiments completed."
}

main "$@"
