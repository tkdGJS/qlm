#!/usr/bin/env bash
set -euo pipefail

# ====== 설정 ======
# sweep 할 값들 (원하는대로 수정)
SLEEPS=(0.01 0.05 0.1 0.5 1.0)

# VQ push 속도 sweep:
#  - MODE=rps : PUSH_RATE_RPS로 sweep (권장)
#  - MODE=interval : PUSH_INTERVAL_S로 sweep
PUSH_MODE="interval"
PUSH_RPS_LIST=(1 2 5 10 20 50)                      # PUSH_MODE=rps 일 때 사용
PUSH_INTERVAL_LIST=(10.0 5.0 1.0 0.5 0.1 0.05 0.01) # PUSH_MODE=interval 일 때 사용

# push 종료 후 큐 드레인 대기(초) - 요청 처리 완료까지 기다리게 하려면 0보다 크게!
DRAIN_TIMEOUT_S_DEFAULT=600

SORT_ALGO="timsort"
SORT_PROFILE="1"

BENCH_CMD=(python benchmarks/hol.py)

# 로그/CSV 저장 폴더 (원하면 변경)
OUT_DIR="results_hol_sweep"
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
  pkill -u "$(id -u)" -TERM -f 'benchmarks/hol.py' || true
  pkill -u "$(id -u)" -TERM -f 'qlm\.|Queue\(|VirtualQueue' || true
  sleep 2
  pkill -u "$(id -u)" -KILL -f 'benchmarks/hol.py' || true
  pkill -u "$(id -u)" -KILL -f 'qlm\.|Queue\(|VirtualQueue' || true
}

kill_all_python_user() {
  kill_vllm_on_port
  kill_hol_related_user
}

cleanup_gpu() {
  local -a pids=()   # <-- 중요: 먼저 선언/초기화 (unbound 방지)

  mapfile -t pids < <(
    timeout 10 nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
    | awk '{print $1}' | sort -u || true
  )
  if ((${#pids[@]} > 0)); then
    local my_uid
    my_uid="$(id -u)"
    local -a my_pids=()
    local -a skipped=()

    # nvidia-smi에 찍힌 PID 중 "내 uid" 소유만 추려서 kill
    for pid in "${pids[@]}"; do
      [[ -z "${pid}" ]] && continue
      # ps가 실패하면(이미 종료 등) 그냥 스킵
      local uid
      uid="$(ps -o uid= -p "${pid}" 2>/dev/null | awk '{print $1}')"
      if [[ -n "${uid}" && "${uid}" == "${my_uid}" ]]; then
        my_pids+=("${pid}")
      else
        skipped+=("${pid}:${uid:-unknown}")
      fi
    done

    if ((${#my_pids[@]} > 0)); then
      log "Killing *my* GPU compute PIDs (uid=${my_uid}): ${my_pids[*]}"
      kill -TERM "${my_pids[@]}" 2>/dev/null || true
      sleep 2
      kill -KILL "${my_pids[@]}" 2>/dev/null || true
    else
      log "No GPU compute PIDs owned by uid=${my_uid} to kill"
    fi

    # 참고용: 내 uid가 아닌 PID들은 kill 대상에서 제외
    if ((${#skipped[@]} > 0)); then
      log "Skipped non-owned GPU PIDs (pid:uid): ${skipped[*]}"
    fi
  else
    log "No GPU compute PIDs found"
  fi

}


reboot_like_cleanup() {
  log "=== Reboot-like cleanup start ==="
  kill_all_python_user
  cleanup_gpu
  log "=== Reboot-like cleanup end ==="
}

run_one() {
  local sleep_s="$1"
  local push_mode="$2"
  local push_val="$3"

  local ts
  ts="$(timestamp)"

  local push_tag
  if [[ "$push_mode" == "rps" ]]; then
    push_tag="pushrps_${push_val}"
  else
    push_tag="pushint_${push_val}"
  fi

  local base="timsort_hol_sleep_${sleep_s}_${push_tag}_${ts}"
  local log_file="${OUT_DIR}/${base}.txt"
  local csv_file="${OUT_DIR}/${base}.csv"

  log "---- Experiment start: QLM_QUEUE_LOOP_SLEEP=${sleep_s}, PUSH_MODE=${push_mode}, PUSH_VAL=${push_val}, DRAIN_TIMEOUT_S=${DRAIN_TIMEOUT_S_DEFAULT} ----"

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
      "${BENCH_CMD[@]}" >>"${log_file}"
  else
    QLM_QUEUE_LOOP_SLEEP="${sleep_s}" \
      QLM_SORT_ALGO="${SORT_ALGO}" \
      QLM_SORT_PROFILE="${SORT_PROFILE}" \
      DRAIN_TIMEOUT_S="${DRAIN_TIMEOUT_S_DEFAULT}" \
      PUSH_INTERVAL_S="${push_val}" \
      PUSH_RATE_RPS="0" \
      "${BENCH_CMD[@]}" >>"${log_file}"
  fi

  log "Benchmark done."

  reboot_like_cleanup

  require_cmd python3
  if [[ -f "log2csv.py" ]]; then
    log "Converting log -> csv: ${csv_file}"
    python3 log2csv.py "${log_file}" -o "${csv_file}"
  else
    log "WARNING: log2csv.py not found in current directory. Skipping csv conversion."
  fi

  log "Post-rest ${REST_SEC}s..."
  sleep "$REST_SEC"

  log "---- Experiment end: QLM_QUEUE_LOOP_SLEEP=${sleep_s}, PUSH_MODE=${push_mode}, PUSH_VAL=${push_val} ----"
  echo
}

main() {
  if [[ "$DROP_CACHES" == "1" || "$GPU_RESET" == "1" ]]; then
    log "Requesting sudo credential (for drop_caches/gpu-reset if enabled)..."
    sudo -v || true
  fi

  log "Output dir: ${OUT_DIR}"
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

  for s in "${SLEEPS[@]}"; do
    if [[ "$PUSH_MODE" == "rps" ]]; then
      for r in "${PUSH_RPS_LIST[@]}"; do
        run_one "$s" "rps" "$r"
      done
    else
      for itv in "${PUSH_INTERVAL_LIST[@]}"; do
        run_one "$s" "interval" "$itv"
      done
    fi
  done

  log "All experiments completed."
}

main "$@"
