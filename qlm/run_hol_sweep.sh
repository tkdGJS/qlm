#!/usr/bin/env bash
set -euo pipefail

# ====== 설정 ======
# sweep 할 값들 (원하는대로 수정)
SLEEPS=(1.0 0.5 0.2 0.1 0.05 0.01)

SORT_ALGO="timsort"
SORT_PROFILE="1"

BENCH_CMD=(python benchmarks/hol.py)

# 로그/CSV 저장 폴더 (원하면 변경)
OUT_DIR="results_hol_sweep"
mkdir -p "$OUT_DIR"

# 실험 전/후 휴식(초) = 5분
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
    # fuser 출력은 "8000/tcp: 1234 5678" 같은 형태가 많아서 숫자만 추출
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

  # 포트 킬이 실패하는 환경 대비: vLLM/uvicorn 계열 패턴도 추가로 정리(현재 유저만)
  pkill -u "$(id -u)" -TERM -f 'vllm|api_server|uvicorn' || true
  sleep 2
  pkill -u "$(id -u)" -KILL -f 'vllm|api_server|uvicorn' || true
}

kill_hol_related_user() {
  log "Killing HOL-related python processes (user=$(id -un))"
  # hol/benchmarks 쪽만 (원하면 패턴 추가)
  pkill -u "$(id -u)" -TERM -f 'benchmarks/hol.py' || true
  pkill -u "$(id -u)" -TERM -f 'qlm\.|Queue\(|VirtualQueue' || true
  sleep 2
  pkill -u "$(id -u)" -KILL -f 'benchmarks/hol.py' || true
  pkill -u "$(id -u)" -KILL -f 'qlm\.|Queue\(|VirtualQueue' || true
}

kill_all_python_user() {
  # "리부트처럼 초기화" 요구 충족: vLLM(포트) 우선 정리 -> hol 관련 정리
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
    # -n: 비번 요구 시 실패로 빠짐(스크립트 멈춤 방지)
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
  cleanup_dram
  log "=== Reboot-like cleanup end ==="
}

run_one() {
  local sleep_s="$1"
  local ts
  ts="$(timestamp)"

  local base="timsort_hol_sleep_${sleep_s}_${ts}"
  local log_file="${OUT_DIR}/${base}.txt"
  local csv_file="${OUT_DIR}/${base}.csv"

  log "---- Experiment start: QLM_QUEUE_LOOP_SLEEP=${sleep_s} ----"

  log "Pre-rest ${REST_SEC}s..."
  sleep "$REST_SEC"

  reboot_like_cleanup

  log "Running benchmark -> ${log_file}"
  # 환경변수 세팅 후 실행, 로그는 파일로 저장
  QLM_QUEUE_LOOP_SLEEP="${sleep_s}" \
    QLM_SORT_ALGO="${SORT_ALGO}" \
    QLM_SORT_PROFILE="${SORT_PROFILE}" \
    "${BENCH_CMD[@]}" >>"${log_file}"

  log "Benchmark done."

  # 실험 직후에도 정리 (사용자 요구: 전/후 모두)
  reboot_like_cleanup

  # log2csv 변환
  require_cmd python3
  if [[ -f "log2csv.py" ]]; then
    log "Converting log -> csv: ${csv_file}"
    python3 log2csv.py "${log_file}" -o "${csv_file}"
  else
    log "WARNING: log2csv.py not found in current directory. Skipping csv conversion."
  fi

  log "Post-rest ${REST_SEC}s..."
  sleep "$REST_SEC"

  log "---- Experiment end: QLM_QUEUE_LOOP_SLEEP=${sleep_s} ----"
  echo
}

main() {
  # sudo 권한 미리 획득(원치 않으면 주석 처리)
  if [[ "$DROP_CACHES" == "1" || "$GPU_RESET" == "1" ]]; then
    log "Requesting sudo credential (for drop_caches/gpu-reset if enabled)..."
    sudo -v || true
  fi

  log "Output dir: ${OUT_DIR}"
  log "SLEEPS: ${SLEEPS[*]}"
  log "Command: ${BENCH_CMD[*]}"
  log "SORT: QLM_SORT_ALGO=${SORT_ALGO}, QLM_SORT_PROFILE=${SORT_PROFILE}"
  log "REST_SEC: ${REST_SEC}"
  log "DROP_CACHES: ${DROP_CACHES}, GPU_RESET: ${GPU_RESET}"
  echo

  for s in "${SLEEPS[@]}"; do
    run_one "$s"
  done

  log "All experiments completed."
}

main "$@"
