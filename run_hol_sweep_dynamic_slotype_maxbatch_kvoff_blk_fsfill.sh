#!/usr/bin/env bash
set -euo pipefail

export PATH=/usr/sbin:/sbin:$PATH

# ====== 설정 ======
# sweep 할 값들 (원하는대로 수정)
#SLEEPS=(0.001 0.005 0.01 0.05 0.1)
#SLEEPS=(0.0001 0.001 0.01)
SLEEPS=(0.001)

# VQ push 속도 sweep:
#  - MODE=rps : PUSH_RATE_RPS로 sweep (권장)
#  - MODE=interval : PUSH_INTERVAL_S로 sweep
PUSH_MODE="interval"
PUSH_RPS_LIST=(1 2 5 10 20 50) # PUSH_MODE=rps 일 때 사용
#PUSH_INTERVAL_LIST=(1.0 0.5 0.1 0.05 0.01 0.005 0.001) # PUSH_MODE=interval 일 때 사용
PUSH_INTERVAL_LIST=(0.1) # PUSH_MODE=interval 일 때 사용
#PUSH_INTERVAL_LIST=(0.01) # PUSH_MODE=interval 일 때 사용

# (추가) max_batch_size sweep: 20~120, 20씩
#MAX_BATCH_SIZES=(100 80 60 40 20)
MAX_BATCH_SIZES=(100)

#CHUNKED_PREFILL_VALUES=(0 1)
CHUNKED_PREFILL_VALUES=(1)
# push 종료 후 큐 드레인 대기(초) - 요청 처리 완료까지 기다리게 하려면 0보다 크게!
#DRAIN_TIMEOUT_S_DEFAULT=600
DRAIN_TIMEOUT_S_DEFAULT=30

SORT_ALGO="timsort"
SORT_PROFILE="1"

BENCH_CMD=(python benchmarks/hol_kvoff.py)

# 로그/CSV 저장 폴더 (원하면 변경)
OUT_DIR="results_hol_sweep_kvoff"
mkdir -p "$OUT_DIR"

# 실험 전/후 휴식(초)
REST_SEC=300
# ===== blktrace/blkparse 설정 =====
TRACE_IO=1
TRACE_MOUNT="/tmp/lmcache_disk" # lmcache disk dir가 있는 파일시스템 기준으로 디바이스 자동 탐지
TRACE_DEV=""                    # 비우면 자동탐지, 직접 지정하려면 "/dev/nvme0n1" 같이 지정
TRACE_DIR="${OUT_DIR}/blktrace" # 결과 저장 폴더

# ===== NVMe filesystem fill sweep 설정 =====
# NOTE: ENABLE_FS_SWEEP=1이면, 매 run_one()마다 아래 디바이스를 **umount -> mkfs.ext4 -> mount** 한 뒤,
#       dumpfile로 파일시스템 사용량을 FILL_PCTS(%) 근처까지 채우고 실험을 시작합니다.
#       (매번 디바이스 내용이 삭제됩니다. 반드시 실험 전용 디바이스를 지정하세요.)
ENABLE_FS_SWEEP=1
NVME_DEV="/dev/nvme3n1"
NVME_MNT="/mnt/lmcache"
LMCACHE_DISK_DIR="${NVME_MNT}/lmcache_disk"   # LMCache가 실제로 쓰는 디렉토리
DUMPFILE_NAME="dumpfile.bin"                 # fill 용 더미 파일
FILL_SAFETY_MIB=512                          # 메타/오차 여유분 (MiB)

# (요청) fill% sweep
FILL_PCTS=(70 80 90)

# blktrace용 mount 기준도 NVMe로 변경 (자동 디바이스 탐지에 사용)
TRACE_MOUNT="${NVME_MNT}"


resolve_trace_dev() {
  local mountp="$1"

  # 사용자가 TRACE_DEV를 지정하면 그걸 우선 사용
  if [[ -n "${TRACE_DEV}" ]]; then
    echo "${TRACE_DEV}"
    return 0
  fi

  # mountp가 올라간 파티션/디바이스 확인
  local part
  part="$(df -P "$mountp" 2>/dev/null | awk 'END{print $1}')"
  if [[ -z "${part}" || ! -e "${part}" ]]; then
    echo "ERROR: cannot resolve device for mountpoint=${mountp} (df returned: ${part})" >&2
    return 1
  fi

  # 파티션이면 부모 디바이스(PKNAME)로 올림 (/dev/nvme0n1p2 -> /dev/nvme0n1)
  local pk
  pk="$(lsblk -no PKNAME "${part}" 2>/dev/null || true)"
  if [[ -n "${pk}" ]]; then
    echo "/dev/${pk}"
  else
    # lsblk가 못 찾으면 파티션 그대로 사용 (환경 따라 이게 더 잘 될 때도 있음)
    echo "${part}"
  fi
}

start_blktrace() {
  local dev="$1"
  local prefix="$2"
  log "Starting blktrace: dev=${dev}, prefix=${prefix}"
  # sudo 필요. (비밀번호 프롬프트 없이 되게 -n 사용)
  sudo -n blktrace -d "${dev}" -o "${prefix}" >/dev/null 2>&1 &
  BLKTRACE_PID=$!
}

stop_blktrace() {
  local pid="$1"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    log "Stopping blktrace (pid=${pid})"
    sudo -n kill -INT "${pid}" >/dev/null 2>&1 || true
    wait "${pid}" 2>/dev/null || true
  fi
}

run_blkparse() {
  local prefix="$1"
  log "Running blkparse: input_prefix=${prefix}"
  # 출력 텍스트는 같은 prefix에 .blkparse.txt로 저장
  blkparse -i "${prefix}" -o "${prefix}.blkparse.txt"
}

# ===== NVMe FS 재초기화 + fill 유틸 =====
is_mounted() {
  local mnt="$1"
  mountpoint -q "$mnt"
}

umount_if_mounted() {
  local mnt="$1"
  if is_mounted "$mnt"; then
    log "Unmounting ${mnt}..."
    sudo -n umount "$mnt" >/dev/null 2>&1 || sudo -n umount -l "$mnt" >/dev/null 2>&1 || {
      log "ERROR: umount failed for ${mnt}"
      return 1
    }
  fi
}

format_and_mount_ext4() {
  local dev="$1"
  local mnt="$2"

  sudo -n mkdir -p "$mnt"

  # 안전: 기존 마운트 해제
  umount_if_mounted "$mnt"

  log "Formatting ${dev} as ext4 (mkfs.ext4 -F -m 0)..."
  if command -v wipefs >/dev/null 2>&1; then
    sudo -n wipefs -a "$dev" >/dev/null 2>&1 || true
  fi
  sudo -n mkfs.ext4 -F -m 0 "$dev" >/dev/null

  log "Mounting ${dev} -> ${mnt} ..."
  sudo -n mount -o noatime "$dev" "$mnt"
}

df_mib() {
  # prints: total_mib used_mib avail_mib usepct (e.g., "100000 1234 98766 1%")
  local mnt="$1"
  df -Pm "$mnt" | awk 'END{print $2, $3, $4, $5}'
}

fill_fs_to_pct() {
  local mnt="$1"
  local pct="$2"
  local dumpfile="${mnt}/${DUMPFILE_NAME}"

  local total used avail usep
  read -r total used avail usep < <(df_mib "$mnt")

  # target_used_mib = total * pct/100 - safety
  local target_used=$(( (total * pct) / 100 ))
  if (( target_used > FILL_SAFETY_MIB )); then
    target_used=$(( target_used - FILL_SAFETY_MIB ))
  fi

  local need=$(( target_used - used ))
  if (( need <= 0 )); then
    log "FS fill: already >= target (used=${used}MiB, target≈${target_used}MiB). Skipping fill."
    return 0
  fi

  log "FS fill: total=${total}MiB, used=${used}MiB -> target≈${target_used}MiB (pct=${pct}%). Writing ${need}MiB to ${dumpfile}"
  sudo -n rm -f "$dumpfile" || true

  # 실제 write 유발(dd). bs=1M * count=need
  sudo -n dd if=/dev/zero of="$dumpfile" bs=1M count="$need" status=progress conv=fsync >/dev/null
  sync || true

  read -r total used avail usep < <(df_mib "$mnt")
  log "FS fill done: now used=${used}MiB / total=${total}MiB (Use%=${usep})"
}

prepare_nvme_fs() {
  local pct="$1"

  if [[ "$ENABLE_FS_SWEEP" != "1" ]]; then
    return 0
  fi

  log "=== NVMe FS prep: dev=${NVME_DEV}, mnt=${NVME_MNT}, fill=${pct}% ==="

  # 마운트 busy 방지: 실험 프로세스 종료
  kill_all_python_user

  format_and_mount_ext4 "${NVME_DEV}" "${NVME_MNT}"
  fill_fs_to_pct "${NVME_MNT}" "${pct}"

  # LMCache 디렉토리 준비 (dumpfile은 유지)
  sudo -n rm -rf "${LMCACHE_DISK_DIR}" || true
  sudo -n mkdir -p "${LMCACHE_DISK_DIR}"
  sudo -n chmod 777 "${LMCACHE_DISK_DIR}" || true
}


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
  pkill -u "$(id -u)" -TERM -f 'benchmarks/hol_kvoff.py' || true
  pkill -u "$(id -u)" -TERM -f 'qlm\.|Queue\(|VirtualQueue' || true
  sleep 2
  pkill -u "$(id -u)" -KILL -f 'benchmarks/hol_kvoff.py' || true
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
  local fill_pct="$6"

  local ts
  ts="$(timestamp)"

  local push_tag
  if [[ "$push_mode" == "rps" ]]; then
    push_tag="pushrps_${push_val}"
  else
    push_tag="pushint_${push_val}"
  fi

  local base="timsort_hol_sleep_${sleep_s}_${push_tag}_mb${mb}_chunked_${chunked_prefill}_fill${fill_pct}_${ts}"
  local log_file="${OUT_DIR}/${base}.txt"
  local csv_file="${OUT_DIR}/${base}.csv"

  rm -rf /tmp/lmcache_prometheus
  mkdir -p /tmp/lmcache_prometheus

  # FS fill sweep (ENABLE_FS_SWEEP=1이면 매 run마다 umount/mkfs/mount + dumpfile fill 수행)
  reboot_like_cleanup
  prepare_nvme_fs "${fill_pct}"

  export PROMETHEUS_MULTIPROC_DIR=/tmp/lmcache_prometheus
  if [[ "$ENABLE_FS_SWEEP" == "1" ]]; then
    export LMCACHE_LOCAL_DISK="file://${LMCACHE_DISK_DIR}/"
  else
    rm -rf /tmp/lmcache_disk
    mkdir -p /tmp/lmcache_disk
    export LMCACHE_LOCAL_DISK="file:///tmp/lmcache_disk/"
  fi
  export QLM_KV_EVENTS_ENDPOINT="tcp://127.0.0.1:5557"

  log "---- Experiment start: max_batch_size=${mb}, QLM_QUEUE_LOOP_SLEEP=${sleep_s}, PUSH_MODE=${push_mode}, PUSH_VAL=${push_val}, CHUNKED_PREFILL=${chunked_prefill}, FILL_PCT=${fill_pct}, DRAIN_TIMEOUT_S=${DRAIN_TIMEOUT_S_DEFAULT} ----"

  # 1) blktrace 시작
  local blktrace_pid=""
  local trace_prefix=""
  if [[ "$TRACE_IO" == "1" ]]; then
    mkdir -p "${TRACE_DIR}"
    local trace_dev
    trace_dev="$(resolve_trace_dev "${TRACE_MOUNT}")"
    trace_prefix="${TRACE_DIR}/${base}"
    start_blktrace "${trace_dev}" "${trace_prefix}"
    blktrace_pid="${BLKTRACE_PID}"
  fi

  # 3) (실험 시작 전) 5분 idle
  log "Pre-rest ${REST_SEC}s (included in blktrace)..."
  sleep "$REST_SEC"

  # 4) 벤치 실행
  log "Running benchmark -> ${log_file}"
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

  # 5) (실험 끝난 뒤) 5분 idle (flush/GC tail 포함시키기)
  log "Post-rest ${REST_SEC}s (included in blktrace)..."
  sleep "$REST_SEC"
  sync || true

  # 6) blktrace 종료 + blkparse
  if [[ "$TRACE_IO" == "1" ]]; then
    stop_blktrace "${blktrace_pid}"
    run_blkparse "${trace_prefix}"
    log "blktrace outputs: ${trace_prefix}.blktrace.*"
    log "blkparse output  : ${trace_prefix}.blkparse.txt"
  fi

  # 7) 이제 cleanup (트레이스에 섞이지 않게)
  reboot_like_cleanup

  require_cmd python3
  if [[ -f "log2csv_monitoring_slotype_lmcache_cols.py" ]]; then
    log "Converting log -> csv: ${csv_file}"
    python3 log2csv_monitoring_slotype_lmcache_cols.py "${log_file}" -o "${csv_file}"
  else
    log "WARNING: log2csv_monitoring_slotype_lmcache_cols.py not found in current directory. Skipping csv conversion."
  fi

  log "---- Experiment end: max_batch_size=${mb}, QLM_QUEUE_LOOP_SLEEP=${sleep_s}, PUSH_MODE=${push_mode}, PUSH_VAL=${push_val}, CHUNKED_PREFILL=${chunked_prefill}, FILL_PCT=${fill_pct} ----"
  echo
}

main() {
  if [[ "$DROP_CACHES" == "1" || "$GPU_RESET" == "1" || "$TRACE_IO" == "1" || "$ENABLE_FS_SWEEP" == "1" ]]; then
    log "Requesting sudo credential (for drop_caches/gpu-reset/blktrace/fs-mkfs if enabled)..."
    sudo -v || true
  fi

  require_cmd awk
  require_cmd mktemp
  require_cmd python3
  if [[ "$TRACE_IO" == "1" ]]; then
    require_cmd blktrace
    require_cmd blkparse
    require_cmd df
    require_cmd lsblk
  fi

  if [[ "$ENABLE_FS_SWEEP" == "1" ]]; then
    require_cmd mountpoint
    require_cmd mount
    require_cmd umount
    require_cmd mkfs.ext4
    require_cmd dd
    require_cmd df
    require_cmd lsblk
    # wipefs는 있으면 사용 (필수 아님)
  fi

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
  log "FILL_PCTS: ${FILL_PCTS[*]} (ENABLE_FS_SWEEP=${ENABLE_FS_SWEEP}, dev=${NVME_DEV}, mnt=${NVME_MNT})"
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

    # (추가) filesystem fill% sweep
    for fill in "${FILL_PCTS[@]}"; do
      log "=== FS fill target: ${fill}% (ENABLE_FS_SWEEP=${ENABLE_FS_SWEEP}, dev=${NVME_DEV}, mnt=${NVME_MNT}) ==="

      for chunked in "${CHUNKED_PREFILL_VALUES[@]}"; do
        for s in "${SLEEPS[@]}"; do
          if [[ "$PUSH_MODE" == "rps" ]]; then
            for r in "${PUSH_RPS_LIST[@]}"; do
              run_one "$s" "rps" "$r" "$mb" "$chunked" "$fill"
            done
          else
            for itv in "${PUSH_INTERVAL_LIST[@]}"; do
              run_one "$s" "interval" "$itv" "$mb" "$chunked" "$fill"
            done
          fi
        done
      done
    done
  done


  log "All experiments completed."
}

main "$@"
