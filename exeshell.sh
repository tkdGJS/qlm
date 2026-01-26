#!/usr/bin/env bash
set -euo pipefail

# 시작 시 한 번만 sudo 인증 (여기서 비번 입력)
sudo -v

# 스크립트가 돌아가는 동안 sudo 타임아웃 갱신(옵션)
# -v 갱신이 실패하면(예: 캐시 만료) 즉시 종료
keep_sudo_alive() {
  while true; do
    sleep 60
    sudo -n -v || exit 1
  done
}
keep_sudo_alive &
KEEPALIVE_PID=$!
trap 'kill $KEEPALIVE_PID 2>/dev/null || true' EXIT

# 원하는 두 스크립트 실행
bash ./run_hol_sweep_dynamic_slotype_maxbatch_priority.sh
bash ./run_hol_sweep_dynamic_slotype_maxbatch.sh
