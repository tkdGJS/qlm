#!/usr/bin/env bash
set -euo pipefail

# 스크립트가 있는 위치를 프로젝트 루트로 사용
QLMPROJDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export QLMPROJDIR

cd "$QLMPROJDIR"

# editable install (권장)
python -m pip install -e .

export PYTHONPATH="$(pwd)"
