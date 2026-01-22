#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

for f in *.txt; do
  echo "[*] Converting: $f"
  python log2csv.py "$f"
done
