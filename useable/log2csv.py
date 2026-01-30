#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parse QLM worker debug logs into CSV.

Expected line example:
[DEBUG] OrigSLO: 13.00 | Insertion Time: 1768460409.3240 | Wait Time: 2.2968 | prompt_tok= 3680 out_tok= 773 total_tok= 4453 | TTFT: 2.3805 | Execution Time: 25.0029 | Diff: 14.2997 | Violation: 14.2997 | TotalViolation: 14.2997 | Finished Time: 1768460436.627092 | SuccessRate: 94.12% (16/17)
"""

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Optional

LINE_RE = re.compile(
    r"""
    ^\[DEBUG\]\s+OrigSLO:\s*(?P<orig_slo>[-+]?\d+(?:\.\d+)?)\s*\|\s*
    Insertion\s+Time:\s*(?P<insertion_time>[-+]?\d+(?:\.\d+)?)\s*\|\s*
    Wait\s+Time:\s*(?P<wait_time>[-+]?\d+(?:\.\d+)?)\s*\|\s*
    prompt_tok=\s*(?P<prompt_tok>\d+)\s+
    out_tok=\s*(?P<out_tok>\d+)\s+
    total_tok=\s*(?P<total_tok>\d+)\s*\|\s*
    TTFT:\s*(?P<ttft>[-+]?\d+(?:\.\d+)?)\s*\|\s*
    Execution\s+Time:\s*(?P<execution_time>[-+]?\d+(?:\.\d+)?)\s*\|\s*
    Diff:\s*(?P<diff>[-+]?\d+(?:\.\d+)?)\s*\|\s*
    Violation:\s*(?P<violation>[-+]?\d+(?:\.\d+)?)\s*\|\s*
    TotalViolation:\s*(?P<total_violation>[-+]?\d+(?:\.\d+)?)\s*\|\s*
    Finished\s+Time:\s*(?P<finished_time>[-+]?\d+(?:\.\d+)?)\s*\|\s*
    SuccessRate:\s*(?P<success_rate_pct>[-+]?\d+(?:\.\d+)?)%\s*
    \((?P<success_count>\d+)\s*/\s*(?P<processed_count>\d+)\)\s*
    $
    """,
    re.VERBOSE,
)

def parse_line(line: str) -> Optional[Dict[str, object]]:
    m = LINE_RE.match(line.strip())
    if not m:
        return None

    g = m.groupdict()
    # Convert types
    def f(x: str) -> float:
        return float(x)

    def i(x: str) -> int:
        return int(x)

    row = {
        "orig_slo": f(g["orig_slo"]),
        "insertion_time": f(g["insertion_time"]),
        "wait_time": f(g["wait_time"]),
        "prompt_tok": i(g["prompt_tok"]),
        "out_tok": i(g["out_tok"]),
        "total_tok": i(g["total_tok"]),
        "ttft": f(g["ttft"]),
        "execution_time": f(g["execution_time"]),
        "diff": f(g["diff"]),
        "violation": f(g["violation"]),
        "total_violation": f(g["total_violation"]),
        "finished_time": f(g["finished_time"]),
        "success_rate_pct": f(g["success_rate_pct"]),
        "success_count": i(g["success_count"]),
        "processed_count": i(g["processed_count"]),
    }
    return row

def main():
    ap = argparse.ArgumentParser(description="Parse QLM debug log lines into CSV.")
    ap.add_argument("log_path", type=Path, help="Input log file path")
    ap.add_argument("-o", "--out", type=Path, default=None, help="Output CSV path (default: <log>.csv)")
    ap.add_argument("--include-raw", action="store_true", help="Include raw log line in CSV")
    args = ap.parse_args()

    in_path: Path = args.log_path
    out_path: Path = args.out or in_path.with_suffix(in_path.suffix + ".csv")

    rows = []
    skipped = 0

    with in_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "[DEBUG] OrigSLO:" not in line:
                continue
            row = parse_line(line)
            if row is None:
                skipped += 1
                continue
            if args.include_raw:
                row["raw"] = line.strip()
            rows.append(row)

    fieldnames = [
        "orig_slo",
        "insertion_time",
        "wait_time",
        "prompt_tok",
        "out_tok",
        "total_tok",
        "ttft",
        "execution_time",
        "diff",
        "violation",
        "total_violation",
        "finished_time",
        "success_rate_pct",
        "success_count",
        "processed_count",
    ]
    if args.include_raw:
        fieldnames.append("raw")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fo:
        w = csv.DictWriter(fo, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[OK] parsed={len(rows)} skipped={skipped} -> {out_path}")

if __name__ == "__main__":
    main()
