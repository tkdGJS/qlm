#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parse QLM worker debug logs into CSV.

Supports:
1) Old format (example)
[DEBUG] OrigSLO: 13.00 | Insertion Time: 1768460409.3240 | Wait Time: 2.2968 | prompt_tok= 3680 out_tok= 773 total_tok= 4453 | TTFT: 2.3805 | Execution Time: 25.0029 | Diff: 14.2997 | Violation: 14.2997 | TotalViolation: 14.2997 | Finished Time: 1768460436.627092 | SuccessRate: 94.12% (16/17)

2) New format (example)
[DEBUG] OrigSLO: 16.00 | slo_type: 0 | Insertion Time: 1769134501.2663 | Wait Time: 757.6409 | prompt_tok= 27 out_tok= 31 total_tok= 58 | TTFT: 0.0898 | TBT_p95=0.0313 | TBT_p99=0.0385 | TTLT: 0.7988 | Diff: 741.7307 | Violation: 741.7307 | TotalViolation: 228903.5883 | Finished Time: 1769135259.717299 | SuccessRate: 2.81% (18/640) | KV=0.138 run=9 wait=0 swap=0 VRAM=14.20GiB/15.00GiB

Also extracts SORT stats lines into a separate CSV:
[SORT] algo=timsort calls=4809 avg=24.146ms max=418.351ms
Optionally enriched with the most recent VQ context line:
[VQ0] groups=9450 head_deadline=... slack=... model=...
"""

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

# ---------- Regex: NEW debug lines ----------
NEW_LINE_RE = re.compile(
    r"""
    ^\[DEBUG\]\s+OrigSLO:\s*(?P<orig_slo>[-+]?\d+(?:\.\d+)?)\s*\|\s*
    slo_type:\s*(?P<slo_type>\d+)\s*\|\s*
    Insertion\s+Time:\s*(?P<insertion_time>[-+]?\d+(?:\.\d+)?)\s*\|\s*
    Wait\s+Time:\s*(?P<wait_time>[-+]?\d+(?:\.\d+)?)\s*\|\s*
    prompt_tok=\s*(?P<prompt_tok>\d+)\s+
    out_tok=\s*(?P<out_tok>\d+)\s+
    total_tok=\s*(?P<total_tok>\d+)\s*\|\s*
    TTFT:\s*(?P<ttft>(?:None|[-+]?\d+(?:\.\d+)?))\s*\|\s*
    TBT_p95=(?P<tbt_p95>(?:NA|[-+]?\d+(?:\.\d+)?))\s*\|\s*
    TBT_p99=(?P<tbt_p99>(?:NA|[-+]?\d+(?:\.\d+)?))\s*\|\s*
    TTLT:\s*(?P<ttlt>(?:NA|[-+]?\d+(?:\.\d+)?))\s*\|\s*
    Diff:\s*(?P<diff>[-+]?\d+(?:\.\d+)?)\s*\|\s*
    Violation:\s*(?P<violation>[-+]?\d+(?:\.\d+)?)\s*\|\s*
    TotalViolation:\s*(?P<total_violation>[-+]?\d+(?:\.\d+)?)\s*\|\s*
    Finished\s+Time:\s*(?P<finished_time>[-+]?\d+(?:\.\d+)?)\s*\|\s*
    SuccessRate:\s*(?P<success_rate_pct>[-+]?\d+(?:\.\d+)?)%\s*
    \((?P<success_count>\d+)\s*/\s*(?P<processed_count>\d+)\)\s*
    \|\s*
    KV=(?P<kv>[-+]?\d+(?:\.\d+)?)\s+
    run=(?P<run_cnt>\d+)\s+
    wait=(?P<wait_cnt>\d+)\s+
    swap=(?P<swap_cnt>\d+)\s+
    VRAM=(?P<vram_used_gib>[-+]?\d+(?:\.\d+)?)GiB/(?P<vram_total_gib>[-+]?\d+(?:\.\d+)?)GiB
    \s*$
    """,
    re.VERBOSE,
)

# ---------- Regex: OLD debug lines ----------
OLD_LINE_RE = re.compile(
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
    \s*$
    """,
    re.VERBOSE,
)

# ---------- Regex: SORT + VQ context ----------
SORT_RE = re.compile(
    r"""
    ^\[SORT\]\s+algo=(?P<algo>\S+)\s+
    calls=(?P<calls>\d+)\s+
    avg=(?P<avg_ms>[-+]?\d+(?:\.\d+)?)ms\s+
    max=(?P<max_ms>[-+]?\d+(?:\.\d+)?)ms\s*$
    """,
    re.VERBOSE,
)

VQ_RE = re.compile(
    r"""
    ^\[VQ(?P<vq_id>\d+)\]\s+
    groups=(?P<groups>\d+)\s+
    head_deadline=(?P<head_deadline>[-+]?\d+(?:\.\d+)?)\s+
    slack=(?P<slack>[-+]?\d+(?:\.\d+)?)\s+
    model=(?P<model>.+?)\s*$
    """,
    re.VERBOSE,
)


def _f_opt(x: str) -> Optional[float]:
    x = x.strip()
    if x in {"None", "NA", "N/A", ""}:
        return None
    return float(x)


def _i(x: str) -> int:
    return int(x)


def parse_debug_line(line: str) -> Optional[Dict[str, Any]]:
    s = line.strip()

    m = NEW_LINE_RE.match(s)
    if m:
        g = m.groupdict()
        return {
            "orig_slo": float(g["orig_slo"]),
            "slo_type": _i(g["slo_type"]),
            "insertion_time": float(g["insertion_time"]),
            "wait_time": float(g["wait_time"]),
            "prompt_tok": _i(g["prompt_tok"]),
            "out_tok": _i(g["out_tok"]),
            "total_tok": _i(g["total_tok"]),
            "ttft": _f_opt(g["ttft"]),
            "tbt_p95": _f_opt(g["tbt_p95"]),
            "tbt_p99": _f_opt(g["tbt_p99"]),
            "ttlt": _f_opt(g["ttlt"]),
            "execution_time": None,  # not present in new logs
            "diff": float(g["diff"]),
            "violation": float(g["violation"]),
            "total_violation": float(g["total_violation"]),
            "finished_time": float(g["finished_time"]),
            "success_rate_pct": float(g["success_rate_pct"]),
            "success_count": _i(g["success_count"]),
            "processed_count": _i(g["processed_count"]),
            "kv": float(g["kv"]),
            "run_cnt": _i(g["run_cnt"]),
            "wait_cnt": _i(g["wait_cnt"]),
            "swap_cnt": _i(g["swap_cnt"]),
            "vram_used_gib": float(g["vram_used_gib"]),
            "vram_total_gib": float(g["vram_total_gib"]),
        }

    m = OLD_LINE_RE.match(s)
    if m:
        g = m.groupdict()
        return {
            "orig_slo": float(g["orig_slo"]),
            "slo_type": None,
            "insertion_time": float(g["insertion_time"]),
            "wait_time": float(g["wait_time"]),
            "prompt_tok": _i(g["prompt_tok"]),
            "out_tok": _i(g["out_tok"]),
            "total_tok": _i(g["total_tok"]),
            "ttft": float(g["ttft"]),
            "tbt_p95": None,
            "tbt_p99": None,
            "ttlt": None,
            "execution_time": float(g["execution_time"]),
            "diff": float(g["diff"]),
            "violation": float(g["violation"]),
            "total_violation": float(g["total_violation"]),
            "finished_time": float(g["finished_time"]),
            "success_rate_pct": float(g["success_rate_pct"]),
            "success_count": _i(g["success_count"]),
            "processed_count": _i(g["processed_count"]),
            "kv": None,
            "run_cnt": None,
            "wait_cnt": None,
            "swap_cnt": None,
            "vram_used_gib": None,
            "vram_total_gib": None,
        }

    return None


def parse_sort_line(line: str) -> Optional[Dict[str, Any]]:
    m = SORT_RE.match(line.strip())
    if not m:
        return None
    g = m.groupdict()
    return {
        "algo": g["algo"],
        "calls": int(g["calls"]),
        "avg_ms": float(g["avg_ms"]),
        "max_ms": float(g["max_ms"]),
    }


def parse_vq_line(line: str) -> Optional[Dict[str, Any]]:
    m = VQ_RE.match(line.strip())
    if not m:
        return None
    g = m.groupdict()
    return {
        "vq_id": int(g["vq_id"]),
        "vq_groups": int(g["groups"]),
        "head_deadline": float(g["head_deadline"]),
        "slack": float(g["slack"]),
        "model": g["model"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse QLM debug log lines into CSV (+ optional SORT stats CSV).")
    ap.add_argument("log_path", type=Path, help="Input log file path")
    ap.add_argument("-o", "--out", type=Path, default=None, help="Output CSV path (default: <log>.csv)")
    ap.add_argument("--out-sort", type=Path, default=None, help="Output SORT CSV path (default: <log>.sort.csv)")
    ap.add_argument("--include-raw", action="store_true", help="Include raw log line in debug CSV")
    args = ap.parse_args()

    in_path: Path = args.log_path
    out_path: Path = args.out or in_path.with_suffix(in_path.suffix + ".csv")
    out_sort_path: Path = args.out_sort or in_path.with_suffix(in_path.suffix + ".sort.csv")

    debug_rows = []
    sort_rows = []
    skipped_debug = 0
    parsed_sort = 0

    # Keep the latest VQ context so we can attach it to SORT rows
    last_vq: Dict[str, Any] = {}

    with in_path.open("r", encoding="utf-8", errors="replace") as f:
        for idx, line in enumerate(f, start=1):
            if "[VQ" in line:
                vq = parse_vq_line(line)
                if vq:
                    last_vq = vq
                continue

            if "[SORT]" in line:
                srow = parse_sort_line(line)
                if srow:
                    parsed_sort += 1
                    row = {"line_idx": idx, **srow}
                    # attach VQ context if we have it
                    row.update({
                        "vq_id": last_vq.get("vq_id"),
                        "vq_groups": last_vq.get("vq_groups"),
                        "head_deadline": last_vq.get("head_deadline"),
                        "slack": last_vq.get("slack"),
                        "model": last_vq.get("model"),
                    })
                    sort_rows.append(row)
                continue

            if "[DEBUG] OrigSLO:" in line:
                drow = parse_debug_line(line)
                if drow is None:
                    skipped_debug += 1
                    continue
                if args.include_raw:
                    drow["raw"] = line.strip()
                debug_rows.append(drow)

    # ---- write debug CSV ----
    debug_fieldnames = [
        "orig_slo",
        "slo_type",
        "insertion_time",
        "wait_time",
        "prompt_tok",
        "out_tok",
        "total_tok",
        "ttft",
        "tbt_p95",
        "tbt_p99",
        "execution_time",
        "ttlt",
        "diff",
        "violation",
        "total_violation",
        "finished_time",
        "success_rate_pct",
        "success_count",
        "processed_count",
        "kv",
        "run_cnt",
        "wait_cnt",
        "swap_cnt",
        "vram_used_gib",
        "vram_total_gib",
    ]
    if args.include_raw:
        debug_fieldnames.append("raw")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fo:
        w = csv.DictWriter(fo, fieldnames=debug_fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in debug_rows:
            w.writerow(r)

    # ---- write sort CSV (only if we parsed any SORT lines) ----
    sort_written = False
    if sort_rows:
        sort_fieldnames = [
            "line_idx",
            "algo",
            "calls",
            "avg_ms",
            "max_ms",
            "vq_id",
            "vq_groups",
            "head_deadline",
            "slack",
            "model",
        ]
        out_sort_path.parent.mkdir(parents=True, exist_ok=True)
        with out_sort_path.open("w", newline="", encoding="utf-8") as fo:
            w = csv.DictWriter(fo, fieldnames=sort_fieldnames, extrasaction="ignore")
            w.writeheader()
            for r in sort_rows:
                w.writerow(r)
        sort_written = True

    msg = f"[OK] debug_parsed={len(debug_rows)} debug_skipped={skipped_debug} -> {out_path}"
    if sort_written:
        msg += f"\n[OK] sort_parsed={len(sort_rows)} -> {out_sort_path}"
    else:
        msg += "\n[OK] sort_parsed=0 (no [SORT] lines found)"
    print(msg)


if __name__ == "__main__":
    main()
