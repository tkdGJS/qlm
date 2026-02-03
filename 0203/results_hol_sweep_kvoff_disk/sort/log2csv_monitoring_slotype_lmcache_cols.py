#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parse QLM debug + sort logs into CSVs.

Outputs (by default):
  1) <log>.csv        : per-request [DEBUG] lines
  2) <log>.sort.csv   : per-sort [SORT] lines

This parser is resilient to log-format changes by extracting fields with
targeted regexes instead of matching the full line strictly.
"""

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any


# -------- regex helpers --------
_RE_VQ = re.compile(
    r"^\[VQ(?P<vq_id>\d+)\]\s+groups=(?P<groups>\d+)\s+head_deadline=(?P<head_deadline>[-+]?\d+(?:\.\d+)?)\s+"
    r"slack=(?P<slack>[-+]?\d+(?:\.\d+)?)\s+model=(?P<model>.+?)\s*$"
)
_RE_DEADLINES = re.compile(r"^\s*deadlines:\s*(?P<deadlines>.+?)\s*$")
_RE_SORT = re.compile(
    r"^\[SORT\]\s+algo=(?P<algo>\S+)\s+calls=(?P<calls>\d+)\s+avg=(?P<avg_ms>[-+]?\d+(?:\.\d+)?)ms\s+"
    r"max=(?P<max_ms>[-+]?\d+(?:\.\d+)?)ms\s*$"
)

def _to_float(x: Optional[str]) -> Optional[float]:
    if x is None:
        return None
    s = x.strip()
    if s in ("None", "NA", "N/A", ""):
        return None
    try:
        return float(s)
    except ValueError:
        return None

def _to_int(x: Optional[str]) -> Optional[int]:
    if x is None:
        return None
    s = x.strip()
    if s in ("None", "NA", "N/A", ""):
        return None
    try:
        return int(s)
    except ValueError:
        return None

def _extract1(pattern: str, line: str) -> Optional[str]:
    m = re.search(pattern, line)
    return m.group(1) if m else None

def _extract2(pattern: str, line: str) -> Optional[Tuple[str, str]]:
    m = re.search(pattern, line)
    if not m:
        return None
    return (m.group(1), m.group(2))

# -------- LMCACHE parsing helpers --------

_NUM_UNIT_RE = re.compile(r"^([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)([A-Za-z]+)$")
_FLOAT_RE = re.compile(r"^[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?$")
_INT_RE = re.compile(r"^[-+]?\d+$")


def _parse_num_and_unit(val: str) -> Tuple[Optional[Any], Optional[str]]:
    """
    Parse a value that may contain a unit suffix, e.g. '56.0MiB'.
    Returns (numeric_or_string_or_None, unit_or_None).
    - If val is NA/None -> (None, None)
    - If numeric w/ unit -> (float(num), unit)
    - If numeric w/o unit -> (int/float, None)
    - Otherwise -> (original string, None)
    """
    s = (val or "").strip()
    if s in ("None", "NA", "N/A", ""):
        return None, None

    m = _NUM_UNIT_RE.match(s)
    if m:
        num_s, unit = m.group(1), m.group(2)
        try:
            return float(num_s), unit
        except ValueError:
            return s, None

    if _FLOAT_RE.match(s):
        if _INT_RE.match(s):
            try:
                return int(s), None
            except ValueError:
                return float(s), None
        return float(s), None

    return s, None


def parse_lmcache_blob(blob: str) -> Dict[str, Any]:
    """
    blob example:
      'lmcache:active_memory_objs_count=7 lmcache:local_cache_usage=56.0MiB ...'
    Produces columns:
      - 'lmcache:active_memory_objs_count' = 7
      - 'lmcache:local_cache_usage(MiB)' = 56.0
    """
    out: Dict[str, Any] = {}
    s = (blob or "").strip()
    if not s:
        return out

    for token in s.split():
        if "=" not in token:
            continue
        k, v = token.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            continue

        parsed, unit = _parse_num_and_unit(v)
        col = f"{k}({unit})" if unit else k
        out[col] = parsed
    return out


# -------- [DEBUG] parsing --------
def parse_debug_line(line: str) -> Optional[Dict[str, Any]]:
    if "[DEBUG] OrigSLO:" not in line:
        return None

    row: Dict[str, Any] = {}

    # core
    row["orig_slo"] = _to_float(_extract1(r"OrigSLO:\s*([-+]?\d+(?:\.\d+)?)", line))
    row["slo_type"] = _to_int(_extract1(r"slo_type:\s*(\d+)", line))
    row["insertion_time"] = _to_float(_extract1(r"Insertion Time:\s*([-+]?\d+(?:\.\d+)?)", line))
    row["wait_time"] = _to_float(_extract1(r"Wait Time:\s*([-+]?\d+(?:\.\d+)?)", line))

    # tokens
    row["prompt_tok"] = _to_int(_extract1(r"prompt_tok=\s*(\d+)", line))
    row["out_tok"] = _to_int(_extract1(r"out_tok=\s*(\d+)", line))
    row["total_tok"] = _to_int(_extract1(r"total_tok=\s*(\d+)", line))

    # latency stats (TTFT, TBT, TTLT)
    row["ttft"] = _to_float(_extract1(r"TTFT:\s*([^\|]+?)\s*\|", line))
    row["tbt_p95"] = _to_float(_extract1(r"TBT_p95=([^\s\|]+)", line))
    row["tbt_p99"] = _to_float(_extract1(r"TBT_p99=([^\s\|]+)", line))

    # Newer format (optional)
    row["tbt_slo"] = _to_float(_extract1(r"TBT_SLO=([^\s\|]+)", line))
    row["tbt_violation"] = _to_float(_extract1(r"TBT_Violation=([^\s\|]+)", line))
    # Can be numeric like "87.50%" or "NA%".
    row["tbt_success_rate_pct"] = _to_float(_extract1(r"TBT_SuccessRate=([^%\s\|]+)%", line))

    row["ttlt"] = _to_float(_extract1(r"TTLT:\s*([-+]?\d+(?:\.\d+)?)", line))

    # deadline metrics
    row["diff"] = _to_float(_extract1(r"Diff:\s*([-+]?\d+(?:\.\d+)?)", line))
    row["violation"] = _to_float(_extract1(r"Violation:\s*([-+]?\d+(?:\.\d+)?)", line))
    row["total_violation"] = _to_float(_extract1(r"TotalViolation:\s*([-+]?\d+(?:\.\d+)?)", line))
    row["finished_time"] = _to_float(_extract1(r"Finished Time:\s*([-+]?\d+(?:\.\d+)?)", line))

    # completion success rate
    row["success_rate_pct"] = _to_float(_extract1(r"SuccessRate:\s*([-+]?\d+(?:\.\d+)?)%", line))
    sp = _extract2(r"\((\d+)\s*/\s*(\d+)\)", line)
    if sp:
        row["success_count"] = _to_int(sp[0])
        row["processed_count"] = _to_int(sp[1])
    else:
        row["success_count"] = None
        row["processed_count"] = None

    # KV / VRAM
    row["kv"] = _to_float(_extract1(r"\bKV=([-+]?\d+(?:\.\d+)?)", line))
    row["run_cnt"] = _to_int(_extract1(r"\brun=(\d+)", line))
    row["wait_cnt"] = _to_int(_extract1(r"\bwait=(\d+)", line))
    # swap can be an integer or NA/None depending on logging.
    row["swap_cnt"] = _to_int(_extract1(r"\bswap=([^\s\|]+)", line))

    vram = _extract2(r"VRAM=([-+]?\d+(?:\.\d+)?)GiB/([-+]?\d+(?:\.\d+)?)GiB", line)
    if vram:
        row["vram_used_gib"] = _to_float(vram[0])
        row["vram_total_gib"] = _to_float(vram[1])
    else:
        row["vram_used_gib"] = None
        row["vram_total_gib"] = None

    # LMCACHE[...] tail (optional): store raw + expand into per-metric columns
    row["lmcache"] = None
    m_lm = re.search(r"\bLMCACHE\[(.*?)\]", line)
    if m_lm:
        row["lmcache"] = m_lm.group(1).strip()
        row.update(parse_lmcache_blob(row["lmcache"]))

    # If the line is too malformed, skip it.
    if row["orig_slo"] is None or row["insertion_time"] is None:
        return None

    return row


# -------- main parsing loop --------
def main():
    ap = argparse.ArgumentParser(description="Parse QLM debug + sort log into CSVs.")
    ap.add_argument("log_path", type=Path, help="Input log file path")
    ap.add_argument("-o", "--out", type=Path, default=None, help="Output CSV path for DEBUG rows (default: <log>.csv)")
    ap.add_argument("--sort-out", type=Path, default=None, help="Output CSV path for SORT rows (default: <log>.sort.csv)")
    ap.add_argument("--include-raw", action="store_true", help="Include raw [DEBUG] line in DEBUG CSV")
    ap.add_argument("--include-raw-sort", action="store_true", help="Include raw [SORT] line in SORT CSV")
    args = ap.parse_args()

    in_path: Path = args.log_path
    out_debug: Path = args.out or in_path.with_suffix(in_path.suffix + ".csv")
    out_sort: Path = args.sort_out or in_path.with_suffix(in_path.suffix + ".sort.csv")

    debug_rows: List[Dict[str, Any]] = []
    sort_rows: List[Dict[str, Any]] = []
    skipped_debug = 0

    last_vq: Dict[str, Any] = {}
    pending_sort: Optional[Dict[str, Any]] = None  # when [SORT] appears before [VQ]

    def flush_pending_sort():
        nonlocal pending_sort
        if pending_sort is not None:
            sort_rows.append(pending_sort)
            pending_sort = None

    with in_path.open("r", encoding="utf-8", errors="replace") as f:
        for idx, line in enumerate(f):
            s = line.rstrip("\n")

            # [VQ] line
            m = _RE_VQ.match(s.strip())
            if m:
                last_vq = {
                    "vq_id": _to_int(m.group("vq_id")),
                    "vq_groups": _to_int(m.group("groups")),
                    "head_deadline": _to_float(m.group("head_deadline")),
                    "slack": _to_float(m.group("slack")),
                    "model": m.group("model").strip(),
                    "deadlines": None,
                }
                # If a SORT was pending and likely belongs to this VQ (common pattern: SORT -> VQ),
                # attach VQ context and flush it now.
                if pending_sort is not None and pending_sort.get("_needs_vq", False):
                    pending_sort.update({
                        "vq_id": last_vq.get("vq_id"),
                        "vq_groups": last_vq.get("vq_groups"),
                        "head_deadline": last_vq.get("head_deadline"),
                        "slack": last_vq.get("slack"),
                        "model": last_vq.get("model"),
                        "deadlines": last_vq.get("deadlines"),
                    })
                    pending_sort.pop("_needs_vq", None)
                    flush_pending_sort()
                continue

            # deadlines line
            md = _RE_DEADLINES.match(s)
            if md and last_vq:
                last_vq["deadlines"] = md.group("deadlines").strip()
                continue

            # [SORT] line
            ms = _RE_SORT.match(s.strip())
            if ms:
                # If another pending sort exists, flush it first
                flush_pending_sort()

                row = {
                    "line_idx": idx,
                    "algo": ms.group("algo"),
                    "calls": _to_int(ms.group("calls")),
                    "avg_ms": _to_float(ms.group("avg_ms")),
                    "max_ms": _to_float(ms.group("max_ms")),
                    # VQ context (may be overwritten if VQ comes right after)
                    "vq_id": last_vq.get("vq_id"),
                    "vq_groups": last_vq.get("vq_groups"),
                    "head_deadline": last_vq.get("head_deadline"),
                    "slack": last_vq.get("slack"),
                    "model": last_vq.get("model"),
                    "deadlines": last_vq.get("deadlines"),
                }
                if args.include_raw_sort:
                    row["raw"] = s.strip()

                # If the next line is a VQ line, we want to bind to that VQ.
                row["_needs_vq"] = True
                pending_sort = row
                continue

            # [DEBUG] line
            if "[DEBUG] OrigSLO:" in s:
                flush_pending_sort()
                row = parse_debug_line(s)
                if row is None:
                    skipped_debug += 1
                    continue

                # attach VQ context if present
                if last_vq:
                    row["vq_id"] = last_vq.get("vq_id")
                    row["vq_groups"] = last_vq.get("vq_groups")
                    row["head_deadline"] = last_vq.get("head_deadline")
                    row["slack"] = last_vq.get("slack")
                    row["model"] = last_vq.get("model")
                    row["deadlines"] = last_vq.get("deadlines")
                else:
                    row["vq_id"] = None
                    row["vq_groups"] = None
                    row["head_deadline"] = None
                    row["slack"] = None
                    row["model"] = None
                    row["deadlines"] = None

                if args.include_raw:
                    row["raw"] = s.strip()
                debug_rows.append(row)
                continue

            # Any other line means pending SORT probably did not belong to a following VQ.
            # Flush it once we see a non-empty, non-VQ line after SORT.
            if pending_sort is not None and pending_sort.get("_needs_vq", False):
                # If this line is empty, keep waiting (sometimes logs have blank lines).
                if s.strip() == "":
                    continue
                pending_sort.pop("_needs_vq", None)
                flush_pending_sort()

    # EOF flush
    if pending_sort is not None:
        pending_sort.pop("_needs_vq", None)
        flush_pending_sort()

    # -------- write DEBUG csv --------
    debug_fields_base = [
        # request fields
        "orig_slo", "slo_type", "insertion_time", "wait_time",
        "prompt_tok", "out_tok", "total_tok",
        "ttft", "tbt_p95", "tbt_p99", "tbt_slo", "tbt_violation", "tbt_success_rate_pct",
        "ttlt",
        "diff", "violation", "total_violation", "finished_time",
        "success_rate_pct", "success_count", "processed_count",
        "kv", "run_cnt", "wait_cnt", "swap_cnt", "vram_used_gib", "vram_total_gib",
        # lmcache raw + expanded columns (expanded columns will be appended dynamically)
        "lmcache",
        # context
        "vq_id", "vq_groups", "head_deadline", "slack", "model", "deadlines",
    ]

    # Add any dynamic fields (e.g., lmcache:* columns) discovered during parsing.
    base_set = set(debug_fields_base)
    extra_fields = sorted({k for r in debug_rows for k in r.keys()} - base_set)
    debug_fields = debug_fields_base + extra_fields

    if args.include_raw:
        debug_fields.append("raw")

    out_debug.parent.mkdir(parents=True, exist_ok=True)
    with out_debug.open("w", newline="", encoding="utf-8") as fo:
        w = csv.DictWriter(fo, fieldnames=debug_fields)
        w.writeheader()
        for r in debug_rows:
            w.writerow(r)

    # -------- write SORT csv --------
    sort_fields = [
        "line_idx", "algo", "calls", "avg_ms", "max_ms",
        "vq_id", "vq_groups", "head_deadline", "slack", "model", "deadlines",
    ]
    if args.include_raw_sort:
        sort_fields.append("raw")

    out_sort.parent.mkdir(parents=True, exist_ok=True)
    with out_sort.open("w", newline="", encoding="utf-8") as fo:
        w = csv.DictWriter(fo, fieldnames=sort_fields)
        w.writeheader()
        for r in sort_rows:
            # internal field
            r.pop("_needs_vq", None)
            w.writerow(r)

    print(f"[OK] DEBUG parsed={len(debug_rows)} skipped={skipped_debug} -> {out_debug}")
    print(f"[OK] SORT  parsed={len(sort_rows)} -> {out_sort}")


if __name__ == "__main__":
    main()
