#!/usr/bin/env python3
# red_black.py
#
# Usage:
#   python3 red_black.py <run_native.log> <run_cachegen.log> <multi_req_lmcache_vram_native.log> <multi_req_lmcache_vram_cachegen.log> [-o output_dir]
#
# Generates (all individual files):
#   1) TTFT timeline overlay        (native=black, cachegen=red)
#   2) TBT_p95 timeline overlay     (native=black, cachegen=red)
#   3) TTLT timeline overlay        (native=black, cachegen=red)
#   4) VRAM usage timeline overlay  (native=black, cachegen=red)
#   5~13) 9 bar charts (thin bars, native black / cachegen red)
#
# Notes:
# - TBT timeline uses per-request TBT_p95 from run logs.
# - If some metrics are NaN/Inf (e.g., estimated TBT when out_tok is missing), those bar charts are skipped safely.

import argparse
import os
import re
import sys
import math
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------
# Regex parsers
# ---------------------------

# Run log DEBUG line (with out_tok)
DEBUG_RE = re.compile(
    r"Insertion Time:\s*([0-9.]+)\s*\|"
    r".*?TTFT:\s*([0-9.]+|None)\s*\|"
    r"\s*TBT_p95=([0-9.]+|NA)\s*\|"
    r"\s*TBT_p99=([0-9.]+|NA)\s*\|"
    r".*?TTLT:\s*([0-9.]+)\s*\|"
    r".*?(?:Output Tokens|out_tok(?:s)?):\s*([0-9]+)"
    r".*?Finished Time:\s*([0-9.]+)",
    re.IGNORECASE,
)

# Run log DEBUG line (without out_tok)
DEBUG_RE_NO_OUTTOK = re.compile(
    r"Insertion Time:\s*([0-9.]+)\s*\|"
    r".*?TTFT:\s*([0-9.]+|None)\s*\|"
    r"\s*TBT_p95=([0-9.]+|NA)\s*\|"
    r"\s*TBT_p99=([0-9.]+|NA)\s*\|"
    r".*?TTLT:\s*([0-9.]+)\s*\|"
    r".*?Finished Time:\s*([0-9.]+)",
    re.IGNORECASE,
)

SAMPLER_RE = re.compile(r"^\[CLIENT_SAMPLER\]\s+([0-9.]+)\s+\S+\s+([0-9.]+)GB")
EVENT_RE = re.compile(
    r"^\[LMCACHE_VRAM\]\[LocalDiskBackend\]\s+([0-9.]+)\s+(serialize|deserialize)\b",
    re.IGNORECASE,
)


# ---------------------------
# Helpers
# ---------------------------

def _to_float_or_nan(s: str) -> float:
    s = str(s).strip()
    if s.lower() in ("none", "na", "nan"):
        return np.nan
    return float(s)

def _bad_metric(v) -> bool:
    try:
        fv = float(v)
        return math.isnan(fv) or math.isinf(fv)
    except Exception:
        return True

def _format_value_label(v) -> str:
    try:
        fv = float(v)
        if math.isnan(fv) or math.isinf(fv):
            return "N/A"
        return f"{fv:.5f}".rstrip("0").rstrip(".")
    except Exception:
        return "N/A"


# ---------------------------
# Parsing functions
# ---------------------------

def parse_run_log(path: str, mode: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if not line.startswith("[DEBUG] OrigSLO:"):
                continue

            m = DEBUG_RE.search(line)
            if m:
                insertion_ts = float(m.group(1))
                ttft = _to_float_or_nan(m.group(2))
                tbt_p95 = _to_float_or_nan(m.group(3))
                tbt_p99 = _to_float_or_nan(m.group(4))
                ttlt = float(m.group(5))
                out_tok = int(m.group(6))
                finished_ts = float(m.group(7))
            else:
                m2 = DEBUG_RE_NO_OUTTOK.search(line)
                if not m2:
                    continue
                insertion_ts = float(m2.group(1))
                ttft = _to_float_or_nan(m2.group(2))
                tbt_p95 = _to_float_or_nan(m2.group(3))
                tbt_p99 = _to_float_or_nan(m2.group(4))
                ttlt = float(m2.group(5))
                out_tok = np.nan
                finished_ts = float(m2.group(6))

            rows.append({
                "mode": mode,
                "insertion_ts": insertion_ts,
                "finished_ts": finished_ts,
                "TTFT": ttft,
                "TBT_p95": tbt_p95,
                "TBT_p99": tbt_p99,
                "TTLT": ttlt,
                "out_tok": out_tok,
            })

    if not rows:
        raise ValueError(f"No parseable [DEBUG] request lines found in {path}")
    return pd.DataFrame(rows)


def parse_vram_log(path: str, mode: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    srows, erows = [], []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            sm = SAMPLER_RE.search(line)
            if sm:
                srows.append({
                    "mode": mode,
                    "ts": float(sm.group(1)),
                    "vram_gb": float(sm.group(2)),
                })
                continue

            em = EVENT_RE.search(line)
            if em:
                erows.append({
                    "mode": mode,
                    "ts": float(em.group(1)),
                    "event": em.group(2).lower(),
                })

    s_df = pd.DataFrame(srows) if srows else pd.DataFrame(columns=["mode", "ts", "vram_gb"])
    e_df = pd.DataFrame(erows) if erows else pd.DataFrame(columns=["mode", "ts", "event"])
    return s_df, e_df


# ---------------------------
# Timeline normalization
# ---------------------------

def normalize_timelines(req_all: pd.DataFrame, vram_all: pd.DataFrame, evt_all: pd.DataFrame):
    mode_t0: Dict[str, float] = {}

    modes = sorted(set(req_all["mode"].dropna().tolist()))
    for mode in modes:
        candidates: List[float] = []

        r = req_all[req_all["mode"] == mode]
        if not r.empty:
            candidates.append(float(r["insertion_ts"].min()))

        v = vram_all[vram_all["mode"] == mode] if not vram_all.empty else pd.DataFrame()
        if not v.empty:
            candidates.append(float(v["ts"].min()))

        e = evt_all[evt_all["mode"] == mode] if not evt_all.empty else pd.DataFrame()
        if not e.empty:
            candidates.append(float(e["ts"].min()))

        mode_t0[mode] = min(candidates) if candidates else 0.0

    req_all = req_all.copy()
    vram_all = vram_all.copy()
    evt_all = evt_all.copy()

    req_all["t_finish"] = req_all.apply(lambda r: r["finished_ts"] - mode_t0[r["mode"]], axis=1)
    req_all["t_insert"] = req_all.apply(lambda r: r["insertion_ts"] - mode_t0[r["mode"]], axis=1)

    if not vram_all.empty:
        vram_all["t"] = vram_all.apply(lambda r: r["ts"] - mode_t0[r["mode"]], axis=1)

    if not evt_all.empty:
        evt_all["t"] = evt_all.apply(lambda r: r["ts"] - mode_t0[r["mode"]], axis=1)

    return req_all, vram_all, evt_all


# ---------------------------
# Plotting functions
# ---------------------------

def overlay_line_plot(
    x_native, y_native, x_cache, y_cache,
    xlabel: str, ylabel: str, title: str, out_path: str
) -> None:
    plt.figure(figsize=(14, 5))
    if len(x_native):
        plt.plot(x_native, y_native, color="black", linewidth=0.9, label="native")
    if len(x_cache):
        plt.plot(x_cache, y_cache, color="red", linewidth=0.9, label="cachegen")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def bar_two(
    title: str, ylabel: str, native_val, cache_val, out_path: str,
    width: float = 0.38
) -> None:
    labels = ["native", "cachegen"]

    # plotting values only (this function is typically called after NaN skip)
    values = [float(native_val), float(cache_val)]
    x = np.arange(2)

    plt.figure(figsize=(7, 5))
    plt.bar(x, values, width=width, color=["black", "red"])
    plt.xticks(x, labels)
    plt.ylabel(ylabel)
    plt.title(title)

    ymax = max(values) if len(values) else 0.0
    pad = ymax * 0.10 if ymax > 0 else 0.05

    for xi, yi in zip(x, values):
        plt.text(xi, yi, _format_value_label(yi), ha="center", va="bottom", fontsize=10)

    plt.ylim(0, ymax + pad * 2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


# ---------------------------
# Metrics computation
# ---------------------------

def compute_summary_metrics(req_df: pd.DataFrame) -> Dict[str, float]:
    ttft = req_df["TTFT"].dropna()
    tbt_p95 = req_df["TBT_p95"].dropna()
    ttlt = req_df["TTLT"].dropna()

    # Estimated TBT = (TTLT - TTFT) / (out_tok - 1), only if out_tok exists and > 1
    est_tbt = pd.Series(dtype=float)
    if "out_tok" in req_df.columns:
        valid = (
            req_df["TTLT"].notna()
            & req_df["TTFT"].notna()
            & req_df["out_tok"].notna()
            & (req_df["out_tok"] > 1)
        )
        if valid.any():
            est_tbt = ((req_df.loc[valid, "TTLT"] - req_df.loc[valid, "TTFT"]) /
                       (req_df.loc[valid, "out_tok"] - 1)).dropna()

    def p95(s: pd.Series) -> float:
        return float(s.quantile(0.95)) if len(s) else np.nan

    return {
        "TTFT_mean": float(ttft.mean()) if len(ttft) else np.nan,
        "TTFT_p95": p95(ttft),
        "TBTp95_mean": float(tbt_p95.mean()) if len(tbt_p95) else np.nan,
        "TBTp95_p95": p95(tbt_p95),
        "estTBT_mean": float(est_tbt.mean()) if len(est_tbt) else np.nan,
        "estTBT_p95": p95(est_tbt),
        "TTLT_mean": float(ttlt.mean()) if len(ttlt) else np.nan,
        "TTLT_p95": p95(ttlt),
        "processed_request_count": int(len(req_df)),
    }


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate native(black) vs cachegen(red) overlay and bar graphs from run/vram logs."
    )
    parser.add_argument("run_native", help="run_native.log")
    parser.add_argument("run_cachegen", help="run_cachegen.log")
    parser.add_argument("vram_native", help="multi_req_lmcache_vram_native.log")
    parser.add_argument("vram_cachegen", help="multi_req_lmcache_vram_cachegen.log")
    parser.add_argument("-o", "--outdir", default="red_black_graph_output", help="Output directory")
    args = parser.parse_args()

    for p in [args.run_native, args.run_cachegen, args.vram_native, args.vram_cachegen]:
        if not os.path.exists(p):
            print(f"[ERROR] File not found: {p}", file=sys.stderr)
            sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)

    # Parse
    req_native = parse_run_log(args.run_native, "native")
    req_cache = parse_run_log(args.run_cachegen, "cachegen")
    vram_native, evt_native = parse_vram_log(args.vram_native, "native")
    vram_cache, evt_cache = parse_vram_log(args.vram_cachegen, "cachegen")

    req_all = pd.concat([req_native, req_cache], ignore_index=True)
    vram_all = pd.concat([vram_native, vram_cache], ignore_index=True)
    evt_all = pd.concat([evt_native, evt_cache], ignore_index=True)

    # Normalize timeline per mode
    req_all, vram_all, evt_all = normalize_timelines(req_all, vram_all, evt_all)

    # Save parsed CSVs
    req_all.to_csv(os.path.join(args.outdir, "parsed_request_metrics.csv"), index=False)
    vram_all.to_csv(os.path.join(args.outdir, "parsed_vram_samples.csv"), index=False)
    evt_all.to_csv(os.path.join(args.outdir, "parsed_lmcache_events.csv"), index=False)

    # Split for plotting
    rn = req_all[req_all["mode"] == "native"].sort_values("t_finish")
    rc = req_all[req_all["mode"] == "cachegen"].sort_values("t_finish")
    vn = vram_all[vram_all["mode"] == "native"].sort_values("t")
    vc = vram_all[vram_all["mode"] == "cachegen"].sort_values("t")

    # 4 overlay line graphs (no deserialize/serialize event lines)
    overlay_line_plot(
        rn["t_finish"].to_numpy(), rn["TTFT"].to_numpy(),
        rc["t_finish"].to_numpy(), rc["TTFT"].to_numpy(),
        "Timeline (s from run start)", "TTFT (s)",
        "TTFT timeline overlay (native=black, cachegen=red)",
        os.path.join(args.outdir, "01_overlay_ttft_native_black_cachegen_red.png")
    )

    overlay_line_plot(
        rn["t_finish"].to_numpy(), rn["TBT_p95"].to_numpy(),
        rc["t_finish"].to_numpy(), rc["TBT_p95"].to_numpy(),
        "Timeline (s from run start)", "TBT_p95 (s)",
        "TBT_p95 timeline overlay (native=black, cachegen=red)",
        os.path.join(args.outdir, "02_overlay_tbt_p95_native_black_cachegen_red.png")
    )

    overlay_line_plot(
        rn["t_finish"].to_numpy(), rn["TTLT"].to_numpy(),
        rc["t_finish"].to_numpy(), rc["TTLT"].to_numpy(),
        "Timeline (s from run start)", "TTLT (s)",
        "TTLT timeline overlay (native=black, cachegen=red)",
        os.path.join(args.outdir, "03_overlay_ttlt_native_black_cachegen_red.png")
    )

    overlay_line_plot(
        vn["t"].to_numpy(), vn["vram_gb"].to_numpy(),
        vc["t"].to_numpy(), vc["vram_gb"].to_numpy(),
        "Timeline (s from run start)", "VRAM usage (GB)",
        "VRAM usage timeline overlay (native=black, cachegen=red)",
        os.path.join(args.outdir, "04_overlay_vram_native_black_cachegen_red.png")
    )

    # Summary metrics
    m_native = compute_summary_metrics(rn)
    m_cache = compute_summary_metrics(rc)

    print("[native metrics]", m_native)
    print("[cachegen metrics]", m_cache)

    summary_df = pd.DataFrame([
        {"mode": "native", **m_native},
        {"mode": "cachegen", **m_cache},
    ])
    summary_df.to_csv(os.path.join(args.outdir, "summary_metrics.csv"), index=False)

    # 9 bar charts (skip NaN/Inf metrics)
    bar_specs = [
        ("mean of TTFT", "TTFT (s)", m_native["TTFT_mean"], m_cache["TTFT_mean"], "05_mean_of_ttft_black_red_thin.png"),
        ("p95 of TTFT", "TTFT (s)", m_native["TTFT_p95"], m_cache["TTFT_p95"], "06_p95_of_ttft_black_red_thin.png"),
        ("mean of TBT_p95 (per-request)", "TBT_p95 (s)", m_native["TBTp95_mean"], m_cache["TBTp95_mean"], "07_mean_of_tbt_p95_per-request_black_red_thin.png"),
        ("p95 of TBT_p95 (per-request)", "TBT_p95 (s)", m_native["TBTp95_p95"], m_cache["TBTp95_p95"], "08_p95_of_tbt_p95_per-request_black_red_thin.png"),
        ("mean of estimated TBT", "Estimated TBT (s)", m_native["estTBT_mean"], m_cache["estTBT_mean"], "09_mean_of_estimated_tbt_black_red_thin.png"),
        ("p95 of estimated TBT", "Estimated TBT (s)", m_native["estTBT_p95"], m_cache["estTBT_p95"], "10_p95_of_estimated_tbt_black_red_thin.png"),
        ("mean of TTLT", "TTLT (s)", m_native["TTLT_mean"], m_cache["TTLT_mean"], "11_mean_of_ttlt_black_red_thin.png"),
        ("p95 of TTLT", "TTLT (s)", m_native["TTLT_p95"], m_cache["TTLT_p95"], "12_p95_of_ttlt_black_red_thin.png"),
        ("mean of processed request count", "Processed requests (count)", m_native["processed_request_count"], m_cache["processed_request_count"], "13_mean_of_processed_request_count_black_red_thin.png"),
    ]

    for title, ylabel, nv, cv, fname in bar_specs:
        if _bad_metric(nv) or _bad_metric(cv):
            print(f"[SKIP] {title}: invalid metric (native={nv}, cachegen={cv})")
            continue
        bar_two(title, ylabel, nv, cv, os.path.join(args.outdir, fname), width=0.38)

    print(f"\n[DONE] Output directory: {os.path.abspath(args.outdir)}")
    print("[FILES]")
    for fn in sorted(os.listdir(args.outdir)):
        print(" -", fn)


if __name__ == "__main__":
    main()
