#!/usr/bin/env python3
"""
Plot KV-offloading impact from per-request snapshot logs.

Inputs: one or more CSV files produced by your logger (1 row = snapshot at request completion).
Outputs: PNG plots (A/B/C + scatter) that show:
  (1) Cause (memory pressure & offload events) -> (2) Cost (offload time) -> (3) Outcome (tail latency).

Usage examples:
  # single run
  python plot_kv_offload_impact.py --inputs cpu_only=/path/run_cpu_only.csv --outdir plots

  # compare CPU-only vs CPU+storage offload
  python plot_kv_offload_impact.py --inputs cpu_only=cpu_only.csv cpu_storage=cpu_storage.csv --outdir plots

Notes:
  - Metrics with *_sum/_count are cumulative across time; this script uses per-snapshot deltas to approximate per-request cost.
  - If you have true bytes counters (e.g., kvo_out_bytes), add them as columns; this script will auto-detect and prefer them.
"""

from __future__ import annotations
import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Helpers
# ----------------------------

def _require_cols(df: pd.DataFrame, cols: List[str], soft: bool = False) -> List[str]:
    missing = [c for c in cols if c not in df.columns]
    if missing and not soft:
        raise KeyError(f"Missing required columns: {missing}")
    return missing

def _to_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _sort_df(df: pd.DataFrame) -> pd.DataFrame:
    # Prefer finished_time (epoch seconds) then insertion_time, else keep order
    keys = [k for k in ["finished_time", "insertion_time"] if k in df.columns]
    if keys:
        df = df.sort_values(keys, kind="mergesort").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    return df

def _relative_time_s(df: pd.DataFrame) -> np.ndarray:
    if "finished_time" in df.columns:
        t = _to_numeric_series(df["finished_time"]).to_numpy()
    elif "insertion_time" in df.columns:
        t = _to_numeric_series(df["insertion_time"]).to_numpy()
    else:
        t = np.arange(len(df), dtype=float)
    # if epoch seconds, normalize to start
    t0 = np.nanmin(t) if np.isfinite(t).any() else 0.0
    return t - t0

def _delta(series: pd.Series) -> pd.Series:
    # Per-snapshot delta; negative deltas are kept (e.g., evictions) unless we clamp later.
    s = _to_numeric_series(series)
    return s.diff().fillna(0.0)

def _safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    denom = denom.replace(0, np.nan)
    out = numer / denom
    return out.fillna(0.0)

def _rolling_mean(a: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return a
    s = pd.Series(a)
    return s.rolling(win, min_periods=1, center=True).mean().to_numpy()


@dataclass
class Derived:
    t: np.ndarray
    x_label: str
    latency_s: np.ndarray
    latency_label: str

    offload_ms: np.ndarray                  # mean GPU->host copy time per snapshot (ms)
    offload_ms_label: str
    store_put_ms: np.ndarray                # mean host->storage put time per snapshot (ms), if any
    store_put_ms_label: str

    offload_amount_gib: np.ndarray          # proxy offload "amount" per snapshot (GiB)
    offload_amount_label: str

    kv_util_pct: np.ndarray
    vram_used_gib: np.ndarray
    run_cnt: np.ndarray
    wait_cnt: np.ndarray


def derive(df: pd.DataFrame, latency_metric: str, xmode: str) -> Derived:
    df = _sort_df(df).copy()

    # X axis
    if xmode == "time":
        t = _relative_time_s(df)
        x_label = "time (sec)"
    else:
        t = np.arange(len(df), dtype=float)
        x_label = "Request sequence (completion order)"

    # Latency metric
    if latency_metric not in df.columns:
        # fallback: best-effort
        for alt in ["tbt_p99", "tbt_p95", "ttft", "ttlt"]:
            if alt in df.columns:
                latency_metric = alt
                break
    latency = _to_numeric_series(df[latency_metric]).fillna(np.nan).to_numpy()
    latency_label = {
        "tbt_p99": "Tail token latency (TBT p99, s)",
        "tbt_p95": "Tail token latency (TBT p95, s)",
        "ttft": "Time to first token (TTFT, s)",
        "ttlt": "Time to last token (TTLT, s)",
    }.get(latency_metric, f"Latency ({latency_metric}, s)")

    # Offload cost: per-snapshot mean store_from_gpu_time
    # These are cumulative counters; convert to per-snapshot deltas first.
    sum_col = "lmcache:store_from_gpu_time_sum"
    cnt_col = "lmcache:store_from_gpu_time_count"
    off_ms = np.zeros(len(df), dtype=float)
    if sum_col in df.columns and cnt_col in df.columns:
        dsum = _delta(df[sum_col])
        dcnt = _delta(df[cnt_col])
        off_ms = (_safe_div(dsum, dcnt) * 1000.0).to_numpy()
    off_ms_label = "KV offload copy time (GPU→host, ms/request)"

    # Storage put cost (only meaningful for CPU+storage case)
    put_sum = "lmcache:store_put_time_sum"
    put_cnt = "lmcache:store_put_time_count"
    put_ms = np.zeros(len(df), dtype=float)
    if put_sum in df.columns and put_cnt in df.columns:
        dsum = _delta(df[put_sum])
        dcnt = _delta(df[put_cnt])
        put_ms = (_safe_div(dsum, dcnt) * 1000.0).to_numpy()
    put_ms_label = "KV storage write time (host→storage, ms/request)"

    # Offload amount (preferred): explicit bytes counters if present.
    # Fallback proxy: increase in host-side cache/storage usage gauges (MiB).
    # This estimates "how much KV ended up on host layers" per snapshot.
    bytes_cols = [c for c in df.columns if any(k in c.lower() for k in ["out_bytes", "offload_bytes", "kvo_out_bytes", "kv_offload_bytes"])]
    off_amt_gib = np.zeros(len(df), dtype=float)
    off_amt_label = "KV offloaded (GiB/request)"
    if bytes_cols:
        # Use the first matched bytes counter (assumed cumulative)
        bc = bytes_cols[0]
        dbytes = _delta(df[bc])
        off_amt_gib = (dbytes.clip(lower=0.0) / (1024**3)).to_numpy()
    else:
        # Proxy: delta of local_cache_usage + local_storage_usage (MiB)
        cache_col = "lmcache:local_cache_usage(MiB)"
        stor_col = "lmcache:local_storage_usage(MiB)"
        d_mib = np.zeros(len(df), dtype=float)
        if cache_col in df.columns:
            d_mib += _delta(df[cache_col]).to_numpy()
        if stor_col in df.columns:
            d_mib += _delta(df[stor_col]).to_numpy()
        off_amt_gib = np.maximum(d_mib, 0.0) / 1024.0  # keep only increases as "offload amount"
        off_amt_label = "Net KV increase on host cache/storage (GiB/request)"

    # System state signals
    kv = _to_numeric_series(df["kv"]).fillna(np.nan).to_numpy() * 100.0 if "kv" in df.columns else np.full(len(df), np.nan)
    vram = _to_numeric_series(df["vram_used_gib"]).fillna(np.nan).to_numpy() if "vram_used_gib" in df.columns else np.full(len(df), np.nan)
    run = _to_numeric_series(df["run_cnt"]).fillna(np.nan).to_numpy() if "run_cnt" in df.columns else np.full(len(df), np.nan)
    wait = _to_numeric_series(df["wait_cnt"]).fillna(np.nan).to_numpy() if "wait_cnt" in df.columns else np.full(len(df), np.nan)

    return Derived(
        t=t,
        x_label=x_label,
        latency_s=latency,
        latency_label=latency_label,
        offload_ms=off_ms,
        offload_ms_label=off_ms_label,
        store_put_ms=put_ms,
        store_put_ms_label=put_ms_label,
        offload_amount_gib=off_amt_gib,
        offload_amount_label=off_amt_label,
        kv_util_pct=kv,
        vram_used_gib=vram,
        run_cnt=run,
        wait_cnt=wait,
    )


def _make_grid(n: int) -> Tuple[int, int]:
    # simple grid: 1xN for <=2, else 2 rows
    if n <= 2:
        return (1, n)
    r = 2
    c = int(np.ceil(n / r))
    return (r, c)


# ----------------------------
# Plots
# ----------------------------

def plot_A_latency_vs_offload(cases: Dict[str, Derived], outpath: str, smooth: int) -> None:
    names = list(cases.keys())
    r, c = _make_grid(len(names))
    fig, axes = plt.subplots(r, c, figsize=(7*c, 4.5*r), squeeze=False)

    for i, name in enumerate(names):
        rr, cc = divmod(i, c)
        ax = axes[rr][cc]
        d = cases[name]

        y_lat = _rolling_mean(d.latency_s, smooth)
        y_off = _rolling_mean(d.offload_ms, smooth)

        ax.plot(d.t, y_lat, label=d.latency_label)
        ax.set_xlabel(d.x_label)
        ax.set_ylabel(d.latency_label)

        ax2 = ax.twinx()
        ax2.plot(d.t, y_off, label=d.offload_ms_label, linestyle="--")
        ax2.set_ylabel(d.offload_ms_label)

        # Combine legends
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="upper left")

        ax.set_title(f"A) Outcome vs offload cost — {name}")

    # hide unused axes
    for j in range(len(names), r*c):
        rr, cc = divmod(j, c)
        axes[rr][cc].axis("off")

    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_B_amount_vs_latency(cases: Dict[str, Derived], outpath: str, smooth: int) -> None:
    names = list(cases.keys())
    r, c = _make_grid(len(names))
    fig, axes = plt.subplots(r, c, figsize=(7*c, 4.5*r), squeeze=False)

    for i, name in enumerate(names):
        rr, cc = divmod(i, c)
        ax = axes[rr][cc]
        d = cases[name]

        y_amt = _rolling_mean(d.offload_amount_gib, smooth)
        y_lat = _rolling_mean(d.latency_s, smooth)

        ax.plot(d.t, y_amt, label=d.offload_amount_label)
        ax.set_xlabel(d.x_label)
        ax.set_ylabel(d.offload_amount_label)

        ax2 = ax.twinx()
        ax2.plot(d.t, y_lat, label=d.latency_label, linestyle="--")
        ax2.set_ylabel(d.latency_label)

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="upper left")

        ax.set_title(f"B) Offload amount proxy vs outcome — {name}")

    for j in range(len(names), r*c):
        rr, cc = divmod(j, c)
        axes[rr][cc].axis("off")

    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_C_chain(cases: Dict[str, Derived], outpath: str, smooth: int) -> None:
    names = list(cases.keys())
    r, c = _make_grid(len(names))
    fig, axes = plt.subplots(2*r, c, figsize=(7*c, 7*r), squeeze=False)

    for i, name in enumerate(names):
        rr, cc = divmod(i, c)
        d = cases[name]

        # Top: memory pressure signals
        ax1 = axes[2*rr][cc]
        kv = _rolling_mean(d.kv_util_pct, smooth)
        vram = _rolling_mean(d.vram_used_gib, smooth)

        if np.isfinite(kv).any():
            ax1.plot(d.t, kv, label="KV cache utilization (%)")
            ax1.set_ylabel("KV cache utilization (%)")
        ax1.set_xlabel(d.x_label)

        ax1b = ax1.twinx()
        if np.isfinite(vram).any():
            ax1b.plot(d.t, vram, label="VRAM used (GiB)", linestyle="--")
            ax1b.set_ylabel("VRAM used (GiB)")

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1b.get_legend_handles_labels()
        if lines or lines2:
            ax1.legend(lines + lines2, labels + labels2, loc="upper left")
        ax1.set_title(f"C1) Memory pressure — {name}")

        # Bottom: queueing + offload amount
        ax2 = axes[2*rr+1][cc]
        run = _rolling_mean(d.run_cnt, smooth)
        wait = _rolling_mean(d.wait_cnt, smooth)
        ax2.plot(d.t, run, label="Running requests (#)")
        ax2.plot(d.t, wait, label="Waiting requests (#)")
        ax2.set_xlabel(d.x_label)
        ax2.set_ylabel("Requests (#)")

        ax2b = ax2.twinx()
        amt = _rolling_mean(d.offload_amount_gib, smooth)
        ax2b.plot(d.t, amt, label=d.offload_amount_label, linestyle="--")
        ax2b.set_ylabel(d.offload_amount_label)

        lines, labels = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2b.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc="upper left")
        ax2.set_title(f"C2) Queueing → offload — {name}")

    # hide unused axes
    for j in range(len(names), r*c):
        rr, cc = divmod(j, c)
        axes[2*rr][cc].axis("off")
        axes[2*rr+1][cc].axis("off")

    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_D_scatter(cases: Dict[str, Derived], outpath: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5))

    # Scatter all cases together for direct comparison; different markers per case.
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    for idx, (name, d) in enumerate(cases.items()):
        x = d.offload_amount_gib
        y = d.latency_s
        kv = d.kv_util_pct

        # bucket by KV utilization
        bins = np.digitize(kv, [70, 90])  # 0: <70, 1: 70-90, 2: >=90
        for b in [0, 1, 2]:
            m = (bins == b) & np.isfinite(x) & np.isfinite(y)
            if not m.any():
                continue
            label = f"{name} | KV <70%" if b == 0 else (f"{name} | KV 70–90%" if b == 1 else f"{name} | KV ≥90%")
            ax.scatter(x[m], y[m], s=18, alpha=0.75, marker=markers[idx % len(markers)], label=label)

    ax.set_xlabel("Offload amount proxy (GiB/request)")
    ax.set_ylabel("Tail token latency (s)")
    ax.set_title("D) Correlation — offload amount vs tail latency (colored by KV pressure)")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_optional_storage_cost(cases: Dict[str, Derived], outpath: str, smooth: int) -> None:
    """Optional: show GPU->host copy vs host->storage write cost (helps CPU+storage vs CPU-only)."""
    names = list(cases.keys())
    r, c = _make_grid(len(names))
    fig, axes = plt.subplots(r, c, figsize=(7*c, 4.5*r), squeeze=False)

    for i, name in enumerate(names):
        rr, cc = divmod(i, c)
        ax = axes[rr][cc]
        d = cases[name]

        ax.plot(d.t, _rolling_mean(d.offload_ms, smooth), label=d.offload_ms_label)
        ax.plot(d.t, _rolling_mean(d.store_put_ms, smooth), label=d.store_put_ms_label, linestyle="--")
        ax.set_xlabel(d.x_label)
        ax.set_ylabel("Time (ms/request)")
        ax.set_title(f"Extra) Offload cost decomposition — {name}")
        ax.legend(loc="upper left")

    for j in range(len(names), r*c):
        rr, cc = divmod(j, c)
        axes[rr][cc].axis("off")

    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


# ----------------------------
# CLI
# ----------------------------

def parse_inputs(items: List[str]) -> Dict[str, str]:
    """Parse --inputs.

    Accepts either:
      - name=path pairs (recommended for multi-case comparison), or
      - bare paths (auto-names derived from filename).

    Examples:
      --inputs cpu_only=run1.csv cpu_storage=run2.csv
      --inputs run1.csv run2.csv
    """
    out: Dict[str, str] = {}
    used: Dict[str, int] = {}

    for it in items:
        if "=" in it:
            name, path = it.split("=", 1)
            name = name.strip()
            path = path.strip()
            if not name:
                raise ValueError(f"Empty case name in: {it}")
        else:
            path = it.strip()
            if not path:
                continue
            base = os.path.basename(path)
            name = os.path.splitext(base)[0]

        if not os.path.exists(path):
            raise FileNotFoundError(path)

        # Ensure unique names
        if name in out:
            used[name] = used.get(name, 1) + 1
            name = f"{name}_{used[name]}"
        out[name] = path

    if not out:
        raise ValueError("No valid inputs provided.")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="One or more case=csv_path pairs.")
    ap.add_argument("--outdir", required=True, help="Output directory for PNGs.")
    ap.add_argument("--latency-metric", default="tbt_p99", help="Prefer tbt_p99; falls back if missing.")
    ap.add_argument("--xmode", choices=["time", "index"], default="time", help="X axis as time or request index.")
    ap.add_argument("--smooth", type=int, default=9, help="Rolling mean window (in points) to reduce jitter.")
    ap.add_argument("--extra-storage-cost", action="store_true", help="Also plot offload copy vs storage write cost.")
    args = ap.parse_args()
    # Base output directory (user-provided)
    os.makedirs(args.outdir, exist_ok=True)

    paths = parse_inputs(args.inputs)

    def _sanitize_tag(s: str) -> str:
        # Keep path-friendly characters only
        out = []
        for ch in s:
            if ch.isalnum() or ch in ("-", "_", "."):
                out.append(ch)
            else:
                out.append("_")
        tag = "".join(out)
        tag = re.sub(r"_+", "_", tag).strip("_")
        return tag or "run"

    # Put outputs under outdir/<input-file-name> (single input) or outdir/compare__<names> (multi input)
    stems = [os.path.basename(p) for p in paths.values()]
    if len(stems) == 1:
        run_tag = _sanitize_tag(stems[0])
    else:
        run_tag = _sanitize_tag("compare__" + "__".join(stems))
    run_outdir = os.path.join(args.outdir, run_tag)
    os.makedirs(run_outdir, exist_ok=True)

    cases: Dict[str, Derived] = {}
    for name, p in paths.items():
        df = pd.read_csv(p)
        cases[name] = derive(df, latency_metric=args.latency_metric, xmode=args.xmode)

    plot_A_latency_vs_offload(cases, os.path.join(run_outdir, "A_latency_vs_offload_ms.png"), smooth=args.smooth)
    plot_B_amount_vs_latency(cases, os.path.join(run_outdir, "B_offload_amount_vs_latency.png"), smooth=args.smooth)
    plot_C_chain(cases, os.path.join(run_outdir, "C_chain_memory_offload_queue.png"), smooth=args.smooth)
    plot_D_scatter(cases, os.path.join(run_outdir, "D_scatter_offload_vs_latency.png"))

    if args.extra_storage_cost:
        plot_optional_storage_cost(cases, os.path.join(run_outdir, "EX_storage_cost_decomposition.png"), smooth=args.smooth)

    print(f"[OK] wrote plots to: {run_outdir}")


if __name__ == "__main__":
    main()
