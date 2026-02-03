#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare KV offloading experiments:
- CPU-only: results_hol_sweep_kvoff
- CPU+storage: results_hol_sweep_kvoff_disk

Auto-match CSV pairs by (sleep, pushint, mb, chunked) parsed from filename like:
timsort_hol_sleep_0.001_pushint_0.01_mb100_chunked_1_YYYYMMDD_HHMMSS.txt.csv

Generates figures per matched pair:
(A) ECDF: TTFT / TBT_p99 / TTLT (facet by slo_type=0/1)
(B) Throughput time-series: tokens/s (1s bin)
(C) Cache usage + rolling latency overlay + event markers (Δ time_to_store/retrieve_count)
(D) Event vs No-event boxplots for TTFT/TBTp99/TTLT (facet by slo_type)
(E) Optional: avg time_to_store/retrieve (Δsum/Δcount) distribution
"""

import argparse
import os
import re
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


FILENAME_RE = re.compile(
    r"""
    sleep_(?P<sleep>[0-9.]+)
    _pushint_(?P<pushint>[0-9.]+)
    _mb(?P<mb>\d+)
    _chunked_(?P<chunked>\d+)
    """,
    re.VERBOSE,
)

# Metrics
LAT_METRICS = [
    ("ttft", "TTFT (s)"),
    ("tbt_p99", "TBT p99 (s)"),
    ("ttlt", "TTLT (s)"),
]

# slo_type labeling
SLO_NAME = {
    0: "Interactive request",
    1: "batch request",
}

# Use only data within this elapsed-time window for ALL plots/aggregations
DEFAULT_MAX_TIME_S = 600.0

# lmcache columns
COL_STORE_COUNT = "lmcache:time_to_store_count"
COL_STORE_SUM   = "lmcache:time_to_store_sum"
COL_RETR_COUNT  = "lmcache:time_to_retrieve_count"
COL_RETR_SUM    = "lmcache:time_to_retrieve_sum"

COL_CACHE_MIB   = "lmcache:local_cache_usage(MiB)"
COL_STOR_MIB    = "lmcache:local_storage_usage(MiB)"


def parse_key_from_filename(fname: str) -> Optional[Tuple[str, str, str, str]]:
    """Return (sleep, pushint, mb, chunked) as strings (exact token from filename)."""
    m = FILENAME_RE.search(fname)
    if not m:
        return None
    return (m.group("sleep"), m.group("pushint"), m.group("mb"), m.group("chunked"))


def discover_csvs(root: Path) -> Dict[Tuple[str, str, str, str], Path]:
    out: Dict[Tuple[str, str, str, str], Path] = {}
    for p in sorted(root.glob("*.csv")):
        key = parse_key_from_filename(p.name)
        if key is None:
            continue
        # If duplicates exist, keep the latest by lexicographic (timestamp is in name)
        if key not in out or p.name > out[key].name:
            out[key] = p
    return out


def ensure_numeric(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def prep_df(df: pd.DataFrame, max_time_s: float = DEFAULT_MAX_TIME_S) -> pd.DataFrame:
    """Sort by finished_time, build elapsed time, event flags, and 1s throughput bins.

    Keeps only rows with elapsed time t in [0, max_time_s].
    """
    df = df.copy()
    if "finished_time" not in df.columns:
        raise ValueError("CSV must have 'finished_time' column.")
    df = df.sort_values("finished_time").reset_index(drop=True)

    t0 = df["finished_time"].min()
    df["t"] = df["finished_time"] - t0
    df["sec"] = np.floor(df["t"]).astype(int)

    # Events from Δcounter
    if COL_STORE_COUNT in df.columns:
        df["store_evt"] = df[COL_STORE_COUNT].diff().fillna(0) > 0
    else:
        df["store_evt"] = False

    if COL_RETR_COUNT in df.columns:
        df["retrieve_evt"] = df[COL_RETR_COUNT].diff().fillna(0) > 0
    else:
        df["retrieve_evt"] = False

    df["any_evt"] = df["store_evt"] | df["retrieve_evt"]

    # Time window filter (apply after computing Δ-based events)
    df = df[(df["t"] >= 0) & (df["t"] <= max_time_s)].reset_index(drop=True)
    return df


def ecdf_xy(x: np.ndarray):
    """Return (x_sorted, y_ecdf)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    x = np.sort(x)
    n = len(x)
    if n == 0:
        return x, np.array([])
    y = np.arange(1, n + 1) / n
    return x, y


def tail_area_under_ecdf(x_sorted: np.ndarray, y_ecdf: np.ndarray) -> float:
    """
    Area of the complementary CDF over the observed range: ∫ (1 - ECDF(x)) dx.
    Units are seconds. Larger => heavier tail / slower distribution (within observed range).
    """
    if x_sorted.size < 2 or y_ecdf.size < 2:
        return 0.0
    return float(np.trapz(1.0 - y_ecdf, x_sorted))


def plot_ecdf_pair(df_cpu: pd.DataFrame, df_disk: pd.DataFrame,
                   outpath: Path, metric: str, xlabel: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for i, slo in enumerate([0, 1]):
        ax = axes[i]
        cpu_vals  = df_cpu.loc[df_cpu["slo_type"] == slo, metric].to_numpy()
        disk_vals = df_disk.loc[df_disk["slo_type"] == slo, metric].to_numpy()
        x1, y1 = ecdf_xy(cpu_vals)
        a1 = tail_area_under_ecdf(x1, y1)
        ax.plot(x1, y1, label=f"CPU-only (area={a1:.4g}s)")
        x2, y2 = ecdf_xy(disk_vals)
        a2 = tail_area_under_ecdf(x2, y2)
        ax.plot(x2, y2, label=f"CPU+storage (area={a2:.4g}s)")
        ax.set_title(SLO_NAME.get(slo, f"slo_type={slo}"))
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("ECDF")
    axes[1].legend(loc="lower right")
    fig.suptitle(f"ECDF: {xlabel} (CPU-only vs CPU+storage)", y=1.02)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close(fig)


def throughput_1s(df: pd.DataFrame) -> pd.Series:
    # tokens/s defined by 1s bin sum(total_tok)
    if "total_tok" not in df.columns:
        raise ValueError("CSV must have 'total_tok' column.")
    return df.groupby("sec")["total_tok"].sum()


def plot_throughput_pair(df_cpu: pd.DataFrame, df_disk: pd.DataFrame, outpath: Path, max_time_s: float = DEFAULT_MAX_TIME_S):
    tp_cpu = throughput_1s(df_cpu)
    tp_disk = throughput_1s(df_disk)

    # align index
    idx = sorted(set(tp_cpu.index.tolist()) | set(tp_disk.index.tolist()))
    max_sec = int(np.floor(max_time_s))
    idx = [i for i in idx if i <= max_sec]
    tp_cpu = tp_cpu.reindex(idx, fill_value=0)
    tp_disk = tp_disk.reindex(idx, fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 3.8))
    # area under throughput curve (tokens/s integrated over seconds) ~= total tokens (1s bins)
    area_cpu = float(np.trapz(tp_cpu.values, tp_cpu.index))
    area_disk = float(np.trapz(tp_disk.values, tp_disk.index))
    ax.plot(tp_cpu.index, tp_cpu.values, label=f"CPU-only (area={area_cpu:.0f} tok)")
    ax.plot(tp_disk.index, tp_disk.values, label=f"CPU+storage (area={area_disk:.0f} tok)")
    ax.set_xlabel("elapsed time (s)")
    ax.set_ylabel("throughput (tokens/s)")
    ax.set_title("Throughput (1s bins)")
    ax.set_xlim(0, max_time_s)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_cache_latency_overlay(df: pd.DataFrame, outpath: Path,
                               mode_label: str, slo_type: int,
                               latency_col: str, latency_label: str,
                               rolling_window: int = 50,
                               max_event_markers: int = 200,
                               max_time_s: float = DEFAULT_MAX_TIME_S):
    """
    Single-experiment overlay:
    - left y: rolling p95 latency (by slo_type)
    - right y: cache usage MiB (cache + storage if available)
    - vertical markers at store/retrieve events (global timeline)
    """
    d = df.copy()
    # rolling on per-slo subset for latency curve
    sub = d.loc[d["slo_type"] == slo_type, ["t", latency_col]].copy()
    sub[latency_col] = pd.to_numeric(sub[latency_col], errors="coerce")
    sub = sub.dropna()
    if sub.empty:
        return

    # rolling p95 (quantile)
    sub["roll_p95"] = sub[latency_col].rolling(rolling_window, min_periods=max(10, rolling_window // 5)).quantile(0.95)

    fig, ax1 = plt.subplots(figsize=(12, 4.0))
    ax1.plot(sub["t"], sub["roll_p95"], label=f"{latency_label} rolling p95")
    ax1.set_xlabel("elapsed time (s)")
    ax1.set_ylabel("latency (s)")
    ax1.grid(True, alpha=0.25)
    ax1.set_xlim(0, max_time_s)

    ax2 = ax1.twinx()
    if COL_CACHE_MIB in d.columns:
        ax2.plot(d["t"], pd.to_numeric(d[COL_CACHE_MIB], errors="coerce"),
                 linestyle="--", label="local_cache_usage (MiB)")
    if COL_STOR_MIB in d.columns:
        ax2.plot(d["t"], pd.to_numeric(d[COL_STOR_MIB], errors="coerce"),
                 linestyle="--", label="local_storage_usage (MiB)")
    ax2.set_ylabel("cache/storage usage (MiB)")
    ax2.set_xlim(0, max_time_s)

    # event markers (store/retrieve on global df)
    store_ts = d.loc[d["store_evt"], "t"].to_numpy()
    retr_ts  = d.loc[d["retrieve_evt"], "t"].to_numpy()

    # cap markers for readability
    if len(store_ts) > max_event_markers:
        store_ts = store_ts[:: max(1, len(store_ts)//max_event_markers)]
    if len(retr_ts) > max_event_markers:
        retr_ts = retr_ts[:: max(1, len(retr_ts)//max_event_markers)]

    for x in store_ts:
        ax1.axvline(x, alpha=0.08)
    for x in retr_ts:
        ax1.axvline(x, alpha=0.08)

    title = f"{mode_label} overlay ({SLO_NAME.get(slo_type, f'slo_type={slo_type}')}): cache usage + latency + events"
    ax1.set_title(title)

    # merge legends
    lines, labels = [], []
    for a in (ax1, ax2):
        l, lab = a.get_legend_handles_labels()
        lines += l
        labels += lab
    ax1.legend(lines, labels, loc="upper right")

    fig.tight_layout()
    fig.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_event_boxpair(df_cpu: pd.DataFrame, df_disk: pd.DataFrame,
                       outpath: Path, metric: str, xlabel: str):
    """
    One plot: boxplot for event/no-event vs mode (4 groups) per slo_type in two panels.
    Groups: CPU-only noevt, CPU-only evt, CPU+storage noevt, CPU+storage evt
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)

    for i, slo in enumerate([0, 1]):
        ax = axes[i]
        def vals(df, any_evt: bool):
            s = df.loc[(df["slo_type"] == slo) & (df["any_evt"] == any_evt), metric]
            return pd.to_numeric(s, errors="coerce").dropna().to_numpy()

        groups = [
            vals(df_cpu,  False),
            vals(df_cpu,  True),
            vals(df_disk, False),
            vals(df_disk, True),
        ]
        labels = ["CPU noevt", "CPU evt", "Disk noevt", "Disk evt"]
        ax.boxplot(groups, labels=labels, showfliers=False)
        ax.set_title(SLO_NAME.get(slo, f"slo_type={slo}"))
        ax.set_xlabel(xlabel)
        ax.grid(True, axis="y", alpha=0.25)

    axes[0].set_ylabel("latency (s)")
    fig.suptitle(f"Event vs No-event: {xlabel} (CPU-only vs CPU+storage)", y=1.02)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close(fig)


def delta_avg_time(df: pd.DataFrame, count_col: str, sum_col: str) -> np.ndarray:
    if count_col not in df.columns or sum_col not in df.columns:
        return np.array([])
    dc = pd.to_numeric(df[count_col], errors="coerce").diff()
    ds = pd.to_numeric(df[sum_col], errors="coerce").diff()
    m = (dc > 0) & (ds.notna())
    out = (ds[m] / dc[m]).to_numpy(dtype=float)
    out = out[np.isfinite(out)]
    return out


def plot_delta_time_dist(df_cpu: pd.DataFrame, df_disk: pd.DataFrame,
                         outpath: Path, title: str,
                         count_col: str, sum_col: str, xlabel: str):
    cpu = delta_avg_time(df_cpu, count_col, sum_col)
    disk = delta_avg_time(df_disk, count_col, sum_col)

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    # ECDF
    x1, y1 = ecdf_xy(cpu)
    a1 = tail_area_under_ecdf(x1, y1)
    ax.plot(x1, y1, label=f"CPU-only (area={a1:.4g}s)")
    x2, y2 = ecdf_xy(disk)
    a2 = tail_area_under_ecdf(x2, y2)
    ax.plot(x2, y2, label=f"CPU+storage (area={a2:.4g}s)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("ECDF")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close(fig)


def safe_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu_dir", type=str, default="results_hol_sweep_kvoff",
                    help="CPU-only offload results dir (CSV files).")
    ap.add_argument("--disk_dir", type=str, default="results_hol_sweep_kvoff_disk",
                    help="CPU+storage offload results dir (CSV files).")
    ap.add_argument("--outdir", type=str, default="plots_kvoff_compare",
                    help="Output directory for figures.")
    ap.add_argument("--rolling", type=int, default=50,
                    help="Rolling window size for overlay plots.")
    ap.add_argument("--max_event_markers", type=int, default=200,
                    help="Max event marker lines per plot.")
    ap.add_argument("--max_time_s", type=float, default=DEFAULT_MAX_TIME_S,
                    help="Use only data within elapsed time [0, max_time_s] for ALL plots.")
    ap.add_argument("--only_pushint", type=str, default="",
                    help="If set (e.g. '0.01'), only process that pushint.")
    args = ap.parse_args()

    cpu_root = Path(args.cpu_dir)
    disk_root = Path(args.disk_dir)
    out_root = Path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)

    cpu_map = discover_csvs(cpu_root)
    disk_map = discover_csvs(disk_root)

    keys = sorted(set(cpu_map.keys()) & set(disk_map.keys()))
    if args.only_pushint:
        keys = [k for k in keys if k[1] == args.only_pushint]

    print(f"[INFO] CPU-only files: {len(cpu_map)}")
    print(f"[INFO] CPU+storage files: {len(disk_map)}")
    print(f"[INFO] Matched pairs: {len(keys)}")
    if not keys:
        print("[WARN] No matched pairs found. Check filename pattern and directories.")
        return

    for (sleep, pushint, mb, chunked) in keys:
        cpu_path = cpu_map[(sleep, pushint, mb, chunked)]
        disk_path = disk_map[(sleep, pushint, mb, chunked)]

        tag = f"sleep_{sleep}_pushint_{pushint}_mb{mb}_chunked_{chunked}"
        pair_out = out_root / safe_name(tag)
        pair_out.mkdir(parents=True, exist_ok=True)

        print(f"\n[PAIR] {tag}")
        print(f"  CPU : {cpu_path}")
        print(f"  DISK: {disk_path}")

        df_cpu = pd.read_csv(cpu_path)
        df_disk = pd.read_csv(disk_path)

        # Ensure numeric for main columns
        ensure_numeric(df_cpu,  ["slo_type", "finished_time", "total_tok", "ttft", "tbt_p99", "ttlt",
                                 COL_STORE_COUNT, COL_STORE_SUM, COL_RETR_COUNT, COL_RETR_SUM, COL_CACHE_MIB, COL_STOR_MIB])
        ensure_numeric(df_disk, ["slo_type", "finished_time", "total_tok", "ttft", "tbt_p99", "ttlt",
                                 COL_STORE_COUNT, COL_STORE_SUM, COL_RETR_COUNT, COL_RETR_SUM, COL_CACHE_MIB, COL_STOR_MIB])

        df_cpu = prep_df(df_cpu, args.max_time_s)
        df_disk = prep_df(df_disk, args.max_time_s)

        # (A) ECDF: ttft/tbt_p99/ttlt
        for col, label in LAT_METRICS:
            outp = pair_out / f"A_ecdf_{col}.png"
            plot_ecdf_pair(df_cpu, df_disk, outp, col, label)

        # (B) Throughput
        outp = pair_out / "B_throughput_tokens_per_s.png"
        plot_throughput_pair(df_cpu, df_disk, outp, max_time_s=args.max_time_s)

        # (C) Cache usage + rolling latency overlay + events (each mode + slo_type)
        # Interactive emphasis: TTFT, Batch emphasis: TTLT
        for mode_label, d in [("CPU-only", df_cpu), ("CPU+storage", df_disk)]:
            outp0 = pair_out / f"C_overlay_{mode_label}_Interactive_TTFT.png"
            plot_cache_latency_overlay(
                d, outp0, mode_label, slo_type=0,
                latency_col="ttft", latency_label="TTFT (s)",
                rolling_window=args.rolling,
                max_event_markers=args.max_event_markers,
                max_time_s=args.max_time_s,
            )
            outp1 = pair_out / f"C_overlay_{mode_label}_Batch_TTLT.png"
            plot_cache_latency_overlay(
                d, outp1, mode_label, slo_type=1,
                latency_col="ttlt", latency_label="TTLT (s)",
                rolling_window=args.rolling,
                max_event_markers=args.max_event_markers,
                max_time_s=args.max_time_s,
            )

        # (D) Event vs No-event boxplots (ttft/tbt_p99/ttlt)
        for col, label in LAT_METRICS:
            outp = pair_out / f"D_event_box_{col}.png"
            plot_event_boxpair(df_cpu, df_disk, outp, col, label)

        # (E) Optional: avg store/retrieve time per event (Δsum/Δcount) ECDF
        outp = pair_out / "E_avg_time_to_store_ecdf.png"
        plot_delta_time_dist(
            df_cpu, df_disk, outp,
            title="avg time_to_store per event (Δsum/Δcount)",
            count_col=COL_STORE_COUNT, sum_col=COL_STORE_SUM,
            xlabel="seconds",
        )
        outp = pair_out / "E_avg_time_to_retrieve_ecdf.png"
        plot_delta_time_dist(
            df_cpu, df_disk, outp,
            title="avg time_to_retrieve per event (Δsum/Δcount)",
            count_col=COL_RETR_COUNT, sum_col=COL_RETR_SUM,
            xlabel="seconds",
        )

        # small summary CSV (optional but handy)
        rows = []
        for slo in [0, 1]:
            for mode, d in [("CPU-only", df_cpu), ("CPU+storage", df_disk)]:
                for col, label in LAT_METRICS:
                    x = pd.to_numeric(d.loc[d["slo_type"] == slo, col], errors="coerce").dropna()
                    if x.empty:
                        continue
                    rows.append({
                        "tag": tag,
                        "mode": mode,
                        "slo_type": slo,
                        "slo_name": SLO_NAME.get(slo, str(slo)),
                        "metric": col,
                        "p50": float(x.quantile(0.50)),
                        "p90": float(x.quantile(0.90)),
                        "p95": float(x.quantile(0.95)),
                        "p99": float(x.quantile(0.99)),
                        "mean": float(x.mean()),
                        "count": int(x.shape[0]),
                    })
        if rows:
            pd.DataFrame(rows).to_csv(pair_out / "summary_quantiles.csv", index=False)

    print(f"\n[DONE] Saved plots under: {out_root.resolve()}")


if __name__ == "__main__":
    main()

