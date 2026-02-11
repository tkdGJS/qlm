#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_run_vs_putget_swap_tokens_v2.py

What it draws (within elapsed time [0, max_time_s], default 600s):

1) run_cnt vs swap in/out tokens/s (estimated)
   - x: elapsed time (s)
   - left y (RED): run_cnt (avg within each 1s bin)
   - right y: swap tokens/s (estimated), shown as:
       * total (put+get)
       * (optional) also put and get separately

2) Interactive-only plot (slo_type=0)
   - x: elapsed time (s)
   - left y (RED): TTFT rolling p95 (over last N completed interactive requests)
   - right y: swap tokens/s (estimated, total put+get)

3) Batch-only plot (slo_type=1)
   - x: elapsed time (s)
   - left y (RED): TTLT rolling p95 (over last N completed batch requests)
   - right y: swap tokens/s (estimated, total put+get)

Important notes about "swap tokens":
- This CSV schema does NOT provide explicit "tokens swapped in/out" counters.
- We estimate swap tokens using:
    tokens_est ≈ (avg_speed_tokens_per_s) * (delta_time_sum_seconds)
  where avg_speed comes from lmcache:*_speed_sum/count and delta_time_sum comes from
  lmcache:*_time_sum.
- By default:
    PUT uses store_put_time_sum (put-stage time)
    GET uses retrieve_to_gpu_time_sum (host->GPU time, i.e., "get" stage)
  You can switch to total pipeline times via --use_total_time.

CPU+storage label meaning:
- It's just a label you pass for the experiment setting where LMCache can spill to a storage tier
  (local_storage_usage increases). Whether that storage is disk/SSD depends on your backend config.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_MAX_TIME_S = 600.0

# Core columns
COL_FIN = "finished_time"
COL_RUN = "run_cnt"
COL_SLO = "slo_type"
COL_TTFT = "ttft"
COL_TTLT = "ttlt"

# lmcache speed (tokens/s)
STORE_SPEED_SUM = "lmcache:store_speed_sum"
STORE_SPEED_CNT = "lmcache:store_speed_count"
RETR_SPEED_SUM  = "lmcache:retrieve_speed_sum"
RETR_SPEED_CNT  = "lmcache:retrieve_speed_count"

# stage times (seconds)
STORE_PUT_SUM   = "lmcache:store_put_time_sum"
RETR_TO_GPU_SUM = "lmcache:retrieve_to_gpu_time_sum"

# total pipeline times (seconds)
TIME_TO_STORE_SUM    = "lmcache:time_to_store_sum"
TIME_TO_RETRIEVE_SUM = "lmcache:time_to_retrieve_sum"


def pos_diff(s: pd.Series) -> pd.Series:
    """diff and clamp negatives to 0 (handles counter resets)."""
    d = pd.to_numeric(s, errors="coerce").diff().fillna(0.0)
    d[d < 0] = 0.0
    return d


def load_and_window(csv_path: Path, max_time_s: float) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if COL_FIN not in df.columns:
        raise ValueError(f"CSV must have column '{COL_FIN}'")
    df = df.sort_values(COL_FIN).reset_index(drop=True)
    t0 = float(pd.to_numeric(df[COL_FIN], errors="coerce").min())
    df["t"] = pd.to_numeric(df[COL_FIN], errors="coerce") - t0
    df = df[(df["t"] >= 0) & (df["t"] <= max_time_s)].reset_index(drop=True)
    df["sec"] = np.floor(df["t"]).astype(int)
    return df


def compute_tokens_per_row(df: pd.DataFrame, use_total_time: bool):
    # pick time columns
    store_time_sum_col = TIME_TO_STORE_SUM if use_total_time else STORE_PUT_SUM
    retr_time_sum_col  = TIME_TO_RETRIEVE_SUM if use_total_time else RETR_TO_GPU_SUM

    missing = []
    for c in [STORE_SPEED_SUM, STORE_SPEED_CNT, store_time_sum_col,
              RETR_SPEED_SUM, RETR_SPEED_CNT, retr_time_sum_col]:
        if c not in df.columns:
            missing.append(c)
    return store_time_sum_col, retr_time_sum_col, missing


def build_swap_tokens_series(df: pd.DataFrame, max_time_s: float, use_total_time: bool):
    """
    Returns:
      idx (0..max_sec),
      put_tok_sec, get_tok_sec, total_tok_sec,
      store_time_sum_col, retr_time_sum_col, missing_cols
    """
    store_time_sum_col, retr_time_sum_col, missing = compute_tokens_per_row(df, use_total_time)

    # token estimation per row (deltas)
    if missing:
        store_tok_row = pd.Series(np.zeros(len(df)), index=df.index)
        retr_tok_row  = pd.Series(np.zeros(len(df)), index=df.index)
    else:
        d_store_speed_sum = pos_diff(df[STORE_SPEED_SUM])
        d_store_speed_cnt = pos_diff(df[STORE_SPEED_CNT])
        d_store_time_sum  = pos_diff(df[store_time_sum_col])

        avg_store_speed = np.where(
            d_store_speed_cnt.to_numpy() > 0,
            (d_store_speed_sum / d_store_speed_cnt).to_numpy(),
            0.0,
        )
        store_tok_row = pd.Series(avg_store_speed * d_store_time_sum.to_numpy(), index=df.index)

        d_retr_speed_sum = pos_diff(df[RETR_SPEED_SUM])
        d_retr_speed_cnt = pos_diff(df[RETR_SPEED_CNT])
        d_retr_time_sum  = pos_diff(df[retr_time_sum_col])

        avg_retr_speed = np.where(
            d_retr_speed_cnt.to_numpy() > 0,
            (d_retr_speed_sum / d_retr_speed_cnt).to_numpy(),
            0.0,
        )
        retr_tok_row = pd.Series(avg_retr_speed * d_retr_time_sum.to_numpy(), index=df.index)

    # aggregate to per-second "tokens/s"
    put_tok_sec = store_tok_row.groupby(df["sec"]).sum()
    get_tok_sec = retr_tok_row.groupby(df["sec"]).sum()

    max_sec = int(np.floor(max_time_s))
    idx = np.arange(0, max_sec + 1)

    put_tok_sec = put_tok_sec.reindex(idx, fill_value=0.0)
    get_tok_sec = get_tok_sec.reindex(idx, fill_value=0.0)
    total_tok_sec = put_tok_sec + get_tok_sec

    return idx, put_tok_sec, get_tok_sec, total_tok_sec, store_time_sum_col, retr_time_sum_col, missing


def build_run_cnt_series(df: pd.DataFrame, max_time_s: float):
    """
    run_cnt is a gauge, so per-second we take average within the second.
    Returns idx and run_cnt_sec (aligned to idx).
    """
    max_sec = int(np.floor(max_time_s))
    idx = np.arange(0, max_sec + 1)

    if COL_RUN in df.columns:
        run_sec = (pd.to_numeric(df[COL_RUN], errors="coerce")
                   .groupby(df["sec"]).mean())
    else:
        run_sec = pd.Series(dtype=float)

    run_sec = run_sec.reindex(idx, fill_value=0.0)
    return idx, run_sec


def rolling_p95_latency_per_sec(df: pd.DataFrame, metric_col: str, slo_value: int,
                                max_time_s: float, window: int, ffill: bool) -> pd.Series:
    """
    Compute rolling p95 over last N completed requests of a given slo_type,
    then downsample to per-second by taking the last rolling value within each second.

    Returns series aligned to idx (0..max_sec). Values are in seconds.
    """
    max_sec = int(np.floor(max_time_s))
    idx = np.arange(0, max_sec + 1)

    if (COL_SLO not in df.columns) or (metric_col not in df.columns):
        return pd.Series(index=idx, data=np.nan)

    sub = df.loc[df[COL_SLO] == slo_value, ["t", "sec", metric_col]].copy()
    sub[metric_col] = pd.to_numeric(sub[metric_col], errors="coerce")
    sub = sub.dropna()
    if sub.empty:
        return pd.Series(index=idx, data=np.nan)

    # Sort by time and compute rolling p95 over requests
    sub = sub.sort_values("t")
    sub["roll_p95"] = sub[metric_col].rolling(window, min_periods=max(10, window // 5)).quantile(0.95)

    # last roll value per second
    per_sec = sub.groupby("sec")["roll_p95"].last()
    per_sec = per_sec.reindex(idx)

    if ffill:
        per_sec = per_sec.ffill()

    return per_sec


def plot_run_vs_swap(idx, run_sec, put_tok_sec, get_tok_sec, total_tok_sec,
                     out_png: Path, max_time_s: float, label: str,
                     show_put_get: bool):
    fig, ax1 = plt.subplots(figsize=(14, 4.2))

    # left axis: run_cnt (RED)
    ax1.plot(idx, run_sec.values, color="red", label=f"{label} run_cnt (avg/s)")
    ax1.set_xlabel("elapsed time (s)")
    ax1.set_ylabel("running requests")
    ax1.set_xlim(0, max_time_s)
    ax1.grid(True, alpha=0.25)

    # right axis: swap tokens/s (estimated)
    ax2 = ax1.twinx()
    ax2.plot(idx, total_tok_sec.values, label=f"{label} swap tokens/s (est, in+out)")
    if show_put_get:
        ax2.plot(idx, put_tok_sec.values, linestyle="--", label=f"{label} swap out tokens/s (est)")
        ax2.plot(idx, get_tok_sec.values, linestyle="--", label=f"{label} swap in tokens/s (est)")
    ax2.set_ylabel("swap tokens per second (estimated)")
    ax2.set_xlim(0, max_time_s)

    # merged legend
    lines, labels = [], []
    for a in (ax1, ax2):
        l, lab = a.get_legend_handles_labels()
        lines += l
        labels += lab
    ax1.legend(lines, labels, loc="upper right")

    ax1.set_title(f"run_cnt vs swap in/out tokens/s (0..{int(max_time_s)}s)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_latency_vs_swap(idx, lat_sec, total_tok_sec, out_png: Path,
                         max_time_s: float, label: str,
                         left_label: str, title: str):
    fig, ax1 = plt.subplots(figsize=(14, 4.2))

    # left axis: latency rolling p95 (RED)
    ax1.plot(idx, lat_sec.values, color="red", label=f"{label} {left_label}")
    ax1.set_xlabel("elapsed time (s)")
    ax1.set_ylabel("latency (s)")
    ax1.set_xlim(0, max_time_s)
    ax1.grid(True, alpha=0.25)

    # right axis: swap tokens/s
    ax2 = ax1.twinx()
    ax2.plot(idx, total_tok_sec.values, label=f"{label} swap tokens/s (est, in+out)")
    ax2.set_ylabel("swap tokens per second (estimated)")
    ax2.set_xlim(0, max_time_s)

    # merged legend
    lines, labels = [], []
    for a in (ax1, ax2):
        l, lab = a.get_legend_handles_labels()
        lines += l
        labels += lab
    ax1.legend(lines, labels, loc="upper right")

    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="input CSV path")
    ap.add_argument("--outdir", default="swap_tokens_plots", help="output directory")
    ap.add_argument("--max_time_s", type=float, default=DEFAULT_MAX_TIME_S, help="time window [0, max_time_s]")
    ap.add_argument("--label", default="", help="label for legend (default: filename stem)")
    ap.add_argument("--use_total_time", action="store_true",
                    help="Use time_to_store_sum/time_to_retrieve_sum instead of put/to_gpu stage times for swap-token estimation.")
    ap.add_argument("--rolling_window", type=int, default=50, help="rolling window (requests) for p95 latency curves")
    ap.add_argument("--no_ffill_latency", action="store_true",
                    help="Do not forward-fill per-second rolling latency values (leaves gaps when no samples in a second).")
    ap.add_argument("--show_put_get", action="store_true",
                    help="In run_cnt plot, also show put and get token curves (dashed).")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_and_window(csv_path, args.max_time_s)
    label = args.label.strip() or csv_path.stem
    ffill_latency = not args.no_ffill_latency

    # swap token series
    idx, put_tok_sec, get_tok_sec, total_tok_sec, store_time_sum_col, retr_time_sum_col, missing = build_swap_tokens_series(
        df, args.max_time_s, args.use_total_time
    )

    # run_cnt series
    _, run_sec = build_run_cnt_series(df, args.max_time_s)

    # 1) run_cnt vs swap tokens
    out_png1 = outdir / f"{csv_path.stem}_run_cnt_vs_swap_tokens.png"
    plot_run_vs_swap(idx, run_sec, put_tok_sec, get_tok_sec, total_tok_sec,
                     out_png1, args.max_time_s, label, args.show_put_get)

    # 2) Interactive: TTFT rolling p95 vs swap tokens
    ttft_p95_sec = rolling_p95_latency_per_sec(
        df, metric_col=COL_TTFT, slo_value=0,
        max_time_s=args.max_time_s, window=args.rolling_window, ffill=ffill_latency
    )
    out_png2 = outdir / f"{csv_path.stem}_interactive_ttft_p95_vs_swap_tokens.png"
    plot_latency_vs_swap(
        idx, ttft_p95_sec, total_tok_sec, out_png2,
        args.max_time_s, label,
        left_label=f"TTFT rolling p95 (win={args.rolling_window})",
        title=f"Interactive: TTFT rolling p95 vs swap in/out tokens/s (0..{int(args.max_time_s)}s)"
    )

    # 3) Batch: TTLT rolling p95 vs swap tokens
    ttlt_p95_sec = rolling_p95_latency_per_sec(
        df, metric_col=COL_TTLT, slo_value=1,
        max_time_s=args.max_time_s, window=args.rolling_window, ffill=ffill_latency
    )
    out_png3 = outdir / f"{csv_path.stem}_batch_ttlt_p95_vs_swap_tokens.png"
    plot_latency_vs_swap(
        idx, ttlt_p95_sec, total_tok_sec, out_png3,
        args.max_time_s, label,
        left_label=f"TTLT rolling p95 (win={args.rolling_window})",
        title=f"batch: TTLT rolling p95 vs swap in/out tokens/s (0..{int(args.max_time_s)}s)"
    )

    # summary
    total_put = float(put_tok_sec.sum())
    total_get = float(get_tok_sec.sum())
    total_swap = float(total_tok_sec.sum())

    summary = []
    summary.append(f"file: {csv_path}")
    summary.append(f"time_window: 0..{args.max_time_s:.0f}s")
    summary.append(f"label: {label}")
    summary.append(f"use_total_time: {args.use_total_time}")
    summary.append(f"rolling_window: {args.rolling_window}")
    summary.append(f"ffill_latency: {ffill_latency}")
    summary.append("")
    summary.append("swap-token estimation uses:")
    summary.append(f"  store_time_sum_col = {store_time_sum_col}")
    summary.append(f"  retr_time_sum_col  = {retr_time_sum_col}")
    summary.append("")
    if missing:
        summary.append("MISSING COLUMNS (swap token estimation becomes zeros):")
        for m in missing:
            summary.append(f"  - {m}")
        summary.append("")
    summary.append("ESTIMATED swap-token totals within window:")
    summary.append(f"  put_tokens_total_est  = {total_put:.0f} tokens")
    summary.append(f"  get_tokens_total_est  = {total_get:.0f} tokens")
    summary.append(f"  swap_tokens_total_est = {total_swap:.0f} tokens (put+get)")
    summary.append("")
    summary.append("NOTE: swap tokens are ESTIMATES derived from avg_speed(tokens/s) * delta_time_sum(s).")
    summary.append("      If you need exact swapped-token counts, add explicit counters in lmcache/vllm instrumentation.")

    out_txt = outdir / f"{csv_path.stem}_swap_tokens_summary.txt"
    out_txt.write_text("\n".join(summary), encoding="utf-8")

    print("[OK] wrote:", out_png1)
    print("[OK] wrote:", out_png2)
    print("[OK] wrote:", out_png3)
    print("[OK] wrote:", out_txt)


if __name__ == "__main__":
    main()
