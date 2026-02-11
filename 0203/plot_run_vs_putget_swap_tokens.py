#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_run_vs_putget_swap_tokens.py

Goal:
- x-axis: elapsed time (s) in [0, 600] by default
- left y-axis: runtime request count (run_cnt)
- right y-axis: estimated "swap tokens" during put/get, binned per second

Important note (because of the CSV schema):
- This CSV does NOT provide explicit "tokens swapped" counters.
- We estimate tokens for put/get using:
    tokens_est ≈ (avg_speed_tokens_per_s) * (delta_time_sum_seconds)
  where avg_speed comes from lmcache:*_speed_sum/count and delta_time_sum comes from
  lmcache:*_time_sum.
- By default:
    PUT  uses store_put_time_sum  (closer to the "put" stage)
    GET  uses retrieve_to_gpu_time_sum (closer to the "get to GPU" stage)
  You can switch to total pipeline times via --use_total_time.

Outputs:
- PNG figure
- a small summary txt

Usage:
  python3 plot_run_vs_putget_swap_tokens.py --csv <file.csv> --outdir out --max_time_s 600
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_MAX_TIME_S = 600.0

# Required (left axis)
COL_RUN = "run_cnt"
COL_FIN = "finished_time"

# lmcache speed (tokens/s)
STORE_SPEED_SUM = "lmcache:store_speed_sum"
STORE_SPEED_CNT = "lmcache:store_speed_count"
RETR_SPEED_SUM  = "lmcache:retrieve_speed_sum"
RETR_SPEED_CNT  = "lmcache:retrieve_speed_count"

# stage times (seconds)
STORE_PUT_SUM   = "lmcache:store_put_time_sum"
RETR_TO_GPU_SUM = "lmcache:retrieve_to_gpu_time_sum"

# total pipeline times (seconds)
TIME_TO_STORE_SUM   = "lmcache:time_to_store_sum"
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
              RETR_SPEED_SUM, RETR_SPEED_CNT, retr_time_sum_col, COL_RUN]:
        if c not in df.columns:
            missing.append(c)
    return store_time_sum_col, retr_time_sum_col, missing


def build_series(df: pd.DataFrame, max_time_s: float, use_total_time: bool):
    store_time_sum_col, retr_time_sum_col, missing = compute_tokens_per_row(df, use_total_time)

    # run_cnt per second (gauge -> average within the second)
    run_sec = (pd.to_numeric(df.get(COL_RUN, 0), errors="coerce")
               .groupby(df["sec"]).mean())

    # token estimation per row (deltas)
    if missing:
        # Fill with zeros so the script still runs, but note in summary.
        store_tok_row = pd.Series(np.zeros(len(df)), index=df.index)
        retr_tok_row  = pd.Series(np.zeros(len(df)), index=df.index)
    else:
        d_store_speed_sum = pos_diff(df[STORE_SPEED_SUM])
        d_store_speed_cnt = pos_diff(df[STORE_SPEED_CNT])
        d_store_time_sum  = pos_diff(df[store_time_sum_col])

        avg_store_speed = np.where(d_store_speed_cnt.to_numpy() > 0,
                                   (d_store_speed_sum / d_store_speed_cnt).to_numpy(),
                                   0.0)
        store_tok_row = pd.Series(avg_store_speed * d_store_time_sum.to_numpy(), index=df.index)

        d_retr_speed_sum = pos_diff(df[RETR_SPEED_SUM])
        d_retr_speed_cnt = pos_diff(df[RETR_SPEED_CNT])
        d_retr_time_sum  = pos_diff(df[retr_time_sum_col])

        avg_retr_speed = np.where(d_retr_speed_cnt.to_numpy() > 0,
                                  (d_retr_speed_sum / d_retr_speed_cnt).to_numpy(),
                                  0.0)
        retr_tok_row = pd.Series(avg_retr_speed * d_retr_time_sum.to_numpy(), index=df.index)

    # aggregate to per-second "tokens/s"
    put_tok_sec = store_tok_row.groupby(df["sec"]).sum()
    get_tok_sec = retr_tok_row.groupby(df["sec"]).sum()

    # build full 0..max_sec index
    max_sec = int(np.floor(max_time_s))
    idx = np.arange(0, max_sec + 1)

    run_sec = run_sec.reindex(idx, fill_value=0.0)
    put_tok_sec = put_tok_sec.reindex(idx, fill_value=0.0)
    get_tok_sec = get_tok_sec.reindex(idx, fill_value=0.0)

    return idx, run_sec, put_tok_sec, get_tok_sec, missing, store_time_sum_col, retr_time_sum_col


def plot(idx, run_sec, put_tok_sec, get_tok_sec, out_png: Path, max_time_s: float,
         label: str, store_time_sum_col: str, retr_time_sum_col: str):
    fig, ax1 = plt.subplots(figsize=(14, 4.2))

    # left axis: running requests
    ax1.plot(idx, run_sec.values, label=f"{label} run_cnt (avg/s)")
    ax1.set_xlabel("elapsed time (s)")
    ax1.set_ylabel("running requests")
    ax1.set_xlim(0, max_time_s)
    ax1.grid(True, alpha=0.25)

    # right axis: tokens per second (estimated)
    ax2 = ax1.twinx()
    ax2.plot(idx, put_tok_sec.values, label=f"{label} put tokens/s (est) [{store_time_sum_col}]")
    ax2.plot(idx, get_tok_sec.values, label=f"{label} get tokens/s (est) [{retr_time_sum_col}]")
    ax2.set_ylabel("swap tokens per second (estimated)")
    ax2.set_xlim(0, max_time_s)

    # merged legend
    lines, labels = [], []
    for a in (ax1, ax2):
        l, lab = a.get_legend_handles_labels()
        lines += l
        labels += lab
    ax1.legend(lines, labels, loc="upper right")

    title = f"Running requests vs put/get swap tokens (0..{int(max_time_s)}s)"
    ax1.set_title(title)

    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="input CSV path")
    ap.add_argument("--outdir", default="run_vs_putget_tokens", help="output directory")
    ap.add_argument("--max_time_s", type=float, default=DEFAULT_MAX_TIME_S, help="time window [0, max_time_s]")
    ap.add_argument("--label", default="", help="label for legend (default: filename stem)")
    ap.add_argument("--use_total_time", action="store_true",
                    help="Use time_to_store_sum/time_to_retrieve_sum instead of put/to_gpu stage times.")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_and_window(csv_path, args.max_time_s)
    label = args.label.strip() or csv_path.stem

    idx, run_sec, put_tok_sec, get_tok_sec, missing, store_time_sum_col, retr_time_sum_col = build_series(
        df, args.max_time_s, args.use_total_time
    )

    out_png = outdir / f"{csv_path.stem}_run_vs_putget_tokens.png"
    plot(idx, run_sec, put_tok_sec, get_tok_sec, out_png, args.max_time_s, label, store_time_sum_col, retr_time_sum_col)

    # summary
    total_put = float(put_tok_sec.sum())
    total_get = float(get_tok_sec.sum())
    summary = []
    summary.append(f"file: {csv_path}")
    summary.append(f"time_window: 0..{args.max_time_s:.0f}s")
    summary.append(f"label: {label}")
    summary.append(f"use_total_time: {args.use_total_time}")
    summary.append(f"store_time_sum_col: {store_time_sum_col}")
    summary.append(f"retr_time_sum_col:  {retr_time_sum_col}")
    summary.append("")
    if missing:
        summary.append("MISSING COLUMNS (tokens estimation disabled -> zeros):")
        for m in missing:
            summary.append(f"  - {m}")
        summary.append("")
    summary.append("ESTIMATED token totals within window:")
    summary.append(f"  put_tokens_total_est = {total_put:.0f} tokens")
    summary.append(f"  get_tokens_total_est = {total_get:.0f} tokens")
    summary.append("")
    summary.append("NOTE: These token counts are ESTIMATES derived from avg_speed(tokens/s) * delta_time_sum(s).")
    summary.append("      If you need exact swapped-token counts, add explicit counters in lmcache/vllm instrumentation.")

    out_txt = outdir / f"{csv_path.stem}_run_vs_putget_tokens_summary.txt"
    out_txt.write_text("\n".join(summary), encoding="utf-8")

    print("[OK] wrote:", out_png)
    print("[OK] wrote:", out_txt)


if __name__ == "__main__":
    main()
