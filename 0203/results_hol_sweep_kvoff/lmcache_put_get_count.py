#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# counters
C_CPU_PUT = "lmcache:store_from_gpu_time_count"
C_STO_PUT = "lmcache:store_put_time_count"
C_STO_GET = "lmcache:time_to_retrieve_count"
C_CPU_GET = "lmcache:retrieve_to_gpu_time_count"

# (optional) time sums if you want avg time later
S_CPU_PUT = "lmcache:store_from_gpu_time_sum"
S_STO_PUT = "lmcache:store_put_time_sum"
S_STO_GET = "lmcache:time_to_retrieve_sum"
S_CPU_GET = "lmcache:retrieve_to_gpu_time_sum"


def safe_diff_counter(x: pd.Series) -> pd.Series:
    """
    Convert cumulative counter -> per-row delta.
    If counter resets (negative diff), clamp to 0.
    """
    d = pd.to_numeric(x, errors="coerce").diff()
    d = d.fillna(0.0)
    d[d < 0] = 0.0
    return d


def prepare(df: pd.DataFrame, max_time_s: float) -> pd.DataFrame:
    if "finished_time" not in df.columns:
        raise ValueError("CSV must have finished_time column")

    df = df.sort_values("finished_time").reset_index(drop=True)
    t0 = df["finished_time"].min()
    df["t"] = df["finished_time"] - t0
    df = df[(df["t"] >= 0) & (df["t"] <= max_time_s)].reset_index(drop=True)
    df["sec"] = np.floor(df["t"]).astype(int)

    # deltas (counts)
    for c in [C_CPU_PUT, C_STO_PUT, C_STO_GET, C_CPU_GET]:
        if c in df.columns:
            df[c + "_d"] = safe_diff_counter(df[c])
        else:
            df[c + "_d"] = 0.0

    # deltas (sums) - optional
    for s in [S_CPU_PUT, S_STO_PUT, S_STO_GET, S_CPU_GET]:
        if s in df.columns:
            df[s + "_d"] = pd.to_numeric(df[s], errors="coerce").diff().fillna(0.0)
            df.loc[df[s + "_d"] < 0, s + "_d"] = 0.0
        else:
            df[s + "_d"] = 0.0

    return df


def per_second_series(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("sec", as_index=True).agg({
        C_CPU_PUT + "_d": "sum",
        C_STO_PUT + "_d": "sum",
        C_STO_GET + "_d": "sum",
        C_CPU_GET + "_d": "sum",
    })
    g = g.rename(columns={
        C_CPU_PUT + "_d": "cpu_put_cnt",
        C_STO_PUT + "_d": "storage_put_cnt",
        C_STO_GET + "_d": "storage_get_cnt",
        C_CPU_GET + "_d": "cpu_get_cnt",
    })
    g.index.name = "sec"
    return g.reset_index()


def plot_rates(df_sec: pd.DataFrame, outdir: Path, prefix: str, max_time_s: float):
    # PUT rates
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.plot(df_sec["sec"], df_sec["cpu_put_cnt"], label="CPU put (/s)")
    ax.plot(df_sec["sec"], df_sec["storage_put_cnt"], label="Storage put (/s)")
    ax.set_xlabel("elapsed time (s)")
    ax.set_ylabel("events per second")
    ax.set_title("PUT events per second (Δcount binned by 1s)")
    ax.set_xlim(0, max_time_s)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(outdir / f"{prefix}_put_rate.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # GET rates
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.plot(df_sec["sec"], df_sec["cpu_get_cnt"], label="CPU get (/s)  [host→GPU]")
    ax.plot(df_sec["sec"], df_sec["storage_get_cnt"], label="Storage get (/s) [retrieve path]")
    ax.set_xlabel("elapsed time (s)")
    ax.set_ylabel("events per second")
    ax.set_title("GET events per second (Δcount binned by 1s)")
    ax.set_xlim(0, max_time_s)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(outdir / f"{prefix}_get_rate.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="input CSV path")
    ap.add_argument("--outdir", default="put_get_out", help="output directory")
    ap.add_argument("--max_time_s", type=float, default=600.0, help="time window [0, max_time_s]")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df = prepare(df, args.max_time_s)
    df_sec = per_second_series(df)

    # totals
    totals = {
        "cpu_put_total": float(df_sec["cpu_put_cnt"].sum()),
        "storage_put_total": float(df_sec["storage_put_cnt"].sum()),
        "storage_get_total": float(df_sec["storage_get_cnt"].sum()),
        "cpu_get_total": float(df_sec["cpu_get_cnt"].sum()),
    }

    # save summary
    summary_path = outdir / "put_get_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"file: {csv_path}\n")
        f.write(f"time_window: 0..{args.max_time_s}s\n")
        for k, v in totals.items():
            f.write(f"{k} = {v:.0f}\n")

    # save per-second csv
    df_sec.to_csv(outdir / "put_get_per_second.csv", index=False)

    # plots
    prefix = csv_path.stem
    plot_rates(df_sec, outdir, prefix, args.max_time_s)

    print("[OK] wrote:", summary_path)
    print("[OK] wrote:", outdir / "put_get_per_second.csv")
    print("[OK] wrote:", outdir / f"{prefix}_put_rate.png")
    print("[OK] wrote:", outdir / f"{prefix}_get_rate.png")


if __name__ == "__main__":
    main()

