#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""plot_swap_throughput_timeseries.py

시간 시퀀스 기반으로 swap_cnt와 처리량(throughput) 감소 관계를 시각화.

사용 컬럼(로그에 존재할 때만 사용):
- finished_time (권장) 또는 insertion_time: 시간축 (단위: seconds)
- processed_count: 처리량 계산용(누적 카운터일 경우 Δ/Δt로 req/s 계산)
- swap_cnt: 스왑 관련 지표(instantaneous 또는 누적 카운터 자동 판별)
- (옵션) run_cnt, wait_cnt, vram_used_gib, vram_total_gib

핵심 플롯(각 파일별 생성):
1) Throughput vs Swap (dual-axis) + swap 구간 shading
2) Normalized throughput (vs swap=0 baseline) + swap overlay
3) Scatter: throughput vs swap metric
4) Lag-correlation: swap metric vs throughput (shifted) 상관

출력 기본 경로: ./maxsize/fig_swap (없으면 자동 생성)

의존성:
  pip install pandas numpy matplotlib seaborn

예시:
  python plot_swap_throughput_timeseries.py --logfile ./sort/xxx.csv
  python plot_swap_throughput_timeseries.py --logdir ./sort --pattern "timsort_hol_sleep_*_mb*_*.csv"
"""

import argparse
import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

FNAME_RE = re.compile(
    r".*sleep_(?P<sleep>[\d\.]+)_pushint_(?P<pushint>[\d\.]+)_mb(?P<mb>\d+)_.*\.csv$"
)


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def savefig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def load_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, engine="python")


def is_mostly_monotone_nondec(x: pd.Series, frac: float = 0.98) -> bool:
    v = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 5:
        return False
    dv = np.diff(v)
    # allow small noise; treat -eps as 0
    eps = 1e-9
    nondec = np.mean(dv >= -eps)
    return nondec >= frac


def pick_time_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["finished_time", "insertion_time"]:
        if c in df.columns:
            return c
    return None


def parse_meta(path: str) -> Tuple[Optional[float], Optional[float], Optional[int]]:
    m = FNAME_RE.match(os.path.basename(path))
    if not m:
        return None, None, None
    return float(m.group("sleep")), float(m.group("pushint")), int(m.group("mb"))


def build_timeseries(
    df: pd.DataFrame,
    time_col: str,
    bin_s: float,
    smooth_bins: int,
    swap_agg: str,
    swap_threshold: float,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Return ts dataframe indexed by t_bin_start (seconds since start), plus a dict of inferred modes."""

    info: Dict[str, str] = {}

    # numeric and sort
    d = df.copy()
    d[time_col] = pd.to_numeric(d[time_col], errors="coerce")
    d = d[np.isfinite(d[time_col])].sort_values(time_col)
    if len(d) == 0:
        raise ValueError(f"No valid time values in column {time_col}")

    t0 = float(d[time_col].iloc[0])
    d["t_rel"] = d[time_col] - t0

    # define bins
    max_t = float(d["t_rel"].max())
    nbins = int(np.floor(max_t / bin_s)) + 1
    edges = np.arange(0, nbins * bin_s + 1e-12, bin_s)
    d["tbin"] = pd.cut(d["t_rel"], bins=edges, include_lowest=True, right=False, labels=False)
    d = d[np.isfinite(d["tbin"])].copy()
    d["tbin"] = d["tbin"].astype(int)

    # throughput
    if "processed_count" in d.columns and is_mostly_monotone_nondec(d["processed_count"]):
        info["throughput_mode"] = "processed_count_counter"
        pc = pd.to_numeric(d["processed_count"], errors="coerce")
        g = d.assign(_pc=pc).groupby("tbin")
        proc_in_bin = (g["_pc"].max() - g["_pc"].min()).fillna(0.0)
        thr = (proc_in_bin / bin_s).rename("throughput_rps")
    else:
        info["throughput_mode"] = "finished_events_per_bin"
        thr = (d.groupby("tbin").size() / bin_s).rename("throughput_rps")

    # swap metric
    if "swap_cnt" in d.columns:
        sc = pd.to_numeric(d["swap_cnt"], errors="coerce")
        g = d.assign(_sc=sc).groupby("tbin")

        if is_mostly_monotone_nondec(sc):
            info["swap_mode"] = "swap_cnt_counter"
            swap_events = (g["_sc"].max() - g["_sc"].min()).fillna(0.0)
            swap_metric = (swap_events / bin_s).rename("swap (count/s)")
        else:
            info["swap_mode"] = f"swap_cnt_instant_{swap_agg}"
            if swap_agg == "max":
                swap_metric = g["_sc"].max().rename("swap_cnt")
            elif swap_agg == "mean":
                swap_metric = g["_sc"].mean().rename("swap_cnt")
            else:
                swap_metric = g["_sc"].median().rename("swap_cnt")
    else:
        info["swap_mode"] = "missing"
        swap_metric = pd.Series(dtype=float, name="swap")

    # optional signals
    def agg_opt(col: str, how: str = "mean") -> pd.Series:
        if col not in d.columns:
            return pd.Series(dtype=float, name=col)
        v = pd.to_numeric(d[col], errors="coerce")
        gg = d.assign(_v=v).groupby("tbin")
        if how == "max":
            return gg["_v"].max().rename(col)
        return gg["_v"].mean().rename(col)

    run_mean = agg_opt("run_cnt", "mean")
    wait_mean = agg_opt("wait_cnt", "mean")
    vram_used_mean = agg_opt("vram_used_gib", "mean")
    vram_total_med = float(np.nanmedian(pd.to_numeric(d.get("vram_total_gib", pd.Series(dtype=float)), errors="coerce"))) if "vram_total_gib" in d.columns else float("nan")

    # combine
    ts = pd.concat([thr, swap_metric, run_mean, wait_mean, vram_used_mean], axis=1).sort_index()
    ts.index = (ts.index.astype(float) * bin_s).rename("t_s")  # bin start time

    # add derived columns
    if "vram_used_gib" in ts.columns and np.isfinite(vram_total_med) and vram_total_med > 0:
        ts["vram_used_ratio"] = ts["vram_used_gib"] / vram_total_med

    # smooth
    if smooth_bins and smooth_bins > 1:
        for col in ts.columns:
            ts[col] = ts[col].rolling(smooth_bins, min_periods=max(1, smooth_bins // 2)).mean()
        info["smoothing"] = f"rolling_mean_{smooth_bins}_bins"
    else:
        info["smoothing"] = "none"

    # swap-active mask for shading
    swap_col = "swap (count/s)" if "swap (count/s)" in ts.columns else ("swap_cnt" if "swap_cnt" in ts.columns else None)
    if swap_col is not None:
        ts["swap_active"] = (ts[swap_col] > swap_threshold).astype(int)
    else:
        ts["swap_active"] = 0

    info["swap_threshold"] = str(swap_threshold)

    return ts, info


def shade_regions(ax: plt.Axes, t: np.ndarray, active: np.ndarray, color: str = "#ffcccb", alpha: float = 0.25):
    """Shade contiguous regions where active==1."""
    if len(t) == 0:
        return
    active = np.asarray(active).astype(int)
    # find segments
    in_seg = False
    start = None
    for i in range(len(t)):
        if active[i] == 1 and not in_seg:
            in_seg = True
            start = t[i]
        if in_seg and (active[i] == 0 or i == len(t) - 1):
            end = t[i] if active[i] == 0 else t[i] + (t[1] - t[0] if len(t) > 1 else 1.0)
            ax.axvspan(start, end, color=color, alpha=alpha, lw=0)
            in_seg = False
            start = None


def plot_dual_axis(ts: pd.DataFrame, outdir: str, title_prefix: str):
    if "throughput_rps" not in ts.columns:
        return

    t = ts.index.to_numpy(dtype=float)
    y_thr = ts["throughput_rps"].to_numpy(dtype=float)

    swap_col = "swap (count/s)" if "swap (count/s)" in ts.columns else ("swap_cnt" if "swap_cnt" in ts.columns else None)

    fig, ax1 = plt.subplots(figsize=(10.5, 4.2))
    ax1.plot(t, y_thr, color="#1f77b4", label="throughput (req/s)")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("throughput (req/s)")

    shade_regions(ax1, t, ts["swap_active"].to_numpy(dtype=int))

    if swap_col is not None:
        ax2 = ax1.twinx()
        ax2.plot(t, ts[swap_col].to_numpy(dtype=float), color="#d62728", alpha=0.85, label=swap_col)
        ax2.set_ylabel(swap_col)

        # legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    else:
        ax1.legend(loc="upper right")

    ax1.set_title(f"{title_prefix} Throughput vs Swap (shaded: swap_active)")
    savefig(os.path.join(outdir, "ts_throughput_vs_swap.png"))


def plot_normalized(ts: pd.DataFrame, outdir: str, title_prefix: str):
    if "throughput_rps" not in ts.columns:
        return

    swap_col = "swap (count/s)" if "swap (count/s)" in ts.columns else ("swap_cnt" if "swap_cnt" in ts.columns else None)

    thr = ts["throughput_rps"].astype(float)
    baseline = thr[ts["swap_active"] == 0].mean()
    if not np.isfinite(baseline) or baseline <= 0:
        baseline = thr.mean()

    ts2 = ts.copy()
    ts2["thr_norm"] = thr / (baseline if baseline else np.nan)

    t = ts2.index.to_numpy(dtype=float)

    fig, ax1 = plt.subplots(figsize=(10.5, 4.2))
    ax1.plot(t, ts2["thr_norm"].to_numpy(dtype=float), color="#1f77b4", label="throughput / baseline")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("normalized throughput")
    ax1.axhline(1.0, color="gray", lw=1, ls="--", alpha=0.7)

    shade_regions(ax1, t, ts2["swap_active"].to_numpy(dtype=int))

    if swap_col is not None:
        ax2 = ax1.twinx()
        ax2.plot(t, ts2[swap_col].to_numpy(dtype=float), color="#d62728", alpha=0.85, label=swap_col)
        ax2.set_ylabel(swap_col)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    else:
        ax1.legend(loc="upper right")

    ax1.set_title(f"{title_prefix} Normalized Throughput vs Swap")
    savefig(os.path.join(outdir, "ts_normalized_throughput.png"))


def plot_scatter(ts: pd.DataFrame, outdir: str, title_prefix: str):
    if "throughput_rps" not in ts.columns:
        return

    swap_col = "swap (count/s)" if "swap (count/s)" in ts.columns else ("swap_cnt" if "swap_cnt" in ts.columns else None)
    if swap_col is None:
        return

    d = ts[["throughput_rps", swap_col]].dropna()
    if len(d) < 10:
        return

    plt.figure(figsize=(6.8, 4.8))
    sns.regplot(data=d, x=swap_col, y="throughput_rps", scatter_kws={"s": 18, "alpha": 0.6}, line_kws={"color": "#1f77b4"})
    plt.title(f"{title_prefix} Throughput vs {swap_col} (binned time points)")
    plt.xlabel(swap_col)
    plt.ylabel("throughput (req/s)")
    savefig(os.path.join(outdir, "scatter_throughput_vs_swap.png"))


def plot_lag_corr(ts: pd.DataFrame, outdir: str, title_prefix: str, max_lag_bins: int):
    if "throughput_rps" not in ts.columns:
        return

    swap_col = "swap (count/s)" if "swap (count/s)" in ts.columns else ("swap_cnt" if "swap_cnt" in ts.columns else None)
    if swap_col is None:
        return

    x = ts[swap_col].astype(float)
    y = ts["throughput_rps"].astype(float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if len(x) < 20:
        return

    # z-score to compare correlations
    xz = (x - x.mean()) / (x.std() if x.std() else 1.0)
    yz = (y - y.mean()) / (y.std() if y.std() else 1.0)

    lags = np.arange(-max_lag_bins, max_lag_bins + 1)
    corrs = []
    for lag in lags:
        if lag < 0:
            a = xz[:lag]
            b = yz[-lag:]
        elif lag > 0:
            a = xz[lag:]
            b = yz[:-lag]
        else:
            a = xz
            b = yz
        if len(a) < 10:
            corrs.append(np.nan)
            continue
        corrs.append(float(np.corrcoef(a, b)[0, 1]))

    plt.figure(figsize=(7.2, 4.2))
    plt.plot(lags, corrs, marker="o")
    plt.axvline(0, color="gray", lw=1)
    plt.xlabel("lag (bins)  [positive: swap leads throughput]")
    plt.ylabel("corr")
    plt.title(f"{title_prefix} Lag-Correlation: {swap_col} vs throughput")
    savefig(os.path.join(outdir, "lagcorr_swap_vs_throughput.png"))


def write_run_info(info: Dict[str, str], outdir: str, path: str):
    p = os.path.join(outdir, "run_inference.txt")
    lines = [f"file: {os.path.basename(path)}"]
    for k in sorted(info.keys()):
        lines.append(f"{k}: {info[k]}")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def process_file(path: str, outdir: str, args: argparse.Namespace):
    df = load_csv(path)
    time_col = pick_time_col(df)
    if time_col is None:
        raise ValueError(f"No time column found (need finished_time or insertion_time): {path}")

    sleep, pushint, mb = parse_meta(path)
    title_prefix = ""
    if mb is not None:
        title_prefix = f"mb{mb} "

    ts, info = build_timeseries(
        df,
        time_col=time_col,
        bin_s=args.bin_s,
        smooth_bins=args.smooth_bins,
        swap_agg=args.swap_agg,
        swap_threshold=args.swap_threshold,
    )

    # add meta info
    if sleep is not None:
        info["sleep"] = str(sleep)
    if pushint is not None:
        info["pushint"] = str(pushint)
    if mb is not None:
        info["mb"] = str(mb)
    info["time_col"] = time_col
    info["bin_s"] = str(args.bin_s)

    # plots
    plot_dual_axis(ts, outdir, title_prefix)
    plot_normalized(ts, outdir, title_prefix)
    plot_scatter(ts, outdir, title_prefix)
    plot_lag_corr(ts, outdir, title_prefix, args.max_lag_bins)

    # also save the binned timeseries
    ts.to_csv(os.path.join(outdir, "binned_timeseries.csv"), index=True)
    write_run_info(info, outdir, path)


def main(args: argparse.Namespace) -> None:
    ensure_dir(args.outdir)

    files: List[str] = []
    if args.logfile:
        files = [args.logfile]
    else:
        files = sorted(glob.glob(os.path.join(args.logdir, args.pattern)))

    if not files:
        raise SystemExit("No input files found.")

    # group by (sleep, pushint) if meta parse succeeds; else put in "misc"
    groups: Dict[str, List[str]] = {}
    for p in files:
        sleep, pushint, mb = parse_meta(p)
        if sleep is None or pushint is None:
            gname = "misc"
        else:
            gname = f"sleep_{sleep}_pushint_{pushint}"
        groups.setdefault(gname, []).append(p)

    for gname, flist in groups.items():
        gdir = os.path.join(args.outdir, gname)
        ensure_dir(gdir)

        for p in flist:
            _, _, mb = parse_meta(p)
            sub = gdir
            if mb is not None:
                sub = os.path.join(gdir, f"mb{mb}")
                ensure_dir(sub)

            print(f"Processing: {os.path.basename(p)} -> {sub}")
            try:
                process_file(p, sub, args)
            except Exception as e:
                print(f"  [error] {os.path.basename(p)}: {e}")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logfile", type=str, default=None, help="single csv file to process")
    parser.add_argument("--logdir", type=str, default=".", help="directory containing csv logs")
    parser.add_argument(
        "--pattern",
        type=str,
        default="timsort_hol_sleep_*_pushint_*_mb*_*.csv",
        help="glob pattern for csv files",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="./maxsize/fig_swap",
        help="output directory for figures (default: ./maxsize/fig_swap)",
    )
    parser.add_argument("--bin_s", type=float, default=1.0, help="bin size in seconds")
    parser.add_argument("--smooth_bins", type=int, default=5, help="rolling mean window in bins (0/1 disables)")
    parser.add_argument("--swap_agg", type=str, default="max", choices=["max", "mean", "median"], help="aggregation for instantaneous swap_cnt")
    parser.add_argument("--swap_threshold", type=float, default=0.0, help="swap_active threshold on swap metric")
    parser.add_argument("--max_lag_bins", type=int, default=30, help="max lag (bins) for lag-correlation plot")

    main(parser.parse_args())

