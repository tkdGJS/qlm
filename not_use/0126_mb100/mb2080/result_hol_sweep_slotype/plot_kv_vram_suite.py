#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""plot_kv_vram_suite.py

목적
- max batch size(mb) 증가에 따라
  1) KV cache 상태(kv)
  2) running/waiting/swap 요청 수(run_cnt, wait_cnt, swap_cnt)
  3) VRAM 사용량(vram_used_gib, vram_total_gib)
  4) VRAM 내 KV-cache 비율(kv/vram_*) 및 그 증가율
을 시각화

가정
- 로그 단위: **second** (time-related cols가 있어도 본 스크립트는 초 단위 그대로 사용)
- kv 컬럼은 "KV cache 규모"를 나타내는 값(대개 GiB 또는 토큰/블록 수)으로 가정.
  단위가 GiB가 아니더라도, 비율 계산은 (kv / vram_*)가 의미 있을 때만 해석하세요.

입력
- timsort_hol_sleep_*_pushint_*_mb*_*.csv
- 필수 컬럼: kv, run_cnt, wait_cnt, swap_cnt, vram_used_gib, vram_total_gib
- (선택) finished_time 또는 insertion_time이 있으면 time-series 스타일 플롯에 활용

출력
- 기본 저장 경로: ./maxsize/fig_kv
- (sleep, pushint) 그룹별 하위 폴더에 저장

사용법
  python plot_kv_vram_suite.py --logdir ./sort

의존성
  pip install pandas numpy matplotlib seaborn
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


def ecdf(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.array([]), np.array([])
    x = np.sort(v)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def q(x: pd.Series, quant: float) -> float:
    v = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    return float(np.quantile(v, quant)) if len(v) else float("nan")


@dataclass
class RunMeta:
    path: str
    sleep: float
    pushint: float
    mb: int


@dataclass
class RunAgg:
    meta: RunMeta
    reqs: int

    kv_mean: float
    kv_p50: float
    kv_p95: float

    run_mean: float
    wait_mean: float
    swap_mean: float

    vram_used_mean: float
    vram_used_p95: float
    vram_total: float

    kv_ratio_used_mean: float
    kv_ratio_total_mean: float
    kv_ratio_used_p95: float
    kv_ratio_total_p95: float


def parse_meta(path: str) -> Optional[RunMeta]:
    m = FNAME_RE.match(os.path.basename(path))
    if not m:
        return None
    return RunMeta(
        path=path,
        sleep=float(m.group("sleep")),
        pushint=float(m.group("pushint")),
        mb=int(m.group("mb")),
    )


def load_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, engine="python")


def require_cols(df: pd.DataFrame, cols: List[str], fname: str) -> bool:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"  [warn] {fname}: missing columns {missing} (skip some plots/metrics)")
        return False
    return True


def aggregate_run(meta: RunMeta, df: pd.DataFrame) -> RunAgg:
    n = len(df)

    # numeric conversions
    def num(col: str) -> pd.Series:
        return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(dtype=float)

    kv = num("kv")
    run_cnt = num("run_cnt")
    wait_cnt = num("wait_cnt")
    swap_cnt = num("swap_cnt")
    vram_used = num("vram_used_gib")
    vram_total = num("vram_total_gib")

    # ratios (avoid divide by zero)
    kv_ratio_used = (kv / vram_used.replace(0, np.nan)) if ("kv" in df.columns and "vram_used_gib" in df.columns) else pd.Series(dtype=float)
    kv_ratio_total = (kv / vram_total.replace(0, np.nan)) if ("kv" in df.columns and "vram_total_gib" in df.columns) else pd.Series(dtype=float)

    # vram_total is usually constant; take median
    vram_total_med = float(np.nanmedian(vram_total.to_numpy(dtype=float))) if len(vram_total) else float("nan")

    return RunAgg(
        meta=meta,
        reqs=n,
        kv_mean=float(np.nanmean(kv)) if len(kv) else float("nan"),
        kv_p50=q(kv, 0.50) if len(kv) else float("nan"),
        kv_p95=q(kv, 0.95) if len(kv) else float("nan"),
        run_mean=float(np.nanmean(run_cnt)) if len(run_cnt) else float("nan"),
        wait_mean=float(np.nanmean(wait_cnt)) if len(wait_cnt) else float("nan"),
        swap_mean=float(np.nanmean(swap_cnt)) if len(swap_cnt) else float("nan"),
        vram_used_mean=float(np.nanmean(vram_used)) if len(vram_used) else float("nan"),
        vram_used_p95=q(vram_used, 0.95) if len(vram_used) else float("nan"),
        vram_total=vram_total_med,
        kv_ratio_used_mean=float(np.nanmean(kv_ratio_used)) if len(kv_ratio_used) else float("nan"),
        kv_ratio_total_mean=float(np.nanmean(kv_ratio_total)) if len(kv_ratio_total) else float("nan"),
        kv_ratio_used_p95=q(kv_ratio_used, 0.95) if len(kv_ratio_used) else float("nan"),
        kv_ratio_total_p95=q(kv_ratio_total, 0.95) if len(kv_ratio_total) else float("nan"),
    )


def write_summary_csv(aggs: List[RunAgg], outdir: str) -> pd.DataFrame:
    rows = []
    for a in aggs:
        rows.append({
            "sleep": a.meta.sleep,
            "pushint": a.meta.pushint,
            "mb": a.meta.mb,
            "reqs": a.reqs,
            "kv_mean": a.kv_mean,
            "kv_p50": a.kv_p50,
            "kv_p95": a.kv_p95,
            "run_cnt_mean": a.run_mean,
            "wait_cnt_mean": a.wait_mean,
            "swap_cnt_mean": a.swap_mean,
            "vram_used_mean_gib": a.vram_used_mean,
            "vram_used_p95_gib": a.vram_used_p95,
            "vram_total_gib": a.vram_total,
            "kv_ratio_used_mean": a.kv_ratio_used_mean,
            "kv_ratio_used_p95": a.kv_ratio_used_p95,
            "kv_ratio_total_mean": a.kv_ratio_total_mean,
            "kv_ratio_total_p95": a.kv_ratio_total_p95,
            "path": a.meta.path,
        })

    df = pd.DataFrame(rows).sort_values(["sleep", "pushint", "mb"])
    df.to_csv(os.path.join(outdir, "summary_kv_vram.csv"), index=False)
    return df


# ------------------------
# Plot helpers
# ------------------------

def plot_lines(df: pd.DataFrame, x: str, ys: List[Tuple[str, str]], title: str, ylabel: str, outpath: str) -> None:
    plt.figure(figsize=(7.5, 4.2))
    for col, label in ys:
        if col not in df.columns:
            continue
        plt.plot(df[x], df[col], marker="o", label=label)
    plt.xlabel(x)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    savefig(outpath)


def plot_counts_vs_mb(df: pd.DataFrame, outdir: str) -> None:
    plot_lines(
        df,
        x="mb",
        ys=[
            ("run_cnt_mean", "# of running request (mean)"),
            ("wait_cnt_mean", "# of wait request (mean)"),
            ("swap_cnt_mean", "# of swap count (mean)"),
        ],
        title="Running/Waiting/Swapped Request Counts vs mb",
        ylabel="# of requests / swaps (mean over samples)",
        outpath=os.path.join(outdir, "counts_vs_mb.png"),
    )


def plot_kv_vram_vs_mb(df: pd.DataFrame, outdir: str) -> None:
    plot_lines(
        df,
        x="mb",
        ys=[("kv_mean", "kv (mean)"), ("kv_p95", "kv (p95)")],
        title="KV Cache Level vs mb",
        ylabel="KV cache utilization (%)",
        outpath=os.path.join(outdir, "kv_vs_mb.png"),
    )

    plot_lines(
        df,
        x="mb",
        ys=[("vram_used_mean_gib", "vram_used (mean GiB)"), ("vram_used_p95_gib", "vram_used (p95 GiB)")],
        title="VRAM Used vs mb",
        ylabel="VRAM utilization (GiB)",
        outpath=os.path.join(outdir, "vram_used_vs_mb.png"),
    )


def plot_kv_ratio_vs_mb(df: pd.DataFrame, outdir: str) -> None:
    # ratios
    plot_lines(
        df,
        x="mb",
        ys=[("kv_ratio_total_mean", "kv/vram_total (mean)"), ("kv_ratio_total_p95", "kv/vram_total (p95)")],
        title="KV Cache Ratio in Total VRAM vs mb",
        ylabel="ratio",
        outpath=os.path.join(outdir, "kv_ratio_total_vs_mb.png"),
    )

    plot_lines(
        df,
        x="mb",
        ys=[("kv_ratio_used_mean", "kv/vram_used (mean)"), ("kv_ratio_used_p95", "kv/vram_used (p95)")],
        title="KV Cache Ratio in Used VRAM vs mb",
        ylabel="ratio",
        outpath=os.path.join(outdir, "kv_ratio_used_vs_mb.png"),
    )


def plot_kv_ratio_growth(df: pd.DataFrame, outdir: str) -> None:
    """증가율 그래프

    1) baseline 대비 상대 증가율: (ratio - ratio_baseline)/ratio_baseline
    2) 인접 mb 간 step growth: (ratio[i]-ratio[i-1]) / ratio[i-1]
    """
    if df.empty:
        return

    d = df.sort_values("mb").copy()

    for col, name in [
        ("kv_ratio_total_mean", "kv/vram_total (mean)"),
        ("kv_ratio_total_p95", "kv/vram_total (p95)"),
        ("kv_ratio_used_mean", "kv/vram_used (mean)"),
        ("kv_ratio_used_p95", "kv/vram_used (p95)"),
    ]:
        if col not in d.columns:
            continue

        base = d[col].iloc[0]
        baseline_growth = (d[col] - base) / (base if np.isfinite(base) and base != 0 else np.nan)
        step_growth = d[col].pct_change()

        plt.figure(figsize=(7.5, 4.2))
        plt.plot(d["mb"], baseline_growth * 100.0, marker="o", label="vs baseline (%)")
        plt.plot(d["mb"], step_growth * 100.0, marker="o", label="step-to-step (%)")
        plt.xlabel("max batch size")
        plt.ylabel("growth (%)")
        plt.title(f"KV Ratio Growth: {name}")
        plt.legend()
        savefig(os.path.join(outdir, f"kv_ratio_growth_{col}.png"))


def plot_distributions(df_by_mb: Dict[int, pd.DataFrame], outdir: str) -> None:
    # ECDF for kv and swap_cnt
    for metric, title, xlabel in [
        ("kv", "KV Distribution (ECDF)", "KV cache utilization (%)"),
        ("swap_cnt", "swap_cnt Distribution (ECDF)", "# of swap count"),
        ("vram_used_gib", "VRAM Used Distribution (ECDF)", "VRAM utilization (GiB)"),
    ]:
        plt.figure(figsize=(7.5, 5.0))
        any_plotted = False
        for mb in sorted(df_by_mb.keys()):
            df = df_by_mb[mb]
            if metric not in df.columns:
                continue
            x, y = ecdf(pd.to_numeric(df[metric], errors="coerce").to_numpy(dtype=float))
            if len(x) == 0:
                continue
            plt.plot(x, y, label=f"mb{mb}")
            any_plotted = True
        if any_plotted:
            plt.xlabel(xlabel)
            plt.ylabel("ECDF")
            plt.title(title)
            plt.legend(ncol=2, fontsize=9)
            savefig(os.path.join(outdir, f"ecdf_{metric}.png"))
        else:
            plt.close()


def plot_time_series_like(df_by_mb: Dict[int, pd.DataFrame], outdir: str) -> None:
    """finished_time(또는 insertion_time) 기준으로 (kv, run_cnt, swap_cnt, vram_used) 변화를 간이 time-series로 그려줌.

    per-request 샘플이 시간순으로 남아있다는 가정.
    """
    for tcol in ["finished_time", "insertion_time"]:
        ok = any(tcol in df.columns for df in df_by_mb.values())
        if ok:
            time_col = tcol
            break
    else:
        return

    # y-labels requested by user
    metrics = [
        ("kv", "KV cache utilization (%)"),
        ("run_cnt", "# of running request"),
        ("wait_cnt", "# of wait request"),
        ("swap_cnt", "# of swap count"),
        ("vram_used_gib", "VRAM utilization (GiB)"),
    ]

    for metric, label in metrics:
        plt.figure(figsize=(10.5, 4.0))
        plotted = False
        for mb in sorted(df_by_mb.keys()):
            df = df_by_mb[mb]
            if time_col not in df.columns or metric not in df.columns:
                continue
            t = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
            v = pd.to_numeric(df[metric], errors="coerce").to_numpy(dtype=float)
            m = np.isfinite(t) & np.isfinite(v)
            if m.sum() < 10:
                continue
            # normalize time within run
            t0 = np.nanmin(t[m])
            tn = t[m] - t0
            vn = v[m]
            # downsample to reduce clutter
            idx = np.linspace(0, len(tn) - 1, num=min(500, len(tn))).astype(int)
            plt.plot(tn[idx], vn[idx], label=f"mb{mb}", alpha=0.8)
            plotted = True

        if plotted:
            plt.xlabel(f"time (s)")
            plt.ylabel(label)
            plt.title(f"{label} over time (per-request samples)")
            plt.legend(ncol=2, fontsize=9)
            savefig(os.path.join(outdir, f"timeseries_like_{metric}.png"))
        else:
            plt.close()

    # --- per-mb: show run_cnt and swap_cnt together in one figure ---
    # (User request: "각 mb 마다 run_cnt와 swap_cnt를 동시에")
    for mb in sorted(df_by_mb.keys()):
        df = df_by_mb[mb]
        if time_col not in df.columns:
            continue
        if ("run_cnt" not in df.columns) or ("swap_cnt" not in df.columns):
            continue

        t = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
        r = pd.to_numeric(df["run_cnt"], errors="coerce").to_numpy(dtype=float)
        s = pd.to_numeric(df["swap_cnt"], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(t) & np.isfinite(r) & np.isfinite(s)
        if m.sum() < 10:
            continue

        t0 = np.nanmin(t[m])
        tn = t[m] - t0
        rn = r[m]
        sn = s[m]

        # downsample to reduce clutter
        idx = np.linspace(0, len(tn) - 1, num=min(800, len(tn))).astype(int)
        tn2, rn2, sn2 = tn[idx], rn[idx], sn[idx]

        fig, ax1 = plt.subplots(figsize=(10.5, 4.0))
        ax1.plot(tn2, rn2, label="# of running request", alpha=0.9)
        ax1.set_xlabel(f"time (s) (normalized)")
        ax1.set_ylabel("# of running request")
        ax1.grid(True, axis="x", alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(tn2, sn2, label="# of swap count", linestyle="--", alpha=0.9)
        ax2.set_ylabel("# of swap count")

        # combine legends from both axes
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="best")

        ax1.set_title(f"run_cnt and swap_cnt over time (mb{mb})")
        savefig(os.path.join(outdir, f"timeseries_like_run_swap_mb{mb}.png"))
def main(args: argparse.Namespace) -> None:
    ensure_dir(args.outdir)

    paths = sorted(glob.glob(os.path.join(args.logdir, args.pattern)))
    if not paths:
        raise SystemExit(f"No files found: {os.path.join(args.logdir, args.pattern)}")

    metas: List[RunMeta] = []
    for p in paths:
        m = parse_meta(p)
        if m is None:
            print(f"[skip] filename not match meta regex: {p}")
            continue
        metas.append(m)

    if not metas:
        raise SystemExit("No parsable log files for meta (sleep/pushint/mb).")

    grouped: Dict[Tuple[float, float], List[RunMeta]] = {}
    for m in metas:
        grouped.setdefault((m.sleep, m.pushint), []).append(m)

    for (sleep, pushint), group in grouped.items():
        group = sorted(group, key=lambda x: x.mb)
        print(f"\n=== Group sleep={sleep}, pushint={pushint} ({len(group)} files) ===")

        aggs: List[RunAgg] = []
        df_by_mb: Dict[int, pd.DataFrame] = {}

        for meta in group:
            fname = os.path.basename(meta.path)
            print(f"  loading: {fname}")
            df = load_csv(meta.path)

            require_cols(
                df,
                ["kv", "run_cnt", "wait_cnt", "swap_cnt", "vram_used_gib", "vram_total_gib"],
                fname,
            )

            df_by_mb[meta.mb] = df
            aggs.append(aggregate_run(meta, df))

        sub = os.path.join(args.outdir, f"sleep_{sleep}_pushint_{pushint}")
        ensure_dir(sub)

        summary_df = write_summary_csv(aggs, sub)

        # main plots
        plot_counts_vs_mb(summary_df, sub)
        plot_kv_vram_vs_mb(summary_df, sub)
        plot_kv_ratio_vs_mb(summary_df, sub)
        plot_kv_ratio_growth(summary_df, sub)

        # distributions and time-series-like plots
        plot_distributions(df_by_mb, sub)
        plot_time_series_like(df_by_mb, sub)

        print(f"  -> saved figures under: {sub}")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default=".", help="directory containing csv logs")
    parser.add_argument(
        "--outdir",
        type=str,
        default="./maxsize/fig_kv",
        help="output directory for figures (default: ./maxsize/fig_kv)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="timsort_hol_sleep_*_pushint_*_mb*_*.csv",
        help="glob pattern for csv files",
    )

    main(parser.parse_args())
