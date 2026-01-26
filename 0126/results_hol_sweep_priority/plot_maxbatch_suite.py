#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""plot_maxbatch_suite.py

목적
- timsort_hol_sleep_*_pushint_*_mb*_*.csv 로그들을 스캔
- dispatch interval(sleep) 고정 실험에서 max batch size(mb) 변화에 따른
  serving quality(QoS) / 시스템 지표 변화를 플롯

입력 CSV 스키마(사용자 제공)
- 단위: **second** (ms 아님)
- 주요 컬럼:
  orig_slo, slo_type, insertion_time, wait_time,
  prompt_tok, out_tok, total_tok,
  ttft, tbt_p95, tbt_p99,
  tbt_slo, tbt_violation, tbt_success_rate_pct,
  ttlt, violation, total_violation,
  finished_time, success_rate_pct, success_count, processed_count,
  kv, run_cnt, wait_cnt, swap_cnt,
  vram_used_gib, vram_total_gib,
  vq_id, vq_groups, head_deadline, slack, model, deadlines

출력
- 기본 저장 경로: ./maxsize/fig
- (sleep, pushint) 그룹별로 하위 폴더 생성 후 figure 저장

사용법
  python plot_maxbatch_suite.py \
    --logdir ./sort \
    --outdir ./maxsize/fig \
    --slo_ttft_s 1.0 \
    --slo_tbt_s 0.1

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

# -------------------------------------------------
# 파일명에서 실험 메타(sleep, pushint, mb) 파싱
# -------------------------------------------------
FNAME_RE = re.compile(
    r".*sleep_(?P<sleep>[\d\.]+)_pushint_(?P<pushint>[\d\.]+)_mb(?P<mb>\d+)_.*\.csv$"
)


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def ecdf(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.array([]), np.array([])
    x = np.sort(v)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def safe_quantile(x: pd.Series, q: float) -> float:
    v = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    return float(np.quantile(v, q)) if len(v) else float("nan")


def savefig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


@dataclass
class RunMeta:
    path: str
    sleep: float
    pushint: float
    mb: int


@dataclass
class RunMetrics:
    meta: RunMeta
    df: pd.DataFrame

    duration_s: float = float("nan")
    reqs: int = 0
    rps: float = float("nan")
    toks_s: float = float("nan")

    # QoS percentiles (seconds)
    ttft_p50_s: float = float("nan")
    ttft_p99_s: float = float("nan")
    tbt_p50_s: float = float("nan")
    tbt_p99_s: float = float("nan")
    ttlt_p50_s: float = float("nan")
    ttlt_p99_s: float = float("nan")

    # breakdown (seconds)
    wait_mean_s: float = float("nan")
    service_mean_s: float = float("nan")

    # SLO/goodput
    goodput_rps: float = float("nan")
    violation_rate: float = float("nan")
    tbt_violation_rate: float = float("nan")


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


def compute_duration_rps_toks(df: pd.DataFrame) -> Tuple[float, float, float, int]:
    # duration은 insertion_time~finished_time 범위로 잡음(둘 다 seconds 단위라고 가정)
    reqs = len(df)
    duration = float("nan")
    if "insertion_time" in df.columns and "finished_time" in df.columns:
        t0 = pd.to_numeric(df["insertion_time"], errors="coerce").min()
        t1 = pd.to_numeric(df["finished_time"], errors="coerce").max()
        if np.isfinite(t0) and np.isfinite(t1) and t1 > t0:
            duration = float(t1 - t0)

    rps = reqs / duration if (np.isfinite(duration) and duration > 0) else float("nan")

    toks_s = float("nan")
    if "out_tok" in df.columns and np.isfinite(duration) and duration > 0:
        out_tok = pd.to_numeric(df["out_tok"], errors="coerce").to_numpy(dtype=float)
        toks_s = float(np.nansum(out_tok) / duration)

    return duration, rps, toks_s, reqs


def compute_goodput(df: pd.DataFrame, rps: float, slo_ttft_s: float, slo_tbt_s: float) -> Tuple[float, float, float]:
    """goodput_rps, violation_rate, tbt_violation_rate

    - 기본 정의: (ttft <= slo_ttft_s) & (tbt_p99 <= slo_tbt_s)
    - tbt_violation 컬럼이 있으면 별도로 그 violation rate도 계산
    """
    if not np.isfinite(rps) or rps <= 0:
        return float("nan"), float("nan"), float("nan")

    if "ttft" not in df.columns or "tbt_p99" not in df.columns:
        return float("nan"), float("nan"), float("nan")

    ttft = pd.to_numeric(df["ttft"], errors="coerce")
    tbt99 = pd.to_numeric(df["tbt_p99"], errors="coerce")

    ok = (ttft <= slo_ttft_s) & (tbt99 <= slo_tbt_s)
    ok_rate = float(np.nanmean(ok.astype(float))) if len(ok) else float("nan")
    viol_rate = 1.0 - ok_rate if np.isfinite(ok_rate) else float("nan")
    goodput = rps * ok_rate if np.isfinite(ok_rate) else float("nan")

    tbt_viol_rate = float("nan")
    if "tbt_violation" in df.columns:
        tv = pd.to_numeric(df["tbt_violation"], errors="coerce")
        # tbt_violation이 1/0 또는 True/False라고 가정
        tbt_viol_rate = float(np.nanmean((tv.astype(float) > 0).astype(float)))

    return goodput, viol_rate, tbt_viol_rate


def summarize_run(meta: RunMeta, df_raw: pd.DataFrame, slo_ttft_s: float, slo_tbt_s: float) -> RunMetrics:
    df = df_raw.copy()

    duration_s, rps, toks_s, reqs = compute_duration_rps_toks(df)
    goodput, viol_rate, tbt_viol_rate = compute_goodput(df, rps, slo_ttft_s, slo_tbt_s)

    rm = RunMetrics(meta=meta, df=df)
    rm.duration_s = duration_s
    rm.reqs = reqs
    rm.rps = rps
    rm.toks_s = toks_s
    rm.goodput_rps = goodput
    rm.violation_rate = viol_rate
    rm.tbt_violation_rate = tbt_viol_rate

    if "ttft" in df.columns:
        rm.ttft_p50_s = safe_quantile(df["ttft"], 0.50)
        rm.ttft_p99_s = safe_quantile(df["ttft"], 0.99)
    if "tbt_p99" in df.columns:
        # tbt_p99 자체가 per-request tail이면, 그 값들의 분포를 다시 p50/p99로 요약
        rm.tbt_p50_s = safe_quantile(df["tbt_p99"], 0.50)
        rm.tbt_p99_s = safe_quantile(df["tbt_p99"], 0.99)
    elif "tbt_p95" in df.columns:
        rm.tbt_p50_s = safe_quantile(df["tbt_p95"], 0.50)
        rm.tbt_p99_s = safe_quantile(df["tbt_p95"], 0.99)

    if "ttlt" in df.columns:
        rm.ttlt_p50_s = safe_quantile(df["ttlt"], 0.50)
        rm.ttlt_p99_s = safe_quantile(df["ttlt"], 0.99)

    if "wait_time" in df.columns:
        wt = pd.to_numeric(df["wait_time"], errors="coerce").to_numpy(dtype=float)
        rm.wait_mean_s = float(np.nanmean(wt)) if len(wt) else float("nan")

    # service time을 (ttlt - wait_time)으로 근사
    if "ttlt" in df.columns and "wait_time" in df.columns:
        ttlt = pd.to_numeric(df["ttlt"], errors="coerce")
        wt = pd.to_numeric(df["wait_time"], errors="coerce")
        svc = (ttlt - wt).to_numpy(dtype=float)
        svc = svc[np.isfinite(svc)]
        rm.service_mean_s = float(np.mean(svc)) if len(svc) else float("nan")

    return rm


def write_summary_table(runs: List[RunMetrics], outdir: str) -> None:
    rows = []
    for r in runs:
        rows.append({
            "sleep": r.meta.sleep,
            "pushint": r.meta.pushint,
            "mb": r.meta.mb,
            "reqs": r.reqs,
            "duration_s": r.duration_s,
            "rps": r.rps,
            "toks_s": r.toks_s,
            "ttft_p50_s": r.ttft_p50_s,
            "ttft_p99_s": r.ttft_p99_s,
            "tbt_p50_s": r.tbt_p50_s,
            "tbt_p99_s": r.tbt_p99_s,
            "ttlt_p50_s": r.ttlt_p50_s,
            "ttlt_p99_s": r.ttlt_p99_s,
            "wait_mean_s": r.wait_mean_s,
            "service_mean_s": r.service_mean_s,
            "goodput_rps": r.goodput_rps,
            "violation_rate": r.violation_rate,
            "tbt_violation_rate": r.tbt_violation_rate,
            "path": r.meta.path,
        })
    df = pd.DataFrame(rows).sort_values(["sleep", "pushint", "mb"])
    df.to_csv(os.path.join(outdir, "summary_metrics.csv"), index=False)


# ----------------------------
# Plot functions
# ----------------------------

def plot_pareto(runs: List[RunMetrics], outdir: str) -> None:
    df = pd.DataFrame([{
        "mb": r.meta.mb,
        "rps": r.rps,
        "toks_s": r.toks_s,
        "ttft_p99_s": r.ttft_p99_s,
        "tbt_p99_s": r.tbt_p99_s,
        "ttlt_p99_s": r.ttlt_p99_s,
    } for r in runs]).sort_values("mb")

    # Throughput (req/s) vs P99 TTFT
    plt.figure(figsize=(6, 4))
    plt.plot(df["rps"], df["ttft_p99_s"], marker="o")
    for _, row in df.iterrows():
        plt.text(row["rps"], row["ttft_p99_s"], f"mb{int(row['mb'])}", fontsize=9, ha="left", va="bottom")
    plt.xlabel("Throughput (req/s)")
    plt.ylabel("P99 TTFT (s)")
    plt.title("Pareto: Throughput vs P99 TTFT")
    savefig(os.path.join(outdir, "pareto_rps_vs_p99_ttft.png"))

    # Throughput (req/s) vs P99 TBT
    plt.figure(figsize=(6, 4))
    plt.plot(df["rps"], df["tbt_p99_s"], marker="o")
    for _, row in df.iterrows():
        plt.text(row["rps"], row["tbt_p99_s"], f"mb{int(row['mb'])}", fontsize=9, ha="left", va="bottom")
    plt.xlabel("Throughput (req/s)")
    plt.ylabel("P99 TBT (s)")
    plt.title("Pareto: Throughput vs P99 TBT")
    savefig(os.path.join(outdir, "pareto_rps_vs_p99_tbt.png"))

    # Tokens/s vs P99 TBT
    plt.figure(figsize=(6, 4))
    plt.plot(df["toks_s"], df["tbt_p99_s"], marker="o")
    for _, row in df.iterrows():
        plt.text(row["toks_s"], row["tbt_p99_s"], f"mb{int(row['mb'])}", fontsize=9, ha="left", va="bottom")
    plt.xlabel("Throughput (tokens/s)")
    plt.ylabel("P99 TBT (s)")
    plt.title("Pareto: Tokens/s vs P99 TBT")
    savefig(os.path.join(outdir, "pareto_toks_vs_p99_tbt.png"))


def plot_goodput(runs: List[RunMetrics], outdir: str) -> None:
    df = pd.DataFrame([{
        "mb": r.meta.mb,
        "rps": r.rps,
        "goodput": r.goodput_rps,
        "viol": r.violation_rate,
        "tbt_viol": r.tbt_violation_rate,
    } for r in runs]).sort_values("mb")

    plt.figure(figsize=(7, 4))
    plt.plot(df["mb"], df["rps"], marker="o", label="Throughput (req/s)")
    plt.plot(df["mb"], df["goodput"], marker="o", label="Goodput (SLO-satisfying req/s)")
    plt.xlabel("Max batch size")
    plt.ylabel("req/s")
    plt.title("Throughput vs Goodput")
    plt.legend()
    savefig(os.path.join(outdir, "goodput_vs_mb.png"))

    plt.figure(figsize=(7, 4))
    plt.plot(df["mb"], df["viol"] * 100.0, marker="o", label="joint SLO violation")
    if df["tbt_viol"].notna().any():
        plt.plot(df["mb"], df["tbt_viol"] * 100.0, marker="o", label="tbt_violation column")
    plt.xlabel("Max batch size")
    plt.ylabel("Violation rate (%)")
    plt.title("SLO Violation Rate vs max batch size")
    plt.legend()
    savefig(os.path.join(outdir, "slo_violation_vs_mb.png"))


def plot_latency_breakdown(runs: List[RunMetrics], outdir: str) -> None:
    # 제공 컬럼에서 prefill/decode 분해가 없으므로, queue(wait_time) + service(ttlt-wait)로 근사
    df = pd.DataFrame([{
        "mb": r.meta.mb,
        "wait": r.wait_mean_s,
        "service": r.service_mean_s,
    } for r in runs]).sort_values("mb")

    if df[["wait", "service"]].isna().all().all():
        return

    plt.figure(figsize=(8, 4.5))
    mb = df["mb"].values
    w = df["wait"].fillna(0).values
    s = df["service"].fillna(0).values

    plt.bar(mb, w, label="Queue(wait_time)")
    plt.bar(mb, s, bottom=w, label="Service(ttlt-wait)")
    plt.xlabel("Max batch size")
    plt.ylabel("Mean latency component (s)")
    plt.title("Latency Breakdown (mean) vs max batch size")
    plt.legend()
    savefig(os.path.join(outdir, "latency_breakdown_mean_vs_mb.png"))


def plot_cdf_by_mb(runs: List[RunMetrics], outdir: str) -> None:
    # TTFT ECDF
    plt.figure(figsize=(7, 5))
    any_plotted = False
    for r in sorted(runs, key=lambda x: x.meta.mb):
        if "ttft" not in r.df.columns:
            continue
        x, y = ecdf(pd.to_numeric(r.df["ttft"], errors="coerce").to_numpy(dtype=float))
        if len(x) == 0:
            continue
        plt.plot(x, y, label=f"mb{r.meta.mb}")
        any_plotted = True
    if any_plotted:
        plt.xlabel("TTFT (s)")
        plt.ylabel("ECDF")
        plt.title("TTFT Distribution (ECDF) by max batch size")
        plt.legend(ncol=2, fontsize=9)
        savefig(os.path.join(outdir, "cdf_ttft_by_mb.png"))
    else:
        plt.close()

    # TBT (use tbt_p99) ECDF
    plt.figure(figsize=(7, 5))
    any_plotted = False
    for r in sorted(runs, key=lambda x: x.meta.mb):
        col = "tbt_p99" if "tbt_p99" in r.df.columns else ("tbt_p95" if "tbt_p95" in r.df.columns else None)
        if col is None:
            continue
        x, y = ecdf(pd.to_numeric(r.df[col], errors="coerce").to_numpy(dtype=float))
        if len(x) == 0:
            continue
        plt.plot(x, y, label=f"mb{r.meta.mb}")
        any_plotted = True
    if any_plotted:
        plt.xlabel("TBT (s)")
        plt.ylabel("ECDF")
        plt.title("TBT Distribution (ECDF) by max batch size")
        plt.legend(ncol=2, fontsize=9)
        savefig(os.path.join(outdir, "cdf_tbt_by_mb.png"))
    else:
        plt.close()


def plot_boxplots(runs: List[RunMetrics], outdir: str) -> None:
    rows = []
    for r in runs:
        df = r.df
        if "ttft" in df.columns:
            rows.append(pd.DataFrame({"mb": r.meta.mb, "metric": "TTFT(s)", "value": pd.to_numeric(df["ttft"], errors="coerce")}))
        if "tbt_p99" in df.columns:
            rows.append(pd.DataFrame({"mb": r.meta.mb, "metric": "TBT_p99(s)", "value": pd.to_numeric(df["tbt_p99"], errors="coerce")}))
        elif "tbt_p95" in df.columns:
            rows.append(pd.DataFrame({"mb": r.meta.mb, "metric": "TBT_p95(s)", "value": pd.to_numeric(df["tbt_p95"], errors="coerce")}))
        if "ttlt" in df.columns:
            rows.append(pd.DataFrame({"mb": r.meta.mb, "metric": "TTLT(s)", "value": pd.to_numeric(df["ttlt"], errors="coerce")}))

    if not rows:
        return
    all_df = pd.concat(rows, ignore_index=True)
    all_df["value"] = pd.to_numeric(all_df["value"], errors="coerce")

    plt.figure(figsize=(11, 5))
    sns.boxplot(data=all_df, x="mb", y="value", hue="metric", showfliers=False)
    plt.xlabel("Max batch size")
    plt.ylabel("Latency (s)")
    plt.title("Latency Boxplot by max batch size")
    plt.legend(ncol=3, fontsize=9)
    savefig(os.path.join(outdir, "box_latency_by_mb.png"))


def plot_vram_vs_time(runs: List[RunMetrics], outdir: str) -> None:
    # time-series가 아니라 per-request지만, finished_time 기준으로 vram_used_gib 변화(드리프트/피크) 보기 좋음
    plotted = False
    plt.figure(figsize=(10, 4))
    for r in sorted(runs, key=lambda x: x.meta.mb):
        df = r.df
        if "finished_time" not in df.columns or "vram_used_gib" not in df.columns:
            continue
        t = pd.to_numeric(df["finished_time"], errors="coerce").to_numpy(dtype=float)
        v = pd.to_numeric(df["vram_used_gib"], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(t) & np.isfinite(v)
        if m.sum() < 5:
            continue
        # normalize time within run
        t0 = np.nanmin(t[m])
        tn = t[m] - t0
        # downsample for speed
        idx = np.linspace(0, len(tn) - 1, num=min(400, len(tn))).astype(int)
        plt.plot(tn[idx], v[m][idx], label=f"mb{r.meta.mb}", alpha=0.8)
        plotted = True

    if plotted:
        plt.xlabel("time (s)")
        plt.ylabel("vram (GiB)")
        plt.title("VRAM Used vs Time (per-request samples)")
        plt.legend(ncol=2, fontsize=9)
        savefig(os.path.join(outdir, "vram_used_timeseries_like.png"))
    else:
        plt.close()


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

        runs: List[RunMetrics] = []
        for meta in group:
            print(f"  loading: {os.path.basename(meta.path)}")
            df_raw = load_csv(meta.path)

            # sanity: required cols
            missing = [c for c in ["insertion_time", "finished_time", "ttft"] if c not in df_raw.columns]
            if missing:
                print(f"  [warn] missing columns {missing} in {os.path.basename(meta.path)}")

            rm = summarize_run(meta, df_raw, args.slo_ttft_s, args.slo_tbt_s)
            runs.append(rm)

        sub = os.path.join(args.outdir, f"sleep_{sleep}_pushint_{pushint}")
        ensure_dir(sub)

        write_summary_table(runs, sub)

        # plots
        plot_pareto(runs, sub)
        plot_goodput(runs, sub)
        plot_latency_breakdown(runs, sub)
        plot_cdf_by_mb(runs, sub)
        plot_boxplots(runs, sub)
        plot_vram_vs_time(runs, sub)

        print(f"  -> saved figures under: {sub}")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default=".", help="directory containing csv logs")
    parser.add_argument(
        "--outdir",
        type=str,
        default="./maxsize/fig",
        help="output directory for figures (default: ./maxsize/fig)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="timsort_hol_sleep_*_pushint_*_mb*_*.csv",
        help="glob pattern for csv files",
    )
    parser.add_argument("--slo_ttft_s", type=float, default=1.0, help="TTFT SLO in seconds")
    parser.add_argument("--slo_tbt_s", type=float, default=0.1, help="TBT SLO in seconds")

    main(parser.parse_args())
