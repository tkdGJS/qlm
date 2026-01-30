#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_hol_v3_per_file_slotype_chunked.py

기존: per_file_plots/<basename>/... 로 저장
추가: basename(파일명)에 포함된 chunked_{0|1} 기준으로 먼저 분리해서
     per_file_plots/chunked_{c}/<basename>/... 로 저장

입력 패턴(기본):
- timsort_hol_sleep_*_pushint_*_mb*_chunked_*_*.csv
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 예: timsort_hol_sleep_0.01_pushint_0.1_mb80_chunked_1_20260127_203539.csv
FNAME_RE = re.compile(r".*timsort_hol_sleep_([0-9.]+)_pushint_([0-9.]+)_.+\.csv$")

def parse_time_series(s: pd.Series) -> pd.Series:
    if np.issubdtype(s.dtype, np.number):
        return s.astype(float)
    ts = pd.to_datetime(s, errors="coerce", utc=True)
    if ts.notna().any():
        return (ts.view("int64") / 1e9).astype(float)
    return pd.to_numeric(s, errors="coerce").astype(float)

def parse_mb_from_name(fname: str):
    m = re.search(r"_mb(\d+)", fname)
    return int(m.group(1)) if m else None

def parse_chunked_from_name(fname: str):
    m = re.search(r"_chunked_(\d+)", fname)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def classify_long_short(df: pd.DataFrame) -> pd.DataFrame:
    if "out_tok" in df.columns:
        ot = pd.to_numeric(df["out_tok"], errors="coerce")
        thr = ot.median()
        df["is_long"] = ot > thr
    elif "prompt_tok" in df.columns:
        pt = pd.to_numeric(df["prompt_tok"], errors="coerce")
        df["is_long"] = pt > pt.median()
    else:
        df["is_long"] = df["service_time"] > df["service_time"].median()
    return df

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def plot_one(csv_path: str, out_dir: str):
    base = os.path.basename(csv_path)
    m = FNAME_RE.match(base)
    sleep = pushint = None
    if m:
        sleep = m.group(1)
        pushint = m.group(2)

    mb = parse_mb_from_name(base)
    chunked = parse_chunked_from_name(base)

    df = pd.read_csv(csv_path)
    if "ttlt" not in df.columns and "execution_time" in df.columns:
        df = df.rename(columns={"execution_time": "ttlt"})

    need = ["insertion_time", "wait_time", "ttlt"]
    for c in need:
        if c not in df.columns:
            raise KeyError(f"{csv_path}: Missing required column '{c}'. cols={list(df.columns)}")

    df["_t_ins"] = parse_time_series(df["insertion_time"])
    df["queue_delay"] = pd.to_numeric(df["wait_time"], errors="coerce")
    df["service_time"] = pd.to_numeric(df["ttlt"], errors="coerce")

    if "orig_slo" in df.columns:
        df["orig_slo_s"] = pd.to_numeric(df["orig_slo"], errors="coerce")
    else:
        df["orig_slo_s"] = np.nan
    df["sojourn_time"] = df["queue_delay"] + df["service_time"]

    df["_t_dsp"] = df["_t_ins"] + df["queue_delay"]
    if "finished_time" in df.columns:
        df["_t_fin"] = parse_time_series(df["finished_time"])
    else:
        df["_t_fin"] = df["_t_dsp"] + df["service_time"]

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["_t_ins","queue_delay","service_time","sojourn_time","_t_dsp","_t_fin"])
    df = df[(df["queue_delay"] >= 0) & (df["service_time"] >= 0)]
    df = df.sort_values("_t_ins").reset_index(drop=True)

    df = classify_long_short(df)
    short = df[~df["is_long"]].copy()
    long_df = df[df["is_long"]].copy()

    title_suffix = f"(sleep={sleep}, pushint={pushint}, mb={mb}, chunked={chunked})" if (sleep is not None and pushint is not None) else f"({base})"

    # ---- (A) short: queue vs service scatter ----
    plt.figure(figsize=(8,6))
    plt.scatter(short["service_time"], short["queue_delay"], s=12, alpha=0.6)
    plt.xlabel("Service time (ttlt) [s]")
    plt.ylabel("Queueing delay (wait_time) [s]")
    plt.title(f"HOL evidence: SHORT queueing vs service {title_suffix}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "A_queue_vs_service_short.png"), dpi=220)
    plt.close()

    # ---- (B) blocking depth vs short queueing ----
    blocking = []
    longs = long_df[["_t_dsp","_t_fin"]].copy()
    for t in df["_t_ins"].values:
        active = longs[(longs["_t_dsp"] <= t) & (longs["_t_fin"] > t)]
        blocking.append(float((active["_t_fin"] - t).sum()))
    df["blocking_depth"] = blocking
    short = df[~df["is_long"]].copy()

    plt.figure(figsize=(8,6))
    plt.scatter(short["blocking_depth"], short["queue_delay"], s=12, alpha=0.6)
    plt.xlabel("Blocking depth at arrival (sum remaining LONG time) [s]")
    plt.ylabel("Queueing delay of SHORT (wait_time) [s]")
    plt.title(f"HOL mechanism: short queueing vs long in-flight work {title_suffix}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "B_blocking_depth.png"), dpi=220)
    plt.close()

    # ---- (C) timeline-ish (all arrivals) ----
    N = min(200, len(df))
    sub = df.copy()
    t0 = sub["_t_ins"].min()
    sub["_d0"] = sub["_t_dsp"] - t0
    sub["_f0"] = sub["_t_fin"] - t0
    sub["arr_idx"] = np.arange(len(sub))

    plt.figure(figsize=(12,6))
    for _, r in sub.iterrows():
        color = "tab:red" if r["is_long"] else "tab:blue"
        plt.plot([r["_d0"], r["_f0"]], [r["arr_idx"], r["arr_idx"]], color=color, linewidth=2)
    plt.xlabel("Time since first arrival [s]")
    plt.ylabel("Dispatch order index")
    plt.title(f"Timeline (first {N}): red=LONG, blue=SHORT {title_suffix}")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "C_timeline_gantt.png"), dpi=220)
    plt.close()

    # ---- (D) timeline + late-wait after EDF deadline (gray) ----
    if "orig_slo" in df.columns:
        sub = df.copy()
        t0 = sub["_t_ins"].min()
        sub["_d0"] = sub["_t_dsp"] - t0
        sub["_f0"] = sub["_t_fin"] - t0
        sub["_dead0"] = (sub["_t_ins"] + sub["orig_slo_s"]) - t0
        sub["arr_idx"] = np.arange(len(sub))

        plt.figure(figsize=(12,6))
        plt.plot([], [], color="0.7", linewidth=1, label="wait time (after SLO deadline)")
        plt.plot([], [], color="tab:red", linewidth=2, label="Batch")
        plt.plot([], [], color="tab:blue", linewidth=2, label="Interactive")

        for _, r in sub.iterrows():
            y = r["arr_idx"]
            if np.isfinite(r["_dead0"]) and (r["_d0"] > r["_dead0"]):
                plt.plot([r["_dead0"], r["_d0"]], [y, y], color="0.7", linewidth=1, solid_capstyle="butt")
            color = "tab:red" if r["is_long"] else "tab:blue"
            plt.plot([r["_d0"], r["_f0"]], [y, y], color=color, linewidth=2, solid_capstyle="butt")

        plt.xlabel("Time since first arrival (sec)")
        plt.ylabel("Dispatch order index")
        plt.title(f"Timeline (first {N}) {title_suffix}")
        plt.grid(True, axis="x", alpha=0.3)
        plt.legend(loc="upper right", frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "D_timeline_deadline_latewait.png"), dpi=220)
        plt.close()

    # ---- HOL corr 저장 ----
    qv = short["queue_delay"].values
    bv = short["blocking_depth"].values
    hol_corr = np.nan
    if len(short) >= 10 and np.std(qv) > 1e-9 and np.std(bv) > 1e-9:
        hol_corr = float(np.corrcoef(qv, bv)[0,1])

    with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
        f.write(f"file={base}\n")
        if sleep is not None and pushint is not None:
            f.write(f"sleep={sleep}\npushint={pushint}\n")
        if mb is not None:
            f.write(f"mb={mb}\n")
        if chunked is not None:
            f.write(f"chunked={chunked}\n")
        f.write(f"n_total={len(df)}\n")
        f.write(f"n_short={len(short)}\n")
        f.write(f"hol_corr(short_queue_delay, blocking_depth)={hol_corr}\n")

    return hol_corr, mb, chunked

def main():
    csvs = sorted(glob.glob("timsort_hol_sleep_*_pushint_*_mb*_chunked_*_*.csv"))
    if not csvs:
        # fallback
        csvs = sorted(glob.glob("timsort_hol_sleep_*_pushint_*_*.csv"))
    if not csvs:
        print("No matching CSVs found.")
        return

    root_out = "per_file_plots"
    ensure_dir(root_out)

    summary = []
    for csv_path in csvs:
        base = os.path.basename(csv_path).replace(".csv", "")
        chunked = parse_chunked_from_name(base)
        ch_label = f"chunked_{chunked}" if chunked in (0, 1) else "chunked_unknown"

        out_dir = os.path.join(root_out, ch_label, base)
        ensure_dir(out_dir)

        try:
            hol_corr, mb, ch = plot_one(csv_path, out_dir)
            print(f"[OK] {csv_path} -> {out_dir} (hol_corr={hol_corr})")
            summary.append({
                "file": os.path.basename(csv_path),
                "chunked": ch,
                "mb": mb,
                "out_dir": out_dir,
                "hol_corr": hol_corr
            })
        except Exception as e:
            print(f"[FAIL] {csv_path}: {e}")

    if summary:
        pd.DataFrame(summary).to_csv(os.path.join(root_out, "summary.csv"), index=False)
        print(f"Saved summary: {os.path.join(root_out, 'summary.csv')}")

if __name__ == "__main__":
    main()
