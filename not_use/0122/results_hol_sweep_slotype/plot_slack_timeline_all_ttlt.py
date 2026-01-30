#!/usr/bin/env python3
import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FNAME_RE = re.compile(r".*timsort_hol_sleep_([0-9.]+)_pushint_([0-9.]+)_.+\.csv$")

def parse_time_series(s: pd.Series) -> pd.Series:
    """Parse numeric seconds or datetime strings -> float seconds."""
    if np.issubdtype(s.dtype, np.number):
        return s.astype(float)
    ts = pd.to_datetime(s, errors="coerce", utc=True)
    if ts.notna().any():
        return (ts.view("int64") / 1e9).astype(float)
    return pd.to_numeric(s, errors="coerce").astype(float)

def classify_long_short(df: pd.DataFrame) -> pd.DataFrame:
    """Only for coloring."""
    if "out_tok" in df.columns:
        v = pd.to_numeric(df["out_tok"], errors="coerce")
        df["is_long"] = v > v.median()
    elif "prompt_tok" in df.columns:
        v = pd.to_numeric(df["prompt_tok"], errors="coerce")
        df["is_long"] = v > v.median()
    else:
        # fallback: ttlt (or execution_time for backward compatibility)
        col = "ttlt" if "ttlt" in df.columns else ("execution_time" if "execution_time" in df.columns else None)
        if col is None:
            df["is_long"] = False
        else:
            v = pd.to_numeric(df[col], errors="coerce")
            df["is_long"] = v > v.median()
    return df

def plot_one(csv_path: str, out_dir: str, maxN: int = 200):
    base = os.path.basename(csv_path)
    m = FNAME_RE.match(base)
    sleep = pushint = None
    if m:
        sleep, pushint = m.group(1), m.group(2)
    title_suffix = f"(sleep={sleep}, pushint={pushint})" if sleep and pushint else f"({base})"

    df = pd.read_csv(csv_path)

    required_base = ["orig_slo", "insertion_time", "wait_time", "finished_time"]
    for c in required_base:
        if c not in df.columns:
            raise KeyError(f"{csv_path}: missing '{c}'. cols={list(df.columns)}")

    # service-time column: prefer ttlt, fallback to execution_time (legacy)
    service_col = "ttlt" if "ttlt" in df.columns else ("execution_time" if "execution_time" in df.columns else None)
    if service_col is None:
        raise KeyError(f"{csv_path}: missing service-time column (expected 'ttlt' or legacy 'execution_time'). cols={list(df.columns)}")

    # time columns
    df["_t_ins"] = parse_time_series(df["insertion_time"])
    df["_t_fin"] = parse_time_series(df["finished_time"])

    df["orig_slo_sec"] = pd.to_numeric(df["orig_slo"], errors="coerce")
    df["queue_delay"] = pd.to_numeric(df["wait_time"], errors="coerce")
    df["service_time"] = pd.to_numeric(df[service_col], errors="coerce")

    # dispatch time for x-axis segment
    df["_t_dsp"] = df["_t_ins"] + df["queue_delay"]

    # y-axis slack at finish: (ins + slo) - finish
    df["slack_finish"] = (df["_t_ins"] + df["orig_slo_sec"]) - df["_t_fin"]

    # clean
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["_t_ins", "_t_dsp", "_t_fin", "slack_finish", "service_time"])
    df = df[(df["queue_delay"] >= 0) & (df["service_time"] >= 0) & (df["_t_fin"] >= df["_t_dsp"])]

    # sort by insertion (consistent with your other plots)
    df = df.sort_values("_t_ins").reset_index(drop=True)

    if len(df) == 0:
        raise RuntimeError(f"{csv_path}: empty after cleaning")

    df = classify_long_short(df)

    N = min(maxN, len(df))
    #sub = df.iloc[:N].copy()
    sub = df.copy()

    # x-axis: time since first arrival (match F style)
    t0 = sub["_t_ins"].min()
    sub["_d0"] = sub["_t_dsp"] - t0
    sub["_f0"] = sub["_t_fin"] - t0

    y = sub["slack_finish"].to_numpy()

    # robust y limits to show + and - together
    lo = float(np.nanpercentile(y, 2))
    hi = float(np.nanpercentile(y, 98))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(np.nanmin(y)), float(np.nanmax(y))
    pad = 0.08 * (hi - lo + 1e-9)
    lo -= pad
    hi += pad

    # ---- Plot (same style as F: 12x6, horizontal run segments, red/blue) ----
    plt.figure(figsize=(12, 6))
    for _, r in sub.iterrows():
        color = "tab:red" if r["is_long"] else "tab:blue"
        plt.plot([r["_d0"], r["_f0"]],
                 [r["slack_finish"], r["slack_finish"]],
                 color=color, linewidth=2)

    plt.axhline(0.0, color="black", linewidth=1.2, alpha=0.85)
    plt.ylim(lo, hi)
    plt.xlabel("Time since first arrival [s]")
    plt.ylabel("Slack at finish = (insertion + SLO) - finished_time [s]\n(+ meets SLO, − violates)")
    plt.title(f"Slack-at-finish timeline (RUN only): red=LONG, blue=SHORT {title_suffix}")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "K_slack_finish_timeline.png"), dpi=220)
    plt.close()

    # small meta for debugging
    with open(os.path.join(out_dir, "K_meta.txt"), "w") as f:
        f.write(f"file={base}\n")
        if sleep and pushint:
            f.write(f"sleep={sleep}\npushint={pushint}\n")
        f.write(f"service_col={service_col}\n")
        f.write(f"N_total={len(df)}\n")
        f.write(f"N_plotted={N}\n")
        f.write(f"y=(_t_ins+orig_slo)-_t_fin\n")

def main():
    ap = argparse.ArgumentParser(description="Generate slack-finish timeline for ALL HOL CSV files using CSV columns only.")
    ap.add_argument("--glob", default="timsort_hol_sleep_*_pushint_*_*.csv",
                    help="input CSV glob (default: timsort_hol_sleep_*_pushint_*_*.csv)")
    ap.add_argument("--out", default="per_file_plots_slack_finish",
                    help="root output directory")
    ap.add_argument("--maxN", type=int, default=200,
                    help="max rows to plot per file (default: 200)")
    args = ap.parse_args()

    csvs = sorted(glob.glob(args.glob))
    if not csvs:
        print("No CSVs found.")
        return

    os.makedirs(args.out, exist_ok=True)

    ok = fail = 0
    for csv_path in csvs:
        stem = os.path.basename(csv_path).replace(".csv", "")
        out_dir = os.path.join(args.out, stem)
        os.makedirs(out_dir, exist_ok=True)
        try:
            plot_one(csv_path, out_dir, maxN=args.maxN)
            print(f"[OK] {csv_path} -> {out_dir}")
            ok += 1
        except Exception as e:
            print(f"[FAIL] {csv_path}: {e}")
            fail += 1

    print(f"Done. ok={ok}, fail={fail}, out_root={args.out}")

if __name__ == "__main__":
    main()

