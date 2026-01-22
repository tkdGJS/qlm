#!/usr/bin/env python3
import os, re, glob, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# filename meta (optional)
# ----------------------------
FNAME_RE = re.compile(r".*timsort_hol_sleep_([0-9.]+)_pushint_([0-9.]+)_.+\.csv$")

def parse_time_series(s: pd.Series) -> pd.Series:
    if np.issubdtype(s.dtype, np.number):
        return s.astype(float)
    ts = pd.to_datetime(s, errors="coerce", utc=True)
    if ts.notna().any():
        return (ts.view("int64") / 1e9).astype(float)
    return pd.to_numeric(s, errors="coerce").astype(float)

def classify_long_short(df: pd.DataFrame) -> pd.DataFrame:
    # color only
    if "out_tok" in df.columns:
        v = pd.to_numeric(df["out_tok"], errors="coerce")
        df["is_long"] = v > v.median()
    elif "prompt_tok" in df.columns:
        v = pd.to_numeric(df["prompt_tok"], errors="coerce")
        df["is_long"] = v > v.median()
    else:
        df["is_long"] = df["service_time"] > df["service_time"].median()
    return df

# ----------------------------
# Log parsing: need ins, slo, finished_time
# Adjust these regex to match your log format if needed.
# ----------------------------
PAT_INS = re.compile(r"(?:insertion_time|enqueue_time|insert(?:ion)?|enq)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
PAT_SLO = re.compile(r"(?:orig_slo|slo|deadline_slo)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
PAT_FIN = re.compile(r"(?:finished_time|finish_time|finished|done_time|complete_time)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)

def parse_log_slack(log_path: str):
    """
    Return arrays sorted by log insertion:
      log_ins, slack_finish=(log_ins+log_slo)-log_finished
    """
    ins_list, slack_list = [], []
    with open(log_path, "r", errors="ignore") as f:
        for line in f:
            m1 = PAT_INS.search(line)
            m2 = PAT_SLO.search(line)
            m3 = PAT_FIN.search(line)
            if not (m1 and m2 and m3):
                continue
            ins = float(m1.group(1))
            slo = float(m2.group(1))
            fin = float(m3.group(1))
            ins_list.append(ins)
            slack_list.append((ins + slo) - fin)

    if not ins_list:
        return None

    order = np.argsort(ins_list)
    log_ins = np.asarray(ins_list)[order]
    slack = np.asarray(slack_list)[order]
    return log_ins, slack

def align_shift(csv_ins_sorted: np.ndarray, log_ins_sorted: np.ndarray):
    """
    Detect if log is relative and csv is absolute epoch-ish. If so, shift log by (csv0-log0).
    """
    med_csv = float(np.nanmedian(csv_ins_sorted))
    med_log = float(np.nanmedian(log_ins_sorted))
    if med_csv > 1e8 and med_log < 1e8:
        shift = float(csv_ins_sorted[0] - log_ins_sorted[0])
        return log_ins_sorted + shift, shift, "shift(log relative -> csv absolute)"
    return log_ins_sorted, 0.0, "no_shift"

def map_slack_to_csv(df: pd.DataFrame, csv_path: str, eps=0.02):
    """
    Attach df['slack_finish'] by matching insertion_time with log insertion_time.
    Uses two-pointer monotone matching after optional shift.
    """
    txt_path = os.path.splitext(csv_path)[0] + ".txt"
    if not os.path.exists(txt_path):
        return df, False, "no_log_txt"

    parsed = parse_log_slack(txt_path)
    if parsed is None:
        return df, False, "log_parse_no_(ins,slo,fin)"

    log_ins, log_slack = parsed

    df_ins = df["_t_ins"].to_numpy()
    order = np.argsort(df_ins)
    df_ins_sorted = df_ins[order]

    log_ins_adj, shift, align_src = align_shift(df_ins_sorted, log_ins)
    # slack is delta time; shift doesn't change slack itself

    out_slack_sorted = np.full(len(df), np.nan, dtype=float)

    i = j = 0
    matched = 0
    n = len(df_ins_sorted)
    m = len(log_ins_adj)

    while i < n and j < m:
        a = df_ins_sorted[i]
        b = log_ins_adj[j]
        if abs(a - b) <= eps:
            out_slack_sorted[i] = log_slack[j]
            matched += 1
            i += 1
            j += 1
        elif a < b:
            i += 1
        else:
            j += 1

    out = df.copy()
    out.loc[order, "slack_finish"] = out_slack_sorted
    frac = matched / max(1, n)

    ok = frac >= 0.3  # 너무 낮으면 파싱/매칭 실패로 보고 스킵
    src = f"log(slack=(ins+slo)-fin) + {align_src} + match(eps={eps}, matched={matched}/{n})"
    return out, ok, src

def plot_F_slack_timeline(csv_path: str, out_dir: str, maxN=200, eps=0.02):
    base = os.path.basename(csv_path)
    m = FNAME_RE.match(base)
    sleep = pushint = None
    if m:
        sleep, pushint = m.group(1), m.group(2)
    title_suffix = f"(sleep={sleep}, pushint={pushint})" if sleep and pushint else f"({base})"

    df = pd.read_csv(csv_path)
    for c in ["insertion_time", "wait_time", "execution_time"]:
        if c not in df.columns:
            raise KeyError(f"{csv_path}: missing {c}. cols={list(df.columns)}")

    df["_t_ins"] = parse_time_series(df["insertion_time"])
    df["queue_delay"] = pd.to_numeric(df["wait_time"], errors="coerce")
    df["service_time"] = pd.to_numeric(df["execution_time"], errors="coerce")
    df["_t_dsp"] = df["_t_ins"] + df["queue_delay"]

    if "finished_time" in df.columns:
        df["_t_fin"] = parse_time_series(df["finished_time"])
    else:
        df["_t_fin"] = df["_t_dsp"] + df["service_time"]

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["_t_ins", "queue_delay", "service_time", "_t_dsp", "_t_fin"])
    df = df[(df["queue_delay"] >= 0) & (df["service_time"] >= 0) & (df["_t_fin"] >= df["_t_dsp"])]
    df = df.sort_values("_t_ins").reset_index(drop=True)
    if len(df) == 0:
        raise RuntimeError("empty after cleaning")

    df = classify_long_short(df)

    # attach slack from log
    df2, ok, src = map_slack_to_csv(df, csv_path, eps=eps)
    if not ok or df2["slack_finish"].notna().sum() < 10:
        with open(os.path.join(out_dir, "F_slack.SKIPPED.txt"), "w") as f:
            f.write(f"Skipped: {src}\n")
        return

    N = min(maxN, len(df2))
    sub = df2.iloc[:N].copy()

    # x-axis same style as F: time since first arrival
    t0 = sub["_t_ins"].min()
    sub["_d0"] = sub["_t_dsp"] - t0
    sub["_f0"] = sub["_t_fin"] - t0

    # y-axis: slack_finish can be +/-
    y = sub["slack_finish"].to_numpy()

    # choose ylim to show + and - nicely (robust)
    lo = float(np.nanpercentile(y, 2))
    hi = float(np.nanpercentile(y, 98))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(np.nanmin(y)), float(np.nanmax(y))
    pad = 0.08 * (hi - lo + 1e-9)
    lo -= pad
    hi += pad

    # ---- Plot: same look/size as F ----
    plt.figure(figsize=(12, 6))
    for _, r in sub.iterrows():
        color = "tab:red" if r["is_long"] else "tab:blue"
        plt.plot([r["_d0"], r["_f0"]], [r["slack_finish"], r["slack_finish"]],
                 color=color, linewidth=2)

    plt.axhline(0.0, color="black", linewidth=1.2, alpha=0.8)  # SLO boundary
    plt.ylim(lo, hi)
    plt.xlabel("Time since first arrival [s]")
    plt.ylabel("Slack at finish = (insertion+slo) - finished_time [s]\n(+ good, − violated)")
    plt.title(f"(F-s) Timeline by slack-at-finish (from log) [{src}] {title_suffix}")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "F_slack_finish_timeline.png"), dpi=220)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="timsort_hol_sleep_*_pushint_*_*.csv")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--out", default="per_file_plots_F_slack")
    ap.add_argument("--maxN", type=int, default=200)
    ap.add_argument("--eps", type=float, default=0.02, help="matching epsilon (seconds) for insertion_time")
    args = ap.parse_args()

    if args.csv:
        csvs = [args.csv]
    else:
        csvs = sorted(glob.glob(args.glob))

    if not csvs:
        print("No CSV found.")
        return

    os.makedirs(args.out, exist_ok=True)

    ok = fail = 0
    for p in csvs:
        out_dir = os.path.join(args.out, os.path.basename(p).replace(".csv", ""))
        os.makedirs(out_dir, exist_ok=True)
        try:
            plot_F_slack_timeline(p, out_dir, maxN=args.maxN, eps=args.eps)
            print(f"[OK] {p} -> {out_dir}")
            ok += 1
        except Exception as e:
            print(f"[FAIL] {p}: {e}")
            fail += 1

    print(f"Done. ok={ok}, fail={fail}, out_root={args.out}")

if __name__ == "__main__":
    main()

