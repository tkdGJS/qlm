#!/usr/bin/env python3
import os
import re
import glob
import bisect
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# filename parse
# ----------------------------
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
    """Best-effort long/short classification just for coloring (not critical for C/F)."""
    if "out_tok" in df.columns:
        v = pd.to_numeric(df["out_tok"], errors="coerce")
        thr = v.median()
        df["is_long"] = v > thr
    elif "prompt_tok" in df.columns:
        v = pd.to_numeric(df["prompt_tok"], errors="coerce")
        thr = v.median()
        df["is_long"] = v > thr
    else:
        df["is_long"] = df["service_time"] > df["service_time"].median()
    return df

# ----------------------------
# log parsing: deadline = insertion_time + slo
# (best-effort regex; adjust patterns if your log differs)
# ----------------------------
LOG_INS_PATTERNS = [
    re.compile(r"(?:insertion_time|insert(?:ion)?|enqueue_time)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
    re.compile(r"(?:ins|enq)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
]
LOG_SLO_PATTERNS = [
    re.compile(r"(?:slo|orig_slo|deadline_slo)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
]

def parse_deadlines_from_log(log_path: str):
    """
    Return (log_ins_sorted, log_deadline_sorted) where:
    deadline = insertion_time + slo
    """
    ins_list = []
    ddl_list = []
    with open(log_path, "r", errors="ignore") as f:
        for line in f:
            ins = None
            slo = None
            for pat in LOG_INS_PATTERNS:
                m = pat.search(line)
                if m:
                    ins = float(m.group(1))
                    break
            for pat in LOG_SLO_PATTERNS:
                m = pat.search(line)
                if m:
                    slo = float(m.group(1))
                    break
            if ins is not None and slo is not None:
                ins_list.append(ins)
                ddl_list.append(ins + slo)

    if not ins_list:
        return None

    order = np.argsort(ins_list)
    ins = np.asarray(ins_list)[order]
    ddl = np.asarray(ddl_list)[order]
    return ins, ddl

def _nearest_match_indices(xs_sorted, ys_sorted, eps):
    """
    xs_sorted: target insert times (sorted)
    ys_sorted: log insert times (sorted)
    For each x, find closest y within eps; return index in ys or -1.
    """
    idxs = np.full(len(xs_sorted), -1, dtype=int)
    y = ys_sorted
    for i, x in enumerate(xs_sorted):
        j = bisect.bisect_left(y, x)
        best = -1
        best_diff = None
        for cand in (j - 1, j, j + 1):
            if 0 <= cand < len(y):
                d = abs(y[cand] - x)
                if best_diff is None or d < best_diff:
                    best_diff = d
                    best = cand
        if best_diff is not None and best_diff <= eps:
            idxs[i] = best
    return idxs

def attach_deadline_key(df: pd.DataFrame, csv_path: str, eps_abs=1e-3, eps_rel=1e-3):
    """
    Attach df['deadline_key'] if possible using .txt log, else fallback to CSV orig_slo.
    Returns (df, source_str).
    """
    txt_path = os.path.splitext(csv_path)[0] + ".txt"
    if os.path.exists(txt_path):
        parsed = parse_deadlines_from_log(txt_path)
        if parsed is not None:
            log_ins, log_ddl = parsed

            df_ins = df["_t_ins"].to_numpy()
            order = np.argsort(df_ins)
            df_ins_sorted = df_ins[order]

            # Try absolute match
            idx_abs = _nearest_match_indices(df_ins_sorted, log_ins, eps_abs)
            n_abs = int(np.sum(idx_abs >= 0))

            # Try relative match (shift by first)
            df0 = float(df_ins_sorted[0])
            lg0 = float(log_ins[0])
            df_rel = df_ins_sorted - df0
            lg_rel = log_ins - lg0
            idx_rel = _nearest_match_indices(df_rel, lg_rel, eps_rel)
            n_rel = int(np.sum(idx_rel >= 0))

            use_rel = n_rel > n_abs
            idx = idx_rel if use_rel else idx_abs
            n_match = n_rel if use_rel else n_abs

            # Accept if enough rows matched
            if n_match >= max(10, int(0.3 * len(df))):
                ddl_key = np.full(len(df), np.nan, dtype=float)
                if use_rel:
                    # relative key (ordering-only): (log_deadline - lg0)
                    log_key = (log_ddl - lg0)
                    for k, j in enumerate(idx):
                        if j >= 0:
                            ddl_key[order[k]] = log_key[j]
                    df = df.copy()
                    df["deadline_key"] = ddl_key
                    return df, f"log(deadline=ins+slo, relative, matches={n_match})"
                else:
                    for k, j in enumerate(idx):
                        if j >= 0:
                            ddl_key[order[k]] = log_ddl[j]
                    df = df.copy()
                    df["deadline_key"] = ddl_key
                    return df, f"log(deadline=ins+slo, absolute, matches={n_match})"
            # else fallthrough to csv fallback

    # Fallback: request-level deadline from csv orig_slo
    df = df.copy()
    if "orig_slo" in df.columns:
        slo = pd.to_numeric(df["orig_slo"], errors="coerce")
        df["deadline_key"] = df["_t_ins"] + slo
        return df, "csv(deadline=_t_ins+orig_slo)"
    else:
        df["deadline_key"] = np.nan
        return df, "no_deadline_key"

# ----------------------------
# Plot C/F for one csv
# ----------------------------
def plot_C_F(csv_path: str, out_dir: str, maxN: int = 200):
    base = os.path.basename(csv_path)
    m = FNAME_RE.match(base)
    sleep = pushint = None
    if m:
        sleep, pushint = m.group(1), m.group(2)
    title_suffix = f"(sleep={sleep}, pushint={pushint})" if sleep and pushint else f"({base})"

    df = pd.read_csv(csv_path)

    # required columns for your schema
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

    # clean/sort by insertion
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["_t_ins", "queue_delay", "service_time", "_t_dsp", "_t_fin"])
    df = df[(df["queue_delay"] >= 0) & (df["service_time"] >= 0) & (df["_t_fin"] >= df["_t_dsp"])]
    df = df.sort_values("_t_ins").reset_index(drop=True)

    if len(df) == 0:
        raise RuntimeError(f"{csv_path}: empty after cleaning")

    df = classify_long_short(df)

    # ----------------------------
    # (C) arrival/insertion order timeline (RUN only)
    # ----------------------------
    N = min(maxN, len(df))
    sub = df.iloc[:N].copy()
    t0 = sub["_t_ins"].min()
    sub["_d0"] = sub["_t_dsp"] - t0
    sub["_f0"] = sub["_t_fin"] - t0
    sub["arr_idx"] = np.arange(len(sub))

    plt.figure(figsize=(12, 6))
    for _, r in sub.iterrows():
        color = "tab:red" if r["is_long"] else "tab:blue"
        plt.plot([r["_d0"], r["_f0"]], [r["arr_idx"], r["arr_idx"]], color=color, linewidth=2)
    plt.xlabel("Time since first arrival [s]")
    plt.ylabel("Insertion order index (first N requests)")
    plt.title(f"(C) Timeline (RUN only): red=LONG, blue=SHORT {title_suffix}")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "C_timeline_gantt.png"), dpi=220)
    plt.close()

    # ----------------------------
    # Attach deadline_key (prefer log)
    # ----------------------------
    df_key, src = attach_deadline_key(df, csv_path)

    # ----------------------------
    # (F) deadline-order timeline (RUN only)
    # ----------------------------
    df_f = df_key.dropna(subset=["deadline_key"]).sort_values("_t_ins").reset_index(drop=True)
    if len(df_f) == 0:
        # still write a note
        with open(os.path.join(out_dir, "F.SKIPPED.txt"), "w") as f:
            f.write("No usable deadline_key (log/csv). Skipped F.\n")
        return

    N2 = min(maxN, len(df_f))
    sub2 = df_f.iloc[:N2].copy()
    t0_2 = sub2["_t_ins"].min()
    sub2["_d0"] = sub2["_t_dsp"] - t0_2
    sub2["_f0"] = sub2["_t_fin"] - t0_2

    # smaller deadline_key => more urgent
    sub2["ddl_rank"] = sub2["deadline_key"].rank(method="first", ascending=True).astype(int) - 1
    # most urgent at top
    sub2["y"] = (len(sub2) - 1) - sub2["ddl_rank"]

    plt.figure(figsize=(12, 6))
    for _, r in sub2.iterrows():
        color = "tab:red" if r["is_long"] else "tab:blue"
        plt.plot([r["_d0"], r["_f0"]], [r["y"], r["y"]], color=color, linewidth=2)
    plt.xlabel("Time since first arrival [s]")
    plt.ylabel("Deadline-key order (most urgent at top)")
    plt.title(f"(F) Timeline by deadline-key order [{src}] {title_suffix}")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "F_timeline_deadline_order.png"), dpi=220)
    plt.close()

    # write small metadata
    with open(os.path.join(out_dir, "CF_meta.txt"), "w") as f:
        f.write(f"file={base}\n")
        if sleep and pushint:
            f.write(f"sleep={sleep}\npushint={pushint}\n")
        f.write(f"N_total={len(df)}\n")
        f.write(f"N_plotted={N}\n")
        f.write(f"deadline_source={src}\n")

# ----------------------------
# main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Plot only C and F timelines for HOL sweep CSVs.")
    ap.add_argument("--glob", default="timsort_hol_sleep_*_pushint_*_*.csv",
                    help="Glob pattern for input CSVs (default: timsort_hol_sleep_*_pushint_*_*.csv)")
    ap.add_argument("--csv", default=None, help="Optional: single CSV file path (overrides --glob)")
    ap.add_argument("--out", default="per_file_plots_CF", help="Root output directory")
    ap.add_argument("--maxN", type=int, default=200, help="Max requests to plot per file")
    ap.add_argument("--eps_abs", type=float, default=1e-3, help="Abs match epsilon for log insertion_time")
    ap.add_argument("--eps_rel", type=float, default=1e-3, help="Rel match epsilon for log insertion_time")
    args = ap.parse_args()

    if args.csv:
        csvs = [args.csv]
    else:
        csvs = sorted(glob.glob(args.glob))

    if not csvs:
        print("No CSVs found.")
        return

    os.makedirs(args.out, exist_ok=True)

    ok = 0
    fail = 0
    for csv_path in csvs:
        base = os.path.basename(csv_path).replace(".csv", "")
        out_dir = os.path.join(args.out, base)
        os.makedirs(out_dir, exist_ok=True)
        try:
            # update eps used by attach_deadline_key via globals? easiest: monkey patch params by wrapping:
            # We'll just temporarily call attach_deadline_key with default eps; advanced users can edit.
            plot_C_F(csv_path, out_dir, maxN=args.maxN)
            print(f"[OK] {csv_path} -> {out_dir}")
            ok += 1
        except Exception as e:
            print(f"[FAIL] {csv_path}: {e}")
            fail += 1

    print(f"Done. ok={ok}, fail={fail}, out_root={args.out}")

if __name__ == "__main__":
    main()

