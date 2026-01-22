#!/usr/bin/env python3
import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Filename meta (optional for title)
# ----------------------------
FNAME_RE = re.compile(r".*timsort_hol_sleep_([0-9.]+)_pushint_([0-9.]+)_.+\.csv$")

# ----------------------------
# Time parsing
# ----------------------------
def parse_time_series(s: pd.Series) -> pd.Series:
    """Parse numeric seconds or datetime strings -> float seconds."""
    if np.issubdtype(s.dtype, np.number):
        return s.astype(float)
    ts = pd.to_datetime(s, errors="coerce", utc=True)
    if ts.notna().any():
        return (ts.view("int64") / 1e9).astype(float)
    return pd.to_numeric(s, errors="coerce").astype(float)

def classify_long_short(df: pd.DataFrame) -> pd.DataFrame:
    """Only for coloring. Not used for correctness of C/F axes."""
    if "out_tok" in df.columns:
        v = pd.to_numeric(df["out_tok"], errors="coerce")
        thr = v.median()
        df["is_long"] = v > thr
    elif "prompt_tok" in df.columns:
        v = pd.to_numeric(df["prompt_tok"], errors="coerce")
        thr = v.median()
        df["is_long"] = v > thr
    else:
        thr = df["service_time"].median()
        df["is_long"] = df["service_time"] > thr
    return df

# ----------------------------
# Log parsing (best-effort)
# We try to parse at least: insertion_time + slo
# Optionally: group_id + is_head to build group-head deadline key (group EDF)
# ----------------------------
PAT_INS = re.compile(r"(?:insertion_time|enqueue_time|insert(?:ion)?|enq)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
PAT_SLO = re.compile(r"(?:orig_slo|slo|deadline_slo)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
PAT_GID = re.compile(r"(?:group_id|gid)\s*[:=]\s*([0-9]+)", re.IGNORECASE)
PAT_HEAD = re.compile(r"(?:is_head|head)\s*[:=]\s*(true|false|0|1)", re.IGNORECASE)

def parse_log_events(log_path: str) -> pd.DataFrame:
    """
    Return DataFrame with columns:
      log_ins, slo, log_deadline (=log_ins+slo), group_id(optional), is_head(optional)
    """
    rows = []
    with open(log_path, "r", errors="ignore") as f:
        for line in f:
            m_ins = PAT_INS.search(line)
            m_slo = PAT_SLO.search(line)
            if not (m_ins and m_slo):
                continue
            ins = float(m_ins.group(1))
            slo = float(m_slo.group(1))

            gid = None
            m_gid = PAT_GID.search(line)
            if m_gid:
                gid = int(m_gid.group(1))

            is_head = None
            m_h = PAT_HEAD.search(line)
            if m_h:
                v = m_h.group(1).lower()
                is_head = (v == "true" or v == "1")

            rows.append({
                "log_ins": ins,
                "slo": slo,
                "log_deadline": ins + slo,
                "group_id": gid,
                "is_head": is_head
            })

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df

def build_deadline_key_from_log(log_df: pd.DataFrame) -> (np.ndarray, np.ndarray, str):
    """
    Return (log_ins_sorted, deadline_key_sorted, source_str)
    deadline_key is:
      - if group_id present: group-head deadline propagated to group members (group EDF key)
      - else: request deadline (ins+slo)
    """
    if log_df.empty:
        return None, None, "no_log"

    # If group_id exists for many rows, attempt group-head EDF
    gid_valid = log_df["group_id"].notna().sum()
    if gid_valid >= max(10, int(0.3 * len(log_df))):
        # Determine head deadline per group
        df = log_df.copy()
        # If is_head is missing, fallback head = earliest log_ins in group
        if df["is_head"].notna().any():
            heads = df[df["is_head"] == True]
            # some groups might lack explicit head marker
            # we’ll fill missing by earliest ins in that group
            head_deadline = {}
            # explicit heads first
            for gid, g in heads.groupby("group_id"):
                # if multiple, take earliest deadline
                head_deadline[int(gid)] = float(g["log_deadline"].min())
            # fill missing
            for gid, g in df.groupby("group_id"):
                gid = int(gid)
                if gid not in head_deadline:
                    # earliest ins as head proxy
                    idx = g["log_ins"].idxmin()
                    head_deadline[gid] = float(df.loc[idx, "log_deadline"])
        else:
            head_deadline = {}
            for gid, g in df.groupby("group_id"):
                gid = int(gid)
                idx = g["log_ins"].idxmin()
                head_deadline[gid] = float(df.loc[idx, "log_deadline"])

        # deadline key for each row = group's head deadline
        def map_key(row):
            gid = row["group_id"]
            if pd.isna(gid):
                return row["log_deadline"]
            return head_deadline.get(int(gid), row["log_deadline"])

        df["deadline_key"] = df.apply(map_key, axis=1)
        df = df.sort_values("log_ins").reset_index(drop=True)
        return df["log_ins"].to_numpy(), df["deadline_key"].to_numpy(), "log(group-head deadline key)"
    else:
        # request-level deadline key
        df = log_df.sort_values("log_ins").reset_index(drop=True)
        return df["log_ins"].to_numpy(), df["log_deadline"].to_numpy(), "log(request deadline=ins+slo)"

# ----------------------------
# Matching log insertion times to CSV insertion times robustly
# ----------------------------
def align_log_to_csv_times(df_ins_sorted: np.ndarray, log_ins_sorted: np.ndarray):
    """
    Detect if log time is relative while csv is absolute epoch, and compute shift.
    Return adjusted log_ins.
    """
    if len(df_ins_sorted) == 0 or len(log_ins_sorted) == 0:
        return log_ins_sorted, 0.0, "no_align"

    med_csv = float(np.nanmedian(df_ins_sorted))
    med_log = float(np.nanmedian(log_ins_sorted))

    # Heuristic: epoch seconds ~ 1e9; relative seconds typically < 1e7
    if med_csv > 1e8 and med_log < 1e8:
        shift = float(df_ins_sorted[0] - log_ins_sorted[0])
        return log_ins_sorted + shift, shift, "shift(log relative -> csv absolute)"
    # Otherwise no shift
    return log_ins_sorted, 0.0, "no_shift"

def map_deadline_key_to_csv(df: pd.DataFrame, csv_path: str,
                            eps: float = 5e-3) -> (pd.DataFrame, str):
    """
    Attach df['deadline_key'] by matching insertion_time against log insertion_time.
    Prefer rank-to-rank if lengths match; fallback to two-pointer nearest match within eps.
    """
    txt_path = os.path.splitext(csv_path)[0] + ".txt"
    if not os.path.exists(txt_path):
        # fallback: csv deadline = _t_ins + orig_slo
        if "orig_slo" in df.columns:
            slo = pd.to_numeric(df["orig_slo"], errors="coerce")
            out = df.copy()
            out["deadline_key"] = out["_t_ins"] + slo
            return out, "csv(deadline=_t_ins+orig_slo)"
        out = df.copy()
        out["deadline_key"] = np.nan
        return out, "no_deadline_key"

    log_df = parse_log_events(txt_path)
    log_ins, log_key, src = build_deadline_key_from_log(log_df)
    if log_ins is None:
        if "orig_slo" in df.columns:
            slo = pd.to_numeric(df["orig_slo"], errors="coerce")
            out = df.copy()
            out["deadline_key"] = out["_t_ins"] + slo
            return out, "csv(deadline=_t_ins+orig_slo) (log empty)"
        out = df.copy()
        out["deadline_key"] = np.nan
        return out, "no_deadline_key (log empty)"

    # Prepare sorted CSV insertion times with index mapping
    df_ins = df["_t_ins"].to_numpy()
    order = np.argsort(df_ins)
    df_ins_sorted = df_ins[order]

    # Align log timebase if needed
    log_ins_adj, shift, align_src = align_log_to_csv_times(df_ins_sorted, log_ins)

    # Adjust key with the same shift if key is absolute time based (it is: deadline timestamp).
    # If shift=0 it does nothing. If shift applied, deadline_key should shift too to stay in same basis.
    log_key_adj = log_key + shift

    # Strategy 1: rank-to-rank if lengths close
    n = len(df_ins_sorted)
    m = len(log_ins_adj)
    out = df.copy()
    out["deadline_key"] = np.nan

    # if lengths equal-ish, do rank-to-rank mapping for stability
    if abs(n - m) <= max(3, int(0.02 * max(n, m))):
        k = min(n, m)
        # map first k in insertion order
        out_key = np.full(n, np.nan, dtype=float)
        out_key[:k] = log_key_adj[:k]
        out.loc[order, "deadline_key"] = out_key
        return out, f"{src} + {align_src} + rank2rank(k={k}, n={n}, m={m})"

    # Strategy 2: two-pointer nearest match (monotone)
    i = 0
    j = 0
    matched = 0
    out_key_sorted = np.full(n, np.nan, dtype=float)
    while i < n and j < m:
        a = df_ins_sorted[i]
        b = log_ins_adj[j]
        if abs(a - b) <= eps:
            out_key_sorted[i] = log_key_adj[j]
            matched += 1
            i += 1
            j += 1
        elif a < b:
            # csv earlier than log; advance csv
            i += 1
        else:
            # log earlier; advance log
            j += 1

    out.loc[order, "deadline_key"] = out_key_sorted
    frac = matched / max(1, n)
    # If matching is too poor, fallback to csv deadline
    if frac < 0.3 and "orig_slo" in out.columns:
        slo = pd.to_numeric(out["orig_slo"], errors="coerce")
        out["deadline_key"] = out["_t_ins"] + slo
        return out, f"csv(deadline=_t_ins+orig_slo) (log match poor: matched={matched}/{n})"

    return out, f"{src} + {align_src} + two-pointer(eps={eps}, matched={matched}/{n})"

# ----------------------------
# Plot C & F
# ----------------------------
def plot_C_F(csv_path: str, out_dir: str, maxN: int = 200, eps_match: float = 5e-3):
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
    # (C) insertion order timeline (RUN only)
    # ----------------------------
    N = min(maxN, len(df))
    sub = df.iloc[:N].copy()
    t0 = sub["_t_ins"].min()
    sub["_d0"] = sub["_t_dsp"] - t0
    sub["_f0"] = sub["_t_fin"] - t0
    sub["y"] = np.arange(len(sub))

    plt.figure(figsize=(12, 6))
    for _, r in sub.iterrows():
        color = "tab:red" if r["is_long"] else "tab:blue"
        plt.plot([r["_d0"], r["_f0"]], [r["y"], r["y"]], color=color, linewidth=2)
    plt.xlabel("Time since first arrival [s]")
    plt.ylabel("Insertion order index (first N)")
    plt.title(f"(C) Timeline (RUN only): red=LONG, blue=SHORT {title_suffix}")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "C_timeline_gantt.png"), dpi=220)
    plt.close()

    # ----------------------------
    # (F) deadline-key order timeline (RUN only)
    # deadline_key is derived from log(deadline=ins+slo) preferably, optionally group-head key
    # ----------------------------
    df2, src = map_deadline_key_to_csv(df, csv_path, eps=eps_match)
    df2 = df2.dropna(subset=["deadline_key"]).sort_values("_t_ins").reset_index(drop=True)
    if len(df2) == 0:
        with open(os.path.join(out_dir, "F.SKIPPED.txt"), "w") as f:
            f.write("No usable deadline_key from log/csv.\n")
        return

    N2 = min(maxN, len(df2))
    sub2 = df2.iloc[:N2].copy()
    t0_2 = sub2["_t_ins"].min()
    sub2["_d0"] = sub2["_t_dsp"] - t0_2
    sub2["_f0"] = sub2["_t_fin"] - t0_2

    # smaller deadline_key => more urgent
    sub2["ddl_rank"] = sub2["deadline_key"].rank(method="first", ascending=True).astype(int) - 1
    sub2["y"] = (len(sub2) - 1) - sub2["ddl_rank"]  # most urgent at top

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

    with open(os.path.join(out_dir, "CF_meta.txt"), "w") as f:
        f.write(f"file={base}\n")
        if sleep and pushint:
            f.write(f"sleep={sleep}\npushint={pushint}\n")
        f.write(f"N_total={len(df)}\n")
        f.write(f"N_plotted={N}\n")
        f.write(f"deadline_key_source={src}\n")

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Plot only C and F timelines with corrected deadline-key mapping.")
    ap.add_argument("--glob", default="timsort_hol_sleep_*_pushint_*_*.csv",
                    help="Glob for CSV inputs")
    ap.add_argument("--csv", default=None, help="Single CSV path (overrides --glob)")
    ap.add_argument("--out", default="per_file_plots_CF", help="Root output directory")
    ap.add_argument("--maxN", type=int, default=200, help="Max requests to plot per file")
    ap.add_argument("--eps_match", type=float, default=5e-3, help="Insertion-time match eps (seconds)")
    args = ap.parse_args()

    if args.csv:
        csvs = [args.csv]
    else:
        csvs = sorted(glob.glob(args.glob))

    if not csvs:
        print("No CSVs found.")
        return

    os.makedirs(args.out, exist_ok=True)

    ok, fail = 0, 0
    for csv_path in csvs:
        base = os.path.basename(csv_path).replace(".csv", "")
        out_dir = os.path.join(args.out, base)
        os.makedirs(out_dir, exist_ok=True)
        try:
            plot_C_F(csv_path, out_dir, maxN=args.maxN, eps_match=args.eps_match)
            print(f"[OK] {csv_path} -> {out_dir}")
            ok += 1
        except Exception as e:
            print(f"[FAIL] {csv_path}: {e}")
            fail += 1

    print(f"Done. ok={ok}, fail={fail}, out_root={args.out}")

if __name__ == "__main__":
    main()

