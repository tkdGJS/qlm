import sys, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def parse_time_series(s):
    if np.issubdtype(s.dtype, np.number):
        return s.astype(float)
    ts = pd.to_datetime(s, errors="coerce", utc=True)
    if ts.notna().any():
        return (ts.view("int64") / 1e9).astype(float)
    return pd.to_numeric(s, errors="coerce").astype(float)

def main(csv_path):
    df = pd.read_csv(csv_path)
    print("Columns:", list(df.columns))

    need = ["insertion_time", "wait_time", "execution_time"]
    for c in need:
        if c not in df.columns:
            raise KeyError(f"Missing required column: {c}")

    # times
    df["_t_ins"] = parse_time_series(df["insertion_time"])
    df["queue_delay"] = pd.to_numeric(df["wait_time"], errors="coerce")
    df["service_time"] = pd.to_numeric(df["execution_time"], errors="coerce")
    df["sojourn_time"] = df["queue_delay"] + df["service_time"]

    # reconstruct dispatch/finish if needed
    df["_t_dsp"] = df["_t_ins"] + df["queue_delay"]
    if "finished_time" in df.columns:
        df["_t_fin"] = parse_time_series(df["finished_time"])
        # sanity: if finished_time exists, prefer it for any checks
    else:
        df["_t_fin"] = df["_t_dsp"] + df["service_time"]

    # clean
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["_t_ins","queue_delay","service_time","sojourn_time","_t_dsp","_t_fin"])
    df = df[(df["queue_delay"] >= 0) & (df["service_time"] >= 0)]
    df = df.sort_values("_t_ins").reset_index(drop=True)

    # classify short/long (prefer out_tok; fallback prompt_tok; else service_time)
    if "out_tok" in df.columns:
        ot = pd.to_numeric(df["out_tok"], errors="coerce")
        # if bimodal, median works fine
        thr = ot.median()
        df["is_long"] = ot > thr
    elif "prompt_tok" in df.columns:
        pt = pd.to_numeric(df["prompt_tok"], errors="coerce")
        df["is_long"] = pt > pt.median()
    else:
        df["is_long"] = df["service_time"] > df["service_time"].median()

    short = df[~df["is_long"]].copy()
    long_df = df[df["is_long"]].copy()

    # ---- (A) short: queue vs service scatter ----
    plt.figure(figsize=(8,6))
    plt.scatter(short["service_time"], short["queue_delay"], s=12, alpha=0.6)
    plt.xlabel("Execution_time [s]")
    plt.ylabel("Queueing delay (wait_time) [s]")
    plt.title("HOL evidence: SHORT requests (queueing vs service)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("A_queue_vs_service_short.png", dpi=220)
    plt.close()

    # ---- (B) blocking depth vs short queueing ----
    # blocking depth at arrival: sum of remaining time of long jobs in-flight at that arrival
    blocking = []
    longs = long_df[["_t_dsp","_t_fin"]].copy()
    for t in df["_t_ins"].values:
        active = longs[(longs["_t_dsp"] <= t) & (longs["_t_fin"] > t)]
        blocking.append(float((active["_t_fin"] - t).sum()))
    df["blocking_depth"] = blocking
    short = df[~df["is_long"]].copy()

    plt.figure(figsize=(8,6))
    plt.scatter(short["blocking_depth"], short["queue_delay"], s=12, alpha=0.6)
    plt.xlabel("Blocking depth (sum remaining requests execution time) [s]")
    plt.ylabel("Queueing delay of SHORT (wait_time) [s]")
    plt.title("HOL mechanism: short queueing vs long in-flight work")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("B_blocking_depth.png", dpi=220)
    plt.close()

    # ---- (C) report HOL index ----
    q = short["queue_delay"].values
    b = short["blocking_depth"].values
    hol_corr = np.nan
    if len(short) >= 10 and np.std(q) > 1e-9 and np.std(b) > 1e-9:
        hol_corr = float(np.corrcoef(q, b)[0,1])
    print(f"HOL corr(short queue_delay, blocking_depth) = {hol_corr:.3f}  (n_short={len(short)})")

    # optional: timeline-ish (first 200 arrivals)
    N = min(200, len(df))
    sub = df.iloc[:N].copy()
    t0 = sub["_t_ins"].min()
    sub["_d0"] = sub["_t_dsp"] - t0
    sub["_f0"] = sub["_t_fin"] - t0
    sub["arr_idx"] = np.arange(len(sub))

    plt.figure(figsize=(12,6))
    for _, r in sub.iterrows():
        color = "tab:red" if r["is_long"] else "tab:blue"
        plt.plot([r["_d0"], r["_f0"]], [r["arr_idx"], r["arr_idx"]], color=color, linewidth=2)
    plt.xlabel("First arrival time [s]")
    plt.ylabel("Arrival order requests")
    plt.title("Timeline (first 200): red=LONG, blue=SHORT")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("C_timeline_gantt.png", dpi=220)
    plt.close()

    print("Saved: A_queue_vs_service_short.png, B_blocking_depth.png, C_timeline_gantt.png")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_hol_v2.py <csv_path>")
        sys.exit(1)
    main(sys.argv[1])

