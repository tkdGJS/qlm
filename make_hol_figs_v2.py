import os, re, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# filename parse
# ----------------------------
FNAME_RE = re.compile(r"timsort_hol_sleep_([0-9.]+)_pushint_([0-9.]+)_.+\.csv$")

def parse_meta(path):
    base = os.path.basename(path)
    m = FNAME_RE.match(base)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))

def parse_time_series(s):
    if np.issubdtype(s.dtype, np.number):
        return s.astype(float)
    ts = pd.to_datetime(s, errors="coerce", utc=True)
    if ts.notna().any():
        return (ts.view("int64") / 1e9).astype(float)
    return pd.to_numeric(s, errors="coerce").astype(float)

# ----------------------------
# HOL metrics for your CSV schema
# required: insertion_time, wait_time, execution_time
# optional: finished_time, out_tok, prompt_tok, success_rate_pct, violation
# ----------------------------
def compute_metrics(csv_path, classify="out_tok"):
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
        # if finished_time is junk, fallback to dsp+service
        bad = df["_t_fin"].isna()
        df.loc[bad, "_t_fin"] = df.loc[bad, "_t_dsp"] + df.loc[bad, "service_time"]
    else:
        df["_t_fin"] = df["_t_dsp"] + df["service_time"]

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["_t_ins","queue_delay","service_time","_t_dsp","_t_fin"])
    df = df[(df["queue_delay"] >= 0) & (df["service_time"] >= 0) & (df["_t_fin"] >= df["_t_dsp"]) & (df["_t_dsp"] >= df["_t_ins"])]
    if len(df) < 30:
        return None

    df = df.sort_values("_t_ins").reset_index(drop=True)

    # classify long/short
    if classify == "out_tok" and "out_tok" in df.columns:
        v = pd.to_numeric(df["out_tok"], errors="coerce")
        thr = v.median()
        df["is_long"] = v > thr
    elif classify == "prompt_tok" and "prompt_tok" in df.columns:
        v = pd.to_numeric(df["prompt_tok"], errors="coerce")
        thr = v.median()
        df["is_long"] = v > thr
    else:
        # fallback: service_time
        thr = df["service_time"].median()
        df["is_long"] = df["service_time"] > thr

    short = df[~df["is_long"]].copy()
    long_df = df[df["is_long"]].copy()
    if len(short) < 10 or len(long_df) < 3:
        return None

    # blocking depth at arrival: sum remaining time of LONG in-flight work
    longs = long_df[["_t_dsp","_t_fin"]].copy()
    blocking = []
    for t in df["_t_ins"].values:
        active = longs[(longs["_t_dsp"] <= t) & (longs["_t_fin"] > t)]
        blocking.append(float((active["_t_fin"] - t).sum()))
    df["blocking_depth"] = blocking
    short = df[~df["is_long"]].copy()

    q = short["queue_delay"].values
    b = short["blocking_depth"].values

    hol_corr = np.nan
    if np.std(q) > 1e-9 and np.std(b) > 1e-9:
        hol_corr = float(np.corrcoef(q, b)[0,1])

    p95_q = float(np.percentile(q, 95))
    p99_q = float(np.percentile(q, 99))

    # "HOL signature" fraction: short queueing dominates short service
    s = short["service_time"].values
    s_med = float(np.median(s)) if np.median(s) > 1e-9 else float(np.mean(s) + 1e-9)
    hol_frac = float(np.mean(q > 10.0 * s_med))

    # optional stability metric
    sr = float(pd.to_numeric(df["success_rate_pct"], errors="coerce").dropna().iloc[-1]) if "success_rate_pct" in df.columns and df["success_rate_pct"].notna().any() else np.nan

    return {
        "n_total": int(len(df)),
        "n_short": int(len(short)),
        "hol_corr": hol_corr,
        "p95_short_q": p95_q,
        "p99_short_q": p99_q,
        "hol_frac": hol_frac,
        "success_rate_pct_end": sr
    }

# ----------------------------
# plotting: heatmap without seaborn
# ----------------------------
def heatmap(df, value_col, title, outpath):
    pv = df.pivot_table(index="sleep", columns="pushint", values=value_col, aggfunc="mean")
    pv = pv.sort_index().sort_index(axis=1)

    plt.figure(figsize=(10.5, 4.8))
    data = pv.values
    masked = np.ma.masked_invalid(data)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="lightgray")
    im = plt.imshow(masked, aspect="auto", origin="lower", cmap=cmap)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    plt.xticks(np.arange(pv.shape[1]), [str(c) for c in pv.columns], rotation=45, ha="right")
    plt.yticks(np.arange(pv.shape[0]), [str(r) for r in pv.index])
    plt.xlabel("pushint (sec)")
    plt.ylabel("sleep (sec)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()

def main():
    csvs = sorted(glob.glob("timsort_hol_sleep_*_pushint_*_*.csv"))
    rows = []
    for p in csvs:
        meta = parse_meta(p)
        if meta is None:
            continue
        sleep, pushint = meta
        m = compute_metrics(p, classify="out_tok")  # change to prompt_tok if needed
        if m is None:
            continue
        m["sleep"] = sleep
        m["pushint"] = pushint
        m["file"] = os.path.basename(p)
        rows.append(m)

    if not rows:
        print("No usable CSV parsed.")
        return

    outdir = "hol_figs_v2"
    os.makedirs(outdir, exist_ok=True)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(outdir, "hol_metrics_all.csv"), index=False)
    print(f"Wrote {outdir}/hol_metrics_all.csv with {len(res)} conditions")

    heatmap(res, "hol_corr",
            "HOL Index = corr(short wait_time, blocking_depth)",
            os.path.join(outdir, "H1_hol_index_corr_heatmap.png"))

    heatmap(res, "p99_short_q",
            "Tail queueing of SHORT (p99 wait_time) [s]",
            os.path.join(outdir, "H2_p99_short_queueing_heatmap.png"))

    heatmap(res, "hol_frac",
            "HOL fraction: P(short wait_time > 10x median(short execution_time))",
            os.path.join(outdir, "H3_hol_fraction_heatmap.png"))

    heatmap(res, "success_rate_pct_end",
            "End success_rate_pct (proxy stability)",
            os.path.join(outdir, "H4_success_rate_end_heatmap.png"))

    # scatter: mechanism vs tail
    plt.figure(figsize=(6.6,5.2))
    plt.scatter(res["hol_corr"], res["p99_short_q"], s=35, alpha=0.8)
    plt.xlabel("HOL Index corr")
    plt.ylabel("p99 short wait_time [s]")
    plt.title("Condition points: HOL mechanism vs tail queueing")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "H5_scatter_hol_vs_tail.png"), dpi=220)
    plt.close()

    print(f"Saved heatmaps & scatter into ./{outdir}/")

if __name__ == "__main__":
    main()

