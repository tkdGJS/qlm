# plot_hol_v3_per_file.py
import os, re, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FNAME_RE = re.compile(r".*timsort_hol_sleep_([0-9.]+)_pushint_([0-9.]+)_.+\.csv$")

def parse_time_series(s: pd.Series) -> pd.Series:
    if np.issubdtype(s.dtype, np.number):
        return s.astype(float)
    ts = pd.to_datetime(s, errors="coerce", utc=True)
    if ts.notna().any():
        return (ts.view("int64") / 1e9).astype(float)
    return pd.to_numeric(s, errors="coerce").astype(float)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def classify_long_short(df: pd.DataFrame) -> pd.DataFrame:
    # same as v2: out_tok median split (fallback prompt_tok, service_time)
    if "out_tok" in df.columns:
        ot = pd.to_numeric(df["out_tok"], errors="coerce")
        thr = np.nanmedian(ot.values)
        df["is_long"] = ot > thr
    elif "prompt_tok" in df.columns:
        pt = pd.to_numeric(df["prompt_tok"], errors="coerce")
        df["is_long"] = pt > np.nanmedian(pt.values)
    else:
        df["is_long"] = df["service_time"] > np.nanmedian(df["service_time"].values)
    df["is_long"] = df["is_long"].fillna(False)
    return df

def infer_deadline(df: pd.DataFrame) -> pd.DataFrame:
    """
    If orig_slo is absolute deadline (seconds), use it directly.
    If orig_slo is budget seconds, deadline = insertion_time + orig_slo.
    Also compute budget = deadline - insertion_time.
    """
    if "orig_slo" not in df.columns:
        df["deadline"] = np.nan
        df["budget"] = np.nan
        df["deadline_mode"] = "missing"
        return df

    ins = df["_t_ins"].values
    slo = parse_time_series(df["orig_slo"]).values

    ins_med = np.nanmedian(ins)
    slo_med = np.nanmedian(slo)

    # Heuristic:
    # - epoch-like seconds ~ 1e9+
    # - if both look epoch-like, treat slo as absolute deadline
    # - else treat slo as budget seconds
    if np.isfinite(ins_med) and np.isfinite(slo_med) and (ins_med > 1e8) and (slo_med > 1e8):
        deadline = slo
        mode = "absolute_deadline"
    else:
        deadline = ins + slo
        mode = "budget_seconds"

    df["deadline"] = deadline
    df["budget"] = df["deadline"] - df["_t_ins"]
    df["deadline_mode"] = mode
    return df

def compute_blocking_depth(df: pd.DataFrame) -> pd.DataFrame:
    # blocking depth at arrival = sum remaining time of active LONG at insertion
    longs = df[df["is_long"]][["_t_dsp","_t_fin"]].copy()
    longs = longs.dropna()
    bd = []
    for t in df["_t_ins"].values:
        if len(longs) == 0 or not np.isfinite(t):
            bd.append(0.0)
            continue
        active = longs[(longs["_t_dsp"] <= t) & (longs["_t_fin"] > t)]
        rem = (active["_t_fin"] - t)
        bd.append(float(np.nansum(rem.values)))
    df["blocking_depth"] = np.asarray(bd, dtype=float)
    df["blocking_depth"] = df["blocking_depth"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df

def plot_one(csv_path: str, out_dir: str):
    base = os.path.basename(csv_path)
    m = FNAME_RE.match(base)
    sleep = pushint = None
    if m:
        sleep = m.group(1)
        pushint = m.group(2)
    title_suffix = f"(sleep={sleep}, pushint={pushint})"

    df = pd.read_csv(csv_path)

    # required columns for A~E
    need = ["insertion_time", "wait_time", "execution_time"]
    for c in need:
        if c not in df.columns:
            raise KeyError(f"{csv_path}: Missing '{c}'. cols={list(df.columns)}")

    # times
    df["_t_ins"] = parse_time_series(df["insertion_time"])
    df["queue_delay"] = pd.to_numeric(df["wait_time"], errors="coerce")
    df["service_time"] = pd.to_numeric(df["execution_time"], errors="coerce")
    df["sojourn_time"] = df["queue_delay"] + df["service_time"]

    # reconstruct dispatch/finish
    df["_t_dsp"] = df["_t_ins"] + df["queue_delay"]
    if "finished_time" in df.columns:
        df["_t_fin"] = parse_time_series(df["finished_time"])
    else:
        df["_t_fin"] = df["_t_dsp"] + df["service_time"]

    # clean
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["_t_ins","queue_delay","service_time","sojourn_time","_t_dsp","_t_fin"])
    df = df[(df["queue_delay"] >= 0) & (df["service_time"] >= 0)]
    df = df.sort_values("_t_ins").reset_index(drop=True)

    df = classify_long_short(df)
    df = compute_blocking_depth(df)
    df = infer_deadline(df)

    short = df[~df["is_long"]].copy()
    long_df = df[df["is_long"]].copy()

    # ---- (A) short: queue vs service scatter ----
    plt.figure(figsize=(8,6))
    plt.scatter(short["service_time"], short["queue_delay"], s=12, alpha=0.6)
    plt.xlabel("Service time (execution_time) [s]")
    plt.ylabel("Queueing delay (wait_time) [s]")
    plt.title(f"(A) SHORT queueing vs service {title_suffix}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "A_queue_vs_service_short.png"), dpi=220)
    plt.close()

    # ---- (B) blocking depth vs short queueing ----
    plt.figure(figsize=(8,6))
    plt.scatter(short["blocking_depth"], short["queue_delay"], s=12, alpha=0.6)
    plt.xlabel("Blocking depth at arrival (sum remaining LONG time) [s]")
    plt.ylabel("Queueing delay of SHORT (wait_time) [s]")
    plt.title(f"(B) SHORT queueing vs long in-flight work {title_suffix}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "B_blocking_depth.png"), dpi=220)
    plt.close()

    # ---- (C) timeline run-only (first 200 arrivals) ----
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
    plt.xlabel("Time since first arrival [s]")
    plt.ylabel("Arrival order index (by insertion)")
    plt.title(f"(C) Timeline (run only, first {N}): red=LONG, blue=SHORT {title_suffix}")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "C_timeline_run_only.png"), dpi=220)
    plt.close()

    # ---- (D) timeline with waiting + run (first 200) ----
    plt.figure(figsize=(12,6))
    for _, r in sub.iterrows():
        y = r["arr_idx"]
        # wait segment: insertion -> dispatch
        plt.plot([r["_d0"] - r["queue_delay"], r["_d0"]], [y, y], color="0.7", linewidth=2)
        # run segment: dispatch -> finish
        color = "tab:red" if r["is_long"] else "tab:blue"
        plt.plot([r["_d0"], r["_f0"]], [y, y], color=color, linewidth=2)
    plt.xlabel("Time since first arrival [s]")
    plt.ylabel("Arrival order index (by insertion)")
    plt.title(f"(D) Timeline (wait+run, first {N}): gray=wait, red=LONG, blue=SHORT {title_suffix}")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "D_timeline_wait_plus_run.png"), dpi=220)
    plt.close()

    # ---- (E) short-only: show waiting directly, colored by blocking depth ----
    shortN = min(250, len(short))
    if shortN > 0:
        ssub = short.iloc[:shortN].copy()
        t0s = df["_t_ins"].min()
        ssub["_ins0"] = ssub["_t_ins"] - t0s
        ssub["_dsp0"] = ssub["_t_dsp"] - t0s
        ssub["_fin0"] = ssub["_t_fin"] - t0s
        ssub["idx"] = np.arange(len(ssub))

        bd = ssub["blocking_depth"].values.astype(float)
        bd = np.nan_to_num(bd, nan=0.0, posinf=0.0, neginf=0.0)

        vmin = float(np.percentile(bd, 5)) if len(bd) else 0.0
        vmax = float(np.percentile(bd, 95)) if len(bd) else 1.0
        if not np.isfinite(vmin): vmin = 0.0
        if not np.isfinite(vmax) or vmax <= vmin: vmax = vmin + 1.0

        cmap = plt.cm.viridis
        def color_for(x):
            x = float(x)
            x = min(max(x, vmin), vmax)
            a = (x - vmin) / (vmax - vmin + 1e-9)
            return cmap(a)

        fig, ax = plt.subplots(figsize=(12, 6))
        for _, r in ssub.iterrows():
            y = r["idx"]
            c = color_for(r["blocking_depth"])
            # wait (insertion->dispatch) thick bar
            ax.plot([r["_ins0"], r["_dsp0"]], [y, y], color=c, linewidth=3)
            # run (dispatch->finish) thin black overlay
            ax.plot([r["_dsp0"], r["_fin0"]], [y, y], color="k", linewidth=1, alpha=0.5)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])  # 중요: mappable로 인식시키기
        cb = fig.colorbar(sm, ax=ax)
        cb.set_label("blocking_depth (sum remaining LONG time) [s]")

        ax.set_xlabel("Time since first arrival [s]")
        ax.set_ylabel("SHORT index (by insertion)")
        ax.set_title(f"(E) SHORT wait bars colored by blocking depth {title_suffix}")
        ax.grid(True, axis="x", alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "E_short_wait_colored_by_blocking.png"), dpi=220)
        plt.close(fig)

    # ---- (F) timeline by deadline order (if deadline exists) ----
    if df["deadline"].notna().any():
        N2 = min(200, len(df))
        sub2 = df.iloc[:N2].copy()
        # order by earliest deadline first => most urgent at top
        sub2 = sub2.sort_values("deadline", ascending=True).reset_index(drop=True)
        sub2["dl_rank"] = np.arange(len(sub2))[::-1]  # top = most urgent
        t0 = df["_t_ins"].min()
        sub2["_d0"] = sub2["_t_dsp"] - t0
        sub2["_f0"] = sub2["_t_fin"] - t0

        plt.figure(figsize=(12,6))
        for _, r in sub2.iterrows():
            color = "tab:red" if r["is_long"] else "tab:blue"
            plt.plot([r["_d0"], r["_f0"]], [r["dl_rank"], r["dl_rank"]], color=color, linewidth=2)
        plt.xlabel("Time since first arrival [s]")
        plt.ylabel("Deadline rank (top=more urgent)")
        plt.title(f"(F) Run timeline by deadline order [{df['deadline_mode'].iloc[0]}] {title_suffix}")
        plt.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "F_timeline_deadline_order.png"), dpi=220)
        plt.close()

        # ---- (G) inversion flags at dispatch: is someone more urgent waiting? ----
        # For each request j at its dispatch time, check if there exists i with:
        # ins_i <= dsp_j, dsp_i > dsp_j (i not started yet), deadline_i < deadline_j
        ins = df["_t_ins"].values
        dsp = df["_t_dsp"].values
        ddl = df["deadline"].values
        inv = np.zeros(len(df), dtype=bool)
        for j in range(len(df)):
            tj = dsp[j]
            if not np.isfinite(tj) or not np.isfinite(ddl[j]):
                continue
            mask = (ins <= tj) & (dsp > tj) & (ddl < ddl[j])
            if np.any(mask):
                inv[j] = True
        df["inversion_at_dispatch"] = inv

        sub3 = df.iloc[:N].copy()
        t0 = sub3["_t_ins"].min()
        sub3["_ins0"] = sub3["_t_ins"] - t0
        sub3["_dsp0"] = sub3["_t_dsp"] - t0
        sub3["_fin0"] = sub3["_t_fin"] - t0
        sub3["arr_idx"] = np.arange(len(sub3))

        plt.figure(figsize=(12,6))
        for _, r in sub3.iterrows():
            y = r["arr_idx"]
            # wait
            plt.plot([r["_ins0"], r["_dsp0"]], [y, y], color="0.7", linewidth=2)
            # run
            color = "tab:red" if r["is_long"] else "tab:blue"
            plt.plot([r["_dsp0"], r["_fin0"]], [y, y], color=color, linewidth=2)
            # inversion marker
            if bool(r.get("inversion_at_dispatch", False)):
                plt.scatter([r["_dsp0"]], [y], color="black", s=18, marker="x", zorder=5)

        plt.xlabel("Time since first arrival [s]")
        plt.ylabel("Arrival order index (by insertion)")
        plt.title(f"(G) Inversions marked at dispatch (x) {title_suffix}")
        plt.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "G_inversion_marked_timeline.png"), dpi=220)
        plt.close()

    # ---- metrics ----
    hol_corr = np.nan
    if len(short) >= 10 and np.std(short["queue_delay"].values) > 1e-9 and np.std(short["blocking_depth"].values) > 1e-9:
        hol_corr = float(np.corrcoef(short["queue_delay"].values, short["blocking_depth"].values)[0,1])

    inv_rate = float(np.mean(df.get("inversion_at_dispatch", pd.Series([False]*len(df))).values)) if len(df) else np.nan

    with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
        f.write(f"file={base}\n")
        if sleep is not None and pushint is not None:
            f.write(f"sleep={sleep}\npushint={pushint}\n")
        f.write(f"deadline_mode={df['deadline_mode'].iloc[0] if 'deadline_mode' in df.columns else 'n/a'}\n")
        f.write(f"n_total={len(df)}\n")
        f.write(f"n_short={len(short)}\n")
        f.write(f"hol_corr(short_queue_delay, blocking_depth)={hol_corr}\n")
        f.write(f"inversion_rate_at_dispatch={inv_rate}\n")

    return hol_corr

def main():
    csvs = sorted(glob.glob("timsort_hol_sleep_*_pushint_*_*.csv"))
    if not csvs:
        print("No matching CSVs found.")
        return

    root_out = "per_file_plots_v3"
    ensure_dir(root_out)

    for csv_path in csvs:
        base = os.path.basename(csv_path).replace(".csv", "")
        out_dir = os.path.join(root_out, base)
        ensure_dir(out_dir)
        try:
            hol_corr = plot_one(csv_path, out_dir)
            print(f"[OK] {csv_path} -> {out_dir} (hol_corr={hol_corr})")
        except Exception as e:
            print(f"[FAIL] {csv_path}: {type(e).__name__}: {e}")

if __name__ == "__main__":
    main()

