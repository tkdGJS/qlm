#!/usr/bin/env python3
# pyright: reportMissingImports=false
# pyright: reportMissingTypeStubs=false
# pyright: reportImplicitStringConcatenation=false
# pyright: reportUnnecessaryComparison=false
# pyright: reportUnreachable=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownParameterType=false
# pyright: reportUnknownLambdaType=false
# pyright: reportMissingParameterType=false
# pyright: reportUnusedCallResult=false
"""six_mode_plot.py
Generates the same plot style as red_black.py, but for 6 modes:
  native_dram, native_disk, native_dram_disk, cachegen_dram, cachegen_disk, cachegen_dram_disk

Edit the 'files' dict at the top if your filenames differ.
Output: ./six_mode_graph_output/*.png + *.csv

CLI:
  python3 six_mode_plot.py --root /path/to/results_root --outdir local_six
"""
import os, re, json, math, argparse, glob
from typing import cast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

files = {
  "native_dram": {
"run": "/home/yuhwa2323/noslab_gpu/result_20260305_203428_native_dram/run_native_dram.log",
"vram": "/home/yuhwa2323/noslab_gpu/result_20260305_203428_native_dram/multi_req_lmcache_vram.log",
"vram_kind": "raw"
  },
  "native_disk": {
"run": "/home/yuhwa2323/noslab_gpu/result_20260305_204737_native_disk/run_native_disk.log",
"vram": "/home/yuhwa2323/noslab_gpu/result_20260305_204737_native_disk/multi_req_lmcache_vram.log",
"vram_kind": "raw"
  },
  "native_dram_disk": {
"run": "/home/yuhwa2323/noslab_gpu/result_20260305_210044_native_dram_disk/run_native_dram_disk.log",
"vram": "/home/yuhwa2323/noslab_gpu/result_20260305_210044_native_dram_disk/multi_req_lmcache_vram.log",
"vram_kind": "raw"
  },
  "cachegen_dram": {
"run": "/home/yuhwa2323/noslab_gpu/result_20260305_211351_cachegen_dram/run_cachegen_dram.log",
"vram": "/home/yuhwa2323/noslab_gpu/result_20260305_211351_cachegen_dram/multi_req_lmcache_vram.log",
"vram_kind": "raw"
  },
  "cachegen_disk": {
"run": "/home/yuhwa2323/noslab_gpu/result_20260305_212658_cachegen_disk/run_cachegen_disk.log",
"vram": "/home/yuhwa2323/noslab_gpu/result_20260305_212658_cachegen_disk/multi_req_lmcache_vram.log",
"vram_kind": "raw"
  },
  "cachegen_dram_disk": {
"run": "/home/yuhwa2323/noslab_gpu/result_20260305_214005_cachegen_dram_disk/run_cachegen_dram_disk.log",
"vram": "/home/yuhwa2323/noslab_gpu/result_20260305_214005_cachegen_dram_disk/multi_req_lmcache_vram.log",
"vram_kind": "raw"
  }
}
order = [
  "native_dram",
  "native_disk",
  "native_dram_disk",
  "cachegen_dram",
  "cachegen_disk",
  "cachegen_dram_disk"
]
colors = {
  "native_dram": "tab:blue",
  "native_disk": "tab:orange",
  "native_dram_disk": "tab:green",
  "cachegen_dram": "tab:red",
  "cachegen_disk": "tab:purple",
  "cachegen_dram_disk": "tab:brown"
}

DEBUG_RE = re.compile(
    r"Insertion Time:\s*([0-9.]+)\s*\|"
    r".*?TTFT:\s*([0-9.]+|None)\s*\|"
    r"\s*TBT_p95=([0-9.]+|NA)\s*\|"
    r"\s*TBT_p99=([0-9.]+|NA)\s*\|"
    r".*?TTLT:\s*([0-9.]+)\s*\|"
    r".*?(?:Output Tokens|out_tok(?:s)?):\s*([0-9]+)"
    r".*?Finished Time:\s*([0-9.]+)",
    re.IGNORECASE,
)
DEBUG_RE_NO_OUTTOK = re.compile(
    r"Insertion Time:\s*([0-9.]+)\s*\|"
    r".*?TTFT:\s*([0-9.]+|None)\s*\|"
    r"\s*TBT_p95=([0-9.]+|NA)\s*\|"
    r"\s*TBT_p99=([0-9.]+|NA)\s*\|"
    r".*?TTLT:\s*([0-9.]+)\s*\|"
    r".*?Finished Time:\s*([0-9.]+)",
    re.IGNORECASE,
)
SAMPLER_RE = re.compile(r"^\[CLIENT_SAMPLER\]\s+([0-9.]+)\s+\S+\s+([0-9.]+)GB")
EVENT_RE = re.compile(r"^\[LMCACHE_VRAM\]\[LocalDiskBackend\]\s+([0-9.]+)\s+(serialize|deserialize)\b", re.IGNORECASE)
PREFIX_HIT_RE = re.compile(r"^\[PREFIX_HIT\]\s+([0-9.]+)\s+([0-9]+)\b")

def _to_float_or_nan(x: object):
    if x is None: return np.nan
    s = str(x).strip().lower()
    if s in ("none","na",""): return np.nan
    try: return float(s)
    except: return np.nan

def parse_run_log(path: str, mode: str) -> pd.DataFrame:
    rows=[]
    with open(path,"r",errors="ignore") as f:
        for line in f:
            if not line.startswith("[DEBUG] OrigSLO:"): continue
            m = DEBUG_RE.search(line)
            if m:
                insertion_ts=float(m.group(1))
                ttft=_to_float_or_nan(m.group(2))
                tbt_p95=_to_float_or_nan(m.group(3))
                tbt_p99=_to_float_or_nan(m.group(4))
                ttlt=float(m.group(5))
                out_tok=int(m.group(6))
                finished_ts=float(m.group(7))
            else:
                m2=DEBUG_RE_NO_OUTTOK.search(line)
                if not m2: continue
                insertion_ts=float(m2.group(1))
                ttft=_to_float_or_nan(m2.group(2))
                tbt_p95=_to_float_or_nan(m2.group(3))
                tbt_p99=_to_float_or_nan(m2.group(4))
                ttlt=float(m2.group(5))
                out_tok=np.nan
                finished_ts=float(m2.group(6))
            rows.append({
                "mode":mode, "insertion_ts":insertion_ts, "finished_ts":finished_ts,
                "TTFT":ttft, "TBT_p95":tbt_p95, "TBT_p99":tbt_p99, "TTLT":ttlt, "out_tok":out_tok,
            })
    if not rows: raise ValueError(f"No parseable [DEBUG] request lines found in {path}")
    return pd.DataFrame(rows)

def parse_vram_raw_log(path: str, mode: str):
    srows, erows, hrows = [], [], []
    with open(path,"r",errors="ignore") as f:
        for line in f:
            sm = SAMPLER_RE.search(line)
            if sm:
                srows.append({"mode":mode,"ts":float(sm.group(1)),"vram_gb":float(sm.group(2))})
                continue
            em = EVENT_RE.search(line)
            if em:
                erows.append({"mode":mode,"ts":float(em.group(1)),"event":em.group(2).lower()})
                continue
            hm = PREFIX_HIT_RE.search(line)
            if hm:
                retrieved_tokens = int(hm.group(2))
                hrows.append({
                    "mode":mode,
                    "ts":float(hm.group(1)),
                    "retrieved_tokens":retrieved_tokens,
                    "hit":1 if retrieved_tokens > 0 else 0,
                })
    s_df = pd.DataFrame(srows) if srows else pd.DataFrame(columns=["mode","ts","vram_gb"])
    e_df = pd.DataFrame(erows) if erows else pd.DataFrame(columns=["mode","ts","event"])
    h_df = pd.DataFrame(hrows) if hrows else pd.DataFrame(columns=["mode","ts","retrieved_tokens","hit"])
    return s_df, e_df, h_df

def parse_vram_json(path: str, mode: str):
    with open(path, "r") as f:
        loaded = cast(object, json.load(f))
    data = loaded if isinstance(loaded, dict) else {}

    srows, erows, hrows = [], [], []

    client_events = data.get("client_events", [])
    if isinstance(client_events, list):
        for ev in client_events:
            if not isinstance(ev, dict):
                continue

            event_name = str(ev.get("event", "")).lower()
            if event_name == "periodic_sample":
                ts = _to_float_or_nan(ev.get("ts", None))
                vram_gb = _to_float_or_nan(ev.get("vram_gb", None))
                if np.isfinite(ts) and np.isfinite(vram_gb):
                    srows.append({"mode": mode, "ts": float(ts), "vram_gb": float(vram_gb)})
                continue

            if event_name in ("prefix_hit", "prefixhit"):
                ts = _to_float_or_nan(ev.get("ts", None))
                raw_hit = ev.get("hit", ev.get("value", ev.get("prefix_hit", None)))
                hit_val = _to_float_or_nan(raw_hit)
                if np.isfinite(ts) and np.isfinite(hit_val):
                    retrieved_tokens = int(round(float(hit_val)))
                    hrows.append({
                        "mode": mode,
                        "ts": float(ts),
                        "retrieved_tokens": retrieved_tokens,
                        "hit": 1 if retrieved_tokens > 0 else 0,
                    })

    lmcache_events = data.get("lmcache_events", [])
    if isinstance(lmcache_events, list):
        for ev in lmcache_events:
            if not isinstance(ev, dict):
                continue
            e = str(ev.get("event", "")).lower()
            if e in ("serialize", "deserialize"):
                ts = _to_float_or_nan(ev.get("ts", None))
                if np.isfinite(ts):
                    erows.append({"mode": mode, "ts": float(ts), "event": e})
    s_df = pd.DataFrame(srows) if srows else pd.DataFrame(columns=["mode","ts","vram_gb"])
    e_df = pd.DataFrame(erows) if erows else pd.DataFrame(columns=["mode","ts","event"])
    h_df = pd.DataFrame(hrows) if hrows else pd.DataFrame(columns=["mode","ts","retrieved_tokens","hit"])
    return s_df, e_df, h_df

def normalize_timelines(req_all: pd.DataFrame, vram_all: pd.DataFrame, evt_all: pd.DataFrame, hit_all: pd.DataFrame):
    mode_t0={}
    modes=sorted(set(req_all["mode"].dropna().tolist()))
    for mode in modes:
        candidates=[]
        r=req_all[req_all["mode"]==mode]
        if not r.empty:
            candidates.append(float(cast(float, np.nanmin(np.asarray(r["insertion_ts"], dtype=float)))))
        v=vram_all[vram_all["mode"]==mode] if not vram_all.empty else pd.DataFrame()
        if not v.empty:
            candidates.append(float(cast(float, np.nanmin(np.asarray(v["ts"], dtype=float)))))
        e=evt_all[evt_all["mode"]==mode] if not evt_all.empty else pd.DataFrame()
        if not e.empty:
            candidates.append(float(cast(float, np.nanmin(np.asarray(e["ts"], dtype=float)))))
        h=hit_all[hit_all["mode"]==mode] if not hit_all.empty else pd.DataFrame()
        if not h.empty:
            candidates.append(float(cast(float, np.nanmin(np.asarray(h["ts"], dtype=float)))))
        mode_t0[mode]=min(candidates) if candidates else 0.0
    req_all=req_all.copy(); vram_all=vram_all.copy(); evt_all=evt_all.copy(); hit_all=hit_all.copy()
    req_all["t_finish"]=req_all.apply(lambda r: r["finished_ts"]-mode_t0[r["mode"]], axis=1)
    req_all["t_insert"]=req_all.apply(lambda r: r["insertion_ts"]-mode_t0[r["mode"]], axis=1)
    if not vram_all.empty:
        vram_all["t"]=vram_all.apply(lambda r: r["ts"]-mode_t0[r["mode"]], axis=1)
    if not evt_all.empty:
        evt_all["t"]=evt_all.apply(lambda r: r["ts"]-mode_t0[r["mode"]], axis=1)
    if not hit_all.empty:
        hit_all["t"]=hit_all.apply(lambda r: r["ts"]-mode_t0[r["mode"]], axis=1)
    return req_all, vram_all, evt_all, hit_all

def compute_hit_metrics(hit_df: pd.DataFrame):
    if hit_df.empty:
        return {
            "prefix_hit_count": 0,
            "prefix_hit_total": 0,
            "prefix_hit_rate": np.nan,
        }
    total = int(len(hit_df))
    hit_count = int(hit_df["hit"].sum())
    hit_rate = float(hit_count / total) if total > 0 else np.nan
    return {
        "prefix_hit_count": hit_count,
        "prefix_hit_total": total,
        "prefix_hit_rate": hit_rate,
    }

def _hit_rate_timeline_series(hit_df: pd.DataFrame):
    if hit_df.empty:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    if "t" not in hit_df.columns or "hit" not in hit_df.columns:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    t = np.asarray(hit_df["t"], dtype=float)
    h = np.asarray(hit_df["hit"], dtype=float)
    if t.size == 0 or h.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    mask = np.isfinite(t) & np.isfinite(h) & (t >= 0.0)
    t = t[mask]
    h = h[mask]
    if t.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    order_idx = np.argsort(t)
    t = t[order_idx]
    h = h[order_idx]

    bins = np.floor(t).astype(int)
    if bins.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    max_bin = int(cast(int, bins[-1]))
    if max_bin < 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    counts = np.bincount(bins, minlength=max_bin + 1).astype(float)
    hits = np.bincount(bins, weights=h, minlength=max_bin + 1).astype(float)

    cum_counts = np.cumsum(counts)
    cum_hits = np.cumsum(hits)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.where(cum_counts > 0.0, (cum_hits / cum_counts) * 100.0, 0.0)
    x = np.arange(max_bin + 1, dtype=float)
    return x, y

def _save_blank_overlay(xlabel: str, ylabel: str, title: str, out_path: str):
    plt.figure(figsize=(14, 5))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def compute_summary_metrics(req_df: pd.DataFrame):
    ttft=req_df["TTFT"].dropna()
    tbt_p95=req_df["TBT_p95"].dropna()
    ttlt=req_df["TTLT"].dropna()
    est_tbt=pd.Series(dtype=float)
    if "out_tok" in req_df.columns:
        valid=(req_df["TTLT"].notna() & req_df["TTFT"].notna() & req_df["out_tok"].notna() & (req_df["out_tok"]>1))
        if valid.any():
            est_tbt=((req_df.loc[valid,"TTLT"]-req_df.loc[valid,"TTFT"])/(req_df.loc[valid,"out_tok"]-1)).dropna()
    def p95(s): return float(s.quantile(0.95)) if len(s) else np.nan
    return {
        "TTFT_mean": float(ttft.mean()) if len(ttft) else np.nan,
        "TTFT_p95": p95(ttft),
        "TBTp95_mean": float(tbt_p95.mean()) if len(tbt_p95) else np.nan,
        "TBTp95_p95": p95(tbt_p95),
        "estTBT_mean": float(est_tbt.mean()) if len(est_tbt) else np.nan,
        "estTBT_p95": p95(est_tbt),
        "TTLT_mean": float(ttlt.mean()) if len(ttlt) else np.nan,
        "TTLT_p95": p95(ttlt),
        "processed_request_count": int(len(req_df)),
    }

def _format_value_label(v: float) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))): return "NA"
    if abs(v) >= 1000: return f"{v:.0f}"
    if abs(v) >= 100:  return f"{v:.1f}"
    if abs(v) >= 10:   return f"{v:.2f}"
    return f"{v:.3f}"

def overlay_line_plot_multi(series, xlabel, ylabel, title, out_path):
    plt.figure(figsize=(14,5))
    for mode,(x,y) in series.items():
        if len(x):
            plt.plot(x,y,linewidth=0.9,label=mode,color=colors.get(mode,None))
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.legend(ncol=3)
    plt.tight_layout()
    plt.savefig(out_path,dpi=160)
    plt.close()

def bar_multi(title, ylabel, values_by_mode, out_path, width=0.7):
    labels=list(values_by_mode.keys())
    vals=[values_by_mode[m] for m in labels]
    plot_vals=[0.0 if (v is None or (isinstance(v,float) and (math.isnan(v) or math.isinf(v)))) else float(v) for v in vals]
    x=list(range(len(labels)))
    plt.figure(figsize=(10,5))
    plt.bar(x,plot_vals,width=width,color=[colors.get(m,None) for m in labels])
    plt.xticks(x,labels,rotation=15,ha="right")
    plt.ylabel(ylabel); plt.title(title)
    ymax=np.nanmax(plot_vals) if len(plot_vals) else 0.0
    if not np.isfinite(ymax): ymax=0.0
    pad=ymax*0.10 if ymax>0 else 0.05
    for xi,raw_v,pv in zip(x,vals,plot_vals):
        plt.text(xi,pv + (pad*0.15 if pv>0 else pad*0.05), _format_value_label(raw_v), ha="center", va="bottom", fontsize=9)
    plt.ylim(0, ymax + pad*2)
    plt.tight_layout()
    plt.savefig(out_path,dpi=180)
    plt.close()

def _extract_ts_from_dirname(name: str):
    m = re.search(r"\b(\d{8})_(\d{6})\b", name)
    if not m:
        m = re.search(r"\b(\d{8})(\d{6})\b", name)
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)))

def _discover_mode_dir(root: str, mode: str) -> str:
    pattern = os.path.join(root, f"*_{mode}")
    matches = [p for p in glob.glob(pattern) if os.path.isdir(p)]
    if not matches:
        raise ValueError(f"No mode directory found for mode={mode!r} under root={root!r} (pattern={pattern!r})")
    if len(matches) == 1:
        return matches[0]
    scored = []
    for p in matches:
        ts = _extract_ts_from_dirname(os.path.basename(p))
        scored.append((ts, p))
    if any(ts is None for ts, _ in scored):
        joined = "\n".join(sorted(matches))
        raise ValueError(
            f"Multiple directories match mode={mode!r} under root={root!r}, and not all have a parseable timestamp:\n{joined}"
        )
    scored.sort(key=lambda t: t[0])
    return scored[-1][1]

def _pick_run_log(mode_dir: str, mode: str) -> str:
    preferred = os.path.join(mode_dir, f"run_{mode}.log")
    if os.path.isfile(preferred):
        return preferred

    matches = sorted(glob.glob(os.path.join(mode_dir, "run_*.log")))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"No run_*.log found under {mode_dir!r} (mode={mode!r})")
    joined = "\n".join(matches)
    raise ValueError(f"Multiple run_*.log files found under {mode_dir!r} (mode={mode!r}); expected exactly 1:\n{joined}")

def _pick_vram_log(mode_dir: str) -> str:
    preferred = os.path.join(mode_dir, "multi_req_lmcache_vram.log")
    if os.path.isfile(preferred):
        return preferred
    fallback = os.path.join(mode_dir, "lmcache_vram.log")
    if os.path.isfile(fallback):
        return fallback
    raise ValueError(
        f"No VRAM log found under {mode_dir!r}; expected multi_req_lmcache_vram.log or lmcache_vram.log"
    )

def build_files_mapping_from_root(root: str):
    if not root:
        raise ValueError("root must be a non-empty path")
    if not os.path.isdir(root):
        raise ValueError(f"root is not a directory: {root!r}")
    mapping = {}
    for mode in order:
        mode_dir = _discover_mode_dir(root, mode)
        mapping[mode] = {
            "run": _pick_run_log(mode_dir, mode),
            "vram": _pick_vram_log(mode_dir),
            "vram_kind": "raw",
        }
    return mapping

def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Generate 6-mode graphs (including prefix hit-rate)")
    p.add_argument("--root", default=None, help="Result root containing the 6 mode directories (auto-discovered)")
    p.add_argument("--outdir", default="six_mode_graph_output", help="Output directory (default: six_mode_graph_output)")
    return p.parse_args(argv)

def main(argv=None):
    args = _parse_args(argv)
    outdir = cast(str, args.outdir)
    os.makedirs(outdir, exist_ok=True)

    files_map = files
    root = cast(object, args.root)
    if root:
        files_map = build_files_mapping_from_root(str(root))

    req_dfs=[]; vram_dfs=[]; evt_dfs=[]; hit_dfs=[]
    for mode in order:
        req_dfs.append(parse_run_log(files_map[mode]["run"], mode))
        if files_map[mode]["vram_kind"]=="raw":
            vram, evt, hit = parse_vram_raw_log(files_map[mode]["vram"], mode)
        else:
            vram, evt, hit = parse_vram_json(files_map[mode]["vram"], mode)
        vram_dfs.append(vram); evt_dfs.append(evt); hit_dfs.append(hit)

    req_all=pd.concat(req_dfs, ignore_index=True)
    vram_all=pd.concat(vram_dfs, ignore_index=True)
    evt_all=pd.concat(evt_dfs, ignore_index=True)
    hit_all=pd.concat(hit_dfs, ignore_index=True)
    req_all, vram_all, evt_all, hit_all = normalize_timelines(req_all, vram_all, evt_all, hit_all)

    req_all.to_csv(os.path.join(outdir,"parsed_request_metrics.csv"), index=False)
    vram_all.to_csv(os.path.join(outdir,"parsed_vram_samples.csv"), index=False)
    evt_all.to_csv(os.path.join(outdir,"parsed_lmcache_events.csv"), index=False)
    hit_all.to_csv(os.path.join(outdir,"parsed_prefix_hits.csv"), index=False)

    for metric, ycol, fname, ylabel in [
        ("TTFT","TTFT","01_overlay_ttft_6modes.png","TTFT (s)"),
        ("TBT_p95","TBT_p95","02_overlay_tbt_p95_6modes.png","TBT_p95 (s)"),
        ("TTLT","TTLT","03_overlay_ttlt_6modes.png","TTLT (s)"),
    ]:
        series={}
        for mode in order:
            df=req_all[req_all["mode"]==mode]
            df=df.iloc[np.argsort(np.asarray(df["t_finish"], dtype=float))]
            series[mode]=(df["t_finish"].to_numpy(), df[ycol].to_numpy())
        overlay_line_plot_multi(series, "Timeline (s from run start)", ylabel, f"{metric} timeline overlay (6 modes)", os.path.join(outdir,fname))

    series={}
    for mode in order:
        df=vram_all[vram_all["mode"]==mode]
        df=df.iloc[np.argsort(np.asarray(df["t"], dtype=float))]
        series[mode]=(df["t"].to_numpy(), df["vram_gb"].to_numpy())
    overlay_line_plot_multi(series, "Timeline (s from run start)", "VRAM usage (GB)", "VRAM usage timeline overlay (6 modes)", os.path.join(outdir,"04_overlay_vram_6modes.png"))

    summary_rows=[]
    for mode in order:
        req_df=req_all[req_all["mode"]==mode]
        req_df=req_df.iloc[np.argsort(np.asarray(req_df["t_finish"], dtype=float))]
        hit_df=hit_all[hit_all["mode"]==mode]
        hit_df=hit_df.iloc[np.argsort(np.asarray(hit_df["t"], dtype=float))]
        summary_rows.append({"mode":mode, **compute_summary_metrics(req_df), **compute_hit_metrics(hit_df)})
    summary_df=pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(outdir,"summary_metrics.csv"), index=False)

    bar_specs = [
        ("mean of TTFT","TTFT (s)","TTFT_mean","05_mean_of_ttft_6modes.png"),
        ("p95 of TTFT","TTFT (s)","TTFT_p95","06_p95_of_ttft_6modes.png"),
        ("mean of TBT_p95 (per-request)","TBT_p95 (s)","TBTp95_mean","07_mean_of_tbt_p95_per-request_6modes.png"),
        ("p95 of TBT_p95 (per-request)","TBT_p95 (s)","TBTp95_p95","08_p95_of_tbt_p95_per-request_6modes.png"),
        ("mean of estimated TBT","Estimated TBT (s)","estTBT_mean","09_mean_of_estimated_tbt_6modes.png"),
        ("p95 of estimated TBT","Estimated TBT (s)","estTBT_p95","10_p95_of_estimated_tbt_6modes.png"),
        ("mean of TTLT","TTLT (s)","TTLT_mean","11_mean_of_ttlt_6modes.png"),
        ("p95 of TTLT","TTLT (s)","TTLT_p95","12_p95_of_ttlt_6modes.png"),
        ("processed request count","Processed requests (count)","processed_request_count","13_processed_request_count_6modes.png"),
    ]
    for title,ylabel,col,fname in bar_specs:
        values_by_mode={m: summary_df.loc[summary_df["mode"]==m, col].iloc[0] for m in order}
        bar_multi(title,ylabel,values_by_mode,os.path.join(outdir,fname))

    hit_rate_pct_by_mode = {
        m: (
            summary_df.loc[summary_df["mode"]==m, "prefix_hit_rate"].iloc[0] * 100.0
            if pd.notna(summary_df.loc[summary_df["mode"]==m, "prefix_hit_rate"].iloc[0])
            else np.nan
        )
        for m in order
    }
    bar_multi(
        "prefix hit rate",
        "Hit rate (%)",
        hit_rate_pct_by_mode,
        os.path.join(outdir, "14_prefix_hit_rate_6modes.png"),
    )

    extra_idx = 15
    subset_metric_specs = [
        ("TTFT", "TTFT", "TTFT (s)", "ttft"),
        ("TBT_p95", "TBT_p95", "TBT_p95 (s)", "tbt_p95"),
        ("TTLT", "TTLT", "TTLT (s)", "ttlt"),
    ]
    subset_groups = [
        ("2modes_dram", "2 modes: native_dram vs cachegen_dram", ["native_dram", "cachegen_dram"]),
        ("2modes_disk", "2 modes: native_disk vs cachegen_disk", ["native_disk", "cachegen_disk"]),
        ("2modes_dram_disk", "2 modes: native_dram_disk vs cachegen_dram_disk", ["native_dram_disk", "cachegen_dram_disk"]),
        ("3modes_native_only", "3 modes: native_dram vs native_disk vs native_dram_disk", ["native_dram", "native_disk", "native_dram_disk"]),
        ("3modes_cachegen_only", "3 modes: cachegen_dram vs cachegen_disk vs cachegen_dram_disk", ["cachegen_dram", "cachegen_disk", "cachegen_dram_disk"]),
    ]

    for group_slug, group_title, modes in subset_groups:
        for metric_label, ycol, ylabel, metric_slug in subset_metric_specs:
            series = {}
            for mode in modes:
                df = req_all[req_all["mode"] == mode]
                df = df.iloc[np.argsort(np.asarray(df["t_finish"], dtype=float))]
                series[mode] = (df["t_finish"].to_numpy(), df[ycol].to_numpy())
            overlay_line_plot_multi(
                series,
                "Timeline (s from run start)",
                ylabel,
                f"{metric_label} timeline overlay ({group_title})",
                os.path.join(outdir, f"{extra_idx:02d}_overlay_{metric_slug}_{group_slug}.png"),
            )
            extra_idx += 1

        series = {}
        for mode in modes:
            df = vram_all[vram_all["mode"] == mode]
            df = df.iloc[np.argsort(np.asarray(df["t"], dtype=float))]
            series[mode] = (df["t"].to_numpy(), df["vram_gb"].to_numpy())
        overlay_line_plot_multi(
            series,
            "Timeline (s from run start)",
            "VRAM usage (GB)",
            f"VRAM usage timeline overlay ({group_title})",
            os.path.join(outdir, f"{extra_idx:02d}_overlay_vram_{group_slug}.png"),
        )
        extra_idx += 1

    hit_overlay_specs = [
        (
            "6modes",
            "6 modes",
            order,
            "overlay_hit_rate_6modes.png",
        ),
        (
            "2modes_dram",
            "2 modes: native_dram vs cachegen_dram",
            ["native_dram", "cachegen_dram"],
            "overlay_hit_rate_2modes_dram.png",
        ),
        (
            "2modes_disk",
            "2 modes: native_disk vs cachegen_disk",
            ["native_disk", "cachegen_disk"],
            "overlay_hit_rate_2modes_disk.png",
        ),
        (
            "2modes_dram_disk",
            "2 modes: native_dram_disk vs cachegen_dram_disk",
            ["native_dram_disk", "cachegen_dram_disk"],
            "overlay_hit_rate_2modes_dram_disk.png",
        ),
        (
            "3modes_native_only",
            "3 modes: native_dram vs native_disk vs native_dram_disk",
            ["native_dram", "native_disk", "native_dram_disk"],
            "overlay_hit_rate_3modes_native_only.png",
        ),
        (
            "3modes_cachegen_only",
            "3 modes: cachegen_dram vs cachegen_disk vs cachegen_dram_disk",
            ["cachegen_dram", "cachegen_disk", "cachegen_dram_disk"],
            "overlay_hit_rate_3modes_cachegen_only.png",
        ),
    ]

    for _slug, group_title, modes, fname in hit_overlay_specs:
        series = {}
        for mode in modes:
            df = cast(pd.DataFrame, hit_all.loc[hit_all["mode"] == mode, :])
            x, y = _hit_rate_timeline_series(df)
            if len(x):
                series[mode] = (x, y)

        out_path = os.path.join(outdir, f"{extra_idx:02d}_{fname}")
        title = f"Hit rate timeline overlay ({group_title})"
        if series:
            overlay_line_plot_multi(
                series,
                "Timeline (s from run start)",
                "Hit rate (%)",
                title,
                out_path,
            )
        else:
            _save_blank_overlay(
                "Timeline (s from run start)",
                "Hit rate (%)",
                title,
                out_path,
            )
        extra_idx += 1

    print("[DONE]", os.path.abspath(outdir))

if __name__=="__main__":
    main()
