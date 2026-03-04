    #!/usr/bin/env python3
    """six_mode_plot.py
    Generates the same plot style as red_black.py, but for 6 modes:
      native_dram, native_disk, native_dram_disk, cachegen_dram, cachegen_disk, cachegen_dram_disk

    Edit the 'files' dict at the top if your filenames differ.
    Output: ./six_mode_graph_output/*.png + *.csv
    """
    import os, re, json, math
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    files = {
  "native_dram": {
    "run": "/mnt/data/run_native_dram.log",
    "vram": "/mnt/data/native_dram_multi_req_lmcache_vram.log",
    "vram_kind": "raw"
  },
  "native_disk": {
    "run": "/mnt/data/run_native_disk.log",
    "vram": "/mnt/data/native_disk_extract_multi_req_lmcache_vram.log.json",
    "vram_kind": "json"
  },
  "native_dram_disk": {
    "run": "/mnt/data/run_native_dram_disk.log",
    "vram": "/mnt/data/native_dram_disk_extract_multi_req_lmcache_vram.log.json",
    "vram_kind": "json"
  },
  "cachegen_dram": {
    "run": "/mnt/data/run_cachegen_dram.log",
    "vram": "/mnt/data/cachegen_dram_multi_req_lmcache_vram.log",
    "vram_kind": "raw"
  },
  "cachegen_disk": {
    "run": "/mnt/data/run_cachegen_disk.log",
    "vram": "/mnt/data/cachegen_disk_extract_multi_req_lmcache_vram.log.json",
    "vram_kind": "json"
  },
  "cachegen_dram_disk": {
    "run": "/mnt/data/run_cachegen_dram_disk.log",
    "vram": "/mnt/data/cachegen_dram_disk_extract_multi_req_lmcache_vram.log.json",
    "vram_kind": "json"
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

    def _to_float_or_nan(x: str):
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
        srows, erows = [], []
        with open(path,"r",errors="ignore") as f:
            for line in f:
                sm = SAMPLER_RE.search(line)
                if sm:
                    srows.append({"mode":mode,"ts":float(sm.group(1)),"vram_gb":float(sm.group(2))})
                    continue
                em = EVENT_RE.search(line)
                if em:
                    erows.append({"mode":mode,"ts":float(em.group(1)),"event":em.group(2).lower()})
        s_df = pd.DataFrame(srows) if srows else pd.DataFrame(columns=["mode","ts","vram_gb"])
        e_df = pd.DataFrame(erows) if erows else pd.DataFrame(columns=["mode","ts","event"])
        return s_df, e_df

    def parse_vram_json(path: str, mode: str):
        with open(path,"r") as f: data=json.load(f)
        srows, erows = [], []
        for ev in data.get("client_events", []):
            if ev.get("event")=="periodic_sample":
                srows.append({"mode":mode,"ts":float(ev["ts"]),"vram_gb":float(ev["vram_gb"])})
        for ev in data.get("lmcache_events", []):
            e=str(ev.get("event","")).lower()
            if e in ("serialize","deserialize"):
                erows.append({"mode":mode,"ts":float(ev["ts"]),"event":e})
        s_df = pd.DataFrame(srows) if srows else pd.DataFrame(columns=["mode","ts","vram_gb"])
        e_df = pd.DataFrame(erows) if erows else pd.DataFrame(columns=["mode","ts","event"])
        return s_df, e_df

    def normalize_timelines(req_all: pd.DataFrame, vram_all: pd.DataFrame, evt_all: pd.DataFrame):
        mode_t0={}
        modes=sorted(set(req_all["mode"].dropna().tolist()))
        for mode in modes:
            candidates=[]
            r=req_all[req_all["mode"]==mode]
            if not r.empty: candidates.append(float(r["insertion_ts"].min()))
            v=vram_all[vram_all["mode"]==mode] if not vram_all.empty else pd.DataFrame()
            if not v.empty: candidates.append(float(v["ts"].min()))
            e=evt_all[evt_all["mode"]==mode] if not evt_all.empty else pd.DataFrame()
            if not e.empty: candidates.append(float(e["ts"].min()))
            mode_t0[mode]=min(candidates) if candidates else 0.0
        req_all=req_all.copy(); vram_all=vram_all.copy(); evt_all=evt_all.copy()
        req_all["t_finish"]=req_all.apply(lambda r: r["finished_ts"]-mode_t0[r["mode"]], axis=1)
        req_all["t_insert"]=req_all.apply(lambda r: r["insertion_ts"]-mode_t0[r["mode"]], axis=1)
        if not vram_all.empty:
            vram_all["t"]=vram_all.apply(lambda r: r["ts"]-mode_t0[r["mode"]], axis=1)
        if not evt_all.empty:
            evt_all["t"]=evt_all.apply(lambda r: r["ts"]-mode_t0[r["mode"]], axis=1)
        return req_all, vram_all, evt_all

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
        x=np.arange(len(labels))
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

    def main():
        outdir="six_mode_graph_output"
        os.makedirs(outdir, exist_ok=True)

        req_dfs=[]; vram_dfs=[]; evt_dfs=[]
        for mode in order:
            req_dfs.append(parse_run_log(files[mode]["run"], mode))
            if files[mode]["vram_kind"]=="raw":
                vram, evt = parse_vram_raw_log(files[mode]["vram"], mode)
            else:
                vram, evt = parse_vram_json(files[mode]["vram"], mode)
            vram_dfs.append(vram); evt_dfs.append(evt)

        req_all=pd.concat(req_dfs, ignore_index=True)
        vram_all=pd.concat(vram_dfs, ignore_index=True)
        evt_all=pd.concat(evt_dfs, ignore_index=True)
        req_all, vram_all, evt_all = normalize_timelines(req_all, vram_all, evt_all)

        req_all.to_csv(os.path.join(outdir,"parsed_request_metrics.csv"), index=False)
        vram_all.to_csv(os.path.join(outdir,"parsed_vram_samples.csv"), index=False)
        evt_all.to_csv(os.path.join(outdir,"parsed_lmcache_events.csv"), index=False)

        for metric, ycol, fname, ylabel in [
            ("TTFT","TTFT","01_overlay_ttft_6modes.png","TTFT (s)"),
            ("TBT_p95","TBT_p95","02_overlay_tbt_p95_6modes.png","TBT_p95 (s)"),
            ("TTLT","TTLT","03_overlay_ttlt_6modes.png","TTLT (s)"),
        ]:
            series={}
            for mode in order:
                df=req_all[req_all["mode"]==mode].sort_values("t_finish")
                series[mode]=(df["t_finish"].to_numpy(), df[ycol].to_numpy())
            overlay_line_plot_multi(series, "Timeline (s from run start)", ylabel, f"{metric} timeline overlay (6 modes)", os.path.join(outdir,fname))

        series={}
        for mode in order:
            df=vram_all[vram_all["mode"]==mode].sort_values("t")
            series[mode]=(df["t"].to_numpy(), df["vram_gb"].to_numpy())
        overlay_line_plot_multi(series, "Timeline (s from run start)", "VRAM usage (GB)", "VRAM usage timeline overlay (6 modes)", os.path.join(outdir,"04_overlay_vram_6modes.png"))

        summary_rows=[]
        for mode in order:
            df=req_all[req_all["mode"]==mode].sort_values("t_finish")
            summary_rows.append({"mode":mode, **compute_summary_metrics(df)})
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

        print("[DONE]", os.path.abspath(outdir))

    if __name__=="__main__":
        main()
