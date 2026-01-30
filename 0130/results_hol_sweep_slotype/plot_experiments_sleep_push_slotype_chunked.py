#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_experiments_sleep_push_slotype_chunked.py

기존: sleep별로 (pushint 라인 7개) 그래프 생성
추가: 파일명에 포함된 chunked_{0|1}, mb{N} 기준으로 먼저 분리해서
     outdir/chunked_{c}/mb{mb}/sleep_{sleep}/... 형태로 저장.

Expected filename examples:
- timsort_hol_sleep_0.0001_pushint_0.01_mb80_chunked_1_20260127_203539.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

PUSH_SPEED = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
SLEEP_ORDER = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0]


def parse_sleep_from_name(path: Path) -> Optional[float]:
    m = re.search(r"sleep_(\d+(?:\.\d+)?)", path.name)
    return float(m.group(1)) if m else None

def parse_push_from_name(path: Path) -> Optional[float]:
    m = re.search(r"pushint_(\d+(?:\.\d+)?)", path.name)
    return float(m.group(1)) if m else None

def parse_mb_from_name(path: Path) -> Optional[int]:
    m = re.search(r"_mb(\d+)", path.name)
    return int(m.group(1)) if m else None

def parse_chunked_from_name(path: Path) -> Optional[int]:
    m = re.search(r"_chunked_(\d+)", path.name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def lighten(color: str, amount: float = 0.6) -> Tuple[float, float, float]:
    r, g, b = mcolors.to_rgb(color)
    return (1 - (1 - r) * amount, 1 - (1 - g) * amount, 1 - (1 - b) * amount)

def build_color_map_for_push(pushes: List[float]) -> Dict[float, str]:
    cmap = plt.get_cmap("tab10")
    colors = [mcolors.to_hex(cmap(i)) for i in range(10)]
    out: Dict[float, str] = {}
    # PUSH_SPEED에 있으면 그 순서대로 정렬
    def k(x: float) -> float:
        return PUSH_SPEED.index(x) if x in PUSH_SPEED else x
    for i, p in enumerate(sorted(pushes, key=k)):
        out[p] = colors[i % len(colors)]
    return out

def load_csvs(paths: List[Path]) -> Dict[float, Dict[float, pd.DataFrame]]:
    """Return Dict[sleep][push] = df (mb/chunked는 여기서 다루지 않음)."""
    data: Dict[float, Dict[float, pd.DataFrame]] = {}
    for p in paths:
        sleep = parse_sleep_from_name(p)
        push = parse_push_from_name(p)
        if sleep is None or push is None:
            print(f"[WARN] sleep/push 값을 파일명에서 못 찾음: {p.name} (skip)")
            continue

        df = pd.read_csv(p).reset_index(drop=True)
        # compat: old column name -> new
        if "ttlt" not in df.columns and "execution_time" in df.columns:
            df = df.rename(columns={"execution_time": "ttlt"})
        data.setdefault(sleep, {})[push] = df
    return data

def plot_multi_line(
    data: Dict[float, pd.DataFrame],
    colors: Dict[float, str],
    ycol: str,
    title: str,
    ylabel: str,
    out_path: Path,
    kind: str = "line",  # "line" | "scatter"
):
    plt.figure(figsize=(12, 5))
    def k(x: float) -> float:
        return PUSH_SPEED.index(x) if x in PUSH_SPEED else x

    for push, df in sorted(data.items(), key=lambda kv: k(kv[0])):
        if ycol not in df.columns:
            print(f"[WARN] push={push} 파일에 컬럼 없음: {ycol} (skip)")
            continue
        x = df.index
        y = df[ycol]
        label = f"push={push}"
        c = colors.get(push, "#1f77b4")
        if kind == "scatter":
            plt.scatter(x, y, s=5, alpha=0.7, color=c, label=label)
        else:
            plt.plot(x, y, linewidth=0.8, color=c, label=label)

    plt.title(title)
    plt.xlabel("CSV row index (order)")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=3, fontsize=9)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()
    print(f"[OK] saved: {out_path}")

def plot_prompt_out_breakdown_per_file(
    data: Dict[float, pd.DataFrame],
    colors: Dict[float, str],
    out_dir: Path,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    def k(x: float) -> float:
        return PUSH_SPEED.index(x) if x in PUSH_SPEED else x

    for push, df in sorted(data.items(), key=lambda kv: k(kv[0])):
        missing = [c for c in ["prompt_tok", "out_tok"] if c not in df.columns]
        if missing:
            print(f"[WARN] push={push} 파일에 컬럼 없음: {missing} (skip)")
            continue

        base = colors.get(push, "#1f77b4")
        prompt_c = lighten(base, 0.35)
        out_c = base

        x = df.index
        prompt = df["prompt_tok"]
        outtok = df["out_tok"]

        plt.figure(figsize=(12, 5))
        plt.bar(x, outtok, bottom=prompt, color=out_c, label="out_tok")
        plt.bar(x, prompt, color=prompt_c, label="prompt_tok")

        plt.title(f"Token breakdown (stacked) | push={push}")
        plt.xlabel("CSV row index (order)")
        plt.ylabel("tokens")
        plt.grid(True, axis="y", alpha=0.25)
        plt.legend()
        plt.tight_layout()

        out_path = out_dir / f"3_token_breakdown_push_{push}.png"
        plt.savefig(out_path, dpi=220)
        plt.close()
        print(f"[OK] saved: {out_path}")

def group_paths_by_chunked_mb(paths: List[Path]) -> Dict[Tuple[str, str], List[Path]]:
    """
    Return Dict[(chunked_label, mb_label)] -> list[Path]
    chunked_label: "chunked_0" | "chunked_1" | "chunked_unknown"
    mb_label: "mb20" | ... | "mb_unknown"
    """
    out: Dict[Tuple[str, str], List[Path]] = {}
    for p in paths:
        ch = parse_chunked_from_name(p)
        mb = parse_mb_from_name(p)
        ch_label = f"chunked_{ch}" if ch in (0, 1) else "chunked_unknown"
        mb_label = f"mb{mb}" if isinstance(mb, int) else "mb_unknown"
        out.setdefault((ch_label, mb_label), []).append(p)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "csvs",
        nargs="*",
        help="CSV paths (or directories). If empty, uses current directory *.csv",
    )
    ap.add_argument(
        "-o",
        "--outdir",
        default="plots",
        help="Output directory for PNGs (default: plots)",
    )
    args = ap.parse_args()

    # collect csv files
    paths: List[Path] = []
    if args.csvs:
        for s in args.csvs:
            p = Path(s)
            if p.is_dir():
                paths += sorted(p.glob("*.csv"))
            else:
                paths.append(p)
    else:
        paths = sorted(Path(".").glob("*.csv"))

    paths = [p for p in paths if p.exists() and p.suffix.lower() == ".csv"]
    if not paths:
        raise SystemExit("No CSV files found.")

    outdir = Path(args.outdir)

    groups = group_paths_by_chunked_mb(paths)
    if not groups:
        raise SystemExit("No CSV files matched for grouping.")

    for (ch_label, mb_label), gpaths in sorted(groups.items()):
        print(f"\n=== Group {ch_label}/{mb_label}: {len(gpaths)} files ===")
        data = load_csvs(gpaths)
        if not data:
            print(f"[WARN] no valid CSVs loaded for {ch_label}/{mb_label} (skip)")
            continue

        base_out = outdir / ch_label / mb_label

        sleeps = sorted(data.keys(), key=lambda x: SLEEP_ORDER.index(x) if x in SLEEP_ORDER else x)
        for sleep in sleeps:
            sleep_str = str(sleep)
            sleep_dir = base_out / f"sleep_{sleep_str}"
            sleep_dir.mkdir(parents=True, exist_ok=True)

            sleep_data = data.get(sleep, {})
            if not sleep_data:
                continue

            pushes = sorted(
                sleep_data.keys(),
                key=lambda x: PUSH_SPEED.index(x) if x in PUSH_SPEED else x
            )
            colors = build_color_map_for_push(pushes)

            plot_multi_line(
                sleep_data, colors,
                ycol="success_rate_pct",
                title=f"(1) Success Rate (%) vs CSV Row Index | {ch_label}/{mb_label} sleep={sleep_str}",
                ylabel="success_rate_pct (%)",
                out_path=sleep_dir / "1_success_rate_pct.png",
                kind="line",
            )

            plot_multi_line(
                sleep_data, colors,
                ycol="ttlt",
                title=f"(2) TTLT vs CSV Row Index | {ch_label}/{mb_label} sleep={sleep_str}",
                ylabel="ttlt (s)",
                out_path=sleep_dir / "2_ttlt.png",
                kind="line",
            )

            plot_prompt_out_breakdown_per_file(
                sleep_data, colors,
                out_dir=sleep_dir / "3_token_breakdown",
            )

            plot_multi_line(
                sleep_data, colors,
                ycol="wait_time",
                title=f"(4) Wait Time vs CSV Row Index | {ch_label}/{mb_label} sleep={sleep_str}",
                ylabel="wait_time (s)",
                out_path=sleep_dir / "4_wait_time.png",
                kind="line",
            )

            plot_multi_line(
                sleep_data, colors,
                ycol="violation",
                title=f"(5) Violation vs CSV Row Index | {ch_label}/{mb_label} sleep={sleep_str}",
                ylabel="violation",
                out_path=sleep_dir / "5_violation.png",
                kind="scatter",
            )

            plot_multi_line(
                sleep_data, colors,
                ycol="orig_slo",
                title=f"(6) SLO vs CSV Row Index | {ch_label}/{mb_label} sleep={sleep_str}",
                ylabel="SLO (sec)",
                out_path=sleep_dir / "6_slo.png",
                kind="line",
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
