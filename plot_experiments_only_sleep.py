#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot QLM experiment CSVs.

Expected columns (from log2csv.py):
- success_rate_pct, execution_time, wait_time, violation, prompt_tok, out_tok
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


SLEEP_ORDER = [0.01, 0.05, 0.1, 0.5, 1.0]


def parse_sleep_from_name(path: Path) -> float | None:
    # e.g., timsort_hol_sleep_0.1_20260115_184128.csv
    m = re.search(r"sleep_(\d+(?:\.\d+)?)", path.name)
    return float(m.group(1)) if m else None


def lighten(color: str, amount: float = 0.6) -> Tuple[float, float, float]:
    """
    amount: 0..1 (higher -> lighter)
    """
    r, g, b = mcolors.to_rgb(color)
    return (1 - (1 - r) * amount, 1 - (1 - g) * amount, 1 - (1 - b) * amount)


def build_color_map(sleeps: List[float]) -> Dict[float, str]:
    # stable, distinct colors
    cmap = plt.get_cmap("tab10")
    colors = [mcolors.to_hex(cmap(i)) for i in range(10)]
    out: Dict[float, str] = {}
    for i, s in enumerate(sorted(sleeps, key=lambda x: SLEEP_ORDER.index(x) if x in SLEEP_ORDER else x)):
        out[s] = colors[i % len(colors)]
    return out


def load_csvs(paths: List[Path]) -> Dict[float, pd.DataFrame]:
    data: Dict[float, pd.DataFrame] = {}
    for p in paths:
        sleep = parse_sleep_from_name(p)
        if sleep is None:
            print(f"[WARN] sleep 값을 파일명에서 못 찾음: {p.name} (skip)")
            continue
        df = pd.read_csv(p)
        # enforce index order = csv row order
        df = df.reset_index(drop=True)
        data[sleep] = df
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
    for sleep, df in sorted(data.items(), key=lambda kv: SLEEP_ORDER.index(kv[0]) if kv[0] in SLEEP_ORDER else kv[0]):
        if ycol not in df.columns:
            print(f"[WARN] {sleep=} 파일에 컬럼 없음: {ycol} (skip)")
            continue
        x = df.index
        y = df[ycol]
        label = f"sleep={sleep}"
        c = colors[sleep]
        if kind == "scatter":
            plt.scatter(x, y, s=10, alpha=0.7, color=c, label=label)
        else:
            plt.plot(x, y, linewidth=1.8, color=c, label=label)

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

    for sleep, df in sorted(data.items(), key=lambda kv: SLEEP_ORDER.index(kv[0]) if kv[0] in SLEEP_ORDER else kv[0]):
        missing = [c for c in ["prompt_tok", "out_tok"] if c not in df.columns]
        if missing:
            print(f"[WARN] {sleep=} 파일에 컬럼 없음: {missing} (skip)")
            continue

        base = colors[sleep]
        prompt_c = lighten(base, 0.35)
        out_c = base

        x = df.index
        prompt = df["prompt_tok"]
        outtok = df["out_tok"]

        plt.figure(figsize=(12, 5))
        plt.bar(x, prompt, color=prompt_c, label="prompt_tok")
        plt.bar(x, outtok, bottom=prompt, color=out_c, label="out_tok")

        plt.title(f"Token breakdown (stacked) | sleep={sleep}")
        plt.xlabel("CSV row index (order)")
        plt.ylabel("tokens")
        plt.grid(True, axis="y", alpha=0.25)
        plt.legend()
        plt.tight_layout()

        out_path = out_dir / f"3_token_breakdown_sleep_{sleep}.png"
        plt.savefig(out_path, dpi=220)
        plt.close()
        print(f"[OK] saved: {out_path}")


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

    data = load_csvs(paths)
    if not data:
        raise SystemExit("No valid CSVs with sleep_*.csv naming were loaded.")

    sleeps = sorted(data.keys(), key=lambda x: SLEEP_ORDER.index(x) if x in SLEEP_ORDER else x)
    colors = build_color_map(sleeps)

    outdir = Path(args.outdir)

    # 1) success_rate_pct (all files, one graph)
    plot_multi_line(
        data, colors,
        ycol="success_rate_pct",
        title="(1) Success Rate (%) vs CSV Row Index",
        ylabel="success_rate_pct (%)",
        out_path=outdir / "1_success_rate_pct.png",
        kind="line",
    )

    # 2) execution_time (all files, one graph)
    plot_multi_line(
        data, colors,
        ycol="execution_time",
        title="(2) Execution Time vs CSV Row Index",
        ylabel="execution_time (s)",
        out_path=outdir / "2_execution_time.png",
        kind="line",
    )

    # 3) prompt_tok + out_tok stacked bars (5 graphs, one per file)
    plot_prompt_out_breakdown_per_file(
        data, colors,
        out_dir=outdir / "3_token_breakdown",
    )

    # 4) wait_time (all files, one graph)
    plot_multi_line(
        data, colors,
        ycol="wait_time",
        title="(4) Wait Time vs CSV Row Index",
        ylabel="wait_time (s)",
        out_path=outdir / "4_wait_time.png",
        kind="line",
    )

    # 5) violation (all files, one graph)  -> scatter가 보통 보기 좋음
    plot_multi_line(
        data, colors,
        ycol="violation",
        title="(5) Violation vs CSV Row Index",
        ylabel="violation",
        out_path=outdir / "5_violation.png",
        kind="scatter",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()

