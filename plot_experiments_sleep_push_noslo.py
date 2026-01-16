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

PUSH_SPEED = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
SLEEP_ORDER = [0.01, 0.05, 0.1, 0.5, 1.0]


def parse_sleep_from_name(path: Path) -> float | None:
    # e.g., timsort_hol_sleep_0.1_20260115_184128.csv
    m = re.search(r"sleep_(\d+(?:\.\d+)?)", path.name)
    return float(m.group(1)) if m else None

def parse_push_from_name(path: Path) -> float | None:
    m = re.search(r"pushint_(\d+(?:\.\d+)?)", path.name)
    return float(m.group(1)) if m else None

def lighten(color: str, amount: float = 0.6) -> Tuple[float, float, float]:
    """
    amount: 0..1 (higher -> lighter)
    """
    r, g, b = mcolors.to_rgb(color)
    return (1 - (1 - r) * amount, 1 - (1 - g) * amount, 1 - (1 - b) * amount)

def build_color_map_for_push(pushes: List[float]) -> Dict[float, str]:
    cmap = plt.get_cmap("tab10")
    colors = [mcolors.to_hex(cmap(i)) for i in range(10)]
    out: Dict[float, str] = {}
    for i, p in enumerate(sorted(pushes, key=lambda x: PUSH_SPEED.index(x) if x in PUSH_SPEED else x)):
        out[p] = colors[i % len(colors)]
    return out

def build_color_map_for_push(pushes: List[float]) -> Dict[float, str]:
    cmap = plt.get_cmap("tab10")
    colors = [mcolors.to_hex(cmap(i)) for i in range(10)]
    out: Dict[float, str] = {}
    for i, p in enumerate(sorted(pushes, key=lambda x: PUSH_SPEED.index(x) if x in PUSH_SPEED else x)):
        out[p] = colors[i % len(colors)]
    return out

def load_csvs(paths: List[Path]) -> Dict[float, Dict[float, pd.DataFrame]]:
    data: Dict[float, Dict[float, pd.DataFrame]] = {}
    for p in paths:
        sleep = parse_sleep_from_name(p)
        push = parse_push_from_name(p)
        if sleep is None or push is None:
            print(f"[WARN] sleep/push 값을 파일명에서 못 찾음: {p.name} (skip)")
            continue

        df = pd.read_csv(p).reset_index(drop=True)
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
    for push, df in sorted(data.items(), key=lambda kv: PUSH_SPEED.index(kv[0]) if kv[0] in PUSH_SPEED else kv[0]):
        if ycol not in df.columns:
            print(f"[WARN] {sleep=} 파일에 컬럼 없음: {ycol} (skip)")
            continue
        x = df.index
        y = df[ycol]
        label = f"push={push}"
        c = colors[push]
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

    for push, df in sorted(data.items(), key=lambda kv: PUSH_SPEED.index(kv[0]) if kv[0] in PUSH_SPEED else kv[0]):
        missing = [c for c in ["prompt_tok", "out_tok"] if c not in df.columns]
        if missing:
            print(f"[WARN] {sleep=} 파일에 컬럼 없음: {missing} (skip)")
            continue

        base = colors[push]
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

    # NOTE: load_csvs() 는 이제 Dict[sleep][push] = df 형태를 반환해야 함
    data = load_csvs(paths)
    if not data:
        raise SystemExit("No valid CSVs with sleep_*/pushint_* naming were loaded.")

    outdir = Path(args.outdir)

    # sleep별 폴더에서 push_speed(7개) 비교 그래프 생성
    sleeps = sorted(data.keys(), key=lambda x: SLEEP_ORDER.index(x) if x in SLEEP_ORDER else x)

    for sleep in sleeps:
        sleep_str = str(sleep)  # "0.01", "0.1", "1.0" 등
        sleep_dir = outdir / f"sleep_{sleep_str}"
        sleep_dir.mkdir(parents=True, exist_ok=True)

        # Dict[push] -> df
        sleep_data = data.get(sleep, {})
        if not sleep_data:
            print(f"[WARN] sleep={sleep_str} 에 해당하는 데이터가 없음 (skip)")
            continue

        pushes = sorted(
            sleep_data.keys(),
            key=lambda x: PUSH_SPEED.index(x) if x in PUSH_SPEED else x
        )

        # NOTE: build_color_map_for_push() 는 push 기준으로 고유 색상 맵을 만들어야 함
        colors = build_color_map_for_push(pushes)

        # 1) success_rate_pct (sleep 고정, push 7개 한 그래프)
        plot_multi_line(
            sleep_data, colors,
            ycol="success_rate_pct",
            title=f"(1) Success Rate (%) vs CSV Row Index | sleep={sleep_str}",
            ylabel="success_rate_pct (%)",
            out_path=sleep_dir / "1_success_rate_pct.png",
            kind="line",
        )

        # 2) execution_time
        plot_multi_line(
            sleep_data, colors,
            ycol="execution_time",
            title=f"(2) Execution Time vs CSV Row Index | sleep={sleep_str}",
            ylabel="execution_time (s)",
            out_path=sleep_dir / "2_execution_time.png",
            kind="line",
        )

        # 3) prompt_tok + out_tok stacked bars (push별 7개 그래프)
        plot_prompt_out_breakdown_per_file(
            sleep_data, colors,
            out_dir=sleep_dir / "3_token_breakdown",
        )

        # 4) wait_time
        plot_multi_line(
            sleep_data, colors,
            ycol="wait_time",
            title=f"(4) Wait Time vs CSV Row Index | sleep={sleep_str}",
            ylabel="wait_time (s)",
            out_path=sleep_dir / "4_wait_time.png",
            kind="line",
        )

        # 5) violation (scatter)
        plot_multi_line(
            sleep_data, colors,
            ycol="violation",
            title=f"(5) Violation vs CSV Row Index | sleep={sleep_str}",
            ylabel="violation",
            out_path=sleep_dir / "5_violation.png",
            kind="scatter",
        )

    print("\nDone.")


if __name__ == "__main__":
    main()

