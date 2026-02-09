#!/usr/bin/env python3
"""
analysis.py

Generic step-wise analysis for REINVENT rollout CSVs.

Defaults:
- Finds "(raw)" columns automatically and computes step-wise means.
- Optionally includes "Score" if present.
- Filters SMILES_state == 0 if that column exists (unless disabled).
- Drops the final step (max step) unless --keep-last-step is set.

Outputs:
- filtered row-level CSV
- step-level means CSV
- plots for each selected value column
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import List

import pandas as pd
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", required=True, help="Input CSV path")

    # If user doesn't pass --outdir, write into the script directory / "stat"
    p.add_argument("--outdir","-o", default=None, help="Output directory for plots/summary")

    # Output filenames
    p.add_argument("--summary-csv", default="step_means.csv", help="Filename for step-mean summary CSV")
    p.add_argument("--filtered-csv", default="filtered_rows.csv", help="Filename for filtered row-level CSV")

    p.add_argument("--step-col", default=None, help="Step column name (case-insensitive). Defaults to 'step'.")
    p.add_argument(
        "--smiles-state-col",
        default="SMILES_state",
        help="SMILES_state column name (case-insensitive).",
    )
    p.add_argument(
        "--score-col",
        default="Score",
        help="Score column name (case-insensitive).",
    )

    p.add_argument(
        "--value-cols",
        default=None,
        help="Comma-separated list of columns to summarize; overrides --raw-pattern.",
    )
    p.add_argument(
        "--raw-pattern",
        default=r"\(raw\)",
        help="Regex for auto-selecting raw columns (case-insensitive).",
    )
    p.add_argument(
        "--include-score",
        action="store_true",
        help="Include the score column in summary/plots if found.",
    )

    p.add_argument("--no-smiles-state-filter", action="store_true", help="Do not filter SMILES_state == 0")
    p.add_argument("--drop-score-zero", action="store_true", help="Drop rows with Score == 0 (if present)")
    p.add_argument("--keep-last-step", action="store_true", help="Keep rows with step == max(step)")
    p.add_argument("--last-step", type=int, default=None, help="Explicit last step to drop (overrides max(step))")
    return p.parse_args()


def _find_column(columns: List[str], target: str | None) -> str | None:
    if not target:
        return None
    for col in columns:
        if col.lower() == target.lower():
            return col
    return None


def _normalize_value_cols(columns: List[str], raw: str) -> List[str]:
    raw_cols = [c for c in columns if re.search(raw, c, flags=re.IGNORECASE)]
    return raw_cols


def _sanitize_name(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return sanitized.lower() or "value"


def main() -> int:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    outdir = Path(args.outdir).expanduser() if args.outdir else (script_dir / "out")
    outdir.mkdir(parents=True, exist_ok=True)

    in_path = Path(args.input).expanduser()
    df = pd.read_csv(in_path, low_memory=False)

    step_col = _find_column(df.columns.tolist(), args.step_col or "step")
    if not step_col:
        raise ValueError(f"Could not find a step column. Columns: {list(df.columns)}")

    smiles_state_col = _find_column(df.columns.tolist(), args.smiles_state_col)
    score_col = _find_column(df.columns.tolist(), args.score_col)

    if args.value_cols:
        requested = [c.strip() for c in args.value_cols.split(",") if c.strip()]
        resolved = []
        for c in requested:
            match = _find_column(df.columns.tolist(), c)
            if not match:
                raise ValueError(f"Requested column not found: '{c}'")
            resolved.append(match)
        value_cols = resolved
    else:
        value_cols = _normalize_value_cols(df.columns.tolist(), args.raw_pattern)

    if args.include_score and score_col and score_col not in value_cols:
        value_cols = [score_col] + value_cols

    if not value_cols:
        raise ValueError(
            "No value columns found. Use --value-cols or adjust --raw-pattern."
        )

    numeric_cols = [step_col] + value_cols
    if score_col and score_col not in numeric_cols:
        numeric_cols.append(score_col)
    if smiles_state_col and smiles_state_col not in numeric_cols:
        numeric_cols.append(smiles_state_col)

    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if smiles_state_col and not args.no_smiles_state_filter:
        df = df[df[smiles_state_col].notna() & (df[smiles_state_col] != 0)].copy()

    if args.drop_score_zero and score_col:
        df = df[df[score_col].notna() & (df[score_col] != 0)].copy()

    if args.last_step is not None:
        last_step = int(args.last_step)
    elif not args.keep_last_step:
        last_step = int(df[step_col].max())
    else:
        last_step = None

    if last_step is not None:
        df = df[df[step_col] != last_step].copy()

    df = df[df[step_col].notna()].copy()

    filtered_path = outdir / args.filtered_csv
    df.to_csv(filtered_path, index=False)

    agg = {
        "n": ("SMILES", "size") if "SMILES" in df.columns else (step_col, "size"),
    }
    for col in value_cols:
        agg[f"mean_{col}"] = (col, "mean")

    step_means = (
        df.groupby(step_col, as_index=False)
        .agg(**agg)
        .sort_values(step_col)
        .reset_index(drop=True)
    )

    summary_path = outdir / args.summary_csv
    step_means.to_csv(summary_path, index=False)

    def plot_one(ycol: str, ylabel: str, fname: str) -> None:
        plt.figure()
        plt.plot(step_means[step_col], step_means[ycol])
        plt.xlabel(step_col)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(outdir / fname, dpi=200)
        plt.close()

    for col in value_cols:
        mean_col = f"mean_{col}"
        fname = f"mean_{_sanitize_name(col)}_vs_{_sanitize_name(step_col)}.png"
        plot_one(mean_col, f"Mean {col}", fname)

    print(f"[OK] Filtered rows saved: {filtered_path}")
    print(f"[OK] Step means saved:    {summary_path}")
    print(f"[OK] Plots saved in:     {outdir.resolve()}")
    print(f"[OK] Value columns used: {value_cols}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
