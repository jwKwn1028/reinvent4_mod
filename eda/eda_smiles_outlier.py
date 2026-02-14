"""
clean_mp.py

Reads a CSV with columns: smiles, mp (Molecular Property)
- Drops empty "Unnamed:*" columns
- Strips/validates SMILES (simple heuristics)
- Converts mp to numeric
- Drops exact duplicates on (smiles, mp_raw)
- Drops non-numeric mp
- Removes extreme mp outliers using IQR*3 rule
- Writes:
  - cleaned CSV (smiles, mp)
  - removed rows CSV with reason codes

Usage:
  python clean_mp.py --input mp.csv --out-clean mp_cleaned.csv --out-removed mp_removed_rows.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ALLOWED_SMILES_RE = re.compile(r"^[A-Za-z0-9\[\]\(\)=#@+\-\\/%.:*]+$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input CSV path (must contain smiles, mp).")
    p.add_argument("--out-clean", default="mp_cleaned.csv", help="Output cleaned CSV path.")
    p.add_argument("--out-removed", default="mp_removed_rows.csv", help="Output removed rows audit CSV path.")
    p.add_argument("--iqr-mult", type=float, default=3.0, help="IQR multiplier for outlier filtering (default 3.0).")
    p.add_argument("--smiles-min-len", type=int, default=2, help="Minimum SMILES string length (default 2).")
    p.add_argument("--low-memory", action="store_true", help="Use pandas low_memory=True (not recommended).")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"ERROR: input file not found: {in_path}", file=sys.stderr)
        return 2

    # Read
    df = pd.read_csv(in_path, low_memory=args.low_memory)

    # Drop near-empty "Unnamed:*" columns (common artifact from index columns)
    unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols, errors="ignore")

    # Basic required columns
    for col in ("smiles", "mp"):
        if col not in df.columns:
            print(f"ERROR: required column '{col}' not found. Columns: {list(df.columns)}", file=sys.stderr)
            return 2

    # Normalize SMILES and mp
    df["smiles"] = df["smiles"].astype("string").str.strip()
    df = df[df["smiles"].notna() & (df["smiles"] != "")].copy()

    # Preserve raw mp for duplicate detection and auditing
    df["mp_raw"] = df["mp"]
    df["mp"] = pd.to_numeric(df["mp"], errors="coerce")

    # Precompute SMILES length
    df["smiles_len"] = df["smiles"].str.len()

    removed_parts: list[pd.DataFrame] = []

    # 1) Drop exact duplicates on (smiles, mp_raw)
    dup_mask = df.duplicated(subset=["smiles", "mp_raw"])
    if dup_mask.any():
        removed_parts.append(df.loc[dup_mask, ["smiles", "mp_raw", "mp"]].assign(
            reason="exact_duplicate(smiles,mp_raw)"
        ))
        df = df.loc[~dup_mask].copy()

    # 2) Drop obviously bad SMILES (heuristics)
    bad_smiles_mask = (df["smiles_len"] < args.smiles_min_len) | (~df["smiles"].astype(str).str.match(ALLOWED_SMILES_RE))
    if bad_smiles_mask.any():
        removed_parts.append(df.loc[bad_smiles_mask, ["smiles", "mp_raw", "mp"]].assign(
            reason=f"bad_smiles(len<{args.smiles_min_len}_or_invalid_chars)"
        ))
        df = df.loc[~bad_smiles_mask].copy()

    # 3) Drop non-numeric mp
    non_numeric_mask = df["mp"].isna()
    if non_numeric_mask.any():
        removed_parts.append(df.loc[non_numeric_mask, ["smiles", "mp_raw", "mp"]].assign(
            reason="non_numeric_mp"
        ))
        df = df.loc[~non_numeric_mask].copy()

    # 4) IQR (InterQuartile Range) outlier filtering (on remaining numeric mp)
    if len(df) == 0:
        print("WARNING: no rows remain after basic cleaning.", file=sys.stderr)
    else:
        q1, q3 = df["mp"].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = float(q1 - args.iqr_mult * iqr)
        upper = float(q3 + args.iqr_mult * iqr)

        outlier_mask = (df["mp"] < lower) | (df["mp"] > upper)
        if outlier_mask.any():
            removed_parts.append(df.loc[outlier_mask, ["smiles", "mp_raw", "mp"]].assign(
                reason=f"mp_outlier_IQR{args.iqr_mult:g}(keep_[{lower:.1f},{upper:.1f}])"
            ))
            df = df.loc[~outlier_mask].copy()

    # Build removed audit table
    removed_df = pd.concat(removed_parts, ignore_index=True) if removed_parts else pd.DataFrame(
        columns=["smiles", "mp_raw", "mp", "reason"]
    )

    # Write outputs
    out_clean = Path(args.out_clean)
    out_removed = Path(args.out_removed)

    out_clean.parent.mkdir(parents=True, exist_ok=True)
    out_removed.parent.mkdir(parents=True, exist_ok=True)

    df[["smiles", "mp"]].to_csv(out_clean, index=False)
    removed_df.to_csv(out_removed, index=False)

    # Console summary
    print("=== Cleaning summary ===")
    print(f"Input:        {in_path}")
    print(f"Cleaned out:  {out_clean}  (rows={len(df):,})")
    print(f"Removed out:  {out_removed} (rows={len(removed_df):,})")
    if len(df) > 0:
        print(f"mp range:     {df['mp'].min():.3g} .. {df['mp'].max():.3g}")
        print(f"mp median:    {df['mp'].median():.3g}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
