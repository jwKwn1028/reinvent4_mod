#!/usr/bin/env python3
"""
filter_dedup.py

Combined script for filtering (lohc, bp, mp) and deduplication (inchi).

1. Filter Logic (from filter.py):
   - Keep rows where "LOHC (raw)" == 1.0 (configurable)
   - BP/MP consistency: (MP (raw) - 273.15) <= (BP (raw))
   - Optional: MP < threshold, BP > threshold
   - Sort by Score (descending)

2. Dedup Logic (from dedup.py):
   - Convert SMILES -> InChI
   - Deduplicate by InChI, keeping row with highest Score
   - Sort by Score (descending)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem

# ----- fixed original column names (from filter.py) -----
LOHC_RAW_COL = "LOHC (raw)"
BP_RAW_COL = "BP (raw)"   # Celsius
MP_RAW_COL = "MP (raw)"   # Kelvin


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Filter REINVENT log CSV by LOHC and BP/MP, then deduplicate by InChI."
    )
    # Common args
    p.add_argument("-i", "--input", required=True, help="Input CSV path")
    p.add_argument("-o", "--output", required=True, help="Output CSV path")

    # --- filter.py args ---
    p.add_argument(
        "--keep-missing-bpmp",
        action="store_true",
        help='Keep rows with missing "BP (raw)" or "MP (raw)"; they bypass the BP/MP rule.',
    )
    p.add_argument(
        "--lohc-value",
        type=float,
        default=1.0,
        help='Required value for "LOHC (raw)" (default: 1.0)',
    )
    p.add_argument(
        "--kelvin-offset",
        type=float,
        default=273.15,
        help='Kelvin→Celsius offset used as MP_C = MP_K - offset (default: 273.15)',
    )
    p.add_argument(
        "--mp-lt",
        type=float,
        default=None,
        help='Optional filter: keep only rows where "MP (raw)" < this value (Kelvin). Example: --mp-lt 300',
    )
    p.add_argument(
        "--bp-gt",
        type=float,
        default=None,
        help='Optional filter: keep only rows where "BP (raw)" > this value (Celsius). Example: --bp-gt 350',
    )

    # --- dedup.py args ---
    p.add_argument("--smiles-col", default="SMILES", help='SMILES column name (default: "SMILES")')
    p.add_argument("--score-col", default="Score", help='Score column name (default: "Score")')
    p.add_argument(
        "--keep-invalid",
        action="store_true",
        help="Keep rows where InChI could not be computed (they will not be deduplicated).",
    )
    p.add_argument(
        "--add-inchi-col",
        action="store_true",
        help='Add an "InChI" column to the output.',
    )
    return p


def require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def smiles_to_inchi(smiles: str) -> str | None:
    if smiles is None:
        return None
    s = str(smiles).strip()
    if not s:
        return None
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None
    try:
        return Chem.MolToInchi(mol)
    except Exception:
        return None


def main(argv: list[str]) -> int:
    args = build_argparser().parse_args(argv)

    df = pd.read_csv(args.input)
    n_input_rows = len(df)

    # Validate columns
    # We use args.score_col instead of the hardcoded SCORE_COL constant from filter.py
    # to respect the --score-col argument from dedup.py
    required_cols = [args.score_col, LOHC_RAW_COL, BP_RAW_COL, MP_RAW_COL]
    
    # Check for extra columns if they exist in the dataframe (handle cases where LOHC/BP/MP might be missing from input if logic requires it)
    # The original filter.py required them.
    require_columns(df, required_cols)
    
    if args.smiles_col not in df.columns:
         raise KeyError(f"Missing SMILES column: {args.smiles_col}")

    # Coerce numerics
    for c in required_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ==========================
    # PART 1: Filtering (filter.py)
    # ==========================
    bp_c = df[BP_RAW_COL]
    mp_k = df[MP_RAW_COL]
    mp_c = mp_k - args.kelvin_offset

    # 1) LOHC filter
    mask = df[LOHC_RAW_COL].eq(args.lohc_value)

    # 2) BP/MP consistency (keep MP(C) <= BP(C))
    valid_bpmp = bp_c.notna() & mp_k.notna()
    if args.keep_missing_bpmp:
        mask &= (~valid_bpmp) | (mp_c <= bp_c)
    else:
        mask &= valid_bpmp & (mp_c <= bp_c)

    # 3) Optional MP(raw) < threshold (Kelvin)
    if args.mp_lt is not None:
        mask &= mp_k.notna() & (mp_k < args.mp_lt)

    # 4) Optional BP(raw) > threshold (Celsius)
    if args.bp_gt is not None:
        mask &= bp_c.notna() & (bp_c > args.bp_gt)

    # Apply filter
    df_filtered = df.loc[mask].copy()
    n_filtered_rows = len(df_filtered)

    # ==========================
    # PART 2: Deduplication (dedup.py)
    # ==========================
    
    # compute InChI
    df_filtered["_InChI"] = df_filtered[args.smiles_col].map(smiles_to_inchi)

    # split valid/invalid
    valid = df_filtered["_InChI"].notna()
    df_valid = df_filtered.loc[valid].copy()
    df_invalid = df_filtered.loc[~valid].copy()

    # sort so the first row per InChI is the highest Score
    df_valid = df_valid.sort_values(args.score_col, ascending=False, na_position="last")
    df_valid = df_valid.drop_duplicates(subset=["_InChI"], keep="first")

    # merge back (optional)
    if args.keep_invalid:
        out = pd.concat([df_valid, df_invalid], ignore_index=True)
    else:
        out = df_valid

    # FINAL: sort output by Score descending
    out = out.sort_values(args.score_col, ascending=False, na_position="last")

    # expose or drop InChI col
    if args.add_inchi_col:
        out = out.rename(columns={"_InChI": "InChI"})
    else:
        out = out.drop(columns=["_InChI"], errors="ignore")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    msg = (
        f"Original Rows: {n_input_rows} | "
        f"After Filter: {n_filtered_rows} | "
        f"Valid InChI: {valid.sum()} | "
        f"Invalid InChI: {(~valid).sum()} | "
        f"Final Output: {len(out)} | "
        f"Saved: {out_path}"
    )
    print(msg, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
