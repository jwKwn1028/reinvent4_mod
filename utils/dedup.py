#!/usr/bin/env python3
"""
dedup_by_inchi.py

Read a CSV, convert SMILES -> InChI (RDKit), and deduplicate by InChI.
For duplicates, KEEP the row with the higher Score (drop lower Score rows).
Final output is sorted by Score descending.

Notes:
- Rows with invalid SMILES (or failed InChI conversion) are dropped by default.
- Uses Standard InChI from RDKit: Chem.MolToInchi(mol)

Example:
  python dedup_by_inchi.py -i input.csv -o dedup.csv --add-inchi-col
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Deduplicate CSV rows by InChI, keeping highest Score per compound.")
    p.add_argument("-i", "--input", required=True, help="Input CSV path")
    p.add_argument("-o", "--output", required=True, help="Output CSV path")
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

    if args.smiles_col not in df.columns:
        raise KeyError(f"Missing SMILES column: {args.smiles_col}")
    if args.score_col not in df.columns:
        raise KeyError(f"Missing Score column: {args.score_col}")

    # numeric score (NaN allowed)
    df[args.score_col] = pd.to_numeric(df[args.score_col], errors="coerce")

    # compute InChI
    df["_InChI"] = df[args.smiles_col].map(smiles_to_inchi)

    # split valid/invalid
    valid = df["_InChI"].notna()
    df_valid = df.loc[valid].copy()
    df_invalid = df.loc[~valid].copy()

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
        f"Input rows: {len(df)} | "
        f"Valid InChI: {valid.sum()} | "
        f"Invalid InChI: {(~valid).sum()} | "
        f"Output rows: {len(out)} | "
        f"Saved: {out_path}"
    )
    print(msg, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
