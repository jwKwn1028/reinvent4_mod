#!/usr/bin/env python3
"""
dedup_inchi.py

Deduplicate a CSV (like your mp.csv) by molecule identity using RDKit InChIKey
computed from the parsed molecule (Mol), not SMILES-string equality.

Key features
- Drops "Unnamed:*" columns
- Parses SMILES -> Mol -> InChIKey
- Optional: outputs canonical SMILES from Mol
- Deduplicates per InChIKey with several mp aggregation modes, including:

  mp_agg = "max_decimals_then_median"
    1) Within each molecule group, prefer the row(s) whose mp_raw has the most
       decimal digits AFTER IGNORING TRAILING ZEROS.
       Example: "12.500" -> 1 decimal (treat as "12.5")
                "12.00"  -> 0 decimals (treat as "12")
    2) If multiple rows tie, choose the median mp (lower-median if even count),
       selecting an existing row (not averaging).

Outputs
- deduplicated CSV
- optional "removed rows" audit CSV

Usage
  python dedup_inchi.py --input mp.csv --out mp_dedup.csv --removed mp_removed.csv \
    --mp-agg max_decimals_then_median --smiles-rep canonical
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import MolToInchiKey
from rdkit.Chem.rdchem import Mol


MpAgg = Literal[
    "median",
    "mean",
    "min",
    "max",
    "first",
    "last",
    "max_decimals_then_median",
]

SmilesRep = Literal["canonical", "original_first", "original_last"]


@dataclass(frozen=True)
class DedupConfig:
    smiles_col: str = "smiles"
    mp_col: str = "mp"

    mp_agg: MpAgg = "median"
    smiles_rep: SmilesRep = "canonical"

    drop_invalid_smiles: bool = True
    drop_non_numeric_mp: bool = True

    keep_inchikey: bool = False


def _mol_from_smiles(smiles: str) -> Optional[Mol]:
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def _inchikey_from_mol(mol: Mol) -> Optional[str]:
    try:
        return MolToInchiKey(mol)
    except Exception:
        return None


def _count_decimals_ignore_trailing_zeros(mp_raw_series: pd.Series) -> pd.Series:
    """
    Count digits after decimal point in mp_raw strings, ignoring trailing zeros.
    Examples:
      "12.500" -> 1
      "12.050" -> 2  (treat "12.05")
      "12.00"  -> 0
      "12"     -> 0
      "1.20e2" -> 1  (mantissa "1.20" -> "1.2")
    Returns int64 Series.
    """
    mp_str = mp_raw_series.astype("string").str.strip()

    # Split exponent (if any)
    mantissa = mp_str.str.split(r"[eE]", n=1).str[0]

    # Decimal part (None if no dot)
    dec_part = mantissa.str.split(".", n=1).str[1]

    # Ignore trailing zeros
    dec_part = dec_part.fillna("").str.rstrip("0")

    # Length of remaining decimal digits
    return dec_part.str.len().astype("int64")


def deduplicate_mp_df_by_inchikey(
    df: pd.DataFrame, cfg: DedupConfig = DedupConfig()
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if cfg.smiles_col not in df.columns or cfg.mp_col not in df.columns:
        raise ValueError(
            f"Expected columns '{cfg.smiles_col}' and '{cfg.mp_col}'. Got: {list(df.columns)}"
        )

    # Drop typical junk columns
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed:")], errors="ignore").copy()

    # Normalize SMILES
    df[cfg.smiles_col] = df[cfg.smiles_col].astype("string").str.strip()
    df = df[df[cfg.smiles_col].notna() & (df[cfg.smiles_col] != "")].copy()

    # Preserve raw mp (string) for precision-based selection
    df["mp_raw"] = df[cfg.mp_col]

    # Numeric mp
    df[cfg.mp_col] = pd.to_numeric(df[cfg.mp_col], errors="coerce")

    removed_parts: list[pd.DataFrame] = []

    # Drop non-numeric mp
    if cfg.drop_non_numeric_mp:
        bad_mp = df[cfg.mp_col].isna()
        if bool(bad_mp.any()):
            removed_parts.append(df.loc[bad_mp].assign(reason="non_numeric_mp"))
            df = df.loc[~bad_mp].copy()

    if len(df) == 0:
        removed_df = pd.concat(removed_parts, ignore_index=True) if removed_parts else pd.DataFrame()
        out_cols = [cfg.smiles_col, cfg.mp_col] + (["inchi_key"] if cfg.keep_inchikey else [])
        return pd.DataFrame(columns=out_cols), removed_df

    # RDKit parsing + InChIKey (molecule-based identity)
    inchikeys: list[Optional[str]] = []
    canon_smiles: list[Optional[str]] = []
    invalid_mask: list[bool] = []

    for s in df[cfg.smiles_col].astype(str).tolist():
        mol = _mol_from_smiles(s)
        if mol is None:
            inchikeys.append(None)
            canon_smiles.append(None)
            invalid_mask.append(True)
            continue

        ik = _inchikey_from_mol(mol)
        if ik is None:
            inchikeys.append(None)
            canon_smiles.append(None)
            invalid_mask.append(True)
            continue

        inchikeys.append(ik)
        canon_smiles.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
        invalid_mask.append(False)

    df["inchi_key"] = inchikeys
    df["smiles_canonical"] = canon_smiles

    # Drop invalid SMILES/InChIKey rows
    invalid = pd.Series(invalid_mask, index=df.index)
    if cfg.drop_invalid_smiles and bool(invalid.any()):
        removed_parts.append(df.loc[invalid].assign(reason="invalid_smiles_or_inchikey"))
        df = df.loc[~invalid].copy()

    removed_df = pd.concat(removed_parts, ignore_index=True) if removed_parts else pd.DataFrame()

    if len(df) == 0:
        out_cols = [cfg.smiles_col, cfg.mp_col] + (["inchi_key"] if cfg.keep_inchikey else [])
        return pd.DataFrame(columns=out_cols), removed_df

    # Representative SMILES choice
    if cfg.smiles_rep == "canonical":
        rep_smiles_col = "smiles_canonical"
        rep_keep = "first"
    elif cfg.smiles_rep == "original_first":
        rep_smiles_col = cfg.smiles_col
        rep_keep = "first"
    else:  # original_last
        rep_smiles_col = cfg.smiles_col
        rep_keep = "last"

    # Stable row index for deterministic tie-breaking
    df["_row"] = np.arange(len(df), dtype=np.int64)

    # --- Special mode: max decimals (ignoring trailing zeros), then choose median row ---
    if cfg.mp_agg == "max_decimals_then_median":
        df["mp_decimals"] = _count_decimals_ignore_trailing_zeros(df["mp_raw"])

        # keep only rows with maximum decimals per InChIKey
        max_dec = df.groupby("inchi_key", sort=False)["mp_decimals"].transform("max")
        df_max = df[df["mp_decimals"] == max_dec].copy()

        # Choose median mp row (lower median if even), stable on _row
        df_max = df_max.sort_values(["inchi_key", cfg.mp_col, "_row"], kind="mergesort")
        df_max["_rank"] = df_max.groupby("inchi_key", sort=False).cumcount()
        n = df_max.groupby("inchi_key", sort=False)[cfg.mp_col].transform("size")
        mid = (n - 1) // 2  # lower median index

        chosen = df_max[df_max["_rank"] == mid].copy()

        # finalize smiles column
        chosen[cfg.smiles_col] = chosen[rep_smiles_col]

        out_cols = [cfg.smiles_col, cfg.mp_col] + (["inchi_key"] if cfg.keep_inchikey else [])
        return chosen[out_cols].reset_index(drop=True), removed_df

    # --- Other aggregation modes ---
    if cfg.mp_agg in ("first", "last"):
        keep = "first" if cfg.mp_agg == "first" else "last"
        chosen = df.drop_duplicates(subset=["inchi_key"], keep=keep).copy()
        chosen[cfg.smiles_col] = chosen[rep_smiles_col]
        out_cols = [cfg.smiles_col, cfg.mp_col] + (["inchi_key"] if cfg.keep_inchikey else [])
        return chosen[out_cols].reset_index(drop=True), removed_df

    # median/mean/min/max as true aggregations
    agg_map = {cfg.mp_col: cfg.mp_agg, rep_smiles_col: rep_keep}
    dedup = df.groupby("inchi_key", sort=False, as_index=False).agg(agg_map)
    dedup = dedup.rename(columns={rep_smiles_col: cfg.smiles_col})

    out_cols = [cfg.smiles_col, cfg.mp_col] + (["inchi_key"] if cfg.keep_inchikey else [])
    return dedup[out_cols].reset_index(drop=True), removed_df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input CSV path (expects columns: smiles, mp).")
    p.add_argument("--out", required=True, help="Output deduplicated CSV path.")
    p.add_argument("--removed", default="", help="Optional output path for removed rows audit CSV.")
    p.add_argument(
        "--mp-agg",
        default="median",
        choices=[
            "median",
            "mean",
            "min",
            "max",
            "first",
            "last",
            "max_decimals_then_median",
        ],
        help="How to choose mp when duplicates exist per InChIKey.",
    )
    p.add_argument(
        "--smiles-rep",
        default="canonical",
        choices=["canonical", "original_first", "original_last"],
        help="Which SMILES to output for each deduplicated molecule.",
    )
    p.add_argument("--keep-inchikey", action="store_true", help="Keep inchi_key column in output.")
    p.add_argument("--no-drop-non-numeric-mp", action="store_true", help="Do not drop non-numeric mp rows.")
    p.add_argument("--no-drop-invalid-smiles", action="store_true", help="Do not drop invalid SMILES/InChIKey rows.")
    p.add_argument("--quiet-rdkit", action="store_true", help="Disable RDKit warnings.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.quiet_rdkit:
        try:
            rdBase.DisableLog("rdApp.*")
        except Exception:
            # Fallback if DisableLog isn't available for some reason
            from rdkit import RDLogger
            RDLogger.logger().setLevel(RDLogger.CRITICAL)

    in_path = Path(args.input)
    out_path = Path(args.out)
    removed_path = Path(args.removed) if args.removed else None

    if not in_path.exists():
        print(f"ERROR: input file not found: {in_path}", file=sys.stderr)
        return 2

    df = pd.read_csv(in_path, low_memory=False)

    cfg = DedupConfig(
        smiles_col="smiles",
        mp_col="mp",
        mp_agg=args.mp_agg,  # type: ignore[arg-type]
        smiles_rep=args.smiles_rep,  # type: ignore[arg-type]
        keep_inchikey=bool(args.keep_inchikey),
        drop_non_numeric_mp=not bool(args.no_drop_non_numeric_mp),
        drop_invalid_smiles=not bool(args.no_drop_invalid_smiles),
    )

    dedup_df, removed_df = deduplicate_mp_df_by_inchikey(df, cfg=cfg)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    dedup_df.to_csv(out_path, index=False)

    if removed_path is not None:
        removed_path.parent.mkdir(parents=True, exist_ok=True)
        removed_df.to_csv(removed_path, index=False)

    print(f"[OK] input rows: {len(df):,}")
    print(f"[OK] dedup rows:  {len(dedup_df):,}")
    if removed_path is not None:
        print(f"[OK] removed rows written: {len(removed_df):,} -> {removed_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
