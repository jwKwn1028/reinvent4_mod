#!/usr/bin/env python3
"""
dedup_inchi.py

Deduplicate one or more CSVs by molecule identity using RDKit InChIKey computed
from parsed Mol, not SMILES string equality.

Multi-input support
- Repeat --input to provide multiple CSV paths.
- Default behavior ("merge mode"):
    * Concatenate all inputs (same schema assumed)
    * Deduplicate globally by InChIKey
    * Write ONE deduplicated output CSV (+ optional removed/audit CSV)
- Optional behavior ("per-file mode", --per-file):
    * Deduplicate each input independently
    * Write ONE output per input (use --out as a directory or a template with {stem})

Key features (same as original)
- Drops "Unnamed:*" columns
- Parses SMILES -> Mol -> InChIKey
- Optional canonical SMILES output
- Deduplicates per InChIKey with mp aggregation modes, including:
  mp_agg = "max_decimals_then_median"
    1) Prefer rows whose mp_raw has most decimal digits after ignoring trailing zeros
    2) If ties, choose the median mp row (lower-median), selecting an existing row

Outputs
- Deduplicated CSV(s)
- Optional removed/audit CSV(s)

Examples
--------
Merge multiple CSVs into one deduplicated dataset:
  python dedup_inchi.py --input A.csv --input B.csv --input C.csv \
    --out mp_dedup.csv --removed mp_removed.csv \
    --smiles-col smiles --mp-col mp \
    --mp-agg max_decimals_then_median --smiles-rep canonical --keep-inchikey

Per-file dedup (one output per input) into a directory:
  python dedup_inchi.py --per-file --input A.csv --input B.csv \
    --out dedup_out/ --removed removed_out/ \
    --smiles-col smiles --mp-col mp --mp-agg median

Per-file dedup with filename template:
  python dedup_inchi.py --per-file --input A.csv --input B.csv \
    --out "dedup_{stem}.csv" --removed "removed_{stem}.csv"
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
try:
    # Most common import location
    from rdkit.Chem.inchi import MolToInchiKey  # type: ignore
except Exception:
    # Fallback for some builds
    from rdkit.Chem import MolToInchiKey  # type: ignore

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
        return str(MolToInchiKey(mol))
    except Exception:
        return None


def _count_decimals_ignore_trailing_zeros(mp_raw_series: pd.Series) -> pd.Series:
    """
    Count digits after decimal point in mp_raw strings, ignoring trailing zeros.

    Examples:
      "12.500" -> 1 (treat as 12.5)
      "12.050" -> 2 (treat as 12.05)
      "12.00"  -> 0 (treat as 12)
      "12"     -> 0
      "1.20e2" -> 1 (mantissa 1.20 -> 1.2)
    """
    mp_str = mp_raw_series.astype("string").fillna("").str.strip()

    # Split exponent (if any): keep mantissa only
    mantissa = mp_str.str.split(r"[eE]", n=1).str[0]

    # Get decimal part if present
    dec_part = mantissa.str.split(".", n=1).str[1]

    # Ignore trailing zeros
    dec_part = dec_part.fillna("").str.rstrip("0")

    return dec_part.str.len().astype("int64")


def deduplicate_mp_df_by_inchikey(
    df: pd.DataFrame, cfg: DedupConfig = DedupConfig()
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      dedup_df: deduplicated output with columns [smiles, mp] (+ inchikey if requested)
      removed_df: audit rows removed for reasons (non_numeric_mp, invalid_smiles_or_inchikey)
    """
    if cfg.smiles_col not in df.columns or cfg.mp_col not in df.columns:
        raise ValueError(
            f"Expected columns '{cfg.smiles_col}' and '{cfg.mp_col}'. Got: {list(df.columns)}"
        )

    # Drop typical junk columns
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed:")], errors="ignore").copy()

    # Normalize SMILES
    df[cfg.smiles_col] = df[cfg.smiles_col].astype("string").fillna("").str.strip()
    df = df[df[cfg.smiles_col] != ""].copy()

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
        if not ik:
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

        max_dec = df.groupby("inchi_key", sort=False)["mp_decimals"].transform("max")
        df_max = df[df["mp_decimals"] == max_dec].copy()

        # Choose median mp row (lower median if even), stable on _row
        df_max = df_max.sort_values(["inchi_key", cfg.mp_col, "_row"], kind="mergesort")
        df_max["_rank"] = df_max.groupby("inchi_key", sort=False).cumcount()
        n = df_max.groupby("inchi_key", sort=False)[cfg.mp_col].transform("size")
        mid = (n - 1) // 2  # lower median index
        chosen = df_max[df_max["_rank"] == mid].copy()

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

    # median/mean/min/max as aggregations
    agg_map = {cfg.mp_col: cfg.mp_agg, rep_smiles_col: rep_keep}
    dedup = df.groupby("inchi_key", sort=False, as_index=False).agg(agg_map)
    dedup = dedup.rename(columns={rep_smiles_col: cfg.smiles_col})

    out_cols = [cfg.smiles_col, cfg.mp_col] + (["inchi_key"] if cfg.keep_inchikey else [])
    return dedup[out_cols].reset_index(drop=True), removed_df


def _read_csv_with_source(path: Path, source_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df[source_col] = str(path)
    return df


def _resolve_out_path(out_arg: str, in_path: Path, per_file: bool) -> Path:
    """
    If per_file:
      - if out_arg is a directory, output = out_arg/<stem>_dedup.csv
      - if out_arg contains "{stem}", format with input stem
      - else: treat as a file path (only valid when single input; we still allow but will overwrite)
    If not per_file: out_arg is a file path.
    """
    outp = Path(out_arg)
    stem = in_path.stem

    if not per_file:
        return outp

    # per-file mode
    if out_arg.endswith("/") or (outp.exists() and outp.is_dir()):
        outp.mkdir(parents=True, exist_ok=True)
        return outp / f"{stem}_dedup.csv"

    if "{stem}" in out_arg:
        return Path(out_arg.format(stem=stem))

    # fallback: treat as file path (caller should ensure it’s sensible)
    return outp


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # Multi-input: repeat --input
    p.add_argument(
        "--input",
        action="append",
        required=True,
        help="Input CSV path. Repeat to provide multiple inputs.",
    )

    p.add_argument(
        "--out",
        required=True,
        help=(
            "Output path. "
            "In merge mode: a single output CSV path. "
            "In --per-file mode: a directory OR a template containing '{stem}'."
        ),
    )
    p.add_argument(
        "--removed",
        default="",
        help=(
            "Optional removed/audit output. "
            "In merge mode: a single CSV path. "
            "In --per-file mode: a directory OR a template containing '{stem}'."
        ),
    )

    p.add_argument("--smiles-col", default="smiles", help="SMILES column name (default: smiles).")
    p.add_argument("--mp-col", default="mp", help="Melting point column name (default: mp).")

    p.add_argument(
        "--mp-agg",
        default="median",
        choices=["median", "mean", "min", "max", "first", "last", "max_decimals_then_median"],
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

    p.add_argument(
        "--per-file",
        action="store_true",
        help="Deduplicate each input independently (writes one output per input). Default is merge+dedup.",
    )
    p.add_argument(
        "--source-col",
        default="source_file",
        help="Column name to store the input filename in merged removed/audit outputs (default: source_file).",
    )

    p.add_argument("--quiet-rdkit", action="store_true", help="Disable RDKit warnings.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.quiet_rdkit:
        try:
            rdBase.DisableLog("rdApp.*")
        except Exception:
            from rdkit import RDLogger
            RDLogger.logger().setLevel(RDLogger.CRITICAL)

    in_paths = [Path(p) for p in args.input]
    for p in in_paths:
        if not p.exists():
            print(f"ERROR: input file not found: {p}", file=sys.stderr)
            return 2

    cfg = DedupConfig(
        smiles_col=str(args.smiles_col),
        mp_col=str(args.mp_col),
        mp_agg=args.mp_agg,  # type: ignore[arg-type]
        smiles_rep=args.smiles_rep,  # type: ignore[arg-type]
        keep_inchikey=bool(args.keep_inchikey),
        drop_non_numeric_mp=not bool(args.no_drop_non_numeric_mp),
        drop_invalid_smiles=not bool(args.no_drop_invalid_smiles),
    )

    per_file = bool(args.per_file)
    removed_arg = str(args.removed).strip()
    source_col = str(args.source_col)

    if per_file:
        # Process each input independently
        for p in in_paths:
            df = _read_csv_with_source(p, source_col=source_col)

            dedup_df, removed_df = deduplicate_mp_df_by_inchikey(df, cfg=cfg)

            out_path = _resolve_out_path(args.out, p, per_file=True)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            dedup_df.to_csv(out_path, index=False)

            if removed_arg:
                rem_path = _resolve_out_path(removed_arg, p, per_file=True)
                rem_path.parent.mkdir(parents=True, exist_ok=True)
                removed_df.to_csv(rem_path, index=False)

            print(f"[OK] {p.name}: input {len(df):,} -> dedup {len(dedup_df):,}")

        return 0

    # Merge mode (default): concatenate then dedup once
    dfs = [_read_csv_with_source(p, source_col=source_col) for p in in_paths]

    # Optional: sanity check schema consistency
    cols0 = list(dfs[0].columns)
    for i, dfi in enumerate(dfs[1:], start=1):
        if list(dfi.columns) != cols0:
            # not fatal if you still have required columns; but warn loudly
            print(
                f"WARNING: column schema differs for input #{i+1} ({in_paths[i].name}). "
                "Proceeding anyway; required columns must exist.",
                file=sys.stderr,
            )

    df_all = pd.concat(dfs, ignore_index=True)

    dedup_df, removed_df = deduplicate_mp_df_by_inchikey(df_all, cfg=cfg)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dedup_df.to_csv(out_path, index=False)

    if removed_arg:
        rem_path = Path(removed_arg)
        rem_path.parent.mkdir(parents=True, exist_ok=True)
        removed_df.to_csv(rem_path, index=False)

    print(f"[OK] inputs: {len(in_paths)} file(s)")
    print(f"[OK] input rows: {len(df_all):,}")
    print(f"[OK] dedup rows:  {len(dedup_df):,}")
    if removed_arg:
        print(f"[OK] removed rows: {len(removed_df):,}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
