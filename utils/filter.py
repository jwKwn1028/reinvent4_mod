#!/usr/bin/env python3
"""
filter_lohc_bp_mp.py

Columns expected (original names):
Agent,Prior,Target,Score,SMILES,SMILES_state,Scaffold,DE,DE (raw),
MP,MP (raw),BP,BP (raw),CapH2,CapH2 (raw),SA,SA (raw),LOHC,LOHC (raw),step

Filtering:
1) Keep rows where "LOHC (raw)" == 1.0 (configurable)
2) BP/MP consistency with units:
   - "BP (raw)" is Celsius
   - "MP (raw)" is Kelvin
   Filter out where MP(C) > BP(C)
   => keep only where (MP (raw) - 273.15) <= (BP (raw))
3) Optional: keep only where "MP (raw)" < <threshold> (Kelvin), via --mp-lt
4) Optional: keep only where "BP (raw)" > <threshold> (Celsius), via --bp-gt
5) Sort by "Score" descending
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# ----- fixed original column names -----
SCORE_COL = "Score"
LOHC_RAW_COL = "LOHC (raw)"
BP_RAW_COL = "BP (raw)"   # Celsius
MP_RAW_COL = "MP (raw)"   # Kelvin


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Filter REINVENT log CSV by LOHC and BP/MP rule (BP °C, MP K), then sort by Score desc."
    )
    p.add_argument("-i", "--input", required=True, help="Input CSV path")
    p.add_argument("-o", "--output", required=True, help="Output CSV path")

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

    # NEW: threshold-based optional filters
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
    return p


def require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def main(argv: list[str]) -> int:
    args = build_argparser().parse_args(argv)

    df = pd.read_csv(args.input)
    require_columns(df, [SCORE_COL, LOHC_RAW_COL, BP_RAW_COL, MP_RAW_COL])

    # Coerce numerics
    for c in [SCORE_COL, LOHC_RAW_COL, BP_RAW_COL, MP_RAW_COL]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

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

    out = df.loc[mask].sort_values(SCORE_COL, ascending=False)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(
        f"Input rows: {len(df)} | Output rows: {len(out)} | Saved: {out_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
