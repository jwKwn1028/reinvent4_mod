#!/usr/bin/env python3
"""
smiles_to_annotated_images.py

Reads a CSV, filters rows where Score != 0.0, converts SMILES to molecule images,
and appends these values under each image:

  DE (raw), MP (raw), BP (raw), CapH2 (raw), SA (raw)

Outputs one PNG per row to an output directory.

Example:
  python smiles_to_annotated_images.py \
    -i stat/filtered/filtered_1_20.csv \
    -o stat/imgs_filtered_1_20 \
    --max-images 500
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

from PIL import Image, ImageDraw, ImageFont


SMILES_COL = "SMILES"
SCORE_COL = "Score"
FIELDS = ["DE (raw)", "MP (raw)", "BP (raw)", "CapH2 (raw)", "SA (raw)"]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Render annotated molecule images for rows with Score != 0.0")
    p.add_argument("-i", "--input", required=True, help="Input CSV path")
    p.add_argument("-o", "--outdir", required=True, help="Output directory for PNGs")
    p.add_argument("--img-size", type=int, default=300, help="Molecule image size (square pixels), default: 300")
    p.add_argument("--dpi", type=int, default=200, help="RDKit image DPI, default: 200")
    p.add_argument("--max-images", type=int, default=None, help="Optional cap on number of images written")
    p.add_argument(
        "--sort-by-score",
        action="store_true",
        help="Sort rows by Score descending before rendering",
    )
    p.add_argument(
        "--keep-invalid-smiles",
        action="store_true",
        help="If set, still write an image for invalid SMILES (blank mol with text). Default: skip invalid.",
    )
    return p


def _safe_stem(s: str, maxlen: int = 80) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s.strip())
    return (s[:maxlen] if len(s) > maxlen else s) or "row"


def _fmt(x: object) -> str:
    if x is None:
        return "NA"
    try:
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return "NA"
        # compact formatting
        return f"{float(x):.6g}"
    except Exception:
        xs = str(x).strip()
        return xs if xs else "NA"


def _make_annotated_image(mol_img: Image.Image, text: str, pad: int = 8) -> Image.Image:
    font = ImageFont.load_default()
    draw_tmp = ImageDraw.Draw(mol_img)

    # Measure text bbox using a temporary draw object
    try:
        bbox = draw_tmp.multiline_textbbox((0, 0), text, font=font, spacing=2)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except Exception:
        # Fallback if older Pillow
        text_w, text_h = draw_tmp.multiline_textsize(text, font=font, spacing=2)

    out_w = max(mol_img.width, text_w + 2 * pad)
    out_h = mol_img.height + text_h + 3 * pad

    out = Image.new("RGB", (out_w, out_h), "white")
    # center molecule
    out.paste(mol_img, ((out_w - mol_img.width) // 2, pad))

    draw = ImageDraw.Draw(out)
    text_x = pad
    text_y = mol_img.height + 2 * pad
    draw.multiline_text((text_x, text_y), text, fill="black", font=font, spacing=2)

    return out


def main() -> int:
    args = build_argparser().parse_args()

    in_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    # Basic column checks
    missing = [c for c in [SMILES_COL, SCORE_COL, *FIELDS] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # Coerce numeric columns used for filtering/printing
    df[SCORE_COL] = pd.to_numeric(df[SCORE_COL], errors="coerce")
    for c in FIELDS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Filter Score != 0.0 (and non-null)
    df = df[df[SCORE_COL].notna() & (df[SCORE_COL] != 0.0)].copy()

    if args.sort_by_score:
        df = df.sort_values(SCORE_COL, ascending=False)

    if args.max_images is not None:
        df = df.head(args.max_images)

    n_written = 0
    n_skipped_invalid = 0

    for idx, row in df.iterrows():
        smiles = str(row[SMILES_COL]).strip()
        mol = Chem.MolFromSmiles(smiles) if smiles else None

        if mol is None and not args.keep_invalid_smiles:
            n_skipped_invalid += 1
            continue

        # Draw molecule (blank if invalid)
        if mol is None:
            mol_img = Image.new("RGB", (args.img_size, args.img_size), "white")
        else:
            mol_img = Draw.MolToImage(mol, size=(args.img_size, args.img_size), dpi=args.dpi)

        # Build annotation text under image
        lines = [
            f"Score: {_fmt(row[SCORE_COL])}",
            f"DE(raw): {_fmt(row['DE (raw)'])}    MP(raw): {_fmt(row['MP (raw)'])}    BP(raw): {_fmt(row['BP (raw)'])}",
            f"CapH2(raw): {_fmt(row['CapH2 (raw)'])}    SA(raw): {_fmt(row['SA (raw)'])}",
        ]
        text = "\n".join(lines)

        out_img = _make_annotated_image(mol_img, text)

        stem = _safe_stem(f"{idx}_score_{_fmt(row[SCORE_COL])}")
        out_path = outdir / f"{stem}.png"
        out_img.save(out_path)
        n_written += 1

    print(
        f"Input rows (Score!=0): {len(df)} | Written: {n_written} | Skipped invalid SMILES: {n_skipped_invalid}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
