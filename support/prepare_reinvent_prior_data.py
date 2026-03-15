#!/usr/bin/env python
"""Prepare REINVENT prior-training data from CH/CHO CSV files."""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

from rdkit import Chem
from rdkit import rdBase

from reinvent.chemistry.utils import compute_scaffold

rdBase.DisableLog("rdApp.*")


DATASET_INPUTS = {
    "CH": Path("data/CH.csv"),
    "CHO": Path("data/CHO.csv"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract SMILES, deduplicate, and write scaffold-based train/validation splits."
    )
    parser.add_argument("--dataset", required=True, type=str.upper, choices=sorted(DATASET_INPUTS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--output-root", type=Path, default=Path("data/priors"))
    return parser.parse_args()


def get_smiles_field(fieldnames: list[str] | None) -> str:
    if not fieldnames:
        raise RuntimeError("Input CSV is missing a header row")

    for field in fieldnames:
        if field.strip().lower() == "smiles":
            return field

    raise RuntimeError(f"Unable to find a SMILES column in header: {fieldnames}")


def load_unique_smiles(input_csv: Path) -> tuple[list[str], int, str]:
    with input_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        smiles_field = get_smiles_field(reader.fieldnames)

        smiles_list = []

        for row in reader:
            smiles = row[smiles_field].strip()

            if smiles:
                smiles_list.append(smiles)

    unique_smiles = list(dict.fromkeys(smiles_list))
    duplicates_removed = len(smiles_list) - len(unique_smiles)

    return unique_smiles, duplicates_removed, smiles_field


def get_scaffold_key(smiles: str) -> tuple[str, bool]:
    mol = Chem.MolFromSmiles(smiles)

    if not mol:
        return f"__INVALID__::{smiles}", True

    scaffold = compute_scaffold(mol, generic=True, isomeric=False)

    if scaffold:
        return scaffold, False

    return "__NO_SCAFFOLD__", False


def split_smiles_by_scaffold(
    smiles_list: list[str], valid_fraction: float, seed: int
) -> tuple[list[str], list[str], int, int, int]:
    if not 0.0 <= valid_fraction < 1.0:
        raise RuntimeError(f"valid_fraction must be in [0.0, 1.0), got {valid_fraction}")

    if not smiles_list:
        raise RuntimeError("No SMILES were extracted from the input CSV")

    scaffold_groups = defaultdict(list)
    invalid_scaffolds = 0

    for smiles in smiles_list:
        scaffold_key, is_invalid = get_scaffold_key(smiles)
        scaffold_groups[scaffold_key].append(smiles)
        invalid_scaffolds += int(is_invalid)

    if valid_fraction == 0.0:
        return list(smiles_list), [], len(scaffold_groups), len(scaffold_groups), 0

    if len(smiles_list) < 2:
        raise RuntimeError("Need at least 2 SMILES to create a validation split")

    valid_count = int(len(smiles_list) * valid_fraction)
    valid_count = max(1, valid_count)
    valid_count = min(len(smiles_list) - 1, valid_count)
    train_target = len(smiles_list) - valid_count

    grouped_smiles = list(scaffold_groups.items())
    rng = random.Random(seed)
    rng.shuffle(grouped_smiles)
    grouped_smiles.sort(key=lambda item: len(item[1]), reverse=True)

    train_smiles = []
    valid_smiles = []
    train_scaffolds = []
    valid_scaffolds = []

    for scaffold_key, group in grouped_smiles:
        if len(train_smiles) < train_target:
            train_smiles.extend(group)
            train_scaffolds.append(scaffold_key)
        else:
            valid_smiles.extend(group)
            valid_scaffolds.append(scaffold_key)

    if not valid_smiles and len(train_scaffolds) > 1:
        moved_scaffold = train_scaffolds.pop()
        moved_group = scaffold_groups[moved_scaffold]
        del train_smiles[-len(moved_group) :]
        valid_smiles.extend(moved_group)
        valid_scaffolds.append(moved_scaffold)

    if train_smiles and valid_smiles:
        rng.shuffle(train_smiles)
        rng.shuffle(valid_smiles)

    print(f"invalid_scaffold_fallbacks={invalid_scaffolds}")

    return (
        train_smiles,
        valid_smiles,
        len(scaffold_groups),
        len(train_scaffolds),
        len(valid_scaffolds),
    )


def write_smiles(output_file: Path, smiles_list: list[str]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w") as handle:
        for smiles in smiles_list:
            handle.write(f"{smiles}\n")


def main() -> None:
    args = parse_args()

    input_csv = DATASET_INPUTS[args.dataset]
    output_dir = args.output_root / args.dataset.lower()

    unique_smiles, duplicates_removed, smiles_field = load_unique_smiles(input_csv)
    train_smiles, valid_smiles, scaffold_count, train_scaffold_count, valid_scaffold_count = (
        split_smiles_by_scaffold(unique_smiles, args.valid_fraction, args.seed)
    )

    all_file = output_dir / "all.smi"
    train_file = output_dir / "train.smi"
    valid_file = output_dir / "valid.smi"

    write_smiles(all_file, unique_smiles)
    write_smiles(train_file, train_smiles)
    write_smiles(valid_file, valid_smiles)

    print(f"dataset={args.dataset}")
    print(f"input_csv={input_csv}")
    print(f"smiles_field={smiles_field}")
    print(f"output_dir={output_dir}")
    print(f"seed={args.seed}")
    print(f"valid_fraction={args.valid_fraction}")
    print(f"unique_smiles={len(unique_smiles)}")
    print(f"duplicates_removed={duplicates_removed}")
    print(f"unique_scaffolds={scaffold_count}")
    print(f"train_smiles={len(train_smiles)}")
    print(f"valid_smiles={len(valid_smiles)}")
    print(f"train_scaffolds={train_scaffold_count}")
    print(f"valid_scaffolds={valid_scaffold_count}")
    print(f"all_file={all_file}")
    print(f"train_file={train_file}")
    print(f"valid_file={valid_file}")


if __name__ == "__main__":
    main()
