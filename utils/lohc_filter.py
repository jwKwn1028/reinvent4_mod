"""
filter_molecules.py

Reads SMILES from a CSV and splits into filtered / unfiltered outputs.

Core filters:
  (2) atom count in [1, 100]
  (3) ring count in [1, 5] AND at least one ring contains a "double-like" bond
      - double-like = DOUBLE bond, and optionally aromatic bonds count too

Additional MP/LOHC-oriented filters (enabled by default, configurable via CLI):
  - single fragment (no salts/mixtures)
  - formal charge == 0
  - HBD <= 1
  - TPSA <= 50
  - 0 <= rotatable bonds <= 9

Outputs (original CSV unchanged):
  - filtered_{original}.csv   : passes all enabled filters
  - unfiltered_{original}.csv : fails any filter

Also:
  - RDKit canonical SMILES in column "rdkit_canonical_smiles"
  - Optional replacement of input SMILES column with canonical SMILES
  - Optional LOHC scaffold validation report
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Lipinski, rdMolDescriptors


def ring_has_double_like_bond(mol: Chem.Mol, count_aromatic_as_double: bool) -> bool:
    """True if any ring contains a DOUBLE bond, or (optionally) any aromatic bond."""
    ri = mol.GetRingInfo()
    for ring_bond_indices in ri.BondRings():
        for bidx in ring_bond_indices:
            b = mol.GetBondWithIdx(int(bidx))
            if b.GetBondType() == Chem.BondType.DOUBLE:
                return True
            if count_aromatic_as_double and b.GetIsAromatic():
                return True
    return False


def canonicalize_smiles(mol: Chem.Mol, isomeric: bool = True) -> str:
    """RDKit canonical SMILES (optionally including stereo)."""
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)


# def symmetry_proxy(mol: Chem.Mol) -> float:
#     """
#     Graph-symmetry proxy in [0, ~1]:
#       proxy = 1 - (#unique canonical ranks / #atoms)
#     Higher => more symmetry/equivalence classes.
#     """
#     n = mol.GetNumAtoms()
#     if n <= 0:
#         return 0.0
#     ranks = Chem.CanonicalRankAtoms(mol, breakTies=False)
#     n_unique = len(set(ranks))
#     return 1.0 - (n_unique / float(n))


def evaluate_smiles(
    smiles: str,
    *,
    atom_min: int,
    atom_max: int,
    ring_min: int,
    ring_max: int,
    count_aromatic_as_double: bool,
    require_single_fragment: bool,
    require_neutral: bool,
    max_hbd: int,
    max_tpsa: float,
    min_rotb: int,
    max_rotb: int,
    canonical_isomeric: bool,
) -> Tuple[bool, str, Optional[Chem.Mol], Optional[str], dict]:
    """
    Returns (passes, reason, mol, canonical_smiles, metrics_dict).
    metrics_dict contains values used for filtering (when available).
    """
    s = (smiles or "").strip()
    if not s:
        return False, "empty_smiles", None, None, {}

    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return False, "invalid_smiles", None, None, {}

    can_smi = canonicalize_smiles(mol, isomeric=canonical_isomeric)

    # --- Metrics used by additional filters ---
    n_frags = len(Chem.GetMolFrags(mol))
    fcharge = Chem.GetFormalCharge(mol)
    hbd = Lipinski.NumHDonors(mol)
    tpsa = float(rdMolDescriptors.CalcTPSA(mol))
    rotb = Lipinski.NumRotatableBonds(mol)

    metrics = {
        "n_frags": n_frags,
        "formal_charge": fcharge,
        "hbd": hbd,
        "tpsa": tpsa,
        "rotb": rotb,
    }

    # --- Additional filters (MP/LOHC-oriented) ---
    if require_single_fragment and n_frags != 1:
        return False, f"not_single_fragment(n_frags={n_frags})", mol, can_smi, metrics

    if require_neutral and fcharge != 0:
        return False, f"nonzero_formal_charge({fcharge})", mol, can_smi, metrics

    if hbd > max_hbd:
        return False, f"hbd_out_of_range({hbd})", mol, can_smi, metrics

    if tpsa > max_tpsa:
        return False, f"tpsa_out_of_range({tpsa:.1f})", mol, can_smi, metrics

    if not (min_rotb <= rotb <= max_rotb):
        return False, f"rotb_out_of_range({rotb})", mol, can_smi, metrics

    # --- Existing filters ---
    n_atoms = mol.GetNumAtoms()
    if not (atom_min <= n_atoms <= atom_max):
        return False, f"atom_count_out_of_range({n_atoms})", mol, can_smi, metrics

    n_rings = rdMolDescriptors.CalcNumRings(mol)
    if not (ring_min <= n_rings <= ring_max):
        return False, f"ring_count_out_of_range({n_rings})", mol, can_smi, metrics

    if not ring_has_double_like_bond(
        mol, count_aromatic_as_double=count_aromatic_as_double
    ):
        return False, "no_double_bond_in_any_ring", mol, can_smi, metrics

    return True, "OK", mol, can_smi, metrics


def validate_lohc_panel(
    *,
    count_aromatic_as_double: bool,
    canonical_isomeric: bool,
    report_path: Path,
) -> None:
    """
    Writes a diagnostic report on a small LOHC-oriented scaffold panel to CSV.
    """
    panel = [
        ("Toluene (lean)", "Cc1ccccc1"),
        ("Methylcyclohexane (rich)", "CC1CCCCC1"),
        ("N-ethylcarbazole / NEC (lean)", "CCn1c2ccccc2c3ccccc13"),
        ("Carbazole (HBD=1 example)", "c1ccc2c(c1)[nH]c3ccccc23"),
        ("Quinoline (lean)", "c1ccc2ncccc2c1"),
        ("Tetralin (partially hydrogenated)", "c1ccc2CCCCc2c1"),
        ("Decalin (rich)", "C1CCC2CCCCC2C1"),
        ("Naphthalene (lean)", "c1ccc2ccccc2c1"),
    ]

    rows = []
    for name, smi in panel:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append(
                {
                    "name": name,
                    "input_smiles": smi,
                    "valid": False,
                    "rings_CalcNumRings": None,
                    "rings_SymmSSSR": None,
                    "double_like_in_ring": None,
                    "n_frags": None,
                    "formal_charge": None,
                    "TPSA": None,
                    "HBD": None,
                    "RotB": None,
                    "rdkit_canonical_smiles": None,
                    "count_aromatic_as_double": count_aromatic_as_double,
                }
            )
            continue

        rings_calc = rdMolDescriptors.CalcNumRings(mol)
        rings_sssr = len(Chem.GetSymmSSSR(mol))
        has_dl = ring_has_double_like_bond(
            mol, count_aromatic_as_double=count_aromatic_as_double
        )

        n_frags = len(Chem.GetMolFrags(mol))
        fcharge = Chem.GetFormalCharge(mol)
        tpsa = float(rdMolDescriptors.CalcTPSA(mol))
        hbd = Lipinski.NumHDonors(mol)
        rotb = Lipinski.NumRotatableBonds(mol)
        can = canonicalize_smiles(mol, isomeric=canonical_isomeric)

        rows.append(
            {
                "name": name,
                "input_smiles": smi,
                "valid": True,
                "rings_CalcNumRings": rings_calc,
                "rings_SymmSSSR": rings_sssr,
                "double_like_in_ring": has_dl,
                "n_frags": n_frags,
                "formal_charge": fcharge,
                "TPSA": tpsa,
                "HBD": hbd,
                "RotB": rotb,
                "rdkit_canonical_smiles": can,
                "count_aromatic_as_double": count_aromatic_as_double,
            }
        )

    report_df = pd.DataFrame(rows)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(report_path, index=False)

    print(f"Wrote LOHC validation report: {report_path}")


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--csv_path", type=Path, help="Input CSV file path")
    ap.add_argument(
        "--col", default="smiles", help="SMILES column name (default: smiles)"
    )

    # Existing filters
    ap.add_argument("--atom-min", type=int, default=1)
    ap.add_argument("--atom-max", type=int, default=100)
    ap.add_argument("--ring-min", type=int, default=1)
    ap.add_argument("--ring-max", type=int, default=9)

    ap.add_argument(
        "--no-count-aromatic-as-double",
        dest="count_aromatic_as_double",
        action="store_false",
        help="Do NOT treat aromatic bonds as satisfying the 'double bond in ring' requirement.",
    )
    ap.set_defaults(count_aromatic_as_double=True)

    ap.add_argument(
        "--strict-double-only",
        action="store_true",
        help="Only explicit DOUBLE bonds satisfy the requirement (aromatic bonds do not).",
    )

    ap.add_argument(
        "--canonical-isomeric",
        action="store_true",
        help="Include stereochemistry in RDKit canonical SMILES (recommended).",
    )
    ap.add_argument(
        "--replace-col-with-canonical",
        action="store_true",
        help="Replace the input SMILES column with rdkit_canonical_smiles in outputs.",
    )

    # Added filters (MP/LOHC-oriented)
    ap.add_argument(
        "--allow-multifragment",
        action="store_true",
        help="Allow multi-fragment molecules (salts/mixtures). Default: require single fragment.",
    )
    ap.add_argument(
        "--allow-charged",
        action="store_true",
        help="Allow non-neutral molecules. Default: require formal charge == 0.",
    )
    ap.add_argument(
        "--max-hbd", type=int, default=1, help="Max H-bond donors (default: 1)"
    )
    ap.add_argument(
        "--max-tpsa", type=float, default=50.0, help="Max TPSA in Å^2 (default: 50)"
    )
    ap.add_argument(
        "--min-rotb", type=int, default=0, help="Min rotatable bonds (default: 0)"
    )
    ap.add_argument(
        "--max-rotb", type=int, default=9, help="Max rotatable bonds (default: 9)"
    )

    # Validation
    ap.add_argument(
        "--validate-lohc-panel",
        action="store_true",
        help="Print a diagnostic report on a small LOHC scaffold panel, then exit.",
    )

    args = ap.parse_args()

    if args.strict_double_only:
        count_aromatic_as_double = False
    else:
        count_aromatic_as_double = bool(args.count_aromatic_as_double)

    canonical_isomeric = bool(args.canonical_isomeric)

    if args.validate_lohc_panel:
        if args.strict_double_only:
            count_aromatic_as_double = False
        else:
            count_aromatic_as_double = bool(args.count_aromatic_as_double)

        canonical_isomeric = bool(args.canonical_isomeric)

        report_path = args.csv_path.with_name(f"{args.csv_path.stem}_report.csv")

        validate_lohc_panel(
            count_aromatic_as_double=count_aromatic_as_double,
            canonical_isomeric=canonical_isomeric,
            report_path=report_path,
        )
        return

    if not args.csv_path.exists():
        raise FileNotFoundError(args.csv_path)

    df = pd.read_csv(args.csv_path)
    if args.col not in df.columns:
        raise ValueError(
            f"Column '{args.col}' not found. Available: {list(df.columns)}"
        )

    require_single_fragment = not bool(args.allow_multifragment)
    require_neutral = not bool(args.allow_charged)

    passes_list = []
    reasons = []
    canon_list = []

    n_frags_list = []
    charge_list = []
    hbd_list = []
    tpsa_list = []
    rotb_list = []

    for smi in df[args.col].astype(str).tolist():
        passes, reason, _mol, can_smi, metrics = evaluate_smiles(
            smi,
            atom_min=args.atom_min,
            atom_max=args.atom_max,
            ring_min=args.ring_min,
            ring_max=args.ring_max,
            count_aromatic_as_double=count_aromatic_as_double,
            require_single_fragment=require_single_fragment,
            require_neutral=require_neutral,
            max_hbd=args.max_hbd,
            max_tpsa=args.max_tpsa,
            min_rotb=args.min_rotb,
            max_rotb=args.max_rotb,
            canonical_isomeric=canonical_isomeric,
        )

        passes_list.append(passes)
        reasons.append(reason)
        canon_list.append(can_smi)

        n_frags_list.append(metrics.get("n_frags"))
        charge_list.append(metrics.get("formal_charge"))
        hbd_list.append(metrics.get("hbd"))
        tpsa_list.append(metrics.get("tpsa"))
        rotb_list.append(metrics.get("rotb"))

    df_out = df.copy()
    df_out["rdkit_canonical_smiles"] = canon_list
    df_out["filter_pass"] = passes_list
    df_out["filter_reason"] = reasons

    df_out["n_frags"] = n_frags_list
    df_out["formal_charge"] = charge_list
    df_out["HBD"] = hbd_list
    df_out["TPSA"] = tpsa_list
    df_out["RotB"] = rotb_list

    if args.replace_col_with_canonical:
        df_out[args.col] = df_out["rdkit_canonical_smiles"]

    filtered_df = df_out[df_out["filter_pass"]].drop(columns=["filter_pass"])
    unfiltered_df = df_out[~df_out["filter_pass"]].drop(columns=["filter_pass"])

    out_dir = args.csv_path.parent
    base = args.csv_path.name
    filtered_path = out_dir / f"filtered_{base}"
    unfiltered_path = out_dir / f"unfiltered_{base}"

    filtered_df.to_csv(filtered_path, index=False)
    unfiltered_df.to_csv(unfiltered_path, index=False)

    print(f"Wrote: {filtered_path}  (n={len(filtered_df)})")
    print(f"Wrote: {unfiltered_path}  (n={len(unfiltered_df)})")
    print("Original CSV unchanged.")


if __name__ == "__main__":
    main()
