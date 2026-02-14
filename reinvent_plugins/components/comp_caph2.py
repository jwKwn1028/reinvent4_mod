"""
CapH2 scoring component.

Calculates the gravimetric hydrogen capacity (CapH2) of a molecule.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from pydantic import Field

from reinvent_plugins.components.add_tag import add_tag
from reinvent_plugins.components.component_results import ComponentResults
from reinvent_plugins.normalize import normalize_smiles

# --- Logic adapted from caph2.py ---



# def pred_rich_form_and_changed_bonds(poor_mol: Chem.Mol):
#     """
#     Create "rich" form by:
#       - Kekulize
#       - Turn DOUBLE bonds in rings into SINGLE bonds

#     Returns:
#       rich_mol, rich_smiles, n_changed_bonds
#     """
#     try:
#         temp = Chem.Mol(poor_mol)  # copy
#         Chem.Kekulize(temp, clearAromaticFlags=True)  # in-place on copy
#     except Exception as e:
#         raise RuntimeError(f"Kekulize failed: {e}") from e

#     rw = Chem.RWMol(temp)
#     n_changed = 0

#     for bond in rw.GetBonds():
#         if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE and bond.IsInRing():
#             bond.SetBondType(Chem.rdchem.BondType.SINGLE)
#             n_changed += 1

#     rich_mol = rw.GetMol()

#     try:
#         Chem.SanitizeMol(rich_mol)  # ensures valence/Hs are consistent
#     except Exception as e:
#         raise RuntimeError(f"Sanitize rich form failed: {e}") from e

#     rich_smi = Chem.MolToSmiles(rich_mol, canonical=True)
#     return rich_mol, rich_smi, n_changed



def pred_rich_form_and_changed_bonds(poor_mol: Chem.Mol):
    aromatic_bond_ids = set()
    for b in poor_mol.GetBonds():
        if b.GetIsAromatic() and b.IsInRing():
            aromatic_bond_ids.add(b.GetIdx())

    temp = Chem.Mol(poor_mol)
    Chem.Kekulize(temp, clearAromaticFlags=True)

    rw = Chem.RWMol(temp)
    n_changed = 0
    for b in rw.GetBonds():
        if b.GetIdx() in aromatic_bond_ids and b.GetBondType() == Chem.rdchem.BondType.DOUBLE:
            b.SetBondType(Chem.rdchem.BondType.SINGLE)
            n_changed += 1

    rich_mol = rw.GetMol()
    Chem.SanitizeMol(rich_mol)
    rich_smi = Chem.MolToSmiles(rich_mol, canonical=True)
    return rich_mol, rich_smi, n_changed



def calc_capH2(poor_smi: str) -> float:
    """
    Compute CapH2 (%).
    Returns the CapH2 value or NaN if calculation fails.
    """
    poor_smi = (poor_smi or "").strip()
    if not poor_smi:
        return float("nan")

    try:
        poor_mol = Chem.MolFromSmiles(poor_smi)
        if poor_mol is None:
            return float("nan")

        rich_mol, rich_smi, n_changed = pred_rich_form_and_changed_bonds(poor_mol)

        mw_poor = Descriptors.MolWt(poor_mol)
        mw_rich = Descriptors.MolWt(rich_mol)
        if mw_rich <= 0:
            return float("nan")

        capH2 = (mw_rich - mw_poor) / mw_rich * 100.0
        return capH2

    except Exception:
        return float("nan")


# --- Component Definition ---

@add_tag("__parameters")
@dataclass
class Parameters:
    """Parameters for the CapH2 component. No parameters needed."""
    pass

@add_tag("__component")
class CompCapH2:
    """
    CapH2 scoring component.
    """

    def __init__(self, params: Parameters):
        self.smiles_type = "rdkit_smiles"
        pass

    @normalize_smiles
    def __call__(self, smiles: List[str]) -> ComponentResults:
        scores = []
        for smi in smiles:
            score = calc_capH2(smi)
            scores.append(score)
        
        return ComponentResults([np.array(scores, dtype=float)])