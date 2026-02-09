"""
LOHC Filter component.

Scores molecules based on a set of filters defined in eda/lohc_filter.py.
Returns 1.0 if the molecule passes all filters, -1.0 otherwise.
"""

from pydantic.dataclasses import dataclass
from typing import List
import numpy as np
from rdkit import Chem
from rdkit.Chem import Lipinski, rdMolDescriptors
from pydantic import Field

from reinvent_plugins.components.add_tag import add_tag
from reinvent_plugins.components.component_results import ComponentResults
from reinvent_plugins.normalize import normalize_smiles

# --- Logic adapted from eda/lohc_filter.py ---

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

def evaluate_smiles(
    smiles: str,
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
) -> bool:
    """
    Returns True if smiles passes all filters, False otherwise.
    """
    s = (smiles or "").strip()
    if not s:
        return False

    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return False

    # --- Metrics used by additional filters ---
    n_frags = len(Chem.GetMolFrags(mol))
    fcharge = Chem.GetFormalCharge(mol)
    hbd = Lipinski.NumHDonors(mol)
    tpsa = float(rdMolDescriptors.CalcTPSA(mol))
    rotb = Lipinski.NumRotatableBonds(mol)

    # --- Additional filters (MP/LOHC-oriented) ---
    if require_single_fragment and n_frags != 1:
        return False

    if require_neutral and fcharge != 0:
        return False

    if hbd > max_hbd:
        return False

    if tpsa > max_tpsa:
        return False

    if not (min_rotb <= rotb <= max_rotb):
        return False

    # --- Existing filters ---
    n_atoms = mol.GetNumAtoms()
    if not (atom_min <= n_atoms <= atom_max):
        return False

    n_rings = rdMolDescriptors.CalcNumRings(mol)
    if not (ring_min <= n_rings <= ring_max):
        return False

    if not ring_has_double_like_bond(mol, count_aromatic_as_double=count_aromatic_as_double):
        return False

    return True

# --- Component Definition ---

@add_tag("__parameters")
@dataclass
class Parameters:
    """Parameters for the LOHC Filter component."""
    
    atom_min: List[int] = Field(default_factory=lambda: [1])
    atom_max: List[int] = Field(default_factory=lambda: [100])
    ring_min: List[int] = Field(default_factory=lambda: [1])
    ring_max: List[int] = Field(default_factory=lambda: [9])
    
    count_aromatic_as_double: List[bool] = Field(default_factory=lambda: [True])
    
    # Note: Logic inverted from CLI args (allow vs require) to match function signature
    require_single_fragment: List[bool] = Field(default_factory=lambda: [True])
    require_neutral: List[bool] = Field(default_factory=lambda: [True])
    
    max_hbd: List[int] = Field(default_factory=lambda: [1])
    max_tpsa: List[float] = Field(default_factory=lambda: [50.0])
    min_rotb: List[int] = Field(default_factory=lambda: [0])
    max_rotb: List[int] = Field(default_factory=lambda: [9])


@add_tag("__component")
class CompLOHCFilter:
    """
    LOHC Filter component.
    Returns 1.0 if pass, -1.0 if fail.
    """

    def __init__(self, params: Parameters):
        self.smiles_type = "rdkit_smiles"
        self.atom_min = params.atom_min[0]
        self.atom_max = params.atom_max[0]
        self.ring_min = params.ring_min[0]
        self.ring_max = params.ring_max[0]
        self.count_aromatic_as_double = params.count_aromatic_as_double[0]
        self.require_single_fragment = params.require_single_fragment[0]
        self.require_neutral = params.require_neutral[0]
        self.max_hbd = params.max_hbd[0]
        self.max_tpsa = params.max_tpsa[0]
        self.min_rotb = params.min_rotb[0]
        self.max_rotb = params.max_rotb[0]

    @normalize_smiles
    def __call__(self, smiles: List[str]) -> ComponentResults:
        scores = []
        for smi in smiles:
            passes = evaluate_smiles(
                smi,
                atom_min=self.atom_min,
                atom_max=self.atom_max,
                ring_min=self.ring_min,
                ring_max=self.ring_max,
                count_aromatic_as_double=self.count_aromatic_as_double,
                require_single_fragment=self.require_single_fragment,
                require_neutral=self.require_neutral,
                max_hbd=self.max_hbd,
                max_tpsa=self.max_tpsa,
                min_rotb=self.min_rotb,
                max_rotb=self.max_rotb,
            )
            scores.append(1.0 if passes else 0.0)
        
        return ComponentResults([np.array(scores, dtype=float)])
