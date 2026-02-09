"""Count number of Ketones"""

__all__ = ["NKetones"]

from dataclasses import dataclass
from typing import List
import numpy as np
from rdkit import Chem

from .component_results import ComponentResults
from reinvent_plugins.mol_cache import molcache
from .add_tag import add_tag
from reinvent_plugins.normalize import normalize_smiles

@add_tag("__parameters")
@dataclass
class Parameters:
    """Parameters for the NKetones component."""
    pass

@add_tag("__component")
class NKetones:
    def __init__(self, params: Parameters):
        self.smiles_type = "rdkit_smiles"
        self.ketone_pattern = Chem.MolFromSmarts('[CX3](=O)([#6])[#6]')

    @normalize_smiles
    def __call__(self, smiles: List[str]) -> ComponentResults:
        scores = []
        for smi in smiles:
            # normalize_smiles decorator handles normalization, but we might receive None or invalid strings if something fails before?
            # actually normalize_smiles ensures we get strings, but let's be safe.
            mol = Chem.MolFromSmiles(smi) if smi else None
            
            if mol:
                matches = mol.GetSubstructMatches(self.ketone_pattern)
                scores.append(len(matches))
            else:
                scores.append(0)
        
        return ComponentResults([np.array(scores, dtype=float)])