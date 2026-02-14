"""Count number of Ketones"""

__all__ = ["NKetones"]

from typing import List
import numpy as np
from rdkit import Chem
from pydantic.dataclasses import dataclass
from .component_results import ComponentResults
from reinvent_plugins.mol_cache import molcache
from .add_tag import add_tag

@add_tag("__parameters")
@dataclass
class Parameters:
    pass

@add_tag("__component")
class NKetones:
    def __init__(self, params: Parameters):
        self.ketone_pattern = Chem.MolFromSmarts('[CX3](=O)([#6])[#6]')

    @molcache
    def __call__(self, mols: List[Chem.Mol]) -> np.array:
        scores = []
        for mol in mols:
            if mol:
                matches = mol.GetSubstructMatches(self.ketone_pattern)
                scores.append(len(matches))
            else:
                scores.append(0)
        
        return ComponentResults([np.array(scores, dtype=float)])
