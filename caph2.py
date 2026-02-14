import csv
import sys
import logging
from collections import Counter

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from tqdm import tqdm


# =========================
# Config
# =========================
OUTPUT_FILE = "CapH2_selections.csv"
INPUT_FILES = [f"selection_canonical{i}.txt" for i in range(1, 6)]

# Log only the first N examples per failure type (still count all failures)
FAILURE_LOG_LIMIT_PER_TYPE = 50
LOG_FILE = "capH2_failures.log"


# =========================
# Logging (implements #4)
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
    ],
)

_failure_counts = Counter()
_failure_logged = Counter()


def log_failure(kind: str, smiles: str, err: Exception):
    """Count failures and log only a limited number per kind."""
    _failure_counts[kind] += 1
    if _failure_logged[kind] < FAILURE_LOG_LIMIT_PER_TYPE:
        logging.warning("Failed (%s): %s | %s", kind, smiles, repr(err))
        _failure_logged[kind] += 1


# =========================
# Core chemistry
# =========================
def pred_rich_form_and_changed_bonds(poor_mol: Chem.Mol):
    """
    Create "rich" form by:
      - Kekulize
      - Turn DOUBLE bonds in rings into SINGLE bonds

    Returns:
      rich_mol, rich_smiles, n_changed_bonds

    Notes:
      - This matches your original logic, but avoids SMILES round-trips (implements #5).
      - If you want aromatic-only hydrogenation, change the bond-selection logic accordingly.
    """
    try:
        temp = Chem.Mol(poor_mol)  # copy
        Chem.Kekulize(temp, clearAromaticFlags=True)  # in-place on copy
    except Exception as e:
        raise RuntimeError(f"Kekulize failed: {e}") from e

    rw = Chem.RWMol(temp)
    n_changed = 0

    for bond in rw.GetBonds():
        if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE and bond.IsInRing():
            bond.SetBondType(Chem.rdchem.BondType.SINGLE)
            n_changed += 1

    rich_mol = rw.GetMol()

    try:
        Chem.SanitizeMol(rich_mol)  # ensures valence/Hs are consistent
    except Exception as e:
        raise RuntimeError(f"Sanitize rich form failed: {e}") from e

    rich_smi = Chem.MolToSmiles(rich_mol, canonical=True)
    return rich_mol, rich_smi, n_changed


def calc_capH2_with_diagnostics(poor_smi: str):
    """
    Compute CapH2 (%) and diagnostics (implements #1).

    Returns:
      dict of outputs, or None if failed
    """
    poor_smi = poor_smi.strip()
    if not poor_smi:
        return None

    try:
        poor_mol = Chem.MolFromSmiles(poor_smi)
        if poor_mol is None:
            raise ValueError("MolFromSmiles returned None")

        rich_mol, rich_smi, n_changed = pred_rich_form_and_changed_bonds(poor_mol)

        mw_poor = Descriptors.MolWt(poor_mol)
        mw_rich = Descriptors.MolWt(rich_mol)
        if mw_rich <= 0:
            raise ValueError(f"Invalid rich MolWt: {mw_rich}")

        capH2 = (mw_rich - mw_poor) / mw_rich * 100.0

        # Diagnostics (#1)
        diag = {
            "SMILES_poor": poor_smi,
            "CapH2_%": round(capH2, 2),
            "MolWt_poor": round(mw_poor, 4),
            "MolWt_rich": round(mw_rich, 4),
            "dMolWt": round(mw_rich - mw_poor, 4),
            "n_ring_double_to_single": n_changed,
            "NumHeavyAtoms": rdMolDescriptors.CalcNumHeavyAtoms(poor_mol),
            "NumRings": rdMolDescriptors.CalcNumRings(poor_mol),
            "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings(poor_mol),
            "SMILES_rich": rich_smi,
        }
        return diag

    except Exception as e:
        # Log failures (#4)
        log_failure("capH2_calc", poor_smi, e)
        return None


# =========================
# Main: stream processing (also helps speed/memory)
# =========================
HEADER = [
    "SMILES_poor",
    "CapH2_%",
    "MolWt_poor",
    "MolWt_rich",
    "dMolWt",
    "n_ring_double_to_single",
    "NumHeavyAtoms",
    "NumRings",
    "NumAromaticRings",
    "SMILES_rich",
]

# Write header
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as out_csv:
    writer = csv.DictWriter(out_csv, fieldnames=HEADER)
    writer.writeheader()

# Process each file
for input_file in INPUT_FILES:
    logging.info("Processing %s ...", input_file)

    try:
        with open(input_file, "r", encoding="utf-8") as infile, open(
            OUTPUT_FILE, "a", newline="", encoding="utf-8"
        ) as out_csv:
            writer = csv.DictWriter(out_csv, fieldnames=HEADER)

            # Stream line-by-line (no full list; improves speed/memory)
            for line in tqdm(infile, desc=input_file):
                smi = line.strip()
                if not smi:
                    continue

                row = calc_capH2_with_diagnostics(smi)
                if row is not None:
                    writer.writerow(row)

    except FileNotFoundError:
        logging.warning("File not found: %s (skipping)", input_file)
    except Exception as e:
        log_failure("file_processing", input_file, e)

# Summary
logging.info("Done. Output: %s", OUTPUT_FILE)
if _failure_counts:
    logging.info("Failure counts (all): %s", dict(_failure_counts))
    logging.info("Failure log (sampled) written to: %s", LOG_FILE)
else:
    logging.info("No failures logged.")
