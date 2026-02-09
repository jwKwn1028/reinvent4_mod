# `filter_molecules.py` — RDKit CSV Filter + Canonicalization

This script reads SMILES from a CSV column, computes RDKit-derived properties, applies a set of structural and polarity/packing filters, and writes two new CSV files:

- `filtered_{original}.csv` — rows that **pass** all enabled filters
- `unfiltered_{original}.csv` — rows that **fail** at least one filter

The original input CSV is not modified.

---

## What the script does

### 1) Input
- Reads a CSV file given by `csv_path`
- Extracts SMILES strings from the column specified by `--col` (default: `molecules`)

### 2) Canonicalization (RDKit)
For each valid SMILES:
- Parses it into an RDKit `Mol`
- Writes RDKit canonical SMILES into a new column:
  - `rdkit_canonical_smiles`
- If `--canonical-isomeric` is set, stereo information is included (when present).
- If `--replace-col-with-canonical` is set, the original SMILES column is replaced in the output files with the canonical SMILES.

> Note: canonicalization is used for **standardizing output**; filtering is based on the RDKit `Mol` object.

### 3) Filters applied

The script applies filters in this order (so the `filter_reason` tells you the first failing condition):

#### A. “MP/LOHC-oriented” filters (enabled by default)
These filters are intended to remove salts/mixtures and very polar / strongly H-bonding molecules, and to bias away from highly symmetric, crystalline packers.

1. **Single fragment**
   - Default: required (`n_frags == 1`)
   - Multi-fragment molecules (e.g., salts/mixtures) are often written as `A.B` in SMILES and typically represent ionic pairs or mixtures.

2. **Neutral formal charge**
   - Default: required (`formal_charge == 0`)

3. **HBD (Hydrogen Bond Donors)**
   - Default: `HBD <= 1`
   - Computed via `Lipinski.NumHDonors(mol)`

4. **TPSA**
   - Default: `TPSA <= 60.0 Å²`
   - Computed via `rdMolDescriptors.CalcTPSA(mol)`

5. **Rotatable bonds**
   - Default: `2 <= RotB <= 12`
   - Computed via `Lipinski.NumRotatableBonds(mol)`

6. **Symmetry proxy**
   - Default: enabled as a **filter** (`symmetry_mode=filter`)
   - Proxy: `1 - (#unique canonical ranks / #atoms)`
     - Uses `Chem.CanonicalRankAtoms(mol, breakTies=False)`
     - Higher values imply more graph-equivalent atoms (more symmetry), which tends to correlate with easier packing/crystallization.
   - Default threshold: `<= 0.25`
   - Modes:
     - `filter`: exclude if above threshold
     - `flag`: keep, but add `symmetry_high_flag` column
     - `off`: ignore symmetry proxy

#### B. Structural filters (your original conditions)
1. **Atom count**
   - Must be within `[atom_min, atom_max]` (defaults: 1–100)
   - Uses `mol.GetNumAtoms()` (explicit atoms; typically heavy atoms for SMILES)

2. **Ring count**
   - Must be within `[ring_min, ring_max]` (defaults: 1–5)
   - Uses `rdMolDescriptors.CalcNumRings(mol)` (RDKit ring perception)

3. **At least one ring contains a “double-like” bond**
   - A “double-like” bond is:
     - an explicit `DOUBLE` bond, OR
     - (optionally) an aromatic bond, if aromatic counting is enabled
   - This is controlled by:
     - `--count-aromatic-as-double` (aromatic bonds count)
     - `--strict-double-only` (only explicit double bonds count; aromatics do not)

---

## Output columns

In addition to your original CSV columns, the outputs include:

### Always added
- `rdkit_canonical_smiles` — canonical SMILES (or `None` if invalid)
- `filter_reason` — first failing filter reason, or `"OK"` if passed
- `n_frags` — number of fragments
- `formal_charge` — formal charge of the molecule
- `HBD` — number of hydrogen bond donors
- `TPSA` — topological polar surface area (Å²)
- `RotB` — number of rotatable bonds
- `symmetry_proxy` — symmetry proxy value

### Conditionally added
- `symmetry_high_flag` — only if `--symmetry-mode flag`

---

## Failure reasons (`filter_reason`)

Examples you may see:

- `empty_smiles`
- `invalid_smiles`
- `not_single_fragment(n_frags=2)`
- `nonzero_formal_charge(1)`
- `hbd_out_of_range(2)`
- `tpsa_out_of_range(73.2)`
- `rotb_out_of_range(0)`
- `symmetry_proxy_high(0.312)`
- `atom_count_out_of_range(142)`
- `ring_count_out_of_range(0)`
- `no_double_bond_in_any_ring`

`filter_reason` stops at the **first** failing condition in the evaluation order.

---

## CLI options (complete list)

### Positional argument
- `csv_path`
  - Path to input CSV file.

### Input
- `--col`
  - **Default:** `molecules`
  - CSV column containing SMILES strings.

### Structural filters
- `--atom-min`
  - **Default:** `1`
- `--atom-max`
  - **Default:** `100`
- `--ring-min`
  - **Default:** `1`
- `--ring-max`
  - **Default:** `5`

### Ring “double-like bond” behavior
- `--count-aromatic-as-double`
  - If set: aromatic bonds are treated as satisfying the “double-like bond in ring” requirement.
- `--strict-double-only`
  - If set: only explicit double bonds satisfy the “double-like bond in ring” requirement (aromatic bonds do not).

> If both are omitted: only explicit `DOUBLE` bonds count (unless you set `--count-aromatic-as-double`).

### Canonicalization options
- `--canonical-isomeric`
  - If set: include stereochemistry in canonical SMILES.
- `--replace-col-with-canonical`
  - If set: replace the input SMILES column (`--col`) with the canonical SMILES in output files.

### Added MP/LOHC-oriented filters
- `--allow-multifragment`
  - If set: allow multi-fragment molecules (default is to require single fragment).
- `--allow-charged`
  - If set: allow non-neutral molecules (default is to require formal charge == 0).
- `--max-hbd`
  - **Default:** `1`
  - Maximum allowed number of H-bond donors.
- `--max-tpsa`
  - **Default:** `60.0`
  - Maximum allowed TPSA (Å²). (Set to `50.0` for a tighter screen.)
- `--min-rotb`
  - **Default:** `2`
  - Minimum allowed rotatable bonds.
- `--max-rotb`
  - **Default:** `12`
  - Maximum allowed rotatable bonds.
- `--symmetry-mode`
  - **Choices:** `filter`, `flag`, `off`
  - **Default:** `filter`
  - How to handle high symmetry:
    - `filter`: exclude if symmetry proxy exceeds threshold
    - `flag`: keep but mark `symmetry_high_flag`
    - `off`: ignore symmetry proxy
- `--symmetry-threshold`
  - **Default:** `0.25`
  - Threshold for symmetry proxy used in `filter` or `flag` modes.

### Validation / diagnostic mode
- `--validate-lohc-panel`
  - Prints a built-in LOHC scaffold report (rings, TPSA, HBD, rotatable bonds, symmetry proxy, etc.) and exits.
  - Useful for sanity-checking ring perception and “double-like bond” behavior.

---

## Example usage

### Typical LOHC-lean screen (aromatics count as “double-like”)
```bash
python filter_molecules.py data.csv \
  --col molecules \
  --count-aromatic-as-double \
  --canonical-isomeric
