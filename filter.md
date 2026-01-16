# REINVENT4 Data Pipeline: Filter Configuration Deep Dive

The `[filter]` section in your configuration TOML file is the gatekeeper for your dataset. It determines which molecules are "clean" enough to enter the training process. 

The REINVENT4 pipeline processes data in two distinct stages:

1.  **Fast Regex Filtering:** A high-speed, text-based pass that discards invalid strings before any expensive chemical parsing occurs.
2.  **RDKit Chemistry Filtering:** A rigorous chemical validation, standardization, and normalization pass using the RDKit library.

---

## 1. Filter Parameters & Logic

### A. Element & Composition Filters (Regex Stage)

These filters run *first* on the raw SMILES string. If a molecule fails here, it never touches RDKit, saving significant computational time.

| Parameter | Type | Default | Description & Implications |
| :--- | :--- | :--- | :--- |
| **`elements`** | `list[str]` | `[]` | **Critical:** Defines the *allowed vocabulary* of elements. <br>• Default set: `C, O, N, S, F, Cl, Br, I` <br>• Your list is **added** to the default set. <br>• **Example:** `["P", "B"]` enables Phosphorus and Boron. <br>• **Note:** Any SMILES token containing an element *not* in this combined list will cause the *entire molecule* to be discarded immediately. |
| **`min_heavy_atoms`** | `int` | `2` | Minimum count of non-hydrogen atoms. Prevents training on trivial fragments like methane or single ions. |
| **`max_heavy_atoms`** | `int` | `90` | Maximum count of non-hydrogen atoms. Limits the size of the model's vocabulary and context window requirements. |
| **`min_carbons`** | `int` | `2` | Ensures organic chemistry relevance. Discards inorganic salts or simple hydrides. |
| **`max_mol_weight`** | `float` | `1200.0` | Maximum molecular weight (Daltons). Useful for filtering out large biomolecules or polymers. |
| **`keep_isotope_molecules`** | `bool` | `True` | • `True`: Accepts SMILES with isotope notation (e.g., `[13C]`). <br>• `False`: Discards any SMILES containing isotope numbers. Use this if your model shouldn't learn specific isotopes. |

### B. Structural Topology Filters (RDKit Stage)

These checks run after RDKit has successfully parsed the molecule.

| Parameter | Type | Default | Description & Implications |
| :--- | :--- | :--- | :--- |
| **`max_num_rings`** | `int` | `12` | Filters out complex polycyclic systems that might be artifacts or irrelevant for drug-like space. |
| **`max_ring_size`** | `int` | `7` | Filters out macrocycles (rings > 7 atoms). <br>• **Important:** If your target domain involves macrocycles, **increase this value**. |

---

## 2. Normalization & Standardization (RDKit Stage)

This is where the chemical representation is "cleaned" to ensure consistency. The pipeline executes these steps in a specific order:

1.  **Fragment Chooser:** If the SMILES contains multiple disconnected fragments (e.g., salts like `CC(=O)O.[Na+]`), REINVENT selects the **largest organic fragment** and discards the rest.
2.  **Normalization:** Applies reaction-transform rules to standardize functional groups.
3.  **Uncharging:** Neutralizes charges where possible.
4.  **Tautomerization:** (Optional) Canonicalizes tautomers.

| Parameter | Type | Default | Detailed Description |
| :--- | :--- | :--- | :--- |
| **`transforms`** | `list[str]` | `["standard"]` | Selects which normalization rulesets to apply. <br>• **`"standard"`**: Applies ~20 RDKit rules (e.g., Nitro to `N+(O-)=O`, Sulfone to `S(=O)(=O)`). <br>• **`"four_valent_nitrogen"`**: A custom rule to fix specific nitrogen valence issues. <br>• You can combine them: `["standard", "four_valent_nitrogen"]`. |
| **`uncharge`** | `bool` | `True` | • `True`: Neutralizes the molecule (adds/removes protons) and then re-ionizes acids/bases to a standard pH-independent state using RDKit's `Uncharger`. <br>• **Recommendation:** Keep `True` for consistent generative outputs. |
| **`canonical_tautomer`** | `bool` | `False` | • `True`: Computes the single canonical tautomer for the molecule. <br>• **Performance Warning:** This is computationally **expensive**. Enabling this on a 2M dataset will significantly increase preprocessing time. Only use if tautomeric consistency is critical for your model. |
| **`inchi_key_deduplicate`** | `bool` | `False` | • `True`: Converts molecules to InChI Keys to find duplicates. This is more robust than SMILES string matching because different SMILES can represent the same molecule. <br>• **Recommendation:** `True` is highly recommended for high-quality datasets. |

### C. Output Formatting

Controls how the final "clean" SMILES string is written to the output file.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| **`keep_stereo`** | `bool` | `True` | • `True`: Preserves stereochemical information (`@`, `@@`, `/`, `\`). <br>• `False`: Strips all stereo tags. Use this if you want to train a model purely on 2D topology. |
| **`kekulize`** | `bool` | `False` | • `True`: Writes SMILES with explicit single/double bonds (`C=C-C=C`) instead of aromatic notation (`c1ccccc1`). <br>• **Note:** Most modern transformers prefer aromatic SMILES, so `False` is usually better. |
| **`randomize_smiles`** | `bool` | `False` | • `True`: Generates a random traversal of the molecular graph. <br>• **Use Case:** Primarily for data augmentation *during* training. For static dataset preparation, `False` (canonical) is standard. |

---

## 3. Debugging & Logging

| Parameter | Type | Default | Usage |
| :--- | :--- | :--- | :--- |
| **`report_errors`** | `bool` | `False` | • `True`: Logs specific error messages from RDKit (e.g., "AtomValenceException", "KekulizeException") for every molecule that fails. <br>• **Warning:** Can generate huge log files if the dataset is dirty. Use on a small subset first to debug data quality. |

## 4. Full Example Configuration

This example setup is robust for a standard "Drug-Like" small molecule dataset, adding support for Phosphorus and ensuring high data quality.

```toml
[filter]
# 1. ALLOWED VOCABULARY
# Add Phosphorus to default set (C, O, N, S, F, Cl, Br, I)
elements = ["P"]

# 2. SIZE & WEIGHT LIMITS
min_heavy_atoms = 5
max_heavy_atoms = 60
min_carbons = 3
max_mol_weight = 900.0

# 3. TOPOLOGY LIMITS
max_num_rings = 10
max_ring_size = 7       # Discard macrocycles > 7 atoms

# 4. NORMALIZATION FLOW
# Apply standard RDKit normalizations
transforms = ["standard"]
# Remove salts, keep largest fragment, then neutralize charges
uncharge = true
# Robust deduplication
inchi_key_deduplicate = true
# Skip expensive tautomerization for speed
canonical_tautomer = false

# 5. OUTPUT FORMAT
keep_stereo = true      # Keep isomerism
kekulize = false        # Use aromatic notation
randomize_smiles = false # Output canonical SMILES
report_errors = true    # Help me see why molecules are failing
```