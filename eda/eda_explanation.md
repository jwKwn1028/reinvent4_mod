# `csv_eda.py` explanation

This document explains what the `csv_eda.py` script does, how it is structured, what it outputs, and **all CLI options** available.

---

## What the script is for

`csv_eda.py` performs exploratory data analysis (EDA) on a CSV that typically contains:

- A **SMILES** column (default: the first column, unless specified)
- One or more **target/property** columns (optional; can be auto-detected from common names)
- Potentially many other columns from chemical ML/RL logs (numeric and non-numeric)

The script produces:

- A single **JSON report** containing dataset metadata and computed statistics
- Optionally, additional **CSV/PNG artifacts** if `--outdir` is provided
- Optionally, a **compare-to report** if `--compare-to` is provided (chemical-space comparison)

---

## Dependencies and “optional” features

The script is designed to run even if some packages are not installed.

### Required
- `pandas`
- `numpy`

### Optional (enables extra sections)
- **scikit-learn**: correlations, variance threshold, PCA, mutual information, t-SNE (compare-to)
- **RDKit**: SMILES parsing, descriptors, scaffolds, fingerprints, chemical-space comparisons
- **umap-learn**: UMAP embedding (compare-to)

The JSON report includes a `versions` section recording the versions of the available libraries.

---

## Key behaviors and improvements (logic choices)

### 1) Empty strings treated as missing (default)
By default, the script converts empty or whitespace-only cells (e.g. `""`, `" "`) to `NaN`.  
This makes missingness statistics and numeric parsing more consistent.

Disable this with:
- `--no-empty-as-missing`

### 2) Robust numeric column classification
A column is classified as numeric only if it satisfies **all** of:

- **Numeric parse ratio** among non-null values ≥ `--numeric-ratio-min` (default `0.95`)
- **Unique numeric values** ≥ `--numeric-unique-min` (default `10`)
- Not **ID-like**, using heuristics:
  - Column name contains `"id"` (case-insensitive), or
  - A substantial fraction look like leading-zero integers (e.g. `00123`)

This avoids treating SMILES-like columns, IDs, and mixed text columns as numeric.

### 3) Bounded outputs to avoid huge JSON
Some fields can blow up in size on large datasets. The script bounds them:

- `first_column_duplicates` stores **count + examples**, not all duplicates
- Correlation “high pairs” are capped via `--max-high-pairs`
- Correlation matrices can be omitted (`--no-corr-matrices`)
- Correlation columns are limited by `--max-corr-cols` (selected by variance if too many)

### 4) Compare-to mode: chemical space + distribution shifts
When `--compare-to` is used and RDKit is installed, the script:
- Samples molecules (if needed) for speed
- Computes similarity and descriptor-based distribution shift metrics
- Optionally creates plots if `--outdir` is set

---

## Outputs: JSON structure (high-level)

The JSON report contains these top-level keys (most important):

- `headers`: list of column names
- `metadata`: dataset size, missingness, memory, duplicates
- `versions`: library versions (python/numpy/pandas/sklearn/rdkit/umap when available)
- `smiles_column`: which column was used as SMILES
- `targets`: target columns selected (explicit or auto)
- `config`: key thresholds used for numeric detection and correlations
- `column_missing`: per-column missing counts and percentages
- `column_unique`: per-column unique counts and percentages
- `column_mode`: per-column mode value and frequency
- `column_top_values`: top-N values per column (bounded by `--top-n`)
- `column_string_length`: min/max/mean/median string lengths, blank count
- `column_entropy`: entropy of value distribution per column
- `column_numeric_count`: numeric parseable count per column
- `column_numeric_meta`: numeric ratio, unique numeric, ID-like heuristic, and final numeric classification
- `column_stats`: numeric descriptive stats (only for columns classified numeric)
- `first_column_duplicates`: number of duplicated values in the first column + examples
- `correlations`: correlations + high-correlation pairs (guarded/bounded)
- `rdkit`: RDKit-based analysis (if RDKit installed and SMILES column exists)
- `sklearn`: scikit-learn analysis (if installed and numeric columns exist)
- `compare_to`: chemical-space comparisons vs another CSV (if `--compare-to` is used)
- `artifacts`: where helper files were written (if `--outdir` is set)

---

## Single-dataset analysis details

### Column EDA
For each column:
- missing count and percent
- unique count and percent
- mode + frequency
- top-N most frequent values (`--top-n`)
- string length summary
- entropy

### Numeric stats (for numeric-classified columns only)
- min/max/mean/median/std/variance
- quartiles (Q1/Q3), IQR, percentiles (p5/p95)
- skew and kurtosis
- counts of zeros and negatives
- IQR outlier count (1.5×IQR rule)
- z-score outlier count (|z| > 3)

### Correlations
Computed only if:
- there are numeric-classified columns
- enough columns survive filtering (≥2)

Important controls:
- Only columns with at least `--corr-min-count` numeric values are considered.
- If more than `--max-corr-cols` columns qualify, the script selects the top-variance columns.
- Correlation matrices can be stored or omitted.

Outputs:
- Pearson + Spearman matrices (optional)
- Lists of high-correlation pairs (|corr| ≥ `--corr-threshold`)

---

## RDKit analysis details (single dataset)

If RDKit is available and the SMILES column exists:

1. **SMILES parsing summary**
   - total rows considered (may be sampled by `--max-rdkit-mols`)
   - empty counts
   - valid mol count
   - parse failures and exceptions
   - a small list of invalid SMILES examples

2. **Descriptor stats**
   Uses a standard descriptor panel (MolWt, LogP, TPSA, HBD/HBA, RotB, ring counts, etc.):
   - count/min/max/mean/median/std/quantiles per descriptor
   - descriptor failure counts

3. **Scaffolds**
   - Murcko scaffold strings
   - number of unique scaffolds
   - top scaffolds (up to 20)

4. **InChIKey duplicates** (if RDKit build includes InChI support)
   - unique InChIKeys, duplicates estimate, failure counts

5. **Fingerprint diversity (approximate)**
   - samples random fingerprint pairs (`--fp-pair-samples`)
   - reports mean/median Tanimoto and mean (1 − Tanimoto)

---

## Compare-to mode (`--compare-to`) details

Compare-to mode computes chemical-space similarity and distribution shifts between dataset **A** (input CSV) and dataset **B** (compare CSV).

Requirements:
- RDKit must be installed for chemistry-related comparisons.
- scikit-learn needed for PCA/t-SNE sections.
- umap-learn needed for UMAP section.

### 1) Similarity coverage (nearest-neighbor max Tanimoto)
- Creates Morgan fingerprints (radius 2, 2048 bits)
- Computes for each molecule in A: max similarity to B (`A→B`)
- And for each in B: max similarity to A (`B→A`)
- Reports:
  - summary stats
  - coverage fractions at thresholds 0.6/0.7/0.8/0.9
  - histograms
  - optional histogram plot if `--outdir` is set

### 2) Descriptor distribution shifts
Using the same RDKit descriptor panel:
- Wasserstein (1D) distance per descriptor
- KL and JS divergence per descriptor (histogram-based, `--kl-bins`)

### 3) PCA 2D (descriptor panel)
- Standardizes descriptors
- Runs PCA 2D
- Reports explained variance ratio and centroids
- Reports silhouette score in PCA2 space (if enough points)
- If `--outdir` is set:
  - saves `compare_pca2_coords.csv`
  - saves `compare_pca2.png`

### 4) Optional t-SNE 2D (descriptor panel)
Enable with `--tsne`.

- Samples up to `--tsne-sample` points
- Standardizes descriptors
- Clamps perplexity based on sample size to avoid invalid settings
- If `--outdir` is set:
  - saves `compare_tsne2_coords.csv`
  - saves `compare_tsne2.png`

### 5) Optional UMAP 2D (descriptor panel)
Enable with `--umap`.

- Samples up to `--umap-sample` points
- Standardizes descriptors
- Clamps `n_neighbors` to be valid for `n_used`
- If `--outdir` is set:
  - saves `compare_umap2_coords.csv`
  - saves `compare_umap2.png`

### 6) Scaffold overlap
- Computes Murcko scaffolds for A and B
- Reports unique counts and Jaccard overlap

### 7) Target/property shift
For each target present in both datasets:
- Wasserstein (1D)
- KL/JS (histogram-based)
- summary stats for A and B
- optional overlay histogram plot if `--outdir` is set

---

## Artifacts written to `--outdir`

If `--outdir` is provided, the script may create:

- `numeric_column_stats.csv` (single dataset numeric stats table)
- `top_scaffolds.csv` (top scaffolds from RDKit analysis)
- `hist_<target>.png` (single dataset target histograms)
- Compare-to plots and coordinate CSVs:
  - `compare_pca2_coords.csv`, `compare_pca2.png`
  - `compare_tsne2_coords.csv`, `compare_tsne2.png` (if `--tsne`)
  - `compare_umap2_coords.csv`, `compare_umap2.png` (if `--umap`)
  - `compare_tanimoto_maxsim_hist.png`

The JSON report stores the outdir path in `artifacts["outdir"]`.

---

## CLI usage

### Basic forms

Single dataset:
```bash
python csv_eda.py INPUT.csv OUTPUT.json
```

If you omit `OUTPUT.json`, it defaults to:
- `eda_<input_basename>.json`

Compare-to mode:
```bash
python csv_eda.py A.csv eda_A.json --compare-to B.csv --outdir eda_out
```

---

## All CLI options

### Positional arguments

- `input_csv`  
  Input CSV path (required)

- `output_json`  
  Output JSON path (optional).  
  Default: `eda_<input_basename>.json`

---

### SMILES / targets

- `--smiles-column <NAME>`  
  Column name treated as SMILES.  
  Default: first column.

- `--target <NAME>`  
  Single target column name (can be combined with `--targets`).

- `--targets "<A,B,C>"` (repeatable)  
  Target column(s). You can:
  - repeat `--targets` multiple times, or
  - provide a comma-separated list in one flag  
  Example:
  ```bash
  --targets "Score,LogD (Lipo) (raw)" --targets "Water Solubility (ESOL) (raw)"
  ```

If neither `--target` nor `--targets` is provided, the script attempts auto-detection using a built-in candidate list.

---

### Column EDA controls

- `--top-n <INT>`  
  Number of most frequent values stored per column.  
  Default: `5`

- `--corr-threshold <FLOAT>`  
  Absolute correlation threshold for including a pair in `high_pairs_*`.  
  Default: `0.9`

---

### Output artifacts

- `--outdir <PATH>`  
  If set, write helper CSVs and plots to this directory.

---

### RDKit sampling and fingerprint settings

- `--max-rdkit-mols <INT>`  
  Maximum SMILES rows parsed with RDKit (sampling is used if dataset is larger).  
  Default: `50000`

- `--fp-pair-samples <INT>`  
  Number of random fingerprint pairs sampled for diversity estimate.  
  Default: `20000`

- `--seed <INT>`  
  Random seed for sampling, fingerprint pair selection, embeddings.  
  Default: `0`

---

### Compare-to mode

- `--compare-to <PATH>`  
  Second CSV file to compare against.  
  Default: none (disabled)

---

### Cleaning and numeric detection

- `--no-empty-as-missing`  
  Do not convert empty/whitespace strings to NaN.

- `--numeric-ratio-min <FLOAT>`  
  Minimum numeric parse ratio among non-null values for a column to be classified as numeric.  
  Default: `0.95`

- `--numeric-unique-min <INT>`  
  Minimum number of unique numeric values for a column to be classified as numeric.  
  Default: `10`

- `--corr-min-count <INT>`  
  Minimum numeric count required for a numeric column to be included in correlation computations.  
  Default: `50`

- `--max-corr-cols <INT>`  
  Maximum number of columns used in correlation matrices and high-pair detection.  
  If exceeded, columns are selected by highest variance.  
  Default: `200`

- `--no-corr-matrices`  
  Do not store full Pearson/Spearman matrices in JSON; still stores high-correlation pairs.

- `--max-high-pairs <INT>`  
  Maximum number of high-correlation pairs to store per correlation type.  
  Default: `10000`

---

### Distribution divergence bins

- `--kl-bins <INT>`  
  Number of histogram bins used for KL/JS divergence computations.  
  Default: `50`

---

### t-SNE (compare-to only)

- `--tsne`  
  Enable t-SNE 2D embedding on descriptor panel in compare-to mode.

- `--tsne-sample <INT>`  
  Max points used for t-SNE (sampled if larger).  
  Default: `5000`

---

### UMAP (compare-to only)

- `--umap`  
  Enable UMAP 2D embedding on descriptor panel in compare-to mode.

- `--umap-sample <INT>`  
  Max points used for UMAP (sampled if larger).  
  Default: `20000`

- `--umap-n-neighbors <INT>`  
  UMAP `n_neighbors`. Will be clamped to be valid for the sampled set size.  
  Default: `15`

- `--umap-min-dist <FLOAT>`  
  UMAP `min_dist`.  
  Default: `0.1`

- `--umap-metric <STR>`  
  UMAP metric (e.g., `euclidean`, `cosine`).  
  Default: `euclidean`

---

## Practical recommendations

- For very large CSVs:
  - Keep `--max-rdkit-mols` at a manageable number (e.g., 50k or less)
  - Consider `--no-corr-matrices` to keep JSON smaller
  - Use `--max-corr-cols` to cap correlation cost

- For compare-to chemical-space checks:
  - Start with PCA + Tanimoto coverage (fast, interpretable)
  - Add `--umap` for a more nonlinear “chemical space” view
  - Use `--tsne` only for smaller samples (it is slower)

---

## Troubleshooting notes (typing / mypy)

If you run mypy and see messages like:
- stubs not installed for pandas
- sklearn/rdkit/umap missing stubs

Common fix for pandas:
```bash
python -m pip install pandas-stubs
```

For sklearn/rdkit/umap, it is common to ignore missing imports in `mypy.ini`, or use `type: ignore` on those imports.

