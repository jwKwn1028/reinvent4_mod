# Reinvent 4 Training Tutorial

## Training Time Estimation

Based on a test run with a subset of the data, here is the estimated training time for the full 2 million molecule dataset.

**Test Run Details:**
- **Dataset Size:** ~957 molecules (1,000 lines from `data/2M.csv`, heavily filtered by `standardize=true`)
- **Epochs:** 2
- **Total Time (real):** ~15.5 seconds (including startup and overhead)
- **Training Loop Time (approx):** ~11 seconds (from progress bar: 00:11)
- **Time per Epoch:** ~5.5 seconds for ~957 molecules
- **Hardware:** CPU (x86_64)

**Extrapolation for 2 Million Molecules:**
- **Scaling Factor:** 2,000,000 / 957 ≈ 2090
- **Estimated Time per Epoch:** 5.5 seconds * 2090 ≈ 11,495 seconds ≈ 3.2 hours
- **Total Time (for 100 epochs, typical):** ~320 hours (approx. 13 days) on a CPU.

**Note:** This is a rough estimation. Training on a GPU would be significantly faster (potentially 10-50x faster), reducing the time to a few hours or a day.

## Loss Function

The loss function used for training the Reinvent Prior model is the **Negative Log Likelihood (NLL) Loss**.

- **Implementation:** `torch.nn.NLLLoss`
- **Location:** `reinvent/models/reinvent/models/model.py`
- **Code Snippet:**
  ```python
  self._nll_loss = tnn.NLLLoss(reduction="none")
  ```
- **Usage:** In the `likelihood` method of the `Model` class:
  ```python
  logits, _ = self.network(sequences[:, :-1])
  log_probs = logits.log_softmax(dim=2)
  return self._nll_loss(log_probs.transpose(1, 2), sequences[:, 1:]).sum(dim=1)
  ```

## Adjustable Hyperparameters

When configuring the model in the `.toml` file, you can adjust several parameters in the `[network]` section.

**Standard Parameters:**
- `num_layers` (int): Number of LSTM/GRU layers (default: 3).
- **Location in Code:** Defined in `reinvent/runmodes/create_model/create_reinvent.py` and passed to `reinvent/models/reinvent/models/model.py`.
- `layer_size` (int): Number of units in each hidden layer (default: 512).
- `dropout` (float): Dropout probability to prevent overfitting (default: 0.0).
- `layer_normalization` (bool): Whether to apply layer normalization (default: false).
- `standardize` (bool): Whether to standardize SMILES strings (default: true).
- **Chemistry Logic Location:** Standardization filters and logic are found in `reinvent/chemistry/standardization/`, specifically `rdkit_standardizer.py` and `filter_registry.py`.

**Additional Parameters:**
- `cell_type` (str): Type of RNN cell to use, either "lstm" or "gru" (default: "lstm").
- `embedding_layer_size` (int): Size of the embedding layer (default: 256).
- `max_sequence_length` (int): Maximum length of the generated SMILES sequence (default: 256). Note: This is often passed during creation or set in the metadata.

**Learning Rate & Scheduler:**
The default learning rate is **1e-4**. This is configured in the `[scheduler]` section of the TOML file.
- **Default LR:** `1.0e-4`
- **Default Scheduler:** `StepLR` (with `step=10`, `gamma=0.95`).
- **Configuration:**
  ```toml
  [scheduler]
  lr = 1e-4
  step = 10
  gamma = 0.95
  ```
- **Definition Location:** Defaults are defined in `reinvent/runmodes/TL/configurations.py`.

## SMILES Standardization and its Impact

SMILES standardization is a critical preprocessing step that occurs before the data is used for training. In Reinvent 4, this is handled by the `RDKitStandardizer` (found in `reinvent/chemistry/standardization/rdkit_standardizer.py`).

### How it Affects Training:

1.  **Reduces Complexity (Canonicalization):**
    A single molecule can be represented by many valid SMILES strings. Standardization forces every molecule into a single **canonical** form. This allows the model to learn a consistent "grammar" and converge much faster, as it doesn't have to learn multiple ways to spell the same molecule.

2.  **Vocabulary Definition:**
    The model's vocabulary is built from the standardized SMILES. This ensures the characters the model learns (like `C`, `N`, `(`, `=`) are derived from clean, consistent data, preventing the vocabulary from becoming bloated with noise or non-standard characters.

3.  **Chemical Quality Control:**
    The standardization pipeline (defined in `filter_registry.py`) performs several key actions:
    - **Largest Fragment:** Strips away solvents and counter-ions (e.g., stripping the HCl from a drug salt).
    - **Charge Neutralization:** Attempts to neutralize formal charges where possible.
    - **Salt Removal:** Removes common salts.
    - **Valid Size:** Filters out molecules that are too small or too large.

4.  **Improved Output Quality:**
    By training exclusively on clean, neutral, and canonical molecules, the model is biased to generate molecules with these same high-quality properties.

### Standardization Comparison:

| Feature | Standardization **ON** (Recommended) | Standardization **OFF** |
| :--- | :--- | :--- |
| **Input Data** | Canonical, Cleaned (Salts/Solvents removed) | Raw (May include salts, noise) |
| **Vocabulary** | Compact and Efficient | Potentially Noisy/Bloated |
| **Learning Task** | Easier (One spelling per molecule) | Harder (Multiple spellings for same molecule) |
| **Model Output** | High validity, neutral, salt-free | Lower validity, may include salts/noise |

## Training Logic & Directory Structure

The training process is handled across several key files in the `reinvent/runmodes/TL/` directory:

1.  **Entry Point:** `reinvent/runmodes/TL/run_transfer_learning.py`
    - Initializes the optimizer (Adam) and the learning rate scheduler.
    - Loads the dataset and the model.
2.  **Base Training Class:** `reinvent/runmodes/TL/learning.py`
    - Contains the main `optimize()` loop.
    - Implements `_train_epoch_common()`, which handles the forward pass, loss calculation, backpropagation, and weight updates.
3.  **Model Specifics:** `reinvent/runmodes/TL/reinvent.py`
    - Inherits from the base `Learning` class.
    - Defines Reinvent-specific data loading and NLL computation.
4.  **Network Architecture:** `reinvent/models/reinvent/models/rnn.py`
    - Defines the actual PyTorch `nn.Module` (LSTM/GRU layers) used by the model.

## Model Storage

- **After Creation:** The initialized model is saved to the path specified by `model_file` in the `[io]` section of your creation config (e.g., `re_2M.model`).
- **During/After Training:** The trained model checkpoints are saved to the path specified by `output_model_file` in the `[parameters]` section of your training config (e.g., `re_2M_trained.model`). The system may save checkpoints like `re_2M_trained.model.10.chkpt`.

## Handling Additional Elements (Si, B, P, Se, As)

By default, REINVENT4's preprocessing filter supports only standard organic elements (`C`, `O`, `N`, `S`, `F`, `Cl`, `Br`, `I`). If your dataset contains other elements (like `Si`, `B`, `P`, `Se`, `As`), they must be explicitly handled in a two-step process to avoid being discarded.

### 1. Element Support in Tokenizer
REINVENT4's tokenizer is data-driven and flexible:
- **Single Letter Atoms (B, P):** Automatically supported.
- **Bracketed Atoms ([Si], [Se], [As]):** Automatically supported because the tokenizer has a specific rule for anything inside square brackets `[...]`.
- **Note:** Most datasets (including `data/2M.csv`) represent these elements inside brackets (e.g., `[Si]`), so no code modification is required.

### 2. The Two-Step Workflow
You cannot just point `create_reinvent.py` to a raw CSV with these elements, as the internal standardizer might filter them out or crash. Instead, use this robust workflow:

**Step A: Explicit Filtering (Preprocessing)**
Create a preprocessing config (e.g., `preprocess.toml`) that explicitly adds your elements to the allow-list.

```toml
[dpl]
input_csv_file = "data/raw_data.csv"
output_smiles_file = "data/filtered_data.smi"
# Separator Note: This controls how COLUMNS are split. 
# Newline (\n) is ALWAYS the row separator. 
# If your file is single-column (one SMILES per line), use a separator 
# that doesn't appear in the data, like "\t" (tab) or " " (space).
separator = "\t"  
# ... other dpl settings ...

[filter]
# CRITICAL: Add all your non-standard elements here
elements = ["P", "Si", "B", "Se", "As", "Ge"]
```

Run the preprocessor:
```bash
python reinvent/datapipeline/preprocess.py configs/preprocess.toml
```

**Step B: Training on Filtered Data**
Update your model creation config (e.g., `create_model.toml`) to point to the **output** of Step A.

```toml
[io]
smiles_file = "data/filtered_data.smi"  # Point to the clean file
model_file = "my_model.model"
# ...
```

Run the model creation:
```bash
python reinvent/runmodes/create_model/create_reinvent.py configs/create_model.toml
```