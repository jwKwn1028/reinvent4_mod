# REINVENT4 Configuration Reference

This document details the configuration parameters available in `.toml` files for REINVENT4. The configuration is divided into **Global Parameters** (top-level), **Run Mode Parameters** (specific to the `run_type`), and **Scoring Components**.

## 1. Global Configuration

These parameters are defined at the top level of the TOML file.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`run_type`** | String | **Required.** The mode of operation. Options: `"staged_learning"` (RL), `"transfer_learning"` (TL), `"sampling"`, `"scoring"`, `"create_model"`. |
| `device` | String | Hardware to run on. Examples: `"cuda:0"`, `"cpu"`. Default: `"cpu"`. |
| `tb_logdir` | String | Path to a directory where TensorBoard logs will be written. |
| `json_out_config` | String | Path to write the full runtime configuration as a JSON file (useful for debugging). |
| `seed` | Integer | Random seed for reproducibility. |
| `parameters` | Table | Contains run-mode specific settings (see below). |

---

## 2. Run Modes

The structure of the `[parameters]` section depends on the `run_type`.

### A. Reinforcement Learning (RL)
**`run_type = "staged_learning"`**

#### `[parameters]`
| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`prior_file`** | String | **Required.** Path to the pre-trained prior model file. |
| **`agent_file`** | String | **Required.** Path to the agent model file (usually same as prior to start). |
| `summary_csv_prefix` | String | Prefix for the summary CSV files generated during training. Default: `"summary"`. |
| `use_checkpoint` | Boolean | If `true`, attempts to resume from a checkpoint. Default: `false`. |
| `purge_memories` | Boolean | If `true`, clears replay buffers when starting. Default: `true`. |
| `smiles_file` | String | Path to a SMILES file (context-dependent usage). |
| `batch_size` | Integer | Batch size for sampling. Default: `100`. |
| `randomize_smiles` | Boolean | Whether to use randomized SMILES strings for input. Default: `true`. |
| `unique_sequences` | Boolean | Ensure sampled sequences are unique. Default: `false`. |
| `temperature` | Float | Sampling temperature. Higher values = more randomness. Default: `1.0`. |

#### `[[stage]]`
A list of learning stages. You can define multiple stages.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`max_steps`** | Integer | **Required.** Maximum number of steps for this stage. |
| `max_score` | Float | Terminate stage if this average score is reached. Default: `1.0`. |
| `min_steps` | Integer | Minimum steps to run before checking termination criteria. Default: `50`. |
| `termination` | String | Termination criterion type. Default: `"simple"`. |
| `chkpt_file` | String | Path to save a checkpoint file for this stage. |
| `[stage.scoring]` | Table | Scoring components for this stage (see **Section 3**). |
| `[stage.diversity_filter]`| Table | (Optional) specific diversity filter for this stage. |

#### `[learning_strategy]`
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `type` | String | Strategy type, e.g., `"dap"`. Default: `"dap"`. |
| `sigma` | Float | Sigma parameter for DAP. Default: `128`. |
| `rate` | Float | Learning rate. Default: `0.0001`. |

#### `[diversity_filter]`
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `type` | String | Filter type: `"PenalizeSameSmiles"`, `"ScaffoldSimilarity"`. |
| `bucket_size` | Integer | Size of the memory bucket. Default: `25`. |
| `minscore` | Float | Minimum score to be considered for the memory. Default: `0.4`. |
| `minsimilarity` | Float | (ScaffoldSimilarity only) Minimum similarity. Default: `0.4`. |
| `penalty_multiplier` | Float | (PenalizeSameSmiles only) Penalty factor. Default: `0.5`. |

#### `[inception]`
Used to seed the memory with known high-scoring molecules.
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `smiles_file` | String | File containing SMILES to seed memory. |
| `memory_size` | Integer | Size of inception memory. Default: `50`. |
| `sample_size` | Integer | Number of samples to draw. Default: `10`. |

---

### B. Transfer Learning (TL)
**`run_type = "transfer_learning"`**

#### `[parameters]`
| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`input_model_file`** | String | **Required.** Path to the pre-trained model. |
| **`output_model_file`** | String | **Required.** Path to save the fine-tuned model. |
| **`smiles_file`** | String | **Required.** File containing SMILES for training. |
| `num_epochs` | Integer | Number of training epochs. |
| `batch_size` | Integer | Training batch size. |
| `sample_batch_size` | Integer | Batch size for sampling validation. Default: `100`. |
| `save_every_n_epochs` | Integer | Save checkpoint frequency. Default: `1`. |
| `validation_smiles_file` | String | Separate file for validation. |
| `standardize_smiles` | Boolean | Canonicalize SMILES. Default: `true`. |
| `max_sequence_length` | Integer | Max token length. Default: `128`. |
| `pairs` | Table | (Transformer only) Config for source/target pairs. |

#### `[scheduler]`
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `lr` | Float | Learning rate. |
| `gamma` | Float | Decay factor (StepLR). |
| `step` | Integer | Step size (StepLR). |

---

### C. Sampling
**`run_type = "sampling"`**

#### `[parameters]`
| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`model_file`** | String | **Required.** Path to the generative model. |
| `output_file` | String | Output CSV path. Default: `"samples.csv"`. |
| `num_smiles` | Integer | Number of molecules to generate. |
| `unique_molecules` | Boolean | Keep only unique molecules. Default: `true`. |
| `randomize_smiles` | Boolean | Default: `true`. |
| `temperature` | Float | Default: `1.0`. |
| `target_smiles_path` | String | (Optional) Path to target SMILES for likelihood calculations. |
| `sample_strategy` | String | e.g., `"multinomial"`. |

#### `[filter]`
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `smarts` | List[String] | List of SMARTS strings to filter generated molecules. |

---

### D. Scoring
**`run_type = "scoring"`**

#### `[parameters]`
| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`smiles_file`** | String | **Required.** Input CSV/SMI file containing molecules to score. |
| `output_csv` | String | Output file path. Default: `"score_results.csv"`. |
| `smiles_column` | String | Column name in CSV containing SMILES. Default: `"SMILES"`. |

#### `[scoring]`
Contains lists of scoring components (see **Section 3**).

---

### E. Create Model
**`run_type = "create_model"`** (Note: Often run via `create_reinvent.py` which uses a slightly different structure).

#### `[io]`
| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`smiles_file`** | String | **Required.** Input SMILES file for vocabulary training. |
| `model_file` | String | Output path for the new model. Default: `"empty.model"`. |

#### `[network]`
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `num_layers` | Integer | Number of LSTM/GRU layers. Default: `3`. |
| `layer_size` | Integer | Hidden size. Default: `512`. |
| `dropout` | Float | Dropout probability. Default: `0.0`. |
| `max_sequence_length` | Integer | Default: `256`. |
| `cell_type` | String | `"lstm"` or `"gru"`. Default: `"lstm"`. |
| `embedding_layer_size` | Integer | Default: `256`. |
| `layer_normalization` | Boolean | Default: `false`. |
| `standardize` | Boolean | Default: `true`. |

#### `[metadata]`
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `data_source` | String | Name of the dataset source. |
| `comment` | String | Description or notes. |

---

## 3. Scoring Components

Scoring components are defined in `[scoring]` or `[stage.scoring]`.
Structure:
```toml
[[component.COMPONENT_NAME.endpoint]]
name = "Component Name"
weight = 1.0
params.PARAM_NAME = VALUE
transform.type = "TRANSFORM_NAME"
```

### Component List

#### Predictive Models
*   **`ChemProp`**
    *   `checkpoint_dir` (List[str]): Path to model checkpoint directory.
    *   `features` (List[str]): e.g., `["morgan"]`, `["rdkit_2d_normalized"]`.
    *   `target_column` (List[str]): Target name in the model.
*   **`DeepChemDMPNN`**
    *   `checkpoint_path` (List[str]): Path to `.ckpt` file.
    *   `transformer_path` (List[str]): Path to `.joblib` transformer.
    *   `features_generators` (List[str]): e.g., `["rdkit_desc_normalized,morgan"]`.
    *   `batch_size`, `depth`, `ffn_hidden`, `ffn_layers`, `dropout_p`, `bias`, `global_features_size`.
*   **`Qptuna`**
    *   `model_file` (List[str]): Path to Qptuna pickle file.
*   **`ExternalProcess`**
    *   `executable` (List[str]): Path to executable.
    *   `args` (List[str]): Command line arguments.
    *   `property` (List[str]): JSON key in output to parse as score.
*   **`REST`**
    *   `server_url`, `server_port`, `server_endpoint`.
    *   `predictor_id`, `predictor_version`.

#### Structure & Filters
*   **`CustomAlerts`** (Filter)
    *   `smarts` (List[str]): SMARTS patterns to penalize/filter.
*   **`MatchingSubstructure`** (Penalty)
    *   `smarts` (List[str]): SMARTS patterns to search.
    *   `use_chirality` (List[bool]).
*   **`GroupCount`**
    *   `smarts` (List[str]): SMARTS patterns to count.
*   **`ReactionFilter`**
    *   `type` (List[str]): `"selective"`, `"nonselective"`, `"definedselective"`.
    *   `reaction_smarts` (List[str]): Reaction SMARTS.
*   **`MMP`** (Matched Molecular Pairs)
    *   `reference_smiles` (List[str]): Reference molecule SMILES.
    *   `num_of_cuts`, `max_variable_heavies`, `max_variable_ratio`.

#### Physicochemical Properties (RDKit)
*   **`RDKitDescriptors`**
    *   `descriptor` (List[str]): RDKit descriptor names (e.g., `"MolWt"`, `"BalabanJ"`).
*   **`MolVolume`**
    *   `grid_spacing` (List[float]), `box_margin` (List[float]).
*   **`PMI`** (Principal Moment of Inertia)
    *   `property` (List[str]): `"npr1"` or `"npr2"`.
*   **Standard Components (No params usually required)**
    *   `Qed`
    *   `MolecularWeight`
    *   `TPSA` (param: `includeSandP`)
    *   `NumRotatableBonds`
    *   `HBondDonors`, `HBondAcceptors`
    *   `SlogP`
    *   `NumRings`, `NumAromaticRings`

#### Shape & Similarity
*   **`TanimotoSimilarity`**
    *   `smiles` (List[str]): Reference SMILES.
    *   `radius` (List[int]): Fingerprint radius (default 2).
    *   `use_counts` (List[bool]), `use_features` (List[bool]).
*   **`ROCSSimilarity`**
    *   `rocs_input` (List[str]): Path to shape query file.
    *   `shape_weight`, `color_weight` (List[float]).
    *   `similarity_measure` (List[str]): e.g., `"Tanimoto"`.

#### External Integrations
*   **`DockStream`**
    *   `configuration_path`, `docker_script_path`, `docker_python_path`.
*   **`Maize`**
    *   `executable`, `workflow`, `config`, `log`, `property`.
*   **`SynthSense`** / **`CAZP`**
    *   AiZynthFinder integration parameters.
*   **`SAScore`** (Synthetic Accessibility)
    *   No parameters.

---

## 4. Score Transformations & Generative Impact

The **Transforms** section is critical for Reinforcement Learning. The RL agent optimizes for a total score between 0.0 and 1.0. Raw component scores (e.g., Molecular Weight of 350, Docking Score of -9.5) must be "transformed" into this 0-1 "desirability" space.

Choosing the right transform tells the agent *how* to improve.

### A. Sigmoid (`type = "sigmoid"`)
**Goal:** Maximize a value (e.g., QED, Solubility).
*   **Logic:** Scores increase smoothly from 0 to 1 as the value increases.
*   **Parameters:**
    *   `low`: The value where the score is approx 0.1-0.2 (start of the slope).
    *   `high`: The value where the score is approx 0.8-0.9 (end of the slope).
    *   `k`: Steepness factor. (Typically `0.25` or `0.5`).
*   **Generative Impact:** The agent is "pulled" towards values higher than `high`.
    *   *Low `k` (gentle slope):* Encourages gradual improvement.
    *   *High `k` (steep slope):* Forces a sharp threshold behavior.

### B. Reverse Sigmoid (`type = "reverse_sigmoid"`)
**Goal:** Minimize a value (e.g., Binding Affinity, TPSA).
*   **Logic:** Scores decrease from 1 to 0 as the value increases.
    *   Values < `low` get scores near 1.0 (Very Good).
    *   Values > `high` get scores near 0.0 (Very Bad).
*   **Parameters:** Same as Sigmoid (`low`, `high`, `k`).
*   **Generative Impact:** The agent is "pushed" away from values higher than `high` and rewarded for going lower than `low`. Ideally suited for docking scores where more negative is better.

### C. Double Sigmoid (`type = "double_sigmoid"`)
**Goal:** Target a specific range (e.g., "Molecular Weight between 300 and 400").
*   **Logic:** A "Bell Curve" or "Window" shape.
    *   Score is ~1.0 inside the [low, high] window.
    *   Score drops to 0.0 outside this window.
*   **Parameters:**
    *   `low`: Center of the left (rising) edge (score ~0.5).
    *   `high`: Center of the right (falling) edge (score ~0.5).
    *   `coef_div`: Divisor for steepness (default `100.0`).
    *   `coef_si`: Steepness of left slope (default `150.0`).
    *   `coef_se`: Steepness of right slope (default `150.0`).
*   **Generative Impact:** Forces the agent to stay within a specific property window. If the agent drifts too high or too low, the reward vanishes.

### D. Step Functions (`step`, `left_step`, `right_step`)
**Goal:** Hard Constraints (Binary Pass/Fail).
*   **Logic:**
    *   `step`: 1.0 if `low <= value <= high`, else 0.0.
    *   `left_step`: 1.0 if `value <= low`, else 0.0.
    *   `right_step`: 1.0 if `value >= high`, else 0.0.
*   **Generative Impact:** **Use with caution in RL.**
    *   Step functions provide *zero gradient*. If a molecule is just outside the range, it gets 0 score. The agent has no "clue" which direction to move to fix it.
    *   *Recommendation:* Use Sigmoids instead of Steps for RL training to provide a "learning slope." Use Steps only for final filtering or `sampling` modes.

### E. Exponential Decay (`type = "exponential_decay"`)
**Goal:** Penalize deviation from zero.
*   **Logic:** $Score = \exp(-k \cdot x)$ (clamped to 1.0 if $x < 0$).
*   **Parameters:**
    *   `k`: Steepness of the decay.
*   **Generative Impact:** Useful for "penalty" components where the raw value represents a count of "bad" features (e.g., number of toxic substructures). The ideal value is 0 (Score=1.0). As counts increase, score drops exponentially.

### F. Value Mapping (`type = "value_mapping"`)
**Goal:** Score categorical data.
*   **Parameters:**
    *   `mapping`: A dictionary, e.g., `{"Active": 1.0, "Inactive": 0.0, "Inconclusive": 0.5}`.
*   **Generative Impact:** Used with predictive models that output class labels instead of probabilities.
