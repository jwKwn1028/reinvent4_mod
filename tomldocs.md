# REINVENT4 Configuration Reference

This document lists all supported TOML parameters for REINVENT4 configuration files.
It covers:
- run_type-based configs consumed by the main REINVENT runner (RL/TL/Sampling/Scoring)
- scoring component configs (inline or external files)
- standalone configs for the data pipeline and create-model utilities

Notes:
- Component names are case-insensitive; underscores/dashes are ignored (e.g., `TanimotoSimilarity` == `tanimoto_similarity`).
- Endpoint params are written as scalars in TOML; REINVENT collects them into lists internally.
- Defaults below reflect code defaults where defined.

---

## 1. Global (run_type configs)

Top-level keys accepted by the main runner:

| Key | Type | Description |
| :--- | :--- | :--- |
| **`run_type`** | String | **Required.** `"staged_learning"`, `"transfer_learning"`, `"sampling"`, `"scoring"`. |
| `device` | String | Torch device string. Default: `"cpu"`. |
| `use_cuda` | Boolean | **Deprecated.** Use `device` instead. |
| `tb_logdir` | String | TensorBoard log directory (RL/TL). |
| `json_out_config` | String | Path to write the fully resolved config as JSON. |
| `seed` | Integer | Random seed for reproducibility. |
| **`parameters`** | Table | **Required.** Run-type-specific settings. |
| `stage` | Array[Table] | RL only: `[[stage]]` list. |
| `learning_strategy` | Table | RL only. |
| `diversity_filter` | Table | RL only (global DF, overrides stage DFs). |
| `intrinsic_penalty` | Table | RL only. |
| `inception` | Table | RL only. |
| `scoring` | Table | Scoring config for `run_type = "scoring"` (RL uses `stage.scoring`). |
| `scheduler` | Table | TL only. |
| `filter` | Table | Sampling only. |
| `responder` | Table | Optional remote monitor configuration. |

### [responder]
| Key | Type | Description |
| :--- | :--- | :--- |
| `endpoint` | String | Remote JSON reporter URL. |
| `frequency` | Integer | Reporting frequency (steps/epochs). Default: `1`. |

If the `RESPONDER_TOKEN` environment variable is set, it is sent as the Authorization header.

---

## 2. Run Modes

### A. Reinforcement Learning (RL)
**`run_type = "staged_learning"`**

#### [parameters]
| Key | Type | Description |
| :--- | :--- | :--- |
| **`prior_file`** | String | **Required.** Path to prior model. |
| **`agent_file`** | String | **Required.** Path to agent model. |
| `summary_csv_prefix` | String | Prefix for per-stage CSV logs. Default: `"summary"`. |
| `use_checkpoint` | Boolean | Resume from checkpoint in `agent_file`. Default: `false`. |
| `purge_memories` | Boolean | Clear DF memories between stages. Default: `true`. |
| `smiles_file` | String | Seed SMILES/fragment file (Libinvent/Linkinvent/Mol2Mol/Pepinvent). Optional. |
| `sample_strategy` | String | Transformer models: `"multinomial"` or `"beamsearch"`. Default: `"multinomial"`. |
| `distance_threshold` | Integer | Mol2Mol distance penalty threshold. Default: `99999`. |
| `batch_size` | Integer | Sampling batch size. Default: `100`. |
| `randomize_smiles` | Boolean | Randomize input SMILES. Default: `true`. |
| `unique_sequences` | Boolean | Enforce unique raw sequences per step. Default: `false`. |
| `temperature` | Float | Sampling temperature. Default: `1.0`. |
| `tb_isim` | Boolean | Log iSIM similarity to TensorBoard. Default: `false`. |

#### [[stage]]
| Key | Type | Description |
| :--- | :--- | :--- |
| **`max_steps`** | Integer | **Required.** Max steps for this stage. |
| `max_score` | Float | Stop stage if average score reaches this. Default: `1.0`. |
| `min_steps` | Integer | Minimum steps before termination check. Default: `50`. |
| `termination` | String | `"simple"`, `"plateau"`, or `"null"`. Default: `"simple"`. |
| `chkpt_file` | String | Checkpoint file for this stage. |
| `scoring` | Table | Stage scoring config (see Section 3). |
| `diversity_filter` | Table | Stage-specific DF (ignored if global DF exists). |

`[stage.scoring]` can reference an external scoring file:
| Key | Type | Description |
| :--- | :--- | :--- |
| `filename` | String | Path to scoring TOML/JSON/YAML. |
| `filetype` | String | Format override: `"toml"`, `"json"`, or `"yaml"`. |

#### [learning_strategy]
| Key | Type | Description |
| :--- | :--- | :--- |
| `type` | String | Reward strategy: `"dap"`, `"sdap"`, `"mauli"`, `"mascof"`, `"dap_reinforce"`, `"mauli_reinforce"`, `"mascof_reinforce"`. |
| `sigma` | Float | Reward scaling. Default: `128`. |
| `rate` | Float | Optimizer learning rate. Default: `0.0001`. |

#### [diversity_filter]
| Key | Type | Description |
| :--- | :--- | :--- |
| `type` | String | `"IdenticalMurckoScaffold"`, `"IdenticalTopologicalScaffold"`, `"ScaffoldSimilarity"`, `"PenalizeSameSmiles"`. |
| `bucket_size` | Integer | Bucket size. Default: `25`. |
| `minscore` | Float | Minimum score to store. Default: `0.4`. |
| `minsimilarity` | Float | ScaffoldSimilarity only. Default: `0.4`. |
| `penalty_multiplier` | Float | PenalizeSameSmiles only. Default: `0.5`. |

#### [intrinsic_penalty]
| Key | Type | Description |
| :--- | :--- | :--- |
| `type` | String | Currently: `"IdenticalMurckoScaffoldRND"`. |
| `penalty_function` | String | `"Step"`, `"Sigmoid"`, `"Linear"`, `"Tanh"`, `"Erf"`. |
| `bucket_size` | Integer | Default: `25`. |
| `minscore` | Float | Default: `0.4`. |
| `learning_rate` | Float | RND predictor learning rate. Default: `1e-4`. |

#### [inception]
| Key | Type | Description |
| :--- | :--- | :--- |
| `smiles_file` | String | Seed SMILES for memory. |
| `memory_size` | Integer | Default: `50`. |
| `sample_size` | Integer | Default: `10`. |
| `deduplicate` | Boolean | Obsolete (always true); kept for compatibility. |

---

### B. Transfer Learning (TL)
**`run_type = "transfer_learning"`**

#### [parameters]
| Key | Type | Description |
| :--- | :--- | :--- |
| **`input_model_file`** | String | **Required.** Path to pretrained model. |
| **`output_model_file`** | String | **Required.** Output model path. |
| **`smiles_file`** | String | **Required.** Training SMILES file. |
| **`num_epochs`** | Integer | **Required.** Number of epochs. |
| **`batch_size`** | Integer | **Required.** Batch size. |
| `sample_batch_size` | Integer | Sampling stats batch size. Default: `100`. |
| `save_every_n_epochs` | Integer | Checkpoint frequency. Default: `1`. |
| `training_zero_epoch_start` | Boolean | Start logging from epoch 0. Default: `false`. |
| `starting_epoch` | Integer | First epoch number (1-based). Default: `1`. |
| `shuffle_each_epoch` | Boolean | Shuffle training data each epoch. Default: `true`. |
| `num_refs` | Integer | Reference count for similarity stats (0 disables). Default: `0`. |
| `validation_smiles_file` | String | Optional validation set. |
| `standardize_smiles` | Boolean | Standardize input SMILES. Default: `true`. |
| `randomize_smiles` | Boolean | Randomize SMILES (Reinvent). Default: `true`. |
| `randomize_all_smiles` | Boolean | Randomize all SMILES (overrides randomize_smiles). Default: `false`. |
| `internal_diversity` | Boolean | Compute internal diversity metric. Default: `false`. |
| `tb_isim` | Boolean | Log iSIM similarity to TensorBoard. Default: `false`. |
| `max_sequence_length` | Integer | Max token length (Transformer). Default: `128`. |
| `clip_gradient_norm` | Float | Non-transformer gradient clipping. Default: `1.0`. |
| `pairs` | Table | Transformer (Mol2Mol) only. See below. |
| `ranking_loss_penalty` | Boolean | Ranking loss (Tanimoto pairs only). Default: `false`. |
| `n_cpus` | Integer | Pair generation workers. Default: `1`. |

#### [parameters.pairs] (Mol2Mol/Transformer)
Common keys:
| Key | Type | Description |
| :--- | :--- | :--- |
| `type` | String | `"tanimoto"`, `"mmp"`, `"scaffold"`, `"precomputed"`. |
| `min_cardinality` | Integer | Min targets per source. Default: `1`. |
| `max_cardinality` | Integer | Max targets per source. Default: `199`. |
| `add_same` | Boolean | Include (s,s) pairs. Default: `false`. |

Type-specific keys:
- **tanimoto**: `lower_threshold` (float), `upper_threshold` (float)
- **mmp**: `hac` (int), `ratio` (float), `max_radius` (int)
- **scaffold**: `generic` (bool)
- **precomputed**: input SMILES file must contain (source,target) pairs

#### [scheduler]
Two scheduler families are used:
- **StepLR** (Reinvent/Libinvent/Linkinvent/Pepinvent):
  - `lr` (float, default `1e-4`)
  - `min` (float, default `1e-7`)
  - `gamma` (float, default `0.95`)
  - `step` (int, default `10`)
- **LambdaLR** (Mol2Mol transformer):
  - `lr` (float, default `1e-4`)
  - `min` (float, default `1e-10`)
  - `beta1` (float, default `0.9`)
  - `beta2` (float, default `0.98`)
  - `eps` (float, default `1e-9`)
  - `factor` (float, default `1.0`)
  - `warmup` (float, default `4000`)

---

### C. Sampling
**`run_type = "sampling"`**

#### [parameters]
| Key | Type | Description |
| :--- | :--- | :--- |
| **`model_file`** | String | **Required.** Model file. |
| **`num_smiles`** | Integer | **Required.** Number of molecules to generate per input. |
| `smiles_file` | String | Optional input SMILES (Libinvent/Linkinvent/Mol2Mol/Pepinvent). |
| `output_file` | String | Output CSV. Default: `"samples.csv"`. |
| `sample_strategy` | String | `"multinomial"` or `"beamsearch"` (transformers). Default: `"multinomial"`. |
| `target_smiles_path` | String | Mol2Mol: target SMILES for NLL check. |
| `target_nll_file` | String | Mol2Mol: output file for target NLLs. Default: `"target_nll.csv"`. |
| `unique_molecules` | Boolean | Keep only unique valid SMILES. Default: `true`. |
| `randomize_smiles` | Boolean | Randomize input SMILES. Default: `true`. |
| `temperature` | Float | Sampling temperature. Default: `1.0`. |

#### [filter]
| Key | Type | Description |
| :--- | :--- | :--- |
| `smarts` | Array[String] | SMARTS patterns to filter out generated molecules. |

---

### D. Scoring
**`run_type = "scoring"`**

#### [parameters]
| Key | Type | Description |
| :--- | :--- | :--- |
| **`smiles_file`** | String | **Required.** Input SMILES/CSV file. |
| `output_csv` | String | Output CSV. Default: `"score_results.csv"`. |
| `smiles_column` | String | Column name for SMILES in CSV. Default: `"SMILES"`. |
| `standardize_smiles` | Boolean | Standardize SMILES before scoring. Default: `true`. |

#### [scoring]
See Section 3.

---

### E. Create Model (create_reinvent.py)
Standalone utility (does **not** use run_type). Config sections:

#### [io]
| Key | Type | Description |
| :--- | :--- | :--- |
| **`smiles_file`** | String | **Required.** Input SMILES for vocab. |
| `model_file` | String | Output model path. Default: `"empty.model"`. |

#### [network]
| Key | Type | Description |
| :--- | :--- | :--- |
| `num_layers` | Integer | Default: `3`. |
| `layer_size` | Integer | Default: `512`. |
| `dropout` | Float | Default: `0.0`. |
| `max_sequence_length` | Integer | Default: `256`. |
| `cell_type` | String | `"lstm"` or `"gru"`. Default: `"lstm"`. |
| `embedding_layer_size` | Integer | Default: `256`. |
| `layer_normalization` | Boolean | Default: `false`. |
| `standardize` | Boolean | Default: `true`. |

#### [metadata]
| Key | Type | Description |
| :--- | :--- | :--- |
| `data_source` | String | Dataset/source name. |
| `comment` | String | Free-form notes. |

---

### F. Data Pipeline (datapipeline)
Standalone utility (no run_type).

Top-level keys:
| Key | Type | Description |
| :--- | :--- | :--- |
| **`input_csv_file`** | String | **Required.** Input CSV/SMI. |
| `smiles_column` | String | Default: `"SMILES"`. |
| `separator` | String | Single character delimiter. Default: `"\t"`. |
| **`output_smiles_file`** | String | **Required.** Output `.smi`. |
| `num_procs` | Integer | Parallel workers. Default: `1`. |
| `chunk_size` | Integer | Batch size for multiprocessing. Default: `500`. |
| `transform_file` | String | Optional custom transform script file. |
| `filter` | Table | Filtering options (see below). |

#### [filter]
| Key | Type | Description |
| :--- | :--- | :--- |
| `elements` | Array[String] | Allowed elements (merged with built-in base set). |
| `transforms` | Array[String] | Normalization transforms. Default: `["standard"]`. |
| `min_heavy_atoms` | Integer | Default: `2`. |
| `max_heavy_atoms` | Integer | Default: `90`. |
| `max_mol_weight` | Float | Default: `1200.0`. |
| `min_carbons` | Integer | Default: `2`. |
| `max_num_rings` | Integer | Default: `12`. |
| `max_ring_size` | Integer | Default: `7`. |
| `keep_stereo` | Boolean | Default: `true`. |
| `keep_isotope_molecules` | Boolean | Default: `true`. |
| `uncharge` | Boolean | Default: `true`. |
| `canonical_tautomer` | Boolean | Default: `false`. |
| `kekulize` | Boolean | Default: `false`. |
| `randomize_smiles` | Boolean | Default: `false`. |
| `report_errors` | Boolean | Default: `false`. |
| `inchi_key_deduplicate` | Boolean | Default: `false`. |

---

## 3. Scoring Configuration Structure

Scoring configs are used in:
- `run_type = "scoring"` as top-level `[scoring]`
- RL stages as `[stage.scoring]`

Skeleton:

```toml
[scoring]
type = "geometric_mean"      # or "arithmetic_mean"
parallel = 1                 # number of processes (1..40)
use_pumas = false            # use PUMAS aggregation/transforms
filename = "stage2_scoring.toml"  # optional external scoring file
filetype = "toml"            # optional override

[[scoring.component]]
[scoring.component.ComponentName]
params.some_param = "value"  # component-level params (optional)

[[scoring.component.ComponentName.endpoint]]
name = "My component"
weight = 1.0                 # scoring components only; filters/penalties ignore weight
params.some_param = "value"  # endpoint-level params (override component-level)
transform.type = "sigmoid"
transform.low = 0.0
transform.high = 1.0
transform.k = 0.25
```

Notes:
- `type` supports `"geometric_mean"` and `"arithmetic_mean"`. Aliases `"custom_product"` and `"custom_sum"` are accepted.
- `parallel` max is `40`.
- `use_pumas = true` switches to PUMAS aggregations/transforms; transform parameters then follow PUMAS definitions.
- Component names are normalized (case-insensitive; underscores/dashes removed).
- Filters and penalties act as masks/penalties and do not contribute weights to aggregation.
- Component-level `params.*` apply to all endpoints; endpoint-level `params.*` override.

---

## 4. Scoring Components (built-in)

### Predictive Models
- **ChemProp**
  - `checkpoint_dir` (String): model directory.
  - `features` (String): e.g., `"morgan"`, `"rdkit_2d_normalized"`.
  - `rdkit_2d_normalized` (Bool): obsolete; use `features = "rdkit_2d_normalized"`.
  - `target_column` (String): target name in the ChemProp model.
- **DeepChemDMPNN**
  - `checkpoint_path` (String): Lightning `.ckpt` path.
  - `transformer_path` (String): DeepChem transformer `.joblib`.
  - `device` (String): `"cpu"` or `"cuda[:N]"`.
  - `batch_size` (Int): inference batch size (default 256 if omitted).
  - `features_generators` (String): comma-separated list (e.g., `"rdkit_desc_normalized,morgan"`).
  - `depth`, `ffn_hidden`, `ffn_layers` (Int), `dropout_p` (Float), `bias` (Bool),
    `global_features_size` (Int, `-1` for auto-detect).
- **Qptuna**
  - `model_file` (String): path to Qptuna pickle file.

### External / Integrations
- **ExternalProcess**
  - `executable` (String): executable path.
  - `args` (String): CLI args.
  - `property` (String): JSON key(s) from `payload` to use as scores.
  - Note: all endpoints share the same `executable`/`args`.
- **REST**
  - `server_url`, `server_port`, `server_endpoint`
  - `predictor_id`, `predictor_version`
  - `header` (String/JSON, optional): HTTP header override.
- **DockStream**
  - `configuration_path`, `docker_script_path`, `docker_python_path`.
- **Icolos**
  - `name` (String): score key in Icolos output.
  - `executable` (String): Icolos executable path.
  - `config_file` (String): Icolos config path.
- **Maize**
  - `executable` (String): Maize binary.
  - `workflow` (String): workflow file (YAML/TOML/JSON).
  - `config` (String, optional): Maize system config.
  - `log` (String, optional): log file.
  - `debug` (Bool, default false), `keep` (Bool, default false).
  - `parameters` (Table): overrides for workflow parameters.
  - `skip_normalize` (Bool, default false), `skip_on_failure` (Bool, default true),
    `pass_fragments` (Bool, default false).
  - `property` (String): which score from the workflow output.
- **SynthSense** / **CAZP** (AiZynthFinder)
  - Component-level params (shared across endpoints):
    - `number_of_steps`, `time_limit_seconds`
    - `stock` (Table), `scorer` (Table)
    - `stock_profile`, `reactions_profile`
  - Endpoint-level params:
    - `score_to_extract`: `"cazp"`, `"sfscore"`, `"rrscore"`, `"route_distance"`,
      `"route_similarity"`, `"route_popularity"`, `"fill_a_plate"`, `"number of reactions"`.
    - `reaction_step_coefficient` (Float): for `"cazp"`/`"sfscore"`.
    - `reference_route_file` (String): for `"rrscore"`/`"route_distance"`.
    - `popularity_threshold`, `penalty_multiplier` (Float): for `"route_popularity"`.
    - `bucket_threshold` (Int), `min_steps_for_penalization` (Int),
      `penalization_enabled` (Bool): for `"fill_a_plate"`.
    - `consider_subroutes`, `min_subroute_length`, `penalize_subroutes`: currently unused (reserved).
  - Cache is disabled automatically for batch-dependent endpoints.

### RDKit Properties (no params unless noted)
- **Qed**, **MolecularWeight**, **GraphLength**, **LargestRingSize**
- **NumAtomStereoCenters**, **NumHeavyAtoms**, **NumHeteroAtoms**
- **HBondAcceptors**, **HBondDonors**, **NumRotBond**
- **Csp3**, **numsp**, **numsp2**, **numsp3**
- **NumRings**, **NumAromaticRings**, **NumAliphaticRings**
- **SlogP**

### RDKit Descriptors
- **RDKitDescriptors**
  - `descriptor` (String): RDKit descriptor name (case-insensitive).

### RDKit 3D / Shape
- **TPSA**
  - `includeSandP` (Bool, default false): include polar S/P.
- **PMI**
  - `property` (String): `"npr1"` or `"npr2"`.
- **MolVolume**
  - `grid_spacing` (Float, default 0.2), `box_margin` (Float, default 2.0).

### Similarity
- **TanimotoSimilarity**
  - `smiles` (Array[String]): reference SMILES.
  - `radius` (Int): fingerprint radius.
  - `use_counts` (Bool), `use_features` (Bool).
- **TanimotoDistance**
  - Deprecated alias of TanimotoSimilarity.

### Structure / Filters / Penalties
- **CustomAlerts** (filter)
  - `smarts` (Array[String]): SMARTS patterns to filter.
- **MatchingSubstructure** (penalty)
  - `smarts` (Array[String]), `use_chirality` (Bool).
- **GroupCount**
  - `smarts` (Array[String]): SMARTS to count.
- **ReactionFilter** (filter)
  - `type` (String): `"selective"`, `"nonselective"`, `"definedselective"`.
  - `reaction_smarts` (Array[String]): reaction SMARTS.
- **MMP**
  - `reference_smiles` (Array[String]): reference molecules.
  - `num_of_cuts` (Int, default 1), `max_variable_heavies` (Int, default 40),
    `max_variable_ratio` (Float, default 0.33).

### Misc
- **RingPrecedence**
  - `database_file` (String): JSON with `rings` / `generic_rings`.
  - `nll_method` (String): `"total"` or `"max"`.
  - `make_generic` (Bool, default false).
- **SAScore**
  - No parameters.
- **ROCSSimilarity** (OpenEye)
  - `rocs_input`, `shape_weight`, `color_weight`, `similarity_measure`
  - `max_stereocenters`, `ewindow`, `maxconfs`
  - `custom_cff` (String, optional)

### Fragment Components (Libinvent/Linkinvent)
No parameters unless noted:
- **FragmentQed**, **FragmentMolecularWeight**, **FragmentTPSA**
- **FragmentDistanceMatrix**, **FragmentNumAtomStereoCenters**
- **FragmentHBondAcceptors**, **FragmentHBondDonors**, **FragmentNumRotBond**
- **FragmentCsp3**, **Fragmentnumsp**, **Fragmentnumsp2**, **Fragmentnumsp3**
- **FragmentEffectiveLength**, **FragmentGraphLength**, **FragmentLengthRatio**
- **FragmentNumHeavyAtoms**, **FragmentNumHeteroAtoms**
- **FragmentNumRings**, **FragmentNumAromaticRings**, **FragmentNumAliphaticRings**
- **FragmentSlogP**

---

## 5. Contrib Components (optional)

These live under `contrib/reinvent_plugins`. To enable them, add `contrib` to `PYTHONPATH`.

- **Pharm2DFP** (RDKit 2D pharmacophore fingerprints)
  - `ref_smiles` (String)
  - `feature_definition` (String): `"base"`, `"minimal"`, or `"gobbi"`.
  - `bins` (Array[Int]): e.g., `[0, 2, 2, 4, 4, 8]` for bins `(0,2),(2,4),(4,8)`.
  - `min_point_count`, `max_point_count` (Int)
  - `similarity` (String): `"tanimoto"`, `"dice"`, `"tversky"`, etc.
  - `similarity_params` (Table): similarity-specific params (e.g., `a`, `b` for tversky).
- **MordredDescriptors**
  - `descriptor` (String), `nprocs` (Int).
- **UnwantedSubstructures** (filter)
  - `catalogs` (Array[String]): RDKit filter catalogs.
- **NIBRSubstructureFilters** (filter)
  - `cutoff` (Int): severity threshold.
- **LillyDescriptors**
  - `descriptors` (String): descriptor names. Requires `LILLY_MOL_ROOT`.
- **LillyPAINS**
  - `assay` (String): one of the assay names defined in `pains_scores.csv`.
  - Requires `LILLY_MOL_ROOT`.
- **LillyMedchemRules**
  - `relaxed` (Bool): use relaxed rules if true.
  - Requires `LILLY_MEDCHEM_RULES_ROOT`.

---

## 6. Score Transforms

Transforms map raw component outputs to 0..1. Use them via `transform.type` and `transform.<param>`.

### Sigmoid (`type = "sigmoid"`)
Maximizes a value.
- `low`, `high` (Float): span of the transition.
- `k` (Float): steepness (e.g., `0.25`, `0.5`).

### Reverse Sigmoid (`type = "reverse_sigmoid"`)
Minimizes a value (same params as sigmoid).

### Double Sigmoid (`type = "double_sigmoid"`)
Targets a window.
- `low`, `high` (Float): window edges.
- `coef_div` (Float, default `100.0`)
- `coef_si` (Float, default `150.0`)
- `coef_se` (Float, default `150.0`)

### Step Functions
Binary constraints.
- `type = "step"`: uses `low` and `high`.
- `type = "left_step"`: uses `low`.
- `type = "right_step"`: uses `high`.

### Exponential Decay (`type = "exponential_decay"`)
Penalizes deviation from zero.
- `k` (Float, must be > 0).

### Value Mapping (`type = "value_mapping"`)
Maps categorical outputs to scores.
- `mapping` (Table): e.g., `{ Active = 1.0, Inactive = 0.0 }`.

---

## 7. Example Configs (run_type)

These examples are minimal, valid configs for each run type. Replace file paths with your own.

### A. Staged Learning (RL)
```toml
run_type = "staged_learning"
device = "cuda:0"
tb_logdir = "tb_logs"
json_out_config = "staged_learning.json"

[parameters]
prior_file = "priors/reinvent.prior"
agent_file = "priors/reinvent.prior"
summary_csv_prefix = "rl"
batch_size = 64
randomize_smiles = true
unique_sequences = false
tb_isim = false

[learning_strategy]
type = "dap"
sigma = 128
rate = 0.0001

[diversity_filter]
type = "IdenticalMurckoScaffold"
bucket_size = 25
minscore = 0.4

[[stage]]
max_steps = 100
min_steps = 25
max_score = 0.6
termination = "simple"
chkpt_file = "stage1.chkpt"

[stage.scoring]
type = "geometric_mean"

[[stage.scoring.component]]
[stage.scoring.component.Qed]

[[stage.scoring.component.Qed.endpoint]]
name = "QED"
weight = 1.0
```

### B. Transfer Learning (TL)
```toml
run_type = "transfer_learning"
device = "cuda:0"
tb_logdir = "tb_logs"
json_out_config = "transfer_learning.json"

[parameters]
input_model_file = "priors/reinvent.prior"
output_model_file = "output/reinvent_tl.model"
smiles_file = "data/train.smi"
num_epochs = 10
batch_size = 128
sample_batch_size = 100
save_every_n_epochs = 1
standardize_smiles = true
randomize_smiles = true
max_sequence_length = 128
clip_gradient_norm = 1.0
tb_isim = false

[scheduler]
lr = 0.0001
gamma = 0.95
step = 10
```

### C. Sampling
```toml
run_type = "sampling"
device = "cuda:0"
json_out_config = "sampling.json"

[parameters]
model_file = "priors/reinvent.prior"
num_smiles = 1000
output_file = "samples.csv"
unique_molecules = true
randomize_smiles = true
temperature = 1.0

[filter]
smarts = [
  "[*;r{8-17}]",
  "[#8][#8]"
]
```

### D. Scoring
```toml
run_type = "scoring"
device = "cpu"
json_out_config = "scoring.json"

[parameters]
smiles_file = "compounds.smi"
output_csv = "scoring.csv"
smiles_column = "SMILES"

[scoring]
type = "geometric_mean"
parallel = 1

[[scoring.component]]
[scoring.component.Qed]

[[scoring.component.Qed.endpoint]]
name = "QED"
weight = 0.5

[[scoring.component]]
[scoring.component.MolecularWeight]

[[scoring.component.MolecularWeight.endpoint]]
name = "MW"
weight = 0.5
transform.type = "double_sigmoid"
transform.low = 200.0
transform.high = 500.0
transform.coef_div = 500.0
transform.coef_si = 20.0
transform.coef_se = 20.0
```

### E. Transfer Learning (Mol2Mol Transformer)
```toml
run_type = "transfer_learning"
device = "cuda:0"
json_out_config = "mol2mol_tl.json"

[parameters]
input_model_file = "priors/mol2mol_scaffold_generic.prior"
output_model_file = "output/mol2mol_tl.model"
smiles_file = "data/mol2mol_train.smi"  # one SMILES per line
num_epochs = 5
batch_size = 64
sample_batch_size = 100
save_every_n_epochs = 1
standardize_smiles = true
max_sequence_length = 128
ranking_loss_penalty = true
n_cpus = 8

[parameters.pairs]
type = "tanimoto"
lower_threshold = 0.7
upper_threshold = 1.0
min_cardinality = 1
max_cardinality = 199
add_same = false

[scheduler]
lr = 0.0001
beta1 = 0.9
beta2 = 0.98
eps = 1e-9
factor = 1.0
warmup = 4000
```

### F. Sampling (Libinvent)
```toml
run_type = "sampling"
device = "cuda:0"
json_out_config = "libinvent_sampling.json"

[parameters]
model_file = "priors/libinvent.prior"
smiles_file = "scaffolds.smi"  # one scaffold per line with attachment points
num_smiles = 200
output_file = "libinvent_samples.csv"
unique_molecules = true
randomize_smiles = true
temperature = 1.0
```

### G. Sampling (Linkinvent)
```toml
run_type = "sampling"
device = "cuda:0"
json_out_config = "linkinvent_sampling.json"

[parameters]
model_file = "priors/linkinvent.prior"
smiles_file = "warheads.smi"  # two warheads per line separated by '|'
num_smiles = 200
output_file = "linkinvent_samples.csv"
unique_molecules = true
randomize_smiles = true
temperature = 1.0
```

### H. Sampling (Pepinvent)
```toml
run_type = "sampling"
device = "cuda:0"
json_out_config = "pepinvent_sampling.json"

[parameters]
model_file = "priors/pepinvent.prior"
smiles_file = "pepinvent.smi"  # masked peptide input per line
num_smiles = 200
output_file = "pepinvent_samples.csv"
unique_molecules = true
randomize_smiles = true
temperature = 1.0
```

---

## 8. Standalone Example Configs (no run_type)

### A. Create Model (create_reinvent.py)
```toml
[io]
smiles_file = "data/train.smi"
model_file = "empty.model"

[network]
num_layers = 3
layer_size = 512
dropout = 0.0
max_sequence_length = 256
cell_type = "lstm"
embedding_layer_size = 256
layer_normalization = false
standardize = true

[metadata]
data_source = "ChEMBL"
comment = "Empty prior for transfer learning"
```

### B. Data Pipeline (datapipeline)
```toml
input_csv_file = "data/input.csv"
smiles_column = "SMILES"
separator = "\t"
output_smiles_file = "processed.smi"
num_procs = 4
chunk_size = 500

[filter]
elements = ["C", "N", "O", "F", "S", "Cl", "Br", "I"]
transforms = ["standard"]
min_heavy_atoms = 2
max_heavy_atoms = 90
max_mol_weight = 1200.0
min_carbons = 2
max_num_rings = 12
max_ring_size = 7
keep_stereo = true
keep_isotope_molecules = true
uncharge = true
canonical_tautomer = false
kekulize = false
randomize_smiles = false
report_errors = false
inchi_key_deduplicate = false
```
