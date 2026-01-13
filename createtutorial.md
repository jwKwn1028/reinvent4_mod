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
