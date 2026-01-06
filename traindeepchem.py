"""
Train single-task DeepChem DMPNN regressors (Lightning + Optuna optional).

Drop-in replacement for your training file with fixes for:
- node /tmp being full (uses TMPDIR if set; otherwise out_dir/_tmp)
- ruff: args not defined (no global args usage inside helpers)
- DeepChem DiskDataset.load_from_disk missing in your DeepChem version
- Optuna failing due to LearningRateMonitor with no logger (conditionally adds LR monitor)
- repeated split/transform in Optuna trials (prepare once, reuse)
- clearer error when target column (e.g. "mp") is missing in the CSV
- safer caching: cache can be rebuilt if corrupt/partial
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, cast

import joblib
import numpy as np
import pandas as pd
import torch

import deepchem as dc
from deepchem.models import DMPNNModel
from sklearn.metrics import r2_score

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

import optuna
from optuna.integration import PyTorchLightningPruningCallback

# -------------------------
# Performance settings
# -------------------------
if torch.cuda.is_available():
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("train_dmpnn_single_task")

# -------------------------
# Global: disable WandB after first failure to prevent Optuna spam
# -------------------------
_WANDB_GLOBALLY_DISABLED = False


# ============================================================
# Tempdir / cache utilities
# ============================================================
def _resolve_tmp_root(out_dir: str) -> str:
    """
    Prefer TMPDIR (set in Slurm script) to avoid node-local /tmp.
    Fall back to out_dir/_tmp.
    """
    tmp_root = os.environ.get("TMPDIR")
    if not tmp_root:
        tmp_root = os.path.join(out_dir, "_tmp")
    os.makedirs(tmp_root, exist_ok=True)
    tempfile.tempdir = tmp_root
    return tmp_root


def _diskdataset_ready(path: str) -> bool:
    # DeepChem DiskDataset has metadata.csv.gzip
    return os.path.isfile(os.path.join(path, "metadata.csv.gzip"))


def _load_diskdataset(path: str) -> dc.data.Dataset:
    """
    Your DeepChem build does NOT have DiskDataset.load_from_disk().
    The compatible load is: DiskDataset(path).
    """
    ds = dc.data.DiskDataset(path)
    if len(ds) == 0:
        raise RuntimeError(f"DiskDataset at {path} is empty/corrupt.")
    return ds


def _safe_select(ds: dc.data.Dataset, inds: List[int], out_dir: str) -> dc.data.Dataset:
    """
    Select indices into a stable directory to avoid hidden temp usage.
    Works on DiskDataset (select_dir) and falls back otherwise.
    """
    os.makedirs(out_dir, exist_ok=True)
    try:
        return ds.select(inds, select_dir=out_dir)  
    except TypeError:
        return ds.select(inds)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _coerce_devices(devices: Any) -> Any:
    if isinstance(devices, str) and devices.isdigit():
        return int(devices)
    return devices


# ============================================================
# DeepChem <-> Lightning data plumbing
# ============================================================
def collate_dataset_fn(batch_data: List[Tuple[Any, Any, Any, Any]], model: Any):
    """
    Collate a batch of (X, y, w, ids) into DMPNN-compatible tensors using the model's generator.

    CRITICAL: Use batch_size=len(X) so generator yields exactly one batch containing
    all incoming items. Prevents silent sample drop when model.batch_size differs.
    """
    X = [item[0] for item in batch_data]
    Y = [item[1] for item in batch_data]
    W = [item[2] for item in batch_data]
    ids = [item[3] for item in batch_data]

    dataset = dc.data.NumpyDataset(X, Y, W, ids)
    generator = model.default_generator(dataset, batch_size=len(X), pad_batches=False)
    processed_batch = next(iter(generator))
    return model._prepare_batch(processed_batch)


class DeepChemDataModule(L.LightningDataModule):
    """
    Custom DataModule to handle Train/Valid/Test splits for DeepChem.
    """

    def __init__(
        self,
        train_ds: dc.data.Dataset,
        valid_ds: dc.data.Dataset,
        test_ds: dc.data.Dataset,
        batch_size: int,
        model: Any,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.test_ds = test_ds
        self.batch_size = int(batch_size)
        self.model = model
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)

    def _create_dataloader(self, dataset: dc.data.Dataset, shuffle: bool):
        if hasattr(dc.data, "_TorchIndexDiskDataset"):
            torch_ds = dc.data._TorchIndexDiskDataset(dataset)
        else:

            class SimpleWrapper(torch.utils.data.Dataset):
                def __init__(self, d: dc.data.Dataset):
                    self.d = d

                def __len__(self) -> int:
                    return len(self.d)

                def __getitem__(self, i: int):
                    return (self.d.X[i], self.d.y[i], self.d.w[i], self.d.ids[i])

            torch_ds = SimpleWrapper(dataset)

        def collate_wrapper(batch):
            return collate_dataset_fn(batch, model=self.model)

        return torch.utils.data.DataLoader(
            torch_ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=collate_wrapper,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def train_dataloader(self):
        return self._create_dataloader(self.train_ds, shuffle=True)

    def val_dataloader(self):
        return self._create_dataloader(self.valid_ds, shuffle=False)

    def test_dataloader(self):
        return self._create_dataloader(self.test_ds, shuffle=False)

    def make_loader(self, dataset: dc.data.Dataset, shuffle: bool = False):
        return self._create_dataloader(dataset, shuffle=shuffle)


class DMPNNLightningModule(L.LightningModule):
    """
    LightningModule wrapper around DeepChem DMPNNModel.
    """

    def __init__(
        self,
        dc_model: DMPNNModel,
        learning_rate: float,
        *,
        lr_plateau_factor: float = 0.5,
        lr_plateau_patience: int = 5,
        lr_plateau_min_lr: float = 1e-6,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["dc_model"])
        self.dc_model = dc_model
        self.net = dc_model.model

        self.learning_rate = float(learning_rate)
        self.lr_plateau_factor = float(lr_plateau_factor)
        self.lr_plateau_patience = int(lr_plateau_patience)
        self.lr_plateau_min_lr = float(lr_plateau_min_lr)

    @staticmethod
    def _unwrap(x: Any) -> Any:
        if isinstance(x, list) and len(x) == 1:
            return x[0]
        return x

    @staticmethod
    def _ensure_tensor(x: Any) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x
        return torch.as_tensor(x)

    def _move_to_device(self, obj: Any) -> Any:
        if obj is None:
            return None
        if isinstance(obj, torch.Tensor):
            return obj.to(self.device, non_blocking=False)
        if hasattr(obj, "to") and callable(getattr(obj, "to")):
            try:
                return obj.to(self.device)
            except Exception:
                return obj
        if isinstance(obj, (list, tuple)):
            return type(obj)(self._move_to_device(x) for x in obj)
        if isinstance(obj, dict):
            return {k: self._move_to_device(v) for k, v in obj.items()}
        return obj

    def _unpack_batch(self, batch: Any) -> Tuple[Any, torch.Tensor, torch.Tensor]:
        if not isinstance(batch, (tuple, list)) or len(batch) < 3:
            raise ValueError(f"Unexpected batch format from _prepare_batch: {type(batch)} / {batch}")

        inputs, labels, weights = batch[0], batch[1], batch[2]
        inputs = self._move_to_device(inputs)
        labels = self._move_to_device(self._ensure_tensor(self._unwrap(labels)).float())
        weights = self._move_to_device(self._ensure_tensor(self._unwrap(weights)).float())
        return inputs, labels, weights

    def _forward_outputs(self, inputs: Any) -> torch.Tensor:
        outputs = self.net(inputs)
        outputs = self._unwrap(outputs)
        if not isinstance(outputs, torch.Tensor):
            outputs = torch.as_tensor(outputs, device=self.device)
        return outputs.float()

    @staticmethod
    def _align_shapes(
        outputs: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if outputs.shape != labels.shape:
            outputs = outputs.view_as(labels)
        if weights.shape != labels.shape:
            weights = weights.view_as(labels)
        return outputs, labels, weights

    def _weighted_mse(self, outputs: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        outputs, labels, weights = self._align_shapes(outputs, labels, weights)
        diff = outputs - labels
        weighted_sq = (diff * diff) * weights
        denom = weights.sum().clamp_min(1e-8)
        return weighted_sq.sum() / denom

    def _weighted_mae(self, outputs: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        outputs, labels, weights = self._align_shapes(outputs, labels, weights)
        abs_err = (outputs - labels).abs() * weights
        denom = weights.sum().clamp_min(1e-8)
        return abs_err.sum() / denom

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        inputs, labels, weights = self._unpack_batch(batch)
        outputs = self._forward_outputs(inputs)
        loss = self._weighted_mse(outputs, labels, weights)
        bs = int(labels.shape[0]) if labels.ndim > 0 else 1
        self.log("train_loss", loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=bs)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        inputs, labels, weights = self._unpack_batch(batch)
        outputs = self._forward_outputs(inputs)
        loss = self._weighted_mse(outputs, labels, weights)
        mae = self._weighted_mae(outputs, labels, weights)
        bs = int(labels.shape[0]) if labels.ndim > 0 else 1
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=bs)
        self.log("val_mae_norm", mae, prog_bar=False, on_step=False, on_epoch=True, batch_size=bs)
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        inputs, labels, weights = self._unpack_batch(batch)
        outputs = self._forward_outputs(inputs)
        loss = self._weighted_mse(outputs, labels, weights)
        mae = self._weighted_mae(outputs, labels, weights)
        bs = int(labels.shape[0]) if labels.ndim > 0 else 1
        self.log("test_loss", loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=bs)
        self.log("test_mae_norm", mae, prog_bar=False, on_step=False, on_epoch=True, batch_size=bs)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.lr_plateau_factor,
            patience=self.lr_plateau_patience,
            min_lr=self.lr_plateau_min_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }


# ============================================================
# Metrics / prediction utilities
# ============================================================
def _untransform_y_array(y: np.ndarray, transformer: Optional[dc.trans.Transformer]) -> np.ndarray:
    y = np.asarray(y)
    if transformer is None:
        return y.reshape(-1)
    y2d = y.reshape(-1, 1) if y.ndim == 1 else y
    try:
        y_un = transformer.untransform(y2d)
    except Exception:
        y_un = y2d
    return np.asarray(y_un).reshape(-1)


@torch.no_grad()
def _predict_norm_with_pl(pl_module: DMPNNLightningModule, loader: torch.utils.data.DataLoader) -> np.ndarray:
    """
    Predict in normalized y space using Lightning forward loop.
    """
    pl_module.eval()
    preds: List[np.ndarray] = []
    for batch in loader:
        inputs, labels, weights = pl_module._unpack_batch(batch)
        outputs = pl_module._forward_outputs(inputs)
        outputs, labels, weights = pl_module._align_shapes(outputs, labels, weights)
        preds.append(outputs.detach().cpu().numpy())

    if not preds:
        return np.zeros((0, 1), dtype=np.float32)

    y_pred = np.concatenate(preds, axis=0)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    elif y_pred.ndim == 2 and y_pred.shape[1] != 1:
        y_pred = y_pred[:, :1]
    return y_pred.astype(np.float32, copy=False)


def _save_predictions_csv(
    out_path: str,
    smiles: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_name: str,
) -> None:
    n = min(len(smiles), len(y_true), len(y_pred))
    df = pd.DataFrame(
        {
            "smiles": smiles[:n],
            f"y_true_{task_name}": y_true[:n],
            f"y_pred_{task_name}": y_pred[:n],
        }
    )
    df.to_csv(out_path, index=False)


def _detect_global_features_size(dataset: dc.data.Dataset) -> int:
    try:
        sample_graph = dataset.X[0]
        gf = getattr(sample_graph, "global_features", None)
        if gf is None:
            return 0
        return int(len(gf))
    except Exception:
        return 0


# ============================================================
# WandB helper (safe)
# ============================================================
def _maybe_wandb_login() -> None:
    if not os.environ.get("WANDB_API_KEY"):
        return
    try:
        import wandb  # noqa: F401

        try:
            wandb.login(key=os.environ.get("WANDB_API_KEY"), relogin=True)
        except Exception:
            pass
    except Exception:
        return


def _make_wandb_logger_safe(
    *,
    project: str,
    entity: Optional[str],
    name: str,
    group: str,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[WandbLogger]:
    global _WANDB_GLOBALLY_DISABLED
    if _WANDB_GLOBALLY_DISABLED:
        return None

    try:
        _maybe_wandb_login()
        try:
            return WandbLogger(
                project=project,
                entity=entity,
                name=name,
                group=group,
                config = config,
                log_model=False,
                reinit="finish_previous",
            )
        except TypeError:
            return WandbLogger(
                project=project,
                entity=entity,
                name=name,
                group=group,
                config=config,
                log_model=False,
                reinit=True,
            )
    except Exception as e:
        logger.warning("WandB init failed. Disabling WandB for this process. Error: %s", str(e))
        _WANDB_GLOBALLY_DISABLED = True
        return None


# ============================================================
# Data preparation (cached once per task)
# ============================================================
@dataclass
class PreparedTaskData:
    train_inds: List[int]
    valid_inds: List[int]
    test_inds: List[int]
    train_raw: dc.data.Dataset
    valid_raw: dc.data.Dataset
    test_raw: dc.data.Dataset
    train_ds: dc.data.Dataset
    valid_ds: dc.data.Dataset
    test_ds: dc.data.Dataset
    transformer: dc.trans.Transformer
    prep_root: str


def _validate_csv_has_columns(csv_path: str, required: List[str]) -> None:
    # Read just headers (and maybe a few rows) to validate columns cheaply.
    df = pd.read_csv(csv_path, nrows=5)
    cols = set(df.columns.astype(str))
    missing = [c for c in required if c not in cols]
    if missing:
        raise KeyError(
            "CSV is missing required columns: "
            + ", ".join(missing)
            + f"\nCSV: {csv_path}\nAvailable columns: {sorted(cols)}"
        )


def load_and_featurize(
    csv_path: str,
    task_name: str,
    *,
    label_column: Optional[str] = None,
    feature_field: str = "smiles",
    id_field: str = "smiles",
    features_generators: Optional[List[str]] = None,
    dataset_cache_dir: Optional[str] = None,
    shard_size: int = 1024,
) -> dc.data.Dataset:
    """
    Creates (or loads) a DiskDataset cached under dataset_cache_dir.
    """
    label_column = label_column or task_name

    logger.info("[%s] Featurizing data from: %s", task_name, csv_path)
    _validate_csv_has_columns(csv_path, required=[feature_field, id_field, label_column])

    if features_generators is not None:
        features_generators = [g for g in features_generators if g and str(g).lower() not in {"none", "null"}]
        if not features_generators:
            features_generators = None

    if dataset_cache_dir is None:
        dataset_cache_dir = os.path.join("models", "_dc_cache", task_name)
    os.makedirs(dataset_cache_dir, exist_ok=True)

    # Load cache if present and valid
    if _diskdataset_ready(dataset_cache_dir):
        try:
            logger.info("[%s] Loading cached DiskDataset: %s", task_name, dataset_cache_dir)
            return _load_diskdataset(dataset_cache_dir)
        except Exception as e:
            logger.warning("[%s] Cache exists but failed to load (%s). Rebuilding.", task_name, e)
            shutil.rmtree(dataset_cache_dir, ignore_errors=True)
            os.makedirs(dataset_cache_dir, exist_ok=True)

    featurizer = dc.feat.DMPNNFeaturizer(features_generators=features_generators)
    loader = dc.data.CSVLoader(
        tasks=[label_column],
        feature_field=feature_field,
        id_field=id_field,
        featurizer=featurizer,
    )

    ds = loader.create_dataset(
        csv_path,
        data_dir=dataset_cache_dir,  # forces writing into cache_dir (not /tmp)
        shard_size=shard_size,
    )
    if len(ds) == 0:
        raise RuntimeError(f"[{task_name}] Dataset is empty after featurize: {csv_path}")
    return ds


def prepare_task_data(
    *,
    dataset: dc.data.Dataset,
    task_name: str,
    out_dir: str,
    seed: int,
) -> PreparedTaskData:
    """
    Do scaffold split + select + normalization fit + transform ONCE.
    Stores everything under out_dir/_prep/task_name and reuses on reruns.
    Check the dimension with cached data
    """
    prep_root = os.path.join(out_dir, "_prep", task_name)
    os.makedirs(prep_root, exist_ok=True)

    # --- NEW SAFETY CHECK ---
    # Check if cached data dimensions match the current input dataset
    train_raw_dir = os.path.join(prep_root, "raw", "train")
    if _diskdataset_ready(train_raw_dir):
        # Load just one sample to check features
        try:
            cached_ds = _load_diskdataset(train_raw_dir)
            current_gf = _detect_global_features_size(dataset)
            cached_gf = _detect_global_features_size(cached_ds)
            
            if current_gf != cached_gf:
                logger.warning(
                    "[%s] Stale cache detected! Current global features: %d, Cached: %d. Rebuilding cache.", 
                    task_name, current_gf, cached_gf
                )
                shutil.rmtree(prep_root)
                os.makedirs(prep_root, exist_ok=True)
        except Exception:
            # If load fails, just rebuild
            shutil.rmtree(prep_root, ignore_errors=True)
            os.makedirs(prep_root, exist_ok=True)
            
    # 1) Split (cached)
    split_npz = os.path.join(prep_root, "split_indices.npz")
    if os.path.exists(split_npz):
        arr = np.load(split_npz)
        train_inds = arr["train"].astype(int).tolist()
        valid_inds = arr["valid"].astype(int).tolist()
        test_inds = arr["test"].astype(int).tolist()
        logger.info("[%s] Loaded cached split indices: %s", task_name, split_npz)
    else:
        splitter = dc.splits.ScaffoldSplitter()
        train_inds, valid_inds, test_inds = splitter.split(dataset, seed=seed)
        np.savez(
            split_npz,
            train=np.array(train_inds, dtype=np.int64),
            valid=np.array(valid_inds, dtype=np.int64),
            test=np.array(test_inds, dtype=np.int64),
        )
        logger.info("[%s] Saved split indices: %s", task_name, split_npz)

    # 2) Select raw splits (cached to disk dirs)
    raw_root = os.path.join(prep_root, "raw")
    train_raw_dir = os.path.join(raw_root, "train")
    valid_raw_dir = os.path.join(raw_root, "valid")
    test_raw_dir = os.path.join(raw_root, "test")

    if _diskdataset_ready(train_raw_dir):
        train_raw = _load_diskdataset(train_raw_dir)
        valid_raw = _load_diskdataset(valid_raw_dir)
        test_raw = _load_diskdataset(test_raw_dir)
        logger.info("[%s] Loaded cached raw splits under: %s", task_name, raw_root)
    else:
        train_raw = _safe_select(dataset, train_inds, train_raw_dir)
        valid_raw = _safe_select(dataset, valid_inds, valid_raw_dir)
        test_raw = _safe_select(dataset, test_inds, test_raw_dir)
        logger.info("[%s] Materialized raw splits under: %s", task_name, raw_root)

    # 3) Fit/load transformer once
    transformer_path = os.path.join(prep_root, "y_transformer.joblib")
    if os.path.exists(transformer_path):
        transformer = joblib.load(transformer_path)
        logger.info("[%s] Loaded cached transformer: %s", task_name, transformer_path)
    else:
        transformer = dc.trans.NormalizationTransformer(transform_y=True, dataset=train_raw)
        joblib.dump(transformer, transformer_path)
        logger.info("[%s] Saved transformer: %s", task_name, transformer_path)

    # 4) Transform once to disk
    trans_root = os.path.join(prep_root, "transformed")
    train_dir = os.path.join(trans_root, "train")
    valid_dir = os.path.join(trans_root, "valid")
    test_dir = os.path.join(trans_root, "test")

    def _load_or_make(raw_ds: dc.data.Dataset, out_path: str) -> dc.data.Dataset:
        if _diskdataset_ready(out_path):
            return _load_diskdataset(out_path)
        os.makedirs(out_path, exist_ok=True)
        return transformer.transform(raw_ds, out_dir=out_path)

    train_ds = _load_or_make(train_raw, train_dir)
    valid_ds = _load_or_make(valid_raw, valid_dir)
    test_ds = _load_or_make(test_raw, test_dir)

    return PreparedTaskData(
        train_inds=train_inds,
        valid_inds=valid_inds,
        test_inds=test_inds,
        train_raw=train_raw,
        valid_raw=valid_raw,
        test_raw=test_raw,
        train_ds=train_ds,
        valid_ds=valid_ds,
        test_ds=test_ds,
        transformer=transformer,
        prep_root=prep_root,
    )


# ============================================================
# Core training (reuses PreparedTaskData)
# ============================================================
def train_single_task_lightning(
    *,
    prepared: PreparedTaskData,
    full_dataset: dc.data.Dataset,
    task_name: str,
    out_dir: str,
    seed: int,
    nb_epoch: int,
    batch_size: int,
    save_splits: bool,
    save_predictions_for: Tuple[str, ...],
    accelerator: str,
    devices: Any,
    strategy: str,
    precision: str,
    learning_rate: float = 1e-4,
    ffn_dropout_p: float = 0.1,
    depth: int = 3,
    ffn_hidden: int = 300,
    ffn_layers: int = 3,
    early_stop_patience: int = 20,
    early_stop_patience_optuna: int = 10,
    lr_plateau_factor: float = 0.5,
    lr_plateau_patience: int = 5,
    lr_plateau_min_lr: float = 1e-6,
    log_every_n_steps: int = 10,
    gradient_clip_val: float = 1.0,
    deterministic: bool = True,
    enable_progress_bar: Optional[bool] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    optuna_trial: Optional[optuna.trial.Trial] = None,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
) -> Tuple[Dict[str, float], Dict[str, str], float]:
    """
    If optuna_trial is provided, trains in a temp trial dir and deletes it after.
    Otherwise trains a "final" model under out_dir/task_name.
    
    Update: 
    - Fixes DDP race condition by guarding file writes with global_rank == 0.
    - Calculates and logs R^2 score.
    """
    # Ensure sklearn is available for R2 calculation

    L.seed_everything(seed, workers=True)

    is_optimizing = optuna_trial is not None
    tmp_root = _resolve_tmp_root(out_dir)

    task_dir = (
        tempfile.mkdtemp(prefix=f"optuna_{task_name}_", dir=tmp_root)
        if is_optimizing
        else os.path.join(out_dir, task_name)
    )
    if not is_optimizing:
        _ensure_dir(task_dir)

    if save_splits and (not is_optimizing):
        if os.environ.get("LOCAL_RANK", "0") == "0":
             np.savez(
                os.path.join(task_dir, f"{task_name}_scaffold_split_indices.npz"),
                train=np.array(prepared.train_inds, dtype=np.int64),
                valid=np.array(prepared.valid_inds, dtype=np.int64),
                test=np.array(prepared.test_inds, dtype=np.int64),
            )

    try:
        train_raw = prepared.train_raw
        valid_raw = prepared.valid_raw
        test_raw = prepared.test_raw
        train_ds = prepared.train_ds
        valid_ds = prepared.valid_ds
        test_ds = prepared.test_ds
        transformer = prepared.transformer

        # Model
        global_features_size = _detect_global_features_size(full_dataset)
        logger.info("[%s] Detected global_features_size: %d", task_name, global_features_size)

        model = DMPNNModel(
            mode="regression",
            n_tasks=1,
            batch_size=batch_size,
            model_dir=task_dir,
            learning_rate=learning_rate,
            n_steps=depth,
            ffn_hidden_dim=ffn_hidden,
            ffn_num_layers=ffn_layers,
            dropout_p=ffn_dropout_p,
            bias=True,
            global_features_size=global_features_size,
        )

        pl_module = DMPNNLightningModule(
            model,
            learning_rate=learning_rate,
            lr_plateau_factor=lr_plateau_factor,
            lr_plateau_patience=lr_plateau_patience,
            lr_plateau_min_lr=lr_plateau_min_lr,
        )

        datamodule = DeepChemDataModule(
            train_ds=train_ds,
            valid_ds=valid_ds,
            test_ds=test_ds,
            batch_size=batch_size,
            model=model,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        loggers: List[Any] = [TensorBoardLogger(save_dir=os.path.join(task_dir, "tb_logs"), name="")]

        wandb_enabled = bool(wandb_project) and (not _WANDB_GLOBALLY_DISABLED)
        if wandb_enabled:
            group = "optuna_search" if is_optimizing else "production_train"
            trial_num = optuna_trial.number if optuna_trial is not None else 0
            run_name = f"trial_{trial_num}" if is_optimizing else (wandb_run_name or f"{task_name}_final")

            hyperparams_config = {
                "task_name": task_name,
                "batch_size": batch_size,
                "epochs": nb_epoch,
                "seed": seed,
                "learning_rate": learning_rate,
                "depth": depth,
                "ffn_hidden": ffn_hidden,
                "ffn_layers": ffn_layers,
                "ffn_dropout_p": ffn_dropout_p,
                "global_features_size": global_features_size,
            }
            
            w_logger = _make_wandb_logger_safe(
                project=cast(str, wandb_project),
                entity=wandb_entity,
                name=run_name,
                group=group,
                config=hyperparams_config, # <--- Pass the dict
            )
            if w_logger is not None:
                loggers.append(w_logger)

        # Callbacks
        callbacks: List[Any] = []

        if loggers:
            callbacks.append(LearningRateMonitor(logging_interval="step"))

        monitor_metric = "val_loss"
        callbacks.append(
            EarlyStopping(
                monitor=monitor_metric,
                patience=early_stop_patience_optuna if is_optimizing else early_stop_patience,
                mode="min",
            )
        )

        ckpt_cb = ModelCheckpoint(
            dirpath=os.path.join(task_dir, "checkpoints"),
            filename="best-{epoch:02d}-{val_loss:.4f}",
            monitor=monitor_metric,
            mode="min",
            save_top_k=1,
            save_last=True,
        )
        callbacks.append(ckpt_cb)

        if optuna_trial is not None:
            callbacks.append(PyTorchLightningPruningCallback(optuna_trial, monitor=monitor_metric))

        trainer = L.Trainer(
            accelerator=accelerator,
            devices=_coerce_devices(devices),
            strategy=strategy,
            precision=cast(Any, precision),
            logger=loggers,
            callbacks=callbacks,
            enable_checkpointing=True,
            max_epochs=nb_epoch,
            log_every_n_steps=log_every_n_steps,
            gradient_clip_val=gradient_clip_val,
            deterministic=deterministic,
            enable_progress_bar=(not is_optimizing) if enable_progress_bar is None else bool(enable_progress_bar),
        )

        trainer.fit(pl_module, datamodule=datamodule)

        # Reload best weights
        best_path = ckpt_cb.best_model_path
        if best_path and os.path.exists(best_path):
            logger.info("[%s] Loading best model from: %s", task_name, best_path)
            checkpoint = torch.load(best_path, map_location="cpu")
            pl_module.load_state_dict(checkpoint["state_dict"])
        else:
            logger.warning("[%s] No best checkpoint found. Using last epoch weights.", task_name)

        # ---------------------------------------------------------------------
        # VALIDATION METRICS (Calculated on all ranks, but only rank 0 usually acts on them)
        # ---------------------------------------------------------------------
        valid_loader = datamodule.val_dataloader()
        y_pred_valid_norm_2d = _predict_norm_with_pl(pl_module, valid_loader)
        y_pred_valid = _untransform_y_array(y_pred_valid_norm_2d, transformer)
        y_true_valid = np.asarray(valid_raw.y).reshape(-1)
        
        val_mae = float(np.mean(np.abs(y_true_valid - y_pred_valid)))
        val_r2 = float(r2_score(y_true_valid, y_pred_valid))

        if is_optimizing:
            return {}, {}, val_mae

        # ---------------------------------------------------------------------
        # SAVE PREDICTIONS & TEST METRICS
        # ---------------------------------------------------------------------
        pred_files: Dict[str, str] = {}
        scores: Dict[str, float] = {}

        # Helper to run prediction and optionally save
        def predict_and_maybe_save(ds_norm: dc.data.Dataset, ds_raw: dc.data.Dataset, split_name: str) -> Tuple[float, float, str]:
            loader = datamodule.make_loader(ds_norm, shuffle=False)
            y_pred_norm_2d = _predict_norm_with_pl(pl_module, loader)
            y_pred = _untransform_y_array(y_pred_norm_2d, transformer)
            y_true = np.asarray(ds_raw.y).reshape(-1)
            
            # Metrics
            mae = float(np.mean(np.abs(y_true - y_pred)))
            r2 = float(r2_score(y_true, y_pred))
            
            out_path = os.path.join(task_dir, f"{task_name}_{split_name}_predictions.csv")

            if trainer.global_rank == 0:
                _save_predictions_csv(out_path, np.asarray(ds_raw.ids), y_true, y_pred, task_name)
            
            return mae, r2, out_path

        splits_to_save = set(save_predictions_for)
        

        # 1. Predict for configured splits (Train/Valid/Test/All)
        if "train" in splits_to_save:
            _, _, path = predict_and_maybe_save(train_ds, train_raw, "train")
            if trainer.global_rank == 0: 
                pred_files["train"] = path
            
        if "valid" in splits_to_save:
            # We already have valid metrics above, but this re-runs to save CSV if requested
            # Optimized: reuse calculation? No, simplistic is safer for now.
            _, _, path = predict_and_maybe_save(valid_ds, valid_raw, "valid")
            if trainer.global_rank == 0: 
                pred_files["valid"] = path

        # 2. Always compute TEST metrics for the final report
        test_mae, test_r2, test_path = predict_and_maybe_save(test_ds, test_raw, "test")
        if "test" in splits_to_save and trainer.global_rank == 0:
            pred_files["test"] = test_path

        if "all" in splits_to_save:
            all_raw = full_dataset
            all_norm = transformer.transform(all_raw)
            _, _, path = predict_and_maybe_save(all_norm, all_raw, "all")
            if trainer.global_rank == 0: 
                pred_files["all"] = path

        scores = {
            "val_MAE": val_mae,
            "val_R2": val_r2,
            "test_MAE": test_mae,
            "test_R2": test_r2
        }

        if trainer.global_rank == 0:
            joblib.dump(transformer, os.path.join(task_dir, "y_transformer.joblib"))
            with open(os.path.join(task_dir, f"{task_name}_scores.json"), "w") as f:
                json.dump(scores, f, indent=2)

        return scores, pred_files, val_mae

    finally:
        try:
            import wandb  # type: ignore
            wandb.finish()
        except Exception:
            pass


        if is_optimizing and os.path.exists(task_dir):
            shutil.rmtree(task_dir, ignore_errors=True)

        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


# ============================================================
# Optuna
# ============================================================
def optimize_task(
    *,
    n_trials: int,
    dataset: dc.data.Dataset,
    prepared: PreparedTaskData,
    task_name: str,
    base_args: Any,
    search_space: Dict[str, Any],
) -> Dict[str, Any]:
    logger.info(">>> Starting Optuna Optimization for %s (%d trials) <<<", task_name, n_trials)

    def objective(trial: optuna.trial.Trial) -> float:
        params: Dict[str, Any] = {}

        if "learning_rate" in search_space:
            cfg = search_space["learning_rate"]
            params["learning_rate"] = trial.suggest_float(
                "learning_rate", cfg.get("low", 1e-5), cfg.get("high", 1e-3), log=cfg.get("log", True)
            )
        else:
            params["learning_rate"] = 1e-4

        if "ffn_dropout_p" in search_space:
            cfg = search_space["ffn_dropout_p"]
            params["ffn_dropout_p"] = trial.suggest_float("ffn_dropout_p", cfg.get("low", 0.0), cfg.get("high", 0.5))
        else:
            params["ffn_dropout_p"] = 0.1

        if "depth" in search_space:
            cfg = search_space["depth"]
            params["depth"] = trial.suggest_int("depth", cfg.get("low", 5), cfg.get("high", 15))
        else:
            params["depth"] = 6

        if "ffn_hidden" in search_space:
            cfg = search_space["ffn_hidden"]
            if isinstance(cfg, list):
                params["ffn_hidden"] = trial.suggest_categorical("ffn_hidden", cfg)
            elif isinstance(cfg, dict):
                params["ffn_hidden"] = trial.suggest_int("ffn_hidden", cfg.get("low", 128), cfg.get("high", 1024))
        else:
            params["ffn_hidden"] = 256

        if "ffn_layers" in search_space:
            cfg = search_space["ffn_layers"]
            params["ffn_layers"] = trial.suggest_int("ffn_layers", cfg.get("low", 2), cfg.get("high", 12))
        else:
            params["ffn_layers"] = 6

        _, _, val_mae = train_single_task_lightning(
            prepared=prepared,
            full_dataset=dataset,
            task_name=task_name,
            out_dir=base_args.out_dir,
            seed=base_args.seed,
            nb_epoch=base_args.epochs,
            batch_size=base_args.batch_size,
            save_splits=False,
            save_predictions_for=tuple(),
            accelerator=base_args.accelerator,
            devices=base_args.devices,
            strategy=base_args.strategy,
            precision=base_args.precision,
            learning_rate=params["learning_rate"],
            ffn_dropout_p=params["ffn_dropout_p"],
            depth=params["depth"],
            ffn_hidden=params["ffn_hidden"],
            ffn_layers=params["ffn_layers"],
            early_stop_patience=base_args.early_stop_patience,
            early_stop_patience_optuna=base_args.early_stop_patience_optuna,
            lr_plateau_factor=base_args.lr_plateau_factor,
            lr_plateau_patience=base_args.lr_plateau_patience,
            lr_plateau_min_lr=base_args.lr_plateau_min_lr,
            log_every_n_steps=base_args.log_every_n_steps,
            gradient_clip_val=base_args.gradient_clip_val,
            deterministic=base_args.deterministic,
            enable_progress_bar=False,
            num_workers=base_args.num_workers,
            pin_memory=base_args.pin_memory,
            optuna_trial=trial,
            wandb_project=base_args.wandb_project,
            wandb_entity=base_args.wandb_entity,
        )
        return float(val_mae)

    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_trials, catch=(RuntimeError, OSError, KeyError))

    logger.info(">>> Optimization Complete for %s <<<", task_name)
    logger.info("Best Trial: %d", study.best_trial.number)
    logger.info("Best Val MAE: %.6f", study.best_value)
    return dict(study.best_params)


# ============================================================
# Config parsing (YAML)
# ============================================================
def _read_yaml_config(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError("PyYAML not installed. pip install pyyaml") from e
    with open(path, "r") as f:
        return cast(Dict[str, Any], yaml.safe_load(f) or {})


def _as_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    if x is None:
        return default
    try:
        return int(x)
    except Exception:
        return default


def _as_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    if x is None:
        return default
    try:
        return float(x)
    except Exception:
        return default


def _as_bool(x: Any, default: Optional[bool] = None) -> Optional[bool]:
    if x is None:
        return default
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _bool_arg(x: Any) -> bool:
    v = _as_bool(x, default=None)
    if v is None:
        raise argparse.ArgumentTypeError(f"Expected boolean, got {x}")
    return v


def _as_list_str(x: Any) -> Optional[List[str]]:
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return [str(v) for v in x]
    if isinstance(x, str):
        return [p.strip() for p in x.split(",") if p.strip()]
    return [str(x)]


def _config_to_parser_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {}

    run = cfg.get("run", {})
    if run.get("out_dir"):
        defaults["out_dir"] = str(run["out_dir"])
    if run.get("epochs") is not None:
        defaults["epochs"] = _as_int(run["epochs"])
    if run.get("batch_size") is not None:
        defaults["batch_size"] = _as_int(run["batch_size"])
    if run.get("seed") is not None:
        defaults["seed"] = _as_int(run["seed"])
    if run.get("save_predictions_for") is not None:
        defaults["save_predictions_for"] = _as_list_str(run["save_predictions_for"])
    if run.get("save_splits") is not None:
        defaults["no_save_splits"] = not bool(_as_bool(run["save_splits"], True))

    fe = cfg.get("featurizer", {})
    if fe.get("features_generators") is not None:
        defaults["features_generators"] = _as_list_str(fe["features_generators"])
    if fe.get("feature_field") is not None:
        defaults["feature_field"] = str(fe["feature_field"])
    if fe.get("id_field") is not None:
        defaults["id_field"] = str(fe["id_field"])
    if fe.get("shard_size") is not None:
        defaults["shard_size"] = _as_int(fe["shard_size"], 1024)

    dmp = cfg.get("dmpnn", {})
    for k in ["learning_rate", "ffn_dropout_p", "depth", "ffn_hidden", "ffn_layers"]:
        if dmp.get(k) is not None:
            defaults[k] = dmp[k]

    cb = cfg.get("callbacks", {})
    if cb.get("early_stop_patience") is not None:
        defaults["early_stop_patience"] = _as_int(cb["early_stop_patience"])
    if cb.get("early_stop_patience_optuna") is not None:
        defaults["early_stop_patience_optuna"] = _as_int(cb["early_stop_patience_optuna"])

    sch = cfg.get("scheduler", {})
    if sch.get("lr_plateau_factor") is not None:
        defaults["lr_plateau_factor"] = _as_float(sch["lr_plateau_factor"])
    if sch.get("lr_plateau_patience") is not None:
        defaults["lr_plateau_patience"] = _as_int(sch["lr_plateau_patience"])
    if sch.get("lr_plateau_min_lr") is not None:
        defaults["lr_plateau_min_lr"] = _as_float(sch["lr_plateau_min_lr"])

    lt = cfg.get("lightning", {})
    for k in ["accelerator", "devices", "strategy", "precision"]:
        if lt.get(k) is not None:
            defaults[k] = str(lt[k])
    if lt.get("log_every_n_steps") is not None:
        defaults["log_every_n_steps"] = _as_int(lt["log_every_n_steps"])
    if lt.get("gradient_clip_val") is not None:
        defaults["gradient_clip_val"] = _as_float(lt["gradient_clip_val"])
    if lt.get("deterministic") is not None:
        defaults["deterministic"] = _as_bool(lt["deterministic"])
    if lt.get("num_workers") is not None:
        defaults["num_workers"] = _as_int(lt["num_workers"], 0)
    if lt.get("pin_memory") is not None:
        defaults["pin_memory"] = _as_bool(lt["pin_memory"], False)

    wb = cfg.get("wandb", {})
    if _as_bool(wb.get("enable")) is False:
        defaults["wandb_project"] = None
    else:
        if wb.get("project") is not None:
            defaults["wandb_project"] = str(wb["project"])
        if wb.get("entity") is not None:
            defaults["wandb_entity"] = str(wb["entity"])

    op = cfg.get("optuna", {})
    if op.get("trials") is not None:
        defaults["optuna_trials"] = _as_int(op["trials"])

    return defaults


def parse_tasks(task_args: List[str]) -> Dict[str, Dict[str, str]]:
    """
    CLI --task supports:
      --task name=path
      --task name=path,label_column
    Returns {name: {"path":..., "label":...}}
    """
    tasks: Dict[str, Dict[str, str]] = {}
    for item in task_args:
        if "=" not in item:
            raise ValueError(f"Invalid --task format: {item}. Use name=path or name=path,label_column.")
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()

        if "," in v:
            path, label = [x.strip() for x in v.split(",", 1)]
        else:
            path, label = v, k

        tasks[k] = {"path": path, "label": label}
    return tasks


def _parse_tasks_from_config(cfg: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """
    Supports config:
      data:
        tasks:
          mp: /path/to.csv
          bp:
            path: /path/to.csv
            label: boiling_point
    """
    tasks_cfg = cfg.get("data", {}).get("tasks", {})
    out: Dict[str, Dict[str, str]] = {}

    if isinstance(tasks_cfg, dict):
        for name, spec in tasks_cfg.items():
            name_s = str(name)
            if isinstance(spec, str):
                out[name_s] = {"path": spec, "label": name_s}
            elif isinstance(spec, dict):
                path = str(spec.get("path", ""))
                label = str(spec.get("label", name_s))
                out[name_s] = {"path": path, "label": label}
    return out


# ============================================================
# Main
# ============================================================
def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre_args, _ = pre.parse_known_args()

    cfg: Dict[str, Any] = {}
    if pre_args.config:
        cfg = _read_yaml_config(pre_args.config)

    defaults = _config_to_parser_defaults(cfg)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=pre_args.config)

    # Run args
    parser.add_argument("--out_dir", type=str, default=defaults.get("out_dir", "models"))
    parser.add_argument("--epochs", type=int, default=defaults.get("epochs", 100))
    parser.add_argument("--batch_size", type=int, default=defaults.get("batch_size", 64))
    parser.add_argument("--seed", type=int, default=defaults.get("seed", 42))
    parser.add_argument("--save_predictions_for", nargs="+", default=defaults.get("save_predictions_for", ["test"]))
    parser.add_argument("--no_save_splits", action="store_true", default=defaults.get("no_save_splits", False))

    # Featurizer
    parser.add_argument(
        "--features_generators",
        nargs="+",
        default=defaults.get("features_generators", ["rdkit_desc_normalized"]),
    )
    parser.add_argument("--feature_field", type=str, default=defaults.get("feature_field", "smiles"))
    parser.add_argument("--id_field", type=str, default=defaults.get("id_field", "smiles"))
    parser.add_argument("--shard_size", type=int, default=defaults.get("shard_size", 1024))

    # Optuna/Wandb
    parser.add_argument("--optuna_trials", type=int, default=defaults.get("optuna_trials", 0))
    parser.add_argument("--wandb_project", type=str, default=defaults.get("wandb_project", None))
    parser.add_argument("--wandb_entity", type=str, default=defaults.get("wandb_entity", None))

    # Lightning
    parser.add_argument("--accelerator", type=str, default=defaults.get("accelerator", "auto"))
    parser.add_argument("--devices", type=str, default=defaults.get("devices", "auto"))
    parser.add_argument("--strategy", type=str, default=defaults.get("strategy", "auto"))
    parser.add_argument("--precision", type=str, default=defaults.get("precision", "32-true"))
    parser.add_argument("--log_every_n_steps", type=int, default=defaults.get("log_every_n_steps", 10))
    parser.add_argument("--gradient_clip_val", type=float, default=defaults.get("gradient_clip_val", 1.0))
    parser.add_argument("--deterministic", type=_bool_arg, default=defaults.get("deterministic", True))
    parser.add_argument("--num_workers", type=int, default=defaults.get("num_workers", 0))
    parser.add_argument("--pin_memory", type=_bool_arg, default=defaults.get("pin_memory", False))

    # Callbacks / scheduler
    parser.add_argument("--early_stop_patience", type=int, default=defaults.get("early_stop_patience", 20))
    parser.add_argument("--early_stop_patience_optuna", type=int, default=defaults.get("early_stop_patience_optuna", 10))
    parser.add_argument("--lr_plateau_factor", type=float, default=defaults.get("lr_plateau_factor", 0.5))
    parser.add_argument("--lr_plateau_patience", type=int, default=defaults.get("lr_plateau_patience", 5))
    parser.add_argument("--lr_plateau_min_lr", type=float, default=defaults.get("lr_plateau_min_lr", 1e-6))

    # Model
    parser.add_argument("--learning_rate", type=float, default=defaults.get("learning_rate", 1e-4))
    parser.add_argument("--ffn_dropout_p", type=float, default=defaults.get("ffn_dropout_p", 0.1))
    parser.add_argument("--depth", type=int, default=defaults.get("depth", 3))
    parser.add_argument("--ffn_hidden", type=int, default=defaults.get("ffn_hidden", 512))
    parser.add_argument("--ffn_layers", type=int, default=defaults.get("ffn_layers", 6))

    # Tasks
    parser.add_argument("--task", action="append", default=None)

    args = parser.parse_args()

    # Ensure tmp root is set early (uses TMPDIR if provided by Slurm)
    _resolve_tmp_root(args.out_dir)

    # Load tasks
    if args.config:
        cfg = _read_yaml_config(args.config)
        tasks = _parse_tasks_from_config(cfg)
    else:
        tasks = {}

    if args.task:
        tasks.update(parse_tasks(args.task))

    if not tasks:
        parser.error("No tasks found. Use --task or config file (data.tasks).")

    _ensure_dir(args.out_dir)

    # features_generators cleanup
    fg = args.features_generators
    if fg is not None:
        fg = [g for g in fg if g and str(g).lower() not in {"none", "null"}]
        if not fg:
            fg = None

    optuna_search_space = cfg.get("optuna", {}).get("search_space", {})

    all_scores: Dict[str, Dict[str, float]] = {}
    all_pred_files: Dict[str, Dict[str, str]] = {}

    for task_name, spec in tasks.items():
        csv_path = spec["path"]
        label_column = spec.get("label", task_name)

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"[{task_name}] CSV not found: {csv_path}")

        dataset = load_and_featurize(
            csv_path,
            task_name=task_name,
            label_column=label_column,
            feature_field=args.feature_field,
            id_field=args.id_field,
            features_generators=fg,
            dataset_cache_dir=os.path.join(args.out_dir, "_dc_cache", task_name),
            shard_size=args.shard_size,
        )

        prepared = prepare_task_data(
            dataset=dataset,
            task_name=task_name,
            out_dir=args.out_dir,
            seed=args.seed,
        )

        final_params = dict(
            learning_rate=args.learning_rate,
            ffn_dropout_p=args.ffn_dropout_p,
            depth=args.depth,
            ffn_hidden=args.ffn_hidden,
            ffn_layers=args.ffn_layers,
        )

        if args.optuna_trials > 0:
            best = optimize_task(
                n_trials=args.optuna_trials,
                dataset=dataset,
                prepared=prepared,
                task_name=task_name,
                base_args=args,
                search_space=optuna_search_space,
            )
            final_params.update(best)
            logger.info(">>> Training Final Model for %s using Optuna best params <<<", task_name)

        logger.info("=== Training task: %s from %s (label column: %s) ===", task_name, csv_path, label_column)

        scores, pred_files, _ = train_single_task_lightning(
            prepared=prepared,
            full_dataset=dataset,
            task_name=task_name,
            out_dir=args.out_dir,
            seed=args.seed,
            nb_epoch=args.epochs,
            batch_size=args.batch_size,
            save_splits=(not args.no_save_splits),
            save_predictions_for=tuple(args.save_predictions_for),
            accelerator=args.accelerator,
            devices=args.devices,
            strategy=args.strategy,
            precision=args.precision,
            early_stop_patience=args.early_stop_patience,
            early_stop_patience_optuna=args.early_stop_patience_optuna,
            lr_plateau_factor=args.lr_plateau_factor,
            lr_plateau_patience=args.lr_plateau_patience,
            lr_plateau_min_lr=args.lr_plateau_min_lr,
            log_every_n_steps=args.log_every_n_steps,
            gradient_clip_val=args.gradient_clip_val,
            deterministic=args.deterministic,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_run_name=f"{task_name}_final",
            **final_params,
        )

        all_scores[task_name] = scores
        all_pred_files[task_name] = pred_files
        logger.info("[%s] scores: %s", task_name, scores)

    summary_rows = [{"task": t, **s} for t, s in all_scores.items()]
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(os.path.join(args.out_dir, "summary_scores.csv"), index=False)

    with open(os.path.join(args.out_dir, "prediction_files.json"), "w") as f:
        json.dump(all_pred_files, f, indent=2)


if __name__ == "__main__":
    main()
