"""
Uni-Mol scoring component.

Uses a pretrained UniMol model (via unimol_tools) to score molecules.
"""

import os
import sys
import shutil
import tempfile
import logging
import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass as std_dataclass
from pydantic.dataclasses import dataclass as pydantic_dataclass
from typing import List, Optional
from pydantic import Field

from reinvent_plugins.components.add_tag import add_tag
from reinvent_plugins.components.component_results import ComponentResults
from reinvent_plugins.normalize import normalize_smiles

logger = logging.getLogger("reinvent")

def ensure_2d_float(y) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    return y

@std_dataclass
class StdScaler:
    mean_: np.ndarray
    std_: np.ndarray

    def transform(self, y_raw: np.ndarray) -> np.ndarray:
        y = ensure_2d_float(y_raw)
        return (y - self.mean_) / self.std_

    def inverse_transform(self, y_scaled: np.ndarray) -> np.ndarray:
        y = ensure_2d_float(y_scaled)
        return y * self.std_ + self.mean_

@add_tag("__parameters")
@pydantic_dataclass
class Parameters:
    """Parameters for the UniMol scoring component."""
    
    model_dir: List[str]
    model_name: List[str] = Field(default_factory=lambda: ["unimolv2"])
    model_size: List[str] = Field(default_factory=lambda: ["84m"])
    batch_size: List[int] = Field(default_factory=lambda: [32])
    use_cuda: List[bool] = Field(default_factory=lambda: [True])
    scaler_mean: List[float] = Field(default_factory=lambda: [0.0])
    scaler_std: List[float] = Field(default_factory=lambda: [1.0])
    unimol_tools_path: List[str] = Field(default_factory=lambda: ["unimol_tools"])

@add_tag("__component")
class CompUnimol:
    """UniMol scoring component."""

    def __init__(self, params: Parameters):
        self.smiles_type = "rdkit_smiles"
        self.model_dir = params.model_dir[0]
        self.model_name = params.model_name[0]
        self.model_size = params.model_size[0]
        self.batch_size = params.batch_size[0]
        self.use_cuda = params.use_cuda[0]
        
        # Scaler parameters
        mean_val = params.scaler_mean[0] if params.scaler_mean else 0.0
        std_val = params.scaler_std[0] if params.scaler_std else 1.0
        self.scaler = StdScaler(
            mean_=np.array([[mean_val]], dtype=np.float32),
            std_=np.array([[std_val]], dtype=np.float32)
        )

        # Handle unimol_tools import
        unimol_path = params.unimol_tools_path[0]
        if not os.path.isabs(unimol_path):
             unimol_path = os.path.abspath(unimol_path)
        
        if os.path.exists(unimol_path):
            if unimol_path not in sys.path:
                sys.path.append(unimol_path)
        elif os.path.exists(os.path.join(os.getcwd(), "unimol_tools")):
             sys.path.append(os.path.join(os.getcwd(), "unimol_tools"))
        
        try:
            from unimol_tools.data import DataHub
            from unimol_tools.models import NNModel
            from unimol_tools.tasks import Trainer
            from unimol_tools.models.nnmodel import NNDataset
            self.DataHub = DataHub
            self.NNModel = NNModel
            self.Trainer = Trainer
            self.NNDataset = NNDataset
        except ImportError as e:
            raise ImportError(f"Could not import unimol_tools. Ensure it is in the path or specified in params. Error: {e}")

        # Load Model
        self.device = torch.device("cuda" if self.use_cuda and torch.cuda.is_available() else "cpu")
        
        if os.path.isfile(self.model_dir):
            ckpt_path = self.model_dir
            # If model_dir is a file, we might need a directory for the trainer to write logs.
            # We can use the parent dir or a temp dir. Let's use parent dir of the model file.
            self.model_base_dir = os.path.dirname(self.model_dir)
        else:
            ckpt_path = os.path.join(self.model_dir, "model_0.pth")
            self.model_base_dir = self.model_dir

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        # Initialize model with a dummy template
        # We need to construct a minimal DataHub to initialize the model structure
        self.tmp_dir = tempfile.mkdtemp(prefix="unimol_init_")
        try:
            dummy_data = {"SMILES": ["C"], "mp": [0.0]}
            hub = self.DataHub(
                data=dummy_data,
                is_train=False,
                task="regression",
                target_cols=["mp"],
                smiles_col="SMILES",
                model_name=self.model_name,
                model_size=self.model_size,
                batch_size=self.batch_size,
                epochs=1,
                learning_rate=1e-4,
                metrics="mse",
                save_path=self.tmp_dir,
                multi_process=False,
                target_normalize="none",
                conf_cache_level=0,
                kfold=1
            )
            template_data = hub.data
            
            self.trainer = self.Trainer(
                task="regression",
                metrics="mse",
                save_path=self.model_base_dir, # Should ideally be read-only or temp, but Trainer might write logs
                batch_size=self.batch_size,
                use_cuda=(self.device.type == "cuda"),
                use_amp=True,
                use_wandb=False,
            )
            
            self.model = self.NNModel(
                template_data,
                self.trainer,
                model_name=self.model_name,
                model_size=self.model_size,
                task="regression",
                dropout=0.0,
            )
            
            # Load weights
            # Using flexible loading from inference.py logic if needed, 
            # but simpler here:
            state_dict = torch.load(ckpt_path, map_location="cpu")
            # Handle nested dicts if any
            if isinstance(state_dict, dict):
                 if "model_state_dict" in state_dict:
                     state_dict = state_dict["model_state_dict"]
                 elif "state_dict" in state_dict:
                     state_dict = state_dict["state_dict"]
                 elif "model" in state_dict:
                     state_dict = state_dict["model"]
            
            self.model.model.load_state_dict(state_dict, strict=False)
            self.model.model.eval()
            self.model.model.to(self.device)
            
        finally:
            shutil.rmtree(self.tmp_dir, ignore_errors=True)

        logger.info(f"Loaded UniMol model from {self.model_dir}")

    @normalize_smiles
    def __call__(self, smiles: List[str]) -> ComponentResults:
        if not smiles:
             return ComponentResults([np.array([], dtype=float)])

        # DataHub writes to disk, so we use a temp dir for each call or reuse one
        tmp_run_dir = tempfile.mkdtemp(prefix="unimol_run_")
        try:
            # Prepare data
            # UniMol expects SMILES and targets. We provide dummy targets.
            data = {"SMILES": list(smiles), "mp": [0.0] * len(smiles)}
            
            hub = self.DataHub(
                data=data,
                is_train=False,
                task="regression",
                target_cols=["mp"],
                smiles_col="SMILES",
                model_name=self.model_name,
                model_size=self.model_size,
                batch_size=self.batch_size,
                epochs=1, # Irrelevant for inference
                learning_rate=1e-4,
                metrics="mse",
                save_path=tmp_run_dir,
                multi_process=False,
                target_normalize="none",
                conf_cache_level=0,
                kfold=1
            )
            
            X = np.asarray(hub.data["unimol_input"], dtype=object)
            y_dummy = np.zeros((len(smiles), 1), dtype=np.float32)
            
            ds = self.NNDataset(X, y_dummy)
            
            # Predict
            # trainer.predict returns (preds, targets, metrics)
            # Preds are scaled
            y_pred_scaled, _, _ = self.trainer.predict(
                self.model.model,
                ds,
                self.model.loss_func,
                self.model.activation_fn,
                dump_dir=tmp_run_dir,
                fold=0,
                target_scaler=None,
                load_model=False, 
            )
            
            # Inverse transform
            y_pred_raw = self.scaler.inverse_transform(ensure_2d_float(y_pred_scaled)).flatten()
            
            return ComponentResults([y_pred_raw])

        except Exception as e:
            logger.error(f"UniMol inference failed: {e}")
            # Return NaNs for all smiles on failure
            return ComponentResults([np.full(len(smiles), np.nan)])
        finally:
             shutil.rmtree(tmp_run_dir, ignore_errors=True)
