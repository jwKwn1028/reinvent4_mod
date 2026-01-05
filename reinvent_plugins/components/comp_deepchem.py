"""
DeepChem DMPNN scoring component (REINVENT4-style).

This component loads a pretrained DeepChem DMPNN regressor checkpoint (Lightning .ckpt)
and an accompanying DeepChem NormalizationTransformer (joblib), then exposes:

    SMILES[]  ->  predicted property (original y-units)  ->  ComponentResults

Example TOML (single endpoint):

# deepchem_dmpnn_scoring.toml
# Example REINVENT4 scoring config for the DeepChemDMPNN component in comp_chemprop.py

# -------------------------------------------------------------------
# 1) Component endpoint: DeepChem DMPNN single-task regressor
# -------------------------------------------------------------------
[[component.DeepChemDMPNN.endpoint]]
name = "DeepChem DMPNN (de) regressor"
weight = 0.7 (requires adjustment for different molecular properties)

# ---- Required paths (relative to REINVENT run directory) ----
param.checkpoint_path  = "models/de/checkpoints/last-v2.ckpt"
param.transformer_path = "models/de/y_transformer.joblib"

# ---- Runtime / batching ----
param.device     = "cuda"      # "cpu" or "cuda" (or "cuda:0")
param.batch_size = 256         # inference batch size used by DeepChem generator

# ---- Featurization (MUST match training) ----
# Training code uses:
#   dc.feat.DMPNNFeaturizer(features_generators=["rdkit_desc_normalized", "morgan"])
param.features_generators = "rdkit_desc_normalized,morgan"

# ---- Model architecture (MUST match training) ----
# These map to the DMPNNModel constructor used in training script:
#   n_steps         -> depth
#   ffn_hidden_dim  -> ffn_hidden
#   ffn_num_layers  -> ffn_layers
#   dropout_p       -> dropout_p
#   bias            -> bias
#   global_features_size -> global_features_size
param.depth = 3
param.ffn_hidden = 300
param.ffn_layers = 3
param.dropout_p = 0.10
param.bias = true

# leave as -1 for auto-detection
# If auto-detect mismatches training, set the correct integer explicitly.
param.global_features_size = -1

# -------------------------------------------------------------------
# 2) Transform: convert raw predicted property into a 0..1 score
# -------------------------------------------------------------------
# You trained a regressor returning the original y-units after untransform.
# Tune these to your property range / goal direction.
transform.type = "reverse_sigmoid"
transform.high = -5.0
transform.low  = -35.0
transform.k    = 0.4


NOTE
- This assumes single-task regression (n_tasks=1), matching your training script.
TODO implement multitask support
- If you trained with different hyperparameters, provide them via params or a meta file.
"""

from __future__ import annotations

__all__ = ["DeepChemDMPNN"]

import logging
import os
from typing import Any, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import torch

from pydantic.dataclasses import dataclass
import deepchem as dc
from deepchem import feat
from deepchem import trans
from .add_tag import add_tag
from .component_results import ComponentResults
from reinvent.scoring.utils import suppress_output
from ..normalize import normalize_smiles

logger = logging.getLogger("reinvent")


def _parse_csv_list(s: str) -> List[str]:
    """Parse 'a,b,c' into ['a','b','c'] with whitespace stripped."""
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _untransform_y_array(
    y: np.ndarray, transformer: Optional[trans.Transformer]
) -> np.ndarray:
    """
    Untransform y back to original units. DeepChem transformers typically expect 2D.
    Returns shape (N,).
    """
    y = np.asarray(y)
    if transformer is None:
        return y.reshape(-1)
    y2d = y.reshape(-1, 1) if y.ndim == 1 else y
    try:
        y_un = transformer.untransform(y2d)
    except Exception:
        y_un = y2d
    return np.asarray(y_un).reshape(-1)


def _device_from_string(device: str) -> torch.device:
    d = (device or "cpu").lower()
    if d.startswith("cuda") and torch.cuda.is_available():
        return torch.device(d)
    return torch.device("cpu")


def _detect_global_features_size_from_smiles(
    featurizer: feat.DMPNNFeaturizer,
    smiles_try: Sequence[str] = ("CC", "CCC", "c1ccccc1"),
) -> int:
    """
    Try to featurize a simple molecule and infer global_features length.
    Returns 0 if no global_features exist.
    """
    for smi in smiles_try:
        try:
            feat = featurizer.featurize([smi])
            if len(feat) == 0:
                continue
            g = feat[0]
            gf = getattr(g, "global_features", None)
            if gf is None:
                return 0
            return int(len(gf))
        except Exception:
            continue
    return 0


def _extract_net_state_dict_from_lightning_ckpt(ckpt_path: str) -> dict:
    """
    Extract the underlying torch model state_dict from a Lightning checkpoint.
    Handles common key-prefix conventions:
      - 'net.*' (as in your LightningModule wrapper)
      - 'dc_model.model.*' (sometimes)
      - already-unprefixed keys
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        # sometimes sd is the dict itself
        sd = ckpt
    else:
        raise ValueError(f"Unexpected checkpoint format at: {ckpt_path}")


    if any(k.startswith("net.") for k in sd.keys()):
        return {k[len("net.") :]: v for k, v in sd.items() if k.startswith("net.")}
    # Alternate prefix
    if any(k.startswith("dc_model.model.") for k in sd.keys()):
        return {
            k[len("dc_model.model.") :]: v
            for k, v in sd.items()
            if k.startswith("dc_model.model.")
        }
    return sd


@add_tag("__parameters")
@dataclass
class Parameters:
    """
    Component parameters.

    All fields are lists because REINVENT endpoints are collected into lists
    even when there is only one endpoint.
    """

    checkpoint_path: List[str]
    transformer_path: List[str]

    device: List[str]
    batch_size: List[int]

    # Optional: must match training if different from defaults, choose best one from Optuna
    features_generators: List[str]  # e.g. "rdkit_desc_normalized,morgan"
    depth: List[int]  
    ffn_hidden: List[int]
    ffn_layers: List[int]
    dropout_p: List[float]
    bias: List[bool]
    global_features_size: List[int]  # -1 means auto-detect, or read from config


@add_tag("__component")
class DeepChemDMPNN:
    """
    DeepChem DMPNN regressor scoring component.
    """

    def __init__(self, params: Parameters):
        self.smiles_type = "rdkit_smiles"  # @normalize_smiles

        # ---- Resolve params (use endpoint 0) ----
        self.checkpoint_path = params.checkpoint_path[0]
        self.transformer_path = params.transformer_path[0]

        device_str = params.device[0] if params.device and params.device[0] else "cpu"
        self.device = _device_from_string(device_str)

        self.batch_size = (
            int(params.batch_size[0])
            if params.batch_size and params.batch_size[0]
            else 256
        )

        feats_str = (
            params.features_generators[0]
            if params.features_generators and params.features_generators[0]
            else "rdkit_desc_normalized,morgan"
        )
        self.features_generators = _parse_csv_list(feats_str)


        self.depth = (
            int(params.depth[0]) if params.depth and params.depth[0] is not None else 3
        )
        self.ffn_hidden = (
            int(params.ffn_hidden[0])
            if params.ffn_hidden and params.ffn_hidden[0] is not None
            else 300
        )
        self.ffn_layers = (
            int(params.ffn_layers[0])
            if params.ffn_layers and params.ffn_layers[0] is not None
            else 3
        )
        self.dropout_p = (
            float(params.dropout_p[0])
            if params.dropout_p and params.dropout_p[0] is not None
            else 0.1
        )
        self.bias = (
            bool(params.bias[0])
            if params.bias and (params.bias[0] is not None)
            else True
        )

        gfs = (
            int(params.global_features_size[0])
            if params.global_features_size
            and params.global_features_size[0] is not None
            else -1
        )

        # ---- Load transformer ----
        if not os.path.exists(self.transformer_path):
            raise FileNotFoundError(f"y_transformer not found: {self.transformer_path}")

        with suppress_output():
            self.y_transformer: Optional[trans.Transformer] = joblib.load(
                self.transformer_path
            )

        # ---- Build featurizer ----
        self.featurizer = feat.DMPNNFeaturizer(
            features_generators=self.features_generators
        )

        # ---- Determine global_features_size ----
        if gfs < 0:
            gfs = _detect_global_features_size_from_smiles(self.featurizer)
            logger.info("Auto-detected global_features_size=%d", gfs)

        self.global_features_size = int(gfs)

        # ---- Instantiate DeepChem model (must match training arch) ----
        # model_dir is required; keep it near the checkpoint to avoid clutter
        model_dir = os.path.abspath(
            os.path.join(os.path.dirname(self.checkpoint_path), "..", "dc_model_cache")
        )
        os.makedirs(model_dir, exist_ok=True)

        self.dc_model = dc.DMPNNModel(
            mode="regression",
            n_tasks=1,
            batch_size=self.batch_size,  # used by generator
            model_dir=model_dir,
            learning_rate=1e-4,  # not used for inference but required by constructor
            n_steps=self.depth,
            ffn_hidden_dim=self.ffn_hidden,
            ffn_num_layers=self.ffn_layers,
            dropout_p=self.dropout_p,
            bias=self.bias,
            global_features_size=self.global_features_size,
        )

        # ---- Load weights from Lightning ckpt ----
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        with suppress_output():
            net_sd = _extract_net_state_dict_from_lightning_ckpt(self.checkpoint_path)
            missing, unexpected = self.dc_model.model.load_state_dict(
                net_sd, strict=False
            )

        if missing:
            logger.warning(
                "Missing keys while loading DMPNN weights (showing up to 10): %s",
                missing[:10],
            )
        if unexpected:
            logger.warning(
                "Unexpected keys while loading DMPNN weights (showing up to 10): %s",
                unexpected[:10],
            )

        self.dc_model.model.to(self.device)
        self.dc_model.model.eval()

        # REINVENT bookkeeping
        self.number_of_endpoints = 1
        logger.info(
            "Loaded DeepChem DMPNN checkpoint=%s | transformer=%s | device=%s",
            self.checkpoint_path,
            self.transformer_path,
            str(self.device),
        )

    def _featurize(self, smiles: List[str]) -> Tuple[List[int], List[Any]]:
        """
        Returns:
          - valid_indices: indices in the original list that featurized successfully
          - feats: list of graph objects aligned with valid_indices
        """
        valid_indices: List[int] = []
        feats: List[Any] = []

        # Featurize one-by-one for robust invalid handling (cheap relative to model)
        for i, smi in enumerate(smiles):
            try:
                arr = self.featurizer.featurize([smi])
                if len(arr) != 1:
                    continue
                g = arr[0]
                # Some failures may return None-like placeholders
                if g is None:
                    continue
                valid_indices.append(i)
                feats.append(g)
            except Exception:
                continue

        return valid_indices, feats

    @torch.no_grad()
    def _predict_norm(self, feats: List[Any], ids: List[str]) -> np.ndarray:
        """
        Predict in normalized y-space using DeepChem's generator + underlying torch module.
        Returns shape (N_valid, 1).
        """
        if not feats:
            return np.zeros((0, 1), dtype=np.float32)

        # dummy y/w required by NumpyDataset
        y = np.zeros((len(feats), 1), dtype=np.float32)
        w = np.ones((len(feats), 1), dtype=np.float32)

        dataset = dc.data.NumpyDataset(
            X=np.asarray(feats, dtype=object), y=y, w=w, ids=np.asarray(ids)
        )

        gen = self.dc_model.default_generator(
            dataset, batch_size=self.batch_size, pad_batches=False
        )

        preds: List[np.ndarray] = []
        self.dc_model.model.eval()

        for batch in gen:
            inputs, labels, weights = self.dc_model._prepare_batch(batch)
            if hasattr(inputs, "to") and callable(getattr(inputs, "to")):
                try:
                    inputs = inputs.to(self.device)
                except Exception:
                    pass
            elif isinstance(inputs, (list, tuple)):
                moved = []
                for x in inputs:
                    if isinstance(x, torch.Tensor):
                        moved.append(x.to(self.device))
                    else:
                        moved.append(x)
                inputs = type(inputs)(moved)

            out = self.dc_model.model(inputs)
            if isinstance(out, (list, tuple)) and len(out) == 1:
                out = out[0]
            if not isinstance(out, torch.Tensor):
                out = torch.as_tensor(out, device=self.device)
            out = out.float().detach().cpu().numpy()

            if out.ndim == 1:
                out = out.reshape(-1, 1)
            elif out.ndim == 2 and out.shape[1] != 1:
                out = out[:, :1]
            preds.append(out.astype(np.float32, copy=False))

        if not preds:
            return np.zeros((0, 1), dtype=np.float32)

        return np.concatenate(preds, axis=0)

    @normalize_smiles
    def __call__(self, smiles: List[str]) -> np.ndarray:
        """
        Args:
          smiles: list of SMILES strings (normalized by decorator)

        Returns:
          ComponentResults with one endpoint: list([scores_per_molecule])
        """
        n = len(smiles)
        out = np.full((n,), np.nan, dtype=np.float32)

        valid_idx, feats = self._featurize(smiles)
        if not valid_idx:
            return ComponentResults([out.astype(float)])

        valid_smiles = [smiles[i] for i in valid_idx]
        y_pred_norm_2d = self._predict_norm(feats, valid_smiles) 
        y_pred = _untransform_y_array(y_pred_norm_2d, self.y_transformer) 

        # Fill in outputs in original order
        for j, i in enumerate(valid_idx):
            try:
                out[i] = float(y_pred[j])
            except Exception:
                out[i] = np.nan

        return ComponentResults([out.astype(float)])


# TODO If Optuna produced non-default depth/ffn_hidden/ffn_layers/dropout_p/global_features_size, set via TOML param (or add model_meta.json and load in __init__).

# If global_features_size auto-detection doesn’t match training, explicitly set param.global_features_size.
