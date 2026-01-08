#!/usr/bin/env python3
"""
MPP scoring component + CLI inference.

Loads a pretrained Lightning checkpoint defined in MPP.py and runs multimodal
inference (graph + text + image) from SMILES.
"""

from __future__ import annotations

__all__ = ["MPP"]

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch_geometric
from torch.utils.data import DataLoader, Dataset
import torchvision as tv
from pydantic import Field
from pydantic.dataclasses import dataclass
from rdkit import Chem
from rdkit.Chem import Draw
from transformers import AutoTokenizer


def _add_project_root_to_syspath() -> None:
    """Ensure local imports work for both module import and script execution."""
    here = Path(__file__).resolve().parent
    repo_root = here.parent.parent
    for p in (here, repo_root):
        p_str = str(p)
        if p_str not in sys.path:
            sys.path.insert(0, p_str)


_add_project_root_to_syspath()

from reinvent_plugins.components.add_tag import add_tag  # noqa: E402
from reinvent_plugins.components.component_results import ComponentResults  # noqa: E402
from reinvent_plugins.normalize import normalize_smiles  # noqa: E402
from graph_featurizer import smiles_to_graph  # noqa: E402
import MPP as mpp  # noqa: E402


logger = logging.getLogger("reinvent")


def _device_from_string(device: str) -> torch.device:
    d = (device or "cpu").lower()
    if d.startswith("cuda") and torch.cuda.is_available():
        return torch.device(d)
    return torch.device("cpu")


def _get_model_cls():
    # MPP_util.py expects MM_Model, but MPP.py defines MM_Mid_Model.
    if hasattr(mpp, "MM_Model"):
        return getattr(mpp, "MM_Model")
    if hasattr(mpp, "MM_Mid_Model"):
        return getattr(mpp, "MM_Mid_Model")
    raise AttributeError("Could not find MM_Model or MM_Mid_Model in MPP module.")


def collate_infer(batch: List[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    batch = [b for b in batch if b is not None]
    if not batch:
        return {}

    out: Dict[str, Any] = {}

    if "graph" in batch[0]:
        out["graph"] = torch_geometric.data.Batch.from_data_list([b["graph"] for b in batch])
    if "input_ids" in batch[0]:
        out["input_ids"] = torch.stack([b["input_ids"] for b in batch])
        out["attention_mask"] = torch.stack([b["attention_mask"] for b in batch])
    if "img" in batch[0]:
        out["img"] = torch.stack([b["img"] for b in batch])
    if "target" in batch[0]:
        out["target"] = torch.stack([b["target"] for b in batch])

    out["row_id"] = torch.tensor([int(b["row_id"]) for b in batch], dtype=torch.long)
    out["smiles"] = [str(b["smiles"]) for b in batch]
    return out


class InferenceDataset(Dataset):
    """Build graph+text+image from SMILES and keep row_id for joining predictions."""

    def __init__(
        self,
        df: pd.DataFrame,
        smiles_col: str = "smiles",
        max_length: int = 200,
        img_dim: int = 256,
        tokenizer: Optional[Any] = None,
        img_transform: Optional[Any] = None,
        tokenizer_name: str = "DeepChem/ChemBERTa-77M-MLM",
    ):
        if smiles_col not in df.columns:
            raise KeyError(f"Missing SMILES column '{smiles_col}'. Columns: {list(df.columns)}")

        self.df = df.copy()
        if "_row_id" not in self.df.columns:
            self.df["_row_id"] = np.arange(len(self.df), dtype=np.int64)

        self.smiles_col = smiles_col
        self.max_length = max_length
        self.img_dim = img_dim

        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(tokenizer_name)
        self.img_transform = img_transform or tv.transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> Optional[Dict[str, Any]]:
        row = self.df.iloc[i]
        row_id = int(row["_row_id"])
        smi = str(row[self.smiles_col]).strip()

        if not smi or smi.lower() == "nan":
            return None

        dummy_target = torch.tensor(0.0, dtype=torch.float32)

        try:
            g = smiles_to_graph(smi, float(dummy_target.item()))
            if g is None or getattr(g, "x", None) is None or g.x.numel() == 0:
                return None

            tok = self.tokenizer(
                smi,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = tok["input_ids"].squeeze(0)
            attention_mask = tok["attention_mask"].squeeze(0)

            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return None
            pil_img = Draw.MolToImage(mol, size=(self.img_dim, self.img_dim))
            img = self.img_transform(pil_img)

            return {
                "row_id": row_id,
                "smiles": smi,
                "graph": g,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "img": img,
                "target": dummy_target,
            }
        except Exception:
            return None


def _move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    if "graph" in batch:
        batch["graph"] = batch["graph"].to(device)
    if "input_ids" in batch:
        batch["input_ids"] = batch["input_ids"].to(device)
    if "attention_mask" in batch:
        batch["attention_mask"] = batch["attention_mask"].to(device)
    if "img" in batch:
        batch["img"] = batch["img"].to(device)
    if "target" in batch:
        batch["target"] = batch["target"].to(device)
    return batch


@torch.no_grad()
def _predict_from_dataframe(
    df: pd.DataFrame,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    max_length: int,
    img_dim: int,
    tokenizer: Any,
    img_transform: Any,
    smiles_col: str = "smiles",
) -> Tuple[List[int], List[str], np.ndarray]:
    dataset = InferenceDataset(
        df=df,
        smiles_col=smiles_col,
        max_length=max_length,
        img_dim=img_dim,
        tokenizer=tokenizer,
        img_transform=img_transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_infer,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )

    all_row_ids: List[int] = []
    all_smiles: List[str] = []
    all_preds: List[np.ndarray] = []

    model.eval()

    for batch in loader:
        if not batch:
            continue
        batch = _move_batch_to_device(batch, device)

        out = model(batch)
        out_t = torch.as_tensor(out).detach().cpu()
        if out_t.ndim == 0:
            out_t = out_t.unsqueeze(0)
        if out_t.ndim == 1:
            out_t = out_t.unsqueeze(1)

        all_preds.append(out_t.numpy())
        all_row_ids.extend(batch["row_id"].cpu().tolist())
        all_smiles.extend(batch["smiles"])

    if not all_preds:
        return all_row_ids, all_smiles, np.zeros((0, 1), dtype=np.float32)

    pred_mat = np.concatenate(all_preds, axis=0)
    return all_row_ids, all_smiles, pred_mat


@add_tag("__parameters")
@dataclass
class Parameters:
    """Parameters for the MPP scoring component."""

    checkpoint_path: List[str]
    device: List[str] = Field(default_factory=lambda: ["cpu"])
    batch_size: List[int] = Field(default_factory=lambda: [64])
    num_workers: List[int] = Field(default_factory=lambda: [0])
    max_length: List[int] = Field(default_factory=lambda: [200])
    img_dim: List[int] = Field(default_factory=lambda: [256])
    tokenizer_name: List[str] = Field(default_factory=lambda: ["DeepChem/ChemBERTa-77M-MLM"])
    output_index: List[int] = Field(default_factory=lambda: [0])


@add_tag("__component")
class MPP:
    """MPP multimodal scoring component (inference only)."""

    def __init__(self, params: Parameters):
        self.smiles_type = "rdkit_smiles"

        self.checkpoint_path = params.checkpoint_path[0]
        device_str = params.device[0] if params.device and params.device[0] else "cpu"
        self.device = _device_from_string(device_str)

        self.batch_size = int(params.batch_size[0]) if params.batch_size else 64
        self.num_workers = int(params.num_workers[0]) if params.num_workers else 0
        self.max_length = int(params.max_length[0]) if params.max_length else 200
        self.img_dim = int(params.img_dim[0]) if params.img_dim else 256

        self.tokenizer_name = (
            params.tokenizer_name[0]
            if params.tokenizer_name and params.tokenizer_name[0]
            else "DeepChem/ChemBERTa-77M-MLM"
        )
        self.output_index = int(params.output_index[0]) if params.output_index else 0

        if not Path(self.checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        ModelCls = _get_model_cls()
        self.model = ModelCls.load_from_checkpoint(
            str(self.checkpoint_path), map_location=self.device
        )
        self.model.eval().to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.img_transform = tv.transforms.ToTensor()

        self.number_of_endpoints = 1
        logger.info(
            "Loaded MPP checkpoint=%s | device=%s | batch_size=%d",
            self.checkpoint_path,
            str(self.device),
            self.batch_size,
        )

    @normalize_smiles
    def __call__(self, smiles: List[str]) -> np.ndarray:
        n = len(smiles)
        out = np.full((n,), np.nan, dtype=np.float32)
        if n == 0:
            return ComponentResults([out.astype(float)])

        df = pd.DataFrame({"smiles": smiles, "_row_id": np.arange(n, dtype=np.int64)})
        row_ids, _, pred_mat = _predict_from_dataframe(
            df=df,
            model=self.model,
            device=self.device,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            max_length=self.max_length,
            img_dim=self.img_dim,
            tokenizer=self.tokenizer,
            img_transform=self.img_transform,
            smiles_col="smiles",
        )

        if pred_mat.size == 0:
            return ComponentResults([out.astype(float)])

        if pred_mat.ndim == 1:
            pred_mat = pred_mat.reshape(-1, 1)
        if self.output_index < 0 or self.output_index >= pred_mat.shape[1]:
            raise ValueError(
                f"output_index={self.output_index} out of range for predictions "
                f"with shape {pred_mat.shape}"
            )

        for j, i in enumerate(row_ids):
            out[i] = float(pred_mat[j, self.output_index])

        return ComponentResults([out.astype(float)])


@torch.no_grad()
def run_inference(
    ckpt_path: Path,
    input_csv: Path,
    output_csv: Path,
    smiles_col: str = "smiles",
    batch_size: int = 64,
    num_workers: int = 0,
    max_length: int = 200,
    img_dim: int = 256,
    tokenizer_name: str = "DeepChem/ChemBERTa-77M-MLM",
) -> None:
    df = pd.read_csv(input_csv)
    df = df.copy()
    df["_row_id"] = np.arange(len(df), dtype=np.int64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ModelCls = _get_model_cls()
    model = ModelCls.load_from_checkpoint(str(ckpt_path), map_location=device)
    model.eval().to(device)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    img_transform = tv.transforms.ToTensor()

    row_ids, smiles_out, pred_mat = _predict_from_dataframe(
        df=df,
        model=model,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        max_length=max_length,
        img_dim=img_dim,
        tokenizer=tokenizer,
        img_transform=img_transform,
        smiles_col=smiles_col,
    )

    if pred_mat.size == 0:
        raise RuntimeError("No valid samples were processed (all rows invalid or unreadable).")

    if pred_mat.ndim == 1:
        pred_mat = pred_mat.reshape(-1, 1)
    k = pred_mat.shape[1]
    pred_cols = ["pred"] if k == 1 else [f"pred_{j}" for j in range(k)]

    pred_df = pd.DataFrame(pred_mat, columns=pred_cols)
    pred_df.insert(0, "_row_id", row_ids)
    pred_df.insert(1, "smiles", smiles_out)

    out_df = df.merge(pred_df.drop(columns=["smiles"]), on="_row_id", how="left")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"[OK] wrote: {output_csv}  (rows={len(out_df)}, pred_cols={pred_cols})")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run inference with an MPP Lightning checkpoint on a SMILES CSV."
    )
    ap.add_argument("--ckpt", required=True, type=Path, help="Path to .ckpt checkpoint")
    ap.add_argument("--input", required=True, type=Path, help="Input CSV containing a SMILES column")
    ap.add_argument("--output", required=True, type=Path, help="Output CSV with prediction column(s)")
    ap.add_argument("--smiles-col", default="smiles", help="Name of the SMILES column")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=0, help="Use 0 if RDKit/Draw causes worker issues")
    ap.add_argument("--max-length", type=int, default=200, help="Tokenizer max_length")
    ap.add_argument("--img-dim", type=int, default=256, help="RDKit image size (square)")
    ap.add_argument(
        "--tokenizer-name",
        default="DeepChem/ChemBERTa-77M-MLM",
        help="Tokenizer name or local path",
    )
    args = ap.parse_args()

    run_inference(
        ckpt_path=args.ckpt,
        input_csv=args.input,
        output_csv=args.output,
        smiles_col=args.smiles_col,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_length=args.max_length,
        img_dim=args.img_dim,
        tokenizer_name=args.tokenizer_name,
    )


if __name__ == "__main__":
    main()
