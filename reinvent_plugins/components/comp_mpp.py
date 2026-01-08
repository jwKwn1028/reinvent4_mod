#!/usr/bin/env python3
"""
inference.py

Run inference for the multimodal MPP model on a CSV containing SMILES.

Example:
  python inference.py \
    --ckpt /path/to/last.ckpt \
    --input /path/to/input.csv \
    --output /path/to/preds.csv \
    --smiles-col smiles

Notes (based on your uploaded code):
- The model forward expects ALL of: graph, input_ids, attention_mask, img.
- We return None on bad samples so the collate function can drop them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch_geometric
from torch.utils.data import DataLoader, Dataset

import torchvision as tv
from rdkit import Chem
from rdkit.Chem import Draw
from transformers import AutoTokenizer

def _add_project_root_to_syspath() -> None:
    """Make imports work whether you run from repo root or from a scripts/ dir."""
    here = Path(__file__).resolve().parent
    candidates = [here, here.parent]
    for p in candidates:
        if (p / "MPP.py").exists() and (p / "graph_featurizer.py").exists():
            sys.path.insert(0, str(p))
            return
    # Fall back to current dir
    sys.path.insert(0, str(here))


_add_project_root_to_syspath()

# Local imports after path fix
from graph_featurizer import smiles_to_graph  # noqa: E402
import MPP as mpp  # noqa: E402


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
        tokenizer_name: str = "DeepChem/ChemBERTa-77M-MLM",
    ):
        if smiles_col not in df.columns:
            raise KeyError(f"Missing SMILES column '{smiles_col}'. Columns: {list(df.columns)}")

        # Stable ids for merge back to original rows
        self.df = df.copy()
        if "_row_id" not in self.df.columns:
            self.df["_row_id"] = np.arange(len(self.df), dtype=np.int64)

        self.smiles_col = smiles_col
        self.max_length = max_length
        self.img_dim = img_dim

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.img_transform = tv.transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> Optional[Dict[str, Any]]:
        row = self.df.iloc[i]
        row_id = int(row["_row_id"])
        smi = str(row[self.smiles_col]).strip()

        if not smi or smi.lower() == "nan":
            return None

        # Dummy target: model forward doesn't use it, but your training loop expects batch["target"].
        dummy_target = torch.tensor(0.0, dtype=torch.float32)

        try:
            # 1) Graph
            g = smiles_to_graph(smi, float(dummy_target.item()))
            if g is None or getattr(g, "x", None) is None or g.x.numel() == 0:
                return None

            # 2) Text
            tok = self.tokenizer(
                smi,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = tok["input_ids"].squeeze(0)
            attention_mask = tok["attention_mask"].squeeze(0)

            # 3) Image
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
            # Skip problematic molecules instead of crashing the whole dataloader.
            return None


def _get_model_cls():
    # MPP_util.py expects MM_Model, but your MPP.py clearly defines MM_Mid_Model.
    # Prefer MM_Model if it exists; otherwise fall back to MM_Mid_Model.
    if hasattr(mpp, "MM_Model"):
        return getattr(mpp, "MM_Model")
    if hasattr(mpp, "MM_Mid_Model"):
        return getattr(mpp, "MM_Mid_Model")
    raise AttributeError("Could not find MM_Model or MM_Mid_Model in MPP module.")


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
) -> None:
    df = pd.read_csv(input_csv)

    # create stable row ids so we can merge back even if some rows are skipped
    df = df.copy()
    df["_row_id"] = np.arange(len(df), dtype=np.int64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ModelCls = _get_model_cls()
    model = ModelCls.load_from_checkpoint(str(ckpt_path), map_location=device)
    model.eval().to(device)

    dataset = InferenceDataset(
        df=df,
        smiles_col=smiles_col,
        max_length=max_length,
        img_dim=img_dim,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_infer,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0 and torch.cuda.is_available()),
    )

    all_row_ids: List[int] = []
    all_smiles: List[str] = []
    all_preds: List[np.ndarray] = []

    for batch in loader:
        if not batch:
            continue

        # Move only the tensors/Batch the model uses
        batch["graph"] = batch["graph"].to(device)
        batch["input_ids"] = batch["input_ids"].to(device)
        batch["attention_mask"] = batch["attention_mask"].to(device)
        batch["img"] = batch["img"].to(device)
        batch["target"] = batch["target"].to(device)

        out = model(batch)
        out_t = torch.as_tensor(out).detach().cpu()
        if out_t.ndim == 1:
            out_t = out_t.unsqueeze(1)  # [B] -> [B,1]

        all_preds.append(out_t.numpy())
        all_row_ids.extend(batch["row_id"].cpu().tolist())
        all_smiles.extend(batch["smiles"])

    if not all_preds:
        raise RuntimeError("No valid samples were processed (all rows invalid or unreadable).")

    pred_mat = np.concatenate(all_preds, axis=0)  # [N_valid, K]
    k = pred_mat.shape[1]

    pred_cols = ["pred"] if k == 1 else [f"pred_{j}" for j in range(k)]
    pred_df = pd.DataFrame(pred_mat, columns=pred_cols)
    pred_df.insert(0, "_row_id", all_row_ids)
    pred_df.insert(1, "smiles", all_smiles)

    # Left-join onto original rows; invalid rows keep NaN predictions
    out_df = df.merge(pred_df.drop(columns=["smiles"]), on="_row_id", how="left")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"[OK] wrote: {output_csv}  (rows={len(out_df)}, pred_cols={pred_cols})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run inference with an MPP Lightning checkpoint on a SMILES CSV.")
    ap.add_argument("--ckpt", required=True, type=Path, help="Path to .ckpt (Lightning checkpoint)")
    ap.add_argument("--input", required=True, type=Path, help="Input CSV containing a SMILES column")
    ap.add_argument("--output", required=True, type=Path, help="Output CSV with prediction column(s)")
    ap.add_argument("--smiles-col", default="smiles", help="Name of the SMILES column (default: smiles)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=0, help="Use 0 if RDKit/Draw causes worker issues")
    ap.add_argument("--max-length", type=int, default=200, help="Tokenizer max_length")
    ap.add_argument("--img-dim", type=int, default=256, help="RDKit image size (square)")
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
    )


if __name__ == "__main__":
    main()
