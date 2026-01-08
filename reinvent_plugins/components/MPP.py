import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from rdkit.Chem import Draw
from rdkit import Chem
import time
import typing
from typing import Callable, Optional, Union, Tuple, Dict, Any, List

import torch, torch_geometric
from torch.utils.data import Dataset, DataLoader
from torch import Tensor, nn
import torch.nn.functional as F
from torch_geometric.typing import (
    Adj,
    NoneType,
    OptPairTensor,
    OptTensor,
    Size,
)
from torch_geometric.nn import MessagePassing, GraphNorm, global_add_pool, global_mean_pool
import pytorch_lightning as pl
import torchvision as tv
from torchvision.models.densenet import _densenet
from transformers import AutoModel, AutoTokenizer

from graph_featurizer import smiles_to_graph

def compute_task_stats(df, task_names):
    stats = {}
    for task in task_names:
        values = df[task].dropna()
        stats[task] = {
            'mean': values.mean(),
            'std': values.std()
        }
    return stats

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    
    if len(batch) == 0:
        return None
    
    keys = batch[0].keys()
    batch_out = {}
    
    if 'graph' in keys:
        batch_out['graph'] = torch_geometric.data.Batch.from_data_list([b['graph'] for b in batch if b['graph'] is not None])
    
    if 'input_ids' in keys:
        batch_out['input_ids'] = torch.stack([b['input_ids'] for b in batch if b['input_ids'] is not None])
        batch_out['attention_mask'] = torch.stack([b['attention_mask'] for b in batch if b['attention_mask'] is not None])
    
    if 'img' in keys:
        batch_out['img'] = torch.stack([b['img'] for b in batch if b['img'] is not None])
    
    if 'target' in keys:
        batch_out['target'] = torch.stack([b['target'] for b in batch if b['target'] is not None])

    return batch_out

class MultiModalDataset(Dataset):
    def __init__(self, 
                 dataframe: pd.DataFrame,
                 modals: List[str] = ['graph', 'text', 'image'],
                 task_names: List[str] = ['Esol'],
                 max_length: int = 200,
                 img_dim: int = 256):
        
        self.modals = modals
        self.task_names = task_names
        self.task_stats = compute_task_stats(dataframe, task_names)
        self.max_length = max_length
        self.img_dim = img_dim
        self.img_transform = tv.transforms.ToTensor()
        self.tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")

        self.samples = []
        self.multi_task = len(task_names) > 1

        for idx, row in dataframe.iterrows():
            smiles = row['smiles']
            targets = []
            for task in task_names:
                if pd.isna(row[task]):
                    targets.append(float("nan"))
                else:
                    if self.multi_task and self.task_stats:
                        mean = self.task_stats[task]['mean']
                        std = self.task_stats[task]['std']
                        targets.append((row[task] - mean) / std)
                    else:
                        targets.append(row[task])
            self.samples.append((idx, smiles, targets if self.multi_task else targets[0]))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        idx_val, smiles, targets = self.samples[idx]

        output = {}

        # 1. Graph
        if 'graph' in self.modals:
            graph = smiles_to_graph(smiles, targets)
            if graph is None or graph.x.size(0) == 0:
                raise IndexError("Invalid graph data")
            output['graph'] = graph

        # 2. Text (SMILES tokenization)
        if 'text' in self.modals:
            try:
                tokenized = self.tokenizer(
                    smiles,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors="pt"
                )
                output['input_ids'] = tokenized["input_ids"].squeeze(0)
                output['attention_mask'] = tokenized["attention_mask"].squeeze(0)
            except Exception:
                raise IndexError("Invalid SMILES tokenization")

        # 3. Image
        if 'image' in self.modals:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError("Invalid SMILES for image.")
            pil_img = Draw.MolToImage(mol, size=(self.img_dim, self.img_dim))
            img = self.img_transform(pil_img)
            output['img'] = img

        # 4. Target
        output['target'] = torch.tensor(targets, dtype=torch.float)

        return output

class AWSConv(MessagePassing):
    def __init__(self, 
                 M_nn: nn.Module, 
                 hidden_dim: int = 64,
                 eps: float = 0.,
                 train_eps: bool = False, 
                 edge_dim: Optional[int] = None,
                 **kwargs
                 ) -> None:
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        
        self.nn = M_nn
        self.initial_eps = eps
        
        if train_eps:
            self.eps = nn.Parameter(torch.empty(1)) # Make eps trainable
        else:
            self.register_buffer('eps', torch.empty(1))
        
        # Edge transform layer
        self.lin = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.SiLU()
        )

        # Custom intermediate MLPs
        self.nn1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            GraphNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=0.1)
        )
        self.nn2 = nn.Sequential(
            nn.Linear(hidden_dim*3, hidden_dim),
            GraphNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=0.1)
        )
        
        self.nn3 = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            GraphNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=0.1)
        )

        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        # Reset intermediate layers
        for layer in self.nn:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.nn1[0].weight)
        nn.init.zeros_(self.nn1[0].bias)
        nn.init.xavier_uniform_(self.nn2[0].weight)
        nn.init.zeros_(self.nn2[0].bias)
        nn.init.xavier_uniform_(self.nn3[0].weight)
        nn.init.zeros_(self.nn3[0].bias)
        nn.init.xavier_uniform_(self.lin[0].weight)
        nn.init.zeros_(self.lin[0].bias)

        self.eps.data.fill_(self.initial_eps)
  
    def forward(
            self,
            x: Union[Tensor, OptPairTensor],
            edge_index: Adj,
            edge_attr: OptTensor = None,
            size: Size = None,
        ) -> Tensor:
    
        if isinstance(x, Tensor):
            x = (x, x)

        # out: [num_atom, hidden_dim]
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
        
        # x_r: [num_atom, hidden_dim]
        x_r = self.nn1(x[1])
        
        # out: [num_atom, hidden_dim*2]
        out = torch.cat([out,x_r],dim=1)
        
        # out: [num_atom, hidden_dim]
        out = self.nn3(out)
        
            
        return self.nn(out)

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        if self.lin is None and x_j.size(-1) != edge_attr.size(-1):
            raise None

        if self.lin is not None:
            edge_attr = self.lin(edge_attr)

        return self.nn2(torch.cat([x_i,x_j,edge_attr],dim=1))

class Base_Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.start_time = None
        self.lr = None
        self.train_losses = []
        self.val_losses = []

    def forward(self, batch):
        pass

    def training_step(self, batch, batch_idx):
        preds = self(batch)
        loss = F.mse_loss(preds, batch['target'])
        self.log("train_loss", loss, on_step=False, on_epoch=True,
                 batch_size=batch["target"].size(0), prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        train_loss = self.trainer.callback_metrics.get("train_loss")
        if train_loss:
            self.train_losses.append(train_loss.item())

    def validation_step(self, batch, batch_idx):
        preds = self(batch)
        loss = F.mse_loss(preds, batch['target'])
        if batch_idx == 0:
            print("y_true(min,max):", batch['target'].min().item(), batch['target'].max().item())
            print("pred_raw(min,max):", preds.min().item(), preds.max().item())
        self.log("val_loss", loss, on_step=False, on_epoch=True,
                 batch_size=batch["target"].size(0), prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics.get("val_loss")
        if val_loss:
            self.val_losses.append(val_loss.item())

    def test_step(self, batch, batch_idx):
        preds = self(batch)
        loss = F.mse_loss(preds, batch['target'])
        return loss

    def on_train_start(self):
        self.start_time = time.time()
        print("Training started...")

    def on_train_end(self):
        total_time = time.time() - self.start_time
        print(f"Training completed in {total_time:.2f} seconds")
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler, "monitor": "val_loss"
        }

class MM_Mid_Model(Base_Model):
    def __init__(self,
                 use_modalities=('graph', 'text', 'image'), 
                 hidden_dim=256, 
                 lr = 5e-4, 
                 num_gnn_layer = 5,
                 conv_type='AWS',
                 Dr=0.1,
                 growth_rate=24,
                 block_config=(4, 6, 8, 6),
                 num_init_features=24,
                 fusion_type = 'weighted',
                 edge_dim=5,
                 train_eps=False,
                 aws_dropout=None,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.num_gnn_layer = num_gnn_layer
        self.edge_dim = edge_dim
        self.lr = lr
        self.block_config = block_config
        self.image_feature_dim = self.get_feature_dim(block_config=block_config)

        # modal flags
        self.use_graph = 'graph' in use_modalities
        self.use_text = 'text' in use_modalities
        self.use_image = 'image' in use_modalities
        num_modal = sum([self.use_graph, self.use_text, self.use_image])
        assert num_modal >= 1, "At least one modality must be enabled."

        # Graph encoder (AWSConv stack)
        self.gnn_pre = nn.Sequential(
            nn.Linear(13,hidden_dim),
            GraphNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, hidden_dim),
            GraphNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=0.1)
            )
        
        self.gnn = nn.ModuleList([
            AWSConv(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                GraphNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU()
            ), edge_dim=5, hidden_dim=hidden_dim) for _ in range(num_gnn_layer)
        ])
        self.gnn_proj = self.gnn_proj = nn.Linear(hidden_dim, hidden_dim) #nn.LayerNorm(hidden_dim)
        

        # Text encoder (ChemBERTa)
        self.text_encoder = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size * 2,
                           self.text_encoder.config.hidden_size * 2)    #nn.LayerNorm(self.text_encoder.config.hidden_size*2)
        self.text_out = nn.Linear(self.text_encoder.config.hidden_size * 2,
                          hidden_dim)
        
        # Image encoder 
        self.image_encoder = _densenet(24,(4,6,8,6),24,None, True)
        self.image_encoder.classifier = nn.Identity()
        self.image_proj = nn.Linear(self.image_feature_dim, self.image_feature_dim)     #nn.LayerNorm(self.image_feature_dim)
        self.image_out = nn.Linear(self.image_feature_dim, hidden_dim)
        
        #Fusion
        self.reg_type = fusion_type
                                               
        # Output MLP (shared for all types)
        if fusion_type == 'cat':
            mlp_input_dim = hidden_dim * 3
        elif fusion_type == 'weighted':
            self.wg = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.wt = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.wi = nn.Linear(hidden_dim, hidden_dim, bias=False)
            # self.bias = nn.Parameter(torch.zeros(hidden_dim))
            mlp_input_dim = hidden_dim * 3
        elif fusion_type == 'crossattention':
            self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, 
                                                    num_heads=4, 
                                                    batch_first=True)
            mlp_input_dim = hidden_dim * 6

        self.reg_mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, batch):
        device = next(self.parameters()).device
        batch['graph'] = batch['graph'].to(device)
        batch['input_ids'] = batch['input_ids'].to(device)
        batch['attention_mask'] = batch['attention_mask'].to(device)
        batch['img'] = batch['img'].to(device)
        
        z_g = self.encode_graph(batch['graph'])
        z_t = self.encode_text(batch['input_ids'], batch['attention_mask'])
        z_i = self.encode_image(batch['img'])

        if self.reg_type == 'cat':
            z = torch.cat([z_g, z_t, z_i], dim=1)

        elif self.reg_type == 'weighted':
            g_proj = self.wg(z_g)
            t_proj = self.wt(z_t)
            i_proj = self.wi(z_i)
            z = torch.cat([g_proj, t_proj, i_proj], dim=1)

        elif self.reg_type == 'crossattention':
            z_g = z_g.unsqueeze(1)
            z_t = z_t.unsqueeze(1)
            z_i = z_i.unsqueeze(1)
            CA_gt, _ = self.cross_attn(z_g, z_t, z_t)
            CA_tg, _ = self.cross_attn(z_t, z_g, z_g)
            CA_gi, _ = self.cross_attn(z_g, z_i, z_i)
            CA_ig, _ = self.cross_attn(z_i, z_g, z_g)
            CA_ti, _ = self.cross_attn(z_t, z_i, z_i)
            CA_it, _ = self.cross_attn(z_i, z_t, z_t)
            z = torch.cat([CA_gt, CA_tg, CA_gi, CA_ig, CA_ti, CA_it], dim=1)
            z = z.flatten(start_dim=1)

        return self.reg_mlp(z).squeeze(-1)

    def encode_graph(self, data):
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
        
        x = self.gnn_pre(x)
        residual = x
        
        for conv in self.gnn:
            x = conv(x, edge_index, edge_attr) + residual
            residual = x
            
        x = global_mean_pool(x, batch)
        
        return self.gnn_proj(x)

    def encode_text(self, input_ids, attention_mask):
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        hidden_states = outputs.last_hidden_state
        cls_token = hidden_states[:, 0, :]
        mean_pooling = torch.mean(hidden_states, dim=1)
        
        combined = torch.cat((cls_token, mean_pooling), dim=1) 
        
        normalized_output = self.text_proj(combined)
        
        return self.text_out(normalized_output)

    def encode_image(self, img):
        x = self.image_encoder(img)
        x = self.image_proj(x)
        
        return self.image_out(x)

    def get_feature_dim(self, 
                        block_config=(4, 6, 8, 6),
                        growth_rate=24,
                        init_features=24, 
                        compression=0.5):
        num_features = init_features
        for i, num_layers in enumerate(block_config):
            num_features += num_layers * growth_rate
            if i != len(block_config) - 1:
                num_features = int(num_features * compression)
        return num_features

def model_selection(config: Dict[str, Any]):
    # unpack with defaults
    task_names = config.get("task_names", [])
    use_modalities = config["use_modalities"]
    conv_type = config.get("conv_type", "AWS")
    num_gnn_layer = config.get("num_gnn_layer", 5)
    hidden_dim = config.get("hidden_dim", 256)
    lr = config.get("lr", 5e-4)
    Dr = config.get("Dr", 0.1)
    densenet_config = config.get("densenet_config", (24, (4, 6, 8, 6), 24))
    fusion_location = config.get("fusion_location", "Mid")
    fusion_type = config.get("fusion_type", "weighted")
    edge_dim = config.get("edge_dim", 5)
    train_eps = config.get("train_eps", True)
    aws_dropout = config.get("aws_dropout", None)

    num_task = len(task_names)
    num_modal = len(use_modalities)

    if num_task == 1:
        
            if fusion_location == 'Mid':
                return MM_Mid_Model(
                    use_modalities=tuple(use_modalities),
                    hidden_dim=hidden_dim, lr=lr,
                    num_gnn_layer=num_gnn_layer, conv_type=conv_type, Dr=Dr,
                    growth_rate=densenet_config[0],
                    block_config=densenet_config[1],
                    num_init_features=densenet_config[2],
                    fusion_type=fusion_type,
                    edge_dim=edge_dim, train_eps=train_eps, aws_dropout=aws_dropout,
                )
            else:
                raise ValueError("Not supported fusion location")