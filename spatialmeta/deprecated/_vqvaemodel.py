# Pytorch
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.distributions import kl_divergence as kld
from torch.optim.lr_scheduler import ReduceLROnPlateau
import einops

# Hugginface Transformers
from transformers import (
    BertConfig,
    PreTrainedModel,
    BertForMaskedLM
)

# Third Party
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump, load
import datasets
import numpy as np
from pathlib import Path
import umap

# Built-in
import time
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.sparse import issparse
from sklearn.utils import class_weight
from collections import Counter
from itertools import chain
from copy import deepcopy
import json
from typing import Callable, Mapping, Union, Iterable, Tuple, Optional, Mapping
import os
import warnings
from sklearn.preprocessing import MinMaxScaler

# Package
from ..model._primitives import *
from ..util.loss import LossFunction
from ..util.logger import get_tqdm
from ..util._trvae_mmd_loss import mmd_loss_calc
from ..external.taming.modules.vqvae.quantize import VectorQuantizer


'''
How to use
adata = sc.read_h5ad("/Volumes/bkup/Ruonan/01_scMeAtlas/00_NBT_data/data/20240227_joint_spatialhvg2000_hvm400.h5ad")
X = adata.X.toarray()
_anndata = sc.AnnData(X=X[:,adata.var.type == 'SM'])
sc.pp.normalize_total(_anndata, target_sum=1e3)
X = np.hstack([_anndata.X, adata.X[:,adata.var.type == 'ST'].toarray()])
adata_normalize_sm = adata.copy()
adata_normalize_sm.X = X
adata.layers['normalized'] = adata_normalize_sm.X
model = model = VQVAE(adata=adata_raw_normalize_sm,vq_k=20)
loss_dict = model.fit(max_epoch=128, lr=1e-4)
cluster_st, cluster_sm = get_vq_cluster(model)
Z = model.get_latent_embedding()
X = model.get_normalized_expression()
adata.layers['reconstruction'] = X
adata.obsm['X_emb']=Z
'''

class BaseVAE(nn.Module):
    def __init__(
        self,
        adata: AnnData,
        hidden_stacks: List[int] = [128], 
        n_latent: int = 64,
        bias: bool = True,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        dropout_rate: float = 0.1,
        activation_fn: Callable = nn.ReLU,
        device: Union[str, torch.device] = "cpu",
        batch_embedding: Literal["embedding", "onehot"] = "onehot",
        encode_libsize: bool = False,
        batch_hidden_dim: int = 8,
        reconstruction_method_st: Literal['mse', 'zg', 'zinb'] = 'zinb',
        reconstruction_method_sm: Literal['mse', 'zg','g'] = 'g'
    ):
        super(BaseVAE, self).__init__()

        self.adata = adata 
    
        adata.hidden_stacks = hidden_stacks
        self.n_hidden = hidden_stacks[-1]
        self.n_latent = n_latent
        self.device = device
        self.reconstruction_method_st = reconstruction_method_st
        self.reconstruction_method_sm = reconstruction_method_sm
        self.encode_libsize = encode_libsize

        self.initialize_dataset()

        self.fcargs = dict(
            bias           = bias, 
            dropout_rate   = dropout_rate, 
            use_batch_norm = use_batch_norm, 
            use_layer_norm = use_layer_norm,
            activation_fn  = activation_fn,
            device         = device
        )
        
        self.encoder_ST = SAE(
            self.in_dim_ST if not self.encode_libsize else self.in_dim_ST + 1,
            stacks = hidden_stacks,
            # n_cat_list = [self.n_batch] if self.n_batch > 0 else None,
            cat_dim = batch_hidden_dim,
            cat_embedding = batch_embedding,
            encode_only = True,
            **self.fcargs
        )  
        
        self.encoder_SM = SAE(
            self.in_dim_SM,
            stacks = hidden_stacks,
            cat_dim = batch_hidden_dim,
            cat_embedding = batch_embedding,
            encode_only = True,
            **self.fcargs
        )
            
         #self.decoder_n_cat_list = decoder_n_cat_list
        self.decoder = FCLayer(
            in_dim = self.n_latent, 
            out_dim = self.n_hidden,
            #n_cat_list = decoder_n_cat_list,
            cat_dim = batch_hidden_dim,
            cat_embedding = batch_embedding,
            use_layer_norm=False,
            use_batch_norm=True,
            dropout_rate=0,
            device=device
        )     
                
        self.encode_libsize = encode_libsize
        
        # The latent cell representation z ~ Logisticnormal(0, I)
        self.z_mean_fc = nn.Linear(self.n_hidden*2, self.n_latent)
        self.z_var_fc = nn.Linear(self.n_hidden*2, self.n_latent)

        self.px_rna_rate_decoder = nn.Linear(
            self.n_hidden, 
            self.in_dim_ST
        )
        
        self.px_rna_scale_decoder = nn.Sequential(
            nn.Linear(self.n_hidden, self.in_dim_ST),
            nn.Softmax(dim=-1)
        )
        
        self.px_rna_dropout_decoder = nn.Linear(
            self.n_hidden, 
            self.in_dim_ST
        )
        
        self.px_sm_rate_decoder = nn.Linear(
            self.n_hidden, 
            self.in_dim_SM
        )
        
        self.px_sm_scale_decoder = nn.Linear(self.n_hidden, self.in_dim_SM)
        
        self.px_sm_dropout_decoder = nn.Linear(
            self.n_hidden,
            self.in_dim_SM
        )
        
        self.to(self.device)

    
    def initialize_dataset(self):
        X = self.adata.X
        self._type=np.array(list(self.adata.var.type.values))
        self.in_dim_SM = X[:,self._type=="SM"].shape[1]
        self.in_dim_ST = X[:,self._type=="ST"].shape[1]
        self._n_record = X.shape[0]
        self._indices = np.array(list(range(self._n_record)))
        _dataset = list(X)
        _shuffle_indices = list(range(len(_dataset)))
        np.random.shuffle(_shuffle_indices)
        self._dataset = np.array([_dataset[i] for i in _shuffle_indices])
        self._shuffle_indices = np.array(
            [x for x, _ in sorted(zip(range(len(_dataset)), _shuffle_indices), key=lambda x: x[1])]
        )

        self._shuffled_indices_inverse = _shuffle_indices

    def as_dataloader(self, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
        return DataLoader(
            self._indices,
            batch_size=batch_size,
            shuffle=shuffle,
        )
        

    def encode(
        self, 
        X: torch.Tensor,
        eps: float = 1e-4
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        X_SM = X[:, self._type == "SM"]
        
        X_ST = X[:, self._type == "ST"]
    
        X_ST = torch.log(X_ST + 1)
        
        
        q_sm = self.encoder_SM.encode(X_SM)
                                    
        q_st = self.encoder_ST.encode(X_ST)
        
        q = torch.hstack((q_sm,q_st))
        
        q_mu = self.z_mean_fc(q)
        q_var = torch.exp(self.z_var_fc(q)) + eps
        z = Normal(q_mu, q_var.sqrt()).rsample()
        H = dict(
            q = q,
            q_mu = q_mu, 
            q_var = q_var,
            z = z
        )

        return H 

    def decode(self, 
        H: Mapping[str, torch.tensor],
        lib_size:torch.tensor, 
    ) -> torch.Tensor:
        z = H["z"] # cell latent representation
        px = self.decoder(z)
        h = None
        px_rna_scale = self.px_rna_scale_decoder(px) 
        px_rna_rate = self.px_rna_rate_decoder(px)
        px_rna_dropout = self.px_rna_dropout_decoder(px)  ## In logits
        px_sm_scale = self.px_sm_scale_decoder(px)
        px_sm_rate = self.px_sm_rate_decoder(px)
        px_sm_dropout = self.px_sm_dropout_decoder(px)  ## In logits
        
        px_rna_scale = px_rna_scale * lib_size.unsqueeze(1)
        
        R = dict(
            h = h,
            px = px,
            px_rna_scale = px_rna_scale,
            px_rna_rate = px_rna_rate,
            px_rna_dropout = px_rna_dropout,
            px_sm_scale = px_sm_scale,
            px_sm_rate = px_sm_rate,
            px_sm_dropout = px_sm_dropout
        )
        return R

    def forward(
        self, 
        X: torch.Tensor,
        reduction: str = "sum", 
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:        
        H=self.encode(X)
        q_mu = H["q_mu"]
        q_var = H["q_var"]
        mean = torch.zeros_like(q_mu)
        scale = torch.ones_like(q_var)
        kldiv_loss = kld(Normal(q_mu, q_var.sqrt()),
                         Normal(mean, scale)).sum(dim = 1)

        X_SM = X[:,self._type=="SM"]
        X_ST = X[:,self._type=="ST"]

        R=self.decode(H, X_ST.sum(1))
                      
        if self.reconstruction_method_st == 'zinb':
            reconstruction_loss_st = LossFunction.zinb_reconstruction_loss(
                X_ST,
                mu = R['px_rna_scale'],
                theta = R['px_rna_rate'].exp(), 
                gate_logits = R['px_rna_dropout'],
                reduction = reduction
            )
            
        elif self.reconstruction_method_st == 'zg':
            reconstruction_loss_st = LossFunction.zi_gaussian_reconstruction_loss(
                X_ST,
                mean=R['px_rna_scale'],
                variance=R['px_rna_rate'].exp(),
                gate_logits=R['px_rna_dropout'],
                reduction=reduction
            )
        elif self.reconstruction_method_st == 'mse':
            reconstruction_loss_st = nn.functional.mse_loss(
                R['px_rna_scale'],
                X_ST,
                reduction=reduction
            )
        if self.reconstruction_method_sm == 'zg':
            reconstruction_loss_sm = LossFunction.zi_gaussian_reconstruction_loss(
                X_SM,
                mean = R['px_sm_scale'],
                variance = R['px_sm_rate'].exp(),
                gate_logits = R['px_sm_dropout'],
                reduction = reduction
            )
        elif self.reconstruction_method_sm == 'mse':
            reconstruction_loss_sm = nn.MSELoss(reduction='mean')(
                R['px_sm_scale'],
                X_SM,
            )
        elif self.reconstruction_method_sm == "g":
            reconstruction_loss_sm = LossFunction.gaussian_reconstruction_loss(
                X_SM,
                mean = R['px_sm_scale'],
                variance = R['px_sm_rate'].exp(),
                reduction = reduction
            )
            
        loss_record = {
            "reconstruction_loss_sm": reconstruction_loss_sm,
            "reconstruction_loss_st": reconstruction_loss_st,
            "kldiv_loss": kldiv_loss,
        }
        return H, R, loss_record
    
    def fit(self,
            max_epoch:int = 35, 
            n_per_batch:int = 128,
            reconstruction_reduction: str = 'sum',
            kl_weight: float = 2.,
            n_epochs_kl_warmup: Union[int, None] = 400,
            optimizer_parameters: Iterable = None,
            weight_decay: float = 1e-6,
            lr: bool = 5e-5,
            random_seed: int = 12,
        ):
        self.train()
        if n_epochs_kl_warmup:
            n_epochs_kl_warmup = min(max_epoch, n_epochs_kl_warmup)
            kl_warmup_gradient = kl_weight / n_epochs_kl_warmup
            kl_weight_max = kl_weight
            kl_weight = 0.
            
        if optimizer_parameters is None:
            optimizer = optim.AdamW(self.parameters(), lr, weight_decay=weight_decay)
        else:
            optimizer = optim.AdamW(optimizer_parameters, lr, weight_decay=weight_decay)        
        pbar = get_tqdm()(range(max_epoch), desc="Epoch", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        loss_record = {
            "reconstruction_loss_sm": 0,
            "reconstruction_loss_st": 0,
            "kldiv_loss": 0,
        }
        epoch_reconstruction_loss_st_list = []
        epoch_reconstruction_loss_sm_list = []
        epoch_kldiv_loss_list = []
        
        epoch_sm_gate_logits_list = []
        
        for epoch in range(1, max_epoch+1):
            self._trained = True
            pbar.desc = "Epoch {}".format(epoch)
            epoch_total_loss = 0
            epoch_reconstruction_loss_sm = 0
            epoch_reconstruction_loss_st = 0 
            epoch_kldiv_loss = 0
            
            epoch_sm_gate_logits = []
            
            X_train = self.as_dataloader(batch_size=n_per_batch, shuffle=True)         
            for b, X in enumerate(X_train):
                P = None
                X = self._dataset[X.cpu().numpy()]
                batch_index, label_index, additional_label_index, additional_batch_index = None, None, None, None   
                X = torch.tensor(np.vstack(list(map(lambda x: x.toarray() if issparse(x) else x, X))))
                X = X.to(self.device)
                lib_size = X.sum(1).to(self.device)
                
                H, R, L = self.forward(
                    X,
                    reduction=reconstruction_reduction,
                )
                epoch_sm_gate_logits.append(
                    R['px_sm_dropout'].detach().cpu().numpy()
                )
                
                
                reconstruction_loss_st = L['reconstruction_loss_st']
                reconstruction_loss_sm = L['reconstruction_loss_sm']
                kldiv_loss = kl_weight * L['kldiv_loss']    

                loss = 1*reconstruction_loss_sm.mean() + 0.5*reconstruction_loss_st.mean() + kldiv_loss.mean()

                avg_reconstruction_loss_st = reconstruction_loss_st.mean()  / n_per_batch
                avg_reconstruction_loss_sm = reconstruction_loss_sm.mean()  / n_per_batch
                avg_kldiv_loss = kldiv_loss.mean()  / n_per_batch

                epoch_reconstruction_loss_sm += avg_reconstruction_loss_sm.item()
                epoch_reconstruction_loss_st += avg_reconstruction_loss_st.item()
                
                epoch_kldiv_loss += avg_kldiv_loss.item()
                epoch_total_loss += loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            pbar.set_postfix({
                'reconst_sm': '{:.2e}'.format(epoch_reconstruction_loss_sm),
                'reconst_st': '{:.2e}'.format(epoch_reconstruction_loss_st),                  
                'kldiv': '{:.2e}'.format(epoch_kldiv_loss),
            }) 
            
            pbar.update(1)        
            epoch_reconstruction_loss_sm_list.append(epoch_reconstruction_loss_sm)
            epoch_reconstruction_loss_st_list.append(epoch_reconstruction_loss_st)
            epoch_kldiv_loss_list.append(epoch_kldiv_loss)
            
            epoch_sm_gate_logits = np.vstack(epoch_sm_gate_logits)
            epoch_sm_gate_logits_list.append(epoch_sm_gate_logits)
            
            if n_epochs_kl_warmup:
                    kl_weight = min( kl_weight + kl_warmup_gradient, kl_weight_max)
            random_seed += 1
                 
        pbar.close()
        self.trained_state_dict = deepcopy(self.state_dict())  
          
        return dict(  
            epoch_reconstruction_loss_st_list=epoch_reconstruction_loss_st_list,
            epoch_reconstruction_loss_sm_list=epoch_reconstruction_loss_sm_list,
            epoch_kldiv_loss_list=epoch_kldiv_loss_list,
            epoch_sm_gate_logits_list=epoch_sm_gate_logits_list
        )

    @torch.no_grad()
    def get_latent_embedding(
        self, 
        latent_key: Literal["z", "q_mu"] = "q_mu", 
        n_per_batch: int = 128,
        show_progress: bool = True
    ) -> np.ndarray:
        self.eval()
        X = self.as_dataloader(batch_size=n_per_batch, shuffle=False)
        Zs = []
        if show_progress:
            pbar = get_tqdm()(X, desc="Latent Embedding", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        
        for x in X:
            x = self._dataset[x.cpu().numpy()]
            batch_index = None             
            x = torch.tensor(np.vstack(list(map(lambda x: x.toarray() if issparse(x) else x, x))))
            if not isinstance(x, torch.FloatTensor):
                x = x.type(torch.FloatTensor)
            x = x.to(self.device)
                    
            H = self.encode(x)
            Zs.append(H[latent_key].detach().cpu().numpy())
            if show_progress:
                pbar.update(1)
        if show_progress:
            pbar.close()
        return np.vstack(Zs)[self._shuffle_indices]
    
    @torch.no_grad()
    def get_normalized_expression(
        self, 
        latent_key: Literal["z", "q_mu"] = "q_mu", 
        n_per_batch: int = 128,
        show_progress: bool = True
    ) -> np.ndarray:
        self.eval()
        X = self.as_dataloader(batch_size=n_per_batch, shuffle=False)
        Zs = []
        if show_progress:
            pbar = get_tqdm()(X, desc="Latent Embedding", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        
        for x in X:
            x = self._dataset[x.cpu().numpy()]
            batch_index = None             
            x = torch.tensor(np.vstack(list(map(lambda x: x.toarray() if issparse(x) else x, x))))
            if not isinstance(x, torch.FloatTensor):
                x = x.type(torch.FloatTensor)
            x = x.to(self.device)
            
            x_ST = x[:,self._type=="ST"]
                    
            H = self.encode(x)
            R = self.decode(H, x_ST.sum(1))
            
            Zs.append(
                np.hstack([
                    R['px_sm_scale'].detach().cpu().numpy(),
                    R['px_rna_scale'].detach().cpu().numpy()
                ])
            )
            if show_progress:
                pbar.update(1)
        if show_progress:
            pbar.close()
        return np.vstack(Zs)[self._shuffle_indices]
    
    
class VQVAE(BaseVAE):
    def __init__(self, vq_k: int, *args, **kwargs):
        super(VQVAE, self).__init__(*args, **kwargs)
        self.quantitizer = VectorQuantizer(
            n_e = vq_k,
            e_dim = self.n_latent,
            beta = 0.25,
        )
        self.z_mean_st_fc = nn.Linear(self.n_hidden, self.n_latent)
        self.z_var_st_fc = nn.Linear(self.n_hidden, self.n_latent)
        self.z_mean_sm_fc = nn.Linear(self.n_hidden, self.n_latent)
        self.z_var_sm_fc = nn.Linear(self.n_hidden, self.n_latent)
        
        
    def encode(
        self, 
        X: torch.Tensor,
        eps: float = 1e-4
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        X_SM = X[:, self._type == "SM"]
        X_ST = X[:, self._type == "ST"]
        X_ST = torch.log(X_ST + 1)
        q_sm = self.encoder_SM.encode(X_SM)
        q_st = self.encoder_ST.encode(X_ST)

        q_mu_st = self.z_mean_st_fc(q_st)
        q_var_st = torch.exp(self.z_var_st_fc(q_st)) + eps
        
        q_mu_sm = self.z_mean_sm_fc(q_sm)
        q_var_sm = torch.exp(self.z_var_sm_fc(q_sm)) + eps
        
        z_st = Normal(q_mu_st, q_var_st.sqrt()).rsample()
        z_sm = Normal(q_mu_sm, q_var_sm.sqrt()).rsample()
        
        z_vq_st, emb_loss_st, info_st = self.quantitizer(
            einops.rearrange(
                z_st,
                '(b w h) d -> b d w h', 
                w = 1,
                h = 1
            )
        )
        z_vq_st = einops.rearrange(z_vq_st, 'b d w h -> (b w h) d')
        
        z_vq_sm, emb_loss_sm, info_sm = self.quantitizer(
            einops.rearrange(
                z_sm,
                '(b w h) d -> b d w h', 
                w = 1,
                h = 1
            )
        )
        z_vq_sm = einops.rearrange(z_vq_sm, 'b d w h -> (b w h) d')
        
        H = dict(
            st = dict(
                q = q_st,
                q_mu = q_mu_st,
                q_var = q_var_st,
                z = z_st,
                z_vq = z_vq_st,
                emb_loss = emb_loss_st,
                info = info_st   
            ),
            sm = dict(
                q = q_sm,
                q_mu = q_mu_sm,
                q_var = q_var_sm,
                z = z_sm,
                z_vq = z_vq_sm,
                emb_loss = emb_loss_sm,
                info = info_sm
            )
        )

        return H 

    def decode(self, 
        H: Mapping[str, torch.tensor],
        lib_size:torch.tensor, 
    ) -> torch.Tensor:
        z_vq_st = H['st']['z_vq']
        z_vq_sm = H['sm']['z_vq']
        
        px_st = self.decoder(z_vq_st)
        px_sm = self.decoder(z_vq_sm)
        
        px_rna_scale = self.px_rna_scale_decoder(px_st) 
        px_rna_rate = self.px_rna_rate_decoder(px_st)
        px_rna_dropout = self.px_rna_dropout_decoder(px_st)  ## In logits
        px_sm_scale = self.px_sm_scale_decoder(px_sm)
        px_sm_rate = self.px_sm_rate_decoder(px_sm)
        px_sm_dropout = self.px_sm_dropout_decoder(px_sm)  ## In logits
        
        px_rna_scale = px_rna_scale * lib_size.unsqueeze(1)
        
        R = dict(
            px_rna_scale = px_rna_scale,
            px_rna_rate = px_rna_rate,
            px_rna_dropout = px_rna_dropout,
            px_sm_scale = px_sm_scale,
            px_sm_rate = px_sm_rate,
            px_sm_dropout = px_sm_dropout
        )
        return R
    
    def forward(
        self, 
        X: torch.Tensor,
        reduction: str = "sum", 
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:        
        H=self.encode(X)
        q_mu_st = H['st']["q_mu"]
        q_var_st= H['st']["q_var"]
        q_mu_sm = H['sm']["q_mu"]
        q_var_sm = H['sm']["q_var"]
        
        mean_st = torch.zeros_like(q_mu_st)
        scale_st = torch.ones_like(q_var_st)
        kldiv_loss_st = kld(Normal(q_mu_st, q_var_st.sqrt()),
                         Normal(mean_st, scale_st)).sum(dim = 1)
        
        mean_sm = torch.zeros_like(q_mu_sm)
        scale_sm = torch.ones_like(q_var_sm)
        kldiv_loss_sm = kld(Normal(q_mu_sm, q_var_sm.sqrt()),
                            Normal(mean_sm, scale_sm)).sum(dim = 1)
        

        X_SM = X[:,self._type=="SM"]
        X_ST = X[:,self._type=="ST"]

        R=self.decode(H, X_ST.sum(1))
                      
        if self.reconstruction_method_st == 'zinb':
            reconstruction_loss_st = LossFunction.zinb_reconstruction_loss(
                X_ST,
                mu = R['px_rna_scale'],
                theta = R['px_rna_rate'].exp(), 
                gate_logits = R['px_rna_dropout'],
                reduction = reduction
            )
            
        elif self.reconstruction_method_st == 'zg':
            reconstruction_loss_st = LossFunction.zi_gaussian_reconstruction_loss(
                X_ST,
                mean=R['px_rna_scale'],
                variance=R['px_rna_rate'].exp(),
                gate_logits=R['px_rna_dropout'],
                reduction=reduction
            )
        elif self.reconstruction_method_st == 'mse':
            reconstruction_loss_st = nn.functional.mse_loss(
                R['px_rna_scale'],
                X_ST,
                reduction=reduction
            )
        if self.reconstruction_method_sm == 'zg':
            reconstruction_loss_sm = LossFunction.zi_gaussian_reconstruction_loss(
                X_SM,
                mean = R['px_sm_scale'],
                variance = R['px_sm_rate'].exp(),
                gate_logits = R['px_sm_dropout'],
                reduction = reduction
            )
        elif self.reconstruction_method_sm == 'mse':
            reconstruction_loss_sm = nn.MSELoss(reduction='mean')(
                R['px_sm_scale'],
                X_SM,
            )
        elif self.reconstruction_method_sm == "g":
            reconstruction_loss_sm = LossFunction.gaussian_reconstruction_loss(
                X_SM,
                mean = R['px_sm_scale'],
                variance = R['px_sm_rate'].exp(),
                reduction = reduction
            )
        
        emb_loss_st = H['st'].pop('emb_loss')
        emb_loss_sm = H['sm'].pop('emb_loss')
        
        mmd_loss = mmd_loss_calc(
            H['st']['q_mu'],
            H['sm']['q_mu']
        )
        
        loss_record = {
            "reconstruction_loss_sm": reconstruction_loss_sm,
            "reconstruction_loss_st": reconstruction_loss_st,
            "kldiv_loss_st": kldiv_loss_st,
            "kldiv_loss_sm": kldiv_loss_sm,
            "emb_loss_st": emb_loss_st,
            "emb_loss_sm": emb_loss_sm,
            "mmd_loss": mmd_loss
        }
        return H, R, loss_record
    
    def fit(self,
            max_epoch:int = 35, 
            n_per_batch:int = 128,
            reconstruction_reduction: str = 'sum',
            kl_weight: float = 2.,
            n_epochs_kl_warmup: Union[int, None] = 400,
            optimizer_parameters: Iterable = None,
            weight_decay: float = 1e-6,
            lr: bool = 5e-5,
            random_seed: int = 12,
        ):
        self.train()
        if n_epochs_kl_warmup:
            n_epochs_kl_warmup = min(max_epoch, n_epochs_kl_warmup)
            kl_warmup_gradient = kl_weight / n_epochs_kl_warmup
            kl_weight_max = kl_weight
            kl_weight = 0.
            
        if optimizer_parameters is None:
            optimizer = optim.AdamW(self.parameters(), lr, weight_decay=weight_decay)
        else:
            optimizer = optim.AdamW(optimizer_parameters, lr, weight_decay=weight_decay)        
        pbar = get_tqdm()(range(max_epoch), desc="Epoch", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        loss_record = {
            "reconstruction_loss_sm": 0,
            "reconstruction_loss_st": 0,
            "kldiv_loss_st": 0,
            "kldiv_loss_sm": 0,
            "emb_loss_st": 0,
            "emb_loss_sm": 0,
            "mmd_loss": 0
        }
        epoch_reconstruction_loss_st_list = []
        epoch_reconstruction_loss_sm_list = []
        epoch_kldiv_loss_st_list = []
        epoch_kldiv_loss_sm_list = []
        epoch_emb_loss_st_list = []
        epoch_emb_loss_sm_list = []
        epoch_mmd_loss_list = []
        epoch_sm_gate_logits_list = []
        
        for epoch in range(1, max_epoch+1):
            self._trained = True
            pbar.desc = "Epoch {}".format(epoch)
            epoch_total_loss = 0
            epoch_reconstruction_loss_sm = 0
            epoch_reconstruction_loss_st = 0 
            epoch_kldiv_loss_st = 0
            epoch_kldiv_loss_sm = 0
            epoch_emb_loss_st = 0
            epoch_emb_loss_sm = 0
            epoch_mmd_loss = 0
            
            epoch_sm_gate_logits = []
            
            X_train = self.as_dataloader(batch_size=n_per_batch, shuffle=True)         
            for b, X in enumerate(X_train):
                P = None
                X = self._dataset[X.cpu().numpy()]
                batch_index, label_index, additional_label_index, additional_batch_index = None, None, None, None   
                X = torch.tensor(np.vstack(list(map(lambda x: x.toarray() if issparse(x) else x, X))))
                X = X.to(self.device)
                lib_size = X.sum(1).to(self.device)
                
                H, R, L = self.forward(
                    X,
                    reduction=reconstruction_reduction,
                )
                epoch_sm_gate_logits.append(
                    R['px_sm_dropout'].detach().cpu().numpy()
                )
                
                
                reconstruction_loss_st = L['reconstruction_loss_st']
                reconstruction_loss_sm = L['reconstruction_loss_sm']
                kldiv_loss_st = kl_weight * L['kldiv_loss_st']
                kldiv_loss_sm = kl_weight * L['kldiv_loss_sm']
                emb_loss_st = L['emb_loss_st']
                emb_loss_sm = L['emb_loss_sm']
                mmd_loss = L['mmd_loss']
                

                loss = 1*reconstruction_loss_sm.mean() + 0.5*reconstruction_loss_st.mean() + kldiv_loss_st.mean() + kldiv_loss_sm.mean() + emb_loss_st.mean() + emb_loss_sm.mean() + mmd_loss.mean()

                avg_reconstruction_loss_st = reconstruction_loss_st.mean()  / n_per_batch
                avg_reconstruction_loss_sm = reconstruction_loss_sm.mean()  / n_per_batch
                avg_kldiv_loss_st = kldiv_loss_st.mean()  / n_per_batch
                avg_kldiv_loss_sm = kldiv_loss_sm.mean()  / n_per_batch
                avg_emb_loss_st = emb_loss_st.mean()  / n_per_batch
                avg_emb_loss_sm = emb_loss_sm.mean()  / n_per_batch
                avg_mmd_loss = mmd_loss.mean()  / n_per_batch

                epoch_reconstruction_loss_sm += avg_reconstruction_loss_sm.item()
                epoch_reconstruction_loss_st += avg_reconstruction_loss_st.item()
                epoch_kldiv_loss_st += avg_kldiv_loss_st.item()
                epoch_kldiv_loss_sm += avg_kldiv_loss_sm.item()
                epoch_emb_loss_st += avg_emb_loss_st.item()
                epoch_emb_loss_sm += avg_emb_loss_sm.item()
                epoch_mmd_loss += avg_mmd_loss.item()
                epoch_total_loss += loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            pbar.set_postfix({
                'reconst_sm': '{:.2e}'.format(epoch_reconstruction_loss_sm),
                'reconst_st': '{:.2e}'.format(epoch_reconstruction_loss_st),                  
                'kldiv_st': '{:.2e}'.format(epoch_kldiv_loss_st),
                'kldiv_sm': '{:.2e}'.format(epoch_kldiv_loss_sm),
            }) 
            
            pbar.update(1)        
            epoch_reconstruction_loss_sm_list.append(epoch_reconstruction_loss_sm)
            epoch_reconstruction_loss_st_list.append(epoch_reconstruction_loss_st)
            epoch_kldiv_loss_st_list.append(epoch_kldiv_loss_st)
            epoch_kldiv_loss_sm_list.append(epoch_kldiv_loss_sm)
            epoch_emb_loss_st_list.append(epoch_emb_loss_st)
            epoch_emb_loss_sm_list.append(epoch_emb_loss_sm)
            epoch_mmd_loss_list.append(epoch_mmd_loss)
            
            epoch_sm_gate_logits = np.vstack(epoch_sm_gate_logits)
            epoch_sm_gate_logits_list.append(epoch_sm_gate_logits)
            
            if n_epochs_kl_warmup:
                kl_weight = min( kl_weight + kl_warmup_gradient, kl_weight_max)
            random_seed += 1
                 
        pbar.close()
        self.trained_state_dict = deepcopy(self.state_dict())  
          
        return dict(  
            epoch_reconstruction_loss_st_list=epoch_reconstruction_loss_st_list,
            epoch_reconstruction_loss_sm_list=epoch_reconstruction_loss_sm_list,
            epoch_kldiv_loss_st_list=epoch_kldiv_loss_st_list,
            epoch_kldiv_loss_sm_list=epoch_kldiv_loss_sm_list,
            epoch_emb_loss_st_list=epoch_emb_loss_st_list,
            epoch_emb_loss_sm_list=epoch_emb_loss_sm_list,
            epoch_mmd_loss_list=epoch_mmd_loss_list,
            epoch_sm_gate_logits_list=epoch_sm_gate_logits_list
        )

    @torch.no_grad()
    def get_vq_cluster(
        self,  
        n_per_batch: int = 128,
        show_progress: bool = True
    ) -> np.ndarray:
        self.eval()
        X = self.as_dataloader(batch_size=n_per_batch, shuffle=False)
        Zs = []
        if show_progress:
            pbar = get_tqdm()(X, desc="VQ Cluster", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        
        all_info_st = []
        all_info_sm = []
        
        for x in X:
            x = self._dataset[x.cpu().numpy()]
            batch_index = None             
            x = torch.tensor(np.vstack(list(map(lambda x: x.toarray() if issparse(x) else x, x))))
            if not isinstance(x, torch.FloatTensor):
                x = x.type(torch.FloatTensor)
            x = x.to(self.device)
            
            x_ST = x[:,self._type=="ST"]
                    
            H = self.encode(x)
            
            info_st = H['st']['info'][2].detach().cpu().numpy()
            info_sm = H['sm']['info'][2].detach().cpu().numpy()
            
            all_info_st.append(info_st)
            all_info_sm.append(info_sm)
            
        all_info_st = np.vstack(all_info_st)
        all_info_sm = np.vstack(all_info_sm)
        
        return all_info_st[self._shuffle_indices], all_info_sm[self._shuffle_indices]
            