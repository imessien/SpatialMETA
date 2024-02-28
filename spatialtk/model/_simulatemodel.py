# Pytorch
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.distributions import kl_divergence as kld
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
from ._primitives import *
from ..util.loss import LossFunction
from ..util.logger import get_tqdm

class SiVAE(nn.Module):
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
        reconstruction_method: Literal['mse', 'zg', 'zinb'] = 'zinb',
    ):
        super(BaseVAE, self).__init__()

        self.adata = adata 
    
        adata.hidden_stacks = hidden_stacks
        self.n_hidden = hidden_stacks[-1]
        self.n_latent = n_latent
        self.device = device
        self.reconstruction_method = reconstruction_method
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

    def initialize_dataset(self):
        X = self.adata.X
        self._type=np.array(list(self.adata.var.type.values))
        self.in_dim_ST = X.shape[1]
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
        
        libsize_ST = torch.log(X.sum(1))
        X = torch.log(X + 1)                          
        q = self.encoder_ST.encode(X)
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
        
        px_rna_scale = px_rna_scale * lib_size.unsqueeze(1)
        
        R = dict(
            h = h,
            px = px,
            px_rna_scale = px_rna_scale,
            px_rna_rate = px_rna_rate,
            px_rna_dropout = px_rna_dropout,
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
        kldiv_loss = kld(Normal(q_mu, q_var.sqrt()), Normal(mean, scale)).sum(dim = 1)
  

        R=self.decode(H, X.sum(1))
                      
        if self.reconstruction_method == 'zinb':
            reconstruction_loss_st = LossFunction.zinb_reconstruction_loss(
                X,
                mu = R['px_rna_scale'],
                theta = R['px_rna_rate'].exp(), 
                gate_logits = R['px_rna_dropout'],
                reduction = reduction
            )

            
        elif self.reconstruction_method == 'zg':
            reconstruction_loss_st = LossFunction.zi_gaussian_reconstruction_loss(
                X,
                mean=R['px_rna_scale'],
                variance=R['px_rna_rate'].exp(),
                gate_logits=R['px_rna_dropout'],
                reduction=reduction
            )

        elif self.reconstruction_method == 'mse':
            reconstruction_loss_st = nn.functional.mse_loss(
                R['px_rna_scale'],
                X,
                reduction=reduction
            )

        loss_record = {
            "reconstruction_loss_st": reconstruction_loss_st,
            "kldiv_loss": kldiv_loss,
        }
        return H, R, loss_record
    
    def fit(self,
            max_epoch:int = 35, 
            n_per_batch:int = 128,
            validation_split: float = .2,
            random_seed: int = 12,
            subset_indices: Union[torch.tensor, np.ndarray] = None,
            reconstruction_reduction: str = 'sum',
            kl_weight: float = 1.,
            optimizer_parameters: Iterable = None,
            weight_decay: float = 1e-6,
            lr: bool = 5e-5,
        ):
        self.train()
        if optimizer_parameters is None:
            optimizer = optim.AdamW(self.parameters(), lr, weight_decay=weight_decay)
        else:
            optimizer = optim.AdamW(optimizer_parameters, lr, weight_decay=weight_decay)        
        pbar = get_tqdm()(range(max_epoch), desc="Epoch", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        loss_record = {
            "reconstruction_loss_st": 0,
            "kldiv_loss": 0,
        }
        for epoch in range(1, max_epoch+1):
            self._trained = True
            pbar.desc = "Epoch {}".format(epoch)
            epoch_total_loss = 0
            epoch_reconstruction_loss_st = 0 
            epoch_kldiv_loss = 0
            X_train = self.as_dataloader(batch_size=n_per_batch, shuffle=True)         
            for b, X in enumerate(X_train):
                P = None
                X = self._dataset[X.cpu().numpy()]
                batch_index, label_index, additional_label_index, additional_batch_index = None, None, None, None   
                X = torch.tensor(np.vstack(list(map(lambda x: x.toarray() if issparse(x) else x, X))))
                X = X.to(self.device)
                lib_size = X.sum(1).to(self.device)
                
                H, R, L = self.forward(
                    X
                )
                reconstruction_loss_st = L['reconstruction_loss_st']
                kldiv_loss = kl_weight * L['kldiv_loss']    

                loss = reconstruction_loss_st.mean() + kldiv_loss.mean()

                avg_reconstruction_loss_st = reconstruction_loss_st.mean()  / n_per_batch
                avg_kldiv_loss = kldiv_loss.mean()  / n_per_batch

                epoch_reconstruction_loss_st += avg_reconstruction_loss_st.item()
                
                epoch_kldiv_loss += avg_kldiv_loss.item()
                epoch_total_loss += loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            pbar.set_postfix({
                'reconst_st': '{:.2e}'.format(epoch_reconstruction_loss_st),                  
                'kldiv': '{:.2e}'.format(epoch_kldiv_loss),
            }) 
            
            pbar.update(1)           
        pbar.close()
        self.trained_state_dict = deepcopy(self.state_dict())            
        
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
            
            x = x
                    
            H = self.encode(x)
            R = self.decode(H, x.sum(1))
            
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