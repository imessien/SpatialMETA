# Pytorch
from collections import Counter
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.distributions import kl_divergence as kld


# Third Party

import numpy as np


# Built-in
import numpy as np

from anndata import AnnData
from scipy.sparse import issparse

from copy import deepcopy
import json
from typing import Callable, Mapping, Union, Iterable, Tuple, Optional, Mapping
import os
import warnings


# Package
from ._primitives import *
from ..util.loss import LossFunction
from ..util.logger import get_tqdm
from ..util._classes import AnnDataSM, AnnDataST, AnnDataJointSMST

def get_k_elements(arr: Iterable, k:int):
    return list(map(lambda x: x[k], arr))

def get_last_k_elements(arr: Iterable, k:int):
    return list(map(lambda x: x[k:], arr))

def get_elements(arr: Iterable, a:int, b:int):
    return list(map(lambda x: x[a:a+b], arr))

class ConditionalVAE(nn.Module):
    """
    This class implements a Conditional Variational Autoencoder (CVAE) for vertical and horizontal integration of ST and SM.
    
    :param adata: AnnDataJointSMST object containing the spatial multi-omics data.
    :param hidden_stacks: List of integers specifying the number of hidden units in each stack of the encoder and decoder, default is [128].
    :param batch_keys: Optional list of strings specifying the batch keys for batch correction.
    :param n_latent: Integer specifying the dimensionality of the latent space, default is 10.
    :param bias: Boolean indicating whether to include bias terms in the linear layers, default is True.
    :param use_batch_norm: Boolean indicating whether to use batch normalization in the linear layers, default is True.
    :param use_layer_norm: Boolean indicating whether to use layer normalization in the linear layers, default is False.
    :param dropout_rate: Float specifying the dropout rate for the linear layers, default is 0.1.
    :param activation_fn: Callable specifying the activation function to use in the linear layers, default is nn.ReLU.
    :param device: String or torch.device specifying the device to use for computation, default is "cpu".
    :param batch_embedding: Literal["embedding", "onehot"] specifying the type of batch embedding to use, default is "onehot".
    :param encode_libsize: Boolean indicating whether to encode library size information, default is False.
    :param batch_hidden_dim: Integer specifying the dimensionality of the batch hidden layer, default is 8.
    :param reconstruction_method_st: Literal['mse', 'zg', 'zinb'] specifying the reconstruction method for the spatial data, default is 'zinb'. mse is mean squared error, zg is zero-inflated Gaussian, and zinb is zero-inflated negative binomial.
    :param reconstruction_method_sm: Literal['mse', 'zg', 'g'] specifying the reconstruction method for the single-cell multi-omics data, default is 'g'. mse is mean squared error, zg is zero-inflated Gaussian, and g is Gaussian.
    
    """
    def __init__(
        self,
        adata: AnnDataJointSMST,
        hidden_stacks: List[int] = [128], 
        batch_keys: Optional[List[str]] = None,
        n_latent: int = 10,
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
        reconstruction_method_sm: Literal['mse', 'zg', 'g'] = 'g'
    ):
        super(ConditionalVAE, self).__init__()

        self.adata = adata 

        self.hidden_stacks = hidden_stacks
        self.n_hidden = hidden_stacks[-1]
        self.n_latent = n_latent
        self.device = device
        self.reconstruction_method_st = reconstruction_method_st
        self.reconstruction_method_sm = reconstruction_method_sm
        self.encode_libsize = encode_libsize
        
        self.batch_hidden_dim = batch_hidden_dim
        self.batch_embedding = batch_embedding
        
        self.batch_keys = [batch_keys] if isinstance(batch_keys, str) else batch_keys

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
            encode_only = True,
            **self.fcargs
        )  
        
        self.encoder_SM = SAE(
            self.in_dim_SM,
            stacks = hidden_stacks,
            encode_only = True,
            **self.fcargs
        )
            
        self.decoder = FCLayer(
            in_dim = self.n_latent, 
            out_dim = self.n_hidden,
            n_cat_list = self.n_batch_keys,
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
        self.z_mean_fc_single = nn.Linear(self.n_hidden, self.n_latent)
        self.z_var_fc_single = nn.Linear(self.n_hidden, self.n_latent)

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
        
        #self.px_sm_scale_decoder = nn.Sequential(
        #    nn.Linear(self.n_hidden, self.in_dim_SM),
        #    nn.ReLU()
        #)
        
        self.px_sm_scale_decoder = nn.Linear(self.n_hidden, self.in_dim_SM)
        
        self.px_sm_dropout_decoder = nn.Linear(
            self.n_hidden,
            self.in_dim_SM
        )
        
        self.to(self.device)
        
    def as_dataloader(self, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
        return DataLoader(
            self._indices,
            batch_size=batch_size,
            shuffle=shuffle,
        )   
        
    def initialize_dataset(self):
        X = self.adata.X
        self._type=np.array(list(self.adata.var.type.values))
        self.in_dim_SM = X[:,self._type=="SM"].shape[1]
        self.in_dim_ST = X[:,self._type=="ST"].shape[1]
        self._n_record = X.shape[0]
        self._indices = np.array(list(range(self._n_record)))
        
        self.n_batch_keys = None 
        self.batch_categories = None 
        self.batch_category_summary = None 
        
        if self.batch_keys is not None:
            for e,i in enumerate(self.batch_keys):
                if i not in self.adata.obs.columns:
                    raise ValueError(f"batch_key {i} is not found in AnnData obs")
                
            self.n_batch_keys = [
                len(np.unique(self.adata.obs[x]))
                for x in self.batch_keys
            ]
            
            self.batch_categories = [
                pd.Categorical(self.adata.obs[x])
                for x in self.batch_keys
            ]
        
            self.batch_category_summary = [
                dict(Counter(x)) for x in self.batch_categories
            ]
        
            for i in range(len(self.batch_category_summary)):
                for k in self.batch_categories[i].categories:
                    if k not in self.batch_category_summary[i].keys():
                        self.batch_category_summary[i][k] = 0
                        
            batch_categories = [np.array(x.codes) for x in self.batch_categories]
            
            _dataset = list(zip(X, *batch_categories))
            
        else:
            _dataset = list(X)
            
            
        _shuffle_indices = list(range(len(_dataset)))
        np.random.shuffle(_shuffle_indices)
        self._dataset = np.array([_dataset[i] for i in _shuffle_indices])
        self._shuffle_indices = np.array(
            [x for x, _ in sorted(zip(range(len(_dataset)), _shuffle_indices), key=lambda x: x[1])]
        )

        self._shuffled_indices_inverse = _shuffle_indices

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
        lib_size: torch.tensor, 
        batch_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        z = H["z"] # cell latent representation
        
        if batch_index is not None:
            
            z = torch.hstack([
                z, 
                batch_index
            ])
        
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
        batch_index: Optional[torch.Tensor] = None, 
        reduction: str = "sum", 
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

        R=self.decode(H, X_ST.sum(1), batch_index)
                      
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

    def fit(
            self,
            max_epoch:int = 35,
            n_per_batch:int = 128,
            mode: Optional[Literal['single','multi']] = None,
            **kwargs
        ):
            """
            Fits the model.
            
            :param max_epoch: Integer specifying the maximum number of epochs to train the model, default is 35.
            :param n_per_batch: Integer specifying the number of samples per batch, default is 128.
            :param mode: Optional string specifying the mode of training. Can be either 'single' or 'multi', default is None.
            :param reconstruction_reduction: String specifying the reduction method for the reconstruction loss, default is 'sum'.
            :param kl_weight: Float specifying the weight of the KL divergence loss, default is 2.
            :param reconstruction_st_weight: Float specifying the weight of the reconstruction loss for the spatial data, default is 1.
            :param reconstruction_sm_weight: Float specifying the weight of the reconstruction loss for the single-cell multi-omics data, default is 1.
            :param n_epochs_kl_warmup: Integer specifying the number of epochs for KL divergence warmup, default is 400.
            :param optimizer_parameters: Iterable specifying the parameters for the optimizer, default is None.
            :param weight_decay: Float specifying the weight decay for the optimizer, default is 1e-6.
            :param lr: Float specifying the learning rate for the optimizer.
            :param random_seed: Integer specifying the random seed, default is 12.
            :param kl_loss_reduction: String specifying the reduction method for the KL divergence loss, default is 'mean'.

            :return: Dictionary containing the training loss values.
            """
            if mode == 'single':
                kwargs['kl_weight'] = 2.
                kwargs['n_epochs_kl_warmup'] = 35

            elif mode == 'multi':
                kwargs['kl_weight'] = 15.
                kwargs['n_epochs_kl_warmup'] = 0
                
            return self.fit_core(
                max_epoch=max_epoch,
                n_per_batch=n_per_batch,
                **kwargs
            )
            
    def fit_core(self,
            max_epoch:int = 35, 
            n_per_batch:int = 128,
            reconstruction_reduction: str = 'sum',
            kl_weight: float = 2.,
            reconstruction_st_weight: float = 1.,
            reconstruction_sm_weight: float = 1.,
            n_epochs_kl_warmup: Union[int, None] = 0,
            optimizer_parameters: Iterable = None,
            weight_decay: float = 1e-6,
            lr: bool = 5e-5,
            random_seed: int = 12,
            kl_loss_reduction: str = 'mean',
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
        epoch_total_loss_list = []
        
        epoch_sm_gate_logits_list = []
        
        for epoch in range(1, max_epoch+1):
            self._trained = True
            pbar.desc = "Epoch {}".format(epoch)
            epoch_total_loss = 0
            epoch_reconstruction_loss_sm = 0
            epoch_reconstruction_loss_st = 0 
            epoch_kldiv_loss = 0
            
            epoch_sm_gate_logits = []
            
            X_train = self.as_dataloader(
                batch_size=n_per_batch, 
                shuffle=True
            )   
            for b, X in enumerate(X_train):
                
                batch_data = self._dataset[X.cpu().numpy()]
                X = get_k_elements(batch_data, 0)
                batch_index = None 
                if self.batch_keys is not None:
                    batch_index = get_last_k_elements(
                        batch_data, 1
                    )
                    batch_index = list(np.vstack(batch_index).T.astype(float))
                    for i in range(len(batch_index)):
                        batch_index[i] = torch.tensor(batch_index[i])
                        if not isinstance(batch_index[i], torch.FloatTensor):
                            batch_index[i] = batch_index[i].type(torch.FloatTensor)
                            
                        batch_index[i] = batch_index[i].to(self.device).unsqueeze(1)
                        
                    batch_index = torch.hstack(batch_index)
                
                 
                X = torch.tensor(np.vstack(list(map(lambda x: x.toarray() if issparse(x) else x, X))))
                X = X.to(self.device)
                
                H, R, L = self.forward(
                    X,
                    batch_index=batch_index,
                    reduction=reconstruction_reduction,
                )
                
                epoch_sm_gate_logits.append(
                    R['px_sm_dropout'].detach().cpu().numpy()
                )
                
                reconstruction_loss_st = reconstruction_st_weight * L['reconstruction_loss_st']
                reconstruction_loss_sm = reconstruction_sm_weight * L['reconstruction_loss_sm']
                kldiv_loss = L['kldiv_loss']    

                #loss = 1*reconstruction_loss_sm.mean() + 0.5*reconstruction_loss_st.mean() + kldiv_loss.mean()

                avg_reconstruction_loss_st = reconstruction_loss_st.mean()  / n_per_batch
                avg_reconstruction_loss_sm = reconstruction_loss_sm.mean()  / n_per_batch
                if kl_loss_reduction == 'mean':
                    avg_kldiv_loss = kldiv_loss.mean()  / n_per_batch
                elif kl_loss_reduction == 'sum':
                    avg_kldiv_loss = kldiv_loss.sum()  / n_per_batch
                loss = avg_reconstruction_loss_sm + avg_reconstruction_loss_st + (avg_kldiv_loss * kl_weight)

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
                'total_loss': '{:.2e}'.format(epoch_total_loss),
            }) 
            
            pbar.update(1)        
            epoch_reconstruction_loss_sm_list.append(epoch_reconstruction_loss_sm)
            epoch_reconstruction_loss_st_list.append(epoch_reconstruction_loss_st)
            epoch_kldiv_loss_list.append(epoch_kldiv_loss)
            epoch_total_loss_list.append(epoch_total_loss)
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
            epoch_sm_gate_logits_list=epoch_sm_gate_logits_list,
            epoch_total_loss_list=epoch_total_loss_list
        )

    @torch.no_grad()
    def get_latent_embedding(
        self, 
        latent_key: Literal["z", "q_mu"] = "q_mu", 
        n_per_batch: int = 128,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Get the latent embedding of the data.
        
        :param latent_key: String specifying the key of the latent variable to return, default is "q_mu".
        :param n_per_batch: Integer specifying the number of samples per batch, default is 128.
        :param show_progress: Boolean indicating whether to show the progress bar, default is True.
        
        :return: Numpy array containing the latent embedding.
        """
        self.eval()
        X = self.as_dataloader(batch_size=n_per_batch, shuffle=False)
        Zs = []
        if show_progress:
            pbar = get_tqdm()(X, desc="Latent Embedding", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        for x in X:
            batch_data = self._dataset[x.cpu().numpy()]
            X = get_k_elements(batch_data, 0)
            x = torch.tensor(np.vstack(list(map(lambda x: x.toarray() if issparse(x) else x, X))))
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
        """
        Get the normalized expression of the data.
        
        :param latent_key: String specifying the key of the latent variable to return, default is "q_mu".
        :param n_per_batch: Integer specifying the number of samples per batch, default is 128.
        :param show_progress: Boolean indicating whether to show the progress bar, default is True.
        
        :return: Numpy array containing the normalized expression.
        """
        self.eval()
        X = self.as_dataloader(batch_size=n_per_batch, shuffle=False)
        Zs = []
        if show_progress:
            pbar = get_tqdm()(X, desc="Latent Embedding", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        
        for x in X:
            batch_data = self._dataset[x.cpu().numpy()]
            X = get_k_elements(batch_data, 0)
            x = torch.tensor(np.vstack(list(map(lambda x: x.toarray() if issparse(x) else x, X))))
            batch_index = None
            if self.batch_keys is not None:
                batch_index = get_last_k_elements(
                    batch_data, 1
                )
                batch_index = list(np.vstack(batch_index).T.astype(float))
                for i in range(len(batch_index)):
                    batch_index[i] = torch.tensor(batch_index[i])
                    if not isinstance(batch_index[i], torch.FloatTensor):
                        batch_index[i] = batch_index[i].type(torch.FloatTensor)
                    batch_index[i] = batch_index[i].to(self.device).unsqueeze(1)
                batch_index = torch.hstack(batch_index)
            if not isinstance(x, torch.FloatTensor):
                x = x.type(torch.FloatTensor)
            x = x.to(self.device)
            
            x_ST = x[:,self._type=="ST"]
                    
            H,R,_ = self.forward(x, batch_index=batch_index)
            
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

class ConditionalVAESM(nn.Module):
    """
    This class implements a Conditional Variational Autoencoder (CVAE) for vertical integration SM.
    
    :param adata: AnnDataSM object containing the SM data.
    :param hidden_stacks: List of integers specifying the number of hidden units in each stack of the encoder and decoder, default is [128].
    :param batch_keys: Optional list of strings specifying the batch keys for batch correction.
    :param n_latent: Integer specifying the dimensionality of the latent space, default is 10.
    :param bias: Boolean indicating whether to include bias terms in the linear layers, default is True.
    :param use_batch_norm: Boolean indicating whether to use batch normalization in the linear layers, default is True.
    :param use_layer_norm: Boolean indicating whether to use layer normalization in the linear layers, default is False.
    :param dropout_rate: Float specifying the dropout rate for the linear layers, default is 0.1.
    :param activation_fn: Callable specifying the activation function to use in the linear layers, default is nn.ReLU.
    :param device: String or torch.device specifying the device to use for computation, default is "cpu".
    :param batch_embedding: Literal["embedding", "onehot"] specifying the type of batch embedding to use, default is "onehot".
    :param encode_libsize: Boolean indicating whether to encode library size information, default is False.
    :param batch_hidden_dim: Integer specifying the dimensionality of the batch hidden layer, default is 8.
    :param reconstruction_method: Literal['mse', 'zg', 'zinb'] specifying the reconstruction method, default is 'g'. mse is mean squared error, zg is zero-inflated Gaussian, and zinb is zero-inflated negative binomial.
    
    return: Dictionary containing the training loss values.
    """
    def __init__(
        self,
        adata: AnnData,
        hidden_stacks: List[int] = [128], 
        batch_keys: Optional[List[str]] = None,
        n_latent: int = 10,
        bias: bool = True,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        dropout_rate: float = 0.1,
        activation_fn: Callable = nn.ReLU,
        device: Union[str, torch.device] = "cpu",
        batch_embedding: Literal["embedding", "onehot"] = "onehot",
        encode_libsize: bool = False,
        batch_hidden_dim: int = 8,
        reconstruction_method: Literal['mse', 'zg', 'zinb','g'] = 'g'
    ):
        
        super(ConditionalVAESM, self).__init__()

        self.adata = adata 

        self.hidden_stacks = hidden_stacks
        self.n_hidden = hidden_stacks[-1]
        self.n_latent = n_latent
        self.device = device
        self.reconstruction_method = reconstruction_method
        self.encode_libsize = encode_libsize
        
        self.batch_keys = [batch_keys] if isinstance(batch_keys, str) else batch_keys

        self.initialize_dataset()

        self.fcargs = dict(
            bias           = bias, 
            dropout_rate   = dropout_rate, 
            use_batch_norm = use_batch_norm, 
            use_layer_norm = use_layer_norm,
            activation_fn  = activation_fn,
            device         = device
        )
        
        
        self.encoder = SAE(
            self.in_dim if not self.encode_libsize else self.in_dim + 1,
            stacks = hidden_stacks,
            encode_only = True,
            **self.fcargs
        )  
        
        self.decoder = FCLayer(
            in_dim = self.n_latent, 
            out_dim = self.n_hidden,
            n_cat_list = self.n_batch_keys,
            cat_dim = batch_hidden_dim,
            cat_embedding = batch_embedding,
            use_layer_norm=False,
            use_batch_norm=True,
            dropout_rate=0,
            device=device
        ) 
        
        self.encode_libsize = encode_libsize
        
        # The latent cell representation z ~ Logisticnormal(0, I)
        self.z_mean_fc = nn.Linear(self.n_hidden, self.n_latent)
        self.z_var_fc = nn.Linear(self.n_hidden, self.n_latent)
        
        self.px_sm_rate_decoder = nn.Linear(
            self.n_hidden, 
            self.in_dim
        )
        
        self.px_sm_scale_decoder = nn.Linear(self.n_hidden, self.in_dim)
        
        self.px_sm_dropout_decoder = nn.Linear(
            self.n_hidden,
            self.in_dim
        )
        
        self.to(self.device)
    
    def as_dataloader(self, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
        return DataLoader(
            self._indices,
            batch_size=batch_size,
            shuffle=shuffle,
        )
        
    def initialize_dataset(self):
        X = self.adata.X
        self._type=np.array(list(self.adata.var.type.values))
        self.in_dim = X.shape[1]
        self._n_record = X.shape[0]
        self._indices = np.array(list(range(self._n_record)))
        
        self.n_batch_keys = None 
        self.batch_categories = None 
        self.batch_category_summary = None 
        
        if self.batch_keys is not None:
            for e,i in enumerate(self.batch_keys):
                if i not in self.adata.obs.columns:
                    raise ValueError(f"batch_key {i} is not found in AnnData obs")
                
            self.n_batch_keys = [
                len(np.unique(self.adata.obs[x]))
                for x in self.batch_keys
            ]
            
            self.batch_categories = [
                pd.Categorical(self.adata.obs[x])
                for x in self.batch_keys
            ]
        
            self.batch_category_summary = [
                dict(Counter(x)) for x in self.batch_categories
            ]
        
            for i in range(len(self.batch_category_summary)):
                for k in self.batch_categories[i].categories:
                    if k not in self.batch_category_summary[i].keys():
                        self.batch_category_summary[i][k] = 0
                        
            batch_categories = [np.array(x.codes) for x in self.batch_categories]
            
            _dataset = list(zip(X, *batch_categories))
            
        else:
            _dataset = list(X)
            
            
        _shuffle_indices = list(range(len(_dataset)))
        np.random.shuffle(_shuffle_indices)
        self._dataset = np.array([_dataset[i] for i in _shuffle_indices])
        self._shuffle_indices = np.array(
            [x for x, _ in sorted(zip(range(len(_dataset)), _shuffle_indices), key=lambda x: x[1])]
        )

        self._shuffled_indices_inverse = _shuffle_indices 
        
    def encode(
        self, 
        X: torch.Tensor,
        eps: float = 1e-4
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        X = torch.log(X + 1)
        
        q = self.encoder.encode(X)
        
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
        lib_size: torch.tensor, 
        batch_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        z = H["z"]
        
        if batch_index is not None:
            z = torch.hstack([
                z, 
                batch_index
            ])
            
        px = self.decoder(z)
        
        h = None
        px_sm_scale = self.px_sm_scale_decoder(px)
        px_sm_rate = self.px_sm_rate_decoder(px)
        px_sm_dropout = self.px_sm_dropout_decoder(px)
        
        R = dict(
            h = h,
            px = px,
            px_sm_scale = px_sm_scale,
            px_sm_rate = px_sm_rate,
            px_sm_dropout = px_sm_dropout
        )
        return R
    
    def forward(
        self,
        X: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        reduction: str = "sum",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        H=self.encode(X)
        q_mu = H["q_mu"]
        q_var = H["q_var"]
        mean = torch.zeros_like(q_mu)
        scale = torch.ones_like(q_var)
        if batch_index is not None:
            mmd_loss = LossFunction.mmd_loss(
                        z = H['q_mu'],
                        cat = batch_index.detach().cpu().numpy(),
                        dim=1,
                    )
        else:
            mmd_loss = torch.tensor(0.0, device=self.device)
        
        kldiv_loss = kld(Normal(q_mu, q_var.sqrt()),
                         Normal(mean, scale)).sum(dim = 1)
        
        R=self.decode(H, X.sum(1), batch_index)
        
        if self.reconstruction_method == 'zinb':
            reconstruction_loss = LossFunction.zinb_reconstruction_loss(
                X,
                mu = R['px_sm_scale'],
                theta = R['px_sm_rate'].exp(), 
                gate_logits = R['px_sm_dropout'],
                reduction = reduction
            )
        elif self.reconstruction_method == 'zg':
            reconstruction_loss = LossFunction.zi_gaussian_reconstruction_loss(
                X,
                mean=R['px_sm_scale'],
                variance=R['px_sm_rate'].exp(),
                gate_logits=R['px_sm_dropout'],
                reduction=reduction
            )
        elif self.reconstruction_method == 'mse':
            reconstruction_loss = nn.functional.mse_loss(
                R['px_sm_scale'],
                X,
                reduction=reduction
            )
        elif self.reconstruction_method == 'g':
            reconstruction_loss = LossFunction.gaussian_reconstruction_loss(
                X,
                mean = R['px_sm_scale'],
                variance = R['px_sm_rate'].exp(),
                reduction = reduction
            )
            
        loss_record = {
            "reconstruction_loss": reconstruction_loss,
            "kldiv_loss": kldiv_loss,
            "mmd_loss": mmd_loss
        }
        return H, R, loss_record
    
    def fit(self,
            max_epoch:int = 35, 
            n_per_batch:int = 128,
            reconstruction_reduction: str = 'sum',
            kl_weight: float = 15.,
            reconstruction_weight: float = 1.,
            n_epochs_kl_warmup: Union[int, None] = 0,
            optimizer_parameters: Iterable = None,
            weight_decay: float = 1e-6,
            lr: bool = 5e-5,
            random_seed: int = 12,
            kl_loss_reduction: str = 'mean',
            mmd_weight: float = 1.,
        ):
        """
        Fits the model.
        
        :param max_epoch: Integer specifying the maximum number of epochs to train the model, default is 35.
        :param n_per_batch: Integer specifying the number of samples per batch, default is 128.
        :param reconstruction_reduction: String specifying the reduction method for the reconstruction loss, default is 'sum'.
        :param kl_weight: Float specifying the weight of the KL divergence loss, default is 15.
        :param reconstruction_weight: Float specifying the weight of the reconstruction loss, default is 1.
        :param n_epochs_kl_warmup: Integer specifying the number of epochs for KL divergence warmup, default is 0.
        :param optimizer_parameters: Iterable specifying the parameters for the optimizer, default is None.
        :param weight_decay: Float specifying the weight decay for the optimizer, default is 1e-6.
        :param lr: Float specifying the learning rate for the optimizer.
        :param random_seed: Integer specifying the random seed, default is 12.
        :param kl_loss_reduction: String specifying the reduction method for the KL divergence loss, default is 'mean'.
        
        :return: Dictionary containing the training loss values.
        """
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
            "reconstruction_loss": 0,
            "kldiv_loss": 0,
            "mmd_loss": 0
        }
        epoch_reconstruction_loss_list = []
        epoch_kldiv_loss_list = []
        epoch_total_loss_list = []
        epoch_mmd_loss_list = []
        
        epoch_sm_gate_logits_list = []
        
        for epoch in range(1, max_epoch+1):
            self._trained = True
            pbar.desc = "Epoch {}".format(epoch)
            epoch_total_loss = 0
            epoch_reconstruction_loss = 0
            epoch_kldiv_loss = 0
            epoch_mmd_loss = 0
            epoch_sm_gate_logits = []
            
            X_train = self.as_dataloader(
                batch_size=n_per_batch, 
                shuffle=True
            )
            for b, X in enumerate(X_train):
                
                batch_data = self._dataset[X.cpu().numpy()]
                X = get_k_elements(batch_data, 0)
                batch_index = None 
                if self.batch_keys is not None:
                    batch_index = get_last_k_elements(
                        batch_data, 1
                    )
                    batch_index = list(np.vstack(batch_index).T.astype(float))
                    for i in range(len(batch_index)):
                        batch_index[i] = torch.tensor(batch_index[i])
                        if not isinstance(batch_index[i], torch.FloatTensor):
                            batch_index[i] = batch_index[i].type(torch.FloatTensor)
                        batch_index[i] = batch_index[i].to(self.device).unsqueeze(1)
                    batch_index = torch.hstack(batch_index)
                
                X = torch.tensor(np.vstack(list(map(lambda x: x.toarray() if issparse(x) else x, X))))
                X = X.to(self.device)
                
                H, R, L = self.forward(
                    X,
                    batch_index=batch_index,
                    reduction=reconstruction_reduction,
                )
                
                epoch_sm_gate_logits.append(
                    R['px_sm_dropout'].detach().cpu().numpy()
                )
                
                reconstruction_loss =  L['reconstruction_loss']
                kldiv_loss = L['kldiv_loss']
                mmd_loss = L['mmd_loss']
                #loss = reconstruction_loss.mean() + kldiv_loss.mean()

                avg_reconstruction_loss = reconstruction_loss.mean()  / n_per_batch
                avg_mmd_loss = mmd_loss.mean()  / n_per_batch
                
                if kl_loss_reduction == 'mean':
                    avg_kldiv_loss = kldiv_loss.mean()  / n_per_batch
                elif kl_loss_reduction == 'sum':
                    avg_kldiv_loss = kldiv_loss.sum()  / n_per_batch
                    
                
                loss = (avg_reconstruction_loss)*reconstruction_weight + \
                (avg_kldiv_loss * kl_weight) + \
                (avg_mmd_loss * mmd_weight)

                epoch_reconstruction_loss += avg_reconstruction_loss.item()
                epoch_kldiv_loss += avg_kldiv_loss.item()
                epoch_total_loss += loss.item()
                epoch_mmd_loss += avg_mmd_loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            pbar.set_postfix({
                'reconst': '{:.2e}'.format(epoch_reconstruction_loss),
                'kldiv': '{:.2e}'.format(epoch_kldiv_loss),
                'total_loss': '{:.2e}'.format(epoch_total_loss),
                'mmd_loss': '{:.2e}'.format(epoch_mmd_loss)
            })
            
            pbar.update(1)
            epoch_reconstruction_loss_list.append(epoch_reconstruction_loss)
            epoch_kldiv_loss_list.append(epoch_kldiv_loss)
            epoch_total_loss_list.append(epoch_total_loss)
            epoch_sm_gate_logits = np.vstack(epoch_sm_gate_logits)
            epoch_sm_gate_logits_list.append(epoch_sm_gate_logits)
            epoch_mmd_loss_list.append(epoch_mmd_loss)
            
            if n_epochs_kl_warmup:
                    kl_weight = min( kl_weight + kl_warmup_gradient, kl_weight_max)
            random_seed += 1
            
        pbar.close()
        self.trained_state_dict = deepcopy(self.state_dict())
        
        return dict(
            epoch_reconstruction_loss_list=epoch_reconstruction_loss_list,
            epoch_kldiv_loss_list=epoch_kldiv_loss_list,
            epoch_sm_gate_logits_list=epoch_sm_gate_logits_list,
            epoch_total_loss_list=epoch_total_loss_list,
            epoch_mmd_loss_list=epoch_mmd_loss_list
        )
        
    @torch.no_grad()
    def get_latent_embedding(
        self, 
        latent_key: Literal["z", "q_mu"] = "q_mu", 
        n_per_batch: int = 128,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Get the latent embedding of the data.
        
        :param latent_key: String specifying the key of the latent variable to return, default is "q_mu".
        :param n_per_batch: Integer specifying the number of samples per batch, default is 128.
        :param show_progress: Boolean indicating whether to show the progress bar, default is True.
        
        :return: Numpy array containing the latent embedding.
        """
        self.eval()
        X = self.as_dataloader(batch_size=n_per_batch, shuffle=False)
        Zs = []
        if show_progress:
            pbar = get_tqdm()(X, desc="Latent Embedding", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        for x in X:
            batch_data = self._dataset[x.cpu().numpy()]
            X = get_k_elements(batch_data, 0)
            x = torch.tensor(np.vstack(list(map(lambda x: x.toarray() if issparse(x) else x, X))))
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
        """
        Get the normalized expression of the data.
        
        :param latent_key: String specifying the key of the latent variable to return, default is "q_mu".
        :param n_per_batch: Integer specifying the number of samples per batch, default is 128.
        :param show_progress: Boolean indicating whether to show the progress bar, default is True.
        
        :return: Numpy array containing the normalized expression.
        """
        self.eval()
        X = self.as_dataloader(batch_size=n_per_batch, shuffle=False)
        Zs = []
        if show_progress:
            pbar = get_tqdm()(X, desc="Latent Embedding", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        
        for x in X:
            batch_data = self._dataset[x.cpu().numpy()]
            X = get_k_elements(batch_data, 0)
            x = torch.tensor(np.vstack(list(map(lambda x: x.toarray() if issparse(x) else x, X))))
            batch_index = None
            if self.batch_keys is not None:
                batch_index = get_last_k_elements(
                    batch_data, 1
                )
                batch_index = list(np.vstack(batch_index).T.astype(float))
                for i in range(len(batch_index)):
                    batch_index[i] = torch.tensor(batch_index[i])
                    if not isinstance(batch_index[i], torch.FloatTensor):
                        batch_index[i] = batch_index[i].type(torch.FloatTensor)
                    batch_index[i] = batch_index[i].to(self.device).unsqueeze(1)
                batch_index = torch.hstack(batch_index)
            if not isinstance(x, torch.FloatTensor):
                x = x.type(torch.FloatTensor)
            x = x.to(self.device)
            
            H,R,_ = self.forward(x, batch_index=batch_index)
            
            Zs.append(
                np.hstack([
                    R['px_sm_scale'].detach().cpu().numpy()
                ])
            )
            if show_progress:
                pbar.update(1)
        if show_progress:
            pbar.close()
        return np.vstack(Zs)[self._shuffle_indices]
 
    