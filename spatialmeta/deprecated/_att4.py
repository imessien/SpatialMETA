# Pytorch
from collections import Counter
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.distributions import kl_divergence as kld


# Third Party
import numpy as np
import scanpy as sc

# Built-in
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

from ._model import ConditionalVAE

def get_k_elements(arr: Iterable, k:int):
    return list(map(lambda x: x[k], arr))

def get_last_k_elements(arr: Iterable, k:int):
    return list(map(lambda x: x[k:], arr))

def get_elements(arr: Iterable, a:int, b:int):
    return list(map(lambda x: x[a:a+b], arr))

def FLATTEN(x): return [i for s in x for i in s]

class ConditionalVAEAtt4(ConditionalVAE):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.v = nn.Parameter(
            torch.rand(self.n_latent,device=self.device)
        )
        self.to_k = nn.Sequential(
            nn.Linear(self.n_latent, self.n_latent),
            nn.Tanh()
        ).to(self.device)
        
        self.decoder_st = FCLayer(
            in_dim = self.n_latent, 
            out_dim = self.n_hidden,
            n_cat_list = self.n_batch_keys,
            cat_dim = self.batch_hidden_dim,
            cat_embedding = self.batch_embedding,
            use_layer_norm=False,
            use_batch_norm=True,
            dropout_rate=0,
            device=self.device
        )
        self.decoder_sm = FCLayer(
            in_dim = self.n_latent, 
            out_dim = self.n_hidden,
            n_cat_list = self.n_batch_keys,
            cat_dim = self.batch_hidden_dim,
            cat_embedding = self.batch_embedding,
            use_layer_norm=False,
            use_batch_norm=True,
            dropout_rate=0,
            device=self.device
        )
        self.to(self.device)
    
    def encode(
        self, 
        X: torch.Tensor,
        eps: float = 1e-4
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        X_SM = X[:, self._type == "SM"]
        X_ST = X[:, self._type == "ST"]
        X_ST = torch.log(X_ST + 1)
        
        q_sm = self.encoder_SM.encode(X_SM).to(self.device)                           
        q_st = self.encoder_ST.encode(X_ST).to(self.device)
        mu_sm = self.z_mean_fc_single(q_sm)
        var_sm = torch.exp(self.z_var_fc_single(q_sm)) + eps
        mu_st = self.z_mean_fc_single(q_st)
        var_st = torch.exp(self.z_var_fc_single(q_st)) + eps
        z_sm = Normal(mu_sm, var_sm.sqrt()).rsample()
        z_st = Normal(mu_st, var_st.sqrt()).rsample()
        
        g_sm = torch.exp(torch.matmul(self.to_k(z_sm), self.v))
        g_st = torch.exp(torch.matmul(self.to_k(z_st), self.v))
        beta_sm = g_sm / (g_sm + g_st)
        beta_st = g_st / (g_sm + g_st)
        
        q_mu = mu_sm * beta_sm.unsqueeze(1) +  mu_st * beta_st.unsqueeze(1)
        q_var = var_sm * beta_sm.unsqueeze(1) +  var_st * beta_st.unsqueeze(1)
        z = z_sm * beta_sm.unsqueeze(1) + z_st * beta_st.unsqueeze(1)
        
        H = dict(
            st = dict(
                q = q_st,
                q_mu = mu_st,
                q_var = var_st,
                z = z_st,
                beta = beta_st,
            ),
            sm = dict(
                q = q_sm,
                q_mu = mu_sm,
                q_var = var_sm,
                z = z_sm,
                beta = beta_sm,
            ),
            #q = q,
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
        z_st = H["st"]["q_mu"]
        z_sm = H["sm"]["q_mu"]
        z = H["z"]
        
        z = z.to(self.device)
        z_st = z_st.to(self.device)
        z_sm = z_sm.to(self.device)
        
        if batch_index is not None:
            z_st = torch.hstack([
                z_st, 
                batch_index.to(self.device)
            ])
            z_sm = torch.hstack([
                z_sm, 
                batch_index.to(self.device)
            ])
            z = torch.hstack([
                z, 
                batch_index.to(self.device)
            ])
        
        R = []

        px_st = self.decoder_st(z.to(self.device))
        px_sm = self.decoder_sm(z.to(self.device))
        px_st_corr = self.decoder_st(z_sm.to(self.device))
        px_sm_corr = self.decoder_sm(z_st.to(self.device))
            

        px_rna_scale = self.px_rna_scale_decoder(px_st) 
        px_rna_rate = self.px_rna_rate_decoder(px_st)
        px_rna_dropout = self.px_rna_dropout_decoder(px_st)  ## In logits
        px_rna_scale = px_rna_scale * lib_size.unsqueeze(1)


        px_sm_scale = self.px_sm_scale_decoder(px_sm)
        #px_sm_scale = F.softmax(px_sm_scale, dim=1) * 1e4
        px_sm_rate = self.px_sm_rate_decoder(px_sm)
        px_sm_dropout = self.px_sm_dropout_decoder(px_sm)  ## In logits
        
        px_rna_corr_scale = self.px_rna_scale_decoder(px_st_corr)
        px_rna_corr_rate = self.px_rna_rate_decoder(px_st_corr)
        px_rna_corr_dropout = self.px_rna_dropout_decoder(px_st_corr)  ## In logits
        px_rna_corr_scale = px_rna_corr_scale * lib_size.unsqueeze(1)

        px_sm_corr_scale = self.px_sm_scale_decoder(px_sm_corr)
        #px_sm_corr_scale = F.softmax(px_sm_corr_scale, dim=1) * 1e4
        px_sm_corr_rate = self.px_sm_rate_decoder(px_sm_corr)
        px_sm_corr_dropout = self.px_sm_dropout_decoder(px_sm_corr)  ## In logits

        R = dict(
            latent = dict(
                    px_st = px_st,
                    px_sm = px_sm,
                    px_rna_scale = px_rna_scale,
                    px_rna_rate = px_rna_rate,
                    px_rna_dropout = px_rna_dropout,
                    px_sm_scale = px_sm_scale,
                    px_sm_rate = px_sm_rate,
                    px_sm_dropout = px_sm_dropout
            ),
            corr = dict(
                px_st = px_st_corr,
                px_sm = px_sm_corr,
                px_rna_scale = px_rna_corr_scale,
                px_rna_rate = px_rna_corr_rate,
                px_rna_dropout = px_rna_corr_dropout,
                px_sm_scale = px_sm_corr_scale,
                px_sm_rate = px_sm_corr_rate,
                px_sm_dropout = px_sm_corr_dropout
            )
            )
        return R
    
    def forward(
        self, 
        X: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None, 
        reduction: str = "sum", 
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:        
        H=self.encode(X)
        mu_st = H["st"]["q_mu"]
        var_st = H["st"]["q_var"]
        mu_sm = H["sm"]["q_mu"]
        var_sm = H["sm"]["q_var"]       
        mean_st = torch.zeros_like(mu_st)
        scale_st = torch.ones_like(var_st)
        mean_sm = torch.zeros_like(mu_sm)
        scale_sm = torch.ones_like(var_sm)
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
            #mmd_loss is 0 and shape is equal to the 
            mmd_loss = torch.tensor(0.0, device=self.device)
        
        kldiv_loss = kld(Normal(q_mu, q_var.sqrt()),
                         Normal(mean, scale)).sum(dim = 1)
        # kldiv_loss_st = kld(Normal(mu_st, var_st.sqrt()),Normal(mean_st, scale_st)).sum(dim = 1)
        # kldiv_loss_sm = kld(Normal(mu_sm, var_sm.sqrt()),Normal(mean_sm, scale_sm)).sum(dim = 1)

        X_SM = X[:,self._type=="SM"]
        X_ST = X[:,self._type=="ST"]

        Rs=self.decode(H, X_ST.sum(1), 
                       batch_index)
        R_latent = Rs['latent']
        R_corr = Rs['corr']
        
        reconstruction_loss_st = 0
        reconstruction_loss_sm = 0
        reconstruction_loss_st_corr = 0
        reconstruction_loss_sm_corr = 0
        
        
        if self.reconstruction_method_st == 'zinb':
            reconstruction_loss_st = LossFunction.zinb_reconstruction_loss(
                X_ST,
                mu = R_latent['px_rna_scale'],
                theta = R_latent['px_rna_rate'].exp(), 
                gate_logits = R_latent['px_rna_dropout'],
                reduction = reduction
            )
            reconstruction_loss_st_corr = LossFunction.zinb_reconstruction_loss(
                X_ST,
                mu = R_corr['px_rna_scale'],
                theta = R_corr['px_rna_rate'].exp(), 
                gate_logits = R_corr['px_rna_dropout'],
                reduction = reduction
            )
            
        elif self.reconstruction_method_st == 'zg':
            reconstruction_loss_st = LossFunction.zi_gaussian_reconstruction_loss(
                X_ST,
                mean=R_latent['px_rna_scale'],
                variance=R_latent['px_rna_rate'].exp(),
                gate_logits=R_latent['px_rna_dropout'],
                reduction=reduction
            )
            reconstruction_loss_st_corr = LossFunction.zi_gaussian_reconstruction_loss(
                X_ST,
                mean=R_corr['px_rna_scale'],
                variance=R_corr['px_rna_rate'].exp(),
                gate_logits=R_corr['px_rna_dropout'],
                reduction=reduction
            )
            
        elif self.reconstruction_method_st == 'mse':
            reconstruction_loss_st = nn.functional.mse_loss(
                R_latent['px_rna_scale'],
                X_ST,
                reduction=reduction
            )
            reconstruction_loss_st_corr = nn.functional.mse_loss(
                R_corr['px_rna_scale'],
                X_ST,
                reduction=reduction
            )
            
        if self.reconstruction_method_sm == 'zg':
            reconstruction_loss_sm = LossFunction.zi_gaussian_reconstruction_loss(
                X_SM,
                mean = R_latent['px_sm_scale'],
                variance = R_latent['px_sm_rate'].exp(),
                gate_logits = R_latent['px_sm_dropout'],
                reduction = reduction
            )
            reconstruction_loss_sm_corr = LossFunction.zi_gaussian_reconstruction_loss(
                X_SM,
                mean = R_corr['px_sm_scale'],
                variance = R_corr['px_sm_rate'].exp(),
                gate_logits = R_corr['px_sm_dropout'],
                reduction = reduction
            )
            
            
        elif self.reconstruction_method_sm == 'mse':
            reconstruction_loss_sm = nn.MSELoss(reduction='mean')(
                R_latent['px_sm_scale'],
                X_SM,
            )
            reconstruction_loss_sm_corr = nn.MSELoss(reduction='mean')(
                R_corr['px_sm_scale'],
                X_SM,
            )            
            
        elif self.reconstruction_method_sm == "g":
            reconstruction_loss_sm = LossFunction.gaussian_reconstruction_loss(
                X_SM,
                mean = R_latent['px_sm_scale'],
                variance = R_latent['px_sm_rate'].exp(),
                reduction = reduction
            )
            reconstruction_loss_sm_corr = LossFunction.gaussian_reconstruction_loss(
                X_SM,
                mean = R_corr['px_sm_scale'],
                variance = R_corr['px_sm_rate'].exp(),
                reduction = reduction
            )            
            
        loss_record = {
            "reconstruction_loss_sm": reconstruction_loss_sm,
            "reconstruction_loss_st": reconstruction_loss_st,
            "reconstruction_loss_sm_corr": reconstruction_loss_sm_corr,
            "reconstruction_loss_st_corr": reconstruction_loss_st_corr,            
            "kldiv_loss": kldiv_loss,
            #"kldiv_loss_st":ldiv_loss_st,
            #"kldiv_loss_sm": kldiv_loss_sm,
            "mmd_loss": mmd_loss
        }
        return H, Rs, loss_record

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
            kl_loss_reduction: str = 'sum',
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
            "reconstruction_loss_sm_corr": 0,
            "reconstruction_loss_st_corr": 0,
            "kldiv_loss": 0,
            "total_loss_sm": 0,
            "total_loss_st": 0,
            "mmd_loss": 0
        }
        epoch_reconstruction_loss_st_list = []
        epoch_reconstruction_loss_sm_list = []
        epoch_reconstruction_loss_st_corr_list = []
        epoch_reconstruction_loss_sm_corr_list = []       
        epoch_kldiv_loss_list = []
        epoch_kldiv_loss_st_list = []
        epoch_kldiv_loss_sm_list = []
        epoch_total_loss_list = []
        epoch_mmd_loss_list = []
        
        #epoch_sm_gate_logits_list = []
        
        for epoch in range(1, max_epoch+1):
            self._trained = True
            pbar.desc = "Epoch {}".format(epoch)
            epoch_total_loss = 0
            epoch_reconstruction_loss_sm = 0
            epoch_reconstruction_loss_st = 0
            epoch_reconstruction_loss_sm_corr = 0
            epoch_reconstruction_loss_st_corr = 0  
            epoch_kldiv_loss = 0
            epoch_kldiv_loss_st = 0
            epoch_kldiv_loss_sm = 0
            epoch_mmd_loss = 0
            
            #epoch_sm_gate_logits = []
            
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
                
                H, Rs, L = self.forward(
                    X,
                    batch_index=batch_index,
                    reduction=reconstruction_reduction,
                )
                
                #for R in Rs:
                #    epoch_sm_gate_logits.append(
                #        R['px_sm_dropout'].detach().cpu().numpy()
                #    )
                
                reconstruction_loss_st = reconstruction_st_weight * L['reconstruction_loss_st']
                reconstruction_loss_sm = reconstruction_sm_weight * L['reconstruction_loss_sm']
                reconstruction_loss_st_corr = reconstruction_st_weight * L['reconstruction_loss_st_corr']
                reconstruction_loss_sm_corr = reconstruction_sm_weight * L['reconstruction_loss_sm_corr']
                kldiv_loss = L['kldiv_loss']
                # kldiv_loss_st = kl_weight * L['kldiv_loss_st']
                # kldiv_loss_sm = kl_weight * L['kldiv_loss_sm'] 
                mmd_loss = L['mmd_loss']

                    #loss = 1*reconstruction_loss_sm.mean() + 0.5*reconstruction_loss_st.mean() + kldiv_loss.mean()

                avg_reconstruction_loss_st = reconstruction_loss_st.mean()  / n_per_batch
                avg_reconstruction_loss_sm = reconstruction_loss_sm.mean()  / n_per_batch
                avg_reconstruction_loss_st_corr = reconstruction_loss_st_corr.mean()  / n_per_batch
                avg_reconstruction_loss_sm_corr = reconstruction_loss_sm_corr.mean()  / n_per_batch
                avg_mmd_loss = mmd_loss.mean()  / n_per_batch
                                
                if kl_loss_reduction == 'mean':
                    avg_kldiv_loss = kldiv_loss.mean()  / n_per_batch
                    # avg_kldiv_loss_st = kldiv_loss_st.mean()  / n_per_batch
                    # avg_kldiv_loss_sm = kldiv_loss_sm.mean()  / n_per_batch
                    
                elif kl_loss_reduction == 'sum':
                    avg_kldiv_loss = kldiv_loss.sum()  / n_per_batch
                    # avg_kldiv_loss_st = kldiv_loss_st.sum()  / n_per_batch
                    # avg_kldiv_loss_sm = kldiv_loss_sm.sum()  / n_per_batch
                    
                loss = avg_reconstruction_loss_sm + \
                    avg_reconstruction_loss_st + \
                    avg_reconstruction_loss_sm_corr + \
                    avg_reconstruction_loss_st_corr + \
                    (avg_kldiv_loss * kl_weight) + \
                    avg_mmd_loss #+ avg_kldiv_loss_sm + avg_kldiv_loss_st

                epoch_reconstruction_loss_sm += avg_reconstruction_loss_sm.item()
                epoch_reconstruction_loss_st += avg_reconstruction_loss_st.item()
                epoch_reconstruction_loss_sm_corr += avg_reconstruction_loss_sm_corr.item()
                epoch_reconstruction_loss_st_corr += avg_reconstruction_loss_st_corr.item()
                
                epoch_mmd_loss += avg_mmd_loss.item()
                                
                epoch_kldiv_loss += avg_kldiv_loss.item()
                # epoch_kldiv_loss_st += avg_kldiv_loss_st.item()
                # epoch_kldiv_loss_sm += avg_kldiv_loss_sm.item()
                
                epoch_total_loss += loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            pbar.set_postfix({
                'reconst_sm': '{:.2e}'.format(epoch_reconstruction_loss_sm),
                'reconst_st': '{:.2e}'.format(epoch_reconstruction_loss_st),
                'reconst_sm_corr': '{:.2e}'.format(epoch_reconstruction_loss_sm_corr),
                'reconst_st_corr': '{:.2e}'.format(epoch_reconstruction_loss_st_corr),                    
                'kldiv': '{:.2e}'.format(epoch_kldiv_loss),
                # 'kldiv_st': '{:.2e}'.format(epoch_kldiv_loss_st),
                # 'kldiv_sm': '{:.2e}'.format(epoch_kldiv_loss_sm),
                'total_loss': '{:.2e}'.format(epoch_total_loss),
                'mmd_loss': '{:.2e}'.format(epoch_mmd_loss)
            }) 
            
            pbar.update(1)        
            epoch_reconstruction_loss_sm_list.append(epoch_reconstruction_loss_sm)
            epoch_reconstruction_loss_st_list.append(epoch_reconstruction_loss_st)
            epoch_reconstruction_loss_sm_corr_list.append(epoch_reconstruction_loss_sm_corr)
            epoch_reconstruction_loss_st_corr_list.append(epoch_reconstruction_loss_st_corr)            
            epoch_kldiv_loss_list.append(epoch_kldiv_loss)
            #epoch_kldiv_loss_st_list.append(epoch_kldiv_loss_st)
            #epoch_kldiv_loss_sm_list.append(epoch_kldiv_loss_sm)
            epoch_total_loss_list.append(epoch_total_loss)
            epoch_mmd_loss_list.append(epoch_mmd_loss)
            #epoch_sm_gate_logits = np.vstack(epoch_sm_gate_logits)
            #epoch_sm_gate_logits_list.append(epoch_sm_gate_logits)
            
            if n_epochs_kl_warmup:
                    kl_weight = min( kl_weight + kl_warmup_gradient, kl_weight_max)
            random_seed += 1
                 
        pbar.close()
        self.trained_state_dict = deepcopy(self.state_dict())  
          
        return dict(  
            epoch_reconstruction_loss_st_list=epoch_reconstruction_loss_st_list,
            epoch_reconstruction_loss_sm_list=epoch_reconstruction_loss_sm_list,
            epoch_reconstruction_loss_st_corr_list=epoch_reconstruction_loss_st_corr_list,
            epoch_reconstruction_loss_sm_corr_list=epoch_reconstruction_loss_sm_corr_list,
            epoch_kldiv_loss_list=epoch_kldiv_loss_list,
            #epoch_kldiv_loss_st_list=epoch_kldiv_loss_st_list,
            #epoch_kldiv_loss_sm_list=epoch_kldiv_loss_sm_list,
            #epoch_sm_gate_logits_list=epoch_sm_gate_logits_list,
            epoch_total_loss_list=epoch_total_loss_list,
            epoch_mmd_loss_list=epoch_mmd_loss_list
        )

    @torch.no_grad()
    def get_latent_embedding(
        self, 
        latent_key: Literal["z", "q_mu"] = "q_mu", 
        n_per_batch: int = 128,
        show_progress: bool = True,
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
        Zs_st = []
        Zs_sm = []
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
            Zs_st.append(H['st'][latent_key].detach().cpu().numpy())
            Zs_sm.append(H['sm'][latent_key].detach().cpu().numpy())
            if show_progress:
                pbar.update(1)
        if show_progress:
            pbar.close()
        return np.vstack(Zs)[self._shuffle_indices], np.vstack(Zs_st)[self._shuffle_indices], np.vstack(Zs_sm)[self._shuffle_indices]
 
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
        Zs_rate = []
        Zs_dropout = []
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
                    
            H,Rs,_ = self.forward(x, batch_index=batch_index)
            Zs.append(
                np.hstack([
                    Rs['latent']['px_sm_scale'].detach().cpu().numpy(),
                    Rs['latent']['px_rna_scale'].detach().cpu().numpy()
                ])
            )
            Zs_rate.append(
                np.hstack([
                    Rs['latent']['px_sm_rate'].detach().cpu().numpy(),
                    Rs['latent']['px_rna_rate'].detach().cpu().numpy()
                ])
            )
            Zs_dropout.append(
                np.hstack([
                    Rs['latent']['px_sm_dropout'].detach().cpu().numpy(),
                    Rs['latent']['px_rna_dropout'].detach().cpu().numpy()
                ])
            )        
            if show_progress:
                pbar.update(1)
        if show_progress:
            pbar.close()
        return np.vstack(Zs)[self._shuffle_indices], np.vstack(Zs_rate)[self._shuffle_indices], np.vstack(Zs_dropout)[self._shuffle_indices]

    @torch.no_grad()
    def get_normalized_expression_corr(
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
        Zs_rate = []
        Zs_dropout = []
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
                    
            H,Rs,_ = self.forward(x, batch_index=batch_index)
            Zs.append(
                np.hstack([
                    Rs['corr']['px_sm_scale'].detach().cpu().numpy(),
                    Rs['corr']['px_rna_scale'].detach().cpu().numpy()
                ])
            )
            Zs_rate.append(
                np.hstack([
                    Rs['corr']['px_sm_rate'].detach().cpu().numpy(),
                    Rs['corr']['px_rna_rate'].detach().cpu().numpy()
                ])
            )
            Zs_dropout.append(
                np.hstack([
                    Rs['corr']['px_sm_dropout'].detach().cpu().numpy(),
                    Rs['corr']['px_rna_dropout'].detach().cpu().numpy()
                ])
            )        
            if show_progress:
                pbar.update(1)
        if show_progress:
            pbar.close()
        return np.vstack(Zs)[self._shuffle_indices], np.vstack(Zs_rate)[self._shuffle_indices], np.vstack(Zs_dropout)[self._shuffle_indices]

    def get_modality_weight(
        self,
        n_per_batch: int = 128,
        show_progress: bool = True,
    ):
        self.eval()
        X = self.as_dataloader(batch_size=n_per_batch, shuffle=False)
        Bs_st = []
        Bs_sm = [] 
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
            b_st = H['st']['beta']
            b_sm = H['sm']['beta']
            Bs_st.append(b_st.detach().cpu().numpy())
            Bs_sm.append(b_sm.detach().cpu().numpy())
            #print(Bs_st)
            #print(Bs_sm)
            if show_progress:
                pbar.update(1)
        if show_progress:
            pbar.close()
        return np.hstack(Bs_st)[self._shuffle_indices], np.hstack(Bs_sm)[self._shuffle_indices]
    #FLATTEN(Bs_st), FLATTEN(Bs_sm)      