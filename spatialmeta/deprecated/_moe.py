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

class ConditionalVAEMOE(ConditionalVAE):
    def get_latent_from_z(self, z_st, z_sm):
        z = 0.5 * (z_st + z_sm)
        return z
    
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
        
        #print(q_sm.shape)
        #print(q_st.shape)
        mu_sm = self.z_mean_fc_single(q_sm)
        mu_st = self.z_mean_fc_single(q_st)
        q_mu = self.z_mean_fc(q)
        
        var_sm = torch.exp(self.z_var_fc_single(q_sm))
        var_st = torch.exp(self.z_var_fc_single(q_st))
        q_var = torch.exp(self.z_var_fc(q)) + eps
        
        z_sm = Normal(mu_sm, var_sm.sqrt()).rsample()
        z_st = Normal(mu_st, var_st.sqrt()).rsample()
        z = Normal(q_mu, q_var.sqrt()).rsample()
        
        H = dict(
            st = dict(
                q = q_st,
                q_mu = mu_st, 
                q_var = var_st,
                z = z_st
            ),
            sm = dict(
                q = q_sm,
                q_mu = mu_sm, 
                q_var = var_sm,
                z = z_sm
            ),
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
        z_st = H["st"]["z"]
        z_sm = H["sm"]["z"]
        z = H["z"]
        if batch_index is not None:
            z_st = torch.hstack([
                z_st, 
                batch_index
            ])
            z_sm = torch.hstack([
                z_sm, 
                batch_index
            ])
        
        R = []
        for z in [z_st, z_sm]:
            px = self.decoder(z)
            
            h = None
            px_rna_scale = self.px_rna_scale_decoder(px) 
            px_rna_rate = self.px_rna_rate_decoder(px)
            px_rna_dropout = self.px_rna_dropout_decoder(px)  ## In logits
            px_sm_scale = self.px_sm_scale_decoder(px)
            px_sm_rate = self.px_sm_rate_decoder(px)
            px_sm_dropout = self.px_sm_dropout_decoder(px)  ## In logits
            
            px_rna_scale = px_rna_scale * lib_size.unsqueeze(1)
            
            R.append(
                dict(
                    h = h,
                    px = px,
                    px_rna_scale = px_rna_scale,
                    px_rna_rate = px_rna_rate,
                    px_rna_dropout = px_rna_dropout,
                    px_sm_scale = px_sm_scale,
                    px_sm_rate = px_sm_rate,
                    px_sm_dropout = px_sm_dropout
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
                
        kldiv_loss = kld(Normal(mu_st, var_st.sqrt()),
                         Normal(mean_st, scale_st)).sum(dim = 1)+kld(Normal(mu_sm, var_sm.sqrt()),Normal(mean_sm, scale_sm)).sum(dim = 1)

        X_SM = X[:,self._type=="SM"]
        X_ST = X[:,self._type=="ST"]

        Rs=self.decode(H, X_ST.sum(1), batch_index)
        
        reconstruction_loss_st = 0
        reconstruction_loss_sm = 0
        
        for R in Rs:
                        
            if self.reconstruction_method_st == 'zinb':
                reconstruction_loss_st += LossFunction.zinb_reconstruction_loss(
                    X_ST,
                    mu = R['px_rna_scale'],
                    theta = R['px_rna_rate'].exp(), 
                    gate_logits = R['px_rna_dropout'],
                    reduction = reduction
                )
                
            elif self.reconstruction_method_st == 'zg':
                reconstruction_loss_st += LossFunction.zi_gaussian_reconstruction_loss(
                    X_ST,
                    mean=R['px_rna_scale'],
                    variance=R['px_rna_rate'].exp(),
                    gate_logits=R['px_rna_dropout'],
                    reduction=reduction
                )
            elif self.reconstruction_method_st == 'mse':
                reconstruction_loss_st += nn.functional.mse_loss(
                    R['px_rna_scale'],
                    X_ST,
                    reduction=reduction
                )
            if self.reconstruction_method_sm == 'zg':
                reconstruction_loss_sm += LossFunction.zi_gaussian_reconstruction_loss(
                    X_SM,
                    mean = R['px_sm_scale'],
                    variance = R['px_sm_rate'].exp(),
                    gate_logits = R['px_sm_dropout'],
                    reduction = reduction
                )
            elif self.reconstruction_method_sm == 'mse':
                reconstruction_loss_sm += nn.MSELoss(reduction='mean')(
                    R['px_sm_scale'],
                    X_SM,
                )
            elif self.reconstruction_method_sm == "g":
                reconstruction_loss_sm += LossFunction.gaussian_reconstruction_loss(
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
                kwargs['n_epochs_kl_warmup'] = 400

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
            n_epochs_kl_warmup: Union[int, None] = 400,
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
        
        #epoch_sm_gate_logits_list = []
        
        for epoch in range(1, max_epoch+1):
            self._trained = True
            pbar.desc = "Epoch {}".format(epoch)
            epoch_total_loss = 0
            epoch_reconstruction_loss_sm = 0
            epoch_reconstruction_loss_st = 0 
            epoch_kldiv_loss = 0
            
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
                kldiv_loss = kl_weight * L['kldiv_loss']    

                    #loss = 1*reconstruction_loss_sm.mean() + 0.5*reconstruction_loss_st.mean() + kldiv_loss.mean()

                avg_reconstruction_loss_st = reconstruction_loss_st.mean()  / n_per_batch
                avg_reconstruction_loss_sm = reconstruction_loss_sm.mean()  / n_per_batch
                if kl_loss_reduction == 'mean':
                    avg_kldiv_loss = kldiv_loss.mean()  / n_per_batch
                elif kl_loss_reduction == 'sum':
                    avg_kldiv_loss = kldiv_loss.sum()  / n_per_batch
                loss = avg_reconstruction_loss_sm + avg_reconstruction_loss_st + avg_kldiv_loss

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
            epoch_kldiv_loss_list=epoch_kldiv_loss_list,
            #epoch_sm_gate_logits_list=epoch_sm_gate_logits_list,
            epoch_total_loss_list=epoch_total_loss_list
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
            z_st = H['st'][latent_key]
            z_sm = H['sm'][latent_key]
            z = self.get_latent_from_z(
                z_st,
                z_sm
            )
            Zs.append(z.detach().cpu().numpy())
            if show_progress:
                pbar.update(1)
        if show_progress:
            pbar.close()
        return np.vstack(Zs)[self._shuffle_indices]

    def get_all_latent_embedding(
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
        
        :return: Tuple of numpy arrays containing the latent embeddings for ST and SM data.
        """
        self.eval()
        X = self.as_dataloader(batch_size=n_per_batch, shuffle=False)
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
            z_st = H['st'][latent_key]
            z_sm = H['sm'][latent_key]
            Zs_st.append(z_st.detach().cpu().numpy())
            Zs_sm.append(z_sm.detach().cpu().numpy())
            
        return np.vstack(Zs_st)[self._shuffle_indices], np.vstack(Zs_sm)[self._shuffle_indices]
    
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
        Zs_st = []
        Zs_sm = []
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
            Zs_st.append(
                np.hstack([
                    Rs[0]['px_sm_scale'].detach().cpu().numpy(),
                    Rs[0]['px_rna_scale'].detach().cpu().numpy()
                ])
            )
            Zs_sm.append(
                np.hstack([
                    Rs[1]['px_sm_scale'].detach().cpu().numpy(),
                    Rs[1]['px_rna_scale'].detach().cpu().numpy()
                ])
            )           
            if show_progress:
                pbar.update(1)
        if show_progress:
            pbar.close()
        return np.vstack(Zs_st)[self._shuffle_indices], np.vstack(Zs_sm)[self._shuffle_indices]


    def get_modality_contribution(
        self,
        method: Literal["cos", "ed"] = "ed",
    ):
        
        def angular_similarity(x, y):
            dot = np.dot(x, y)
            norms = np.linalg.norm(x) * np.linalg.norm(y)
            cos_similarity = dot / norms
            angular = (1 - np.arccos(cos_similarity) / np.pi)
            return angular
        
        st_latent, sm_latent = self.get_all_latent_embedding()
        
        st_latent = np.vstack(st_latent)
        st_latent = sc.AnnData(X=st_latent)
        
        sm_latent = np.vstack(sm_latent)
        sm_latent = sc.AnnData(X=sm_latent)
        
        joint_latent = sc.AnnData(X=(sm_latent.X + st_latent.X) * 0.5)
        
        if method == "cos":
            ang_sm = np.array([angular_similarity(x, y) for x, y in zip(joint_latent.X, sm_latent.X)])
            ang_st = np.array([angular_similarity(x, y) for x, y in zip(joint_latent.X, st_latent.X)])
            contribution_st_sm = ang_st - ang_sm + 0.5
            return contribution_st_sm
        elif method == "ed":
            from scipy.stats import pearsonr 
            cor_sm = np.array(
                [pearsonr(x,y)[0] for x,y in zip(joint_latent.X, sm_latent.X)]
            )
            cor_st = np.array(
                [pearsonr(x,y)[0] for x,y in zip(joint_latent.X, st_latent.X)]
            )
            return cor_sm, cor_st
        