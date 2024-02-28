import torch
from torch.nn import functional as F
from torch.distributions import Normal, kl_divergence as kldiv
import torch
import torch.nn as nn
import torch.nn.functional as F

from .distributions import ZeroInflatedNegativeBinomial
from .distributions import ZeroInflatedGaussian
from typing import Literal

class LossFunction:
    @staticmethod
    def mse(recon_x:   torch.tensor, 
            x:         torch.tensor, 
            reduction: str = "sum"):
        """
        The reconstruction error in the form of mse
        """
        return F.mse_loss(recon_x, x, reduction = reduction)
    

    @staticmethod
    def kld(q: torch.tensor,
            p: torch.tensor):
        kl_loss = kldiv(q.log(), p, reduction="sum", log_target=False)
        kl_loss.requires_grad_(True)
        return kl_loss

    @staticmethod
    def kl1(mu:  torch.tensor, 
            var: torch.tensor):
        return kldiv(Normal(mu, torch.sqrt(var)), Normal(0, 1)).sum(dim=1)

    @staticmethod
    def kl2(mu1: torch.tensor, 
           var1: torch.tensor, 
           mu2:  torch.tensor, 
           var2: torch.tensor):
        return kldiv(Normal(mu1, var1.sqrt()), Normal(mu2, var2.sqrt()))
    
    @staticmethod
    def zinb_reconstruction_loss(X:            torch.tensor, 
                                 total_counts: torch.tensor = None,
                                 logits:       torch.tensor = None,
                                 mu:           torch.tensor = None,
                                 theta:        torch.tensor = None,
                                 gate_logits:  torch.tensor = None,
                                 reduction:    str = "sum"):
        if ((total_counts == None) and (logits == None)):
            if ((mu == None) and (theta == None )):
                raise ValueError
            logits = (mu / theta).log()
            total_counts = theta + 1e-6
            znb = ZeroInflatedNegativeBinomial(
                total_count=total_counts, 
                logits=logits,
                gate_logits=gate_logits
            )   
        else: 
            znb = ZeroInflatedNegativeBinomial(
                total_count=total_counts, 
                logits=logits, 
                gate_logits=gate_logits
            )
        if reduction == "sum":
            reconst_loss = -znb.log_prob(X).sum(dim = 1)
        elif reduction == "mean":
            reconst_loss = -znb.log_prob(X).mean(dim = 1)
        return reconst_loss

    @staticmethod 
    def zi_gaussian_reconstruction_loss(
        X,
        mean,
        variance,
        gate_logits,
        reduction: Literal['sum','mean'] = 'sum'):
        zg = ZeroInflatedGaussian(
            mean=mean,
            variance=variance,
            gate_logits=gate_logits
        )   
        if reduction == "sum":
            reconst_loss = -zg.log_prob(X).sum(dim = 1)
        elif reduction == "mean":
            reconst_loss = -zg.log_prob(X).mean(dim = 1)
        return reconst_loss 
    
    @staticmethod 
    def gaussian_reconstruction_loss(
        X,
        mean,
        variance,
        reduction: Literal['sum','mean'] = 'sum'):
        g = Normal(
            mean,
            variance,
        )   
        if reduction == "sum":
            reconst_loss = -g.log_prob(X).sum(dim = 1)
        elif reduction == "mean":
            reconst_loss = -g.log_prob(X).mean(dim = 1)
        return reconst_loss     