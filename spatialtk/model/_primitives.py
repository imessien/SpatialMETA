import torch
from torch import nn, optim
from torch.nn import functional as F
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Union, Mapping, Iterable, List, Optional
from torch.distributions import Normal
from torch.utils.data import DataLoader
from torch import nn, einsum
import warnings

from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, repeat

from enum import Enum
import numpy as np

from functools import partial

from ..util.compat import Literal


def one_hot_(labels, return_dict = False):
    n_labels_ = np.unique(labels)
    n_labels = dict(zip(n_labels_, range(len(n_labels_))))
    if return_dict:
        return {"one_hot": F.one_hot( torch.tensor(list(map(lambda x: n_labels[x], labels)))), "labels": n_labels}
    return F.one_hot( torch.tensor(list(map(lambda x: n_labels[x], labels))))

def one_hot(index: torch.Tensor, n_cat: int) -> torch.Tensor:
    """One hot a tensor of categories."""
    onehot = torch.zeros(index.size(0), n_cat, device=index.device)
    onehot.scatter_(1, index.type(torch.long), 1)
    return onehot.type(torch.float32)



class FCLayer(nn.Module):
    """FCLayer Fully-Connected Layers for a neural network """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_cat_list: Iterable[int] = None,
        cat_dim: int = 8,
        cat_embedding: Literal["embedding", "onehot"] = "onehot",
        bias: bool = True,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        activation_fn: nn.Module = nn.ReLU,
        activation_dim: int = None,
        device: str = "cuda"
    ):
        super(FCLayer, self).__init__()
        if n_cat_list is not None:
            # Categories
            if not all(map(lambda x: x > 1, n_cat_list)):
                warnings.warn("category list contains values less than 1")
            self.n_category = len(n_cat_list)
            self._cat_dim = cat_dim
            self._cat_embedding = cat_embedding
            if cat_embedding == "embedding":
                self.cat_dimension = self.n_category * cat_dim # Total dimension of categories using embedding
            else:
                self.cat_dimension = sum(n_cat_list) # Total dimension of categories using one-hot
            self.n_cat_list = n_cat_list
            if cat_embedding == "embedding":
                self.cat_embedding = nn.ModuleList(
                    [nn.Embedding(n, cat_dim) for n in n_cat_list]
                )
            else: 
                self.cat_embedding = [
                    partial(one_hot, n_cat=n) for n in n_cat_list
                ]

        else:
            # No categories will be included
            self.n_category = 0
            self.n_cat_list = None
        
        self._fclayer = nn.Sequential(
                *list(filter(lambda x:x, 
                        [
                            nn.Linear(in_dim, out_dim, bias=bias) 
                            if self.n_category == 0 
                            else nn.Linear(in_dim + self.cat_dimension, out_dim, bias=bias),
                            nn.BatchNorm1d(out_dim, momentum=0.01, eps=0.001) if use_batch_norm else None,
                            nn.LayerNorm(out_dim, elementwise_affine=False) if use_layer_norm else None,
                            activation_fn(dim=activation_dim) if activation_dim else activation_fn() if activation_fn else None,
                            nn.Dropout(p = dropout_rate) if dropout_rate > 0 else None
                        ]
                    )
                )
            )
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device 

    def forward(self, X: torch.Tensor, cat_list: torch.Tensor =  None, return_category_embedding: bool = False):
        category_embedding = []
        if self.n_category > 0:
            if cat_list != None:
                if (len(cat_list) != self.n_category):
                    raise ValueError("Number of category list should be equal to {}".format(self.n_category))

                for i, (n_cat, cat) in enumerate(zip(self.n_cat_list, cat_list)):
                    assert(n_cat > 1)
                    if self._cat_embedding == "embedding":
                        category_embedding.append(self.cat_embedding[i](cat))
                    else: 
                        category_embedding.append(self.cat_embedding[i](cat.unsqueeze(0).T))
            else:
                if X.shape[1] != self.in_dim + self.n_category:
                    raise ValueError("Dimension of X should be equal to {} + {} but found {} if cat_list is provided".format(self.in_dim, self.n_category, X.shape[1]))
                cat_list = X[:, -self.n_category:].type(torch.LongTensor).T.to(self.device)
                for i, (n_cat, cat) in enumerate(zip(self.n_cat_list, cat_list)):
                    if self._cat_embedding == "embedding":
                        category_embedding.append(self.cat_embedding[i](cat))
                    else: 
                        category_embedding.append(self.cat_embedding[i](cat.unsqueeze(0).T))
               
            category_embedding = torch.hstack(category_embedding).to(self.device)
            if return_category_embedding:
                return self._fclayer(torch.hstack([X[:,:self.in_dim], category_embedding])), category_embedding
            else: 
                return self._fclayer(torch.hstack([X[:,:self.in_dim], category_embedding]))
        else:
            return self._fclayer(X)

    def to(self, device:str):
        super(FCLayer, self).to(device)
        self.device=device 
        return self


class FCDEF(Enum):
    ENCODER = 0x0
    DECODER = 0x1


class SAE(nn.Module):
    ''' Stacked Autoencoders. 
        Fitting includes stacked fitting and fine-tuning:
            Fine-tuning step removes the decoder and use clustering method
            to fine-tune the encoder.
        parameters:
            dim:    int 
            stacks: Iterable[int]
            n_cat_list: Iterable[int]
            cat_dim: int
    '''
    def __init__(
            self, 
            dim:int, 
            stacks:Iterable[int] = [512, 128, 64], 
            n_cat_list: Iterable[int] = None,
            cat_dim: int = 8,
            bias: bool = True,
            dropout_rate: float = 0.1,
            use_batch_norm: bool = True,
            use_layer_norm: bool = False,
            cat_embedding: Literal["embedding", "onthot"] = "onehot",
            activation_fn: nn.Module = nn.ReLU,
            encode_only: bool = False,
            decode_only: bool = False,
            device="cuda"
    ):
        super(SAE, self).__init__()
        fcargs = dict(
            bias=bias, 
            dropout_rate=dropout_rate, 
            use_batch_norm=use_batch_norm, 
            use_layer_norm=use_layer_norm,
            activation_fn=activation_fn,
            device=device,
            cat_embedding = cat_embedding
        )
        self.dim = dim
        self.num_layers = len(stacks)
        self.n_cat_list = n_cat_list
        self.cat_dim = cat_dim
        self.n_category = len(n_cat_list) if n_cat_list != None else 0
        self.stacks = stacks
        layers = [None] * len(stacks)
        self.n_layers = len(stacks)
        if (encode_only & decode_only):
            raise ValueError("SAE instance cannot be both encode and decode only")
        for i,j in enumerate(stacks):
            if i == 0:
                layers[i] = [FCLayer(dim, 
                             stacks[i], 
                             n_cat_list, 
                             cat_dim,
                             **fcargs)
                             if not decode_only 
                             else None, 
                             FCLayer(stacks[i], dim, **fcargs) 
                             if not encode_only 
                             else None]
            else:
                layers[i] = [FCLayer(stacks[i-1], stacks[i], **fcargs)
                             if not decode_only 
                             else None, 
                             FCLayer(stacks[i], stacks[i-1], **fcargs) 
                             if not encode_only 
                             else None ]
        layers = [i for s in layers for i in s]
        self.layers = nn.ModuleList(layers)
        self.device = device
        self.loss = []
        self.encode_only = encode_only
        self.decode_only = decode_only

    def get_layer(self, codec:str, layer:int):
        i = 0 if codec == FCDEF.ENCODER else 1
        return self.layers[layer * 2 + i]

    def encode(self, x: torch.Tensor):
        '''
        encode features in the nth layer 
        '''
        if self.decode_only:
            raise TypeError("This is an decoder-only SAE instance")
        h = None
        for i in range(self.num_layers):
            layer = self.get_layer(FCDEF.ENCODER, i)
            if i == self.num_layers - 1:
                if i == 0:
                    h = layer(x)
                else:
                    h = layer(h)
            else:
                if i == 0: 
                    h = layer(x)
                else:
                    h = layer(h)
        return h
    
    def decode(self, z: torch.Tensor):
        '''
        decode features in the nth layer 
        '''
        if self.encode_only:
            raise TypeError("This is an encoder-only SAE instance")
        h = None
        for i in range(self.num_layers):
            layer = self.get_layer(FCDEF.DECODER, self.num_layers - 1 - i)
            if i == self.num_layers - 1:
                if i == 0:
                    h = layer(z)
                else:
                    h = layer(h)
            else:
                if i == 0:
                    h = layer(z)
                else:
                    h = layer(h)
        return h

    def forward(self, x: torch.Tensor):
        z = self.encode(x.view(-1, self.dim))
        return self.decode(z), z