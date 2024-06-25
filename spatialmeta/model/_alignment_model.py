# ------------------------------------------------------------------------
# Copyright (c) 2024 STDI. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable STAlign (https://github.com/JEFworks-Lab/STalign)
# ------------------------------------------------------------------------

from torch.utils.data.sampler import SubsetRandomSampler
from itertools import zip_longest
import einops
from scipy.spatial import Delaunay
from sklearn.preprocessing import MaxAbsScaler
from sklearn.neighbors import NearestNeighbors

from scipy.spatial import Delaunay
import numpy as np
from scipy.spatial import KDTree

from torch.distributions import Normal, kl_divergence as kld
from torch.utils.data import DataLoader, SubsetRandomSampler
from typing import Union, Tuple, List, Callable, Iterable, Literal
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from itertools import zip_longest
import numpy as np
from scipy.spatial import Delaunay, KDTree
from einops import rearrange
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import issparse

from ._model import SAE, FCLayer, LossFunction, get_tqdm
from ..external.stalign.STalign import (
    rasterize_with_signal,
    extent_from_x,
    to_A,
    interp
)
from ._primitives import Linear
from ..external.stalign.STalign import *
from ..plotting._utils import get_spatial_image, get_spatial_scalefactors_dict
from ..plotting._plotting import create_subplots, create_fig
from ..util._classes import AnnDataST, AnnDataSM

def filter_spatial_outlier_spots(
    coordinates: np.ndarray, 
    subset: bool = True
) -> np.ndarray:
    '''
    Filter out spatial outlier spots using the nearest neighbor method.

    :param coordinates: np.ndarray. The spatial coordinates of the spots, should be [N, 2]
    :param subset: bool. If True, return the subset of the coordinates without the outliers. 
        If False, return the boolean mask of the outliers.

    :return: np.ndarray. The subset of the coordinates without the outliers or the boolean mask of the outliers.
    '''
    neighbors = NearestNeighbors(n_neighbors=100)
    neighbors.fit(coordinates)
    D,I=neighbors.kneighbors(coordinates)
    distance1 = D[:,1:].mean(1)
    Q3 = np.percentile(distance1 , 99)
    outliers1 = (distance1 > Q3)
    distance2 = D[:,1:].min(1)

    Q3 = np.percentile(distance2 , 75)
    Q1 = np.percentile(distance2 , 25)
    IQR = max(np.mean(distance2) * 0.01, Q3-Q1)
    outliers2 = (distance2 > np.mean(distance2) + IQR)
    outliers = outliers1 & outliers2
    if subset:
        return coordinates[~outliers]
    else:
        outliers


def point_alignment_error(pointsI: torch.Tensor, pointsJ: torch.Tensor) -> torch.Tensor:
    """
    Compute the point alignment error between two sets of points.


    :param pointsI: torch.Tensor. The first set of points, should be [N, 2]
    :param pointsJ: torch.Tensor. The second set of points, should be [N, 2]

    :return: torch.Tensor. The point alignment error.
    """

    tree = KDTree(pointsI.detach().cpu().numpy())
    _, I_indices = tree.query(pointsJ.detach().cpu().numpy())

    tree = KDTree(pointsJ.detach().cpu().numpy())
    _, J_indices = tree.query(pointsI.detach().cpu().numpy())

    error = (torch.mean(torch.pow(pointsI - pointsJ[J_indices], 2)) +
             torch.mean(torch.pow(pointsJ - pointsI[I_indices], 2))) / 2
    return error


def nearest_neighbor_torch(
    src: torch.Tensor, 
    dst: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find the nearest (Euclidean distance) neighbor in dst for each point in src.

    :param src: torch.Tensor. The source points, should be [N, 2]
    :param dst: torch.Tensor. The destination points, should be [N, 2]

    :return: Tuple[torch.Tensor, torch.Tensor]. The indices of the nearest neighbor in dst for each point in src.
    """
    dist = torch.cdist(src, dst)
    min_indices = torch.argmin(dist, dim=1)
    dist_2 = torch.cdist(dst, src)
    min_indices_2 = torch.argmin(dist_2, dim=1)
    return min_indices, min_indices_2


def point_alignment_error_2(
    pointsI: torch.Tensor, 
    pointsJ: torch.Tensor, 
    wi: float = 1, 
    wj: float = 1
) -> torch.Tensor:
    '''
    Compute the point alignment error between two sets of points.

    :param pointsI: torch.Tensor. The first set of points, should be [N, 2]
    :param pointsJ: torch.Tensor. The second set of points, should be [N, 2]
    :param wi: float. The weight for the first set of points.
    :param wj: float. The weight for the second set of points.

    :return: torch.Tensor. The point alignment error.
    '''
    indices1, indices2 = nearest_neighbor_torch(pointsI, pointsJ)
    error = (
        torch.mean(torch.sqrt(torch.pow(pointsI - pointsJ[indices1], 2)))
        * wi
        / pointsI.shape[0]
        + torch.mean(torch.sqrt(torch.pow(pointsJ - pointsI[indices2], 2)))
        * wj
        / pointsJ.shape[0]
    )
    return error


def toR(theta: torch.tensor) -> torch.tensor:
    """
    Construct a 2D rotation matrix from the given angle theta.
    
    :param theta: torch.tensor. A scalar value of angle of rotation.

    :return: torch.tensor. The 2D rotation matrix.
    """
    # Construct rotation matrix
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    # Construct the rotation matrix
    rot_matrix = torch.stack([cos_theta, -sin_theta, sin_theta, cos_theta], dim=0)
    rot_matrix = rot_matrix.view(2, 2).t()
    return rot_matrix


def alpha_shape(
    points: np.ndarray, 
    alpha: float, 
    only_outer: bool = True
):
    """
    Compute the alpha shape (concave hull) of a set of points.
    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border or also inner edges.

    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
        the indices in the points array.
    """
    assert points.shape[0] > 3, "Need at least four points"
    def add_edge(edges, i, j):
        """
        Add an edge between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it's not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))
    tri = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    return np.array([ (points[[i, j], 0],points[[i, j], 1]) for i,j in edges])


class AlignmentVAE(nn.Module):
    '''
    AlignmentVAE is a class for joint modeling and aligning two spatial transcriptomics datasets.

    :param adata_st: AnnDataST. The spatial transcriptomics dataset.
    :param adata_sm: AnnDataSM. The spatial transcriptomics dataset.
    :param hidden_stacks: List[int]. The hidden layer sizes of the encoder and decoder.
    :param n_latent: int. The latent dimension.
    :param bias: bool. If True, use bias in the linear layers.
    :param use_batch_norm: bool. If True, use batch normalization.
    :param use_layer_norm: bool. If True, use layer normalization.
    :param dropout_rate: float. The dropout rate.
    :param activation_fn: Callable. The activation function.
    :param device: Union[str, torch.device]. The device to run the model.
    :param batch_embedding: Literal["embedding", "onehot"]. The batch embedding method.
    :param encode_libsize: bool. If True, encode the library size.
    :param batch_hidden_dim: int. The batch hidden dimension.
    :param reconstruction_method_st: Literal['mse', 'zg', 'zinb']. The reconstruction method for the spatial transcriptomics dataset.
    :param reconstruction_method_sm: Literal['mse', 'zg','g']. The reconstruction method for the spatial metabolomics dataset.

    '''
    def __init__(
        self,
        *,
        adata_st: AnnDataST,
        adata_sm: AnnDataSM,
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

        super(AlignmentVAE, self).__init__()
        self.adata_st = adata_st
        self.adata_sm = adata_sm
        self.in_dim_st = adata_st.n_vars 
        self.in_dim_sm = adata_sm.n_vars

        self.hidden_stacks = hidden_stacks
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

        self.encoder_st = SAE(
            self.in_dim_st if not self.encode_libsize else self.in_dim_st + 1,
            stacks = hidden_stacks,
            # n_cat_list = [self.n_batch] if self.n_batch > 0 else None,
            cat_dim = batch_hidden_dim,
            cat_embedding = batch_embedding,
            encode_only = True,
            **self.fcargs
        )  

        self.encoder_sm = SAE(
            self.in_dim_sm,
            stacks = hidden_stacks,
            cat_dim = batch_hidden_dim,
            cat_embedding = batch_embedding,
            encode_only = True,
            **self.fcargs
        )

        # self.decoder_n_cat_list = decoder_n_cat_list
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
        self.z_mean_st_fc = nn.Linear(self.n_hidden, self.n_latent)
        self.z_var_st_fc = nn.Linear(self.n_hidden, self.n_latent)
        self.z_mean_sm_fc = nn.Linear(self.n_hidden, self.n_latent)
        self.z_var_sm_fc = nn.Linear(self.n_hidden, self.n_latent)

        self.px_rna_rate_decoder = nn.Linear(
            self.n_hidden, 
            self.in_dim_st
        )

        self.px_rna_scale_decoder = nn.Sequential(
            nn.Linear(self.n_hidden, self.in_dim_st),
            nn.Softmax(dim=-1)
        )

        self.px_rna_dropout_decoder = nn.Linear(
            self.n_hidden, 
            self.in_dim_st
        )

        self.px_sm_rate_decoder = nn.Linear(
            self.n_hidden, 
            self.in_dim_sm
        )

        self.px_sm_scale_decoder = nn.Linear(self.n_hidden, self.in_dim_sm)

        self.px_sm_dropout_decoder = nn.Linear(
            self.n_hidden,
            self.in_dim_sm
        )

        self.st_spatial_coord = adata_st.obsm['spatial']
        self.sm_spatial_coord = adata_sm.obsm['spatial']

        # Alignments

        self.to(self.device)

    def initialize_dataset(self):
        X_st = self.adata_st.X
        X_sm = self.adata_sm.X

        self.n_record_st = X_st.shape[0]
        self.n_record_sm = X_sm.shape[0]
        self._indices_st = np.arange(self.n_record_st)
        self._indices_sm = np.arange(self.n_record_sm)

        _dataset_st = list(X_st)
        _dataset_sm = list(X_sm)

        _shuffle_indices_st = list(range(len(_dataset_st)))
        _shuffle_indices_sm = list(range(len(_dataset_sm)))

        np.random.shuffle(_shuffle_indices_st)
        np.random.shuffle(_shuffle_indices_sm)

        self._dataset_st = np.array([_dataset_st[i] for i in _shuffle_indices_st])
        self._dataset_sm = np.array([_dataset_sm[i] for i in _shuffle_indices_sm])

        self._shuffle_indices_st = np.array(
            [x for x,_ in sorted(zip(range(len(_dataset_st)), _shuffle_indices_st),
            key=lambda x: x[1])]
        )
        self._shuffle_indices_sm = np.array(
            [x for x,_ in sorted(zip(range(len(_dataset_sm)), _shuffle_indices_sm),
            key=lambda x: x[1])]
        )


    def as_multi_dataloader(
        self,
        n_per_batch: int = 128,
        subset_indices_st: Union[torch.tensor, np.ndarray] = None,
        subset_indices_sm: Union[torch.tensor, np.ndarray] = None,
        train_test_split: bool = False,
        random_seed: bool = 42,
        validation_split: bool = .2,
        shuffle: bool = True,
    ):
        indices_st = self._indices_st if subset_indices_st is None else subset_indices_st
        indices_sm = self._indices_sm if subset_indices_sm is None else subset_indices_sm
        np.random.seed(random_seed)
        if shuffle:
            np.random.shuffle(indices_st)
            np.random.shuffle(indices_sm)
        if train_test_split:
            split_st = int(np.floor(validation_split * self.n_record_st))
            split_sm = int(np.floor(validation_split * self.n_record_sm))
            if split_st % n_per_batch == 1:
                n_per_batch += 1
            if split_sm % n_per_batch == 1:
                n_per_batch += 1
            train_indices_st, val_indices_st = indices_st[split_st:], indices_st[:split_st]
            train_indices_sm, val_indices_sm = indices_sm[split_sm:], indices_sm[:split_sm]

            train_sampler_st = SubsetRandomSampler(train_indices_st)
            train_sampler_sm = SubsetRandomSampler(train_indices_sm)

            return {
                "st": (
                    DataLoader(indices_st, n_per_batch, sampler=train_sampler_st),
                    DataLoader(indices_st, n_per_batch, sampler=train_sampler_st)
                ),
                "sm": (
                    DataLoader(indices_sm, n_per_batch, sampler=train_sampler_sm),
                    DataLoader(indices_sm, n_per_batch, sampler=train_sampler_sm)
                ),
            }
        else:
            return {
                "st": (
                    DataLoader(indices_st, n_per_batch),
                ),
                "sm": (
                    DataLoader(indices_sm, n_per_batch),
                ),
            }

    def encode(
        self, batch_data, eps: float = 1e-8
    ):
        st_dict, sm_dict = None, None
        if batch_data['st'] is not None:              
            q_st = self.encoder_st.encode(batch_data['st']['X'])
            q_mu_st = self.z_mean_st_fc(q_st)
            q_var_st = torch.exp(self.z_var_st_fc(q_st)) + eps
            z_st = Normal(q_mu_st, q_var_st.sqrt()).rsample()
            st_dict = dict(
                q = q_st,
                q_mu = q_mu_st,
                q_var = q_var_st,
                z = z_st
            )

        if batch_data['sm'] is not None:
            q_sm = self.encoder_sm.encode(batch_data['sm']['X'])   
            q_mu_sm = self.z_mean_sm_fc(q_sm)
            q_var_sm = torch.exp(self.z_var_sm_fc(q_sm)) + eps
            z_sm = Normal(q_mu_sm, q_var_sm.sqrt()).rsample()
            sm_dict = dict(
                q = q_sm,
                q_mu = q_mu_sm,
                q_var = q_var_sm,
                z = z_sm
            )

        H = dict(
            st = st_dict,
            sm = sm_dict
        )

        return H 

    def decode(
        self, H, lib_size:torch.tensor
    ):
        if H['st'] is not None:
            z_vq_st = H['st']['z']
            px_st = self.decoder(z_vq_st)
            px_rna_scale_orig = self.px_rna_scale_decoder(px_st) 
            px_rna_rate = self.px_rna_rate_decoder(px_st)
            px_rna_dropout = self.px_rna_dropout_decoder(px_st)  ## In logits
            px_rna_scale = px_rna_scale_orig * lib_size.unsqueeze(1)

        if H['sm'] is not None:
            z_vq_sm = H['sm']['z']
            px_sm = self.decoder(z_vq_sm)
            px_sm_scale = self.px_sm_scale_decoder(px_sm)
            px_sm_rate = self.px_sm_rate_decoder(px_sm)
            px_sm_dropout = self.px_sm_dropout_decoder(px_sm)  ## In logits

        R = dict(
            st = dict(
                px_rna_scale_orig = px_rna_scale_orig,
                px_rna_scale = px_rna_scale,
                px_rna_rate = px_rna_rate,
                px_rna_dropout = px_rna_dropout,
            ) if H['st'] is not None else None,
            sm = dict(
                px_sm_scale = px_sm_scale,
                px_sm_rate = px_sm_rate,
                px_sm_dropout = px_sm_dropout
            ) if H['sm'] is not None else None
        )
        return R

    def forward(
        self,
        batch_data,
        reduction: str = "sum", 
        **kwargs
    ):

        reconstruction_loss_sm = torch.tensor(0., device=self.device)
        reconstruction_loss_st = torch.tensor(0., device=self.device)
        kldiv_loss_st = torch.tensor(0., device=self.device)
        kldiv_loss_sm = torch.tensor(0., device=self.device)

        H=self.encode(batch_data)
        R=self.decode(H, batch_data['st']['lib_size'])

        if H['st'] is not None:
            q_mu_st = H['st']["q_mu"]
            q_var_st= H['st']["q_var"]

            mean_st = torch.zeros_like(q_mu_st)
            scale_st = torch.ones_like(q_var_st)
            kldiv_loss_st = kld(Normal(q_mu_st, q_var_st.sqrt()),
                            Normal(mean_st, scale_st)).sum(dim = 1)
            X_st = batch_data['st']['X']

            if self.reconstruction_method_st == 'zinb':
                reconstruction_loss_st = LossFunction.zinb_reconstruction_loss(
                    X_st,
                    mu = R['st']['px_rna_scale'],
                    theta = R['st']['px_rna_rate'].exp(), 
                    gate_logits = R['st']['px_rna_dropout'],
                    reduction = reduction
                )

            elif self.reconstruction_method_st == 'zg':
                reconstruction_loss_st = LossFunction.zi_gaussian_reconstruction_loss(
                    X_st,
                    mean=R['st']['px_rna_scale'],
                    variance=R['st']['px_rna_rate'].exp(),
                    gate_logits=R['st']['px_rna_dropout'],
                    reduction=reduction
                )
            elif self.reconstruction_method_st == 'mse':
                reconstruction_loss_st = nn.functional.mse_loss(
                    R['st']['px_rna_scale'],
                    X_st,
                    reduction=reduction
                )
        if H['sm'] is not None:
            q_mu_sm = H['sm']["q_mu"]
            q_var_sm = H['sm']["q_var"]
            mean_sm = torch.zeros_like(q_mu_sm)
            scale_sm = torch.ones_like(q_var_sm)
            kldiv_loss_sm = kld(Normal(q_mu_sm, q_var_sm.sqrt()),
                                Normal(mean_sm, scale_sm)).sum(dim = 1)
            X_sm = batch_data['sm']['X']

            if self.reconstruction_method_sm == 'zg':
                reconstruction_loss_sm = LossFunction.zi_gaussian_reconstruction_loss(
                    X_sm,
                    mean = R['sm']['px_sm_scale'],
                    variance = R['sm']['px_sm_rate'].exp(),
                    gate_logits = R['sm']['px_sm_dropout'],
                    reduction = reduction
                )
            elif self.reconstruction_method_sm == 'mse':
                reconstruction_loss_sm = nn.MSELoss(reduction='mean')(
                    R['sm']['px_sm_scale'],
                    X_sm,
                )
            elif self.reconstruction_method_sm == "g":
                reconstruction_loss_sm = LossFunction.gaussian_reconstruction_loss(
                    X_sm,
                    mean = R['sm']['px_sm_scale'],
                    variance = R['sm']['px_sm_rate'].exp(),
                    reduction = reduction
                )

        loss_record = {
            "reconstruction_loss_sm": reconstruction_loss_sm,
            "reconstruction_loss_st": reconstruction_loss_st,
            "kldiv_loss_st": kldiv_loss_st,
            "kldiv_loss_sm": kldiv_loss_sm,
        }
        return H, R, loss_record

    def fit_vae(
        self,
        max_epoch:int = 35, 
        n_per_batch:int = 128,
        kl_weight: float = 2.,
        n_epochs_kl_warmup: Union[int, None] = 400,
        optimizer_parameters: Iterable = None,
        weight_decay: float = 1e-6,
        lr: bool = 5e-5,
        random_seed: int = 12,
        validation_split: float = 0.1,
    ):
        """
        Fit the VAE model.

        :param max_epoch: int. The maximum number of epochs.
        :param n_per_batch: int. The number of samples per batch.
        :param kl_weight: float. The weight of the KL divergence loss.
        :param n_epochs_kl_warmup: Union[int, None]. The number of epochs for KL divergence warmup.
        :param optimizer_parameters: Iterable. The optimizer parameters.
        :param weight_decay: float. The weight decay.
        :param lr: float. The learning rate.
        :param random_seed: int. The random seed.
        :param validation_split: float. The validation split.

        :return: Dict. The loss record.
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
            "reconstruction_loss_sm": 0,
            "reconstruction_loss_st": 0,
            "kldiv_loss": 0,
        }
        epoch_reconstruction_loss_st_list = []
        epoch_reconstruction_loss_sm_list = []
        epoch_kldiv_loss_list = []

        for epoch in range(1, max_epoch+1):
            dataloaders = self.as_multi_dataloader(
                n_per_batch=n_per_batch,
                train_test_split = True,
                validation_split = validation_split,
                random_seed=random_seed,
            )

            X_train_st = dataloaders['st'][0]
            X_test_st = dataloaders['st'][1]
            X_train_sm = dataloaders['sm'][0]
            X_test_sm = dataloaders['sm'][1] 

            for b, (X_st, X_sm) in enumerate(zip(X_train_st, X_train_sm)):
                epoch_reconstruction_loss_sm = 0
                epoch_reconstruction_loss_st = 0
                epoch_kldiv_loss = 0
                epoch_total_loss = 0

                batch_data = self._prepare_batch(X_st, X_sm)
                H, R, L = self.forward(batch_data)
                reconstruction_loss_st = L['reconstruction_loss_st']
                reconstruction_loss_sm = L['reconstruction_loss_sm']
                kldiv_loss = kl_weight * (L['kldiv_loss_st'].mean() + L['kldiv_loss_sm'].mean())

                loss = reconstruction_loss_sm.mean() + reconstruction_loss_st.mean() + kldiv_loss

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

            if n_epochs_kl_warmup:
                kl_weight = min( kl_weight + kl_warmup_gradient, kl_weight_max)
            random_seed += 1

        pbar.close()
        self.trained_state_dict = deepcopy(self.state_dict())  

        return dict(  
            epoch_reconstruction_loss_st_list=epoch_reconstruction_loss_st_list,
            epoch_reconstruction_loss_sm_list=epoch_reconstruction_loss_sm_list,
            epoch_kldiv_loss_list=epoch_kldiv_loss_list,
        )

    @torch.no_grad()
    def get_latent_embedding(
        self, 
        latent_key: Literal["z", "q_mu"] = "q_mu", 
        n_per_batch: int = 128,
        show_progress: bool = True
    ) -> np.ndarray:
        self.eval()
        Zs_st = []
        Zs_sm = []
        
        dataloaders = self.as_multi_dataloader(
            subset_indices_st=list(range(self.n_record_st)),
            subset_indices_sm=list(range(self.n_record_sm)),
            n_per_batch=n_per_batch,
            train_test_split = False,
            shuffle = False
        )

        X_train_st = dataloaders['st'][0]
        X_train_sm = dataloaders['sm'][0]

        if show_progress:
            pbar = get_tqdm()(max(len(X_train_st), len(X_train_sm)), desc="Latent Embedding", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

        for b, (X_st, X_sm) in enumerate(zip_longest(X_train_st, X_train_sm)):
            batch_data = self._prepare_batch(X_st, X_sm)
            H = self.encode(batch_data)

            if H['st'] is not None:
                Zs_st.append(H['st'][latent_key].detach().cpu().numpy())
            if H['sm'] is not None:
                Zs_sm.append(H['sm'][latent_key].detach().cpu().numpy())
            if show_progress:
                pbar.update(1)
        if show_progress:
            pbar.close()
        return np.vstack(Zs_st)[self._shuffle_indices_st], np.vstack(Zs_sm)[self._shuffle_indices_sm]

    @torch.no_grad()
    def get_st_sm_reconstruction(
        self,
        n_per_batch: int = 128,
        reconstruction_key: Literal["px_scale", "px_rate", "px_dropout"] = "px_scale",
        show_progress: bool = True
    ):
        self.eval()
        Zs_st = []
        Zs_sm = []
        dataloaders = self.as_multi_dataloader(
            subset_indices_st=list(range(self.n_record_st)),
            subset_indices_sm=list(range(self.n_record_sm)),
            n_per_batch=n_per_batch,
            train_test_split = False,
            shuffle=False
        )

        X_train_st = dataloaders['st'][0]
        X_train_sm = dataloaders['sm'][0]

        if show_progress:
            pbar = get_tqdm()(max(len(X_train_st), len(X_train_sm)), desc="Latent Embedding", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

        for b, (X_st, X_sm) in enumerate(zip_longest(X_train_st, X_train_sm)):
            batch_data = self._prepare_batch(X_st, X_sm)
            H,R,L = self.forward(batch_data)

            if H['st'] is not None:
                Zs_st.append(
                    R['st'][reconstruction_key.split("_")[0] + '_rna_' + reconstruction_key.split("_")[1] + '_orig'].detach().cpu().numpy()
                )
            if H['sm'] is not None:
                Zs_sm.append(R['sm'][reconstruction_key.split("_")[0] + '_sm_' + reconstruction_key.split("_")[1]].detach().cpu().numpy())

            if show_progress:
                pbar.update(1)
        if show_progress:
            pbar.close()
        return np.vstack(Zs_st)[self._shuffle_indices_st], np.vstack(Zs_sm)[self._shuffle_indices_sm]

    def get_rasterized_feature_map(self) -> dict:
        adata_st = self.adata_st
        adata_sm = self.adata_sm
        J, scale = get_spatial_image(adata_st)
        x_st, y_st = self.st_spatial_coord[:, 0] * scale, self.st_spatial_coord[:, 1] * scale
        x_sm, y_sm = self.sm_spatial_coord[:, 0], self.sm_spatial_coord[:, 1]
        z_st, z_sm = self.get_latent_embedding(latent_key="q_mu")
        output_sm = rasterize_with_signal(x_sm, y_sm, z_sm, dx=0.4, blur=0.01)
        xJ = [
            torch.arange(J.shape[1], device=self.device),
            torch.arange(J.shape[0], device=self.device),
        ]
        xI = [
            torch.tensor(output_sm[0], device=self.device),
            torch.tensor(output_sm[1], device=self.device),
        ]
        I = (
            torch.nn.functional.normalize(
                torch.tensor(output_sm[2], dtype=torch.float32, device=self.device),
                p=2.0,
                dim=0,
            )
            * 100
        )

        pointsI = torch.tensor(np.vstack([x_sm, y_sm]).T, dtype=torch.float32)
        pointsJ = torch.tensor(np.vstack([x_st, y_st]).T, dtype=torch.float32)
        maskI = torch.from_numpy(self.inside_delaunay_mask([xI[0].detach().cpu().numpy(),xI[1].detach().cpu().numpy()], pointsI)).to(self.device)
        maskJ = torch.from_numpy(self.inside_delaunay_mask([xJ[0].detach().cpu().numpy(),xJ[1].detach().cpu().numpy()], pointsJ)).to(self.device)
        pointsI = pointsI.to(self.device)
        pointsJ = pointsJ.to(self.device)
        J = J / 255
        J = torch.tensor( np.dot(J[...,:3], [0.2989, 0.5870, 0.1140]) ) # Gray scale
        J = J.unsqueeze(0)
        J = J.to(torch.float32).to(self.device)

        coordinates = np.indices(np.array(J.shape)[[2,1]]).transpose(2,1,0)
        coordinates_flat = einops.rearrange(coordinates, 'a b c -> (a b) c')

        neighbors = NearestNeighbors(n_neighbors=1)
        neighbors.fit(self.st_spatial_coord * scale)
        distances, indices = neighbors.kneighbors(coordinates_flat)

        G = einops.rearrange(z_st[indices.flatten()], '(w h) c-> h w c', w = J.shape[-2], h = J.shape[-1])
        maskJ = einops.rearrange(distances>10,'(w h) c-> h w c', w = J.shape[-2], h = J.shape[-1])[:,:,0]
        G[maskJ]=0
        maskJ = torch.from_numpy(maskJ).to(self.device)

        G = torch.from_numpy(G).to(self.device)
        G = einops.rearrange(G, 'w h c -> c h w')

        fig, axes = create_subplots(2,2,figsize=(10,10))
        axes=axes.flatten()
        axes[0].imshow(I.mean(0).detach().cpu().numpy())
        axes[0].set_title('SM rasterized latent feature')

        axes[1].scatter(
            adata_sm.obsm['spatial'][:,0],-adata_sm.obsm['spatial'][:,1],
            c=z_sm.mean(-1),
            s=0.7
        )
        axes[1].set_title('SM latent feature')

        axes[2].imshow(J.mean(0).detach().cpu().numpy())
        axes[2].set_title('ST spatial image')
        axes[3].imshow(G.detach().cpu().numpy().mean(0))
        axes[3].set_title('ST latent feature')

        return dict(
            data = dict(
                I = I, # intend to be the SM feature in the image space
                J = J, # intend to be the spatial image of the spatial image
                G = G, # intend to be the ST feature in the image space as J
                maskI = maskI, # mask for the SM feature
                maskJ = maskJ, # mask for the ST feature
                pointsI = pointsI, # spatial coordinates for the SM feature
                pointsJ = pointsJ, # spatial coordinates for the ST feature,
                z_st = z_st,
                z_sm = z_sm,
                x_st = x_st,
                y_st = y_st,
                x_sm = x_sm,
                y_sm = y_sm,
                scale = scale,
                xI = xI,
                xJ = xJ,
            ),
            fig = dict(
                feature = fig
            )
        )

    def random_sample_inside_image_spot(
        self, 
        data, 
        threshold1: float = 0.65, 
        threshold2: float = 0.2
    ):
        J = data['data']['J']
        maskJ = data['data']['maskJ']
        x_sm, x_st = data['data']['x_sm'], data['data']['x_st']
        y_sm, y_st = data['data']['y_sm'], data['data']['y_st']
        scale = data['data']['scale']
        z_st, z_sm = data['data']['z_st'], data['data']['z_sm']
        pointsI = data['data']['pointsI']
        pointsJ = data['data']['pointsJ']

        adata_st = self.adata_st
        Jtrue = (J[0] <= J[0][maskJ.T].mean() * threshold1).detach().cpu().numpy()
        fig,axes=create_subplots(1,3, figsize=(10,5))
        fig.set_size_inches=(15,5)
        axes[2].imshow(Jtrue)
        w,h = J.shape[1], J.shape[2]
        spot_diamter_in_pixel = (get_spatial_scalefactors_dict(adata_st)['spot_diameter_fullres'] * scale)
        spot_gap_in_pixel = spot_diamter_in_pixel * (110 / 65)
        ws = np.round(w / spot_gap_in_pixel)
        hs = np.round(h / spot_gap_in_pixel)
        xv,yv=np.meshgrid(np.arange(ws) * spot_gap_in_pixel, np.arange(hs) * spot_gap_in_pixel)
        for i in range(0,yv.shape[1],2):
            yv[:,i] += spot_gap_in_pixel / 2
        xv,yv=xv.flatten(),yv.flatten()
        spot_coordinate = np.array([xv,yv]).T

        import einops
        from sklearn.neighbors import NearestNeighbors
        coordinates = np.indices((
            w,h
        )).transpose(1, 2, 0)
        coordinates_flat = einops.rearrange(coordinates, 'a b c -> (b a) c')

        neighbors = NearestNeighbors(n_neighbors=1)
        neighbors.fit(spot_coordinate)
        distance, indices = neighbors.kneighbors(coordinates_flat)
        mask_d = (distance.min(1) < spot_diamter_in_pixel / 2).reshape((
            h, w
        )).astype(np.uint8)
        indices_mapping = {k:[] for k in range(spot_coordinate.shape[0])}
        for i,j,k in zip(indices.flatten(),mask_d.flatten(),coordinates_flat):
            if j:
                indices_mapping[i].append(k)
        indices_mapping = {k:np.vstack(v) for k,v in indices_mapping.items()}
        spot_coordinate = spot_coordinate[[Jtrue[None,None,:,:][:,:,v[:,0],v[:,1]].mean() > threshold2 for v in indices_mapping.values()]]
        spot_coordinate = spot_coordinate[:,[1,0]]
        spot_coordinate = filter_spatial_outlier_spots(spot_coordinate)
        spot_coordinate = torch.tensor(spot_coordinate,dtype=torch.float32).to(self.device)
        initial_scale = (spot_coordinate[:,0].max() - spot_coordinate[:,1].min()) / (x_sm.max() - x_sm.min())


        pointsI_edge = torch.from_numpy(
            alpha_shape(pointsI.detach().cpu().numpy(), alpha=1)
        ).to(pointsI.dtype).to(pointsI.device)
        pointsI_edge = einops.rearrange(pointsI_edge, 'a b c -> (a c) b')
        spot_coordinate_edge = torch.from_numpy(
            alpha_shape(spot_coordinate.detach().cpu().numpy(), alpha=100)
        ).to(spot_coordinate.dtype).to(spot_coordinate.device)
        spot_coordinate_edge = einops.rearrange(spot_coordinate_edge, 'a b c -> (a c) b')

        axes[0].imshow(J[0].detach().cpu().numpy())
        axes[0].set_title('ST spatial image')

        axes[1].imshow(J[0].detach().cpu().numpy())
        axes[0].scatter(
            adata_st.obsm['spatial'][:,0] * scale,
            adata_st.obsm['spatial'][:,1] * scale,
            s=3,
            lw=0,
            c=z_st.mean(-1),
            cmap='Reds',
            marker='s'
        )
        axes[0].imshow(maskJ.T.detach().cpu().numpy(), alpha=0.3)

        axes[1].scatter(
            spot_coordinate[:,0].detach().cpu().numpy(),spot_coordinate[:,1].detach().cpu().numpy(), s=4,
            lw=0,
            c='white'
        )
        axes[1].set_title('Randomly sampled inside-tissue spots')
        
        data['fig']['spot_coordinate'] = fig
        data['data']['spot_coordinate'] = spot_coordinate
        data['data']['initial_scale'] = initial_scale
        data['data']['spot_coordinate_edge'] = spot_coordinate_edge
        data['data']['pointsI_edge'] = pointsI_edge
        return data 

    def fit_alignment(
        self,
        data: dict,
        initial_scale: bool = None,
        a: float = 50.0,
        p: float = 2.0,
        expand: float = 2.0,
        nt: float =3,
        niter: int =500,
        diffeo_start: float =0,
        epV: float =2e-1,
        sigmaM: float =1.0,
        sigmaR: float =5e5,
        align_st_feature: bool = False,
        debug_path: float =None
    ):
        """
        Fit the alignment model for spatial transcriptomics (ST) 
        features and spatial metabolomics (SM) features based on
        gradient descent based iterative closest point.

        :param data: dict. The data dictionary.
        :param initial_scale: bool. The initial scale for the image
        :param a: float. Smoothness scale of velocity field.
        :param p: float. Power of Laplacian in velocity regularization.
        :param expand: float. The expansion factor.
        :param nt: float. Number of timesteps for integrating velocity field.
        :param niter: int. The number of iterations.
        :param diffeo_start: float. The starting step of diffeomorphism.
        :param epV: float. Gradient descent step size for velocity field. The default value was set to a small value 
                           (2e-1) to avoid divergence, compared to the original implementation so the user may need 
                            to adjust this value to allow velocity field to converge.
        :param sigmaM: float. Standard deviation of image matching term for Gaussian mixture modeling in cost function. 
        :param sigmaR: float. Standard deviation of regularization term for Gaussian mixture modeling in cost function.
        :param align_st_feature: bool. Whether to align the ST feature with SM feature.
        :param debug_path: float. The temporary file path that save intermediate results in alignment.
        
        """
        if align_st_feature:
            return self._fit_alignment_with_st_feature_impl(
                data,
                initial_scale=initial_scale,
                a=a,
                p=p,
                expand=expand,
                nt=nt,
                niter=niter,
                diffeo_start=diffeo_start,
                epV=epV,
                sigmaM=sigmaM,
                sigmaR=sigmaR,
                debug_path = debug_path,
            )
        else:
            return self._fit_alignment_impl(
                data,
                initial_scale=initial_scale,
                a=a,
                p=p,
                expand=expand,
                nt=nt,
                niter=niter,
                diffeo_start=diffeo_start,
                epV=epV,
                sigmaM=sigmaM,
                sigmaR=sigmaR,
                debug_path = debug_path,
            )

    def _fit_alignment_impl(
        self,
        data: dict,
        initial_scale: bool = None,
        a: float = 50.0,
        p: float = 2.0,
        expand: float = 2.0,
        nt: float =3,
        niter: int =500,
        diffeo_start: float =0,
        epV: float =2e-1,
        sigmaM: float =1.0,
        sigmaR: float =5e5,
        debug_path: float =None
    ):
        self.eval()

        if debug_path is not None:
            os.system(f"rm -rf {debug_path}/*png")

        initial_scale = data['data']['initial_scale'] if initial_scale is None else initial_scale

        spot_coordinate = data['data']['spot_coordinate']
        spot_coordinate_edge = data['data']['spot_coordinate_edge']
        pointsI_edge = data['data']['pointsI_edge']
        pointsI = data['data']['pointsI']
        pointsJ = data['data']['pointsJ']
        spot_coordinate = spot_coordinate.to(self.device)
        spot_coordinate_edge = spot_coordinate_edge.to(self.device)
        pointsI_edge = pointsI_edge.to(self.device)
        pointsI = pointsI.to(self.device)
        pointsJ = pointsJ.to(self.device)
        I = data['data']['I']
        J = data['data']['J']
        G = data['data']['G']
        maskI = data['data']['maskI']
        maskJ = data['data']['maskJ']
        x_sm = data['data']['x_sm']
        y_sm = data['data']['y_sm']
        x_st = data['data']['x_st']
        y_st = data['data']['y_st']
        scale = data['data']['scale']
        init_bias = torch.tensor(1.)
        xI = data['data']['xI']
        xJ = data['data']['xJ']


        latent_adaptor_last = Linear(I.shape[0],J.shape[0],init='final').to(self.device)
        latent_adaptor_last.bias = torch.nn.Parameter(torch.tensor([init_bias],device=self.device, requires_grad=True))
        latent_adaptor = nn.Sequential(
            Linear(I.shape[0], I.shape[0], init='normal'),
            nn.ReLU(),
            nn.LayerNorm(I.shape[0]),
            nn.Dropout(0.1),
            latent_adaptor_last
        ).to(self.device)
        latent_optimizer = torch.optim.AdamW(
            latent_adaptor.parameters(), lr=1e-3
        )

        theta = torch.tensor(0,device=self.device, dtype=torch.float32, requires_grad=True)
        # L = torch.eye(2, device=self.device, dtype=torch.float32, requires_grad=True)
        T = torch.zeros(2, device=self.device, dtype=torch.float32, requires_grad=True)
        S = torch.tensor([torch.log(torch.tensor(initial_scale)),torch.log(torch.tensor(initial_scale))], device=self.device, dtype=torch.float32, requires_grad=True)
        # scale = torch.tensor(1.0, device=self.device, dtype=torch.float32, requires_grad=True)
        minv = torch.as_tensor([x[0] for x in xI], device=self.device, dtype=torch.float32)
        maxv = torch.as_tensor([x[-1] for x in xI], device=self.device, dtype=torch.float32)
        minv, maxv = (minv + maxv) * 0.5 + 0.5 * torch.tensor(
            [-1.0, 1.0], device=self.device, dtype=torch.float32
        )[..., None] * (maxv - minv) * expand
        xv = [
            torch.arange(m, M, a * 0.5, device=self.device, dtype=torch.float32)
            for m, M in zip(minv, maxv)
        ]
        XV = torch.stack(torch.meshgrid(xv), -1)
        v = torch.zeros(
            (nt, XV.shape[0], XV.shape[1], XV.shape[2]),
            device=self.device,
            dtype=torch.float32,
            requires_grad=True,
        )

        dv = torch.as_tensor([x[1] - x[0] for x in xv], device=self.device, dtype=torch.float32)
        fv = [
            torch.arange(n, device=self.device, dtype=torch.float32) / n / d
            for n, d in zip(XV.shape, dv)
        ]
        FV = torch.stack(torch.meshgrid(fv), -1)
        LL = (
            1.0
            + 2.0 * a**2 * torch.sum((1.0 - torch.cos(2.0 * np.pi * FV * dv)) / dv**2, -1)
        ) ** (p * 2.0)
        K = 1.0 / LL
        DV = torch.prod(dv)
        WM = torch.ones(J[0].shape, dtype=J.dtype, device=J.device) * 0.5
        WB = torch.ones(J[0].shape, dtype=J.dtype, device=J.device) * 0.4
        WA = torch.ones(J[0].shape, dtype=J.dtype, device=J.device) * 0.1  
        xI = [torch.tensor(x, device=self.device, dtype=torch.float32) for x in xI]
        xJ = [torch.tensor(x, device=self.device, dtype=torch.float32) for x in xJ]
        XI = torch.stack(torch.meshgrid(*xI, indexing="ij"), -1)
        XJ = torch.stack(torch.meshgrid(*xJ, indexing="ij"), -1)
        dJ = [x[1] - x[0] for x in xJ]
        extentJ = (
            xJ[1][0].item() - dJ[1].item() / 2.0,
            xJ[1][-1].item() + dJ[1].item() / 2.0,
            xJ[0][-1].item() + dJ[0].item() / 2.0,
            xJ[0][0].item() - dJ[0].item() / 2.0,
        )


        os.system(f"rm {debug_path}" + '/*')
        fit_history = {}
        pbar = get_tqdm()(range(niter), desc="LLDDMM", bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
        for it in range(niter):
            if it % 10 == 0 and it > 0:
                XII = einops.rearrange(XI.detach().cpu().numpy(), 'w h c -> (h w) c')
                neighbors = NearestNeighbors(n_neighbors=1)
                neighbors.fit(XII)
                _, indices = neighbors.kneighbors(einops.rearrange(
                    Xs.detach().cpu().numpy(),
                    'w h c -> (w h) c'
                ))
                AI = einops.rearrange(
                    einops.rearrange(I, 'c w h -> (w h) c')[indices],
                    '(w h) 1 c -> c h w', w = wj, h = hj
                )
                for _ in range(100):
                    fAIt = latent_adaptor(einops.rearrange(AI.detach(), ' c w h -> h w c'))
                    ## TODO:
                    EM = torch.sum(einops.rearrange((
                        einops.rearrange(J.detach(), 'c w h -> h w c') - fAIt
                    ), 'w h c -> c w h') ** 2 * WM.T.detach() * ~maskJ.detach() ) * 10 / sigmaM**2
    
                    EM.backward()
                    latent_optimizer.step()
                    latent_optimizer.zero_grad()
            
            L = toR(theta)
            A = to_A(L, T)
            # Ai
            Li = torch.linalg.inv(L)
            wj, hj = XJ.shape[:2]
            pointsIt = torch.clone(pointsI).to(J.device)
            pointsIt_bias = pointsIt.mean(0) 
            Xs = (((einops.rearrange(XJ, 'w h c -> (w h) c') - pointsIt_bias) - A[:2, -1])  @ L)
            Xs = einops.rearrange(Xs, '(w h) c -> w h c', w = wj, h = hj)
            # now diffeo, not semilagrange here
            for t in range(nt - 1, -1, -1):
                Xs = (
                    Xs
                    + interp(xv, -v[t].permute(2, 0, 1), Xs.permute(2, 0, 1)).permute(
                        1, 2, 0
                    )
                    / nt
                )
            Xs /= (torch.e ** S) 
            Xs += pointsIt_bias
            # and points
            pointsIt_rot_bias = (pointsIt_bias @ L) - pointsIt_bias
            pointsIt -= pointsIt_bias
            pointsIt *= (torch.e ** S)
            if pointsIt.shape[0] > 0:
                for t in range(nt):
                    pointsIt += (
                        interp(xv, v[t].permute(2, 0, 1), pointsIt.T[..., None])[..., 0].T
                        / nt
                    )
                pointsIt = (A[:2, :2] @ pointsIt.T + A[:2, -1][..., None]).T
            pointsIt += pointsIt_bias
            pointsI_edget = torch.clone(pointsI_edge).to(J.device)
            pointsI_edget -= pointsIt_bias
            pointsI_edget *= (torch.e ** S)
            if pointsI_edget.shape[0] > 0:
                for t in range(nt):
                    pointsI_edget += (
                        interp(xv, v[t].permute(2, 0, 1), pointsI_edget.T[..., None])[..., 0].T
                        / nt
                    )
                pointsI_edget = (A[:2, :2] @ pointsI_edget.T + A[:2, -1][..., None]).T
            pointsI_edget += pointsIt_bias
            xIs = [
                xI[0].detach(), 
                xI[1].detach()
            ]
            EM = torch.tensor(0., device=self.device)
            if it > 0 and it % 10 == 0:   
                XII = einops.rearrange(XI.detach().cpu().numpy(), 'w h c -> (h w) c')
                neighbors = NearestNeighbors(n_neighbors=1)
                neighbors.fit(XII)
                _, indices = neighbors.kneighbors(einops.rearrange(
                    Xs.detach().cpu().numpy(),
                    'w h c -> (w h) c'
                ))
                AI = einops.rearrange(
                    einops.rearrange(I, 'c w h -> (w h) c')[indices],
                    '(w h) 1 c -> c h w', w = wj, h = hj
                )
                # transform the contrast
                B = torch.ones(
                    1 + AI.shape[0], AI.shape[1] * AI.shape[2], device=AI.device, dtype=AI.dtype
                )
                B[1 : AI.shape[0] + 1] = AI.reshape(AI.shape[0], -1)
                with torch.no_grad():
                    BB = B @ (B * WM.ravel()).T
                    BJ = B @ ((J * WM).reshape(J.shape[0], J.shape[1] * J.shape[2])).T
                    small = 0.1
                    coeffs = torch.linalg.solve(
                        BB + small * torch.eye(BB.shape[0], device=BB.device, dtype=BB.dtype),
                        BJ,
                    )
                fAI = ((B.T @ coeffs).T).reshape(J.shape)
                fAIt = latent_adaptor(einops.rearrange(AI, ' c w h -> h w c'))
                ## TODO:
                EM = torch.mean(einops.rearrange((
                    einops.rearrange(J,'c w h -> h w c') - fAIt
                ), 'w h c -> c w h') ** 2 * WM.T * ~maskJ ) * 10 / sigmaM**2

            ER = (
                torch.mean(
                    torch.sum(torch.abs(torch.fft.fftn(v, dim=(1, 2))) ** 2, dim=(0, -1))
                    * LL
                )
                * DV
                / 2.0
                / v.shape[1]
                / v.shape[2]
                / sigmaR**2
            ) 
            EP = point_alignment_error_2(
                pointsI_edget, spot_coordinate_edge
            ) * 10
            if it > 0 and it % 10 == 0:
                E = EP + ER + EM
            else:
                E = EP + ER
            tosave = [EM.item(), ER.item(), EP.item()]
            if debug_path is not None and it % 10 == 0:
                import matplotlib.pyplot as plt
                fig,axes=plt.subplots(2,2,figsize=(12,12))
                axes=axes.flatten()
                axes[0].scatter(pointsIt[:,0].detach().cpu().numpy(),pointsIt[:,1].detach().cpu().numpy(),c='r', s=1, label='I')
                axes[0].scatter(pointsJ[:,0].detach().cpu().numpy(),pointsJ[:,1].detach().cpu().numpy(),c='b', s=1, label='J')
                axes[0].set_title(f'point alignment error: {EP.item()}')
                axes[0].invert_yaxis()
                if it > 0 and it % 10 == 0:
                    axes[1].imshow(
                        einops.rearrange(fAIt, 'w h c -> h w c').mean(-1).detach().cpu().numpy(),
                        #vmin=J.min(),
                        #vmax=J.max()
                    )
                    axes[1].set_title(f'EM: {EM.item()}')
                axes[2].scatter(pointsIt[:,0].detach().cpu().numpy(),pointsIt[:,1].detach().cpu().numpy(),c='r', s=1, label='I')
                axes[2].scatter(pointsI_edget[:,0].detach().cpu().numpy(),pointsI_edget[:,1].detach().cpu().numpy(),c='orange', s=1, label='I')
                axes[2].scatter(spot_coordinate[:,0].detach().cpu().numpy(),spot_coordinate[:,1].detach().cpu().numpy(),c='black', s=1, label='J')
                axes[2].scatter(spot_coordinate_edge[:,0].detach().cpu().numpy(),spot_coordinate_edge[:,1].detach().cpu().numpy(),c='blue', s=1, label='J')
                axes[2].set_title(f'point alignment error: {EP.item()}')
                axes[2].invert_yaxis()
                axes[3].imshow(
                    J.mean(0).detach().cpu().numpy(),
                )
                axes[3].set_title("J")
                axes[1].set_xbound(axes[3].get_xbound())
                axes[1].set_ybound(axes[3].get_ybound())
                if debug_path is not None:
                    fig.savefig(
                        os.path.join(
                            debug_path,
                            f'point_alignment_{it}.png'
                        )
                    )
                    plt.close()
            postfix = dict(zip(['EM','ER','EP'], tosave))
            postfix['angle'] = theta.item()
            pbar.set_postfix(postfix)
            E.backward()
            with torch.no_grad():
                if it % 10 == 0 and it > 0:
                    latent_optimizer.step()
                    latent_optimizer.zero_grad()
                    
                if not torch.isnan(theta.grad).any() or not torch.isnan(T.grad).any():
                    theta -= (5e-2 / (1.0 + (it >= diffeo_start) * 9)) * theta.grad
                if not torch.isnan(T.grad).any():
                    T -= 2000 * T.grad
                if not torch.isnan(S.grad).any():
                    S -= S.grad * 1e-4
                T.grad.zero_()
                # theta += torch.rand(theta.shape).to(S.device) * 0.05
                theta.grad.zero_()
                # S += torch.rand(S.shape).to(S.device) * 0.05
                S.grad.zero_()
                # v grad
                vgrad = v.grad
                if not torch.isnan(vgrad).any():
                    # smooth it
                    vgrad = torch.fft.ifftn(
                        torch.fft.fftn(vgrad, dim=(1, 2)) * K[..., None], dim=(1, 2)
                    ).real
                    if it >= diffeo_start:
                        v -= vgrad * epV
                    v.grad.zero_()

            A = to_A(L, T)
            fit_history[it] = dict(
                loss = tosave,
                theta = theta.item(),
                T = T.detach().cpu().numpy(),
                S = S.detach().cpu().numpy(),
                v = v.detach().cpu().numpy(),
                A = A.detach().cpu().numpy(),
                L = L.detach().cpu().numpy(),
                pointsIt = pointsIt.detach().cpu().numpy(),
            )
            pbar.update()
        pbar.close()
        return fit_history

    def _fit_alignment_with_st_feature_impl(
        self,
        data,
        initial_scale=None,
        a=50.0,
        p=2.0,
        expand=2.0,
        nt=3,
        niter=500,
        diffeo_start=0,
        epV=2e-1,
        sigmaM=1.0,
        sigmaR=5e5,
        debug_path=None
    ):
        self.eval()

        if debug_path is not None:
            os.system(f"rm -rf {debug_path}/*png")

        initial_scale = data['data']['initial_scale'] if initial_scale is None else initial_scale

        spot_coordinate = data['data']['spot_coordinate']
        spot_coordinate_edge = data['data']['spot_coordinate_edge']
        pointsI_edge = data['data']['pointsI_edge']
        pointsI = data['data']['pointsI']
        pointsJ = data['data']['pointsJ']
        spot_coordinate = spot_coordinate.to(self.device)
        spot_coordinate_edge = spot_coordinate_edge.to(self.device)
        pointsI_edge = pointsI_edge.to(self.device)
        pointsI = pointsI.to(self.device)
        pointsJ = pointsJ.to(self.device)
        I = data['data']['I']
        J = data['data']['J']
        G = data['data']['G']
        maskI = data['data']['maskI']
        maskJ = data['data']['maskJ']
        x_sm = data['data']['x_sm']
        y_sm = data['data']['y_sm']
        x_st = data['data']['x_st']
        y_st = data['data']['y_st']
        scale = data['data']['scale']
        init_bias = torch.tensor(1.)
        xI = data['data']['xI']
        xJ = data['data']['xJ']

        # specific for the ST feature
        J = torch.cat([G,J], dim=0)


        latent_adaptor_1_last = Linear(I.shape[0],J.shape[0],init='final').to(self.device)
        latent_adaptor_1_last.bias = torch.nn.Parameter(torch.tensor([init_bias],device=self.device, requires_grad=True))
        latent_adaptor_1 = nn.Sequential(
            Linear(I.shape[0], I.shape[0], init='normal'),
            nn.ReLU(),
            nn.LayerNorm(I.shape[0]),
            nn.Dropout(0.1),
            latent_adaptor_1_last
        ).to(self.device)
        latent_adaptor_1_rev = nn.Linear(J.shape[0],I.shape[0]).to(self.device)
        
        # specific for the ST feature
        latent_adaptor_2_last = Linear(J.shape[0],J.shape[0],init='final').to(self.device)
        latent_adaptor_2_last.bias = torch.nn.Parameter(torch.tensor([init_bias],device=self.device, requires_grad=True))
        latent_adaptor_2 = nn.Sequential(
            Linear(J.shape[0], J.shape[0], init='normal'),
            nn.ReLU(),
            nn.LayerNorm(J.shape[0]),
            nn.Dropout(0.1),
            latent_adaptor_2_last
        ).to(self.device)
        latent_adaptor_2_rev = nn.Linear(J.shape[0],J.shape[0]).to(self.device)

        from itertools import chain
        latent_optimizer = torch.optim.AdamW(chain(
            latent_adaptor_1.parameters(),
            latent_adaptor_1_rev.parameters(),
            latent_adaptor_2.parameters(),
            latent_adaptor_2_rev.parameters(), 
        ), lr=1e-3)

        theta = torch.tensor(0,device=self.device, dtype=torch.float32, requires_grad=True)
        # L = torch.eye(2, device=self.device, dtype=torch.float32, requires_grad=True)
        T = torch.zeros(2, device=self.device, dtype=torch.float32, requires_grad=True)
        S = torch.tensor([torch.log(torch.tensor(initial_scale)),torch.log(torch.tensor(initial_scale))], device=self.device, dtype=torch.float32, requires_grad=True)
        # scale = torch.tensor(1.0, device=self.device, dtype=torch.float32, requires_grad=True)
        minv = torch.as_tensor([x[0] for x in xI], device=self.device, dtype=torch.float32)
        maxv = torch.as_tensor([x[-1] for x in xI], device=self.device, dtype=torch.float32)
        minv, maxv = (minv + maxv) * 0.5 + 0.5 * torch.tensor(
            [-1.0, 1.0], device=self.device, dtype=torch.float32
        )[..., None] * (maxv - minv) * expand
        xv = [
            torch.arange(m, M, a * 0.5, device=self.device, dtype=torch.float32)
            for m, M in zip(minv, maxv)
        ]
        XV = torch.stack(torch.meshgrid(xv), -1)
        v = torch.zeros(
            (nt, XV.shape[0], XV.shape[1], XV.shape[2]),
            device=self.device,
            dtype=torch.float32,
            requires_grad=True,
        )
        extentV = extent_from_x(xv)
        dv = torch.as_tensor([x[1] - x[0] for x in xv], device=self.device, dtype=torch.float32)
        fv = [
            torch.arange(n, device=self.device, dtype=torch.float32) / n / d
            for n, d in zip(XV.shape, dv)
        ]
        extentF = extent_from_x(fv)
        FV = torch.stack(torch.meshgrid(fv), -1)
        LL = (
            1.0
            + 2.0 * a**2 * torch.sum((1.0 - torch.cos(2.0 * np.pi * FV * dv)) / dv**2, -1)
        ) ** (p * 2.0)
        K = 1.0 / LL
        DV = torch.prod(dv)
        Ki = torch.fft.ifftn(K).realWM = torch.ones(J[0].shape, dtype=J.dtype, device=J.device) * 0.5
        WM = torch.ones(J[0].shape, dtype=J.dtype, device=J.device) * 0.5
        WB = torch.ones(J[0].shape, dtype=J.dtype, device=J.device) * 0.4
        WA = torch.ones(J[0].shape, dtype=J.dtype, device=J.device) * 0.1  
        xI = [torch.tensor(x, device=self.device, dtype=torch.float32) for x in xI]
        xJ = [torch.tensor(x, device=self.device, dtype=torch.float32) for x in xJ]
        XI = torch.stack(torch.meshgrid(*xI, indexing="ij"), -1)
        XJ = torch.stack(torch.meshgrid(*xJ, indexing="ij"), -1)
        dJ = [x[1] - x[0] for x in xJ]
        extentJ = (
            xJ[1][0].item() - dJ[1].item() / 2.0,
            xJ[1][-1].item() + dJ[1].item() / 2.0,
            xJ[0][-1].item() + dJ[0].item() / 2.0,
            xJ[0][0].item() - dJ[0].item() / 2.0,
        )
        
        os.system(f"rm {debug_path}" + '/*')
        fit_history = {}
        pbar = get_tqdm()(range(niter), desc="LLDDMM", bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
        for it in range(niter):
            if it % 10 == 0 and it > 0:
                XII = einops.rearrange(XI.detach().cpu().numpy(), 'w h c -> (h w) c')
                neighbors = NearestNeighbors(n_neighbors=1)
                neighbors.fit(XII)
                _, indices = neighbors.kneighbors(einops.rearrange(
                    Xs.detach().cpu().numpy(),
                    'w h c -> (w h) c'
                ))
                AI = einops.rearrange(
                    einops.rearrange(I, 'c w h -> (w h) c')[indices],
                    '(w h) 1 c -> c h w', w = wj, h = hj
                )
                for _ in range(100):
                    fAIt = latent_adaptor_1(einops.rearrange(AI.detach(), ' c w h -> h w c'))
                    JI = einops.rearrange(J.detach(), 'c w h -> h w c')
                    JIt = latent_adaptor_2(JI)
                    fAItr = latent_adaptor_1_rev(fAIt)
                    JItr = latent_adaptor_2_rev(JIt)

                    ## TODO:
                    EM = torch.sum(einops.rearrange((
                       JIt  - fAIt
                    ), 'w h c -> c w h') ** 2 * WM.T.detach() * ~maskJ.detach() ) * 10 / sigmaM**2

                    EL = (torch.sum(einops.rearrange((
                       JI  - JItr
                    ), 'w h c -> c w h') ** 2 * WM.T.detach() * ~maskJ.detach() ) + torch.sum(einops.rearrange((
                       einops.rearrange(AI, 'c w h -> h w c')  - fAItr
                    ), 'w h c -> c w h') ** 2)) * 10 / sigmaM**2
                    
                    EML = EM + EL
                    EML.backward()

                    latent_optimizer.step()
                    latent_optimizer.zero_grad()
            
            L = toR(theta)
            A = to_A(L, T)
            # Ai
            Li = torch.linalg.inv(L)
            wj, hj = XJ.shape[:2]
            pointsIt = torch.clone(pointsI).to(J.device)
            pointsIt_bias = pointsIt.mean(0) 
            Xs = (((einops.rearrange(XJ, 'w h c -> (w h) c') - pointsIt_bias) - A[:2, -1])  @ L)
            Xs = einops.rearrange(Xs, '(w h) c -> w h c', w = wj, h = hj)
            # now diffeo, not semilagrange here
            for t in range(nt - 1, -1, -1):
                Xs = (
                    Xs
                    + interp(xv, -v[t].permute(2, 0, 1), Xs.permute(2, 0, 1)).permute(
                        1, 2, 0
                    )
                    / nt
                )
            Xs /= (torch.e ** S) 
            Xs += pointsIt_bias
            # and points
            pointsIt_rot_bias = (pointsIt_bias @ L) - pointsIt_bias
            pointsIt -= pointsIt_bias
            pointsIt *= (torch.e ** S)
            if pointsIt.shape[0] > 0:
                for t in range(nt):
                    pointsIt += (
                        interp(xv, v[t].permute(2, 0, 1), pointsIt.T[..., None])[..., 0].T
                        / nt
                    )
                pointsIt = (A[:2, :2] @ pointsIt.T + A[:2, -1][..., None]).T
            pointsIt += pointsIt_bias
            pointsI_edget = torch.clone(pointsI_edge).to(J.device)
            pointsI_edget -= pointsIt_bias
            pointsI_edget *= (torch.e ** S)
            if pointsI_edget.shape[0] > 0:
                for t in range(nt):
                    pointsI_edget += (
                        interp(xv, v[t].permute(2, 0, 1), pointsI_edget.T[..., None])[..., 0].T
                        / nt
                    )
                pointsI_edget = (A[:2, :2] @ pointsI_edget.T + A[:2, -1][..., None]).T
            pointsI_edget += pointsIt_bias
            xIs = [
                xI[0].detach(), 
                xI[1].detach()
            ]
            EM = torch.tensor(0., device=self.device)
            if it > 0 and it % 10 == 0:   
                XII = einops.rearrange(XI.detach().cpu().numpy(), 'w h c -> (h w) c')
                neighbors = NearestNeighbors(n_neighbors=1)
                neighbors.fit(XII)
                _, indices = neighbors.kneighbors(einops.rearrange(
                    Xs.detach().cpu().numpy(),
                    'w h c -> (w h) c'
                ))
                AI = einops.rearrange(
                    einops.rearrange(I, 'c w h -> (w h) c')[indices],
                    '(w h) 1 c -> c h w', w = wj, h = hj
                )
                # transform the contrast
                B = torch.ones(
                    1 + AI.shape[0], AI.shape[1] * AI.shape[2], device=AI.device, dtype=AI.dtype
                )
                B[1 : AI.shape[0] + 1] = AI.reshape(AI.shape[0], -1)
                with torch.no_grad():
                    BB = B @ (B * WM.ravel()).T
                    BJ = B @ ((J * WM).reshape(J.shape[0], J.shape[1] * J.shape[2])).T
                    small = 0.1
                    coeffs = torch.linalg.solve(
                        BB + small * torch.eye(BB.shape[0], device=BB.device, dtype=BB.dtype),
                        BJ,
                    )
                fAI = ((B.T @ coeffs).T).reshape(J.shape)
                fAIt = latent_adaptor_1(einops.rearrange(AI, ' c w h -> h w c'))
                JI = einops.rearrange(J.detach(), 'c w h -> h w c')
                JIt = latent_adaptor_2(JI)
                ## TODO:
                EM = torch.mean(einops.rearrange((
                    JIt - fAIt
                ), 'w h c -> c w h') ** 2 * WM.T * ~maskJ ) * 10 / sigmaM**2

            ER = (
                torch.mean(
                    torch.sum(torch.abs(torch.fft.fftn(v, dim=(1, 2))) ** 2, dim=(0, -1))
                    * LL
                )
                * DV
                / 2.0
                / v.shape[1]
                / v.shape[2]
                / sigmaR**2
            ) 
            EP = point_alignment_error_2(pointsI_edget, spot_coordinate_edge) * 10
            if it > 0 and it % 10 == 0:
                E = EP + ER + EM
            else:
                E = EP + ER
            tosave = [EM.item(), ER.item(), EP.item()]
            if debug_path is not None and it % 10 == 0 and it > 0:
                import matplotlib.pyplot as plt
                fig,axes=plt.subplots(2,2,figsize=(12,12))
                axes=axes.flatten()
                axes[0].scatter(pointsIt[:,0].detach().cpu().numpy(),pointsIt[:,1].detach().cpu().numpy(),c='r', s=1, label='I')
                axes[0].scatter(pointsJ[:,0].detach().cpu().numpy(),pointsJ[:,1].detach().cpu().numpy(),c='b', s=1, label='J')
                axes[0].set_title(f'point alignment error: {EP.item()}')
                axes[0].invert_yaxis()
                if it > 0 and it % 10 == 0:
                    axes[1].imshow(
                        einops.rearrange(fAIt, 'w h c -> h w c').mean(-1).detach().cpu().numpy(),
                        #vmin=J.min(),
                        #vmax=J.max()
                    )
                    axes[1].set_title(f'EM: {EM.item()}')

                axes[2].scatter(pointsIt[:,0].detach().cpu().numpy(),pointsIt[:,1].detach().cpu().numpy(),c='r', s=1, label='I')
                axes[2].scatter(pointsI_edget[:,0].detach().cpu().numpy(),pointsI_edget[:,1].detach().cpu().numpy(),c='orange', s=1, label='I')
                axes[2].scatter(spot_coordinate[:,0].detach().cpu().numpy(),spot_coordinate[:,1].detach().cpu().numpy(),c='black', s=1, label='J')
                axes[2].scatter(spot_coordinate_edge[:,0].detach().cpu().numpy(),spot_coordinate_edge[:,1].detach().cpu().numpy(),c='blue', s=1, label='J')
                axes[2].set_title(f'point alignment error: {EP.item()}')
                axes[2].invert_yaxis()
                
                axes[3].imshow(
                    JIt.mean(-1).detach().cpu().numpy(),
                )
                axes[3].set_title("J")
                axes[1].set_xbound(axes[3].get_xbound())
                axes[1].set_ybound(axes[3].get_ybound())
                if debug_path is not None:
                    fig.savefig(
                        os.path.join(
                            debug_path,
                            f'point_alignment_{it}.png'
                        )
                    )
                    plt.close()
            postfix = dict(zip(['EM','ER','EP'], tosave))
            postfix['angle'] = theta.item()
            pbar.set_postfix(postfix)
            E.backward()
            with torch.no_grad():
                if it % 10 == 0 and it > 0:
                    latent_optimizer.step()
                    latent_optimizer.zero_grad()
                    
                if not torch.isnan(theta.grad).any() or not torch.isnan(T.grad).any():
                    theta -= (5e-2 / (1.0 + (it >= diffeo_start) * 9)) * theta.grad
                if not torch.isnan(T.grad).any():
                    T -= 2000 * T.grad
                if not torch.isnan(S.grad).any():
                    S -= S.grad * 1e-4
                T.grad.zero_()
                # theta += torch.rand(theta.shape).to(S.device) * 0.05
                theta.grad.zero_()
                # S += torch.rand(S.shape).to(S.device) * 0.05
                S.grad.zero_()
                # v grad
                vgrad = v.grad
                if not torch.isnan(vgrad).any():
                    # smooth it
                    vgrad = torch.fft.ifftn(
                        torch.fft.fftn(vgrad, dim=(1, 2)) * K[..., None], dim=(1, 2)
                    ).real
                    if it >= diffeo_start:
                        v -= vgrad * epV
                    v.grad.zero_()

            A = to_A(L, T)
            fit_history[it] = dict(
                loss = tosave,
                theta = theta.item(),
                T = T.detach().cpu().numpy(),
                S = S.detach().cpu().numpy(),
                v = v.detach().cpu().numpy(),
                A = A.detach().cpu().numpy(),
                L = L.detach().cpu().numpy(),
                pointsIt = pointsIt.detach().cpu().numpy(),
            )
            pbar.update()
        pbar.close()
        return fit_history


    def _prepare_batch(
        self,
        X_st,
        X_sm,
    ):
        stdict, smdict = None, None
        if X_st is not None:
            x_st = self._dataset_st[X_st.cpu().numpy()]
            x_st = torch.tensor(
                np.vstack(list(map(lambda x: x.toarray() if issparse(x) else x, x_st)))
            )
            x_st = x_st.to(self.device)
            lib_size_st = x_st.sum(1).to(self.device)
            coord_st = torch.tensor(
                self.st_spatial_coord[X_st.detach().cpu().numpy()], device=self.device
            ).to(torch.float32)
            stdict = dict(
                X=x_st,
                lib_size=lib_size_st,
                spatial_coord=coord_st
            )

        if X_sm is not None:
            x_sm = self._dataset_sm[X_sm.cpu().numpy()]
            x_sm = torch.tensor(
                np.vstack(list(map(lambda x: x.toarray() if issparse(x) else x, x_sm)))
            )
            x_sm = x_sm.to(self.device)
            lib_size_sm = x_sm.sum(1).to(self.device)
            coord_sm = torch.tensor(
                self.sm_spatial_coord[X_sm.detach().cpu().numpy()], device=self.device
            ).to(torch.float32)
            smdict = dict(
                X=x_sm,
                lib_size=lib_size_sm,
                spatial_coord=coord_sm
            )

        return dict(
            st=stdict,
            sm=smdict
        )

    def inside_delaunay_mask(self, xI, pointsI):
        # Compute Delaunay triangulation
        tri = Delaunay(pointsI)

        # Create a mask for the image
        mask = np.zeros((xI[0].shape[0], xI[1].shape[0]), dtype=bool)

        # Generate all pixel coordinates
        xx, yy = np.meshgrid(xI[0],xI[1])
        pixel_coordinates = np.vstack((xx.flatten(), yy.flatten())).T

        # Find simplex for each pixel
        simplex_indices = tri.find_simplex(pixel_coordinates)

        # Mark pixels inside the Delaunay hull
        mask.T[(simplex_indices >= 0).reshape(len(xI[1]),len(xI[0]))] = True
        return mask
