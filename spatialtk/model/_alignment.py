import pytorch3d
from typing import List, NamedTuple, Optional, TYPE_CHECKING, Union
import torch
from pytorch3d.ops import knn_points
from pytorch3d.structures import utils as strutil
# Built-in
import time
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

class Alignment(NamedTuple):
    def __init__(self, adata: AnnData, **kwargs):
        self.adata = adata
        self.kwargs = kwargs
        self._check_data()
        self._check_kwargs
    
    def image_transform(site_df, horizontal_flip=False, 
                    vertical_flip=False, rotation=None):
        # Create copies of the input data frames for modification
        new_site_df = site_df.copy()
        # Apply horizontal flip
        if horizontal_flip:
            new_site_df["x_coordinates"] = -new_site_df["x_coordinates"]
        # Apply vertical flip
        if vertical_flip:
            new_site_df["y_coordinates"] = -new_site_df["y_coordinates"]
        # Apply rotation to SM
        if rotation is not None:
            radians = np.deg2rad(rotation)
            cos_theta = np.cos(radians)
            sin_theta = np.sin(radians)
            x = new_site_df["x_coordinates"]
            y = new_site_df["y_coordinates"]
            new_site_df["x_coordinates"] = x * cos_theta - y * sin_theta
            new_site_df["y_coordinates"] = x * sin_theta + y * cos_theta
        return new_site_df
    
    def transform_and_align_coordinates(adata_SM, adata_ST, SM_site_df=None, ST_site_df=None):
        if SM_site_df is None:
            SM_site_df = adata_SM.obs.loc[:, ["x_coordinates", "y_coordinates"]]
        if ST_site_df is None:
            ST_site_df = adata_ST.obs.loc[:, ["x_coordinates", "y_coordinates"]]
        SM_tensors = torch.tensor(SM_site_df.values, dtype=torch.float32)
        ST_tensors = torch.tensor(ST_site_df.values, dtype=torch.float32)
        width_SM=SM_site_df.x_coordinates.max()-SM_site_df.x_coordinates.min()
        height_SM=SM_site_df.y_coordinates.max()-SM_site_df.y_coordinates.min()
        width_ST=ST_site_df.x_coordinates.max()-ST_site_df.x_coordinates.min()
        height_ST=ST_site_df.y_coordinates.max()-ST_site_df.y_coordinates.min()
        #scaling = ((height_ST / height_SM)+(width_ST/width_SM))/2
        #scaling = torch.max(ST_tensors) / torch.max(SM_tensors)
        scaling_width = width_ST / width_SM
        scaling_height = height_ST / height_SM
        ST_tensor_scaled = ST_tensors.clone()
        ST_tensor_scaled[:, 0] = ST_tensor_scaled[:, 0] / scaling_width
        ST_tensor_scaled[:, 1] = ST_tensor_scaled[:, 1] / scaling_height
        #ST_tensor_scaled = ST_tensors / scaling
        icp_solution = pytorch3d.ops.iterative_closest_point(
            SM_tensors.to(torch.float32).unsqueeze(0),
            ST_tensor_scaled.unsqueeze(0),
        )
        _x = SM_site_df.to_numpy()
        _new_x = torch.zeros(_x.shape)
        _new_x[:,0] = torch.tensor(_x[:,0])
        _new_x[:,1] = torch.tensor(_x[:,1] )
        new_sm_coordinate = (torch.tensor(_new_x).to(torch.float32) @ icp_solution.RTs.R + icp_solution.RTs.T)
        adata_SM.obs['new_x_coordinates'] = new_sm_coordinate.numpy()[0, :, 0] * scaling_width.item()
        adata_SM.obs['new_y_coordinates'] = new_sm_coordinate.numpy()[0, :, 1] * scaling_height.item()
        adata_SM.obsm["new_coordinates"] = adata_SM.obs.loc[:, ['new_x_coordinates', 'new_y_coordinates']].to_numpy()
        adata_ST.obsm['coordinates'] = adata_ST.obs.loc[:, ['x_coordinates', 'y_coordinates']].to_numpy()
        return adata_SM, adata_ST
    