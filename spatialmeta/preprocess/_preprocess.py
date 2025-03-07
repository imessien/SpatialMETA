# Pytorch
import torch

# Built-in
import time
import random
import os
from typing import List, NamedTuple, Optional, TYPE_CHECKING, Union, Tuple
import scipy
# Third-party
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
import pyimzml
from pyimzml.ImzMLParser import ImzMLParser
import intervaltree
import einops
from sklearn.neighbors import NearestNeighbors
from scipy import sparse
from shapely.geometry import Point
from shapely.geometry import Polygon
from scipy import spatial
from scipy.spatial import ConvexHull
from scipy.sparse import csr_matrix
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests
from collections import defaultdict
from scipy.stats import hypergeom
from pathlib import Path
from sklearn.preprocessing import MaxAbsScaler


# Local
from ..util._decorators import ignore_warning
from ._creat_hex_grid import create_hex_grid
from ..util._squidpy_spatial_variable_genes import spatial_neighbors, spatial_autocorr
from ..util._classes import AnnDataSM, AnnDataST, AnnDataJointSMST

MODULE_PATH = Path(__file__).parent

@ignore_warning(level="ignore")
def read_sm_csv_as_anndata(
    sm_file: str
) -> AnnDataSM:
    """
    Read SM csv file as AnnData object.

    :param sm_file: str. The file path and name of the SM csv file.

    :return: AnnDataSM. The AnnData object with SM data.
    
    """
    if not os.path.exists(sm_file):
        raise FileNotFoundError(f"SM file {sm_file} not found.")

    SM_df = pd.read_csv(sm_file)
    original_x_coord = SM_df["x"]
    original_y_coord = SM_df["y"]
    X = SM_df.drop(["x", "y"], axis=1)
    var_list = list(X.columns.values)
    adata_SM = AnnDataSM(
        sparse.csr_matrix(X),
        var=pd.DataFrame(
            var_list, 
            index=var_list, 
            columns=["m/z"]
        ),
        obs=pd.DataFrame(
            np.array([original_x_coord, original_y_coord]).T,
            columns=["x_coord_original", "y_coord_original"],
        ),
    )
    adata_SM.obsm["spatial"] = adata_SM.obs.loc[
        :, ["x_coord_original", "y_coord_original"]
    ].to_numpy()
    adata_SM.obs["total_intensity"] = np.array(adata_SM.X.sum(1)).flatten()
    adata_SM.obs["mean_intensity"] = np.array(adata_SM.X.mean(1)).flatten()
    adata_SM = adata_SM[adata_SM.obs["total_intensity"] > 0]
    return adata_SM


def _merge_and_get_unique_arrays(mass_data):
    all_values = [value for array in mass_data.values() for value in array]
    unique_values = list(set(all_values))
    return unique_values


def _calculate_ppm_range(observed_mz, ppm_tolerance=5):
    ppm_range = observed_mz * (ppm_tolerance / 1e6)
    lower_limit = observed_mz - ppm_range
    upper_limit = observed_mz + ppm_range
    return pd.Series(
        [ppm_range, lower_limit, upper_limit],
        index=["ppm_range", "lower_limit", "upper_limit"],
    )


@ignore_warning(level="ignore")
def get_mz_reference(
    p: ImzMLParser, 
    ppm_tolerance: int = 5
) -> pd.DataFrame:
    """
    Get m/z reference from ImzMLParser object.

    :param p: ImzMLParser. The ImzMLParser object.
    :param ppm_tolerance: int. The ppm tolerance, defalut is 5.

    :return: pd.DataFrame. The m/z reference for the SM data, and the column names are "m/z", "Interval Width (+/- Da)".
    """
    my_spectra = []
    mass_data = {}
    for idx, (x, y, z) in enumerate(p.coordinates):
        mzs, intensities = p.getspectrum(idx)
        mass_data[idx] = mzs
        my_spectra.append([mzs, intensities, (x, y, z)])
    unique_values = _merge_and_get_unique_arrays(mass_data)
    unique_valueslues_ppm = [
        _calculate_ppm_range(x, ppm_tolerance) for x in unique_values
    ]
    unique_valueslues_ppm_df = pd.DataFrame(unique_valueslues_ppm)
    unique_mz_df = pd.DataFrame(
        {
            "mz": unique_values,
            "ppm_range": unique_valueslues_ppm_df["ppm_range"].values,
            "lower_limit": unique_valueslues_ppm_df["lower_limit"].values,
            "upper_limit": unique_valueslues_ppm_df["upper_limit"].values,
        }
    )
    unique_mz_df = unique_mz_df.sort_values(by="mz").reset_index(drop=True)
    merged_intervals = pd.DataFrame(columns=["lower_limit", "upper_limit"])
    current_lower = unique_mz_df["lower_limit"].iloc[0]
    current_upper = unique_mz_df["upper_limit"].iloc[0]
    for i in range(1, len(unique_mz_df)):
        # If the lower limit of the next interval is less than or equal to the current upper limit, merge the intervals
        if unique_mz_df["lower_limit"].iloc[i] <= current_upper:
            current_upper = max(current_upper, unique_mz_df["upper_limit"].iloc[i])
        else:
            merged_intervals = merged_intervals.append(
                {"lower_limit": current_lower, "upper_limit": current_upper},
                ignore_index=True,
            )
            current_lower = unique_mz_df["lower_limit"].iloc[i]
            current_upper = unique_mz_df["upper_limit"].iloc[i]
    merged_intervals = merged_intervals.append(
        {"lower_limit": current_lower, "upper_limit": current_upper}, ignore_index=True
    )
    merged_intervals["new_mz"] = (
        merged_intervals["lower_limit"] + merged_intervals["upper_limit"]
    ) / 2
    merged_intervals["new_ppm_range"] = (
        merged_intervals["upper_limit"] - merged_intervals["lower_limit"]
    ) / 2
    mz_reference = merged_intervals[["new_mz", "new_ppm_range"]]
    mz_reference.columns = ["m/z", "Interval Width (+/- Da)"]
    return mz_reference


def _get_interval(t, v):
    r = t.at(v)
    if len(r) > 0:
        return list(r)[0][-1]
    else:
        return None


@ignore_warning(level="ignore")
def read_sm_imzml_as_anndata(
    p: ImzMLParser, 
    mz_reference: pd.DataFrame
) -> AnnDataSM:
    """
    Read SM imzML file as AnnData object.

    :param p: ImzMLParser. The ImzMLParser object.
    :param mz_reference: pd.DataFrame. The m/z reference.

    :return: AnnDataSM. The AnnData object with SM data.
    """
    my_spectra = []
    for idx, (x, y, z) in enumerate(p.coordinates):
        mzs, intensities = p.getspectrum(idx)
        my_spectra.append([mzs, intensities, (x, y, z)])
    t = intervaltree.IntervalTree()
    for i, j in mz_reference.iloc[:, :2].to_numpy():
        t[i - j : i + j] = str(i)
    mz_mapping_result = [[_get_interval(t, x) for x in y[0]] for y in my_spectra]
    mz_reorder_indices = dict(
        zip(list(map(str, mz_reference.iloc[:, 0])), range(len(mz_reference)))
    )
    new_signal_merged = []
    # Reorder the signal to match the mz_reference
    for e, m in enumerate(mz_mapping_result):
        signal = my_spectra[e][1]
        new_signal = np.zeros(len(mz_reference))
        for r, v in enumerate(m):
            if v in mz_reorder_indices.keys():
                new_signal[mz_reorder_indices[v]] += signal[r]
        new_signal_merged.append(new_signal)
    new_signal_merged = np.vstack(new_signal_merged)
    x_coord_original = [spectra[2][0] for spectra in my_spectra]
    y_coord_original = [spectra[2][1] for spectra in my_spectra]
    var_list = mz_reference["m/z"].values
    adata_SM = AnnDataSM(
        X=sparse.csr_matrix(new_signal_merged),
        obs=pd.DataFrame(
            np.array([x_coord_original, y_coord_original]).T,
            columns=["x_coord_original", "y_coord_original"],
        ),
        var=pd.DataFrame(var_list, index=var_list, columns=["m/z"]),
    )
    adata_SM.obsm["spatial"] = adata_SM.obs.loc[
        :, ["x_coord_original", "y_coord_original"]
    ].to_numpy()
    adata_SM.obs["total_intensity"] = np.array(adata_SM.X.sum(1)).flatten()
    adata_SM.obs["mean_intensity"] = np.array(adata_SM.X.mean(1)).flatten()
    adata_SM = adata_SM[adata_SM.obs["total_intensity"] > 0]
    return adata_SM


def merge_sm_pos_neg(
    adata_SM_pos: AnnDataSM, 
    adata_SM_neg: AnnDataSM
) -> AnnDataSM:
    """
    Merge the positive and negative SM data.

    :param adata_SM_pos: AnnDataSM. The AnnData object with positive SM data.
    :param adata_SM_neg: AnnDataSM. The AnnData object with negative SM data.

    :return: AnnDataSM. The merged AnnData object with SM data.
    """
    _adata_SM_pos = adata_SM_pos.copy()
    _adata_SM_neg = adata_SM_neg.copy()
    adata_SM_neg_obs_df = _adata_SM_neg.obs
    adata_SM_pos_obs_df = _adata_SM_pos.obs
    adata_SM_neg_obs_df.columns = [
        "x_coord_original",
        "y_coord_original",
        "total_intensity_neg",
        "mean_intensity_neg",
    ]
    adata_SM_pos_obs_df.columns = [
        "x_coord_original",
        "y_coord_original",
        "total_intensity_pos",
        "mean_intensity_pos",
    ]
    obs_merge_df = adata_SM_pos_obs_df.merge(
        adata_SM_neg_obs_df, on=["x_coord_original", "y_coord_original"]
    )
    obs_merge_df["spot_index"] = (
        obs_merge_df["x_coord_original"].astype(str)
        + "_"
        + obs_merge_df["y_coord_original"].astype(str)
    )
    _adata_SM_pos.obs["spot_index"] = (
        _adata_SM_pos.obs["x_coord_original"].astype(str)
        + "_"
        + _adata_SM_pos.obs["y_coord_original"].astype(str)
    )
    _adata_SM_neg.obs["spot_index"] = (
        _adata_SM_neg.obs["x_coord_original"].astype(str)
        + "_"
        + _adata_SM_neg.obs["y_coord_original"].astype(str)
    )
    adata_SM_pos_overlap = _adata_SM_pos[
        _adata_SM_pos.obs["spot_index"].isin(obs_merge_df["spot_index"])
    ]
    adata_SM_neg_overlap = _adata_SM_neg[
        _adata_SM_neg.obs["spot_index"].isin(obs_merge_df["spot_index"])
    ]
    merge_X_data = np.concatenate(
        (adata_SM_pos_overlap.X.toarray(), adata_SM_neg_overlap.X.toarray()), axis=1
    )
    pos_var_list = list(_adata_SM_pos.var.index)
    neg_var_list = list(_adata_SM_neg.var.index)
    pos_var_df = pd.DataFrame({"name": pos_var_list, "type": "SM_pos"})
    pos_var_df.index = pos_var_list
    neg_var_df = pd.DataFrame({"name": neg_var_list, "type": "SM_neg"})
    neg_var_df.index = neg_var_list
    joint_var_df = pd.concat([pos_var_df, neg_var_df])
    merge_adata_SM = AnnDataSM(
        csr_matrix(merge_X_data),
        var=joint_var_df,
        obs=obs_merge_df,
        obsm=adata_SM_pos_overlap.obsm,
        uns=_adata_SM_pos.uns,
    )
    return merge_adata_SM


def calculate_scale_factor(
    adata_SM: AnnDataSM, 
    adata_ST: AnnDataST, 
    spatial_key_SM: str = "spatial", 
    spatial_key_ST: str = "spatial"
) -> Tuple[float, float]:
    """
    Calculate the scaling factor between SM and ST data.

    :param adata_SM: AnnDataSM. The AnnData object with SM data.
    :param adata_ST: AnnDataST. The AnnData object with ST data.
    :param spatial_key_SM: str. The spatial key for SM data, default is "spatial".
    :param spatial_key_ST: str. The spatial key for ST data, default is "spatial".

    :return: Tuple[float, float]. The scaling factor for width and height.
    """
    SM_site_df = pd.DataFrame(adata_SM.obsm[spatial_key_SM])
    ST_site_df = pd.DataFrame(adata_ST.obsm[spatial_key_ST])
    SM_site_df.columns = ["x_coord", "y_coord"]
    ST_site_df.columns = ["x_coord", "y_coord"]
    width_SM = SM_site_df.x_coord.max() - SM_site_df.x_coord.min()
    height_SM = SM_site_df.y_coord.max() - SM_site_df.y_coord.min()
    width_ST = ST_site_df.x_coord.max() - ST_site_df.x_coord.min()
    height_ST = ST_site_df.y_coord.max() - ST_site_df.y_coord.min()
    scaling_width = width_ST / width_SM
    scaling_height = height_ST / height_SM
    return scaling_width, scaling_height


def spot_transform_by_manual(
    adata: AnnData,
    horizontal_flip: bool = False,
    vertical_flip: bool = False,
    rotation: int = None,
    scale_width: float = 1,
    scale_height: float = 1,
    translation_x: Optional[float] = None,
    translation_y: Optional[float] = None,
    spatial_key_SM: str = "spatial",
    new_spatial_key_SM: str = "new1_spatial",
):
    """
    Transform the spatial coordinates of SM data by manual.
    
    :param adata: AnnData. The AnnData object.
    :param horizontal_flip: bool. The horizontal flip flag, if True, flip the spatial coordinates horizontally,default is False.
    :param vertical_flip: bool. The vertical flip flag, if True, flip the spatial coordinates vertically,default is False.
    :param rotation: int. The rotation angle, if not None, rotate the spatial coordinates,default is None.
    :param scale_width: float. The scaling factor for width,default is 1.
    :param scale_height: float. The scaling factor for height,default is 1.
    :param translation_x: Optional[float]. The translation factor for x coordinate,default is None.
    :param translation_y: Optional[float]. The translation factor for y coordinate,default is None.
    :param spatial_key_SM: str. The spatial key for SM data, default is "spatial".
    :param new_spatial_key_SM: str. The new spatial key for SM data, default is "new1_spatial".
    
    :return: AnnDataSM. The AnnData object with transformed SM data.
    """
    site_df = pd.DataFrame(adata.obsm[spatial_key_SM])
    site_df.columns = ["x_coord", "y_coord"]
    # Create copies of the input data frames for modification
    new_site_df = site_df.copy()
    # Apply horizontal flip
    if horizontal_flip:
        new_site_df["x_coord"] = -new_site_df["x_coord"]
    # Apply vertical flip
    if vertical_flip:
        new_site_df["y_coord"] = -new_site_df["y_coord"]
    # Apply rotation to SM
    if rotation is not None:
        radians = np.deg2rad(rotation)
        cos_theta = np.cos(radians)
        sin_theta = np.sin(radians)
        x = new_site_df["x_coord"]
        y = new_site_df["y_coord"]
        new_site_df["x_coord"] = x * cos_theta - y * sin_theta
        new_site_df["y_coord"] = x * sin_theta + y * cos_theta
    # Apply scaling
    new_site_df["x_coord"] = new_site_df["x_coord"] * scale_width
    new_site_df["y_coord"] = new_site_df["y_coord"] * scale_height
    # Apply translation
    if translation_x is not None:
        new_site_df["x_coord"] = new_site_df["x_coord"] + translation_x
    if translation_y is not None:
        new_site_df["y_coord"] = new_site_df["y_coord"] + translation_y
    # make x_coord and y_coord positive
    if new_site_df.iloc[:, 0].min() < 0:
        new_site_df.iloc[:, 0] = (
            new_site_df.iloc[:, 0] + new_site_df.iloc[:, 0].abs().max()
        )
    if new_site_df.iloc[:, 1].min() < 0:
        new_site_df.iloc[:, 1] = (
            new_site_df.iloc[:, 1] + new_site_df.iloc[:, 1].abs().max()
        )
    adata.obsm[new_spatial_key_SM] = new_site_df.to_numpy()


def new_spot_sample(
    adata_SM,
    adata_ST,
    spatial_key_SM="spatial",
    spatial_key_ST="spatial",
    min_diam=500,
) -> pd.DataFrame:
    """
    Generate new spots by resampling in the intersection of the convex hull of SM and ST spots.
    
    :param adata_SM: AnnDataSM. The AnnData object with SM data.
    :param adata_ST: AnnDataST. The AnnData object with ST data.
    :param spatial_key_SM: str. The spatial key for SM data, default is "spatial".
    :param spatial_key_ST: str. The spatial key for ST data, default is "spatial".
    :param min_diam: int. The minimum diameter of the hexagonal grid, default is 500.
    
    :return: pd.DataFrame. The new spots in the intersection of the convex hull of SM and ST spots.
    """
    width_max_SM = adata_SM.obsm[spatial_key_SM][:, 0].max()
    height_max_SM = adata_SM.obsm[spatial_key_SM][:, 1].max()
    width_max_ST = adata_ST.obsm[spatial_key_ST][:, 0].max()
    height_max_ST = adata_ST.obsm[spatial_key_ST][:, 1].max()
    # calculate the max bewteen width and height
    max_wh = max(width_max_SM, height_max_SM, width_max_ST, height_max_ST) * 1.2
    nx = int(max_wh / min_diam)
    ny = int(max_wh / min_diam)
    x_shift = (nx * min_diam) / 2
    y_shift = (ny * min_diam) / 2
    hex_grid_coord = create_hex_grid(
        nx=nx,
        ny=ny,
        min_diam=min_diam,
        x_shift=x_shift,
        y_shift=y_shift,
        do_plot=False,
        edge_color=(0, 0, 0),
    )
    SM_coord = pd.DataFrame(adata_SM.obsm[spatial_key_SM])
    ST_coord = pd.DataFrame(adata_ST.obsm[spatial_key_ST])
    SM_coord.columns = ["x_coord", "y_coord"]
    ST_coord.columns = ["x_coord", "y_coord"]
    hull1 = ConvexHull(SM_coord)
    hull2 = ConvexHull(ST_coord)
    vertices1 = SM_coord.iloc[hull1.vertices]
    vertices2 = ST_coord.iloc[hull2.vertices]
    polygon1 = Polygon(vertices1.values)
    polygon2 = Polygon(vertices2.values)
    intersection = polygon1.intersection(polygon2)
    new_dot_df = pd.DataFrame(hex_grid_coord[0])
    new_dot_df.columns = ["x_coord", "y_coord"]
    new_dot_df["point"] = new_dot_df.apply(
        lambda row: Point(row["x_coord"], row["y_coord"]), axis=1
    )
    new_dot_df["in_intersection"] = new_dot_df["point"].apply(
        lambda point: intersection.contains(point)
    )
    new_dot_in_df = new_dot_df[new_dot_df.in_intersection]
    return new_dot_in_df

def ST_spot_sample(
    adata_ST: AnnDataST,
    spatial_key_ST: str = "spatial",
) -> pd.DataFrame:
    """
    Sample the ST spots.
    
    :param adata_ST: AnnDataST. The AnnData object with ST data.
    :param spatial_key_ST: str. The spatial key for ST data, default is "spatial".
    
    :return: pd.DataFrame. The ST spots.
    """
    ST_coord = pd.DataFrame(adata_ST.obsm[spatial_key_ST])
    ST_coord.columns = ["x_coord", "y_coord"]
    return ST_coord

def SM_spot_sample(
    adata_SM: AnnDataSM,
    spatial_key_SM: str = "spatial",
) -> pd.DataFrame:
    """
    Sample the SM spots.
    
    :param adata_SM: AnnDataSM. The AnnData object with SM data.
    :param spatial_key_SM: str. The spatial key for SM data, default is "spatial".
    
    :return: pd.DataFrame. The SM spots.
    """
    SM_coord = pd.DataFrame(adata_SM.obsm[spatial_key_SM])
    SM_coord.columns = ["x_coord", "y_coord"]
    return SM_coord

@ignore_warning(level="ignore")
def spot_align_byknn(
    new_dot_in_df: pd.DataFrame,
    adata_SM: AnnData,
    adata_ST: AnnData,
    spatail_key_SM: str = "spatial",
    spatial_key_ST: str = "spatial",
    min_diam: int = 500,
    min_dist_fold: float = 1.5,
) -> Tuple[AnnDataSM, AnnDataST]:
    """
    Reassignment the new spots to the SM and ST data by KNN.
    
    :param new_dot_in_df: pd.DataFrame. The new spots in the intersection of the convex hull of SM and ST spots, oytput of function 'new_spot_sample()'.
    :param adata_SM: AnnDataSM. The AnnData object with SM data.
    :param adata_ST: AnnDataST. The AnnData object with ST data.
    :param spatail_key_SM: str. The spatial key for SM data, default is "spatial".
    :param spatial_key_ST: str. The spatial key for ST data, default is "spatial".
    :param min_diam: int. The minimum distance of the spot, which is same as the min_diam in function 'new_spot_sample()', default is 500.
    :param min_dist_fold: float. The minimum distance fold, used to filter the nearest spots, defaults to 1.5. For example, if min_diam is 500 and min_dist_fold is 1.5, the minimum distance for filtering is 500 * 1.5 = 750. This filters out spots greater than this distance.
    
    :return: Tuple[AnnDataSM, AnnDataST]. The AnnData object with SM and ST data after reassignment.
    """
    
    min_dist = min_diam * min_dist_fold
    new_dot_in_df["spot_name"] = "spot_" + new_dot_in_df.reset_index().index.astype(str)
    new_dot_in_df.index = new_dot_in_df.reset_index().index
    new_dot_coords = new_dot_in_df[["x_coord", "y_coord"]].values
    adata_SM_coords = adata_SM.obsm[spatail_key_SM]
    adata_ST_coords = adata_ST.obsm[spatial_key_ST]
    knn_SM = NearestNeighbors(n_neighbors=5, algorithm="auto")
    knn_ST = NearestNeighbors(n_neighbors=5, algorithm="auto")
    knn_SM.fit(adata_SM_coords)
    knn_ST.fit(adata_ST_coords)
    _, indices_SM = knn_SM.kneighbors(new_dot_coords)
    _, indices_ST = knn_ST.kneighbors(new_dot_coords)
    distances_SM, indices_SM = knn_SM.kneighbors(new_dot_coords)
    distances_ST, indices_ST = knn_ST.kneighbors(new_dot_coords)
    knn_result_df = pd.DataFrame(
        columns=[
            "spot_name",
            "nearest_spots_SM",
            "nearest_spots_ST",
            "distances_SM",
            "distances_ST",
            "filtered_nearest_spots_SM",
            "filtered_nearest_spots_ST",
            "filtered_nearest_spots_SM_number",
            "filtered_nearest_spots_ST_number",
        ]
    )
    for i in range(len(new_dot_in_df)):
        spot_name = new_dot_in_df.loc[i, "spot_name"]
        nearest_spots_SM = indices_SM[i]
        nearest_spots_ST = indices_ST[i]
        spots_distances_SM = distances_SM[i]
        spots_distances_ST = distances_ST[i]
        spots_filtered_indices_SM = nearest_spots_SM[
            [j for j, dist in enumerate(spots_distances_SM) if dist <= min_dist]
        ]
        spots_filtered_indices_ST = nearest_spots_ST[
            [j for j, dist in enumerate(spots_distances_ST) if dist <= min_dist]
        ]
        knn_result_tmp_df = pd.DataFrame(
            {
                "spot_name": spot_name,
                "nearest_spots_SM": [nearest_spots_SM],
                "nearest_spots_ST": [nearest_spots_ST],
                "distances_SM": [spots_distances_SM],
                "distances_ST": [spots_distances_ST],
                "filtered_nearest_spots_SM": [spots_filtered_indices_SM],
                "filtered_nearest_spots_ST": [spots_filtered_indices_ST],
                "filtered_nearest_spots_SM_number": [len(spots_filtered_indices_SM)],
                "filtered_nearest_spots_ST_number": [len(spots_filtered_indices_ST)],
            }
        )
        knn_result_df = pd.concat([knn_result_df,knn_result_tmp_df])
    # set knn_result_df index in order
    knn_result_df.index = range(knn_result_df.shape[0])
    knn_adata_SM = np.zeros(
        (knn_result_df.shape[0], adata_SM.shape[1]), dtype=np.float32
    )
    knn_adata_ST = np.zeros(
        (knn_result_df.shape[0], adata_ST.shape[1]), dtype=np.float32
    )
    for i, row in knn_result_df.iterrows():
        # print(i)
        if row["filtered_nearest_spots_SM_number"] > 0:
            nearest_indices_SM = row["filtered_nearest_spots_SM"]
            knn_adata_SM[i] = adata_SM.X[nearest_indices_SM].mean(axis=0)
        if row["filtered_nearest_spots_ST_number"] > 0:
            nearest_indices_ST = row["filtered_nearest_spots_ST"]
            knn_adata_ST[i] = np.round(
                adata_ST.X[nearest_indices_ST].mean(axis=0), decimals=0
            ).astype(np.float32)
    var_list_SM = list(adata_SM.var.index)
    adata_SM_new = AnnDataSM(
        csr_matrix(knn_adata_SM),
        var=pd.DataFrame(var_list_SM, index=var_list_SM, columns=["m/z"]),
        obs=pd.DataFrame(
            np.array(
                [
                    new_dot_in_df["x_coord"].values,
                    new_dot_in_df["y_coord"].values,
                    new_dot_in_df["spot_name"].values,
                ]
            ).T,
            columns=["x_coord", "y_coord", "spot_name"],
        ),
        obsm={"spatial": new_dot_coords},
        uns=adata_SM.uns,
    )
    var_list_ST = list(adata_ST.var.index)
    adata_ST_new = AnnDataST(
        csr_matrix(knn_adata_ST),
        var=pd.DataFrame(var_list_ST, index=var_list_ST, columns=["genename"]),
        obs=pd.DataFrame(
            np.array(
                [
                    new_dot_in_df["x_coord"].values,
                    new_dot_in_df["y_coord"].values,
                    new_dot_in_df["spot_name"].values,
                ]
            ).T,
            columns=["x_coord", "y_coord", "spot_name"],
        ),
        obsm={"spatial": new_dot_coords},
        uns=adata_ST.uns,
    )
    adata_SM_new.obs["x_coord"] = adata_SM_new.obs["x_coord"].astype(int)
    adata_SM_new.obs["y_coord"] = adata_SM_new.obs["y_coord"].astype(int)
    adata_ST_new.obs["x_coord"] = adata_ST_new.obs["x_coord"].astype(int)
    adata_ST_new.obs["y_coord"] = adata_ST_new.obs["y_coord"].astype(int)
    return adata_SM_new, adata_ST_new

def joint_adata_sm_st(
    adata_SM_new: AnnDataSM, 
    adata_ST_new: AnnDataST
) -> AnnDataJointSMST:
    """
    Merge the SM and ST data into a joint AnnData object.
    
    :param adata_SM_new: AnnDataSM. The AnnData object with SM data after reassignment.
    :param adata_ST_new: AnnDataST. The AnnData object with ST data after reassignment.
    
    :return: AnnDataJointSMST. The joint AnnData object with SM and ST data.
    """
    overlap_obs_list = set(adata_ST_new.obs["spot_name"]).intersection(
        set(adata_SM_new.obs["spot_name"])
    )
    adata_SM_new_overlap = adata_SM_new[
        adata_SM_new.obs["spot_name"].isin(overlap_obs_list)
    ]
    adata_ST_new_overlap = adata_ST_new[
        adata_ST_new.obs["spot_name"].isin(overlap_obs_list)
    ]
    joint_X_data = np.concatenate(
        (adata_SM_new_overlap.X.toarray(), adata_ST_new_overlap.X.toarray()), axis=1
    )
    SM_obs_df = adata_SM_new_overlap.obs
    ST_obs_df = adata_ST_new_overlap.obs
    # joint SM_obs_df and ST_obs_df by spot_name,x_coord, y_coord
    joint_obs_df = SM_obs_df.merge(ST_obs_df, on=["spot_name", "x_coord", "y_coord"])
    SM_var_list = list(adata_SM_new.var.index)
    ST_var_list = list(adata_ST_new.var.index)
    SM_var_df = pd.DataFrame({"name": SM_var_list, "type": "SM"})
    SM_var_df.index = SM_var_list
    ST_var_df = pd.DataFrame({"name": ST_var_list, "type": "ST"})
    ST_var_df.index = ST_var_list
    joint_var_df = pd.concat([SM_var_df, ST_var_df])
    joint_adata = AnnDataJointSMST(
        csr_matrix(joint_X_data),
        var=joint_var_df,
        obs=joint_obs_df,
        obsm=adata_ST_new_overlap.obsm,
        uns=adata_ST_new.uns,
    )
    joint_adata.obs["x_coord"] = joint_adata.obs["x_coord"].astype(int)
    joint_adata.obs["y_coord"] = joint_adata.obs["y_coord"].astype(int)
    return joint_adata


@ignore_warning(level="ignore")
def normalize_total_joint_adata_sm_st(
    joint_adata: AnnDataJointSMST,
    target_sum_SM: Optional[int] = 1e4,
    target_sum_ST: Optional[int] = 1e4,
):
    """
    Normalize the total intensity of the SM and ST data in the joint AnnData object.
    
    :param joint_adata: AnnDataJointSMST. The joint AnnData object with SM and ST data.
    :param target_sum_SM: Optional[int]. The target sum for SM data, default is 1e4.
    :param target_sum_ST: Optional[int]. The target sum for ST data, default is 1e4.
    """
    if target_sum_SM is not None:
        joint_adata_SM = joint_adata[:, joint_adata.var.type == "SM"]
        sc.pp.normalize_total(joint_adata_SM, target_sum=target_sum_SM)
        joint_adata.X[:, joint_adata.var.type == "SM"] = joint_adata_SM.X
    if target_sum_ST is not None:
        joint_adata_ST = joint_adata[:, joint_adata.var.type == "ST"]
        sc.pp.normalize_total(joint_adata_ST, target_sum=target_sum_ST)
        joint_adata_ST.X = joint_adata_ST.X.astype(joint_adata.X.dtype)

def scale_joint_adata_sm_st(
    joint_adata: AnnDataJointSMST,
    scale_range_SM: Optional[Tuple[float, float]] = (0, 10),
    scale_range_ST: Optional[Tuple[float, float]] = (0, 10),
):
    """
    Scale the SM and ST data in the joint AnnData object.
    
    :param joint_adata: AnnDataJointSMST. The joint AnnData object with SM and ST data.
    :param scale_range_SM: Optional[Tuple[float, float]]. The scale range for SM data, default is (0, 10).
    :param scale_range_ST: Optional[Tuple[float, float]]. The scale range for ST data, default is (0, 10).
    """
    if scale_range_SM is not None:
        joint_adata_SM = joint_adata[:, joint_adata.var.type == "SM"]
        sc.pp.scale(joint_adata_SM, zero_center=False, max_value=scale_range_SM[1])
        joint_adata.X[:, joint_adata.var.type == "SM"] = joint_adata_SM.X
    if scale_range_ST is not None:
        joint_adata_ST = joint_adata[:, joint_adata.var.type == "ST"]
        sc.pp.scale(joint_adata_ST, zero_center=False, max_value=scale_range_ST[1])
        joint_adata_ST.X = joint_adata_ST.X.astype(joint_adata.X.dtype)
        
def maxabsscale_joint_adata_sm_st(
    joint_adata: AnnDataJointSMST,
    scale_range_SM: Optional[Tuple[float, float]] = (0, 10),
    scale_range_ST: Optional[Tuple[float, float]] = (0, 10),
    global_scale: bool = False
):
    """
    Scale the SM and ST data in the joint AnnData object.
    
    :param joint_adata: AnnDataJointSMST. The joint AnnData object with SM and ST data.
    :param scale_range_SM: Optional[Tuple[float, float]]. The scale range for SM data, default is (0, 10).
    :param scale_range_ST: Optional[Tuple[float, float]]. The scale range for ST data, default is (0, 10).
    """
    if global_scale:
        if scale_range_SM is not None:
            joint_adata_SM = joint_adata[:, joint_adata.var.type == "SM"].copy()
            b, c = joint_adata_SM.X.shape
            if scipy.sparse.issparse(joint_adata_SM.X) or isinstance(joint_adata_SM.X, anndata._core.views.SparseCSRView):
                joint_adata_SM.X = joint_adata_SM.X.toarray()
            joint_adata_SM.X = einops.rearrange(
                (MaxAbsScaler().fit_transform(
                    einops.rearrange(joint_adata_SM.X, 'b c -> (b c)')[None,:]
                ) * scale_range_SM[1]).flatten(),
                '(b c) -> b c',
                b=b,
                c=c
            )
            joint_adata.X[:, joint_adata.var.type == "SM"] = joint_adata_SM.X
        if scale_range_ST is not None:
            joint_adata_ST = joint_adata[:, joint_adata.var.type == "ST"].copy()
            b, c = joint_adata_ST.X.shape
            if scipy.sparse.issparse(joint_adata_ST.X) or isinstance(joint_adata_ST.X, anndata._core.views.SparseCSRView):
                joint_adata_ST.X = joint_adata_ST.X.toarray()
            joint_adata_ST.X = einops.rearrange(
                (MaxAbsScaler().fit_transform(
                    einops.rearrange(joint_adata_ST.X, 'b c -> (b c)')[None,:]
                ) * scale_range_ST[1]).flatten(),
                '(b c) -> b c',
                b=b,
                c=c
            )
            joint_adata.X[:, joint_adata.var.type == "ST"] = joint_adata_ST.X
    else:
        if scale_range_SM is not None:
            joint_adata_SM = joint_adata[:, joint_adata.var.type == "SM"]
            joint_adata_SM.X = MaxAbsScaler().fit_transform(joint_adata_SM.X) * scale_range_SM[1]
            joint_adata.X[:, joint_adata.var.type == "SM"] = joint_adata_SM.X
        if scale_range_ST is not None:
            joint_adata_ST = joint_adata[:, joint_adata.var.type == "ST"]
            joint_adata_ST.X = MaxAbsScaler().fit_transform(joint_adata_ST.X) * scale_range_SM[1]
            joint_adata_ST.X = joint_adata_ST.X.astype(joint_adata.X.dtype)
        


def compute_batch_variable_genes_core(
    adata: AnnData,
    batch_key: str,
    min_frac: float = 0.8,
    min_logfc: float = 3,
) -> List[str]:
    """
    Compute the batch variable genes.
    
    :param adata: AnnData. The AnnData object.
    :param batch_key: str. The batch key.
    :param min_frac: float. The minimum fraction, default is 0.8.
    :param min_logfc: float. The minimum log fold change, default is 3.
    
    :return: List[str]. The batch variable genes.
    """
    
    if len(np.unique(adata.obs[batch_key])) == 1:
        return {
            adata.obs[batch_key].iloc[0]: np.ones(adata.shape[1])
        }
    sc.tl.rank_genes_groups(adata, groupby=batch_key, method='t-test')
    l = list(adata.var.index)
    result = []
    columns = list(pd.DataFrame(adata.uns['rank_genes_groups']['logfoldchanges']).columns)
    for i,j,k in zip(pd.DataFrame(adata.uns['rank_genes_groups']['logfoldchanges']).to_numpy(),
        pd.DataFrame(adata.uns['rank_genes_groups']['names']).to_numpy(),
        pd.DataFrame(adata.uns['rank_genes_groups']['pvals_adj']).to_numpy(),
    ):
        for e,(m,n,p) in enumerate(zip(i,j,-np.log10(k))):
            result.append((columns[e],m,n,p))
    result = pd.DataFrame(result, columns=['groups','logfoldchanges','gene_name','pvals_adj'])

    mask_genes = {}

    selected_genes = {}
    for i in np.unique(adata.obs[batch_key]):

        sel = result.loc[
            np.array(result['logfoldchanges'] > 1)  &
            np.array(result['pvals_adj'].replace(np.inf, 300) > 10) &
            np.array(result.iloc[:,0] == i),
            ['groups','gene_name']
        ]
        selected_genes[i] = list(sel['gene_name'])


    ht = sc.pl.dotplot(
        adata,
        np.unique(FLATTEN(selected_genes.values())),
        groupby=batch_key, 
        show=False, 
        return_fig=True
    )
    size_df = ht.dot_size_df

    for i in np.unique(adata.obs[batch_key]):
        mask_genes[i] = []
        m = dict(zip(
            result.loc[result['groups'] == i,'gene_name'],
            result.loc[result['groups'] == i,'logfoldchanges']
        ))
        for j in selected_genes[i]:
            if size_df.loc[i,j] > min_frac and m[j] > min_logfc:
                mask_genes[i].append(j)

    return list(set(FLATTEN([list(v) for v in mask_genes.values()])))

def FLATTEN(x): return [i for s in x for i in s]

@ignore_warning(level="ignore")
def spatial_variable_joint_adata_sm_st(
    joint_adata: AnnDataJointSMST,
    n_top_genes: int = 2000,
    n_top_metabolites: int = 800,
    add_key: str = "highly_variable_moranI",
    batch_key: Optional[str] = None,
    min_samples: int = 2,
    min_frac: float = 0.8,
    min_logfc: float = 3,
):
    """
    Calculate the spatial variables for the joint AnnData object and remove the batch-specific spatial variables.
    
    :param joint_adata: AnnDataJointSMST. The joint AnnData object with SM and ST data.
    :param n_top_genes: int. The number of top genes, default is 2000.
    :param n_top_metabolites: int. The number of top metabolites, default is 800.
    :param add_key: str. The key for the spatial variables, default is "highly_variable_moranI".
    :param batch_key: Optional[str]. The batch key, default is None.
    :param min_samples: int. The minimum number of samples, default is 2.
    :param min_frac: float. The minimum fraction, default is 0.8.
    :param min_logfc: float. The minimum log fold change, default is 3.
    """
    if batch_key is None:
        joint_adata_SM = joint_adata[:, joint_adata.var.type == "SM"]
        joint_adata_ST = joint_adata[:, joint_adata.var.type == "ST"]
        spatial_neighbors(joint_adata_ST)
        spatial_autocorr(joint_adata_ST, mode="moran", genes=joint_adata_ST.var_names)
        top_ST = (
            joint_adata_ST.uns["moranI"][
                np.array(~joint_adata_ST.uns["moranI"]["I"].isna())
            ]
            .sort_values("I", ascending=False)
            .index[:n_top_genes]
        )
        #joint_adata_ST.var[add_key] = joint_adata_ST.var_names.isin(top_ST)
        spatial_neighbors(joint_adata_SM)
        spatial_autocorr(joint_adata_SM, mode="moran", genes=joint_adata_SM.var_names)
        top_SM = (
            joint_adata_SM.uns["moranI"][
                np.array(~joint_adata_SM.uns["moranI"]["I"].isna())
            ]
            .sort_values("I", ascending=False)
            .index[:n_top_metabolites]
        )
        #joint_adata_SM.var[add_key] = joint_adata_SM.var_names.isin(top_SM)
        ###concat var according the order of joint_adata
        top_SM_ST = top_ST.append(top_SM)
        ###concat var according the order of joint_adata
        joint_adata.var[add_key] = joint_adata.var_names.isin(top_SM_ST)
    else:
        #The individual Spatial_variables are calculated for each batch and then merged, keeping only the Spatial_variables that appear in at least 2 samples
        batch_list = joint_adata.obs[batch_key].unique()
        variable_occurrences = defaultdict(lambda: {'ST': 0, 'SM': 0})
        for batch in batch_list:
            joint_adata_batch = joint_adata[joint_adata.obs[batch_key] == batch]
            joint_adata_SM = joint_adata_batch[:, joint_adata_batch.var.type == "SM"]
            joint_adata_ST = joint_adata_batch[:, joint_adata_batch.var.type == "ST"]
            spatial_neighbors(joint_adata_ST)
            spatial_autocorr(joint_adata_ST, mode="moran", genes=joint_adata_ST.var_names)
            top_ST = (
                joint_adata_ST.uns["moranI"][
                    np.array(~joint_adata_ST.uns["moranI"]["I"].isna())
                ]
                .sort_values("I", ascending=False)
                .index[:n_top_genes]
            )
            for gene in top_ST:
                variable_occurrences[gene]['ST'] += 1
                
            spatial_neighbors(joint_adata_SM)
            spatial_autocorr(joint_adata_SM, mode="moran", genes=joint_adata_SM.var_names)
            top_SM = (
                joint_adata_SM.uns["moranI"][
                    np.array(~joint_adata_SM.uns["moranI"]["I"].isna())
                ]
                .sort_values("I", ascending=False)
                .index[:n_top_metabolites]
            )
            for meatabolite in top_SM:
                variable_occurrences[meatabolite]['SM'] += 1
            
        retained_spatial_variables = []
        for variable, counts in variable_occurrences.items():
            if counts['ST'] >= min_samples or counts['SM'] >= min_samples:
                retained_spatial_variables.append(variable)
        joint_adata_ST = joint_adata[:, joint_adata.var.type == "ST"]
        joint_adata_SM = joint_adata[:, joint_adata.var.type == "SM"]        
        batch_spatial_variable_ST = compute_batch_variable_genes_core(
            joint_adata_ST,
            batch_key,
            min_frac=min_frac,
            min_logfc=min_logfc,
        )
        batch_spatial_variable_SM = compute_batch_variable_genes_core(
            joint_adata_SM,
            batch_key,
            min_frac=min_frac,
            min_logfc=min_logfc,
        )
        # retained_spatial_variables_final remove batch_spatial_variable_ST and batch_spatial_variable_SM
        retained_spatial_variables_final = list(
            set(retained_spatial_variables)
            - set(batch_spatial_variable_ST)
            - set(batch_spatial_variable_SM)
        )
        joint_adata.var[add_key] = joint_adata.var_names.isin(retained_spatial_variables_final)

@ignore_warning(level="ignore")
def highly_variable_joint_adata_sm_st(
    joint_adata: AnnDataJointSMST,
    n_top_genes: int = 2000,
    n_top_metabolites: int = 800,
    add_key: str = "highly_variable",
    batch_key: Optional[str] = None,
    **kwargs
):
    """
    Calculate the highly variable genes and metabolites for the joint AnnData object and add the results to the AnnData object.
    
    :param joint_adata: AnnDataJointSMST. The joint AnnData object with SM and ST data.
    :param n_top_genes: int. The number of top genes, default is 2000.
    :param n_top_metabolites: int. The number of top metabolites, default is 800.
    :param add_key: str. The key for the highly variable genes and metabolites, default is "highly_variable".
    :param batch_key: Optional[str]. The batch key, default is None.
    :param **kwargs: dict. The keyword arguments for sc.pp.highly_variable_genes.
    """
    joint_adata_SM = joint_adata[:, joint_adata.var.type == "SM"]
    joint_adata_ST = joint_adata[:, joint_adata.var.type == "ST"]
    sc.pp.highly_variable_genes(joint_adata_ST, 
                                n_top_genes = n_top_genes,
                                batch_key = batch_key,
                                **kwargs)
    sc.pp.highly_variable_genes(joint_adata_SM, 
                                n_top_genes = n_top_metabolites, 
                                batch_key = batch_key,
                                **kwargs)
    top_ST = joint_adata_ST[:,joint_adata_ST.var.highly_variable].var_names
    top_SM = joint_adata_SM[:,joint_adata_SM.var.highly_variable].var_names
    top_SM_ST = top_ST.append(top_SM)
    joint_adata.var[add_key] = joint_adata.var_names.isin(top_SM_ST)
    
@ignore_warning(level="ignore")
def spatial_variable(
    adata: Union[AnnData, AnnDataSM, AnnDataST],
    *,
    layer: Optional[str] = None,
    n_top_variable: int = 2000, 
    add_key: str = "highly_variable_moranI",
    batch_key: Optional[str] = None,
    min_samples: int = 2,
    min_frac: float = 0.8,
    min_logfc: float = 3,
):
    """
    Calculate the spatial variables and add the results to the AnnData object.
    
    :param adata: AnnData. The AnnData object.
    :param layer: Optional[str]. The layer key, default is None.
    :param n_top_variable: int. The number of top variables, default is 2000.
    :param add_key: str. The key for the spatial variables, default is "highly_variable_moranI".
    :param batch_key: Optional[str]. The batch key, default is None.
    :param min_samples: int. The minimum number of samples, default is 2.
    :param min_frac: float. The minimum fraction, default is 0.8.
    :param min_logfc: float. The minimum log fold change, default is 3.
    """
    if batch_key is None:
        spatial_neighbors(adata)
        spatial_autocorr(
            adata, 
            mode = "moran", 
            genes = adata.var_names,
            layer = layer
        )
        s = (
            adata.uns["moranI"][np.array(~adata.uns["moranI"]["I"].isna())]
            .sort_values("I", ascending=False)
            .index[:n_top_variable]
        )
        adata.var[add_key] = adata.var_names.isin(s)
    else:
        batch_list = adata.obs[batch_key].unique()
        variable_occurrences = defaultdict(int)
        for batch in batch_list:
            adata_batch = adata[adata.obs[batch_key] == batch]
            spatial_neighbors(adata_batch)
            spatial_autocorr(
                adata_batch, 
                mode = "moran", 
                genes = adata_batch.var_names,
                layer = layer
            )
            s = (
                adata_batch.uns["moranI"][np.array(~adata_batch.uns["moranI"]["I"].isna())]
                .sort_values("I", ascending=False)
                .index[:n_top_variable]
            )
            for gene in s:
                variable_occurrences[gene] += 1
        retained_spatial_variables = []
        for variable, count in variable_occurrences.items():
            if count >= min_samples:
                retained_spatial_variables.append(variable)
        batch_spatial_variable = compute_batch_variable_genes_core(
            adata,
            batch_key,
            min_frac=min_frac,
            min_logfc=min_logfc,
        )
        retained_spatial_variables_final = list(
            set(retained_spatial_variables) - set(batch_spatial_variable)
        )
        adata.var[add_key] = adata.var_names.isin(retained_spatial_variables_final)


@ignore_warning(level="ignore")
def rank_gene_and_metabolite_groups(
    adata: AnnData, 
    var_object: str = "type", 
    use_raw: bool = False, 
    groupby_ST: str = "leiden",
    groupby_SM: str = "leiden",
    key_added_ST: str = "rank_genes_groups",
    key_added_SM: str = "rank_metabolites_groups",
    **kwargs
):
    """
    Rank gene and metabolite groups and add the results to the AnnData object.
    
    :param adata: AnnData. The AnnData object.
    :param var_object: str. The variable object, default is "type".
    :param use_raw: bool. The flag of using raw data, default is False.
    :param groupby_ST: str. The groupby key for ST data, default is "leiden".
    :param groupby_SM: str. The groupby key for SM data, default is "leiden".
    :param key_added_ST: str. The key for the rank gene groups, default is "rank_genes_groups".
    :param key_added_SM: str. The key for the rank metabolite groups, default is "rank_metabolites_groups".
    :param **kwargs: dict. The keyword arguments for sc.tl.rank_genes_groups.
    """
    adata_ST = sc.AnnData(
        X=adata.X[:, adata.var[var_object] == "ST"],
        obs=adata.obs,
        var=adata.var[adata.var[var_object] == "ST"],
        uns=adata.uns,
        obsm=adata.obsm,
        obsp=adata.obsp,
    )
    adata_SM = sc.AnnData(
        X=adata.X[:, adata.var[var_object] == "SM"],
        obs=adata.obs,
        var=adata.var[adata.var[var_object] == "SM"],
        uns=adata.uns,
        obsm=adata.obsm,
        obsp=adata.obsp,
    )
    if use_raw:
        adata_ST_raw = sc.AnnData(
            X=adata.raw.X[:, adata.var[var_object] == "ST"],
            obs=adata.obs,
            var=adata.raw.var[adata.var[var_object] == "ST"],
            uns=adata.uns,
            obsm=adata.obsm,
            obsp=adata.obsp,
        )
        adata_ST.raw = adata_ST_raw
        adata_SM_raw = sc.AnnData(
            X=adata.raw.X[:, adata.var[var_object] == "SM"],
            obs=adata.obs,
            var=adata.raw.var[adata.var[var_object] == "SM"],
            uns=adata.uns,
            obsm=adata.obsm,
            obsp=adata.obsp,
        )
        adata_SM.raw = adata_SM_raw
    sc.tl.rank_genes_groups(adata_ST, 
                            use_raw=use_raw,
                            groupby=groupby_ST,
                            key_added=key_added_ST,
                            **kwargs)
    sc.tl.rank_genes_groups(adata_SM,
                            use_raw=use_raw,
                            groupby=groupby_SM,
                            key_added=key_added_SM,
                            **kwargs)
    adata.uns[key_added_ST] = adata_ST.uns[key_added_ST]
    adata.uns[key_added_SM] = adata_SM.uns[key_added_SM]


@ignore_warning(level="ignore")
def corrcoef_stsm_inall(
    adata: AnnDataJointSMST, 
    inputlist=None, 
    list_type="gene", 
    use_raw=True, 
    ntop=10
):
    """
    Calculate the correlation coefficients between the ST and SM data in the joint AnnData object and add the results to the AnnData object uns as 'corrcoef_stsm_inall_top' and 'corrcoef_stsm_inall'.
    
    :param adata: AnnDataJointSMST. The joint AnnData object with SM and ST data.
    :param inputlist: Optional[list]. The input list, default is None.
    :param list_type: str. The list type, default is "gene".
    :param use_raw: bool. The flag of using raw data, default is True.
    :param ntop: int. The number of top genes or metabolites, default is 10.
    """
    if use_raw==True:
        _adata = adata.raw.to_adata()
    elif use_raw==False:
        _adata = adata
    results_df = pd.DataFrame()
    if list_type=="gene":
        l1 = inputlist
        l2 = _adata.var.index[_adata.var['type']=="SM"]
        corr_mat = np.zeros((len(l1),len(l2)))
        corr_mat = pd.DataFrame(corr_mat, index=l1, columns=l2)
        for gene in inputlist:
            ST_tmp = _adata.X.toarray()[:,_adata.var.index.isin([gene])]
            SM_X = _adata.X.toarray()[:,_adata.var['type']=="SM"]
            correlation_scores = []
            for i in range(SM_X.shape[1]):
                correlation_score = np.corrcoef(ST_tmp.squeeze(), SM_X[:, i])[0, 1]
                corr_mat.loc[gene,l2[i]] = correlation_score
                correlation_scores.append(correlation_score)
            metabolites = _adata.var.index[_adata.var['type']=="SM"]
            df = pd.DataFrame({'metabolite': metabolites, 'correlation_score': correlation_scores})
            df = df.sort_values(by='correlation_score', ascending=False)
            results_tmp_df = df.head(ntop)
            results_tmp_df['gene']=gene
            results_df = pd.concat([results_df, results_tmp_df])
    elif list_type=="metabolite":
        l1 = inputlist
        l2 = _adata.var.index[_adata.var['type']=="ST"]
        corr_mat = np.zeros((len(l1),len(l2)))
        corr_mat = pd.DataFrame(corr_mat, index=l1, columns=l2)
        for metabolite in inputlist:
            SM_tmp = _adata.X.toarray()[:,_adata.var.index.isin([metabolite])]
            ST_X = _adata.X.toarray()[:,_adata.var['type']=="ST"]
            correlation_scores = []
            for i in range(ST_X.shape[1]):
                correlation_score = np.corrcoef(SM_tmp.squeeze(), ST_X[:, i])[0, 1]
                corr_mat.loc[metabolite,l2[i]] = correlation_score
                correlation_scores.append(correlation_score)
            genes = _adata.var.index[_adata.var['type']=="ST"]
            df = pd.DataFrame({'gene': genes, 'correlation_score': correlation_scores})
            df = df.sort_values(by='correlation_score', ascending=False)
            results_tmp_df = df.head(ntop)
            results_tmp_df['metabolite']=metabolite
            results_df = pd.concat([results_df, results_tmp_df])
    adata.uns['corrcoef_stsm_inall_top'] = results_df
    adata.uns['corrcoef_stsm_inall'] = corr_mat  


@ignore_warning(level="ignore") 
def corrcoef_stsm_ingroup(
    adata: AnnDataJointSMST,
    inputlist=None,
    list_type="gene",
    groupby="leiden",
    use_raw=True,
    ntop=5
):
    """
    Calculate the correlation coefficients between the ST and SM data in the joint AnnData object in each group and add the results to the AnnData object uns as 'corrcoef_stsm_ingroup_top' and 'corrcoef_stsm_ingroup'.
    
    :param adata: AnnDataJointSMST. The joint AnnData object with SM and ST data.
    :param inputlist: Optional[list]. The input list, default is None.
    :param list_type: str. The list type, default is "gene".
    :param groupby: str. The groupby key, default is "leiden".
    :param use_raw: bool. The flag of using raw data, default is True.
    :param ntop: int. The number of top genes or metabolites, default is 5.
    
    """
    if use_raw==True:
        _adata = adata.raw.to_adata()
    elif use_raw==False:
        _adata = adata
    results_df = pd.DataFrame()
    results_dict = {}
    if list_type=="gene":
        for cluster in _adata.obs[groupby].unique():
            l1 = inputlist
            l2 = list(_adata.var.index[_adata.var['type']=="SM"])
            corr_mat = np.zeros((len(l1),len(l2)))
            for gene in inputlist:
                ST_tmp = _adata.X.toarray()[_adata.obs[groupby]==cluster,_adata.var.index==gene]
                row_mask = _adata.obs[groupby]==cluster
                SM_X = _adata.X.toarray()[row_mask][:,_adata.var['type']=="SM"]
                correlation_scores = []
                for i in range(SM_X.shape[1]):
                    correlation_score = np.corrcoef(ST_tmp.squeeze(), SM_X[:, i])[0, 1]
                    corr_mat[l1.index(gene),i] = correlation_score
                    correlation_scores.append(correlation_score)
                metabolites = _adata.var.index[_adata.var['type']=="SM"]
                df = pd.DataFrame({'metabolite': metabolites, 'correlation_score': correlation_scores})
                df = df.sort_values(by='correlation_score', ascending=False)
                results_tmp_df = df.head(ntop)
                results_tmp_df['gene']=gene
                results_tmp_df['cluster']=cluster
                results_df = pd.concat([results_df, results_tmp_df])
            results_tmp_dict = {"cluster":cluster,
                           "gene":l1,
                           "metabolite":l2,
                           "data":corr_mat}
            results_dict[cluster] = results_tmp_dict
    elif list_type=="metabolite":
        for cluster in _adata.obs[groupby].unique():
            l1 = inputlist
            l2 = list(_adata.var.index[_adata.var['type']=="ST"])
            corr_mat = np.zeros((len(l1),len(l2)))
            for metabolite in inputlist:
                SM_tmp = _adata.X.toarray()[_adata.obs[groupby]==cluster,_adata.var.index==metabolite]
                row_mask = _adata.obs[groupby]==cluster
                ST_X = _adata.X.toarray()[row_mask][:,_adata.var['type']=="ST"]
                correlation_scores = []
                for i in range(ST_X.shape[1]):
                    correlation_score = np.corrcoef(SM_tmp.squeeze(), ST_X[:, i])[0, 1]
                    corr_mat[l1.index(metabolite),i] = correlation_score
                    correlation_scores.append(correlation_score)
                genes = _adata.var.index[_adata.var['type']=="ST"]
                df = pd.DataFrame({'gene': genes, 'correlation_score': correlation_scores})
                df = df.sort_values(by='correlation_score', ascending=False)
                results_tmp_df = df.head(ntop)
                results_tmp_df['metabolite']=metabolite
                results_tmp_df['cluster']=cluster
                results_df = results_df.append(results_tmp_df)
            results_tmp_dict = {"cluster":cluster,
                           "gene":l1,
                           "metabolite":l2,
                           "data":corr_mat}
            results_dict[cluster] = results_tmp_dict
    adata.uns['corrcoef_stsm_ingroup_top'] = results_df
    adata.uns['corrcoef_stsm_ingroup'] = results_dict

def normalize_non_inplace(adata: Union[AnnData, AnnDataSM, AnnDataST]):
    import copy
    adata.layers['normalized'] = copy.deepcopy(adata.X)
    sc.pp.normalize_total(adata, layer='normalized')
    sc.pp.log1p(adata, layer='normalized')

def calculate_metabolite_enrichment(
    metabolite_list: List[str],
    cutoff: float = 0.05,
    type: str = "sub_class"
) -> pd.DataFrame:
    """
    Calculate the metabolite enrichment.
    
    :param metabolite_list: List[str]. The list of metabolites.
    :param cutoff: float. The cutoff, default is 0.05.
    :param type: str. The type, default is "sub_class".
    
    :return: pd.DataFrame. The metabolite enrichment results.
    """
    KEGG_df = pd.read_csv(MODULE_PATH / "../data/hmdb.csv", index_col=0)
    enrichment_results = []
    classes = KEGG_df['class'].unique()
    subclasses = KEGG_df['sub_class'].unique()
    
    if type == "sub_class":
        for cls in subclasses:
            metabolites_in_class = KEGG_df[KEGG_df['sub_class'] == cls]['accession']
            metabolites_in_class_set = set(metabolites_in_class)
            metabolites_in_class_set_overlap = set(metabolites_in_class_set.intersection(metabolite_list))
            k = len(metabolites_in_class_set.intersection(metabolite_list))
            K = len(metabolites_in_class_set)
            N = len(set(KEGG_df['accession']))
            n = len(metabolite_list)
            p_value = hypergeom.sf(k - 1, N, K, n)
            enrichment_results.append({
                'class': cls,
                'p_value': p_value,
                'overlap': k,
                'total_class_metabolites': K,
                'metaoblites_in_class': metabolites_in_class_set_overlap,
            })
            
    if type == "class":
         for cls in classes:
            metabolites_in_class = KEGG_df[KEGG_df['class'] == cls]['accession']
            metabolites_in_class_set = set(metabolites_in_class)
            metabolites_in_class_set_overlap = metabolites_in_class_set.intersection(metabolite_list)
            k = len(metabolites_in_class_set.intersection(metabolite_list))
            K = len(metabolites_in_class_set)
            N = len(set(KEGG_df['accession']))
            n = len(metabolite_list)
            p_value = hypergeom.sf(k - 1, N, K, n)
            enrichment_results.append({
                'class': cls,
                'p_value': p_value,
                'overlap': k,
                'total_class_metabolites': K,
                'metaoblites_in_class': metabolites_in_class_set_overlap,
            })

    enrichment_results_df = pd.DataFrame(enrichment_results)
    enrichment_results_cutoff_df = enrichment_results_df[enrichment_results_df['p_value'] < cutoff]
            
    return enrichment_results_cutoff_df

def spatial_distance_cluster(
    adata: AnnData,
    groupby: str = "leiden",
    spatial_key: str = "spatial",
    metric: str = "euclidean",
    use_raw: bool = False,
    **kwargs
) -> pd.DataFrame:
    """
    Calculate the spatial distance between clusters.
    
    :param adata: AnnData. The AnnData object.
    :param groupby: str. The groupby key, default is "leiden".
    :param spatial_key: str. The spatial key, default is "spatial".
    :param metric: str. The metric, default is "euclidean".
    :param use_raw: bool. The flag of using raw data, default is False.
    
    :return: pd.DataFrame. The spatial distance between clusters.
    """
    
    if use_raw:
        adata = adata.raw.to_adata()
    elif use_raw==False:
        adata = adata
    celltype_ls = adata.obs[groupby].unique()
    celltype_distance = []
    for i in celltype_ls:
        for j in celltype_ls:
            if i != j:
                celltype_i = adata[adata.obs[groupby] == i].obsm[spatial_key]
                celltype_j = adata[adata.obs[groupby] == j].obsm[spatial_key]
                distance = spatial.distance_matrix(celltype_i, celltype_j)
                celltype_distance.append({
                    'from': i,
                    'to': j,
                    'distance': distance.mean(),
                    'weight': 1/distance.mean(),
                })
    celltype_distance_df = pd.DataFrame(celltype_distance)
    #min max scale
    celltype_distance_df['scale_distance'] = (celltype_distance_df['distance'] - celltype_distance_df['distance'].min()) / (celltype_distance_df['distance'].max() - celltype_distance_df['distance'].min())
    celltype_distance_df['scale_weight'] = (celltype_distance_df['weight'] - celltype_distance_df['weight'].min()) / (celltype_distance_df['weight'].max() - celltype_distance_df['weight'].min())
    return celltype_distance_df
   
def calculate_dot_df(
    adata: AnnData,
    groupby: str = "leiden",
    spatial_key: str = "spatial",
    use_raw: bool = False,
    **kwargs
) -> pd.DataFrame:
    """
    Calculate the dot dataframe.
    
    :param adata: AnnData. The AnnData object.
    :param groupby: str. The groupby key, default is "leiden".
    :param spatial_key: str. The spatial key, default is "spatial".
    :param use_raw: bool. The flag of using raw data, default is False.
    
    :return: pd.DataFrame. The dot dataframe.
    """
    if use_raw:
        adata = adata.raw.to_adata()
    elif use_raw==False:
        adata = adata
    dot_df = pd.DataFrame()
    celltype_ls = adata.obs[groupby].unique()
    cell_number_total = adata.shape[0]
    for celltype in celltype_ls:
        celltype_df = adata[adata.obs[groupby] == celltype]
        dot_tmp_df = pd.DataFrame({
            'dot_name': [celltype],
            'cell_number': [celltype_df.shape[0]],
            'dot_size' : [100*(celltype_df.shape[0] / cell_number_total)]
        })
        dot_df = pd.concat([dot_df, dot_tmp_df])
    dot_df['scale_dot_size'] = (dot_df['dot_size'] - dot_df['dot_size'].min()) / (dot_df['dot_size'].max() - dot_df['dot_size'].min())
    return dot_df 

def merge_and_assign_var_data(
    joint_adata : AnnDataJointSMST,
    var_anno_df : pd.DataFrame,
    columns_to_assign : List[str]
    ):
    """
    Merge the var_anno_df with joint_adata.var and joint_adata.raw.var and assign the columns to joint_adata.var and joint_adata.raw.var.
    
    :param joint_adata: AnnDataJointSMST. The joint AnnData object with SM and ST data.
    :param var_anno_df: pd.DataFrame. The var annotation dataframe.
    :param columns_to_assign: List[str]. The columns to assign.
    
    return: None
    """
    # Merging with joint_adata.var
    var_df = joint_adata.var
    var_anno_df.index = var_anno_df['name'].values
    var_anno_df['name'] = var_anno_df['name'].astype(str)
    var_df['name'] = var_df['name'].astype(str)
    var_merge_df = var_df.merge(var_anno_df, on='name', how='left')
    
    # Assign columns from var_merge_df to joint_adata.var
    joint_adata.var[columns_to_assign] = var_merge_df[columns_to_assign].values
    
    # Merging with joint_adata.raw.var
    var_raw_df = joint_adata.raw.var
    var_anno_df.index = var_anno_df['name'].values
    var_anno_df['name'] = var_anno_df['name'].astype(str)
    var_raw_df['name'] = var_raw_df['name'].astype(str)
    var_merge_df = var_raw_df.merge(var_anno_df, on='name', how='left')
    
    # Assign columns from var_merge_df to joint_adata.raw.var
    joint_adata.raw.var[columns_to_assign] = var_merge_df[columns_to_assign].values
