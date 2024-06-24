import scanpy as sc
import anndata as AnnData
from scipy.spatial import distance
import numpy as np
from ..util._classes import AnnDataSM, AnnDataST, AnnDataJointSMST

def filter_cells_sm(
    adata_SM: AnnDataSM,
    min_total_intensity: int = None,
    min_mean_intensity: int = None,
    max_total_intensity: int = None,
    max_mean_intensity: int = None
) -> AnnDataSM:
    """
    Filter cells based on total intensity and mean intensity.
    
    :param adata_SM: AnnDataSM. The AnnDataSM object.
    :param min_total_intensity: int, minimum total intensity, default None.
    :param min_mean_intensity: int, minimum mean intensity, default None.
    :param max_total_intensity: int, maximum total intensity, default None.
    :param max_mean_intensity: int, maximum mean intensity, default None.
    
    :return: AnnDataSM. The filtered AnnDataSM object.
    """
    if min_total_intensity is not None:
        adata_SM_filtered_min_total_intensity = adata_SM[adata_SM.obs['total_intensity']>min_total_intensity]
        n_filtered_min_total_intensity = adata_SM.shape[0] - adata_SM_filtered_min_total_intensity.shape[0]
        print(f"Filtered {n_filtered_min_total_intensity} cells with total intensity less than {min_total_intensity}")
    else:
        adata_SM_filtered_min_total_intensity = adata_SM
    if min_mean_intensity is not None:
        adata_SM_filtered_min_mean_intensity = adata_SM[adata_SM.obs['mean_intensity']>min_mean_intensity]
        n_filtered_min_mean_intensity = adata_SM.shape[0] - adata_SM_filtered_min_mean_intensity.shape[0]
        print(f"Filtered {n_filtered_min_mean_intensity} cells with mean intensity less than {min_mean_intensity}")
    else:
        adata_SM_filtered_min_mean_intensity = adata_SM
    if max_total_intensity is not None:
        adata_SM_filtered_max_total_intensity = adata_SM[adata_SM.obs['total_intensity']<max_total_intensity]
        n_filtered_max_total_intensity = adata_SM.shape[0] - adata_SM_filtered_max_total_intensity.shape[0]
        print(f"Filtered {n_filtered_max_total_intensity} cells with total intensity more than {max_total_intensity}")
    else:
        adata_SM_filtered_max_total_intensity = adata_SM
    if max_mean_intensity is not None:
        adata_SM_filtered_max_mean_intensity = adata_SM[adata_SM.obs['mean_intensity']<max_mean_intensity]
        n_filtered_max_mean_intensity = adata_SM.shape[0] - adata_SM_filtered_max_mean_intensity.shape[0]
        print(f"Filtered {n_filtered_max_mean_intensity} cells with mean intensity more than {max_mean_intensity}")
    else:
        adata_SM_filtered_max_mean_intensity = adata_SM
    overlap = adata_SM_filtered_min_total_intensity.obs.index.intersection(adata_SM_filtered_min_mean_intensity.obs.index).intersection(adata_SM_filtered_max_total_intensity.obs.index).intersection(adata_SM_filtered_max_mean_intensity.obs.index)
    adata_SM_filtered = adata_SM[overlap]
    return adata_SM_filtered
        
def filter_metabolites_sm(adata_SM:AnnData,
    min_cells: int = None,
    max_cells: int = None
) -> AnnDataSM:
    """
    Filter metabolites based on the number of cells.
    
    :param adata_SM: AnnDataSM. The AnnDataSM object.
    :param min_cells: int, minimum number of cells, default None.
    :param max_cells: int, maximum number of cells, default None.
    
    :return: AnnDataSM. The filtered AnnDataSM object.
    """
    if min_cells is not None:
        adata_SM_filtered_min_cells = adata_SM[:,adata_SM.X.sum(0)>min_cells]
        n_filtered_min_cells = adata_SM.shape[1] - adata_SM_filtered_min_cells.shape[1]
        print(f"Filtered {n_filtered_min_cells} metabolites with less than {min_cells} cells")
    else:
        adata_SM_filtered_min_cells = adata_SM
    if max_cells is not None:
        adata_SM_filtered_max_cells = adata_SM[:,adata_SM.X.sum(0)<max_cells]
        n_filtered_max_cells = adata_SM.shape[1] - adata_SM_filtered_max_cells.shape[1]
        print(f"Filtered {n_filtered_max_cells} metabolites with more than {max_cells} cells")
    else:
        adata_SM_filtered_max_cells = adata_SM
    overlap = adata_SM_filtered_min_cells.var.index.intersection(adata_SM_filtered_max_cells.var.index)
    adata_SM_filtered = adata_SM[:,overlap]
    return adata_SM_filtered

def calculate_min_diam(
    adata: AnnData,
    spatial_key: str = 'spatial',
) -> float:
    """
    Calculate the minimum distance between spots.
    
    :param adata: AnnData. The AnnData object.
    :param spatial_key: str, the key of spatial data in adata.obsm, default 'spatial'.
    
    :return: float. The minimum distance between spots.
    """
    spatial_df = adata.obsm[spatial_key]
    # Calculate the minimum distance between cells
    min_distance = float('inf')
    for i in range(spatial_df.shape[0]):
        for j in range(i+1, spatial_df.shape[0]):
            d = distance.euclidean(spatial_df[i], spatial_df[j])
            if d < min_distance:
                min_distance = d
    return min_distance

def calculate_qc_metrics_sm(
    adata_SM: AnnDataSM
):
    """
    Calculate the total intensity and mean intensity of each spot.
    
    :param adata_SM: AnnDataSM. The AnnDataSM object.
    """
    adata_SM.obs["total_intensity"] = np.array(adata_SM.X.sum(1)).flatten()
    adata_SM.obs["mean_intensity"] = np.array(adata_SM.X.mean(1)).flatten()

def add_obs_to_adata(
    object_adata: AnnData,
    adata: AnnData,
    obs_key: str,
):
    """
    Add obs to object_adata from adata.
    
    :param object_adata: AnnData. The AnnData object.
    :param adata: AnnData. The AnnData object.
    :param obs_key: str, the key of obs to be added.
    """
    obs_df = adata.obs.loc[:,["spot_name", "x_coord", "y_coord",obs_key]]
    joint_obs_df = object_adata.obs
    #merge by spot_name, x_coord, y_coord
    joint_obs_df = joint_obs_df.merge(obs_df, on=["spot_name", "x_coord", "y_coord"],how = "left")
    object_adata.obs[obs_key] = joint_obs_df[obs_key].values
 
def add_hvf_to_jointadata(
    joint_adata: AnnDataJointSMST,
    adata_SM: AnnDataSM,
    adata_ST: AnnDataST,
    hvf_key_SM: str = "highly_variable_moranI",
    hvf_key_ST: str = "highly_variable_moranI",
    hvf_key_joint: str = "highly_variable_moranI"
):
    """
    Add highly variable features to joint_adata.
    
    :param joint_adata: AnnDataJointSMST. The AnnDataJointSMST object.
    :param adata_SM: AnnDataSM. The AnnDataSM object.
    :param adata_ST: AnnDataST. The AnnDataST object.
    :param hvf_key_SM: str, the key of highly variable features in adata_SM.var, default "highly_variable_moranI".
    :param hvf_key_ST: str, the key of highly variable features in adata_ST.var, default "highly_variable_moranI".
    :param hvf_key_joint: str, the key of highly variable features in joint_adata.var, default "highly_variable_moranI".
    """
    SM_var_df =  adata_SM.var
    ST_var_df =  adata_ST.var
    SM_hvf = list(SM_var_df.index[SM_var_df[hvf_key_SM]].values)
    ST_hvf = list(ST_var_df.index[ST_var_df[hvf_key_ST]].values)
    #joint_hvf = SM_hvf + ST_hvf,and joint_adata.var['hvf_key_joint'] is True if the features are in joint_hvf
    joint_hvf = SM_hvf + ST_hvf
    joint_adata.var[hvf_key_joint] = False
    joint_adata.var.loc[joint_hvf,hvf_key_joint] = True
    
def removeHSP_MT_RPL_DNAJ(adata):
    return adata[:,list(map(lambda x: 
                            not (x.startswith("MT-") 
                                 or x.startswith("RPS") 
                                 or x.startswith("RPL") 
                                 or x.startswith("HSP") 
                                 or x.startswith("DNAJ")), 
                            adata.var.index))].copy()
    
def removeHsp_mt_Rpl_Dnaj(adata):
    return adata[:,list(map(lambda x: 
                            not (x.startswith("mt-") 
                                 or x.startswith("Rps") 
                                 or x.startswith("Rpl") 
                                 or x.startswith("Hsp") 
                                 or x.startswith("DNAJ")), 
                            adata.var.index))].copy()
    