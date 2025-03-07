from typing import List, NamedTuple, Optional, TYPE_CHECKING, Union, Tuple
import matplotlib
import matplotlib.pyplot as plt
import pyimzml
from pyimzml.ImzMLParser import ImzMLParser
import pandas as pd
import anndata as AnnData
import matplotlib.patches as mpatch
from adjustText import adjust_text
import seaborn as sns
import numpy as np
import tqdm

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import convolve2d
from scipy.stats import pearsonr
import networkx as nx

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.size"] = "10"
matplotlib.rcParams["font.weight"] = 100
matplotlib.rcParams["axes.linewidth"] = 2
matplotlib.rcParams["axes.edgecolor"] = "#000000"


from colour import Color
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
import scanpy as sc
import dotplot
from svgpathtools import parse_path
from ..util._decorators import ignore_warning
from ._utils import get_spatial_image
from .utils._fit_curve import fit_p_value, fit_curve, valid_curves, normalize_func
from ..preprocess._preprocess import spatial_distance_cluster, calculate_dot_df
from ..util._classes import AnnDataSM, AnnDataST, AnnDataJointSMST


def make_colormap(
    colors: list,
    show_palette: bool = False,
):
    """
    Make a colormap from a list of colors.

    :param colors: list of colors to use in the colormap.
    :param show_palette: whether to display the colormap as a palette, default is False.

    :return: LinearSegmentedColormap object.

    """
    color_ramp = LinearSegmentedColormap.from_list(
        "my_list", [Color(c1).rgb for c1 in colors]
    )
    if show_palette:
        plt.figure(figsize=(15, 3))
        plt.imshow(
            [list(np.arange(0, len(colors), 0.1))],
            interpolation="nearest",
            origin="lower",
            cmap=color_ramp,
        )
        plt.xticks([])
        plt.yticks([])
    return color_ramp


def showionimage(p, mz_value=800, tol=200, z=1):
    im = pyimzml.ImzMLParser.getionimage(p, mz_value, tol)
    plt.imshow(im).set_interpolation("nearest")
    plt.colorbar()
    plt.show()


def create_fig(
    figsize: tuple = (8, 4),
):
    """
    Create a figure with the specified size and axis properties.

    :param figsize: tuple specifying the size of the figure, default is (8, 4).

    :return: figure and axis objects.
    """
    fig, ax = plt.subplots()
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    # ax.spines['bottom'].set_color('none')
    # ax.spines['left'].set_color('none')
    for line in ax.yaxis.get_ticklines():
        line.set_markersize(5)
        line.set_color("#585958")
        line.set_markeredgewidth(0.5)
    for line in ax.xaxis.get_ticklines():
        line.set_markersize(5)
        line.set_markeredgewidth(0.5)
        line.set_color("#585958")
    ax.set_xbound(0, 10)
    ax.set_ybound(0, 10)
    fig.set_size_inches(figsize)
    return fig, ax


def create_subplots(nrow: int, ncol: int, figsize: tuple = (8, 4)):
    """
    Create a figure with the specified size and axis properties.

    :param nrow: number of rows in the subplot.
    :param ncol: number of columns in the subplot.
    :param figsize: tuple specifying the size of the figure, default is (8, 4).

    :return: figure and axis objects.
    """
    fig, axes = plt.subplots(nrow, ncol)
    for ax in axes.flatten():
        ax.spines["right"].set_color("none")
        ax.spines["top"].set_color("none")
        for line in ax.yaxis.get_ticklines():
            line.set_markersize(5)
            line.set_color("#585958")
            line.set_markeredgewidth(0.5)
        for line in ax.xaxis.get_ticklines():
            line.set_markersize(5)
            line.set_markeredgewidth(0.5)
            line.set_color("#585958")
    fig.set_size_inches(figsize)
    return fig, axes


def plot_spot_sm_st(
    adata_SM: AnnDataSM,
    adata_ST: AnnDataST,
    SM_spatial_key: str = "spatial",
    ST_spatial_key: str = "spatial",
    stacked: bool = False,
    ST_color: str = "#C7C8CC",
    SM_color: str = "#C499BA",
    s: int = 10,
    **kwargs,
):
    """
    Plot the spatial distribution of spots from Spatial Transcriptomics and Spatial Metabolomics data.

    :param adata_SM: AnnData object containing the Spatial Metabolomics data.
    :param adata_ST: AnnData object containing the Spatial Transcriptomics data.
    :param SM_spatial_key: key in adata_SM.obsm where the spatial coordinates are stored, default is "spatial".
    :param ST_spatial_key: key in adata_ST.obsm where the spatial coordinates are stored, default is "spatial".
    :param stacked: whether to plot the data in a single plot or side-by-side, default is False.
    :param ST_color: color to use for the Spatial Transcriptomics data, default is "#C7C8CC".
    :param SM_color: color to use for the Spatial Metabolomics data, default is "#C499BA".
    :param s: size of the scatter points, default is 10.
    :param kwargs: additional keyword arguments to pass to the scatter function.

    :return: None.
    """
    if stacked:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.scatter(
            adata_ST.obsm[ST_spatial_key][:, 0],
            adata_ST.obsm[ST_spatial_key][:, 1],
            c=ST_color,
            s=s,
            **kwargs,
        )
        ax.scatter(
            adata_SM.obsm[SM_spatial_key][:, 0],
            adata_SM.obsm[SM_spatial_key][:, 1],
            c=SM_color,
            s=s,
            **kwargs,
        )
        # add legend
        ax.legend(["Spatial Transcriptomics", "Spatial Metabolomics"])
    else:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].scatter(
            adata_ST.obsm[ST_spatial_key][:, 0],
            adata_ST.obsm[ST_spatial_key][:, 1],
            c=ST_color,
            s=10,
            **kwargs,
        )
        # add label tittle
        ax[0].set_title("Spatial Transcriptomics")
        ax[1].scatter(
            adata_SM.obsm[SM_spatial_key][:, 0],
            adata_SM.obsm[SM_spatial_key][:, 1],
            c=SM_color,
            s=10,
            **kwargs,
        )
        # add label tittle
        ax[1].set_title("Spatial Metabolomics")


def plot_newdot_sm_st(
    new_dot_in_df: pd.DataFrame,
    adata_SM: AnnDataSM,
    adata_ST: AnnDataST,
    ST_spatial_key: str = "spatial",
    SM_spatial_key: str = "spatial",
    ST_color: str = "#C7C8CC",
    SM_color: str = "#C499BA",
    new_dot_color: str = "#FF0000",
):
    """
    Plot the spatial distribution of new spots in the Spatial Metabolomics and Spatial Transcriptomics data.

    :param new_dot_in_df: DataFrame containing the spatial coordinates of the new spots.
    :param adata_SM: AnnData object containing the Spatial Metabolomics data.
    :param adata_ST: AnnData object containing the Spatial Transcriptomics data.
    :param SM_spatial_key: key in adata_SM.obsm where the spatial coordinates are stored, default is "spatial".
    :param ST_spatial_key: key in adata_ST.obsm where the spatial coordinates are stored, default is "spatial".
    :param ST_color: color to use for the Spatial Transcriptomics data, default is "#C7C8CC".
    :param SM_color: color to use for the Spatial Metabolomics data, default is "#C499BA".
    :param new_dot_color: color to use for the new spots, default is "#FF0000".

    :return: None.
    """
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.scatter(
        new_dot_in_df["x_coord"],
        new_dot_in_df["y_coord"],
        c=new_dot_color,
        s=3,
    )
    ax.scatter(
        adata_ST.obsm[ST_spatial_key][:, 0],
        adata_ST.obsm[ST_spatial_key][:, 1],
        c=ST_color,
        s=3,
    )
    ax.scatter(
        adata_SM.obsm[SM_spatial_key][:, 0],
        adata_SM.obsm[SM_spatial_key][:, 1],
        c=SM_color,
        s=3,
    )
    # add legend
    ax.legend(["New Spots", "Spatial Transcriptomics", "Spatial Metabolomics"])


def plot_markerfeature(
    adata: AnnData,
    groupby: str,
    palette: dict,
    marker_feature_list: list,
    figsize: tuple = (10, 4),
    logfoldchanges_max: int = 10,
    logfoldchanges_min: int = -2,
    uns_key: str = "rank_genes_groups",
    save_path: str = None,
):
    """
    Create a scatter plot of the marker features in the AnnData object.

    :param adata: AnnData object containing the data.
    :param groupby: key in adata.obs to use for grouping the data.
    :param palette: dictionary containing the colors to use for each group.
    :param marker_feature_list: list of marker features to highlight.
    :param figsize: tuple specifying the size of the figure, default is (10, 4).
    :param logfoldchanges_max: maximum logfoldchanges value to consider, default is 10.
    :param logfoldchanges_min: minimum logfoldchanges value to consider, default is -2.
    :param uns_key: key in adata.uns where the data is stored, default is 'rank_genes_groups'.
    :param save_path: path to save the figure, default is None.

    :return: None.
    """
    # Initialize figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Calculate fraction of features for each group
    group_frac_dict = {}
    for size_obs_group in np.unique(adata.obs[groupby]):
        _adata = adata[adata.obs[groupby] == size_obs_group]
        frac = dict(
            zip(
                _adata.var.index,
                np.array((_adata.X > 0).sum(0) / _adata.shape[0]).flatten(),
            )
        )
        group_frac_dict[size_obs_group] = frac

    # Prepare result DataFrame
    result = []
    columns = list(pd.DataFrame(adata.uns[uns_key]["logfoldchanges"]).columns)
    for i, j, k in zip(
        pd.DataFrame(adata.uns[uns_key]["logfoldchanges"]).to_numpy(),
        pd.DataFrame(adata.uns[uns_key]["names"]).to_numpy(),
        pd.DataFrame(adata.uns[uns_key]["pvals_adj"]).to_numpy(),
    ):
        for e, (m, n, p) in enumerate(zip(i, j, -np.log10(k))):
            result.append((columns[e], m, n, p))

    result = pd.DataFrame(
        result, columns=[groupby, "logfoldchanges", "gene_name", "pvals_adj"]
    )
    result = result.sort_values(groupby)
    result["size"] = list(
        map(
            lambda x: group_frac_dict[x[0]][x[1]],
            zip(result[groupby], result["gene_name"]),
        )
    )
    # Make sure the groupby column uses pandas categorical type and add jitter
    result[groupby] = result[groupby].astype("category")
    result[groupby + "_codes"] = result[groupby].cat.codes + np.random.uniform(
        -0.3, 0.3, len(result)
    )

    # Create scatterplots for different conditions
    # Positive logfoldchanges, low absolute logfoldchanges, low pvals_adj
    _data = result[
        (result["logfoldchanges"] > 0)
        & (np.abs(result["logfoldchanges"]) < logfoldchanges_max)
        & (result["logfoldchanges"] > logfoldchanges_min)
        & (
            (np.abs(result["logfoldchanges"]) < 0.25)
            | (np.abs(result["pvals_adj"]) <= 2)
        )
    ]

    sns.scatterplot(
        data=_data,
        x=groupby + "_codes",
        y="logfoldchanges",
        ax=ax,
        marker="o",
        size="size",
        sizes=(0, 12),
        color="#D7D7D7",
        zorder=-1,
        legend=False,
    )

    # Positive logfoldchanges, moderate absolute logfoldchanges, high pvals_adj
    _data = result[
        (result["logfoldchanges"] > 0)
        & (result["logfoldchanges"] < logfoldchanges_max)
        & (result["logfoldchanges"] > logfoldchanges_min)
        & (np.abs(result["logfoldchanges"]) >= 0.25)
        & (np.abs(result["pvals_adj"]) > 2)
    ]

    for subtype in np.unique(_data[groupby]):
        st = sns.scatterplot(
            data=_data[_data[groupby] == subtype],
            x=groupby + "_codes",
            y="logfoldchanges",
            ax=ax,
            marker="o",
            size="size",
            sizes=(0, 12),
            color=palette[subtype],
            zorder=-1,
            legend=False,
        )
        
    # Negative logfoldchanges, low absolute logfoldchanges, low pvals_adj
    _data = result[
        (result["logfoldchanges"] < 0)
        & (result["logfoldchanges"] < logfoldchanges_max)
        & (result["logfoldchanges"] > logfoldchanges_min)
        & (
            (np.abs(result["logfoldchanges"]) < 0.25)
            | (np.abs(result["pvals_adj"]) <= 2)
        )
    ]

    sns.scatterplot(
        data=_data,
        x=groupby + "_codes",
        y="logfoldchanges",
        ax=ax,
        marker="o",
        size="size",
        sizes=(0, 12),
        color="#D7D7D7",
        zorder=-1,
        legend=False,
    )

    # Now you can safely use `st.get_children()` to retrieve the plot children
    all_collections = list(
        filter(
            lambda x: isinstance(x, matplotlib.collections.PathCollection),
            st.get_children(),
        )
    )
    all_text = []

    # Loop through each subtype and gene list and logfc is small logfoldchanges_max
    for e, (logfc_arr, name_arr, pval_arr, gene_list) in enumerate(
        zip(
            pd.DataFrame(adata.uns[uns_key]["logfoldchanges"]).to_numpy().T,
            pd.DataFrame(adata.uns[uns_key]["names"]).to_numpy().T,
            pd.DataFrame(adata.uns[uns_key]["pvals_adj"]).to_numpy().T,
            (
                marker_feature_list
                if marker_feature_list is not None
                else [[] for _ in range(len(name_arr))]
            ),  # Handle empty list case
        )
    ):
        # Loop through each gene name and its corresponding data
        for _, (logfc, gene, pval) in enumerate(zip(logfc_arr, name_arr, pval_arr)):
            # Check if the gene is in the list for the current subtype
            if np.abs(logfc) < logfoldchanges_max:
                if gene in gene_list:
                    # Find the coordinates in the result DataFrame
                    coords = result.loc[
                        (result[groupby].cat.codes == e) & (result["gene_name"] == gene),
                        groupby + "_codes",
                    ].iloc[0]
                    x, y = coords, logfc
                    # Add a scatter point

                    c_ls = list(np.unique(_data[groupby]))
                    ax.scatter(
                        x,
                        y,
                        lw=0.3,
                        edgecolor="black",
                        c=palette[c_ls[e]],
                        s=group_frac_dict[c_ls[e]][gene] * 12,
                    )

                    # Annotate the gene name
                    all_text.append(
                        ax.annotate(
                            xy=(x, y),
                            text=gene,
                            size=8,
                            xytext=(1, 40),
                            textcoords="offset points",
                            ha="center",
                            arrowprops=dict(
                                arrowstyle="-",
                                mutation_scale=0.005,
                                color="black",
                                lw=0.5,
                                ls="-",
                            ),
                        )
                    )
    # Modify the x-axis tick labels to show the group names
    x_tick_positions = range(len(c_ls))
    # Set the x-tick positions and labels explicitly
    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels(c_ls, rotation=45, ha="right")
    # Display the plot
    plt.show()

    # Save the figure if save_path is provided
    if save_path:
        fig.savefig(save_path)

def plot_marker_gene_metabolite(
    adata,
    groupby: str,
    palette: dict,
    ST_marker_feature_list: list,
    SM_marker_feature_list: list,
    figsize: tuple = (10, 4),
    save_path: str = None,
    logfoldchanges_max_ST: int = 10,
    logfoldchanges_max_SM: int = 10,
    key_ST: str = "rank_genes_groups",
    key_SM: str = "rank_metabolites_groups",
):
    """
    Create a scatter plot of the marker genes and metabolites in the AnnData object.

    :param adata: AnnDataSMST object containing the ST and SM data.
    :param groupby: key in adata.obs to use for grouping the data.
    :param palette: dictionary containing the colors to use for each group.
    :param ST_marker_feature_list: list of marker genes to highlight.
    :param SM_marker_feature_list: list of marker metabolites to highlight.
    :param figsize: tuple specifying the size of the figure, default is (10, 4).
    :param save_path: path to save the figure, default is None.
    :param logfoldchanges_max_ST: maximum logfoldchanges value to consider for ST, default is 10.
    :param logfoldchanges_max_SM: maximum logfoldchanges value to consider for SM, default is 10.
    :param key_ST: key in adata.uns where the ST data is stored, default is 'rank_genes_groups'.
    :param key_SM: key in adata.uns where the SM data is stored, default is 'rank_features_groups'.

    :return: None.
    """
    # Initialize figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    # Calculate fraction of features for each group
    group_frac_dict = {}
    for size_obs_group in np.unique(adata.obs[groupby]):
        _adata = adata[adata.obs[groupby] == size_obs_group]
        frac = dict(
            zip(
                _adata.var.index,
                np.array((_adata.X > 0).sum(0) / _adata.shape[0]).flatten(),
            )
        )
        group_frac_dict[size_obs_group] = frac

    # Prepare result ST DataFrame
    result_ST = []
    columns = list(pd.DataFrame(adata.uns[key_ST]["logfoldchanges"]).columns)
    for i, j, k in zip(
        pd.DataFrame(adata.uns[key_ST]["logfoldchanges"]).to_numpy(),
        pd.DataFrame(adata.uns[key_ST]["names"]).to_numpy(),
        pd.DataFrame(adata.uns[key_ST]["pvals_adj"]).to_numpy(),
    ):
        for e, (m, n, p) in enumerate(zip(i, j, -np.log10(k))):
            result_ST.append((columns[e], m, n, p))

    result_ST = pd.DataFrame(
        result_ST, columns=[groupby, "logfoldchanges", "gene_name", "pvals_adj"]
    )
    result_ST = result_ST.sort_values(groupby)
    result_ST["size"] = list(
        map(
            lambda x: group_frac_dict[x[0]][x[1]],
            zip(result_ST[groupby], result_ST["gene_name"]),
        )
    )
    result_ST[groupby] = result_ST[groupby].astype("category")
    result_ST[groupby + "_codes"] = result_ST[groupby].cat.codes + np.random.uniform(
        -0.3, 0.3, len(result_ST)
    )

    # Prepare result SM DataFrame
    result_SM = []
    columns = list(pd.DataFrame(adata.uns[key_SM]["logfoldchanges"]).columns)
    for i, j, k in zip(
        pd.DataFrame(adata.uns[key_SM]["logfoldchanges"]).to_numpy(),
        pd.DataFrame(adata.uns[key_SM]["names"]).to_numpy(),
        pd.DataFrame(adata.uns[key_SM]["pvals_adj"]).to_numpy(),
    ):
        for e, (m, n, p) in enumerate(zip(i, j, -np.log10(k))):
            result_SM.append((columns[e], m, n, p))

    result_SM = pd.DataFrame(
        result_SM, columns=[groupby, "logfoldchanges", "metabolite_name", "pvals_adj"]
    )
    result_SM = result_SM.sort_values(groupby)
    result_SM["size"] = list(
        map(
            lambda x: group_frac_dict[x[0]][x[1]],
            zip(result_SM[groupby], result_SM["metabolite_name"]),
        )
    )
    result_SM[groupby] = result_SM[groupby].astype("category")
    result_SM[groupby + "_codes"] = result_SM[groupby].cat.codes + np.random.uniform(
        -0.3, 0.3, len(result_SM)
    )

    # Create scatterplots for different conditions
    # Positive logfoldchanges, low absolute logfoldchanges, low pvals_adj
    _data = result_ST[
        (result_ST["logfoldchanges"] > 0)
        & (np.abs(result_ST["logfoldchanges"]) < logfoldchanges_max_ST)
        & (
            (np.abs(result_ST["logfoldchanges"]) < 0.25)
            | (np.abs(result_ST["pvals_adj"]) <= 2)
        )
    ]

    sns.scatterplot(
        data=_data,
        x=groupby + "_codes",
        y="logfoldchanges",
        ax=ax,
        marker="o",
        size="size",
        sizes=(0, 12),
        color="#D7D7D7",
        zorder=-1,
        legend=False,
    )
    # Positive logfoldchanges, moderate absolute logfoldchanges, high pvals_adj
    _data = result_ST[
        (result_ST["logfoldchanges"] > 0)
        & (np.abs(result_ST["logfoldchanges"]) < logfoldchanges_max_ST)
        & (np.abs(result_ST["logfoldchanges"]) >= 0.25)
        & (np.abs(result_ST["pvals_adj"]) > 2)
    ]

    for subtype in np.unique(_data[groupby]):
        st = sns.scatterplot(
            data=_data[_data[groupby] == subtype],
            x=groupby + "_codes",
            y="logfoldchanges",
            ax=ax,
            marker="o",
            size="size",
            sizes=(0, 12),
            color=palette[subtype],
            zorder=-1,
            legend=False,
        )
    # Positive logfoldchanges, low absolute logfoldchanges, low pvals_adj
    _data = result_SM[
        (result_SM["logfoldchanges"] > 0)
        & (np.abs(result_SM["logfoldchanges"]) < logfoldchanges_max_SM)
        & (
            (np.abs(result_SM["logfoldchanges"]) < 0.25)
            | (np.abs(result_SM["pvals_adj"]) <= 2)
        )
    ]
    _data["logfoldchanges"] = -_data["logfoldchanges"]
    sns.scatterplot(
        data=_data,
        x=groupby + "_codes",
        y="logfoldchanges",
        ax=ax,
        marker="o",
        size="size",
        sizes=(0, 12),
        color="#D7D7D7",
        zorder=-1,
        legend=False,
    )
    # Positive logfoldchanges, moderate absolute logfoldchanges, high pvals_adj
    _data = result_SM[
        (result_SM["logfoldchanges"] > 0)
        & (np.abs(result_SM["logfoldchanges"]) < logfoldchanges_max_SM)
        & (np.abs(result_SM["logfoldchanges"]) >= 0.25)
        & (np.abs(result_SM["pvals_adj"]) > 2)
    ]
    _data["logfoldchanges"] = -_data["logfoldchanges"]
    for subtype in np.unique(_data[groupby]):
        sm = sns.scatterplot(
            data=_data[_data[groupby] == subtype],
            x=groupby + "_codes",
            y="logfoldchanges",
            ax=ax,
            marker="o",
            size="size",
            sizes=(0, 12),
            color=palette[subtype],
            zorder=-1,
            legend=False,
        )
    # Now you can safely use `st.get_children()` to retrieve the plot children
    all_collections = list(
        filter(
            lambda x: isinstance(x, matplotlib.collections.PathCollection),
            st.get_children(),
        )
    )
    all_text = []

    # Loop through each subtype and gene list
    for e, (logfc_arr, name_arr, pval_arr, gene_list) in enumerate(
        zip(
            pd.DataFrame(adata.uns[key_ST]["logfoldchanges"]).to_numpy().T,
            pd.DataFrame(adata.uns[key_ST]["names"]).to_numpy().T,
            pd.DataFrame(adata.uns[key_ST]["pvals_adj"]).to_numpy().T,
            (
                ST_marker_feature_list
                if ST_marker_feature_list is not None
                else [[] for _ in range(len(name_arr))]
            ),  # Handle empty list case
        )
    ):
        # Loop through each gene name and its corresponding data
        for _, (logfc, gene, pval) in enumerate(zip(logfc_arr, name_arr, pval_arr)):
            # Check if the gene is in the list for the current subtype
            if np.abs(logfc) < logfoldchanges_max_ST:
                if gene in gene_list:
                    # Find the coordinates in the result DataFrame
                    coords = result_ST.loc[
                        (result_ST[groupby].cat.codes == e)
                        & (result_ST["gene_name"] == gene),
                        groupby + "_codes",
                    ].iloc[0]
                    x, y = coords, logfc
                    # Add a scatter point

                    c_ls = list(np.unique(_data[groupby]))
                    ax.scatter(
                        x,
                        y,
                        lw=0.3,
                        edgecolor="black",
                        c=palette[c_ls[e]],
                        s=group_frac_dict[c_ls[e]][gene] * 12,
                    )

                    # Annotate the gene name
                    all_text.append(
                        ax.annotate(
                            xy=(x, y),
                            text=gene,
                            size=8,
                            xytext=(1, 40),
                            textcoords="offset points",
                            ha="center",
                            arrowprops=dict(
                                arrowstyle="-",
                                mutation_scale=0.005,
                                color="black",
                                lw=0.5,
                                ls="-",
                            ),
                        )
                    )
    # Loop through each subtype and metabolite list
    for e, (logfc_arr, name_arr, pval_arr, gene_list) in enumerate(
        zip(
            pd.DataFrame(adata.uns[key_SM]["logfoldchanges"]).to_numpy().T,
            pd.DataFrame(adata.uns[key_SM]["names"]).to_numpy().T,
            pd.DataFrame(adata.uns[key_SM]["pvals_adj"]).to_numpy().T,
            (
                SM_marker_feature_list
                if SM_marker_feature_list is not None
                else [[] for _ in range(len(name_arr))]
            ),  # Handle empty list case
        )
    ):
        # Loop through each metabolite name and its corresponding data
        for _, (logfc, metabolite, pval) in enumerate(
            zip(logfc_arr, name_arr, pval_arr)
        ):
            if np.abs(logfc) < logfoldchanges_max_SM:
                if metabolite in gene_list:
                    coords = result_SM.loc[
                        (result_SM[groupby].cat.codes == e)
                        & (result_SM["metabolite_name"] == metabolite),
                        groupby + "_codes",
                    ].iloc[0]
                    x, y = coords, logfc
                    ax.scatter(
                        x,
                        -y,
                        lw=0.3,
                        edgecolor="black",
                        c=palette[c_ls[e]],
                        s=group_frac_dict[c_ls[e]][metabolite] * 12,
                    )
                    all_text.append(
                        ax.annotate(
                            xy=(x, -y),
                            text=metabolite,
                            size=8,
                            xytext=(1, -40),
                            textcoords="offset points",
                            ha="center",
                            arrowprops=dict(
                                arrowstyle="-",
                                mutation_scale=0.005,
                                color="black",
                                lw=0.5,
                                ls="-",
                            ),
                        )
                    )
    # Modify the y-axis tick labels to show absolute values
    ax.set_yticklabels(
        [
            np.abs(float(label.get_text().replace("−", "-")))
            for label in ax.get_yticklabels()
        ]
    )
    # Modify the x-axis tick labels to show the group names
    x_tick_positions = range(len(c_ls))
    # Set the x-tick positions and labels explicitly
    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels(c_ls, rotation=45, ha="right")

    # Display the plot
    plt.show()
    # Save the figure if save_path is provided
    if save_path:
        fig.savefig(save_path)


def plot_corrcoef_stsm_inall(
    adata: AnnDataJointSMST,
    row_cluster: bool = True,
    col_cluster: bool = True,
    figsize: tuple = (10, 10),
):
    """
    Create a clustermap of the correlation coefficients between genes and metabolites.

    :param adata: AnnDataJointSMST object containing the ST and SM data.
    :param row_cluster: whether to cluster the rows, default is True.
    :param col_cluster: whether to cluster the columns, default is True.
    :param figsize: tuple specifying the size of the figure, default is (10, 10).

    :return: None.
    """
    gene_ls = list(np.unique(adata.uns["corrcoef_stsm_inall_top"]["gene"]))
    metabolite_ls = list(np.unique(adata.uns["corrcoef_stsm_inall_top"]["metabolite"]))
    corr_mat = np.zeros((len(gene_ls), len(metabolite_ls)))
    pbar = tqdm.tqdm(total=len(gene_ls) * len(metabolite_ls))
    for i in gene_ls:
        for j in metabolite_ls:
            r = adata.uns["corrcoef_stsm_inall"].loc[i, j]
            if isinstance(r, np.float64):
                corr_mat[gene_ls.index(i), metabolite_ls.index(j)] = r
            elif len(r) > 0:
                corr_mat[gene_ls.index(i), metabolite_ls.index(j)] = r[0]
            pbar.update(1)
    pbar.close()
    palette = [
        "#6b9080",
        "#a4c3b2",
        "#cce3de",
        "#eaf4f4",
        "#f6fff8",
        "#ffe5d9",
        "#ffcad4",
        "#f4acb7",
        "#9d8189",
    ]
    sns.clustermap(
        pd.DataFrame(corr_mat, index=gene_ls, columns=metabolite_ls),
        cmap=make_colormap(palette),
        fmt=".1f",
        row_cluster=row_cluster,
        col_cluster=col_cluster,
        figsize=figsize,
    )


def plot_corrcoef_stsm_ingroup(
    adata: AnnDataJointSMST,
    row_cluster: bool = True,
    cluster: str = "cluster_0",
    col_cluster: bool = True,
    figsize: tuple = (10, 10),
):
    """
    Create a clustermap of the correlation coefficients between genes and metabolites in a specific cluster.
    
    :param adata: AnnDataJointSMST object containing the ST and SM data.
    :param row_cluster: whether to cluster the rows, default is True.
    :param cluster: cluster to consider, default is 'cluster_0'.
    :param col_cluster: whether to cluster the columns, default is True.
    :param figsize: tuple specifying the size of the figure, default is (10, 10).
    
    :return: None.
    """
    gene_ls = list(np.unique(adata.uns["corrcoef_stsm_ingroup_top"]["gene"]))
    tmp_df = adata.uns["corrcoef_stsm_ingroup_top"]
    metabolite_ls = list(
        np.unique(tmp_df["metabolite"][tmp_df.cluster == cluster].values)
    )
    corr_mat = np.zeros((len(gene_ls), len(metabolite_ls)))
    pbar = tqdm.tqdm(total=len(gene_ls) * len(metabolite_ls))
    corrcoef_stsm_inall_mat = adata.uns["corrcoef_stsm_ingroup"][cluster]["data"]
    index_ls = list(adata.uns["corrcoef_stsm_ingroup"][cluster]["gene"])
    colnames_ls = list(adata.uns["corrcoef_stsm_ingroup"][cluster]["metabolite"])
    corrcoef_stsm_inall_df = pd.DataFrame(
        corrcoef_stsm_inall_mat, index=index_ls, columns=colnames_ls
    )
    for i in gene_ls:
        for j in metabolite_ls:
            r = corrcoef_stsm_inall_df.loc[i, j]
            corr_mat[gene_ls.index(i), metabolite_ls.index(j)] = r
            pbar.update(1)
    pbar.close()
    palette = [
        "#6b9080",
        "#a4c3b2",
        "#cce3de",
        "#eaf4f4",
        "#f6fff8",
        "#ffe5d9",
        "#ffcad4",
        "#f4acb7",
        "#9d8189",
    ]
    sns.clustermap(
        pd.DataFrame(corr_mat, index=gene_ls, columns=metabolite_ls),
        cmap=make_colormap(palette),
        fmt=".1f",
        row_cluster=row_cluster,
        col_cluster=col_cluster,
        figsize=figsize,
    )


def _hex_to_rgb(value):
    value = value.lstrip("#")
    lv = len(value)
    return tuple(int(value[i : i + lv // 3], 16) / 255 for i in range(0, lv, lv // 3))


def _get_valid_cell_type(adata, min_spots: int):
    all_prediction_types = np.unique(adata.obs["celltype_predict"])
    valid_spots = []
    for i in all_prediction_types:
        num_spots = np.sum(adata.obs["celltype_predict"] == i)
        if num_spots >= min_spots:
            valid_spots.append(i)
    return valid_spots


def plot_spatial_deconvolution(
    adata: sc.AnnData,
    palettes: dict = None,
    min_spots: int = 10,
    img_alpha: float = 0.5,
    show_color_bar: bool = True,
    s: float = 10,
    vmin: float = 0.5,
    show: bool = False,
    valid_cell_types: list = None,
    ax=None,
    marker="H",
    use_pie_chart: bool = False,
    key_celltype_predict_proportions: str = "celltype_predict_proportions",
    key_celltype_predict: str = "celltype_predict",
):
    """
    Create a spatial plot of the cell type predictions.
    
    :param adata: AnnData object containing the data.
    :param palettes: dictionary containing the colors to use for each group, default is None.
    :param min_spots: minimum number of spots to consider, default is 10.
    :param img_alpha: alpha value for the image, default is 0.5.
    :param show_color_bar: whether to show the color bar, default is True.
    :param s: size of the spots, default is 10.
    :param vmin: minimum value for the spots, default is 0.5.
    :param show: whether to show the plot, default is False.
    :param valid_cell_types: list of valid cell types, default is None.
    :param ax: axis to use for the plot, default is None.
    :param marker: marker to use for the spots, default is 'H'.
    :param use_pie_chart: whether to use a pie chart, default is False.
    :param key_celltype_predict_proportions: key in adata.obs containing the cell type proportions, default is 'celltype_predict_proportions'.
    :param key_celltype_predict: key in adata.obs containing the cell type predictions, default is 'celltype_predict'.
    
    :return: None.
    """
    spatial_coord = adata.obsm["spatial"].copy()
    # fig, (ax1,ax2) = plt.subplots(1,2,figsize=(5,5))
    # fig.set_size_inches(10, 5)

    if palettes is None:
        palettes = sc.pl.palettes.godsnot_102

    if ax is None:
        fig, ax1 = plt.subplots(figsize=(5, 5))
    else:
        ax1 = ax
    img, scale = get_spatial_image(adata)
    ax1.imshow(img, alpha=img_alpha)
    if valid_cell_types is None:
        valid_cell_types = _get_valid_cell_type(adata, min_spots)

    if isinstance(list(palettes.values())[0], str):
        palettes = {k: _hex_to_rgb(v) for k, v in palettes.items()}

    if use_pie_chart:
        c = list(adata.obs.columns).index(key_celltype_predict_proportions)
        for e, (x, y) in enumerate(
            zip(spatial_coord[:, 0] * scale, spatial_coord[:, 1] * scale)
        ):
            ct = dict(
                zip(
                    list(palettes.keys()),
                    adata.obs.iloc[e].loc[list(palettes.keys())].to_numpy().flatten(),
                )
            )
            ax1.pie(
                ct.values(),
                radius=s * scale,
                colors=list(palettes.values()),
                center=(x, y),
            )
    else:
        if len(valid_cell_types) > 1:
            for i in valid_cell_types:
                alphas = adata.obs.loc[
                    adata.obs[key_celltype_predict] == i,
                    key_celltype_predict_proportions,
                ]
                alphas = (alphas - alphas.min()) / (
                    alphas.max() - alphas.min()
                ) * 0.5 + vmin
                ax1.scatter(
                    spatial_coord[adata.obs[key_celltype_predict] == i, 0] * scale,
                    spatial_coord[adata.obs[key_celltype_predict] == i, 1] * scale,
                    c=list(map(lambda x: tuple(list(palettes[i]) + [x]), alphas)),
                    s=s * scale,
                    lw=0,
                    marker=marker,
                )
            else:
                alphas = adata.obs.loc[
                    adata.obs[key_celltype_predict] == i,
                    key_celltype_predict_proportions,
                ]

    plt.grid(False)
    ax1.axis("off")

    if show_color_bar:
        sc.pl._tools.scatterplots._add_categorical_legend(
            ax1,
            pd.Categorical(valid_cell_types),
            palette=palettes,
            legend_loc="right margin",
            legend_fontweight="bold",
            legend_fontsize=None,
            legend_fontoutline=None,
            multi_panel=False,
            na_color="F7F7F7",
            na_in_legend=False,
            scatter_array=False,
        )
    if show:
        plt.show()
    return fig, ax1

@ignore_warning(level="ignore")
def plot_gene_corrcoef_sm_ingroup(
    adata: AnnDataJointSMST,
    genename: str,
    groupby: str = "leiden", 
    use_raw: bool = True,
    ntop: int = 5,
):
    """
    Create a dotplot of the correlation coefficients between genes and metabolites in a specific cluster.
    
    :param adata: AnnDataJointSMST object containing the ST and SM data.
    :param genename: name of the gene.
    :param groupby: key in adata.obs to use for grouping the data, default is 'leiden'.
    :param use_raw: whether to use the raw data, default is True.
    :param ntop: number of top genes to consider, default is 5.
    
    :return: None.
    """
    if use_raw == True:
        _adata = adata.raw.to_adata()
    elif use_raw == False:
        _adata = adata
    results_df = pd.DataFrame()
    l1 = adata.obs[groupby].unique()
    l2 = list(_adata.var.index[_adata.var["type"] == "SM"])
    corr_mat = np.zeros((len(l1), len(l2)))
    corr_mat = pd.DataFrame(corr_mat, index=l1, columns=l2)
    for cluster in _adata.obs[groupby].unique():
        ST_tmp = _adata.X.toarray()[
            _adata.obs[groupby] == cluster, _adata.var.index == genename
        ]
        row_mask = _adata.obs[groupby] == cluster
        SM_X = _adata.X.toarray()[row_mask][:, _adata.var["type"] == "SM"]
        correlation_scores = []
        for i in range(SM_X.shape[1]):
            correlation_score = np.corrcoef(ST_tmp.squeeze(), SM_X[:, i])[0, 1]
            corr_mat.loc[cluster, l2[i]] = correlation_score
            correlation_scores.append(correlation_score)
        metabolites = _adata.var.index[_adata.var["type"] == "SM"]
        df = pd.DataFrame(
            {"metabolite": metabolites, "correlation_score": correlation_scores}
        )
        df = df.sort_values(by="correlation_score", ascending=False)
        results_tmp_df = df.head(ntop)
        results_tmp_df["gene"] = genename
        results_tmp_df["cluster"] = cluster
        results_df = results_df.append(results_tmp_df)
    metabolite_top_ls = list(results_df["metabolite"].unique())
    results_out_df = corr_mat.loc[:, metabolite_top_ls]
    results_out_df["cluster"] = results_out_df.index
    results_out_long_df = pd.melt(
        results_out_df,
        id_vars=["cluster"],
        var_name=["metabolite"],
        value_name="correlation_score",
    )
    new_keys = {
        "item_key": "cluster",
        "group_key": "metabolite",
        "sizes_key": "correlation_score",
    }
    dp = dotplot.DotPlot.parse_from_tidy_data(results_out_long_df, **new_keys)
    sct = dp.plot(size_factor=100)


@ignore_warning(level="ignore")
def plot_metabolite_corrcoef_st_ingroup(
    adata: AnnDataJointSMST,
    metabolite: str,
    groupby: str = "leiden",
    use_raw: bool = True,
    ntop: int = 5,
):
    """
    Create a dotplot of the correlation coefficients between genes and metabolites in a specific cluster.
    
    :param adata: AnnDataJointSMST object containing the ST and SM data.
    :param metabolite: name of the metabolite.
    :param groupby: key in adata.obs to use for grouping the data, default is 'leiden'.
    :param use_raw: whether to use the raw data, default is True.
    :param ntop: number of top genes to consider, default is 5.
    
    :return: None.
    """
    if use_raw == True:
        _adata = adata.raw.to_adata()
    elif use_raw == False:
        _adata = adata
    results_df = pd.DataFrame()
    l1 = adata.obs[groupby].unique()
    l2 = list(_adata.var.index[_adata.var["type"] == "ST"])
    corr_mat = np.zeros((len(l1), len(l2)))
    corr_mat = pd.DataFrame(corr_mat, index=l1, columns=l2)
    for cluster in _adata.obs[groupby].unique():
        SM_tmp = _adata.X.toarray()[
            _adata.obs[groupby] == cluster, _adata.var.index == metabolite
        ]
        row_mask = _adata.obs[groupby] == cluster
        ST_X = _adata.X.toarray()[row_mask][:, _adata.var["type"] == "ST"]
        correlation_scores = []
        for i in range(ST_X.shape[1]):
            correlation_score = np.corrcoef(SM_tmp.squeeze(), ST_X[:, i])[0, 1]
            corr_mat.loc[cluster, l2[i]] = correlation_score
            correlation_scores.append(correlation_score)
        genes = _adata.var.index[_adata.var["type"] == "ST"]
        df = pd.DataFrame({"gene": genes, "correlation_score": correlation_scores})
        df = df.sort_values(by="correlation_score", ascending=False)
        results_tmp_df = df.head(ntop)
        results_tmp_df["metabolite"] = metabolite
        results_tmp_df["cluster"] = cluster
        results_df = results_df.append(results_tmp_df)
    gene_top_ls = list(results_df["gene"].unique())
    results_out_df = corr_mat.loc[:, gene_top_ls]
    results_out_df["cluster"] = results_out_df.index
    results_out_long_df = pd.melt(
        results_out_df,
        id_vars=["cluster"],
        var_name=["gene"],
        value_name="correlation_score",
    )
    new_keys = {
        "item_key": "cluster",
        "group_key": "gene",
        "sizes_key": "correlation_score",
    }
    dp = dotplot.DotPlot.parse_from_tidy_data(results_out_long_df, **new_keys)
    sct = dp.plot(size_factor=100)


@ignore_warning(level="ignore")
def plot_volcano_corrcoef_gene(
    adata: AnnDataJointSMST,
    metabolite: str,
    use_raw: bool = True,
    threshold: float = 0.25,
    nonmarker_size: int = 8,
    marker_size: int = 20,
    marker_alpha: float = 1,
    color_threshold: float = 0.25,
    color_above: str = "#D2649A",
    color_below: str = "#40A578",
    color_above_font: str = "#D2649A",
    color_below_font: str = "#40A578",
    fontsize: int = 8,
    color_neutral: str = "grey",
    figsize: tuple = (6, 5),
    title: str = "Volcano Plot for Correlation Coefficients",
    show: bool = True,
):
    """
    Create a volcano plot of the correlation coefficients between genes and metabolites.
    
    :param adata: AnnDataJointSMST object containing the ST and SM data.
    :param metabolite: name of the metabolite.
    :param use_raw: whether to use the raw data, default is True.
    :param threshold: threshold for the correlation coefficient, default is 0.25.
    :param nonmarker_size: size of the non-marker, default is 8.
    :param marker_size: size of the marker, default is 20.
    :param marker_alpha: alpha value for the marker, default is 1.
    :param color_threshold: threshold for the color, default is 0.25.
    :param color_above: color for values above the threshold, default is '#D2649A'.
    :param color_below: color for values below the threshold, default is '#40A578'.
    :param color_above_font: color for values above the threshold, default is '#D2649A'.
    :param color_below_font: color for values below the threshold, default is '#40A578'.
    :param fontsize: font size for the labels, default is 8.
    :param color_neutral: color for neutral values, default is 'grey'.
    :param figsize: tuple specifying the size of the figure, default is (6, 5).
    :param title: title of the plot, default is 'Volcano Plot for Correlation Coefficients'.
    :param show: whether to show the plot, default is True.
    
    :return: None.
    """
    if use_raw:
        _adata = adata.raw.to_adata()
    else:
        _adata = adata

    # Calculate correlation score and p-values
    SM_X = _adata.X.toarray()[:, _adata.var.index == metabolite]
    ST_X = _adata.X.toarray()[:, _adata.var["type"] == "ST"]
    correlation_scores = []
    p_values = []

    for i in range(ST_X.shape[1]):
        correlation_score, p_value = pearsonr(SM_X.squeeze(), ST_X[:, i])
        correlation_scores.append(correlation_score)
        p_values.append(p_value)

    genes = _adata.var.index[_adata.var["type"] == "ST"]
    df = pd.DataFrame(
        {"gene": genes, "correlation_score": correlation_scores, "p_value": p_values}
    )
    df = df.sort_values(by="correlation_score", ascending=False)
    results_df = df

    # Plot volcano plot
    plt.figure(figsize=figsize)
    colors = np.where(
        results_df["correlation_score"] > color_threshold,
        color_above,
        np.where(
            results_df["correlation_score"] < -color_threshold,
            color_below,
            color_neutral,
        ),
    )
    sizes = np.where(
        np.abs(results_df["correlation_score"]) > threshold, marker_size, nonmarker_size
    )  # Adjust marker sizes
    plt.scatter(
        results_df["correlation_score"],
        -np.log10(results_df["p_value"]),
        c=colors,
        s=sizes,
        alpha=marker_alpha,
    )
    plt.xlabel("Correlation coefficient (r)")
    plt.ylabel("-log10(P-value)")
    plt.title(title)
    plt.axvline(x=threshold, color=color_above, linestyle="--")
    plt.axvline(x=-threshold, color=color_below, linestyle="--")

    # Add labels to significant points
    for i, row in results_df.iterrows():
        if np.abs(row["correlation_score"]) > threshold:
            if row["correlation_score"] > threshold:
                plt.text(
                    row["correlation_score"] + 0.02,
                    -np.log10(row["p_value"]),
                    row["gene"],
                    fontsize=fontsize,
                    color=color_above_font,
                    ha="left",
                    va="center",
                )
            else:
                plt.text(
                    row["correlation_score"] - 0.02,
                    -np.log10(row["p_value"]),
                    row["gene"],
                    fontsize=fontsize,
                    color=color_below_font,
                    ha="right",
                    va="center",
                )
    if show == True:
        plt.show()


@ignore_warning(level="ignore")
def plot_volcano_corrcoef_metabolite(
    adata: AnnDataJointSMST,
    gene: str,
    use_raw: bool = True,
    threshold: float = 0.25,
    nonmarker_size: int = 8,
    marker_size: int = 20,
    marker_alpha: float = 1,
    color_threshold: float = 0.25,
    color_above: str = "#D2649A",
    color_below: str = "#40A578",
    color_above_font: str = "#D2649A",
    color_below_font: str = "#40A578",
    fontsize: int = 8,
    color_neutral: str = "grey",
    figsize: tuple = (6, 5),
    title: str = "Volcano Plot for Correlation Coefficients",
    show: bool = True,
):
    """
    Create a volcano plot of the correlation coefficients between genes and metabolites.
    
    :param adata: AnnDataJointSMST object containing the ST and SM data.
    :param gene: name of the gene.
    :param use_raw: whether to use the raw data, default is True.
    :param threshold: threshold for the correlation coefficient, default is 0.25.
    :param nonmarker_size: size of the non-marker, default is 8.
    :param marker_size: size of the marker, default is 20.
    :param marker_alpha: alpha value for the marker, default is 1.
    :param color_threshold: threshold for the color, default is 0.25.
    :param color_above: color for values above the threshold, default is '#D2649A'.
    :param color_below: color for values below the threshold, default is '#40A578'.
    :param color_above_font: color for values above the threshold, default is '#D2649A'.
    :param color_below_font: color for values below the threshold, default is '#40A578'.
    :param fontsize: font size for the labels, default is 8.
    :param color_neutral: color for neutral values, default is 'grey'.
    :param figsize: tuple specifying the size of the figure, default is (6, 5).
    :param title: title of the plot, default is 'Volcano Plot for Correlation Coefficients'.
    :param show: whether to show the plot, default is True.
    
    :return: None.
    """
    if use_raw:
        _adata = adata.raw.to_adata()
    else:
        _adata = adata

    # Calculate correlation score and p-values
    SM_X = _adata.X.toarray()[:, _adata.var["type"] == "SM"]
    ST_X = _adata.X.toarray()[:, _adata.var.index == gene]
    correlation_scores = []
    p_values = []

    for i in range(SM_X.shape[1]):
        correlation_score, p_value = pearsonr(SM_X[:, i], ST_X.squeeze())
        correlation_scores.append(correlation_score)
        p_values.append(p_value)

    metabolites = _adata.var.index[_adata.var["type"] == "SM"]
    df = pd.DataFrame(
        {
            "metabolite": metabolites,
            "correlation_score": correlation_scores,
            "p_value": p_values,
        }
    )
    df = df.sort_values(by="correlation_score", ascending=False)
    results_df = df

    # Plot volcano plot
    plt.figure(figsize=figsize)
    colors = np.where(
        results_df["correlation_score"] > color_threshold,
        color_above,
        np.where(
            results_df["correlation_score"] < -color_threshold,
            color_below,
            color_neutral,
        ),
    )
    sizes = np.where(
        np.abs(results_df["correlation_score"]) > threshold, marker_size, nonmarker_size
    )  # Adjust marker sizes
    plt.scatter(
        results_df["correlation_score"],
        -np.log10(results_df["p_value"]),
        c=colors,
        s=sizes,
        alpha=marker_alpha,
    )
    plt.xlabel("Correlation coefficient (r)")
    plt.ylabel("-log10(P-value)")
    plt.title(title)
    plt.axvline(x=threshold, color=color_above, linestyle="--")
    plt.axvline(x=-threshold, color=color_below, linestyle="--")

    # Add labels to significant points
    for i, row in results_df.iterrows():
        if np.abs(row["correlation_score"]) > threshold:
            if row["correlation_score"] > threshold:
                plt.text(
                    row["correlation_score"] + 0.02,
                    -np.log10(row["p_value"]),
                    row["metabolite"],
                    fontsize=fontsize,
                    color=color_above_font,
                    ha="left",
                    va="center",
                )
            else:
                plt.text(
                    row["correlation_score"] - 0.02,
                    -np.log10(row["p_value"]),
                    row["metabolite"],
                    fontsize=fontsize,
                    color=color_below_font,
                    ha="right",
                    va="center",
                )
    if show == True:
        plt.show()


def plot_features_trajectory(
    adata: AnnData,
    features: list,
    bins: int=100,
    palette: Union[dict, list]=None,
    figsize: tuple=(10, 5),
    scale: bool=False,
    save_path: str=None,
):
    """
    Create a line plot of the mean values of features along a trajectory.
    
    :param adata: AnnData object containing the data.
    :param features: list of features to plot.
    :param bins: number of bins to use for grouping the data, default is 100.
    :param palette: dictionary or list of colors to use for the features, default is None.
    :param figsize: tuple specifying the size of the figure, default is (10, 5).
    :param scale: whether to scale the values, default is False.
    :param save_path: path to save the plot, default is None.
    
    :return: None.
    """
    # Define default palette as a list if none is provided
    default_palette = [
        "#1f77b4",
        "#ff7f0e",
        "#279e68",
        "#d62728",
        "#aa40fc",
        "#8c564b",
        "#e377c2",
        "#b5bd61",
        "#17becf",
        "#aec7e8",
        "#ffbb78",
        "#98df8a",
        "#ff9896",
        "#c5b0d5",
        "#c49c94",
        "#f7b6d2",
        "#dbdb8d",
        "#9edae5",
        "#ad494a",
        "#8c6d31",
    ]

    # Use the default palette if none is provided
    if palette is None:
        palette = default_palette

    # Verify each feature exists in the adata object
    for feature in features:
        if feature not in adata.var.index:
            raise ValueError(f"Feature '{feature}' not found in adata.")

    plt.figure(figsize=figsize)

    # Iterate over each feature and plot
    for idx, feature in enumerate(features):
        # Get feature data and locations
        feature_index = list(adata.var.index).index(feature)
        arr = (
            adata.layers["normalized"][
                adata.uns["trajectory"]["trajectory_1"]["indices"], feature_index
            ]
            .toarray()
            .flatten()
        )
        locations = np.array(adata.uns["trajectory"]["trajectory_1"]["locations"])

        # Filter out zero values
        locations = locations[arr != 0]
        arr = arr[arr != 0]

        # Apply 0-1 scaling if scale parameter is True
        if scale:
            arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

        # Create a DataFrame with locations and values
        df = pd.DataFrame({"locations": locations, "value": arr})

        # Group data into bins and calculate mean and std
        df["bin"] = pd.cut(df["locations"], bins)
        grouped_data = df.groupby("bin")["value"].agg(["mean", "std"])

        # Calculate midpoints of bins
        x = [(interval.left + interval.right) / 2 for interval in grouped_data.index]
        mean_values = grouped_data["mean"]
        std_values = grouped_data["std"]

        # Determine color from palette
        if isinstance(palette, dict):
            color = palette.get(feature, default_palette[idx % len(default_palette)])
        else:
            color = palette[idx % len(palette)]

        # Plot line and shaded region
        sns.lineplot(x=x, y=mean_values, label=feature, color=color)
        plt.fill_between(
            x,
            mean_values - std_values,
            mean_values + std_values,
            color=color,
            alpha=0.2,
        )

    # Set axis labels and show legend
    plt.xlabel("Location")
    plt.ylabel("Mean Value")
    plt.legend()

    if save_path:
        plt.savefig(save_path)
    # Show the plot
    plt.show()


def _unique_preserve_order(arr):
    unique_values, indices = np.unique(arr, return_index=True)
    sorted_indices = np.sort(indices)
    unique_elements_preserved_order = arr[sorted_indices]
    return unique_elements_preserved_order


def _smooth_rows(X, window_size):
    # Define the kernel for smoothing
    kernel = np.ones((1, window_size)) / window_size
    smoothed_X = convolve2d(X, kernel, mode="same", boundary="wrap")
    return smoothed_X


def plot_clustermap_with_smoothing(
    adata: AnnData,
    window_size: int=200,
    cmap: str ="vlag",
    feature_top: int=10,
    key: str="rank_genes_groups",
    save_path: str=None,
    figsize: tuple=(10, 10),
    return_data: bool=False,
    **kwargs,
):
    """
    Create a clustermap of the top features with smoothing applied.
    
    :param adata: AnnData object containing the data.
    :param window_size: size of the window for smoothing, default is 200.
    :param cmap: colormap to use for the clustermap, default is 'vlag'.
    :param feature_top: number of top features to consider, default is 10.
    :param key: key in adata.uns to use for the features, default is 'rank_genes_groups'.
    :param save_path: path to save the plot, default is None.
    :param figsize: tuple specifying the size of the figure, default is (10, 10).
    :param return_data: whether to return the data, default is False.
    
    :return: None.
    """
    varnames = _unique_preserve_order(
        pd.DataFrame(adata.uns[key]["names"]).head(feature_top).to_numpy().T.flatten()
    )

    X = adata[:, varnames].layers["normalized"].toarray()
    X = _smooth_rows(X.T, window_size)

    # Generate a clustermap with specified settings
    g = sns.clustermap(
        pd.DataFrame(X, index=varnames),
        row_cluster=False,
        col_cluster=False,
        standard_scale="var",  # Standardize by columns (1)
        cmap=cmap,
        **kwargs,
    )
    g.ax_heatmap.set_xticklabels([])
    plt.gcf().set_size_inches(figsize)
    if save_path:
        plt.savefig(save_path)
    plt.show()
    # return dict format
    if return_data:
        data_dict = {"X": X, "varnames": varnames}
        return data_dict


def plot_trajectory_with_arrows(
    adata: AnnData,
    path_key: str = "trajectory_1",
    img_key: str = "scaledres",
    color: str = "trajectory_1",
    fig=None,
    ax=None,
    arrow_head_width: int = 15,
    arrow_width: float = 0.05,
    show: bool = False,
    **kwargs,
):
    """
    Create a spatial plot of the trajectory with arrows.
    
    :param adata: AnnData object containing the data.
    :param path_key: key in adata.uns containing the path, default is 'trajectory_1'.
    :param img_key: key in adata.uns containing the image, default is 'scaledres'.
    :param color: key in adata.obs containing the color, default is 'trajectory_1'.
    :param fig: figure to use for the plot, default is None.
    :param ax: axis to use for the plot, default is None.
    :param arrow_head_width: width of the arrow head, default is 15.
    :param arrow_width: width of the arrow, default is 0.05.
    :param show: whether to show the plot, default is False.
    
    :return: None.
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots()

    sc.pl.spatial(adata, color=color, ax=ax, show=show, img_key=img_key, **kwargs)
    path = parse_path(adata.uns["trajectory"][path_key]["path"])

    for i in range(len(path)):
        start = path[i].start
        end = path[i].end
        ax.arrow(
            start.real,
            start.imag,
            end.real - start.real,
            end.imag - start.imag,
            head_width=arrow_head_width,
            width=arrow_width,
            edgecolor="white",
        )


def plot_network(
    adata: AnnData,
    groupby: str = "leiden",
    spatial_key: str = "spatial",
    use_raw: bool = False,
    palette: dict = None,
    edge_use_scale: bool = True,
    node_use_scale: bool = True,
    node_scale_factor: float = 10.0,
    edge_weight_threshold: float = 0.1,
    edge_scale_factor: float = 1.0,
    seed: int = None,
    top_n_neighbors: int = 5,
    show_weight: bool = False,
    show_labels: bool = True,
    node_min_size: float = 1.0,
    split_by: str = None,
    return_data: bool = False,
    show: bool = True,
    iterations: int = 50,
):
    """
    Create a network plot of the data. And the network plot is generated based on the spatial distance and spot size.
    
    :param adata: AnnData object containing the data.
    :param groupby: key in adata.obs to use for grouping the data, default is 'leiden'.
    :param spatial_key: key in adata.obsm containing the spatial data, default is 'spatial'.
    :param use_raw: whether to use the raw data, default is False.
    :param palette: dictionary containing the colors to use for each group, default is None.
    :param edge_use_scale: whether to use scale for the edge, default is True.
    :param node_use_scale: whether to use scale for the node, default is True.
    :param node_scale_factor: factor to scale the node, default is 10.0.
    :param edge_weight_threshold: threshold for the edge weight, default is 0.1.
    :param edge_scale_factor: factor to scale the edge, default is 1.0.
    :param seed: seed to use for random number generation, default is None.
    :param top_n_neighbors: number of top neighbors to consider, default is 5.
    :param show_weight: whether to show the weight, default is False.
    :param show_labels: whether to show the labels, default is True.
    :param node_min_size: minimum size of the node, default is 1.0.
    :param split_by: key in adata.obs to use for splitting the data, default is None.
    :param return_data: whether to return the data, default is False.
    :param show: whether to show the plot, default is True.
    :param iterations: number of iterations to use for the layout, default is 50.
    
    :return: None.
    """
    np.random.seed(seed)
    if split_by is None:
        distance_df = spatial_distance_cluster(
            adata, groupby=groupby, spatial_key=spatial_key, useraw=use_raw
        )
        dot_df = calculate_dot_df(
            adata, groupby=groupby, spatial_key=spatial_key, use_raw=use_raw
        )
    else:
        distance_total_df = pd.DataFrame()
        dot_total_df = pd.DataFrame()
        for i in adata.obs[split_by].unique():
            adata_tmp = adata[adata.obs[split_by] == i]
            distance_df_tmp = spatial_distance_cluster(
                adata_tmp, groupby=groupby, spatial_key=spatial_key, use_raw=use_raw
            )
            dot_df_tmp = calculate_dot_df(
                adata_tmp, groupby=groupby, spatial_key=spatial_key, use_raw=use_raw
            )
            distance_total_df = pd.concat([distance_total_df, distance_df_tmp])
            dot_total_df = pd.concat([dot_total_df, dot_df_tmp])
            distance_total_df["split_by"] = i
            dot_total_df["split_by"] = i
        # group by split_by and calculate mean
        distance_total_df = distance_total_df[["from", "to", "distance"]]
        dot_total_df = dot_total_df[["dot_name", "cell_number"]]
        distance_df = distance_total_df.groupby(["from", "to"]).mean().reset_index()
        dot_df = dot_total_df.groupby(["dot_name"]).mean().reset_index()
        distance_df["weight"] = 1 / distance_df["distance"]
        distance_df["scale_distance"] = (
            distance_df["distance"] - distance_df["distance"].min()
        ) / (distance_df["distance"].max() - distance_df["distance"].min())
        distance_df["scale_weight"] = (
            distance_df["weight"] - distance_df["weight"].min()
        ) / (distance_df["weight"].max() - distance_df["weight"].min())
        dot_df["dot_size"] = dot_df["cell_number"] / dot_df["cell_number"].sum()
        dot_df["scale_dot_size"] = (dot_df["dot_size"] - dot_df["dot_size"].min()) / (
            dot_df["dot_size"].max() - dot_df["dot_size"].min()
        )
    distance_df_unique = distance_df.copy()
    distance_df_unique[["from", "to"]] = pd.DataFrame(
        np.sort(distance_df_unique[["from", "to"]], axis=1),
        index=distance_df_unique.index,
    )
    distance_df_unique.drop_duplicates(subset=["from", "to"], inplace=True)
    G = nx.Graph()
    for idx, row in distance_df_unique.iterrows():
        if edge_use_scale:
            weight = row["scale_weight"]
        else:
            weight = row["weight"]
        if weight >= edge_weight_threshold:
            G.add_edge(row["from"], row["to"], weight=weight)

    node_df = dot_df.copy()
    node_colors = {}
    node_sizes = {}
    if palette is None:
        palette = sc.pl.palettes.default_102
    if isinstance(palette, dict):
        node_colors = {node: palette[node] for node in G.nodes}
    else:
        node_colors = dict(zip(G.nodes, palette))
    if node_use_scale:
        node_sizes = {
            node: max(
                node_df[node_df["dot_name"] == node]["scale_dot_size"].iloc[0]
                * node_scale_factor,
                node_min_size,
            )
            for node in G.nodes
        }
    else:
        node_sizes = {
            node: max(
                node_df[node_df["dot_name"] == node]["dot_size"].iloc[0]
                * node_scale_factor,
                node_min_size,
            )
            for node in G.nodes
        }
    pos = nx.spring_layout(G, iterations=iterations)
    # from fa2 import ForceAtlas2
    # nx.to_scipy_sparse_matrix = nx.to_scipy_sparse_array
    # forceatlas2 = ForceAtlas2()
    # pos = forceatlas2.forceatlas2_networkx_layout(G, pos=None, iterations=50)

    edge_widths = [G[u][v]["weight"] * edge_scale_factor for u, v in G.edges]
    for node in G.nodes:
        neighbors = list(G.neighbors(node))
        if len(neighbors) > top_n_neighbors:
            sorted_neighbors = sorted(neighbors, key=lambda x: G[node][x]["weight"])
            for neighbor in sorted_neighbors[: len(sorted_neighbors) - top_n_neighbors]:
                G.remove_edge(node, neighbor)
    nx.draw(
        G,
        pos,
        with_labels=show_labels,
        node_color=list(node_colors.values()),
        node_size=[node_sizes[n] for n in G.nodes],
        width=edge_widths,
    )
    if show_weight:
        labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    if return_data:
        return G, node_df, distance_df_unique
    if show:
        plt.show()
