import scanpy as sc
import pandas as pd
import squidpy as sq
import numpy as np
from scipy.spatial import distance_matrix
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    homogeneity_score,
    completeness_score,
    accuracy_score
)
from scipy.spatial.distance import pdist, squareform
import warnings
import gc
import scib
import random
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import mutual_info_classif
import psutil
import logging 

warnings.filterwarnings("ignore")

LOG_FILE_PATH = Path("./benchmark_evaluation_log.txt")
LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)


def log_memory_usage():
    """Log current memory usage"""
    process = psutil.Process()
    mem_info = process.memory_info()
    logging.info(f"Memory usage: {mem_info.rss / 1024 / 1024:.2f} MB")


def set_seed(seed=2024):
    """Set fixed random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    sc.settings.seed = seed
    sc.settings.verbosity = 0 # 0: errors, 1: warnings, 2: info, 3: hints
    sc.settings.set_figure_params(dpi=80, facecolor='white')
    logging.info(f"Seed set to {seed}")

def fx_1NN(i, location_in):
    """Calculate the distance to the nearest neighbor for a point"""
    location_in = np.array(location_in)
    dist_array = distance_matrix(location_in[i,:][None,:], location_in)[0,:]
    dist_array[i] = np.inf
    return np.min(dist_array)

def fx_kNN(i, location_in, k, cluster_in):
    """Check if majority of k nearest neighbors have different cluster labels"""
    location_in = np.array(location_in)
    cluster_in = np.array(cluster_in)
    dist_array = distance_matrix(location_in[i,:][None,:], location_in)[0,:]
    dist_array[i] = np.inf
    ind = np.argsort(dist_array)[:k]
    cluster_use = np.array(cluster_in)
    if np.sum(cluster_use[ind]!=cluster_in[i]) > (k/2):
        return 1
    else:
        return 0

def _compute_CHAOS(clusterlabel, location):
    """Compute CHAOS metric for spatial clustering evaluation"""
    clusterlabel = np.array(clusterlabel)
    location = np.array(location)
    if location.ndim == 1:
        location = location.reshape(-1,1)
    if location.shape[0] != len(clusterlabel):
        logging.error(f"_compute_CHAOS: Mismatch in lengths clusterlabel ({len(clusterlabel)}) and location ({location.shape[0]})")
        return np.nan

    if location.shape[0] > 0 and np.all(location == location[0,:]):
        logging.warning("_compute_CHAOS: All spatial locations are identical. CHAOS metric may not be meaningful. Returning NaN.")
        return np.nan

    matched_location = StandardScaler().fit_transform(location)

    clusterlabel_unique = np.unique(clusterlabel)
    dist_val = np.zeros(len(clusterlabel_unique))
    total_points_processed = 0

    for i, k_val in enumerate(clusterlabel_unique):
        location_cluster = matched_location[clusterlabel==k_val,:]
        if location_cluster.shape[0] <= 2:
            continue
        n_location_cluster = location_cluster.shape[0]
        results = [fx_1NN(j, location_cluster) for j in range(n_location_cluster)]
        dist_val[i] = np.sum(results)
        total_points_processed += n_location_cluster

    if total_points_processed == 0:
        logging.warning("_compute_CHAOS: No points processed (all clusters too small or other issue?). Returning NaN.")
        return np.nan
    return np.sum(dist_val)/total_points_processed


def _compute_PAS(clusterlabel, location):
    """Compute Proportion of Ambiguous Spots (PAS) metric"""
    clusterlabel = np.array(clusterlabel)
    location = np.array(location)
    if location.ndim == 1:
        location = location.reshape(-1,1)
    if location.shape[0] != len(clusterlabel):
        logging.error(f"_compute_PAS: Mismatch in lengths clusterlabel ({len(clusterlabel)}) and location ({location.shape[0]})")
        return np.nan
    if location.shape[0] == 0:
        logging.warning("_compute_PAS: Location array is empty. Returning NaN.")
        return np.nan

    matched_location = location
    results = [fx_kNN(i, matched_location, k=10, cluster_in=clusterlabel) for i in range(matched_location.shape[0])]
    return np.sum(results)/len(clusterlabel) if len(clusterlabel) > 0 else np.nan

def compute_ARI(adata, gt_key, pred_key):
    if gt_key not in adata.obs or pred_key not in adata.obs:
        logging.warning(f"compute_ARI: Key {gt_key} or {pred_key} not in adata.obs. Skipping.")
        return np.nan
    return adjusted_rand_score(adata.obs[gt_key], adata.obs[pred_key])

def compute_NMI(adata, gt_key, pred_key):
    if gt_key not in adata.obs or pred_key not in adata.obs:
        logging.warning(f"compute_NMI: Key {gt_key} or {pred_key} not in adata.obs. Skipping.")
        return np.nan
    return normalized_mutual_info_score(adata.obs[gt_key], adata.obs[pred_key])

def compute_CHAOS(adata, pred_key, spatial_key='spatial'):
    if pred_key not in adata.obs or spatial_key not in adata.obsm:
        logging.warning(f"compute_CHAOS (adata): Key {pred_key} or {spatial_key} not in adata.obs/obsm. Skipping.")
        return np.nan
    return _compute_CHAOS(adata.obs[pred_key].astype(str).values, adata.obsm[spatial_key])

def compute_PAS(adata, pred_key, spatial_key='spatial'):
    if pred_key not in adata.obs or spatial_key not in adata.obsm:
        logging.warning(f"compute_PAS (adata): Key {pred_key} or {spatial_key} not in adata.obs/obsm. Skipping.")
        return np.nan
    return _compute_PAS(adata.obs[pred_key].astype(str).values, adata.obsm[spatial_key])

def compute_ASW(adata, pred_key, spatial_key='spatial'):
    if pred_key not in adata.obs or spatial_key not in adata.obsm:
        logging.warning(f"compute_ASW: Key {pred_key} or {spatial_key} not in adata.obs/obsm. Skipping.")
        return np.nan
    labels = adata.obs[pred_key].astype(str).values
    n_labels = len(np.unique(labels))
    if n_labels < 2 or n_labels > adata.shape[0] - 1:
        logging.warning(f"compute_ASW: Number of labels ({n_labels}) is invalid for silhouette score. Must be 2 <= n_labels <= n_samples - 1. Skipping.")
        return np.nan

    try:
        d = squareform(pdist(adata.obsm[spatial_key]))
        return silhouette_score(X=d, labels=labels, metric='precomputed')
    except Exception as e:
        logging.error(f"Error in compute_ASW: {e}")
        return np.nan


def compute_HOM(adata, gt_key, pred_key):
    if gt_key not in adata.obs or pred_key not in adata.obs:
        logging.warning(f"compute_HOM: Key {gt_key} or {pred_key} not in adata.obs. Skipping.")
        return np.nan
    return homogeneity_score(adata.obs[gt_key], adata.obs[pred_key])

def compute_COM(adata, gt_key, pred_key):
    if gt_key not in adata.obs or pred_key not in adata.obs:
        logging.warning(f"compute_COM: Key {gt_key} or {pred_key} not in adata.obs. Skipping.")
        return np.nan
    return completeness_score(adata.obs[gt_key], adata.obs[pred_key])

def compute_isolated_aws(adata, gt_key, batch_key=None, embedding_key='multi_emb'):
    if gt_key not in adata.obs:
        logging.warning(f"compute_isolated_aws: Ground truth key '{gt_key}' not found in adata.obs. Skipping.")
        return np.nan
    if embedding_key not in adata.obsm:
        logging.warning(f"compute_isolated_aws: Embedding key '{embedding_key}' not found in adata.obsm. Skipping.")
        return np.nan

    labels = adata.obs[gt_key]
    n_labels = len(np.unique(labels))
    if n_labels < 2 or n_labels > adata.shape[0] - 1:
        logging.warning(f"compute_isolated_aws: Number of labels for gt_key '{gt_key}' ({n_labels}) is invalid. Skipping.")
        return np.nan

    if batch_key is not None and batch_key not in adata.obs:
        logging.warning(f"compute_isolated_aws: Batch key '{batch_key}' provided but not found in adata.obs. Proceeding without batch correction for this metric.")
        batch_key = None # Fallback to pseudo_batch

    try:
        if batch_key is not None:
            return scib.metrics.isolated_labels_asw(adata=adata, label_key=gt_key, batch_key=batch_key, embed=embedding_key)
        else:
            adata.obs['pseudo_batch'] = 0
            result = scib.metrics.isolated_labels_asw(adata=adata, label_key=gt_key,  batch_key ='pseudo_batch', embed=embedding_key)
            del adata.obs['pseudo_batch']
            return result
    except Exception as e:
        logging.error(f"Error in compute_isolated_aws: {e}")
        if 'pseudo_batch' in adata.obs: del adata.obs['pseudo_batch'] # Ensure cleanup
        return np.nan


def compute_clisi_graph(adata, gt_key, batch_key=None, embedding_key='multi_emb'):
    if gt_key not in adata.obs:
        logging.warning(f"compute_clisi_graph: Ground truth key '{gt_key}' not found in adata.obs. Skipping.")
        return np.nan
    if embedding_key not in adata.obsm:
        logging.warning(f"compute_clisi_graph: Embedding key '{embedding_key}' not found in adata.obsm. Skipping.")
        return np.nan

    if batch_key is not None and batch_key not in adata.obs:
        logging.warning(f"compute_clisi_graph: Batch key '{batch_key}' provided but not found in adata.obs. Proceeding without batch for this metric.")
        batch_key = None

    try:
        if batch_key is not None:
            return scib.metrics.clisi_graph(adata=adata, label_key=gt_key, use_rep=embedding_key, batch_key=batch_key, type_='embed') # type_ added
        else:
            adata.obs['pseudo_batch'] = 0
            # Note: scib's clisi_graph for embed type might not use batch_key directly in some versions,
            # but it's good practice to pass it if available or use a pseudo one for consistency.
            # If type_='graph', batch_key is used. For type_='embed', it might be less critical.
            result = scib.metrics.clisi_graph(adata=adata, label_key=gt_key, type_="embed", use_rep=embedding_key, batch_key='pseudo_batch')
            del adata.obs['pseudo_batch']
            return result
    except Exception as e:
        logging.error(f"Error in compute_clisi_graph: {e}")
        if 'pseudo_batch' in adata.obs: del adata.obs['pseudo_batch']
        return np.nan


def compute_gt_silhouette(adata, gt_key, embedding_key='multi_emb'):
    if embedding_key not in adata.obsm:
        logging.warning(f"compute_gt_silhouette: Embedding key '{embedding_key}' not found in adata.obsm. Skipping.")
        return np.nan
    if gt_key not in adata.obs:
        logging.warning(f"compute_gt_silhouette: Ground truth key '{gt_key}' not found in adata.obs. Skipping.")
        return np.nan

    labels = adata.obs[gt_key]
    n_labels = len(np.unique(labels))
    if n_labels < 2 or n_labels > adata.shape[0] - 1:
        logging.warning(f"compute_gt_silhouette: Number of labels for gt_key '{gt_key}' ({n_labels}) is invalid. Skipping.")
        return np.nan
    try:
        return scib.metrics.silhouette(adata=adata, label_key=gt_key, embed=embedding_key)
    except Exception as e:
        logging.error(f"Error in compute_gt_silhouette: {e}")
        return np.nan


def marker_score(adata_orig, domain_key, top_n=5, min_samples_per_group=2, spatial_neighbor_key="spatial_neighbors"):
    if adata_orig is None:
        logging.warning(f"marker_score: Input adata_orig is None for domain '{domain_key}'. Skipping.")
        return np.nan, np.nan
    adata = adata_orig.copy()
    if adata.shape[0] == 0 or adata.shape[1] == 0:
        logging.warning(f"marker_score: Input adata is empty for domain '{domain_key}'. Skipping.")
        return np.nan, np.nan
    if domain_key not in adata.obs:
        logging.error(f"marker_score: Domain key '{domain_key}' not found in adata.obs.")
        return np.nan, np.nan
    if not pd.api.types.is_categorical_dtype(adata.obs[domain_key]):
        adata.obs[domain_key] = adata.obs[domain_key].astype('category')

    group_counts = adata.obs[domain_key].value_counts()
    valid_groups = group_counts[group_counts >= min_samples_per_group].index.tolist()

    if len(valid_groups) < 2:
        logging.warning(f"marker_score: Not enough groups (found {len(valid_groups)}) with >= {min_samples_per_group} samples for domain '{domain_key}'. Skipping.")
        return np.nan, np.nan

    adata_filtered = adata[adata.obs[domain_key].isin(valid_groups)].copy()
    adata_filtered.obs[domain_key] = adata_filtered.obs[domain_key].cat.remove_unused_categories()

    if not adata_filtered.var_names.empty:
        if not pd.api.types.is_string_dtype(adata_filtered.var_names.dtype) and not all(isinstance(x, str) for x in adata_filtered.var_names):
            logging.info(f"marker_score: Converting adata_filtered.var_names to string for domain '{domain_key}'.")
            try:
                adata_filtered.var_names = adata_filtered.var_names.astype(str)
            except Exception as e:
                logging.error(f"marker_score: Could not convert var_names to string for domain '{domain_key}': {e}. Skipping.")
                return np.nan, np.nan

    if not adata_filtered.var_names.is_unique:
        logging.info(f"marker_score: Making adata_filtered.var_names unique for domain '{domain_key}'.")
        adata_filtered.var_names_make_unique()


    if adata_filtered.shape[0] < 2 or len(adata_filtered.obs[domain_key].cat.categories) < 2:
        logging.warning(f"marker_score: adata_filtered has < 2 samples or < 2 categories for domain '{domain_key}'. Skipping.")
        return np.nan, np.nan

    X_data = adata_filtered.X
    is_all_zero = False
    if hasattr(X_data, "nnz"):
        is_all_zero = (X_data.nnz == 0)
    else:
        is_all_zero = np.all(X_data == 0)

    is_constant_to_first_val = False
    if X_data.shape[0] > 0 and X_data.shape[1] > 0:
        first_val = X_data[0,0]
        if hasattr(X_data, "nnz"):
            if first_val == 0: is_constant_to_first_val = (X_data.nnz == 0)
            else:
                if X_data.nnz == 0: is_constant_to_first_val = False
                else:
                    is_constant_to_first_val = (np.all(X_data.data == first_val) and
                                                (X_data.nnz == (X_data.shape[0] * X_data.shape[1])))
        else:
            is_constant_to_first_val = np.all(X_data == first_val)

    if is_all_zero or is_constant_to_first_val:
        logging.warning(f"marker_score: adata_filtered.X is all zeros or constant for domain '{domain_key}'. Skipping.")
        return np.nan, np.nan

    sc.pp.normalize_total(adata_filtered, target_sum=1e4)
    sc.pp.log1p(adata_filtered)

    n_genes_param = min(max(25, top_n * 2), adata_filtered.shape[1])
    if n_genes_param == 0:
         logging.warning(f"marker_score: adata_filtered has 0 genes for domain '{domain_key}'. Skipping rank_genes_groups.")
         return np.nan, np.nan

    try:
        sc.tl.rank_genes_groups(adata_filtered, groupby=domain_key, use_raw=False, method='wilcoxon', n_genes=n_genes_param)
    except Exception as e:
        logging.error(f"Error during rank_genes_groups for domain '{domain_key}': {e}")
        return np.nan, np.nan


    selected_genes = []
    if 'names' in adata_filtered.uns['rank_genes_groups']:
        all_ranked_genes = []
        for group_name in adata_filtered.uns['rank_genes_groups']['names'].dtype.names:
            group_genes_array = adata_filtered.uns['rank_genes_groups']['names'][group_name]
            if hasattr(group_genes_array, 'ndim') and group_genes_array.ndim == 0:
                gene_name = group_genes_array.item()
                if gene_name is not None and str(gene_name) != 'nan': all_ranked_genes.append(str(gene_name))
            else:
                try:
                    gene_list = list(group_genes_array[:top_n])
                    all_ranked_genes.extend([str(g) for g in gene_list if g is not None and str(g) != 'nan'])
                except (IndexError, TypeError) as e:
                    logging.warning(f"marker_score: Error processing group_genes for group '{group_name}', domain '{domain_key}': {e}")
        selected_genes = list(np.unique(all_ranked_genes))

    if not selected_genes:
        logging.warning(f"marker_score: No marker genes found for domain '{domain_key}' with top_n={top_n} after rank_genes_groups. Skipping.")
        return np.nan, np.nan

    if spatial_neighbor_key not in adata_filtered.obsp:
        if 'spatial' not in adata_filtered.obsm_keys():
            logging.error(f"marker_score: 'spatial' coordinates not found for domain '{domain_key}'. Cannot compute spatial_neighbors. Skipping.")
            return np.nan, np.nan
        try:
            sq.gr.spatial_neighbors(adata_filtered, spatial_key='spatial')
        except Exception as e:
            logging.error(f"Error computing spatial_neighbors for domain '{domain_key}': {e}")
            return np.nan, np.nan


    valid_selected_genes = [g for g in selected_genes if g in adata_filtered.var_names]
    if not valid_selected_genes:
        logging.warning(f"marker_score: No valid marker genes found in adata_filtered.var_names for domain '{domain_key}'. Skipping spatial_autocorr.")
        return np.nan, np.nan

    if adata_filtered[:, valid_selected_genes].X.shape[1] == 0:
        logging.warning(f"marker_score: Subset for valid_selected_genes has 0 genes. Skipping spatial_autocorr for domain '{domain_key}'.")
        return np.nan, np.nan

    X_subset_data = adata_filtered[:, valid_selected_genes].X
    is_subset_all_zero = (X_subset_data.nnz == 0) if hasattr(X_subset_data, "nnz") else np.all(X_subset_data == 0)
    if is_subset_all_zero:
        logging.warning(f"marker_score: Expression for valid_selected_genes is all zeros in domain '{domain_key}'. Skipping spatial_autocorr.")
        return np.nan, np.nan

    moranI, gearyC = np.nan, np.nan
    try:
        sq.gr.spatial_autocorr(adata_filtered, mode="moran", genes=valid_selected_genes, n_perms=100, n_jobs=-1)
        moranI = np.nanmedian(adata_filtered.uns["moranI"]['I']) if "moranI" in adata_filtered.uns and 'I' in adata_filtered.uns["moranI"] and len(adata_filtered.uns["moranI"]['I']) > 0 else np.nan
    except Exception as e:
        logging.error(f"Error calculating Moran's I for domain {domain_key}: {e}")
    try:
        sq.gr.spatial_autocorr(adata_filtered, mode="geary", genes=valid_selected_genes, n_perms=100, n_jobs=-1)
        gearyC = np.nanmedian(adata_filtered.uns["gearyC"]['C']) if "gearyC" in adata_filtered.uns and 'C' in adata_filtered.uns["gearyC"] and len(adata_filtered.uns["gearyC"]['C']) > 0 else np.nan
    except Exception as e:
        logging.error(f"Error calculating Geary's C for domain {domain_key}: {e}")

    return moranI, gearyC

def calculate_gene_specificity(adata_orig, domain_key, layers="counts", zero_percentage_threshold=99, min_cells_per_group=4):
    if adata_orig is None:
        logging.warning(f"calculate_gene_specificity: Input adata_orig is None for domain '{domain_key}'. Skipping.")
        return np.nan
    adata = adata_orig.copy()
    if adata.shape[0] == 0 or adata.shape[1] == 0:
        logging.warning(f"calculate_gene_specificity: Input adata is empty for domain '{domain_key}'. Skipping.")
        return np.nan

    if layers not in adata.layers:
        logging.warning(f"calculate_gene_specificity: Layer '{layers}' not found. Using .X.")
        if adata.X is None:
            logging.error("calculate_gene_specificity: adata.X is also None. Cannot proceed.")
            return np.nan
    else:
        adata.X = adata.layers[layers].copy()

    if domain_key not in adata.obs:
        logging.error(f"calculate_gene_specificity: Domain key '{domain_key}' not found.")
        return np.nan
    if not pd.api.types.is_categorical_dtype(adata.obs[domain_key]):
        adata.obs[domain_key] = adata.obs[domain_key].astype('category')

    count_dict = adata.obs[domain_key].value_counts()
    valid_groups = count_dict[count_dict >= min_cells_per_group].index
    if len(valid_groups) < 1:
        logging.warning(f"calculate_gene_specificity: No groups with >= {min_cells_per_group} cells for domain '{domain_key}'. Skipping.")
        return np.nan
    adata = adata[adata.obs[domain_key].isin(valid_groups)].copy()
    adata.obs[domain_key] = adata.obs[domain_key].cat.remove_unused_categories()
    labels = adata.obs[domain_key].values

    if hasattr(adata.X, "toarray"): X_dense = adata.X.toarray()
    else: X_dense = adata.X.copy()

    filtered_genes_mask = np.array([
        (np.sum(X_dense[:, i] == 0) / len(X_dense[:, i]) * 100 if len(X_dense[:, i]) > 0 else 100) <= zero_percentage_threshold
        for i in range(adata.n_vars)
    ])

    if not np.any(filtered_genes_mask):
        logging.warning(f"calculate_gene_specificity: No genes passed zero_percentage_threshold={zero_percentage_threshold} for domain '{domain_key}'. Skipping.")
        return np.nan

    adata = adata[:, filtered_genes_mask].copy()
    labels = adata.obs[domain_key].values

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    if hasattr(adata.X, "toarray"): X_processed = adata.X.toarray()
    else: X_processed = adata.X

    specificity_scores = {}
    unique_labels_in_adata = np.unique(labels)
    if len(unique_labels_in_adata) == 0:
        logging.warning(f"calculate_gene_specificity: No unique labels left after filtering for domain '{domain_key}'. Skipping.")
        return np.nan

    for i, gene in enumerate(adata.var_names):
        gene_expression = X_processed[:, i]
        mean_total_expression_of_gene = gene_expression.mean()
        if mean_total_expression_of_gene == 0:
            specificity_scores[gene] = [0.0] * len(unique_labels_in_adata)
            continue
        gene_specificities_for_classes = [
            (gene_expression[labels == class_id].mean() if np.sum(labels == class_id) > 0 else 0.0) / mean_total_expression_of_gene
            for class_id in unique_labels_in_adata
        ]
        specificity_scores[gene] = gene_specificities_for_classes

    if not specificity_scores: return np.nan
    specificity_df = pd.DataFrame(specificity_scores, index=unique_labels_in_adata)
    return specificity_df.max(axis=0).mean() if not specificity_df.empty else np.nan


def logistic_regression_feature_importance(adata_orig, domain_key, layers="counts", min_cells_per_group=4):
    if adata_orig is None:
        logging.warning(f"logistic_regression: Input adata_orig is None for domain '{domain_key}'. Skipping.")
        return np.nan
    adata = adata_orig.copy()
    if adata.shape[0] == 0 or adata.shape[1] == 0:
        logging.warning(f"logistic_regression: Input adata is empty for domain '{domain_key}'. Skipping.")
        return np.nan

    if layers not in adata.layers:
        logging.warning(f"logistic_regression: Layer '{layers}' not found. Using .X.")
        if adata.X is None:
            logging.error("logistic_regression: adata.X is also None. Cannot proceed.")
            return np.nan
    else:
        adata.X = adata.layers[layers].copy()

    if domain_key not in adata.obs:
        logging.error(f"logistic_regression: Domain key '{domain_key}' not found.")
        return np.nan
    if not pd.api.types.is_categorical_dtype(adata.obs[domain_key]):
        adata.obs[domain_key] = adata.obs[domain_key].astype('category')

    count_dict = adata.obs[domain_key].value_counts()
    valid_groups = count_dict[count_dict >= min_cells_per_group].index
    if len(valid_groups) < 2:
        logging.warning(f"logistic_regression: Not enough groups (found {len(valid_groups)}) with >= {min_cells_per_group} cells for domain '{domain_key}'. Skipping.")
        return np.nan

    adata = adata[adata.obs[domain_key].isin(valid_groups)].copy()
    adata.obs[domain_key] = adata.obs[domain_key].cat.remove_unused_categories()

    if adata.shape[0] < 10:
        logging.warning(f"logistic_regression: Too few samples ({adata.shape[0]}) after filtering for domain '{domain_key}'. Skipping.")
        return np.nan

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    labels = adata.obs[domain_key].values
    X = adata.X
    y = labels

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError as e:
        logging.warning(f"logistic_regression: train_test_split failed for domain '{domain_key}': {e}. Skipping.")
        return np.nan

    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        logging.warning(f"logistic_regression: Train or test set is empty after split for domain '{domain_key}'. Skipping.")
        return np.nan

    model = LogisticRegression(max_iter=2000, multi_class='ovr', random_state=42, solver='liblinear')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def calculate_mutual_information(adata_orig, domain_key, layers="counts", min_cells_per_group=4):
    if adata_orig is None:
        logging.warning(f"mutual_information: Input adata_orig is None for domain '{domain_key}'. Skipping.")
        return np.nan
    adata = adata_orig.copy()
    if adata.shape[0] == 0 or adata.shape[1] == 0:
        logging.warning(f"mutual_information: Input adata is empty for domain '{domain_key}'. Skipping.")
        return np.nan

    if layers not in adata.layers:
        logging.warning(f"mutual_information: Layer '{layers}' not found. Using .X.")
        if adata.X is None:
            logging.error("mutual_information: adata.X is also None. Cannot proceed.")
            return np.nan
    else:
        adata.X = adata.layers[layers].copy()

    if domain_key not in adata.obs:
        logging.error(f"mutual_information: Domain key '{domain_key}' not found.")
        return np.nan
    if not pd.api.types.is_categorical_dtype(adata.obs[domain_key]):
        adata.obs[domain_key] = adata.obs[domain_key].astype('category')

    count_dict = adata.obs[domain_key].value_counts()
    valid_groups = count_dict[count_dict >= min_cells_per_group].index
    if len(valid_groups) < 1:
        logging.warning(f"mutual_information: No groups with >= {min_cells_per_group} cells for domain '{domain_key}'. Skipping.")
        return np.nan

    adata = adata[adata.obs[domain_key].isin(valid_groups)].copy()
    adata.obs[domain_key] = adata.obs[domain_key].cat.remove_unused_categories()

    if adata.shape[0] == 0:
        logging.warning(f"mutual_information: Adata is empty after filtering for domain '{domain_key}'. Skipping.")
        return np.nan

    if hasattr(adata.X, "toarray"): X = adata.X.toarray()
    else: X = adata.X
    y = adata.obs[domain_key].cat.codes.values

    mi_scores = mutual_info_classif(X, y, random_state=42)
    return np.mean(mi_scores) if len(mi_scores) > 0 else np.nan

def compute_batch_asw(adata, batch_key="sample", label_key="leiden", embed_key="X_emb"):
    """Wrapper for scib.metrics.silhouette_batch"""
    try:
        return scib.metrics.silhouette_batch(
            adata=adata,
            batch_key=batch_key,
            label_key=label_key,
            embed=embed_key
        )
    except Exception as e:
        logging.error(f"Error in compute_batch_asw: {e}")
        return np.nan

def compute_ilisi_graph(adata, batch_key="sample", type_="embed", use_rep="X_emb"):
    """Wrapper for scib.metrics.ilisi_graph"""
    try:
        return scib.metrics.ilisi_graph(
            adata=adata,
            batch_key=batch_key,
            type_=type_,
            use_rep=use_rep
        )
    except FileNotFoundError as e:
        logging.error(f"ilisi_graph failed: {e}")
        return np.nan
    except TypeError as e:
        logging.error(f"TypeError in ilisi_graph: {e}")
        return np.nan
    except Exception as e:
        logging.error(f"Error in compute_ilisi_graph: {e}")
        return np.nan

def compute_pcr_comparison(adata_pre, adata, covariate="sample", embed_key="X_emb"):
    """Wrapper for scib.metrics.pcr_comparison"""
    try:
        return scib.metrics.pcr_comparison(
            adata_pre=adata_pre,
            adata_post=adata,
            covariate=covariate,
            embed=embed_key
        )
    except Exception as e:
        logging.error(f"Error in compute_pcr_comparison: {e}")
        return np.nan

def run_modality_benchmark(adata, embedding_dir: Path):
    results_dict = {}
    leiden_key = 'leiden'
    gt_key = 'pathological_annotation'

    expected_metric_keys = [
        'leiden_clusters', 'CHAOS', 'PAS', 'ASW',
        'ARI', 'NMI', 'isolated_asw', 'clisi_graph', 'gt_silhouette'
    ]
    st_top_n_list = [5, 10, 20, 50, 100, 200, 500]
    sm_top_n_list = [5, 10, 20, 50, 100, 200]
    threshold_list = [100, 99, 98, 95]

    for n in st_top_n_list: expected_metric_keys.extend([f'ST_moranI_top{n}', f'ST_gearyC_top{n}'])
    for n in sm_top_n_list: expected_metric_keys.extend([f'SM_moranI_top{n}', f'SM_gearyC_top{n}'])
    for t in threshold_list: expected_metric_keys.extend([f'ST_specificity_thresh{t}', f'SM_specificity_thresh{t}'])
    expected_metric_keys.extend(['ST_logistic', 'SM_logistic', 'ST_mi', 'SM_mi'])

    def initialize_na_results():
        return {key: np.nan for key in expected_metric_keys}

    set_seed(2024) # Seed set here for benchmark specific randomness, if any.
    if 'multi_emb' not in adata.obsm:
        logging.error("run_multi_benchmark: 'multi_emb' not found in adata.obsm. Cannot proceed.")
        return initialize_na_results()

    leiden_file = embedding_dir / "leiden_clusters_1.parquet"
    if not leiden_file.exists():
        logging.error(f"run_multi_benchmark: Precomputed Leiden clusters file not found: {leiden_file}")
        return initialize_na_results()

    try:
        leiden_df = pd.read_parquet(leiden_file)
    except Exception as e:
        logging.error(f"run_multi_benchmark: Could not read Leiden clusters file {leiden_file}: {e}")
        return initialize_na_results()

    barcode_col, cluster_col = 'barcode', 'leiden_cluster'
    if barcode_col not in leiden_df.columns or cluster_col not in leiden_df.columns:
        logging.error(f"run_multi_benchmark: Leiden file {leiden_file} missing '{barcode_col}' or '{cluster_col}' column.")
        return initialize_na_results()

    adata_obs_names_str = adata.obs_names.astype(str)
    leiden_df[barcode_col] = leiden_df[barcode_col].astype(str)
    leiden_df.set_index(barcode_col, inplace=True, drop=False)

    common_barcodes = adata_obs_names_str.intersection(leiden_df.index)
    if len(common_barcodes) == 0:
        logging.error(f"run_multi_benchmark: No common barcodes between AnnData and Leiden file.")
        logging.debug(f" AnnData obs_names sample: {list(adata_obs_names_str[:3])}")
        logging.debug(f" Leiden file index sample: {list(leiden_df.index[:3])}")
        return initialize_na_results()

    logging.info(f"run_multi_benchmark: Found {len(common_barcodes)} common barcodes for mapping Leiden clusters.")
    adata.obs[leiden_key] = 'unknown'
    adata.obs.loc[common_barcodes, leiden_key] = leiden_df.loc[common_barcodes, cluster_col].values
    adata.obs[leiden_key] = adata.obs[leiden_key].astype('category')
    logging.info(f"run_multi_benchmark: Successfully mapped Leiden clusters. Unique Leiden clusters: {list(adata.obs[leiden_key].unique())}")
    logging.info(f"Leiden cluster counts: {adata.obs[leiden_key].value_counts().to_dict()}")

    results_dict['leiden_clusters'] = adata.obs[leiden_key].nunique()
    if 'unknown' in adata.obs[leiden_key].cat.categories and adata.obs[leiden_key].nunique() == 1:
        results_dict['leiden_clusters'] = 0
    elif results_dict['leiden_clusters'] == 0:
        logging.warning("run_multi_benchmark: leiden_clusters is 0 after mapping.")

    if gt_key in adata.obs:
        adata.obs[gt_key] = adata.obs[gt_key].astype('category')
        results_dict['ARI'] = compute_ARI(adata, gt_key, leiden_key)
        results_dict['NMI'] = compute_NMI(adata, gt_key, leiden_key)
        results_dict['isolated_asw'] = compute_isolated_aws(adata, gt_key, embedding_key='multi_emb')
        results_dict['clisi_graph'] = compute_clisi_graph(adata, gt_key, embedding_key='multi_emb')
        results_dict['gt_silhouette'] = compute_gt_silhouette(adata, gt_key, embedding_key='multi_emb')
        logging.info(f"run_multi_benchmark: Computed ground truth metrics using '{gt_key}' and embedding 'multi_emb'")
    else:
        logging.warning(f"run_multi_benchmark: Ground truth key '{gt_key}' not found. Skipping ground truth metrics.")
        results_dict.update({k: np.nan for k in ['ARI', 'NMI', 'isolated_asw', 'clisi_graph', 'gt_silhouette']})

    results_dict['CHAOS'] = compute_CHAOS(adata, leiden_key, spatial_key='spatial')
    results_dict['PAS'] = compute_PAS(adata, leiden_key, spatial_key='spatial')
    results_dict['ASW'] = compute_ASW(adata, leiden_key, spatial_key='spatial')

    if 'counts' not in adata.layers:
        logging.error("run_multi_benchmark: 'counts' layer not found. Skipping ST/SM specific metrics.")
        for k in expected_metric_keys:
            if k.startswith(('ST_', 'SM_')) and k not in results_dict: results_dict[k] = np.nan
        return results_dict

    adata.X = adata.layers['counts'].copy()

    if 'type' not in adata.var.columns:
        logging.error("run_multi_benchmark: 'type' column not found in adata.var. Skipping ST/SM specific metrics.")
        for k in expected_metric_keys:
            if k.startswith(('ST_', 'SM_')) and k not in results_dict: results_dict[k] = np.nan
        return results_dict

    adata_SM_orig = adata[:,adata.var.type=="SM"].copy() if "SM" in adata.var.type.unique() else None
    adata_ST_orig = adata[:,adata.var.type=="ST"].copy() if "ST" in adata.var.type.unique() else None

    if adata_ST_orig is not None: adata_ST_orig.obs[leiden_key] = adata.obs.loc[adata_ST_orig.obs_names, leiden_key].astype('category')
    if adata_SM_orig is not None: adata_SM_orig.obs[leiden_key] = adata.obs.loc[adata_SM_orig.obs_names, leiden_key].astype('category')

    for n in st_top_n_list:
        if adata_ST_orig is not None and adata_ST_orig.shape[1] > 0:
            moranI_ST, gearyC_ST = marker_score(adata_ST_orig, leiden_key, top_n=n)
            results_dict[f'ST_moranI_top{n}'] = moranI_ST; results_dict[f'ST_gearyC_top{n}'] = gearyC_ST
        else: results_dict[f'ST_moranI_top{n}'] = results_dict[f'ST_gearyC_top{n}'] = np.nan
    if adata_ST_orig is None or adata_ST_orig.shape[1] == 0: logging.warning("run_multi_benchmark: No ST data or genes. ST marker scores are NaN.")

    for n in sm_top_n_list:
        if adata_SM_orig is not None and adata_SM_orig.shape[1] > 0:
            moranI_SM, gearyC_SM = marker_score(adata_SM_orig, leiden_key, top_n=n)
            results_dict[f'SM_moranI_top{n}'] = moranI_SM; results_dict[f'SM_gearyC_top{n}'] = gearyC_SM
        else: results_dict[f'SM_moranI_top{n}'] = results_dict[f'SM_gearyC_top{n}'] = np.nan
    if adata_SM_orig is None or adata_SM_orig.shape[1] == 0: logging.warning("run_multi_benchmark: No SM data or genes. SM marker scores are NaN.")

    for thresh in threshold_list:
        results_dict[f'SM_specificity_thresh{thresh}'] = calculate_gene_specificity(adata_SM_orig, leiden_key, layers="counts", zero_percentage_threshold=thresh) if adata_SM_orig else np.nan
        results_dict[f'ST_specificity_thresh{thresh}'] = calculate_gene_specificity(adata_ST_orig, leiden_key, layers="counts", zero_percentage_threshold=thresh) if adata_ST_orig else np.nan

    results_dict['ST_logistic'] = logistic_regression_feature_importance(adata_ST_orig, leiden_key, "counts") if adata_ST_orig else np.nan
    results_dict['SM_logistic'] = logistic_regression_feature_importance(adata_SM_orig, leiden_key, "counts") if adata_SM_orig else np.nan
    results_dict['ST_mi'] = calculate_mutual_information(adata_ST_orig, leiden_key, "counts") if adata_ST_orig else np.nan
    results_dict['SM_mi'] = calculate_mutual_information(adata_SM_orig, leiden_key, "counts") if adata_SM_orig else np.nan

    if 'adata_SM_orig' in locals() and adata_SM_orig is not None: del adata_SM_orig
    if 'adata_ST_orig' in locals() and adata_ST_orig is not None: del adata_ST_orig
    gc.collect(); log_memory_usage()
    return results_dict

def run_all_modality_adata(adata_base_file: str, embedding_mat_base: str, save_csv_path: str):
    logging.info(f"Starting main processing loop for {Path(embedding_mat_base).name}!")
    all_results = []
    log_memory_usage()

    for file_path in Path(adata_base_file).glob("**/*hvf2800.h5ad"):
        adata = None
        embedding_df = None
        dataset_name = file_path.stem
        logging.info(f"Processing AnnData file: {file_path.name} for method {Path(embedding_mat_base).name}")
        log_memory_usage()

        try:
            adata = sc.read_h5ad(file_path)
            logging.info(f"Successfully read: {file_path.name}, shape: {adata.shape}")

            embedding_dir = Path(embedding_mat_base) / dataset_name
            embedding_file = embedding_dir / "embedding_table.parquet"

            if not embedding_file.exists():
                logging.warning(f"run_all_adata: No embedding file '{embedding_file.name}' in {embedding_dir} for {dataset_name}. Skipping.")
                all_results.append({'file_name': file_path.name, 'embedding_method': Path(embedding_mat_base).name, 'error': f"Missing embedding file"})
                continue

            embedding_df = pd.read_parquet(embedding_file)
            logging.info(f"Read embedding file. Shape: {embedding_df.shape}")

            adata.obs_names = adata.obs_names.astype(str)
            if 'barcode' in embedding_df.columns:
                embedding_df['barcode'] = embedding_df['barcode'].astype(str)
                embedding_df.set_index('barcode', inplace=True)
            else:
                embedding_df.index = embedding_df.index.astype(str)

            if len(embedding_df) != adata.n_obs:
                logging.error(f"run_all_adata: Embedding length ({len(embedding_df)}) for {dataset_name} != adata ({adata.n_obs}). Skipping.")
                all_results.append({'file_name': file_path.name, 'embedding_method': Path(embedding_mat_base).name, 'error': "Embedding length mismatch"})
                continue

            if adata.obs_names.isin(embedding_df.index).all():
                current_embedding_values = embedding_df.loc[adata.obs_names].values
                logging.info(f"run_all_adata: Aligned embedding for {dataset_name}.")
            else:
                missing_bc = adata.obs_names[~adata.obs_names.isin(embedding_df.index)]
                logging.error(f"run_all_adata: {len(missing_bc)} AnnData barcodes not in embedding for {dataset_name}. Sample missing: {list(missing_bc[:3])}. Skipping.")
                all_results.append({'file_name': file_path.name, 'embedding_method': Path(embedding_mat_base).name, 'error': "Barcodes mismatch"})
                continue

            adata.obsm["multi_emb"] = current_embedding_values
            logging.info(f"run_all_adata: Set adata.obsm['multi_emb'] for {dataset_name}. Shape: {adata.obsm['multi_emb'].shape}")

            results = run_modality_benchmark(adata, embedding_dir)
            results['file_name'] = file_path.name
            results['embedding_method'] = Path(embedding_mat_base).name
            all_results.append(results)
            logging.info(f"Finished benchmarks for {file_path.name} with method {results['embedding_method']}.")

        except FileNotFoundError as e:
            logging.error(f"run_all_adata: File not found for {file_path.name}: {e}")
            all_results.append({'file_name': file_path.name, 'embedding_method': Path(embedding_mat_base).name, 'error': f"FileNotFoundError: {e}"})
        except Exception as e:
            logging.critical(f"CRITICAL ERROR processing {file_path.name} with method {Path(embedding_mat_base).name}: {e}")
            import traceback
            logging.error(traceback.format_exc())
            all_results.append({'file_name': file_path.name, 'embedding_method': Path(embedding_mat_base).name, 'error': f"General Exception: {str(e)[:100]}"})
        finally:
            if adata is not None: del adata
            if embedding_df is not None: del embedding_df
            gc.collect()
            log_memory_usage()

    if not all_results:
        logging.warning(f"run_all_adata: No results generated for method {Path(embedding_mat_base).name}!")
        results_df = pd.DataFrame([{'file_name': 'N/A', 'embedding_method': Path(embedding_mat_base).name, 'error': 'No .h5ad files processed or all failed'}])
    else:
        results_df = pd.DataFrame(all_results)
    try:
        results_df.to_csv(save_csv_path, index=False)
        logging.info(f"All results for method {Path(embedding_mat_base).name} saved to {save_csv_path}")
    except Exception as e:
        logging.error(f"ERROR saving results CSV for method {Path(embedding_mat_base).name} to {save_csv_path}: {e}")

    if 'results_df' in locals(): del results_df
    gc.collect(); log_memory_usage()
    return save_csv_path


def run_multi_modality():
    logging.info("Starting benchmark evaluation script (Direct Version with Memory Logging)")
    set_seed(2024) 
    log_memory_usage()

    adata_source_dir = "/mnt/volume3/trn/10_spatialMETA_revision/06_spatialmeta_groundtruth"
    embedding_method_base_dirs = [
        #  "/mnt/volume1/2023SRTP/library/benchmark/Upload_summary/cross_modality/tool_results/TotalVI_run",
        # "/mnt/volume1/2023SRTP/library/benchmark/Upload_summary/cross_modality/tool_results/PCA_baseline",
        #  "/mnt/volume1/2023SRTP/library/benchmark/Upload_summary/cross_modality/tool_results/spaMultiVAE",
        # "/mnt/volume1/2023SRTP/library/benchmark/Upload_summary/cross_modality/tool_results/SpatialGlue_run",
        "/mnt/volume1/2023SRTP/library/benchmark/Upload_summary/cross_modality/tool_results/SpatialMETA_run",
        # "/mnt/volume1/2023SRTP/library/benchmark/Upload_summary/cross_modality/tool_results/Stabmap_run",
        #  "/mnt/volume1/2023SRTP/library/benchmark/Upload_summary/cross_modality/tool_results/Seurat_BNN_run",
         "/mnt/volume1/2023SRTP/library/benchmark/Upload_summary/cross_modality/tool_results/MISO_2m"
    ]

    if not Path(adata_source_dir).is_dir():
        logging.critical(f"AnnData source directory '{adata_source_dir}' does not exist. Aborting.")
        return

    csv_output_dir = Path("/mnt/volume1/2023SRTP/library/benchmark/Upload_summary/cross_modality/new_results_direct_memlog")
    csv_output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"CSV output directory: {csv_output_dir}")

    for emb_method_path_str in embedding_method_base_dirs:
        emb_method_path = Path(emb_method_path_str)
        method_name = emb_method_path.name
        logging.info(f"Processing method: {method_name}")
        log_memory_usage()

        if not emb_method_path.is_dir():
            logging.warning(f"Embedding method base path '{emb_method_path}' does not exist. Skipping.")
            error_df = pd.DataFrame([{'file_name': 'N/A',
                                      'embedding_method': method_name,
                                      'error': f"Method directory not found: {emb_method_path}"}])
            final_benchmark_csv_path = csv_output_dir / f"{method_name}_benchmark_summary.csv"
            try:
                error_df.to_csv(final_benchmark_csv_path, index=False)
                logging.info(f"Saved error placeholder for missing method '{method_name}' to '{final_benchmark_csv_path}'")
            except Exception as e:
                logging.error(f"ERROR saving error placeholder CSV for {method_name}: {e}")
            continue

        final_benchmark_csv_path = csv_output_dir / f"{method_name}_benchmark_summary.csv"
        logging.info(f"Output CSV for method {method_name} will be: {final_benchmark_csv_path}")

        try:
            run_all_modality_adata(
                adata_base_file=adata_source_dir,
                embedding_mat_base=emb_method_path_str,
                save_csv_path=str(final_benchmark_csv_path)
            )
            logging.info(f"Completed benchmark for method '{method_name}'. Results potentially saved to '{final_benchmark_csv_path}'")
        except Exception as e:
            logging.critical(f"CRITICAL ERROR during run_all_adata for method {method_name}: {e}")
            import traceback
            logging.error(traceback.format_exc())
            error_df = pd.DataFrame([{'file_name': 'N/A',
                                      'embedding_method': method_name,
                                      'error': f"Catastrophic failure in run_all_adata: {str(e)[:100]}"}])
            try:
                error_df.to_csv(final_benchmark_csv_path, index=False)
                logging.info(f"Saved catastrophic error summary for method '{method_name}' to '{final_benchmark_csv_path}'")
            except Exception as e_save:
                logging.error(f"ERROR saving catastrophic error CSV for {method_name}: {e_save}")

        gc.collect()
        log_memory_usage()

    logging.info("Benchmark evaluation script (Direct Version with Memory Logging) completed")

# if __name__ == '__main__':
#     run_multi_modality()

def run_sample_benchmark(adata, file_path):
    results_dict = {}
    merge_adata_pre = None
    
    leiden_key = 'leiden'
    gt_key = 'pathological_annotation'
    embedding_key = 'X_emb'

    set_seed(2024)
    try:
        results_dict['cell_count'] = adata.shape[0]

        # Compute batch metrics
        results_dict['batch_asw'] = compute_batch_asw(adata)
        results_dict['ilisi_graph'] = compute_ilisi_graph(adata)

        # Compute PCR comparison
        merge_adata_pre = sc.read_h5ad(file_path)
        results_dict['pcr'] = compute_pcr_comparison(merge_adata_pre, adata)

        for sample in adata.obs['sample'].unique():
            adata_sample = adata[adata.obs['sample'] == sample].copy()

            ground_truth = sc.read(f'/mnt/volume3/trn/10_spatialMETA_revision/06_spatialmeta_groundtruth/adata_joint_{sample}_hvf2800.h5ad')
            adata_sample.obs[gt_key] = ground_truth.obs[gt_key]
            del ground_truth

            results_dict[f'{sample}_cell_count'] = adata_sample.shape[0]

            if gt_key in adata_sample.obs:
                adata_sample.obs[gt_key] = adata_sample.obs[gt_key].astype('category')
                results_dict[f'{sample}_ARI'] = compute_ARI(adata_sample, gt_key, leiden_key)
                results_dict[f'{sample}_NMI'] = compute_NMI(adata_sample, gt_key, leiden_key)
                
                results_dict[f'{sample}_isolated_asw'] = compute_isolated_aws(adata_sample, gt_key, embedding_key=embedding_key)
                results_dict[f'{sample}_clisi_graph'] = compute_clisi_graph(adata_sample, gt_key, embedding_key=embedding_key)
                results_dict[f'{sample}_gt_silhouette'] = compute_gt_silhouette(adata_sample, gt_key, embedding_key=embedding_key)
                logging.info(f"Computed ground truth metrics using '{gt_key}' and embedding '{embedding_key}'")
            else:
                logging.warning(f"run_multi_benchmark: Ground truth key '{gt_key}' not found. Skipping ground truth related metrics (ARI, NMI, isolated_asw, clisi_graph, gt_silhouette).")
                results_dict.update({k: np.nan for k in ['ARI', 'NMI', 'isolated_asw', 'clisi_graph', 'gt_silhouette']})

            results_dict[f'{sample}_CHAOS'] = compute_CHAOS(adata_sample, leiden_key, spatial_key='spatial')
            results_dict[f'{sample}_PAS'] = compute_PAS(adata_sample, leiden_key, spatial_key='spatial')
            results_dict[f'{sample}_ASW'] = compute_ASW(adata_sample, leiden_key, spatial_key='spatial')
            
            adata_sample.X = adata_sample.layers['counts'].copy()
            adata_sample_SM = adata_sample[:,adata_sample.var.type=="SM"].copy() if "SM" in adata_sample.var.type.unique() else None
            adata_sample_ST = adata_sample[:,adata_sample.var.type=="ST"].copy() if "ST" in adata_sample.var.type.unique() else None
            gc.collect()
            
            st_top_n = [5, 10, 20, 50, 100, 200, 500]
            sm_top_n = [5, 10, 20, 50, 100, 200]
            
            if adata_sample_ST is not None and adata_sample_ST.shape[1] > 0:
                if leiden_key not in adata_sample_ST.obs:
                    logging.warning(f"run_multi_benchmark: Leiden key '{leiden_key}' not found in ST sample {sample}. Skipping ST marker scores.")
                elif len(adata_sample_ST.obs[leiden_key].unique()) < 2:
                    logging.warning(f"run_multi_benchmark: Not enough unique Leiden clusters for ST sample {sample} to compute marker scores. Skipping.")
                else:
                    for n in st_top_n:
                        logging.info(f"{sample}_ST marker score top {n}")
                        try:
                            moranI_ST, gearyC_ST = marker_score(adata_sample_ST, leiden_key, top_n=n)
                            results_dict[f'{sample}_ST_moranI_top{n}'] = moranI_ST
                            results_dict[f'{sample}_ST_gearyC_top{n}'] = gearyC_ST
                        except ValueError as ve:
                            logging.warning(f"marker_score failed for ST sample {sample}, top {n}: {ve}")
                            results_dict[f'{sample}_ST_moranI_top{n}'] = np.nan
                            results_dict[f'{sample}_ST_gearyC_top{n}'] = np.nan
                        except Exception as e:
                            logging.error(f"Error computing marker_score for ST sample {sample}, top {n}: {e}")
                            results_dict[f'{sample}_ST_moranI_top{n}'] = np.nan
                            results_dict[f'{sample}_ST_gearyC_top{n}'] = np.nan
                        gc.collect()
            else:
                logging.warning(f"No ST data or genes for sample {sample}. Skipping ST marker scores.")
                for n in st_top_n:
                    results_dict[f'{sample}_ST_moranI_top{n}'] = np.nan
                    results_dict[f'{sample}_ST_gearyC_top{n}'] = np.nan

            if adata_sample_SM is not None and adata_sample_SM.shape[1] > 0:
                if leiden_key not in adata_sample_SM.obs:
                    logging.warning(f"run_multi_benchmark: Leiden key '{leiden_key}' not found in SM sample {sample}. Skipping SM marker scores.")
                elif len(adata_sample_SM.obs[leiden_key].unique()) < 2:
                    logging.warning(f"run_multi_benchmark: Not enough unique Leiden clusters for SM sample {sample} to compute marker scores. Skipping.")
                else:
                    for n in sm_top_n:
                        logging.info(f"{sample}_SM marker score top {n}")
                        try:
                            moranI_SM, gearyC_SM = marker_score(adata_sample_SM, leiden_key, top_n=n)
                            results_dict[f'{sample}_SM_moranI_top{n}'] = moranI_SM
                            results_dict[f'{sample}_SM_gearyC_top{n}'] = gearyC_SM
                        except ValueError as ve:
                            logging.warning(f"marker_score failed for SM sample {sample}, top {n}: {ve}")
                            results_dict[f'{sample}_SM_moranI_top{n}'] = np.nan
                            results_dict[f'{sample}_SM_gearyC_top{n}'] = np.nan
                        except Exception as e:
                            logging.error(f"Error computing marker_score for SM sample {sample}, top {n}: {e}")
                            results_dict[f'{sample}_SM_moranI_top{n}'] = np.nan
                            results_dict[f'{sample}_SM_gearyC_top{n}'] = np.nan
                        gc.collect()
            else:
                logging.warning(f"No SM data or genes for sample {sample}. Skipping SM marker scores.")
                for n in sm_top_n:
                    results_dict[f'{sample}_SM_moranI_top{n}'] = np.nan
                    results_dict[f'{sample}_SM_gearyC_top{n}'] = np.nan

            thresholds = [100, 99, 98, 95]
            for thresh in thresholds:
                results_dict[f'{sample}_SM_specificity_thresh{thresh}'] = calculate_gene_specificity(adata_sample_SM, leiden_key, layers="counts", zero_percentage_threshold=thresh) if adata_sample_SM else np.nan
                results_dict[f'{sample}_ST_specificity_thresh{thresh}'] = calculate_gene_specificity(adata_sample_ST, leiden_key, layers="counts", zero_percentage_threshold=thresh) if adata_sample_ST else np.nan
                gc.collect()
            
            results_dict[f'{sample}_ST_logistic'] = logistic_regression_feature_importance(adata_sample_ST, leiden_key, "counts") if adata_sample_ST else np.nan
            results_dict[f'{sample}_SM_logistic'] = logistic_regression_feature_importance(adata_sample_SM, leiden_key, "counts") if adata_sample_SM else np.nan
            
            results_dict[f'{sample}_ST_mi'] = calculate_mutual_information(adata_sample_ST, leiden_key, "counts") if adata_sample_ST else np.nan
            results_dict[f'{sample}_SM_mi'] = calculate_mutual_information(adata_sample_SM, leiden_key, "counts") if adata_sample_SM else np.nan
            
            if 'adata_sample_SM' in locals() and adata_sample_SM is not None: del adata_sample_SM
            if 'adata_sample_ST' in locals() and adata_sample_ST is not None: del adata_sample_ST
            gc.collect(); log_memory_usage()

        gc.collect()
        
    except Exception as e:
        logging.error(f"Error in run_multi_benchmark: {str(e)}")
        logging.error(traceback.format_exc())
    
    finally:
        del adata
        if merge_adata_pre is not None:
            del merge_adata_pre
        gc.collect()
        log_memory_usage()
    
    return results_dict

def run_cross_sample():
    logging.info("Starting cross-sample benchmark evaluation script")
    set_seed(2024)
    log_memory_usage()

    methods = ['pca','scvi','scanvi','scpoli','totalvi','spavae','seurat_cca','seurat_rpca','spatialmeta']
    sample_name = "merge_adata_mouse_brain_svf"
    # sample_name = "merge_adata_RCC_M1_svf"
    
    csv_output_dir = Path("/mnt/volume0/cyr_qyc/qyc/cross_sample/results_new/benchmark/index")
    csv_output_dir.mkdir(parents=True, exist_ok=True)
    final_benchmark_csv_path = csv_output_dir / f"{sample_name}_bench_sample.csv"
    
    logging.info(f"CSV output will be saved to: {final_benchmark_csv_path}")
    
    for method in methods:
        logging.info(f"Processing method: {method}")
        log_memory_usage()
        
        try:
            adata = sc.read_h5ad(f"/mnt/volume3/trn/10_spatialMETA_revision/data/20250104_merge_data/{sample_name}.h5ad")
            logging.info(f"Successfully read AnnData for {sample_name}, shape: {adata.shape}")
            
            # Load embedding
            embedding_path = f"/mnt/volume0/cyr_qyc/qyc/cross_sample/results_new/{method}/{sample_name}/result.parquet"
            if Path(embedding_path).exists():
                adata.obsm['X_emb'] = pd.read_parquet(embedding_path).values
                logging.info(f"Loaded embedding from parquet for {method}/{sample_name}")
            else:
                embedding_path = f"/mnt/volume0/cyr_qyc/qyc/cross_sample/results_new/{method}/{sample_name}/result.csv"
                if Path(embedding_path).exists():
                    adata.obsm['X_emb'] = pd.read_csv(embedding_path, index_col=0).values
                    logging.info(f"Loaded embedding from CSV for {method}/{sample_name}")
                else:
                    logging.error(f"No embedding file found for {method}/{sample_name}")
                    continue

            # Load leiden clusters
            leiden_path = f"/mnt/volume0/cyr_qyc/qyc/cross_sample/results_new/{method}/{sample_name}/leiden.parquet"
            if Path(leiden_path).exists():
                leiden_df = pd.read_parquet(leiden_path)
                if 'leiden' in leiden_df.columns and len(leiden_df) == adata.shape[0]:
                    leiden_df.index = adata.obs_names
                    adata.obs['leiden'] = leiden_df['leiden'].astype('category')
                    logging.info(f"Loaded leiden clusters for {method}/{sample_name}")
                else:
                    logging.warning(f"Leiden file for {method}/{sample_name} is malformed or size mismatch")
                    continue
            else:
                logging.warning(f"No leiden file found for {method}/{sample_name}")
                continue

            # Run benchmark
            results = run_sample_benchmark(adata, f"/mnt/volume3/trn/10_spatialMETA_revision/data/20250104_merge_data/{sample_name}.h5ad")
            results['method'] = method
            
            # Save results
            results_df = pd.DataFrame(results, index=[0])
            if not final_benchmark_csv_path.exists():
                results_df.to_csv(final_benchmark_csv_path, index=False)
                logging.info(f"Created new results file for {method}")
            else:
                results_df.to_csv(final_benchmark_csv_path, mode='a', header=False, index=False)
                logging.info(f"Appended results for {method}")
            
        except Exception as e:
            logging.critical(f"CRITICAL ERROR processing {method}: {e}")
            import traceback
            logging.error(traceback.format_exc())
            error_df = pd.DataFrame([{
                'method': method,
                'error': f"Catastrophic failure: {str(e)[:100]}"
            }])
            try:
                if not final_benchmark_csv_path.exists():
                    error_df.to_csv(final_benchmark_csv_path, index=False)
                else:
                    error_df.to_csv(final_benchmark_csv_path, mode='a', header=False, index=False)
            except Exception as e_save:
                logging.error(f"ERROR saving error summary for {method}: {e_save}")
        
        finally:
            if 'adata' in locals():
                del adata
            gc.collect()
            log_memory_usage()
    
    logging.info("Cross-sample benchmark evaluation script completed")

# if __name__ == "__main__":
#     run_cross_sample()