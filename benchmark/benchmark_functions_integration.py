#import packages
import scanpy as sc
import pandas as pd
import squidpy as sq
import numpy as np
from scipy.spatial import *
from sklearn.preprocessing import *
from pathlib import Path
from sklearn.metrics import *
from scipy.spatial.distance import *
import warnings
import gc
import scib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import mutual_info_classif
import numpy as np
from numpy.linalg import norm
warnings.filterwarnings("ignore")

def __init__(self):
    pass

def res_search(adata,target_k = 7, res_start = 0.1, res_step = 0.1, res_epochs = 10): 
    
    """
        adata: the Anndata object, a dataset.
        target_k: int, expected number of clusters.
        res_start: float, starting value of resolution. default: 0.1.
        res_step: float, step of resoution. default: 0.1.
        res_epochs: int, epoch of resolution. default: 10.
    """

    print(f"searching resolution to k={target_k}")
    res = res_start
    sc.tl.leiden(adata, resolution=res)

    old_k = len(adata.obs['leiden'].cat.categories)
    print("Res = ", res, "Num of clusters = ", old_k)

    run = 0
    while old_k != target_k:
        old_sign = 1 if (old_k<target_k) else -1
        sc.tl.leiden(adata, resolution=res+res_step*old_sign)
        new_k = len(adata.obs['leiden'].cat.categories)
        print("Res = ", res+res_step*old_sign, "Num of clusters = ", new_k)
        if new_k == target_k:
            res = res+res_step*old_sign
            print("recommended res = ", str(res))
            return res
        new_sign = 1 if (new_k<target_k) else -1
        if new_sign==old_sign:
            res = res+res_step*old_sign
            print("Res changed to", res)
            old_k = new_k
        else:
            res_step = res_step/2
            print("Res changed to", res)
        if run>res_epochs:
            print("Exact resolution not found")
            print("Recommended res = ", str(res))
            return res
        run+=1
    print("Recommended res = ", str(res))
    return res

def _compute_CHAOS(clusterlabel, location):

    clusterlabel = np.array(clusterlabel)
    location = np.array(location)
    matched_location = StandardScaler().fit_transform(location)

    clusterlabel_unique = np.unique(clusterlabel)
    dist_val = np.zeros(len(clusterlabel_unique))
    count = 0
    for k in clusterlabel_unique:
        location_cluster = matched_location[clusterlabel==k,:]
        if len(location_cluster)<=2:
            continue
        n_location_cluster = len(location_cluster)
        results = [fx_1NN(i,location_cluster) for i in range(n_location_cluster)]
        dist_val[count] = np.sum(results)
        count = count + 1

    return np.sum(dist_val)/len(clusterlabel)



def fx_1NN(i,location_in):
    location_in = np.array(location_in)
    dist_array = distance_matrix(location_in[i,:][None,:],location_in)[0,:]
    dist_array[i] = np.inf
    return np.min(dist_array)


def fx_kNN(i,location_in,k,cluster_in):

    location_in = np.array(location_in)
    cluster_in = np.array(cluster_in)
    dist_array = distance_matrix(location_in[i,:][None,:],location_in)[0,:]
    dist_array[i] = np.inf
    ind = np.argsort(dist_array)[:k]
    cluster_use = np.array(cluster_in)
    if np.sum(cluster_use[ind]!=cluster_in[i])>(k/2):
        return 1
    else:
        return 0
def _compute_PAS(clusterlabel,location):
    
    clusterlabel = np.array(clusterlabel)
    location = np.array(location)
    matched_location = location
    results = [fx_kNN(i,matched_location,k=10,cluster_in=clusterlabel) for i in range(matched_location.shape[0])]
    return np.sum(results)/len(clusterlabel)
    

def markerFC(adata_valid,marker_list,sdm_key):
    rst_dict = {}
    sdm_unique = adata_valid.obs[sdm_key].cat.categories
    for marker in marker_list:
        mean_exp_list = []
        for sdm in sdm_unique:
            mean_exp_list.append(np.mean(adata_valid[adata_valid.obs[sdm_key]==sdm][:,marker].X))
        max_sdm_idx = np.argmax(mean_exp_list)
#         print(sdm_unique[max_sdm_idx])

        max_sdm_value = np.max(mean_exp_list)
        other_sdm_value = np.mean(adata_valid[adata_valid.obs[sdm_key]!=sdm_unique[max_sdm_idx]][:,marker].X)
        cur_fc = max_sdm_value/other_sdm_value
        rst_dict[marker] = cur_fc
    return rst_dict
def compute_ARI(adata,gt_key,pred_key):
    return adjusted_rand_score(adata.obs[gt_key],adata.obs[pred_key])

def compute_NMI(adata,gt_key,pred_key):
    return normalized_mutual_info_score(adata.obs[gt_key],adata.obs[pred_key])

def compute_CHAOS(adata,pred_key,spatial_key='spatial'):
    return _compute_CHAOS(adata.obs[pred_key],adata.obsm[spatial_key])

def compute_PAS(adata,pred_key,spatial_key='spatial'):
    return _compute_PAS(adata.obs[pred_key],adata.obsm[spatial_key])

def compute_ASW(adata,pred_key,spatial_key='spatial'):
    d = squareform(pdist(adata.obsm[spatial_key]))
    return silhouette_score(X=d,labels=adata.obs[pred_key],metric='precomputed')

def compute_HOM(adata,gt_key,pred_key):
    return homogeneity_score(adata.obs[gt_key],adata.obs[pred_key])

def compute_COM(adata,gt_key,pred_key):
    return completeness_score(adata.obs[gt_key],adata.obs[pred_key])


def marker_score(adata, domain_key, top_n=5, min_samples=2):
    adata = adata.copy()
    
    group_counts = adata.obs[domain_key].value_counts()
    print("Group counts:", group_counts)
    
    valid_groups = group_counts[group_counts >= min_samples].index.tolist()
    
    print("Valid groups:", valid_groups)
    
    if len(valid_groups) < 2:
        raise ValueError(f"Not enough groups with at least {min_samples} samples")
    
    adata_filtered = adata[adata.obs[domain_key].isin(valid_groups)].copy()
    sc.pp.normalize_per_cell(adata_filtered)
    sc.pp.log1p(adata_filtered)
    
    group_counts_filtered = adata_filtered.obs[domain_key].value_counts()
    valid_groups_filtered = group_counts_filtered[group_counts_filtered >= 2].index.tolist()
    adata_filtered = adata_filtered[adata_filtered.obs[domain_key].isin(valid_groups_filtered)].copy()
    
    sc.tl.rank_genes_groups(adata_filtered, groupby=domain_key, use_raw=False)
    
    selected_genes = []
    for i in range(min(top_n, len(adata_filtered.uns['rank_genes_groups']['names']))):
        toadd = list(adata_filtered.uns['rank_genes_groups']['names'][i])
        selected_genes.extend(toadd)
    
    selected_genes = np.unique(selected_genes)
    
    sq.gr.spatial_neighbors(adata_filtered)
    sq.gr.spatial_autocorr(
        adata_filtered,
        mode="moran",
        genes=selected_genes,
        n_perms=100,
        n_jobs=-1,
    )
    sq.gr.spatial_autocorr(
        adata_filtered,
        mode="geary",
        genes=selected_genes,
        n_perms=100,
        n_jobs=-1,
    )
    
    moranI = np.median(adata_filtered.uns["moranI"]['I'])
    gearyC = np.median(adata_filtered.uns["gearyC"]['C'])
    
    return moranI, gearyC

def calculate_gene_specificity(adata, domain_key,layers="counts",zero_percentage_threshold=99):
    adata.X = adata.layers[layers].copy()
    labels = adata.obs[domain_key].values
    specificity_scores = {}
    adata = adata.copy()
    count_dict = adata.obs[domain_key].value_counts()
    adata = adata[adata.obs[domain_key].isin(count_dict.keys()[count_dict>3].values)]

    filtered_genes = []
    for gene in adata.var_names:
        gene_expression = adata[:, gene].X.toarray().flatten()
        
        # Calculate the percentage of spots where the expression is 0
        zero_percentage = np.sum(gene_expression == 0) / len(gene_expression) * 100
        
        # Only keep the gene if its expression is non-zero in more than the threshold percentage of spots
        if zero_percentage <= zero_percentage_threshold:
            filtered_genes.append(gene)
    
    # Only proceed with the filtered genes
    adata = adata[:, filtered_genes]
    
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    
    for gene in filtered_genes:
        gene_expression = adata[:, gene].X.toarray().flatten()
        scores = []
        
        for class_id in np.unique(labels):
            class_expression = gene_expression[labels == class_id].mean()
            total_expression = gene_expression.mean()
            specificity = class_expression / total_expression
            scores.append(specificity)
        
        specificity_scores[gene] = scores
    
    specificity_df = pd.DataFrame(specificity_scores, index=np.unique(labels))
    return specificity_df.max().mean()




def logistic_regression_feature_importance(adata, domain_key,layers="counts"):
    adata.X = adata.layers[layers].copy()
    adata = adata.copy()
    count_dict = adata.obs[domain_key].value_counts()
    adata = adata[adata.obs[domain_key].isin(count_dict.keys()[count_dict>3].values)]
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    labels = adata.obs[domain_key].values
    X = adata.X
    y = labels
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(max_iter=1000, multi_class='ovr')
    model.fit(X_train, y_train)
    
    feature_importance = np.abs(model.coef_)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy



def calculate_mutual_information(adata, domain_key,layers="counts"):
    adata.X = adata.layers[layers].copy()
    X = adata.X.toarray()  
    y = adata.obs[domain_key].values  

    mi_scores = mutual_info_classif(X, y)
    

    mi_df = pd.DataFrame({'Gene': adata.var_names, 'Mutual Information': mi_scores})
    mi_df = mi_df.sort_values(by='Mutual Information', ascending=False)
    
    return mi_df['Mutual Information'].mean()



# calculate reconstruction
def mean_absolute_error(X, X_reconstructed):
    return np.mean(np.abs(X - X_reconstructed))
def pearson_correlation_coefficient(X, X_reconstructed):
    pcc_list = []
    for i in range(X.shape[1]):
        x = X[:, i]
        x_reconstructed = X_reconstructed[:, i]
        # Avoid division by zero
        if np.std(x) == 0 or np.std(x_reconstructed) == 0:
            pcc_list.append(np.nan)  # Handle constant rows
        else:
            pcc = np.corrcoef(x, x_reconstructed)[0, 1]
            pcc_list.append(pcc)
    mean_pcc = np.nanmean(pcc_list)  
    return pcc_list, mean_pcc
def cosine_similarity(X, X_reconstructed):
    cos_sim_list = []
    for i in range(X.shape[1]):
        x = X[:, i]
        x_reconstructed = X_reconstructed[:, i]
        # Compute cosine similarity
        cos_sim = np.dot(x, x_reconstructed) / (norm(x) * norm(x_reconstructed))
        cos_sim_list.append(cos_sim)
    mean_cos_sim = np.mean(cos_sim_list)
    return cos_sim_list, mean_cos_sim
