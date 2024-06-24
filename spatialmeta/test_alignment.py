import spatialtk
from spatialtk.model._alignment_model import AlignmentVAE
import scanpy as sc
import sinfonia
import torch
import copy
from spatialtk.util._squidpy_spatial_variable_genes import spatial_autocorr, spatial_neighbors


a=50.0
p=2.0
expand=2.0
nt=3
niter=5000
diffeo_start=0,
epL=1e-4
epT=1e-4,
epV=2e-1
epA=1e-5
sigmaM=1.0
sigmaB=2.0
sigmaA=5.0
sigmaR=5e5
sigmaL=1
sigmaP=2e1
muB=None,
muA=None


def normalize_non_inplace(adata):
    adata.layers['normalized'] = copy.deepcopy(adata.X)
    sc.pp.normalize_total(adata, layer='normalized')
    sc.pp.log1p(adata, layer='normalized')

adata_st = sc.read_h5ad("/Users/snow/Documents/wd/stalign/data/m3_DHB_ST_raw.h5ad")
# adata_st = sc.read_h5ad("/mnt/volume1/trn/07_NBT/SpatialToolkit/extdata/m3_DHB_ST_raw.h5ad")
normalize_non_inplace(adata_st)

adata_st = sinfonia.statistics.spatially_variable_genes(
    adata_st,
    n_top_genes=2000,
    inplace=True,
    layer='normalized'
)
adata_st = adata_st[:,adata_st.var['spatially_variable']]
xmin,xmax = adata_st.obsm['spatial'][:,0].min(), adata_st.obsm['spatial'][:,0].max()
# scale x to 0-100 and keep aspect ratio
adata_st.obsm['spatial'] = (adata_st.obsm['spatial'] - xmin) / (xmax - xmin) * 100

adata_sm = sc.read_h5ad("/Users/snow/Documents/wd/stalign/data/m3_DHB_SM_raw.h5ad")
# adata_sm = sc.read_h5ad("/mnt/volume1/trn/07_NBT/SpatialToolkit/extdata/m3_DHB_SM_raw.h5ad")
sc.pp.normalize_total(adata_sm, target_sum=1e3)

adata_sm = sinfonia.statistics.spatially_variable_genes(
    adata_sm,
    n_top_genes=500,
    inplace=True
)

#spatial_neighbors(adata_sm)
#spatial_autocorr(adata_sm)

adata_sm = adata_sm[:,adata_sm.var['spatially_variable']]
xmin,xmax = adata_sm.obsm['spatial'][:,0].min(), adata_sm.obsm['spatial'][:,0].max()
# scale x to 0-100 and keep aspect ratio

spatialtk.pp.spot_transform_bymanul(adata_sm,
                                    rotation=90,
                                    SM_spatial_key="spatial",
                                    new_SM_spatial_key="spatial"
                                   )

adata_sm.obsm['spatial'] = (adata_sm.obsm['spatial'] - xmin) / (xmax - xmin) * 100


model = AlignmentVAE(
    adata_st=adata_st,
    adata_sm=adata_sm,
    n_latent=10,
    # device='mps'
)

