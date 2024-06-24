import scanpy as sc
from STalign.STalign import *
from STalign.LLDDMM import *
import matplotlib.pyplot as plt

device='cpu' if not torch.cuda.is_available() else 'cuda:0'
adata_ST = sc.read_h5ad("./data/adata_ST.h5ad")
adata_SM = sc.read_h5ad("./data/adata_SM.h5ad")
adata_SM.obsm['spatial'] = adata_SM.obs.to_numpy()
# sc.pp.highly_variable_genes(adata_ST, flavor='seurat_v3', n_top_genes=4000)
# adata_ST = adata_ST[:,adata_ST.var['highly_variable']]
# sc.tl.pca(adata_ST, svd_solver='randomized')
# sc.tl.pca(adata_SM, svd_solver='randomized')

x_st,y_st,G_st=adata_ST.obsm['spatial'][:,0], adata_ST.obsm['spatial'][:,1], adata_ST.obsm['X_pca']
x_sm,y_sm,G_sm=adata_SM.obsm['spatial'][:,0], adata_SM.obsm['spatial'][:,1], adata_SM.obsm['X_pca']

output_st = rasterize_with_signal(x_st, y_st, G_st)
output_sm = rasterize_with_signal(x_sm, y_sm, G_sm)

xI = [
    torch.tensor(output_sm[0], device=device),
    torch.tensor(output_sm[1], device=device),
]
I = torch.nn.functional.normalize(torch.tensor(output_sm[2], dtype=torch.float32, device=device), p=2.0, dim = 0) * 100
xJ = [
    torch.tensor(output_st[0], device=device),
    torch.tensor(output_st[1], device=device),
]
J = torch.nn.functional.normalize(torch.tensor(output_st[2], dtype=torch.float32, device=device), p=2.0, dim = 0) * 100
pointsI = torch.tensor(np.vstack([x_sm, y_sm]).T, dtype=torch.double)
pointsJ = torch.tensor(np.vstack([x_st, y_st]).T, dtype=torch.double)


output = LLDDMM(
    xI = xI,
    I = I,
    xJ = xJ,
    J = J,
    pointsI = pointsI, 
    pointsJ = pointsJ
)


new_xy_st = transform_points_source_to_target(output['xv'], output['v'], output['A'], torch.tensor(np.vstack([x_st,y_st]).T, dtype=torch.double))
adata_ST.obsm['X_spatial_transformed'] = new_xy_st.detach().numpy()


pointsI=None
pointsJ=None
L=None
T=None
A=None
v=None
xv=None
a=500.0
p=2.0
expand=2.0
nt=3
niter=5000
diffeo_start=0
epL=2e-8
epT=2e-1
epV=2e3
sigmaM=1.0
sigmaB=2.0
sigmaA=5.0
sigmaR=5e5
sigmaP=2e1
device='cpu'
dtype=torch.float64
muB=None
muA=None
draw=False
