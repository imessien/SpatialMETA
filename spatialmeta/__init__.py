from ._metadata import __version__, within_flit
import importlib
import subprocess
import warnings


if not within_flit():  # see function docstring on why this is there
    # the actual API
    # (start with settings as several tools are using it)
    from . import model as model
    from . import tool as tl
    from . import preprocess as pp
    from . import plotting as pl
    from . import util as ut
    from . import data as data
    from . import external as ext

    from .util._classes import AnnDataSM, AnnDataST, AnnDataJointSMST

    from anndata import AnnData, concat
    from anndata import (
        read_h5ad,
        read_csv,
        read_excel,
        read_hdf,
        read_loom,
        read_mtx,
        read_text,
        read_umi_tools,
    )

    from .util._classes import AnnDataST, AnnDataSM
    def read_h5ad_st(*args, **kwargs):
        adata = read_h5ad(*args, **kwargs)
        return AnnDataST.from_anndata(adata)
    
    def read_h5ad_sm(*args, **kwargs):
        adata = read_h5ad(*args, **kwargs)
        return AnnDataSM.from_anndata(adata)
    

    # has to be done at the end, after everything has been imported
    import sys
    sys.modules.update({f'{__name__}.{m}': globals()[m] for m in ['model', 'tl', 'pp', 'pl', 'ut', 'data', 'ext']})
    del sys