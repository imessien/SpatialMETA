import scanpy as sc
from anndata import AnnData

class AnnDataSM(AnnData):
    """
    Anndata object for Spatial Metabolomics data
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._spatialtk_type = 'AnnDataSM'

    @classmethod
    def from_anndata(cls, adata, *args, **kwargs):
        if not isinstance(adata, sc.AnnData):
            raise ValueError("Input must be a scanpy AnnData object")
        return cls(adata, *args, **kwargs)

    def _gen_repr(self, n_obs, n_vars) -> str:
        if self.isbacked:
            backed_at = f" backed at {str(self.filename)!r}"
        else:
            backed_at = ""
        descr = f"Spatial Metabolomics AnnData object with n_obs × n_vars = {n_obs} × {n_vars}{backed_at}"
        for attr in [
            "obs",
            "var",
            "uns",
            "obsm",
            "varm",
            "layers",
            "obsp",
            "varp",
        ]:
            keys = getattr(self, attr).keys()
            if len(keys) > 0:
                descr += f"\n    {attr}: {str(list(keys))[1:-1]}"
        return descr

    def __repr__(self) -> str:
        if self.is_view:
            return "View of " + self._gen_repr(self.n_obs, self.n_vars)
        else:
            return self._gen_repr(self.n_obs, self.n_vars)

class AnnDataST(AnnData):
    """
    Anndata object for Spatial Transcriptomics data
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._spatialtk_type = 'AnnDataST'

    @classmethod
    def from_anndata(cls, adata, *args, **kwargs):
        if not isinstance(adata, sc.AnnData):
            raise ValueError("Input must be a scanpy AnnData object")
        return cls(adata, *args, **kwargs)
    
    def _gen_repr(self, n_obs, n_vars) -> str:
        if self.isbacked:
            backed_at = f" backed at {str(self.filename)!r}"
        else:
            backed_at = ""
        descr = f"Spatial Transcriptomics AnnData object with n_obs × n_vars = {n_obs} × {n_vars}{backed_at}"
        for attr in [
            "obs",
            "var",
            "uns",
            "obsm",
            "varm",
            "layers",
            "obsp",
            "varp",
        ]:
            keys = getattr(self, attr).keys()
            if len(keys) > 0:
                descr += f"\n    {attr}: {str(list(keys))[1:-1]}"
        return descr

    def __repr__(self) -> str:
        if self.is_view:
            return "View of " + self._gen_repr(self.n_obs, self.n_vars)
        else:
            return self._gen_repr(self.n_obs, self.n_vars)
        
class AnnDataJointSMST(AnnData):
    """
    Anndata object for Joint Spatial Metabolomics and Transcriptomics data
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._spatialtk_type = 'AnnDataJointSTSM'

    def _gen_repr(self, n_obs, n_vars) -> str:
        if self.isbacked:
            backed_at = f" backed at {str(self.filename)!r}"
        else:
            backed_at = ""
        descr = f"Joint Spatial Transcriptomics and Metabolomics\n" + "AnnData object with n_obs × n_vars = {n_obs} × {n_vars}{backed_at}"
        for attr in [
            "obs",
            "var",
            "uns",
            "obsm",
            "varm",
            "layers",
            "obsp",
            "varp",
        ]:
            keys = getattr(self, attr).keys()
            if len(keys) > 0:
                descr += f"\n    {attr}: {str(list(keys))[1:-1]}"
        return descr

    def __repr__(self) -> str:
        if self.is_view:
            return "View of " + self._gen_repr(self.n_obs, self.n_vars)
        else:
            return self._gen_repr(self.n_obs, self.n_vars)