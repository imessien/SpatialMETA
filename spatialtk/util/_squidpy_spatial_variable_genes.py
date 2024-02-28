import scanpy as sc
from anndata import AnnData 
import pandas as pd
from typing import (
    Any, Callable, Optional, Sequence, Union, Iterable, Tuple, Dict
)
from scipy.sparse import csr_matrix
from functools import partial
import numpy as np
from enum import Enum
from scanpy.get import _get_obs_rep
from scanpy.metrics._gearys_c import _gearys_c
from scanpy.metrics._morans_i import _morans_i
from scipy.spatial import Delaunay
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from itertools import chain
from scipy.sparse import (
    SparseEfficiencyWarning,
    block_diag,
    csr_matrix,
    isspmatrix_csr,
    spmatrix,
)
from enum import unique 
from queue import Queue
from os import cpu_count
from scipy import stats
from statsmodels.stats.multitest import multipletests
from threading import Thread
import joblib as jl
import functools
import inspect
import warnings
from contextlib import contextmanager
from enum import Enum
from multiprocessing import Manager, cpu_count
from numpy.random import default_rng


from ..util.compat import Literal


@unique
class Signal(Enum):
    """Signaling values when informing parallelizer."""

    NONE = 0
    UPDATE = 1
    FINISH = 2
    UPDATE_FINISH = 3

@unique
class CoordType(Enum):
    GRID = "grid"
    GENERIC = "generic"

@unique
class Transform(Enum):
    SPECTRAL = "spectral"
    COSINE = "cosine"
    NONE = None

class SigQueue(Queue):  # type: ignore[misc]
    """Signalling queue."""


class SpatialAutocorr(str, Enum):
    MORAN = "moran"
    GEARY = "geary"

class cprop:
    def __init__(self, f: Callable[..., str]):
        self.f = f

    def __get__(self, obj: Any, owner: Any) -> str:
        return self.f(owner)
    
class Key:
    class obs:
        pass

    class obsp:
        @classmethod
        def spatial_dist(cls, value: str= None) -> str:
            return f"{Key.obsm.spatial}_distances" if value is None else f"{value}_distances"

        @classmethod
        def spatial_conn(cls, value: str= None) -> str:
            return f"{Key.obsm.spatial}_connectivities" if value is None else f"{value}_connectivities"
        
    class obsm:
        @cprop
        def spatial(cls) -> str:
            return "spatial"

    class uns:
        @cprop
        def spatial(cls) -> str:
            return Key.obsm.spatial

        @cprop
        def image_key(cls) -> str:
            return "images"

        @cprop
        def image_res_key(cls) -> str:
            return "hires"

        @cprop
        def image_seg_key(cls) -> str:
            return "segmentation"

        @cprop
        def scalefactor_key(cls) -> str:
            return "scalefactors"

        @cprop
        def size_key(cls) -> str:
            return "spot_diameter_fullres"

        @classmethod
        def spatial_neighs(cls, value: str = None) -> str:
            return f"{Key.obsm.spatial}_neighbors" if value is None else f"{value}_neighbors"

        @classmethod
        def ligrec(cls, cluster: str, value: str = None) -> str:
            return f"{cluster}_ligrec" if value is None else value

        @classmethod
        def nhood_enrichment(cls, cluster: str) -> str:
            return f"{cluster}_nhood_enrichment"

        @classmethod
        def centrality_scores(cls, cluster: str) -> str:
            return f"{cluster}_centrality_scores"

        @classmethod
        def interaction_matrix(cls, cluster: str) -> str:
            return f"{cluster}_interactions"

        @classmethod
        def co_occurrence(cls, cluster: str) -> str:
            return f"{cluster}_co_occurrence"

        @classmethod
        def ripley(cls, cluster: str, mode: str) -> str:
            return f"{cluster}_ripley_{mode}"

        @classmethod
        def colors(cls, cluster: str) -> str:
            return f"{cluster}_colors"
        

def _build_grid(
    coords: np.ndarray, n_neighs: int, n_rings: int, delaunay: bool = False, set_diag: bool = False
) -> Tuple[csr_matrix, csr_matrix]:
    if n_rings > 1:
        Adj: csr_matrix = _build_connectivity(
            coords,
            n_neighs=n_neighs,
            neigh_correct=True,
            set_diag=True,
            delaunay=delaunay,
            return_distance=False,
        )
        Res, Walk = Adj, Adj
        for i in range(n_rings - 1):
            Walk = Walk @ Adj
            Walk[Res.nonzero()] = 0.0
            Walk.eliminate_zeros()
            Walk.data[:] = i + 2.0
            Res = Res + Walk
        Adj = Res
        Adj.setdiag(float(set_diag))
        Adj.eliminate_zeros()

        Dst = Adj.copy()
        Adj.data[:] = 1.0
    else:
        Adj = _build_connectivity(coords, n_neighs=n_neighs, neigh_correct=True, delaunay=delaunay, set_diag=set_diag)
        Dst = Adj.copy()

    Dst.setdiag(0.0)

    return Adj, Dst


def _build_connectivity(
    coords: np.ndarray,
    n_neighs: int,
    radius: float  = None,
    delaunay: bool = False,
    neigh_correct: bool = False,
    set_diag: bool = False,
    return_distance: bool = False,
) -> csr_matrix:
    N = coords.shape[0]
    if delaunay:
        tri = Delaunay(coords)
        indptr, indices = tri.vertex_neighbor_vertices
        Adj = csr_matrix((np.ones_like(indices, dtype=np.float64), indices, indptr), shape=(N, N))

        if return_distance:
            # fmt: off
            dists = np.array(list(chain(*(
                euclidean_distances(coords[indices[indptr[i] : indptr[i + 1]], :], coords[np.newaxis, i, :])
                for i in range(N)
                if len(indices[indptr[i] : indptr[i + 1]])
            )))).squeeze()
            Dst = csr_matrix((dists, indices, indptr), shape=(N, N))
            # fmt: on
    else:
        r = 1 if radius is None else radius if isinstance(radius, (int, float)) else max(radius)
        tree = NearestNeighbors(n_neighbors=n_neighs, radius=r, metric="euclidean")
        tree.fit(coords)

        if radius is None:
            results = tree.kneighbors()
            dists, row_indices = (result.reshape(-1) for result in results)
            col_indices = np.repeat(np.arange(N), n_neighs)
            if neigh_correct:
                dist_cutoff = np.median(dists) * 1.3  # there's a small amount of sway
                mask = dists < dist_cutoff
                row_indices, col_indices = row_indices[mask], col_indices[mask]
                dists = dists[mask]
        else:
            results = tree.radius_neighbors()
            dists = np.concatenate(results[0])
            row_indices = np.concatenate(results[1])
            col_indices = np.repeat(np.arange(N), [len(x) for x in results[1]])

        Adj = csr_matrix((np.ones_like(row_indices, dtype=np.float64), (row_indices, col_indices)), shape=(N, N))
        if return_distance:
            Dst = csr_matrix((dists, (row_indices, col_indices)), shape=(N, N))

    # radius-based filtering needs same indices/indptr: do not remove 0s
    Adj.setdiag(1.0 if set_diag else Adj.diagonal())
    if return_distance:
        Dst.setdiag(0.0)
        return Adj, Dst

    return Adj

def outer(indices: np.ndarray, indptr: np.ndarray, degrees: np.ndarray) -> np.ndarray:
    res = np.empty_like(indices, dtype=np.float64)
    start = 0
    for i in range(len(indptr) - 1):
        ixs = indices[indptr[i] : indptr[i + 1]]
        res[start : start + len(ixs)] = degrees[i] * degrees[ixs]
        start += len(ixs)

    return res


def _transform_a_spectral(a: spmatrix) -> spmatrix:
    if not isspmatrix_csr(a):
        a = a.tocsr()
    if not a.nnz:
        return a

    degrees = np.squeeze(np.array(np.sqrt(1.0 / a.sum(axis=0))))
    a = a.multiply(outer(a.indices, a.indptr, degrees))
    a.eliminate_zeros()

    return a


def _transform_a_cosine(a: spmatrix) -> spmatrix:
    return cosine_similarity(a, dense_output=False)

def _spatial_neighbor(
    adata: AnnData,
    spatial_key: str = "spatial",
    coord_type: str = 'None',
    n_neighs: int = 6,
    radius: float = None,
    delaunay: bool = False,
    n_rings: int = 1,
    transform: str = None,
    set_diag: bool = False,
    percentile: float = None,
) -> Tuple[csr_matrix, csr_matrix]:
    coords = adata.obsm[spatial_key]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SparseEfficiencyWarning)
        if coord_type == CoordType.GRID:
            Adj, Dst = _build_grid(coords, n_neighs=n_neighs, n_rings=n_rings, delaunay=delaunay, set_diag=set_diag)
        elif coord_type == CoordType.GENERIC:
            Adj, Dst = _build_connectivity(
                coords, n_neighs=n_neighs, radius=radius, delaunay=delaunay, return_distance=True, set_diag=set_diag
            )
        else:
            raise NotImplementedError(f"Coordinate type `{coord_type}` is not yet implemented.")

    if coord_type == CoordType.GENERIC and isinstance(radius, Iterable):
        minn, maxx = sorted(radius)[:2]
        mask = (Dst.data < minn) | (Dst.data > maxx)
        a_diag = Adj.diagonal()

        Dst.data[mask] = 0.0
        Adj.data[mask] = 0.0
        Adj.setdiag(a_diag)

    if percentile is not None and coord_type == CoordType.GENERIC:
        threshold = np.percentile(Dst.data, percentile)
        Adj[Dst > threshold] = 0.0
        Dst[Dst > threshold] = 0.0

    Adj.eliminate_zeros()
    Dst.eliminate_zeros()

    # check transform
    if transform == Transform.SPECTRAL:
        Adj = _transform_a_spectral(Adj)
    elif transform == Transform.COSINE:
        Adj = _transform_a_cosine(Adj)
    elif transform == Transform.NONE:
        pass
    else:
        raise NotImplementedError(f"Transform `{transform}` is not yet implemented.")

    return Adj, Dst


def _save_data(adata: AnnData, *, attr: str, key: str, data: Any, prefix: bool = True, time: Any = None) -> None:
    obj = getattr(adata, attr)
    obj[key] = data


def spatial_neighbors(
    adata: AnnData,
    spatial_key: str = 'spatial',
    library_key: str = None,
    coord_type: str  = None,
    n_neighs: int = 6,
    radius: float = None,
    delaunay: bool = False,
    n_rings: int = 1,
    percentile: float = None,
    transform: str = None,
    set_diag: bool = False,
    key_added: str = "spatial",
    copy: bool = False,
) -> Tuple[csr_matrix, csr_matrix]:
    """
    Create a graph from spatial coordinates.

    Parameters
    ----------
    %(adata)s
    %(spatial_key)s
    %(library_key)s
    coord_type
        Type of coordinate system. Valid options are:

            - `{c.GRID.s!r}` - grid coordinates.
            - `{c.GENERIC.s!r}` - generic coordinates.
            - `None` - `{c.GRID.s!r}` if ``spatial_key`` is in :attr:`anndata.AnnData.uns`
              with ``n_neighs = 6`` (Visium), otherwise use `{c.GENERIC.s!r}`.
    n_neighs
        Depending on the ``coord_type``:

            - `{c.GRID.s!r}` - number of neighboring tiles.
            - `{c.GENERIC.s!r}` - number of neighborhoods for non-grid data. Only used when ``delaunay = False``.
    radius
        Only available when ``coord_type = {c.GENERIC.s!r}``. Depending on the type:

            - :class:`float` - compute the graph based on neighborhood radius.
            - :class:`tuple` - prune the final graph to only contain edges in interval `[min(radius), max(radius)]`.
    delaunay
        Whether to compute the graph from Delaunay triangulation. Only used when ``coord_type = {c.GENERIC.s!r}``.
    n_rings
        Number of rings of neighbors for grid data. Only used when ``coord_type = {c.GRID.s!r}``.
    percentile
        Percentile of the distances to use as threshold. Only used when ``coord_type = {c.GENERIC.s!r}``.
    transform
        Type of adjacency matrix transform. Valid options are:

            - `{t.SPECTRAL.s!r}` - spectral transformation of the adjacency matrix.
            - `{t.COSINE.s!r}` - cosine transformation of the adjacency matrix.
            - `{t.NONE.v}` - no transformation of the adjacency matrix.
    set_diag
        Whether to set the diagonal of the spatial connectivities to `1.0`.
    key_added
        Key which controls where the results are saved if ``copy = False``.
    %(copy)s

    Returns
    -------
    If ``copy = True``, returns a :class:`tuple` with the spatial connectivities and distances matrices.

    Otherwise, modifies the ``adata`` with the following keys:

        - :attr:`anndata.AnnData.obsp` ``['{{key_added}}_connectivities']`` - the spatial connectivities.
        - :attr:`anndata.AnnData.obsp` ``['{{key_added}}_distances']`` - the spatial distances.
        - :attr:`anndata.AnnData.uns`  ``['{{key_added}}']`` - :class:`dict` containing parameters.
    """

    transform = Transform.NONE if transform is None else Transform(transform)
    if coord_type is None:
        if radius is not None:
            print(
                f"Graph creation with `radius` is only available when `coord_type = {CoordType.GENERIC!r}` specified. "
                f"Ignoring parameter `radius = {radius}`."
            )
        coord_type = CoordType.GRID if 'spatial' in adata.uns else CoordType.GENERIC
    else:
        coord_type = CoordType(coord_type)

    if library_key is not None:
        libs = adata.obs[library_key].cat.categories
    else:
        libs = [None]

    _build_fun = partial(
        _spatial_neighbor,
        spatial_key=spatial_key,
        coord_type=coord_type,
        n_neighs=n_neighs,
        radius=radius,
        delaunay=delaunay,
        n_rings=n_rings,
        transform=transform,
        set_diag=set_diag,
        percentile=percentile,
    )

    if library_key is not None:
        mats: list[Tuple[spmatrix, spmatrix]] = []
        ixs = []  # type: ignore[var-annotated]
        for lib in libs:
            ixs.extend(np.where(adata.obs[library_key] == lib)[0])
            mats.append(_build_fun(adata[adata.obs[library_key] == lib]))
        ixs = np.argsort(ixs)  # type: ignore[assignment] # invert
        Adj = block_diag([m[0] for m in mats], format="csr")[ixs, :][:, ixs]
        Dst = block_diag([m[1] for m in mats], format="csr")[ixs, :][:, ixs]
    else:
        Adj, Dst = _build_fun(adata)

    neighs_key = Key.uns.spatial_neighs(key_added)
    conns_key = Key.obsp.spatial_conn(key_added)
    dists_key = Key.obsp.spatial_dist(key_added)

    neighbors_dict = {
        "connectivities_key": conns_key,
        "distances_key": dists_key,
        "params": {"n_neighbors": n_neighs, "radius": radius},
    }

    if copy:
        return Adj, Dst

    _save_data(adata, attr="obsp", key=conns_key, data=Adj)
    _save_data(adata, attr="obsp", key=dists_key, data=Dst, prefix=False)
    _save_data(adata, attr="uns", key=neighs_key, data=neighbors_dict, prefix=False)


def _g_moments(w: Union[spmatrix, np.ndarray]) -> Tuple[float, float, float]:
    """
    Compute moments of adjacency matrix for analytic p-value calculation.

    See `pysal <https://pysal.org/libpysal/_modules/libpysal/weights/weights.html#W>`_ implementation.
    """
    # s0
    s0 = w.sum()

    # s1
    t = w.transpose() + w
    t2 = t.multiply(t)  # type: ignore[union-attr]
    s1 = t2.sum() / 2.0

    # s2
    s2array: np.ndarray = np.array(w.sum(1) + w.sum(0).transpose()) ** 2
    s2 = s2array.sum()

    return s0, s1, s2


def _analytic_pval(score: np.ndarray, g: Union[spmatrix, np.ndarray], params: Dict[str, Any]) -> Tuple[np.ndarray, float]:
    """
    Analytic p-value computation.

    See `Moran's I <https://pysal.org/esda/_modules/esda/moran.html#Moran>`_ and
    `Geary's C <https://pysal.org/esda/_modules/esda/geary.html#Geary>`_ implementation.
    """
    s0, s1, s2 = _g_moments(g)
    n = g.shape[0]
    s02 = s0 * s0
    n2 = n * n
    v_num = n2 * s1 - n * s2 + 3 * s02
    v_den = (n - 1) * (n + 1) * s02

    Vscore_norm = v_num / v_den - (1.0 / (n - 1)) ** 2
    seScore_norm = Vscore_norm ** (1 / 2.0)

    z_norm = (score - params["expected"]) / seScore_norm
    p_norm = np.empty(score.shape)
    p_norm[z_norm > 0] = 1 - stats.norm.cdf(z_norm[z_norm > 0])
    p_norm[z_norm <= 0] = stats.norm.cdf(z_norm[z_norm <= 0])

    if params["two_tailed"]:
        p_norm *= 2.0

    return p_norm, Vscore_norm


def _p_value_calc(
    score: np.ndarray,
    sims: np.ndarray,
    weights: Union[spmatrix, np.ndarray],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Handle p-value calculation for spatial autocorrelation function.

    Parameters
    ----------
    score
        (n_features,).
    sims
        (n_simulations, n_features).
    params
        Object to store relevant function parameters.

    Returns
    -------
    pval_norm
        p-value under normality assumption
    pval_sim
        p-values based on permutations
    pval_z_sim
        p-values based on standard normal approximation from permutations
    """
    p_norm, var_norm = _analytic_pval(score, weights, params)
    results = {"pval_norm": p_norm, "var_norm": var_norm}

    if sims is None:
        return results

    n_perms = sims.shape[0]
    large_perm = (sims >= score).sum(axis=0)
    # subtract total perm for negative values
    large_perm[(n_perms - large_perm) < large_perm] = n_perms - large_perm[(n_perms - large_perm) < large_perm]
    # get p-value based on permutation
    p_sim = (large_perm + 1) / (n_perms + 1)

    # get p-value based on standard normal approximation from permutations
    e_score_sim = sims.sum(axis=0) / n_perms
    se_score_sim = sims.std(axis=0)
    z_sim = (score - e_score_sim) / se_score_sim
    p_z_sim = np.empty(z_sim.shape)

    p_z_sim[z_sim > 0] = 1 - stats.norm.cdf(z_sim[z_sim > 0])
    p_z_sim[z_sim <= 0] = stats.norm.cdf(z_sim[z_sim <= 0])

    var_sim = np.var(sims, axis=0)

    results["pval_z_sim"] = p_z_sim
    results["pval_sim"] = p_sim
    results["var_sim"] = var_sim

    return results


def parallelize(
    callback: Callable[..., Any],
    collection: Sequence[Any],
    n_jobs: int = 1,
    n_split: int = None,
    unit: str = "",
    use_ixs: bool = False,
    backend: str = "loky",
    extractor: Callable[[Sequence[Any]], Any] = None,
    show_progress_bar: bool = True,
    use_runner: bool = False,
    **_: Any,
) -> Any:
    """
    Parallelize function call over a collection of elements.

    Parameters
    ----------
    callback
        Function to parallelize. Can either accept a whole chunk (``use_runner=False``) or just a single
        element (``use_runner=True``).
    collection
        Sequence of items to split into chunks.
    n_jobs
        Number of parallel jobs.
    n_split
        Split ``collection`` into ``n_split`` chunks.
        If <= 0, ``collection`` is assumed to be already split into chunks.
    unit
        Unit of the progress bar.
    use_ixs
        Whether to pass indices to the callback.
    backend
        Which backend to use for multiprocessing. See :class:`joblib.Parallel` for valid options.
    extractor
        Function to apply to the result after all jobs have finished.
    show_progress_bar
        Whether to show a progress bar.
    use_runner
        Whether the ``callback`` handles only 1 item from the ``collection`` or a chunk.
        The latter grants more control, e.g. using :func:`numba.prange` instead of normal iteration.

    Returns
    -------
    The result depending on ``callable``, ``extractor``.
    """
    if show_progress_bar:
        try:
            import ipywidgets  # noqa: F401
            from tqdm.auto import tqdm
        except ImportError:
            try:
                from tqdm.std import tqdm
            except ImportError:
                tqdm = None
    else:
        tqdm = None

    def runner(iterable: Iterable[Any], *args: Any, queue: SigQueue = None, **kwargs: Any) -> list[Any]:
        result: list[Any] = []

        for it in iterable:
            res = callback(it, *args, **kwargs)
            if res is not None:
                result.append(result)
            if queue is not None:
                queue.put(Signal.UPDATE)

        if queue is not None:
            queue.put(Signal.FINISH)

        return result

    def update(pbar: tqdm.std.tqdm, queue: SigQueue, n_total: int) -> None:
        n_finished = 0
        while n_finished < n_total:
            try:
                res = queue.get()
            except EOFError as e:
                if not n_finished != n_total:
                    raise RuntimeError(f"Finished only `{n_finished}` out of `{n_total}` tasks.") from e
                break

            assert isinstance(res, Signal), f"Invalid type `{type(res).__name__}`."

            if res in (Signal.FINISH, Signal.UPDATE_FINISH):
                n_finished += 1
            if pbar is not None and res in (Signal.UPDATE, Signal.UPDATE_FINISH):
                pbar.update()

        if pbar is not None:
            pbar.close()

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if pass_queue and show_progress_bar:
            pbar = None if tqdm is None else tqdm(total=col_len, unit=unit)
            queue = Manager().Queue()
            thread = Thread(target=update, args=(pbar, queue, len(collections)))
            thread.start()
        else:
            pbar, queue, thread = None, None, None

        res = jl.Parallel(n_jobs=n_jobs, backend=backend)(
            jl.delayed(runner if use_runner else callback)(
                *((i, cs) if use_ixs else (cs,)),
                *args,
                **kwargs,
                queue=queue,
            )
            for i, cs in enumerate(collections)
        )

        if thread is not None:
            thread.join()

        return res if extractor is None else extractor(res)

    if n_jobs is None:
        n_jobs = 1
    if n_jobs == 0:
        raise ValueError("Number of jobs cannot be `0`.")
    elif n_jobs < 0:
        n_jobs = cpu_count() + 1 + n_jobs

    if n_split is None:
        n_split = n_jobs

    if n_split <= 0:
        col_len = sum(map(len, collection))
        collections = collection
    else:
        col_len = len(collection)
        step = int(np.ceil(len(collection) / n_split))
        collections = list(
            filter(len, (collection[i * step : (i + 1) * step] for i in range(int(np.ceil(col_len / step)))))
        )

    if use_runner:
        use_ixs = False
    pass_queue = not hasattr(callback, "py_func")  # we'd be inside a numba function

    return wrapper

def _score_helper(
    ix: int,
    perms: Sequence[int],
    mode: SpatialAutocorr,
    g: spmatrix,
    vals: np.ndarray,
    seed: int = None,
    queue: SigQueue = None,
) -> pd.DataFrame:
    score_perms = np.empty((len(perms), vals.shape[0]))
    rng = default_rng(None if seed is None else ix + seed)
    func = _morans_i if mode == SpatialAutocorr.MORAN else _gearys_c

    for i in range(len(perms)):
        idx_shuffle = rng.permutation(g.shape[0])
        score_perms[i, :] = func(g[idx_shuffle, :], vals)

        if queue is not None:
            queue.put(Signal.UPDATE)

    if queue is not None:
        queue.put(Signal.FINISH)

    return score_perms


def _get_n_cores(n_cores: int) -> int:
    """
    Make number of cores a positive integer.

    This is useful for especially logging.

    Parameters
    ----------
    n_cores
        Number of cores to use.

    Returns
    -------
    int
        Positive integer corresponding to how many cores to use.
    """
    if n_cores == 0:
        raise ValueError("Number of cores cannot be `0`.")
    if n_cores is None:
        return 1
    if n_cores < 0:
        return cpu_count() + 1 + n_cores

    return n_cores

def spatial_autocorr(
    adata: AnnData,
    connectivity_key: str = "spatial_connectivities",
    genes: Union[str, Sequence[str]] = None,
    mode: Literal["moran", "geary"] = 'moran',
    transformation: bool = True,
    n_perms: int = None,
    two_tailed: bool = False,
    corr_method: str = "fdr_bh",
    attr: Literal["obs", "X", "obsm"] = "X",
    layer: str = None,
    seed: int = None,
    use_raw: bool = False,
    copy: bool = False,
    n_jobs: int = None,
    backend: str = "loky",
    show_progress_bar: bool = True,
) -> pd.DataFrame:
    genes = None

    def extract_X(adata: AnnData, genes: str = None):
        if genes is None:
            if "highly_variable" in adata.var:
                genes = adata[:, adata.var["highly_variable"]].var_names.values
            else:
                genes = adata.var_names.values
        elif isinstance(genes, str):
            genes = [genes]

        if not use_raw:
            return _get_obs_rep(adata[:, genes], use_raw=False, layer=layer).T, genes
        if adata.raw is None:
            raise AttributeError("No `.raw` attribute found. Try specifying `use_raw=False`.")
        genes = list(set(genes) & set(adata.raw.var_names))
        return adata.raw[:, genes].X.T, genes

    def extract_obs(adata: AnnData, cols = None):
        if cols is None:
            df = adata.obs.select_dtypes(include=np.number)
            return df.T.to_numpy(), df.columns
        if isinstance(cols, str):
            cols = [cols]
        return adata.obs[cols].T.to_numpy(), cols

    def extract_obsm(adata: AnnData, ixs: int = None):
        if layer not in adata.obsm:
            raise KeyError(f"Key `{layer!r}` not found in `adata.obsm`.")
        if ixs is None:
            ixs = np.arange(adata.obsm[layer].shape[1])  # type: ignore[assignment]
        ixs = list(np.ravel([ixs]))
        return adata.obsm[layer][:, ixs].T, ixs

    if attr == "X":
        vals, index = extract_X(adata, genes)  # type: ignore[arg-type]
    elif attr == "obs":
        vals, index = extract_obs(adata, genes)  # type: ignore[arg-type]
    elif attr == "obsm":
        vals, index = extract_obsm(adata, genes)  # type: ignore[arg-type]
    else:
        raise NotImplementedError(f"Extracting from `adata.{attr}` is not yet implemented.")

    mode = SpatialAutocorr(mode)  # type: ignore[assignment]
    params = {"transformation": transformation, "two_tailed": two_tailed}

    if mode == SpatialAutocorr.MORAN:
        params["func"] = _morans_i
        params["stat"] = "I"
        params["expected"] = -1.0 / (adata.shape[0] - 1)  # expected score
        params["ascending"] = False
        params['mode'] = 'moran'
    elif mode == SpatialAutocorr.GEARY:
        params["func"] = _gearys_c
        params["stat"] = "C"
        params["expected"] = 1.0
        params["ascending"] = True
        params['mode'] = 'geary'
    else:
        raise NotImplementedError(f"Mode `{mode}` is not yet implemented.")

    g = adata.obsp[connectivity_key].copy()
    if transformation:  # row-normalize
        normalize(g, norm="l1", axis=1, copy=False)

    score = params["func"](g, vals)

    n_jobs = _get_n_cores(n_jobs)
    if n_perms is not None:
        perms = np.arange(n_perms)

        score_perms = parallelize(
            _score_helper,
            collection=perms,
            extractor=np.concatenate,
            use_ixs=True,
            n_jobs=n_jobs,
            backend=backend,
            show_progress_bar=show_progress_bar,
        )(mode=mode, g=g, vals=vals, seed=seed)
    else:
        score_perms = None

    with np.errstate(divide="ignore"):
        pval_results = _p_value_calc(score, score_perms, g, params)

    df = pd.DataFrame({params["stat"]: score, **pval_results}, index=index)

    if corr_method is not None:
        for pv in filter(lambda x: "pval" in x, df.columns):
            _, pvals_adj, _, _ = multipletests(df[pv].values, alpha=0.05, method=corr_method)
            df[f"{pv}_{corr_method}"] = pvals_adj

    df.sort_values(by=params["stat"], ascending=params["ascending"], inplace=True)
    _save_data(adata, attr="uns", key=params["mode"] + params["stat"], data=df)