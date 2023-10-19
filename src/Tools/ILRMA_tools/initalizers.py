"""Initialize models of IVA or ILRMA."""
import numpy as np
from Tools.ILRMA_tools.util import tensor_T
from constant import N_BASIS  # , kwargs

kwargs = {"init_basis": "ramdom"}


def init_demix(observed, **kwargs):
    """
    Initialize NMF parameters to be updated in ILRMA.
    Parameters
    ----------
    observed: ndarray (n_frame, n_freq, n_src)
    kwargs: dict
    Returns
    -------
    demix_matrix: (n_freq, n_src, n_src)
        Initialized demixing matrix.
    """
    # from bss.utils import optimal_demix

    _, n_freq, n_src = observed.shape

    key = kwargs["init_demix"]
    allowed = {
        "identity": np.tile(np.eye(n_src, dtype=complex), (n_freq, 1, 1)),
        "random": np.random.rand(n_freq, n_src, n_src)
        + 1j * np.random.rand(n_freq, n_src, n_src),
        # "optimal": optimal_demix(kwargs["source"], observed),
    }
    demix = allowed[key]

    return demix


def init_basis(source, **kwargs):
    """
    Initialize basis matrix in ILRMA.
    Parameters
    ----------
    source: ndarray (n_frame, n_freq, n_src)
    kwargs: dict
    Returns
    -------
    basis: ndarray (n_src, n_freq, n_basis)
        Initialized basis matrix.
        Each element is non negative.
    """
    _, n_freq, n_src = source.shape
    n_basis = kwargs["n_basis"]
    # n_basis = N_BASIS

    basis = {
        "random": np.random.uniform(low=0.01, high=1.0, size=(n_src, n_freq, n_basis))
        + 0.1,
        "ones": np.ones((n_src, n_freq, n_basis)),
    }[kwargs["init_basis"]]

    return basis


def init_activ(source, **kwargs):
    """
    Initialize NMF parameters to be updated in ILRMA.
    Parameters
    ----------
    source: ndarray (n_frame, n_freq, n_src)
    kwargs: dict
    Returns
    -------
    activ: ndarray (n_src, n_frame, n_basis)
        Initialized activation matrix.
        Each element is non negative.
    """
    n_frame, _, n_src = source.shape
    n_basis = kwargs["n_basis"]
    # n_basis = N_BASIS

    activ = {
        "random": np.random.uniform(low=0.01, high=1.0, size=(n_src, n_frame, n_basis))
        + 0.1,
        "ones": np.ones((n_src, n_frame, n_basis)),
    }[kwargs["init_activ"]]

    return activ


def init_model(source, **kwargs):
    """
    Initialize NMF parameters to be updated in ILRMA.
    Parameters
    ----------
    source: ndarray (n_frame, n_freq, n_src)
    kwargs: dict
    Returns
    -------
    model: ndarray (n_src, n_freq, n_frame)
        Initialized activation matrix.
        Each element is non negative.
    """
    model = {
        "nmf": kwargs["basis"] @ tensor_T(kwargs["activ"]),
        "optimal": np.abs(source.transpose([2, 1, 0])) ** 2,
    }[kwargs["init_model"]]

    if "eps" in kwargs.keys():
        eps = kwargs["eps"]
        model[model < eps] = eps

    return model
