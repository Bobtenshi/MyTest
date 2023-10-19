import numpy as np


def tensor_T(A):
    """Compute transpose for tensor."""
    return A.swapaxes(-2, -1)


def tensor_H(A):
    """Compute Hermitian transpose for tensor."""
    return np.conj(A).swapaxes(-2, -1)


def solve_2x2HEAD(V1, V2, method="ono", eig_reverse=False):
    """
    Solve a 2x2 HEAD problem with given two positive semi-definite matrices.
    Parameters
    ----------
    V1: (n_freq, 2, 2)
    V2: (n_freq, 2, 2)
    method: "numpy" or "ono"
        If "numpy", `eigval` is calculated by using `numpy.linalg.eig`.
        If "ono", `eigval` is calculated by the method presented in Ono2012IWAENC.
    eig_reverse: bool
        If True, eigenvalues is sorted in *ascending* order.
        Default is False.
        This parameter will be deprecated in the future.
    Returns
    -------
    eigval: (n_freq, 2)
        eigenvalues, must be real numbers
    eigvec: (2, n_freq, 2)
        eigenvectors corresponding to the eigenvalues
    """
    V_hat = np.array([V1, V2])
