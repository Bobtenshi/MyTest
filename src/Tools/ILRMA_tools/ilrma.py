"""Implementation of independent low-rank matrix analysis (ILRMA)."""
import numpy as np

from Tools.ILRMA_tools.update_models import update_source_model, update_spatial_model
from Tools.ILRMA_tools.util import tensor_T

update_rules_single = ["IP1", "ISS1"]
update_rules_double = ["IP2", "ISS2"]


def ilrma(
    x,
    n_basis=2,
    method="IP1",
    # normalize_param=False,
    normalize_param=True,
    seed=None,
    B0=None,
    A0=None,
    W0=None,
):
    """
    Separates mixture with ILRMA.
    Parameters
    ----------
    x: ndarray (n_frame, n_freq, n_src)
        STFT representation of input mixture signal.
    n_basis: int, optional
        The number of bases of NMF.
        Default is 2.
    method: {"IP1", "IP2", "ISS1"}, optional
        The update method of demixing matrices.
    normalize_param: bool, optional
        If True, the parameters is normalized in each iteration.
        Default is False.
    B0: ndarray (n_src, n_freq, n_basis), optional
        Initial basis matrices.
        If None, `B0` is initialized as `np.ones((n_src, n_freq, n_basis))`.
        Default is None.
    A0: ndarray (n_src, n_frame, n_basis), optional
        If None, `A0` is initialized as `np.random.uniform(low=0.1, high=1.0, size=(n_src, n_frame, n_basis))`.
        Default is None.
    W0: ndarray (n_freq, n_src, n_src), optional
        If None, `W0[:, :, :n_src]` as `np.tile(np.eye(n_src, dtype=x.dtype), (n_freq, 1, 1))`.
        Default is None.
    Returns
    -------
    Updated `W`, `B`, `A`, and `R`.
    """
    n_frame, n_freq, n_src = x.shape

    # The frequency axis is first to improve computational efficiency
    # shape: (n_freq, n_src, n_frame)
    x = x.transpose([1, 2, 0]).copy()

    # initial demixing matrix
    if W0 is None:
        W0 = np.zeros((n_freq, n_src, n_src), dtype=x.dtype)
        W0[:, :, :n_src] = np.tile(np.eye(n_src, dtype=x.dtype), (n_freq, 1, 1))
    W = W0.copy()

    # save current numpy RNG state and set a known seed
    rng_state = None
    if seed is not None:
        rng_state = np.random.get_state()
        np.random.seed(seed)

    # source model parametes
    if B0 is None:
        B0 = np.ones((n_src, n_freq, n_basis))
    if A0 is None:
        A0 = np.random.uniform(low=0.1, high=1.0, size=(n_src, n_frame, n_basis))
    B, A = B0.copy(), A0.copy()

    # variance of source model
    # shape: (n_src, n_freq, n_frame)
    R = B @ tensor_T(A)

    # initialize the demixed outputs
    # shape: (n_freq, n_src, n_frame)
    y = W @ x
    y_power = np.square(abs(y))

    # if normalize_param:
    #    normalize(y_power, W, B, R)

    # TODO: インデックスを全部渡すようにupdate_rule変える？
    if method in update_rules_double:
        for s in range(0, n_src * 2, 2):
            m, n = s % n_src, (s + 1) % n_src
            tgt_idx = np.array([m, n])

            for l in tgt_idx:
                B[l], A[l] = update_source_model(y_power[:, l, :], B[l], A[l])
            R[tgt_idx, :, :] = B[tgt_idx, :, :] @ tensor_T(A[tgt_idx, :, :])

            W[:, :, :] = update_spatial_model(x, R, W, row_idx=tgt_idx, method=method)
    elif method in update_rules_single:
        for s in range(n_src):
            B[s], A[s] = update_source_model(y_power[:, s, :], B[s], A[s])
            R[s] = B[s] @ A[s].T

            W[:, :, :] = update_spatial_model(
                # abs(x.copy()), R, W, row_idx=s, method=method)
                x,
                R,
                W,
                row_idx=s,
                method=method,
            )

    # restore numpy RNG former state
    if seed is not None:
        np.random.set_state(rng_state)

    return W, B, A, R
