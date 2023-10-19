"""Update BSS paramters."""
import numpy as np
from .util import tensor_H, tensor_T, solve_2x2HEAD

# from icecream import ic

NP_EPS = np.finfo(np.float64).eps
methods_dict = {"IP1": (1, 1), "IP2": (2, 2), "ISS1": (1, 1), "ISS2": (2, 2)}


# TODO: write test code
debug = False


def print_residual_2x2head(u1, u2, V1, V2):
    """
    Print the residual of HEAD problem.
    Parameters
    ----------
    u1, u2: (n_freq, 2)
    V1, V2: (n_freq, 2, 2)
    """
    head11 = (u1[:, None, :].conj() @ V1 @ u1[:, :, None]).squeeze().real.mean()
    head21 = (u2[:, None, :].conj() @ V1 @ u1[:, :, None]).squeeze().real.mean()

    head12 = (u1[:, None, :].conj() @ V2 @ u2[:, :, None]).squeeze().real.mean()
    head22 = (u2[:, None, :].conj() @ V2 @ u2[:, :, None]).squeeze().real.mean()
    ic(np.array([[head11, head12], [head21, head22]]))


def update_source_model(y_power, B, A, eps=NP_EPS):
    """
    Update source model parameters `B`, `A` corresponding one source with multiplicative update rules.
    Parameters
    ----------
    y_power: ndarray (n_freq, n_frame)
        Power spectrograms of demixed signals.
    B: ndarray (n_freq, n_basis)
        Basis matrices of source spectrograms.
    A: ndarray (n_frame, n_basis)
        Activity matrices of source spectrograms.
    eps: float, optional
        `B` and `A` are pre-processed `B[B < eps] = eps` to improve numerical stability.
        Default is `np.finfo(np.float64).eps`.
    Returns
    -------
    Updated B and A
    """
    R = B @ A.T
    iR = np.reciprocal(R)

    B *= (y_power * np.square(iR)) @ A / (iR @ A)
    B[B < eps] = eps

    R = B @ A.T
    iR = np.reciprocal(R)

    A *= (y_power.T * np.square(iR.T)) @ B / (iR.T @ B)
    A[A < eps] = eps

    return B, A


def _ip_1(x, R, W, row_idx):
    """
    Update one demixing vector with IP1 algorithm.
    Parameters
    ----------
    x: ndarray (n_freq, n_src, n_frame)
        Power spectrograms of demixed signals.
    R: ndarray (n_freq, n_frame)
    R: ndarray (n_src, n_freq, n_frame)
        Variance matrices of source spectrograms.
    W: ndarray (n_freq, n_src, n_src)
        Demixing filters.
    row_idx: int
        The index of row vector of W.
    Returns
    -------
    Updated W
    """
    _, n_src, n_frame = x.shape
    iR = np.reciprocal(R)  # element-wise reciprocal number of R

    # shape: (n_freq, n_src, n_src)
    C = (x * iR[row_idx, :, None, :]) @ tensor_H(x) / n_frame
    w = np.conj(np.linalg.solve(W @ C, np.eye(n_src)[None, :, row_idx]))
    denom = (w[:, None, :] @ C) @ np.conj(w[:, :, None])
    W[:, row_idx, :] = w / np.sqrt(denom[:, :, 0])

    return W


def _ip_2(x, R, W, row_idx):
    """
    Update two demixing vectors with IP-2 algorithm.
    Parameters
    ----------
    x: ndarray (n_freq, n_src, n_frame)
        Power spectrograms of demixed signals.
    R: ndarray (n_src, n_freq, n_frame)
        Variance matrices of source spectrograms.
    W: ndarray (n_freq, n_src, n_src)
        Demixing filters.
    row_idx: ndarray (2,)
        The indeces of row vector of W.
    Returns
    -------
    Updated W
    """
    _, n_src, n_frames = x.shape

    # shape: (2, n_freq, n_src, n_src)
    V = (
        (x[None, :, :, :] / R[row_idx, :, None, :])
        @ tensor_H(x[None, :, :, :])
        / n_frames
    )

    # shape: (2, n_freq, n_src, 2)
    try:
        P = np.linalg.solve(W[None, :, :, :] @ V, np.eye(n_src)[None, None, :, row_idx])
    except:
        return W

    # shape: (2, n_freq, 2, 2)
    U = tensor_H(P) @ V @ P

    # Eigen vectors of U[1] @ inv(U[0])
    # shape: (2, n_freq, 2)
    _, u = solve_2x2HEAD(U[0], U[1])
    if debug:
        print_residual_2x2head(u[0], u[1], U[0], U[1])

    W[:, row_idx, :, None] = (P @ u[:, :, :, None]).swapaxes(0, 1).conj()

    return W


def _iss_1(x, R, W, row_idx, flooring=True, eps=NP_EPS):
    """
    Update all demixing vectors with ISS algorithm.
    Parameters
    ----------
    x: ndarray (n_freq, n_src, n_frame)
        Power spectrograms of demixed signals.
    R: ndarray (n_src, n_freq, n_frame)
        Variance matrices of source spectrograms.
    W: ndarray (n_freq, n_src, n_src)
        Demixing filters.
    row_idx: int
        The index of row vector of W.
    flooring: bool, optional
        If True, flooring is processed.
        Default is True.
    Returns
    -------
    Updated W
    """
    y = W @ x
    n_freq, n_src, n_frame = y.shape

    # (n_freq, n_src, n_frame)
    iR = np.reciprocal(R.transpose([1, 0, 2]))

    # update coefficients of W
    v_denom = np.zeros((n_freq, n_src), dtype=x.dtype)
    v = np.zeros((n_freq, n_src), dtype=x.dtype)

    # separation
    v[:, :, None] = (y * iR) @ np.conj(y[:, row_idx, :, None])
    v_denom[:, :, None] = iR @ np.square(np.abs(y[:, row_idx, :, None]))

    # flooring is processed to improve numerical stability
    if flooring:
        v_denom[v_denom < eps] = eps
    v[:, :] /= v_denom
    v[:, row_idx] -= 1 / np.sqrt(v_denom[:, row_idx] / n_frame)

    # update demixing matrices and demixed signals
    a = v[:, :, None] * W[:, row_idx, None, :]
    W[:, :, :] -= v[:, :, None] * W[:, row_idx, None, :]
    y[:, :, :] -= v[:, :, None] * y[:, row_idx, None, :]

    return W


def _iss_2(x, R, W, row_idx):
    """
    Update all demixing vectors with ISS-2 algorithm.
    Parameters
    ----------
    x: ndarray (n_freq, n_src, n_frame)
        Power spectrograms of demixed signals.
    R: ndarray (n_src, n_freq, n_frame)
        Variance matrices of source spectrograms.
    W: ndarray (n_freq, n_src, n_src)
        Demixing filters.
    row_idx: ndarray (2,)
        The indeces of row vector of W.
    Returns
    -------
    Updated W
    """
    n_freq, n_src, n_frames = x.shape

    # shape: (n_src, n_freq, n_src, n_src)
    V = (x[None, :, :, :] / R[:, :, None, :]) @ tensor_H(x[None, :, :, :]) / n_frames

    u = np.zeros((n_freq, n_src, 2), dtype=x.dtype)

    # TODO: vectorize
    no_tgt = [s for s in range(n_src) if s not in row_idx]
    for s in no_tgt:
        # (n_freq, 2, 2)
        y_cors_mat = W[:, row_idx, :] @ V[s, :, :, :] @ tensor_H(W[:, row_idx, :])

        # (n_freq, 1, 2)
        y_cors_vec = W[:, s, None, :] @ V[s, :, :, :] @ tensor_H(W[:, row_idx, :])

        # (n_freq, 2, 1)
        u[:, s, :, None] = np.linalg.solve(y_cors_mat, tensor_T(y_cors_vec))

    # (2, n_freq, 2, 2)
    V_hat = (
        W[None, :, row_idx, :] @ V[row_idx, :, :, :] @ tensor_H(W[None, :, row_idx, :])
    )

    # (2, n_freq, 2)
    # TODO: investigate why `eig_reverse` needs to be True.
    _, u_hat = solve_2x2HEAD(V_hat[0], V_hat[1], eig_reverse=True)
    if debug:
        print_residual_2x2head(u_hat[0], u_hat[1], V_hat[0], V_hat[1])

    # (n_freq, 2, 2)
    u_hat = u_hat.transpose([1, 0, 2])

    u[:, row_idx, :] = -u_hat.conj()
    u[:, row_idx, [0, 1]] += 1

    W[:, :, :] -= u @ W[:, row_idx, :]

    return W


def update_spatial_model(x, R, W, row_idx, method="IP1"):
    """
    Update demixing matrix W.
    Parameters
    ----------
    x: ndarray (n_freq, n_src, n_frame)
        Input mixture signal.
    R: ndarray (n_src, n_freq, n_frame)
        Variance matrices of source spectrograms.
    W: ndarray (n_freq, n_src)
        Demixing matrices.
    row_idx: int or ndarray (2,)
        The index of row vector of W.
    method: string
    Returns
    -------
    Updated W
    """
    allowed_methods = {
        "IP1": _ip_1,
        "IP2": _ip_2,
        "ISS1": _iss_1,
    }

    W_new = allowed_methods[method](x, R, W.copy(), row_idx)

    return W_new
