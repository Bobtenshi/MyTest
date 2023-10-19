from Tools.initalizers import init_activ, init_basis, init_demix, init_model
import time
import numpy as np
from constant import EPSI, USE_NORM  # , ITERATIONS
from Tools.update_models import update_source_model, update_spatial_model
from Tools.ilrma import ilrma
from Tools.calc_cost import cost_function


NP_EPS = np.finfo(np.float64).eps
update_rules_single = ["IP1", "ISS1", "IPA"]
update_rules_double = ["IP2"]
EPS = 1e-15


def floor_model(R, eps):
    R[R < eps] = eps


def exp_ILRMA(X, hydra):
    # check errors and set default values

    I, J, M = X.shape
    N = M
    if N > I:
        raise ValueError(
            "The input spectrogram might be wrong."
            "The size of it must be (freq x frame x ch)."
        )
    # X = whitening(X.copy())

    # Initialization
    W = np.zeros((I, M, M), dtype=X.dtype)
    W[:, :, :M] = np.tile(np.eye(M, dtype=X.dtype), (I, 1, 1))
    Y = X @ W.conj()

    # Source model parameters
    B = init_basis(X.transpose(1, 0, 2), n_basis=2, init_basis="random")
    A = init_activ(X.transpose(1, 0, 2), n_basis=2, init_activ="random")

    # Source model: (n_src, n_freq, n_frame)
    R = init_model(X, init_model="nmf", basis=B, activ=A)

    # n_ch x n_freq x n_frame
    # Y_abs = abs(Y).astype(np.float32).transpose(2, 0, 1)
    # Y_pow = np.power(abs(Y), 2).transpose(2, 0, 1)
    # P = np.maximum(Y_pow, EPS)

    # Initialize
    cost_updR_list, cost_updW_list, cost_init_list = [], [], []
    bss_res_list = []
    process_time_list = []
    sum_process_time_list = []
    source_model_list = []
    sum_time = 0

    # mixture sources for calc SDR Imp.
    bss_res_list.append(X)
    source_model_list.append(R)
    process_time_list.append(0)
    sum_process_time_list.append(0)

    # calc objective function
    cost_init_list.append(cost_function(X, R, W))
    cost_updR_list.append(cost_function(X, R, W))
    cost_updW_list.append(cost_function(X, R, W))

    # Define update rule of W
    method = hydra.params.rule_updateW

    # Algorithm for FastMVAE2
    # Iterative update
    for it in range(hydra.const.iteration):
        # W = update_w(X, R, W)
        # TODO: インデックスを全部渡すようにupdate_rule変える？

        # update W, B, and A
        time_start = time.perf_counter()

        W_n, B_n, A_n, R_n = ilrma(
            X.transpose([1, 0, 2]).copy(),
            B0=B.copy(),
            A0=A.copy(),
            W0=W.copy(),
            method=method,
            # normalize_param=cfg.bss.normalize_param,
        )

        # flooring source model
        floor_model(R_n, EPS)

        cost_init_list.append(cost_function(X, R, W))
        cost_updR_list.append(cost_function(X, R_n, W))
        cost_updW_list.append(cost_function(X, R_n, W_n))
        W = W_n.copy()
        B = B_n.copy()
        A = A_n.copy()
        R = R_n.copy()

        time_end = time.perf_counter()

        # Y = W@X or X@W
        if hydra.params.rule_updateW == "IP1_original":
            Y = X @ W.conj()
        else:
            x = X.transpose([0, 2, 1])
            y = W @ x
            Y = y.transpose([0, 2, 1])

        # Add results for each itr
        bss_res_list.append(Y)
        source_model_list.append(R)
        process_time_list.append(time_end - time_start)
        sum_time += time_end - time_start
        sum_process_time_list.append(sum_time)

    return (
        Y,
        W,
        bss_res_list,
        process_time_list,
        sum_process_time_list,
        cost_init_list,
        cost_updR_list,
        cost_updW_list,
        source_model_list,
    )
