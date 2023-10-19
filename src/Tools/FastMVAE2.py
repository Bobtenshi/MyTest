# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FastMVAE2.py
# author: Li Li (lili-0805@ieee.org)
#
# source tools/venv/bin/activate

# from turtle import color
import torch
import net
import numpy as np
import numpy.linalg as LA
from constant import EPSI, USE_NORM
import time
import os
import matplotlib.pyplot as plt

from Tools.ILRMA_tools.update_models import update_source_model, update_spatial_model
from Tools.calc_cost import cost_function


NP_EPS = np.finfo(np.float64).eps
update_rules_single = ["IP1", "ISS1", "IPA"]
update_rules_double = ["IP2"]
color_palette = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]


def FastMVAE2(X, hydra):
    os.environ["CUDA_VISIBLE_DEVICES"] = hydra.const.gpu_card

    # check errors and set default values
    I, J, M = X.shape
    N = M
    if N > I:
        raise ValueError(
            "The input spectrogram might be wrong."
            "The size of it must be (freq x frame x ch)."
        )

    # Initialization
    W = np.zeros((I, M, N), dtype=np.complex)
    for i in range(I):
        W[i, :, :] = np.eye(N)
    Y = X @ W.conj()

    P = np.maximum(np.abs(Y) ** 2, EPSI)  # P = |Y|^2
    R = P.copy()  # Source model: (n_freq, n_frame, n_src)

    if USE_NORM:
        W, R, P = local_normalize(W, R, P, I, J)

    P = P.transpose(2, 0, 1)  # I,J,M -> M,I,J
    R = R.transpose(2, 0, 1)  # I,J,M -> M,I,J
    Q = np.zeros((N, I, J))

    # load trained networks
    nn_freq = I - 1
    encoder = net.ChimeraACVAE_Encoder(nn_freq, hydra.arg.label_dim)
    decoder = net.ChimeraACVAE_Decoder(nn_freq, hydra.arg.label_dim)
    checkpoint = torch.load(hydra.arg.model_path, map_location="cpu")
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # GPUが使えるなら使用
    if device == "cuda":
        encoder.cuda(device)
        decoder.cuda(device)
    encoder.eval()

    # Initialize
    bss_res_list = []
    process_time_list = []
    sum_process_time_list = []
    cost_updR_list, cost_updW_list, cost_init_list = [], [], []
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
        print(
            f"\r   Modeling method: {hydra.params.modeling_method}, Update rule: {hydra.params.rule_updateW}, Iteration: {it}",
            end="",
        )

        Y_abs = abs(Y).astype(np.float32).transpose(2, 0, 1)
        gv = np.mean(np.power(Y_abs[:, 1:I, :], 2), axis=(1, 2), keepdims=True)
        Y_abs_array_norm = Y_abs / np.sqrt(gv)
        y_abs = torch.from_numpy(
            np.asarray(Y_abs_array_norm[:, None, 1:, :], dtype="float32")
        ).to(device)

        # update label and latent variable
        z_mu, _, lhat = encoder(y_abs[:, :, :, :])
        # update logvar
        logvar = decoder(z_mu, lhat)  # (n_src, 1, freq, time)

        Q[:, 1:I, :] = np.squeeze(np.exp(logvar.detach().to("cpu").numpy()), axis=1)
        Q = np.maximum(Q, EPSI)  # (n_src, freq, time)
        gv = np.mean(
            np.divide(P[:, 1:I, :], Q[:, 1:I, :]), axis=(1, 2), keepdims=True
        )  # (n_src, 1 ,1)
        Rhat = np.multiply(Q, gv)  # (n_src, freq, time)
        Rhat[:, 0, :] = R[:, 0, :].copy()
        # R = Rhat.transpose(1, 2, 0)

        time_start = time.perf_counter()
        W_n = W.copy()
        if method in update_rules_double:  # 2行ずつ更新
            for _ in range(hydra.params.n_updateW):
                for s in range(0, M * 2, 2):
                    m, n = s % M, (s + 1) % M
                    tgt_idx = np.array([m, n])

                    W_n[:, :, :] = update_spatial_model(
                        X.transpose([0, 2, 1]),
                        Rhat,
                        W_n,
                        row_idx=tgt_idx,
                        method=method,
                    )
        elif method in update_rules_single:
            rowIdxList = np.concatenate((np.arange(M), np.arange(M)), axis=0)
            for _ in range(hydra.params.n_updateW):
                for s in rowIdxList:
                    W_n[:, :, :] = update_spatial_model(
                        X.transpose([0, 2, 1]), Rhat, W_n, row_idx=s, method=method
                    )

        time_end = time.perf_counter()

        # Y = W@X or X@W
        if hydra.params.rule_updateW == "IP1_original":
            Y = X @ W_n.conj()
        else:
            # (freq, time, mic) => (time, freq, mic)
            x = X.transpose([1, 0, 2])
            # (freq, mic, mic) @ (freq, mic, time) -> (freq, mic, time)
            y = W_n @ x.transpose([1, 2, 0])
            # (freq, mic, time) => (freq, time, mic)
            Y = y.transpose([0, 2, 1])

        P = np.maximum(np.abs(Y) ** 2, EPSI)  # P = |Y|^2

        if USE_NORM:
            W_n, Rhat, P_n = local_normalize(W_n, Rhat.transpose([1, 2, 0]), P, I, J)

        P_n = P_n.transpose(2, 0, 1)  # I,J,M -> M,I,J
        Rhat = Rhat.transpose(2, 0, 1)  # I,J,M -> M,I,J

        cost_init_list.append(cost_function(X, R, W))
        cost_updR_list.append(cost_function(X, Rhat, W))
        cost_updW_list.append(cost_function(X, Rhat, W_n))

        R = Rhat.copy()
        W = W_n.copy()
        P = P_n.copy()

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


# ======== Local functions ========
def local_normalize(W, R, P, I, J, *args):
    lamb = np.sqrt(np.sum(np.sum(P, axis=0), axis=0) / (I * J))

    W = W / np.squeeze(lamb)
    lambPow = lamb**2
    P = P / lambPow
    R = R / lambPow
    if len(args) == 1:
        T = args[0]
        T = T / lambPow
        return W, R, P, T
    elif len(args) == 0:
        return W, R, P
