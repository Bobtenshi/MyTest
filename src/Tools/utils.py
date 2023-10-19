# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
# simurate_pyroom.py
# auther: yamaji syuhei (hei8maji@gmail.com)

import os
from tkinter.messagebox import NO
import hydra
import requests
import numpy as np

def get_root():
    # print(f'Current working directory: {os.getcwd()}')
    CURRENT_ROOT = os.getcwd()
    TMP_ROOT = hydra.utils.get_original_cwd()
    os.chdir(TMP_ROOT)
    # os.chdir('..')
    PROJECT_ROOT = os.getcwd()

    return CURRENT_ROOT, PROJECT_ROOT


def SendLine(msg=None):
    url = "https://notify-api.line.me/api/notify"
    token = "NXOF52tUd8CQ0qAWVXhh7k1RXeaeOQ8T8YwSLd8c9wB"
    headers = {"Authorization": "Bearer " + token}
    if msg != None:
        payload = {"message": msg}
    r = requests.post(url, headers=headers, params=payload)


def back_projection(Y, X):
    I, J, M = Y.shape

    if X.shape[2] == 1:
        A = np.zeros((1, M, I), dtype=np.complex)
        Z = np.zeros((I, J, M), dtype=np.complex)
        for i in range(I):
            Yi = np.squeeze(Y[i, :, :]).T  # channels x frames (M x J)
            Yic = np.conjugate(Yi.T)
            A[0, :, i] = X[i, :, 0] @ Yic @ np.linalg.inv(Yi @ Yic)

        A[np.isnan(A)] = 0
        A[np.isinf(A)] = 0
        for m in range(M):
            for i in range(I):
                Z[i, :, m] = A[0, m, i] * Y[i, :, m]

    elif X.shape[2] == M:
        A = np.zeros(M, M, I)
        Z = np.zeros(I, J, M, M)
        for i in range(I):
            for m in range(M):
                Yi = np.squeeze(Y[i, :, :]).T
                Yic = np.conjugate(Yi.T)
                A[0, :, i] = X[i, :, m] @ Yic @ np.linalg.inv(Yi @ Yic)
        A[np.isnan(A)] = 0
        A[np.isinf(A)] = 0
        for n in range(M):
            for m in range(M):
                for i in range(I):
                    Z[i, :, n, m] = A[m, n, i] * Y[i, :, n]

    else:
        print("The number of channels in X must be 1 or equal to that in Y.")

    return Z


class Pycolor:
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    PURPLE = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    RETURN = "\033[07m"  # 反転
    ACCENT = "\033[01m"  # 強調
    FLASH = "\033[05m"  # 点滅
    RED_FLASH = "\033[05;41m"  # 赤背景+点滅
    END = "\033[0m"
