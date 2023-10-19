import numpy as np
import pandas as pd
import os
import datetime


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


class Logger:
    def __init__(self, logf, add=True):
        if add and os.path.isfile(logf):
            self.out = open(logf, "a")
            self.out.write(f"\n{datetime.datetime.now()}\n")
        else:
            if os.path.isfile(logf):
                os.remove(logf)
            self.out = open(logf, "a")
            self.out.write(f"{datetime.datetime.now()}\n")

    def __del__(self):
        if self.out is not None:
            self.close()

    def __call__(self, msg):
        print(msg)
        self.out.write(f"{msg}\n")
        self.out.flush()

    def close(self):
        self.out.close()
        self.out = None


def make_df():
    df = pd.DataFrame(
        {
            "Data_name": [],
            "NumberOfSources": [],
            "ModelingMethod": [],
            "Itration": [],
            "Time": [],
            "SI-SDR": [],
            "CostI": [],
            "CostR": [],
            "CostW": [],
        }
    )
    return df


def df_bss_res(file_name, n_src, bss_name, itr, time, SDR, costI, costR, costW):
    df = pd.DataFrame(
        {
            "Data_name": file_name,
            "NumberOfSources": int(n_src),
            "ModelingMethod": bss_name,
            "Itration": int(itr),
            "Time": time,
            "SI-SDR": SDR,
            "CostI": costI,
            "CostR": costR,
            "CostW": costW,
        }
    )

    return df
