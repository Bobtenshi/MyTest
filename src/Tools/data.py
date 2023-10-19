# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
# data.py
#   modified data.py in MVAE
# author: Li Li (lili-0805@ieee.org)


import os
import numpy as np
from scipy.io import wavfile
from scipy import signal
import glob


def dat_load_trunc(files, dpath, seglen, maxbsize):
    i = 0
    X_train = np.array([])
    for fname in files:
        path_fname = os.path.join(dpath, fname)
        X = np.load(path_fname)
        freqnum = X.shape[0]

        if i == 0:
            X_train = X[1:freqnum, :].T
            framenumlist = np.array([X.shape[1]])
        else:
            X_train = np.append(X_train, X[1:freqnum, :].T, axis=0)
            framenumlist = np.append(framenumlist, X.shape[1])
        i += 1

    framenum, n_f = X_train.shape
    if framenum > seglen:
        framenum = int(framenum / seglen) * seglen
        X_train = X_train[0:framenum, :]
        X_train = X_train.reshape(-1, seglen, n_f)
    else:
        X_tmp = np.zeros((seglen, n_f))
        X_tmp[0:framenum, :] = X_train
        X_train = X_tmp.reshape(-1, seglen, n_f)
    n_seg = X_train.shape[0]

    Y = X_train.real[np.newaxis]
    Y = np.append(Y, X_train.imag[np.newaxis], axis=0)
    Y = np.transpose(Y, (1, 0, 3, 2))
    # Y.shape: (n_seg, 2(real & img), n_f, seglen)

    if n_seg > maxbsize:
        Y = Y[0:maxbsize]
        framenumlist = framenumlist[0:maxbsize]

    return Y, framenumlist


def load_mix_z(files, dpath, maxbsize):
    n_seg = 0
    X_train = np.array([])
    z_train = np.array([])
    for fname in files:
        path_fname = os.path.join(dpath, fname)
        data = np.load(path_fname)
        X, z = data["x"], data["z"]

        if n_seg == 0:
            X_train = X
            z_train = z
        else:
            X_train = np.append(X_train, X, axis=0)
            z_train = np.append(z_train, z, axis=0)
        n_seg += 1
    if n_seg > maxbsize:
        X_train = X_train[:maxbsize]
        z_train = z_train[:maxbsize]

    return X_train, z_train


def prenorm(statpath, X):
    # X must be a 4D array with size (N, n_ch, freqnum, framenum)
    # statpath is a path for a txt file containing mean and
    # standard deviation of X
    # The txt file is assumed to contain a 1D array with size 2 where
    # the first and second elements are the mean and standard deviation of X.
    if statpath is None or not os.path.exists(statpath):
        X_abs = np.linalg.norm(X, axis=1, keepdims=True)
        gv = np.mean(np.power(X_abs, 2), axis=(0, 1, 2, 3), keepdims=True)
        gs = np.sqrt(gv)
        X = X / gs

        np.save(statpath, [gv, gs])

    else:
        gs = np.load(statpath)[1]
        X = X / gs

    return X, gs


def load_wav(wpath, keyword, fs_resample):
    # wdir = sorted(os.listdir(wpath))
    wdir = sorted(glob.glob(f"{wpath}/{keyword}.wav", recursive=True))

    # define max length of data
    maxlen = 0
    chnum = 0
    for fname in wdir:
        # path_fname = os.path.join(wpath, fname)
        path_fname = os.path.join(fname)
        # print(path_fname)
        fs, data = wavfile.read(path_fname)
        ddim = data.ndim
        if len(data) > maxlen:
            maxlen = len(data)
        if ddim == 1:
            data = data.reshape(len(data), -1)
        chnum += data.shape[1]

    # load data
    sig = np.asarray([]).reshape(maxlen, 0)
    for fname in wdir:
        # path_fname = os.path.join(wpath, fname)
        path_fname = os.path.join(fname)
        fs, data = wavfile.read(path_fname)
        ddim = data.ndim
        if ddim == 1:
            data = data.reshape(len(data), -1)
        data_tmp = np.zeros((maxlen, data.shape[1]))
        data_tmp[0 : len(data), :] = data

        sig = np.append(sig, data_tmp, axis=1)

    # resample data
    J = int(np.ceil(sig.shape[0] * fs_resample / fs))
    y = signal.resample(sig, J)
    nsamples = len(y)
    return y
