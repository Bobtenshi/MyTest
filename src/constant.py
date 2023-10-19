# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
# simurate_pyroom.py
# auther: yamaji syuhei (hei8maji@gmail.com)
#
import sys

FIG_SIZZE = {"mm": 1 / 25.4, "pt": 1 / 72}
SEED = 10
EPSI = sys.float_info.epsilon

# signal processing
SAMP_RATE = 16000
WIN_TYPE = "hamming"
WIN_LEN = 2048
WIN_SHIFT = 1024
RATIO = 4
N_FREQ = WIN_LEN // 2 + 1
N_BASIS = 2

# separation algorithm
REF_MIC = 0
USE_NORM = True
USE_WHITENING = False

# network training
SEG_LEN = 128
MAX_BATCH_SIZE = 16

# save freq of BSS itr
ITR_FREQ = 1

DEFAULT_SPK = {
    "folders": [
        "SF1+SF2_r0.20_rt78ms_2src_2mic",
        "SF1+SF2_r0.80_rt351ms_2src_2mic",
        "SF1+SF2_ANE_2src_2mic",
        "SF1+SF2_E2A_2src_2mic",
    ],
    "label_dim": 4,
}

MORE_SPK = {
    "folders": [
        "2src_2mic",
        "3src_3mic",
        #"6src_6mic",
        # "9src_9mic",
        # "12src_12mic",
    ],
    "label_dim": 101,
}
