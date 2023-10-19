import os
import argparse
import re
import torch
import numpy as np
from utils import Logger, Pycolor, num2str, make_df, df_bss_res
from constant import SAMP_RATE, ITR_FREQ
from natsort import natsorted
from Tools.metrics import si_bss_eval
from Tools.separation import bss_eval_sources
import net_dev
import librosa
from torchinfo import summary
import glob
import pandas as pd
from utils import num2str, make_df
from show_glaph import calc_imp


def calc_sdr_by_tmp_dnn(hydra, model_path):

    #  setting for tmp calc sdr
    hydra.params.select = [1, 20]
    hydra.params.multiprocessing = False
    # hydra.params.multiprocessing = True

    hydra.arg.model_path = model_path
    hydra.arg.train_data = f"{hydra.const.data_root}train/{hydra.const.dataset}/"

    # output directories
    hydra.const.test_mode = "in_trainning"
    hydra.arg.save_output_root = (
        f"./output/{hydra.const.test_mode}/{hydra.const.dataset}/"
    )
    hydra.arg.save_result_root = (
        f"./result/{hydra.const.test_mode}/{hydra.const.dataset}/"
    )

    # import modules
    import constant as constant
    from Tools import utils
    import source_separation
    import sys

    # define experiment path
    # hydra.arg.test_dataset_name = utils.get_dataset_folder_name(
    #    hydra.const.dataset, hydra.const.test_dataset
    # )
    hydra.arg.test_dataset_name = "jvs"
    hydra.arg.test_data = f"{hydra.const.data_root}test/{hydra.arg.test_dataset_name}/"
    hydra.arg.output_path = f"{hydra.arg.save_output_root}"

    if hydra.params.modeling_method == "None":
        hydra.params.modeling_method = (
            f"As_{hydra.params.retrain_model_type}-Loss_{hydra.params.retrain_loss}"
        )

    default_spk_group = [
        "default_spk_dep",
        "default_spk_ind",
    ]
    more_spk_group = ["more_spk_ind", "more_spk_ind_repeat", "jvs"]

    if hydra.arg.test_dataset_name in default_spk_group:
        from constant import DEFAULT_SPK as SPK_DATA
    elif hydra.arg.test_dataset_name in more_spk_group:
        from constant import MORE_SPK as SPK_DATA
    else:
        sys.exit(f"test_dataset:{hydra.arg.test_dataset_name} is None.")
    hydra.arg.test_dataset_folders = SPK_DATA["folders"]
    hydra.arg.label_dim = SPK_DATA["label_dim"]
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # ---------------------------------------------
    # stage 1: test (pre-)trained model
    # ---------------------------------------------
    # run FastMVAE2
    # test_fmvae2.main(hydra, exp_params)
    print("#------------------------------------------------------------------")
    print("# stage 1: Start to separate the mixtures with trained model...    ")
    print("#------------------------------------------------------------------")
    source_separation.main(hydra)
    print(Pycolor.GREEN + "Separation is done." + Pycolor.END)

    # ---------------------------------------------
    # stage 2:  eval mode
    # ---------------------------------------------
    import evaluation

    print("#------------------------------------------")
    print("# stage 2: Start to evaluate the outputs...")
    print("#------------------------------------------")
    evaluation.main(hydra)
    print(Pycolor.GREEN + "Evaluation is done." + Pycolor.END)

    sdr_list = []
    # データの読み込み
    res_root = hydra.arg.output_path.replace("output", "result")
    data_paths = glob.glob(f"{res_root}**/*res.csv", recursive=True)
    for f in data_paths:
        tmp_df = pd.read_csv(f)
        df_imp_sdr = calc_imp(tmp_df)[0]
        sdr_list.append(df_imp_sdr.loc[30, "imp_sdr"])

    return sdr_list
