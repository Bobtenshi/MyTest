# !/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import Tools.data as data
import numpy as np
from scipy.io import wavfile
from scipy import signal
from utils import back_projection
from constant import SAMP_RATE, WIN_TYPE, WIN_LEN, WIN_SHIFT, RATIO
from constant import SEED, USE_WHITENING, REF_MIC
from Tools.FastMVAE2 import FastMVAE2

# from Tools.FastMVAE2_ import FastMVAE2
from Tools.ILRMA_tools.main_ilrma import exp_ILRMA
from natsort import natsorted

BSS_methods = {
    "ILRMA": exp_ILRMA,
    "ChimeraACVAE": FastMVAE2,
    "As_AE-Loss_ISdiv": FastMVAE2,
    "As_AE-Loss_ISdiv_6ch": FastMVAE2,
    "As_AE-Loss_ISdiv_236ch": FastMVAE2,
    "As_AE-Loss_PIT": FastMVAE2,
}


# def _main(folder, hydra):
def _main(args):
    folder = args[0]
    hydra = args[1]

    save_path = os.path.join(
        hydra.arg.output_dir,
        hydra.params.modeling_method,
        # hydra.params.rule_updateW,
        # f"{hydra.params.n_updateW}_times_update",
        folder,
    )

    # Input data and resample
    mix = data.load_wav(os.path.join(hydra.arg.input_dir, folder), "mix?", SAMP_RATE)
    src_img = data.load_wav(
        os.path.join(hydra.arg.input_dir, folder), "src_img?", SAMP_RATE
    )
    ns = mix.shape[1]  # (160000, n_mic)

    # STFT
    frames_ = np.floor((mix.shape[0] + 2 * WIN_SHIFT) / WIN_SHIFT)
    frames = int(np.ceil(frames_ / RATIO) * RATIO)

    X = np.zeros((int(WIN_LEN / 2 + 1), int(frames), mix.shape[1]), dtype="complex")
    for n in range(mix.shape[1]):
        f, t, X[:, : int(frames_), n] = signal.stft(
            mix[:, n], nperseg=WIN_LEN, window=WIN_TYPE, noverlap=WIN_LEN - WIN_SHIFT
        )

    ALLOW_SAVE = True
    (
        _,
        W,
        bss_res_list,
        process_time_list,
        sum_process_time_list,
        cost_init_list,
        cost_updR_list,
        cost_updW_list,
        source_model_list,
    ) = BSS_methods[hydra.params.modeling_method](
        X,
        hydra,
    )

    for (itr, Y, pro_time, sum_pro_time, costI, costR, costW, source_model) in zip(
        range(len(bss_res_list)),
        bss_res_list,
        process_time_list,
        sum_process_time_list,
        cost_init_list,
        cost_updR_list,
        cost_updW_list,
        source_model_list,
    ):

        # projection back
        XbP = np.zeros((X.shape[0], X.shape[1], 1), dtype="complex")
        XbP[:, :, 0] = X[:, :, REF_MIC]
        Z = back_projection(Y, XbP)

        # iSTFT
        sep = np.zeros([mix.shape[0], ns])
        for n in range(ns):

            sep_ = signal.istft(Z[:, :, n], window=WIN_TYPE)[1]
            sep[:, n] = sep_[: mix.shape[0]]

        # save wav files
        itr_save_path = os.path.join(save_path, f"itr_{itr}")
        os.makedirs(itr_save_path, exist_ok=True)
        os.makedirs(
            os.path.join(save_path, "src_imgs"), exist_ok=True
        )  # src_img用FOLDER
        os.makedirs(
            os.path.join(save_path, "mixtures"), exist_ok=True
        )  # src_img用FOLDER

        for n in range(ns):
            wavfile.write(
                os.path.join(itr_save_path, f"estimated_signal{n}.wav"),
                SAMP_RATE,
                sep[:, n],  # .astype(np.int16),
            )
            wavfile.write(
                os.path.join(save_path, "src_imgs", f"src_img{n}.wav"),
                SAMP_RATE,
                src_img[:, n],  # .astype(np.int16),
            )
            wavfile.write(
                os.path.join(save_path, "mixtures", f"mixture{n}.wav"),
                SAMP_RATE,
                mix[:, n],  # .astype(np.int16),
            )

        # plot用にsave
        Y_abs = abs(Y).astype(np.float32).transpose(2, 0, 1)
        gv = np.mean(np.power(Y_abs[:, 1:, :], 2), axis=(1, 2), keepdims=True)

        np.savez(
            os.path.join(itr_save_path, "params"),
            Time=pro_time,
            SumTime=sum_pro_time,
            W=W,
            UpdateRule=hydra.params.rule_updateW,
            CostI=costI,
            CostR=costR,
            CostW=costW,
            Y_abs=Y_abs,
            source_model=source_model,
            gv=gv,
        )
    print(f"\n   Separated signals are saved in {save_path}")
    # except:
    #    print("   Error in source sep.")


# -----------------------------------------------------
# main
# -----------------------------------------------------
def main(hydra):
    np.random.seed(seed=SEED)
    for folder in hydra.arg.test_dataset_folders:
        print(f"   {folder} is working.")
        all_input_dir_names = os.listdir(os.path.join(hydra.arg.test_data, folder))
        natsorted(all_input_dir_names)

        hydra.arg.input_dir = os.path.join(hydra.arg.test_data, folder)
        hydra.arg.output_dir = os.path.join(hydra.arg.output_path, folder)

        if hydra.params.select[1] < len(all_input_dir_names):
            input_dir_names = all_input_dir_names[
                hydra.params.select[0] : hydra.params.select[1] + 1
            ]
        else:
            input_dir_names = all_input_dir_names

        # multi-core or single-core processing
        if hydra.params.multiprocessing:
            from concurrent.futures import ProcessPoolExecutor
            from itertools import repeat

            try:
                with ProcessPoolExecutor(max_workers=8) as executor:
                    executor.map(_main, zip(input_dir_names, repeat(hydra)))
                print("   Processed with multi-processing.")

            except:
                print("   Error in source sep.")

        else:
            for dir in input_dir_names:
                _main([dir, hydra])
                print("   Processed with for-roop.")
