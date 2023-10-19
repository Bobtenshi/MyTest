import imp
import os
import torch
import numpy as np
from utils import Logger, make_df, df_bss_res
from constant import SAMP_RATE, ITR_FREQ
from natsort import natsorted
from Tools.metrics import si_bss_eval
from Tools.stft_tools import spectrogram, stft

import librosa
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import glob

gpu = 0 if torch.cuda.is_available() else -1


def evaluation(args):
    try:
        hydra = args[2]
        sep_path = os.path.join(args[1], args[0])
        ref_path = os.path.join(args[1], "src_imgs")
        mix_path = os.path.join(args[1], "mixtures")

        # log
        # logprint = Logger(os.path.join(sep_path, "bss_result.txt"), add=False)
        # print(sep_path)

        sep_file = []
        for s in os.listdir(sep_path):
            if s.endswith(".wav"):
                sep_file.append(s)

        ref_file = sorted(glob.glob(f"{ref_path}/*.wav", recursive=True))
        mix_file = sorted(glob.glob(f"{mix_path}/*.wav", recursive=True))
        max_len = 0
        n_ch = len(sep_file)

        for w in sep_file:
            y = librosa.core.load(f"{sep_path}/{w}", sr=SAMP_RATE)[0]
            # _, y = wavfile.read(f"{sep_path}/{w}")
            if max_len < len(y):
                max_len = len(y)
        for w in ref_file:
            y = librosa.core.load(f"{w}", sr=SAMP_RATE)[0]
            # _, y = wavfile.read(f"{sep_path}/{w}")
            if max_len < len(y):
                max_len = len(y)

        sep_sig = np.zeros((n_ch, max_len))
        ref_sig = np.zeros((n_ch, max_len))

        for idx, f in enumerate(ref_file):
            y = librosa.core.load(f"{f}", sr=SAMP_RATE)[0]
            # _, y = wavfile.read(f"{f}")
            ref_sig[idx, : len(y)] = y

        if args[0] == "itr_0":
            for idx, f in enumerate(mix_file):
                y = librosa.core.load(f"{f}", sr=SAMP_RATE)[0]  # 分離前=観測信号をLoad
                # _, y = wavfile.read(f"{f}")
                sep_sig[idx, : len(y)] = y
        else:
            for idx in range(n_ch):
                y = librosa.core.load(f"{sep_path}/{sep_file[idx]}", sr=SAMP_RATE)[
                    0
                ]  # 分離信号をLoad
                # _, y = wavfile.read(f"{sep_path}/{sep_file[idx]}")
                sep_sig[idx, : len(y)] = y

        # Load processing-time for each itr & update rule
        params_path = os.path.join(sep_path, "params.npz")
        params = np.load(params_path)
        time, update_rule, costI, costR, costW = (
            params["Time"],
            params["UpdateRule"],
            params["CostI"],
            params["CostR"],
            params["CostW"],
        )

        if hydra.params.rule_eval == "SI_SDR":
            sdr, _, _, _ = si_bss_eval(ref_sig.T, sep_sig.T, scaling=True)
            # pesq, stoi = pesq_stoi(ref_sig, sep_sig, SAMP_RATE, False)
            # pesq, stoi = np.zeros(n_ch, dtype=float), np.zeros(n_ch, dtype=float)
        np.savez(
            f"{sep_path}/bss_result.npz",
            SDR=sdr,
            Time=time,
            UpdateRule=update_rule,
            CostI=costI,
            CostR=costR,
            CostW=costW,
        )

        # if args[0] == "itr_2" or args[0] == "itr_30":
        #
        #    S = stft(ref_sig[:, :].T, 2048, 1024, window="hamming")
        #    Y = stft(sep_sig[:, :].T, 2048, 1024, window="hamming")
        #
        #    spectrogram(params["source_model"][0, :, :], f"r_{args[0]}_0")
        #    spectrogram(S[0][:, :,0], f"s_{args[0]}_0")
        #    spectrogram(Y[0][:, :,0], f"p_{args[0]}_0")
        #
        #    spectrogram(params["source_model"][1, :, :], f"r_{args[0]}_1")
        #    spectrogram(S[0][:, :,1], f"s_{args[0]}_1")
        #    spectrogram(Y[0][:, :,1], f"p_{args[0]}_1")

        print(f"\r{sep_path} is done")
    except:
        print("err")


def save_res_as_pd(data_root, folders, itr_dirs, hydra):
    savepath = os.path.join(data_root, hydra.params.modeling_method)
    # folder毎に独立に処理
    for f in folders:
        try:
            # Make folder
            itr_savepath = os.path.join(savepath, f)
            os.makedirs(itr_savepath, exist_ok=True)
            # setup pandas df
            df = make_df()
            df = df.astype(object)

            # itr 毎にSDR類を計算 logging
            for itr, itr_dir in enumerate(itr_dirs):
                # res(itr, 処理時間，目的関数値...等)をLoad
                result = np.load(os.path.join(savepath, f, itr_dir, "bss_result.npz"))
                Time, Update_rule, costI, costR, costW = (
                    result["Time"],
                    result["UpdateRule"],
                    result["CostI"],
                    result["CostR"],
                    result["CostW"],
                )

                # pandasに値を格納
                df = df.append(
                    df_bss_res(
                        data_root,  # data_name
                        result["SDR"].shape[0],  # n_src
                        hydra.params.modeling_method,  # ILRMA, Chimera,Prop.
                        itr,
                        Time,
                        result["SDR"],  # SDR value
                        costI,
                        costR,
                        costW,
                    )
                )
            df.to_csv(os.path.join(itr_savepath, "res.csv"), index=False)
        except:
            print(f"Error: Can not save 'bss results' from {itr_savepath}")


##########################################
# 全体の流れを記述している
#   1.SDR計算
#   2.pandasで保存
##########################################
def eval_flow(sep_root, folders, hydra=None):
    print(sep_root)
    print(folders)

    # Calc SI-SDR
    for idx, f in enumerate(folders):
        try:
            sep_dir = os.path.join(sep_root, hydra.params.modeling_method, f)
            ref_dir = os.path.join(sep_dir, "src_imgs")  # SDR計算用ソースイメージの場所

            # ITR毎の分離信号の場所
            itr_dirs = os.listdir(sep_dir)
            itr_dirs = [f for f in natsorted(itr_dirs) if "itr" in f]

            # 並列処理
            if hydra.params.multiprocessing:
                estimate_sig_dir = [
                    os.path.join(sep_dir, itr_dir) for itr_dir in itr_dirs
                ]
                with ProcessPoolExecutor(max_workers=10) as executor:
                    executor.map(
                        evaluation,
                        zip(itr_dirs, repeat(sep_dir), repeat(hydra)),
                    )

            # for-roop処理 (debug用)
            else:
                for itr_dir in itr_dirs:
                    evaluation([itr_dir, sep_dir, hydra])  # 2s/itr
        except:
            print(f"{sep_dir} is not found")

    # Save SI-SDR as pandas(.csv)
    save_res_as_pd(sep_root, folders, itr_dirs, hydra)


##########################################
# main関数
##########################################
def main(hydra):
    os.environ["CUDA_VISIBLE_DEVICES"] = hydra.const.gpu_card

    for folder in hydra.arg.test_dataset_folders:
        print(f"   {folder} is working.")

        hydra.arg.input_dir = os.path.join(hydra.arg.test_data, folder)
        hydra.arg.ref_dir = os.path.join(hydra.arg.test_data, folder)
        hydra.arg.output_dir = os.path.join(hydra.arg.output_path, folder)

        # テスト用ファイルの個数指定
        folders = [
            f"{i}" for i in range(hydra.params.select[0], hydra.params.select[1] + 1)
        ]
        # hydra.arg.sources = int(re.search(r"\d+src_", hydra.arg.ref_dir).group().replace("src_", ""))

        eval_flow(
            hydra.arg.output_dir,
            folders,
            hydra,
        )
