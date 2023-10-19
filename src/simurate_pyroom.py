# !/usr/bin/env python
# -*- coding: utf-8 -*-
# simurate_pyroom.py
# auther: yamaji syuhei (hei8maji@gmail.com)

import sys
import os
import numpy as np
import random
import glob
#import librosa
import pyroomacoustics as pra
import hydra
from scipy.io import wavfile
from Tools.utils import back_projection
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from natsort import natsorted
from constant import SAMP_RATE, WIN_TYPE, WIN_LEN, WIN_SHIFT, RATIO, SEG_LEN
from Tools.stft_tools import stft, istft, spectrogram, power
from Tools.utils import get_root, SendLine
from Tools.FastMVAE2 import FastMVAE2


def make_room(nch, rt60, room_dim):
    e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
    room = pra.ShoeBox(
        room_dim,
        fs=16000,
        materials=pra.Material(e_absorption),
        max_order=max_order,
    )
    return room


def mix_signals(samples):
    dry_signal_list = []
    src_img_list = []
    ch = len(samples)
    rt60 = 0.3
    room_size = [8.0, 6.0]
    center_position = [3, 3]

    # マイク位置，音源位置を決定
    mic_pos = pra.circular_2D_array(center=center_position, M=ch, phi0=0, radius=0.1)

    # マイク配置確認用room
    room_pos = make_room(ch, rt60, room_size)  # 部屋の初期化，毎回同じ部屋を作成
    room_pos.add_microphone_array(pra.MicrophoneArray(mic_pos, room_pos.fs))

    # src_imgをシミュレート
    for idx, src_path in enumerate(samples):
        fs, signal = wavfile.read(src_path)

        room = make_room(ch, rt60, room_size)  # 部屋の初期化，毎回同じ部屋を作成
        room.add_microphone_array(pra.MicrophoneArray(mic_pos, room.fs))  # マイクの配置

        sig_position = random.randint(
            0, signal.shape[0] - fs - 10
        )  # 音源の配置 配置場所=signal_posのidx番目, どこの10sかはランダム

        signal_pos = pra.circular_2D_array(
            center=center_position, M=ch, phi0=np.random.randint(1, 360), radius=2.5
        )
        room.add_source(
            signal_pos[:, idx],
            signal=signal[sig_position : sig_position + room.fs * 10],
        )
        room_pos.add_source(
            signal_pos[:, idx],
            signal=signal[sig_position : sig_position + room.fs * 10],
        )
        room.simulate()
        src_img_list.append(room.mic_array.signals[:, : room.fs * 10].T)

    return np.array(src_img_list), fs, room_pos  # room.mic_array.signals.T

def run_chimerabss(X, hydra):
    hydra.arg.label_dim = 101
    hydra.arg.save_model_root = f"{hydra.const.model_root}trained/jvs/ChimeraACVAE/"
    hydra.arg.model_path = f"{hydra.arg.save_model_root}/Chimera.model"
    hydra.const.iteration = 30

    _, _, bss_res_list, _, _, _, _, _, source_model_list = FastMVAE2(X, hydra)

    # projection back of estimation for each itaration
    bss_res_bp_list = []
    XbP = np.zeros((X.shape[0], X.shape[1], 1), dtype="complex")
    XbP[:, :, 0] = X[:, :, 1]
    for idx, Y in enumerate(bss_res_list):
        if idx == 0:
            Z = Y
        else:
            Z = back_projection(Y, XbP)
        bss_res_bp_list.append(Z)

    return bss_res_list, source_model_list


def mix_bss_save(params):
    i = params[0]
    src_data = params[1]
    hydra = params[2]
    nch = params[3]
    root = params[4]
    save_root = f"{root}/{i}"

    if not os.path.isdir(save_root):
        os.makedirs(save_root)
    if not os.path.isdir(f"{save_root}/res/"):
        os.makedirs(f"{save_root}/res/")

    # mixture-specをload or 作成
    # drysourceをn個選択（重複なし）
    samples = random.sample(src_data, nch)
    # Mix by pyroom-acoustics
    src_imgs, fs, room = mix_signals(samples)  # (src_num, N samples, mic_num)
    mixture = np.sum(src_imgs, axis=0)  # 同じマイクごとに加算


    #src_imgsを整形
    src_img = np.zeros_like(mixture)
    for j in range(nch):
        src_img[:, j] = src_imgs[j, :, 0].copy()

    for j in range(nch):  # save src_img & mixture_src
        wavfile.write(
            f"{save_root}/src_img{j}.wav",
            fs,
            src_imgs[j, :, 0].astype(np.float32),  # 0番目のマイクでのｊ番目の音源，
        )
        wavfile.write(
            f"{save_root}/mix{j}.wav",
            fs,
            mixture[:, j].astype(np.float32),
        )
    # STFT
    X, _ = stft(
        mixture, hydra.const.win_len, hydra.const.win_shift, window=hydra.const.win_type
    )  # X : (freq, time-frame, ch)
    S, _ = stft(
        src_img, hydra.const.win_len, hydra.const.win_shift, window=hydra.const.win_type
    )  # S : (freq, time-frame, ch)


    # spec_save_path = f'{mix_folder}/{i}.npy'
    np.save(f"{save_root}/mix_spec.npy", X)
    np.save(f"{save_root}/srcimg_spec.npy", S)

    if hydra.params.train_or_test == "train" or hydra.params.train_or_test == "test" :
        # BSS by fMVAE
        Y_list, source_model_list = run_chimerabss(X[:, :SEG_LEN, :], hydra)

        # save 分離途中信号 for CentaurVAE data-set
        for idx, Y in enumerate(Y_list):
            Y_abs = abs(Y).astype(np.float32).transpose(2, 0, 1)
            source_model = source_model_list[idx]
            # np.random.shuffle(Y_abs)
            gv = np.mean(np.power(Y_abs[:, 1:, :], 2), axis=(1, 2), keepdims=True)

            np.savez(
                f"{save_root}/res/file{i}_itr{idx}_nch{nch}_.npz",
                Y_abs=Y_abs,
                S_abs = abs(S.transpose([2, 0, 1])),
                source_model=source_model,
                gv=gv,
            )

    fig, ax = room.plot()
    fig.savefig(f"{save_root}/mic_music_position.png")
    print(f"filr_{i} has finished...")


@hydra.main(config_name="hydras")
def main(hydra):
    # get project-root and cd project-root
    _, _ = get_root()

    # set valuses
    os.environ["CUDA_VISIBLE_DEVICES"] = hydra.const.gpu_card

    data_num = 20

    # find data-folders
    if hydra.params.train_or_test == "train":
        data_name = "wsj0"
        src_path = natsorted(glob.glob(f"./data/trimmed_wav/{data_name}/*_[1-5]*.wav"))
    elif hydra.params.train_or_test == "test":
        data_name = "jvs"
        src_path = natsorted(glob.glob(f"./data/trimmed_wav/{data_name}/*_*.wav"))

    # define save root
    root = f"./data/{hydra.params.train_or_test}/{data_name}/{hydra.params.nch}src_{hydra.params.nch}mic"

    # mix_bss_save([0, src_path, hydra, hydra.params.nch, root])
    with ProcessPoolExecutor(max_workers=1) as executor:
        executor.map(
            mix_bss_save,
            zip(
                range(data_num),
                repeat(src_path),
                repeat(hydra),
                repeat(hydra.params.nch),
                repeat(root),
            ),
        )
    # notify
    if hydra.params.line_notify:
        SendLine("Making dataset is done.")


if __name__ == "__main__":
    main()

# python3 src/simurate_pyroom.py params.nch=6 params.train_or_test="test","train" -m
# python3 src/simurate_pyroom.py params.nch=2 params.train_or_test="train" -m
# python3 src/simurate_pyroom.py params.nch=2 params.train_or_test="test" -m
# python3 src/simurate_pyroom.py params.train_or_test="test","train" -m
