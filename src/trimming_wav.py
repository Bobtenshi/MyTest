import enum
from posixpath import pathsep
import numpy as np
import glob
import tqdm
import random
import librosa
from constant import SAMP_RATE
from scipy.io import loadmat, wavfile
import os


def trimm_jvs():
    for folder_n, path_each_human in enumerate(
        tqdm.tqdm(glob.glob(f"data/jvs_ver1/*/"))
    ):

        wav_array = np.empty((0))
        wav_array_trimmed = np.empty((0))
        wav_paths = glob.glob(f"{path_each_human}parallel*/wav24kHz16bit/*.wav")
        random.shuffle(wav_paths)
        for wav_path in wav_paths:
            # wav_array = np.append(wav_array, (librosa.load(wav_path, sr=SAMP_RATE)[0]*32767).astype(int)) # connatenate wav file
            wav_array = np.append(
                wav_array, librosa.load(wav_path, sr=SAMP_RATE)[0]
            )  # connatenate wav file

        # wav_array_trimmed, index = librosa.effects.trim(wav_array, top_db=1, frame_length = 128, hop_length = 64)
        points = librosa.effects.split(
            wav_array, top_db=50, frame_length=2048, hop_length=512
        )
        for point in points:
            wav_array_trimmed = np.append(
                wav_array_trimmed, wav_array[point[0] : point[1]]
            )  # connatenate wav file
            wav_array_trimmed = np.append(
                wav_array_trimmed, np.zeros(3200, dtype="float32")
            )  # connatenate wav file

        print(len(wav_array) / 16000)  # about 800 s
        print(len(wav_array_trimmed) / 16000)  # about 800 s

        os.makedirs(f"data/trimmed_wav/jvs/", exist_ok=True)
        wavfile.write(
            f"data/trimmed_wav/jvs/trimmed_{folder_n}.wav",
            SAMP_RATE,
            wav_array_trimmed[:],
        )


if __name__ == "__main__":
    trimm_jvs()
