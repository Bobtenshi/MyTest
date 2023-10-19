import numpy as np
import glob
import tqdm
import random
import librosa
from constant import SAMP_RATE
from natsort import natsorted
from scipy.io import wavfile
import os


folder_n = 0
for path_each_dataset in tqdm.tqdm(
    natsorted(glob.glob(f"data/WSJ0/11-1[4-5]*"))
):  # 11.1 - 11.15
    for path_each_situation in natsorted(glob.glob(f"{path_each_dataset}/wsj0/*")):
        for path_each_human in natsorted(glob.glob(f"{path_each_situation}/*")):

            folder_n += 1
            wav_array = np.empty((0))
            wav_array_trimmed = np.empty((0))

            wav_paths = natsorted(glob.glob(f"{path_each_human}/*.wav"))
            random.shuffle(wav_paths)
            for wav_path in wav_paths:
                wav_array = np.append(
                    wav_array, librosa.load(wav_path, sr=SAMP_RATE)[0]
                )  # connatenate wav file

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

            os.makedirs(f"data/trimmed_wav/wsj0/", exist_ok=True)
            wavfile.write(
                f"data/trimmed_wav/wsj0/trimmed_{folder_n}.wav",
                SAMP_RATE,
                wav_array_trimmed[:],
            )
print(folder_n)
