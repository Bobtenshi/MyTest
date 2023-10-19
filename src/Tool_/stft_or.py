#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import wave
import math
from scipy.fftpack import fft, ifft
from scipy import hamming
from scipy.io import wavfile
from scipy.signal import resample
import scipy.linalg as LA
import matplotlib.pyplot as plt

NP_EPS = np.finfo(np.float64).eps
FIG_SIZZE = {"mm": 1 / 25.4, "pt": 1 / 72}
axes_size = 0.3
plt.rcParams["font.size"] = 2
plt.rcParams["axes.labelsize"] = "x-small"
plt.rcParams["xtick.labelsize"] = "x-small"
plt.rcParams["ytick.labelsize"] = "x-small"
plt.rcParams["axes.linewidth"] = axes_size
plt.rcParams["xtick.major.size"] = axes_size * 5
plt.rcParams["xtick.major.width"] = axes_size
plt.rcParams["ytick.major.size"] = axes_size * 5
plt.rcParams["ytick.major.width"] = axes_size


### for STFT ###
def stft(signal, fftSize, shiftSize, window="hamming"):
    """
    Parameters
    ----------
    signal: input signal
    fftSize: frame length
    shiftSize: frame shift
    window: window function
    Returns
    -------
    S: spectrogram of input signal (fftSize/2+1 x frame x ch)
    window: used window function (fftSize x 1)
    """
    signal = np.array(signal)

    if window == "hamming":
        # todo 色々対応
        window = hamming(fftSize + 1)[:fftSize]

    nch = signal.shape[1]

    zeroPadSize = fftSize - shiftSize
    signal = np.concatenate(
        [np.zeros([zeroPadSize, nch]), signal, np.zeros([fftSize, nch])]
    )
    length = signal.shape[0]

    frames = int(np.floor((length - fftSize + shiftSize) / shiftSize))
    I = int(fftSize / 2 + 1)
    S = np.zeros([I, frames, nch], dtype=np.complex128)

    for ch in range(nch):
        for j in range(frames):
            sp = j * shiftSize
            spectrum = fft(signal[sp : sp + fftSize, ch] * window)
            S[:, j, ch] = spectrum[:I]

    return S, window


def whitening(X, dnum=2):
    # M == dnumで固定 (todo)
    I, J, M = X.shape
    Y = np.zeros(X.shape, dtype=np.complex128)

    def _whitening(Xi):
        V = Xi @ Xi.T.conjugate() / J  # covariance matrix (M, M)
        eig_val, P = LA.eig(V)
        D = np.diag(eig_val)

        idx = np.argsort(eig_val)
        D = D[idx, idx]
        P = P[:, idx]
        return (
            np.diag(D ** (-0.5)) @ P.T.conjugate() @ Xi
        ).T  # (M, M) * (M, M) * (M, J)

    for i in range(I):
        Y[i] = _whitening(X[i].T)

    return Y


def power(S):
    return np.real(S) ** 2 + np.imag(S) ** 2


def spectrogram(S, output_path=None):
    import pylab as pl

    pl.figure()
    I, J = S.shape
    X, Y = pl.meshgrid(pl.arange(J + 1), pl.arange(I + 1))

    S = power(S)
    # pl.pcolormesh(X, Y, np.log(S))
    pl.pcolormesh(X, Y, np.log(S))
    pl.xlabel("Time-frame ")
    pl.ylabel("Frequency [Hz]")
    pl.colorbar(orientation="vertical")
    pl.clim(-30, 10)

    if output_path is None:
        pl.show()
    else:
        pl.savefig(output_path)


def spectrogram_diff(S, REF, itr, output_path=None):
    """_summary_

    Args:
        S (DNN output ):(mic, freq, time-frame)
        REF(DNN output ):(mic, freq, time-frame)
        output_path :path

    """
    M, I, J = S.shape

    if M > 3:
        fig, axes = plt.subplots(1, M // 2, constrained_layout=True)
        fig.set_size_inches(320 * FIG_SIZZE["pt"], 80 * FIG_SIZZE["pt"])

        for clm in range(M // 2):
            Spec = np.maximum(power(S[clm, :, :] - REF[clm, :, :]), NP_EPS)
            im = axes[clm].imshow(
                np.log(Spec), interpolation="nearest", aspect=0.08, clim=(-10, 10)
            )
            axes[clm].set_title(f"diff_R_itr{itr}_mic{clm}")
            axes[clm].invert_yaxis()
            if clm == 0:
                axes[clm].set_xlabel("Time-frame ")
                axes[clm].set_ylabel("Frequency [Hz]")
    else:
        fig, axes = plt.subplots(1, M, constrained_layout=True)
        fig.set_size_inches(320 * FIG_SIZZE["pt"], 160 * FIG_SIZZE["pt"])
        for clm in range(M):
            Spec = power(S[clm, :, :] - REF[clm, :, :])
            im = axes[clm].imshow(
                np.log(Spec), interpolation="nearest", aspect=0.08, clim=(-10, 10)
            )
            axes[clm].set_title(f"diff_R_itr{itr}_mic{clm}")
            axes[clm].invert_yaxis()
            if clm == 0:
                axes[clm].set_xlabel("Time-frame ")
                axes[clm].set_ylabel("Frequency [Hz]")
    if output_path is None:
        plt.show()
    else:
        # plt.tight_layout() # 追加
        fig.colorbar(im)
        plt.savefig(output_path)


def spectrogram_inout(S, R, itr, output_path=None):
    """_summary_

    Args:
        S (DNN input ):(mic, freq, time-frame)
        R (DNN output):( mic, freq, time-frame)
        output_path :path

    """
    # import pylab as pl
    fig = plt.figure()
    M, I, J = S.shape
    X, Y = np.meshgrid(np.arange(J + 1), np.arange(I + 1))

    fig, axes = plt.subplots(2, M // 2, constrained_layout=True)
    fig.set_size_inches(320 * FIG_SIZZE["pt"], 160 * FIG_SIZZE["pt"])

    for row in range(2):
        for clm in range(M // 2):
            if row % 2 == 0:
                Spec = np.maximum(power(S[clm, :, :]), NP_EPS)
                im = axes[row, clm].imshow(
                    np.log(Spec), interpolation="nearest", aspect=0.08, clim=(-30, 10)
                )
                axes[row, clm].set_title(f"input_mic{clm}")
                axes[row, clm].invert_yaxis()
                if clm == 0:
                    axes[row, clm].set_xlabel("Time-frame ")
                    axes[row, clm].set_ylabel("Frequency [Hz]")

            else:
                Spec = np.maximum(power(R[clm, :, :]), NP_EPS)
                im = axes[row, clm].imshow(
                    np.log(Spec), interpolation="nearest", aspect=0.08, clim=(-30, 10)
                )
                axes[row, clm].set_title(f"output_mic{clm}")
                axes[row, clm].invert_yaxis()
                if clm == 0:
                    axes[row, clm].set_xlabel("Time-frame ")
                    axes[row, clm].set_ylabel("Frequency [Hz]")

    if output_path is None:
        plt.show()
    else:
        fig.colorbar(im)
        # plt.tight_layout() # 追加
        plt.savefig(output_path)


### for ISTFT ###
def optSynWnd(analysisWindow, shiftSize):
    fftSize = analysisWindow.shape[0]
    synthesizedWindow = np.zeros(fftSize)
    for i in range(shiftSize):
        amp = 0
        for j in range(1, int(fftSize / shiftSize) + 1):
            amp += analysisWindow[i + (j - 1) * shiftSize] ** 2
        for j in range(1, int(fftSize / shiftSize) + 1):
            synthesizedWindow[i + (j - 1) * shiftSize] = (
                analysisWindow[i + (j - 1) * shiftSize] / amp
            )

    return synthesizedWindow


def istft(S, shiftSize, window, length):
    """
    % [inputs]
    %           S: STFT of input signal (fftSize/2+1 x frames x nch)
    %   shiftSize: frame shift (default: fftSize/2)
    %      window: window function used in STFT (fftSize x 1) or choose used
    %              function from below.
    %              'hamming'    : Hamming window (default)
    %              'hann'       : von Hann window
    %              'rectangular': rectangular window
    %              'blackman'   : Blackman window
    %              'sine'       : sine window
    %      length: length of original signal (before STFT)
    %
    % [outputs]
    %   waveform: time-domain waveform of the input spectrogram (length x nch)
    %
    """
    freq, frames, nch = S.shape
    fftSize = (freq - 1) * 2
    invWindow = optSynWnd(window, shiftSize)

    spectrum = np.zeros(fftSize, dtype=np.complex128)
    tmpSignal = np.zeros([(frames - 1) * shiftSize + fftSize, nch])
    for ch in range(nch):
        for j in range(frames):
            spectrum[: int(fftSize / 2) + 1] = S[:, j, ch]
            spectrum[0] /= 2
            spectrum[int(fftSize / 2)] /= 2
            sp = j * shiftSize
            tmpSignal[sp : sp + fftSize, ch] += (
                np.real(ifft(spectrum, fftSize) * invWindow) * 2
            )

    waveform = tmpSignal[fftSize - shiftSize : (frames - 1) * shiftSize + fftSize]

    waveform = waveform[:length]
    return waveform


def main(input_path, output_path):
    """
    input_path : wav
    output_path : png
    """
    fs, signal = wavfile.read(input_path)
    S, window = stft(signal, fftSize=4096 * 2, shiftSize=2048)
    S = power(S)
    spectrogram(S[:, :, 0], output_path)


if __name__ == "__main__":
    # exit(main('input/synth_res.wav', 'test.png'))

    fs, signal1 = wavfile.read("input/guitar_res.wav")
    signal1 = signal1 / 32768.0
    assert fs == 16000
    fs, signal2 = wavfile.read("input/synth_res.wav")
    signal2 = signal2 / 32768.0
    assert fs == 16000
    nch = 2

    mix = np.zeros([len(signal1), nch])

    for ch in range(nch):
        mix[:, ch] = signal1[:, ch] + signal2[:, ch]

    S, window = stft(mix, fftSize=4096, shiftSize=2048)
    mix2 = istft(S, 2048, window, mix.shape[0])
    wavfile.write("est.wav", 16000, mix2)
