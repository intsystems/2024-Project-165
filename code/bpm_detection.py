# Copyright 2012 Free Software Foundation, Inc.
#
# This file is part of The BPM Detector Python
#
# The BPM Detector Python is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
#
# The BPM Detector Python is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with The BPM Detector Python; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.

import argparse
import array
import math
import pdb
import wave

import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy import signal

import os
from pydub import AudioSegment
from collections import Counter


def read_wav(filename):
    # open file, get metadata for audio
    try:
        wf = wave.open(filename, "rb")
    except IOError as e:
        print(e)
        return

    # typ = choose_type( wf.getsampwidth() ) # TODO: implement choose_type
    nsamps = wf.getnframes()
    assert nsamps > 0

    fs = wf.getframerate()
    assert fs > 0

    # Read entire file and make into an array
    samps = list(array.array("i", wf.readframes(nsamps)))

    try:
        assert nsamps == len(samps)
    except AssertionError:
        print(nsamps, "not equal to", len(samps))

    return samps, fs


def convert_mp3_to_wav(mp3_filename):
    # Путь к новому WAV файлу
    wav_filename = mp3_filename.replace('.mp3', '.wav')

    # Загрузка и конвертация MP3 в WAV
    audio = AudioSegment.from_mp3(mp3_filename)
    audio.export(wav_filename, format="wav")

    return wav_filename


def read_audio_file(filename):
    # Проверка расширения файла
    _, file_extension = os.path.splitext(filename)

    if file_extension.lower() == '.mp3':
        # Если файл MP3, конвертировать его в WAV
        print("Конвертация MP3 в WAV...")
        filename = convert_mp3_to_wav(filename)

    # Чтение WAV файла
    samps, fs = read_wav(filename)
    return samps, fs


# print an error when no data can be found
def no_audio_data():
    print("No audio data for sample, skipping...")
    return None, None


# simple peak detection
def peak_detect(data):
    max_val = np.amax(abs(data))
    peak_ndx = np.where(data == max_val)
    if len(peak_ndx[0]) == 0:  # if nothing found then the max must be negative
        peak_ndx = np.where(data == -max_val)
    return peak_ndx


def bpm_detector(data, fs):
    cA = []
    cD = []
    correl = []
    cD_sum = []
    levels = 4
    max_decimation = 2 ** (levels - 1)
    min_ndx = math.floor(60.0 / 220 * (fs / max_decimation))
    max_ndx = math.floor(60.0 / 40 * (fs / max_decimation))

    for loop in range(0, levels):
        cD = []
        # 1) DWT
        if loop == 0:
            [cA, cD] = pywt.dwt(data, "db4")
            cD_minlen = len(cD) / max_decimation + 1
            cD_sum = np.zeros(math.floor(cD_minlen))
        else:
            [cA, cD] = pywt.dwt(cA, "db4")

        # 2) Filter
        cD = signal.lfilter([0.01], [1 - 0.99], cD)

        # 4) Subtract out the mean.

        # 5) Decimate for reconstruction later.
        cD = abs(cD[:: (2 ** (levels - loop - 1))])
        cD = cD - np.mean(cD)

        # 6) Recombine the signal before ACF
        # Essentially, each level the detail coefs (i.e. the HPF values) are concatenated to the beginning of the array
        cD_sum = cD[0:math.floor(cD_minlen)] + cD_sum

    if [b for b in cA if b != 0.0] == []:
        return no_audio_data()

    # Adding in the approximate data as well...
    cA = signal.lfilter([0.01], [1 - 0.99], cA)
    cA = abs(cA)
    cA = cA - np.mean(cA)
    cD_sum = cA[0 : math.floor(cD_minlen)] + cD_sum

    # ACF
    correl = np.correlate(cD_sum, cD_sum, "full")

    midpoint = math.floor(len(correl) / 2)
    correl_midpoint_tmp = correl[midpoint:]
    peak_ndx = peak_detect(correl_midpoint_tmp[min_ndx:max_ndx])
    if len(peak_ndx) > 1:
        return no_audio_data()

    peak_ndx_adjusted = peak_ndx[0] + min_ndx
    bpm = 60.0 / peak_ndx_adjusted * (fs / max_decimation)
    return bpm, correl


def adjust_bpm_counts(bpm_counts):
    """
    Adjust BPM counts by adding counts from half and quarter BPM values.

    Parameters:
        bpm_counts (Counter): A Counter object with BPM values as keys and their counts as values.

    Returns:
        int: The BPM value with the highest adjusted count.
    """
    adjusted_counts = Counter(bpm_counts)  # Make a copy to adjust without altering the original

    for bpm in list(bpm_counts):
        half_bpm = int(bpm / 2)
        quarter_bpm = int(bpm / 4)

        if half_bpm * 2 == bpm and half_bpm in bpm_counts:
            adjusted_counts[bpm] += bpm_counts[half_bpm]
        if quarter_bpm * 4 == bpm and quarter_bpm in bpm_counts:
            adjusted_counts[bpm] += bpm_counts[quarter_bpm]

    # Find the BPM with the maximum adjusted count
    max_bpm = max(adjusted_counts, key=adjusted_counts.get)
    return max_bpm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process .wav file to determine the Beats Per Minute.")
    parser.add_argument("--filename", required=True, help=".wav file for processing")
    parser.add_argument(
        "--window",
        type=float,
        default=3,
        help="Size of the the window (seconds) that will be scanned to determine the bpm. Typically less than 10 seconds. [3]",
    )

    args = parser.parse_args()
    samps, fs = read_audio_file(args.filename)
    data = []
    correl = []
    bpm = 0
    n = 0
    nsamps = len(samps)
    window_samps = int(args.window * fs)
    samps_ndx = 0  # First sample in window_ndx
    max_window_ndx = math.floor(nsamps / window_samps)
    bpms = np.zeros(max_window_ndx)
    bpms_rounded = np.zeros(max_window_ndx)

    # Iterate through all windows
    for window_ndx in range(0, max_window_ndx):

        # Get a new set of samples
        # print(n,":",len(bpms),":",max_window_ndx_int,":",fs,":",nsamps,":",samps_ndx)
        data = samps[samps_ndx : samps_ndx + window_samps]
        if not ((len(data) % window_samps) == 0):
            raise AssertionError(str(len(data)))

        bpm, correl_temp = bpm_detector(data, fs)
        if bpm is None:
            continue
        print(bpm, int(np.round(bpm)[0]))
        bpms[window_ndx] = bpm
        bpms_rounded[window_ndx] = int(np.round(bpm)[0])
        correl = correl_temp

        # Iterate at the end of the loop
        samps_ndx = samps_ndx + window_samps

        # Counter for debug...
        n = n + 1

    bpm = np.median(bpms)
    print("Completed!  Estimated Beats Per Minute:", bpm)
    bpms_rounded = bpms_rounded.astype(np.int64)
    print("Completed! Estimated Rounded BPM:", np.bincount(bpms_rounded).argmax())
    bpm_counter = Counter(bpms_rounded)
    print("Completed! Estimated Counter-Rounded BPM:", adjust_bpm_counts(bpm_counter))
    print("\nCounter:", bpm_counter)

    n = range(0, len(correl))
    # plt.plot(n, abs(correl))
    # plt.show(block=True)