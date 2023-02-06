import glob
import sys
import matplotlib.pyplot as plt
from librosa import get_duration
import numpy as np

# Matlab
import matlab.engine


def get_logMel_shapes(npz_loc):
    """

    :param npz_loc:
    :return:
    """
    mel_data = np.load(npz_loc)
    list_of_mel_shape = []
    n = 0
    for name in mel_data:
        array = mel_data[name]
        list_of_mel_shape.append(np.shape(array)[0])
        if n % 1000 == 0:
            print(n, "Done")
        n = n + 1
    print("All Done")
    return list_of_mel_shape


def get_audio_durs(wav_root):
    """

    :param wav_root:
    :return:
    """
    durs = []
    n = 0
    for filepath in glob.glob(wav_root, recursive=True):
        t = get_duration(filename=filepath)
        durs.append(t)
        if n % 1000 == 0:
            print(n, "Done")
        n = n + 1
    print("All Done")
    return durs


def compare_dur_and_logMel_shape(logMels, durations):
    plt.hist(logMels)
    plt.figure()
    plt.hist(durations)
    plt.show()
    return


def run_matlab_engine(matlabroot):
    """

    :param matlabroot:
    :return:
    """
    sys.path.append(matlabroot)
    path_1 = r"matlab"

    eng = matlab.engine.start_matlab()
    eng.addpath(path_1)

    return eng


def get_file_info(filepath):
    """

    :param filepath:
    :return:
    """
    file_info = {}
    split = filepath.split("\\")
    filename = split[-1]
    language = split[-2]

    file_info["filepath"] = filepath
    file_info["filename"] = filename
    file_info["language"] = language

    return file_info