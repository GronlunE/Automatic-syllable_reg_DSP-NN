import glob
import wave
import contextlib
import sys
import matplotlib as plt

# Matlab
import matlab.engine


def get_audio_durs(wav_root):
    durs = []
    for filepath in glob.glob(wav_root, recursive=True):
        with contextlib.closing(wave.open(filepath, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
            durs.append(duration)

    plt.hist(durs)
    plt.show()


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
