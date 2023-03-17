import glob
import sys
import matplotlib.pyplot as plt
from librosa import get_duration
import numpy as np
import pandas as pd
from mat73 import loadmat
from config import*
from scipy.io import savemat

# Matlab
import matlab.engine


def get_logMel_shapes():
    """

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


def get_audio_durs():
    """

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


def run_matlab_engine():
    """

    :return:
    """
    sys.path.append(matlabroot)
    path_1 = r"matlab"
    path_2 = r"matlab\gammatone"
    path_3 = r"matlab\thetaOsc"

    eng = matlab.engine.start_matlab()
    eng.addpath(path_1, path_2, path_3)

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


def write_csv():
    language_dirs = [r"resources\test_audio\english"]

    npz_data = np.load(npz_loc)
    sylls = loadmat(tensordata_loc)["syllables"].flatten()
    print(sylls)
    n = 0
    for filepath in language_dirs:
        csv_dict = {"Filename": [], "Audio duration": [], "LogMel shape": [], "Syllables": []}
        language = filepath.split("\\")[-1]
        syllables = []
        filenames = []
        durations = []
        LM_shapes = []
        directory = filepath + "\\*.wav*"
        for file in glob.glob(directory, recursive=True):
            filename = file.split("\\")[-1]
            # syll = int(taglib.File(file).tags["SYLLABLE_COUNT"][0])
            syll = sylls[n]
            t = round(get_duration(filename=file), 2)
            LM_shape = np.shape(npz_data[filename])[0]

            filenames.append(filename.removesuffix(".wav"))
            LM_shapes.append(LM_shape)
            durations.append(t)
            syllables.append(syll)
            n = n + 1

        csv_dict["Filename"] = filenames
        csv_dict["Audio duration"] = durations
        csv_dict["Syllables"] = syllables
        csv_dict["LogMel shape"] = LM_shapes

        csv_df = pd.DataFrame(csv_dict)
        csv_df.to_csv(filepath + "\\" + language + ".csv", index=False)


def get_filepaths(root):
    """

    :param root:
    :return:
    """
    filepaths = []

    for filepath in glob.glob(root, recursive=True):
        filepaths.append(filepath)

    return filepaths


def set_labels():

    # Load the existing mat file
    tensordata = loadmat(tensordata_loc)

    # Define the labels
    labels = np.empty(17996, dtype='object')
    labels[:6000] = 'french'
    labels[6000:11998] = 'polish'
    labels[11998:] = 'spanish'

    # Add the labels to the mat file
    tensordata['labels'] = labels

    # Save the updated mat file
    savemat(tensordata_loc, tensordata)

    # Add the labels to the mat file
    tensordata['labels'] = labels

    # Save the updated mat file
    savemat(tensordata_loc, tensordata)
