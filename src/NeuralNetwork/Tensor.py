from os import path
import glob

import numpy as np
from scipy.io import savemat, loadmat

import taglib
from Miscellaneous import run_matlab_engine, get_file_info


def build_logMel_npz(wav_root, matlab_home, npz_loc):
    """

    :return:
    """
    npz_dict = {}
    n = 0
    eng = run_matlab_engine(matlab_home=matlab_home)

    for filepath in glob.glob(wav_root, recursive=True):
        file_info = get_file_info(filepath)
        filename = file_info["filename"]

        logMel = np.array(eng.logMel(filepath)).astype(float)
        npz_dict[filename] = logMel

        if n % 1000 == 0:
            print(n, "Done")
        n = n+1

    print("All done")
    np.savez(npz_loc, **npz_dict)


def form_dict(wav_root, npz_loc):
    """


    :return:
    """
    n = 0
    data_dict = {}
    mel_data = np.load(npz_loc)
    for file in glob.glob(wav_root, recursive=True):

        # Get filename and language
        file_data = get_file_info(file)
        file_name = file_data["filename"]

        # Get syllables for for the audio
        wav_file = taglib.File(file)
        syllables = int(wav_file.tags["SYLLABLE_COUNT"][0])

        # Get log_Mel for the audio
        log_mel = mel_data[file_name]

        # Add to dict
        data_dict[file_name] = [syllables, log_mel]

        if n % 1000 == 0:
            print(n, "Done")
        n = n + 1

    print("All done")
    return data_dict


def assemble_tensor(wav_logMels, T=650):
    """

    :param wav_logMels:
    :param T:
    :return:
    """
    n = 0

    # Number of samples
    N = len(wav_logMels)

    # Feature dimension
    D = wav_logMels[0].shape[1]

    # Initialize a numpy array with the desired shape
    output_tensor = np.zeros((N, T, D))

    for i, logMel in enumerate(wav_logMels):

        # Crop if longer than T frames
        if logMel.shape[0] > T:
            logMel = logMel[:T, :]

        # Pad if shorter than T frames
        elif logMel.shape[0] < T:
            padding = np.zeros((T - logMel.shape[0], D))
            logMel = np.concatenate((logMel, padding), axis=0)

        # Assign the processed logMel to the output tensor
        output_tensor[i, :, :] = logMel

        if n % 1000 == 0:
            print(n, "Done")
        n = n+1

    print("All done")
    return output_tensor


def build_training_data(wav_root, npz_loc, matlab_home, tensordata_loc):
    """

    :return:
    """
    if not path.exists(tensordata_loc):
        if not path.exists(npz_loc):
            print("Building logMels...")
            build_logMel_npz(wav_root=wav_root, matlab_home=matlab_home, npz_loc=npz_loc)

        print("Unpacking syllables and logMels...")
        # Form "filename: [syllables, log-Mel]" dict for the existing audio files
        data_dict = form_dict(wav_root=wav_root, npz_loc=npz_loc)

        list_of_log_mels = []
        syllables = []

        # Divide the dict into lists of log-Mel values and syllable values index-wise
        for key in data_dict.keys():
            syllables.append(data_dict[key][0])
            list_of_log_mels.append(data_dict[key][1])
        syll_train = np.array(syllables)

        # Form Tensor
        print("Forming tensor...")
        tensor = assemble_tensor(list_of_log_mels)

        savemat(tensordata_loc, {"tensor": tensor, "syll_train": syll_train})

    else:
        print("Loading tensor and syllable data from memory...")
        mat_data = loadmat(tensordata_loc)
        tensor = mat_data["tensor"]
        syll_train = np.transpose(mat_data["syll_train"])

    return tensor, syll_train
