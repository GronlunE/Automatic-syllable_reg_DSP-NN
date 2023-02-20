import glob
import numpy as np
from os import path
from scipy.io import savemat, loadmat
import scipy.io as spio
from librosa import get_duration
import pandas as pd

# To open metadata from .wav files
import taglib

# Own implementation
from Miscellaneous import run_matlab_engine, get_file_info


def build_logMel_npz(wav_root, matlabroot, npz_loc):
    """

    :return:
    """
    npz_dict = {}
    n = 0
    eng = run_matlab_engine(matlabroot=matlabroot)

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


def build_data(wav_root, npz_loc, matlabroot, tensordata_loc):
    """

    :return:
    """
    if not path.exists(tensordata_loc):
        if not path.exists(npz_loc):
            print("Building logMels...")
            build_logMel_npz(wav_root=wav_root, matlabroot=matlabroot, npz_loc=npz_loc)

        print("Unpacking syllables and logMels...\n")
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
        print("Forming tensor...\n")
        tensor = assemble_tensor(list_of_log_mels)

        savemat(tensordata_loc, {"tensor": tensor, "syllables": syll_train})

    else:
        print("Loading tensor and syllable data from memory...\n")
        mat_data = loadmat(tensordata_loc)
        tensor = mat_data["tensor"]
        syll_train = np.transpose(mat_data["syllables"])

    return tensor, syll_train


def import_test_mat(wav_root, npz_loc, tensordata_loc, language):

    mat_loc = r"resources\test_audio\english\SWB_anno.mat"
    mat2_loc = r"resources\test_audio\estonian\SKK_anno.mat"

    def loadmat(filename):
        '''
        this function should be called instead of direct spio.loadmat
        as it cures the problem of not properly recovering python dictionaries
        from mat files. It calls the function check keys to cure all entries
        which are still mat-objects
        '''
        data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
        return _check_keys(data)

    def _check_keys(dict):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in dict:
            if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
                dict[key] = _todict(dict[key])
        return dict

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        dict = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                dict[strg] = _todict(elem)
            else:
                dict[strg] = elem
        return dict

    anno = loadmat(mat_loc)
    syllables = list(anno["anno"]["syllables"])
    filenames = list(anno["anno"]["filename"])

    anno_2 = loadmat(mat2_loc)
    syllables_2 = list(anno_2["anno"]["syllables"])
    filenames_2 = list(anno_2["anno"]["filename"])

    s = syllables

    n = 0

    filename_list = []
    mel_data = np.load(npz_loc)

    for filepath in glob.glob(wav_root, recursive=True):
        file = filepath.split("\\")[-1]
        filename_list.append(file)

    test_tensor_dict = {}

    test_syllables = []
    list_of_log_mels = []
    list_of_durs = []
    current_files = []
    y = []

    if language == "english":
        y = filenames
        s = syllables
    elif language == "estonian":
        y = filenames_2
        s = syllables_2

    for filename in y:

        file = filename.split("/")[-1]
        wav = r"resources\test_audio" + "\\" + language + "\\" + file

        if file in filename_list:

            current_files.append(file.removesuffix(".wav"))
            sylls = len(list(s[y.index(filename)]))
            test_syllables.append(sylls)
            t = round(get_duration(filename = wav), 2)
            list_of_durs.append(t)
            list_of_log_mels.append(mel_data[file])

            if n % 1000 == 0:
                print(n, "Done")
            n = n+1

    tensor = assemble_tensor(list_of_log_mels)

    thetaSylls = pd.read_csv(r"resources\csv" + "\\" + language + ".csv")["Syllables"].tolist()

    test_tensor_dict["tensor"] = tensor
    test_tensor_dict["syllables"] = np.array(thetaSylls)
    test_tensor_dict["true_syllables"] = np.array(test_syllables)

    print("test tensor shape is:", np.shape(np.array(tensor)))
    print("test syllables shape is:", np.shape(np.array(test_syllables)))

    savemat(tensordata_loc, test_tensor_dict)
    print("All done")
