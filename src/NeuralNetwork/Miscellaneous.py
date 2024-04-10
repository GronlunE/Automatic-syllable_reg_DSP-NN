"""
Created on Wed Apr 10 16:30:00 2024

@author: GronlunE

Purpose:
Contains miscellaneous functions used in the other scripts of the project.

"""


import glob
import sys
import matplotlib.pyplot as plt
from librosa import get_duration
import numpy as np
import pandas as pd
from config import*
from scipy.io import savemat
import scipy.io as spio
import os
import wave
from pydub import AudioSegment

# Matlab
import matlab.engine


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


def get_audio_durs(root):
    """

    :return:
    """
    durs = []
    n = 0
    for filepath in glob.glob(root, recursive=True):
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


def pred_to_csv(mat_file_path):
    # Load the .mat file
    data = loadmat(mat_file_path)
    start_index = 1

    # Determine the starting index for the enumerate based on whether Command_1 exists
    if 'Command_1' not in data:
        start_index = 2

    # Create an empty dictionary to store the data
    data_dict = {}
    data_keys = list(data.keys())[3:]

    # Loop through the commands and extract the necessary data
    for command in data_keys:

        label = data[command]['Call'][1]

        if isinstance(label, np.ndarray):
            label = ",".join([s[:3] for s in label])

        elif not label.isnumeric():
            label = label[:3]

        # Get the MAE and MAPE values for Estonian and English
        est_mae = data[command]['Predictions']['estonian']['MAE']
        est_mape = data[command]['Predictions']['estonian']['MAPE']
        eng_mae = data[command]['Predictions']['english']['MAE']
        eng_mape = data[command]['Predictions']['english']['MAPE']

        # Add the data to the dictionary
        data_dict[label] = {'Estonian MAE': est_mae, 'Estonian MAPE': est_mape,
                            'English MAE': eng_mae, 'English MAPE': eng_mape}

        filename = "langdep_preds.csv"

        # Convert the dictionary to a Pandas DataFrame
        df = pd.DataFrame.from_dict(data_dict, orient='index')

        # Save the DataFrame as a CSV file with the constructed filename
        df.to_csv(rf"F:\Thesis\{filename}")


def csv_to_latex(csv_loc):

    # Read in the CSV file as a DataFrame
    df = pd.read_csv(csv_loc)

    # Export the DataFrame as a LaTeX table
    with open(csv_loc, 'w') as f:
        f.write(df.to_latex(index=False))


def dur_hist():

    # Calculate the optimal bin size for each histogram using the Freedman-Diaconis rule
    def freedman_diaconis(data):
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        h = 2 * iqr / (len(data) ** (1 / 3))
        bin_size = np.ceil((max(data) - min(data)) / h)
        return int(bin_size)

    # Define the directories you want to search for .wav files
    directory1 = r"G:\AudioData\languages\original"
    directory2 = r"resources\audio"

    # Create empty lists to store the durations of all .wav files
    durations1 = []
    durations2 = []

    # Loop through all files and subdirectories in the first directory
    for root, dirs, files in os.walk(directory1):
        for file in files:
            # Check if the file is a .wav file
            if file.endswith(".wav"):
                # Open the .wav file and get its duration
                with wave.open(os.path.join(root, file), "rb") as f:
                    frames = f.getnframes()
                    rate = f.getframerate()
                    duration = frames / float(rate)
                    durations1.append(duration)

    # Loop through all files and subdirectories in the second directory
    for root, dirs, files in os.walk(directory2):
        for file in files:
            # Check if the file is a .wav file
            if file.endswith(".wav"):
                # Open the .wav file and get its duration
                with wave.open(os.path.join(root, file), "rb") as f:
                    frames = f.getnframes()
                    rate = f.getframerate()
                    duration = frames / float(rate)
                    durations2.append(duration)

    # Calculate the optimal number of bins for both datasets
    bins1 = freedman_diaconis(np.array(durations1))
    bins2 = freedman_diaconis(np.array(durations2))

    plt.rcParams.update({'font.size': 26})
    plt.figure(figsize=(12, 8))

    # Plot the histogram for the first directory
    plt.hist(durations1, bins=bins1)
    plt.title("Original")
    plt.xlabel("Duration (s)")
    plt.ylabel("Count")

    # Save the histogram for the first directory as a png file
    plt.savefig("original_recordings.pdf")

    # Clear the figure
    plt.clf()

    plt.rcParams.update({'font.size': 26})
    plt.figure(figsize=(12, 8))

    # Plot the histogram for the second directory
    plt.hist(durations2, bins=bins2)
    plt.title("Segmented")
    plt.xlabel("Duration (s)")
    plt.ylabel("Count")

    # Save the histogram for the second directory as a png file
    plt.savefig("segmented_recordings.pdf")


def calculate_mean_syllables_per_duration(language):
    csv_folder = r"resources/csv"
    filename = f"{language}.csv"
    csv_path = os.path.join(csv_folder, filename)

    df = pd.read_csv(csv_path)

    syllables_header = "TrueSyllables" if language in ["Estonian", "English"] else "Syllables"
    df["syllables_per_duration"] = df[syllables_header] / df["Audio duration"]

    mean_syllables_per_duration = df["syllables_per_duration"].mean()
    return mean_syllables_per_duration


def data_table():
    """

    :return:
    """

    # Function to get audio file durations in seconds
    def get_duration(file_path):
        audio = AudioSegment.from_file(file_path)
        duration = len(audio) / 1000  # Duration in seconds
        return duration

    # Define the paths to the language folders
    french = r"resources/audio/french"
    polish = r"resources/audio/polish"
    spanish = r"resources/audio/spanish"
    estonian = r"resources/test_audio/estonian"
    english = r"resources/test_audio/english"
    training_folders = [french, polish, spanish]
    train_lang_name = ["French", "Polish", "Spanish"]
    testing_folders = [estonian, english]
    test_lang_name = ["Estonian", "English"]

    # Initialize lists to store language data
    training_data = []
    testing_data = []

    # Loop through the training folders
    for folder in training_folders:
        i = training_folders.index(folder)
        language_data = {'Language': train_lang_name[i]}

        # Get a list of audio files in the folder
        audio_files = [file for file in os.listdir(folder) if file.endswith('.wav')]

        # Get file count
        language_data['File Count'] = len(audio_files)

        # Get total duration
        total_duration = sum([get_duration(os.path.join(folder, file)) for file in audio_files])
        language_data['Total Duration (hrs)'] = round(total_duration / 3600, 2)

        # Get mean duration
        mean_duration = total_duration / len(audio_files)
        language_data['Mean Duration (secs)'] = round(mean_duration, 2)

        # Get mean syllables/duration
        mean_syll_per_dur = calculate_mean_syllables_per_duration(train_lang_name[i])
        language_data['Mean Syllables/Duration (per/s)'] = round(mean_syll_per_dur, 2)

        training_data.append(language_data)

    # Loop through the testing folders
    for folder in testing_folders:
        i = testing_folders.index(folder)
        language_data = {'Language': test_lang_name[i]}

        # Get a list of audio files in the folder
        audio_files = [file for file in os.listdir(folder) if file.endswith('.wav')]

        # Get file count
        language_data['File Count'] = len(audio_files)

        # Get total duration
        total_duration = sum([get_duration(os.path.join(folder, file)) for file in audio_files])
        language_data['Total Duration (hrs)'] = round(total_duration / 3600, 2)

        # Get mean duration
        mean_duration = total_duration / len(audio_files)
        language_data['Mean Duration (secs)'] = round(mean_duration, 2)

        # Get mean syllables/duration
        mean_syll_per_dur = calculate_mean_syllables_per_duration(test_lang_name[i])
        language_data['Mean Syllables/Duration (per/s)'] = round(mean_syll_per_dur, 2)

        testing_data.append(language_data)

    training_df = pd.DataFrame(training_data)
    testing_df = pd.DataFrame(testing_data)

    training_df.to_csv('training_data.csv', index=False)
    testing_df.to_csv('testing_data.csv', index=False)
