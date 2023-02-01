import glob
import sys
import taglib
from os import path

import numpy as np
import matplotlib.pyplot as plt
import matlab.engine

from tensorflow import keras
from keras import layers

"""
[input layer]
[RNN layer 1] (return-sequences=true),128 dim
[RNN layer 2]  (return-sequences=false), 128 dim
[dense layer] 1dim (GELU, ELU or RELU):
"""


def run_malab_engine(matlab_home = r"C:\Program Files\MATLAB\R2022b"):
    """

    :param matlab_home:
    :return:
    """
    sys.path.append(matlab_home)
    path_1 = r"matlab_logMel"

    eng = matlab.engine.start_matlab()
    eng.addpath(path_1)

    return eng


def build_logMel_npz(root = r"resources\audio\**\*.wav", save_loc = r"resources\logMel.npz"):
    """

    :param save_loc:
    :param root:
    :return:
    """
    npz_dict = {}
    n = 0

    eng = run_malab_engine()
    for filepath in glob.glob(root, recursive=True):

        file_info = get_file_info(filepath)
        filename = file_info["filename"]

        logMel = np.array(eng.logMel(filepath)).astype(float)
        npz_dict[filename] = logMel

        if n % 1000 == 0:
            print(n, "Done")
        n = n+1

    np.savez(save_loc, **npz_dict)


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


def form_dict(root, languages):
    """

    :param languages:
    :param root:
    :return:
    """
    n = 0
    data_dict = {}
    mel_data = np.load(r"resources\logMel.npz")
    for file in glob.glob(root, recursive=True):

        # Get filename and language
        file_data = get_file_info(file)
        file_name = file_data["filename"]
        file_language = file_data["language"]

        if file_language in languages:

            # Get syllables for for the audio
            wav_file = taglib.File(file)
            syllables = int(wav_file.tags["SYLLABLE_COUNT"][0])

            # Get log_Mel for the audio
            log_mel = mel_data[file_name]

            # Add to dict
            data_dict[file_name] = [syllables, log_mel]

        if n % 1000 == 0:
            print(n, "Done")
        n = n+1

    return data_dict


def form_tensor(input_data, T=650):
    """

    :param input_data:
    :param T:
    :return:
    """
    # Input data as a list of matrices
    # input_data = [matrix_1, matrix_2, ..., matrix_n]

    n = 0

    # Number of samples
    N = len(input_data)

    # Feature dimension
    D = input_data[0].shape[1]

    # Initialize a numpy array with the desired shape
    output_tensor = np.zeros((N, T, D))

    for i, matrix in enumerate(input_data):

        # Crop if longer than T frames
        if matrix.shape[0] > T:
            matrix = matrix[:T, :]

        # Pad if shorter than T frames
        elif matrix.shape[0] < T:
            padding = np.zeros((T - matrix.shape[0], D))
            matrix = np.concatenate((matrix, padding), axis=0)

        # Assign the processed matrix to the output tensor
        output_tensor[i, :, :] = matrix

        if n % 1000 == 0:
            print(n, "Done")
        n = n+1

    return output_tensor, N, T, D


def plot_model(model):
    """

    :param model:
    :return:
    """

    loss = model.history['loss']
    val_loss = model.history['val_loss']
    epochs = range(1, len(val_loss) + 1)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()


def run_RNN(root = r"resources\audio\**\*.wav"):
    """

    :param root:
    :return:
    """
    if not path.exists(r"src\resources\logMel.npz"):
        print("Building logMels...")
        build_logMel_npz()

    print("Unpacking syllables and logMels...")
    # Form "filename: [syllables, log-Mel]" dict for the existing audio files
    languages = ["french"]
    data_dict = form_dict(root, languages)
    list_of_log_mels = []
    syllables = []

    # Divide the dict into lists of log-Mel values and syllable values index-wise
    for key in data_dict.keys():
        syllables.append(data_dict[key][0])
        list_of_log_mels.append(data_dict[key][1])
    syll_train = np.array(syllables)

    # Form Tensor
    print("Forming tensor...")
    tensor, N, T, D = form_tensor(list_of_log_mels)

    print("Tensor dimensions:", [N, T, D])
    print("Syllable dimensions:", np.shape(syll_train))

    # Input Layer
    inputs = keras.Input(shape=(None, D))

    # GRU Layer 1
    gru_1 = layers.GRU(128, return_sequences=True)(inputs)

    # GRU Layer 2
    gru_2 = layers.GRU(128, return_sequences=False)(gru_1)

    # Dense Layer
    dense = layers.Dense(1, activation='relu')(gru_2)

    # Create the model
    model = keras.Model(inputs=inputs, outputs=dense)

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    history = model.fit(tensor, syll_train,
                        epochs=10,
                        batch_size=32,
                        validation_split=0.2,
                        callbacks=[keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 10 ** (epoch/30))])

    plot_model(history)


run_RNN()
