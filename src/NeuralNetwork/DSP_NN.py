import glob
import sys
import wave
import contextlib

from scipy.io import savemat, loadmat
import taglib
from os import path

import numpy as np
import matplotlib.pyplot as plt
import matlab.engine

from tensorflow import keras
# from keras.layers import Dense, Dropout, Embedding, LSTM, merging, add, MaxPooling1D
# from keras.layers import Flatten, Conv2D, MaxPooling2D, UpSampling2D, Conv1D, UpSampling1D, AveragePooling1D, Multiply
from keras.layers import Input, TimeDistributed
from keras.layers import Dense, Dropout, Conv1D, GRU
from keras import regularizers
from keras.models import Model

"""
[input layer]
[RNN layer 1] (return-sequences=true),128 dim
[RNN layer 2]  (return-sequences=false), 128 dim
[dense layer] 1dim (GELU, ELU or RELU):
"""


def get_audio_durs(root = r"resources\audio\french\*.wav"):
    durs = []
    for filepath in glob.glob(root, recursive=True):
        with contextlib.closing(wave.open(filepath, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
            durs.append(duration)

    plt.hist(durs)
    plt.show()


def run_matlab_engine(matlab_home =r"C:\Program Files\MATLAB\R2022b"):
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
    eng = run_matlab_engine()

    for filepath in glob.glob(root, recursive=True):
        file_info = get_file_info(filepath)
        filename = file_info["filename"]

        logMel = np.array(eng.logMel(filepath)).astype(float)
        npz_dict[filename] = logMel

        if n % 1000 == 0:
            print(n, "Done")
        n = n+1

    print("All done")
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


def form_dict(root):
    """

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


def form_tensor(input_data, T=650):
    """

    :param input_data:
    :param T:
    :return:
    """

    # Input data as a list of matrices
    # input_data = [matrix_1, matrix_2, ..., matrix_n]
    list_of_N = []
    for matrix in input_data:
        list_of_N.append(np.shape(matrix)[0])
    plt.hist(list_of_N)
    plt.show()

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

    print("All done")
    return output_tensor


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


def run_RNN(root):
    """

    :param root:
    :return:
    """

    def wavenet_model(X_in, n_channels):
        """
        Created on Thu Feb  2 17:30:38 2023

        @author: rasaneno
        """
        # Define WaveNet encoder model
        conv_length = [2, 2, 2, 2, 2]
        # pooling_length = [1, 1, 1, 1, 1]
        conv_dilation = [1, 2, 4, 8, 16]
        actreg = 0.00000001
        # relu_actreg = 0.00000001

        dropout_rate = 0.1

        sequence1 = Input(shape=(X_in.shape[1:]))
        # sequence2 = Input(shape=(X_in.shape[1:]))
        encoder1 = Conv1D(n_channels, conv_length[0], dilation_rate=conv_dilation[0], activation='sigmoid',
                          padding='causal', activity_regularizer=regularizers.l2(actreg))
        encoder2 = Conv1D(n_channels, conv_length[1], dilation_rate=conv_dilation[1], activation='sigmoid',
                          padding='causal', activity_regularizer=regularizers.l2(actreg))
        encoder3 = Conv1D(n_channels, conv_length[2], dilation_rate=conv_dilation[2], activation='sigmoid',
                          padding='causal', activity_regularizer=regularizers.l2(actreg))
        encoder4 = Conv1D(n_channels, conv_length[3], dilation_rate=conv_dilation[3], activation='sigmoid',
                          padding='causal', activity_regularizer=regularizers.l2(actreg))
        encoder5 = Conv1D(n_channels, conv_length[4], dilation_rate=conv_dilation[4], activation='sigmoid',
                          padding='causal', activity_regularizer=regularizers.l2(actreg))

        encoder1_tanh = Conv1D(n_channels, conv_length[0], dilation_rate=conv_dilation[0], activation='tanh',
                               padding='causal', activity_regularizer=regularizers.l2(actreg))
        encoder2_tanh = Conv1D(n_channels, conv_length[1], dilation_rate=conv_dilation[1], activation='tanh',
                               padding='causal', activity_regularizer=regularizers.l2(actreg))
        encoder3_tanh = Conv1D(n_channels, conv_length[2], dilation_rate=conv_dilation[2], activation='tanh',
                               padding='causal', activity_regularizer=regularizers.l2(actreg))
        encoder4_tanh = Conv1D(n_channels, conv_length[3], dilation_rate=conv_dilation[3], activation='tanh',
                               padding='causal', activity_regularizer=regularizers.l2(actreg))
        encoder5_tanh = Conv1D(n_channels, conv_length[4], dilation_rate=conv_dilation[4], activation='tanh',
                               padding='causal', activity_regularizer=regularizers.l2(actreg))

        """
        pooler1 = MaxPooling1D(pooling_length[0], 1, padding='same')
        pooler2 = MaxPooling1D(pooling_length[1], 1, padding='same')
        pooler3 = MaxPooling1D(pooling_length[2], 1, padding='same')
        pooler4 = MaxPooling1D(pooling_length[3], 1, padding='same')
        pooler5 = MaxPooling1D(pooling_length[4], 1, padding='same')
        """
        skip_scaler1 = TimeDistributed(Dense(n_channels, activation='linear'))
        skip_scaler2 = TimeDistributed(Dense(n_channels, activation='linear'))
        skip_scaler3 = TimeDistributed(Dense(n_channels, activation='linear'))
        skip_scaler4 = TimeDistributed(Dense(n_channels, activation='linear'))
        skip_scaler5 = TimeDistributed(Dense(n_channels, activation='linear'))

        res_scaler1 = TimeDistributed(Dense(n_channels, activation='linear'))
        res_scaler2 = TimeDistributed(Dense(n_channels, activation='linear'))
        res_scaler3 = TimeDistributed(Dense(n_channels, activation='linear'))
        res_scaler4 = TimeDistributed(Dense(n_channels, activation='linear'))
        # res_scaler5 = TimeDistributed(Dense(n_channels, activation='linear'))

        post_scaler1 = TimeDistributed(Dense(n_channels, activation='linear'))
        post_scaler2 = TimeDistributed(Dense(n_channels, activation='linear'))
        post_scaler3 = TimeDistributed(Dense(n_channels, activation='linear'))
        post_scaler4 = TimeDistributed(Dense(n_channels, activation='linear'))
        post_scaler5 = TimeDistributed(Dense(n_channels, activation='linear'))

        summer = keras.layers.Add()
        multiplier = keras.layers.Multiply()
        # concatenator = keras.layers.Concatenate()

        do1 = Dropout(dropout_rate)
        do2 = Dropout(dropout_rate)
        do3 = Dropout(dropout_rate)
        do4 = Dropout(dropout_rate)
        do5 = Dropout(dropout_rate)

        # Create 5-layer WaveNet encoder
        l1_skip = skip_scaler1(do1(multiplier([encoder1(sequence1), encoder1_tanh(sequence1)])))
        l1_res = res_scaler1(l1_skip)
        l2_skip = skip_scaler2(do2(multiplier([encoder2(l1_res), encoder2_tanh(l1_res)])))
        l2_res = res_scaler2(summer([l1_res, l2_skip]))
        l3_skip = skip_scaler3(do3(multiplier([encoder3(l2_res), encoder3_tanh(l2_res)])))
        l3_res = res_scaler3(summer([l2_res, l3_skip]))
        l4_skip = skip_scaler4(do4(multiplier([encoder4(l3_res), encoder4_tanh(l3_res)])))
        l4_res = res_scaler4(summer([l3_res, l4_skip]))
        l5_skip = skip_scaler5(do5(multiplier([encoder5(l4_res), encoder5_tanh(l4_res)])))
        # l5_res = res_scaler5(summer([l4_res, l5_skip]))

        # Merge layers into postnet with addition
        # convstack_out = summer([l1_skip,l2_skip])
        # convstack_out = summer([convstack_out,l3_skip])
        # convstack_out = summer([convstack_out,l4_skip])
        # convstack_out = summer([convstack_out,l5_skip])
        convstack_out = summer([post_scaler1(l1_skip), post_scaler2(l2_skip)])
        convstack_out = summer([convstack_out, post_scaler3(l3_skip)])
        convstack_out = summer([convstack_out, post_scaler4(l4_skip)])
        convstack_out = summer([convstack_out, post_scaler5(l5_skip)])

        # Future predictions from current observations
        integrator = GRU(n_channels, return_sequences=False)(convstack_out)
        mapper = Dense(1, activation='relu')(integrator)
        model_ = Model(inputs=sequence1, outputs=mapper)

        return model_

    def build_training_data():
        """

        :return:
        """
        if not path.exists(r"resources\tensordata.mat"):
            if not path.exists(r"resources\logMel.npz"):
                print("Building logMels...")
                build_logMel_npz()

            print("Unpacking syllables and logMels...")
            # Form "filename: [syllables, log-Mel]" dict for the existing audio files
            data_dict = form_dict(root)

            list_of_log_mels = []
            syllables = []

            # Divide the dict into lists of log-Mel values and syllable values index-wise
            for key in data_dict.keys():
                syllables.append(data_dict[key][0])
                list_of_log_mels.append(data_dict[key][1])
            y = np.array(syllables)

            # Form Tensor
            print("Forming tensor...")
            x = form_tensor(list_of_log_mels)

            savemat(r"resources\tensordata.mat", {"tensor": x, "syll_train": y})

        else:
            print("Loading tensor and syllable data from memory...")
            mat_data = loadmat(r"resources\tensordata.mat")
            x = mat_data["tensor"]
            y = np.transpose(mat_data["syll_train"])

        return x, y

    # Get tensor and syllables for the audio segments
    tensor, syll_train = build_training_data()

    print("Tensor dimensions:", np.shape(tensor))
    print("Syllable dimensions:", np.shape(syll_train))

    # D = tensor.shape[2]
    # T = tensor.shape[1]
    N = tensor.shape[0]

    # Shuffle data (so that validation split also contains data from all languages)
    ord_ = np.arange(N)
    np.random.shuffle(ord_)
    tensor = tensor[ord_, :, :]
    syll_train = syll_train[ord_]

    # Get model
    model = wavenet_model(tensor, 64)

    # Compile the model
    model.compile(optimizer='adam', loss='mean_absolute_percentage_error',
                  metrics=[keras.metrics.MeanAbsoluteError(), keras.metrics.MeanAbsolutePercentageError()])

    # Train the model
    history = model.fit(tensor, syll_train,
                        epochs=20,
                        batch_size=32,
                        validation_split=0.2, )
    plot_model(history)

    return


run_RNN(r"resources\audio\**\*.wav")
