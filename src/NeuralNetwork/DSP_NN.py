"""
Created on Wed Apr 10 16:30:00 2024

@author: GronlunE

Purpose:

Contains functions used to train the models from models.py
"""

import numpy as np
from scipy.io import loadmat
from config import*
import pandas as pd

# Keras
import tensorflow as tf
from keras.metrics import MeanAbsoluteError, MeanAbsolutePercentageError
from models import wavenet_model, sylnet_model, TransformerBlock


def run_prediction(model, batch_size, language):
    """

    :param language:
    :param batch_size:
    :param model:
    :return:
    """

    print("\nLoading testing data...", "\n")

    if language == "estonian":

        mat_data = loadmat(estonian_tensordata_loc)
        df = pd.read_csv(estonian_csv_loc)
        audio_durations = df['Audio duration'].to_numpy()
        test_tensor = mat_data["tensor"]

    elif language == "english":

        mat_data = loadmat(english_tensordata_loc)
        df = pd.read_csv(english_csv_loc)
        audio_durations = df['Audio duration'].to_numpy()
        test_tensor = mat_data["tensor"]

    else:

        mat_data = loadmat(tensordata_loc)
        df = pd.read_csv(test_csv_loc)
        audio_durations = df['Audio duration'].to_numpy()
        test_tensor = mat_data["tensor"]

    if language == "english":
        print(f"Testing against true {language} values\n")
        test_syll = np.transpose(mat_data["true_syllables"])

    elif language == "estonian":
        print(f"Testing against true {language} values\n")
        test_syll = np.transpose(mat_data["true_syllables"])

    else:
        print("Testing against thetaseg values\n")
        test_syll = np.transpose(mat_data["syllables"])

    # Find indices of tensors with duration less than or equal to T seconds
    T = test_tensor.shape[1] / 100
    valid_indices = np.where(audio_durations <= T)[0]

    # Filter to only keep valid indices
    test_tensor = test_tensor[valid_indices, :, :]
    test_syll = test_syll[valid_indices]

    valid_indices = np.where(test_syll != 0)[0]

    # Remove any zeros from the annotations
    test_tensor = test_tensor[valid_indices, :, :]
    test_syll = test_syll[valid_indices]

    # Find the indices of the NaN values in the tensor
    nan_indices = np.argwhere(np.isnan(test_tensor))

    # Create a mask for the non-NaN values in the tensor
    mask = np.ones(test_tensor.shape[0], dtype=bool)
    mask[nan_indices[:, 0]] = False

    # Remove the NaN values from the tensor and update the label array
    test_tensor = test_tensor[mask]
    test_syll = test_syll[mask]

    print("Tensor dimensions:", np.shape(np.array(test_tensor)))
    print("Syllable dimensions:", np.shape(np.array(test_syll)), "\n")

    syll_estimates = model.predict(test_tensor, batch_size=batch_size)

    mae = np.nanmean(np.abs(syll_estimates[:, 0, 0] - test_syll))
    mape = np.nanmean(np.abs(syll_estimates[:, 0, 0] - test_syll) / test_syll) * 100

    print(f"\nMeanAbsoluteError: {mae}")
    print(f"MeanAbsolutePercentageError: {mape}\n")

    return mae, mape


def train_NeuralNet(tensor, syllables, epochs, batch_size, dims):
    """

    :param tensor:
    :param syllables:
    :param epochs:
    :param batch_size:
    :param dims:
    :return:
    """

    tensor[tensor == -np.inf] = 20*np.log10(eps)
    print("Tensor dimensions:", np.shape(tensor))
    print("Syllable dimensions:", np.shape(syllables))

    # strategy = tf.distribute.Strategy()
    # print("Number of devices: {}".format(strategy.num_replicas_in_sync))
    model = sylnet_model(tensor, dims)

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='mean_absolute_percentage_error',
        metrics=[MeanAbsoluteError(), MeanAbsolutePercentageError()]
    )

    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_mean_absolute_percentage_error', patience=10)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=r"resources/most_recent_model",
        monitor='val_mean_absolute_percentage_error',
        mode='min',
        save_best_only=True,
        custom_objects={'TransformerBlock': TransformerBlock}
        )

    # Train the model
    history = model.fit(tensor, syllables,
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[earlystop, model_checkpoint_callback],
                        validation_split = 0.3)

    return model, history


def set_baseline(language):

    if language == "estonian":

        data = loadmat(estonian_tensordata_loc)

    else:

        data = loadmat(english_tensordata_loc)

    syllables = data["syllables"]
    true_syllables = data["true_syllables"]
    true_syllables[true_syllables == 0] = 1

    MAE = []
    MAPE = []

    for i in range(np.size(syllables)):
        testSyllables = true_syllables[i]
        thetaSyllables = syllables[i]

        MAE.append(np.absolute(testSyllables-thetaSyllables))
        MAPE.append(np.absolute(((testSyllables-thetaSyllables)/testSyllables)*100))

    MAE = np.mean(np.array(MAE))
    MAPE = np.mean(np.array(MAPE))

    print(f"Baseline to beat for: {language}\n")
    print(f"MAE: {MAE}")
    print(f"MAPE: {MAPE}")
    print("\n")

    return MAE, MAPE


def run_cross_validation(language, dims=32):
    """

    :param language:
    :param dims:
    :return:
    """
    # -------------------------------------------------DATA PROCESSING--------------------------------------------------
    # Load in the data from the .mat file according to language use.
    if language == "estonian":

        data = loadmat(estonian_tensordata_loc)

    else:

        data = loadmat(english_tensordata_loc)

    tensor = data["tensor"]
    N = tensor.shape[0]
    syllables = data["syllables"]
    true_syllables = data["true_syllables"]
    tensor = tensor[true_syllables != 0]
    syllables = syllables[true_syllables != 0]
    true_syllables = true_syllables[true_syllables != 0]
    print(N)
    tensor = tensor[syllables != 0]
    syllables = syllables[syllables != 0]
    true_syllables = true_syllables[syllables != 0]
    print(N)
    # Find the indices of the NaN values in the tensor
    nan_indices = np.argwhere(np.isnan(tensor))
    print(nan_indices)

    # Remove the NaN values from the tensor and update the syllables array
    tensor = np.delete(tensor, nan_indices[:, 0], axis=0)
    syllables = np.delete(syllables, nan_indices[:, 0], axis=0)
    true_syllables = np.delete(true_syllables, nan_indices[:, 0], axis=0)

    N = tensor.shape[0]
    print(N)

    ord_ = np.arange(N)
    np.random.shuffle(ord_)
    tensor = tensor[ord_, :, :]
    syllables = syllables[ord_]
    true_syllables = true_syllables[ord_]

    # Compile model
    model = sylnet_model(tensor, dims)

    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=r"resources/wavenet_model.h5",
        monitor='val_loss',
        mode='min',
        save_best_only=True,
    )
    # --------------------------------------------------CROSS-FOLD------------------------------------------------------
    index_list = list(range(N))
    fold_maes_dsp = []
    fold_mapes_dsp = []
    fold_maes_true = []
    fold_mapes_true = []

    for i in range(1):

        print(f"Fold: {i} For: {language}")

        # Derive data for the fold.
        train_index = index_list[0:int(0.6 * len(index_list))]
        test_index = index_list[int(0.6 * len(index_list)):]
        index_list = np.roll(index_list, -int(len(index_list) * 0.1))

        tensor_train = tensor[train_index]
        tensor_test = tensor[test_index]
        syll_train_dsp = syllables[train_index]
        syll_train_true = true_syllables[train_index]
        syll_test_true = true_syllables[test_index]

        # Train the model with dsp labels and predict syllables.
        print("With DSP labels for")

        # Reset model for training
        model.compile(optimizer='adam', loss='mean_absolute_percentage_error',
                      metrics=[MeanAbsoluteError(), MeanAbsolutePercentageError()])

        # Train the model
        model.fit(tensor, true_syllables,
                  epochs=100,
                  batch_size=32,
                  callbacks=[earlystop, model_checkpoint_callback],
                  validation_split=0.3)
        syll_pred_dsp = model.predict(tensor, batch_size=32)

        # MAE and MAPE for dsp-label trained model.
        fold_maes_dsp.append(np.mean(np.abs(syll_test_true - syll_pred_dsp)))
        fold_mapes_dsp.append(np.mean(np.abs((syll_test_true - syll_pred_dsp) / syll_test_true) * 100))

        # Reset model for training.
        model.load_weights(initial_weights)
        model.compile(optimizer='adam', loss='mean_absolute_percentage_error',
                      metrics=[MeanAbsoluteError(), MeanAbsolutePercentageError()])

        # Train the model with true labels and predict syllables.
        print("With true labels")
        model.fit(tensor, true_syllables,
                  epochs=100,
                  batch_size=32,
                  callbacks=[earlystop, model_checkpoint_callback],
                  validation_split=0.3)
        syll_pred_true = model.predict(tensor, batch_size=32)

        # MAE and MAPE for true-label trained model.
        fold_maes_true.append(np.mean(np.abs(syll_test_true - syll_pred_true)))
        fold_mapes_true.append(np.mean(np.abs((syll_test_true - syll_pred_true) / syll_test_true) * 100))

    mean_mae_True = np.mean([score for score in fold_maes_true])
    mean_mape_True = np.mean([score for score in fold_mapes_true])
    mean_mae_DSP = np.mean([score for score in fold_maes_dsp])
    mean_mape_DSP = np.mean([score for score in fold_mapes_dsp])

    print(f"Overall True cross-fold validation score for {language}: MAE={mean_mae_True}, MAPE={mean_mape_True}")
    print(f"Overall DSP cross-fold validation score for {language}: MAE={mean_mae_DSP}, MAPE={mean_mape_DSP}")

    return mean_mae_True, mean_mape_True, mean_mae_DSP, mean_mape_DSP

