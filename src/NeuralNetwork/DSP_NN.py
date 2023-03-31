import keras.callbacks
import numpy as np
import matplotlib.pyplot as plt
from mat73 import loadmat
from config import*

# Keras
import tensorflow as tf
from keras import layers
from keras.layers import Input, TimeDistributed, Add, Multiply
from keras.layers import Dense, Dropout, Conv1D
from keras.metrics import MeanAbsoluteError, MeanAbsolutePercentageError
from keras import regularizers
from keras.models import Model
import keras.losses


def causal_attention_mask(batch_size, n_dest, n_src, dtype):

    """

    Mask the upper half of the dot product matrix in self attention.
    This prevents flow of information from future tokens to current token.
    1's in the lower triangle, counting from the lower right corner.
    """
    i = tf.range(n_dest)[:, None]
    j = tf.range(n_src)
    m = i >= j - n_src + n_dest
    mask = tf.cast(m, dtype)
    mask = tf.reshape(mask, [1, n_dest, n_src])
    mult = tf.concat(
        [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
    )
    return tf.tile(mask, mult)


class TransformerBlock(layers.Layer):

    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads, embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, *args, **kwargs):
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
        attention_output = self.att(inputs, inputs, attention_mask=causal_mask)
        attention_output = self.dropout1(attention_output)
        out1 = self.layernorm1(inputs + attention_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)


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

    summer = Add()
    multiplier = Multiply()
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
    integrator = TransformerBlock(n_channels, 8, n_channels)(convstack_out)
    integrator2 = Conv1D(1, X_in.shape[1])(integrator)
    mapper = Dense(1, activation='relu')(integrator2)
    model = Model(inputs=sequence1, outputs=mapper)

    #

    return model


def plot_model(model):
    """

    :param model:
    :return:
    """

    loss = model.history['loss']
    mean_abs_err = model.history['mean_absolute_error']
    mean_abs_pct_err = model.history['mean_absolute_percentage_error']

    val_loss = model.history['val_loss']
    val_mean_abs_err = model.history['val_mean_absolute_error']
    val_mean_abs_pct_err = model.history['val_mean_absolute_percentage_error']

    epochs = range(1, len(val_loss) + 1)

    plt.figure()
    plt.plot(epochs, loss, 'g', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.figure()
    plt.plot(epochs, mean_abs_err, 'g', label='Training loss')
    plt.plot(epochs, val_mean_abs_err, 'r', label='Validation loss')
    plt.title('MA and validation MA error')
    plt.legend()

    plt.figure()
    plt.plot(epochs, mean_abs_pct_err, 'g', label='Training loss')
    plt.plot(epochs, val_mean_abs_pct_err, 'r', label='Validation loss')
    plt.title('MA% and validation MA% error')
    plt.legend()

    plt.show()


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

    elif language == "english":

        mat_data = loadmat(english_tensordata_loc)

    else:

        mat_data = loadmat(tensordata_loc)

    test_tensor = mat_data["tensor"]

    if language == "english" or language == "estonian":
        print("Testing against true values\n")
        test_syll = np.transpose(mat_data["true_syllables"])
    else:
        print("Testing against thetaseg values\n")
        test_syll = np.transpose(mat_data["syllables"])

    test_tensor = test_tensor[test_syll != 0]
    test_syll = test_syll[test_syll != 0]

    # Find the indices of the NaN values in the tensor
    nan_indices = np.argwhere(np.isnan(test_tensor))

    # Remove the NaN values from the tensor and update the label array
    test_tensor = np.delete(test_tensor, nan_indices[:, 0], axis=0)
    test_syll = np.delete(test_syll, nan_indices[:, 0], axis=0)

    print("Tensor dimensions:", np.shape(np.array(test_tensor)))
    print("Syllable dimensions:", np.shape(np.array(test_syll)), "\n")

    syll_estimates = model.predict(test_tensor, batch_size=batch_size)

    mae = np.nanmean(np.abs(syll_estimates[:, 0, 0] - test_syll))
    mape = np.nanmean(np.abs(syll_estimates[:, 0, 0] - test_syll) / test_syll) * 100

    print(f"\nMeanAbsoluteError: {mae}")
    print(f"MeanAbsolutePercentageError: {mape}\n")

    return mae, mape


def run_WaveNet(tensor, syllables, epochs, batch_size, dims):
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
    model = wavenet_model(tensor, dims)

    # Compile the model
    model.compile(optimizer='adam', loss='mean_absolute_percentage_error',
                  metrics=[MeanAbsoluteError(), MeanAbsolutePercentageError()])

    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=r"resources/wavenet_model.h5",
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        )

    # Train the model
    history = model.fit(tensor, syllables,
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[earlystop, model_checkpoint_callback],
                        validation_split=0.3)

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

        if i % 1000 == 0:
            print(i, "Done")

    print("All Done")
    print("\n")

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
    # -----------------------------------------------------DATA---------------------------------------------------------
    # Load in the data from the .mat file
    if language == "estonian":

        data = loadmat(estonian_tensordata_loc)

    else:

        data = loadmat(english_tensordata_loc)

    tensor = data["tensor"]
    syllables = data["syllables"]
    true_syllables = data["true_syllables"]
    tensor = tensor[true_syllables != 0]
    syllables = syllables[true_syllables != 0]
    true_syllables = true_syllables[true_syllables != 0]

    # Find the indices of the NaN values in the tensor
    nan_indices = np.argwhere(np.isnan(tensor))

    # Remove the NaN values from the tensor and update the label array
    tensor = np.delete(tensor, nan_indices[:, 0], axis=0)
    syllables = np.delete(syllables, nan_indices[:, 0], axis=0)

    N = tensor.shape[0]

    ord_ = np.arange(N)
    np.random.shuffle(ord_)
    tensor = tensor[ord_, :, :]
    syllables = syllables[ord_]
    true_syllables = true_syllables[ord_]

    # Compile model
    model = wavenet_model(tensor, dims)
    model.compile(optimizer='adam', loss='mean_absolute_percentage_error', )
    model.save_weights(initial_weights)

    # --------------------------------------------------CROSS-FOLD------------------------------------------------------
    index_list = list(range(N))
    fold_maes_dsp = []
    fold_mapes_dsp = []
    fold_maes_true = []
    fold_mapes_true = []

    for i in range(10):

        print(f"Fold: {i} For: {language}")

        # Derive data for the fold.
        train_index = index_list[0:int(0.9 * len(index_list))]
        test_index = index_list[int(0.9 * len(index_list)) + 1:]
        np.roll(index_list, -int(len(index_list) * 0.1))

        X_train = tensor[train_index]
        X_test = tensor[test_index]
        y_train_dsp = syllables[train_index]
        y_train_true = true_syllables[train_index]
        true_test = true_syllables[test_index]

        # Reset model for training.
        model.load_weights(initial_weights)

        # Train the model with dsp labels and predict syllables.
        print("With DSP labels for")
        model.fit(X_train, y_train_dsp, epochs=5, batch_size=32)
        pred_dsp = model.predict(X_test)

        # MAE and MAPE for dsp-label trained model.
        fold_maes_dsp.append(np.mean(np.abs(true_test - pred_dsp)))
        fold_mapes_dsp.append(np.mean(np.abs((true_test - pred_dsp) / true_test) * 100))

        # Reset model for training.
        model.load_weights(initial_weights)

        # Train the model with true labels and predict syllables.
        print("With true labels")
        model.fit(X_train, y_train_true, epochs=5, batch_size=32)
        pred_true = model.predict(X_test)

        # MAE and MAPE for true-label trained model.
        fold_maes_true.append(np.mean(np.abs(true_test - pred_true)))
        fold_mapes_true.append(np.mean(np.abs((true_test - pred_true) / true_test) * 100))

    mean_mae_True = np.mean([score for score in fold_maes_true])
    mean_mape_True = np.mean([score for score in fold_mapes_true])
    mean_mae_DSP = np.mean([score for score in fold_maes_dsp])
    mean_mape_DSP = np.mean([score for score in fold_mapes_dsp])

    print(f"Overall True cross-validation score for {language}: MAE={mean_mae_True}, MAPE={mean_mape_True}")
    print(f"Overall DSP cross-validation score for {language}: MAE={mean_mae_DSP}, MAPE={mean_mape_DSP}")

    return mean_mae_True, mean_mape_True, mean_mae_DSP, mean_mape_DSP

