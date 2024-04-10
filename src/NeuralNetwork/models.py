"""
Created on Tue Feb  7 09:24:33 2023

@author: rasaneno

Purpose:

Defines the architecture of the neural network.
Based on https://github.com/shreyas253/SylNet, https://ieeexplore.ieee.org/document/8769944.
Contains two different models. One more basic and the other more akin to SylNet.
"""

# Keras
import tensorflow as tf
from keras import layers
import keras.losses
from keras.layers import Dense, Dropout, Input, TimeDistributed, Conv1D, MaxPooling1D
from keras import regularizers
from keras.models import Model
import keras.callbacks


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
    # Define WaveNet encoder model
    conv_length = [2, 2, 2, 2, 2]
    # pooling_length = [1,1,1,1,1]
    conv_dilation = [1, 2, 4, 8, 16]
    actreg = 0.0000000001
    dropout_rate = 0.05

    sequence1 = Input(shape=(X_in.shape[1:]))
    encoder1 = Conv1D(n_channels,conv_length[0], dilation_rate=conv_dilation[0], activation='sigmoid',padding='causal', activity_regularizer=regularizers.l2(actreg))
    encoder2 = Conv1D(n_channels,conv_length[1], dilation_rate=conv_dilation[1], activation='sigmoid',padding='causal', activity_regularizer=regularizers.l2(actreg))
    encoder3 = Conv1D(n_channels,conv_length[2], dilation_rate=conv_dilation[2], activation='sigmoid',padding='causal', activity_regularizer=regularizers.l2(actreg))
    encoder4 = Conv1D(n_channels,conv_length[3], dilation_rate=conv_dilation[3], activation='sigmoid',padding='causal', activity_regularizer=regularizers.l2(actreg))
    encoder5 = Conv1D(n_channels,conv_length[4], dilation_rate=conv_dilation[4], activation='sigmoid',padding='causal', activity_regularizer=regularizers.l2(actreg))

    encoder1_tanh = Conv1D(n_channels, conv_length[0], dilation_rate=conv_dilation[0], activation='tanh', padding='causal', activity_regularizer=regularizers.l2(actreg))
    encoder2_tanh = Conv1D(n_channels, conv_length[1], dilation_rate=conv_dilation[1], activation='tanh', padding='causal', activity_regularizer=regularizers.l2(actreg))
    encoder3_tanh = Conv1D(n_channels, conv_length[2], dilation_rate=conv_dilation[2], activation='tanh', padding='causal', activity_regularizer=regularizers.l2(actreg))
    encoder4_tanh = Conv1D(n_channels, conv_length[3], dilation_rate=conv_dilation[3], activation='tanh', padding='causal', activity_regularizer=regularizers.l2(actreg))
    encoder5_tanh = Conv1D(n_channels, conv_length[4], dilation_rate=conv_dilation[4], activation='tanh', padding='causal', activity_regularizer=regularizers.l2(actreg))

    # pooler1 = MaxPooling1D(pooling_length[0],1,padding='same')
    # pooler2 = MaxPooling1D(pooling_length[1],1,padding='same')
    # pooler3 = MaxPooling1D(pooling_length[2],1,padding='same')
    # pooler4 = MaxPooling1D(pooling_length[3],1,padding='same')
    # pooler5 = MaxPooling1D(pooling_length[4],1,padding='same')

    skip_scaler1 = TimeDistributed(Dense(n_channels, activation='linear'))
    skip_scaler2 = TimeDistributed(Dense(n_channels, activation='linear'))
    skip_scaler3 = TimeDistributed(Dense(n_channels, activation='linear'))
    skip_scaler4 = TimeDistributed(Dense(n_channels, activation='linear'))
    skip_scaler5 = TimeDistributed(Dense(n_channels, activation='linear'))

    res_scaler1 = TimeDistributed(Dense(n_channels, activation='linear'))
    res_scaler2 = TimeDistributed(Dense(n_channels, activation='linear'))
    res_scaler3 = TimeDistributed(Dense(n_channels, activation='linear'))
    res_scaler4 = TimeDistributed(Dense(n_channels, activation='linear'))
    # res_scaler5 =  TimeDistributed(Dense(n_channels,activation='linear'))

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

    l1_skip = skip_scaler1(do1(multiplier([encoder1(sequence1),encoder1_tanh(sequence1)])))
    l1_res = res_scaler1(l1_skip)
    l2_skip = skip_scaler2(do2(multiplier([encoder2(l1_res),encoder2_tanh(l1_res)])))
    l2_res = res_scaler2(summer([l1_res,l2_skip]))
    l3_skip = skip_scaler3(do3(multiplier([encoder3(l2_res),encoder3_tanh(l2_res)])))
    l3_res = res_scaler3(summer([l2_res,l3_skip]))
    l4_skip = skip_scaler4(do4(multiplier([encoder4(l3_res),encoder4_tanh(l3_res)])))
    l4_res = res_scaler4(summer([l3_res,l4_skip]))
    l5_skip = skip_scaler5(do5(multiplier([encoder5(l4_res),encoder5_tanh(l4_res)])))
    # l5_res = res_scaler5(summer([l4_res,l5_skip]))

    # Merge layers into postnet with addition
    convstack_out = summer([post_scaler1(l1_skip),post_scaler2(l2_skip)])
    convstack_out = summer([convstack_out,post_scaler3(l3_skip)])
    convstack_out = summer([convstack_out,post_scaler4(l4_skip)])
    convstack_out = summer([convstack_out,post_scaler5(l5_skip)])

    integrator = TransformerBlock(n_channels, 8, n_channels)(convstack_out)
    integrator2 = Conv1D(1,X_in.shape[1])(integrator)
    mapper = Dense(1,activation='sigmoid')(integrator2)
    model = Model(inputs=sequence1,outputs=mapper)

    return model


def sylnet_model(X_in,n_channels=128,dropout_rate=0.5):
    # Define WaveNet encoder model
    conv_length = [5,5,5,5,5,5,5,5,5]
    # pooling_length = [1,1,1,1,1]
    # conv_dilation = [1,2,4,8,16,32,64,128,256]
    conv_dilation = [1,2,4,6,8,12,16,32,64]
    actreg = 0.0000000001

    sequence1 = Input(shape=(X_in.shape[1:]))
    encoder1 = Conv1D(n_channels,conv_length[0],dilation_rate=conv_dilation[0],activation='sigmoid',padding='causal',activity_regularizer=regularizers.l2(actreg))
    encoder2 = Conv1D(n_channels,conv_length[1],dilation_rate=conv_dilation[1],activation='sigmoid',padding='causal',activity_regularizer=regularizers.l2(actreg))
    encoder3 = Conv1D(n_channels,conv_length[2],dilation_rate=conv_dilation[2],activation='sigmoid',padding='causal',activity_regularizer=regularizers.l2(actreg))
    encoder4 = Conv1D(n_channels,conv_length[3],dilation_rate=conv_dilation[3],activation='sigmoid',padding='causal',activity_regularizer=regularizers.l2(actreg))
    encoder5 = Conv1D(n_channels,conv_length[4],dilation_rate=conv_dilation[4],activation='sigmoid',padding='causal',activity_regularizer=regularizers.l2(actreg))
    encoder6 = Conv1D(n_channels,conv_length[5],dilation_rate=conv_dilation[5],activation='sigmoid',padding='causal',activity_regularizer=regularizers.l2(actreg))
    encoder7 = Conv1D(n_channels,conv_length[6],dilation_rate=conv_dilation[6],activation='sigmoid',padding='causal',activity_regularizer=regularizers.l2(actreg))
    encoder8 = Conv1D(n_channels,conv_length[7],dilation_rate=conv_dilation[7],activation='sigmoid',padding='causal',activity_regularizer=regularizers.l2(actreg))

    encoder1_tanh = Conv1D(n_channels,conv_length[0],dilation_rate=conv_dilation[0],activation='tanh',padding='causal',activity_regularizer=regularizers.l2(actreg))
    encoder2_tanh = Conv1D(n_channels,conv_length[1],dilation_rate=conv_dilation[1],activation='tanh',padding='causal',activity_regularizer=regularizers.l2(actreg))
    encoder3_tanh = Conv1D(n_channels,conv_length[2],dilation_rate=conv_dilation[2],activation='tanh',padding='causal',activity_regularizer=regularizers.l2(actreg))
    encoder4_tanh = Conv1D(n_channels,conv_length[3],dilation_rate=conv_dilation[3],activation='tanh',padding='causal',activity_regularizer=regularizers.l2(actreg))
    encoder5_tanh = Conv1D(n_channels,conv_length[4],dilation_rate=conv_dilation[4],activation='tanh',padding='causal',activity_regularizer=regularizers.l2(actreg))
    encoder6_tanh = Conv1D(n_channels,conv_length[5],dilation_rate=conv_dilation[5],activation='tanh',padding='causal',activity_regularizer=regularizers.l2(actreg))
    encoder7_tanh = Conv1D(n_channels,conv_length[6],dilation_rate=conv_dilation[6],activation='tanh',padding='causal',activity_regularizer=regularizers.l2(actreg))
    encoder8_tanh = Conv1D(n_channels,conv_length[7],dilation_rate=conv_dilation[7],activation='tanh',padding='causal',activity_regularizer=regularizers.l2(actreg))
    skip_scaler1 = TimeDistributed(Dense(n_channels,activation='linear'))
    skip_scaler2 = TimeDistributed(Dense(n_channels,activation='linear'))
    skip_scaler3 = TimeDistributed(Dense(n_channels,activation='linear'))
    skip_scaler4 = TimeDistributed(Dense(n_channels,activation='linear'))
    skip_scaler5 = TimeDistributed(Dense(n_channels,activation='linear'))
    skip_scaler6 = TimeDistributed(Dense(n_channels,activation='linear'))
    skip_scaler7 = TimeDistributed(Dense(n_channels,activation='linear'))
    skip_scaler8 = TimeDistributed(Dense(n_channels,activation='linear'))

    res_scaler1 = TimeDistributed(Dense(n_channels,activation='linear'))
    res_scaler2 = TimeDistributed(Dense(n_channels,activation='linear'))
    res_scaler3 = TimeDistributed(Dense(n_channels,activation='linear'))
    res_scaler4 = TimeDistributed(Dense(n_channels,activation='linear'))
    res_scaler5 = TimeDistributed(Dense(n_channels,activation='linear'))
    res_scaler6 = TimeDistributed(Dense(n_channels,activation='linear'))
    res_scaler7 = TimeDistributed(Dense(n_channels,activation='linear'))

    summer = keras.layers.Add()
    multiplier = keras.layers.Multiply()

    do1 = Dropout(dropout_rate)
    do2 = Dropout(dropout_rate)
    do3 = Dropout(dropout_rate)
    do4 = Dropout(dropout_rate)
    do5 = Dropout(dropout_rate)
    do6 = Dropout(dropout_rate)
    do7 = Dropout(dropout_rate)
    do8 = Dropout(dropout_rate)

    l1_act = do1(multiplier([encoder1(sequence1),encoder1_tanh(sequence1)]))
    l1_skip = skip_scaler1(l1_act)
    l1_res = res_scaler1(l1_act)
    l2_act = do2(multiplier([encoder2(l1_res),encoder2_tanh(l1_res)]))
    l2_skip = skip_scaler2(l2_act)
    l2_res = (summer([l1_res,res_scaler2(l2_act)]))
    l3_act = do3(multiplier([encoder3(l2_res),encoder3_tanh(l2_res)]))
    l3_skip = skip_scaler3(l3_act)
    l3_res = (summer([l2_res,res_scaler3(l3_act)]))
    l4_act = do4(multiplier([encoder4(l3_res),encoder4_tanh(l3_res)]))
    l4_skip = skip_scaler4(l4_act)
    l4_res = (summer([l3_res,res_scaler4(l4_act)]))
    l5_act = do5(multiplier([encoder5(l4_res),encoder5_tanh(l4_res)]))
    l5_skip = skip_scaler5(l5_act)
    l5_res = (summer([l4_res,res_scaler5(l5_act)]))
    l6_act = do6(multiplier([encoder6(l5_res),encoder6_tanh(l5_res)]))
    l6_skip = skip_scaler6(l6_act)
    l6_res = (summer([l5_res,res_scaler6(l6_act)]))
    l7_act = do7(multiplier([encoder7(l6_res),encoder7_tanh(l6_res)]))
    l7_skip = skip_scaler7(l7_act)
    l7_res = (summer([l6_res,res_scaler7(l7_act)]))
    l8_act = do8(multiplier([encoder8(l7_res),encoder8_tanh(l7_res)]))
    l8_skip = skip_scaler8(l8_act)
    # l5_res = res_scaler5(summer([l4_res,l5_skip]))

    # Merge layers into postnet with addition
    convstack_out = summer([l1_skip, l2_skip])
    convstack_out = summer([convstack_out, l3_skip])
    convstack_out = summer([convstack_out, l4_skip])
    convstack_out = summer([convstack_out, l5_skip])
    convstack_out = summer([convstack_out, l6_skip])
    convstack_out = summer([convstack_out, l7_skip])
    convstack_out = summer([convstack_out, l8_skip])

    integrator = Conv1D(n_channels,5,activation='relu',padding='causal')(convstack_out)
    # integrator2 = LSTM(n_channels,return_sequences=False)(integrator)
    integrator2 = TransformerBlock(n_channels,8,n_channels)(integrator)
    mapper = Dense(1,activation='relu')(integrator2)
    mapper2 = MaxPooling1D(X_in.shape[1])(mapper)

    model = Model(inputs=sequence1,outputs=mapper2)
    #
    return model
