"""
models_dl.py
============
Deep-learning model builders.

Every builder follows the same signature:

    build_<name>(input_shape, num_classes_device, num_classes_attack=None)
        → compiled keras.Model

Rules
-----
* input_shape  = (n_features, 1)   — 3-D tensor expected by sequential models
* Single-output : num_classes_attack=None  → one softmax head named 'output'
* Multi-output  : num_classes_attack≠None  → two heads named 'device' & 'attack'
* All models are compiled with Adam + sparse_categorical_crossentropy,
  matching the original 1D-CNN approach.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Flatten, Reshape,
    Conv1D, MaxPooling1D, AveragePooling1D, GlobalAveragePooling1D,
    LSTM, Bidirectional, GRU, SimpleRNN,
    BatchNormalization, Add, Activation,
    Lambda,
)
from tensorflow.keras import regularizers


# ═══════════════════════════════════════════════════════════════════════════
# Shared utilities
# ═══════════════════════════════════════════════════════════════════════════

def _compile_and_return(inputs, shared_out, num_classes_device,
                        num_classes_attack):
    """
    Attach output head(s) and compile the model.

    Parameters
    ----------
    inputs             : keras Input tensor
    shared_out         : last shared tensor before the output head(s)
    num_classes_device : number of device-type classes
    num_classes_attack : number of attack-category classes, or None
    """
    if num_classes_attack is not None:
        device_out = Dense(num_classes_device, activation='softmax',
                           name='device')(shared_out)
        attack_out = Dense(num_classes_attack, activation='softmax',
                           name='attack')(shared_out)
        model = Model(inputs=inputs, outputs=[device_out, attack_out])
        model.compile(
            optimizer='adam',
            loss={
                'device': 'sparse_categorical_crossentropy',
                'attack': 'sparse_categorical_crossentropy',
            },
            metrics={'device': 'accuracy', 'attack': 'accuracy'},
        )
    else:
        out   = Dense(num_classes_device, activation='softmax',
                      name='output')(shared_out)
        model = Model(inputs=inputs, outputs=out)
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
    return model


# ═══════════════════════════════════════════════════════════════════════════
# 1.  1D-CNN  (original, kept unchanged)
# ═══════════════════════════════════════════════════════════════════════════

def build_1dcnn(input_shape, num_classes_device, num_classes_attack=None):
    """
    Two-block 1-D Convolutional Neural Network.
    Conv(64) → Pool → Conv(128) → Pool → Flatten → Dense(128) → Dropout(0.5)
    """
    inputs = Input(shape=input_shape)
    x = Conv1D(64,  kernel_size=3, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    return _compile_and_return(inputs, x, num_classes_device, num_classes_attack)


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Autoencoder  (classification via latent representation)
# ═══════════════════════════════════════════════════════════════════════════

def build_autoencoder(input_shape, num_classes_device, num_classes_attack=None):
    """
    Encoder–Decoder Autoencoder used for classification.

    The encoder compresses the input to a latent space; the decoder
    reconstructs the input (auxiliary reconstruction loss).  The latent
    vector is also fed to classification head(s).

    Architecture
    ------------
    Input → Flatten → Encoder (Dense 128→64→32) → Decoder (Dense 64→128)
    → Reconstruction output  +  Classification head(s) from latent space
    """
    n_features = input_shape[0]          # (n_features, 1)

    inputs = Input(shape=input_shape)
    x = Flatten()(inputs)

    # Encoder
    enc = Dense(128, activation='relu')(x)
    enc = Dropout(0.3)(enc)
    enc = Dense(64,  activation='relu')(enc)
    enc = Dropout(0.3)(enc)
    latent = Dense(32, activation='relu', name='latent')(enc)

    # Decoder (reconstruction branch — for regularisation)
    dec = Dense(64,  activation='relu')(latent)
    dec = Dense(128, activation='relu')(dec)
    reconstruction = Dense(n_features, activation='linear',
                           name='reconstruction')(dec)

    # Classification head(s) built from latent
    cls = Dense(64, activation='relu')(latent)
    cls = Dropout(0.3)(cls)

    if num_classes_attack is not None:
        device_out = Dense(num_classes_device, activation='softmax',
                           name='device')(cls)
        attack_out = Dense(num_classes_attack, activation='softmax',
                           name='attack')(cls)
        model = Model(inputs=inputs,
                      outputs=[device_out, attack_out, reconstruction])
        model.compile(
            optimizer='adam',
            loss={
                'device':         'sparse_categorical_crossentropy',
                'attack':         'sparse_categorical_crossentropy',
                'reconstruction': 'mse',
            },
            loss_weights={'device': 1.0, 'attack': 1.0, 'reconstruction': 0.1},
            metrics={'device': 'accuracy', 'attack': 'accuracy'},
        )
    else:
        out = Dense(num_classes_device, activation='softmax',
                    name='output')(cls)
        model = Model(inputs=inputs, outputs=[out, reconstruction])
        model.compile(
            optimizer='adam',
            loss={'output': 'sparse_categorical_crossentropy',
                  'reconstruction': 'mse'},
            loss_weights={'output': 1.0, 'reconstruction': 0.1},
            metrics={'output': 'accuracy'},
        )
    return model


# ═══════════════════════════════════════════════════════════════════════════
# 3.  LSTM
# ═══════════════════════════════════════════════════════════════════════════

def build_lstm(input_shape, num_classes_device, num_classes_attack=None):
    """
    Stacked LSTM.
    LSTM(64, return_sequences) → LSTM(64) → Dense(64) → Dropout
    """
    inputs = Input(shape=input_shape)
    x = LSTM(64, return_sequences=True)(inputs)
    x = Dropout(0.3)(x)
    x = LSTM(64)(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    return _compile_and_return(inputs, x, num_classes_device, num_classes_attack)


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Bidirectional LSTM
# ═══════════════════════════════════════════════════════════════════════════

def build_bilstm(input_shape, num_classes_device, num_classes_attack=None):
    """
    Bidirectional LSTM.
    BiLSTM(64, return_sequences) → BiLSTM(64) → Dense(64) → Dropout
    """
    inputs = Input(shape=input_shape)
    x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(64))(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    return _compile_and_return(inputs, x, num_classes_device, num_classes_attack)


# ═══════════════════════════════════════════════════════════════════════════
# 5.  GRU
# ═══════════════════════════════════════════════════════════════════════════

def build_gru(input_shape, num_classes_device, num_classes_attack=None):
    """
    Stacked GRU.
    GRU(64, return_sequences) → GRU(64) → Dense(64) → Dropout
    """
    inputs = Input(shape=input_shape)
    x = GRU(64, return_sequences=True)(inputs)
    x = Dropout(0.3)(x)
    x = GRU(64)(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    return _compile_and_return(inputs, x, num_classes_device, num_classes_attack)


# ═══════════════════════════════════════════════════════════════════════════
# 6.  CNN-GRU
# ═══════════════════════════════════════════════════════════════════════════

def build_cnn_gru(input_shape, num_classes_device, num_classes_attack=None):
    """
    Hybrid CNN + GRU.
    Conv(64) → Pool → Conv(128) → Pool → GRU(64) → Dense(64) → Dropout
    """
    inputs = Input(shape=input_shape)
    x = Conv1D(64,  kernel_size=3, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = GRU(64)(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    return _compile_and_return(inputs, x, num_classes_device, num_classes_attack)


# ═══════════════════════════════════════════════════════════════════════════
# 7.  CNN-LSTM
# ═══════════════════════════════════════════════════════════════════════════

def build_cnn_lstm(input_shape, num_classes_device, num_classes_attack=None):
    """
    Hybrid CNN + LSTM.
    Conv(64) → Pool → Conv(128) → Pool → LSTM(64) → Dense(64) → Dropout
    """
    inputs = Input(shape=input_shape)
    x = Conv1D(64,  kernel_size=3, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = LSTM(64)(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    return _compile_and_return(inputs, x, num_classes_device, num_classes_attack)


# ═══════════════════════════════════════════════════════════════════════════
# 8.  MLP
# ═══════════════════════════════════════════════════════════════════════════

def build_mlp(input_shape, num_classes_device, num_classes_attack=None):
    """
    Multi-Layer Perceptron.
    Flatten → Dense(256) → BN → Dropout → Dense(128) → BN → Dropout → Dense(64)
    """
    inputs = Input(shape=input_shape)
    x = Flatten()(inputs)
    for units in (256, 128, 64):
        x = Dense(units, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
    return _compile_and_return(inputs, x, num_classes_device, num_classes_attack)


# ═══════════════════════════════════════════════════════════════════════════
# 9.  ResNet1D
# ═══════════════════════════════════════════════════════════════════════════

def _residual_block(x, filters, kernel_size=3):
    """One residual block: two Conv → BN → ReLU layers with a skip connection."""
    shortcut = x
    x = Conv1D(filters, kernel_size=kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(filters, kernel_size=kernel_size, padding='same')(x)
    x = BatchNormalization()(x)

    # Match channel dimension if needed
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, kernel_size=1, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


def build_resnet1d(input_shape, num_classes_device, num_classes_attack=None):
    """
    1-D Residual Network.
    Initial Conv → ResBlock(64) → ResBlock(128) → ResBlock(128)
    → GlobalAvgPool → Dense(128) → Dropout
    """
    inputs = Input(shape=input_shape)
    x = Conv1D(64, kernel_size=7, padding='same', activation='relu')(inputs)
    x = MaxPooling1D(pool_size=2)(x)

    x = _residual_block(x, filters=64)
    x = MaxPooling1D(pool_size=2)(x)

    x = _residual_block(x, filters=128)
    x = _residual_block(x, filters=128)
    x = GlobalAveragePooling1D()(x)

    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    return _compile_and_return(inputs, x, num_classes_device, num_classes_attack)


# ═══════════════════════════════════════════════════════════════════════════
# 10. Simple RNN
# ═══════════════════════════════════════════════════════════════════════════

def build_rnn(input_shape, num_classes_device, num_classes_attack=None):
    """
    Stacked SimpleRNN.
    RNN(64, return_sequences) → RNN(64) → Dense(64) → Dropout
    """
    inputs = Input(shape=input_shape)
    x = SimpleRNN(64, return_sequences=True)(inputs)
    x = Dropout(0.3)(x)
    x = SimpleRNN(64)(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    return _compile_and_return(inputs, x, num_classes_device, num_classes_attack)


# ═══════════════════════════════════════════════════════════════════════════
# 11. Echo State Network  (ESN / Reservoir Computing)
# ═══════════════════════════════════════════════════════════════════════════
#
# Keras does not provide a built-in reservoir layer, so we implement a
# minimal but complete ESN using a custom Keras layer.
# The reservoir weights are random and *not* trained (trainable=False).
# Only the read-out Dense layer is trained.
# ═══════════════════════════════════════════════════════════════════════════

class ReservoirLayer(tf.keras.layers.Layer):
    """
    Echo-State Network reservoir layer.

    For each time step t the reservoir state is updated as:
        h_t = tanh( W_in · x_t  +  W_res · h_{t-1} )

    W_in  and W_res are drawn randomly and kept fixed (trainable=False).
    W_res is scaled so that its spectral radius ≈ `spectral_radius`.

    The time-step loop is wrapped with @tf.function so Keras graph-mode
    tracing works correctly.
    """

    def __init__(self, units: int = 128, spectral_radius: float = 0.9,
                 sparsity: float = 0.1, seed: int = 42, **kwargs):
        super().__init__(**kwargs)
        self.units           = units
        self.spectral_radius = spectral_radius
        self.sparsity        = sparsity
        self.seed            = seed

    # ------------------------------------------------------------------
    def build(self, input_shape):
        n_in = input_shape[-1]
        rng  = np.random.default_rng(self.seed)

        # Input weight matrix
        W_in = rng.uniform(-1, 1, size=(n_in, self.units)).astype(np.float32)
        self.W_in = self.add_weight(
            name='W_in', shape=W_in.shape,
            initializer=tf.constant_initializer(W_in),
            trainable=False,
        )

        # Sparse reservoir weight matrix
        W_res = rng.uniform(-1, 1, size=(self.units, self.units)).astype(np.float32)
        mask  = (rng.uniform(0, 1, size=W_res.shape) > self.sparsity).astype(np.float32)
        W_res *= mask

        # Scale to desired spectral radius
        eigvals      = np.linalg.eigvals(W_res)
        spectral_now = float(np.max(np.abs(eigvals)))
        if spectral_now > 0:
            W_res = (W_res / spectral_now * self.spectral_radius).astype(np.float32)

        self.W_res = self.add_weight(
            name='W_res', shape=W_res.shape,
            initializer=tf.constant_initializer(W_res),
            trainable=False,
        )
        super().build(input_shape)

    # ------------------------------------------------------------------
    def compute_output_shape(self, input_shape):
        """Tell Keras the output shape so graph-mode tracing works."""
        return (input_shape[0], self.units)

    # ------------------------------------------------------------------
    @tf.function
    def call(self, inputs):
        """
        inputs : (batch, time_steps, n_features)
        returns: (batch, units)  — reservoir state after the last time step
        """
        # tf.while_loop lets us iterate over T in graph mode
        batch_size  = tf.shape(inputs)[0]
        time_steps  = tf.shape(inputs)[1]
        h_init      = tf.zeros((batch_size, self.units), dtype=inputs.dtype)

        W_in  = tf.cast(self.W_in,  inputs.dtype)
        W_res = tf.cast(self.W_res, inputs.dtype)

        def body(t, h):
            x_t = inputs[:, t, :]
            h   = tf.tanh(tf.matmul(x_t, W_in) + tf.matmul(h, W_res))
            return t + 1, h

        def cond(t, h):
            return t < time_steps

        _, h_final = tf.while_loop(cond, body, [0, h_init])
        return h_final

    # ------------------------------------------------------------------
    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(units=self.units, spectral_radius=self.spectral_radius,
                        sparsity=self.sparsity, seed=self.seed))
        return cfg


def build_esn(input_shape, num_classes_device, num_classes_attack=None):
    """
    Echo State Network (Reservoir Computing).
    Reservoir(128 units, ρ≈0.9) → Dense(64) → Dropout → output head(s)

    The reservoir is randomly initialised and frozen; only the read-out
    layer is trained, making ESNs extremely fast to fit.
    """
    inputs = Input(shape=input_shape)
    # ESN expects (batch, time, features); our input is already 3-D
    x = ReservoirLayer(units=128, spectral_radius=0.9, sparsity=0.1,
                       name='reservoir')(inputs)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    return _compile_and_return(inputs, x, num_classes_device, num_classes_attack)


# ═══════════════════════════════════════════════════════════════════════════
# Registry — maps config name → builder function
# ═══════════════════════════════════════════════════════════════════════════

DL_BUILDERS: dict[str, callable] = {
    "1D-CNN":      build_1dcnn,
    "Autoencoder": build_autoencoder,
    "LSTM":        build_lstm,
    "BiLSTM":      build_bilstm,
    "GRU":         build_gru,
    "CNN-GRU":     build_cnn_gru,
    "CNN-LSTM":    build_cnn_lstm,
    "MLP":         build_mlp,
    "ResNet1D":    build_resnet1d,
    "RNN":         build_rnn,
    "ESN":         build_esn,
}
