import os
import sys
import glob

# Ensure CUDA 12 libraries from virtualenv site-packages are visible to TensorFlow
_venv_nvidia = os.path.join(sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages", "nvidia")
if os.path.exists(_venv_nvidia):
    _nv_paths = glob.glob(f"{_venv_nvidia}/*/lib")
    if _nv_paths:
        current_ld = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = ":".join(_nv_paths) + ((":" + current_ld) if current_ld else "")
    _cuda_nvcc = os.path.join(_venv_nvidia, "cuda_nvcc")
    if os.path.exists(_cuda_nvcc) and "XLA_FLAGS" not in os.environ:
        os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={_cuda_nvcc}"

import tensorflow as tf

# Enable GPU memory growth if GPU is present
try:
    _gpus = tf.config.list_physical_devices('GPU')
    for _gpu in _gpus:
        tf.config.experimental.set_memory_growth(_gpu, True)
except Exception:
    pass

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LeakyReLU, Input
from tensorflow.keras.optimizers import Adam
from src.config import LEARNING_RATE, DEFAULT_N_MODELS

@tf.keras.utils.register_keras_serializable(name="two_sigmoid")
def two_sigmoid(x):
    """
    Activation function: 2 * sigmoid(x).
    Range: (0, 2).
    - 1.0 = price unchanged
    - > 1.0 = price increase (up to 200%)
    - < 1.0 = price decrease (down to 0%)
    """
    return 2.0 * tf.sigmoid(x)

@tf.keras.utils.register_keras_serializable(name="directional_accuracy")
def directional_accuracy(y_true, y_pred):
    """
    Binary directional accuracy: measures whether the prediction correctly
    predicts the price moving UP (> 1.0) or DOWN (< 1.0).
    """
    true_up = tf.cast(y_true > 1.0, tf.float32)
    pred_up = tf.cast(y_pred > 1.0, tf.float32)
    return tf.reduce_mean(tf.cast(tf.equal(true_up, pred_up), tf.float32))

def opportunity_cost_loss(y_true, y_pred):
    """Legacy loss preserved for backward compatibility."""
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred)

def create_dense_model(input_dim=2048, lr=LEARNING_RATE, n_models=DEFAULT_N_MODELS):
    """
    Creates a Deep Neural Network (or N-Model Ensemble) for predicting 32-candle ahead price ratios.
    
    If n_models == 1:
        Creates a single 6-layer MLP Sequential model with 2*sigmoid output.
    If n_models > 1:
        Creates an Ensemble Model with N independent parallel MLP branches,
        each with its own independent random weights, learning diverse trading signals.
    """
    if n_models <= 1:
        model = Sequential([
            Input(shape=(input_dim,)),
            Dense(1024),
            LeakyReLU(negative_slope=0.01),
            Dense(512),
            LeakyReLU(negative_slope=0.01),
            Dense(256),
            LeakyReLU(negative_slope=0.01),
            Dense(64),
            LeakyReLU(negative_slope=0.01),
            Dense(16),
            LeakyReLU(negative_slope=0.01),
            Dense(8),
            LeakyReLU(negative_slope=0.01),
            Dense(1, activation=two_sigmoid)
        ])
        model.compile(
            optimizer=Adam(learning_rate=lr, clipnorm=1.0),
            loss='mse',
            metrics=['mae', directional_accuracy]
        )
        return model
    else:
        inp = Input(shape=(input_dim,))
        outputs = []
        for i in range(n_models):
            x = Dense(1024, name=f"dense_1024_m{i}")(inp)
            x = LeakyReLU(negative_slope=0.01, name=f"lrelu_1_m{i}")(x)
            x = Dense(512, name=f"dense_512_m{i}")(x)
            x = LeakyReLU(negative_slope=0.01, name=f"lrelu_2_m{i}")(x)
            x = Dense(256, name=f"dense_256_m{i}")(x)
            x = LeakyReLU(negative_slope=0.01, name=f"lrelu_3_m{i}")(x)
            x = Dense(64, name=f"dense_64_m{i}")(x)
            x = LeakyReLU(negative_slope=0.01, name=f"lrelu_4_m{i}")(x)
            x = Dense(16, name=f"dense_16_m{i}")(x)
            x = LeakyReLU(negative_slope=0.01, name=f"lrelu_5_m{i}")(x)
            x = Dense(8, name=f"dense_8_m{i}")(x)
            x = LeakyReLU(negative_slope=0.01, name=f"lrelu_6_m{i}")(x)
            out = Dense(1, activation=two_sigmoid, name=f"out_m{i}")(x)
            outputs.append(out)

        model = Model(inputs=inp, outputs=outputs, name=f"ensemble_{n_models}_mlp")
        model.compile(
            optimizer=Adam(learning_rate=lr, clipnorm=1.0),
            loss=['mse'] * n_models,
            metrics=[['mae', directional_accuracy]] * n_models
        )
        return model

# Alias for backward compatibility
create_lstm_model = create_dense_model
create_model = create_dense_model
