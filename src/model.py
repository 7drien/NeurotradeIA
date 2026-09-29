import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Input
from tensorflow.keras.optimizers import Adam

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

def create_dense_model(input_dim=1024, lr=0.001):
    """
    Creates a Deep Neural Network for predicting 16-candle ahead price ratios.
    
    Architecture:
    - Input: Vector of 1024 values (128 candles * 8 features)
    - Hidden Layer 1: Dense(256) + LeakyReLU
    - Hidden Layer 2: Dense(64) + LeakyReLU
    - Hidden Layer 3: Dense(16) + LeakyReLU
    - Hidden Layer 4: Dense(8) + LeakyReLU
    - Output Layer: Dense(1) + 2*sigmoid activation
    """
    model = Sequential([
        Input(shape=(input_dim,)),
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
        optimizer=Adam(learning_rate=lr),
        loss='mse',
        metrics=['mae', directional_accuracy]
    )
    return model

# Alias for backward compatibility
create_lstm_model = create_dense_model
create_model = create_dense_model
