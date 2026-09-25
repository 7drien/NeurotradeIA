from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, GaussianNoise
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow as tf

# --- Custom Loss Function (Legacy) ---
def opportunity_cost_loss(y_true, y_pred):
    """
    Legacy loss function preserved if any old code references it.
    """
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred)


def create_lstm_model(input_shape):
    """
    Creates a leaner GRU/LSTM model compiled for binary classification (Meta-Labeling).
    Reduced regularization to allow the model to dynamically learn the patterns without underfitting.
    """
    model = Sequential([
        # Light Data Augmentation
        GaussianNoise(0.005, input_shape=input_shape),
        
        # Sequence modeling with LSTM, light regularization to avoid being static
        LSTM(64, return_sequences=True, kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(0.2),
        
        LSTM(32, return_sequences=False, kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(0.2),

        Dense(32, activation='relu', kernel_regularizer=l2(1e-4)),
        Dropout(0.1),
        Dense(2, activation='softmax')
    ])
    
    # Compile the model with less extreme Label Smoothing
    model.compile(
        optimizer=Adam(learning_rate=0.001), 
        loss=CategoricalCrossentropy(label_smoothing=0.05), 
        metrics=['accuracy']
    )
    return model
