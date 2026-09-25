from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, BatchNormalization, MaxPooling1D
from tensorflow.keras.losses import categorical_crossentropy
import tensorflow as tf

# --- Custom Loss Function ---
def opportunity_cost_loss(y_true, y_pred):
    """
    A custom loss function that penalizes the model for predicting 'Hold'
    when there was a clear 'Buy' or 'Sell' opportunity.
    """
    # Standard cross-entropy loss
    base_loss = categorical_crossentropy(y_true, y_pred)

    # --- Penalty Logic ---
    # Identify where the model predicted 'Hold' (class 1)
    predicted_hold = tf.cast(tf.argmax(y_pred, axis=1) == 1, tf.float32)
    
    # Identify where the actual outcome was 'Buy' (class 2) or 'Sell' (class 0)
    actual_trade = tf.cast(tf.argmax(y_true, axis=1) != 1, tf.float32)
    
    # Calculate the opportunity cost: penalty is applied only when
    # the model predicted 'Hold' but should have traded.
    # We can make the penalty a factor of the base loss.
    penalty_factor = 2.0 
    opportunity_cost = penalty_factor * base_loss * predicted_hold * actual_trade
    
    return base_loss + opportunity_cost

from tensorflow.keras.regularizers import l2

def create_lstm_model(input_shape):
    """
    Creates a leaner GRU model compiled for binary classification (Meta-Labeling).
    Reduced complexity and added L2 regularization to prevent overfitting on noisy data.
    """
    model = Sequential([
        # Simpler sequence modeling with GRU, heavily regularized
        LSTM(64, return_sequences=True, input_shape=input_shape, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.4),
        
        LSTM(32, return_sequences=False, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.4),

        Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.2),
        Dense(2, activation='softmax')
    ])
    
    # Compile the model with standard cross-entropy for binary task
    model.compile(
        optimizer='adam', 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    return model
