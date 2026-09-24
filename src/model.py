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

def create_lstm_model(input_shape):
    """
    Creates a deep CNN-LSTM model compiled with our custom loss function.
    """
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape, padding='causal'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        LSTM(100, return_sequences=True),
        Dropout(0.3),
        BatchNormalization(),
        
        LSTM(50, return_sequences=False),
        Dropout(0.3),
        BatchNormalization(),

        Dense(50, activation='relu'),
        Dense(3, activation='softmax')
    ])
    
    # Compile the model with the new custom loss function
    model.compile(
        optimizer='adam', 
        loss=opportunity_cost_loss, 
        metrics=['accuracy']
    )
    return model
