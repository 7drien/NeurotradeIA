from tensorflow.keras.callbacks import Callback
from src.backtesting import simulate_backtest # Import the simulation function

class UILoggerCallback(Callback):
    """
    A custom Keras callback to send training progress to the UI queue.
    """
    def __init__(self, queue):
        super().__init__()
        self.queue = queue

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch."""
        if logs:
            # We can still send this for logging purposes, even if not plotted
            self.queue.put({'type': 'train_update', 'epoch': epoch, 'logs': logs.copy()})

    def on_train_end(self, logs=None):
        """Called at the end of training."""
        self.queue.put({'type': 'train_finished'})


class BacktestOnEpochEnd(Callback):
    """
    A Keras callback to run a backtest at a specified frequency and send the
    FULL results to the UI for plotting.
    """
    def __init__(self, queue, frequency=5):
        super().__init__()
        self.queue = queue
        self.frequency = frequency

    def on_epoch_end(self, epoch, logs=None):
        """
        At the end of a specified epoch, run a backtest and send full results.
        """
        if (epoch + 1) % self.frequency == 0:
            print(f"\n--- Running backtest for epoch {epoch + 1} ---")
            
            # The model is saved by ModelCheckpoint, so simulate_backtest will load the best version.
            backtest_results = simulate_backtest()
            
            if backtest_results:
                # Send the ENTIRE results dictionary to the UI
                self.queue.put({'type': 'backtest_update', 'results': backtest_results})
            else:
                print("Backtest simulation failed for this epoch.")
