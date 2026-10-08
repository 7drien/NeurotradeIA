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
    A Keras callback to calculate the model oscillator on the 20% test slice of BTC-USD
    at a specified frequency and send the FULL backtest results to the UI for plotting.
    """
    def __init__(self, queue, frequency=1, params=None, test_data=None):
        super().__init__()
        self.queue = queue
        self.frequency = frequency
        self.params = params if params is not None else {}
        self.test_data = test_data
        self.current_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        """
        At the end of each specified epoch, run inference on the 20% test slice of BTC-USD
        using the current model weights and send full results to the UI.
        """
        self.current_epoch = epoch + 1
        if (epoch + 1) % self.frequency == 0:
            print(f"\n--- Running backtest & indicator calculation for epoch {epoch + 1} ---")
            
            backtest_results = simulate_backtest(
                params=self.params,
                model=self.model,
                test_data=self.test_data
            )
            
            if backtest_results:
                preds = backtest_results['preds']
                print(f"Epoch {epoch + 1}: Indicator successfully computed on {len(preds)} test candles (20% end of BTC-USD). Predictions range: [{preds.min():.4f}, {preds.max():.4f}]")
                backtest_results['epoch'] = epoch + 1
                # Send the ENTIRE results dictionary to the UI
                self.queue.put({'type': 'backtest_update', 'results': backtest_results, 'epoch': epoch + 1})
            else:
                print("Backtest simulation failed for this epoch.")

    def on_train_end(self, logs=None):
        """
        Runs final inference on the restored best weights after early stopping
        or full training completion, ensuring cache and UI display the best model.
        """
        final_epoch = self.current_epoch if self.current_epoch > 0 else 1
        print(f"\n--- Running final backtest evaluation on best model weights (Epoch {final_epoch}) ---")
        backtest_results = simulate_backtest(
            params=self.params,
            model=self.model,
            test_data=self.test_data
        )
        if backtest_results:
            backtest_results['epoch'] = final_epoch
            self.queue.put({'type': 'backtest_update', 'results': backtest_results, 'epoch': final_epoch})


