import tkinter as tk
from tkinter import ttk
import threading
from queue import Queue, Empty

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from src.train import train_model_with_callback
from src.backtesting import plot_backtest_results

ui_queue = Queue()

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("NeurotradeIA - Backtest-Centric Training")
        self.geometry("1200x800")

        # --- UI Layout ---
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Top frame for controls
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=5)

        self.train_button = ttk.Button(control_frame, text="Start Training & Live Backtesting", command=self.run_training)
        self.train_button.pack(side=tk.LEFT, padx=10)
        
        self.epoch_label = ttk.Label(control_frame, text="Epoch: N/A", font=("Helvetica", 12))
        self.epoch_label.pack(side=tk.LEFT, padx=10)
        
        self.return_label = ttk.Label(control_frame, text="Return: N/A", font=("Helvetica", 12))
        self.return_label.pack(side=tk.LEFT, padx=10)

        # Main plot area for backtest results
        plot_frame = ttk.Frame(main_frame, padding="5")
        plot_frame.pack(fill=tk.BOTH, expand=True)

        # --- Matplotlib Figure for Backtesting ---
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Start the queue checker
        self.after(100, self.check_queue)

    def run_training(self):
        """Runs the training process in a separate thread."""
        self.train_button.config(state="disabled")
        print("Starting training...")
        
        # Clear previous plots and labels
        self.ax1.clear()
        self.ax2.clear()
        self.epoch_label.config(text="Epoch: 0")
        self.return_label.config(text="Return: 0.00%")
        self.canvas.draw()

        # The training function now needs to accept the queue
        thread = threading.Thread(target=train_model_with_callback, args=(ui_queue,))
        thread.start()

    def check_queue(self):
        """Checks the queue for messages and updates the UI."""
        try:
            message = ui_queue.get_nowait()
            msg_type = message.get('type')

            if msg_type == 'backtest_update':
                results = message.get('results')
                if results:
                    # Update the main plot with the new backtest results
                    plot_backtest_results(results, self.fig, self.ax1, self.ax2)
                    self.canvas.draw()
                    
                    # Update labels
                    final_value = results['portfolio_value'][-1]
                    initial_capital = results['initial_capital']
                    returns = (final_value - initial_capital) / initial_capital * 100
                    self.return_label.config(text=f"Return: {returns:.2f}%")

            elif msg_type == 'train_update':
                # Update epoch label
                epoch = message.get('epoch', 0) + 1
                self.epoch_label.config(text=f"Epoch: {epoch}")

            elif msg_type == 'train_finished':
                print("Training process finished.")
                self.train_button.config(state="normal")

        except Empty:
            pass
        finally:
            self.after(100, self.check_queue)

if __name__ == "__main__":
    app = App()
    app.mainloop()
