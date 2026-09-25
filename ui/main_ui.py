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
        self.title("NeurotradeIA - Institutional Quant Dashboard")
        self.geometry("1400x900")
        
        # Apply a dark theme or modern look
        style = ttk.Style(self)
        if 'clam' in style.theme_names():
            style.theme_use('clam')
        style.configure("TFrame", background="#1e1e1e")
        style.configure("TLabel", background="#1e1e1e", foreground="#ffffff", font=("Segoe UI", 12))
        style.configure("TButton", font=("Segoe UI", 12, "bold"), padding=10)
        style.configure("Header.TLabel", font=("Segoe UI", 16, "bold"), foreground="#00d2ff")
        
        self.configure(bg="#1e1e1e")

        # --- UI Layout ---
        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Top frame for controls and metrics
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 20))

        title_lbl = ttk.Label(header_frame, text="Neural Network Meta-Labeling", style="Header.TLabel")
        title_lbl.pack(side=tk.LEFT)

        self.train_button = ttk.Button(header_frame, text="▶ START TRAINING (EPOCH-BY-EPOCH)", command=self.run_training)
        self.train_button.pack(side=tk.RIGHT, padx=10)
        
        # Metrics bar
        metrics_frame = ttk.Frame(main_frame)
        metrics_frame.pack(fill=tk.X, pady=10)
        
        self.epoch_label = ttk.Label(metrics_frame, text="Epoch: N/A", font=("Segoe UI", 14, "bold"))
        self.epoch_label.pack(side=tk.LEFT, padx=20)
        
        self.return_label = ttk.Label(metrics_frame, text="Return: 0.00%", font=("Segoe UI", 14, "bold"), foreground="#4caf50")
        self.return_label.pack(side=tk.LEFT, padx=20)

        self.winrate_label = ttk.Label(metrics_frame, text="Win Rate: 0.00%", font=("Segoe UI", 14, "bold"), foreground="#4caf50")
        self.winrate_label.pack(side=tk.LEFT, padx=20)

        self.progress = ttk.Progressbar(metrics_frame, mode='indeterminate', length=200)

        # Main plot area for backtest results
        plot_frame = tk.Frame(main_frame, bg="#1e1e1e")
        plot_frame.pack(fill=tk.BOTH, expand=True)

        # --- Matplotlib Figure for Backtesting ---
        # Dark style for matplotlib
        plt.style.use('dark_background')
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        self.fig.patch.set_facecolor('#1e1e1e')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Start the queue checker
        self.after(100, self.check_queue)

    def run_training(self):
        """Runs the training process in a separate thread."""
        self.train_button.config(state="disabled")
        self.progress.pack(side=tk.RIGHT, padx=20)
        self.progress.start(10)
        print("Starting training...")
        
        # Clear previous plots and labels
        self.ax1.clear()
        self.ax2.clear()
        self.epoch_label.config(text="Epoch: 0")
        self.return_label.config(text="Return: 0.00%")
        self.winrate_label.config(text="Win Rate: 0.00%")
        self.canvas.draw()

        # The training function now needs to accept the queue
        thread = threading.Thread(target=train_model_with_callback, args=(ui_queue,))
        thread.start()

    def check_queue(self):
        """Checks the queue for messages and updates the UI."""
        try:
            while not ui_queue.empty():
                message = ui_queue.get_nowait()
                msg_type = message.get('type')
    
                if msg_type == 'backtest_update':
                    results = message.get('results')
                    if results:
                        # Update the main plot with the new backtest results
                        plot_backtest_results(results, self.fig, self.ax1, self.ax2)
                        
                        # Style axes for dark theme
                        for ax in [self.ax1, self.ax2]:
                            ax.set_facecolor('#2d2d2d')
                            ax.grid(color='#444444')
                            
                        self.canvas.draw()
                        
                        # Update labels using vectorbt portfolio
                        pf = results['portfolio']
                        final_value = pf.value().iloc[-1]
                        initial_capital = results['initial_capital']
                        returns = (final_value - initial_capital) / initial_capital * 100
                        winrate = pf.trades.win_rate() * 100
                        
                        self.return_label.config(text=f"Return: {returns:.2f}%", foreground="#4caf50" if returns >= 0 else "#f44336")
                        self.winrate_label.config(text=f"Win Rate: {winrate:.2f}%")
    
                elif msg_type == 'train_update':
                    # Update epoch label
                    epoch = message.get('epoch', 0) + 1
                    self.epoch_label.config(text=f"Epoch: {epoch}")
    
                elif msg_type == 'train_finished':
                    print("Training process finished.")
                    self.train_button.config(state="normal")
                    self.progress.stop()
                    self.progress.pack_forget()

        except Empty:
            pass
        finally:
            self.after(100, self.check_queue)

if __name__ == "__main__":
    app = App()
    app.mainloop()
