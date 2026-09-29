import os
import tkinter as tk
from tkinter import ttk
import threading
from queue import Queue, Empty

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from src.train import train_model_with_callback
from src.backtesting import plot_backtest_results, run_backtest_with_threshold

ui_queue = Queue()

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("NeurotradeIA - 128-Candle Oscillator Dashboard")
        self.geometry("1450x950")
        
        # Apply modern dark theme
        style = ttk.Style(self)
        if 'clam' in style.theme_names():
            style.theme_use('clam')
        style.configure("TFrame", background="#1e1e1e")
        style.configure("TLabel", background="#1e1e1e", foreground="#ffffff", font=("Segoe UI", 11))
        style.configure("TButton", font=("Segoe UI", 11, "bold"), padding=8)
        style.configure("Header.TLabel", font=("Segoe UI", 15, "bold"), foreground="#00d2ff")
        style.configure("SubHeader.TLabel", font=("Segoe UI", 12, "bold"), foreground="#00e676")
        
        self.configure(bg="#1e1e1e")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # --- UI Layout ---
        main_frame = ttk.Frame(self, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left Panel (Settings & Controls)
        left_panel = ttk.Frame(main_frame, width=320)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        
        # Right Panel (Metrics & Matplotlib Plots)
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # --- Section 1: Model & Training Configuration ---
        settings_lbl = ttk.Label(left_panel, text="Training Configuration", style="Header.TLabel")
        settings_lbl.pack(pady=(0, 15), anchor=tk.W)
        
        self.epochs_var = tk.StringVar(value="50")
        self.batch_size_var = tk.StringVar(value="32")
        self.n_steps_var = tk.StringVar(value="128")
        self.k_steps_var = tk.StringVar(value="16")
        self.days_var = tk.StringVar(value="700")
        
        self.create_input_field(left_panel, "Epochs:", self.epochs_var)
        self.create_input_field(left_panel, "Batch Size:", self.batch_size_var)
        self.create_input_field(left_panel, "Sequence (128 Candles):", self.n_steps_var)
        self.create_input_field(left_panel, "Horizon (16 Candles):", self.k_steps_var)
        self.create_input_field(left_panel, "Days to Load:", self.days_var)
        
        self.train_button = ttk.Button(left_panel, text="▶ START TRAINING", command=self.run_training)
        self.train_button.pack(fill=tk.X, pady=(15, 10))

        # --- Section 2: Fast Oscillator Strategy (Dual Thresholds: Entry & Exit) ---
        sep = ttk.Separator(left_panel, orient='horizontal')
        sep.pack(fill=tk.X, pady=15)

        strat_lbl = ttk.Label(left_panel, text="Dual-Threshold Strategy", style="SubHeader.TLabel")
        strat_lbl.pack(pady=(0, 5), anchor=tk.W)

        desc_lbl = ttk.Label(
            left_panel, 
            text="Long: Buy > 1+x_in  | Exit < 1+x_out\nShort: Sell < 1-x_in | Exit > 1-x_out\nRecalculates instantly without retraining.",
            font=("Segoe UI", 9),
            foreground="#aaaaaa"
        )
        desc_lbl.pack(pady=(0, 8), anchor=tk.W)

        self.x_entry_var = tk.StringVar(value="0.01")
        self.x_exit_var = tk.StringVar(value="0.005")

        self.create_input_field(left_panel, "Entry Threshold x_in (e.g. 0.01 = 1%):", self.x_entry_var)
        self.create_input_field(left_panel, "Exit Threshold x_out (e.g. 0.005 = 0.5%):", self.x_exit_var)

        # Quick preset buttons
        preset_lbl = ttk.Label(left_panel, text="Quick Presets (Entry / Exit):", font=("Segoe UI", 9), foreground="#aaaaaa")
        preset_lbl.pack(pady=(6, 2), anchor=tk.W)

        preset_frame1 = ttk.Frame(left_panel)
        preset_frame1.pack(fill=tk.X, pady=2)
        ttk.Button(preset_frame1, text="0.01 / 0.005", width=12, command=lambda: self.set_thresholds("0.01", "0.005")).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_frame1, text="0.008 / 0.003", width=12, command=lambda: self.set_thresholds("0.008", "0.003")).pack(side=tk.LEFT, padx=2)

        preset_frame2 = ttk.Frame(left_panel)
        preset_frame2.pack(fill=tk.X, pady=2)
        ttk.Button(preset_frame2, text="0.015 / 0.008", width=12, command=lambda: self.set_thresholds("0.015", "0.008")).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_frame2, text="0.02 / 0.01", width=12, command=lambda: self.set_thresholds("0.02", "0.01")).pack(side=tk.LEFT, padx=2)

        self.recalc_button = ttk.Button(
            left_panel, 
            text="⚡ RECALCULATE BACKTEST", 
            command=self.run_recalculate_backtest
        )
        self.recalc_button.pack(fill=tk.X, pady=(12, 10))
        
        # --- Top Header & Metrics Bar (Right Panel) ---
        header_frame = ttk.Frame(right_panel)
        header_frame.pack(fill=tk.X, pady=(0, 10))

        title_lbl = ttk.Label(header_frame, text="128-Candle MLP Oscillator & Backtesting Engine", style="Header.TLabel")
        title_lbl.pack(side=tk.LEFT)
        
        metrics_frame = ttk.Frame(right_panel)
        metrics_frame.pack(fill=tk.X, pady=5)
        
        self.epoch_label = ttk.Label(metrics_frame, text="Epoch: N/A", font=("Segoe UI", 13, "bold"))
        self.epoch_label.pack(side=tk.LEFT, padx=15)
        
        self.return_label = ttk.Label(metrics_frame, text="Return: 0.00%", font=("Segoe UI", 13, "bold"), foreground="#4caf50")
        self.return_label.pack(side=tk.LEFT, padx=15)

        self.winrate_label = ttk.Label(metrics_frame, text="Win Rate: 0.00%", font=("Segoe UI", 13, "bold"), foreground="#4caf50")
        self.winrate_label.pack(side=tk.LEFT, padx=15)

        self.trades_label = ttk.Label(metrics_frame, text="Trades: 0", font=("Segoe UI", 13, "bold"), foreground="#00d2ff")
        self.trades_label.pack(side=tk.LEFT, padx=15)

        self.progress = ttk.Progressbar(metrics_frame, mode='indeterminate', length=180)

        # Plot area (3 Subplots: Price, Oscillator, Equity)
        plot_frame = tk.Frame(right_panel, bg="#1e1e1e")
        plot_frame.pack(fill=tk.BOTH, expand=True)

        plt.style.use('dark_background')
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(
            3, 1, figsize=(16, 12),
            gridspec_kw={'height_ratios': [2.2, 1.2, 1.2]},
            sharex=True
        )
        self.fig.patch.set_facecolor('#1e1e1e')
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.set_facecolor('#2d2d2d')
            ax.grid(color='#444444', alpha=0.3)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Start listening to UI queue and attempt initial chart load
        self.after(100, self.check_queue)
        self.after(600, self.initial_load_backtest)

    def create_input_field(self, parent, label_text, var):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=4)
        lbl = ttk.Label(frame, text=label_text, font=("Segoe UI", 9))
        lbl.pack(side=tk.TOP, anchor=tk.W)
        entry = ttk.Entry(frame, textvariable=var, font=("Segoe UI", 10))
        entry.pack(side=tk.TOP, fill=tk.X)

    def set_thresholds(self, in_val_str, out_val_str):
        self.x_entry_var.set(in_val_str)
        self.x_exit_var.set(out_val_str)
        self.run_recalculate_backtest()

    def run_training(self):
        """Runs model training in a background daemon thread."""
        self.train_button.config(state="disabled")
        self.recalc_button.config(state="disabled")
        self.progress.pack(side=tk.RIGHT, padx=20)
        self.progress.start(10)
        print("Starting training process...")
        
        try:
            params = {
                'epochs': int(self.epochs_var.get()),
                'batch_size': int(self.batch_size_var.get()),
                'n_steps': int(self.n_steps_var.get()),
                'k_steps': int(self.k_steps_var.get()),
                'days_to_load': int(self.days_var.get()),
                'x_entry': float(self.x_entry_var.get()),
                'x_exit': float(self.x_exit_var.get())
            }
        except ValueError:
            print("Invalid input parameters. Please check values.")
            self.progress.stop()
            self.progress.pack_forget()
            self.train_button.config(state="normal")
            self.recalc_button.config(state="normal")
            return
        
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.epoch_label.config(text="Epoch: 0")
        self.canvas.draw()

        thread = threading.Thread(target=train_model_with_callback, args=(ui_queue, params))
        thread.daemon = True
        thread.start()

    def run_recalculate_backtest(self):
        """
        Recalculates the backtest instantly using cached predictions for x_entry and x_exit.
        Does NOT re-train the model.
        """
        try:
            x_in = float(self.x_entry_var.get())
            x_out = float(self.x_exit_var.get())
        except ValueError:
            print("Invalid thresholds. Please enter decimal numbers like 0.01 and 0.005.")
            return

        print(f"Recalculating backtest with x_entry={x_in:.4f}, x_exit={x_out:.4f}...")
        try:
            results = run_backtest_with_threshold(x_entry=x_in, x_exit=x_out)
            if results:
                self.update_backtest_ui(results)
            else:
                print("Could not run backtest. Train the model first to generate predictions.")
        except Exception as e:
            print(f"Backtest recalculation error: {e}")

    def update_backtest_ui(self, results):
        """Renders backtest results across all 3 subplots and updates metric labels."""
        if not results:
            return
        
        plot_backtest_results(results, self.fig, self.ax1, self.ax2, self.ax3)
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.set_facecolor('#2d2d2d')
            ax.grid(color='#444444', alpha=0.3)
        self.canvas.draw()
        
        pf = results['portfolio']
        final_value = pf.value().iloc[-1]
        initial_capital = results['initial_capital']
        returns = (final_value - initial_capital) / initial_capital * 100
        total_trades = pf.trades.count()
        winrate = pf.trades.win_rate() * 100 if total_trades > 0 else 0.0
        
        self.return_label.config(
            text=f"Return: {returns:.2f}%", 
            foreground="#4caf50" if returns >= 0 else "#f44336"
        )
        self.winrate_label.config(text=f"Win Rate: {winrate:.2f}%")
        self.trades_label.config(text=f"Trades: {total_trades}")

    def initial_load_backtest(self):
        """Loads and displays existing cached backtest on startup if available."""
        if os.path.exists("data/cached/model_predictions.pkl") or os.path.exists("best_model.keras"):
            try:
                x_in = float(self.x_entry_var.get())
                x_out = float(self.x_exit_var.get())
                results = run_backtest_with_threshold(x_entry=x_in, x_exit=x_out)
                if results:
                    self.update_backtest_ui(results)
            except Exception as e:
                print(f"Initial backtest load skipped: {e}")

    def check_queue(self):
        """Processes background worker events from training."""
        try:
            while not ui_queue.empty():
                message = ui_queue.get_nowait()
                msg_type = message.get('type')
    
                if msg_type == 'backtest_update':
                    results = message.get('results')
                    if results:
                        self.update_backtest_ui(results)
    
                elif msg_type == 'train_update':
                    epoch = message.get('epoch', 0) + 1
                    self.epoch_label.config(text=f"Epoch: {epoch}")
    
                elif msg_type == 'train_finished':
                    print("Training process finished.")
                    self.train_button.config(state="normal")
                    self.recalc_button.config(state="normal")
                    self.progress.stop()
                    self.progress.pack_forget()

        except Empty:
            pass
        finally:
            self.after(100, self.check_queue)
            
    def on_closing(self):
        import os
        print("Closing application...")
        self.destroy()
        os._exit(0)

if __name__ == "__main__":
    app = App()
    app.mainloop()
