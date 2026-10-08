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

# Color Palette - Obsidian Quant Terminal
BG_MAIN = "#0c0e14"
BG_CARD = "#141721"
BORDER_CARD = "#232738"
BG_INPUT = "#1b1f2d"
BORDER_INPUT = "#2d3448"
BORDER_FOCUS = "#38bdf8"
TEXT_WHITE = "#f8fafc"
TEXT_MUTED = "#94a3b8"
TEXT_SUB = "#64748b"

ACCENT_BLUE = "#0284c7"
ACCENT_BLUE_HOVER = "#0369a1"
ACCENT_PURPLE = "#6366f1"
ACCENT_PURPLE_HOVER = "#4f46e5"
ACCENT_GREEN = "#10b981"
ACCENT_AMBER = "#f59e0b"
ACCENT_CYAN = "#06b6d4"
ACCENT_ROSE = "#f43f5e"

def get_font_family():
    try:
        from tkinter import font
        families = font.families()
        for f in ["Segoe UI", "Ubuntu", "DejaVu Sans", "Helvetica", "Arial"]:
            if f in families:
                return f
    except Exception:
        pass
    return "DejaVu Sans"

def get_device_info():
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            return "NVIDIA CUDA GPU Active", ACCENT_GREEN
    except Exception:
        pass
    return "CPU Fallback Mode", TEXT_MUTED

class ModernEntry(tk.Entry):
    def __init__(self, master, textvariable=None, font_family="DejaVu Sans", **kwargs):
        super().__init__(
            master,
            textvariable=textvariable,
            bg=BG_INPUT,
            fg=TEXT_WHITE,
            insertbackground="#ffffff",
            relief="flat",
            highlightbackground=BORDER_INPUT,
            highlightcolor=BORDER_FOCUS,
            highlightthickness=1,
            font=(font_family, 9),
            **kwargs
        )

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("NeurotradeAI - Quantitative Dashboard")
        self.geometry("1450x950")
        self.configure(bg=BG_MAIN)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.font_family = get_font_family()

        # Modern TTK style configuration
        style = ttk.Style(self)
        if 'clam' in style.theme_names():
            style.theme_use('clam')
        style.configure(
            "Horizontal.TProgressbar", 
            troughcolor=BG_CARD, 
            background=ACCENT_BLUE, 
            bordercolor=BORDER_CARD, 
            lightcolor=ACCENT_BLUE, 
            darkcolor=ACCENT_BLUE
        )

        # Main horizontal container (zero scrollbars needed)
        main_frame = tk.Frame(self, bg=BG_MAIN, padx=12, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # =========================================================================
        # LEFT PANEL: Parameters & Controls (Card-Based Layout, No Scrollbar)
        # =========================================================================
        left_panel = tk.Frame(main_frame, bg=BG_MAIN, width=335)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))
        left_panel.pack_propagate(False)

        # --- Card 1: Model & Training Configuration ---
        c1 = self.create_card(left_panel)
        c1.pack(fill=tk.X, pady=(0, 10))

        c1_header = tk.Frame(c1, bg=BG_CARD)
        c1_header.pack(fill=tk.X, pady=(0, 8))
        tk.Label(c1_header, text="MODEL CONFIGURATION", font=(self.font_family, 9, "bold"), fg=BORDER_FOCUS, bg=BG_CARD).pack(side=tk.LEFT)

        self.epochs_var = tk.StringVar(value="200")
        self.batch_size_var = tk.StringVar(value="128")
        self.lr_var = tk.StringVar(value="0.0003")
        self.n_models_var = tk.StringVar(value="3")
        self.days_var = tk.StringVar(value="All")
        self.n_steps_var = tk.StringVar(value="256")
        self.k_steps_var = tk.StringVar(value="32")

        self.create_two_inputs_row(c1, "Epochs:", self.epochs_var, "Batch Size:", self.batch_size_var)
        self.create_two_inputs_row(c1, "Learning Rate:", self.lr_var, "Ensemble (n models):", self.n_models_var)
        self.create_two_inputs_row(c1, "Sequence (256):", self.n_steps_var, "Horizon (32):", self.k_steps_var)
        self.create_single_input_row(c1, "Days to Load:", self.days_var)

        self.early_stopping_var = tk.BooleanVar(value=True)
        self.es_check = tk.Checkbutton(
            c1, 
            text="Enable Early Stopping (patience=15)", 
            variable=self.early_stopping_var,
            bg=BG_CARD,
            fg=TEXT_WHITE,
            selectcolor=BG_INPUT,
            activebackground=BG_CARD,
            activeforeground=TEXT_WHITE,
            relief="flat",
            highlightthickness=0,
            font=(self.font_family, 8)
        )
        self.es_check.pack(anchor=tk.W, pady=(4, 8))

        self.train_button = tk.Button(
            c1,
            text="▶  START TRAINING",
            font=(self.font_family, 9, "bold"),
            bg=ACCENT_BLUE,
            fg="#ffffff",
            activebackground=ACCENT_BLUE_HOVER,
            activeforeground="#ffffff",
            relief="flat",
            bd=0,
            cursor="hand2",
            pady=7,
            command=self.run_training
        )
        self.train_button.pack(fill=tk.X)

        # --- Card 2: Fast Oscillator Strategy (VectorBT) ---
        c2 = self.create_card(left_panel)
        c2.pack(fill=tk.X, pady=(0, 10))

        c2_header = tk.Frame(c2, bg=BG_CARD)
        c2_header.pack(fill=tk.X, pady=(0, 2))
        tk.Label(c2_header, text="FAST OSCILLATOR STRATEGY", font=(self.font_family, 9, "bold"), fg=ACCENT_PURPLE, bg=BG_CARD).pack(side=tk.LEFT)
        tk.Label(c2_header, text="● VectorBT", font=(self.font_family, 7, "bold"), fg=TEXT_SUB, bg=BG_CARD).pack(side=tk.RIGHT)

        rule_lbl = tk.Label(
            c2, 
            text="Long: Buy > 1+x_in  |  Exit < 1+x_out\nShort: Sell < 1-x_in  |  Exit > 1-x_out", 
            font=(self.font_family, 7), 
            fg=TEXT_SUB, 
            bg=BG_CARD, 
            justify=tk.LEFT
        )
        rule_lbl.pack(anchor=tk.W, pady=(0, 6))

        self.x_entry_var = tk.StringVar(value="0.05")
        self.x_exit_var = tk.StringVar(value="-0.02")
        self.fees_var = tk.StringVar(value="0.1")

        self.create_two_inputs_row(c2, "Entry x_in (e.g. 0.05):", self.x_entry_var, "Exit x_out (e.g. -0.02):", self.x_exit_var)
        self.create_single_input_row(c2, "Fees per Trade (%):", self.fees_var)

        # Quick Presets
        tk.Label(c2, text="Quick Presets (Entry / Exit):", font=(self.font_family, 7, "bold"), fg=TEXT_MUTED, bg=BG_CARD).pack(anchor=tk.W, pady=(6, 2))
        
        p_row1 = tk.Frame(c2, bg=BG_CARD)
        p_row1.pack(fill=tk.X, pady=1)
        self.create_preset_btn(p_row1, "0.01 / 0.005", "0.01", "0.005").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        self.create_preset_btn(p_row1, "0.008 / 0.003", "0.008", "0.003").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

        p_row2 = tk.Frame(c2, bg=BG_CARD)
        p_row2.pack(fill=tk.X, pady=1)
        self.create_preset_btn(p_row2, "0.015 / 0.008", "0.015", "0.008").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        self.create_preset_btn(p_row2, "0.02 / 0.01", "0.02", "0.01").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

        self.recalc_button = tk.Button(
            c2,
            text="⚡  RECALCULATE BACKTEST",
            font=(self.font_family, 9, "bold"),
            bg=ACCENT_PURPLE,
            fg="#ffffff",
            activebackground=ACCENT_PURPLE_HOVER,
            activeforeground="#ffffff",
            relief="flat",
            bd=0,
            cursor="hand2",
            pady=7,
            command=self.run_recalculate_backtest
        )
        self.recalc_button.pack(fill=tk.X, pady=(8, 0))

        # --- Card 3: Hardware & Partition Status ---
        c3 = self.create_card(left_panel)
        c3.pack(fill=tk.X)

        device_str, device_color = get_device_info()
        c3_header = tk.Frame(c3, bg=BG_CARD)
        c3_header.pack(fill=tk.X)
        tk.Label(c3_header, text=f"● {device_str}", font=(self.font_family, 8, "bold"), fg=device_color, bg=BG_CARD).pack(side=tk.LEFT)

        tk.Label(
            c3, 
            text="Data: 80% Train | 20% Out-of-sample Test",
            font=(self.font_family, 7), 
            fg=TEXT_SUB, 
            bg=BG_CARD, 
            justify=tk.LEFT
        ).pack(anchor=tk.W, pady=(3, 0))

        # =========================================================================
        # RIGHT PANEL: Top Header, Modern KPI Badges & Matplotlib Charts
        # =========================================================================
        right_panel = tk.Frame(main_frame, bg=BG_MAIN)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Header Title Bar
        header_frame = tk.Frame(right_panel, bg=BG_MAIN)
        header_frame.pack(fill=tk.X, pady=(0, 6))

        title_left = tk.Frame(header_frame, bg=BG_MAIN)
        title_left.pack(side=tk.LEFT)
        tk.Label(title_left, text="NEUROTRADE AI", font=(self.font_family, 13, "bold"), fg=TEXT_WHITE, bg=BG_MAIN).pack(side=tk.LEFT)
        tk.Label(title_left, text=" | MLP Oscillator & Strategy Terminal", font=(self.font_family, 10), fg=TEXT_SUB, bg=BG_MAIN).pack(side=tk.LEFT, padx=6)

        header_right = tk.Frame(header_frame, bg=BG_MAIN)
        header_right.pack(side=tk.RIGHT)
        self.progress = ttk.Progressbar(header_right, mode='indeterminate', length=140)

        # KPI Metric Cards Bar (7 Cards)
        metrics_bar = tk.Frame(right_panel, bg=BG_MAIN)
        metrics_bar.pack(fill=tk.X, pady=(0, 8))

        kpis = [
            ("EPOCH", "N/A", TEXT_MUTED),
            ("STRATEGY RETURN", "+0.00%", ACCENT_GREEN),
            ("BUY & HOLD BTC", "+0.00%", ACCENT_AMBER),
            ("SHARPE RATIO", "0.00", ACCENT_PURPLE),
            ("WIN RATE", "0.0%", ACCENT_CYAN),
            ("TOTAL TRADES", "0", BORDER_FOCUS),
            ("MAX DRAWDOWN", "0.00%", ACCENT_ROSE),
        ]

        self.kpi_labels = {}
        for key, default_val, default_color in kpis:
            card = tk.Frame(metrics_bar, bg=BG_CARD, highlightbackground=BORDER_CARD, highlightthickness=1, padx=8, pady=5)
            card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
            
            lbl_title = tk.Label(card, text=key, font=(self.font_family, 7, "bold"), fg=TEXT_SUB, bg=BG_CARD)
            lbl_title.pack(anchor=tk.W)
            
            lbl_val = tk.Label(card, text=default_val, font=(self.font_family, 11, "bold"), fg=default_color, bg=BG_CARD)
            lbl_val.pack(anchor=tk.W, pady=(1, 0))
            self.kpi_labels[key] = lbl_val

        # Aliases for backward compatibility
        self.epoch_label = self.kpi_labels["EPOCH"]
        self.return_label = self.kpi_labels["STRATEGY RETURN"]
        self.bh_label = self.kpi_labels["BUY & HOLD BTC"]
        self.sharpe_label = self.kpi_labels["SHARPE RATIO"]
        self.winrate_label = self.kpi_labels["WIN RATE"]
        self.trades_label = self.kpi_labels["TOTAL TRADES"]
        self.max_dd_label = self.kpi_labels["MAX DRAWDOWN"]

        # Matplotlib Plot Canvas
        plot_frame = tk.Frame(right_panel, bg=BG_MAIN)
        plot_frame.pack(fill=tk.BOTH, expand=True)

        plt.style.use('dark_background')
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(
            3, 1, figsize=(16, 12),
            gridspec_kw={'height_ratios': [2.2, 1.2, 1.2]},
            sharex=True
        )
        self.fig.patch.set_facecolor(BG_MAIN)
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.set_facecolor(BG_CARD)
            ax.grid(color=BORDER_CARD, alpha=0.6, linestyle='--')
            for spine in ax.spines.values():
                spine.set_color(BORDER_CARD)
            ax.tick_params(colors=TEXT_MUTED, labelsize=8)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Queue checking and initial cached backtest load
        self._check_queue_id = self.after(100, self.check_queue)
        self._initial_load_id = self.after(600, self.initial_load_backtest)

    def create_card(self, parent):
        return tk.Frame(parent, bg=BG_CARD, highlightbackground=BORDER_CARD, highlightthickness=1, padx=12, pady=10)

    def create_two_inputs_row(self, parent, label1, var1, label2, var2):
        row_frame = tk.Frame(parent, bg=BG_CARD)
        row_frame.pack(fill=tk.X, pady=2)

        col1 = tk.Frame(row_frame, bg=BG_CARD)
        col1.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))
        tk.Label(col1, text=label1, font=(self.font_family, 7, "bold"), fg=TEXT_MUTED, bg=BG_CARD).pack(anchor=tk.W)
        ModernEntry(col1, textvariable=var1, font_family=self.font_family).pack(fill=tk.X, pady=(2, 0))

        col2 = tk.Frame(row_frame, bg=BG_CARD)
        col2.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(3, 0))
        tk.Label(col2, text=label2, font=(self.font_family, 7, "bold"), fg=TEXT_MUTED, bg=BG_CARD).pack(anchor=tk.W)
        ModernEntry(col2, textvariable=var2, font_family=self.font_family).pack(fill=tk.X, pady=(2, 0))

    def create_single_input_row(self, parent, label_text, var):
        frame = tk.Frame(parent, bg=BG_CARD)
        frame.pack(fill=tk.X, pady=2)
        tk.Label(frame, text=label_text, font=(self.font_family, 7, "bold"), fg=TEXT_MUTED, bg=BG_CARD).pack(anchor=tk.W)
        ModernEntry(frame, textvariable=var, font_family=self.font_family).pack(fill=tk.X, pady=(2, 0))

    def create_preset_btn(self, parent, text, in_val, out_val):
        return tk.Button(
            parent,
            text=text,
            bg=BG_INPUT,
            fg=TEXT_WHITE,
            activebackground=BORDER_CARD,
            activeforeground="#ffffff",
            relief="flat",
            bd=0,
            highlightbackground=BORDER_INPUT,
            highlightthickness=1,
            font=(self.font_family, 8),
            cursor="hand2",
            pady=3,
            command=lambda: self.set_thresholds(in_val, out_val)
        )

    def set_thresholds(self, in_val_str, out_val_str):
        self.x_entry_var.set(in_val_str)
        self.x_exit_var.set(out_val_str)
        self.run_recalculate_backtest()

    def run_training(self):
        """Runs model training in a background daemon thread."""
        self.train_button.config(state="disabled", bg="#1e293b", cursor="arrow")
        self.recalc_button.config(state="disabled", bg="#1e293b", cursor="arrow")
        self.progress.pack(side=tk.RIGHT, padx=10)
        self.progress.start(10)
        print("Starting training process...")

        try:
            days_str = self.days_var.get().strip()
            days_to_load = int(days_str) if days_str.isdigit() and int(days_str) > 0 else None
            fees_pct = float(self.fees_var.get())
            n_models_val = int(self.n_models_var.get()) if self.n_models_var.get().isdigit() else 3
            if n_models_val < 1:
                n_models_val = 1

            params = {
                'epochs': int(self.epochs_var.get()),
                'batch_size': int(self.batch_size_var.get()),
                'learning_rate': float(self.lr_var.get()),
                'n_models': n_models_val,
                'n_steps': int(self.n_steps_var.get()),
                'k_steps': int(self.k_steps_var.get()),
                'days_to_load': days_to_load,
                'early_stopping': self.early_stopping_var.get(),
                'patience': 15,
                'x_entry': float(self.x_entry_var.get()),
                'x_exit': float(self.x_exit_var.get()),
                'fees_pct': fees_pct,
                'fees': fees_pct / 100.0
            }
        except ValueError:
            print("Invalid input parameters. Please check values.")
            self.progress.stop()
            self.progress.pack_forget()
            self.train_button.config(state="normal", bg=ACCENT_BLUE, cursor="hand2")
            self.recalc_button.config(state="normal", bg=ACCENT_PURPLE, cursor="hand2")
            return

        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.clear()
            ax.set_facecolor(BG_CARD)
            ax.grid(True, color=BORDER_CARD, alpha=0.6, linestyle='--')
            for spine in ax.spines.values():
                spine.set_color(BORDER_CARD)
            ax.tick_params(colors=TEXT_MUTED, labelsize=8)

        self.kpi_labels["EPOCH"].config(text="0")
        self.canvas.draw()

        thread = threading.Thread(target=train_model_with_callback, args=(ui_queue, params))
        thread.daemon = True
        thread.start()

    def run_recalculate_backtest(self):
        """
        Recalculates the backtest instantly using cached predictions for x_entry, x_exit, and fees.
        Does NOT re-train the model.
        """
        try:
            x_in = float(self.x_entry_var.get())
            x_out = float(self.x_exit_var.get())
            fees_pct = float(self.fees_var.get())
            fees = fees_pct / 100.0
        except ValueError:
            print("Invalid inputs. Please enter decimal numbers for thresholds and fees.")
            return

        print(f"Recalculating backtest with x_entry={x_in:.4f}, x_exit={x_out:.4f}, fees={fees_pct:.2f}%...")
        try:
            results = run_backtest_with_threshold(x_entry=x_in, x_exit=x_out, fees=fees)
            if results:
                self.update_backtest_ui(results)
            else:
                print("Could not run backtest. Train the model first to generate predictions.")
        except Exception as e:
            print(f"Backtest recalculation error: {e}")

    def update_backtest_ui(self, results):
        """Renders backtest results across all 3 subplots and updates metric badge cards."""
        if not results:
            return

        plot_backtest_results(results, self.fig, self.ax1, self.ax2, self.ax3)
        self.canvas.draw()

        pf = results['portfolio']
        returns = results.get('returns', 0.0)
        bh_return = results.get('bh_return', 0.0)
        sharpe = results.get('sharpe_ratio', 0.0)
        max_dd = results.get('max_drawdown', 0.0)
        total_trades = results.get('total_trades', int(pf.trades.count().sum()) if hasattr(pf.trades.count(), 'sum') else int(pf.trades.count()))
        winrate = results.get('win_rate', 0.0)

        # Update modern KPI badges
        self.kpi_labels["STRATEGY RETURN"].config(
            text=f"{returns:+.2f}%", 
            fg=ACCENT_GREEN if returns >= 0 else ACCENT_ROSE
        )
        self.kpi_labels["BUY & HOLD BTC"].config(
            text=f"{bh_return:+.2f}%",
            fg=ACCENT_AMBER if bh_return >= 0 else ACCENT_ROSE
        )
        self.kpi_labels["SHARPE RATIO"].config(
            text=f"{sharpe:.2f}",
            fg=ACCENT_GREEN if sharpe >= 1.0 else (ACCENT_PURPLE if sharpe >= 0.0 else ACCENT_ROSE)
        )
        self.kpi_labels["WIN RATE"].config(
            text=f"{winrate:.1f}%",
            fg=ACCENT_CYAN
        )
        self.kpi_labels["TOTAL TRADES"].config(
            text=f"{total_trades}",
            fg=BORDER_FOCUS
        )
        self.kpi_labels["MAX DRAWDOWN"].config(
            text=f"{max_dd:.2f}%",
            fg=ACCENT_ROSE if max_dd < 0 else TEXT_MUTED
        )
        if 'epoch' in results:
            self.kpi_labels["EPOCH"].config(text=f"{results['epoch']}", fg=TEXT_WHITE)

    def initial_load_backtest(self):
        """Loads and displays existing cached backtest on startup if available."""
        if os.path.exists("data/cached/model_predictions.pkl"):
            try:
                x_in = float(self.x_entry_var.get())
                x_out = float(self.x_exit_var.get())
                fees_pct = float(self.fees_var.get())
                fees = fees_pct / 100.0
                results = run_backtest_with_threshold(x_entry=x_in, x_exit=x_out, fees=fees)
                if results and len(results.get('preds', [])) > 500:
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
                    epoch = message.get('epoch')
                    if epoch is not None:
                        self.kpi_labels["EPOCH"].config(text=f"{epoch}", fg=TEXT_WHITE)

                elif msg_type == 'train_update':
                    epoch = message.get('epoch', 0) + 1
                    self.kpi_labels["EPOCH"].config(text=f"{epoch}", fg=TEXT_WHITE)

                elif msg_type == 'train_finished':
                    print("Training process finished.")
                    self.train_button.config(state="normal", bg=ACCENT_BLUE, cursor="hand2")
                    self.recalc_button.config(state="normal", bg=ACCENT_PURPLE, cursor="hand2")
                    self.progress.stop()
                    self.progress.pack_forget()

        except Empty:
            pass
        finally:
            self._check_queue_id = self.after(100, self.check_queue)

    def on_closing(self):
        print("Closing application...")
        if hasattr(self, '_check_queue_id'):
            try:
                self.after_cancel(self._check_queue_id)
            except Exception:
                pass
        if hasattr(self, '_initial_load_id'):
            try:
                self.after_cancel(self._initial_load_id)
            except Exception:
                pass
        self.destroy()
        os._exit(0)

if __name__ == "__main__":
    app = App()
    app.mainloop()
