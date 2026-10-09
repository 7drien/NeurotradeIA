import os
import sys
import threading
from queue import Queue, Empty

import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from src.train import train_model_with_callback
from src.backtesting import plot_backtest_results, run_backtest_with_threshold

ui_queue = Queue()

# Color Palette - Obsidian Quant Terminal
BG_MAIN = "#0b0e14"
BG_CARD = "#141722"
BORDER_CARD = "#232838"
BG_INPUT = "#1a1e2c"
BORDER_INPUT = "#2c3346"
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
        import tkinter.font as tkfont
        families = tkfont.families()
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
            try:
                details = tf.config.experimental.get_device_details(gpus[0])
                name = details.get('device_name', 'CUDA GPU')
                return f"CUDA: {name}", ACCENT_GREEN
            except Exception:
                return "CUDA: GPU Active", ACCENT_GREEN
    except Exception:
        pass
    return "CPU Active", TEXT_MUTED

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Configure CustomTkinter theme and appearance
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        self.title("NeurotradeAI - Quantitative Dashboard")
        self.geometry("1450x950")
        self.minsize(1200, 800)
        self.configure(fg_color=BG_MAIN)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.font_family = get_font_family()

        # Main horizontal container
        main_frame = ctk.CTkFrame(self, fg_color=BG_MAIN, corner_radius=0)
        main_frame.pack(fill="both", expand=True, padx=12, pady=10)

        # =========================================================================
        # LEFT PANEL: Parameters & Controls (Card-Based Layout, Scrollable)
        # =========================================================================
        left_panel = ctk.CTkScrollableFrame(
            main_frame,
            width=335,
            fg_color=BG_MAIN,
            corner_radius=0,
            scrollbar_button_color=BORDER_CARD,
            scrollbar_button_hover_color=BORDER_INPUT
        )
        left_panel.pack(side="left", fill="y", padx=(0, 12))

        # --- Card 1: Model & Training Configuration ---
        c1 = self.create_card(left_panel)
        c1.pack(fill="x", pady=(0, 10))

        c1_header = ctk.CTkFrame(c1, fg_color="transparent")
        c1_header.pack(fill="x", pady=(0, 8))
        ctk.CTkLabel(
            c1_header,
            text="MODEL CONFIGURATION",
            font=(self.font_family, 10, "bold"),
            text_color=BORDER_FOCUS
        ).pack(side="left")

        self.epochs_var = ctk.StringVar(value="200")
        self.batch_size_var = ctk.StringVar(value="128")
        self.lr_var = ctk.StringVar(value="0.0003")
        self.n_models_var = ctk.StringVar(value="3")
        self.days_var = ctk.StringVar(value="All")
        self.n_steps_var = ctk.StringVar(value="256")
        self.k_steps_var = ctk.StringVar(value="32")

        self.create_two_inputs_row(c1, "Epochs:", self.epochs_var, "Batch Size:", self.batch_size_var)
        self.create_two_inputs_row(c1, "Learning Rate:", self.lr_var, "Ensemble (n models):", self.n_models_var)
        self.create_two_inputs_row(c1, "Sequence (256):", self.n_steps_var, "Horizon (32):", self.k_steps_var)
        self.create_single_input_row(c1, "Days to Load:", self.days_var)

        self.early_stopping_var = ctk.BooleanVar(value=True)
        self.es_check = ctk.CTkCheckBox(
            c1,
            text="Enable Early Stopping (patience=15)",
            variable=self.early_stopping_var,
            font=(self.font_family, 10),
            text_color=TEXT_WHITE,
            fg_color=ACCENT_BLUE,
            hover_color=ACCENT_BLUE_HOVER,
            border_color=BORDER_INPUT,
            border_width=1,
            corner_radius=4,
            checkbox_width=18,
            checkbox_height=18
        )
        self.es_check.pack(anchor="w", pady=(6, 10))

        self.train_button = ctk.CTkButton(
            c1,
            text="▶  START TRAINING",
            font=(self.font_family, 11, "bold"),
            fg_color=ACCENT_BLUE,
            hover_color=ACCENT_BLUE_HOVER,
            text_color="#ffffff",
            corner_radius=6,
            height=34,
            cursor="hand2",
            command=self.run_training
        )
        self.train_button.pack(fill="x")

        # --- Card 2: Fast Oscillator Strategy (VectorBT) ---
        c2 = self.create_card(left_panel)
        c2.pack(fill="x", pady=(0, 10))

        c2_header = ctk.CTkFrame(c2, fg_color="transparent")
        c2_header.pack(fill="x", pady=(0, 2))
        ctk.CTkLabel(
            c2_header,
            text="FAST OSCILLATOR STRATEGY",
            font=(self.font_family, 10, "bold"),
            text_color=ACCENT_PURPLE
        ).pack(side="left")
        ctk.CTkLabel(
            c2_header,
            text="● VectorBT",
            font=(self.font_family, 8, "bold"),
            text_color=TEXT_SUB
        ).pack(side="right")

        rule_lbl = ctk.CTkLabel(
            c2,
            text="Long: Buy > 1+x_in  |  Exit < 1+x_out\nShort: Sell < 1-x_in  |  Exit > 1-x_out",
            font=(self.font_family, 8),
            text_color=TEXT_SUB,
            justify="left",
            anchor="w"
        )
        rule_lbl.pack(fill="x", pady=(0, 6))

        self.x_entry_var = ctk.StringVar(value="0.05")
        self.x_exit_var = ctk.StringVar(value="-0.02")
        self.fees_var = ctk.StringVar(value="0.1")

        self.create_two_inputs_row(c2, "Entry x_in (e.g. 0.05):", self.x_entry_var, "Exit x_out (e.g. -0.02):", self.x_exit_var)
        self.create_single_input_row(c2, "Fees per Trade (%):", self.fees_var)

        # Quick Presets
        ctk.CTkLabel(
            c2,
            text="Quick Presets (Entry / Exit):",
            font=(self.font_family, 8, "bold"),
            text_color=TEXT_MUTED,
            anchor="w"
        ).pack(fill="x", pady=(6, 2))

        p_row1 = ctk.CTkFrame(c2, fg_color="transparent")
        p_row1.pack(fill="x", pady=1)
        self.create_preset_btn(p_row1, "0.01 / 0.005", "0.01", "0.005").pack(side="left", fill="x", expand=True, padx=(0, 2))
        self.create_preset_btn(p_row1, "0.008 / 0.003", "0.008", "0.003").pack(side="left", fill="x", expand=True, padx=(2, 0))

        p_row2 = ctk.CTkFrame(c2, fg_color="transparent")
        p_row2.pack(fill="x", pady=1)
        self.create_preset_btn(p_row2, "0.015 / 0.008", "0.015", "0.008").pack(side="left", fill="x", expand=True, padx=(0, 2))
        self.create_preset_btn(p_row2, "0.02 / 0.01", "0.02", "0.01").pack(side="left", fill="x", expand=True, padx=(2, 0))

        self.recalc_button = ctk.CTkButton(
            c2,
            text="⚡  RECALCULATE BACKTEST",
            font=(self.font_family, 11, "bold"),
            fg_color=ACCENT_PURPLE,
            hover_color=ACCENT_PURPLE_HOVER,
            text_color="#ffffff",
            corner_radius=6,
            height=34,
            cursor="hand2",
            command=self.run_recalculate_backtest
        )
        self.recalc_button.pack(fill="x", pady=(8, 0))

        # --- Card 3: Indicator Statistical Distribution & Quantiles ---
        c3 = self.create_card(left_panel)
        c3.pack(fill="x")

        c3_header = ctk.CTkFrame(c3, fg_color="transparent")
        c3_header.pack(fill="x", pady=(0, 2))
        ctk.CTkLabel(
            c3_header,
            text="STATISTICAL DISTRIBUTION",
            font=(self.font_family, 10, "bold"),
            text_color=ACCENT_CYAN
        ).pack(side="left")
        ctk.CTkLabel(
            c3_header,
            text="● Quantiles",
            font=(self.font_family, 8, "bold"),
            text_color=TEXT_SUB
        ).pack(side="right")

        # Mini Matplotlib Distribution Plot (Histogram & Quantile Lines)
        self.dist_fig, self.dist_ax = plt.subplots(figsize=(3.1, 0.95), dpi=90)
        self.dist_fig.patch.set_facecolor(BG_CARD)
        self.dist_fig.subplots_adjust(left=0.08, right=0.96, top=0.92, bottom=0.28)
        self.format_dist_axis()
        self.dist_canvas = FigureCanvasTkAgg(self.dist_fig, master=c3)
        self.dist_canvas.get_tk_widget().pack(fill="x", pady=(2, 4))

        # Quantile & Extreme Value Badges Table
        self.stat_labels = {}
        stats_frame = ctk.CTkFrame(c3, fg_color="transparent")
        stats_frame.pack(fill="x")

        self.create_stat_row(stats_frame, "MIN", "min", ACCENT_ROSE, "MAX", "max", ACCENT_GREEN)
        self.create_stat_row(stats_frame, "P01 (1%)", "q01", "#f87171", "P99 (99%)", "q99", "#4ade80")
        self.create_stat_row(stats_frame, "P05 (5%)", "q05", "#fb923c", "P95 (95%)", "q95", "#38bdf8")
        self.create_stat_row(stats_frame, "Q25 (Q1)", "q25", TEXT_MUTED, "Q75 (Q3)", "q75", TEXT_MUTED)
        self.create_stat_row(stats_frame, "MEDIAN", "median", ACCENT_CYAN, "STD (σ)", "std", ACCENT_PURPLE)

        # =========================================================================
        # RIGHT PANEL: Top Header, Modern KPI Badges & Matplotlib Charts
        # =========================================================================
        right_panel = ctk.CTkFrame(main_frame, fg_color=BG_MAIN, corner_radius=0)
        right_panel.pack(side="right", fill="both", expand=True)

        # Header Title Bar
        header_frame = ctk.CTkFrame(right_panel, fg_color=BG_MAIN, corner_radius=0)
        header_frame.pack(fill="x", pady=(0, 6))

        title_left = ctk.CTkFrame(header_frame, fg_color=BG_MAIN, corner_radius=0)
        title_left.pack(side="left")
        ctk.CTkLabel(
            title_left,
            text="NEUROTRADE AI",
            font=(self.font_family, 15, "bold"),
            text_color=TEXT_WHITE
        ).pack(side="left")

        device_str, device_color = get_device_info()
        device_badge = ctk.CTkFrame(
            title_left,
            fg_color=BG_CARD,
            border_color=BORDER_CARD,
            border_width=1,
            corner_radius=6
        )
        device_badge.pack(side="left", padx=(12, 0))
        ctk.CTkLabel(
            device_badge,
            text=f"● {device_str}",
            font=(self.font_family, 9, "bold"),
            text_color=device_color
        ).pack(padx=8, pady=3)

        header_right = ctk.CTkFrame(header_frame, fg_color=BG_MAIN, corner_radius=0)
        header_right.pack(side="right")
        self.progress = ctk.CTkProgressBar(
            header_right,
            mode="indeterminate",
            width=140,
            progress_color=ACCENT_BLUE,
            fg_color=BG_CARD
        )

        # KPI Metric Cards Bar (7 Cards)
        metrics_bar = ctk.CTkFrame(right_panel, fg_color=BG_MAIN, corner_radius=0)
        metrics_bar.pack(fill="x", pady=(0, 8))

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
            card = ctk.CTkFrame(
                metrics_bar,
                fg_color=BG_CARD,
                border_color=BORDER_CARD,
                border_width=1,
                corner_radius=8
            )
            card.pack(side="left", fill="both", expand=True, padx=2)

            lbl_title = ctk.CTkLabel(
                card,
                text=key,
                font=(self.font_family, 8, "bold"),
                text_color=TEXT_SUB,
                anchor="w"
            )
            lbl_title.pack(anchor="w", padx=8, pady=(4, 0))

            lbl_val = ctk.CTkLabel(
                card,
                text=default_val,
                font=(self.font_family, 13, "bold"),
                text_color=default_color,
                anchor="w"
            )
            lbl_val.pack(anchor="w", padx=8, pady=(0, 4))
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
        plot_frame = ctk.CTkFrame(right_panel, fg_color=BG_MAIN, corner_radius=0)
        plot_frame.pack(fill="both", expand=True)

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
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

        # Queue checking and initial cached backtest load
        self._check_queue_id = self.after(100, self.check_queue)
        self._initial_load_id = self.after(600, self.initial_load_backtest)

    def create_card(self, parent):
        return ctk.CTkFrame(
            parent,
            fg_color=BG_CARD,
            border_color=BORDER_CARD,
            border_width=1,
            corner_radius=8
        )

    def create_two_inputs_row(self, parent, label1, var1, label2, var2):
        row_frame = ctk.CTkFrame(parent, fg_color="transparent")
        row_frame.pack(fill="x", pady=2)

        col1 = ctk.CTkFrame(row_frame, fg_color="transparent")
        col1.pack(side="left", fill="x", expand=True, padx=(0, 3))
        ctk.CTkLabel(
            col1,
            text=label1,
            font=(self.font_family, 8, "bold"),
            text_color=TEXT_MUTED,
            anchor="w"
        ).pack(fill="x")
        ctk.CTkEntry(
            col1,
            textvariable=var1,
            height=28,
            corner_radius=6,
            border_width=1,
            fg_color=BG_INPUT,
            border_color=BORDER_INPUT,
            text_color=TEXT_WHITE,
            font=(self.font_family, 10)
        ).pack(fill="x", pady=(1, 0))

        col2 = ctk.CTkFrame(row_frame, fg_color="transparent")
        col2.pack(side="right", fill="x", expand=True, padx=(3, 0))
        ctk.CTkLabel(
            col2,
            text=label2,
            font=(self.font_family, 8, "bold"),
            text_color=TEXT_MUTED,
            anchor="w"
        ).pack(fill="x")
        ctk.CTkEntry(
            col2,
            textvariable=var2,
            height=28,
            corner_radius=6,
            border_width=1,
            fg_color=BG_INPUT,
            border_color=BORDER_INPUT,
            text_color=TEXT_WHITE,
            font=(self.font_family, 10)
        ).pack(fill="x", pady=(1, 0))

    def format_dist_axis(self):
        self.dist_ax.set_facecolor(BG_CARD)
        for spine in self.dist_ax.spines.values():
            spine.set_color(BORDER_CARD)
        self.dist_ax.spines['top'].set_visible(False)
        self.dist_ax.spines['left'].set_visible(False)
        self.dist_ax.spines['right'].set_visible(False)
        self.dist_ax.tick_params(colors=TEXT_MUTED, labelsize=7, length=2, pad=1)
        self.dist_ax.set_yticks([])

    def create_stat_row(self, parent, label1, key1, color1, label2, key2, color2):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=1)

        c1 = ctk.CTkFrame(row, fg_color=BG_INPUT, border_color=BORDER_CARD, border_width=1, corner_radius=4)
        c1.pack(side="left", fill="x", expand=True, padx=(0, 2))
        ctk.CTkLabel(c1, text=label1, font=(self.font_family, 8), text_color=TEXT_MUTED).pack(side="left", padx=5, pady=2)
        val1 = ctk.CTkLabel(c1, text="-", font=(self.font_family, 9, "bold"), text_color=color1)
        val1.pack(side="right", padx=5, pady=2)
        self.stat_labels[key1] = val1

        c2 = ctk.CTkFrame(row, fg_color=BG_INPUT, border_color=BORDER_CARD, border_width=1, corner_radius=4)
        c2.pack(side="right", fill="x", expand=True, padx=(2, 0))
        ctk.CTkLabel(c2, text=label2, font=(self.font_family, 8), text_color=TEXT_MUTED).pack(side="left", padx=5, pady=2)
        val2 = ctk.CTkLabel(c2, text="-", font=(self.font_family, 9, "bold"), text_color=color2)
        val2.pack(side="right", padx=5, pady=2)
        self.stat_labels[key2] = val2

    def create_single_input_row(self, parent, label_text, var):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", pady=2)
        ctk.CTkLabel(
            frame,
            text=label_text,
            font=(self.font_family, 8, "bold"),
            text_color=TEXT_MUTED,
            anchor="w"
        ).pack(fill="x")
        ctk.CTkEntry(
            frame,
            textvariable=var,
            height=28,
            corner_radius=6,
            border_width=1,
            fg_color=BG_INPUT,
            border_color=BORDER_INPUT,
            text_color=TEXT_WHITE,
            font=(self.font_family, 10)
        ).pack(fill="x", pady=(1, 0))

    def create_preset_btn(self, parent, text, in_val, out_val):
        return ctk.CTkButton(
            parent,
            text=text,
            fg_color=BG_INPUT,
            hover_color=BORDER_CARD,
            border_width=1,
            border_color=BORDER_INPUT,
            text_color=TEXT_WHITE,
            corner_radius=5,
            height=24,
            font=(self.font_family, 9),
            cursor="hand2",
            command=lambda: self.set_thresholds(in_val, out_val)
        )

    def set_thresholds(self, in_val_str, out_val_str):
        self.x_entry_var.set(in_val_str)
        self.x_exit_var.set(out_val_str)
        self.run_recalculate_backtest()

    def run_training(self):
        """Runs model training in a background daemon thread."""
        self.train_button.configure(state="disabled", fg_color="#1e293b")
        self.recalc_button.configure(state="disabled", fg_color="#1e293b")
        self.progress.pack(side="right", padx=10)
        self.progress.start()
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
            self.train_button.configure(state="normal", fg_color=ACCENT_BLUE)
            self.recalc_button.configure(state="normal", fg_color=ACCENT_PURPLE)
            return

        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.clear()
            ax.set_facecolor(BG_CARD)
            ax.grid(True, color=BORDER_CARD, alpha=0.6, linestyle='--')
            for spine in ax.spines.values():
                spine.set_color(BORDER_CARD)
            ax.tick_params(colors=TEXT_MUTED, labelsize=8)

        self.kpi_labels["EPOCH"].configure(text="0")
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
        self.update_distribution_stats(results)

        pf = results['portfolio']
        returns = results.get('returns', 0.0)
        bh_return = results.get('bh_return', 0.0)
        sharpe = results.get('sharpe_ratio', 0.0)
        max_dd = results.get('max_drawdown', 0.0)
        total_trades = results.get('total_trades', int(pf.trades.count().sum()) if hasattr(pf.trades.count(), 'sum') else int(pf.trades.count()))
        winrate = results.get('win_rate', 0.0)

        # Update modern KPI badges
        self.kpi_labels["STRATEGY RETURN"].configure(
            text=f"{returns:+.2f}%",
            text_color=ACCENT_GREEN if returns >= 0 else ACCENT_ROSE
        )
        self.kpi_labels["BUY & HOLD BTC"].configure(
            text=f"{bh_return:+.2f}%",
            text_color=ACCENT_AMBER if bh_return >= 0 else ACCENT_ROSE
        )
        self.kpi_labels["SHARPE RATIO"].configure(
            text=f"{sharpe:.2f}",
            text_color=ACCENT_GREEN if sharpe >= 1.0 else (ACCENT_PURPLE if sharpe >= 0.0 else ACCENT_ROSE)
        )
        self.kpi_labels["WIN RATE"].configure(
            text=f"{winrate:.1f}%",
            text_color=ACCENT_CYAN
        )
        self.kpi_labels["TOTAL TRADES"].configure(
            text=f"{total_trades}",
            text_color=BORDER_FOCUS
        )
        self.kpi_labels["MAX DRAWDOWN"].configure(
            text=f"{max_dd:.2f}%",
            text_color=ACCENT_ROSE if max_dd < 0 else TEXT_MUTED
        )
        if 'epoch' in results:
            self.kpi_labels["EPOCH"].configure(text=f"{results['epoch']}", text_color=TEXT_WHITE)

    def update_distribution_stats(self, results):
        """Updates the mini distribution histogram and quantiles table in Card 3."""
        if not results:
            return
        preds = results.get('preds', None)
        if preds is None:
            return

        import numpy as np
        preds_vals = preds.values.flatten() if hasattr(preds, 'values') else np.array(preds).flatten()
        if len(preds_vals) == 0:
            return

        stats = results.get('stats', None)
        if stats is None:
            stats = {
                "min": float(np.min(preds_vals)),
                "max": float(np.max(preds_vals)),
                "q01": float(np.percentile(preds_vals, 1)),
                "q05": float(np.percentile(preds_vals, 5)),
                "q25": float(np.percentile(preds_vals, 25)),
                "median": float(np.percentile(preds_vals, 50)),
                "q75": float(np.percentile(preds_vals, 75)),
                "q95": float(np.percentile(preds_vals, 95)),
                "q99": float(np.percentile(preds_vals, 99)),
                "mean": float(np.mean(preds_vals)),
                "std": float(np.std(preds_vals)),
            }

        # Update numerical labels
        for k, lbl in self.stat_labels.items():
            if k in stats:
                lbl.configure(text=f"{stats[k]:.4f}")

        # Update mini distribution plot
        self.dist_ax.clear()
        self.format_dist_axis()

        n_bins = min(35, max(15, len(preds_vals) // 40))
        counts, bins, _ = self.dist_ax.hist(
            preds_vals, bins=n_bins, density=True,
            color='#0ea5e9', alpha=0.55,
            edgecolor='#0284c7', linewidth=0.5
        )

        med = stats.get('median', 1.0)
        self.dist_ax.axvline(med, color='#38bdf8', linestyle='-', linewidth=1.2, alpha=0.9)

        q05 = stats.get('q05', 0.98)
        q95 = stats.get('q95', 1.02)
        self.dist_ax.axvline(q05, color='#94a3b8', linestyle=':', linewidth=0.9, alpha=0.8)
        self.dist_ax.axvline(q95, color='#94a3b8', linestyle=':', linewidth=0.9, alpha=0.8)

        x_entry = results.get('x_entry', None)
        if x_entry is not None:
            buy_e = 1.0 + x_entry
            sell_e = 1.0 - x_entry
            b_min, b_max = float(bins.min()), float(bins.max())
            if b_min <= buy_e <= b_max:
                self.dist_ax.axvline(buy_e, color='#22c55e', linestyle='--', linewidth=1.1, alpha=0.85)
            if b_min <= sell_e <= b_max:
                self.dist_ax.axvline(sell_e, color='#ef4444', linestyle='--', linewidth=1.1, alpha=0.85)

        self.dist_canvas.draw()

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
                        self.kpi_labels["EPOCH"].configure(text=f"{epoch}", text_color=TEXT_WHITE)

                elif msg_type == 'train_update':
                    epoch = message.get('epoch', 0) + 1
                    self.kpi_labels["EPOCH"].configure(text=f"{epoch}", text_color=TEXT_WHITE)

                elif msg_type == 'train_finished':
                    print("Training process finished.")
                    self.train_button.configure(state="normal", fg_color=ACCENT_BLUE)
                    self.recalc_button.configure(state="normal", fg_color=ACCENT_PURPLE)
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
        try:
            plt.close('all')
        except Exception:
            pass
        self.destroy()
        sys.exit(0)

if __name__ == "__main__":
    app = App()
    app.mainloop()
