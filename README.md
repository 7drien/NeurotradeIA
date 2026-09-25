# NeurotradeIA: Deep Learning Model for Trading

> **⚠️ Disclaimer**: This project is intended purely for visualization, educational purposes, and experimentation. It does not constitute financial advice, and the models or strategies developed here should not be used for live trading with real capital.

NeurotradeIA is a comprehensive pipeline designed to train deep learning models that predict the success of trading signals using **Triple Barrier Meta-Labeling**. The project seamlessly integrates data preparation, advanced preprocessing, model training, robust backtesting, and a graphical user interface (GUI) into a single cohesive ecosystem.

---

## 🚀 Technical Characteristics

### 🧠 Deep Learning Architecture & Strategy
* **Supported Models**: The architecture supports sequential neural networks, primarily focused on **LSTMs (Long Short-Term Memory)** with customizable layers.
* **Meta-Labeling & Triple Barrier Method**: Instead of simply predicting up or down, the model evaluates underlying signals (like Bollinger Bands with z-scores) and predicts whether taking a trade will hit a profit target, hit a stop loss, or expire (Failure/Hold vs Success/Trade).
* **Strict Chronological Validation**: To prevent data leakage and look-ahead bias, the dataset is split chronologically into **Train -> Validation**. 
  * *Train* is strictly the past.
  * *Validation* is the period immediately following the train set.
  * No global shuffling is permitted, and normalization is exclusively fitted on the training segment.

### ⚡ Performance Optimization
* **Data Caching**: Preprocessed DataFrames and scaler objects (`scaler.joblib`, `yfinance.cache`) are cached to allow for nearly instant data loading upon startup, avoiding the redundant recalculation of features or re-reading of heavy data on every run.
* **Cython Extensions**: Infrastructure exists to accelerate computationally heavy time-series operations (such as generating sliding windows) in C using **Cython**.

### 🔄 End-to-End Pipeline
1. **Data Ingestion**: Automated financial data downloading via `yfinance`.
2. **Preprocessing & Feature Engineering**: Calculates advanced technical indicators (z-score Bollinger Bands, MACD, RSI, ATR) and generates sequences (sliding windows of context candles).
3. **Training & Validation**: 
   * Dynamic learning rate scheduling and early stopping based on real-time validation metrics.
   * Auto-saving the best-performing model weights natively as `.keras`.
4. **Walk-Forward Backtesting**: Realistic simulation of trading strategies based on the model's signals triggered during validation.
   * Incorporates transaction costs.
   * Evaluated simultaneously with model training via custom Keras callbacks.

### 🖥️ User Interface (Tkinter)
* **Training Hub**: A dedicated Tkinter GUI (`main.py` and `ui/main_ui.py`) allows users to:
  * Dynamically configure hyperparameters (Epochs, Batch Size, Sequence Length, LSTM Units, Days to Load).
  * Monitor the loss curves, win rates, and validation metric curves in real-time.
* **Threaded Execution**: The training loop runs in an isolated background thread (daemon), ensuring the Tkinter UI remains responsive and does not freeze during intensive epochs.

---

## 🗂️ Project Structure

```text
project_root/
│
├── data/
│   ├── raw/                 # Raw data downloaded via yfinance
│   ├── processed/           # Transformed and normalized data
│   ├── cached/              # Cache files for accelerated loading
│
├── ui/
│   ├── main_ui.py           # Main Tkinter interface with dynamic settings
│   ├── model_tester.py      # Architecture testing utilities
│
├── src/
│   ├── __init__.py
│   ├── config.py            # Global default parameters
│   ├── config_loader.py     # Configuration utilities
│   ├── data_loader.py       # Downloading and caching logic
│   ├── features.py          # Technical indicators (BB z-score, MACD, etc.)
│   ├── preprocessing.py     # Normalization, meta-labeling, and sequence creation
│   ├── model.py             # Neural network definitions (LSTM, etc.)
│   ├── train.py             # Training loop with Meta-Labeling support
│   ├── callbacks.py         # Keras callbacks for UI logging and Backtesting
│   ├── backtesting.py       # Backtest engine with costs
│   ├── trading_env.py       # Custom Gymnasium trading environment for RL
│   ├── rl_train.py          # Stable-baselines3 PPO Reinforcement Learning training
│   ├── cython_extensions/   # Compiled Cython code for calculations
│   │   ├── fast_ops.pyx     # Optimized functions
│   │   ├── setup.py         # Cython compilation script
│
├── README.md                # Project documentation
├── requirements.txt         # Required Python libraries
├── main.py                  # Entry point for the application
└── .gitignore               # Ignored files
```

---

## 📦 Installation Guide

Follow these steps to install and run the project on your local machine.

### 1. Prerequisites
Ensure you have **Python 3.8+** installed on your system. 

### 2. Install Dependencies
Navigate to the project root directory and create a virtual environment (recommended). Then install the required Python packages using pip:

```bash
pip install -r requirements.txt
```
*Key libraries include: `tensorflow`, `pandas`, `numpy`, `yfinance`, `scikit-learn`, `gymnasium`, `stable-baselines3`, and `cython`.*

### 3. Compile Cython Extensions (Optional)
To benefit from Cython optimizations for fast operations, compile the C extensions:

```bash
cd src/cython_extensions/
python setup.py build_ext --inplace
cd ../../
```

### 4. Run the Application
Once dependencies are installed, launch the main Tkinter GUI:

```bash
python main.py
```

---

## 📍 Important Notes & Best Practices

* **Zero Data Leakage**: Never mix future and present data. Normalization parameters (mean, standard deviation) must strictly be calculated on the training set and applied to the validation/test sets.
* **Testing Horizons**: Experiment with multiple horizons for the sequence context size and target barriers.
* **Baselines**: Always compare complex neural networks against simple baseline models (e.g., Buy & Hold or simple moving average crossovers).
* **Market Noise**: Financial markets are mostly noise with slight drift. Beware of overfitting; if training loss approaches zero, the model is likely memorizing noise.

---

## 🔮 Future Roadmap

* **Advanced Reinforcement Learning**: Expand the `gymnasium` and `stable-baselines3` implementations to optimize dynamic position sizing and portfolio management on live data streams.
* **Advanced Ensembling**: Test probabilistic models and ensemble methods to gauge signal uncertainty.
* **Rolling Retrain**: Automate periodic model retraining to adapt to evolving market regimes (drift monitoring).
